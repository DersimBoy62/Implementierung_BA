#!/usr/bin/env python3
"""V6.0: Exact prior-preserving elimination of ONE redundant M2 temporal feature.

Experimental add-on. Does not edit V5.2, V5.8 or V5.9; no outer-label scoring.
Full state feature g=(h_1..h_K, delta_1..delta_K) has delta_K=-sum_{i<K}delta_i.
Keep ALL K h occupancies (they jointly encode an intercept), retain K-1 delta's.
For f_full=A f_red, the *induced* regression prior is beta_red=A.T beta_full:
    mu_red=A.T mu_full, V_red=A.T V_full A (same inverse-gamma a,b).
A diagonal V_red obtained by dropping the last delta is NOT equivalent.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
V59=ROOT/'research_math_v59'
V55=ROOT/'research_convergence_v55'
V52=ROOT/'final_candidate_v52'
EXPECTED_V59='ed2de4bc67631701071ef9a84570a1682fe863cb6617a63f2e2f55cdfdcd570d'


def check(pred, msg):
    if not bool(pred): raise AssertionError(msg)


def dependencies():
    source=V59/'reduced_markov_logits_v59.py'
    check(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==EXPECTED_V59,
          'Required original V5.9 source missing or modified')
    if str(V59) not in sys.path:sys.path.insert(0,str(V59))
    import reduced_markov_logits_v59 as v59
    check(Path(v59.__file__).resolve()==source.resolve(),'Wrong imported V5.9 module')
    v58,mods=v59.checked_v58()
    return v59,v58,mods


def state_selector(K):
    """r_state=S @ g_full; g_full = A @ r_state for every M2 path."""
    check(K>=2,'At least two hidden states')
    S=np.zeros((2*K-1,2*K))
    for j in range(K):S[j,j]=1.
    for j in range(K-1):S[K+j,K+j]=1.
    A=S.T.copy()
    A[2*K-1,K:2*K-1]=-1.
    check(np.max(abs(S@A-np.eye(2*K-1)))<1e-14,'Wrong reduction map')
    return S,A


def full_feature_map(K,nphys):
    """A maps [all h, first K-1 delta, s] -> [all h, all delta, s]."""
    check(nphys>=0,'Wrong physical dimension')
    S_state,A_state=state_selector(K)
    full=2*K+nphys;red=2*K-1+nphys
    A=np.zeros((full,red))
    A[:2*K,:2*K-1]=A_state
    if nphys:A[2*K:,2*K-1:]=np.eye(nphys)
    S=np.zeros((red,full))
    S[:2*K-1,:2*K]=S_state
    if nphys:S[2*K-1:,2*K:]=np.eye(nphys)
    check(np.max(abs(S@A-np.eye(red)))<1e-14,'Wrong full reduction map')
    return S,A


def reduce_head_prior(prior,K,nphys,th):
    S,A=full_feature_map(K,nphys)
    check(prior.mu.shape==(A.shape[0],) and prior.V.shape==(A.shape[0],A.shape[0]),
          'Original prior dimensionality mismatch')
    mu=A.T@prior.mu
    V=A.T@prior.V@A
    check(np.linalg.eigvalsh(V).min()>0,'Induced prior is singular')
    return th.Head(mu,(V+V.T)*.5,float(prior.a),float(prior.b))


def reduce_head_moments(m_g,H_g,s,K,th):
    mg,Hg=th.augment(m_g,H_g,s)
    S,_=full_feature_map(K,len(s))
    m=S@mg;H=S@Hg@S.T
    return m,(H+H.T)*.5


def embed_head_coefficients(y,heads,ss,K,th):
    """Pull the reduced-head expected log-likelihood back to FULL M2 state features."""
    b_red,G_red,c=th.both_compressed_coeff(y,heads,ss,2*K-1)
    S,_=state_selector(K)
    b=S.T@b_red;G=S.T@G_red@S
    return b,(G+G.T)*.5,float(c)


def make_reduced_local(x,p,heads,y,ss,spec,steps,prev_full,v59,mods,th):
    dyn,full,fm,_,base,_,moments,obj_full=mods
    lp=base.elog_dir(p.pi);la=base.elog_dir(p.A);ell=full.elog_emission_full(x,p)
    T,K=ell.shape
    fb_v=fm.init_logits_from_fb(base.forward_backward(lp,la,ell))
    r_fb=v59.full_to_reduced(fb_v,T,K)
    b,G,c=embed_head_coefficients(y,heads,ss,K,th)
    def fg(r):
        f,g,*_=v59.reduced_objective_and_grad(r,lp,la,ell,b,G,spec,obj_full)
        return -f,-g
    fb_loss,_=fg(r_fb)
    has_old=prev_full is not None
    old_red=v59.full_to_reduced(prev_full,T,K) if has_old else None
    old_loss=fg(old_red)[0] if has_old else None
    choose_old=bool(has_old and old_loss<=fb_loss+1e-10)
    start=old_red if choose_old else r_fb
    start_loss=old_loss if choose_old else fb_loss
    if steps:
        opt=minimize(fg,start,jac=True,method='L-BFGS-B',
                     options={'maxiter':steps,'ftol':1e-10,'gtol':1e-5,'maxls':15})
        cand_loss,_=fg(opt.x)
        accepted=bool(np.isfinite(cand_loss) and cand_loss<start_loss-1e-10)
        best=opt.x if accepted else start
        niter=int(opt.nit)
    else:
        cand_loss=start_loss;best=start;accepted=False;niter=0
    found_loss,_=fg(best)
    check(found_loss<=fb_loss+1e-7,'Lost original FB q')
    if has_old:check(found_loss<=old_loss+1e-7,'Lost preceding q')
    fv=v59.reduced_to_full(best,T,K)
    q0,Q=fm.unpack_logits(fv,T,K)
    fb=fm.markov_marginals(q0,Q)
    m,H=moments(q0,Q,spec)
    return ({'fb':fb,'m_g':m,'H_g':H,
             'head_moments':[reduce_head_moments(m,H,np.asarray(s),K,th) for s in ss],
             'v':fv,'accepted':accepted,'iterations':niter,
             'local_gain':max(0.,start_loss-cand_loss)},
            {'selected_previous_q':choose_old,'accepted':accepted,'nit':niter,
             'full_logits':v59.dimensions(T,K)[0],
             'reduced_logits':v59.dimensions(T,K)[1],
             'reduced_head_dimension':len(heads[0].mu)})


def train_m2_reduced(rings,ydata,sdata,K,C,epochs,seed,steps,physical_var,verbose=False):
    v59,v58,mods=dependencies()
    dyn,full,fm,th,base,Spec,moments,obj_full=mods
    check(epochs>0 and steps>=0 and C>=0 and K>=2,'Invalid training controls')
    check(rings and ydata and set(ydata).issubset(rings) and set(sdata)==set(rings),
          'Wrong training subsets')
    check(all(len(z)==2 for z in sdata.values()),'Two physical feature vectors required')
    d=next(iter(rings.values())).shape[1]
    check(all(np.asarray(x).ndim==2 and np.asarray(x).shape[1]==d and len(x)>=4 and np.isfinite(x).all() for x in rings.values()),
          'Malformed process observation matrix')
    spec=Spec(K,'M2')
    nphys=tuple(len(sdata[next(iter(rings))][j]) for j in range(2))
    check(all(tuple(len(sdata[r][j]) for j in range(2))==nphys for r in rings),
          'Inconsistent physical feature counts')
    prior=full.make_priors_full(K,d)
    full_priors=tuple(dyn.prior_head(spec,nphys[j],physical_var) for j in range(2))
    red_priors=tuple(reduce_head_prior(full_priors[j],K,nphys[j],th) for j in range(2))
    sampled=np.vstack([x[::max(1,len(x)//120)] for x in rings.values()])
    centers=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sampled).cluster_centers_
    p=full.FullProc(prior.pi.copy(),prior.A.copy(),centers,np.full(K,4.),prior.nu.copy(),prior.W.copy())
    heads=list(red_priors)
    weights={rid:(1. if rid in ydata else C) for rid in rings}
    prev_local={};prev_total=None;history=[]
    for ep in range(1,epochs+1):
        local={};details=[]
        for rid,x in rings.items():
            if weights[rid]==0:continue
            if rid in ydata:
                prev=prev_local[rid]['v'] if rid in prev_local else None
                loc,detail=make_reduced_local(x,p,heads,ydata[rid],sdata[rid],spec,steps,
                                              prev,v59,mods,th)
                details.append(detail);local[rid]=loc
            else:
                old=dyn.local_unlabeled(x,p,sdata[rid],spec)
                old['head_moments']=[reduce_head_moments(old['m_g'],old['H_g'],np.asarray(s),K,th)
                                     for s in sdata[rid]]
                local[rid]=old
        e_local=v58.elbo_parts(rings,ydata,local,p,heads,prior,red_priors,C,mods)
        p_new=full.update_process_full(rings,local,prior,weights)
        e_proc=v58.elbo_parts(rings,ydata,local,p_new,heads,prior,red_priors,C,mods)
        h_new=[th.fit_head(ydata,local,j,red_priors[j]) for j in range(2)]
        e_heads=v58.elbo_parts(rings,ydata,local,p_new,h_new,prior,red_priors,C,mods)
        dl=None if prev_total is None else e_local['total']-prev_total
        dp=e_proc['total']-e_local['total'];dh=e_heads['total']-e_proc['total']
        violations=[{'stage':name,'decrease':float(delta)} for name,delta in
                    (('local',dl),('process',dp),('heads',dh))
                    if delta is not None and delta<-1e-6]
        row={'epoch':ep,'elbo':float(e_heads['total']),'delta_local':dl,
             'delta_process':float(dp),'delta_heads':float(dh),
             'n_block_violations':len(violations),'violations':violations,
             'head_dimensions_reduced':[len(h.mu) for h in h_new],
             'head_dimensions_full':[len(h.mu) for h in full_priors],
             'local_iterations_sum':sum(d['nit'] for d in details),
             'components':e_heads}
        history.append(row)
        if verbose:print('V60 M2-prior-preserving epoch',ep,'ELBO',round(row['elbo'],9),
                         'blocks',(None if dl is None else round(dl,7),round(dp,7),round(dh,7)),
                         'violations',len(violations),flush=True)
        p=p_new;heads=h_new;prev_local=local;prev_total=e_heads['total']
    return {'history':history,'head_priors':red_priors,'head_posteriors':heads}


def synthetic_test():
    v59,v58,mods=dependencies()
    dyn,full,fm,th,base,Spec,moments,obj_full=mods
    rng=np.random.default_rng(6101)
    records=[]
    for K in (2,3,4):
        T=K+6;spec=Spec(K,'M2')
        z=rng.normal(size=v59.dimensions(T,K)[1])*.4
        q0,Q=fm.unpack_logits(v59.reduced_to_full(z,T,K),T,K)
        mm,HH=moments(q0,Q,spec)
        S,A=state_selector(K)
        check(abs(mm[:K].sum()-1.)<1e-13,'Occupancy identity failed')
        check(abs(mm[K:].sum())<1e-13,'Delta mean identity failed')
        check(np.max(abs(mm-A@(S@mm)))<1e-12,'Exact feature span failed')
        check(np.max(abs(HH-A@(S@HH@S.T)@A.T))<2e-12,'Exact second moments span failed')
        for phys in (0,2):
            full_prior=dyn.prior_head(spec,phys,physical_var=.38526108747593474)
            red_prior=reduce_head_prior(full_prior,K,phys,th)
            Sf,Af=full_feature_map(K,phys)
            if phys:
                delta_index=K; cov=red_prior.V[delta_index:delta_index+K-1,delta_index:delta_index+K-1]
                expected=np.full((K-1,K-1),.15)+np.eye(K-1)*.15
                check(np.max(abs(cov-expected))<1e-14,'Wrong induced temporal covariance')
            local={};ydata={}
            for j in range(5):
                q=rng.normal(size=v59.dimensions(T+j,K)[1])*.5
                p0,pT=fm.unpack_logits(v59.reduced_to_full(q,T+j,K),T+j,K)
                m,H=moments(p0,pT,spec)
                s=rng.normal(size=phys)
                m_full,H_full=th.augment(m,H,s)
                r_m,r_H=reduce_head_moments(m,H,s,K,th)
                check(np.max(abs(m_full-Af@r_m))<2e-12,'Augmented mean not in subspace')
                check(np.max(abs(H_full-Af@r_H@Af.T))<2e-12,'Augmented second moment not in subspace')
                rid=f'S{j}';ydata[rid]=[float(rng.normal()),float(rng.normal())]
                local[rid]={'head_moments':[(m_full,H_full),(r_m,r_H)]}
            # fit separately with same ring labels, two heads but different coordinate maps
            f_loc={r:{'head_moments':[p['head_moments'][0]]} for r,p in local.items()}
            r_loc={r:{'head_moments':[p['head_moments'][1]]} for r,p in local.items()}
            f_post=th.fit_head(ydata,f_loc,0,full_prior)
            r_post=th.fit_head(ydata,r_loc,0,red_prior)
            check(np.max(abs(r_post.mu-Af.T@f_post.mu))<2e-9,'Induced posterior mean mismatch')
            check(np.max(abs(r_post.V-Af.T@f_post.V@Af))<2e-9,'Induced posterior covariance mismatch')
            check(abs(r_post.a-f_post.a)<1e-12 and abs(r_post.b-f_post.b)<2e-9,
                  'NIG hyperparameter mismatch')
            klf=base.kl_reg(f_post.mu,f_post.V,f_post.a,f_post.b,
                             full_prior.mu,full_prior.V,full_prior.a,full_prior.b)
            klr=base.kl_reg(r_post.mu,r_post.V,r_post.a,r_post.b,
                             red_prior.mu,red_prior.V,red_prior.a,red_prior.b)
            check(abs(klf-klr)<3e-9,'Regression KL changed after exact posterior update')
            for rid in local:
                (mf,Hf),(mr,Hr)=local[rid]['head_moments']
                y=float(ydata[rid][0])
                lf=th.quality_logterm(y,mf,Hf,f_post)
                lr=th.quality_logterm(y,mr,Hr,r_post)
                check(abs(lf-lr)<2e-9,'Expected quality loglikelihood changed')
                # Predictive mean/variance used by actual dynamic_quality_v2.predict.
                pred_f=float(f_post.mu@mf);pred_r=float(r_post.mu@mr)
                var_f=float((f_post.b/(f_post.a-1))*(1+np.trace(f_post.V@Hf))
                            +max(0.,f_post.mu@Hf@f_post.mu-pred_f**2))
                var_r=float((r_post.b/(r_post.a-1))*(1+np.trace(r_post.V@Hr))
                            +max(0.,r_post.mu@Hr@r_post.mu-pred_r**2))
                check(abs(pred_f-pred_r)<2e-9 and abs(var_f-var_r)<2e-9,
                      'Posterior predictive moment changed under M2 reparameterization')
            # A naive feature drop would leave variance .15 rather than the exact .30.
            check(abs(red_prior.V[K,K]-.30)<1e-14,
                  'Wrong variance for retained delta coefficient (sum of two .15 priors)')
            # Compare FULL and reduced local q(z) objective and analytic gradient.
            ss=[rng.normal(size=phys),rng.normal(size=phys)]
            full_heads=[f_post,f_post];red_heads=[r_post,r_post];yy=rng.normal(size=2)
            bf,Gf,cf=th.both_compressed_coeff(yy,full_heads,ss,spec.D)
            br,Gr,cr=embed_head_coefficients(yy,red_heads,ss,K,th)
            lp=rng.normal(size=K);la=rng.normal(size=(K,K));ell=rng.normal(size=(T,K))
            ff,gf,*_=obj_full(v59.reduced_to_full(z,T,K),lp,la,ell,bf,Gf,spec)
            fr,gr,*_=obj_full(v59.reduced_to_full(z,T,K),lp,la,ell,br,Gr,spec)
            check(abs((ff+cf)-(fr+cr))<3e-9,'Local quality objective changed by reduction')
            check(np.max(abs(gf-gr))<4e-9,'Local analytic logits gradient changed by reduction')
            records.append({'K':K,'n_phys':phys,'full_dimension':2*K+phys,
                            'reduced_dimension':2*K-1+phys,'kl_abs_error':abs(klf-klr),
                            'posterior_b_abs_error':abs(r_post.b-f_post.b),
                            'local_gradient_max_abs_error':float(np.max(abs(gf-gr)))})
    # Check explicitly that dropping BOTH occupancy and delta without intercept is invalid.
    K=2
    h=np.array([[.1,.9],[.7,.3]])
    target=np.array([1.2,3.4])
    # One state column alone cannot produce both responses without intercept.
    check(abs(np.linalg.lstsq(h[:,:1],target,rcond=None)[0][0]*h[0,0]-target[0])>.1,
          'Unexpected full occupancy equivalence without intercept')
    x={f'S{j}':rng.normal(size=(8+j,2)) for j in range(4)}
    y={'S0':np.array([.12,-.23]),'S1':np.array([-.3,.43]),'S2':np.array([.22,.04])}
    sd={rid:[np.array([.1]),np.array([-.1,.2])] for rid in x}
    short=train_m2_reduced(x,y,sd,K=2,C=.2,epochs=3,seed=20261111,
                           steps=8,physical_var=.38526108747593474)
    check(all(h['n_block_violations']==0 for h in short['history']),
          'Synthetic train has block ELBO violations')
    print('V60 SELF-TEST PASS: exact M2 affine subspace, induced NIG prior/posterior,',
          'KL/loglikelihood/local objective/grad identities K2/K3/K4,',
          '3-epoch block ELBO; frozen V52/V58/V59 SHA PASS; NO DM/TEST DATA')
    return records


def run_real(args):
    v59,v58,mods=dependencies()
    if str(V55) not in sys.path:sys.path.insert(0,str(V55))
    import convergence_probe_v55 as probe
    import main_bayesian_ssr_v52 as main
    check(2<=args.epochs<=40,'Allowed only 2..40 research epochs')
    path=Path(args.data)
    cfg,manifest=probe.preflight(path,Path(args.model_dir) if args.model_dir else None)
    out=Path(args.out)
    check(out.suffix.lower()=='.json' and out.parent.is_dir() and not out.exists() and not out.is_symlink(),
          'Output must be new JSON in existing directory')
    check(V52.resolve() not in out.resolve(strict=False).parents,'Cannot write to frozen mathematical core')
    if args.model_dir:check(Path(args.model_dir).resolve() not in out.resolve(strict=False).parents,
                            'Cannot write into stored baseline model')
    x,y,sdata,_=probe.build_training_arrays(path)
    check(len(x)==12 and len(y)==8,'Unexpected 8L+4U dev split')
    reference=None
    if args.reference:
        ref=Path(args.reference)
        check(ref.is_file(),'V59 reference missing')
        reference=json.loads(ref.read_text(encoding='utf-8'))
        check(reference.get('status')=='V59_REDUCED_LOGITS_MATH_EQUIVALENCE_RESEARCH_ONLY',
              'Unexpected V59 reference status')
        check(reference.get('source_sha256')==cfg['dataset_sha256'] and reference.get('epochs')==args.epochs,
              'Reference dataset or epochs mismatch')
    print('V60 TRAIN-ONLY exact M2 regression-prior reparam, no heldout quality scoring',flush=True)
    result=train_m2_reduced(x,y,sdata,K=cfg['K'],C=cfg['C'],epochs=args.epochs,
                            seed=cfg['random_seed'],steps=cfg['local_steps'],
                            physical_var=cfg['physical_var'],verbose=True)
    hist=result['history']
    compare=None
    if reference:
        diff=[a['elbo']-b['elbo'] for a,b in zip(hist,reference['history'])]
        compare={'max_abs_epoch_ELBO_difference':max(abs(z) for z in diff),
                 'final_ELBO_difference':diff[-1]}
    data={'status':'V60_M2_INDUCED_PRIOR_EXACT_REPARAM_RESEARCH_ONLY',
          'source_sha256':cfg['dataset_sha256'],
          'reference_model_sha256':None if manifest is None else manifest['model_sha256'],
          'seed':cfg['random_seed'],'epochs':args.epochs,
          'K':cfg['K'],'C':cfg['C'],'local_steps':cfg['local_steps'],
          'regression_heads_induced_full_to_reduced_prior':True,
          'historical_outer_scored':False,'model_weights_saved':False,
          'head_full_dimensions':hist[-1]['head_dimensions_full'],
          'head_reduced_dimensions':hist[-1]['head_dimensions_reduced'],
          'mathematics_core_sha256':v58.CORE_HASHES,
          'history':hist,'v59_reference_comparison':compare,
          'limitation':'Exact Bayesian reparameterization, not evidence of superior predictions or runtimes.'}
    with out.open('xb') as f:
        f.write((json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps({'status':data['status'],'out':str(out),
                      'final_elbo':hist[-1]['elbo'],
                      'max_abs_reference_elbo_diff':None if not compare else compare['max_abs_epoch_ELBO_difference'],
                      'block_violations':sum(x['n_block_violations'] for x in hist),
                      'full_head_dimensions':data['head_full_dimensions'],
                      'reduced_head_dimensions':data['head_reduced_dimensions']},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    subs=ap.add_subparsers(dest='cmd',required=True)
    subs.add_parser('self-test')
    p=subs.add_parser('run')
    p.add_argument('--data',required=True)
    p.add_argument('--model-dir')
    p.add_argument('--reference')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--out',required=True)
    args=ap.parse_args()
    if args.cmd=='self-test':synthetic_test()
    else:run_real(args)

if __name__=='__main__':main()

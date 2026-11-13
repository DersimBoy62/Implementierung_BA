#!/usr/bin/env python3
"""V6.1 research-only: audit Normal-Wishart and Dirichlet process posterior updates.

Preserves the V6.0 training objective and update order; stores aggregate numerical
statistics, no fitted model/prediction outputs. All cores and overlays SHA-gated.
Investigates the epoch 10->11 jump using per-block additive ELBO accounting.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
V60=ROOT/'research_math_v60'
V55=ROOT/'research_convergence_v55'
V52=ROOT/'final_candidate_v52'
V60_SHA='051987b95f83ecd4170154622cecda389fe0962dc7e9842f4c1fc169d0859d13'

def check(cond,msg):
    if not bool(cond): raise AssertionError(msg)

def deps():
    src=V60/'m2_prior_reduction_v60.py'
    check(src.is_file(),'Required V6.0 mathematical source missing')
    check(hashlib.sha256(src.read_bytes()).hexdigest()==V60_SHA,'V6.0 source checksum differs: DO NOT continue')
    if str(V60) not in sys.path:sys.path.insert(0,str(V60))
    import m2_prior_reduction_v60 as v60
    check(Path(v60.__file__).resolve()==src.resolve(),'Wrong V6.0 module imported')
    v59,v58,mods=v60.dependencies()
    return v60,v59,v58,mods

def weighted_stats(rings,ydata,local,C,K):
    L=np.zeros(K);U=np.zeros(K);startL=np.zeros(K);startU=np.zeros(K)
    transL=np.zeros((K,K));transU=np.zeros((K,K))
    by_ring=[]
    for rid,x in rings.items():
        if rid not in local:continue
        g=local[rid]['fb'].gamma;xi=local[rid]['fb'].xi
        n=g.sum(0);tr=xi.sum(0)
        if rid in ydata:
            L+=n;transL+=tr;startL+=g[0]
        else:
            U+=n;transU+=tr;startU+=g[0]
        by_ring.append({'ring_id':rid,'labeled':rid in ydata,'T':len(x),
                        'state_mass':(n/len(x)).tolist()})
    return {'nL':L.tolist(),'nU':U.tolist(),'n_weighted':(L+C*U).tolist(),
            'n_weighted_total':float((L+C*U).sum()),
            'transition_counts_weighted':(transL+C*transU).tolist(),
            'first_state_counts_weighted':(startL+C*startU).tolist(),
            'per_ring_state_mass':by_ring}

def post_summary(p):
    from numpy.linalg import eigvalsh
    means=[];variances=[];cond=[]
    for k in range(len(p.nu)):
        means.append(float(np.linalg.norm(p.mu[k])))
        # Actual Normal-Wishart posterior predictive x covariance, not an HMM residual correlation.
        d=p.mu.shape[1]
        cv=((p.kappa[k]+1)/(p.kappa[k]*(p.nu[k]-d-1)))*np.linalg.inv(p.W[k])
        ev=eigvalsh(cv)
        check(np.all(ev>0),'Predictive covariance not positive definite')
        variances.append({'min_eigenvalue':float(ev.min()),
                          'max_eigenvalue':float(ev.max()),
                          'trace':float(ev.sum()),'logdet':float(np.log(ev).sum())})
        cond.append(float(ev.max()/ev.min()))
    return {'mu_norm_per_state':means,
            'kappa':p.kappa.tolist(),'nu':p.nu.tolist(),
            'pred_cov_eigen':variances,'pred_cov_condition':cond,
            'transition_mean':(p.A/p.A.sum(1)[:,None]).tolist(),
            'pi_mean':(p.pi/p.pi.sum()).tolist()}

def posterior_change(old,new):
    # Direct state indexing is consistent between adjacent iterations; re-labeling not applied.
    return {'mu_fro':float(np.linalg.norm(new.mu-old.mu)),
            'mu_per_state':[float(np.linalg.norm(a-b)) for a,b in zip(old.mu,new.mu)],
            'W_fro':float(np.linalg.norm(new.W-old.W)),
            'transition_alpha_fro':float(np.linalg.norm(new.A-old.A)),
            'initial_alpha_fro':float(np.linalg.norm(new.pi-old.pi)),
            'max_change_predictive_transition':float(np.max(np.abs(
                new.A/new.A.sum(1)[:,None]-old.A/old.A.sum(1)[:,None])))}

def term_delta(pre,post):
    d={k:float(post[k]-pre[k]) for k in pre if k!='total'}
    result={'emission_NW':d['l_x']+d['u_x_weighted']-d['kl_NW'],
            'transition_Dirichlet':d['l_A']+d['u_A_weighted']-d['kl_A'],
            'initial_Dirichlet':d['l_pi']+d['u_pi_weighted']-d['kl_pi']}
    reconstructed=sum(result.values())
    target=float(post['total']-pre['total'])
    check(abs(reconstructed-target)<1e-5,f'Process stage additive identity failed {reconstructed-target}')
    # Quality / entropy terms cannot change when only global process posterior is updated.
    unchanged=('l_entropy','u_entropy_weighted','y_head1','y_head2','kl_heads')
    check(max(abs(d[k]) for k in unchanged)<1e-7,'Process update changed a non-process ELBO component')
    return {'component_deltas':d,'process_groups':{k:float(v) for k,v in result.items()},
            'process_total':target}

def emission_ring_deltas(rings,ydata,local,p_old,p_new,C,full):
    result=[];K=len(p_new.pi)
    for rid,x in rings.items():
        if rid not in local:continue
        gamma=local[rid]['fb'].gamma
        old=full.elog_emission_full(x,p_old);new=full.elog_emission_full(x,p_new)
        term=np.sum(gamma*(new-old),axis=0)*(1. if rid in ydata else C)
        result.append({'ring_id':rid,'labeled':rid in ydata,'weighted_emission_gain':float(term.sum()),
                       'per_state':term.tolist()})
    return result

def trace(rings,ydata,sdata,K,C,epochs,seed,steps,physical_var,verbose=False):
    v60,v59,v58,mods=deps()
    dyn,full,fm,th,base,Spec,moments,obj_full=mods
    check(epochs>=2 and K>=2 and C>=0 and steps>=0,'Bad run controls')
    check(set(ydata).issubset(rings) and set(sdata)==set(rings),'Data split error')
    spec=Spec(K,'M2');d=next(iter(rings.values())).shape[1]
    nphys=tuple(len(sdata[next(iter(rings))][j]) for j in range(2))
    prior=full.make_priors_full(K,d)
    full_priors=tuple(dyn.prior_head(spec,nphys[j],physical_var) for j in range(2))
    red_priors=tuple(v60.reduce_head_prior(full_priors[j],K,nphys[j],th) for j in range(2))
    sampled=np.vstack([x[::max(1,len(x)//120)] for x in rings.values()])
    centers=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sampled).cluster_centers_
    p=full.FullProc(prior.pi.copy(),prior.A.copy(),centers,np.full(K,4.),prior.nu.copy(),prior.W.copy())
    heads=list(red_priors);prev_local={};prev_total=None;history=[]
    weights={rid:(1. if rid in ydata else C) for rid in rings}
    for ep in range(1,epochs+1):
        local={};local_details=[]
        for rid,x in rings.items():
            if weights[rid]==0:continue
            if rid in ydata:
                prev=prev_local[rid]['v'] if rid in prev_local else None
                obj,detail=v60.make_reduced_local(x,p,heads,ydata[rid],sdata[rid],spec,steps,prev,
                                                  v59,mods,th)
                local[rid]=obj;detail['ring_id']=rid;local_details.append(detail)
            else:
                obj=dyn.local_unlabeled(x,p,sdata[rid],spec)
                obj['head_moments']=[v60.reduce_head_moments(obj['m_g'],obj['H_g'],np.asarray(s),K,th)
                                     for s in sdata[rid]]
                local[rid]=obj
        pre=v58.elbo_parts(rings,ydata,local,p,heads,prior,red_priors,C,mods)
        after_p=full.update_process_full(rings,local,prior,weights)
        post=v58.elbo_parts(rings,ydata,local,after_p,heads,prior,red_priors,C,mods)
        head_new=[th.fit_head(ydata,local,j,red_priors[j]) for j in range(2)]
        after=v58.elbo_parts(rings,ydata,local,after_p,head_new,prior,red_priors,C,mods)
        pd=term_delta(pre,post)
        nd=weighted_stats(rings,ydata,local,C,K)
        # Direct check of closed-form sufficient stats for Dirichlet and NW counts.
        check(np.allclose(after_p.kappa,prior.kappa+np.array(nd['n_weighted']),atol=1e-8),'NW kappa not prior + weighted counts')
        check(np.allclose(after_p.nu,prior.nu+np.array(nd['n_weighted']),atol=1e-8),'NW nu not prior + weighted counts')
        check(np.allclose(after_p.pi,prior.pi+np.array(nd['first_state_counts_weighted']),atol=1e-8),'Dirichlet pi update mismatch')
        check(np.allclose(after_p.A,prior.A+np.array(nd['transition_counts_weighted']),atol=1e-8),'Dirichlet transitions update mismatch')
        total=float(after['total']); dlocal=None if prev_total is None else float(pre['total']-prev_total)
        dhe=float(total-post['total'])
        if dlocal is not None:check(dlocal>=-1e-6,f'Local monotonicity violation at {ep}: {dlocal}')
        check(pd['process_total']>=-1e-6 and dhe>=-1e-6,'Process/head coordinate update decreased ELBO')
        row={'epoch':ep,'elbo':total,'delta_total':None if prev_total is None else total-prev_total,
             'delta_local':dlocal,'delta_process':pd['process_total'],'delta_heads':dhe,
             'process_posterior_ELBO_split':pd,'sufficient_statistics':nd,
             'posterior_before':post_summary(p),'posterior_after':post_summary(after_p),
             'posterior_change':posterior_change(p,after_p),
             'weighted_emission_by_ring':emission_ring_deltas(rings,ydata,local,p,after_p,C,full),
             'final_components':after,
             'local_selected_old':sum(a['selected_previous_q'] for a in local_details),
             'local_iterations_sum':sum(a['nit'] for a in local_details)}
        history.append(row)
        if verbose:
            groups=pd['process_groups'];print(f"V61 ep{ep:02d} ELBO={total:.6f} processΔ={pd['process_total']:.6f} "
                    f"NWemissionΔ={groups['emission_NW']:.6f} transitionsΔ={groups['transition_Dirichlet']:.6f} "
                    f"muMove={row['posterior_change']['mu_fro']:.6f}",flush=True)
        p=after_p;heads=head_new;prev_local=local;prev_total=total
    return history

def synthetic_test():
    v60,v59,v58,mods=deps()
    rng=np.random.default_rng(6102026)
    r={f'S{i}':rng.normal(size=(7+i,3)) for i in range(4)}
    y={'S0':np.array([.1,.3]),'S1':np.array([-.4,.8]),'S2':np.array([.7,-.2])}
    ss={rid:[rng.normal(size=2),rng.normal(size=1)] for rid in r}
    ours=trace(r,y,ss,2,.2,3,20261111,8,.25)
    target=v60.train_m2_reduced(r,y,ss,2,.2,3,20261111,8,.25)['history']
    check(max(abs(a['elbo']-b['elbo']) for a,b in zip(ours,target))<1e-8,'Different model after instrumentation')
    for item in ours:
        x=item['process_posterior_ELBO_split']
        check(abs(x['process_total']-sum(x['process_groups'].values()))<1e-5,'Additive ELBO split violated')
        check(item['delta_process']>-1e-6,'Process update not optimal')
    print('V61 SELF-TEST PASS: immutable core/overlays, NW-Dirichlet sufficient stats, additive process-ELBO decomposition, 3-epoch equivalence; NO DM DATA',flush=True)

def real_run(args):
    v60,v59,v58,mods=deps()
    if str(V55) not in sys.path:sys.path.insert(0,str(V55))
    import convergence_probe_v55 as probe
    cfg,manifest=probe.preflight(Path(args.data),Path(args.model_dir))
    check(2<=args.epochs<=40,'Epochs 2..40 only')
    out=Path(args.out)
    check(out.suffix=='.json' and out.parent.is_dir() and not out.exists() and not out.is_symlink(),
          'Output must be NEW JSON, parent existing, no symlink')
    check(V52.resolve() not in out.resolve(strict=False).parents,'Cannot write into immutable core')
    check(Path(args.model_dir).resolve() not in out.resolve(strict=False).parents,'Cannot overwrite stored model')
    x,y,s,_=probe.build_training_arrays(Path(args.data))
    check(len(x)==12 and len(y)==8,'Expected 8 L + 4 U development-only rings')
    ref=None
    if args.reference:
        ref=json.loads(Path(args.reference).read_text(encoding='utf-8'))
        check(ref.get('status')=='V60_M2_INDUCED_PRIOR_EXACT_REPARAM_RESEARCH_ONLY','Incorrect V6.0 reference')
        check(ref.get('source_sha256')==cfg['dataset_sha256'] and ref.get('epochs')==args.epochs,'Reference mismatch')
    hist=trace(x,y,s,cfg['K'],cfg['C'],args.epochs,cfg['random_seed'],cfg['local_steps'],cfg['physical_var'],verbose=True)
    error=None if ref is None else max(abs(a['elbo']-b['elbo']) for a,b in zip(hist,ref['history']))
    if error is not None:check(error<1e-5,f'Instrumentation changed baseline trajectory: {error}')
    focus=[]
    for h in hist:
        if h['epoch'] in (9,10,11,12,13,14):
            focus.append({'epoch':h['epoch'],'elbo':h['elbo'],
                          'delta_process':h['delta_process'],
                          'process_groups':h['process_posterior_ELBO_split']['process_groups'],
                          'weighted_state_n':h['sufficient_statistics']['n_weighted'],
                          'mu_change':h['posterior_change']['mu_fro'],
                          'pred_cov':h['posterior_after']['pred_cov_eigen']})
    data={'status':'V61_GLOBAL_PROCESS_POSTERIOR_MATH_FORENSICS_RESEARCH_ONLY',
          'source_sha256':cfg['dataset_sha256'],
          'reference_model_sha256':manifest['model_sha256'],'config_sha256':probe.m.release.FREEZE_CONFIG_HASH,
          'source_sha256_v60':V60_SHA,
          'mathematics_core_sha256':v58.CORE_HASHES,
          'seed':cfg['random_seed'],'K':cfg['K'],'C':cfg['C'],'epochs':args.epochs,
          'only_dev_train_data':True,'quality_scoring':False,'full_model_checkpoint_saved':False,
          'reference_max_abs_ELBO_diff':error,
          'history':hist,'focus_9_to_14':focus,
          'limitations':'Fixed-q process update decomposition; does not establish physical interpretation, prediction improvement or global optimum'}
    with out.open('xb') as stream:
        stream.write((json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    ep11=hist[10];print(json.dumps({'status':data['status'],'file':str(out),'n_epochs':len(hist),
               'max_abs_reference_diff':error,'epoch11_process_gain':ep11['delta_process'],
               'epoch11_process_gain_split':ep11['process_posterior_ELBO_split']['process_groups'],
               'epoch11_weighted_state_n':ep11['sufficient_statistics']['n_weighted'],
               'final_ELBO':hist[-1]['elbo']},indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sp=ap.add_subparsers(dest='cmd',required=True);sp.add_parser('self-test')
    run=sp.add_parser('run')
    run.add_argument('--data',required=True);run.add_argument('--model-dir',required=True)
    run.add_argument('--reference',required=True);run.add_argument('--out',required=True)
    run.add_argument('--epochs',type=int,default=30)
    args=ap.parse_args();synthetic_test() if args.cmd=='self-test' else real_run(args)
if __name__=='__main__':main()

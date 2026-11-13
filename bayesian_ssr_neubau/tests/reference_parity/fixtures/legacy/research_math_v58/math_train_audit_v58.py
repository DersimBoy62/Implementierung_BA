#!/usr/bin/env python3
"""V5.8: mathematischer Code-Audit der coordinate-ascent VI (Forschungszweig).

FROZEN V5.2 CORE IS READ-ONLY. Keine Testlabels, keine Vorhersagebewertung.
Experiments compare original local FB restart with previous-local warm-start
and explicitly bookkeep the *same* weighted variational objective after every
block update. No claim of global optimum or predictive improvement.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parent
V52=ROOT/'final_candidate_v52'
V55=ROOT/'research_convergence_v55'
if str(V52) not in sys.path:sys.path.insert(0,str(V52))
if str(V52/'final_core') not in sys.path:sys.path.insert(0,str(V52/'final_core'))

# The checksum catalog is the exact V5.2 mathematical code reviewed in V5.7.
CORE_HASHES={
 'compare_dynamic_quality_cv4.py':'90a6f474081c5ab4d11f81da5ebf86d9841efa8119eaf4c8d61ee95bd9c51eab',
 'compare_two_head_cv4.py':'79ef34dbb97981425e4d28d0765d172f9b8d932855302864b099556ab2882467',
 'dm_data.py':'cafe07406e125c253d7cf9de0aef9440c68a90810a0d60d839451852c36598b1',
 'dynamic_markov_v2.py':'db481edcad1497e448cd6b63591da08fe8e189962346837eda0d5973f66d5924',
 'dynamic_quality_v2.py':'fdfecbe62c12271cbbecdb269cfe69be56c1097755c2cb9d7d4bc6e73beb2348',
 'full_markov_b1.py':'3509e7e0c529f57492708619b46025163ab283a2ece2a7286ec32d67c540afba',
 'fullcov_b1.py':'701b0dfa099338e24226736fb6f1d263ad103d94ffa10b12a298770b0b739aab',
 'quality_link_challenge.py':'1a69d044f2c22fec843e53c75d2820f2d19d35677829040df9fe8260dd858197',
 'two_head_b1.py':'75705fe3090ca78d7b062f182ae991503d51b12e8081126a18cb99fc427c539b',
 'vb_b1.py':'d59ba4caaabfdb1ca2079ceefc14b918ac6eef83e90a2d7abe231d7be91e1e7f',
}

def check(cond, message):
    if not cond:raise AssertionError(message)

def check_core():
    for file,expected in CORE_HASHES.items():
        p=V52/'final_core'/file
        check(p.is_file(),f'Missing frozen core: {file}')
        got=hashlib.sha256(p.read_bytes()).hexdigest()
        check(got==expected,f'Frozen mathematical core has changed: {file}')
    import dynamic_quality_v2 as dyn
    import fullcov_b1 as full
    import full_markov_b1 as fm
    import two_head_b1 as th
    import vb_b1 as base
    from dynamic_markov_v2 import Spec, moments, objective_logits
    return dyn,full,fm,th,base,Spec,moments,objective_logits


def choose_local(x,p,heads,y,ss,spec,steps,prev_v,mods):
    """Maximize same local ELBO from best of process-FB and previous local q.

    No comparison across different global q(theta): BOTH candidates are
    re-evaluated under current global process/head posterior parameters.
    """
    dyn,full,fm,th,base,Spec,moments,objective_logits=mods
    lp=base.elog_dir(p.pi); la=base.elog_dir(p.A); ell=full.elog_emission_full(x,p)
    fb0=base.forward_backward(lp,la,ell)
    fb_v=fm.init_logits_from_fb(fb0)
    b,G,c=th.both_compressed_coeff(y,heads,ss,spec.D)
    def fg(v):
        val,grad,_,_=objective_logits(v,lp,la,ell,b,G,spec)
        return -val, -grad
    fb_loss,_=fg(fb_v)
    check(np.isfinite(fb_loss),'Nonfinite local FB objective')
    old_loss=None
    old_allowed=prev_v is not None
    if old_allowed:
        prev_v=np.asarray(prev_v,float)
        check(prev_v.shape==fb_v.shape and np.all(np.isfinite(prev_v)),
              'Malformed saved previous local Markov logits')
        old_loss,_=fg(prev_v)
        check(np.isfinite(old_loss),'Nonfinite previous local Markov objective')
    # Prefer previous local q on an equality: safeguard vs previous epoch.
    use_prev=bool(old_allowed and old_loss<=fb_loss+1e-10)
    start=prev_v if use_prev else fb_v
    start_loss=old_loss if use_prev else fb_loss
    if steps>0:
        sol=minimize(fg,start,jac=True,method='L-BFGS-B',
                     options={'maxiter':steps,'ftol':1e-10,'gtol':1e-5,'maxls':15})
        final_loss,_=fg(sol.x)
        accept=bool(np.isfinite(final_loss) and final_loss<start_loss-1e-10)
        v=sol.x if accept else start
        nit=int(sol.nit);message=str(sol.message)
    else:
        final_loss=start_loss;accept=False;v=start;nit=0;message='No optimizer iterations requested'
    chosen_loss,_=fg(v)
    # Numerically check both baselines are not worsened, with 1e-7 abs margin.
    check(chosen_loss<=fb_loss+1e-7, 'New local VI worse than process FB candidate')
    if old_allowed: check(chosen_loss<=old_loss+1e-7,'New local VI worse than previous q(z)')
    q0,Q=fm.unpack_logits(v,len(x),spec.K)
    fb=fm.markov_marginals(q0,Q)
    mm,HH=moments(q0,Q,spec)
    return ({'fb':fb,'m_g':mm,'H_g':HH,'head_moments':[th.augment(mm,HH,s) for s in ss],
             'local_gain':float(max(0.,start_loss-final_loss)) if np.isfinite(final_loss) else 0.,
             'accepted':accept,'iterations':nit,'v':v.copy()},
            {'used_prev':use_prev,'previous_available':old_allowed,
             'objective_fb':float(-fb_loss+c),
             'objective_old':None if old_loss is None else float(-old_loss+c),
             'objective_start':float(-start_loss+c),
             'objective_chosen':float(-chosen_loss+c),
             'gain_over_fb':float(fb_loss-chosen_loss),
             'gain_over_old':None if old_loss is None else float(old_loss-chosen_loss),
             'nit':nit,'accepted':accept,'message':message})


def elbo_parts(rings,ydata,local,p,heads,prior,priors_heads,C,mods):
    """Audit exact formula of locked dynamic_quality_v2.train, term by term."""
    dyn,full,fm,th,base,*_=mods
    lp=base.elog_dir(p.pi);la=base.elog_dir(p.A)
    result={'l_pi':0.,'l_A':0.,'l_x':0.,'l_entropy':0.,
            'u_pi_weighted':0.,'u_A_weighted':0.,'u_x_weighted':0.,'u_entropy_weighted':0.,
            'y_head1':0.,'y_head2':0.,
            'kl_pi':float(base.kl_dir(p.pi,prior.pi)),
            'kl_A':float(sum(base.kl_dir(a,b) for a,b in zip(p.A,prior.A))),
            'kl_NW':float(sum(full.kl_normal_wishart(p.mu[k],p.kappa[k],p.nu[k],p.W[k],
                            prior.mu[k],prior.kappa[k],prior.nu[k],prior.W[k]) for k in range(len(p.pi)))),
            'kl_heads':float(sum(base.kl_reg(h.mu,h.V,h.a,h.b,p0.mu,p0.V,p0.a,p0.b)
                            for h,p0 in zip(heads,priors_heads)))}
    for rid,x in rings.items():
        if rid not in local:continue
        fb=local[rid]['fb'];g=fb.gamma
        e_pi=float(g[0]@lp)
        e_A=float(np.sum(fb.xi*la[None,:,:]))
        e_x=float(np.sum(g*full.elog_emission_full(x,p)))
        ent=-float(np.sum(g[0]*np.log(np.maximum(g[0],1e-300))))
        conditional=fb.xi/np.maximum(g[:-1,:,None],1e-300)
        ent-=float(np.sum(fb.xi*np.log(np.maximum(conditional,1e-300))))
        if rid in ydata:
            for key,component in zip(('l_pi','l_A','l_x','l_entropy'),(e_pi,e_A,e_x,ent)):
                result[key]+=component
            for j in range(2):
                mg,Hg=local[rid]['head_moments'][j]
                result[f'y_head{j+1}']+=th.quality_logterm(float(ydata[rid][j]),mg,Hg,heads[j])
        else:
            for key,component in zip(('u_pi_weighted','u_A_weighted','u_x_weighted','u_entropy_weighted'),(e_pi,e_A,e_x,ent)):
                result[key]+=C*component
    total=sum(v for k,v in result.items() if not k.startswith('kl_'))-sum(v for k,v in result.items() if k.startswith('kl_'))
    result['total']=float(total)
    check(np.all(np.isfinite(list(result.values()))),'Nonfinite ELBO component')
    independently=float(th.kl_all(p,heads,prior,priors_heads))
    check(abs(independently-sum(v for k,v in result.items() if k.startswith('kl_')))<1e-6,
          'Global KL decomposition not equal to frozen core')
    return result


def train_audited(rings,ydata,sdata,K,C,epochs,seed,steps,physical_var=0.25,mode='warmstart',verbose=False):
    mods=check_core()
    dyn,full,fm,th,base,Spec,moments,objective_logits=mods
    check(mode in ('warmstart','original_restart'),'Unknown mode')
    spec=Spec(K,'M2')
    check(rings and ydata and set(ydata).issubset(rings) and set(sdata)==set(rings),
          'Invalid ring sets')
    check(C>=0 and steps>=0 and epochs>=1, 'Invalid iteration controls')
    d=next(iter(rings.values())).shape[1]
    check(all(np.asarray(x).ndim==2 and x.shape[1]==d and len(x)>=4 for x in rings.values()),'Wrong X dimensions')
    nphys=(len(sdata[next(iter(rings))][0]),len(sdata[next(iter(rings))][1]))
    prior=full.make_priors_full(K,d)
    priors_heads=tuple(dyn.prior_head(spec,nphys[j],physical_var) for j in range(2))
    sampled=np.vstack([x[::max(1,len(x)//120)] for x in rings.values()])
    centroids=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sampled).cluster_centers_
    p=full.FullProc(prior.pi.copy(),prior.A.copy(),centroids,np.full(K,4.),prior.nu.copy(),prior.W.copy())
    heads=list(priors_heads)
    weights={rid:(1. if rid in ydata else C) for rid in rings}
    previous_local={};history=[]; records=[]
    previous_total=None
    for ep in range(epochs):
        local={};ldetails=[]
        for rid,x in rings.items():
            if weights[rid]==0:continue
            if rid in ydata:
                previous_v=None
                if mode=='warmstart' and rid in previous_local:previous_v=previous_local[rid]['v']
                loc,detail=choose_local(x,p,heads,ydata[rid],sdata[rid],spec,steps,previous_v,mods)
                detail.update(epoch=ep+1,ring_id=rid)
                local[rid]=loc;ldetails.append(detail)
            else:
                local[rid]=dyn.local_unlabeled(x,p,sdata[rid],spec)
        e=elbo_parts(rings,ydata,local,p,heads,prior,priors_heads,C,mods)
        after_local=e['total']
        p2=full.update_process_full(rings,local,prior,weights)
        e_p=elbo_parts(rings,ydata,local,p2,heads,prior,priors_heads,C,mods)
        heads2=[th.fit_head(ydata,local,j,priors_heads[j]) for j in range(2)]
        e_h=elbo_parts(rings,ydata,local,p2,heads2,prior,priors_heads,C,mods)
        final=e_h['total']
        dlocal=None if previous_total is None else after_local-previous_total
        dp=e_p['total']-after_local;dh=final-e_p['total']
        row={'epoch':ep+1,'elbo':float(final),'after_local':float(after_local),
             'after_process':float(e_p['total']),'delta_local':None if dlocal is None else float(dlocal),
             'delta_process':float(dp),'delta_heads':float(dh),
             'delta_total':None if previous_total is None else float(final-previous_total),
             'labeled_prev_selected':sum(v['used_prev'] for v in ldetails),
             'labeled_steps_accepted':sum(v['accepted'] for v in ldetails),
             'state_mass':(sum(local[r]['fb'].gamma.sum(0) for r in local)/sum(len(rings[r]) for r in local)).tolist(),
             'components':{k:float(v) for k,v in e_h.items()},
             'global_mu_fro_change':float(np.linalg.norm(p2.mu-p.mu)),
             'global_A_alpha_change':float(np.linalg.norm(p2.A-p.A)),
             'global_W_fro_change':float(np.linalg.norm(p2.W-p.W))}
        for stage,value in [('local',dlocal),('process',dp),('heads',dh)]:
            if value is not None and value < -1e-6:
                row.setdefault('violations',[]).append({'stage':stage,'decrease':float(value)})
        history.append(row);records.extend(ldetails)
        if verbose:
            print('V58',mode,'ep',ep+1,'ELBO',round(final,6),
                  'local Δ',None if dlocal is None else round(dlocal,6),
                  'proc Δ',round(dp,6),'heads Δ',round(dh,6),
                  'violations',len(row.get('violations',[])),flush=True)
        previous_local=local;p=p2;heads=heads2;previous_total=final
    return {'history':history,'local':records}


def synthetic_test():
    mods=check_core()
    dyn,full,fm,th,base,Spec,moments,objective_logits=mods
    rng=np.random.default_rng(2905)
    spec=Spec(2,'M2')
    x=rng.normal(size=(7,2))
    p0=full.make_priors_full(2,2)
    p=full.FullProc(p0.pi.copy(),p0.A.copy(),np.array([[-.5,0.2],[0.4,-.5]]),
                    np.full(2,4.),p0.nu.copy(),p0.W.copy())
    heads=[th.Head(np.array([.9,-.6, .2,-.1,.05]),np.eye(5)*.1,2.5,1.5),
           th.Head(np.array([-.3,.8,-.4,.2,-.03]),np.eye(5)*.15,2.5,1.5)]
    yy=np.array([0.4,-.2]); ss=[np.array([0.3]),np.array([-.1])]
    previous=fm.init_logits_from_fb(base.forward_backward(base.elog_dir(p.pi),base.elog_dir(p.A),full.elog_emission_full(x,p)))
    prev=previous+rng.normal(scale=.15,size=len(previous))
    q,detail=choose_local(x,p,heads,yy,ss,spec,12,prev,mods)
    check(detail['gain_over_fb'] >= -1e-7 and detail['gain_over_old']>=-1e-7,
          'Max of FB and previous must not be degraded')
    check(abs(q['fb'].gamma.sum(1)-1).max()<1e-11,'Bad local marginals')
    check(np.min(np.linalg.eigvalsh(q['H_g']))>=-1e-9,'Quality second moment not PSD')
    # At zero new optimizer steps the chosen result must not degrade old q.
    q2,detail2=choose_local(x,p,heads,yy,ss,spec,0,q['v'],mods)
    check(detail2['gain_over_old']>=-1e-7,'Zero-step candidate selection should preserve old q')
    # Independent identity: delta of the local M2 objective must equal delta
    # of the full weighted ELBO when process/head global factors are fixed.
    lp=base.elog_dir(p.pi);la=base.elog_dir(p.A);ell=full.elog_emission_full(x,p)
    b,G,c=th.both_compressed_coeff(yy,heads,ss,spec.D)
    from_old,_grad,_,_=objective_logits(previous,lp,la,ell,b,G,spec)
    from_new,_grad,_,_=objective_logits(q['v'],lp,la,ell,b,G,spec)
    q00,Q0=fm.unpack_logits(previous,len(x),2)
    m0,H0=moments(q00,Q0,spec)
    fb0=fm.markov_marginals(q00,Q0)
    l0={'fb':fb0,'head_moments':[th.augment(m0,H0,s) for s in ss]}
    full_old=elbo_parts({'S1':x},{'S1':yy},{'S1':l0},p,heads,p0,heads,0.2,mods)['total']
    full_new=elbo_parts({'S1':x},{'S1':yy},{'S1':q},p,heads,p0,heads,0.2,mods)['total']
    check(abs((from_new-from_old)-(full_new-full_old))<1e-8,
          'Local M2 objective delta differs from full ELBO delta')
    # End-to-end synthetic coordinate proof with two labeled and one U ring.
    rr={'S1':rng.normal(size=(8,2)),'S2':rng.normal(size=(9,2)),'S3':rng.normal(size=(7,2))}
    yd={'S1':np.array([0.3,-.4]),'S2':np.array([-.2,.5])}
    st={r:[np.array([.1]),np.array([.3])] for r in rr}
    outcome=train_audited(rr,yd,st,2,0.2,4,20261111,8,mode='warmstart')
    check(len(outcome['history'])==4 and len(outcome['local'])==8,'Bad synthetic training output')
    for ep in outcome['history']:
        check(abs(ep['components']['total']-ep['elbo'])<1e-8,'ELBO decomposition mismatch')
        check(not ep.get('violations'),'Coordinate-ascent decreased synthetic ELBO: '+str(ep.get('violations')))
    print('V58 SELF-TEST PASS: 10 V52 mathematical SHA, competing Markov starts,')
    print(' exact M2 moments, 4 synthetic epochs, per-block ELBO accounting; NO DM DATA')


def real_run(args):
    check(V55.is_dir(),'Original V5.5 preprocessing overlay required')
    if str(V55) not in sys.path:sys.path.insert(0,str(V55))
    import convergence_probe_v55 as probe
    import main_bayesian_ssr_v52 as m
    check(args.epochs in range(2,41),'Research epoch budget 2..40 only')
    check(args.mode in ('warmstart','original_restart'),'Invalid mode')
    cfg,manifest=probe.preflight(pathlib.Path(args.data),pathlib.Path(args.model_dir))
    out=pathlib.Path(args.out)
    check(out.suffix.lower()=='.json' and out.parent.is_dir() and not out.exists() and not out.is_symlink(),
          'Output JSON must be NEW; parent must exist')
    check(V52.resolve() not in out.resolve(strict=False).parents,'No writing in frozen original code')
    check(pathlib.Path(args.model_dir).resolve() not in out.resolve(strict=False).parents,'No writing to original model')
    x,y,ss,rv=probe.build_training_arrays(pathlib.Path(args.data))
    check(len(x)==12 and len(y)==8,'Unexpected development split')
    print('V58 TRAIN ONLY, mathematical coordinate-ascent audit, no prediction scoring',flush=True)
    result=train_audited(x,y,ss,K=cfg['K'],C=cfg['C'],epochs=args.epochs,
         seed=cfg['random_seed'],steps=cfg['local_steps'],physical_var=cfg['physical_var'],
         mode=args.mode,verbose=True)
    outdict={'status':'V58_EXPLORATORY_CODE_MATH_COORDINATE_AUDIT_NOT_FINAL_MODEL',
             'mode':args.mode,'dataset_sha256':cfg['dataset_sha256'],
             'config_sha256':m.release.FREEZE_CONFIG_HASH,
             'frozen_model_sha256':manifest['model_sha256'],
             'frozen_core_sha256':CORE_HASHES,
             'train_only':True,'quality_prediction_scoring':False,
             'historical_old_test_used':False,
             'training_rings':list(sorted(x)),
             'quality_label_count':len(y),'epochs':args.epochs,
             'history':result['history'],
             'local_optimizer_details':result['local'],
             'warnings':['No new independent test performed.',
                         'A coordinate-wise ELBO increase does NOT establish global optimum or generalization.',
                         'Warmstart changes training algorithm and is exploratory, not an approved final model.',
                         'Monotonic diagnostics compare full weighted objectives at successive substeps; tolerances are numerical.']}
    with out.open('xb') as fp:fp.write((json.dumps(outdict,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8'))
    print(json.dumps({'status':outdict['status'],'mode':args.mode,'output':str(out),
          'final_elbo':result['history'][-1]['elbo'],
          'n_block_violations':sum(len(v.get('violations',[])) for v in result['history']),
          'max_block_decrease':float(min([0.]+[w['decrease'] for v in result['history'] for w in v.get('violations',[])])),
          'max_elbo_gain_epoch11':float(result['history'][10]['delta_total']) if args.epochs>=11 else None},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    cmds=parser.add_subparsers(dest='cmd',required=True)
    cmds.add_parser('self-test')
    r=cmds.add_parser('run')
    r.add_argument('--data',required=True);r.add_argument('--model-dir',required=True)
    r.add_argument('--epochs',type=int,default=30)
    r.add_argument('--mode',choices=('warmstart','original_restart'),default='warmstart')
    r.add_argument('--out',required=True)
    a=parser.parse_args()
    if a.cmd=='self-test':synthetic_test()
    else:real_run(a)

if __name__=='__main__':main()

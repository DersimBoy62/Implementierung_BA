#!/usr/bin/env python3
"""V5.9 Research: gauge-fixed (K-1)-logit local Markov VI for the V5.2 M2 model.

Read-only SHA-gated frozen cores, no held-out quality scoring, no model freeze.
The model family and exact local ELBO are unchanged; the optimizer coordinates are not.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V58 = ROOT/'research_math_v58'
V55 = ROOT/'research_convergence_v55'
V52 = ROOT/'final_candidate_v52'
V58_SHA = '585d51325443a7ef222f9afec4610779137372441263b1071c3317b247dde1bd'


def checked_v58():
    path = V58/'math_train_audit_v58.py'
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != V58_SHA:
        raise RuntimeError('Required V5.8 source absent or changed; do not patch original script')
    if str(V58) not in sys.path:sys.path.insert(0,str(V58))
    import math_train_audit_v58 as v58
    if Path(v58.__file__).resolve()!=path.resolve():
        raise RuntimeError('Wrong V5.8 module imported')
    modules=v58.check_core()  # SHA-checks all ten frozen V5.2 math modules
    return v58,modules


def check(cond, message):
    if not bool(cond):raise AssertionError(message)


def dimensions(T,K):
    if int(T)!=T or int(K)!=K or T<2 or K<2:raise ValueError('Require integer T>=2, K>=2')
    return K + (T-1)*K*K, (K-1)*(1+(T-1)*K)


def reduced_to_full(r,T,K):
    """Map K-1 log-odds per row to the original K logits (last logit=0).

    Bijection between R^{(K-1)*(1+(T-1)*K)} and strictly positive
    row-normalized Markov distributions q0, Q. No probabilistic restriction.
    """
    nfull,nred=dimensions(T,K)
    r=np.asarray(r,dtype=float)
    if r.shape!=(nred,) or not np.isfinite(r).all():raise ValueError('Invalid reduced logits')
    init=np.r_[r[:K-1],0.]
    rows=np.zeros((T-1,K,K))
    rows[...,:K-1]=r[K-1:].reshape(T-1,K,K-1)
    ans=np.r_[init,rows.ravel()]
    assert ans.shape==(nfull,)
    return ans


def full_to_reduced(v,T,K):
    """Choose a unique gauge for ANY original full-logit representation."""
    nfull,nred=dimensions(T,K)
    v=np.asarray(v,dtype=float)
    if v.shape!=(nfull,) or not np.isfinite(v).all():raise ValueError('Invalid full logits')
    q0=v[:K];Q=v[K:].reshape(T-1,K,K)
    r=np.r_[q0[:K-1]-q0[-1],(Q[...,:K-1]-Q[...,-1:]).ravel()]
    assert r.shape==(nred,)
    return r


def full_gradient_to_reduced(full_grad,T,K):
    """Chain rule for the *last-logit-fixed-to-zero* embedding.

    Full gradient has zero per-row sums from softmax. Reduced gradient is
    simply the first K-1 components in each row, NOT their differences.
    """
    nfull,nred=dimensions(T,K)
    g=np.asarray(full_grad,dtype=float)
    if g.shape!=(nfull,) or not np.isfinite(g).all():raise ValueError('Invalid full gradient')
    ans=np.r_[g[:K-1],g[K:].reshape(T-1,K,K)[...,:K-1].ravel()]
    assert ans.shape==(nred,)
    return ans


def reduced_objective_and_grad(r,lp,la,ell,b,G,spec,obj_full):
    T,K=ell.shape
    v=reduced_to_full(r,T,K)
    f,g,m,H=obj_full(v,lp,la,ell,b,G,spec)
    return f,full_gradient_to_reduced(g,T,K),m,H


def local_reduced(x,p,heads,y,ss,spec,steps,prev_full,mods):
    """Gauge-reduced equivalent of V5.8 safeguarded warm-start local VI."""
    dyn,full,fm,th,base,_,moments,obj_full=mods
    lp=base.elog_dir(p.pi);la=base.elog_dir(p.A);ell=full.elog_emission_full(x,p)
    fb0=base.forward_backward(lp,la,ell)
    T,K=ell.shape
    fb_full=fm.init_logits_from_fb(fb0)
    fb_red=full_to_reduced(fb_full,T,K)
    b,G,c=th.both_compressed_coeff(y,heads,ss,spec.D)
    def fg(r):
        f,g,*_=reduced_objective_and_grad(r,lp,la,ell,b,G,spec,obj_full)
        return -f,-g
    fb_loss,_=fg(fb_red)
    check(np.isfinite(fb_loss),'Non-finite FB local objective')
    old_allowed=prev_full is not None
    old_red=None;old_loss=None
    if old_allowed:
        old_red=full_to_reduced(prev_full,T,K)
        old_loss,_=fg(old_red)
        check(np.isfinite(old_loss),'Non-finite previous local objective')
    use_prev=bool(old_allowed and old_loss<=fb_loss+1e-10)
    start=old_red if use_prev else fb_red
    start_loss=old_loss if use_prev else fb_loss
    if steps>0:
        sol=minimize(fg,start,jac=True,method='L-BFGS-B',
                     options={'maxiter':steps,'ftol':1e-10,'gtol':1e-5,'maxls':15})
        alt,_=fg(sol.x)
        accepted=bool(np.isfinite(alt) and alt<start_loss-1e-10)
        chosen=sol.x if accepted else start
        niter=int(sol.nit);message=str(sol.message)
    else:
        alt=start_loss;accepted=False;chosen=start;niter=0;message='zero local steps'
    selected_loss,_=fg(chosen)
    check(selected_loss<=fb_loss+1e-7,'Reduced local VI degraded process-FB start')
    if old_allowed:check(selected_loss<=old_loss+1e-7,'Reduced local VI degraded saved q')
    full_v=reduced_to_full(chosen,T,K)
    q0,Q=fm.unpack_logits(full_v,T,K)
    fb=fm.markov_marginals(q0,Q)
    m,H=moments(q0,Q,spec)
    return ({'fb':fb,'m_g':m,'H_g':H,
             'head_moments':[th.augment(m,H,s) for s in ss],
             'local_gain':max(0.,start_loss-alt) if np.isfinite(alt) else 0.,
             'accepted':accepted,'iterations':niter,'v':full_v},
            {'used_prev':use_prev,'previous_available':old_allowed,
             'objective_start':float(-start_loss+c),
             'objective_chosen':float(-selected_loss+c),
             'gain_over_fb':float(fb_loss-selected_loss),
             'gain_over_old':None if not old_allowed else float(old_loss-selected_loss),
             'nit':niter,'accepted':accepted,'message':message,
             'n_full_logits':dimensions(T,K)[0],
             'n_reduced_logits':dimensions(T,K)[1]})


def train_reduced(rings,ydata,sdata,K,C,epochs,seed,steps,physical_var=0.25,verbose=True):
    """Same frozen global conjugate updates and full weighted ELBO as V5.8."""
    v58,mods=checked_v58()
    dyn,full,fm,th,base,Spec,moments,obj_full=mods
    spec=Spec(K,'M2')
    check(rings and ydata and set(ydata).issubset(rings) and set(sdata)==set(rings),
          'Invalid labeled/unlabeled data split')
    check(C>=0 and epochs>=1 and steps>=0,'Invalid controls')
    d=next(iter(rings.values())).shape[1]
    check(all(x.ndim==2 and len(x)>=4 and x.shape[1]==d and np.isfinite(x).all() for x in rings.values()),
          'Invalid process arrays')
    nphys=(len(sdata[next(iter(rings))][0]),len(sdata[next(iter(rings))][1]))
    prior=full.make_priors_full(K,d)
    priors_heads=tuple(dyn.prior_head(spec,nphys[j],physical_var) for j in range(2))
    sampled=np.vstack([x[::max(1,len(x)//120)] for x in rings.values()])
    centroids=KMeans(n_clusters=K,n_init=5,random_state=seed).fit(sampled).cluster_centers_
    p=full.FullProc(prior.pi.copy(),prior.A.copy(),centroids,np.full(K,4.),prior.nu.copy(),prior.W.copy())
    heads=list(priors_heads)
    weights={rid:(1. if rid in ydata else C) for rid in rings}
    prev_local={};prev_total=None;history=[];local_records=[]
    for ep in range(1,epochs+1):
        local={};details=[]
        for rid,x in rings.items():
            if weights[rid]==0:continue
            if rid in ydata:
                previous_v=prev_local[rid]['v'] if rid in prev_local else None
                loc,rec=local_reduced(x,p,heads,ydata[rid],sdata[rid],spec,steps,previous_v,mods)
                rec.update(epoch=ep,ring_id=rid)
                local[rid]=loc;details.append(rec)
            else:
                local[rid]=dyn.local_unlabeled(x,p,sdata[rid],spec)
        e_l=v58.elbo_parts(rings,ydata,local,p,heads,prior,priors_heads,C,mods)
        pnew=full.update_process_full(rings,local,prior,weights)
        e_p=v58.elbo_parts(rings,ydata,local,pnew,heads,prior,priors_heads,C,mods)
        headsnew=[th.fit_head(ydata,local,j,priors_heads[j]) for j in range(2)]
        e_h=v58.elbo_parts(rings,ydata,local,pnew,headsnew,prior,priors_heads,C,mods)
        dlocal=None if prev_total is None else e_l['total']-prev_total
        dprocess=e_p['total']-e_l['total']
        dheads=e_h['total']-e_p['total']
        violations=[]
        for stage,value in (('local',dlocal),('process',dprocess),('heads',dheads)):
            if value is not None and value<-1e-6:violations.append({'stage':stage,'decrease':float(value)})
        row={'epoch':ep,'elbo':e_h['total'],
             'after_local':e_l['total'],'after_process':e_p['total'],
             'delta_local':dlocal,'delta_process':dprocess,'delta_heads':dheads,
             'local_prev_selected':sum(int(z['used_prev']) for z in details),
             'local_optimizer_accepted':sum(int(z['accepted']) for z in details),
             'local_optimizer_iterations':sum(z['nit'] for z in details),
             'n_full_logit_parameters':sum(z['n_full_logits'] for z in details),
             'n_reduced_logit_parameters':sum(z['n_reduced_logits'] for z in details),
             'components':e_h,'violations':violations}
        history.append(row);local_records.extend(details)
        if verbose:print('V59 reduced M2 epoch',ep,'ELBO',round(e_h['total'],8),
                         'd(local/proc/heads)',str(tuple(None if v is None else round(v,8)
                         for v in (dlocal,dprocess,dheads))),
                         'violations',len(violations),flush=True)
        heads=headsnew;p=pnew;prev_local=local;prev_total=e_h['total']
    return {'history':history,'local_optimizer_details':local_records}


def synthetic_tests():
    v58,mods=checked_v58()
    dyn,full,fm,th,base,Spec,moments,obj_full=mods
    rng=np.random.default_rng(20261112)
    results=[]
    for T,K,variant in ((5,2,'M2'),(6,3,'M2'),(5,2,'M3'),(4,3,'M0')):
        spec=Spec(K,variant);D=spec.D
        r=rng.normal(size=dimensions(T,K)[1])*.5
        vf=reduced_to_full(r,T,K)
        # Add different arbitrary additive constants per softmax row; q is invariant.
        vv=np.r_[vf[:K]+rng.normal(),
                 (vf[K:].reshape(T-1,K,K)+rng.normal(size=(T-1,K,1))).ravel()]
        red2=full_to_reduced(vv,T,K)
        q0,Q=fm.unpack_logits(vv,T,K)
        q1,Q1=fm.unpack_logits(reduced_to_full(red2,T,K),T,K)
        check(max(np.max(abs(q0-q1)),np.max(abs(Q-Q1)))<3e-14,'Probability gauge invariance failed')
        check(np.max(abs(red2-r))<2e-14,'Full/reduced log-odds roundtrip failed')
        lp=rng.normal(size=K);la=rng.normal(size=(K,K));ell=rng.normal(size=(T,K))
        b=rng.normal(size=D)*.3
        M=rng.normal(size=(D,D));G=(M+M.T)*.05
        ffull,gfull,mfull,Hfull=obj_full(vf,lp,la,ell,b,G,spec)
        fred,gred,mred,Hred=reduced_objective_and_grad(r,lp,la,ell,b,G,spec,obj_full)
        fshift,gshift,_,_=obj_full(vv,lp,la,ell,b,G,spec)
        err_grad=float(np.max(abs(gred-full_gradient_to_reduced(gfull,T,K))))
        check(abs(ffull-fred)<2e-12 and abs(ffull-fshift)<2e-11,'Local objective changed by gauge')
        check(err_grad<2e-12,'Chain-rule analytic gradient changed')
        check(np.max(abs(mfull-mred))<2e-12 and np.max(abs(Hfull-Hred))<2e-12,'M2 moments changed')
        check(abs(gfull[:K].sum())<1e-11 and np.max(abs(gfull[K:].reshape(T-1,K,K).sum(-1)))<1e-11,
              'Full softmax gradient has a gauge-direction component')
        # Independent central finite differences of the *reduced-coordinate* objective.
        epsilon=1e-5
        fd=np.empty(len(r))
        for j in range(len(r)):
            delta=np.zeros(len(r));delta[j]=epsilon
            plus=reduced_objective_and_grad(r+delta,lp,la,ell,b,G,spec,obj_full)[0]
            minus=reduced_objective_and_grad(r-delta,lp,la,ell,b,G,spec,obj_full)[0]
            fd[j]=(plus-minus)/(2*epsilon)
        fd_error=float(np.max(abs(gred-fd)))
        check(fd_error<8e-6,f'Finite difference failed (T={T},K={K},{variant}), {fd_error}')
        results.append({'T':T,'K':K,'variant':variant,'full':dimensions(T,K)[0],
                        'reduced':dimensions(T,K)[1],'max_analytic_gradient_error':err_grad,
                        'max_finite_difference_error':fd_error})
    # Verify both optimizer coordinate systems can improve a SAME synthetic initial q.
    T,K=7,2;spec=Spec(K,'M2');D=spec.D
    r=rng.normal(size=dimensions(T,K)[1])*.3
    lp=rng.normal(size=K);la=rng.normal(size=(K,K));ell=rng.normal(size=(T,K))
    b=rng.normal(size=D)*.04;G=np.zeros((D,D))
    def fgfull(v):
        f,g,_,_=obj_full(v,lp,la,ell,b,G,spec)
        return -f,-g
    def fgred(r):
        f,g,_,_=reduced_objective_and_grad(r,lp,la,ell,b,G,spec,obj_full)
        return -f,-g
    orig=fgfull(reduced_to_full(r,T,K))[0]
    sfull=minimize(fgfull,reduced_to_full(r,T,K),jac=True,method='L-BFGS-B',options={'maxiter':50})
    sred=minimize(fgred,r,jac=True,method='L-BFGS-B',options={'maxiter':50})
    check(sfull.fun<=orig+1e-9 and sred.fun<=orig+1e-9,'Optimizer did not improve local objective')
    # Compare same computed q, not claiming identical optimizer trajectories.
    result=train_reduced({'S1':rng.normal(size=(8,2)),'S2':rng.normal(size=(9,2)),
                         'S3':rng.normal(size=(7,2))},
                         {'S1':np.array([.2,-.3]),'S2':np.array([-.1,.4])},
                         {rid:[np.array([.1]),np.array([-.1])] for rid in ('S1','S2','S3')},
                         K=2,C=.2,epochs=3,seed=20261111,steps=8,verbose=False)
    check(not any(row['violations'] for row in result['history']),'Synthetic train violated coordinate ELBO')
    print('V59 SELF-TEST PASS: 10 frozen V52 SHA; gauge/Markov moment/ELBO',
          'identity, analytic chain rule, finite differences (K2/K3/M0/M2/M3),',
          'optimizer and 3-epoch coordinate ELBO; NO DM/TEST TRAINING')
    return results


def read_reference(path,cfg,epochs):
    if path is None:return None
    p=Path(path)
    if not p.is_file():raise FileNotFoundError('V5.8 reference JSON not found')
    ref=json.loads(p.read_text(encoding='utf-8'))
    check(ref.get('status')=='V58_EXPLORATORY_CODE_MATH_COORDINATE_AUDIT_NOT_FINAL_MODEL',
          'Unexpected reference type')
    check(ref.get('mode')=='warmstart','Reference must use V5.8 warmstart mode')
    check(ref.get('dataset_sha256')==cfg['dataset_sha256'],'Dataset hash mismatch')
    check(ref.get('epochs')==epochs,'Epoch count mismatch')
    if len(ref.get('history',[]))!=epochs:raise ValueError('Reference history incomplete')
    return ref


def real_run(args):
    v58,mods=checked_v58()
    check(V55.is_dir(),'V5.5 input-splitting script is required')
    if str(V55) not in sys.path:sys.path.insert(0,str(V55))
    import convergence_probe_v55 as probe
    import main_bayesian_ssr_v52 as mainv52
    check(args.epochs in range(2,41),'2..40 epochs for this exploratory run')
    cfg,manifest=probe.preflight(Path(args.data),Path(args.model_dir))
    out=Path(args.out)
    check(out.suffix=='.json' and out.parent.is_dir() and not out.exists() and not out.is_symlink(),
          'Output must be a NEW .json with existing parent directory (write once)')
    check(V52.resolve() not in out.resolve(strict=False).parents,'Never write into frozen V5.2 core')
    check(Path(args.model_dir).resolve() not in out.resolve(strict=False).parents,
          'Never write into old frozen model')
    reference=read_reference(args.reference,cfg,args.epochs)
    rings,ydata,sdata,_=probe.build_training_arrays(Path(args.data))
    check(len(rings)==12 and len(ydata)==8,'Unexpected 8L+4U development split')
    print('V59 RESEARCH ONLY | REDUCED K-1 MARKOV LOGITS | SAME weighted ELBO',
          '| NO HELD-OUT TEST SCORING',flush=True)
    result=train_reduced(rings,ydata,sdata,cfg['K'],cfg['C'],args.epochs,
                         cfg['random_seed'],cfg['local_steps'],cfg['physical_var'],verbose=True)
    history=result['history']
    compare=None
    if reference is not None:
        differences=[a['elbo']-b['elbo'] for a,b in zip(history,reference['history'])]
        compare={'reference_path':str(args.reference),
                 'max_abs_elbo_difference':float(max(abs(x) for x in differences)),
                 'final_elbo_difference_reduced_minus_original':float(differences[-1]),
                 'per_epoch_difference':differences,
                 'caveat':'Numerically distinct optimizers may trace distinct paths despite exact objective equivalence.'}
    outdata={'status':'V59_REDUCED_LOGITS_MATH_EQUIVALENCE_RESEARCH_ONLY',
             'source_sha256':cfg['dataset_sha256'],
             'frozen_model_sha256':manifest['model_sha256'],
             'frozen_config_sha256':mainv52.release.FREEZE_CONFIG_HASH,
             'math_core_sha256':v58.CORE_HASHES,
             'seed':cfg['random_seed'],'C':cfg['C'],'K':cfg['K'],'local_steps':cfg['local_steps'],
             'reduced_markov_logits':True,'mode':'warmstart','epochs':args.epochs,
             'training_rings':sorted(rings),'quality_labels_count':len(ydata),
             'heldout_quality_scoring':False,'trained_model_saved':False,
             'history':history,'local_optimizer_details':result['local_optimizer_details'],
             'reference_comparison':compare,
             'limitations':['Exact parameterization of positive Markov q but altered optimizer geometry.',
                            'No claim of identical optimization trajectories, global optimum or generalization.',
                            'Existing model and source core not modified.']}
    with out.open('xb') as file:
        file.write((json.dumps(outdata,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps({'status':outdata['status'],'file':str(out),
                      'elbo_final':history[-1]['elbo'],
                      'max_absolute_reference_ELBO_deviation':None if compare is None else compare['max_abs_elbo_difference'],
                      'n_block_violations':sum(len(h['violations']) for h in history),
                      'max_local_logits_full':max(z['n_full_logits'] for z in result['local_optimizer_details']),
                      'max_local_logits_reduced':max(z['n_reduced_logits'] for z in result['local_optimizer_details'])},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('self-test')
    rr=sub.add_parser('run')
    rr.add_argument('--data',required=True)
    rr.add_argument('--model-dir',required=True)
    rr.add_argument('--epochs',type=int,default=30)
    rr.add_argument('--reference',default=None)
    rr.add_argument('--out',required=True)
    args=ap.parse_args()
    if args.cmd=='self-test':synthetic_tests()
    else:real_run(args)

if __name__=='__main__':main()

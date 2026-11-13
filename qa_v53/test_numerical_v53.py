"""V5.3 pre-freeze numerical QA for Bayesian-SSR V5.2 (synthetic inputs only).

Run from Implementierung_BA:
    .\\.venv\\Scripts\\python.exe .\\qa_v53\\test_numerical_v53.py

This test does NOT train a model, read DM process data, evaluate historical tests,
modify the ten frozen mathematical modules, or grant final model approval.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import pathlib
import platform
import sys
from datetime import datetime, timezone

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
V52 = HERE.parent / 'final_candidate_v52'
CORE = V52 / 'final_core'
if str(V52) not in sys.path: sys.path.insert(0, str(V52))
if str(CORE) not in sys.path: sys.path.insert(0, str(CORE))

import release_integrity_v52 as release
import dynamic_markov_v2 as mk
import full_markov_b1 as fm
import fullcov_b1 as fc
import dynamic_quality_v2 as dyn
import two_head_b1 as th
import vb_b1 as base

RESULTS = []


def checked(name, fn):
    value = fn()
    result = {'test': name, 'status': 'PASS', 'detail': value}
    RESULTS.append(result)
    print('PASS', name, (': '+str(value)) if value is not None else '', flush=True)


def must(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def close(a, b, tol=1e-8, message='unexpected numerical difference'):
    difference = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    must(difference <= tol, f'{message}: diff={difference:.6g}, tol={tol:.6g}')
    return difference


def test_freeze():
    cfg = release.validate_registered_config(V52/'PROPOSED_MODEL_FREEZE.json', CORE, V52)
    must(cfg['freeze_status'] == 'PROVISIONAL_USER_APPROVAL_PENDING', 'unexpected freeze status')
    must(cfg['K'] == 2 and cfg['C'] == 0.2 and cfg['local_steps'] == 8, 'wrong scientific config')
    return 'all 10 mathematical core SHA256 and V5.2 registered config verified'


def path_stats(q0, Q, spec):
    # Independent, explicit exact enumeration for K2 / T5, not reusing moment recursion.
    T = Q.shape[0]+1
    node, edge = mk.increments(T, spec)
    expected = np.zeros(spec.D)
    second = np.zeros((spec.D, spec.D))
    probsum = 0.
    for states in itertools.product(range(spec.K), repeat=T):
        prob = q0[states[0]]
        reward = node[0, states[0]].copy()
        for t in range(1, T):
            prob *= Q[t-1, states[t-1], states[t]]
            reward += node[t, states[t]] + edge[t-1, states[t-1], states[t]]
        probsum += prob
        expected += prob*reward
        second += prob*np.outer(reward, reward)
    must(abs(probsum-1) < 1e-12, 'enumerated probabilities not normalized')
    return expected, second


def test_m2_enumeration():
    rng = np.random.default_rng(531)
    T = 5; spec = mk.Spec(2, 'M2')
    q0 = rng.dirichlet([2, 3]); Q = rng.dirichlet([2,3], size=(T-1,2))
    m, H = mk.moments(q0, Q, spec)
    exact_m, exact_H = path_stats(q0, Q, spec)
    e = max(close(m,exact_m, 1e-12), close(H,exact_H,1e-12))
    return {'paths': 2**T, 'max_abs_error':e}


def test_local_objective_exact_path_enumeration():
    rng=np.random.default_rng(535);T=5;K=2;spec=mk.Spec(K,'M2')
    q0=rng.dirichlet([2.,3.]);Q=rng.dirichlet([4.,3.],size=(T-1,K))
    v=fm.init_logits_from_fb(fm.markov_marginals(q0,Q))
    q0,Q=fm.unpack_logits(v,T,K)
    logpi=np.log([.67,.33]);logA=np.log([[.83,.17],[.21,.79]])
    ell=rng.normal(size=(T,K))*.3
    b=rng.normal(size=spec.D)*.4
    mat=rng.normal(size=(spec.D,spec.D))*.04;G=(mat+mat.T)/2.
    observed,_,_,_=mk.objective_logits(v,logpi,logA,ell,b,G,spec)
    node,edge=mk.increments(T,spec)
    exact=0.
    for states in itertools.product(range(K),repeat=T):
        p=q0[states[0]]
        reward=node[0,states[0]].copy()
        term=logpi[states[0]]+ell[0,states[0]]-np.log(q0[states[0]])
        for t in range(1,T):
            j,k=states[t-1],states[t]
            p*=Q[t-1,j,k]
            reward+=node[t,k]+edge[t-1,j,k]
            term+=logA[j,k]+ell[t,k]-np.log(Q[t-1,j,k])
        exact+=p*(term+b@reward+reward@G@reward)
    e=close(observed,exact,1e-12)
    return {'paths':2**T,'max_abs_error':e,'scope':'local Markov+quality path objective, not full global ELBO'}


def test_m2_invariants():
    rng=np.random.default_rng(532); spec=mk.Spec(2,'M2')
    for T in (4,5,17,101):
        q0=rng.dirichlet([1.5,3]);Q=rng.dirichlet([3.,2.],size=(T-1,2))
        m,H=mk.moments(q0,Q,spec)
        must(np.isfinite(m).all() and np.isfinite(H).all(),'non-finite moments')
        close(np.sum(m[:2]),1.,1e-12)
        close(np.sum(m[2:]),0.,1e-12)
        close(H,H.T,1e-12)
        eig=np.linalg.eigvalsh(H)
        must(float(eig.min())>=-1e-10, 'second moments not PSD')
        must(float(np.linalg.eigvalsh(H-np.outer(m,m)).min())>=-1e-10,'covariance not PSD')
    return 'T=4,5,17,101: normalized, symmetric and PSD to numerical tolerance'


def logits_objective(v, logpi,logA,ell,b,G,spec):
    f,g,_,_=mk.objective_logits(v,logpi,logA,ell,b,G,spec)
    return f,g


def test_m2_gradient():
    rng=np.random.default_rng(533); T=6; K=2; spec=mk.Spec(K,'M2')
    q0=rng.dirichlet([2.,2.]); Q=rng.dirichlet([3.,4.],size=(T-1,K))
    v=fm.init_logits_from_fb(fm.markov_marginals(q0,Q))
    # objective_logits only uses unpack_logits; direct log probabilities also represent the same q.
    logpi=np.log(np.array([0.61,0.39]))
    logA=np.log(np.array([[0.9,0.1],[0.22,0.78]]))
    ell=rng.normal(0.,.3,size=(T,K))
    b=rng.normal(size=spec.D)*0.4
    U=rng.normal(size=(spec.D,spec.D))*0.1;G=(U+U.T)/2
    f,g=logits_objective(v,logpi,logA,ell,b,G,spec)
    must(math.isfinite(f) and np.isfinite(g).all(),'nonfinite objective/gradient')
    # Stable coordinate finite differences with centered differences, independent of reverse AD.
    eps=1e-5
    probes=np.unique(np.linspace(0,len(v)-1,7,dtype=int))
    errs=[]
    for idx in probes:
        vv=v.copy();vv[idx]+=eps
        fp,_=logits_objective(vv,logpi,logA,ell,b,G,spec)
        vv[idx]-=2*eps
        fm_,_=logits_objective(vv,logpi,logA,ell,b,G,spec)
        errs.append(abs((fp-fm_)/(2*eps)-g[idx]))
    e=max(errs)
    must(e<2e-6, f'gradient finite differences too far: {e}')
    return {'coordinates':len(probes),'max_abs_error':e}


def test_m2_long_sequence():
    spec=mk.Spec(2,'M2');T=2048
    q0=np.array([0.54,0.46]);Q=np.tile(np.array([[.998,.002],[.003,.997]]),(T-1,1,1))
    m,H=mk.moments(q0,Q,spec)
    must(np.isfinite(m).all() and np.isfinite(H).all(),'long-sequence moments not finite')
    close(m[:2].sum(),1.,2e-10)
    close(m[2:].sum(),0.,2e-10)
    must(np.linalg.eigvalsh(H-np.outer(m,m)).min()>-1e-9,'long covariance not PSD')
    return {'T':T,'occupancy':[float(x) for x in m[:2]]}


def test_m2_near_deterministic():
    spec=mk.Spec(2,'M2'); T=80
    tiny=1e-12
    q0=np.array([1.-tiny,tiny]);Q=np.tile(np.array([[1.-tiny,tiny],[tiny,1.-tiny]]),(T-1,1,1))
    m,H=mk.moments(q0,Q,spec)
    must(np.isfinite(m).all() and np.isfinite(H).all(), 'near deterministic probabilities caused nonfinite moments')
    close(m[:2].sum(),1.,2e-10)
    return {'T':T,'min_probability':tiny}


def test_kl_self():
    for d in (2,6):
        m=np.linspace(-.2,.2,d);W=np.diag(np.linspace(.03,.3,d));nu=d+3.
        kl=fc.kl_normal_wishart(m,.31,nu,W,m,.31,nu,W)
        must(abs(kl)<1e-8,f'KL(q||q) nonzero: {kl}')
    return 'Normal-Wishart KL(q||q)=0 for d=2 and d=6'


def test_kl_perturbed():
    d=6;W=np.diag(np.geomspace(1e-8,1,d));m=np.zeros(d);nu=d+5.
    self_kl=fc.kl_normal_wishart(m,.3,nu,W,m,.3,nu,W)
    alt=fc.kl_normal_wishart(m+np.array([1e-3,0,0,0,0,0]),.35,nu+1,W,m,.3,nu,W)
    must(np.isfinite([self_kl,alt]).all() and abs(self_kl)<1e-7 and alt>0,'near-degenerate KL failure')
    return {'scale_condition':float(np.linalg.cond(W)),'self_kl':float(self_kl),'perturbed_kl':float(alt)}


def test_invalid_wishart_scale_rejected():
    W=np.diag([1.,-0.1])
    try:fc.expected_log_precision(5.,W)
    except ArithmeticError:return 'indefinite precision Wishart scale correctly rejected'
    raise AssertionError('indefinite Wishart scale silently accepted')


def test_emission_and_process_update():
    rng=np.random.default_rng(534);K=2;d=6;T=50
    prior=fc.make_priors_full(K,d)
    x0=rng.normal(0.,.2,size=(T,1))
    x=np.hstack([x0+i*1e-8 for i in range(d)]) # intentionally highly collinear
    g=np.tile(np.array([[.7,.3]]),(T,1))
    xi=np.tile(np.array([[[.63,.07],[.06,.24]]]),(T-1,1,1))
    class FB:pass
    fb=FB();fb.gamma=g;fb.xi=xi
    p=fc.update_process_full({'SYNTH':x},{'SYNTH':{'fb':fb}},prior,{'SYNTH':1.})
    lp=fc.elog_emission_full(x,p)
    must(np.isfinite(lp).all() and np.isfinite(p.W).all(),'nonfinite emissions/posterior')
    for k in range(K):
        must(np.linalg.eigvalsh(p.W[k]).min()>0,'Wishart W not PD')
        kl=fc.kl_normal_wishart(p.mu[k],p.kappa[k],p.nu[k],p.W[k],
                               prior.mu[k],prior.kappa[k],prior.nu[k],prior.W[k])
        must(math.isfinite(kl) and kl>=-1e-8,'nonfinite/negative KL update')
    return {'T':T,'features':d,'worst_W_condition':float(max(np.linalg.cond(w) for w in p.W))}


def test_nig_quality_and_predictive():
    spec=mk.Spec(2,'M2');q0=np.array([.6,.4]);T=15
    Q=np.tile(np.array([[.96,.04],[.13,.87]]),(T-1,1,1))
    m,H=mk.moments(q0,Q,spec)
    priors=[dyn.prior_head(spec,2,0.38526108747593474) for _ in range(2)]
    sample={'A':{'head_moments':[th.augment(m,H,np.array([.1,.2])),th.augment(m,H,np.array([-.1,.2]))]},
            'B':{'head_moments':[th.augment(m,H,np.array([.2,.4])),th.augment(m,H,np.array([-.2,.4]))]}}
    y={'A':np.array([.6,1.2]),'B':np.array([.8,1.5])}
    for j in range(2):
        head=th.fit_head(y,sample,j,priors[j])
        must(np.linalg.eigvalsh(head.V).min()>0 and head.b>0 and head.a>1,'invalid quality NIG posterior')
        a,A=sample['A']['head_moments'][j]
        mu=float(head.mu@a)
        var=(head.b/(head.a-1.))*(1.+np.trace(head.V@A)) + max(0.,head.mu@A@head.mu-mu*mu)
        must(np.isfinite([mu,var]).all() and var>0,'invalid predictive mean or variance')
    return 'both synthetic Bayesian NIG heads yield finite positive predictive variance'


def synthetic_local_pair(y):
    spec=mk.Spec(2,'M2');prior=fc.make_priors_full(2,6)
    p=fc.FullProc(prior.pi.copy(),prior.A.copy(),np.zeros((2,6)),
                  np.full(2,4.),prior.nu.copy(),prior.W.copy())
    T=20;x=np.zeros((T,6));x[:,0]=np.sin(np.arange(T)/3.)*.1
    features=[np.zeros(2),np.zeros(2)]
    heads=[dyn.prior_head(spec,2,.38526108747593474) for _ in range(2)]
    for head in heads:
        # Deliberately strengthened synthetic head, to make coupling observable.
        head.mu[:4]=np.array([3.,-3.,1.,-1.]);head.a=8.;head.b=.25
    a=dyn.local_joint(x,p,heads,np.array([y,y]),features,spec,0)
    b=dyn.local_joint(x,p,heads,np.array([y,y]),features,spec,8)
    return a,b


def test_local_steps_zero_vs_eight():
    a,b=synthetic_local_pair(1.)
    close(a['m_g'][:2].sum(),1.,1e-12)
    close(b['m_g'][:2].sum(),1.,1e-12)
    must(a['local_gain']==0. and not a['accepted'] and a['iterations']==0,'steps=0 changed local q')
    must(b['accepted'] and b['local_gain']>1e-5,'quality-informed local optimization did not improve synthetic objective')
    delta=float(np.max(np.abs(a['m_g']-b['m_g'])))
    must(delta>1e-3,'no observable local M2 change after 8 VI steps')
    return {'local_gain':b['local_gain'],'max_M2_moment_shift':delta,'iterations':b['iterations']}


def test_local_label_direction():
    _,pos=synthetic_local_pair(1.);_,neg=synthetic_local_pair(-1.)
    must(pos['m_g'][0]>0.6 and neg['m_g'][0]<0.4,'synthetic heads not responding to label direction')
    must(pos['accepted'] and neg['accepted'],'synthetic local label response not optimized')
    return {'occupancy_z1_y_plus':float(pos['m_g'][0]),'occupancy_z1_y_minus':float(neg['m_g'][0])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json-out',default=None,help='Optional NEW results JSON path; will not overwrite')
    args=parser.parse_args()
    tests=[
        ('locked_V52_config_and_ten_math_hashes',test_freeze),
        ('M2_exact_path_enumeration',test_m2_enumeration),
        ('local_objective_exact_path_enumeration',test_local_objective_exact_path_enumeration),
        ('M2_probability_PSD_invariants',test_m2_invariants),
        ('M2_analytic_vs_finite_difference_gradient',test_m2_gradient),
        ('M2_long_2048_steps',test_m2_long_sequence),
        ('M2_near_deterministic_transitions',test_m2_near_deterministic),
        ('Normal_Wishart_KL_self_identity',test_kl_self),
        ('Normal_Wishart_near_degenerate_W',test_kl_perturbed),
        ('invalid_Wishart_scale_fail_closed',test_invalid_wishart_scale_rejected),
        ('full_covariance_update_collinear_process',test_emission_and_process_update),
        ('two_NIG_heads_positive_predictive_variance',test_nig_quality_and_predictive),
        ('fixed_global_local_VI_steps_0_vs_8',test_local_steps_zero_vs_eight),
        ('fixed_global_label_direction_response',test_local_label_direction),
    ]
    try:
        for name,test in tests:checked(name,test)
    except Exception as exc:
        print('FAIL', name, repr(exc), file=sys.stderr, flush=True)
        RESULTS.append({'test':name,'status':'FAIL','error':repr(exc)})
        ok=False
    else:ok=True
    report={'suite':'DM-V53-SYNTHETIC-NUMERICAL-QA-PREFREEZE-v1',
            'run_utc':datetime.now(timezone.utc).isoformat(),
            'v52_release':release.RELEASE,'v52_config_sha256':release.FREEZE_CONFIG_HASH,
            'environment':release.environment_snapshot(),
            'no_training':True,'no_dataset_deserialization':True,'no_historical_test':True,
            'synthetic_only':True,'global_core_unchanged':True,
            'n_pass':sum(r['status']=='PASS' for r in RESULTS),
            'n_total':len(tests),'results':RESULTS,
            'caveats':['No user Windows 3.11 verification yet',
                       'No empirical local feedback on real 8L+4U data',
                       'Synthetic strong quality heads are deliberately not fitted to DM labels',
                       'Not an independent quality/generalization test',
                       'Not a global proof of convergence or numerical stability',
                       'V5.2 model freeze remains provisional']}
    if args.json_out:
        out=pathlib.Path(args.json_out)
        if not out.parent.is_dir():raise FileNotFoundError('Output parent directory must already exist')
        with out.open('x',encoding='utf-8') as fh:
            json.dump(report,fh,indent=2,ensure_ascii=False,allow_nan=False);fh.write('\n')
    print(f"V5.3 QA: {report['n_pass']}/{len(tests)} synthetische numerische Tests PASS; "
          "KEIN TRAINING / KEINE DM-DATEN / KEIN ALTER TEST", flush=True)
    return 0 if ok else 1


if __name__=='__main__':sys.exit(main())

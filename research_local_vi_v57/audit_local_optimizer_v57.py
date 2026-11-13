#!/usr/bin/env python3
"""V5.7 read-only-on-core audit of the real local L-BFGS-B optimizer.

In-memory observation hooks only; no changes to SHA-locked V5.2 core or model,
no use of historical external-test ring labels, no prediction scoring.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, hashlib
from collections import Counter
import numpy as np
from scipy.optimize import minimize as scipy_minimize

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parent
V55 = PROJECT / 'research_convergence_v55'
if str(V55) not in sys.path: sys.path.insert(0, str(V55))
import convergence_probe_v55 as probe
m = probe.m

VALID_STEPS = (8, 16)
VALID_SEED = 20261111
EPOCHS = 30


def check(test, msg):
    if not test: raise ValueError(msg)


def as_real(value):
    x=float(value)
    check(np.isfinite(x),'Nonfinite optimizer metric')
    return x


def run_optimizer_audit(original, fn, x0, **kwargs):
    # One read-only evaluation of the same pure objective used by scipy.
    f0, g0 = fn(x0)
    check(np.isfinite(f0) and np.all(np.isfinite(g0)), 'Nonfinite initial local objective/gradient')
    sol = original(fn, x0, **kwargs)
    f1, g1 = fn(sol.x)
    check(np.isfinite(f1) and np.all(np.isfinite(g1)), 'Nonfinite final local objective/gradient')
    data = {
        'nit':int(sol.nit), 'nfev':int(sol.nfev),
        'success':bool(sol.success), 'status':int(sol.status),
        'message':str(sol.message),
        'hit_iteration_budget':bool(int(sol.nit)>=kwargs['options']['maxiter']),
        'initial_objective_to_minimize':as_real(f0),
        'final_objective_to_minimize':as_real(f1),
        'objective_reduction':as_real(f0-f1),
        'initial_grad_inf_norm':as_real(np.max(np.abs(g0))),
        'final_grad_inf_norm':as_real(np.max(np.abs(g1))),
        'dimension':int(len(x0)),
    }
    return sol, data


def read_reference(path, seed, steps, source_hash, config_hash):
    obj=json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    check(obj['source_sha256']==source_hash and obj['config_sha256']==config_hash,'Reference data/config differs')
    if obj['status']=='EXPLORATORY_TRAIN_ONLY_CONVERGENCE_DIAGNOSTIC_NOT_MODEL_SELECTION':
        rseed=int(obj['seed']); rsteps=8
    else:
        check(obj['status']=='V56_EXPLORATORY_TRAIN_ONLY_STABILITY_NOT_MODEL_SELECTION','Unrecognized training reference report')
        rseed=int(obj['seed']); rsteps=int(obj['local_vi_maxiter'])
    check(rseed==seed and rsteps==steps, 'Reference seed or local maxiter mismatch')
    h=np.asarray(obj['metrics']['elbo_per_epoch'],dtype=float)
    check(h.shape==(EPOCHS,) and np.isfinite(h).all(),'Reference history must have 30 finite epochs')
    return h


def run(args):
    check(args.local_steps in VALID_STEPS, 'Only 8 or 16 iteration controlled cases')
    check(args.seed==VALID_SEED,'Only original 20261111 seed for controlled local comparison')
    data = pathlib.Path(args.data)
    model_dir=pathlib.Path(args.model_dir)
    cfg,man=probe.preflight(data,model_dir)
    out=pathlib.Path(args.out)
    check(out.suffix.lower()=='.json', 'JSON output is required')
    check(out.parent.is_dir() and not out.exists() and not out.is_symlink(),'Output must be NEW and parent directory must exist (write-once)')
    check(probe.V52.resolve() not in out.resolve(strict=False).parents, 'Forbidden output in frozen code')
    check(model_dir.resolve() not in out.resolve(strict=False).parents, 'Forbidden output in frozen model')
    x,y,headfeatures,rv=probe.build_training_arrays(data)
    check(len(y)==8 and len(x)==12,'Wrong development training ring counts')
    id_to_ring={id(value): key for key,value in x.items() if key in y}
    check(len(id_to_ring)==8,'Cannot track the eight labeled input arrays')
    reference=read_reference(pathlib.Path(args.reference),args.seed,args.local_steps,cfg['dataset_sha256'],m.release.FREEZE_CONFIG_HASH)

    original_minimize=m.dyn.minimize
    original_local_joint=m.dyn.local_joint
    pending=[]
    records=[]
    def tracing_minimize(fun,x0,**kwargs):
        # Preserve scipy's inputs, options and returned result. Only log pure fn evaluations.
        check(kwargs.get('method')=='L-BFGS-B','Unexpected optimizer method')
        check(int(kwargs.get('options',{}).get('maxiter',-1))==args.local_steps,'Optimizer budget differs')
        sol,d=run_optimizer_audit(original_minimize,fun,x0,**kwargs)
        pending.append(d)
        return sol
    def tracing_local_joint(xarr,*a,**kw):
        check(id(xarr) in id_to_ring,'Labeled ring array could not be identified')
        before=len(pending)
        res=original_local_joint(xarr,*a,**kw)
        check(len(pending)==before+1,'Exactly one optimizer call expected per labeled ring')
        d=pending.pop()
        d['ring_id']=id_to_ring[id(xarr)]
        d['epoch']=len(records)//len(y)+1
        d['accepted']=bool(res['accepted'])
        d['actual_local_gain']=as_real(res['local_gain'])
        check(d['nit']==int(res['iterations']),'Captured nit differs from core result')
        records.append(d)
        return res
    print(f'V57 RESEARCH ONLY | local optimizer trace | seed={args.seed} maxiter={args.local_steps} epochs=30',flush=True)
    try:
        m.dyn.minimize=tracing_minimize
        m.dyn.local_joint=tracing_local_joint
        model=m.dyn.train(x,y,headfeatures,variant='M2',K=cfg['K'],C=cfg['C'],epochs=EPOCHS,
            steps=args.local_steps,seed=args.seed,physical_var=cfg['physical_var'],verbose=True)
    finally:
        m.dyn.minimize=original_minimize
        m.dyn.local_joint=original_local_joint
    check(len(records)==8*EPOCHS and len(pending)==0,'Missing local optimizer records')
    got=np.asarray([r['elbo'] for r in model['history']],dtype=float)
    maxdiff=as_real(np.max(np.abs(got-reference)))
    check(maxdiff<1e-5, f'Instrumented training differs from original: max abs ELBO diff {maxdiff}')
    per_epoch=[]
    for ep in range(1,EPOCHS+1):
        rs=[r for r in records if r['epoch']==ep]
        check(len(rs)==8 and len(set(z['ring_id'] for z in rs))==8,'Incomplete per-epoch ring diagnostics')
        per_epoch.append({'epoch':ep,'max_final_grad_inf_norm':as_real(max(z['final_grad_inf_norm'] for z in rs)),
            'median_final_grad_inf_norm':as_real(np.median([z['final_grad_inf_norm'] for z in rs])),
            'n_hit_maxiter':sum(z['hit_iteration_budget'] for z in rs),
            'n_optimizer_success':sum(z['success'] for z in rs),
            'n_local_update_accepted':sum(z['accepted'] for z in rs),
            'sum_actual_local_gain':as_real(sum(z['actual_local_gain'] for z in rs)),
            'max_iterations':max(z['nit'] for z in rs)})
    res={'status':'V57_LOCAL_OPTIMIZER_TRACE_RESEARCH_ONLY_NOT_FINAL_FREEZE',
         'source_sha256':cfg['dataset_sha256'], 'config_sha256':m.release.FREEZE_CONFIG_HASH,
         'reference_model_sha256':man['model_sha256'],
         'math_core_file_sha256':cfg['core_math_file_sha256'],
         'seed':args.seed, 'local_vi_maxiter':args.local_steps,'epochs':EPOCHS,
         'training_array_count_labeled':8,'training_array_count_masked_unlabeled':4,
         'historical_outer_rings_not_used':list(m.LOCKED),
         'quality_scoring_performed':False, 'saved_model_modified':False,
         'comparison_reference':str(args.reference),
         'max_abs_ELBO_difference_from_uninstrumented_run':maxdiff,
         'optimizer_result_status_counts':dict(Counter(str(z['status']) for z in records)),
         'optimizer_result_messages':dict(Counter(z['message'] for z in records)),
         'n_optimizer_calls':len(records),
         'n_calls_at_iteration_limit':sum(z['hit_iteration_budget'] for z in records),
         'n_optimizer_success':sum(z['success'] for z in records),
         'n_local_update_accepted':sum(z['accepted'] for z in records),
         'per_epoch':per_epoch,'per_ring_per_epoch':records,
         'limitation':'Final gradient is in redundant softmax logits; numerical stationarity of these local objectives does not prove globally optimal variational inference, predictive quality, or external generalization.'}
    with out.open('xb') as fp:fp.write((json.dumps(res,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
    print(json.dumps({'status':res['status'],'out':str(out),'n_optimizer_calls':res['n_optimizer_calls'],
                      'n_hit_maxiter':res['n_calls_at_iteration_limit'],
                      'n_optimizer_success':res['n_optimizer_success'],
                      'n_local_update_accepted':res['n_local_update_accepted'],
                      'max_abs_ELBO_diff_vs_uninstrumented':maxdiff,
                      'last_epoch_max_final_grad_inf_norm':per_epoch[-1]['max_final_grad_inf_norm']},indent=2))


def self_test():
    def quad(v):
        return float(np.sum((v-np.array([2.,-1.]))**2)),2*(v-np.array([2.,-1.]))
    o, rec=run_optimizer_audit(scipy_minimize,quad,np.array([5.,5.]),jac=True,method='L-BFGS-B',options={'maxiter':8,'ftol':1e-10,'gtol':1e-5,'maxls':15})
    check(o.success and rec['final_grad_inf_norm']<1e-5,'Quadratic optimizer tracer failed')
    check(rec['dimension']==2 and rec['objective_reduction']>0,'Incorrect tracer metrics')
    cfg=m.load_config()
    check(cfg['K']==2 and abs(cfg['C']-0.2)<1e-12 and cfg['local_steps']==8,'Unexpected frozen V52 reference config')
    print('V57 LOCAL OPTIMIZER TRACE SELF-TEST: synthetic scipy trace, frozen V52 code/config PASS; NO DM/TRAINING')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('self-test')
    r=sub.add_parser('run')
    r.add_argument('--data',required=True);r.add_argument('--model-dir',required=True)
    r.add_argument('--seed',type=int,choices=(VALID_SEED,),default=VALID_SEED)
    r.add_argument('--local-steps',type=int,choices=VALID_STEPS,required=True)
    r.add_argument('--reference',required=True,help='Previous V55 or V56 30-epoch history JSON of same seed/budget')
    r.add_argument('--out',required=True)
    args=ap.parse_args()
    if args.cmd=='self-test': self_test()
    else:run(args)
if __name__=='__main__':main()

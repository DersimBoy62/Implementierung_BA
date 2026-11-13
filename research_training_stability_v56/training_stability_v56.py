#!/usr/bin/env python3
"""Research-only training stability: seeds, local VI budgets, fixed 30 epochs.

DOES NOT modify registered V5.2 artifacts or run quality/outer-test scoring.
Uses exact 10 SHA-locked frozen core modules; writes result JSON once.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys
from datetime import datetime, timezone
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parent
V55 = PROJECT / 'research_convergence_v55'
if str(V55) not in sys.path: sys.path.insert(0, str(V55))
import convergence_probe_v55 as probe
m = probe.m

SEEDS = (20261111, 20261112, 20261113)
STEPS = (8, 16)
EPOCHS = 30


def require(v,msg):
    if not v: raise ValueError(msg)


def report_metrics(history):
    v = np.array([ep['elbo'] for ep in history], dtype=float)
    q = np.array([ep['state_mass'] for ep in history], dtype=float)
    require(v.shape==(EPOCHS,) and np.isfinite(v).all(), 'Unexpected / nonfinite ELBO history')
    require(q.shape==(EPOCHS,2) and np.isfinite(q).all(), 'Unexpected state-mass history')
    require(np.max(abs(q.sum(axis=1)-1))<1e-7,'State masses not normalized')
    dif = np.diff(v)
    rel = abs(dif)/np.maximum(1.0,abs(v[:-1]))
    plateau_start = None
    for i in range(len(rel)-4):
        if (rel[i:i+5] < 1e-4).all():
            plateau_start = int(i+1) # 1-based epoch at start of five transitions
            break
    return {'epochs':EPOCHS,'elbo_per_epoch':v.tolist(),'epoch_gains':dif.tolist(),
        'final_elbo':float(v[-1]),'gain_after_epoch10':float(v[-1]-v[9]),
        'last_five_rel_gains_below_1e4':bool((rel[-5:]<1e-4).all()),
        'max_last_five_abs_gain':float(max(abs(dif[-5:]))),
        'last_five_rel_gains':rel[-5:].tolist(),
        'plateau_start_epoch_descriptive_only':plateau_start,
        'state_mass_final':q[-1].tolist(), 'state_mass_sorted_final':sorted(q[-1].tolist()),
        'state_mass_max_change_last_five':float(np.max(abs(np.diff(q[-6:],axis=0)))),
        'local_gain_per_epoch':[float(ep.get('local_gain',0)) for ep in history],
        'negative_epoch_steps':int((dif<-1e-6).sum()),
        'numerical_negative_epoch_steps':int((dif<0).sum())}


def run(args):
    require(args.seed in SEEDS,'Seed must belong to the declared diagnostic set')
    require(args.local_steps in STEPS,'Local budget must be 8 or 16')
    data=pathlib.Path(args.data);model_dir=pathlib.Path(args.model_dir)
    cfg,man=probe.preflight(data, model_dir)
    out=pathlib.Path(args.out)
    require(out.parent.is_dir() and not out.exists() and not out.is_symlink(),
            'Output must be NEW and its directory already exist (write once)')
    require(out.suffix.lower()=='.json', 'JSON output required')
    require(out.resolve(strict=False) != data.resolve() and out.resolve(strict=False) != model_dir.resolve(), 'Bad output destination')
    require(probe.V52.resolve() not in out.resolve(strict=False).parents,'Cannot write inside immutable core')
    require(model_dir.resolve() not in out.resolve(strict=False).parents,'Cannot write inside original model')
    x,y,heads,rv=probe.build_training_arrays(data)
    require(len(y)==8 and len(rv)==12,'Unregistered training view')
    print('V56 RESEARCH ONLY: fixed K2/C0.2/P0/M2; seed', args.seed,'local steps', args.local_steps,
          'epochs', EPOCHS,'no held-out quality scoring',flush=True)
    model=m.dyn.train(x,y,heads,variant='M2',K=cfg['K'],C=cfg['C'],epochs=EPOCHS,
        steps=args.local_steps,seed=args.seed,physical_var=cfg['physical_var'],verbose=True)
    v=report_metrics(model['history'])
    if args.seed==SEEDS[0] and args.local_steps==STEPS[0]:
        # Strictly cross-check same baseline if somebody reruns the original condition.
        _,pack=m.release.verify_bundle(model_dir,cfg,probe.V52,strict_environment=True,load_pickle=True)
        base=np.array([z['elbo'] for z in pack['model']['history']],dtype=float)
        v['first_ten_baseline_max_abs_ELBO_diff']=float(np.max(abs(np.array(v['elbo_per_epoch'][:10])-base)))
        require(v['first_ten_baseline_max_abs_ELBO_diff']<=1e-7,'Reference epoch mismatch')
    raw={'status':'V56_EXPLORATORY_TRAIN_ONLY_STABILITY_NOT_MODEL_SELECTION',
         'seed':args.seed,'local_vi_maxiter':args.local_steps,'epochs':EPOCHS,
         'model_sha256_reference':man['model_sha256'],'source_sha256':cfg['dataset_sha256'],
         'config_sha256':m.release.FREEZE_CONFIG_HASH,'mathematics_module_hashes':cfg['core_math_file_sha256'],
         'labeled_ring_ids':list(m.LABELS),'unlabeled_ring_ids':list(m.U_ONLY),
         'outer_test_rings_not_selected_or_scored':list(m.LOCKED),
         'prediction_metrics_computed':False,'model_artifact_saved':False,
         'methodology_warning':'A flat training ELBO does not imply predictive improvement, global optimality, calibrated uncertainty, or independent generalization. Local optimizer success and gradient norms are not logged by the frozen core.',
         'metrics':v}
    with out.open('xb') as fp:fp.write((json.dumps(raw,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps({'status':raw['status'],'output':str(out),'seed':args.seed,
                      'local_steps':args.local_steps,'final_elbo':v['final_elbo'],
                      'plateau_start':v['plateau_start_epoch_descriptive_only'],
                      'last5_small':v['last_five_rel_gains_below_1e4']},indent=2))


def load_row(path):
    p=pathlib.Path(path)
    obj=json.loads(p.read_text(encoding='utf8'))
    ok={'EXPLORATORY_TRAIN_ONLY_CONVERGENCE_DIAGNOSTIC_NOT_MODEL_SELECTION','V56_EXPLORATORY_TRAIN_ONLY_STABILITY_NOT_MODEL_SELECTION'}
    require(obj.get('status') in ok,'Unexpected research report status '+str(path))
    require(obj['model_sha256_reference']=='56c3605ba3e0157125ecb7b2adaaddcb160a15c16edf22f2e28e5b232f2ae3c4','Different reference')
    require(obj['source_sha256']=='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5','Different data')
    require(obj['config_sha256']=='88f9da6aa51be71db7968a8222cb36e5e4838b8a8431988bd7c54c09bace51ab','Different config')
    require(obj['epochs_diagnostic']==EPOCHS if 'epochs_diagnostic' in obj else obj['epochs']==EPOCHS,'Incorrect budget')
    if 'epochs_diagnostic' in obj:
        metric=obj['metrics']; v=np.asarray(metric['elbo_per_epoch'])
        seed=obj['seed']; steps=8
        return {'path':str(p),'seed':seed,'local_steps':steps,'final_elbo':float(v[-1]),
                'after10':float(v[-1]-v[9]),'last5_small':metric['last5_relative_gain_below_1e_minus4'],
                'state_mass_sorted_final':sorted(metric['state_mass_final'])}
    metric=obj['metrics']
    return {'path':str(p),'seed':obj['seed'],'local_steps':obj['local_vi_maxiter'],
            'final_elbo':metric['final_elbo'],'after10':metric['gain_after_epoch10'],
            'last5_small':metric['last_five_rel_gains_below_1e4'],
            'state_mass_sorted_final':metric['state_mass_sorted_final']}


def compare(args):
    rows=[load_row(p) for p in args.reports]
    require(len(set((r['seed'],r['local_steps']) for r in rows))==len(rows),'Duplicate seed/budget combinations')
    require(all(r['seed'] in SEEDS and r['local_steps'] in STEPS for r in rows),'Outside diagnostic grid')
    result={'status':'V56_TRAIN_STABILITY_COMPARISON_EXPLORATORY','n_runs':len(rows),
            'range_final_elbo':float(max(x['final_elbo'] for x in rows)-min(x['final_elbo'] for x in rows)),
            'rows':sorted(rows,key=lambda r:(r['local_steps'],r['seed'])),
            'warning':'Descriptive training ELBO comparisons only; not predictive model ranking or test evidence.'}
    if args.out:
        out=pathlib.Path(args.out)
        require(out.parent.is_dir() and not out.exists() and not out.is_symlink(),'Comparison output must be new')
        with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


def self_test():
    require(len(SEEDS)==3 and len(STEPS)==2 and EPOCHS==30, 'Wrong fixed design')
    synthetic=[{'elbo':float(-100+10*(1-0.5**i)),'state_mass':[.25+.0001*i,.75-.0001*i],'local_gain':0.01} for i in range(30)]
    out=report_metrics(synthetic)
    require(out['epochs']==30 and np.isfinite(out['final_elbo']),'Synthetic metrics incorrect')
    require(out['last_five_rel_gains_below_1e4'],'Synthetic plateau expected')
    require(out['negative_epoch_steps']==0,'Unexpected synthetic decrease')
    m.load_config()
    print('V56 SELF-TEST: synthetic history / allowed design / fixed V52 math/config PASS; NO TRAINING / NO DM DATA')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    subs=ap.add_subparsers(dest='cmd',required=True)
    subs.add_parser('self-test')
    r=subs.add_parser('run')
    r.add_argument('--data',required=True);r.add_argument('--model-dir',required=True)
    r.add_argument('--seed',type=int,required=True,choices=SEEDS)
    r.add_argument('--local-steps',type=int,required=True,choices=STEPS)
    r.add_argument('--out',required=True)
    c=subs.add_parser('compare')
    c.add_argument('--reports',nargs='+',required=True)
    c.add_argument('--out')
    args=ap.parse_args()
    if args.cmd=='self-test':self_test()
    elif args.cmd=='run':run(args)
    elif args.cmd=='compare':compare(args)
if __name__=='__main__':main()

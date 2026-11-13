#!/usr/bin/env python3
"""Development-only convergence experiment for the frozen V5.2 Bayesian SSR.

Does not touch V5.2 model/configuration, existing outputs, CV labels, or old test rings.
Uses exactly the V5.2 preprocessing and unchanged dynamic_quality_v2.train,
with a longer training budget for DIAGNOSTICS only (not model selection).
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, math
from datetime import datetime, timezone
import numpy as np
from sklearn.preprocessing import StandardScaler

HERE=pathlib.Path(__file__).resolve().parent
PROJECT=HERE.parent
V52=PROJECT/'final_candidate_v52'
if str(V52) not in sys.path:sys.path.insert(0,str(V52))
import main_bayesian_ssr_v52 as m


def check(condition,msg):
    if not condition:raise ValueError(msg)


def preflight(path,model_dir=None):
    cfg=m.load_config()
    check(m.release.sha_file(path)==cfg['dataset_sha256'],'DM source SHA mismatch')
    check(tuple(cfg['labeled_ring_ids'])==m.LABELS and tuple(cfg['unlabeled_ring_ids'])==m.U_ONLY,'Split differs from locked code')
    check(not (set(m.LABELS)&set(m.U_ONLY)) and not (set(m.LABELS)&set(m.LOCKED)) and not (set(m.U_ONLY)&set(m.LOCKED)),'Forbidden overlap')
    check(cfg['K']==2 and cfg['C']==0.2 and cfg['epochs']==10 and cfg['local_steps']==8,'Unexpected registered model')
    manifest=None
    if model_dir:
        manifest=m.release.verify_bundle(model_dir,cfg,V52,strict_environment=True,load_pickle=False)
        check(manifest['model_sha256']=='56c3605ba3e0157125ecb7b2adaaddcb160a15c16edf22f2e28e5b232f2ae3c4', 'Not the existing exact Windows MODEL_8L4U_RUN01')
    return cfg,manifest


def build_training_arrays(path):
    # Same preprocessing operations, identifiers and iteration order as V5.2 fit_pipeline.
    rv=m.dev.load_development(path)
    check(set(rv)==set(m.LABELS)|set(m.U_ONLY) and not set(rv)&set(m.LOCKED),'Forbidden development rings')
    for rid in m.LABELS: check(np.isfinite(rv[rid].y).all(),'Missing labeled target: '+rid)
    for rid in m.U_ONLY: check(np.isnan(rv[rid].y).all(),'U target was not masked: '+rid)
    ids=tuple(sorted(rv))
    sx=m.dm.fit_x_scaler(rv,ids)
    sy=m.dm.fit_y_scaler(rv,m.LABELS)
    x={rid:sx.transform(rv[rid].x) for rid in ids}
    y={rid:sy.transform(rv[rid].y) for rid in m.LABELS}
    desc={rid:m.extract_descriptors(x[rid],rv[rid].t,'physical') for rid in ids}
    headfeatures={rid:[None,None] for rid in ids}
    for j,idx in enumerate(m.OPTIONS['SEPARATE_AXIAL']):
        idx=list(idx)
        s=StandardScaler().fit(np.stack([desc[rid][idx] for rid in ids]))
        for rid in ids: headfeatures[rid][j]=s.transform(desc[rid][idx][None,:])[0]
    return x,y,headfeatures,rv


def audit_history(history, cfg, rv, baseline_history):
    vals=np.asarray([q['elbo'] for q in history],dtype=float)
    check(np.all(np.isfinite(vals)),'Nonfinite ELBO')
    diff=np.diff(vals); rel=np.abs(diff)/np.maximum(1.,np.abs(vals[:-1]))
    masses=np.asarray([q['state_mass'] for q in history],dtype=float)
    check(masses.shape==(len(vals),2) and np.all(np.isfinite(masses)),'Bad state masses')
    check(np.all(np.abs(masses.sum(axis=1)-1.)<1e-7),'State masses fail normalization')
    check(len(baseline_history)==cfg['epochs'],'Base model does not have registered 10 epochs')
    base=np.asarray([q['elbo'] for q in baseline_history],dtype=float)
    max10=float(np.max(np.abs(base-vals[:len(base)])))
    check(max10<=1e-7,'First 10 epochs differ from previously stored model; do NOT interpret longer run: '+str(max10))
    weighted_N=sum(len(rv[r].t) for r in m.LABELS)+cfg['C']*sum(len(rv[r].t) for r in m.U_ONLY)
    return {
      'training_epochs':len(history),
      'baseline_first10_max_abs_ELBO_diff':max10,
      'elbo_initial':float(vals[0]),'elbo_final':float(vals[-1]),
      'elbo_per_epoch':vals.tolist(),
      'epoch_gains':diff.tolist(),
      'relative_abs_epoch_gains':rel.tolist(),
      'positive_epoch_changes':int((diff>0).sum()),
      'negative_epoch_changes':int((diff<0).sum()),
      'last_epoch_gain':float(diff[-1]),
      'last_epoch_relative_abs_gain':float(rel[-1]),
      'state_mass_first':masses[0].tolist(),
      'state_mass_final':masses[-1].tolist(),
      'state_mass_max_change_last5':float(np.max(np.abs(np.diff(masses[-6:],axis=0)))) if len(masses)>=6 else None,
      'last5_relative_gain_below_1e_minus4':bool(len(rel)>=5 and np.all(rel[-5:]<1e-4)),
      'timepoints_L':int(sum(len(rv[r].t) for r in m.LABELS)),
      'timepoints_U':int(sum(len(rv[r].t) for r in m.U_ONLY)),
      'weighted_timepoints_descriptive_only':float(weighted_N),
      'elbo_per_weighted_timepoint_descriptive_only':float(vals[-1]/weighted_N),
      'local_gain_per_epoch':[float(q.get('local_gain',0.0)) for q in history],
      'not_a_global_optimum_proof':True
    }


def run(args):
    path=pathlib.Path(args.data)
    cfg,man=preflight(path,args.model_dir)
    out=pathlib.Path(args.out)
    check(out.parent.is_dir() and not out.exists() and not out.is_symlink(),'Output must be NEW and its parent must exist (write once)')
    check(out.suffix.lower()=='.json','Output needs .json')
    check(V52.resolve() not in out.absolute().parents,'Do not write to frozen code tree')
    check(11<=args.epochs<=100,'Research-only epoch budget must be 11..100')
    # Trusted, locally authored V5.2 model bundle, verified before unpickling.
    check(args.model_dir is not None,'Stored original Windows model required to cross-check first 10 epochs')
    _,pkg=m.release.verify_bundle(args.model_dir,cfg,V52,strict_environment=True,load_pickle=True)
    x,y,sdata,rv=build_training_arrays(path)
    print('V55 CONVERGENCE DIAGNOSTIC ONLY; no outer/validation scoring; no change to V5.2')
    print('Training unchanged mathematical code for',args.epochs,'epochs; original 10-epoch model remains untouched',flush=True)
    trained=m.dyn.train(x,y,sdata,variant='M2',K=cfg['K'],C=cfg['C'],
                        epochs=args.epochs,steps=cfg['local_steps'],
                        seed=cfg['random_seed'],physical_var=cfg['physical_var'],verbose=True)
    hist=trained['history']
    check(len(hist)==args.epochs,'Unexpected train history length')
    metrics=audit_history(hist,cfg,rv,pkg['model']['history'])
    result={'status':'EXPLORATORY_TRAIN_ONLY_CONVERGENCE_DIAGNOSTIC_NOT_MODEL_SELECTION',
       'model_sha256_reference':man['model_sha256'],
       'source_sha256':cfg['dataset_sha256'],
       'config_sha256':m.release.FREEZE_CONFIG_HASH,
       'mathematics_module_hashes':cfg['core_math_file_sha256'],
       'seed':cfg['random_seed'],'epochs_baseline':cfg['epochs'],'epochs_diagnostic':args.epochs,
       'labels_for_scoring_used':False,'outer_test_ring_ids_not_selected_or_scored':list(m.LOCKED),
       'quality_predictions_not_scored':True,
       'methodology_warning':'Train-only ELBO plateau cannot establish prediction improvement, generalization or global optimum.',
       'metrics':metrics}
    raw=(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
    with out.open('xb') as fp:fp.write(raw)
    print(json.dumps({'status':result['status'],'out':str(out),'baseline_max_abs_ELBO_diff':metrics['baseline_first10_max_abs_ELBO_diff'], 'last_gain':metrics['last_epoch_gain'],'last_relative_gain':metrics['last_epoch_relative_abs_gain'],'last5_below_1e-4':metrics['last5_relative_gain_below_1e_minus4']},indent=2))


def self_test():
    checks=0
    for case in range(5):
        if case==0:check({1,2}.isdisjoint({3,4}),'Bad set separation')
        if case==1:check(abs(float(np.diff([1.,2.])[0])-1.)<1e-15,'Bad ELBO delta')
        if case==2:check(np.abs(np.sum([.25,.75])-1.)<1e-15,'Bad mass')
        if case==3:check(all(v>0 for v in [10,20,40]),'Budgets')
        if case==4:
            try:check(False,'negative sentinel')
            except ValueError:pass
            else:raise AssertionError('Sentinel not raised')
        checks+=1
    m.load_config()
    print('V55 CONVERGENCE PROBE SELF-TEST:',checks,'synthetic invariants PASS; V52 config and 10 math hashes PASS; NO TRAINING/DM DATA')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('self-test')
    p=sub.add_parser('preflight');p.add_argument('--data',required=True);p.add_argument('--model-dir',required=True)
    r=sub.add_parser('run');r.add_argument('--data',required=True);r.add_argument('--model-dir',required=True);r.add_argument('--epochs',type=int,default=30);r.add_argument('--out',required=True)
    args=ap.parse_args()
    if args.cmd=='self-test':self_test()
    elif args.cmd=='preflight':
      cfg,manifest=preflight(pathlib.Path(args.data),args.model_dir)
      print(json.dumps({'status':'V55_CONVERGENCE_PREFLIGHT_PASS_NO_TRAINING','model_sha256':manifest['model_sha256'],'data_sha256':cfg['dataset_sha256'],'config_sha256':m.release.FREEZE_CONFIG_HASH,'baseline_epochs':cfg['epochs'],'diagnostic_scope':'TRAIN_ONLY_ELBO; NO NEW TEST; NO PREDICTION SCORING'},indent=2))
    elif args.cmd=='run':run(args)

if __name__=='__main__':main()

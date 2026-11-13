"""V5.2 provisional Bayesian SSR entrypoint: strict freeze + atomic model release.

No automatic training. The original 10 Bayesian HMM/VI modules and V5.1 safety
checks are unchanged and SHA-locked. Previously opened outer test ring data
are neither part of development fitting nor used for any assessment here.
"""
from __future__ import annotations
import argparse, json, pathlib, sys

ROOT=pathlib.Path(__file__).resolve().parent
CORE=ROOT/'final_core'
if str(CORE) not in sys.path:sys.path.insert(0,str(CORE))

import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
import compare_dynamic_quality_cv4 as dev
import dynamic_quality_v2 as dyn
from compare_two_head_cv4 import OPTIONS
from quality_link_challenge import extract_descriptors
import input_safety_v51 as safety
import release_integrity_v52 as release

CONFIG_PATH=ROOT/'PROPOSED_MODEL_FREEZE.json'
FEATURE_COLUMNS=release.FEATURES
LABELS=release.LABELED
U_ONLY=release.U_ONLY
LOCKED=release.LOCKED


def load_config():
    cfg=release.validate_registered_config(CONFIG_PATH,CORE,ROOT)
    if tuple(dm.SETS['V1_Kern_6'])!=FEATURE_COLUMNS:
        raise ValueError('Unexpected live code feature mapping')
    index_maps=OPTIONS['SEPARATE_AXIAL']
    if tuple(tuple(x) for x in index_maps)!=((0,1,3,5,6,10),(0,2,4,7,11)):
        raise ValueError('Unexpected live descriptor implementation')
    if safety.POLICY_VERSION!='DM-V51-PROCESS-ONLY-SAFETY-2026-11-12-v1':
        raise ValueError('Unexpected safety policy version')
    return cfg


def fit_pipeline(data_path,outdir):
    """Explicit training only; no validation score, never re-evaluate old test."""
    cfg=load_config()
    dest=release.check_output_destination(ROOT,outdir)  # fail before training
    if release.sha_file(data_path)!=cfg['dataset_sha256']:
        raise ValueError('Unknown/training data file: SHA mismatch; refusing to deserialize')
    rv=dev.load_development(data_path) # only the registered 8L+4U development view
    if set(rv)!=set(LABELS)|set(U_ONLY) or set(rv)&set(LOCKED):
        raise ValueError('Forbidden development ring allocation')
    for rid in LABELS:
        if not np.isfinite(rv[rid].y).all():raise ValueError('Missing labeled target '+rid)
    for rid in U_ONLY:
        if not np.isnan(rv[rid].y).all():raise ValueError('U quality not masked '+rid)
    ids=tuple(sorted(rv))
    sx=dm.fit_x_scaler(rv,ids)
    sy=dm.fit_y_scaler(rv,LABELS)
    x={rid:sx.transform(rv[rid].x) for rid in ids}
    y={rid:sy.transform(rv[rid].y) for rid in LABELS}
    desc={rid:extract_descriptors(x[rid],rv[rid].t,'physical') for rid in ids}
    head_scalers=[];headfeatures={rid:[None,None] for rid in ids}
    for j,idx in enumerate(OPTIONS['SEPARATE_AXIAL']):
        scaler=StandardScaler().fit(np.stack([desc[rid][list(idx)] for rid in ids]))
        head_scalers.append(scaler)
        for rid in ids:headfeatures[rid][j]=scaler.transform(desc[rid][list(idx)][None,:])[0]
    model=dyn.train(x,y,headfeatures,variant='M2',K=cfg['K'],C=cfg['C'],
                    epochs=cfg['epochs'],steps=cfg['local_steps'],seed=cfg['random_seed'],
                    physical_var=cfg['physical_var'])
    policy=safety.make_training_policy(rv,LABELS,U_ONLY,LOCKED)
    package={'model':model,'x_scaler':sx,'y_scaler':sy,'input_safety_policy':policy,
             'physical_scalers':head_scalers,'config_sha256':release.FREEZE_CONFIG_HASH,
             'source_data_sha256':cfg['dataset_sha256'],'train_ids_L':LABELS,
             'train_ids_U':U_ONLY,'core_sha256':cfg['core_math_file_sha256'],
             'v52_release':release.RELEASE}
    return release.store_model_atomic(dest,package,model['history'],cfg,ROOT)


def predict_one(model_dir,ring_id,input_csv,output_json,completion_record):
    """Trust only locally produced V5.2 model bundles with exact environment."""
    cfg=load_config()
    ring_id=safety.normalize_and_check_new_ring_id(ring_id) # BEFORE model access
    output_json=pathlib.Path(output_json)
    if output_json.exists() or output_json.is_symlink():raise FileExistsError('Prediction output must be new')
    if not completion_record or not pathlib.Path(completion_record).is_file():
        raise ValueError('Independent ring-completion record required')
    man,package=release.verify_bundle(model_dir,cfg,ROOT,strict_environment=True,load_pickle=True)
    t,rawx,input_report=safety.validate_raw_logger(input_csv,package['input_safety_policy'])
    closeout=safety.verify_completion_attestation(completion_record,ring_id,input_csv,input_report)
    sx=package['x_scaler'];sy=package['y_scaler']
    normx=sx.transform(rawx)
    physical=extract_descriptors(normx,t,'physical')
    groups=OPTIONS['SEPARATE_AXIAL']
    headx=[package['physical_scalers'][j].transform(physical[list(idx)][None,:])[0]
           for j,idx in enumerate(groups)]
    try:mu,sd=dyn.predict(normx,headx,package['model'])
    except (ArithmeticError,FloatingPointError) as exc:
        raise ValueError('HMM numerical inference failed; cannot export quality prediction') from exc
    pred=mu*sy.scale+sy.mean;uncertainty=sd*sy.scale
    if not(np.isfinite(pred).all() and np.isfinite(uncertainty).all() and (uncertainty>0).all()):
        raise ArithmeticError('Nonfinite prediction or uncertainty')
    result={'ring_id':ring_id,'model_config_id':cfg['freeze_id'],
            'model_release':release.RELEASE,
            'prediction_mm':dict(zip(('outer_roundness','height_runout'),pred.tolist())),
            'predictive_sd_mm':dict(zip(('outer_roundness','height_runout'),uncertainty.tolist())),
            'input_csv_sha256':release.sha_file(input_csv),
            'model_artifact_sha256':man['model_sha256'],
            'model_manifest_sha256':release.sha_file(pathlib.Path(model_dir)/'TRAIN_MANIFEST.json'),
            'environment_sha256':man['environment_sha256'],
            'input_policy_sha256':man['input_safety_policy_sha256'],
            'completion_record':closeout,'input_safety_check':input_report,
            'full_process_window':'F2','n_process_samples':len(t),
            'warning':'Full-run end-of-process quality forecast; NOT online control; SD NOT a calibrated 95% interval.'}
    # Write-once; each report has a checksum binding its model and manifest.
    with output_json.open('x',encoding='utf-8') as fh:
        json.dump(result,fh,indent=2,ensure_ascii=False,allow_nan=False);fh.write('\n')
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    sub=a.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train',help='PROVISIONAL development-only fit; NOT the official final run')
    t.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl');t.add_argument('--out',required=True)
    p=sub.add_parser('predict');p.add_argument('--model-dir',required=True);p.add_argument('--ring-id',required=True)
    p.add_argument('--csv',required=True);p.add_argument('--completion-record',required=True);p.add_argument('--out',required=True)
    c=sub.add_parser('verify-config')
    v=sub.add_parser('verify-model');v.add_argument('--model-dir',required=True)
    prep=sub.add_parser('prepare-policy');prep.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl');prep.add_argument('--out',required=True)
    check=sub.add_parser('verify-input');check.add_argument('--policy',required=True);check.add_argument('--csv',required=True)
    args=a.parse_args()
    cfg=load_config()
    if args.command=='verify-config':
        print(json.dumps({'status':'V52_CONFIG_AND_ALL_10_MATH_HASHES_VERIFIED',
                          'freeze_status':cfg['freeze_status'],'release':release.RELEASE,
                          'environment':release.environment_snapshot()},indent=2,ensure_ascii=False))
    elif args.command=='verify-model':
        man=release.verify_bundle(args.model_dir,cfg,ROOT,strict_environment=True,load_pickle=False)
        print(json.dumps({'status':'MODEL_HASH_MANIFEST_CODE_ENV_VERIFIED_NO_UNPICKLING',
                          'model_sha256':man['model_sha256'],'release':man['release']},indent=2))
    elif args.command=='train':
        man=fit_pipeline(args.data,args.out)
        print('DEVELOPMENT-ONLY MODEL SAVED ATOMICALLY:',args.out,man['model_sha256'])
    elif args.command=='predict':
        result=predict_one(args.model_dir,args.ring_id,args.csv,args.out,args.completion_record)
        print(json.dumps(result,indent=2,ensure_ascii=False))
    elif args.command=='prepare-policy':
        if release.sha_file(args.data)!=release.SOURCE_HASH:raise ValueError('Unknown dataset SHA')
        rv=dev.load_development(args.data)
        policy=safety.make_training_policy(rv,LABELS,U_ONLY,LOCKED)
        with pathlib.Path(args.out).open('x',encoding='utf-8') as fh:
            json.dump(policy,fh,indent=2,ensure_ascii=False,allow_nan=False);fh.write('\n')
        print('Development-only input policy generated, no training',args.out)
    elif args.command=='verify-input':
        policy=json.loads(pathlib.Path(args.policy).read_text(encoding='utf-8'))
        t,x,report=safety.validate_raw_logger(args.csv,policy)
        print(json.dumps({'status':'V51_PROCESS_INPUT_QA_PASS','report':report},indent=2,ensure_ascii=False))

if __name__=='__main__':main()

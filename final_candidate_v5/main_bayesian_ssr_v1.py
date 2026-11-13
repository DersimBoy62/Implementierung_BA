"""Provisional development-only production-style training and prediction wrapper.

This wrapper retains all original mathematical V3 kernels byte-for-byte in
final_core/. It NEVER evaluates the previously opened outer test rings.
All validation statements belong to the research scripts, not this final fit.

Only load trusted, locally generated pickles with verified SHA256.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, pathlib, pickle, sys
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent
CORE = ROOT/'final_core'
if str(CORE) not in sys.path: sys.path.insert(0,str(CORE))

import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
import compare_dynamic_quality_cv4 as dev
import dynamic_quality_v2 as dyn
from compare_two_head_cv4 import OPTIONS
from quality_link_challenge import extract_descriptors, DATA_SHA

CONFIG_PATH = ROOT/'PROPOSED_MODEL_FREEZE.json'
FEATURE_COLUMNS = tuple(dm.SETS['V1_Kern_6'])
LABELS = ('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18')
U_ONLY = ('DM01','DM02','DM10','DM14')
LOCKED = ('DM07','DM08','DM11','DM15')


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _write_new_json(path, content):
    with pathlib.Path(path).open('x', encoding='utf-8') as f:
        json.dump(content, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def load_config():
    cfg=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if cfg.get('freeze_status') != 'PROVISIONAL_USER_APPROVAL_PENDING':
        raise ValueError('This candidate wrapper is only for the registered provisional config')
    if (cfg['K'],cfg['C'],cfg['physical_var'],cfg['epochs'],cfg['local_steps'],cfg['random_seed']) != (2,0.2,0.38526108747593474,10,8,20261111):
        raise ValueError('Unregistered candidate parameters')
    if (tuple(cfg['labeled_ring_ids']),tuple(cfg['unlabeled_ring_ids']),tuple(cfg['never_use_for_selection_ids'])) != (LABELS,U_ONLY,LOCKED):
        raise ValueError('Development split changed')
    if cfg['dataset_sha256'] != DATA_SHA or tuple(cfg['input_columns'])!=FEATURE_COLUMNS:
        raise ValueError('Data or feature specification changed')
    if cfg['physical_descriptor_indices'] != {'outside':[0,1,3,5,6,10],'height':[0,2,4,7,11]}:
        raise ValueError('Physical descriptor mapping changed')
    for fname,expected in cfg['core_math_file_sha256'].items():
        if sha(CORE/fname)!=expected:
            raise RuntimeError('Original mathematical kernel has changed: '+fname)
    return cfg


def fit_pipeline(data_path: str | pathlib.Path, outdir: str | pathlib.Path):
    cfg=load_config()
    out=pathlib.Path(outdir).resolve()
    if out == ROOT.resolve() or out == CORE.resolve(): raise ValueError('Output must not overwrite code')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output not empty; refusing to overwrite')
    if sha(data_path)!=cfg['dataset_sha256']:
        raise ValueError('Pickle input digest not equal to authorized dataset')
    # Loader creates an L+U development view; labels of U are NaN; locked rings are discarded.
    rv=dev.load_development(data_path)
    assert tuple(sorted(r for r in rv if np.isfinite(rv[r].y).all())) == LABELS
    assert tuple(sorted(r for r in rv if np.isnan(rv[r].y).all())) == U_ONLY
    assert not (set(rv)&set(LOCKED))
    ids=tuple(sorted(rv))
    sx=dm.fit_x_scaler(rv,ids)
    sy=dm.fit_y_scaler(rv,LABELS)
    x={rid:sx.transform(rv[rid].x) for rid in ids}
    y={rid:sy.transform(rv[rid].y) for rid in LABELS}
    desc={rid:extract_descriptors(x[rid],rv[rid].t,'physical') for rid in ids}
    head_scalers=[]
    headfeatures={rid:[None,None] for rid in ids}
    for j,idx in enumerate(OPTIONS['SEPARATE_AXIAL']):
        scaler=StandardScaler().fit(np.stack([desc[rid][list(idx)] for rid in ids]))
        head_scalers.append(scaler)
        for rid in ids:
            headfeatures[rid][j]=scaler.transform(desc[rid][list(idx)][None,:])[0]
    model=dyn.train(x,y,headfeatures,variant='M2',K=cfg['K'],C=cfg['C'],
                    epochs=cfg['epochs'],steps=cfg['local_steps'],seed=cfg['random_seed'],
                    physical_var=cfg['physical_var'])
    if not np.isfinite(model['history'][-1]['elbo']):
        raise ArithmeticError('Nonfinite training objective')
    package={'model':model,'x_scaler':sx,'y_scaler':sy,
             'physical_scalers':head_scalers,'config_sha256':sha(CONFIG_PATH),
             'source_data_sha256':cfg['dataset_sha256'], 'train_ids_L':LABELS,
             'train_ids_U':U_ONLY, 'core_sha256':cfg['core_math_file_sha256']}
    out.mkdir(parents=True, exist_ok=True)
    model_path=out/'TRAINED_MODEL.pkl'
    with model_path.open('xb') as f:pickle.dump(package,f,protocol=pickle.HIGHEST_PROTOCOL)
    manifest={'protocol':cfg['freeze_id'],'status':'PROVISIONAL MODEL FIT ON REUSED DEVELOPMENT DATA; NO NEW TEST',
              'trained_at_utc':datetime.now(timezone.utc).isoformat(),
              'input_pickle_sha256':cfg['dataset_sha256'],'config_sha256':sha(CONFIG_PATH),
              'entrypoint_sha256':sha(__file__),
              'model_pickle_sha256':sha(model_path),
              'original_core_math_hashes':cfg['core_math_file_sha256'],
              'train_labeled_ids':list(LABELS),'train_U_process_only_ids':list(U_ONLY),
              'locked_old_test_ids_not_used':list(LOCKED),
              'final_elbo':float(model['history'][-1]['elbo']),
              'final_state_mass':model['history'][-1]['state_mass'],
              'n_epochs':len(model['history']),
              'warning':'Training RMSE or outputs from these fitted rings are not independent validation.'}
    _write_new_json(out/'TRAIN_MANIFEST.json', manifest)
    return manifest


def extract_new_ring_csv(path):
    """Accept process-only logger CSV with F2 gating and 6 feature channels.

    Requires raw acquisition order, 0.1-second (or other positive) timestamps,
    and the three process-status columns. Never reads finished quality labels.
    """
    path=pathlib.Path(path)
    required=('time',*FEATURE_COLUMNS,'phase804','LaserOK','MessEin')
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or set(reader.fieldnames)!=set(required) or len(set(reader.fieldnames))!=len(reader.fieldnames):
            raise ValueError('Only the declared process columns are permitted (no metrology/target columns); expected '+', '.join(required))
        if any(name.startswith('ovality.') or name.startswith('planarity.') for name in reader.fieldnames):
            raise ValueError('Post-process quality columns must not be present in prediction input')
        rows=[]
        for row in reader:
            rows.append([float(row[k]) for k in required])
    if not rows:raise ValueError('Empty logger')
    vals=np.asarray(rows,float)
    if not np.isfinite(vals).all() or not np.all(np.diff(vals[:,0])>0):
        raise ValueError('Invalid timestamps or process values')
    gate=np.isin(vals[:,-3],[1.,2.]) & (vals[:,-2]==1.) & (vals[:,-1]==1.)
    idx=np.flatnonzero(gate)
    if len(idx)<6 or not np.all(np.diff(idx)==1):
        raise ValueError('F2 window must be contiguous and contain at least six timepoints')
    return vals[idx,0], vals[idx,1:1+len(FEATURE_COLUMNS)]


def _load_owned_model(directory, cfg):
    directory=pathlib.Path(directory)
    man=json.loads((directory/'TRAIN_MANIFEST.json').read_text(encoding='utf-8'))
    mpath=directory/'TRAINED_MODEL.pkl'
    if man.get('config_sha256')!=sha(CONFIG_PATH) or man.get('model_pickle_sha256')!=sha(mpath) or man.get('entrypoint_sha256')!=sha(__file__):
        raise RuntimeError('Trained model/config hash mismatch: refusing to unpickle')
    if man.get('input_pickle_sha256')!=cfg['dataset_sha256'] or tuple(man.get('locked_old_test_ids_not_used',[]))!=LOCKED:
        raise RuntimeError('Model provenance mismatch')
    # Only unpickle a model artifact WE created ourselves, after digest check.
    with mpath.open('rb') as f:package=pickle.load(f)
    if package.get('config_sha256')!=sha(CONFIG_PATH) or tuple(package['train_ids_L'])!=LABELS:
        raise RuntimeError('Package config/train IDs mismatch')
    return package


def predict_one(model_dir, ring_id, input_csv, output_json):
    cfg=load_config()
    ring_id=str(ring_id).strip()
    if not ring_id or ring_id in set(LABELS)|set(U_ONLY)|set(LOCKED):
        raise ValueError('For this command, use a genuinely NEW ring ID, not historical DM development/test rings')
    if pathlib.Path(output_json).exists(): raise FileExistsError('Prediction report must be new')
    package=_load_owned_model(model_dir,cfg)
    t,rawx=extract_new_ring_csv(input_csv)
    sx=package['x_scaler'];sy=package['y_scaler']
    normx=sx.transform(rawx)
    physical=extract_descriptors(normx,t,'physical')
    groups=OPTIONS['SEPARATE_AXIAL']
    headx=[package['physical_scalers'][j].transform(physical[list(ix)][None,:])[0]
           for j,ix in enumerate(groups)]
    try:
        mu,sd=dyn.predict(normx,headx,package['model'])
    except (ArithmeticError, FloatingPointError) as exc:
        raise ValueError('HMM inference did not converge numerically for these process signals; no prediction exported. Check sensor units, scaling, and domain shift.') from exc
    pred=mu*sy.scale+sy.mean; uncertainty=sd*sy.scale
    if not (np.isfinite(pred).all() and np.isfinite(uncertainty).all() and (uncertainty>0).all()):
        raise ArithmeticError('Non-finite prediction or uncertainty')
    result={'ring_id':ring_id,'model_config_id':cfg['freeze_id'],
            'prediction_mm':dict(zip(('outer_roundness','height_runout'),pred.tolist())),
            'predictive_sd_mm':dict(zip(('outer_roundness','height_runout'),uncertainty.tolist())),
            'input_csv_sha256':sha(input_csv),'full_process_window': 'F2',
            'n_process_samples':len(t),
            'warning':'Full-run end-of-process quality forecast, NOT validated online process control; SD not calibrated 95% interval.'}
    _write_new_json(output_json,result)
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    sub=a.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train'); t.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl');t.add_argument('--out',required=True)
    p=sub.add_parser('predict');p.add_argument('--model-dir',required=True);p.add_argument('--ring-id',required=True);p.add_argument('--csv',required=True);p.add_argument('--out',required=True)
    args=a.parse_args()
    if args.command=='train':
        man=fit_pipeline(args.data,args.out)
        print('PROVISIONAL TRAINED MODEL SAVED; last ELBO',round(man['final_elbo'],4))
        print('TRAIN LABELS ONLY',man['train_labeled_ids'],'U QUALITY MASKED',man['train_U_process_only_ids'])
        print('PREVIOUS TEST EXCLUDED',man['locked_old_test_ids_not_used'])
    else:
        r=predict_one(args.model_dir,args.ring_id,args.csv,args.out)
        print(json.dumps(r,indent=2,ensure_ascii=False))

if __name__=='__main__':main()

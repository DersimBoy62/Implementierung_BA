#!/usr/bin/env python3
"""V5.4 PRE-FREEZE QA overlay; NOT a final model release.

Commands:
  self-test         synthetic fail-closed audit tests; no dataset
  audit-v53         read-only independent check of four recorded CV4 JSONs
  synthetic-replay  two 10-epoch synthetic trainings + true model pickle replay

Never reads DM training pickle, never scores historical outer-test rings, never
modifies V5.2/V5.3 files, and never changes hyperparameters based on outcomes.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, pickle, sys, tempfile
import numpy as np

HERE=pathlib.Path(__file__).resolve().parent
PROJECT=HERE.parent
V52=PROJECT/'final_candidate_v52'
if str(V52) not in sys.path: sys.path.insert(0,str(V52))
if str(V52/'final_core') not in sys.path: sys.path.insert(0,str(V52/'final_core'))

import main_bayesian_ssr_v52 as main
import release_integrity_v52 as integrity
import dynamic_quality_v2 as dyn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

EXPECTED_PROTOCOL='V53_PAIRED_LOCAL_0_VS_8_DEV_CV4_2026-11-12_v1'
EXPECTED_L=integrity.LABELED
EXPECTED_U=integrity.U_ONLY
EXPECTED_LOCKED=integrity.LOCKED
FOLDS={
  1:('DM04','DM05'),
  2:('DM06','DM09'),
  3:('DM12','DM16'),
  4:('DM17','DM18'),
}
TARGETS=('outside_ovality_mm','height_runout_mm')


def must(ok,msg):
    if not ok:raise ValueError(msg)

def sha_file(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def finite(x):return np.all(np.isfinite(np.asarray(x,dtype=float)))

def close(a,b,atol=2e-11):return np.allclose(a,b,rtol=0,atol=atol)

def audit_documents(folder):
    cfg=main.load_config()  # validates all math hashes and config prior to reading results
    folder=pathlib.Path(folder)
    must(folder.is_dir(),'Four-fold results directory missing')
    docs=[];shas={}
    for fold in FOLDS:
        file=folder/f'FOLD_{fold}_PAIRED_STEPS_0_8.json'
        must(file.is_file() and not file.is_symlink(),f'Missing or symlinked original fold {fold}')
        docs.append(json.loads(file.read_text(encoding='utf-8')))
        shas[file.name]=sha_file(file)
    sp=folder/'V53_PAIRED_DEVELOPMENT_CV4_SUMMARY.json'
    must(sp.is_file() and not sp.is_symlink(),'Original V53 summary missing or symlinked')
    summ=json.loads(sp.read_text(encoding='utf-8'))
    shas[sp.name]=sha_file(sp)
    must(summ.get('protocol')==EXPECTED_PROTOCOL,'Summary protocol mismatch')
    must(summ.get('source_sha256')==integrity.SOURCE_HASH,'Summary dataset hash mismatch')
    must(summ.get('config_sha256')==integrity.FREEZE_CONFIG_HASH,'Summary config hash mismatch')
    must(summ.get('outer_test_ids_NEVER_USED')==list(EXPECTED_LOCKED),'Summary historical-test register mismatch')
    must(summ.get('target_order')==list(TARGETS),'Summary target order mismatch')
    must(summ.get('n_unique_dev_validation_rings')==8,'Summary unique validation count mismatch')
    err={0:{},8:{}}; ring_records={}; convergence=[];feedback=[]
    for fold,d in zip(FOLDS,docs):
        val=FOLDS[fold]
        tr=tuple(r for r in EXPECTED_L if r not in val)
        all_fit=set(tr)|set(EXPECTED_U)
        must(d.get('fold')==fold and d.get('protocol')==EXPECTED_PROTOCOL,f'Fold {fold} protocol mismatch')
        must(d.get('source_sha256')==integrity.SOURCE_HASH,f'Fold {fold} dataset SHA mismatch')
        must(d.get('configuration_sha256')==integrity.FREEZE_CONFIG_HASH,f'Fold {fold} config SHA mismatch')
        must(tuple(d['train_L'])==tr and tuple(d['train_U_process_only'])==EXPECTED_U and tuple(d['validation'])==val,f'Fold {fold} split mismatch')
        fit=d.get('scaler_fit_ids',{})
        must(set(fit.get('X',()))==all_fit and len(fit['X'])==len(all_fit),f'Fold {fold} X leakage/duplication')
        must(set(fit.get('physical_per_head',()))==all_fit and len(fit['physical_per_head'])==len(all_fit),f'Fold {fold} descriptor leakage/duplication')
        must(set(fit.get('y',()))==set(tr) and len(fit['y'])==len(tr),f'Fold {fold} target leakage/duplication')
        must(set(d.get('arms',{}))=={'steps_0','steps_8'},f'Fold {fold} arm mismatch')
        must([r['ring_id'] for r in d['paired_validation']]==list(val),f'Fold {fold} validation rows/order mismatch')
        for steps in (0,8):
            arm=d['arms'][f'steps_{steps}']; h=arm['history']
            must(len(h)==cfg['epochs'],'Wrong epoch count')
            must([row['epoch'] for row in h]==list(range(1,cfg['epochs']+1)),'Wrong epoch order')
            es=np.asarray([row['elbo'] for row in h],dtype=float)
            gains=np.asarray([row['local_gain'] for row in h],dtype=float)
            must(finite(es) and finite(gains) and np.all(gains>=-1e-12),'Non-finite history or negative local gain')
            must(close(arm['final_ELBO'],es[-1]),'Final ELBO inconsistent')
            for row in h:
                states=np.asarray(row['state_mass'],float)
                must(states.shape==(cfg['K'],) and finite(states) and (states>=-1e-9).all() and close(states.sum(),1,atol=2e-8),'Invalid state mass')
            # Scientific finding: existing field is last epoch, not a sum of epochs.
            must(close(arm['final_train_local_gain_sum'],gains[-1]),'Original local_gain field not equal final epoch')
            deltas=np.diff(es)
            must(finite(deltas),'Non-finite ELBO step')
            convergence.append({
                'fold':fold,'steps':steps,'elbo_first':float(es[0]),'elbo_final':float(es[-1]),
                'elbo_last_increment':float(deltas[-1]),'elbo_smallest_increment':float(deltas.min()),
                'total_elbo_increase':float(es[-1]-es[0]),
                'last_increment_fraction_of_total':float(deltas[-1]/(es[-1]-es[0])) if es[-1]!=es[0] else None,
                'last_epoch_local_gain':float(gains[-1]),'sum_over_epoch_local_gain':float(gains.sum()),
                'increasing_through_last_epoch':bool(deltas[-1]>0),
                'any_decrease':bool((deltas<0).any())})
            for rid in val:
                pred=arm['validation_predictions_without_labels'][rid]
                must(len(pred['prediction_mm'])==2 and finite(pred['prediction_mm']),'Invalid prediction')
                must(len(pred['predictive_sd_mm'])==2 and finite(pred['predictive_sd_mm']) and min(pred['predictive_sd_mm'])>0,'Invalid predictive sd')
            if steps==8:
                fb=arm.get('fixed_global_local_feedback_train_L',[])
                must(len(fb)==len(tr) and set(r['ring_id'] for r in fb)==set(tr),'Wrong local posterior shift ring set')
                for r in fb:
                    must(r['optimizer_iterations']<=8 and r['optimizer_iterations']>=0,'Local optimizer exceeded specified steps')
                    must(finite([r['mean_abs_gamma_shift'],r['max_abs_gamma_shift'],r['max_abs_M2_first_moment_shift'],r['max_abs_M2_second_moment_shift'],r['local_gain_fixed_globals']]),'Nonfinite feedback')
                    must(r['max_abs_gamma_shift']>=r['mean_abs_gamma_shift']>=0,'Shift bounds invalid')
                    feedback.append({'fold':fold,'ring_id':r['ring_id'],'accepted':bool(r['optimizer_accepted']),
                                     'max_abs_gamma_shift':r['max_abs_gamma_shift'],'local_gain':r['local_gain_fixed_globals']})
        for r in d['paired_validation']:
            rid=r['ring_id'];must(rid not in ring_records,f'Duplicate validation result {rid}')
            truth=np.asarray(r['truth_mm'],float)
            must(truth.shape==(2,) and finite(truth),'Invalid truth shape')
            for s in (0,8):
                p=np.asarray(d['arms'][f'steps_{s}']['validation_predictions_without_labels'][rid]['prediction_mm'],float)
                e=np.asarray(r[f'error_steps_{s}_mm'],float)
                ae=np.asarray(r[f'abs_error_steps_{s}_mm'],float)
                must(close(p-truth,e) and close(np.abs(e),ae),f'Ring {rid} error mismatch')
                err[s][rid]=e
            must(close(np.abs(err[8][rid])-np.abs(err[0][rid]),r['delta_abs_error_8_minus_0_mm']),f'Ring {rid} paired delta mismatch')
            ring_records[rid]={'fold':fold,'truth_mm':truth.tolist(),
                               'delta_abs_error_8_minus_0_mm':r['delta_abs_error_8_minus_0_mm']}
    must(set(ring_records)==set(EXPECTED_L),'Missing or unexpected validation rings')
    def aggregate(s):
        z=np.stack([err[s][rid] for rid in EXPECTED_L]);return np.sqrt(np.mean(z*z,axis=0)),np.mean(np.abs(z),axis=0),z
    r0,a0,e0=aggregate(0);r8,a8,e8=aggregate(8)
    for key,val in {'RMSE_steps0_mm':r0,'RMSE_steps8_mm':r8,'MAE_steps0_mm':a0,'MAE_steps8_mm':a8,
                    'delta_RMSE_8_minus_0_mm':r8-r0,'delta_MAE_8_minus_0_mm':a8-a0,
                    'count_abs_error_better_with_8':(np.abs(e8)<np.abs(e0)).sum(axis=0)}.items():
        must(close(summ[key],val),f'CV4 summary metric mismatch: {key}')
    must(set(r['ring_id'] for r in summ['ringwise'])==set(EXPECTED_L),'Summary ring set mismatch')
    for r in summ['ringwise']:
        must(close(r['delta_abs_error_8_minus_0_mm'],ring_records[r['ring_id']]['delta_abs_error_8_minus_0_mm']),'Summary ringwise discrepancy')
    return {
      'status':'V54_READONLY_V53_INTEGRITY_AUDIT_PASS_NOT_MODEL_FREEZE',
      'science_status':'EXPLORATORY_REUSED_8_DEVELOPMENT_LABELS_NO_INDEPENDENT_GENERALIZATION',
      'source_sha256':integrity.SOURCE_HASH,'config_sha256':integrity.FREEZE_CONFIG_HASH,
      'original_result_sha256':shas,
      'n_unique_development_rings':len(ring_records),'n_paired_model_fits':8,'n_local_checks':len(feedback),
      'n_local_optimizer_accepted':sum(r['accepted'] for r in feedback),
      'maximum_abs_gamma_shift':max(r['max_abs_gamma_shift'] for r in feedback),
      'RMSE_steps0_mm':r0.tolist(),'RMSE_steps8_mm':r8.tolist(),
      'RMSE_delta_8_minus_0_mm':(r8-r0).tolist(),
      'MAE_steps0_mm':a0.tolist(),'MAE_steps8_mm':a8.tolist(),
      'MAE_delta_8_minus_0_mm':(a8-a0).tolist(),
      'ringwise':ring_records,'elbo_diagnostics':convergence,
      'known_original_reporting_bug':'final_train_local_gain_sum names the final-epoch local_gain, not sum of all epochs',
      'model_convergence':'NOT_ESTABLISHED_BY_10_EPOCHS',
      'model_choice':'UNCHANGED_NO_RETUNING',
      'historical_test':'DM07_DM08_DM11_DM15_NOT_USED_IN_THIS_QA'
    }


def synthetic_data():
    # No DM ID, process values, original labels, or pickle are opened.
    rng=np.random.default_rng(20261112)
    N=6; xs={}; ys={}; head={}
    for i in range(N):
        t=np.linspace(0,1,12+i)
        raw=np.stack([0.4+0.2*t+0.1*i+rng.normal(0,.04,len(t)),
                      -0.3+0.1*t+0.05*i+rng.normal(0,.03,len(t)),
                      0.4+0.15*t+rng.normal(0,.02,len(t)),
                      0.2-0.05*t+rng.normal(0,.015,len(t)),
                      -0.1+0.12*t+rng.normal(0,.025,len(t)),
                      0.3*t+0.04*i+rng.normal(0,.02,len(t))],axis=1)
        xs[f'SYN{i:02d}']=raw
        head[f'SYN{i:02d}']=[np.array([float(i),.3,-.4,.7,-.2,.5]),np.array([float(i)*.2,.1,.5,-.3,.6])]
        if i<4:ys[f'SYN{i:02d}']=np.array([.6+.13*i,5.1+.35*i])
    return xs,ys,head


def synthetic_replay():
    cfg=main.load_config()
    xs,ys,head=synthetic_data()
    def fit():
        with threadpool_limits(limits=1):
            return dyn.train(xs,ys,head,variant='M2',K=cfg['K'],C=cfg['C'],
                             epochs=cfg['epochs'],seed=cfg['random_seed'],
                             steps=cfg['local_steps'],physical_var=cfg['physical_var'],verbose=False)
    m1=fit();m2=fit()
    def check_predict(m):
        with threadpool_limits(limits=1):
            return dyn.predict(xs['SYN04'],head['SYN04'],m)
    p1,sd1=check_predict(m1)
    p2,sd2=check_predict(m2)
    h1=np.asarray([r['elbo'] for r in m1['history']],float)
    h2=np.asarray([r['elbo'] for r in m2['history']],float)
    must(finite(h1) and finite(h2) and finite(p1) and finite(sd1) and (sd1>0).all(),'Nonfinite model/prediction')
    must(np.allclose(h1,h2,atol=1e-10,rtol=1e-12),'Same-process synthetic two-fit ELBO differs')
    must(np.allclose(p1,p2,atol=1e-10,rtol=1e-12) and np.allclose(sd1,sd2,atol=1e-10,rtol=1e-12),'Same-process synthetic two-fit prediction differs')
    raw=pickle.dumps(m1,protocol=pickle.HIGHEST_PROTOCOL)
    with tempfile.TemporaryDirectory() as tmp:
        p=pathlib.Path(tmp)/'SYNTHETIC_ONLY_NOT_A_RELEASE.pkl'
        p.write_bytes(raw)
        h=sha_file(p)
        must(h==hashlib.sha256(raw).hexdigest(),'Synthetic binary SHA discrepancy')
        r=pickle.loads(p.read_bytes()) # locally constructed/trusted pickle only!
        pr,sr=check_predict(r)
    must(np.array_equal(p1,pr) and np.array_equal(sd1,sr),'Trained-model trusted pickle replay changed prediction')
    return {'status':'V54_SYNTHETIC_TWO_FITS_AND_REAL_MODEL_PICKLE_REPLAY_PASS',
            'scientific_scope':'Synthetic same-process, same-environment only; does not validate DM prediction accuracy or cross-platform bitwise determinism',
            'epochs':cfg['epochs'],'local_steps':cfg['local_steps'],
            'max_abs_ELBO_repeat_diff':float(np.max(np.abs(h1-h2))),
            'max_abs_prediction_repeat_diff':float(np.max(np.abs(p1-p2))),
            'max_abs_SD_repeat_diff':float(np.max(np.abs(sd1-sd2))),
            'trusted_pickle_replay_prediction_exact':True,'synthetic_pickle_sha256':h,
            'no_real_data_loaded':True,'release_provenance_not_claimed':True}


def self_test():
    checks=[]
    def check(name,fn):fn();checks.append(name);print('PASS',name,flush=True)
    def assert_reject(fn):
        try:fn()
        except (ValueError,KeyError,TypeError):return
        raise AssertionError('Tampered input accepted')
    cfg=main.load_config()
    check('V52_SHA_and_config_lock',lambda:must(cfg['K']==2 and cfg['local_steps']==8,'Unexpected model configuration'))
    check('folds_predeclared_disjoint',lambda:must(set(sum((list(x) for x in FOLDS.values()),[]))==set(EXPECTED_L),'Fold definition mismatch'))
    check('source_and_test_exclusion',lambda:must(not(set(EXPECTED_L)|set(EXPECTED_U))&set(EXPECTED_LOCKED),'Historical test ring leaked'))
    check('target_binding',lambda:must(tuple(cfg['quality_targets'])==integrity.TARGETS,'Different quality targets'))
    check('hashes_match_locked_reference',lambda:must(len(cfg['core_math_file_sha256'])==10,'Math module count mismatch'))
    check('finite_guard_rejects_nan',lambda:assert_reject(lambda:must(finite([float('nan')]),'Synthetic nan detected')))
    def write_once_guard():
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'guard.json'
            with path.open('x') as f:f.write('{}')
            try:
                with path.open('x') as f:f.write('overwrite')
            except FileExistsError:pass
            else:raise AssertionError('Write-once overwrite unexpectedly permitted')
    check('write_once_json_guard',write_once_guard)
    print(f'V54 PRE-FREEZE SELF-TEST: {len(checks)}/{len(checks)} PASS; NO DM PICKLE, NO TRAINING',flush=True)


def audit_negative_tests(folder):
    folder=pathlib.Path(folder)
    originals={p.name:json.loads(p.read_text(encoding='utf-8')) for p in
               [*(folder/f'FOLD_{i}_PAIRED_STEPS_0_8.json' for i in FOLDS),
                folder/'V53_PAIRED_DEVELOPMENT_CV4_SUMMARY.json']}
    tests=[]
    def tamper_case(name, file, mutate):
        with tempfile.TemporaryDirectory() as td:
            place=pathlib.Path(td)
            for fn,d in originals.items():
                cp=json.loads(json.dumps(d))
                if fn==file:mutate(cp)
                (place/fn).write_text(json.dumps(cp,allow_nan=False),encoding='utf-8')
            try:audit_documents(place)
            except (ValueError,KeyError,TypeError):
                tests.append(name);print('PASS',name,flush=True)
                return
            raise AssertionError(f'Tampered original accepted: {name}')
    f1='FOLD_1_PAIRED_STEPS_0_8.json'
    tamper_case('replaced_validation_ring_refused',f1,lambda d:d['validation'].__setitem__(0,'DM07'))
    tamper_case('validation_in_y_scaler_refused',f1,lambda d:d['scaler_fit_ids']['y'].append('DM04'))
    tamper_case('tampered_prediction_residual_refused',f1,lambda d:d['paired_validation'][0]['error_steps_8_mm'].__setitem__(0,123.))
    tamper_case('tampered_ELBO_history_refused',f1,lambda d:d['arms']['steps_8']['history'][-1].__setitem__('elbo',123.))
    tamper_case('changed_model_fingerprint_refused',f1,lambda d:d.__setitem__('configuration_sha256','0'*64))
    sf='V53_PAIRED_DEVELOPMENT_CV4_SUMMARY.json'
    tamper_case('tampered_summary_metric_refused',sf,lambda d:d['RMSE_steps8_mm'].__setitem__(0,42.))
    tamper_case('historical_test_register_tamper_refused',sf,lambda d:d['outer_test_ids_NEVER_USED'].__setitem__(0,'DM04'))
    print(f'V54 NEGATIVE AUDIT TESTS: {len(tests)}/{len(tests)} PASS; NO TRAINING, NO DM PICKLE',flush=True)


def main_cli():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='command',required=True)
    sp.add_parser('self-test')
    aa=sp.add_parser('audit-v53')
    aa.add_argument('--results-dir',default='results/v53_feedback_devonly')
    aa.add_argument('--out',required=True,help='write-once new report path outside original V53 results recommended')
    sp.add_parser('synthetic-replay')
    neg=sp.add_parser('test-audit-negative')
    neg.add_argument('--results-dir',default='results/v53_feedback_devonly')
    args=p.parse_args()
    if args.command=='self-test':self_test()
    if args.command=='audit-v53':
        res=audit_documents(args.results_dir)
        out=pathlib.Path(args.out)
        must(out.parent.is_dir(),'Output parent must exist; no automatic parent creation')
        with out.open('x',encoding='utf-8') as f: json.dump(res,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
        print(json.dumps({k:v for k,v in res.items() if k not in ('elbo_diagnostics','ringwise','original_result_sha256')},indent=2,ensure_ascii=False))
        print('WRITE-ONCE AUDIT:',out)
    if args.command=='synthetic-replay':
        print(json.dumps(synthetic_replay(),indent=2,ensure_ascii=False))
    if args.command=='test-audit-negative':audit_negative_tests(args.results_dir)


if __name__=='__main__':main_cli()

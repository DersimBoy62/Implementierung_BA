"""V5.4 read-only negative/contract QA of V5.2 NEW-RING prediction boundary.

Does NOT train, read the DM dataset, assess past tests, create quality predictions,
or fake actual operator/machine completion. All generated CSVs are SYNTHETIC.
The optional --model-dir is verified by SHA and manifest WITHOUT unpickling.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / 'final_candidate_v52'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import main_bayesian_ssr_v52 as main
import input_safety_v51 as safety
import release_integrity_v52 as release

EXPECTED_MODEL_SHA = '56c3605ba3e0157125ecb7b2adaaddcb160a15c16edf22f2e28e5b232f2ae3c4'
PROTOCOL = 'V54_NEW_RING_PREDICTION_BOUNDARY_READONLY_2026-11-12_v1'


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def denied(function, kind=(ValueError, FileExistsError, FileNotFoundError)):
    try:
        function()
    except kind:
        return
    raise AssertionError('Invalid prediction request unexpectedly accepted')


def synthetic_rows(n=150, pre=5, post=5):
    # Independently generated SYNTHETIC process-like fixture, not a real new ring.
    rows=[]
    for i in range(pre+n+post):
        j=max(0, min(n-1,i-pre))
        p=j/(n-1) if n>1 else 0.0
        active = pre<=i<pre+n
        rows.append([round(0.1*i,8), 43+8*p, 40+10*p,
                     .37+.15*p, .070-.012*p, .075-.020*p,
                     .004+.001*math.sin(2*math.pi*p),
                     1 if active else 6, int(active), int(active)])
    return rows


def save_csv(path,rows,header=None):
    with path.open('x',newline='',encoding='utf-8') as fh:
        wr=csv.writer(fh)
        wr.writerow(list(header if header is not None else safety.REQUIRED))
        wr.writerows(rows)
    return path


def safe_policy():
    return json.loads((SRC/'DEVELOPMENT_INPUT_POLICY_V51.json').read_text(encoding='utf-8'))


def self_test():
    checks=[]
    def step(name, fun):
        fun();checks.append(name);print('PASS',name)

    def config_and_shas():
        c=main.load_config()
        require(c['freeze_status']=='PROVISIONAL_USER_APPROVAL_PENDING','Not provisional')
        require(c['K']==2 and c['C']==0.2 and c['epochs']==10 and c['local_steps']==8,'Config changed')
        require(len(c['core_math_file_sha256'])==10,'Math modules changed')
    step('locked_v52_config_and_all_ten_core_hashes',config_and_shas)

    with tempfile.TemporaryDirectory(prefix='v54-boundary-synthetic-') as td:
        root=pathlib.Path(td)
        policy=safe_policy()
        valid=save_csv(root/'valid_SYNTHETIC.csv',synthetic_rows())
        def positive_process_only():
            t,x,report=safety.validate_raw_logger(valid,policy)
            require(len(t)==150 and x.shape==(150,6) and report['f2_rows']==150,'Valid synthetic F2 rejected')
        step('valid_synthetic_process_only_logger_no_quality',positive_process_only)

        def old_ids():
            for name in ('DM01','dm04','DM-0004','ＤＭ０４','DM04_COPY','dm15','DM18-v2'):
                denied(lambda name=name:safety.normalize_and_check_new_ring_id(name))
            require(safety.normalize_and_check_new_ring_id(' NEW19 ')=='NEW19','Valid new ID rejected')
        step('historical_ring_aliases_rejected_before_model_io',old_ids)

        incomplete=save_csv(root/'missing_end.csv',synthetic_rows()[:-5])
        step('missing_process_trailer_rejected',lambda:denied(lambda:safety.validate_raw_logger(incomplete,policy)))
        no_pre=save_csv(root/'missing_start.csv',synthetic_rows()[5:])
        step('missing_process_preamble_rejected',lambda:denied(lambda:safety.validate_raw_logger(no_pre,policy)))
        short=save_csv(root/'short.csv',synthetic_rows(n=6))
        step('unrealistically_short_f2_rejected',lambda:denied(lambda:safety.validate_raw_logger(short,policy)))
        gap=synthetic_rows()
        for row in gap[55:]:row[0]+=20.0
        broken=save_csv(root/'gap.csv',gap)
        step('time_axis_gap_rejected',lambda:denied(lambda:safety.validate_raw_logger(broken,policy)))
        mm=synthetic_rows()
        for row in mm:row[3]*=1000
        wrong_unit=save_csv(root/'unit_mm.csv',mm)
        step('mm_instead_of_m_rejected',lambda:denied(lambda:safety.validate_raw_logger(wrong_unit,policy)))
        leak=save_csv(root/'extra_target.csv',[x+[1.4] for x in synthetic_rows()],
                      list(safety.REQUIRED)+['ovality.ovality_radial_mm'])
        step('quality_target_column_forbidden_in_predictor_csv',lambda:denied(lambda:safety.validate_raw_logger(leak,policy)))

        # This is NOT an operator closeout: it is explicitly FALSE and must be refused.
        _,_,rep=safety.validate_raw_logger(valid,policy)
        false_closeout={'schema':'DM-V51-END-OF-RUN-ATTESTATION-v1',
                       'ring_id':'NEW19',
                       'raw_logger_sha256':safety.hash_file(valid),
                       'completed_ring_confirmed':False,
                       'evidence_type':'operator_documented',
                       'evidence_reference':'SYNTHETIC_NOT_REAL',
                       'confirmed_by':'SYNTHETIC_TEST_ONLY',
                       'f2_samples':rep['f2_rows'],
                       'f2_last_time_s':rep['f2_tlast_s']}
        record=root/'false_attestation.json'
        record.write_text(json.dumps(false_closeout,allow_nan=False),encoding='utf-8')
        step('false_or_missing_completion_must_fail',lambda:denied(lambda:safety.verify_completion_attestation(record,'NEW19',valid,rep)))

        # Confirm model IO is NOT reached when completion is missing.
        step('missing_completion_rejected_before_model_access',
             lambda:denied(lambda:main.predict_one(root/'NONEXISTENT_MODEL','NEW19',valid,root/'never.json',root/'missing.json')))
        out=root/'already_exists.json';out.write_text('{}',encoding='utf-8')
        step('write_once_prediction_output_must_not_be_overwritten',
             lambda:denied(lambda:main.predict_one(root/'NONEXISTENT_MODEL','NEW19',valid,out,root/'missing.json')))
        require(out.read_text(encoding='utf-8')=='{}','Existing output changed')
        require(not (root/'never.json').exists(),'Unexpected prediction output')

    print(f'V54 PREDICTION BOUNDARY: {len(checks)}/{len(checks)} PASS; '
          'SYNTHETIC NEGATIVE GATES ONLY; NO TRAINING / NO DM DATASET / NO PREDICTIONS')
    return len(checks)


def audit_model(model_dir, out):
    # The optional artifact verification is read-only and NEVER unpickles.
    cfg=main.load_config()
    manifest=release.verify_bundle(model_dir,cfg,SRC,strict_environment=True,load_pickle=False)
    require(manifest['model_sha256']==EXPECTED_MODEL_SHA,'Not the recorded RUN01 development model')
    require(manifest['status']=='PROVISIONAL_DEVELOPMENT_ONLY','Unexpected model status')
    require(manifest['training_input_sha256']==release.SOURCE_HASH,'Wrong original dataset fingerprint')
    out=pathlib.Path(out)
    require(out.parent.is_dir() and not out.exists() and not out.is_symlink(),
            'Output must be new; pre-existing reports are immutable')
    payload={
      'protocol':PROTOCOL,
      'status':'DEVELOPMENT_MODEL_BOUNDARY_READY_NOT_REAL_NEW_RING_PREDICTION',
      'model_sha256':manifest['model_sha256'],
      'manifest_sha256':release.sha_file(pathlib.Path(model_dir)/'TRAIN_MANIFEST.json'),
      'source_sha256':manifest['training_input_sha256'],
      'config_sha256':manifest['config_sha256'],
      'model_environment':manifest['environment'],
      'checks':'MODEL_SHA_MANIFEST_CODE_ENV_VERIFIED_NO_UNPICKLING',
      'synthetic_negative_contract_checks':12,
      'full_real_new_ring_positive_predict':'NOT_TESTED__NEW_RAW_CSV_AND_GENUINE_CLOSEOUT_REQUIRED',
      'independent_generalization':'NOT_ESTABLISHED',
      'model_convergence':'NOT_ESTABLISHED_BY_10_EPOCHS',
      'no_dm_dataset_loaded':True,'no_training':True,
      'no_quality_scoring':True,'no_old_test_used':True,
      'operator_attestation_authenticity':'MANUAL_VERIFICATION_REQUIRED',
      'safety_warning':'Never unpickle models not personally created in the trusted local project; manifest SHA alone is not an external signature.'}
    with out.open('x',encoding='utf-8') as f:
        json.dump(payload,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':payload['status'],'model_sha256':payload['model_sha256'],
                      'audit_written_once':str(out)},ensure_ascii=False,indent=2))


def main_cli():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('self-test')
    au=sub.add_parser('audit-model')
    au.add_argument('--model-dir',required=True)
    au.add_argument('--out',required=True)
    args=p.parse_args()
    if args.cmd=='self-test':self_test()
    elif args.cmd=='audit-model':audit_model(args.model_dir,args.out)

if __name__=='__main__':main_cli()

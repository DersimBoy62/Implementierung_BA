"""No pickle, no training, no quality labels, no historical-ring prediction."""
import csv
import json
import pathlib
import tempfile
from unittest import mock
import numpy as np
import input_safety_v51 as safe
import main_bayesian_ssr_v51 as drv

HERE=pathlib.Path(__file__).resolve().parent
POLICY=json.loads((HERE/'DEVELOPMENT_INPUT_POLICY_V51.json').read_text(encoding='utf-8'))


def rows(n=150,preamble=5,trailer=5):
    data=[]
    for i in range(preamble+n+trailer):
        j=max(0,i-preamble)
        f2=preamble<=i<preamble+n
        phase=1 if f2 else 6
        theta=j/max(n-1,1)
        data.append([round(i*.1,8),
                     43+8*theta, 40+10*theta,
                     .37+.15*theta, .070-.012*theta,
                     .075-.020*theta, .004+.001*np.sin(2*np.pi*theta),
                     phase,int(f2),int(f2)])
    return data


def export(folder,rs,fields=None):
    path=folder/'raw.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(fields or safe.REQUIRED);w.writerows(rs)
    return path


def refuses(fn,fragment=None):
    try:fn()
    except (ValueError,RuntimeError,FileExistsError) as e:
        if fragment and fragment.lower() not in str(e).lower():
            raise AssertionError('Unexpected rejection reason: '+str(e))
    else:raise AssertionError('Bad input was accepted')


def run_tests():
    checks=[]
    def step(name,fn):
        fn();checks.append(name);print('PASS',name)

    def t_config_core_hash():
        assert drv.load_config()['K']==2
        assert len(drv.load_config()['core_math_file_sha256'])==10
    step('unchanged_full_bayesian_model_config_and_10_hashed_math_kernels',t_config_core_hash)

    def t_historical_aliases():
        for old in ('DM01','dm04','dm-0004',' DM 04 ','ＤＭ０４','DM04_COPY','DM13','dm15','DM18-v2'):
            refuses(lambda old=old:safe.normalize_and_check_new_ring_id(old),'historical')
        assert safe.normalize_and_check_new_ring_id(' new_19 ')=='NEW_19'
        assert safe.normalize_and_check_new_ring_id('DM19')=='DM19'
        for invalid in ('','DM 19','??new','a','new!','N'*33):
            refuses(lambda invalid=invalid:safe.normalize_and_check_new_ring_id(invalid))
    step('new_ring_identifier_canonicalization_and_historical_alias_rejection',t_historical_aliases)

    def t_valid_qa():
        with tempfile.TemporaryDirectory() as d:
            p=export(pathlib.Path(d),rows())
            t,x,rep=safe.validate_raw_logger(p,POLICY)
            assert len(t)==150 and x.shape==(150,6) and rep['raw_rows']==160
            assert abs(t[-1]-t[0]-14.9)<1e-8
    step('complete_raw_logger_positive_with_phase6_before_and_after',t_valid_qa)

    def t_premature_stop():
        with tempfile.TemporaryDirectory() as d:
            p=export(pathlib.Path(d),rows()[:-5])
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'final phase-6')
    step('missing_end_of_run_phase6_rejected',t_premature_stop)

    def t_missing_preamble():
        with tempfile.TemporaryDirectory() as d:
            p=export(pathlib.Path(d),rows()[5:])
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'initial phase-6')
    step('truncated_start_rejected',t_missing_preamble)

    def t_too_short():
        with tempfile.TemporaryDirectory() as d:
            p=export(pathlib.Path(d),rows(n=6))
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'too short')
    step('six_point_f2_ended_but_incomplete_rejected',t_too_short)

    def t_timestamp_gap():
        with tempfile.TemporaryDirectory() as d:
            a=rows()
            for r in a[50:]:r[0]+=20
            p=export(pathlib.Path(d),a)
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'timestamp gap')
    step('twenty_second_gap_rejected_before_inference',t_timestamp_gap)

    def t_clock_and_duplicate():
        with tempfile.TemporaryDirectory() as d:
            a=rows();a[51][0]=a[50][0]
            p=export(pathlib.Path(d),a)
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'timestamp')
    step('duplicate_time_rejected',t_clock_and_duplicate)

    def t_disconnected_f2():
        with tempfile.TemporaryDirectory() as d:
            a=rows();a[65][-3:]=[6,0,0]
            p=export(pathlib.Path(d),a)
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'islands')
    step('fragmented_process_window_rejected',t_disconnected_f2)

    def t_quality_leak():
        with tempfile.TemporaryDirectory() as d:
            a=[r+[1.2] for r in rows()]
            p=export(pathlib.Path(d),a,list(safe.REQUIRED)+['ovality.ovality_radial_mm'])
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'exact')
    step('quality_or_unexpected_columns_rejected',t_quality_leak)

    def t_scale_mm():
        with tempfile.TemporaryDirectory() as d:
            a=rows()
            for r in a:r[3]*=1000
            p=export(pathlib.Path(d),a)
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'Ist_D150')
    step('millimetre_instead_of_metre_sensor_rejected',t_scale_mm)

    def t_outside_domain():
        with tempfile.TemporaryDirectory() as d:
            a=rows()
            for r in a:r[6]=0.8
            p=export(pathlib.Path(d),a)
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'RWG55')
    step('large_finite_but_out_of_domain_ring_growth_rejected',t_outside_domain)

    def t_sensor_flatline():
        with tempfile.TemporaryDirectory() as d:
            a=rows()
            for r in a:r[3]=.4
            p=export(pathlib.Path(d),a)
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'flatline')
    step('stuck_geometric_sensor_rejected',t_sensor_flatline)

    def t_nan():
        with tempfile.TemporaryDirectory() as d:
            a=rows();a[10][3]=float('nan')
            p=export(pathlib.Path(d),a)
            refuses(lambda:safe.validate_raw_logger(p,POLICY),'non-finite')
    step('nan_sensor_value_rejected',t_nan)

    def record(csv_file,report,ring='NEW01'):
        return {'schema':'DM-V51-END-OF-RUN-ATTESTATION-v1',
                'ring_id':ring,'raw_logger_sha256':safe.hash_file(csv_file),
                'completed_ring_confirmed':True,'evidence_type':'operator_documented',
                'evidence_reference':'QA-SYNTHETIC-ONLY-NOT-REAL','confirmed_by':'qa fixture',
                'f2_samples':report['f2_rows'],'f2_last_time_s':report['f2_tlast_s']}

    def t_attestation_binding():
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);p=export(root,rows());_,_,rep=safe.validate_raw_logger(p,POLICY)
            a=record(p,rep);e=root/'closeout.json';e.write_text(json.dumps(a))
            good=safe.verify_completion_attestation(e,'new01',p,rep)
            assert good['evidence_type']=='operator_documented'
            a['f2_samples']-=1;e.write_text(json.dumps(a))
            refuses(lambda:safe.verify_completion_attestation(e,'NEW01',p,rep),'sample count')
            a=record(p,rep);a['raw_logger_sha256']='0'*64;e.write_text(json.dumps(a))
            refuses(lambda:safe.verify_completion_attestation(e,'NEW01',p,rep),'exact raw logger')
    step('independent_completion_record_binds_exact_logger_and_F2_extent',t_attestation_binding)

    def t_missing_attestation():
        with tempfile.TemporaryDirectory() as d:
            p=export(pathlib.Path(d),rows())
            refuses(lambda:drv.predict_one('bad_model','NEW01',p,pathlib.Path(d)/'p.json',pathlib.Path(d)/'missing.json'))
            # Missing attestation is rejected before attempting to load model.
    step('predict_requires_explicit_model_and_completion_record',t_missing_attestation)

    def t_locked_prediction_prior_to_model_access():
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(drv,'_load_owned_model',side_effect=AssertionError('MUST NOT ACCESS MODEL')):
                for old in ('DM04','dm-04','DM15','DM04_COPY'):
                    refuses(lambda old=old:drv.predict_one('bad_model',old,'missing.csv',pathlib.Path(d)/'out.json','missing.json'))
    step('historical_prediction_rejected_before_model_or_csv_access',t_locked_prediction_prior_to_model_access)

    def t_bad_policy():
        bad=dict(POLICY);bad['units']=['millimetres']*6
        with tempfile.TemporaryDirectory() as d:
            p=export(pathlib.Path(d),rows())
            refuses(lambda:safe.validate_raw_logger(p,bad),'units')
    step('wrong_units_in_policy_rejected',t_bad_policy)

    print(f'{len(checks)} V5.1 TESTS PASSED; NO TRAINING, NO PICKLE, NO OUTER LABELS')
    return checks

if __name__=='__main__':run_tests()

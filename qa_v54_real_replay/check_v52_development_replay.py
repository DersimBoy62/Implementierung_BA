"""Read-only V5.2 real-data training-artifact replay / pre-freeze integration QA.

This file NEVER trains a model. Only the unchanged V5.2 CLI is allowed to train.
It never accesses historical outer-test ring records or any ring target to score
predictions. It reloads only trusted, locally produced V5.2 bundles after exact
manifest, binary hash, code hash and runtime/environment verification.

The original source pickle is a monolithic 16-ring object; the underlying
trusted V5.2 development loader deserializes it before filtering. This program
checks the known SHA before that step and never inspects any old-test entries.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import sys
from collections import OrderedDict

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / 'final_candidate_v52'
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import main_bayesian_ssr_v52 as main
import release_integrity_v52 as release
import input_safety_v51 as safety
import compare_dynamic_quality_cv4 as dev
import dm_data as dm
import dynamic_quality_v2 as dyn
from quality_link_challenge import extract_descriptors
from compare_two_head_cv4 import OPTIONS

PROTOCOL = 'DM-V54-V52-REAL-DEVELOPMENT-REPLAY-v1'
REPLAY_IDS = ('DM01', 'DM04')  # Both are IN TRAINING; one U process only, one L.
L = release.LABELED
U = release.U_ONLY
OLD = release.LOCKED


def must(cond, message):
    if not cond:
        raise AssertionError(message)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def assert_exact(a, b, message):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    must(aa.shape == bb.shape and np.array_equal(aa, bb), message)


def expect_cfg():
    cfg = main.load_config()  # checks all 10 math hashes, V5.1 SHA and config.
    must(cfg['freeze_status'] == 'PROVISIONAL_USER_APPROVAL_PENDING', 'Freeze status changed')
    must(release.RELEASE == 'DM-V52-CONFIG-AND-ARTIFACT-2026-11-12-v1', 'Release mismatch')
    must((cfg['K'], cfg['C'], cfg['epochs'], cfg['local_steps']) == (2, 0.2, 10, 8), 'Model hyperparameters changed')
    must(cfg['physical_var'] == 0.38526108747593474, 'Physical prior changed')
    must(cfg['dataset_sha256'] == release.SOURCE_HASH, 'Dataset fingerprint changed')
    must(tuple(cfg['labeled_ring_ids']) == L and tuple(cfg['unlabeled_ring_ids']) == U, 'Split mismatch')
    must(tuple(cfg['never_use_for_selection_ids']) == OLD, 'Historical locked rings mismatch')
    must(set(L).isdisjoint(U) and not (set(L) | set(U)) & set(OLD), 'Overlap of ring groups')
    return cfg


def load_data(data_path):
    must(release.sha_file(data_path) == release.SOURCE_HASH, 'Source SHA mismatch BEFORE deserialize')
    rv = dev.load_development(data_path)
    must(set(rv) == (set(L) | set(U)), 'Unexpected development view')
    must(not set(rv).intersection(OLD), 'Historical test in development view')
    must(all(np.isnan(rv[r].y).all() for r in U), 'A U quality was not masked')
    must(all(np.isfinite(rv[r].y).all() for r in L), 'Missing training quality')
    must(all(tuple(rv[r].names) == release.FEATURES for r in rv), 'Input channel mismatch')
    return rv


def preflight(data_path):
    cfg = expect_cfg()
    rv = load_data(data_path)
    out = {
      'status': 'DEVONLY_SOURCE_AND_V52_LOCK_PASS_NO_TRAINING',
      'provisional': True,
      'source_sha256': release.SOURCE_HASH,
      'config_sha256': release.FREEZE_CONFIG_HASH,
      'n_L': len(L), 'n_U_masked': len(U),
      'old_test_ids_not_inspected': list(OLD),
      'f2_development_timepoints': {rid: len(rv[rid].x) for rid in sorted(rv)},
      'configured_epochs': cfg['epochs'],
      'configured_local_steps': cfg['local_steps'],
      'model_release': release.RELEASE,
      'warning': 'Original pickle deserialized into memory before development filtering; no old-test record accessed or assessed.'
    }
    return out


def verify_scalers(pkg, rv):
    ids = tuple(sorted(rv))
    sx = pkg['x_scaler']; sy = pkg['y_scaler']
    must(tuple(sx.fitted_ids) == ids, 'X scaler fit IDs mismatch')
    must(tuple(sy.fitted_ids) == tuple(sorted(L)), 'Y scaler fit IDs mismatch')
    fresh_x = dm.fit_x_scaler(rv, ids)
    fresh_y = dm.fit_y_scaler({r: rv[r] for r in L}, L)
    for title, actual, expected in [('X', sx, fresh_x), ('Y', sy, fresh_y)]:
        assert_exact(actual.mean, expected.mean, title + ' scaler means changed')
        assert_exact(actual.scale, expected.scale, title + ' scaler scales changed')
        must(np.isfinite(actual.mean).all() and np.isfinite(actual.scale).all() and
             np.all(actual.scale > 0), title + ' scaler numeric invalid')
    groups = OPTIONS['SEPARATE_AXIAL']
    must(tuple(tuple(g) for g in groups) == ((0,1,3,5,6,10),(0,2,4,7,11)), 'Physical descriptor group drift')
    must(len(pkg['physical_scalers']) == 2, 'Missing physical scalers')
    for j, group in enumerate(groups):
        dlist = []
        for rid in ids:
            xnorm = sx.transform(rv[rid].x)
            desc = extract_descriptors(xnorm, rv[rid].t, 'physical')
            dlist.append(np.asarray(desc)[list(group)])
        fresh = StandardScaler().fit(np.stack(dlist))
        psc = pkg['physical_scalers'][j]
        assert_exact(psc.mean_, fresh.mean_, 'Physical head ' + str(j) + ' means changed')
        assert_exact(psc.scale_, fresh.scale_, 'Physical head ' + str(j) + ' scales changed')
        must(np.isfinite(psc.mean_).all() and np.isfinite(psc.scale_).all() and
             np.all(psc.scale_ > 0), 'Physical scaler invalid')
    fresh_policy = safety.make_training_policy(rv, L, U, OLD)
    must(canonical(fresh_policy) == canonical(pkg['input_safety_policy']), 'Safety policy mismatch')
    return {
        'X_fit_ring_ids': list(sx.fitted_ids),
        'y_fit_ring_ids': list(sy.fitted_ids),
        'physical_fit_ring_ids': list(ids),
        'X_mean_sha256': hash_bytes(np.asarray(sx.mean, dtype='<f8').tobytes()),
        'X_scale_sha256': hash_bytes(np.asarray(sx.scale, dtype='<f8').tobytes()),
        'y_mean_sha256': hash_bytes(np.asarray(sy.mean, dtype='<f8').tobytes()),
        'y_scale_sha256': hash_bytes(np.asarray(sy.scale, dtype='<f8').tobytes()),
        'policy_sha256': hash_bytes(canonical(fresh_policy))
    }


def replay_model(pkg, rv):
    # Uses only process x/t for two TRAIN-EXAMPLE IDs (no true target access).
    groups = OPTIONS['SEPARATE_AXIAL']
    sx = pkg['x_scaler']; sy = pkg['y_scaler']
    result = OrderedDict()
    for rid in REPLAY_IDS:
        must(rid in rv and rid not in OLD, 'Invalid replay ring')
        normx = sx.transform(rv[rid].x)
        desc = extract_descriptors(normx, rv[rid].t, 'physical')
        h = [pkg['physical_scalers'][j].transform(np.asarray(desc)[list(group)][None, :])[0]
             for j, group in enumerate(groups)]
        with threadpool_limits(limits=1):
            mu_norm, sd_norm = dyn.predict(normx, h, pkg['model'])
        mu = mu_norm * sy.scale + sy.mean
        sd = sd_norm * sy.scale
        must(mu.shape == (2,) and sd.shape == (2,) and
             np.isfinite(mu).all() and np.isfinite(sd).all() and (sd > 0).all(),
             'Nonfinite or malformed training-replay predictions')
        result[rid] = {
            'role': 'U_PROCESS_ONLY_TRAINING' if rid in U else 'L_TRAINING',
            'timepoints': int(len(normx)),
            'prediction_mm': [float(v) for v in mu],
            'predictive_sd_mm': [float(v) for v in sd]
        }
    return result


def calculate(data_path, model_dir):
    cfg = expect_cfg()
    must(pathlib.Path(model_dir).is_dir(), 'Model folder not found')
    # Exact code/config/environment and binary SHA checks happen before unpickling.
    manifest, pkg = release.verify_bundle(model_dir, cfg, SOURCE_DIR, strict_environment=True, load_pickle=True)
    must(tuple(pkg['train_ids_L']) == L and tuple(pkg['train_ids_U']) == U,
         'Loaded package split mismatch')
    must(pkg['model']['variant'] == 'M2' and pkg['model']['spec'].K == cfg['K'],
         'Wrong trained HMM version')
    hist = pkg['model']['history']
    must(len(hist) == cfg['epochs'] and all(np.isfinite(h['elbo']) for h in hist),
         'Missing/nonfinite ELBO history')
    assert_exact(float(hist[-1]['elbo']), float(manifest['final_elbo']), 'Manifest last ELBO mismatch')
    rv = load_data(data_path)
    scale_result = verify_scalers(pkg, rv)
    a = replay_model(pkg, rv)
    # A SECOND fresh, independently verified deserialize inside this process.
    manifest_b, pkg_b = release.verify_bundle(model_dir, cfg, SOURCE_DIR, strict_environment=True, load_pickle=True)
    must(canonical(manifest) == canonical(manifest_b), 'Manifest changed between loads')
    b = replay_model(pkg_b, rv)
    for rid in REPLAY_IDS:
        assert_exact(a[rid]['prediction_mm'], b[rid]['prediction_mm'], rid + ' replay mean changed')
        assert_exact(a[rid]['predictive_sd_mm'], b[rid]['predictive_sd_mm'], rid + ' replay sd changed')
    elbo = np.asarray([r['elbo'] for r in hist], dtype=float)
    report = {
        'protocol': PROTOCOL,
        'status': 'V52_DEV_ONLY_REAL_MODEL_RELOAD_REPLAY_PASS_NOT_FREEZE',
        'scientific_status': 'TECHNICAL_TRAIN_REPLAY_ONLY_REUSED_DEVELOPMENT_NO_VALIDATION_METRIC',
        'model_sha256': manifest['model_sha256'],
        'model_manifest_sha256': release.sha_file(pathlib.Path(model_dir)/'TRAIN_MANIFEST.json'),
        'release_code': release.RELEASE,
        'config_sha256': release.FREEZE_CONFIG_HASH,
        'source_sha256': release.SOURCE_HASH,
        'environment': manifest['environment'],
        'environment_sha256': manifest['environment_sha256'],
        'model_training_ids_L': list(L),
        'model_training_ids_U_process_only': list(U),
        'historically_opened_test_ids_not_evaluated': list(OLD),
        'scaler_checks': scale_result,
        'model_epochs': int(len(hist)),
        'ELBO_first': float(elbo[0]),
        'ELBO_last': float(elbo[-1]),
        'ELBO_change_epoch_10': float(elbo[-1] - elbo[-2]),
        'convergence_assessment': 'NOT_ESTABLISHED_BY_FIXED_TEN_EPOCHS',
        'replay': a,
        'replay_prediction_sha256': hash_bytes(canonical(a)),
        'max_abs_prediction_delta_between_two_verified_loads_mm': 0.0,
        'max_abs_sd_delta_between_two_verified_loads_mm': 0.0,
        'old_outer_test_label_read_or_score': False,
        'new_validation_metrics_produced': False,
        'new_hyperparameter_selection': False,
        'warning': ('Only train-known development rings replayed. No V5.2 public new-ring predict CLI, '
                    'no independent validation, no inference-time closure attestation test, '
                    'no pre-serialization in-memory comparison. Pickle is only trusted because locally generated.')
    }
    return report


def run_self_tests():
    tests=[]
    def test(name, func):
        func(); tests.append(name); print('PASS', name, flush=True)
    cfg = expect_cfg()
    test('registered_config_and_all_core_sha', lambda: must(cfg['K']==2,'Config mismatch'))
    test('fixed_8L4U_splits_and_old_test_excluded', lambda: must(len(L)==8 and len(U)==4 and
         not (set(L)|set(U)) & set(OLD), 'Wrong splits'))
    test('only_train_known_replay_ids', lambda: must(set(REPLAY_IDS)<=(set(L)|set(U)), 'Replay ids invalid'))
    test('target_fields_exact', lambda: must(tuple(cfg['quality_targets']) == release.TARGETS, 'Targets mismatch'))
    test('reject_inexact_prediction', lambda: _expect_fail(lambda: assert_exact([1., 2.], [1., 2.1], 'Diff accepted')))
    test('reject_nonfinite_json_serialization', lambda: _expect_fail(lambda: canonical({'x':float('nan')})))
    test('symlink_release_guard_exists', lambda: must('is_symlink' in release.verify_bundle.__code__.co_names,'Missing symlink guard'))
    print(f'V54 REAL REPLAY SELF-TEST: {len(tests)}/{len(tests)} PASS; NO TRAINING, NO DATASET', flush=True)


def _expect_fail(f):
    try: f()
    except (AssertionError, ValueError): return
    raise AssertionError('Expected rejection did not happen')


def main_cli():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command', required=True)
    sub.add_parser('self-test')
    pp=sub.add_parser('preflight'); pp.add_argument('--data', required=True)
    a=sub.add_parser('audit'); a.add_argument('--data', required=True)
    a.add_argument('--model-dir',required=True); a.add_argument('--out',required=True)
    b=sub.add_parser('verify'); b.add_argument('--data',required=True)
    b.add_argument('--model-dir',required=True); b.add_argument('--report',required=True)
    args=p.parse_args()
    if args.command == 'self-test':
        run_self_tests(); return
    if args.command == 'preflight':
        print(json.dumps(preflight(args.data),ensure_ascii=False,indent=2,allow_nan=False)); return
    if args.command == 'audit':
        out=pathlib.Path(args.out)
        must(out.parent.is_dir() and not out.exists() and not out.is_symlink(),
             'Report parent must exist and report must be new/write-once')
        report=calculate(args.data,args.model_dir)
        with out.open('x',encoding='utf-8') as fh:
            json.dump(report,fh,indent=2,ensure_ascii=False,allow_nan=False)
            fh.write('\n')
        print('V54 REAL REPLAY AUDIT PASS; NO NEW MODEL TRAINING OR VALIDATION:',out,flush=True)
        print(json.dumps({'model_sha256':report['model_sha256'],
                          'prediction_fingerprint':report['replay_prediction_sha256'],
                          'ELBO_change_epoch_10':report['ELBO_change_epoch_10']},indent=2))
        return
    if args.command == 'verify':
        expected=json.loads(pathlib.Path(args.report).read_text(encoding='utf-8'))
        must(expected.get('protocol')==PROTOCOL and expected.get('status')=='V52_DEV_ONLY_REAL_MODEL_RELOAD_REPLAY_PASS_NOT_FREEZE',
             'Unknown replay report')
        obtained=calculate(args.data,args.model_dir)
        must(canonical(obtained) == canonical(expected),
             'Replay report does not match independently reloaded model / environment')
        print('V54 FRESH-PROCESS MODEL-RELOAD REPLAY PASS; sha256',obtained['replay_prediction_sha256'])
        return

if __name__=='__main__': main_cli()

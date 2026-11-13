"""Read-only diagnostics of locked V5.2 development model history.

No training, no DM dataset load, no predictions, no quality scoring. The model
must be a trusted local V5.2 model with the expected Windows-run fingerprint.
Never run this on untrusted pickles: SHA256 plus a locally editable manifest is
not a provenance signature.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / 'final_candidate_v52'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import main_bayesian_ssr_v52 as main
import release_integrity_v52 as release

PROTOCOL = 'DM-V54-READONLY-HISTORY-2026-11-12-v1'
EXPECTED_MODEL_SHA256 = '56c3605ba3e0157125ecb7b2adaaddcb160a15c16edf22f2e28e5b232f2ae3c4'
EXPECTED_REPLAY_SHA256 = '9c0566b93663297570c9a4a2ff5f36abc5196e1f81bb910d182620756c6dfc52'
EXPECTED_DATA_SHA256 = 'bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canon(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(path):
    return release.sha_file(path)


def exact_config():
    cfg = main.load_config()
    require(cfg['freeze_status'] == 'PROVISIONAL_USER_APPROVAL_PENDING', 'The recorded freeze status changed')
    require(cfg['K'] == 2 and cfg['C'] == 0.2 and cfg['epochs'] == 10 and cfg['local_steps'] == 8,
            'Registered model settings changed')
    require(cfg['dataset_sha256'] == EXPECTED_DATA_SHA256, 'Dataset identity changed')
    return cfg


def inspect_history(history, n_epochs=10, k=2):
    require(type(history) is list and len(history) == n_epochs, 'Unexpected number of epochs')
    rows = []
    for ix, item in enumerate(history):
        require(isinstance(item, dict), 'Malformed history entry')
        epoch = item.get('epoch')
        require(type(epoch) is int and epoch == ix + 1, 'Epoch numbering drift')
        elbo = float(item['elbo'])
        gain = float(item['local_gain'])
        masses = [float(x) for x in item['state_mass']]
        require(math.isfinite(elbo) and math.isfinite(gain) and len(masses) == k and
                all(math.isfinite(v) and -1e-8 <= v <= 1 + 1e-8 for v in masses) and
                abs(sum(masses) - 1.0) < 1e-5, 'Nonfinite/invalid ELBO, local gain or masses')
        prior = rows[-1]['elbo'] if rows else None
        diff = elbo - prior if prior is not None else None
        rows.append({'epoch': epoch, 'elbo': elbo, 'delta_to_previous': diff,
                     'state_mass': masses, 'local_gain_reported_for_this_epoch': gain})
    deltas = [row['delta_to_previous'] for row in rows[1:]]
    increases = sum(v > 0 for v in deltas)
    decreases = sum(v < 0 for v in deltas)
    final = deltas[-1]
    tail = deltas[-3:]
    abs_last = max(abs(rows[-1]['elbo']), 1)
    return {
        'epochs': rows,
        'first_elbo': rows[0]['elbo'], 'last_elbo': rows[-1]['elbo'],
        'net_elbo_change': rows[-1]['elbo'] - rows[0]['elbo'],
        'increase_count_1_to_10': increases,
        'decrease_count_1_to_10': decreases,
        'last_epoch_change': final,
        'last_epoch_relative_change_vs_abs_final_elbo': final / abs_last,
        'last_three_epoch_changes': tail,
        'max_abs_epoch_change': max(abs(v) for v in deltas),
        'last_epoch_local_gain': rows[-1]['local_gain_reported_for_this_epoch'],
        'sum_reported_epoch_local_gains': sum(row['local_gain_reported_for_this_epoch'] for row in rows),
        'significant_last_gain_not_predeclared': 'No a priori convergence tolerance defined',
        'convergence_conclusion': 'NOT_ESTABLISHED_BY_FIXED_10_EPOCHS',
    }


def analyze(model_dir, replay_report):
    cfg = exact_config()
    model_dir = pathlib.Path(model_dir)
    manifest, pkg = release.verify_bundle(model_dir,cfg,CODE,strict_environment=True,load_pickle=True)
    require(manifest['model_sha256'] == EXPECTED_MODEL_SHA256, 'Wrong model SHA (only RUN01 permitted)')
    require(manifest['status'] == 'PROVISIONAL_DEVELOPMENT_ONLY', 'Status must stay provisional')
    require(pkg['source_data_sha256'] == EXPECTED_DATA_SHA256, 'Source data SHA changed')
    require(tuple(pkg['train_ids_L']) == release.LABELED and
            tuple(pkg['train_ids_U']) == release.U_ONLY, 'Development split changed')
    replay_path = pathlib.Path(replay_report)
    replay = json.loads(replay_path.read_text(encoding='utf-8'))
    require(replay['status'] == 'V52_DEV_ONLY_REAL_MODEL_RELOAD_REPLAY_PASS_NOT_FREEZE',
            'Replay report status invalid')
    require(replay['model_sha256'] == EXPECTED_MODEL_SHA256 and
            replay['replay_prediction_sha256'] == EXPECTED_REPLAY_SHA256,
            'Replay result is not the recorded Windows run')
    require(replay['config_sha256'] == release.FREEZE_CONFIG_HASH and
            replay['source_sha256'] == EXPECTED_DATA_SHA256,
            'Replay report config/data drift')
    data = inspect_history(pkg['model']['history'],cfg['epochs'],cfg['K'])
    require(data['last_elbo'] == manifest['final_elbo'], 'Manifest ELBO mismatch')
    require(abs(data['last_epoch_change'] - replay['ELBO_change_epoch_10']) < 1e-9,
            'Last ELBO gain does not match V5.4 replay audit')
    return {
        'protocol': PROTOCOL, 'status':'V54_TRAINING_HISTORY_REVIEWED_NOT_FINAL_FREEZE',
        'model_sha256': EXPECTED_MODEL_SHA256, 'replay_prediction_sha256': EXPECTED_REPLAY_SHA256,
        'replay_report_sha256': sha(replay_path),
        'manifest_sha256': sha(model_dir/'TRAIN_MANIFEST.json'),
        'config_sha256': release.FREEZE_CONFIG_HASH,
        'source_sha256':EXPECTED_DATA_SHA256,
        'environment': manifest['environment'],
        'registered_config':{'K':cfg['K'],'C':cfg['C'],'epochs':cfg['epochs'],
                             'local_steps':cfg['local_steps'],'random_seed':cfg['random_seed']},
        'history':data,
        'no_training_no_dataset_load_no_old_test_no_validation':True,
        'methodical_limit': ('Descriptive ELBO diagnostics, not a predeclared mathematical convergence criterion; '
                            'do not retune parameters or use reused development RMSE to extend training.'),
        'safety_notice': 'Only trusted locally generated and signed-off model pickle; hashes are not external signatures.'
    }


def self_test():
    # The original V5.2 config + core module hashes are tested as part of this step.
    exact_config()
    checks = []
    def step(label, callback):
        callback(); checks.append(label); print('PASS',label)
    mock = [ {'epoch':i, 'elbo':-100.0 + i*i, 'state_mass':[0.4,0.6],
              'local_gain': i*0.001} for i in range(1,11) ]
    step('valid_history_and_tail_growth', lambda: require(inspect_history(mock)['last_epoch_change'] == 19, 'Tail mismatch'))
    step('last_gain_is_not_all_epoch_gain', lambda: require(abs(inspect_history(mock)['sum_reported_epoch_local_gains']-0.055)<1e-14,'Gain-sum mismatch'))
    def reject(modifier):
        import copy
        bad = copy.deepcopy(mock)
        modifier(bad)
        try:inspect_history(bad)
        except (ValueError,TypeError,KeyError):return
        raise AssertionError('Malformed history accepted')
    step('reject_missing_epoch',lambda:reject(lambda a:a.pop()))
    step('reject_wrong_epoch_number',lambda:reject(lambda a:a[3].update(epoch=6)))
    step('reject_nan_elbo',lambda:reject(lambda a:a[5].update(elbo=float('nan'))))
    step('reject_bad_mass',lambda:reject(lambda a:a[8].update(state_mass=[0.9,0.9])))
    step('reject_missing_local_gain',lambda:reject(lambda a:a[8].pop('local_gain')))
    step('locked_configuration_and_10_math_hashes_unchanged',lambda:require(exact_config()['epochs']==10,'Config bad'))
    print(f'V54 READ-ONLY HISTORY SELFTEST: {len(checks)}/{len(checks)} PASS; NO TRAINING, NO DM DATASET, NO MODEL LOAD')


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test')
    au=sub.add_parser('audit');au.add_argument('--model-dir',required=True)
    au.add_argument('--replay-report',required=True);au.add_argument('--out',required=True)
    ve=sub.add_parser('verify');ve.add_argument('--model-dir',required=True)
    ve.add_argument('--replay-report',required=True);ve.add_argument('--report',required=True)
    args=parser.parse_args()
    if args.command=='self-test':
        self_test();return
    if args.command=='audit':
        out=pathlib.Path(args.out)
        require(out.parent.is_dir() and not out.exists() and not out.is_symlink(),'Output must be new, parent directory must exist')
        doc=analyze(args.model_dir,args.replay_report)
        with out.open('x',encoding='utf-8') as fh:
            json.dump(doc,fh,indent=2,ensure_ascii=False,allow_nan=False);fh.write('\n')
        print(json.dumps({'status':doc['status'],'last_epoch_gain':doc['history']['last_epoch_change'],
                          'convergence':doc['history']['convergence_conclusion'],
                          'positive_elbo_steps':doc['history']['increase_count_1_to_10'],
                          'negative_elbo_steps':doc['history']['decrease_count_1_to_10'],
                          'audit_written_once':str(out)},indent=2,ensure_ascii=False))
        return
    if args.command=='verify':
        expected=json.loads(pathlib.Path(args.report).read_text(encoding='utf-8'))
        actual=analyze(args.model_dir,args.replay_report)
        require(canon(expected)==canon(actual),'Read-only audit does not reproduce exactly')
        print('V54 READ-ONLY HISTORY VERIFICATION PASS; NO TRAINING / NO DATASET / NO OLD TEST')
        return

if __name__=='__main__':main_cli()

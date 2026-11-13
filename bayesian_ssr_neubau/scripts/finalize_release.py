"""Verify real artifacts and bind test evidence to the released model and sources."""
from pathlib import Path
import hashlib
import json
import numpy as np
from bayesian_ring_ssr.data_io import DATA_SHA256,sha256,read_process_file
from bayesian_ring_ssr.persistence import load_model,write_json
from bayesian_ring_ssr.posterior_predictive import predict_new_ring
from bayesian_ring_ssr.metrics import regression_metrics
from bayesian_ring_ssr.split_guard import LABELED_IDS,UNLABELED_IDS,LOCKED_IDS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    project = Path(__file__).resolve().parents[1]
    root = project.parent
    checks = {}
    assert sha256(root/'input/DM_16_aktive_Ringe.pkl')==DATA_SHA256
    checks['data_sha256'] = True
    verification = read(project/'docs/verification_001.json')
    assert verification['exitcode']==0 and verification['tests']==105
    assert verification['failures']==verification['errors']==verification['skipped']==0
    checks['105_executed_tests'] = True
    graph = read(project/'docs/IMPORTGRAPH.json')
    for entry in graph['modules'].values():
        assert sha256(root/entry['path'])==entry['sha256']
    checks['tested_source_hashes_match_import_audit'] = True
    source_hashes = {p.relative_to(root).as_posix():sha256(p) for p in (project/'src').rglob('*.py')}
    bundle = project/'outputs/baseline_8L4U_001'
    model = load_model(bundle)
    assert len(model.history)==30
    assert min(row['delta'] for row in model.update_history)>=-1e-6
    assert len(model.update_history)==450
    checks['30_epochs_450_validated_coordinate_blocks'] = True
    for filename in ('checkpoint_epoch_001.json','checkpoint_epoch_030.json','epoch_metrics.jsonl','diagnostics.json'):
        assert (bundle/filename).is_file()
    ring = read_process_file(project/'outputs/prediction_process_DM04.csv',ring_id='DM04')
    assert predict_new_ring(model,ring)==read(project/'outputs/prediction_DM04_001.json')
    checks['raw_process_cli_prediction_matches_reloaded_model'] = True
    train_audit = read(project/'docs/TRAINING_30E_AUDIT.json')
    assert train_audit['model_sha256']==sha256(bundle/'model.json')
    reference = read(root/train_audit['post_training_reference']['path'])
    old_hashes = reference['mathematics_core_sha256']
    for name,digest in old_hashes.items():
        assert sha256(root/'final_candidate_v52/final_core'/name)==digest
    checks['all_ten_historical_math_sources_unchanged'] = True
    cv_dir = project/'outputs/development_cv_001'
    manifest = read(cv_dir/'SHA256_MANIFEST.json')
    for name,digest in manifest['files'].items():
        assert sha256(cv_dir/name)==digest
    cv = read(cv_dir/'development_cv.json')
    assert len(cv['folds'])==20
    for arm in cv['arms']:
        rows = [r for r in cv['per_ring'] if r['arm']==arm]
        assert len(rows)==8 and sorted(r['ring_id'] for r in rows)==list(LABELED_IDS)
        actual = regression_metrics([r['truth_mm'] for r in rows],[r['pred_mm'] for r in rows])
        for metric in ('rmse_mm','mae_mm'):
            np.testing.assert_allclose(actual[metric],cv['metrics'][arm][metric],atol=1e-14)
    for fold in cv['folds']:
        split = fold['fold']
        L,U,V = set(split['L']),set(split['U']),set(split['VAL'])
        assert len(L)==6 and len(U)==4 and len(V)==2
        assert not (L|U)&V and not (L|U|V)&set(LOCKED_IDS)
        if 'scalers' in fold['model']:
            scalers = fold['model']['scalers']
            assert set(scalers['y_scaler']['fit_ids'])==L
            Xids = L|U if fold['arm'] in ('shared_c0','ssl') else L
            assert set(scalers['x_scaler']['fit_ids'])==Xids
            assert set(scalers['descriptor_scaler']['fit_ids'])==Xids
        if 'history' in fold['model']:
            assert len(fold['model']['history'])==30
            assert all(h['epoch_gain']>=-1e-6 for h in fold['model']['history'])
    checks['four_folds_five_arms_all_scalers_train_only'] = True
    for file in ('README.md','CHANGELOG.md','requirements-lock-windows-py311.txt',
                 'docs/MATHEMATISCHES_MODELL.md','docs/QUELLENSTATUS.md','docs/Inventur.csv',
                 'docs/ABGLEICH_A_B_C_D_V60.md','docs/TESTPROTOKOLL.md','docs/ABSCHLUSSBERICHT.md'):
        assert (project/file).is_file()
    report = dict(date='2026-11-13',status='BASELINE_COMPLETE; not final model freeze',checks=checks,
        model_sha256=sha256(bundle/'model.json'),test_report_sha256=sha256(project/'docs/verification_001.json'),
        cv_report_sha256=sha256(cv_dir/'development_cv.json'),source_sha256=source_hashes,
        math_test_status='105 executed tests passed; evidence bound to model and current sources by this release validation',
        model_embedded_math_test_status=model.math_test_status,
        note='The immutable training object retained NOT_RECORDED for test status; actual executed test evidence is bound here after verification.',
        cv_metrics=cv['metrics'],limitations=['historically reused development labels','conditional temporal emission independence',
            'modular variational predictive uncertainty; uncalibrated','no independent external evaluation'])
    write_json(project/'docs/RELEASE_VALIDATION.json',report)
    exclusions = {'__pycache__','.pytest_cache'}
    files = {}
    for path in sorted(project.rglob('*')):
        relative = path.relative_to(project)
        if not path.is_file() or any(p in exclusions or p.endswith('.egg-info') for p in relative.parts) or path.name=='RELEASE_SHA256.json':
            continue
        files[relative.as_posix()] = sha256(path)
    write_json(project/'RELEASE_SHA256.json',dict(date='2026-11-13',files=files,
        excludes='Python/pytest caches and editable-install metadata only'))
    print(json.dumps(dict(status=report['status'],checks=checks,release_files=len(files),cv_metrics=cv['metrics']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

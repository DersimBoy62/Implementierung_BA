"""Read-only summary of development-only two-objective Optuna study.

Does not open any ring pickle or train any model. Verifies each immutable
trial report, Optuna values and the research manifest. Optional comparison
with the manual 16-configuration K grid (reads report JSONs only).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import optuna

NORM_REFERENCE=(0.5528856197,3.2026)
LOCKED=('DM07','DM08','DM11','DM15')


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(2**20),b''):h.update(c)
    return h.hexdigest()


def pareto_indices(rows):
    obj=[np.asarray(r['rmse_mm'],float) for r in rows]
    return [i for i,p in enumerate(obj) if not any(np.all(x<=p) and np.any(x<p) for j,x in enumerate(obj) if j!=i)]


def summarize(study_dir,manual_root=None):
    directory=Path(study_dir)
    manifest=json.loads((directory/'STUDY_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['protocol']!='DM-V3-OPTUNA-EXPLORATORY-v1':raise ValueError('Unexpected protocol')
    if tuple(manifest['outer_test_ids_already_evaluated_in_previous_study_and_excluded'])!=LOCKED:
        raise ValueError('Invalid protected ring split')
    db=(directory/'optuna_study.sqlite3').resolve()
    study=optuna.load_study(
        study_name=f'DM_V3_{manifest["search_phase"]}_{"smoke" if manifest["smoke"] else "cv4"}',
        storage=f'sqlite:///{db.as_posix()}')
    completed=[]
    for t in study.trials:
        if t.state!=optuna.trial.TrialState.COMPLETE:continue
        fname=t.user_attrs.get('report')
        if fname!=f'TRIAL_{t.number:04d}.json':raise ValueError('Report name mismatch')
        p=directory/fname
        if digest(p)!=t.user_attrs['report_sha256']:raise ValueError('Trial report modified')
        r=json.loads(p.read_text(encoding='utf-8'))
        if r['trial_number']!=t.number or r['source_manifest_sha256']!=digest(directory/'STUDY_MANIFEST.json'):
            raise ValueError('Manifest/report mismatch')
        if r['source_data_sha256']!=manifest['input_pickle_sha256']:
            raise ValueError('Data mismatch')
        if tuple(r['outer_ids_explicitly_excluded'])!=LOCKED:raise ValueError('Outer protection mismatch')
        if r['configuration']['K'] not in manifest['allowed_K'] or r['configuration']['variant'] not in manifest['allowed_quality_coupling']:
            raise ValueError('Search domain changed')
        if r['configuration']['preprocessing'] not in manifest['allowed_preprocessing']:
            raise ValueError('Preprocessing domain changed')
        assert len(r['per_ring'])==2*manifest['folds']
        if len(set(x['ring'] for x in r['per_ring']))!=len(r['per_ring']):raise ValueError('Duplicate ring')
        if set(x['ring'] for x in r['per_ring'])-set(r['development_labeled_ids']):raise ValueError('Invalid val ring')
        e=np.array([x['residual_mm'] for x in r['per_ring']],float)
        rmse=np.sqrt(np.mean(e*e,axis=0));mae=np.mean(np.abs(e),axis=0)
        if not np.allclose(rmse,r['rmse_mm'],atol=1e-10,rtol=1e-10) or not np.allclose(mae,r['mae_mm'],atol=1e-10,rtol=1e-10):
            raise ValueError('Metric inconsistency')
        if not np.allclose(rmse/manifest['objective_normalizers_rmse_mm'],t.values,atol=1e-10,rtol=1e-10):
            raise ValueError('Objective values inconsistent')
        completed.append({'trial':t.number,'config':r['configuration'],'rmse_mm':rmse.tolist(),
                          'mae_mm':mae.tolist(),'objectives_normalized':list(t.values),
                          'effective_states_folds':[f['effective_states_entropy_exp'] for f in r['fold_diagnostics']]})
    pareto=[completed[i] for i in pareto_indices(completed)]
    output={'protocol':manifest['protocol'],'phase':manifest['search_phase'],'smoke':manifest['smoke'],
            'completed_trials':len(completed),'total_trials':len(study.trials),
            'study_manifest_sha256':digest(directory/'STUDY_MANIFEST.json'),
            'trials':completed,'pareto':pareto,
            'status':'Development CV reused many times; Pareto is EXPLORATORY only.'}
    if manual_root:
        import summarize_state_grid as manual
        manual_rows=[]
        for p in sorted(Path(manual_root).rglob('STATES_*_FOLDS4.json')):
            try:
                config,metrics=manual.read_report(p)
            except (ValueError,AssertionError):
                raise ValueError(f'Invalid manual report: {p}')
            manual_rows.append({'preprocessing':config[0],'variant':config[1],'K':config[2],
                                'rmse_mm':metrics['rmse_mm']})
        if len(manual_rows)!=16 or len(set((x['preprocessing'],x['variant'],x['K']) for x in manual_rows))!=16:
            raise ValueError('Expected exactly 16 unique manual comparisons')
        output['manual_grid']={'n':16,'pareto':[manual_rows[i] for i in pareto_indices(manual_rows)]}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--manual-root',default=None)
    p.add_argument('--out',default=None)
    a=p.parse_args()
    r=summarize(a.study,a.manual_root)
    print('STUDY',r['phase'],'COMPLETE',r['completed_trials'],'/',r['total_trials'],'PARETO',len(r['pareto']))
    for x in r['pareto']:
        print(' OPTUNA PARETO',x['trial'],x['config'],'RMSE',np.round(x['rmse_mm'],4))
    if 'manual_grid' in r:
        print('MANUAL 16 CONFIGURATIONS PARETO:',r['manual_grid']['pareto'])
    print('NUR ENTWICKLUNGSDATEN. KEIN NEUER UNABHAENGIGER TEST.')
    if a.out:
        pp=Path(a.out);pp.parent.mkdir(parents=True,exist_ok=True)
        with pp.open('x',encoding='utf-8') as f:
            json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False)
        print('REPORT',pp.resolve())

if __name__=='__main__':main()

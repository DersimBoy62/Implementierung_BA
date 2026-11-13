"""Read-only audit of C dose-response CV reports. No training, no pickle access."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import scan_u_weight_dose_v3 as dose


def load_run(folder,pairing):
    root=Path(folder)
    p=root/'C_DOSE_PROTOCOL.json';s=root/'C_DOSE_SUMMARY.json'
    pro=json.loads(p.read_text(encoding='utf-8'))
    report=json.loads(s.read_text(encoding='utf-8'))
    if pro['protocol']!=dose.PROTOCOL or pro['pairing']!=pairing or report['protocol']!=dose.PROTOCOL:
        raise ValueError('Dose-response protocol/pairing mismatch')
    if report['source_protocol_sha256']!=dose.sha(p):raise ValueError('Summary checksum mismatch')
    if pro['fixed_C_grid']!=list(dose.FIXED_C):raise ValueError('Grid changed')
    if sorted(report['results'])!=['13','22']:raise ValueError('Candidate set changed')
    for t in ('13','22'):
        cfg=pro['configurations'][t]; v=report['results'][t]
        if v['trial_configuration']!=cfg or v['pairing']!=pairing:
            raise ValueError('Candidate config changed')
        levels=dose.requested_levels(cfg['C'])
        curves=v['dose_curve']
        if len(curves)!=len(levels) or not np.allclose([r['C'] for r in curves],levels,atol=0,rtol=0):
            raise ValueError('Unexpected C dose levels')
        for r in curves:
            if (len(r['rmse_mm'])!=2 or len(r['delta_mse_vs_shared_C0_mm2'])!=2 or
                not np.isfinite(r['rmse_mm']).all() or (np.array(r['rmse_mm'])<0).any()):
                raise ValueError('Bad dose response metrics')
            expect=np.array(r['rmse_mm'])**2-np.array(curves[0]['rmse_mm'])**2
            if not np.allclose(expect,r['delta_mse_vs_shared_C0_mm2'],rtol=1e-8,atol=1e-8):
                raise ValueError('RMSE/paired MSE inconsistencies')
        for c in levels:
            if c==0 or c==cfg['C']:continue
            name=root/dose.summary_file_name(int(t),c)
            raw=json.loads(name.read_text(encoding='utf-8'))
            if raw['dose_protocol_sha256']!=dose.sha(p):raise ValueError('Report checksum changed')
            errors=np.asarray([x['residual_mm'] for x in raw['cv4_report']['per_ring']],float)
            if (errors.shape!=(8,2) or set(x['ring'] for x in raw['cv4_report']['per_ring'])!=set(dose.opt.EXPECTED_L)):
                raise ValueError('Unexpected per-ring errors in dose trial')
            mm=np.sqrt(np.mean(errors**2,axis=0))
            if not np.allclose(mm, raw['cv4_report']['rmse_mm'],rtol=1e-8,atol=1e-8):
                raise ValueError('Stored trial metrics inconsistent')
    return pro,report


def aggregate(runs):
    expected=('canonical','20261114','20261115')
    if any(k not in expected for k in runs):raise ValueError('Unregistered pairing')
    if not runs:raise ValueError('No source results')
    refs=[r[0] for r in runs.values()]
    baseline=refs[0]
    for p in refs[1:]:
        if p['data_sha256']!=baseline['data_sha256'] or p['core_code_sha256']!=baseline['core_code_sha256'] or p['configurations']!=baseline['configurations']:
            raise ValueError('Different source/model candidate across pairing studies')
    results={}
    for trial in ('13','22'):
        cs=[x['C'] for x in runs['canonical'][1]['results'][trial]['dose_curve']]
        for pair,(proto,summary) in runs.items():
            if [x['C'] for x in summary['results'][trial]['dose_curve']]!=cs:
                raise ValueError('Dose grid differs across pairings')
        curve=[]
        for i,c in enumerate(cs):
            all_rmse=np.array([r[1]['results'][trial]['dose_curve'][i]['rmse_mm'] for r in runs.values()])
            dmse=np.array([r[1]['results'][trial]['dose_curve'][i]['delta_mse_vs_shared_C0_mm2'] for r in runs.values()])
            curve.append({'C':c,'mean_of_split_RMSE_mm':np.mean(all_rmse,axis=0).tolist(),
                          'min_split_RMSE_mm':np.min(all_rmse,axis=0).tolist(),
                          'max_split_RMSE_mm':np.max(all_rmse,axis=0).tolist(),
                          'splits_better_than_C0_by_target':np.sum(dmse<0,axis=0).tolist(),
                          'split_count':len(runs)})
        results[trial]=curve
    return {'protocol':dose.PROTOCOL,'studied_pairings':list(runs),
            'reused_rings_notice':'Three different pairings of SAME eight labeled development rings; not three independent tests.',
            'minima_not_confirmed':'Do not choose a final C and claim generalization from these repeatedly reused folds.',
            'candidate_C_profiles':results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical',required=True)
    p.add_argument('--split-a')
    p.add_argument('--split-b')
    p.add_argument('--out',help='Optional NEW JSON summary file (no overwrite)')
    a=p.parse_args()
    if bool(a.split_a)!=bool(a.split_b):raise ValueError('Provide both alternative split runs or neither')
    paths={'canonical':a.canonical}
    if a.split_a:paths.update({'20261114':a.split_a,'20261115':a.split_b})
    runs={key:load_run(value,key) for key,value in paths.items()}
    data=aggregate(runs)
    for t,points in data['candidate_C_profiles'].items():
        print('TRIAL',t,'SPLITS',len(runs),'C dose response (mm; low better)')
        for x in points:
            print(f'  C={x["C"]:.6g} mean-RMSE {np.round(x["mean_of_split_RMSE_mm"],4)} '
                  f'beneficial_vs_C0 {x["splits_better_than_C0_by_target"]}/{len(runs)}')
    if a.out:
        dose.write_new(a.out,data)
        print('READ-ONLY DOSE SUMMARY SAVED',Path(a.out).resolve())
    print('ALL DATA DEV-ONLY; NO MODEL TRAINING HERE; NO NEW INDEPENDENT TEST')


if __name__=='__main__':main()

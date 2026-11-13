"""Ringwise, paired SSR ablation audit. Does NOT retrain or access raw/test rings.

Reads the output of audit_unlabeled_ablation.py. Descriptive audit only:
 - RMSE / MAE in original mm from exactly 8 held-out inner-development rings;
 - two distinct sources of 'unlabeled gain': preprocessing and actual ELBO U contribution;
 - paired per-ring change in squared errors and win counts;
 - leave-one-validation-ring-out from SCORING ONLY (not model fitting / CV).

No p-values and no test data. Many previously viewed inner validation rings:
no unbiased model-selection or uncertainty/generalization claim.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

MODES=('SUP_L_ONLY','C0_SHARED_PREPROC','SSL_U_C1')
MODELS=('FULL_STATE','HYBRID_SIX','PHYS_NIG_TWELVE')
COMPARISONS=(('SUP_L_ONLY','C0_SHARED_PREPROC','U_PREPROC_AND_INIT'),
             ('C0_SHARED_PREPROC','SSL_U_C1','U_ELBO_ONLY'),
             ('SUP_L_ONLY','SSL_U_C1','OVERALL_U'))
TARGETS=('Aussenunrundheit','Hoehenschlag')
OUTER=('DM07','DM08','DM11','DM15')


def validate_report(report):
    if tuple(report['outer_test_locked'])!=OUTER:
        raise ValueError('Outer test IDs do not match original seal')
    if len(report['folds'])!=4:
        raise ValueError('Exactly four existing inner CV folds expected')
    all_val=[]
    for fold in report['folds']:
        L=set(fold['train_L']);U=set(fold['train_U']);V=set(fold['inner_val'])
        if len(L)!=6 or len(U)!=4 or len(V)!=2:
            raise ValueError('Unexpected ring count in fold')
        if L&U or L&V or U&V or (L|U|V)&set(OUTER):
            raise ValueError('Fold leakage detected')
        if set(fold['modes'])!=set(MODES):
            raise ValueError('Wrong model modes')
        for mode in MODES:
            dat=fold['modes'][mode]
            if set(dat['quality_fit_ids'])!=L:
                raise ValueError(f'Quality scaler leakage: fold {fold["fold"]}')
            expected=L if mode=='SUP_L_ONLY' else L|U
            if set(dat['x_fit_ids'])!=expected:
                raise ValueError(f'X scaler leakage: fold {fold["fold"]}, mode {mode}')
            if abs(float(dat['unlabeled_training_weight'])-(1 if mode=='SSL_U_C1' else 0))>1e-12:
                raise ValueError('Wrong U training weight')
            if set(dat['per_model'])!=set(MODELS):
                raise ValueError('Wrong models in mode')
            for model in MODELS:
                entries=dat['per_model'][model]
                if len(entries)!=2 or {x['ring'] for x in entries}!=V:
                    raise ValueError('Validation ring mismatch')
                for row in entries:
                    if not np.all(np.isfinite(row['residual_mm'])) or len(row['residual_mm'])!=2:
                        raise ValueError('Bad numeric residual')
        all_val.extend(V)
    if len(all_val)!=8 or len(set(all_val))!=8:
        raise ValueError('CV validation rings must be eight unique rings')
    return tuple(sorted(all_val))


def errors(report, mode, model, val_ids):
    lut={v['ring']:v['residual_mm']
         for fold in report['folds']
         for v in fold['modes'][mode]['per_model'][model]}
    if set(lut)!=set(val_ids):raise ValueError('Missing/extra predictions')
    a=np.asarray([lut[x] for x in val_ids],dtype=float)
    if a.shape!=(8,2):raise ValueError('Wrong residual matrix dimensions')
    return a


def score(E):
    return {'rmse_mm':np.sqrt(np.mean(E**2,axis=0)).tolist(),
            'mae_mm':np.mean(np.abs(E),axis=0).tolist()}


def paired_comparison(E_before, E_after, ids):
    """Negative delta MSE means the second mode is better (on these 8 rings)."""
    B=np.asarray(E_before,float);A=np.asarray(E_after,float)
    if B.shape!=(8,2) or A.shape!=(8,2):raise ValueError('Exactly 8x2 errors')
    delta=A*A-B*B
    loo=np.stack([(delta.sum(axis=0)-delta[i])/7 for i in range(8)])
    improved=delta<0
    worse=delta>0
    sumdelta=delta.sum(axis=0)
    return {'change_mse_mm2':np.mean(delta,axis=0).tolist(),
            'improved_ring_count':np.sum(improved,axis=0).tolist(),
            'worse_ring_count':np.sum(worse,axis=0).tolist(),
            'unchanged_ring_count':(8-np.sum(improved,axis=0)-np.sum(worse,axis=0)).tolist(),
            'scoring_loo_delta_mse_range_mm2':{'min':np.min(loo,axis=0).tolist(),
                                              'max':np.max(loo,axis=0).tolist()},
            'scoring_loo_gain_has_same_sign_as_all':[
                bool(np.all(np.sign(loo[:,j])==np.sign(sumdelta[j]))) if abs(sumdelta[j])>0 else bool(np.all(loo[:,j]==0))
                for j in range(2)],
            'per_ring': [{'ring':rid,
                          'err_before_mm':B[i].tolist(),
                          'err_after_mm':A[i].tolist(),
                          'delta_squared_error_mm2':delta[i].tolist()} for i,rid in enumerate(ids)]}


def audit(report):
    ids=validate_report(report)
    results={}
    for model in MODELS:
        err={mode:errors(report,mode,model,ids) for mode in MODES}
        comparisons={title:paired_comparison(err[a],err[b],ids) for a,b,title in COMPARISONS}
        results[model]={'mode_scores':{mode:score(E) for mode,E in err.items()},
                        'comparisons':comparisons}
    return {'kind':'DESCRIPTIVE paired ablation audit - existing folds - NO retraining',
            'target_order':TARGETS, 'validation_rings':ids,
            'outer_test_locked':OUTER,
            'warning':'Scoring-LOO only drops one ring when computing aggregate results; never retrains the model. Development rings already used repeatedly. No confirmatory p values or generalization claim.',
            'comparison_definitions':{
                'U_PREPROC_AND_INIT':'SUP_L_ONLY -> C0_SHARED_PREPROC; multiple preprocessing/initialization factors change simultaneously',
                'U_ELBO_ONLY':'C0_SHARED_PREPROC -> SSL_U_C1; identical input scaling, initial split and seeds, C differs',
                'OVERALL_U':'SUP_L_ONLY -> SSL_U_C1; not isolated single factor'},
            'model_audits':results}


def save(audit_result,out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Nonempty directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    j=out/'SSR_ROBUSTHEIT.json'
    j.write_text(json.dumps(audit_result,indent=2,ensure_ascii=False),encoding='utf-8')
    c=out/'SSR_RING_DIFFS.csv'
    with c.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f,delimiter=';')
        w.writerow(('model','comparison','ring','target','before_error_mm','after_error_mm','delta_squared_error_mm2'))
        for model,dat in audit_result['model_audits'].items():
            for comp,results in dat['comparisons'].items():
                for row in results['per_ring']:
                    for i,target in enumerate(TARGETS):
                        w.writerow((model,comp,row['ring'],target,row['err_before_mm'][i],row['err_after_mm'][i],row['delta_squared_error_mm2'][i]))
    return j,c


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',default='results/ssr_ablation_cv4/SSR_ABLATION_CV4.json')
    ap.add_argument('--out',default='results/ssr_robustheit_cv4')
    a=ap.parse_args()
    r=json.loads(Path(a.report).read_text(encoding='utf-8'))
    d=audit(r);js,csvp=save(d,a.out)
    print('Gleiches bestehendes CV4: 8 gelabelte Validierungsringe, nur Ergebnisse gelesen; KEIN Retraining')
    for model,dat in d['model_audits'].items():
        print('\n',model)
        for mode,met in dat['mode_scores'].items():
            print(' ',mode,'RMSE mm',np.round(met['rmse_mm'],4))
        for cmp,met in dat['comparisons'].items():
            print(' ',cmp,'Δ MSE(mm²)',np.round(met['change_mse_mm2'],4),
                  'besser von 8',met['improved_ring_count'],
                  'LOO-Vorzeichen stabil',met['scoring_loo_gain_has_same_sign_as_all'])
    print('OUTER-TEST GESPERRT',OUTER)
    print('BERICHT',js.resolve());print('RINGDIFFERENZEN',csvp.resolve())

if __name__=='__main__': main()

"""Vier vorab festgelegte innere CV-Ringpaare, äußere vier TEST-Ringe bleiben gesperrt.

Jeder der acht ursprünglich gelabelten Entwicklungsringe wird genau einmal intern
validiert; 6 L + 4 U trainieren in jedem Fold. Noch KEINE nested äußere Evaluation.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--local-steps',type=int,default=8)
    ap.add_argument('--out',default='results/covariance_cv4_v1_f2')
    ap.add_argument('--features',default='V1_Kern_6',choices=['V1_Kern_6','V2_Dynamik_10','V3_Zentrierung_12','V4_Umformung_10'])
    ap.add_argument('--window',default='F2',choices=['F0','F1','F2'])
    ap.add_argument('--K',type=int,default=3)
    ap.add_argument('--C',type=float,default=1.)
    ap.add_argument('--seed',type=int,default=20261111)
    ap.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    folds=[]
    script=Path(__file__).parent/'compare_covariance_dev.py'
    for fold in range(4):
        target=out/f'fold_{fold+1}'
        cmd=[sys.executable,str(script),'--fold-index',str(fold),'--epochs',str(args.epochs),
             '--local-steps',str(args.local_steps),'--out',str(target),'--features',args.features,
             '--window',args.window,'--K',str(args.K),'--C',str(args.C),'--seed',str(args.seed),
             '--data',args.data,'--models','both']
        print('\n=== INNER FOLD',fold+1,'/4 ===',flush=True)
        subprocess.run(cmd,check=True)
        folds.append(json.loads((target/'covariance_inner_comparison.json').read_text(encoding='utf-8')))
    reference=folds[0]['split']['outer_test_locked']
    Lvals=[]
    for f in folds:
        assert f['split']['outer_test_locked']==reference
        assert not (set(f['split']['inner_val']) & set(f['split']['train_L']))
        Lvals.extend(f['split']['inner_val'])
    assert len(Lvals)==len(set(Lvals))==8
    # Einfache train-L-Mittelwert-Baseline, identisch foldlokal und ohne TEST-Zugriff.
    import dm_data as dm
    raw=dm.load_raw(args.data,expected_sha='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5')
    baseline_sq=[];baseline_sq_mm=[]
    for f in folds:
        train_y=np.stack([[float(raw[r]['zielwerte'][t]) for t in dm.TARGETS] for r in f['split']['train_L']])
        mu=train_y.mean(0);sd=train_y.std(0);sd=np.where(sd>1e-10,sd,1.)
        for r in f['split']['inner_val']:
            yt=np.array([float(raw[r]['zielwerte'][t]) for t in dm.TARGETS])
            baseline_sq.append(((yt-mu)/sd)**2)
            baseline_sq_mm.append((yt-mu)**2)
    baseline={'normalized_rmse_per_target':np.sqrt(np.mean(baseline_sq,axis=0)).tolist(),
              'rmse_mm_per_target':np.sqrt(np.mean(baseline_sq_mm,axis=0)).tolist()}
    print('Trainingsmittelwert-BASELINE CV4 NRMSE',np.round(baseline['normalized_rmse_per_target'],4),
          '| RMSE mm',np.round(baseline['rmse_mm_per_target'],4))
    metrics={}
    for model in ('diag','full'):
        rows=[row for f in folds for row in f['comparison'][model]['inner_val']]
        assert len(rows)==8
        e=np.array([r['error_norm_train_sd'] for r in rows])
        # Normierung variiert je nach innerem Trainingsfold, wie bei CV üblich.
        metrics[model]={'normalized_rmse_per_target':np.sqrt(np.mean(e**2,axis=0)).tolist(),
                        'mean_X_logp_approx_per_timepoint_per_ring':float(np.mean([r['X_posterior_predictive_logp_approx_per_step'] for r in rows])),
                        'rmse_mm_per_target':np.sqrt(np.mean([(np.asarray(r['y_pred'])-np.asarray(r['y_true']))**2 for r in rows],axis=0)).tolist(),
                        'n_inner_validated_rings':len(rows),
                        'per_ring':rows}
        print(model,'CV4 NRMSE Außen,Höhenschlag',np.round(metrics[model]['normalized_rmse_per_target'],4),
              '| mean approx X log p/time/ring',round(metrics[model]['mean_X_logp_approx_per_timepoint_per_ring'],4))
    report={'status':'4-FOLD INNER CV; KEIN ÄUSSERES GENERALISIERUNGSERGEBNIS',
            'config':vars(args),'outer_test_locked':reference,
            'method':'8 ursprünglich gelabelte Entwicklungsringe in 4 disjunkte VAL-Paare; 4 U bleiben U. '
                     'Alle Datenumformungen foldlokal, mit Training-L und Training-U für X; nur Training-L für Y.',
            'note':'K und Merkmale wurden nicht durch diese CV ausgewählt. X log score ist nur approximiert '
                   '(Posterior-Mittel der Dirichlet-Übergänge, Posterior-predictive Student-t-Emissionen). '
                   'Keine hyperparametrische Suche / äußere Eval. Zeitpunkte korrelieren.',
            'metrics':metrics,'baseline':baseline}
    (out/'CV4_ZUSAMMENFASSUNG.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('CV4-Bericht:',(out/'CV4_ZUSAMMENFASSUNG.json').resolve())
    print('OUTER-TEST gesperrt:',reference)

if __name__=='__main__':main()

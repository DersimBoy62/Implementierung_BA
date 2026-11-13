"""TRAIN-ONLY Vergleich: diagonale NIG vs vollständige Gauß-Wishart Emissionen.

Ausschließlich innerhalb der 12 Entwicklungsringe: 2 aus ehemals L als INNER-VAL,
6 verbleibende L + 4 U sind Trainingsbasis. Die vier äußeren TEST-Ringe werden
weder skaliert noch trainiert noch ausgewertet. Keine Optuna-Optimierung.

Ergebnisse sind eine einzige, explorative innere Aufteilung und KEINE unabhängige
abschließende Generalisierungsaussage.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import dm_data as dm
import vb_b1 as vb
import fullcov_b1 as fc

SOURCE_SHA='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    a.add_argument('--out',default='results/covariance_inner_dev_v1_f2')
    a.add_argument('--features',choices=list(dm.SETS),default='V1_Kern_6')
    a.add_argument('--window',choices=['F0','F1','F2'],default='F2')
    a.add_argument('--K',type=int,default=3)
    a.add_argument('--epochs',type=int,default=3)
    a.add_argument('--C',type=float,default=1.)
    a.add_argument('--local-steps',type=int,default=8)
    a.add_argument('--seed',type=int,default=20261111)
    a.add_argument('--models',choices=['diag','full','both'],default='both')
    a.add_argument('--fold-index',type=int,choices=[0,1,2,3],default=3)
    args=a.parse_args()
    raw=dm.load_raw(args.data,expected_sha=SOURCE_SHA)
    split=dm.ring_split(raw,seed=args.seed)
    # Nur Entwicklungsringe überhaupt in Merkmals-View übernehmen
    dev_ids=set(split['L'])|set(split['U'])
    assert dev_ids.isdisjoint(split['TEST_GESPERRT']) and len(dev_ids)==12
    dataset=dm.view({rid:raw[rid] for rid in sorted(dev_ids)},args.features,args.window)
    dm.refuse_post_quality_inputs(dataset)
    # Vorab deterministisch reservierte 2 Ringe, ohne Targetwerte anzusehen
    inner_val=tuple(sorted(split['L'])[2*args.fold_index:2*args.fold_index+2])
    train_L=tuple(i for i in split['L'] if i not in inner_val)
    train_U=tuple(split['U'])
    train_ids=tuple(sorted(set(train_L)|set(train_U)))
    assert len(train_L)==6 and len(train_U)==4 and len(train_ids)==10
    assert not (set(inner_val)&set(train_ids))
    xscale=dm.fit_x_scaler(dataset,train_ids)
    yscale=dm.fit_y_scaler(dataset,train_L)
    X={rid:xscale.transform(dataset[rid].x) for rid in train_ids}
    Y={rid:yscale.transform(dataset[rid].y) for rid in train_L}
    val_X={rid:xscale.transform(dataset[rid].x) for rid in inner_val}
    models=['diag','full'] if args.models=='both' else [args.models]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    results={}
    print('Train L',train_L,'Train U',train_U,'INNER-VAL',inner_val,
          'OUTER-TEST GESPERRT',split['TEST_GESPERRT'])
    print('Nur train angepasst: X-Scaler',xscale.fitted_ids,'Y-Scaler',yscale.fitted_ids)
    for model in models:
        print('\nTrainiere',model,'Emissionen...')
        if model=='diag':
            fit=vb.train(X,Y,K=args.K,C=args.C,epochs=args.epochs,seed=args.seed,
                         mode='FULL_JOINT',max_local_iter=args.local_steps,verbose=True)
            predict=vb.predict; score=fc.predictive_logx_diag
        else:
            fit=fc.train_full(X,Y,K=args.K,C=args.C,epochs=args.epochs,seed=args.seed,
                              max_local_iter=args.local_steps,verbose=True)
            predict=fc.predict_full; score=fc.predictive_logx_full
        rows=[]
        for rid in inner_val:
            m,sd,_,_=predict(val_X[rid],fit['process'],fit['quality'])
            pred=m*yscale.scale+yscale.mean
            actual=dataset[rid].y
            # pro Zielgröße separat, normiert mit den TRAIN-L-Standardabweichungen
            normerr=(pred-actual)/yscale.scale
            ll=float(score(val_X[rid],fit['process']))
            rows.append({'ring':rid,'T':len(val_X[rid]),'y_true':actual.tolist(),
                         'y_pred':pred.tolist(),'error_norm_train_sd':normerr.tolist(),
                         'X_posterior_predictive_logp_approx_per_step':ll/len(val_X[rid])})
            print(' INNER-VAL',rid,'quality normalized residual',np.round(normerr,4),
                  'X approximate log-score / step',round(ll/len(val_X[rid]),4))
        errors=np.array([row['error_norm_train_sd'] for row in rows])
        nrmse=np.sqrt((errors**2).mean(axis=0))
        mXscore=float(np.mean([row['X_posterior_predictive_logp_approx_per_step'] for row in rows]))
        results[model]={'history':fit['history'], 'inner_val':rows,
                        'inner_nrmse_for_2_targets':nrmse.tolist(),
                        'mean_inner_X_logp_per_step_approx':mXscore}
        print('Ergebnis',model,'INNER-NRMSE',np.round(nrmse,4),
              'mean inner X log p per step',round(mXscore,4))
    report={'scope':'EXPLORATIVER INNER-VAL MODELLVERGLEICH (nur 2 Ringe)',
            'warning':'Kein äußerer Test, keine Optuna-Suche, kein valides finales Modellranking. '
                      'X-Logscore nutzt Dirichlet-Posterior-MITTEL statt Integration der HMM-Übergangsparameter.',
            'source_sha256':SOURCE_SHA,'config':vars(args),
            'split':{'train_L':train_L,'train_U':train_U,'inner_val':inner_val,
                     'outer_test_locked':split['TEST_GESPERRT']},
            'scaler_fit_X_ids':xscale.fitted_ids,'scaler_fit_Y_ids':yscale.fitted_ids,
            'features':dataset[train_L[0]].names,'comparison':results}
    (out/'covariance_inner_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nBericht:',(out/'covariance_inner_comparison.json').resolve())
    print('ÄUSSERE TESTRINGE NICHT EVALUIERT; Ergebnisse auf 2 inneren VAL-Ringen explorativ.')

if __name__=='__main__':main()

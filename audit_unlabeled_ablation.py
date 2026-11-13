"""Harte SSR-Ablation für die DM-Bachelorarbeit: Wo helfen ungelabelte Ringe?

Drei Zustände der Datenverwendung:
  SUP_L_ONLY:          X/Y-Skalierung nur TRAIN-L, Initialisierung nur L, U völlig absent.
  C0_SHARED_PREPROC:   X/phys-Skalierung und KMeans mit TRAIN L+U, aber C=0 im ELBO.
  SSL_U_C1:            TRAIN L+U für X/Initialisierung und C=1 im ELBO.

Alle vier inneren Folds, nur original train/val L/U, outer test rigoros ausgeschlossen.
Keines der U-Labels wird im Training der Qualität gelesen. Die drei Pipelines
sind nicht kausal isolierte Ein-Faktor-Experimente; der C0_SHARED_PREPROC-vs-C1
Vergleich hält jedoch Preprocessing/Initialisierung konstant.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
import fullcov_b1 as full
import hybrid_b1 as hybrid
from quality_link_challenge import (DATA_SHA,FIXED_SEED,LOCKED,
                                    extract_descriptors,normal_inverse_gamma,nig_predict)

MODES=('SUP_L_ONLY','C0_SHARED_PREPROC','SSL_U_C1')
MODELS=('FULL_STATE','HYBRID_SIX','PHYS_NIG_TWELVE')
HYBRID_INDICES=(0,1,2,5,6,7)

def group_specs(s,fold):
    if fold not in (1,2,3,4):raise ValueError('fold 1..4')
    val=tuple(sorted(s['L'])[2*(fold-1):2*fold])
    L=tuple(r for r in s['L'] if r not in val)
    U=tuple(s['U'])
    assert len(L)==6 and len(U)==4 and len(val)==2
    assert not (set(L)|set(U))&set(val)
    assert not (set(L)|set(U)|set(val))&set(s['TEST_GESPERRT'])
    return L,U,val


def evaluate(data, out, folds=(1,2,3,4), epochs=10, steps=8, K=3):
    raw=dm.load_raw(data,expected_sha=DATA_SHA)
    split=dm.ring_split(raw,seed=FIXED_SEED)
    if tuple(split['TEST_GESPERRT'])!=LOCKED:raise AssertionError('Outer test mismatch')
    dev=tuple(sorted(set(split['L'])|set(split['U'])))
    rings=dm.view({r:raw[r] for r in dev},'V1_Kern_6','F2')
    dm.refuse_post_quality_inputs(rings)
    path=Path(out)
    if path.exists() and any(path.iterdir()):raise FileExistsError('Out nonempty '+str(path))
    path.mkdir(parents=True,exist_ok=True)
    rows={m:{a:[] for a in MODELS} for m in MODES}
    baseline=[]; fold_reports=[]
    for fold in folds:
        L,U,val=group_specs(split,fold)
        sy=dm.fit_y_scaler({r:rings[r] for r in L},L)
        fold_rep={'fold':fold,'train_L':L,'train_U':U,'inner_val':val,'outer_test_locked':LOCKED,'modes':{}}
        for r in val:
            baseline.append({'ring':r,'fold':fold,'residual_mm':(sy.mean-rings[r].y).tolist()})
        for mode in MODES:
            tr=L if mode=='SUP_L_ONLY' else tuple(sorted((*L,*U)))
            sx=dm.fit_x_scaler({r:rings[r] for r in tr},tr)
            X={r:sx.transform(rings[r].x) for r in (*tr,*val)}
            Y={r:sy.transform(rings[r].y) for r in L}
            features={r:extract_descriptors(X[r],rings[r].t,'physical') for r in (*tr,*val)}
            physics6={r:features[r][list(HYBRID_INDICES)] for r in features}
            scale6=StandardScaler().fit(np.array([physics6[r] for r in tr]))
            six={r:scale6.transform(physics6[r][None,:])[0] for r in physics6}
            scale12=StandardScaler().fit(np.array([features[r] for r in tr]))
            twelve={r:scale12.transform(features[r][None,:])[0] for r in features}
            assert set(Y)==set(L)
            C=1. if mode=='SSL_U_C1' else 0.
            # C0_SHARED_PREPROC: all train X passed, but global q updates weight U=0.
            x_train={r:X[r] for r in tr}
            print(f'FOLD {fold} MODE {mode} fitX {len(tr)} labeled {len(L)} unlabeled weight {C}',flush=True)
            fullmod=full.train_full(x_train,Y,K=K,C=C,epochs=epochs,
                                   seed=FIXED_SEED,max_local_iter=steps)
            hybridmod=hybrid.train(x_train,Y,{r:six[r] for r in tr},K=K,C=C,
                                   epochs=epochs,seed=FIXED_SEED,steps=steps)
            nig=normal_inverse_gamma(np.array([twelve[r] for r in L]),
                                     np.array([Y[r] for r in L]))
            valrows={a:[] for a in MODELS}
            for rid in val:
                fmu,fsd,_,_=full.predict_full(X[rid],fullmod['process'],fullmod['quality'])
                hmu,hsd=hybrid.predict(X[rid],six[rid],hybridmod)
                nmu,nsd=nig_predict(nig,twelve[rid][None,:])
                preds={'FULL_STATE':(fmu,fsd),
                       'HYBRID_SIX':(hmu,hsd),
                       'PHYS_NIG_TWELVE':(nmu[0],nsd[0])}
                for mn,(mu,sd) in preds.items():
                    yhat=mu*sy.scale+sy.mean
                    rr={'ring':rid,'fold':fold,'true_mm':rings[rid].y.tolist(),
                        'pred_mm':yhat.tolist(),'residual_mm':(yhat-rings[rid].y).tolist(),
                        'predictive_sd_mm':(sd*sy.scale).tolist()}
                    rows[mode][mn].append(rr);valrows[mn].append(rr)
                    print(' ',mn,rid,'residual(mm)',np.round(rr['residual_mm'],3),flush=True)
            fold_rep['modes'][mode]={
                'x_fit_ids':sx.fitted_ids, 'physics6_fit_ids':tr,'physics12_fit_ids':tr,
                'quality_fit_ids':sy.fitted_ids, 'unlabeled_training_weight':C,
                'per_model':valrows,
                'final_elbo':{'FULL_STATE':fullmod['history'][-1]['elbo'],
                              'HYBRID_SIX':hybridmod['history'][-1]['elbo']}
            }
        fold_reports.append(fold_rep)
    def scores(rr):
        z=np.asarray([r['residual_mm'] for r in rr],dtype=float)
        return {'rmse_mm':np.sqrt(np.mean(z*z,axis=0)).tolist(),
                'mae_mm':np.mean(np.abs(z),axis=0).tolist(),
                'n_val_rings':len(rr)}
    metrics={mode:{mn:scores(rows[mode][mn]) for mn in MODELS} for mode in MODES}
    rep={'kind':'exploratory strict supervised-vs-semisupervised ablation; NO test evaluation',
         'caution':'CV fold revisitation; not unbiased model-selection evidence; scaling and kmeans change SUP_L_ONLY vs other modes',
         'config':{'K':K,'epochs':epochs,'local_steps':steps,'data':str(data),'feature_set':'V1_Kern_6','window':'F2'},
         'modes':list(MODES),'models':list(MODELS),'outer_test_locked':LOCKED,
         'baseline':scores(baseline),'metrics':metrics,'folds':fold_reports}
    (path/'SSR_ABLATION_CV4.json').write_text(json.dumps(rep,indent=2,ensure_ascii=False),encoding='utf8')
    print('\nBASELINE:',np.round(rep['baseline']['rmse_mm'],4),flush=True)
    for mode in MODES:
        for model in MODELS:
            print(mode,model,'RMSE mm',np.round(metrics[mode][model]['rmse_mm'],4),flush=True)
    print('OUTER TEST LOCKED',LOCKED,flush=True)
    print('REPORT',(path/'SSR_ABLATION_CV4.json').resolve(),flush=True)
    return rep

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ap.add_argument('--out',default='results/ssr_ablation_cv4')
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--local-steps',type=int,default=8)
    ap.add_argument('--K',type=int,default=3)
    ap.add_argument('--folds',type=int,nargs='+',default=[1,2,3,4],choices=[1,2,3,4])
    a=ap.parse_args()
    evaluate(a.data,a.out,folds=tuple(a.folds),epochs=a.epochs,steps=a.local_steps,K=a.K)

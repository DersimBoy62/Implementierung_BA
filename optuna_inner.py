"""Optuna-Innenvalidierung (nicht automatisch starten).

Der eingefrorene äußere TEST-Split bleibt unberührt. Der Code bildet einen
kategorialen Suchraum für Fenstervariante, Features, Zustandszahl, C und Priors.
Dies ist ein **erster** innerer Suchlauf, noch keine vollständige Nested-CV-
Auswertung: Das spätere äußere K-fold-Protokoll fehlt bewusst noch.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from dm_data import load_raw,ring_split,view,fit_x_scaler,fit_y_scaler,refuse_post_quality_inputs
from vb_b1 import train,predict


def inner_score(raw,outer_split,conf,seed=20261111,inner_folds=3,epochs=3):
    ids=tuple(sorted(set(outer_split['L'])|set(outer_split['U'])))
    if set(ids)&set(outer_split['TEST_GESPERRT']):raise AssertionError('Leakage outer TEST')
    rings=view(raw,conf['features'],conf['window'])
    refuse_post_quality_inputs(rings)
    kf=KFold(n_splits=inner_folds,shuffle=True,random_state=seed)
    fold_scores=[]
    for fold,(tr,va) in enumerate(kf.split(ids)):
        train_ids=[ids[i] for i in tr];val_ids=[ids[i] for i in va]
        rng=np.random.default_rng(seed+fold)
        perm=list(rng.permutation(train_ids))
        nl=max(3,int(np.ceil(.7*len(perm))))
        L=perm[:nl];U=perm[nl:]
        fit_ids=list(L)+list(U) if conf['C']>0 else list(L)
        # Nur zulässige Trainingsringe werden im Scaler verwendet.
        xs=fit_x_scaler(rings,fit_ids);ys=fit_y_scaler(rings,L)
        X={i:xs.transform(rings[i].x) for i in fit_ids}
        Y={i:ys.transform(rings[i].y) for i in L}
        out=train(X,Y,K=conf['K'],C=conf['C'],epochs=epochs,
                  seed=seed+fold,mode=conf.get('mode','OFF'),
                  prior_options={'transition_diag_boost':conf['transition_diag_boost'],
                                 'emission_kappa':conf['emission_kappa'],
                                 'quality_prior_scale':conf['quality_prior_scale']})
        errors=[]
        for i in val_ids:
            # y aus dem Validierungsring wird NUR zur Bewertung gelesen.
            pred,_,_,_=predict(xs.transform(rings[i].x),out['process'],out['quality'])
            reference=ys.transform(rings[i].y)
            errors.append((pred-reference)**2)
        sq=np.mean(errors,axis=0)
        fold_scores.append(float(np.sqrt(sq).mean()))
    return float(np.mean(fold_scores)),fold_scores


def objective(trial, raw,split,seed,epochs):
    conf={
        'features':trial.suggest_categorical('features',['V1_Kern_6','V2_Dynamik_10','V3_Zentrierung_12','V4_Umformung_10']),
        'window':trial.suggest_categorical('window',['F1','F2']),
        'K':trial.suggest_int('K',2,5),
        'C':trial.suggest_categorical('C',[0.,0.5,1.]),
        'transition_diag_boost':trial.suggest_categorical('transition_diag_boost',[2.,4.,8.]),
        'emission_kappa':trial.suggest_categorical('emission_kappa',[.1,.2,1.]),
        'quality_prior_scale':trial.suggest_categorical('quality_prior_scale',[.5,2.,5.]),
        'mode':'OFF' # vor umfassender JOINT-Implementierung nicht als Modellwahl-Achse optimieren.
    }
    score,folds=inner_score(raw,split,conf,seed,epochs=epochs)
    trial.set_user_attr('inner_fold_losses',folds)
    return score


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ap.add_argument('--trials',type=int,default=12)
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--seed',type=int,default=20261111)
    ap.add_argument('--run',action='store_true',help='Aktiviert Optuna-Innenvalidierung tatsächlich')
    args=ap.parse_args()
    raw=load_raw(args.data,expected_sha='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5')
    split=ring_split(raw,args.seed)
    print('Äußerer TEST-GESPERRT-Split:',split['TEST_GESPERRT'])
    if not args.run:
        print('Tuning bewusst NICHT gestartet. Optuna-Suchraum nur vorbereitet. --run zur Aktivierung.')
        return
    import optuna
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(lambda t:objective(t,raw,split,args.seed,args.epochs),n_trials=args.trials)
    path=Path('results/optuna_inner');path.mkdir(parents=True,exist_ok=True)
    (path/'best_inner_only.json').write_text(json.dumps({'config':study.best_params,
               'inner_score':study.best_value,'outer_test_untouched':list(split['TEST_GESPERRT'])},indent=2),encoding='utf-8')
    print('Beste NUR interne Konfiguration:',study.best_params,'Score',study.best_value)
    print('ACHTUNG: Keine externe Generalisierungseinschätzung, vollständige Nested-CV fehlt')

if __name__=='__main__':main()

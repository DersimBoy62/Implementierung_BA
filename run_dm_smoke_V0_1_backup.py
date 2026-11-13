"""Erster technischer Modell-SMOKETEST; KEINE Optimierung, KEIN Testbericht.

Ausführung: python run_dm_smoke.py --data input/DM_16_aktive_Ringe.pkl --epochs 2
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
from dm_data import load_raw,view,ring_split,fit_x_scaler,fit_y_scaler,refuse_post_quality_inputs,sha256
from vb_b1 import train,predict


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ap.add_argument('--out',default='results/smoke_v1_f2_k3_off_c1')
    ap.add_argument('--features',default='V1_Kern_6',choices=['V1_Kern_6','V2_Dynamik_10','V3_Zentrierung_12','V4_Umformung_10'])
    ap.add_argument('--window',default='F2',choices=['F0','F1','F2'])
    ap.add_argument('--K',type=int,default=3)
    ap.add_argument('--C',type=float,default=1.0)
    ap.add_argument('--epochs',type=int,default=2)
    ap.add_argument('--mode',choices=['OFF','TILT_JOINT','FULL_JOINT'],default='OFF')
    ap.add_argument('--local-steps',type=int,default=8)
    ap.add_argument('--seed',type=int,default=20261111)
    ap.add_argument('--transition_diag_boost',type=float,default=4.0)
    ap.add_argument('--emission_kappa',type=float,default=0.2)
    ap.add_argument('--quality_prior_scale',type=float,default=2.0)
    args=ap.parse_args()
    raw=load_raw(args.data,expected_sha='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5')
    rings=view(raw,args.features,args.window)
    refuse_post_quality_inputs(rings)
    split=ring_split(rings,seed=args.seed)
    L,U=split['L'],split['U']
    training_ids=L+U if args.C>0 else L
    assert not (set(split['TEST_GESPERRT']) & set(training_ids))
    xfit=fit_x_scaler(rings,training_ids)
    yfit=fit_y_scaler(rings,L)
    X={i:xfit.transform(rings[i].x) for i in training_ids}
    Y={i:yfit.transform(rings[i].y) for i in L}
    print('L=',L,' U=',U,' test GESPERRT=',split['TEST_GESPERRT'])
    print('Train ONLY source:',sorted(X),'label targets:',len(Y),'d:',len(rings[L[0]].names))
    result=train(X,Y,K=args.K,C=args.C,epochs=args.epochs,seed=args.seed,mode=args.mode,max_local_iter=args.local_steps,
                 prior_options={'transition_diag_boost':args.transition_diag_boost,
                                'emission_kappa':args.emission_kappa,
                                'quality_prior_scale':args.quality_prior_scale},verbose=True)
    summary={'source_sha256':sha256(args.data),
             'info':'TECHNISCHER SMOKETEST; KEINE gehaltene TEST-Metrik und KEINE optimale Modellwahl',
             'source':'A–D mathematische Struktur; TILT_JOINT ist eingeschränkte lokale Familie',
             'config':vars(args),'split':{k:list(v) for k,v in split.items()},
             'xfit_ids':list(xfit.fitted_ids),'yfit_ids':list(yfit.fitted_ids),
             'channel_names':list(rings[L[0]].names),'x_mean':xfit.mean.tolist(),'x_scale':xfit.scale.tolist(),
             'y_mean':yfit.mean.tolist(),'y_scale':yfit.scale.tolist(),
             'history':result['history']}
    # Nur Trainingsringe: Plausibilitätskontrolle, KEIN Generalisierungsergebnis.
    checks=[]
    for rid in L:
        pmean,pstd,_,_=predict(X[rid],result['process'],result['quality'])
        pred=pmean*yfit.scale+yfit.mean; sig=pstd*yfit.scale
        checks.append({'ring':rid,'y_true':rings[rid].y.tolist(),'y_train_prediction':pred.tolist(), 'predictive_std_P1_approx':sig.tolist()})
    summary['train_only_diagnostics']=checks
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'smoke_report.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    # Speichern ausschließlich geprüfter Modellparameter – nicht Ringtestdaten.
    np.savez_compressed(out/'vb_parameters.npz',pi=result['process'].pi,A=result['process'].A,mu=result['process'].mu,
                        kappa=result['process'].kappa,ax=result['process'].a,bx=result['process'].b,
                        qmu=result['quality'].mu,qV=result['quality'].V,qa=result['quality'].a,qb=result['quality'].b)
    print('Reports:',out.resolve())
    print('TEST weiterhin gesperrt – weder skaliert noch trainiert noch evaluiert')

if __name__=='__main__':main()

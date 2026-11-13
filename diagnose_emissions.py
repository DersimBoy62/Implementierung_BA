"""Prüft bedingte Korrelation und marginale Emissionsform auf TRAIN-Ringen.

Lädt ein existierendes smoke_report.json und vb_parameters.npz. Alle
Daten-Splits, Kanalgruppen und Trainings-Scaler kommen aus dem Bericht.
Die ursprünglichen Trainingsskripte und Pickles werden NICHT verändert.

Standard-Modus `process`: erneute Prozess-only Zustandsinferenz q(Z|X).
Alternativ `joint`: erneute lokale B1-Inferenz q(Z|X,Y) NUR für L-Ringe;
es handelt sich dann um eine NEUE lokale Inferenz unter finalem globalem
Posterior (nicht um exakt gespeicherte lokale q aus der letzten Epoche).
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

from dm_data import load_raw, view, refuse_post_quality_inputs, sha256
from vb_b1 import ProcPosterior, QualityPosterior, local_inference

SOURCE_SHA = 'bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'


def weighted_moments_1d(x, w):
    s = float(w.sum())
    if s <= 0: return float('nan'), float('nan'), float('nan'), float('nan')
    m = float(w @ x / s)
    v = float(w @ ((x-m)**2) / s)
    if v < 1e-15: return m, float(np.sqrt(max(v, 0))), float('nan'), float('nan')
    sig = np.sqrt(v)
    z = (x-m)/sig
    skew = float(w @ (z**3) / s)
    kurt = float(w @ (z**4) / s-3.)
    return m, float(sig), skew, kurt


def weighted_covariance(X, w):
    """State weighted pooled covariance on training steps (not independence corrected)."""
    w = np.asarray(w, dtype=np.float64)
    den = float(w.sum())
    if den <= 0: raise ValueError('Leerer Zustand')
    mu = np.einsum('n,nd->d',w,X)/den
    E = X-mu
    cov = np.einsum('n,ni,nj->ij',w,E,E,optimize=True)/den
    sd = np.sqrt(np.maximum(np.diag(cov), 0.))
    normalizer = np.outer(sd,sd)
    with np.errstate(divide='ignore',invalid='ignore'):
        corr = cov/normalizer
    corr = np.where(normalizer>1e-14,corr,np.nan)
    return mu,cov,corr,sd


def load_report_and_model(report_file):
    path=Path(report_file)
    if not path.is_file(): raise FileNotFoundError(path)
    report=json.loads(path.read_text(encoding='utf-8'))
    modelpath=path.parent/'vb_parameters.npz'
    if not modelpath.is_file(): raise FileNotFoundError(f'Fehlende Modellparameter: {modelpath}')
    with np.load(modelpath,allow_pickle=False) as data:
        p=ProcPosterior(data['pi'],data['A'],data['mu'],data['kappa'],data['ax'],data['bx'])
        q=QualityPosterior(data['qmu'],data['qV'],data['qa'],data['qb'])
    return report,p,q,modelpath


def diagnose(report_file, out, inference='process', min_abs_corr=0.8):
    report,p,q,modelpath=load_report_and_model(report_file)
    config=report['config']
    if not (0<min_abs_corr<=1): raise ValueError('min_abs_corr muss zwischen 0 und 1 liegen')
    split=report['split']
    L=set(split['L']); U=set(split['U']); test=set(split['TEST_GESPERRT'])
    train_ids=set(report['xfit_ids'])
    expected_train=L | (U if config['C']>0 else set())
    if train_ids!=expected_train or train_ids & test or L & U or (L|U)&test:
        raise AssertionError('Split- oder Scaler-Trainings-IDs inkonsistent. ABBRUCH.')
    if set(report['yfit_ids']) != L:
        raise AssertionError('Y-Scaler wurde nicht ausschließlich auf L trainiert.')
    f = Path(config['data'])
    if not f.exists() or sha256(f)!=SOURCE_SHA or report['source_sha256']!=SOURCE_SHA:
        raise RuntimeError('Datenquelle oder SHA256 nicht wie im ursprünglichen Training; ABBRUCH')
    raw=load_raw(f,expected_sha=SOURCE_SHA)
    # Aufruf von view nur fuer Trainingsringe: Testsequenzen werden nicht analysiert.
    training_raw={rid:raw[rid] for rid in sorted(train_ids)}
    data=view(training_raw,features=config['features'],window=config['window'])
    refuse_post_quality_inputs(data)
    names=tuple(report['channel_names'])
    if tuple(data[next(iter(train_ids))].names)!=names or p.mu.shape[1]!=len(names):
        raise AssertionError('Kanalzuordnung zum trainierten Modell inkonsistent')
    mean=np.asarray(report['x_mean'],dtype=float)
    scale=np.asarray(report['x_scale'],dtype=float)
    ymean=np.asarray(report['y_mean'],dtype=float)
    yscale=np.asarray(report['y_scale'],dtype=float)
    if np.any(scale<=0) or np.any(yscale<=0):raise ValueError('Ungültige Skalierung')
    ys=[]; ws=[]; ring_info=[]
    for rid in sorted(train_ids):
        x=(data[rid].x-mean)/scale
        if x.shape[1]!=len(names):raise AssertionError('Uneinheitliche Anzahl Merkmale')
        if inference=='joint' and rid in L:
            from full_markov_b1 import local_full_joint
            y=(data[rid].y-ymean)/yscale
            ret=local_full_joint(x,p,q,y,max_iter=config.get('local_steps',8))
            label='q(Z|X,Y), neu inferiert'
        else:
            ret=local_inference(x,p,mode='OFF')
            label='q(Z|X), neu inferiert'
        gamma=ret['fb'].gamma
        if not np.allclose(gamma.sum(1),1.,atol=1e-7):raise AssertionError('Gamma-Normierung')
        # C ist Teil der gewählten Trainingszielgewichtung.
        ring_weight=1. if rid in L else float(config['C'])
        ws.append(gamma*ring_weight)
        ys.append(x)
        ring_info.append({'ring_id':rid,'kind':'L' if rid in L else 'U','T':int(len(x)),
                          'posterior':label, 'state_fractions':(gamma.mean(0)).tolist()})
    XX=np.vstack(ys)
    WW=np.vstack(ws)
    K=WW.shape[1]
    d=len(names)
    pairs=[]; per_feature=[]; per_state=[]
    for k in range(K):
        w=WW[:,k]
        mu,cov,corr,sd=weighted_covariance(XX,w)
        neff=float(w.sum()**2 / np.dot(w,w))
        above=[]
        for i in range(d):
            mk,sk,skew,kurt=weighted_moments_1d(XX[:,i],w)
            per_feature.append({'state':k+1,'feature':names[i],
                                'weighted_mean':mk, 'weighted_std':sk,
                                'weighted_skewness':skew,'weighted_excess_kurtosis':kurt})
            for j in range(i+1,d):
                r=float(corr[i,j])
                is_flag=np.isfinite(r) and abs(r)>=min_abs_corr
                if is_flag:above.append((names[i],names[j],r))
                pairs.append({'state':k+1,'feature_a':names[i], 'feature_b':names[j],
                              'weighted_correlation':r, 'abs_correlation':abs(r),
                              'flag_above_threshold':is_flag})
        per_state.append({'state':k+1,'weighted_timepoint_mass':float(w.sum()),
                          'fraction_weighted_timepoints':float(w.sum()/WW.sum()),
                          'timepoint_kish_neff_not_independent_samples':neff,
                          'high_corr_pair_count':len(above),
                          'max_abs_offdiagonal_corr':float(np.nanmax(np.abs(corr-np.diag(np.diag(corr))))),
                          'state_model_mu':p.mu[k].tolist(), 'state_weighted_mean':mu.tolist(),
                          'top_pairs':[{'a':a,'b':b,'r':r} for a,b,r in sorted(above,key=lambda z:-abs(z[2]))[:12]]})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def write_csv(filename,records):
        with (out/filename).open('w',newline='',encoding='utf-8-sig') as f:
            if records:
                wr=csv.DictWriter(f,fieldnames=list(records[0].keys()),delimiter=';')
                wr.writeheader();wr.writerows(records)
    write_csv('korrelationen_nach_zustand.csv',pairs)
    write_csv('marginale_verteilungen_nach_zustand.csv',per_feature)
    # direct ring occupancy table
    write_csv('zustandsanteile_je_trainingsring.csv',[
        {'ring_id':r['ring_id'],'typ':r['kind'],'zeitpunkte':r['T'],
         **{f'zustand_{i+1}_anteil':a for i,a in enumerate(r['state_fractions'])},
         'posterior':r['posterior']} for r in ring_info])
    result={'status':'DESKRIPTIVE TRAININGS-DIAGNOSTIK, KEINE TEST-EVALUATION',
            'config':config, 'source_model_report':str(report_file),'source_model_parameters':str(modelpath),
            'posterior_kind':inference, 'posterior_explanation':{
                'process':'q(Z|X) aus dem gespeicherten finalen Prozessposterior neu berechnet. '
                          'Nicht gleich den beim letzten lokalen Schritt gespeicherten FULL_JOINT-Verteilungen.',
                'joint':'q(Z|X,Y) fuer L, q(Z|X) fuer U unter finalen globalen Posterioren neu berechnet. '
                        'Nicht identisch zu den urspruenglichen lokalen Verteilungen vor dem letzten globalen Update.'}[inference],
            'data_source_sha256':SOURCE_SHA, 'n_train_rings':len(train_ids),
            'n_train_timepoints':len(XX), 'n_test_rings_accessed_for_analysis':0,
            'no_target_values_used_as_input_features':True,
            'features':names,'threshold_abs_corr':min_abs_corr,
            'per_state':per_state,
            'warnings':['Gewichtete Zustandskorrelationen sind diagnostisch. Sie ersetzen nicht das '
                        'HMM-Modellvergleichs- oder Validierungsverfahren.',
                        'Die Zeitpunkte sind seriell abhaengig: Kish-Neff ist KEINE '
                        'Anzahl unabhaengiger Beobachtungen.',
                        'Korrelation innerhalb der erneut inferierten Zustaende ist '
                        'nicht identisch zur Guete der eigentlichen emissionsbezogenen '
                        'Gaußlikelihood; dies ist eine deskriptive Approximation.',
                        'Kein einziger der vier gesperrten Testringe wurde im Modellfit '
                        'oder in dieser Analyse verwendet.']}
    (out/'emissions_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Audit bereit:',out.resolve())
    print('Modellbericht:',report_file, '| Inferenz:',inference)
    print('Trainingsringe:',len(train_ids), '| Zeitpunkte:',len(XX),'| Testringe nicht ausgewertet:',len(test))
    for x in per_state:
        print(' Zustand',x['state'], 'Masse',round(x['fraction_weighted_timepoints'],4),
              '| Paare |r| >=',min_abs_corr, ':',x['high_corr_pair_count'])
        for row in x['top_pairs'][:4]:
            print('   ',row['a'],'/',row['b'], 'r=',round(row['r'],4))
    print('ACHTUNG: Diagnose mit neu inferierten Zustaenden, keine Test-Qualitaetsmetriken.')
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--report',default='results/full_joint_k3_e10_l8/smoke_report.json')
    ap.add_argument('--out',default='results/emissionsaudit_v1_f2_k3_e10')
    ap.add_argument('--posterior-mode',choices=['process','joint'],default='process')
    ap.add_argument('--corr-threshold',type=float,default=0.8)
    a=ap.parse_args()
    diagnose(a.report,a.out,a.posterior_mode,a.corr_threshold)

if __name__=='__main__':main()

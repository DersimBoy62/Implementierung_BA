"""Leakage-sicherer DM-Datenzugang. Nur vertrauenswürdige, eigene Pickles laden.

Nicht ausführen auf fremden Pickle-Dateien. Alle Fit-Transformationen sind foldlokal.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, pickle
from pathlib import Path
import numpy as np

TARGETS = ('ovality.ovality_radial_mm', 'planarity.height_deviation_mm')
SETS = {
    'V1_Kern_6': ('F_rad63','F_ax64','Ist_D150','Ist_s151','Ist_h152','RWG55'),
    'V2_Dynamik_10': ('F_rad63','F_ax64','Ist_D150','Ist_s151','Ist_h152','RWG55','h_punkt800','s_punkt801','nisr1289','niax1295'),
    'V3_Zentrierung_12': ('F_rad63','F_ax64','Ist_D150','Ist_s151','Ist_h152','RWG55','h_punkt800','s_punkt801','IFzE166','IFzA167','xw_ZA281','xw_ZE282'),
    'V4_Umformung_10': ('Ist_D150','Ist_s151','Ist_h152','RWG55','del_h710','del_s712','br711','ba713','h_punkt800','s_punkt801'),
}

@dataclass
class Ring:
    ring_id: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    names: tuple[str,...]

@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray
    fitted_ids: tuple[str,...]
    def transform(self, x):
        return (x-self.mean)/self.scale


def sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def load_raw(path, expected_sha=None):
    path=Path(path)
    actual=sha256(path)
    if expected_sha is not None and actual!=expected_sha:
        raise RuntimeError('Eingabe-Pickle weicht von geprüfter Quelle ab: SHA256-Mismatch')
    with path.open('rb') as f: obj=pickle.load(f)
    if len(obj)!=16 or len(set(obj))!=16 or 'DM03' in obj or 'DM13' in obj:
        raise ValueError('Erwartet genau 16 reguläre DM-Ringe, ohne DM03/DM13')
    for rid,r in obj.items():
        if not r['train_eligible'] or r['ring_id']!=rid:raise ValueError(f'Statusfehler: {rid}')
        l=r['logger']
        a=np.asarray(l['matrix'])
        if a.ndim!=2 or a.shape[1]!=115 or not np.isfinite(a).all():raise ValueError(f'Logger-Matrix: {rid}')
        if not (np.diff(a[:,0])>0).all():raise ValueError(f'Zeitachse: {rid}')
        if tuple(l['matrix_columns']) != ('time', *l['signal_names']):raise ValueError(f'Kanalnamen: {rid}')
        for t in TARGETS:
            if not np.isfinite(float(r['zielwerte'][t])):raise ValueError(f'Qualitätslabel {rid}: {t}')
    return obj


def view(raw, features='V1_Kern_6', window='F2'):
    if features not in SETS:raise KeyError('Unbekannte Kanalkonfiguration')
    if window not in ('F0','F1','F2'):raise KeyError('Unbekanntes Prozessfenster')
    names=SETS[features]; answer={}
    for rid, r in raw.items():
        l=r['logger']; a=l['matrix']; idx={s:i for i,s in enumerate(l['matrix_columns'])}
        if not all(s in idx for s in names):raise KeyError(f'Fehlende Kanäle {rid}')
        mask=np.ones(len(a),dtype=bool)
        if window in ('F1','F2'):
            mask &= np.isin(a[:,idx['phase804']],(1.,2.))
        if window=='F2':
            mask &= (a[:,idx['LaserOK']]==1) & (a[:,idx['MessEin']]==1)
        ids=np.flatnonzero(mask)
        if len(ids)<3:raise ValueError(f'Zu wenige gültige Zeitpunkte bei {rid}')
        # Kein unzulässiges Zusammenkleben getrennter Phaseninseln:
        if not np.all(np.diff(ids)==1):raise ValueError(f'Nicht zusammenhängendes {window}-Fenster bei {rid}')
        x=a[np.ix_(ids, [idx[n] for n in names])].astype(float,copy=True)
        y=np.array([float(r['zielwerte'][t]) for t in TARGETS],dtype=float)
        answer[rid]=Ring(rid,a[ids,0].copy(),x,y,tuple(names))
    return answer


def ring_split(ids, seed=20261111, test_size=4, labeled_size=8):
    rng=np.random.default_rng(seed)
    perm=list(map(str,rng.permutation(sorted(ids))))
    if len(perm)!=16 or not 0 < test_size < len(perm):raise ValueError('Split erwartet 16 Ringe')
    test=tuple(sorted(perm[:test_size]))
    dev=perm[test_size:]
    if labeled_size<2 or labeled_size>=len(dev):raise ValueError('Unzulässige Labelanzahl')
    return {'L':tuple(sorted(dev[:labeled_size])), 'U':tuple(sorted(dev[labeled_size:])), 'TEST_GESPERRT':test}


def fit_x_scaler(rings, allowed_ids):
    ids=tuple(sorted(allowed_ids))
    if not ids:raise ValueError('Leerer X-Fit')
    xs=np.vstack([rings[i].x for i in ids])
    mu=xs.mean(axis=0)
    std=xs.std(axis=0)
    std=np.where(std>1e-10,std,1.0)
    return Scaler(mu,std,ids)


def fit_y_scaler(rings, labeled_ids):
    ids=tuple(sorted(labeled_ids))
    a=np.stack([rings[i].y for i in ids])
    mu=a.mean(axis=0)
    std=a.std(axis=0)
    return Scaler(mu,np.where(std>1e-10,std,1.0),ids)


def refuse_post_quality_inputs(rings):
    for r in rings.values():
        if any('ovality.' in s or 'planarity.' in s or 'diameter.' in s for s in r.names):
            raise AssertionError('Datenleckage! Messauswertung in X enthalten')
        if r.y.shape!=(2,):raise AssertionError('Nur zwei Targets erlaubt')

"""Synthetic guardrail checks for provisional final-driver wrapper, no ring pickle loaded."""
from __future__ import annotations
import csv, json, pathlib, tempfile
import numpy as np
import main_bayesian_ssr_v1 as m


def test(name,fn):
    fn();print('PASS',name)

def t_config_and_hashes():
    c=m.load_config()
    assert c['freeze_status']=='PROVISIONAL_USER_APPROVAL_PENDING'
    assert m.FEATURE_COLUMNS == tuple(c['input_columns'])
    assert c['K']==2 and c['C']==0.2


def t_no_restricted_prediction():
    with tempfile.TemporaryDirectory() as temp:
        for rid in (*m.LABELS,*m.U_ONLY,*m.LOCKED):
            try:m.predict_one('unused',rid,'unused','x.json')
            except ValueError:pass
            else:raise AssertionError(f'Accepted historical ring {rid}')


def t_csv_gate_and_features():
    cols=['time',*m.FEATURE_COLUMNS,'phase804','LaserOK','MessEin']
    with tempfile.TemporaryDirectory() as temp:
        path=pathlib.Path(temp)/'ring.csv'
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(cols)
            for i in range(10):
                w.writerow([i*.1, 10+i, 20+i, 600+i, 60+i*.1, 55+i*.01, .5+i*.03, 1 if i<6 else 6,1,1])
        t,x=m.extract_new_ring_csv(path)
        assert t.shape==(6,) and x.shape==(6,6)
        assert x[0,0]==10


def t_reject_noncontiguous_window():
    cols=['time',*m.FEATURE_COLUMNS,'phase804','LaserOK','MessEin']
    with tempfile.TemporaryDirectory() as temp:
        path=pathlib.Path(temp)/'ring.csv'
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(cols)
            for i in range(10):
                w.writerow([i*.1,*([i+1]*6),1 if i!=4 else 6,1,1])
        try:m.extract_new_ring_csv(path)
        except ValueError:pass
        else:raise AssertionError('Noncontiguous F2 window accepted')


def t_reject_quality_leakage():
    cols=['time',*m.FEATURE_COLUMNS,'phase804','LaserOK','MessEin','ovality.ovality_radial_mm']
    with tempfile.TemporaryDirectory() as temp:
        path=pathlib.Path(temp)/'ring.csv'
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(cols)
            for i in range(10):w.writerow([i*.1,*([i+1]*6),1,1,1,0.01])
        try:m.extract_new_ring_csv(path)
        except ValueError:pass
        else:raise AssertionError('Quality leakage not rejected')


def t_atomic_no_replace():
    with tempfile.TemporaryDirectory() as temp:
        p=pathlib.Path(temp)/'result.json'
        m._write_new_json(p,{'value':1})
        try:m._write_new_json(p,{'value':2})
        except FileExistsError:pass
        else:raise AssertionError('Overwrote existing research output')
        assert json.loads(p.read_text())=={'value':1}


def t_core_descriptor_head_map():
    cfg=m.load_config()
    assert [list(x) for x in m.OPTIONS['SEPARATE_AXIAL']]==[cfg['physical_descriptor_indices']['outside'],cfg['physical_descriptor_indices']['height']]
    t=np.linspace(0,12,20);x=np.zeros((20,6));x[:,0]=t;x[:,2]=100+2*t
    d=m.extract_descriptors(x,t,'physical')
    assert d.shape==(12,) and np.all(np.isfinite(d))


def t_gated_json_hash():
    with tempfile.TemporaryDirectory() as temp:
        p=pathlib.Path(temp)/'a.json';m._write_new_json(p,{'a':np.float64(2).item()})
        h=m.sha(p);assert len(h)==64


if __name__=='__main__':
    for fn in (t_config_and_hashes,t_no_restricted_prediction,t_csv_gate_and_features,
               t_reject_noncontiguous_window,t_reject_quality_leakage,t_atomic_no_replace,
               t_core_descriptor_head_map,t_gated_json_hash):test(fn.__name__,fn)
    print('8 PROVISIONAL PIPELINE CONTRACT TESTS PASSED; NO RING PICKLE LOADED')

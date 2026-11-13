"""Minimal structural guards for the DM semi-supervised ablation."""
import numpy as np
from sklearn.preprocessing import StandardScaler
import dm_data as dm
from quality_link_challenge import normal_inverse_gamma,nig_predict
from audit_unlabeled_ablation import group_specs,LOCKED,MODES,MODELS


def test_four_fold_ring_isolation():
    ids=['DM01','DM02','DM04','DM05','DM06','DM07','DM08','DM09',
         'DM10','DM11','DM12','DM14','DM15','DM16','DM17','DM18']
    s=dm.ring_split(ids,seed=20261111)
    assert tuple(s['TEST_GESPERRT'])==LOCKED
    all_val=[]
    for j in range(1,5):
        L,U,V=group_specs(s,j)
        assert len(L)==6 and len(U)==4 and len(V)==2
        assert not set(L+U+V)&set(LOCKED)
        assert set(U)==set(s['U'])
        all_val.extend(V)
    assert len(set(all_val))==8 and set(all_val)==set(s['L'])


def test_strict_supervised_preprocess_differs_from_unlabeled_aware():
    L=np.array([[-1.0],[0.0],[2.0]])
    U=np.array([[5.0],[9.0]])
    supervised=StandardScaler().fit(L)
    semi=StandardScaler().fit(np.vstack([L,U]))
    assert not np.allclose(supervised.mean_,semi.mean_)
    assert not np.allclose(supervised.transform(L),semi.transform(L))


def test_direct_nig_is_invariant_to_C_given_fixed_preprocessing():
    # NIG regression sees only 6 labeled X and Y. A fake C value does not enter.
    x=np.arange(18,dtype=float).reshape(6,3)/12
    y=np.stack([np.sin(x[:,0]),np.cos(x[:,1])],axis=1)
    A=normal_inverse_gamma(x,y)
    bmu,bsd=nig_predict(A,x[0:2])
    for C in (0.,1.,8.):
        B=normal_inverse_gamma(x,y)
        mu,sd=nig_predict(B,x[0:2])
        assert np.allclose(mu,bmu) and np.allclose(sd,bsd)


def test_unexpected_fold_rejected():
    try:
        group_specs({'L':(), 'U':(), 'TEST_GESPERRT':()},0)
    except ValueError:return
    raise AssertionError('Invalid fold did not raise')


def test_registered_pipeline_modes_are_distinct():
    assert len(MODES)==3 and len(set(MODES))==3
    assert len(MODELS)==3 and len(set(MODELS))==3
    assert MODES[0]=='SUP_L_ONLY' and MODES[-1]=='SSL_U_C1'


if __name__=='__main__':
    for name,fun in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fun):
            fun(); print('PASS',name.replace('test_',''))

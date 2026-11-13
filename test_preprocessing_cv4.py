"""Read-only synthetic/scaler checks + optional real development-only source check.
No fit ever needs ring quality data. No outer test evaluation.
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
from compare_preprocessing_cv4 import fit_process_scaler, NORMAL_IQR, LOCKED, EXPECTED_L, EXPECTED_U
import dm_data as dm


def ring(a):
    return SimpleNamespace(x=np.asarray(a,float), y=object())


def test_p0_identical_to_original():
    rings={'DM01':ring([[1.,2.],[2.,4.],[5.,8.]]), 'DM02':ring([[3.,3.],[4.,5.]])}
    ref=dm.fit_x_scaler(rings,('DM01','DM02'))
    got=fit_process_scaler(rings,('DM01','DM02'),'P0')
    assert np.array_equal(ref.mean,got.mean)
    assert np.array_equal(ref.scale,got.scale)
    assert got.fitted_ids==ref.fitted_ids


def test_p1_equal_ring_mixture_exact():
    rings={'DM01':ring([[0.,0.],[0.,0.]]),'DM02':ring([[10.,20.]]*6)}
    got=fit_process_scaler(rings,('DM01','DM02'),'P1')
    assert np.allclose(got.mean,[5.,10.])
    assert np.allclose(got.scale,[5.,10.])
    p0=fit_process_scaler(rings,('DM01','DM02'),'P0')
    assert np.allclose(p0.mean,[7.5,15.])


def test_p2_robust_normal_consistency_and_no_clipping():
    rings={'DM01':ring([[0],[1],[2],[3],[4],[10000]])}
    got=fit_process_scaler(rings,('DM01',),'P2')
    expected=np.percentile(rings['DM01'].x,[25,50,75],axis=0)
    assert np.allclose(got.mean,expected[1])
    assert np.allclose(got.scale,(expected[2]-expected[0])/NORMAL_IQR)
    assert np.max(got.transform(rings['DM01'].x))>1000  # no clipping / winsorization


def test_missing_outer_and_invalid_rejected():
    x={'DM01':ring([[0],[1]])}
    for ids in (('DM01','DM01'),('DM01','DM02'),(),('DM07',)):
        try:
            fit_process_scaler(x,ids,'P1')
        except (ValueError,AssertionError):
            pass
        else:
            raise AssertionError(f'Must reject {ids}')
    x['DM02']=ring([[0],[float('nan')]])
    try:fit_process_scaler(x,('DM01','DM02'),'P2')
    except ValueError:pass
    else:raise AssertionError('NaN not blocked')


def test_never_reads_quality():
    class ForbiddenQuality:
        def __getattribute__(self,name):
            if name=='y':raise AssertionError('preprocessor accessed quality label')
            return object.__getattribute__(self,name)
    rr={'DM01':ForbiddenQuality()}
    rr['DM01'].x=np.array([[0.,2.],[1.,4.],[3.,5.]])
    for m in ('P0','P1','P2'):
        assert len(fit_process_scaler(rr,('DM01',),m).mean)==2


def test_registered_development_boundary():
    assert set(EXPECTED_L).isdisjoint(set(EXPECTED_U))
    assert not (set(LOCKED)&(set(EXPECTED_L)|set(EXPECTED_U)))
    assert len(EXPECTED_L)==8 and len(EXPECTED_U)==4 and len(LOCKED)==4


if __name__=='__main__':
    tests=[test_p0_identical_to_original,test_p1_equal_ring_mixture_exact,
           test_p2_robust_normal_consistency_and_no_clipping,
           test_missing_outer_and_invalid_rejected,test_never_reads_quality,
           test_registered_development_boundary]
    for t in tests:
        t()
        print('PASS',t.__name__)
    print('ALLE 6 SYNTHETISCHEN VORVERARBEITUNGSTESTS BESTANDEN. KEINE TESTLABELS.')

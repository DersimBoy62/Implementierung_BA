from dataclasses import replace
import numpy as np
import pytest
from bayesian_ring_ssr.coordinate_train import fit
from bayesian_ring_ssr.posterior_predictive import predict_new_ring
from bayesian_ring_ssr.persistence import save_model,load_model
from bayesian_ring_ssr.schema import UnlabeledRing


def test_T10_real_synthetic_HMM_joint_training_and_monotonic_blocks(synthetic_model,synthetic_rings):
    model = synthetic_model
    l,u,test = synthetic_rings
    assert len(model.heads)==2 and [len(h.mean) for h in model.heads]==[9,8]
    assert len(model.history)==3
    assert model.history[-1]['elbo'] > model.update_history[0]['ELBO_before']+1.
    assert min(row['delta'] for row in model.update_history)>=-1e-6
    assert any(row.get('optimizer',{}).get('accepted') for row in model.update_history)
    assert set(model.local_chains)==set(l)|set(u)
    assert not hasattr(next(iter(u.values())),'targets')
    pred = predict_new_ring(model,test)
    assert len(pred['mean_mm'])==len(pred['sd_mm'])==2
    assert pred['unit']=='mm' and np.isfinite(pred['mean_mm']).all() and min(pred['sd_mm'])>0
    # A held-out new process has no quality interface and never entered any scaler.
    assert test.ring_id not in model.preprocessing.x_scaler.fit_ids
    assert test.ring_id not in model.preprocessing.y_scaler.fit_ids
    with pytest.raises(TypeError,match='process-only'):
        predict_new_ring(model,next(iter(l.values())))


def test_T8_exact_json_load_vs_memory_and_tamper_detection(synthetic_model,synthetic_rings,tmp_path):
    model = synthetic_model
    test = synthetic_rings[2]
    bundle = tmp_path/'model'
    save_model(model,bundle)
    loaded = load_model(bundle)
    assert predict_new_ring(loaded,test)==predict_new_ring(model,test)
    assert loaded.preprocessing.to_dict()==model.preprocessing.to_dict()
    with pytest.raises(FileExistsError):
        save_model(model,bundle)
    with (bundle/'model.json').open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError,match='integrity'):
        load_model(bundle)


def test_T8_seed_reproducibility_and_configured_other_seed(synthetic_rings,small_config):
    l,u,test = synthetic_rings
    a,b = fit(l,u,small_config),fit(l,u,small_config)
    np.testing.assert_allclose(a.process.A,b.process.A,atol=1e-12,rtol=0)
    np.testing.assert_allclose(a.heads[0].mean,b.heads[0].mean,atol=1e-12,rtol=0)
    assert predict_new_ring(a,test)==predict_new_ring(b,test)
    other = fit(l,u,replace(small_config,seed=20261112))
    assert other.config.seed==20261112
    assert np.isfinite(predict_new_ring(other,test)['mean_mm']).all()
    # A different seed can still give the same KMeans centers: no robustness claim.


def test_T9_wrong_U_type_and_duplicate_ring(synthetic_rings,small_config):
    l,u,_ = synthetic_rings
    with pytest.raises(TypeError,match='UnlabeledRing'):
        fit(l,[next(iter(l.values()))],small_config)
    with pytest.raises(ValueError,match='Doppelte'):
        fit(list(l.values())+[next(iter(l.values()))],u,small_config)

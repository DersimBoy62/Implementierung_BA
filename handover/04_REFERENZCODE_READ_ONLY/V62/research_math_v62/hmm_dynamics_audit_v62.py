#!/usr/bin/env python3
"""V6.2 -- inspect within-ring HMM time dynamics and emission residuals.

Research-only, unchanged V6.1/V6.0/V5.2 mathematical cores. This instrument
replays V6.1 training, records aggregate posterior Markov diagnostics from
its actual local q(z_1:T), and compares the exact ELBO curve to V6.1.
No historical outer labels, no quality predictions, no model checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V61_PATH = ROOT / 'research_math_v61' / 'process_posterior_audit_v61.py'
V61_SHA = 'be17e95a4c87da9ecb1f79b2d93587e6f7920a12fd4ded3f1d84c2fb07b751c2'
V55 = ROOT / 'research_convergence_v55'
CORE = ROOT / 'final_candidate_v52'
SELECTED = (1, 9, 10, 11, 12, 20, 30)


def check(cond, msg):
    if not bool(cond):
        raise AssertionError(msg)


def dependencies():
    check(V61_PATH.is_file(), 'Missing released V6.1 source')
    check(hashlib.sha256(V61_PATH.read_bytes()).hexdigest() == V61_SHA,
          'Released V6.1 source does not match approved SHA: abort')
    if str(V61_PATH.parent) not in sys.path:
        sys.path.insert(0, str(V61_PATH.parent))
    import process_posterior_audit_v61 as v61
    check(Path(v61.__file__).resolve() == V61_PATH.resolve(), 'Imported incorrect V6.1 script')
    v61.deps()  # SHA-gates the original V6.0,V5.9,V5.8 and frozen V5.2 math cores
    return v61


def vec(x):
    return np.asarray(x, dtype=float)


def soft_dynamics(fb, rid, is_labeled, weight, bins=3):
    """Expected switches use pairwise xi, not merely argmax(gamma).

    The two concepts differ sharply when q is uncertain. Time bin summaries
    are observational diagnostics; bins are NOT physical rolling phases.
    """
    g = vec(fb.gamma)
    xi = vec(fb.xi)
    check(g.ndim == 2 and np.isfinite(g).all(), f'{rid}: invalid gamma')
    T, K = g.shape
    check(T >= 2 and xi.shape == (T-1, K, K), f'{rid}: invalid xi dimensions')
    check(np.isfinite(xi).all() and min(g.min(), xi.min()) >= -1e-9,
          f'{rid}: negative/nonfinite state probability')
    check(np.max(np.abs(g.sum(1) - 1)) < 1e-6, f'{rid}: gamma not normalized')
    check(np.max(np.abs(xi.sum((1,2)) - 1)) < 1e-6, f'{rid}: xi not normalized')
    err1 = float(np.max(np.abs(xi.sum(2) - g[:-1])))
    err2 = float(np.max(np.abs(xi.sum(1) - g[1:])))
    check(max(err1, err2) < 1e-6, f'{rid}: xi/gamma inconsistent')
    off = ~np.eye(K, dtype=bool)
    n_change = float(xi[:, off].sum())
    hard = np.argmax(g, axis=1)
    hard_changes = int(np.count_nonzero(np.diff(hard)))
    mass = g.mean(0)
    ent = -float((g * np.log(np.maximum(g, 1e-300))).sum(axis=1).mean())
    intervals = np.array_split(np.arange(T), bins)
    thirds = [g[s].mean(0).tolist() for s in intervals]
    difference = float(0.5 * np.sum(np.abs(vec(thirds[-1]) - vec(thirds[0]))))
    rate = n_change / (T-1)
    return {
        'ring_id': str(rid), 'labeled': bool(is_labeled), 'T': int(T),
        'data_weight': float(weight),
        'state_mass': mass.tolist(),
        'dominant_state_mass': float(mass.max()),
        'posterior_expected_switches': n_change,
        'posterior_expected_switch_rate': rate,
        'hard_argmax_gamma_switches': hard_changes,
        'hard_argmax_gamma_switch_rate': hard_changes/(T-1),
        'mean_marginal_state_entropy_nats': ent,
        'mean_max_gamma': float(g.max(1).mean()),
        'fraction_ambiguous_max_gamma_below_0_8': float(np.mean(g.max(1) < 0.8)),
        'occupancy_by_normalized_time_third': thirds,
        'early_late_state_distribution_TV': difference,
        'max_xi_marginal_inconsistency': max(err1,err2),
    }


def corr_lag1(z):
    z = vec(z).ravel()
    if len(z) < 4: return None
    a, b = z[:-1], z[1:]
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a)*np.linalg.norm(b)
    if den <= 1e-12: return None
    return float(np.clip(np.dot(a,b)/den, -1., 1.))


def lag_residuals(x, fb, posterior_emission_means, rid):
    """Exploratory posterior-weighted residual autocorrelation, not a formal test.

    Since q(z) was inferred from x, residuals have endogeneity. High acf is
    diagnostic of remaining temporal structure; low acf proves nothing.
    """
    x = vec(x)
    g = vec(fb.gamma)
    mu = vec(posterior_emission_means)
    check(mu.shape == (g.shape[1], x.shape[1]), 'Bad emission mean dimensions')
    r = x - g @ mu
    raw = [corr_lag1(x[:,j]) for j in range(x.shape[1])]
    corrected = [corr_lag1(r[:,j]) for j in range(x.shape[1])]
    present_raw = [float(y) for y in raw if y is not None]
    present_res = [float(y) for y in corrected if y is not None]
    return {'ring_id':str(rid),'n_channels':int(x.shape[1]),
            'raw_lag1_per_channel':raw,
            'posterior_mean_residual_lag1_per_channel':corrected,
            'median_abs_raw_lag1':None if not present_raw else float(np.median(np.abs(present_raw))),
            'median_abs_residual_lag1':None if not present_res else float(np.median(np.abs(present_res)))}


def aggregate_epoch(rows, trans_mean):
    check(len(rows)>0,'Missing ring diagnostics')
    weights = np.asarray([r['data_weight'] for r in rows])
    T = np.asarray([r['T'] for r in rows])
    switches = np.asarray([r['posterior_expected_switches'] for r in rows])
    hard_switch = np.asarray([r['hard_argmax_gamma_switches'] for r in rows])
    maxdist = np.asarray([r['dominant_state_mass'] for r in rows])
    timesteps = float(np.dot(weights,T-1))
    off_expected = float(np.dot(weights,switches))
    check(timesteps > 0,'Invalid weighted timepoints')
    tm = vec(trans_mean)
    check(tm.shape[0]==tm.shape[1] and np.max(np.abs(tm.sum(1)-1))<1e-7, 'Bad posterior transition means')
    d = np.diag(tm)
    expected_sojourn = [float(1/(1-p)) if p < 1-1e-12 else None for p in d]
    return {'rings':len(rows),'rings_single_state_dominant_gt_0_95':int(np.sum(maxdist>.95)),
            'rings_single_state_dominant_gt_0_99':int(np.sum(maxdist>.99)),
            'rings_zero_hard_argmax_switches':int(np.sum(hard_switch==0)),
            'total_unweighted_expected_switches':float(switches.sum()),
            'total_weighted_expected_switches':off_expected,
            'overall_weighted_expected_switch_rate':off_expected/timesteps,
            'total_unweighted_hard_argmax_switches':int(hard_switch.sum()),
            'posterior_transition_mean':tm.tolist(),
            'posterior_mean_sojourn_steps':expected_sojourn,
            'median_ring_expected_switches':float(np.median(switches)),
            'max_ring_xi_gamma_inconsistency':float(max(r['max_xi_marginal_inconsistency'] for r in rows))}


def audit_trace(rings, ydata, sdata, config):
    """Instrument V6.1 in memory, without editing any math module or changing its q."""
    v61 = dependencies()
    selected = set(SELECTED)
    call_count=[0]
    epoch_agg=[]
    epoch_snap=[]
    original_stats = v61.weighted_stats
    original_emit = v61.emission_ring_deltas
    local_cache={}

    def stats_hook(rings_, ydata_, local_, C, K):
        ret = original_stats(rings_,ydata_,local_,C,K)
        call_count[0]+=1
        ep=call_count[0]
        rows=[]
        for rid in sorted(rings_):
            if rid not in local_:continue
            rows.append(soft_dynamics(local_[rid]['fb'],rid,rid in ydata_,
                                      1. if rid in ydata_ else C))
        check(sum(r['T'] for r in rows)==sum(a['T'] for a in ret['per_ring_state_mass']),
              'Ring timepoints disagree')
        trans = vec(ret['transition_counts_weighted'])
        switch_weighted = sum(r['posterior_expected_switches']*r['data_weight'] for r in rows)
        check(abs(switch_weighted - (trans.sum()-np.trace(trans)))<1e-5,
              'Expected-switch sum does not match weighted Dirichlet sufficient statistic')
        local_cache['rows']=rows
        if ep in selected:
            epoch_snap.append({'epoch':ep,'rings':rows})
        return ret

    def emission_hook(rings_, ydata_, local_, p_old, p_new, C, full):
        ret = original_emit(rings_, ydata_, local_, p_old, p_new, C, full)
        ep=call_count[0]
        rows=local_cache['rows']
        means=p_new.mu
        a = aggregate_epoch(rows,p_new.A/p_new.A.sum(1,keepdims=True))
        a['epoch']=ep
        epoch_agg.append(a)
        if ep in selected:
            for snapshot in epoch_snap:
                if snapshot['epoch']==ep:
                    snapshot['posterior_mean_emission_residual_lag1']=[
                        lag_residuals(rings_[rid],local_[rid]['fb'],means,rid)
                        for rid in sorted(rings_) if rid in local_]
                    break
        return ret

    try:
        v61.weighted_stats = stats_hook
        v61.emission_ring_deltas = emission_hook
        hist=v61.trace(rings,ydata,sdata,config['K'],config['C'],config['epochs'],
                       config['seed'],config['local_steps'],config['physical_var'],verbose=False)
    finally:
        v61.weighted_stats=original_stats
        v61.emission_ring_deltas=original_emit
    check(len(hist)==len(epoch_agg)==config['epochs'], 'Missing epochs in audit')
    check(call_count[0]==config['epochs'],'Unexpected weighted_stats call count')
    return hist,epoch_agg,epoch_snap


def self_test():
    dependencies()
    # Case 1: a fully persistent Markov path (NOT model training).
    states=[0]*10
    K=2;g=np.eye(K)[states]
    xi=np.array([np.outer(g[t],g[t+1]) for t in range(len(states)-1)])
    from types import SimpleNamespace
    a=soft_dynamics(SimpleNamespace(gamma=g,xi=xi),'S1',True,1.)
    check(a['posterior_expected_switches']==0 and a['hard_argmax_gamma_switches']==0,
          'Persistent path switches wrong')
    # Case 2: alternating states, expected switches must be T-1.
    states=[0,1,0,1,0,1,0,1,0,1]
    g=np.eye(K)[states]
    xi=np.array([np.outer(g[t],g[t+1]) for t in range(len(states)-1)])
    b=soft_dynamics(SimpleNamespace(gamma=g,xi=xi),'S2',False,.2)
    check(abs(b['posterior_expected_switches']-9)<1e-10 and b['hard_argmax_gamma_switches']==9,
          'Alternating path switches wrong')
    c=aggregate_epoch([a,b],[[.99,.01],[.02,.98]])
    check(abs(c['total_weighted_expected_switches']-1.8)<1e-12,
          'Weighted switches incorrect')
    X=np.arange(30,dtype=float).reshape(10,3)
    r=lag_residuals(X,SimpleNamespace(gamma=np.ones((10,2))/2),np.zeros((2,3)),'S1')
    check(all(v is not None and np.isfinite(v) for v in r['raw_lag1_per_channel']),
          'Autocorrelation cannot be evaluated')
    # Gradient/closed-form and 3 synthetic epoch checks remain SHA-gated by V6.1.
    print('V62 SELF-TEST PASS: immutable V61/V60/V59/V58/V52 code, exact xi/gamma identities, '
          'switch counts, weights, time thirds, lag-1 diagnostics; NO DM / NO TRAINING',flush=True)


def run(args):
    v61=dependencies()
    if str(V55) not in sys.path:sys.path.insert(0,str(V55))
    import convergence_probe_v55 as probe
    cfg, manifest=probe.preflight(Path(args.data),Path(args.model_dir))
    check(cfg['K']>=2 and cfg['C']>=0, 'Incorrect process model config')
    check(args.epochs==30, 'This research comparison is fixed to 30 epochs')
    path=Path(args.out)
    check(path.suffix=='.json' and path.parent.is_dir() and not path.exists() and not path.is_symlink(),
          'Output must be a NEW JSON with an existing parent directory')
    check(CORE.resolve() not in path.resolve(strict=False).parents,'Cannot write into frozen math core')
    check(Path(args.model_dir).resolve() not in path.resolve(strict=False).parents,'Cannot overwrite stored model')
    ref=json.loads(Path(args.reference).read_text(encoding='utf-8'))
    check(ref.get('status')=='V61_GLOBAL_PROCESS_POSTERIOR_MATH_FORENSICS_RESEARCH_ONLY',
          'Incorrect V6.1 reference')
    check(ref['source_sha256']==cfg['dataset_sha256'] and ref['epochs']==args.epochs,
          'Reference dataset/epochs mismatch')
    check(ref['reference_model_sha256']==manifest['model_sha256'],'Wrong model reference')
    check(ref['K']==cfg['K'] and abs(ref['C']-cfg['C'])<1e-12,'Reference K/C mismatch')
    rings,ydata,sdata,_=probe.build_training_arrays(Path(args.data))
    check(len(rings)==12 and len(ydata)==8,'Expected 8 labeled+4 masked development rings')
    check(not set(rings).intersection({'DM07','DM08','DM11','DM15'}),'Historical outer rings are included')
    print('V62 RESEARCH ONLY | Replay V61 with observational ring-time/HMM diagnostics; '
          'NO TEST SCORING / NO MODEL SAVED',flush=True)
    hist,agg,snap=audit_trace(rings,ydata,sdata,
           {'K':cfg['K'],'C':cfg['C'],'epochs':args.epochs,'seed':cfg['random_seed'],
            'local_steps':cfg['local_steps'],'physical_var':cfg['physical_var']})
    error=float(max(abs(a['elbo']-b['elbo']) for a,b in zip(hist,ref['history'])))
    check(error<1e-5,f'Diagnostic code changed ELBO trajectory: {error}')
    for e in (1,9,10,11,12,20,30):
        s=agg[e-1];print(f"V62 ep{e:02d} ELBO={hist[e-1]['elbo']:.6f}  "
            f"weighted expected switches={s['total_weighted_expected_switches']:.5f}  "
            f"rings dominant>.95={s['rings_single_state_dominant_gt_0_95']}/12  "
            f"zero-hard-switch rings={s['rings_zero_hard_argmax_switches']}/12",flush=True)
    data={'status':'V62_HMM_WITHIN_RING_DYNAMICS_DIAGNOSTIC_RESEARCH_ONLY',
          'source_sha256':cfg['dataset_sha256'],'config_sha256':ref['config_sha256'],
          'reference_model_sha256':manifest['model_sha256'],'source_sha256_v61':V61_SHA,
          'K':cfg['K'],'C':cfg['C'],'seed':cfg['random_seed'],'epochs':args.epochs,
          'quality_scoring':False,'historical_outer_rings_excluded':True,
          'new_model_saved':False,'max_abs_reference_ELBO_diff':error,
          'history_summary':agg,'selected_epoch_ring_diagnostics':snap,
          'interpretation_caveats':[
            'Argmax-gamma switches are NOT Viterbi switches; expected switches from xi are primary.',
            'Time thirds are arbitrary normalized positions, not physical rolling phases.',
            'Posterior transition means reflect priors and pooled counts, not a direct guarantee of phase detection.',
            'Emission lag1 residuals use q inferred from x and are descriptive, not a formal independence test.',
            'No new independent generalization evaluation or quality metrics were performed.']}
    with path.open('xb') as stream:
        stream.write((json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    last=agg[-1]
    print(json.dumps({'status':data['status'],'out':str(path),
       'max_abs_reference_ELBO_diff':error,
       'final_expected_switches_weighted':last['total_weighted_expected_switches'],
       'final_expected_switch_rate':last['overall_weighted_expected_switch_rate'],
       'final_rings_dominant_state_gt_0_95':last['rings_single_state_dominant_gt_0_95'],
       'final_rings_no_hard_switches':last['rings_zero_hard_argmax_switches'],
       'final_transition_mean':last['posterior_transition_mean']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sp=parser.add_subparsers(dest='cmd',required=True)
    sp.add_parser('self-test')
    p=sp.add_parser('run');p.add_argument('--data',required=True)
    p.add_argument('--model-dir',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=30)
    a=parser.parse_args()
    if a.cmd=='self-test':self_test()
    else:run(a)

if __name__=='__main__':main()

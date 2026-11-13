#!/usr/bin/env python3
"""V6.3 read-only model-structure audit: logger phase boundary vs posterior HMM
switches, and within-phase residual serial dependence.

NO modified math core, no test scoring, no checkpoint. Phase is observational
metadata not a training feature. Unchanged 30-epoch V6.1 replay is required
because V6.2 intentionally did not store gamma/xi paths or global parameters.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
V62=ROOT/'research_math_v62'/'hmm_dynamics_audit_v62.py'
V62_SHA='0a3df4ca6a43eca56988a0b2ae89903ae2273d55c4aa38011fcfc5b5f118a3f8'
V55=ROOT/'research_convergence_v55'
CORE=ROOT/'final_candidate_v52'


def check(c, m):
    if not bool(c): raise AssertionError(m)


def deps():
    check(V62.is_file() and hashlib.sha256(V62.read_bytes()).hexdigest()==V62_SHA,
          'V6.2 immutable source SHA mismatch')
    if str(V62.parent) not in sys.path: sys.path.insert(0,str(V62.parent))
    import hmm_dynamics_audit_v62 as v62
    check(Path(v62.__file__).resolve()==V62.resolve(), 'Wrong imported V6.2')
    v61=v62.dependencies()
    return v62,v61


def corr_lag1(a):
    x=np.asarray(a,float).ravel()
    if x.size<5 or not np.isfinite(x).all(): return None
    l=x[:-1]-x[:-1].mean(); r=x[1:]-x[1:].mean()
    d=np.linalg.norm(l)*np.linalg.norm(r)
    return None if d<=1e-12 else float(np.clip(l@r/d,-1.,1.))


def median_abs_lag1(x):
    vals=[corr_lag1(x[:,i]) for i in range(x.shape[1])]
    vals=[abs(v) for v in vals if v is not None]
    return None if not vals else float(np.median(vals))


def piecewise_detrend(x,phase,kind='linear'):
    """Remove within-phase intercept or intercept+linear time trend per sensor.
    Not a generative emission correction. Does not use quality labels.
    """
    x=np.asarray(x,float);ph=np.asarray(phase)
    check(x.ndim==2 and len(ph)==len(x) and np.isfinite(x).all(),'Bad detrending input')
    r=np.empty_like(x)
    breaks=np.r_[0,np.flatnonzero(np.diff(ph)!=0)+1,len(ph)]
    for left,right in zip(breaks[:-1],breaks[1:]):
        seg=x[left:right]; n=len(seg)
        if kind=='mean' or n<3:
            r[left:right]=seg-seg.mean(0)
        else:
            t=np.linspace(-1.,1.,n)
            design=np.column_stack([np.ones(n),t])
            coef=np.linalg.lstsq(design,seg,rcond=None)[0]
            r[left:right]=seg-design@coef
    return r


def within_phase_median_abs_lag1(x,phase):
    vals=[]
    starts=np.r_[0,np.flatnonzero(np.diff(phase)!=0)+1,len(phase)]
    for a,b in zip(starts[:-1],starts[1:]):
        for j in range(x.shape[1]):
            c=corr_lag1(x[a:b,j])
            if c is not None:vals.append(abs(c))
    return None if not vals else float(np.median(vals))


def boundary_metrics(g,xi,phase,dt,window_seconds=2.):
    g=np.asarray(g,float);xi=np.asarray(xi,float);phase=np.asarray(phase)
    T,K=g.shape
    check(T==len(phase) and xi.shape==(T-1,K,K),'Incompatible phase/path dimensions')
    check(np.isfinite(g).all() and np.isfinite(xi).all(), 'Nonfinite Markov probabilities')
    check(np.all(np.abs(g.sum(1)-1)<1e-7), 'Bad gamma normalization')
    check(np.all(np.abs(xi.sum((1,2))-1)<1e-7), 'Bad xi normalization')
    b=np.flatnonzero(phase[:-1]!=phase[1:])
    check(len(b)==1 and phase[b[0]]==2 and phase[b[0]+1]==1,
          'Expected exactly one phase804 change 2->1 in F2')
    k=int(b[0]); w=int(round(window_seconds/dt))
    check(w>=1,'Invalid boundary window')
    lo=max(0,k-w); hi=min(T-1,k+w+1)
    mass=np.sum(xi[:, ~np.eye(K,dtype=bool)],axis=1)
    msum=float(mass.sum()); near=float(mass[lo:hi].sum())
    hard=np.flatnonzero(np.diff(np.argmax(g,axis=1))!=0)
    sub0=g[:k+1].mean(axis=0); sub1=g[k+1:].mean(axis=0)
    window_n=hi-lo
    return {'n':int(T),'phase_boundary_edge':k,'phase_boundary_seconds_from_F2_start':float((k+1)*dt),
            'phase2_count':int(k+1),'phase1_count':int(T-k-1),
            'posterior_switch_mass_total':msum,
            'posterior_switch_mass_near_phase_boundary':near,
            'near_boundary_time_edge_fraction':float(window_n/(T-1)),
            'near_boundary_switch_fraction':None if msum<1e-10 else float(near/msum),
            'expected_near_switches_if_uniform':float(msum*window_n/(T-1)),
            'hard_argmax_switch_count':int(len(hard)),
            'closest_hard_switch_distance_seconds':None if not len(hard) else float(np.min(abs(hard-k))*dt),
            'state_occupancy_mean_phase2':sub0.tolist(),
            'state_occupancy_mean_phase1':sub1.tolist(),
            'phase_occupancy_total_variation':float(0.5*np.abs(sub0-sub1).sum()),
            'timestep_sec':float(dt)}


def get_phase_metadata(raw_path,rv,ids):
    # A verified trusted OWN pickle only. Data access checks source SHA before unpickling.
    if str(CORE/'final_core') not in sys.path:sys.path.insert(0,str(CORE/'final_core'))
    import dm_data as dm
    raw=dm.load_raw(raw_path,expected_sha='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5')
    ret={}
    for rid in sorted(ids):
        item=raw[rid]['logger']; a=np.asarray(item['matrix']); cols=item['matrix_columns']
        ix={s:i for i,s in enumerate(cols)}
        mask=np.isin(a[:,ix['phase804']],[1.,2.]) & (a[:,ix['LaserOK']]==1) & (a[:,ix['MessEin']]==1)
        v=a[mask]; tt=v[:,0]; ph=v[:,ix['phase804']]
        check(len(tt)==len(rv[rid].t) and np.max(np.abs(tt-rv[rid].t))<1e-9,
              f'{rid}: F2 timestamps not equal to exact training view')
        check(np.all(np.diff(tt)>0) and np.all(np.abs(np.diff(tt)-.1)<1e-6),
              f'{rid}: expected contiguous 0.1 sec logger')
        check(np.sum(np.diff(ph)!=0)==1,f'{rid}: expected single phase boundary')
        ret[rid]={'ph':ph,'t':tt}
    return ret


def do_trace(v61,rings,ydata,sdata,cfg):
    # Capture only final-epoch gamma/xi and posterior emission means in memory.
    original_stats=v61.weighted_stats
    original_emit=v61.emission_ring_deltas
    capture={'local':{},'mu':None}
    def stats_hook(rings_,ydata_,local_,C_,K_):
        capture['local']={rid:{'gamma':np.asarray(obj['fb'].gamma).copy(),
                               'xi':np.asarray(obj['fb'].xi).copy()} for rid,obj in local_.items()}
        return original_stats(rings_,ydata_,local_,C_,K_)
    def emit_hook(rings_,ydata_,local_,p_old,p_new,C_,full_):
        capture['mu']=np.asarray(p_new.mu,float).copy()
        return original_emit(rings_,ydata_,local_,p_old,p_new,C_,full_)
    v61.weighted_stats=stats_hook
    v61.emission_ring_deltas=emit_hook
    try:
        hist=v61.trace(rings,ydata,sdata,cfg['K'],cfg['C'],cfg['epochs'],cfg['seed'],
                       cfg['local_steps'],cfg['physical_var'],verbose=False)
    finally:
        v61.weighted_stats=original_stats
        v61.emission_ring_deltas=original_emit
    check(len(hist)==cfg['epochs'] and len(capture['local'])==len(rings) and capture['mu'] is not None,
          'Trace capture is incomplete')
    return hist,capture


def analyse(rings,meta,capture,selected):
    rows=[]
    means=capture['mu']
    for rid in sorted(rings):
        x=np.asarray(rings[rid],float);p=capture['local'][rid]
        gamma=p['gamma'];xi=p['xi'];ph=meta[rid]['ph']
        b=boundary_metrics(gamma,xi,ph,dt=.1)
        resid=x-gamma@means
        lineardet=piecewise_detrend(resid,ph,'linear')
        rawdet=piecewise_detrend(x,ph,'linear')
        row={**b,'ring_id':rid,'labeled':bool(rid in selected),
             'residual_lag1_median_abs':median_abs_lag1(resid),
             'within_phase_residual_lag1_median_abs':within_phase_median_abs_lag1(resid,ph),
             'within_phase_linearly_detrended_residual_lag1_median_abs':within_phase_median_abs_lag1(lineardet,ph),
             'within_phase_linearly_detrended_raw_lag1_median_abs':within_phase_median_abs_lag1(rawdet,ph),
             'first_difference_residual_lag1_median_abs':median_abs_lag1(np.diff(resid,axis=0)),
             'first_difference_raw_lag1_median_abs':median_abs_lag1(np.diff(x,axis=0))}
        rows.append(row)
    near=sum(r['posterior_switch_mass_near_phase_boundary'] for r in rows)
    total=sum(r['posterior_switch_mass_total'] for r in rows)
    uniform=sum(r['expected_near_switches_if_uniform'] for r in rows)
    exposure=sum(r['near_boundary_time_edge_fraction']*(r['n']-1) for r in rows)/sum(r['n']-1 for r in rows)
    return rows,{
       'n_rings':len(rows),'n_phase_boundary_transitions':len(rows),
       'posterior_switches_near_phases_total_unweighted':near,
       'posterior_expected_switches_all_unweighted':total,
       'posterior_switch_share_near_phase_boundary':None if total<1e-10 else near/total,
       'uniform_in_time_expected_near_switches_unweighted':uniform,
       'observed_to_uniform_near_switch_mass_ratio':None if uniform<1e-10 else near/uniform,
       'total_window_exposure_fraction_unweighted_edges':exposure,
       'n_rings_hard_switch_within_2_seconds':sum(r['closest_hard_switch_distance_seconds'] is not None and r['closest_hard_switch_distance_seconds']<=2 for r in rows),
       'median_abs_lag1_residual':float(np.median([r['residual_lag1_median_abs'] for r in rows if r['residual_lag1_median_abs'] is not None])),
       'median_abs_lag1_within_phase_detrended_residual':float(np.median([r['within_phase_linearly_detrended_residual_lag1_median_abs'] for r in rows if r['within_phase_linearly_detrended_residual_lag1_median_abs'] is not None])),
       'median_abs_lag1_first_difference_residual':float(np.median([r['first_difference_residual_lag1_median_abs'] for r in rows if r['first_difference_residual_lag1_median_abs'] is not None]))}


def self_test():
    deps()
    T=100; ph=np.r_[np.full(70,2),np.full(30,1)]
    g=np.ones((T,2))*np.array([.85,.15]);xi=np.einsum('ti,tj->tij',g[:-1],g[1:])
    for t in range(T-1):
        xi[t] /=xi[t].sum()
    bm=boundary_metrics(g,xi,ph,.1)
    check(bm['phase_boundary_edge']==69 and abs(bm['phase_boundary_seconds_from_F2_start']-7)<1e-12,
          'Phase boundary indexing wrong')
    g=np.eye(2)[[0]*70+[1]*30]
    xi=np.array([np.outer(g[t],g[t+1]) for t in range(T-1)])
    bm=boundary_metrics(g,xi,ph,.1)
    check(abs(bm['posterior_switch_mass_total']-1)<1e-12 and
          abs(bm['posterior_switch_mass_near_phase_boundary']-1)<1e-12 and
          bm['closest_hard_switch_distance_seconds']==0,
          'Known phase-aligned switch not recovered')
    t=np.arange(T)*.1
    y=np.column_stack([1+.03*t+np.sin(2*np.pi*t/15),3-.07*t+.3*np.cos(t)])
    det=piecewise_detrend(y,ph)
    check(det.shape==y.shape and np.isfinite(det).all(), 'Bad detrending')
    check(abs(piecewise_detrend(np.column_stack([t, 2*t+7]),ph)).max()<1e-12,
          'Linear phase detrending must remove exact linear signals')
    check(median_abs_lag1(np.random.default_rng(17).normal(size=(100,2))) is not None,
          'Missing ACF diagnostic')
    print('V63 SELF-TEST PASS: full core SHA, phase-window switch identity, F2 edge alignment, '
          'piecewise detrend and residual diagnostics; NO DM / NO TRAINING',flush=True)


def run(args):
    v62,v61=deps()
    if str(V55) not in sys.path:sys.path.insert(0,str(V55))
    import convergence_probe_v55 as probe
    cfg,man=probe.preflight(args.data,args.model_dir)
    out=Path(args.out)
    check(out.suffix=='.json' and out.parent.is_dir() and not out.exists() and not out.is_symlink(),
          'Output must be a NEW JSON in an existing directory (write-once)')
    check(CORE.resolve() not in out.resolve(strict=False).parents,'Cannot write into immutable core')
    check(Path(args.model_dir).resolve() not in out.resolve(strict=False).parents,'Cannot write into stored model')
    ref62=json.loads(Path(args.reference_v62).read_text(encoding='utf-8'))
    ref61=json.loads(Path(args.reference_v61).read_text(encoding='utf-8'))
    check(ref62.get('status')=='V62_HMM_WITHIN_RING_DYNAMICS_DIAGNOSTIC_RESEARCH_ONLY' and
          ref61.get('status')=='V61_GLOBAL_PROCESS_POSTERIOR_MATH_FORENSICS_RESEARCH_ONLY',
          'Wrong references')
    check(ref62['source_sha256']==cfg['dataset_sha256'] and
          ref61['source_sha256']==cfg['dataset_sha256'] and
          ref62['reference_model_sha256']==man['model_sha256'] and
          ref61['reference_model_sha256']==man['model_sha256'], 'Wrong data/model reference')
    check(ref62['epochs']==ref61['epochs']==30 and cfg['K']==2 and cfg['C']==.2,
          'Incorrect registered research configuration')
    rings,ydata,sdata,rv=probe.build_training_arrays(args.data)
    check(len(rings)==12 and len(ydata)==8 and not set(rings).intersection({'DM07','DM08','DM11','DM15'}),
          'Not exactly 12 permitted development rings')
    meta=get_phase_metadata(args.data,rv,rings)
    print('V63 READONLY TRAIN-REPLAY: phase804 physical logging vs inferred HMM switches; '
          'residual temporal diagnostic, NO quality scoring or new model saved',flush=True)
    hist,capture=do_trace(v61,rings,ydata,sdata,{
      'K':cfg['K'],'C':cfg['C'],'epochs':30,'seed':cfg['random_seed'],
      'local_steps':cfg['local_steps'],'physical_var':cfg['physical_var']})
    err=max(abs(a['elbo']-b['elbo']) for a,b in zip(hist,ref61['history']))
    check(err<1e-5,f'Original ELBO trajectory changed: {err}')
    rows,agg=analyse(rings,meta,capture,ydata)
    stored={r['ring_id']:r for s in ref62['selected_epoch_ring_diagnostics'] if s['epoch']==30 for r in s['rings']}
    check(len(stored)==12,'Missing ep30 ringwise V62 reference')
    maxswitch=max(abs(r['posterior_switch_mass_total']-stored[r['ring_id']]['posterior_expected_switches']) for r in rows)
    check(maxswitch<1e-5,f'HMM dynamics changed from V62: {maxswitch}')
    result={'status':'V63_PHASE_BOUNDARY_AND_EMISSION_TIME_AUDIT_RESEARCH_ONLY',
        'source_sha256':cfg['dataset_sha256'],'original_model_sha256':man['model_sha256'],
        'v62_source_sha256':V62_SHA,'epochs':30,'K':cfg['K'],'C':cfg['C'],
        'no_quality_scoring':True,'no_model_saved':True,
        'max_abs_ref_v61_ELBO_diff':float(err),
        'max_abs_ref_v62_ring_switch_count_diff':float(maxswitch),
        'fixed_phase_alignment_window_seconds':2.0,
        'summary':agg,'per_ring':rows,
        'limitations':[
         'phase804 is observed metadata: association alone does not prove causal or physical state meaning.',
         'Expected switches are xi offdiagonal sums, hard switches are argmax gamma, not Viterbi.',
         'Posterior gamma and fitted means depend on observations; residual ACF is not a formal test of conditional independence.',
         'Phasewise linear detrending is descriptive, NOT a generative autoregressive model.',
         'No independent test set or held-out quality predictions assessed.']}
    with out.open('xb') as f:
        f.write((json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps({'status':result['status'],'output':str(out),
      'max_abs_ref_v61_ELBO_diff':err,'max_abs_ref_v62_switch_diff':maxswitch,
      **agg},indent=2,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='cmd',required=True)
    sp.add_parser('self-test')
    r=sp.add_parser('run')
    for a in ['data','model-dir','reference-v61','reference-v62','out']:
        r.add_argument('--'+a,required=True)
    args=p.parse_args()
    self_test() if args.cmd=='self-test' else run(args)

if __name__=='__main__':main()

"""Train-only HMM state covariance audit after within-ring linear/quadratic time detrending.

Exploratory diagnostic, NOT proof that a diagonal emission model is adequate.
Each degree uses the same eligible ring subset within each state; outer test and
inner validation rings are never used for fitting, local inference or scoring.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import dm_data as dm
import vb_b1 as vb
import fullcov_b1_df as fc
from audit_state_residuals import SOURCE_SHA, covariance_corr


def temporal_state_covariances(X, times, W, min_ring_mass=10.0, degrees=(0, 1, 2)):
    """Weighted covariance of within-ring detrended residuals for one state.

    Ring-specific polynomial of elapsed time is fitted separately to every
    channel using fractional state membership as sample weights. All degree
    comparisons use exactly the same eligible rings/weights. State weights
    come from an *X-only* posterior, and are held fixed during this audit.
    """
    if not X or set(X) != set(times) or set(X) != set(W):
        raise ValueError('X, times and W must share nonempty ring keys')
    if min_ring_mass < 0 or not np.isfinite(min_ring_mass):
        raise ValueError('min_ring_mass must be finite nonnegative')
    degrees = tuple(int(p) for p in degrees)
    if len(set(degrees)) != len(degrees) or not degrees or any(p not in (0, 1, 2) for p in degrees):
        raise ValueError('Choose distinct polynomial degrees from 0,1,2')
    d = next(iter(X.values())).shape[1]
    scatter = {p:np.zeros((d,d)) for p in degrees}
    used=[]; excluded=[]; mass=0.0; total_mass=0.0
    ring_records=[]
    for rid in sorted(X):
        x=np.asarray(X[rid],float)
        t=np.asarray(times[rid],float)
        w=np.asarray(W[rid],float)
        if x.ndim != 2 or x.shape[1] != d or t.shape != (len(x),) or w.shape != (len(x),):
            raise ValueError(f'inconsistent shapes for {rid}')
        if not (np.isfinite(x).all() and np.isfinite(t).all() and np.isfinite(w).all()):
            raise ValueError(f'nonfinite entries {rid}')
        if (w < 0).any() or (w > 1+1e-7).any() or (np.diff(t) <= 0).any():
            raise ValueError(f'invalid weights/times {rid}')
        s=float(w.sum());total_mass += s
        if s < min_ring_mass or len(x)<8 or t[-1] <= t[0]:
            excluded.append(rid);continue
        # physical timestamp-based progress, not index-based; F2 timestamps can skip rows.
        tau = 2*(t-t[0])/(t[-1]-t[0])-1
        effective_n = s*s/max(np.dot(w,w),1e-30)
        if effective_n < 10 or np.linalg.matrix_rank(np.vander(tau, 3, increasing=True)) < 3:
            excluded.append(rid);continue
        used.append(rid);mass += s
        sq=np.sqrt(w)
        row={'ring':rid, 'state_mass':s, 'effective_n':float(effective_n), 'n_points':len(x)}
        for degree in degrees:
            Z=np.vander(tau, degree+1, increasing=True)
            # Weighted least squares with QR/SVD; avoids normal equations.
            coeff=np.linalg.lstsq(Z*sq[:,None], x*sq[:,None], rcond=1e-12)[0]
            residual=x-Z@coeff
            block=(residual.T*w)@residual
            scatter[degree] += block
            r=covariance_corr(block/s)
            row[f'corr_degree_{degree}']=r
        ring_records.append(row)
    if mass <= 0:raise ValueError('No rings with adequate state support')
    cov={p:(m/mass + m.T/mass)*.5 for p,m in scatter.items()}
    return {'state_mass_included':float(mass),'state_mass_total':float(total_mass),
            'coverage':float(mass/max(total_mass,1e-30)), 'included_rings':used,
            'excluded_rings':excluded,'cov':cov,
            'corr':{p:covariance_corr(m) for p,m in cov.items()},
            'ring_records':ring_records}


def produce(a):
    raw=dm.load_raw(a.data,expected_sha=SOURCE_SHA)
    split=dm.ring_split(raw,seed=a.seed)
    L=tuple(split['L']);U=tuple(split['U']);val=tuple(sorted(L)[2*(a.fold-1):2*a.fold]);
    trL=tuple(r for r in L if r not in val)
    train=tuple(sorted(set(trL)|set(U)))
    outer=set(split['TEST_GESPERRT'])
    assert len(trL)==6 and len(U)==4 and len(val)==2 and not (set(train)&(outer|set(val)))
    # Restrict the actual Ring objects to TRAIN before constructing any view.
    ds=dm.view({rid:raw[rid] for rid in train},a.features,a.window)
    dm.refuse_post_quality_inputs(ds)
    sx=dm.fit_x_scaler(ds,train);sy=dm.fit_y_scaler(ds,trL)
    X={rid:sx.transform(ds[rid].x) for rid in train}
    T={rid:ds[rid].t for rid in train}
    Y={rid:sy.transform(ds[rid].y) for rid in trL}
    names=list(ds[train[0]].names)
    if not {'Ist_D150','Ist_h152'}.issubset(names):
        raise ValueError('Chosen features must contain both Ist_D150 and Ist_h152')
    di,hi=names.index('Ist_D150'),names.index('Ist_h152')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Output folder not empty: {out}')
    out.mkdir(parents=True,exist_ok=True)
    rows=[];detail=[]
    for model in (['diag','full'] if a.models=='both' else [a.models]):
        print(f'Trainiere {model}, Fold {a.fold}, K={a.K}, Epochen {a.epochs}; nur TRAIN ...',flush=True)
        if model=='diag':
            fit=vb.train(X,Y,K=a.K,C=a.C,epochs=a.epochs,seed=a.seed,mode='FULL_JOINT',
                         max_local_iter=a.local_steps,verbose=False)
            p=fit['process'];infer=lambda x:vb.local_inference(x,p,mode='OFF')['fb'].gamma
        else:
            fit=fc.train_full(X,Y,K=a.K,C=a.C,epochs=a.epochs,seed=a.seed,
                              max_local_iter=a.local_steps,
                              prior_options={'prior_precision':a.prior_precision,
                                             'prior_df_offset':a.df_offset},verbose=False)
            p=fit['process'];infer=lambda x:fc.infer_process_full(x,p)['fb'].gamma
        GS={rid:infer(X[rid]) for rid in train}
        for rid in train:
            assert GS[rid].shape==(len(X[rid]),a.K) and np.allclose(GS[rid].sum(1),1,atol=1e-7)
        for k in range(a.K):
            w={rid:GS[rid][:,k] for rid in train}
            try:
                audit=temporal_state_covariances(X,T,w,min_ring_mass=a.min_ring_mass)
            except ValueError:
                print(f'  STATE {k+1}: no adequately supported rings',flush=True)
                rows.append({'model':model,'state':k+1,'status':'insufficient support'})
                continue
            model_r=0.
            if model=='full':
                d=len(names)
                # Posterior E[Lambda^{-1}] is inv(W)/(nu-d-1). Scalar
                # uncertainty of the Gaussian mu does not change its correlation.
                model_cov=np.linalg.inv(p.W[k])/(p.nu[k]-d-1)
                model_r=float(covariance_corr(model_cov)[di,hi])
            r0=float(audit['corr'][0][di,hi]);r1=float(audit['corr'][1][di,hi]);r2=float(audit['corr'][2][di,hi]);
            rows.append({'model':model,'state':k+1,'status':'ok','state_mass_total':audit['state_mass_total'],
                         'mass_coverage':audit['coverage'],'ring_count':len(audit['included_rings']),
                         'r_ring_centered':r0,'r_after_linear_time':r1,'r_after_quadratic_time':r2,
                         'full_model_expected_corr':model_r if model=='full' else 0.,
                         'rings_included':','.join(audit['included_rings'])})
            for i in range(len(names)):
                for j in range(i+1,len(names)):
                    detail.append({'model':model,'state':k+1,'a':names[i],'b':names[j],
                                   'r_centered':float(audit['corr'][0][i,j]),
                                   'r_linear':float(audit['corr'][1][i,j]),
                                   'r_quadratic':float(audit['corr'][2][i,j])})
            print(f'  STATE {k+1} | mass cover {audit["coverage"]:.1%} | rings {len(audit["included_rings"])} | '
                  f'r(D,h) center={r0:+.4f} linear={r1:+.4f} quadratic={r2:+.4f}'
                  +(f' | Full expected emission r={model_r:+.4f}' if model=='full' else ''),flush=True)
    def write_csv(name,data):
        if not data:return
        fields=list(dict.fromkeys(k for r in data for k in r))
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fields,delimiter=';');w.writeheader();w.writerows(data)
    write_csv('time_trend_by_state.csv',rows)
    write_csv('time_trend_all_pairs.csv',detail)
    result={'warning':'Exploratory in-sample analysis only. Posterior q(Z|X), not q(Z|X,Y). '
              'Detrending uses inferred state weights and is not a valid fit test. '
              'A small time-detrended correlation does NOT prove diagonal HMM is adequate: '
              'the implemented stationary emission does not itself condition on elapsed time. '
              'No outer test or inner validation rings examined. Model state ids not aligned.',
            'configuration':vars(a),'train_L':list(trL),'train_U':list(U),'inner_val_excluded':list(val),
            'outer_test_locked':sorted(outer),'features':names,'summary':rows}
    (out/'TIME_TREND_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Reports:',out.resolve(), 'INNER-VAL excluded:',val,'OUTER TEST locked:',sorted(outer),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ap.add_argument('--out',default='results/time_trend_audit_fold1_k3')
    ap.add_argument('--fold',type=int,choices=[1,2,3,4],default=1)
    ap.add_argument('--K',type=int,default=3)
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--local-steps',type=int,default=8)
    ap.add_argument('--C',type=float,default=1.)
    ap.add_argument('--seed',type=int,default=20261111)
    ap.add_argument('--models',choices=['both','diag','full'],default='both')
    ap.add_argument('--features',default='V1_Kern_6',choices=list(dm.SETS))
    ap.add_argument('--window',default='F2',choices=['F0','F1','F2'])
    ap.add_argument('--min-ring-mass',type=float,default=10.)
    ap.add_argument('--prior-precision',type=float,default=1.)
    ap.add_argument('--df-offset',type=int,default=3)
    a=ap.parse_args()
    if a.df_offset<2 or a.min_ring_mass < 0:ap.error('df-offset>=2 and min-ring-mass>=0 required')
    produce(a)

if __name__=='__main__':main()

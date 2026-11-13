"""Train-only state-conditional residual/correlation audit, diagonal vs full covariance.

Ring-centering removes between-ring offsets *within a given inferred HMM state*;
it does NOT remove dynamic time trends or prove conditional independence. Exploratory.
No original library files or outer-test/inner-validation ring features are inspected.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import dm_data as dm
import vb_b1 as vb
import fullcov_b1_df as fc

SOURCE_SHA = 'bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'


def covariance_corr(cov):
    std = np.sqrt(np.maximum(np.diag(cov), 0.))
    den = np.outer(std, std)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 1e-12, cov / den, np.nan)


def state_covariances(X, W, min_ring_mass=1e-5):
    """Pooled E[(x-E_w x)(...)], within-ring covariance after subtracting E_w[x|ring].

    X/W: matching dictionaries ring -> (T,d), ring -> (T,), unweighted ring
    observation sequence and inferred fractional state memberships.
    """
    if set(X) != set(W) or not X:
        raise ValueError('Ring keys do not match or empty')
    d = next(iter(X.values())).shape[1]
    total = 0.
    weighted_sum = np.zeros(d)
    weighted_second = np.zeros((d,d))
    within_scatter = np.zeros((d,d))
    ring_rows = []
    for rid in sorted(X):
        x=np.asarray(X[rid],float); w=np.asarray(W[rid],float)
        if x.ndim != 2 or x.shape[1]!=d or w.shape!=(len(x),):raise ValueError('shapes differ')
        if not np.isfinite(x).all() or not np.isfinite(w).all() or (w<0).any():
            raise ValueError('nonfinite or negative')
        s=float(w.sum())
        if s<min_ring_mass: continue
        m=np.einsum('n,nd->d',w,x)/s
        e=x-m
        sc=np.einsum('n,ni,nj->ij',w,e,e,optimize=True)
        within_scatter+=sc
        total+=s
        weighted_sum+=s*m
        weighted_second+=sc+s*np.outer(m,m)
        rcov=sc/s
        ring_rows.append({'ring':rid,'mass':s,'cov':rcov,'corr':covariance_corr(rcov)})
    if total<=0: raise ValueError('zero state mass')
    mu=weighted_sum/total
    pooled=weighted_second/total-np.outer(mu,mu)
    within=within_scatter/total
    pooled=(pooled+pooled.T)/2
    within=(within+within.T)/2
    return {'mass':total,'pooled_cov':pooled,'within_cov':within,
            'pooled_corr':covariance_corr(pooled),'within_corr':covariance_corr(within),
            'rings':ring_rows}


def produce(args):
    raw=dm.load_raw(args.data,expected_sha=SOURCE_SHA)
    split=dm.ring_split(raw,seed=args.seed)
    L=tuple(split['L']);U=tuple(split['U']);outer=set(split['TEST_GESPERRT'])
    val=tuple(sorted(L)[2*(args.fold-1):2*args.fold])
    trL=tuple(r for r in L if r not in val);trU=U
    train_ids=tuple(sorted(set(trL)|set(trU)))
    assert len(trL)==6 and len(trU)==4 and len(val)==2
    assert not (set(train_ids)&set(val)) and not (set(train_ids)&outer)
    # Refuse even to construct a model view of excluded/validation rings.
    dataset=dm.view({r:raw[r] for r in train_ids},args.features,args.window)
    dm.refuse_post_quality_inputs(dataset)
    xs=dm.fit_x_scaler(dataset,train_ids);ys=dm.fit_y_scaler(dataset,trL)
    X={r:xs.transform(dataset[r].x) for r in train_ids}
    Y={r:ys.transform(dataset[r].y) for r in trL}
    names=list(dataset[train_ids[0]].names)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory must be empty/new: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    summary=[];pairs=[];per_ring=[]
    for model in (['diag','full'] if args.models=='both' else [args.models]):
        print(f'Trainiere Modell={model}, Fold={args.fold}, K={args.K}, d={len(names)}, nur Trainingsringe ...',flush=True)
        if model=='diag':
            fit=vb.train(X,Y,K=args.K,C=args.C,epochs=args.epochs,seed=args.seed,
                         mode='FULL_JOINT',max_local_iter=args.local_steps,verbose=False)
            infer=lambda x: vb.local_inference(x,fit['process'],mode='OFF')['fb'].gamma
        else:
            fit=fc.train_full(X,Y,K=args.K,C=args.C,epochs=args.epochs,seed=args.seed,
                              max_local_iter=args.local_steps,
                              prior_options={'prior_precision':args.prior_precision,
                                             'prior_df_offset':args.df_offset},verbose=False)
            infer=lambda x: fc.infer_process_full(x,fit['process'])['fb'].gamma
        gs={r:infer(X[r]) for r in train_ids}
        if any(g.shape!=(len(X[r]),args.K) or not np.allclose(g.sum(1),1,atol=1e-7) for r,g in gs.items()):
            raise AssertionError('invalid state marginal gamma')
        for k in range(args.K):
            stats=state_covariances(X,{r:gs[r][:,k] for r in train_ids})
            for i in range(len(names)):
                for j in range(i+1,len(names)):
                    pair={'model':model,'state':k+1,'a':names[i],'b':names[j],
                          'pooled_r':float(stats['pooled_corr'][i,j]),
                          'ring_centered_r':float(stats['within_corr'][i,j]),
                          'state_mass':stats['mass']}
                    pairs.append(pair)
                    for info in stats['rings']:
                        per_ring.append({'model':model,'state':k+1,'ring':info['ring'],'state_mass_ring':info['mass'],
                                         'a':names[i],'b':names[j],
                                         'within_ring_r':float(info['corr'][i,j])})
            max_pair=max([r for r in pairs if r['model']==model and r['state']==k+1],
                         key=lambda row: abs(row['ring_centered_r']) if np.isfinite(row['ring_centered_r']) else -1)
            summary.append({'model':model,'state':k+1,'state_mass':stats['mass'],
                            'state_mass_fraction':float(stats['mass']/sum(map(len,X.values()))),
                            'top_pair_ring_centered':{'a':max_pair['a'],'b':max_pair['b'],
                                                      'r':max_pair['ring_centered_r']},
                            'rings_with_state_mass_at_least_10':sum(z['mass']>=10 for z in stats['rings'])})
            if 'Ist_D150' in names and 'Ist_h152' in names:
                d_idx=names.index('Ist_D150');h_idx=names.index('Ist_h152')
                print(f'  State {k+1} mass {stats["mass"]:.1f} pooled r(D,h) '
                      f'{stats["pooled_corr"][d_idx,h_idx]:+.4f}, '
                      f'ring-centered r(D,h) {stats["within_corr"][d_idx,h_idx]:+.4f}, '
                      f'rings mass>=10: {summary[-1]["rings_with_state_mass_at_least_10"]}',flush=True)
    def write_csv(name, records):
        if not records:return
        with (out/name).open('w',newline='',encoding='utf-8-sig') as fh:
            wr=csv.DictWriter(fh,fieldnames=records[0].keys(),delimiter=';')
            wr.writeheader();wr.writerows(records)
    write_csv('state_pair_correlations.csv',pairs)
    write_csv('ringwise_state_pair_correlations.csv',per_ring)
    data={'status':'EXPLORATORY IN-SAMPLE TRAIN-ONLY DIAGNOSTICS',
          'warning':'The pooled and ring-centered correlations use posterior gamma inferred with X only. '
                    'They do not directly assess generalization or demonstrate conditional independence. '
                    'Ring centering removes between-ring offsets, but not within-ring time trends. '
                    'State indices are not aligned across fitted models.',
          'config':vars(args),'train_L':list(trL),'train_U':list(trU),'excluded_inner_val':list(val),
          'outer_test_locked':sorted(outer),'n_training_rings':len(train_ids),
          'n_training_steps':sum(map(len,X.values())),'features':names,
          'summary':summary,'pairs':pairs}
    (out/'STATE_RESIDUAL_AUDIT.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Fertig:',out.resolve(),'INNER VAL',val,'OUTER TEST',sorted(outer),flush=True)
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ap.add_argument('--out',default='results/state_residual_audit_fold1_k3')
    ap.add_argument('--fold',type=int,choices=[1,2,3,4],default=1)
    ap.add_argument('--K',type=int,default=3)
    ap.add_argument('--models',choices=['both','diag','full'],default='both')
    ap.add_argument('--features',default='V1_Kern_6',choices=list(dm.SETS))
    ap.add_argument('--window',default='F2',choices=['F0','F1','F2'])
    ap.add_argument('--seed',type=int,default=20261111)
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--local-steps',type=int,default=8)
    ap.add_argument('--C',type=float,default=1.0)
    ap.add_argument('--prior-precision',type=float,default=1.0)
    ap.add_argument('--df-offset',type=int,default=3)
    a=ap.parse_args()
    if a.df_offset<2:ap.error('--df-offset must be >= 2')
    produce(a)

if __name__=='__main__':main()

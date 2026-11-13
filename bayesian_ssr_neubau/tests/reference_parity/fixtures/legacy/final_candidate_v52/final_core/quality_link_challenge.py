"""Aggressive non-HMM comparisons of ring-quality link. Ringwise, label-leakage safe.

NOT a fully Bayesian SSR replacement: PCA is unsupervised point-estimate and the
quality heads are conjugate Bayesian regression / exact GP with fixed kernels.
All selected feature transforms are fitted on train L+U; y is fitted ONLY on L.
Never use outer test rings in any fit, selection, score or diagnostic.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import t as student_t, norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import dm_data as dm

DATA_SHA='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'
FIXED_SEED=20261111
LOCKED=('DM07','DM08','DM11','DM15')


def extract_descriptors(x,t,mode='physical'):
    """X is already scaled with TRAIN-only timepoint scaler, t is raw seconds.

    All features known at the END of the batch, not usable as an unchanged online predictor.
    Input contains only logger values; no finished-ring metrology target.
    """
    x=np.asarray(x,float); t=np.asarray(t,float)
    if x.ndim!=2 or x.shape[1]!=6 or len(x)!=len(t) or len(t)<6:
        raise ValueError('Expected F2 V1 six-channel ring signals')
    if not np.isfinite(x).all() or not np.isfinite(t).all() or (np.diff(t)<=0).any():
        raise ValueError('Invalid ring signals/time')
    duration=t[-1]-t[0]
    if duration<=0:raise ValueError('Empty duration')
    tau=(t-t[0])/duration
    if mode=='physical':
        # 6 input channels: F_rad, F_ax, Ist_D, Ist_s, Ist_h, RWG
        # Time-independent log duration; start/end changes; loads and variation.
        # Standardized units, learned feature scaling within each fold.
        return np.array([
            np.log(duration),
            x[:,0].mean(), x[:,1].mean(),                  # average forces
            x[:,0].std(), x[:,1].std(),                    # variable forces
            x[-1,2]-x[0,2],x[-1,3]-x[0,3],x[-1,4]-x[0,4], # deformation
            x[:,5].mean(),x[:,5].std(),                    # radial speed
            x[:,0].max(),x[:,1].max(),                    # peak loads
        ],float)
    if mode=='path':
        # Equal normalized time support; no arbitrary interpolation across missing phase gaps.
        grid=(np.arange(6)+.5)/6
        z=np.stack([np.interp(grid,tau,x[:,j]) for j in range(6)],axis=1)
        return np.r_[z.ravel(),np.log(duration)]
    if mode=='stats':
        z=np.stack([x.mean(0),x.std(0),x[-1]-x[0],np.ptp(x,axis=0)],axis=1)
        return np.r_[z.ravel(),np.log(duration)]
    raise ValueError(mode)

def make_latent(train_feats, val_feats, dimension=3):
    """Train-only standardization then train-only PCA and train-only PC scaling."""
    ids=tuple(sorted(train_feats))
    x=np.array([train_feats[r] for r in ids])
    scaler=StandardScaler().fit(x)
    xs=scaler.transform(x)
    n_pc=min(dimension,len(ids)-1,xs.shape[1])
    pca=PCA(n_components=n_pc,svd_solver='full').fit(xs)
    xt=pca.transform(xs)
    pcs=StandardScaler().fit(xt)
    u={r:pcs.transform(pca.transform(scaler.transform(f[None,:])))[0] for r,f in train_feats.items()}
    v={r:pcs.transform(pca.transform(scaler.transform(f[None,:])))[0] for r,f in val_feats.items()}
    if any(not np.isfinite(z).all() for z in list(u.values())+list(v.values())):raise ArithmeticError('Nonfinite PC descriptors')
    return u,v,{'fit_ids':ids,'n_pc':n_pc,'explained_variance_ratio':pca.explained_variance_ratio_.tolist()}


def normal_inverse_gamma(X,Y,V0=1.0,a0=2.5,b0=1.5):
    """Exact conjugate Bayesian linear regression for TWO conditionally independent targets.
    Y standardized on labeled TRAIN only; intercept prior mean 0 (training Y mean).
    X are latent PCA features + constant bias? bias excluded because response centered.
    """
    X=np.asarray(X,float);Y=np.asarray(Y,float)
    n,d=X.shape
    if Y.shape!=(n,2) or n<2 or V0<=0 or a0<=0 or b0<=0:raise ValueError('NIG config')
    L=np.eye(d)/V0+X.T@X
    c,low=cho_factor(L,check_finite=True)
    V=cho_solve((c,low),np.eye(d))
    M=cho_solve((c,low),X.T@Y)
    a=a0+n/2
    b=np.array([b0+.5*(np.sum(Y[:,i]**2)-M[:,i]@L@M[:,i]) for i in range(2)])
    if np.any(b<=0):raise ArithmeticError('Bayesian linear regression b <= 0')
    return {'M':M,'V':V,'a':a,'b':b}


def nig_predict(p,X):
    X=np.asarray(X,float)
    mu=X@p['M']
    var=(1+np.sum((X@p['V'])*X,axis=1))[:,None]*(p['b']/p['a'])[None,:]
    # Student-t predictive SD, only valid df>2
    sd=np.sqrt(var*p['a']/(p['a']-1))
    return mu,sd


def matern32_k(a,b,length):
    a=np.asarray(a,float);b=np.asarray(b,float)
    d=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
    v=np.sqrt(3)*d/length
    return (1+v)*np.exp(-v)


def gp_fit_predict(X,Y,Xtest,noise_variance=0.5):
    """Exact GP posterior with fixed Matérn-3/2 hyperparameters learned from X ONLY.
    No marginal-likelihood optimization on 6 quality labels. Independent outputs.
    """
    X=np.asarray(X,float);Y=np.asarray(Y,float);Xtest=np.asarray(Xtest,float)
    if Y.shape!=(len(X),2) or noise_variance<=0:raise ValueError('GP shape/noise')
    distances=pdist(X)
    positive=distances[distances>1e-12]
    length=float(np.median(positive)) if len(positive) else 1.0
    length=max(length,1e-6)
    K=matern32_k(X,X,length)+np.eye(len(X))*(noise_variance+1e-9)
    c,low=cho_factor(K)
    Ks=matern32_k(Xtest,X,length)
    mu=Ks@cho_solve((c,low),Y)
    v=cho_solve((c,low),Ks.T)
    var=np.maximum(1.+noise_variance-np.sum(Ks*v.T,axis=1),1e-10)
    return mu,np.sqrt(var)[:,None]*np.ones((1,2)),length


def evaluate(data='input/DM_16_aktive_Ringe.pkl',out='results/quality_link_challenge_cv4',
             reference=None,seed=FIXED_SEED,features='V1_Kern_6',window='F2',n_pcs=3):
    if seed!=FIXED_SEED:raise ValueError('Split fixed to initial sealed test; other seed refused')
    if features!='V1_Kern_6' or window!='F2':raise ValueError('Registered experiment V1/F2 only')
    raw=dm.load_raw(data,expected_sha=DATA_SHA)
    s=dm.ring_split(raw,seed=FIXED_SEED)
    if tuple(s['TEST_GESPERRT'])!=LOCKED:raise AssertionError('Outer holdout changed')
    dev=tuple(sorted(set(s['L'])|set(s['U'])))
    dview=dm.view({i:raw[i] for i in dev},features,window)
    dm.refuse_post_quality_inputs(dview)
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Nonempty directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    models=['PHYS_NIG','STATS_PCA_GP','PATH_PCA_GP','PATH_PCA_NIG']
    scores={m:[] for m in models}
    base=[]
    folds=[]
    for k in range(4):
        val=tuple(sorted(s['L'])[2*k:2*k+2]); trL=tuple(r for r in s['L'] if r not in val)
        trU=tuple(s['U']);tr=tuple(sorted(set(trL)|set(trU)))
        assert len(val)==2 and len(trL)==6 and len(trU)==4 and len(tr)==10
        assert not set(val)&set(tr) and not set(s['TEST_GESPERRT'])&(set(val)|set(tr))
        sx=dm.fit_x_scaler({r:dview[r] for r in tr},tr)
        sy=dm.fit_y_scaler({r:dview[r] for r in trL},trL)
        # TRAIN-and-UNLABELED information used ONLY for X feature PCA and scaling.
        # NO y from U or VAL is read into model fitting.
        ytrain=np.array([sy.transform(dview[r].y) for r in trL])
        models_out={}
        for mode in ('physical','stats','path'):
            feat_tr={rid:extract_descriptors(sx.transform(dview[rid].x),dview[rid].t,mode) for rid in tr}
            feat_val={rid:extract_descriptors(sx.transform(dview[rid].x),dview[rid].t,mode) for rid in val}
            if mode=='physical':
                # PHYS direct 12 physics-informed features, no PCA. Prior regularizes p>n.
                fs=StandardScaler().fit(np.array([feat_tr[r] for r in tr]))
                zztr={r:fs.transform(feat_tr[r][None,:])[0] for r in tr}
                zzv={r:fs.transform(feat_val[r][None,:])[0] for r in val}
                transform={'fit_ids':tr,'n_desc':len(feat_tr[tr[0]]),'n_pc':None}
            else:
                zztr,zzv,transform=make_latent(feat_tr,feat_val,n_pcs)
            assert set(transform['fit_ids'])==set(tr)
            X=np.array([zztr[r] for r in trL]);XV=np.array([zzv[r] for r in val])
            which=['PHYS_NIG'] if mode=='physical' else (['STATS_PCA_GP'] if mode=='stats' else ['PATH_PCA_GP','PATH_PCA_NIG'])
            for label in which:
                if label.endswith('_GP'):
                    pred,sd,length=gp_fit_predict(X,ytrain,XV)
                    details={'lengthscale_training_X_only':length,'noise_var':0.5}
                    halfwidth_norm=norm.ppf(.975)*sd
                else:
                    pp=normal_inverse_gamma(X,ytrain)
                    pred,sd=nig_predict(pp,XV)
                    details={'a_post':pp['a'],'b_post':pp['b'].tolist()}
                    predictive_student_scale=sd/np.sqrt(pp['a']/(pp['a']-1))
                    halfwidth_norm=student_t.ppf(.975,2*pp['a'])*predictive_student_scale
                pred_mm=pred*sy.scale+sy.mean
                sd_mm=sd*sy.scale
                per=[]
                for j,r in enumerate(val):
                    actual=dview[r].y
                    yv=sy.transform(actual)
                    if label.endswith('_GP'):
                        ylog=norm.logpdf(yv,loc=pred[j],scale=sd[j])-np.log(sy.scale)
                    else:
                        ylog=student_t.logpdf(yv,df=2*pp['a'],loc=pred[j],
                                              scale=predictive_student_scale[j])-np.log(sy.scale)
                    row={'ring':r,'truth_mm':actual.tolist(),'pred_mm':pred_mm[j].tolist(),
                         'residual_mm':(pred_mm[j]-actual).tolist(),'residual_norm':((pred_mm[j]-actual)/sy.scale).tolist(),
                         'predictive_sd_mm':sd_mm[j].tolist(),
                         'predictive_logpdf_y_per_target':ylog.tolist(),
                         'interval_95_halfwidth_mm':(halfwidth_norm[j]*sy.scale).tolist(),
                         'interval_95_covered':[bool(abs(pred_mm[j,z]-actual[z]) <= halfwidth_norm[j,z]*sy.scale[z]) for z in range(2)]}
                    scores[label].append(row);per.append(row)
                models_out[label]={'descriptor_mode':mode,'transformation':transform,'model':details,'val':per}
        for r in val:
            actual=dview[r].y
            base.append({'ring':r,'residual_mm':(sy.mean-actual).tolist(),'residual_norm':((sy.mean-actual)/sy.scale).tolist()})
        fold={'fold':k+1,'train_labeled':trL,'train_unlabeled':trU,'inner_val':val,
              'outer_test_locked':s['TEST_GESPERRT'],'x_scaler_fit_ids':sx.fitted_ids,
              'y_scaler_fit_ids':sy.fitted_ids,'quality_models':models_out}
        folds.append(fold)
        print(f'FOLD {k+1} TRAIN-L={trL} U={trU} INNER-VAL={val} OUTER-TEST LOCKED={s["TEST_GESPERRT"]}',flush=True)
        for label in models:
            ee=np.array([v['residual_mm'] for v in models_out[label]['val']])
            print(f'  {label:16s} | ring errors mm',np.round(ee,3).tolist(),flush=True)
    def calc(rows):
        e=np.array([r['residual_mm'] for r in rows]);en=np.array([r['residual_norm'] for r in rows])
        return {'rmse_mm':np.sqrt(np.mean(e*e,axis=0)).tolist(),
                'nrmse':np.sqrt(np.mean(en*en,axis=0)).tolist(),
                'mae_mm':np.mean(np.abs(e),axis=0).tolist(), 'n_val_rings':len(rows)}
    overall={'baseline':calc(base)}
    for label,rows in scores.items():
        overall[label]=calc(rows)
        overall[label]['empirical_95_coverage_8rings']=(np.mean([r['interval_95_covered'] for r in rows],axis=0)).tolist()
        overall[label]['mean_predictive_logpdf_y_per_target']=(np.mean([r['predictive_logpdf_y_per_target'] for r in rows],axis=0)).tolist()
        overall[label]['per_ring']=rows
    rep={'status':'EXPLORATORY four INNER folds; 8 ring-wise validation objects, no final selection',
         'purpose':'RADICAL quality-link comparison: no hidden HMM state needed for quality prediction',
         'model_warning':'These are deliberately structurally different COMPARATORS, not fully Bayesian jointly-trained SSR. GP kernel fixed; PCA point-estimate. Intervals use exact Student-t for NIG, Gaussian for fixed-noise GP, but coverage on only 8 rings is imprecise.',
         'config':{'data':data,'seed':seed,'features':features,'window':window,'n_pcs':n_pcs},
         'outer_test_locked':s['TEST_GESPERRT'],'folds':folds,'metrics':overall}
    if reference:
        ref=json.loads(Path(reference).read_text(encoding='utf-8'))
        if tuple(ref['outer_test_locked'])!=LOCKED:raise ValueError('Outer reference mismatch')
        cfg=ref['config']
        for nm in ('features','window','seed'):
            if cfg[nm]!=rep['config'][nm]:raise ValueError('CV reference mismatch: '+nm)
        rep['reference']={'baseline':ref['baseline'],
                          'diag':{k:v for k,v in ref['metrics']['diag'].items() if k!='per_ring'},
                          'full':{k:v for k,v in ref['metrics']['full'].items() if k!='per_ring'}}
    (out/'QUALITY_LINK_CV4.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf8')
    print('\n=== FOUR-FOLD RESULTS (8 inner validation rings) ===')
    for label,vals in overall.items():
        print(f'{label:16s} RMSE mm',np.round(vals['rmse_mm'],4),'NRMSE',np.round(vals['nrmse'],4))
    print('OUTER TEST SEALED:',s['TEST_GESPERRT'])
    print('REPORT:',(out/'QUALITY_LINK_CV4.json').resolve())
    return rep


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    ap.add_argument('--out',default='results/quality_link_challenge_cv4')
    ap.add_argument('--n-pcs',type=int,default=3,choices=[2,3,4])
    ap.add_argument('--reference-cv4',default=None)
    a=ap.parse_args()
    evaluate(data=a.data,out=a.out,reference=a.reference_cv4,n_pcs=a.n_pcs)
if __name__=='__main__':main()

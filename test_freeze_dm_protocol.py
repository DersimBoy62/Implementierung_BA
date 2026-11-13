"""Small tests for read-only experimental protocol freeze. No external test ring data."""
import copy
from freeze_dm_protocol import verify, OUTER,LABELED

def sample():
    metrics={}
    folds=[]
    rows={m:[] for m in ('SUP_L_ONLY','C0_SHARED_PREPROC','SSL_U_C1')}
    labs=('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18')
    U=('DM01','DM02','DM10','DM14')
    for i in range(4):
        v=labs[i*2:i*2+2]; l=tuple(x for x in labs if x not in v)
        modes={}
        for m in rows:
            t=list(l) if m=='SUP_L_ONLY' else sorted(set(l)|set(U))
            rr=[]
            for x in v:
                rr.append({'ring':x,'residual_mm':[0.,0.],'predictive_sd_mm':[1.,1.], 'pred_mm':[1.,2.]})
                rows[m].append(rr[-1]);
            modes[m]={'x_scaler_fit':t,'physics_scaler_fit':t,'y_scaler_fit':list(l),
                      'quality_train_ids':list(l),'unlabeled_elbo_weight':float(m=='SSL_U_C1'),'rows':rr}
        folds.append({'fold':i+1,'inner_val':v,'train_labeled':l,'train_unlabeled':U,
                      'outer_test_locked':OUTER,'modes':modes})
    for m in rows:metrics[m]={'rmse_mm':[0.,0.]}
    a={'outer_test_locked':OUTER,'folds':folds,'metrics':metrics,'per_ring':rows,
       'config':{'features':'V1_Kern_6','window':'F2','K':3,'epochs':10,'local_steps':8,
                 'seed':20261111,'head_variant':'SEPARATE_AXIAL'}}
    b={'outer_test_locked':OUTER,'folds':[None]*4,
       'per_ring':[{'ring':r,'pred_mm':[1.,2.]} for r in LABELED],
       'config':{'variant':'SEPARATE_AXIAL','K':3,'C':1.,'epochs':10,'local_steps':8,
                 'seed':20261111,'features':'V1_Kern_6','window':'F2',
                 'heads':[['log_duration','avg_radial_force','std_radial_force',
                           'delta_outer_diameter','delta_wall_thickness','max_radial_force'],
                          ['log_duration','avg_axial_force','std_axial_force',
                           'delta_height','max_axial_force']]}}
    return a,b

def fail(f,a,b):
    try:verify(a,b)
    except (AssertionError,ValueError):return
    raise AssertionError(f+' not rejected')

if __name__=='__main__':
    a,b=sample();assert verify(a,b)==0
    print('PASS accepts_registered_4fold_and_two_heads')
    aa=copy.deepcopy(a);aa['folds'][0]['modes']['SUP_L_ONLY']['y_scaler_fit'].append('DM07')
    fail('outer label in Y scaler',aa,b)
    print('PASS blocks_test_label_leakage')
    aa=copy.deepcopy(a);aa['folds'][1]['modes']['SSL_U_C1']['x_scaler_fit'].append('DM06')
    fail('inner val in X scaler',aa,b)
    print('PASS blocks_inner_validation_leakage')
    aa=copy.deepcopy(a);aa['config']['epochs']=12
    fail('unregistered epoch change',aa,b)
    print('PASS blocks_hyperparameter_change')
    bb=copy.deepcopy(b);bb['per_ring'][0]['pred_mm'][0]+=1e-4
    fail('C1 predictions do not reproduce previous run',a,bb)
    print('PASS blocks_reference_mismatch')

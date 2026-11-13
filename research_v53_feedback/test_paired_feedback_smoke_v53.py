"""Synthetic end-to-end smoke test: uses NO DM data, no user quality labels."""
import numpy as np
import paired_feedback_dev_v53 as m


def run():
    cfg=m.hard_guard()
    rng=np.random.default_rng(2153)
    rv={}
    for i,r in enumerate((*m.L,*m.U)):
        n=13+i%4
        t=np.arange(n,dtype=float)*.1
        u=np.linspace(0,1,n)
        x=np.stack([40+10*u+rng.normal(0,.5,n),18+9*u+rng.normal(0,.4,n),
                    .5+.2*u+rng.normal(0,.001,n),.07-.01*u+rng.normal(0,.001,n),
                    .06-.004*u+rng.normal(0,.001,n),.005+.02*u+rng.normal(0,.0005,n)],axis=1)
        y=np.array([.4+.09*i,3.1-.12*i]) if r in m.L else np.full(2,np.nan)
        rv[r]=m.dm.Ring(r,t,x,y,tuple(m.dm.SETS['V1_Kern_6']))
    # Exactly the same two-arm paired training/control flow, on manufactured arrays.
    res=m.fold_run(cfg,rv,1)
    assert len(res['paired_validation'])==2
    assert len(res['arms']['steps_8']['fixed_global_local_feedback_train_L'])==6
    assert all(len(res['arms'][f'steps_{s}']['history'])==10 for s in m.STEPS)
    assert np.isfinite([x for r in res['paired_validation'] for x in r['error_steps_8_mm']]).all()
    print('V53 SYNTHETIC E2E SMOKE PASS: two arms, 10 epochs, fixed-global feedback, paired validation, no DM data')

if __name__=='__main__':run()

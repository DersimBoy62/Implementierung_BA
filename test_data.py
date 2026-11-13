from dm_data import load_raw,view,ring_split,fit_x_scaler,fit_y_scaler,refuse_post_quality_inputs

p='input/DM_16_aktive_Ringe.pkl'
r=load_raw(p,expected_sha='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5')
assert len(r)==16
for w,total in [('F0',10385),('F1',9847),('F2',9292)]:
    data=view(r,features='V1_Kern_6',window=w)
    assert sum(len(x.t) for x in data.values())==total
    refuse_post_quality_inputs(data)
    print('PASS',w,total)
for v in ['V1_Kern_6','V2_Dynamik_10','V3_Zentrierung_12','V4_Umformung_10']:
    data=view(r,features=v,window='F2')
    assert set(data)==set(r)
    print('PASS',v,len(data['DM01'].names))
data=view(r)
sp=ring_split(data)
assert len(sp['L'])==8 and len(sp['U'])==4 and len(sp['TEST_GESPERRT'])==4
assert not(set(sp['L'])&set(sp['U'])|set(sp['TEST_GESPERRT'])&set(sp['U'])|set(sp['L'])&set(sp['TEST_GESPERRT']))
xs=fit_x_scaler(data,sp['L']+sp['U']);ys=fit_y_scaler(data,sp['L'])
assert not(set(xs.fitted_ids)&set(sp['TEST_GESPERRT']))
assert not(set(ys.fitted_ids)&set(sp['TEST_GESPERRT']))
print('PASS ring_split_and_fit_only_train')

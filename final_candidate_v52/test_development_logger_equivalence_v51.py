"""Reconstruct 12 permitted L/U *raw process* CSVs; verify exact F2 extraction.

No learning or quality evaluation. Loads the trusted serialized dataset after its
SHA256 check; it deserializes all ring objects but accesses only LOGGER entries
for the existing allowed 8 L + 4 U, never outer-ring features or quality labels.
"""
from __future__ import annotations
import argparse,csv,json,pathlib,tempfile
import numpy as np
import main_bayesian_ssr_v51 as drv
import input_safety_v51 as gate


def run(data,policy):
    cfg=drv.load_config()
    if drv.sha(data)!=cfg['dataset_sha256']:
        raise ValueError('Unrecognized dataset digest; refusing to deserialize')
    rings=drv.dev.load_development(data)
    p=json.loads(pathlib.Path(policy).read_text(encoding='utf-8'))
    ids=tuple(sorted(set(drv.LABELS)|set(drv.U_ONLY)))
    assert set(rings)==set(ids)
    # This second deserialization is restricted to the already verified local
    # trusted dataset and is needed to inspect the full logger before F2.
    import pickle
    with pathlib.Path(data).open('rb') as fh:all_raw=pickle.load(fh)
    results=[]
    with tempfile.TemporaryDirectory() as d:
        for rid in ids:
            raw=all_raw[rid]['logger']
            col=tuple(raw['matrix_columns'])
            ix=[col.index(n) for n in gate.REQUIRED]
            logfile=pathlib.Path(d)/(rid+'.csv')
            with logfile.open('w',newline='',encoding='utf-8') as fh:
                w=csv.writer(fh);w.writerow(gate.REQUIRED)
                w.writerows(np.asarray(raw['matrix'])[:,ix])
            t,x,report=gate.validate_raw_logger(logfile,p)
            assert np.array_equal(t,rings[rid].t), f'Changed F2 timebase for {rid}'
            assert np.array_equal(x,rings[rid].x), f'Changed F2 features for {rid}'
            assert np.isfinite(x).all()
            results.append({'ring_id':rid,'raw_rows':report['raw_rows'],'F2_rows':report['f2_rows'],
                            'matches_original_F2_bitwise':True})
            print('PASS REAL_PROCESS_ONLY',rid,'raw',report['raw_rows'],'F2',report['f2_rows'])
    print('PASS EXACT 12/12 LOGGER-F2-COMPATIBILITY; NO QUALITY OR OLD OUTER TEST EVALUATION')
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='input/DM_16_aktive_Ringe.pkl')
    parser.add_argument('--policy',default=str(pathlib.Path(__file__).parent/'DEVELOPMENT_INPUT_POLICY_V51.json'))
    args=parser.parse_args();run(args.data,args.policy)

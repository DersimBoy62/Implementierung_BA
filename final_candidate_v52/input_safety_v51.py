"""V5.1 fail-closed input gates for complete ring-roll *logger-only* CSV.

No target/quality access. Training-derived envelopes come from 8 L + 4 U F2
process series only. Old outer test rings excluded by the caller. Nothing is
imputed, interpolated, clipped or silently resampled. Phase-6 trailer confirms
that F2 ended; operator / machine completion record confirms ring completion.
Neither one guarantees a good ring or a reliable prediction for novel regimes.
"""
from __future__ import annotations
import csv
import hashlib
import json
import pathlib
import re
import unicodedata
from typing import Mapping
import numpy as np

FEATURE_COLUMNS=('F_rad63','F_ax64','Ist_D150','Ist_s151','Ist_h152','RWG55')
REQUIRED=('time',*FEATURE_COLUMNS,'phase804','LaserOK','MessEin')
UNITS=('percent_machine_signal','percent_machine_signal','m','m','m','m_per_s')
POLICY_VERSION='DM-V51-PROCESS-ONLY-SAFETY-2026-11-12-v1'
DT_S=0.1
DT_ABS_TOL_S=0.002  # pre-specified 2% logger clock tolerance, no interpolation
MIN_IDLE_SAMPLES=3  # explicit phase 6 with both laser and measurement off
MIN_TRAIN_RATIO=0.75 # conservative coverage gate of shortest DEVELOPMENT ring
DOMAIN_MARGIN_FRACTION=0.25 # ±25% of observed full L/U feature range, OOD not physics
SPAN_FRACTION=0.02 # 2% of smallest observed development signal excursion


def hash_file(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def normalize_and_check_new_ring_id(ring_id):
    raw=unicodedata.normalize('NFKC',str(ring_id)).strip().upper()
    # First look for historical aliases, including separators / leading zeros /
    # common suffixes (DM04, dm-0004, DM04_COPY, DM 04, ...).
    match=re.fullmatch(r'DM[\s_-]*0*(\d+)(?:[^\d].*)?',raw)
    if match and 1<=int(match.group(1))<=18:
        raise ValueError('Historical DM01-DM18 identifiers and aliases are forbidden for new-ring prediction')
    if not re.fullmatch(r'[A-Z][A-Z0-9_-]{1,31}',raw):
        raise ValueError('Ring ID must be 2-32 ASCII alphanumeric/underscore/hyphen chars starting with letter')
    return raw


def _v(row, field, line):
    try:return float(row[field])
    except (KeyError,TypeError,ValueError) as exc:raise ValueError(f'Invalid numeric field {field} on data row {line}') from exc


def read_full_logger_csv(path):
    """Return all rows, including inactive *before* and *after* the F2 window."""
    path=pathlib.Path(path)
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or tuple(reader.fieldnames)!=REQUIRED:
            raise ValueError('Expected EXACT process-only raw logger header/order: '+','.join(REQUIRED))
        rows=[]
        for line,r in enumerate(reader,2):
            if None in r or any(r[k] is None or not r[k].strip() for k in REQUIRED):
                raise ValueError(f'Missing/malformed CSV values on row {line}')
            rows.append([_v(r,k,line) for k in REQUIRED])
    if not rows:raise ValueError('Empty process logger')
    vals=np.asarray(rows,float)
    if vals.ndim!=2 or vals.shape[1]!=len(REQUIRED) or not np.isfinite(vals).all():
        raise ValueError('Non-finite process values')
    return vals


def _f2_and_end_check(vals):
    t=vals[:,0]
    if len(t)<20:raise ValueError('Logger too short for complete process run')
    dt=np.diff(t)
    if not np.all((dt>0)&(np.abs(dt-DT_S)<=DT_ABS_TOL_S)):
        bad=np.flatnonzero((dt<=0)|(np.abs(dt-DT_S)>DT_ABS_TOL_S))
        raise ValueError(f'Timestamp gap/clock mismatch near index {int(bad[0])}: expected {DT_S}±{DT_ABS_TOL_S}s; interpolation not permitted')
    phase,laser,mess=vals[:,-3],vals[:,-2],vals[:,-1]
    if not np.all(np.isin(laser,[0,1])) or not np.all(np.isin(mess,[0,1])):
        raise ValueError('LaserOK and MessEin must be exact binary flags')
    if not np.all(np.isin(phase,[1,2,6])):
        raise ValueError('Unexpected process phase; require original DM protocol review')
    idle=(phase==6)&(laser==0)&(mess==0)
    # DM06 legitimately has LaserOK=MessEin=1 during its initial phase-6 preamble.
    # The pre-F2 gate uses PHASE ONLY; the post-F2 completion gate uses all three flags.
    if not np.all(phase[:MIN_IDLE_SAMPLES]==6):
        raise ValueError('Missing initial phase-6 preamble: F2 might be truncated at start')
    if not np.all(idle[-MIN_IDLE_SAMPLES:]):
        raise ValueError('Missing final phase-6/laser-off/measurement-off trailer: process completion is not observable')
    f2=np.isin(phase,[1,2])&(laser==1)&(mess==1)
    ids=np.flatnonzero(f2)
    if len(ids)<2:raise ValueError('No sustained F2 process window')
    if not np.all(np.diff(ids)==1):raise ValueError('F2 islands are disconnected; refuse silent concatenation')
    end=int(ids[-1]);start=int(ids[0])
    if not (start>=MIN_IDLE_SAMPLES and end<len(t)-MIN_IDLE_SAMPLES):
        raise ValueError('Both raw preamble and completed F2 trailer required')
    if not np.all(~f2[end+1:]):raise ValueError('Unexpected later active F2 samples')
    # After active F2: at most five 0.1-second transition samples before
    # demonstrated phase-6/off; existing development raw log has 0 or 1.
    first_idle=np.flatnonzero(idle[end+1:])
    if len(first_idle)==0 or first_idle[0]>5:
        raise ValueError('No timely inactive phase-6 state after the F2 window')
    if np.any(~idle[end+1+int(first_idle[0]):]):
        raise ValueError('Non-idle statuses after process-end marker; inspect export')
    return ids


def make_training_policy(devrings: Mapping[str,object], labeled_ids, u_ids, locked_ids):
    """Training-only profile; consume only process x/t of allowed L/U rings."""
    ids=tuple(sorted(set(labeled_ids)|set(u_ids)))
    if len(set(labeled_ids))!=8 or len(set(u_ids))!=4 or (set(labeled_ids)&set(u_ids)):
        raise ValueError('Safety policy requires exact 8 disjoint L and 4 U process rings')
    if len(ids)!=12 or set(ids)!=set(devrings) or set(ids)&set(locked_ids):
        raise ValueError('Safety envelope must come exclusively from exactly 8 L + 4 U development rings')
    allx=[];n=[];spans=[]
    for rid in ids:
        ring=devrings[rid]
        x=np.asarray(ring.x,float);t=np.asarray(ring.t,float)
        if x.ndim!=2 or x.shape[1]!=6 or len(x)!=len(t) or len(x)<2 or not np.isfinite(x).all():
            raise ValueError(f'Invalid process-only training view {rid}')
        if not np.all(np.abs(np.diff(t)-DT_S)<=DT_ABS_TOL_S):
            raise ValueError(f'Unexpected process timebase in training view: {rid}')
        allx.append(x);n.append(len(t));spans.append(np.ptp(x,axis=0))
    xx=np.vstack(allx);lo=xx.min(axis=0);hi=xx.max(axis=0)
    span=hi-lo
    guard_lo=lo-DOMAIN_MARGIN_FRACTION*span
    guard_hi=hi+DOMAIN_MARGIN_FRACTION*span
    # Required dynamic movement on the three geometric sensors; forces/RWG
    # may be constant on genuine short runs and are not subject to span gate.
    minspans=np.min(np.asarray(spans),axis=0)
    policy={
      'protocol':POLICY_VERSION,
      'purpose':'DEV-derived conservative input QA; not physical limits, not independent validation',
      'training_ids_L':sorted(labeled_ids),'training_ids_U':sorted(u_ids),
      'never_used_test_ids':sorted(locked_ids),
      'channels':list(FEATURE_COLUMNS),'units':list(UNITS),
      'timebase_s':DT_S,'timebase_abs_tolerance_s':DT_ABS_TOL_S,
      'min_initial_and_final_phase6_samples':MIN_IDLE_SAMPLES,
      'minimum_f2_samples':int(np.ceil(MIN_TRAIN_RATIO*min(n))),
      'minimum_f2_duration_s':round(MIN_TRAIN_RATIO*(min(n)-1)*DT_S,5),
      'training_f2_min_samples':int(min(n)), 'training_f2_max_samples':int(max(n)),
      'feature_min_train':lo.tolist(),'feature_max_train':hi.tolist(),
      'feature_allowed_min':guard_lo.tolist(),'feature_allowed_max':guard_hi.tolist(),
      'allowed_domain_margin_fraction':DOMAIN_MARGIN_FRACTION,
      'min_geometric_signal_span':[float(max(v*SPAN_FRACTION,1e-7)) for v in minspans[2:5]],
      'training_development_ring_count':len(ids),
      'caveat':'Gate screens obvious export errors; it does not prove valid physical operation or model calibration.'
    }
    return policy


def validate_raw_logger(path,policy):
    if policy.get('protocol')!=POLICY_VERSION or tuple(policy.get('channels',[]))!=FEATURE_COLUMNS:
        raise ValueError('Unknown/unsupported V5.1 input policy')
    if tuple(policy.get('units',[]))!=UNITS or policy.get('timebase_s')!=DT_S:
        raise ValueError('Unexpected sensor units or timebase in input policy')
    vals=read_full_logger_csv(path)
    ids=_f2_and_end_check(vals)
    t=vals[ids,0].copy();x=vals[np.ix_(ids,range(1,7))].copy()
    if len(ids)<policy['minimum_f2_samples'] or float(t[-1]-t[0])+1e-7<policy['minimum_f2_duration_s']:
        raise ValueError(f'F2 too short for development domain: {len(ids)} samples, {t[-1]-t[0]:.2f}s; not a verified complete process')
    lo=np.asarray(policy['feature_allowed_min'],float)
    hi=np.asarray(policy['feature_allowed_max'],float)
    if lo.shape!=(6,) or hi.shape!=(6,) or not np.all(lo<hi) or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError('Malformed input-domain profile')
    outside=(x<lo)|(x>hi)
    if np.any(outside):
        i,j=np.argwhere(outside)[0]
        raise ValueError(f'Potential unit error / unseen process domain in {FEATURE_COLUMNS[int(j)]}: {x[i,j]:.6g} outside [{lo[j]:.6g}, {hi[j]:.6g}]. Review, no extrapolation.')
    if np.any(x[:,2:5]<=0):
        raise ValueError('Diameter, wall thickness and height must be positive in metres')
    minspan=np.asarray(policy['min_geometric_signal_span'],float)
    if np.any(np.ptp(x[:,2:5],axis=0)<minspan):
        raise ValueError('Geometric sensor flatline / insufficient physical evolution over F2')
    report={
      'input_safety_protocol':POLICY_VERSION,
      'raw_rows':len(vals),'f2_rows':len(t),
      'f2_t0_s':float(t[0]),'f2_tlast_s':float(t[-1]),
      'initial_phase6_proven':True,'final_inactive_phase6_proven':True,
      'sample_step_s':DT_S,
      'process_terminated_in_logger':True,
      'fraction_timepoints_outside_observed_training_range':[
        float(np.mean((x[:,j]<policy['feature_min_train'][j])|(x[:,j]>policy['feature_max_train'][j]))) for j in range(6)],
      'not_proven':'Raw phase-6 termination alone does not prove successfully finished ring or quality.'
    }
    return t,x,report


def verify_completion_attestation(path,ring_id,csv_path,report):
    """Independent human/machine closeout record bound to CSV SHA and F2 length.

    The program verifies format/binding, NOT the operator's truthfulness.
    """
    record=json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    required={'schema','ring_id','raw_logger_sha256','completed_ring_confirmed','evidence_type','evidence_reference',
              'confirmed_by','f2_samples','f2_last_time_s'}
    if set(record)!=required or record.get('schema')!='DM-V51-END-OF-RUN-ATTESTATION-v1':
        raise ValueError('Unrecognized or incomplete independent process completion attestation')
    canonical=normalize_and_check_new_ring_id(record['ring_id'])
    if canonical!=normalize_and_check_new_ring_id(ring_id):
        raise ValueError('Completion record ring ID does not match requested new ring')
    if record['completed_ring_confirmed'] is not True:
        raise ValueError('A finished-ring confirmation is required; phase 6 is not enough')
    if record['evidence_type'] not in ('operator_documented','machine_end_of_ring_event'):
        raise ValueError('Unrecognized completion evidence type')
    for k in ('evidence_reference','confirmed_by'):
        if not isinstance(record[k],str) or len(record[k].strip())<3:
            raise ValueError('Completion record requires a genuine traceable evidence reference')
    if record['raw_logger_sha256']!=hash_file(csv_path):
        raise ValueError('Completion evidence is not bound to this exact raw logger CSV')
    if not (type(record['f2_samples']) is int and record['f2_samples']==report['f2_rows']):
        raise ValueError('Completion evidence F2 sample count differs from CSV')
    if not (type(record['f2_last_time_s']) in (int,float) and np.isfinite(record['f2_last_time_s']) and abs(record['f2_last_time_s']-report['f2_tlast_s'])<=1e-5):
        raise ValueError('Completion evidence F2 end time differs from CSV')
    return {'evidence_type':record['evidence_type'],'evidence_reference':record['evidence_reference'],
            'completion_record_sha256':hash_file(path),
            'caveat':'Record binding verified; factual completion must be substantiated by its independent source.'}

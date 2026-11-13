#!/usr/bin/env python3
"""V5.6 read-only forensic check: do different KMeans seeds really yield different model starts?

Uses the exact frozen V5.2 process-scaling and sampled-frame KMeans expression.
No model fitting, no predictions, no error/quality scoring, no old outer-test rings.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH = ROOT / 'research_convergence_v55'
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
import convergence_probe_v55 as probe
m = probe.m
SEEDS = (20261111, 20261112, 20261113)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_kmeans(sampled, seed, k):
    # EXACT expression used by the SHA-locked dynamic_quality_v2.train.
    model = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(sampled)
    return model.cluster_centers_, model.labels_, float(model.inertia_)


def paired_summary(a, b, labels_a, labels_b, inertia_a, inertia_b):
    require(a.shape == b.shape, 'KMeans centers have different shapes')
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    ra, cb = linear_sum_assignment(distances)
    aligned_b = np.empty_like(a)
    aligned_b[ra] = b[cb]
    # Align B cluster IDs to A cluster IDs before comparing assignments.
    b_to_a = {int(j):int(i) for i,j in zip(ra,cb)}
    reindexed_b_labels = np.array([b_to_a[int(z)] for z in labels_b])
    d = np.linalg.norm(a - aligned_b,axis=1)
    return {
      'matched_centroid_euclidean_distances':d.tolist(),
      'max_matched_centroid_distance':float(np.max(d)),
      'mean_matched_centroid_distance':float(np.mean(d)),
      'same_centers_within_1e_minus_10':bool(np.max(d) <= 1e-10),
      'label_disagreement_fraction_after_optimal_alignment':float(np.mean(labels_a != reindexed_b_labels)),
      'inertia_difference':float(inertia_b-inertia_a),
      'has_materially_distinct_initialization':bool(np.max(d)>1e-6 or np.any(labels_a != reindexed_b_labels))}


def describe(sampled, k):
    fits = {s:exact_kmeans(sampled, s, k) for s in SEEDS}
    base = fits[SEEDS[0]]
    return {
      'n_sampled_rows':int(sampled.shape[0]),
      'n_features':int(sampled.shape[1]),
      'n_states':int(k),
      'seeds':{str(s):{'inertia':fits[s][2],
              'centroid_hash_sha256':hashlib.sha256(np.ascontiguousarray(fits[s][0]).tobytes()).hexdigest(),
              'cluster_sample_counts':np.bincount(fits[s][1],minlength=k).tolist()}
              for s in SEEDS},
      'pairs_vs_original_seed':{str(s):paired_summary(base[0],fits[s][0],base[1],fits[s][1],base[2],fits[s][2])
              for s in SEEDS[1:]}
    }


def self_test():
    # Exact cluster assignment permutation cannot imply materially different initialization.
    aa=np.array([[1.,2.],[7.,8.]])
    bb=aa[[1,0]]
    rr=paired_summary(aa,bb,np.array([0,0,1,1]),np.array([1,1,0,0]),1.,1.)
    require(rr['same_centers_within_1e_minus_10'], 'Permutations not handled')
    require(rr['label_disagreement_fraction_after_optimal_alignment']==0., 'Cluster IDs mismatched')
    rr2=paired_summary(aa,bb+np.array([[0.2,0.2],[0.2,0.2]]),np.array([0,0,1,1]),np.array([1,1,0,0]),1.,1.)
    require(rr2['has_materially_distinct_initialization'], 'Perturbation not detected')
    sample=np.vstack([np.linspace(-1,-.9,20)[:,None],np.linspace(0,.1,20)[:,None],np.linspace(1,1.1,20)[:,None]])
    z=describe(np.column_stack([sample, sample*2]),2)
    require(z['n_sampled_rows']==60 and len(z['pairs_vs_original_seed'])==2,'KMeans synthetic check wrong')
    m.load_config()  # verify the 10 mathematical SHA locks, without any real DM data.
    print('V56 SEED FORENSICS SELF-TEST: KMeans pairs, permutation alignment, perturbation, fixed code PASS; NO DM / NO TRAINING')


def audit(args):
    path=pathlib.Path(args.data)
    cfg, manifest=probe.preflight(path, pathlib.Path(args.model_dir))
    out=pathlib.Path(args.out) if args.out else None
    if out:
        require(out.parent.is_dir() and not out.exists() and not out.is_symlink(),
                'Report destination must be new and directory must exist (write once)')
        require(out.suffix.lower()=='.json', 'JSON output needed')
        require(probe.V52.resolve() not in out.resolve(strict=False).parents,'No writes to frozen V5.2 code')
        require(path.resolve()!=out.resolve(strict=False),'Do not overwrite source data')
    # Same ring allocation, sorted ring order, and train-only X-scaling as V5.2.
    # This loader is SHA-gated and masks process-only ring quality. No y is used here.
    rv=m.dev.load_development(path)
    require(set(rv)==set(m.LABELS)|set(m.U_ONLY) and not set(rv)&set(m.LOCKED), 'Development ring split changed')
    for rid in m.U_ONLY:
        require(np.isnan(rv[rid].y).all(), 'Unlabeled ring was not masked')
    ids=tuple(sorted(rv))
    sx=m.dm.fit_x_scaler(rv,ids)
    xs={rid:sx.transform(rv[rid].x) for rid in ids}
    sampled=np.vstack([x[::max(1,len(x)//120)] for x in xs.values()])
    details=describe(sampled,cfg['K'])
    result={'status':'READONLY_V56_SEED_INITIALIZATION_AUDIT_NOT_TRAINING',
        'source_sha256':cfg['dataset_sha256'],'config_sha256':m.release.FREEZE_CONFIG_HASH,
        'reference_model_sha256':manifest['model_sha256'],
        'code_training_uses':'KMeans(n_clusters=K, n_init=5, random_state=seed).fit(sampled)',
        'training_sample_source':'all 8L + 4U masked development process-X, V5.2 X-scaler, ring IDs sorted',
        'held_out_old_test_not_used':list(m.LOCKED),
        'quality_labels_not_used_for_this_audit':True,
        'no_model_fitted':True,
        'caveat':'KMeans output agreement does not prove all training parameters identical; distinct initial centers with same final ELBO is not proof of global convergence.',
        'kmeans':details}
    if out:
        out.open('x',encoding='utf8').write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('self-test')
    a=sub.add_parser('audit')
    a.add_argument('--data',required=True)
    a.add_argument('--model-dir',required=True)
    a.add_argument('--out')
    args=p.parse_args()
    if args.cmd=='self-test': self_test()
    else: audit(args)


if __name__=='__main__':
    main()

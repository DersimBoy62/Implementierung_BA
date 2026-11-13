"""Fixed four-fold development evaluation; historical outer rings excluded."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from .data_io import load_development,DATA_SHA256
from .split_guard import development_folds,guard_training_ids
from .coordinate_train import fit
from .posterior_predictive import predict_new_ring
from .preprocessing import fit_preprocessing
from .nig_heads import NIGHead
from .physics_descriptors import PHYSICAL_DESCRIPTOR_INDICES
from .metrics import regression_metrics
from .persistence import write_json

ARMS = ("train_mean","physical_nig","strict_c0","shared_c0","ssl")


def cross_validate(data_path,config,out,arms=ARMS,progress=None):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    if not arms or len(set(arms))!=len(arms) or set(arms)-set(ARMS):
        raise ValueError("Unknown or duplicated development evaluation arm")
    # The loader's label boundary includes only registered L; each fit receives
    # exactly its six L-TRAIN objects. Validation never enters fit/scaler calls.
    all_l,all_u = load_development(data_path)
    rows,folds = [],[]
    for fold in development_folds():
        train = {rid:all_l[rid] for rid in fold["L"]}
        val = {rid:all_l[rid] for rid in fold["VAL"]}
        guard_training_ids(train,all_u,val)
        for arm in arms:
            if progress:
                progress(dict(fold=fold["fold"],arm=arm,status="started"))
            if arm in ("ssl","shared_c0","strict_c0"):
                cfg = config if arm=="ssl" else replace(config,C=0.,include_unlabeled_preprocessing=arm=="shared_c0")
                model = fit(train,all_u,cfg,DATA_SHA256)
                predictions = {rid:predict_new_ring(model,r.process_only())["mean_mm"] for rid,r in val.items()}
                model_metadata = dict(config=cfg.to_dict(),scalers=model.preprocessing.to_dict(),
                    initializations="fresh TRAIN-X KMeans",history=model.history,split_manifest=model.split_manifest)
            elif arm=="physical_nig":
                # Strict L-only X/descriptor scaling; one intercept + same X-only groups.
                prep = fit_preprocessing(train,(),include_unlabeled=False)
                head_models = []
                for j,name in enumerate(("outside","height")):
                    idx = list(PHYSICAL_DESCRIPTOR_INDICES[name])
                    design = [np.r_[1.,prep.descriptors(r)[idx]] for r in train.values()]
                    prior = NIGHead(np.zeros(len(idx)+1),np.diag(np.r_[2.,np.full(len(idx),config.physical_var)]),2.5,1.5)
                    ys = [prep.y_scaler.transform(r.targets)[j] for r in train.values()]
                    head_models.append(prior.update(ys,[(v,np.outer(v,v)) for v in design]))
                predictions = {}
                for rid,r in val.items():
                    scaled = [head_models[j].mean@np.r_[1.,prep.descriptors(r)[list(PHYSICAL_DESCRIPTOR_INDICES[name])]]
                              for j,name in enumerate(("outside","height"))]
                    predictions[rid] = prep.y_scaler.inverse_transform(scaled).tolist()
                model_metadata = dict(scalers=prep.to_dict(),heads=[h.to_dict() for h in head_models],policy="L-only; intercept variance2; same physical groups")
            else:
                mean = np.stack([r.targets for r in train.values()]).mean(axis=0)
                predictions = {rid:mean.tolist() for rid in val}
                model_metadata = dict(train_label_mean_mm=mean.tolist(),fit_ids=sorted(train))
            fold_rows = [dict(fold=fold["fold"],arm=arm,ring_id=rid,truth_mm=r.targets.tolist(),pred_mm=predictions[rid]) for rid,r in val.items()]
            rows.extend(fold_rows)
            fold_report = dict(fold=fold,arm=arm,model=model_metadata,rows=fold_rows)
            folds.append(fold_report)
            write_json(out/f"fold_{fold['fold']}_{arm}.json",fold_report)
            if progress:
                progress(dict(fold=fold["fold"],arm=arm,status="complete"))
    metrics = {arm:regression_metrics([r["truth_mm"] for r in rows if r["arm"]==arm],[r["pred_mm"] for r in rows if r["arm"]==arm]) for arm in arms}
    report = dict(status="COMPLETED; descriptive reused-development CV; no independent external validation",
        dataset_sha256=DATA_SHA256,arms=list(arms),metrics=metrics,per_ring=rows,folds=folds,
        limitation="Eight labeled rings; repeated historical development; no calibrated intervals or statistical superiority claim")
    write_json(out/"development_cv.json",report)
    return report


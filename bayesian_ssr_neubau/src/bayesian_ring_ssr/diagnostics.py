"""Descriptive state/phase and residual summaries; no formal GOF claim."""
import numpy as np


def lag1(X):
    result = []
    for column in X.T:
        a,b = column[:-1],column[1:]
        result.append(float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else None)
    return result


def training_diagnostics(model, process_rings):
    output = {}
    for rid,chain in model.local_chains.items():
        ring = process_rings[rid]
        X = model.preprocessing.transform_process(ring).X
        gamma,xi = chain.gamma,chain.xi
        residual = X-gamma@np.stack([e.mean for e in model.process.emissions])
        expected_switch = xi.sum(axis=(1,2))-np.trace(xi,axis1=1,axis2=2)
        out = dict(expected_switches=float(expected_switch.sum()),mean_state=gamma.mean(0).tolist(),
                   residual_lag1=lag1(residual),residual_first_difference_lag1=lag1(np.diff(residual,axis=0)),
                   gamma_normalization_maxerror=float(np.max(np.abs(gamma.sum(1)-1))),
                   xi_row_maxerror=float(np.max(np.abs(xi.sum(2)-gamma[:-1]))),
                   xi_column_maxerror=float(np.max(np.abs(xi.sum(1)-gamma[1:]))))
        if ring.phases is not None:
            phases = ring.phases
            detrended = residual.copy()
            for p in (1,2):
                select = phases==p
                if np.sum(select)>2:
                    design = np.column_stack((np.ones(np.sum(select)),ring.times[select]-ring.times[select][0]))
                    detrended[select] -= design@np.linalg.lstsq(design,residual[select],rcond=None)[0]
            out["phase_detrended_lag1"] = lag1(detrended)
            boundary = np.flatnonzero(np.diff(phases)!=0)
            out["phase_boundary_times"] = ring.times[boundary+1].tolist()
            edge_times = ring.times[1:]
            near = np.zeros(len(edge_times),dtype=bool)
            for index in boundary:
                near |= np.abs(edge_times-ring.times[index+1])<=2.+1e-8
            out["expected_switches_within_2s"] = float(expected_switch[near].sum())
        output[rid] = out
    return dict(status="descriptive; residuals depend on inferred gamma; no formal test",rings=output)


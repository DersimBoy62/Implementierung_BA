"""Ring-level descriptive development metrics in millimetres."""
import numpy as np


def regression_metrics(truth, prediction):
    truth,prediction = np.asarray(truth,float),np.asarray(prediction,float)
    if truth.shape!=prediction.shape or truth.ndim!=2 or truth.shape[1]!=2 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Expected finite ring-by-two arrays")
    error = prediction-truth
    return dict(n_rings=len(error),rmse_mm=np.sqrt(np.mean(error**2,axis=0)).tolist(),mae_mm=np.mean(np.abs(error),axis=0).tolist())

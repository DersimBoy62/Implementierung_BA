"""Process-only prediction of the first two marginal quality moments."""
import numpy as np
from .schema import RingProcessSequence,LabeledRing,TARGETS
from .coordinate_train import head_descriptors
from .forward_backward import FB
from .hmmmoments import moments
from .nig_heads import augment


def predict_new_ring(model, ring):
    if not isinstance(ring,RingProcessSequence) or isinstance(ring,LabeledRing):
        raise TypeError("Prediction accepts only a process-only ring; call process_only() explicitly")
    X = model.preprocessing.transform_process(ring).X
    chain = FB(*model.process.potentials(X)).chain
    m,H = moments(chain)
    descriptors = head_descriptors(model.preprocessing,ring)
    scaled = [model.heads[j].predictive_moments(*augment(m,H,descriptors[j])) for j in range(2)]
    mean = model.preprocessing.y_scaler.inverse_transform(np.array([v[0] for v in scaled]))
    sd = np.sqrt(np.array([v[1] for v in scaled]))*model.preprocessing.y_scaler.scale
    return dict(ring_id=ring.ring_id,targets=list(TARGETS),mean_mm=mean.tolist(),sd_mm=sd.tolist(),unit="mm",
                uncertainty="Exact first/second moments within fitted structured VI; Markov mixture quantiles and calibration unverified",
                inference="process-only forward-backward with expected log potentials",state_moments=m.tolist())

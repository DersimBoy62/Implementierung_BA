"""Zwölf deterministische END-OF-RING-Kennwerte, aus TRAIN-X-skalierten Kanälen."""
from __future__ import annotations
import numpy as np

DESCRIPTOR_NAMES = (
    "log_duration", "avg_radial_force", "avg_axial_force", "std_radial_force",
    "std_axial_force", "delta_outer_diameter", "delta_wall_thickness", "delta_height",
    "avg_ring_growth_rate", "std_ring_growth_rate", "max_radial_force", "max_axial_force",
)
PHYSICAL_DESCRIPTOR_INDICES = {"outside": (0, 1, 3, 5, 6, 10), "height": (0, 2, 4, 7, 11)}


def extract_descriptors(X, times):
    x, t = np.asarray(X, dtype=float), np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 6 or x.shape != (len(t), 6):
        raise ValueError("Deskriptoren erwarten T>=6 und sechs geordnete F2-Kanäle")
    if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Ungültige Deskriptor-Eingabe oder Zeitordnung")
    # Kanalweise Reduktion erhält auch die historische float64-Summationsfolge.
    # axis=0 ist algebraisch gleich, kann aber rundungsbedingt abweichen.
    mean = np.array([np.mean(x[:, j]) for j in range(6)])
    std = np.array([np.std(x[:, j], ddof=0) for j in range(6)])
    change = x[-1] - x[0]
    return np.array((np.log(t[-1] - t[0]), mean[0], mean[1], std[0], std[1],
                     change[2], change[3], change[4], mean[5], std[5],
                     x[:, 0].max(), x[:, 1].max()), dtype=np.float64)


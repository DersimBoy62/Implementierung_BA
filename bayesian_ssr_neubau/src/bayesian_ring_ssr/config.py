"""Explicit, validated baseline controls; no implicit hyperparameter search."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import math


@dataclass(frozen=True)
class Config:
    K: int = 2
    C: float = 0.2
    variant: str = "M2"
    physical_var: float = 0.38526108747593474
    seed: int = 20261111
    local_maxiter: int = 8
    train_epochs: int = 30
    process_window: str = "F2"
    include_unlabeled_preprocessing: bool = True
    monotonic_atol: float = 1e-6

    def __post_init__(self):
        for name in ("K", "seed", "local_maxiter", "train_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.K < 2 or self.seed < 0 or self.local_maxiter < 0 or self.train_epochs < 1:
            raise ValueError("Invalid integer configuration")
        if not math.isfinite(self.C) or self.C < 0:
            raise ValueError("C must be finite and nonnegative")
        if not math.isfinite(self.physical_var) or self.physical_var <= 0:
            raise ValueError("physical_var must be positive")
        if self.variant != "M2" or self.process_window != "F2":
            raise ValueError("This baseline implements M2/F2 only")
        if not isinstance(self.include_unlabeled_preprocessing, bool):
            raise ValueError("include_unlabeled_preprocessing must be boolean")
        if not math.isfinite(self.monotonic_atol) or not 0 < self.monotonic_atol <= 1e-6:
            raise ValueError("monotonic_atol must be in (0,1e-6]")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


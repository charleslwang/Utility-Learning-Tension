# experiments/core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
from numpy.typing import NDArray

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------------
# Model config / capacity
# -------------------------
@dataclass
class ModelCfg:
    degree: int

class Capacity:
    @staticmethod
    def cap(cfg: ModelCfg) -> int:
        # proxy B(d) = d+1 (≈ VC/pdim up to constants)
        return cfg.degree + 1

# -------------------------
# Data generation
# -------------------------
def make_synthetic(n: int, sigma: float, flip_prob: float, rng: np.random.Generator,
                   dim: int = 1) -> Tuple[NDArray, NDArray]:
    """
    1D binary classification with piecewise cubic + heteroskedastic noise + label flips.
    If dim > 1, appends (dim-1) nuisance features.
    """
    # 1. Generate input distribution: mixture of two Gaussians
    n1 = rng.binomial(n, 0.5)
    x1 = rng.normal(-0.5, 0.2, size=n1)
    x2 = rng.normal(0.6, 0.25, size=n - n1)
    x = np.concatenate([x1, x2])
    rng.shuffle(x)
    x = np.clip(x, -1.5, 1.5).astype(np.float32)
    
    # 2. Piecewise cubic latent function with kink at 0.2
    f_x = 0.8*x - 1.6*x**2 + 0.9*x**3 + 0.8 * (x > 0.2) * (x - 0.2)**2
    
    # 3. Heteroskedastic noise: higher variance in the tails
    noise_std = 0.25 + 0.35 * (np.abs(x) > 0.8)
    epsilon = rng.normal(0, noise_std)
    
    # 4. Generate noisy labels
    y_prob = f_x + epsilon
    y = (y_prob >= 0).astype(int)
    
    # 5. Add label flips (10-20%)
    flip_mask = rng.uniform(0, 1, size=n) < flip_prob
    y[flip_mask] = 1 - y[flip_mask]
    
    # 6. Add nuisance dimensions if needed
    if dim > 1:
        z = rng.normal(0, 1, size=(n, dim-1))
        X = np.column_stack([x.reshape(-1, 1), z])
    else:
        X = x.reshape(-1, 1)
    
    return X, y

# -------------------------
# Features / trainer / eval
# -------------------------
class PolyERM:
    def __init__(self, degree: int, solver: str = 'lbfgs',
                 C: float = 0.1, max_iter: int = 10000, tol: float = 1e-6):
        """
        Polynomial logistic ERM with scaling after feature expansion.
        Smaller C = stronger L2 (more stable); higher max_iter + tighter tol to converge.
        """
        self.degree = degree
        self.solver = solver
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

        self.pipe = Pipeline(steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("scale", StandardScaler()),
            ("clf",   LogisticRegression(
                        solver=self.solver,
                        penalty='l2',
                        C=self.C,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        n_jobs=-1,
                        random_state=42
                     ))
        ])

    def fit(self, X: NDArray, y: NDArray) -> "PolyERM":
        self.pipe.fit(X, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        p = self.pipe.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

    def loss01(self, X: NDArray, y: NDArray) -> float:
        yhat = self.predict(X)
        return float(np.mean(yhat != y))

# -------------------------
# Proposer (H-axis)
# -------------------------
class DegreeProposer:
    def __init__(self, step: int = 1):
        self.step = step
    def next(self, cfg: ModelCfg) -> ModelCfg:
        return ModelCfg(degree=cfg.degree + self.step)

# -------------------------
# Gates
# -------------------------
from dataclasses import dataclass
@dataclass
class GateParams:
    epsV: float
    tau: float
    K: int

class TwoGate:
    def accept(self,
               val_old: float,
               val_new: float,
               cfg_new: ModelCfg,
               gp: GateParams) -> bool:
        cap_ok = (Capacity.cap(cfg_new) <= gp.K)
        val_ok = (val_new <= val_old - (2.0 * gp.epsV + gp.tau))
        return cap_ok and val_ok

class Destructive:
    def accept(self,
               val_old: float,
               val_new: float,
               cfg_new: ModelCfg,
               gp: GateParams) -> bool:
        # Permissive utility: accept any non-worse validation
        return val_new <= val_old

# -------------------------
# VC-style epsV calculator
# -------------------------
def eps_vc(K: int, n_v: int, delta_v: float, c0: float = 2.0) -> float:
    """
    ε_V ≈ c0 * sqrt((K + log(1/δ))/n_v)
    """
    return float(c0 * np.sqrt((K + np.log(1.0 / max(1e-12, delta_v))) / max(1, n_v)))

# -------------------------
# Oracle risk (for display only)
# -------------------------
def oracle_risk(X_test: NDArray, y_test: NDArray, K: int) -> Dict[int, float]:
    """
    Sweep degrees with B(d)<=K and report best test risk.
    Returns dict: degree -> test risk (also useful to plot curve).
    """
    risks = {}
    for d in range(K):  # B(d)=d+1<=K  => d<=K-1
        m = PolyERM(d).fit(X_test, y_test)  # NB: fitting on test just for oracle curve viz
        risks[d] = m.loss01(X_test, y_test)
    return risks

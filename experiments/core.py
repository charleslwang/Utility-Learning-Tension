from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

@dataclass
class ModelCfg:
    degree: int

class Capacity:
    @staticmethod
    def cap(cfg: ModelCfg) -> int:
        return cfg.degree + 1

def make_synthetic(n: int, sigma: float, flip_prob: float,
                   rng: np.random.Generator, dim: int = 1,
                   mode: str = "1d") -> Tuple[NDArray, NDArray]:
    if mode == "logit_poly":
        if dim < 2:
            raise ValueError("mode='logit_poly' requires dim>=2")
        X = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
        x1, x2 = X[:, 0], X[:, 1]
        logit = 3.0*x1 - 2.0*x2 + 1.5*(x1*x2) + rng.normal(0, sigma, size=n)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.uniform(0, 1, size=n) < p).astype(int)
    else:
        n1 = rng.binomial(n, 0.5)
        x1 = rng.normal(-0.5, 0.2, size=n1)
        x2 = rng.normal(0.6, 0.25, size=n - n1)
        x = np.concatenate([x1, x2])
        rng.shuffle(x)
        x = np.clip(x, -1.5, 1.5).astype(np.float32)
        f_x = 0.8*x - 1.6*x**2 + 0.9*x**3 + 0.8 * (x > 0.2) * (x - 0.2)**2
        noise_std = sigma + 0.35 * (np.abs(x) > 0.8)
        epsilon = rng.normal(0, noise_std)
        y_prob = f_x + epsilon
        y = (y_prob >= 0).astype(int)
        if dim > 1:
            z = rng.normal(0, 1, size=(n, dim-1))
            X = np.column_stack([x.reshape(-1, 1), z]).astype(np.float32)
        else:
            X = x.reshape(-1, 1).astype(np.float32)

    flips = rng.uniform(0, 1, size=n) < flip_prob
    y[flips] = 1 - y[flips]
    return X, y

class PolyERM:
    def __init__(self, degree: int, solver: str = 'saga',
                 C: float = 1.0, max_iter: int = 5000, tol: float = 1e-4):
        self.degree = degree
        if degree == 0:
            steps = [
                ("poly", PolynomialFeatures(degree=0, include_bias=True)),
                ("clf",  LogisticRegression(solver=solver, penalty='l2', C=C,
                                           max_iter=max_iter, tol=tol, n_jobs=-1,
                                           random_state=42))
            ]
        else:
            steps = [
                ("poly",  PolynomialFeatures(degree=degree, include_bias=True)),
                ("scale", StandardScaler(with_mean=False)),
                ("clf",   LogisticRegression(solver=solver, penalty='l2', C=C,
                                            max_iter=max_iter, tol=tol, n_jobs=-1,
                                            random_state=42))
            ]
        self.pipe = Pipeline(steps=steps)

    def fit(self, X: NDArray, y: NDArray) -> "PolyERM":
        self.pipe.fit(X, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        p = self.pipe.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

    def loss01(self, X: NDArray, y: NDArray) -> float:
        yhat = self.predict(X)
        return float(np.mean(yhat != y))

def eps_vc(K: int, n_v: int, delta_v: float, c0: float = 2.0) -> float:
    return float(c0 * np.sqrt((K + np.log(1.0 / max(1e-12, delta_v))) / max(1, n_v)))
    
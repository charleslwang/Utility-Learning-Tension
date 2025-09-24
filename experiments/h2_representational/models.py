"""Models and VC bound utilities for Experiment H.2."""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

@dataclass
class PTFLearner:
    d: int
    lambda_reg: float = 1e-3          # was 1e-6; stronger reg helps convergence
    max_iter: int = 2000              # was 10000; 2k is plenty once conditioning is fixed
    tol: float = 1e-5                 # was 1e-6; slightly looser tolerance helps
    random_state: int = 42

    def __post_init__(self):
        self.model = Pipeline([
            # KEY CHANGE: remove duplicate bias; LR will handle the intercept.
            ('poly',   PolynomialFeatures(degree=self.d, include_bias=False)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0 / (self.lambda_reg + 1e-10),
                penalty='l2',
                solver='liblinear',   # was 'lbfgs'
                max_iter=2000,        # was 10000
                tol=1e-4,             # was 1e-6
                random_state=self.random_state,
                fit_intercept=True,   # explicit (default True), pairs with include_bias=False if you switch later
            ))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PTFLearner":
        p = float(np.mean(y))
        if p < 0.4 or p > 0.6:
            self.model.named_steps['clf'].class_weight = 'balanced'
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.model.score(X, y)

    def get_weights(self) -> np.ndarray:
        return self.model.named_steps['clf'].coef_

    def get_decision_boundary(self, x_range: Tuple[float, float] = (-1, 1), n_points: int = 1000):
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = self.model.decision_function(x.reshape(-1, 1))
        return x, y

def compute_vc_bound(d: int, n: int, delta: float = 0.05, c: float = 2.0) -> float:
    """VC bound proxy for binary PTFs: ~ sqrt((d+1 + log 1/delta)/n)."""
    return c * np.sqrt((d + 1 + np.log(1.0 / delta)) / max(n, 1))

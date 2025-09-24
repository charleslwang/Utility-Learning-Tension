"""Data generation and loading for Experiment H.2."""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val:   np.ndarray
    y_val:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    metadata: Dict[str, Any]

def generate_polynomial_data(
    d_star: int,
    m: int,
    val_ratio: float = 0.2,
    test_size: int = 50000,
    noise_level: float = 0.0,
    seed: int = 42
) -> Dataset:
    rng = np.random.RandomState(seed)

    # true polynomial coefficients
    w_true = rng.randn(d_star + 1)

    # --- keep labels from collapsing to a single class ---
    # Center the threshold so the polynomial crosses near zero over [-1,1].
    X_grid = np.linspace(-1, 1, 2001)
    f_grid = sum(w_true[i] * (X_grid ** i) for i in range(d_star + 1))
    t = np.median(f_grid)          # roughly centers sign boundary
    w_true[0] -= t                 # shift bias term


    def true_function(X: np.ndarray) -> np.ndarray:
        X_poly = np.column_stack([X**i for i in range(d_star + 1)])
        return (X_poly @ w_true >= 0).astype(int)

    n_val = int(m * val_ratio)
    n_train = m - n_val

    X_train = rng.uniform(-1, 1, size=n_train)
    X_val   = rng.uniform(-1, 1, size=n_val)
    X_test  = rng.uniform(-1, 1, size=test_size)

    y_train = true_function(X_train)
    y_val   = true_function(X_val)
    y_test  = true_function(X_test)

    def add_noise(y: np.ndarray, p: float) -> np.ndarray:
        if p <= 0: return y
        flip = rng.random(len(y)) < p
        z = y.copy()
        z[flip] = 1 - z[flip]
        return z

    def label_fn(X_vec: np.ndarray) -> np.ndarray:
        X_poly = np.column_stack([X_vec**i for i in range(d_star + 1)])
        return (X_poly @ w_true >= 0).astype(int)

    def ensure_two_classes(X: np.ndarray, y: np.ndarray, make_labels_fn, rng, max_tries: int = 50):
        """If y has one class, resample X and relabel; try up to max_tries."""
        tries = 0
        while len(np.unique(y)) < 2 and tries < max_tries:
            X = rng.uniform(-1, 1, size=len(X))
            y = make_labels_fn(X)
            tries += 1
        return X, y

    # Enforce two classes on train/val (using clean labels)
    X_train, y_train = ensure_two_classes(X_train, y_train, label_fn, rng)
    X_val,   y_val   = ensure_two_classes(X_val,   y_val,   label_fn, rng)

    y_train = add_noise(y_train, noise_level)
    y_val   = add_noise(y_val,   noise_level)

    meta = dict(
        d_star=d_star, m=m,
        n_train=n_train, n_val=n_val, n_test=test_size,
        noise_level=noise_level, seed=seed, w_true=w_true
    )

    return Dataset(
        X_train=X_train.reshape(-1, 1),
        y_train=y_train,
        X_val=X_val.reshape(-1, 1),
        y_val=y_val,
        X_test=X_test.reshape(-1, 1),
        y_test=y_test,
        metadata=meta
    )

def save_dataset(dataset: Dataset, path: str) -> None:
    import os, joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(dataset, path)

def load_dataset(path: str) -> Dataset:
    import joblib
    return joblib.load(path)

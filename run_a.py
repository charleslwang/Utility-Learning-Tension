# experiments/run_a.py
from __future__ import annotations
import argparse, csv, os
import numpy as np
from numpy.typing import NDArray

from config import DataConfig
from utils import ensure_dir, set_all_seeds
from core import make_synthetic
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ----------------- helpers -----------------
def sigmoid(z: NDArray) -> NDArray:
    z = np.clip(z, -500.0, 500.0)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 1e-12, 1 - 1e-12)

def logistic_loss_and_grad(W: NDArray, X: NDArray, y: NDArray, l2: float) -> tuple[float, NDArray]:
    """
    X: (n, d), y in {0,1}, W: (d,)
    returns (loss, grad) for logistic regression with L2.
    """
    z = X @ W
    p = sigmoid(z)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * l2 * np.sum(W * W)
    grad = (X.T @ (p - y)) / X.shape[0] + l2 * W
    return float(loss), grad

def to_poly_features(X: NDArray, degree: int) -> NDArray:
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = poly.fit_transform(X)
    # scale after expansion for numerical stability; keep with_mean=False for sparse safety
    if degree > 0:
        Xp = StandardScaler(with_mean=False).fit_transform(Xp)
    return Xp.astype(np.float64)

# ----------------- single run -----------------
def run_once(
    seed: int,
    m: int,
    degree: int,
    T: int,
    policy: str,
    B_abs: float | None,
    l2: float,
    batch_size: int,
    data_cfg: DataConfig,
    out_path: str,
    eta0: float,
) -> None:
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    # data (fixed representation experiment)
    X, y = make_synthetic(
        n=data_cfg.m + data_cfg.n_v + data_cfg.n_test,
        sigma=data_cfg.sigma,
        flip_prob=data_cfg.flip_prob,
        rng=rng,
        dim=data_cfg.dim,
        mode="1d",        # your 1D generator for A-axis
    )
    X_tr, y_tr = X[:data_cfg.m], y[:data_cfg.m]
    X_te, y_te = X[-data_cfg.n_test:], y[-data_cfg.n_test:]

    # fixed representation: polynomial features of degree d
    Xtr = to_poly_features(X_tr, degree)
    Xte = to_poly_features(X_te, degree)

    n, d = Xtr.shape
    W = np.zeros(d, dtype=np.float64)

    step_mass = 0.0
    # If capped, enforce a hard step-mass budget B_abs; otherwise unlimited
    B_m = B_abs if (policy == "cap" and B_abs is not None) else np.inf

    # initial metrics
    p_tr = sigmoid(Xtr @ W)
    tr_loss = -np.mean(y_tr * np.log(p_tr) + (1 - y_tr) * np.log(1 - p_tr))
    p_te = sigmoid(Xte @ W)
    te_loss = -np.mean(y_te * np.log(p_te) + (1 - y_te) * np.log(1 - p_te))
    gap = te_loss - tr_loss

    with open(out_path, "a", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([seed, m, policy, degree, 0, f"{step_mass:.6f}", f"{tr_loss:.6f}", f"{te_loss:.6f}", f"{gap:.6f}"])

        for t in range(1, T + 1):
            # minibatch
            idx = rng.choice(n, size=min(batch_size, n), replace=False)
            Xb, yb = Xtr[idx], y_tr[idx]

            # gradient and step size
            _, g = logistic_loss_and_grad(W, Xb, yb, l2=l2)
            eta_t = eta0 / np.sqrt(t)

            # respect the budget if capped
            if step_mass + eta_t > B_m:
                break

            # SGD update
            W -= eta_t * g
            step_mass += eta_t

            # log every 10 iters
            if t % 10 == 0:
                p_tr = sigmoid(Xtr @ W)
                tr_loss = -np.mean(y_tr * np.log(p_tr) + (1 - y_tr) * np.log(1 - p_tr))
                p_te = sigmoid(Xte @ W)
                te_loss = -np.mean(y_te * np.log(p_te) + (1 - y_te) * np.log(1 - p_te))
                gap = te_loss - tr_loss
                wr.writerow([seed, m, policy, degree, t, f"{step_mass:.6f}", f"{tr_loss:.6f}", f"{te_loss:.6f}", f"{gap:.6f}"])

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["no_cap", "cap"], default="no_cap")
    ap.add_argument("--degree", type=int, default=5, help="fixed polynomial degree (representation)")
    ap.add_argument("--T", type=int, default=20000, help="total SGD steps")
    ap.add_argument("--eta0", type=float, default=0.05, help="base learning rate (eta_t = eta0 / sqrt(t))")
    ap.add_argument("--B", type=float, default=None, help="ABSOLUTE step-mass budget for cap policy (e.g., 2.5)")
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--m", type=int, nargs="+", default=[500])
    ap.add_argument("--n_v", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=50000)
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--flip", type=float, default=0.20)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out_dir", type=str, default="experiments/outputs")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    out_path = os.path.join(args.out_dir, f"a_axis_{args.policy}.csv")

    # header
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerow(["seed", "m", "policy", "degree", "iter", "step_mass", "train_loss", "test_loss", "gen_gap"])

    for m in args.m:
        print(f"\n=== m={m} policy={args.policy} degree={args.degree} ===")
        data_cfg = DataConfig(
            m=m,
            n_v=args.n_v,
            n_test=args.n_test,
            dim=args.dim,
            sigma=args.sigma,
            flip_prob=args.flip,
        )
        for seed in range(args.seeds):
            run_once(
                seed=seed,
                m=m,
                degree=args.degree,
                T=args.T,
                policy=args.policy,
                B_abs=args.B,
                l2=args.l2,
                batch_size=args.batch_size,
                data_cfg=data_cfg,
                out_path=out_path,
                eta0=args.eta0,
            )
        print(f"Completed m={m}")

    print(f"\nWrote: {out_path}")

if __name__ == "__main__":
    main()

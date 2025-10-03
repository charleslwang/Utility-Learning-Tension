from __future__ import annotations
import argparse, csv, os
import numpy as np
from numpy.typing import NDArray

from config import DataConfig
from utils import ensure_dir, set_all_seeds
from core import make_synthetic
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def sigmoid(z: NDArray) -> NDArray:
    z = np.clip(z, -500.0, 500.0)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 1e-12, 1 - 1e-12)

def logistic_loss(W: NDArray, X: NDArray, y: NDArray, l2: float) -> tuple[float, NDArray]:
    z = X @ W
    p = sigmoid(z)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * l2 * np.sum(W * W)
    grad = (X.T @ (p - y)) / X.shape[0] + l2 * W
    return float(loss), grad

def to_poly(X: NDArray, degree: int) -> NDArray:
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = poly.fit_transform(X)
    if degree > 0:
        Xp = StandardScaler(with_mean=False).fit_transform(Xp)
    return Xp.astype(np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["no_cap", "cap"], required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--T", type=int, default=50000)  # increased
    ap.add_argument("--eta0", type=float, default=0.01)  # constant LR
    ap.add_argument("--B", type=float, default=2.5)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_v", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--flip", type=float, default=0.20)
    ap.add_argument("--output_dir", type=str, default="experiments/outputs")
    ap.add_argument("--log_every", type=int, default=250)  # reduced logging
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    out_path = os.path.join(args.output_dir, f"a_axis_{args.policy}.csv")

    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerow(["seed", "m", "policy", "degree", "iter", "step_mass", 
                                "train_loss", "test_loss", "gen_gap"])

    print(f"Running policy '{args.policy}' for {args.seeds} seeds with m={args.m}...")
    
    for seed in range(args.seeds):
        set_all_seeds(seed)
        rng = np.random.default_rng(seed)

        X, y = make_synthetic(
            n=args.m + args.n_v + args.n_test,
            sigma=args.sigma, flip_prob=args.flip,
            rng=rng, dim=args.dim, mode="1d"
        )
        X_tr, y_tr = X[:args.m], y[:args.m]
        X_te, y_te = X[-args.n_test:], y[-args.n_test:]

        Xtr = to_poly(X_tr, args.degree)
        Xte = to_poly(X_te, args.degree)

        n, d = Xtr.shape
        W = np.zeros(d, dtype=np.float64)
        step_mass = 0.0
        B_m = args.B if args.policy == "cap" else np.inf

        # Log initial state
        p_tr = sigmoid(Xtr @ W)
        tr_loss = -np.mean(y_tr * np.log(p_tr) + (1 - y_tr) * np.log(1 - p_tr))
        p_te = sigmoid(Xte @ W)
        te_loss = -np.mean(y_te * np.log(p_te) + (1 - y_te) * np.log(1 - p_te))
        gap = te_loss - tr_loss

        with open(out_path, "a", newline="") as f:
            wr = csv.writer(f)
            wr.writerow([seed, args.m, args.policy, args.degree, 0, 
                        f"{step_mass:.6f}", f"{tr_loss:.6f}", f"{te_loss:.6f}", f"{gap:.6f}"])

            for t in range(1, args.T + 1):
                idx = rng.choice(n, size=min(args.batch_size, n), replace=True)
                Xb, yb = Xtr[idx], y_tr[idx]

                _, g = logistic_loss(W, Xb, yb, l2=args.l2)
                eta_t = args.eta0  # constant learning rate

                if step_mass + eta_t > B_m:
                    break

                W -= eta_t * g
                step_mass += eta_t

                if t % args.log_every == 0:
                    p_tr = sigmoid(Xtr @ W)
                    tr_loss = -np.mean(y_tr * np.log(p_tr) + (1 - y_tr) * np.log(1 - p_tr))
                    p_te = sigmoid(Xte @ W)
                    te_loss = -np.mean(y_te * np.log(p_te) + (1 - y_te) * np.log(1 - p_te))
                    gap = te_loss - tr_loss
                    wr.writerow([seed, args.m, args.policy, args.degree, t, 
                               f"{step_mass:.6f}", f"{tr_loss:.6f}", f"{te_loss:.6f}", f"{gap:.6f}"])
        
        if (seed + 1) % 5 == 0:
            print(f"  ... completed seed {seed + 1}/{args.seeds}")

    print(f"\nFinished. Wrote results to: {out_path}")

if __name__ == "__main__":
    main()

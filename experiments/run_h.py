# experiments/run_h.py
from __future__ import annotations

import argparse
import csv
import math
import os
import numpy as np

from utils import ensure_dir, set_all_seeds
from core import make_synthetic, ModelCfg, PolyERM, Capacity, eps_vc


def write_header(path: str) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "m", "seed", "policy", "degree", "accepted", "cap_ok", "val_ok",
            "train_loss", "val_loss", "test_loss", "epsV", "tau", "K"
        ])


def log_row(path: str, *, m: int, seed: int, policy: str, degree: int,
            accepted: bool, cap_ok: bool, val_ok: bool,
            tr: float, va: float, te: float, epsV: float, tau: float, K: int) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            m, seed, policy, degree, int(accepted), int(cap_ok), int(val_ok),
            f"{tr:.6f}", f"{va:.6f}", f"{te:.6f}", f"{epsV:.6f}", f"{tau:.6f}", K
        ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["twogate", "dest_train", "dest_val", "dest_val_nocap"], required=True)

    # === Fast & robust defaults (you can override via CLI) ===
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--m", type=int, default=150)
    p.add_argument("--n_v", type=int, default=60)          # small V -> reuse pressure
    p.add_argument("--n_test", type=int, default=2000)     # stable test estimate
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--sigma", type=float, default=1.2)     # harder data
    p.add_argument("--flip", type=float, default=0.35)     # Bayes floor

    # === Capacity + VC knobs ===
    p.add_argument("--K_max", type=int, default=60)
    p.add_argument("--K_mult", type=float, default=0.7)    # default K(m) = floor(K_mult * sqrt(m))
    p.add_argument("--K_override", type=int, default=None, # if set, overrides K directly (use degree+1)
                   help="If set, overrides K(capacity) directly; use degree+1.")
    p.add_argument("--c0", type=float, default=0.10)       # smaller VC constant -> realistic epsV
    p.add_argument("--tau_mult", type=float, default=0.20) # tau = tau_mult * epsV
    p.add_argument("--delta_v", type=float, default=0.05)

    # === TwoGate warm-up controls ===
    p.add_argument("--warmup_frac", type=float, default=0.5,
                   help="Fraction of max_degree to run with a lenient TwoGate margin.")
    p.add_argument("--warmup_scale", type=float, default=0.25,
                   help="During warm-up, require only warmup_scale*(epsV+tau) improvement.")

    # === Sweep range and model hyperparams ===
    p.add_argument("--max_degree", type=int, default=50)
    p.add_argument("--logreg_C", type=float, default=10.0) # allow overfit at high degree for contrast
    p.add_argument("--output_dir", type=str, default="experiments/outputs")
    args = p.parse_args()

    ensure_dir(args.output_dir)
    out_path = os.path.join(args.output_dir, f"h_axis_{args.policy}.csv")
    write_header(out_path)

    print(f"Running policy '{args.policy}' for {args.seeds} seeds with m={args.m}...")

    for seed in range(args.seeds):
        set_all_seeds(seed)
        rng = np.random.default_rng(seed)

        # --- Data ---
        X, y = make_synthetic(
            n=args.m + args.n_v + args.n_test,
            sigma=args.sigma, flip_prob=args.flip,
            rng=rng, dim=max(2, args.dim), mode="logit_poly"
        )
        X_tr, y_tr = X[:args.m], y[:args.m]
        X_va, y_va = X[args.m:args.m + args.n_v], y[args.m:args.m + args.n_v]
        X_te, y_te = X[-args.n_test:], y[-args.n_test:]

        # --- Gates ---
        K_default = int(math.floor(args.K_mult * math.sqrt(args.m)))
        K = args.K_override if args.K_override is not None else K_default
        K = min(args.K_max, K)

        epsV = eps_vc(K=K, n_v=args.n_v, delta_v=args.delta_v, c0=args.c0)
        tau = args.tau_mult * epsV

        # TwoGate warm-up boundary (in degrees), and ensure capacity doesn't stop before warm-up
        warmup_deg = max(1, int(args.max_degree * args.warmup_frac))
        K = max(K, warmup_deg + 1)  # cap is degree+1, so allow degrees up to warm-up

        # --- Start at degree 0 ---
        model_cur = PolyERM(0, C=args.logreg_C).fit(X_tr, y_tr)
        tr_cur = model_cur.loss01(X_tr, y_tr)
        va_cur = model_cur.loss01(X_va, y_va)
        te_cur = model_cur.loss01(X_te, y_te)

        log_row(out_path, m=args.m, seed=seed, policy=args.policy, degree=0,
                accepted=True, cap_ok=True, val_ok=True,
                tr=tr_cur, va=va_cur, te=te_cur, epsV=epsV, tau=tau, K=K)

        # --- Sequential self-modification over complexity ---
        for d in range(1, args.max_degree + 1):
            cand = PolyERM(d, C=args.logreg_C).fit(X_tr, y_tr)
            tr_new = cand.loss01(X_tr, y_tr)
            va_new = cand.loss01(X_va, y_va)
            te_new = cand.loss01(X_te, y_te)

            if args.policy == "twogate":
                cap_ok = (Capacity.cap(ModelCfg(d)) <= K)
                margin = (epsV + tau)
                if d <= warmup_deg:
                    # lenient warm-up
                    val_ok = (va_new <= va_cur - args.warmup_scale * margin)
                else:
                    # strict margin
                    val_ok = (va_new <= va_cur - margin)
                accepted = cap_ok and val_ok

            elif args.policy == "dest_val_nocap":
                cap_ok = True
                val_ok = (va_new <= va_cur - (epsV + tau))
                accepted = val_ok

            elif args.policy == "dest_val":
                cap_ok = True
                val_ok = (va_new <= va_cur)
                accepted = val_ok

            elif args.policy == "dest_train":
                cap_ok = True
                val_ok = (tr_new < tr_cur + 1e-3)
                accepted = val_ok

            else:
                raise ValueError(f"Unknown policy: {args.policy}")

            log_row(out_path, m=args.m, seed=seed, policy=args.policy, degree=d,
                    accepted=accepted, cap_ok=cap_ok, val_ok=val_ok,
                    tr=tr_new, va=va_new, te=te_new, epsV=epsV, tau=tau, K=K)

            if accepted:
                tr_cur, va_cur, te_cur = tr_new, va_new, te_new

        if (seed + 1) % 4 == 0:
            print(f"  ... completed seed {seed + 1}/{args.seeds}")

    print(f"\nFinished. Wrote results to: {out_path}")


if __name__ == "__main__":
    main()

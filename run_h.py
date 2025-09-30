# experiments/run_h.py
from __future__ import annotations
import argparse, csv, os, math, warnings
from typing import Tuple
import numpy as np

warnings.filterwarnings(
    'ignore',
    message='optree is installed but the version is too old',
    category=FutureWarning,
    module='torch.utils._pytree'
)

from config import DataConfig, GateConfig, RunConfig
from utils import ensure_dir, set_all_seeds
from core import (
    make_synthetic, ModelCfg, PolyERM,
    TwoGate, Capacity, GateParams
)

def compute_epsV(K: int, n_v: int, delta_v: float, c0: float) -> float:
    # uniform over the fixed capped reference family G_{K(m)}
    return c0 * math.sqrt((K + math.log(1.0 / delta_v)) / n_v)

def maybe_write_header(out_path: str):
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        with open(out_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "m","seed","policy","degree",
                "train_loss","val_loss","test_loss",
                "accepted","cap_ok","val_ok",
                "epsV","tau","K"
            ])

def log_row(out_path: str, m: int, seed: int, policy: str, degree: int,
            train_loss: float, val_loss: float, test_loss: float,
            accepted: bool, cap_ok: bool, val_ok: bool,
            epsV: float, tau: float, K: int):
    with open(out_path, "a", newline="") as f:
        csv.writer(f).writerow([
            m, seed, policy, degree,
            f"{train_loss:.6f}", f"{val_loss:.6f}", f"{test_loss:.6f}",
            int(accepted), int(cap_ok), int(val_ok),
            f"{epsV:.6f}", f"{tau:.6f}", K
        ])

def run_once(seed: int, m: int,
             data_cfg: DataConfig, gate_cfg: GateConfig,
             policy_name: str, out_path: str,
             destructive_slack: float, destructive_use_train: bool):

    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    # ----- data -----
    X, y = make_synthetic(
        n = data_cfg.m + data_cfg.n_v + data_cfg.n_test,
        sigma = data_cfg.sigma,
        flip_prob = data_cfg.flip_prob,
        rng = rng,
        dim = data_cfg.dim
    )
    X_tr = X[:data_cfg.m];          y_tr = y[:data_cfg.m]
    X_va = X[data_cfg.m:data_cfg.m+data_cfg.n_v];  y_va = y[data_cfg.m:data_cfg.m+data_cfg.n_v]
    X_te = X[-data_cfg.n_test:];    y_te = y[-data_cfg.n_test:]

    # ----- fixed, ex-ante gates for this run -----
    K = min(gate_cfg.K_max, int(math.floor(gate_cfg.K_mult * math.sqrt(m))))
    epsV = compute_epsV(K=K, n_v=data_cfg.n_v, delta_v=gate_cfg.delta_v, c0=gate_cfg.c0)
    tau  = gate_cfg.tau_mult * epsV
    gp_fixed = GateParams(K=K, epsV=epsV, tau=tau)

    # ----- initialize model at degree 0 -----
    degree = 0
    model = PolyERM(degree).fit(X_tr, y_tr)
    train_loss = model.loss01(X_tr, y_tr)
    val_loss   = model.loss01(X_va, y_va)
    test_loss  = model.loss01(X_te, y_te)
    log_row(out_path, m, seed, policy_name, degree,
            train_loss, val_loss, test_loss,
            True, True, True, epsV, tau, K)

    # ----- propose increasing degrees -----
    while True:
        proposal = degree + 1
        if proposal > gate_cfg.K_max:
            break

        new_model = PolyERM(proposal).fit(X_tr, y_tr)
        tr_new = new_model.loss01(X_tr, y_tr)
        va_new = new_model.loss01(X_va, y_va)
        te_new = new_model.loss01(X_te, y_te)

        if policy_name == "twogate":
            cap_ok = (Capacity.cap(ModelCfg(degree=proposal)) <= gp_fixed.K)
            val_ok = (va_new <= val_loss - (2.0 * gp_fixed.epsV + gp_fixed.tau))
            accepted = cap_ok and val_ok
        else:  # destructive
            capacity_increases = (proposal > degree)
            if destructive_use_train:
                # Accept if capacity grows AND training doesn't worsen much
                train_ok = (tr_new <= train_loss + destructive_slack)
                accepted = capacity_increases and train_ok
            else:
                # Accept if capacity grows AND validation doesn't worsen much
                val_threshold_ok = (va_new <= val_loss + destructive_slack)
                accepted = capacity_increases and val_threshold_ok
            cap_ok = True  # no cap gate
            val_ok = accepted  # for logging

        log_row(out_path, m, seed, policy_name, proposal,
                tr_new, va_new, te_new, accepted, cap_ok, val_ok,
                epsV, tau, K)

        if not accepted:
            break

        # commit
        degree, model = proposal, new_model
        train_loss, val_loss, test_loss = tr_new, va_new, te_new

def main():
    p = argparse.ArgumentParser()
    # policies / sweep
    p.add_argument("--policy", choices=["twogate", "destructive"], default="twogate")
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--m", type=int, nargs="+", default=[500, 1000, 2000, 5000])

    # data
    p.add_argument("--n_v", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=50000)
    p.add_argument("--dim", type=int, default=1)
    p.add_argument("--sigma", type=float, default=0.6)
    p.add_argument("--flip", type=float, default=0.15)

    # gates
    p.add_argument("--K_max", type=int, default=60)
    p.add_argument("--K_mult", type=float, default=0.5, help="K(m)=K_mult*sqrt(m)")
    p.add_argument("--c0", type=float, default=2.0)
    p.add_argument("--tau_mult", type=float, default=0.5)
    p.add_argument("--delta_v", type=float, default=0.05)

    # destructive utility knobs
    p.add_argument("--destructive_slack", type=float, default=0.003)
    p.add_argument("--destructive_use_train", action="store_true")

    # io
    p.add_argument("--output_dir", type=str, default="experiments/outputs")

    args = p.parse_args()

    # configs
    gate_cfg = GateConfig(
        K_max=args.K_max, K_mult=args.K_mult,
        c0=args.c0, tau_mult=args.tau_mult, delta_v=args.delta_v
    )

    ensure_dir(args.output_dir)
    out_path = os.path.join(args.output_dir, f"h_axis_{args.policy}.csv")
    maybe_write_header(out_path)

    for m in args.m:
        print(f"\n=== m={m} policy={args.policy} ===")
        data_cfg = DataConfig(
            m=m, n_v=args.n_v, n_test=args.n_test,
            dim=args.dim, sigma=args.sigma, flip_prob=args.flip
        )
        for seed in range(args.seeds):
            run_once(seed, m, data_cfg, gate_cfg, args.policy, out_path,
                     destructive_slack=args.destructive_slack,
                     destructive_use_train=args.destructive_use_train)
            print(f"seed {seed} done.")

    print(f"\nWrote: {out_path}")

if __name__ == "__main__":
    main()

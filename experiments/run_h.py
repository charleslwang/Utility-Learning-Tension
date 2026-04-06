import argparse
import csv
import math
import os
from typing import Dict, List

try:
    from experiments.neural_common import AdaptiveMLP, correction_audit, evaluate_all_splits, fit_classifier, make_split_bundle
    from experiments.utils import ensure_dir, get_device, parse_int_list, set_all_seeds
except ImportError:
    from neural_common import AdaptiveMLP, correction_audit, evaluate_all_splits, fit_classifier, make_split_bundle
    from utils import ensure_dir, get_device, parse_int_list, set_all_seeds


def vc_margin(capacity_proxy: int, n_v: int, delta_v: float, c0: float) -> float:
    return c0 * math.sqrt((capacity_proxy + math.log(1.0 / max(delta_v, 1e-8))) / max(n_v, 1))



def build_model(input_dim: int, width: int, depth: int, dropout: float) -> AdaptiveMLP:
    return AdaptiveMLP(input_dim=input_dim, hidden_width=width, depth=depth, dropout=dropout)



def fieldnames() -> List[str]:
    return [
        "seed",
        "m",
        "policy",
        "proposal_step",
        "candidate_width",
        "accepted",
        "cap_ok",
        "val_ok",
        "width_cap",
        "epsV",
        "tau",
        "candidate_train_loss",
        "candidate_train_acc",
        "candidate_val_loss",
        "candidate_val_acc",
        "candidate_test_iid_loss",
        "candidate_test_iid_acc",
        "candidate_test_shift_loss",
        "candidate_test_shift_acc",
        "candidate_gen_gap",
        "candidate_correction_train_pre_acc",
        "candidate_correction_train_post_acc",
        "candidate_correction_train_gain",
        "candidate_correction_train_pre_loss",
        "candidate_correction_train_post_loss",
        "candidate_correction_eval_pre_acc",
        "candidate_correction_eval_post_acc",
        "candidate_correction_eval_gain",
        "candidate_correction_eval_pre_loss",
        "candidate_correction_eval_post_loss",
        "candidate_correction_param_shift_l2",
        "candidate_correction_pre_acc",
        "candidate_correction_post_acc",
        "candidate_correction_gain",
        "candidate_correction_pre_loss",
        "candidate_correction_post_loss",
        "active_width",
        "active_train_loss",
        "active_train_acc",
        "active_val_loss",
        "active_val_acc",
        "active_test_iid_loss",
        "active_test_iid_acc",
        "active_test_shift_loss",
        "active_test_shift_acc",
        "active_gen_gap",
        "active_correction_train_pre_acc",
        "active_correction_train_post_acc",
        "active_correction_train_gain",
        "active_correction_train_pre_loss",
        "active_correction_train_post_loss",
        "active_correction_eval_pre_acc",
        "active_correction_eval_post_acc",
        "active_correction_eval_gain",
        "active_correction_eval_pre_loss",
        "active_correction_eval_post_loss",
        "active_correction_param_shift_l2",
        "active_correction_pre_acc",
        "active_correction_post_acc",
        "active_correction_gain",
        "active_correction_pre_loss",
        "active_correction_post_loss",
    ]


def evaluate_candidate(args: argparse.Namespace, seed: int, width: int, device) -> Dict[str, float]:
    bundle = make_split_bundle(
        m=args.m,
        n_v=args.n_v,
        n_test=args.n_test,
        n_shift=args.n_shift,
        n_correction=args.n_correction,
        input_dim=args.input_dim,
        shortcut_strength=args.shortcut_strength,
        label_noise=args.label_noise,
        feature_noise=args.feature_noise,
        seed=seed,
        shift_alignment=args.shift_alignment,
        correction_alignment=args.correction_alignment,
        correction_eval_alignment=args.correction_eval_alignment,
    )
    model = build_model(args.input_dim, width, args.depth, args.dropout)
    model = fit_classifier(
        model=model,
        train_split=bundle.train,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
    )
    metrics = evaluate_all_splits(model, bundle, device=device, batch_size=args.eval_batch_size)
    audit = correction_audit(
        model=model,
        correction_split=bundle.correction,
        correction_eval_split=bundle.correction_eval,
        device=device,
        steps=args.correction_steps,
        lr=args.correction_lr,
        batch_size=args.correction_batch_size,
        weight_decay=args.correction_weight_decay,
    )
    metrics.update(audit)
    return metrics


def make_row(seed: int, args: argparse.Namespace, proposal_step: int, width: int, accepted: bool, cap_ok: bool, val_ok: bool, width_cap: int, eps_v: float, tau: float, candidate: Dict[str, float], active: Dict[str, float]) -> Dict[str, float]:
    return {
        "seed": seed,
        "m": args.m,
        "policy": args.policy,
        "proposal_step": proposal_step,
        "candidate_width": width,
        "accepted": accepted,
        "cap_ok": cap_ok,
        "val_ok": val_ok,
        "width_cap": width_cap,
        "epsV": eps_v,
        "tau": tau,
        "candidate_train_loss": candidate["train_loss"],
        "candidate_train_acc": candidate["train_acc"],
        "candidate_val_loss": candidate["val_loss"],
        "candidate_val_acc": candidate["val_acc"],
        "candidate_test_iid_loss": candidate["test_iid_loss"],
        "candidate_test_iid_acc": candidate["test_iid_acc"],
        "candidate_test_shift_loss": candidate["test_shift_loss"],
        "candidate_test_shift_acc": candidate["test_shift_acc"],
        "candidate_gen_gap": candidate["gen_gap"],
        "candidate_correction_train_pre_acc": candidate["correction_train_pre_acc"],
        "candidate_correction_train_post_acc": candidate["correction_train_post_acc"],
        "candidate_correction_train_gain": candidate["correction_train_gain"],
        "candidate_correction_train_pre_loss": candidate["correction_train_pre_loss"],
        "candidate_correction_train_post_loss": candidate["correction_train_post_loss"],
        "candidate_correction_eval_pre_acc": candidate["correction_eval_pre_acc"],
        "candidate_correction_eval_post_acc": candidate["correction_eval_post_acc"],
        "candidate_correction_eval_gain": candidate["correction_eval_gain"],
        "candidate_correction_eval_pre_loss": candidate["correction_eval_pre_loss"],
        "candidate_correction_eval_post_loss": candidate["correction_eval_post_loss"],
        "candidate_correction_param_shift_l2": candidate["correction_param_shift_l2"],
        "candidate_correction_pre_acc": candidate["correction_pre_acc"],
        "candidate_correction_post_acc": candidate["correction_post_acc"],
        "candidate_correction_gain": candidate["correction_gain"],
        "candidate_correction_pre_loss": candidate["correction_pre_loss"],
        "candidate_correction_post_loss": candidate["correction_post_loss"],
        "active_width": active["width"],
        "active_train_loss": active["train_loss"],
        "active_train_acc": active["train_acc"],
        "active_val_loss": active["val_loss"],
        "active_val_acc": active["val_acc"],
        "active_test_iid_loss": active["test_iid_loss"],
        "active_test_iid_acc": active["test_iid_acc"],
        "active_test_shift_loss": active["test_shift_loss"],
        "active_test_shift_acc": active["test_shift_acc"],
        "active_gen_gap": active["gen_gap"],
        "active_correction_train_pre_acc": active["correction_train_pre_acc"],
        "active_correction_train_post_acc": active["correction_train_post_acc"],
        "active_correction_train_gain": active["correction_train_gain"],
        "active_correction_train_pre_loss": active["correction_train_pre_loss"],
        "active_correction_train_post_loss": active["correction_train_post_loss"],
        "active_correction_eval_pre_acc": active["correction_eval_pre_acc"],
        "active_correction_eval_post_acc": active["correction_eval_post_acc"],
        "active_correction_eval_gain": active["correction_eval_gain"],
        "active_correction_eval_pre_loss": active["correction_eval_pre_loss"],
        "active_correction_eval_post_loss": active["correction_eval_post_loss"],
        "active_correction_param_shift_l2": active["correction_param_shift_l2"],
        "active_correction_pre_acc": active["correction_pre_acc"],
        "active_correction_post_acc": active["correction_post_acc"],
        "active_correction_gain": active["correction_gain"],
        "active_correction_pre_loss": active["correction_pre_loss"],
        "active_correction_post_loss": active["correction_post_loss"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["twogate", "dest_train", "dest_val", "dest_val_nocap"], required=True)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n_v", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--n_shift", type=int, default=2000)
    parser.add_argument("--n_correction", type=int, default=128)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--shortcut_strength", type=float, default=2.6)
    parser.add_argument("--label_noise", type=float, default=0.08)
    parser.add_argument("--feature_noise", type=float, default=0.9)
    parser.add_argument("--widths", type=str, default="4,8,16,32,64")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--K_mult", type=float, default=1.1)
    parser.add_argument("--K_override", type=int, default=None)
    parser.add_argument("--c0", type=float, default=0.08)
    parser.add_argument("--tau_mult", type=float, default=0.15)
    parser.add_argument("--delta_v", type=float, default=0.05)
    parser.add_argument("--correction_steps", type=int, default=12)
    parser.add_argument("--correction_lr", type=float, default=0.08)
    parser.add_argument("--correction_batch_size", type=int, default=32)
    parser.add_argument("--correction_weight_decay", type=float, default=0.0)
    parser.add_argument("--shift_alignment", type=float, default=-0.35)
    parser.add_argument("--correction_alignment", type=float, default=0.0)
    parser.add_argument("--correction_eval_alignment", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="experiments/outputs")
    args = parser.parse_args()

    widths = parse_int_list(args.widths)
    width_cap = args.K_override if args.K_override is not None else int(math.floor(args.K_mult * math.sqrt(args.m)))
    width_cap = max(widths[0], width_cap)
    eps_v = vc_margin(capacity_proxy=width_cap, n_v=args.n_v, delta_v=args.delta_v, c0=args.c0)
    tau = args.tau_mult * eps_v

    ensure_dir(args.output_dir)
    output_path = os.path.join(args.output_dir, f"h_axis_{args.policy}.csv")
    device = get_device(args.device)

    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames())
        writer.writeheader()
        for seed in range(args.seeds):
            set_all_seeds(seed)
            incumbent_metrics = evaluate_candidate(args, seed=seed, width=widths[0], device=device)
            incumbent_metrics["width"] = widths[0]
            writer.writerow(
                make_row(
                    seed=seed,
                    args=args,
                    proposal_step=0,
                    width=widths[0],
                    accepted=True,
                    cap_ok=widths[0] <= width_cap,
                    val_ok=True,
                    width_cap=width_cap,
                    eps_v=eps_v,
                    tau=tau,
                    candidate=incumbent_metrics,
                    active=incumbent_metrics,
                )
            )
            handle.flush()

            for proposal_step, width in enumerate(widths[1:], start=1):
                candidate_metrics = evaluate_candidate(args, seed=seed, width=width, device=device)
                candidate_metrics["width"] = width
                if args.policy == "twogate":
                    cap_ok = width <= width_cap
                    val_ok = candidate_metrics["val_loss"] <= incumbent_metrics["val_loss"] - (eps_v + tau)
                    accepted = cap_ok and val_ok
                elif args.policy == "dest_val_nocap":
                    cap_ok = True
                    val_ok = candidate_metrics["val_loss"] <= incumbent_metrics["val_loss"] - (eps_v + tau)
                    accepted = val_ok
                elif args.policy == "dest_val":
                    cap_ok = True
                    val_ok = candidate_metrics["val_loss"] <= incumbent_metrics["val_loss"]
                    accepted = val_ok
                elif args.policy == "dest_train":
                    cap_ok = True
                    val_ok = candidate_metrics["train_loss"] <= incumbent_metrics["train_loss"] - 1e-4
                    accepted = val_ok
                else:
                    raise ValueError(f"Unknown policy: {args.policy}")

                if accepted:
                    incumbent_metrics = candidate_metrics

                writer.writerow(
                    make_row(
                        seed=seed,
                        args=args,
                        proposal_step=proposal_step,
                        width=width,
                        accepted=accepted,
                        cap_ok=cap_ok,
                        val_ok=val_ok,
                        width_cap=width_cap,
                        eps_v=eps_v,
                        tau=tau,
                        candidate=candidate_metrics,
                        active=incumbent_metrics,
                    )
                )
                handle.flush()

            if (seed + 1) % 4 == 0:
                print(f"Completed seed {seed + 1}/{args.seeds}")

    print(f"Wrote H-axis results to {output_path}")


if __name__ == "__main__":
    main()

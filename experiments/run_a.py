import argparse
import csv
import math
import os
from typing import Dict, List

import torch
from torch import nn

try:
    from experiments.neural_common import TeachabilityMLP, evaluate_teachability_model, make_loader, make_split_bundle, teachability_probe
    from experiments.utils import ensure_dir, get_device, set_all_seeds
except ImportError:
    from neural_common import TeachabilityMLP, evaluate_teachability_model, make_loader, make_split_bundle, teachability_probe
    from utils import ensure_dir, get_device, set_all_seeds



def fieldnames() -> List[str]:
    return [
        "seed",
        "m",
        "policy",
        "stage",
        "step_mass",
        "step_budget",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "test_iid_loss",
        "test_iid_acc",
        "test_shift_loss",
        "test_shift_acc",
        "gen_gap",
        "probe_train_pre_acc",
        "probe_train_post_acc",
        "probe_train_gain",
        "probe_train_pre_loss",
        "probe_train_post_loss",
        "probe_eval_pre_acc",
        "probe_eval_post_acc",
        "probe_eval_gain",
        "probe_eval_pre_loss",
        "probe_eval_post_loss",
        "probe_param_shift_l2",
        "probe_pre_acc",
        "probe_post_acc",
        "probe_gain",
        "probe_pre_loss",
        "probe_post_loss",
    ]



def train_stage(
    model: TeachabilityMLP,
    train_split,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
) -> int:
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(train_split, batch_size=batch_size, shuffle=True)
    updates = 0
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            updates += 1
    return updates



def make_row(seed: int, args: argparse.Namespace, stage: int, step_mass: float, metrics: Dict[str, float], probe: Dict[str, float]) -> Dict[str, float]:
    return {
        "seed": seed,
        "m": args.m,
        "policy": args.policy,
        "stage": stage,
        "step_mass": step_mass,
        "step_budget": args.B,
        "train_loss": metrics["train_loss"],
        "train_acc": metrics["train_acc"],
        "val_loss": metrics["val_loss"],
        "val_acc": metrics["val_acc"],
        "test_iid_loss": metrics["test_iid_loss"],
        "test_iid_acc": metrics["test_iid_acc"],
        "test_shift_loss": metrics["test_shift_loss"],
        "test_shift_acc": metrics["test_shift_acc"],
        "gen_gap": metrics["test_iid_loss"] - metrics["train_loss"],
        "probe_train_pre_acc": probe["probe_train_pre_acc"],
        "probe_train_post_acc": probe["probe_train_post_acc"],
        "probe_train_gain": probe["probe_train_gain"],
        "probe_train_pre_loss": probe["probe_train_pre_loss"],
        "probe_train_post_loss": probe["probe_train_post_loss"],
        "probe_eval_pre_acc": probe["probe_eval_pre_acc"],
        "probe_eval_post_acc": probe["probe_eval_post_acc"],
        "probe_eval_gain": probe["probe_eval_gain"],
        "probe_eval_pre_loss": probe["probe_eval_pre_loss"],
        "probe_eval_post_loss": probe["probe_eval_post_loss"],
        "probe_param_shift_l2": probe["probe_param_shift_l2"],
        "probe_pre_acc": probe["probe_pre_acc"],
        "probe_post_acc": probe["probe_post_acc"],
        "probe_gain": probe["probe_gain"],
        "probe_pre_loss": probe["probe_pre_loss"],
        "probe_post_loss": probe["probe_post_loss"],
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["no_cap", "cap"], required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--m", type=int, default=384)
    parser.add_argument("--n_v", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--n_shift", type=int, default=2000)
    parser.add_argument("--n_correction", type=int, default=96)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--shortcut_strength", type=float, default=2.6)
    parser.add_argument("--label_noise", type=float, default=0.08)
    parser.add_argument("--feature_noise", type=float, default=0.95)
    parser.add_argument("--stage_epochs", type=int, default=1)
    parser.add_argument("--stages", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--probe_steps", type=int, default=10)
    parser.add_argument("--probe_lr", type=float, default=0.1)
    parser.add_argument("--probe_lr_scale", type=float, default=1.0)
    parser.add_argument("--probe_batch_size", type=int, default=32)
    parser.add_argument("--B", type=float, default=2.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="experiments/outputs")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    output_path = os.path.join(args.output_dir, f"a_axis_{args.policy}.csv")
    device = get_device(args.device)

    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames())
        writer.writeheader()
        for seed in range(args.seeds):
            set_all_seeds(seed)
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
            )
            model = TeachabilityMLP(args.input_dim, args.width, depth=args.depth, dropout=args.dropout).to(device)
            for parameter in model.correction_head.parameters():
                parameter.requires_grad = False
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            batches_per_epoch = math.ceil(args.m / args.batch_size)
            mass_per_stage = args.lr * batches_per_epoch * args.stage_epochs
            step_mass = 0.0

            initial_metrics = evaluate_teachability_model(model, bundle, device=device, batch_size=args.eval_batch_size)
            initial_probe = teachability_probe(
                model=model,
                correction_split=bundle.correction,
                correction_eval_split=bundle.correction_eval,
                device=device,
                steps=args.probe_steps,
                lr=args.probe_lr,
                lr_scale=args.probe_lr_scale,
                batch_size=args.probe_batch_size,
            )
            writer.writerow(make_row(seed=seed, args=args, stage=0, step_mass=step_mass, metrics=initial_metrics, probe=initial_probe))
            handle.flush()

            for stage in range(1, args.stages + 1):
                projected = step_mass + mass_per_stage
                if args.policy == "cap" and projected > args.B:
                    break
                train_stage(
                    model=model,
                    train_split=bundle.train,
                    device=device,
                    optimizer=optimizer,
                    epochs=args.stage_epochs,
                    batch_size=args.batch_size,
                )
                step_mass = projected
                metrics = evaluate_teachability_model(model, bundle, device=device, batch_size=args.eval_batch_size)
                probe = teachability_probe(
                    model=model,
                    correction_split=bundle.correction,
                    correction_eval_split=bundle.correction_eval,
                    device=device,
                    steps=args.probe_steps,
                    lr=args.probe_lr,
                    lr_scale=args.probe_lr_scale,
                    batch_size=args.probe_batch_size,
                )
                writer.writerow(make_row(seed=seed, args=args, stage=stage, step_mass=step_mass, metrics=metrics, probe=probe))
                handle.flush()

            if (seed + 1) % 4 == 0:
                print(f"Completed seed {seed + 1}/{args.seeds}")

    print(f"Wrote A-axis results to {output_path}")


if __name__ == "__main__":
    main()

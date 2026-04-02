import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from experiments.utils import ensure_dir
except ImportError:
    from utils import ensure_dir


sns.set_theme(style="whitegrid", context="talk")


POLICY_ORDER_H = ["twogate", "dest_val_nocap", "dest_val", "dest_train"]
POLICY_ORDER_A = ["cap", "no_cap"]
COLORS = {
    "twogate": "#1b9e77",
    "dest_val_nocap": "#d95f02",
    "dest_val": "#7570b3",
    "dest_train": "#e7298a",
    "cap": "#1b9e77",
    "no_cap": "#d95f02",
}



def load_axis_files(input_dir: str, prefix: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.startswith(prefix) and file_name.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(input_dir, file_name)))
    if not frames:
        raise FileNotFoundError(f"No files matching {prefix}*.csv found in {input_dir}")
    return pd.concat(frames, ignore_index=True)



def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_h_axis(input_dir: str, output_dir: str) -> None:
    df = load_axis_files(input_dir, prefix="h_axis_")
    df["policy"] = pd.Categorical(df["policy"], categories=POLICY_ORDER_H, ordered=True)
    df = df.sort_values(["policy", "seed", "proposal_step"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.lineplot(
        data=df,
        x="proposal_step",
        y="active_test_shift_acc",
        hue="policy",
        palette=COLORS,
        ax=axes[0],
        errorbar=("se", 1.0),
    )
    axes[0].set_title("Committed OOD accuracy vs proposal step")
    axes[0].set_xlabel("Proposal step")
    axes[0].set_ylabel("Committed OOD accuracy")

    sns.lineplot(
        data=df,
        x="proposal_step",
        y="active_correction_gain",
        hue="policy",
        palette=COLORS,
        ax=axes[1],
        errorbar=("se", 1.0),
        legend=False,
    )
    axes[1].set_title("Committed correction gain vs proposal step")
    axes[1].set_xlabel("Proposal step")
    axes[1].set_ylabel("Correction gain after fixed audit")
    savefig(os.path.join(output_dir, "h_axis_neural_trajectories.png"))

    final_rows = df.sort_values(["seed", "policy", "proposal_step"]).groupby(["seed", "policy"], as_index=False).tail(1)
    summary = final_rows[["policy", "active_width", "active_test_shift_acc", "active_correction_gain"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=summary, x="policy", y="active_test_shift_acc", order=POLICY_ORDER_H, palette=COLORS, ax=axes[0], errorbar=("se", 1.0))
    axes[0].set_title("Final committed OOD accuracy")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=summary, x="policy", y="active_correction_gain", order=POLICY_ORDER_H, palette=COLORS, ax=axes[1], errorbar=("se", 1.0))
    axes[1].set_title("Final committed correction gain")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)
    savefig(os.path.join(output_dir, "h_axis_neural_summary.png"))

    table = final_rows.groupby("policy").agg(
        final_active_width_mean=("active_width", "mean"),
        final_active_width_std=("active_width", "std"),
        final_shift_acc_mean=("active_test_shift_acc", "mean"),
        final_shift_acc_std=("active_test_shift_acc", "std"),
        final_correction_gain_mean=("active_correction_gain", "mean"),
        final_correction_gain_std=("active_correction_gain", "std"),
    )
    table.to_csv(os.path.join(output_dir, "h_axis_neural_summary.csv"))



def plot_a_axis(input_dir: str, output_dir: str) -> None:
    df = load_axis_files(input_dir, prefix="a_axis_")
    df["policy"] = pd.Categorical(df["policy"], categories=POLICY_ORDER_A, ordered=True)
    df = df.sort_values(["policy", "seed", "step_mass", "stage"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.lineplot(
        data=df,
        x="step_mass",
        y="test_shift_acc",
        hue="policy",
        palette=COLORS,
        ax=axes[0],
        errorbar=("se", 1.0),
    )
    axes[0].set_title("OOD accuracy vs step-mass")
    axes[0].set_xlabel("Cumulative step-mass")
    axes[0].set_ylabel("OOD accuracy")

    sns.lineplot(
        data=df,
        x="step_mass",
        y="probe_gain",
        hue="policy",
        palette=COLORS,
        ax=axes[1],
        errorbar=("se", 1.0),
        legend=False,
    )
    axes[1].set_title("Teachability probe gain vs step-mass")
    axes[1].set_xlabel("Cumulative step-mass")
    axes[1].set_ylabel("Probe gain")
    savefig(os.path.join(output_dir, "a_axis_step_mass_curves.png"))

    final_rows = df.sort_values(["seed", "policy", "step_mass", "stage"]).groupby(["seed", "policy"], as_index=False).tail(1)
    summary = final_rows[["policy", "test_shift_acc", "probe_gain", "step_mass"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=summary, x="policy", y="test_shift_acc", order=POLICY_ORDER_A, palette=COLORS, ax=axes[0], errorbar=("se", 1.0))
    axes[0].set_title("Final OOD accuracy")
    axes[0].set_xlabel("")

    sns.barplot(data=summary, x="policy", y="probe_gain", order=POLICY_ORDER_A, palette=COLORS, ax=axes[1], errorbar=("se", 1.0))
    axes[1].set_title("Final teachability probe gain")
    axes[1].set_xlabel("")
    savefig(os.path.join(output_dir, "a_axis_summary.png"))

    table = final_rows.groupby("policy").agg(
        final_step_mass_mean=("step_mass", "mean"),
        final_step_mass_std=("step_mass", "std"),
        final_shift_acc_mean=("test_shift_acc", "mean"),
        final_shift_acc_std=("test_shift_acc", "std"),
        final_probe_gain_mean=("probe_gain", "mean"),
        final_probe_gain_std=("probe_gain", "std"),
    )
    table.to_csv(os.path.join(output_dir, "a_axis_summary.csv"))



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=["h", "a"], required=True)
    parser.add_argument("--input_dir", type=str, default="experiments/outputs")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    ensure_dir(output_dir)
    if args.axis == "h":
        plot_h_axis(args.input_dir, output_dir)
    else:
        plot_a_axis(args.input_dir, output_dir)


if __name__ == "__main__":
    main()

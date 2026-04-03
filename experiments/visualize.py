import argparse
import os
from typing import Dict, List, Tuple

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



def add_mean_std_display(table: pd.DataFrame, metric_roots: List[str]) -> pd.DataFrame:
    formatted = table.copy()
    for metric_root in metric_roots:
        mean_col = f"{metric_root}_mean"
        std_col = f"{metric_root}_std"
        if mean_col in formatted.columns and std_col in formatted.columns:
            formatted[std_col] = formatted[std_col].fillna(0.0)
            formatted[f"{metric_root}_mean_pm_std"] = formatted.apply(
                lambda row: f"{row[mean_col]:.3f} ± {row[std_col]:.3f}",
                axis=1,
            )
    return formatted



def make_paired_wins(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    metrics: List[Tuple[str, bool]],
) -> pd.DataFrame:
    merged = left.merge(right, on="seed", suffixes=("_left", "_right"))
    rows: List[Dict[str, float]] = []
    for metric_name, higher_is_better in metrics:
        left_values = merged[f"{metric_name}_left"]
        right_values = merged[f"{metric_name}_right"]
        if higher_is_better:
            left_wins = (left_values > right_values).sum()
            right_wins = (right_values > left_values).sum()
        else:
            left_wins = (left_values < right_values).sum()
            right_wins = (right_values < left_values).sum()
        ties = len(merged) - left_wins - right_wins
        rows.append(
            {
                "left_policy": left_name,
                "right_policy": right_name,
                "metric": metric_name,
                "paired_seed_count": len(merged),
                "left_mean": left_values.mean(),
                "right_mean": right_values.mean(),
                "left_minus_right_mean": (left_values - right_values).mean(),
                "left_wins": int(left_wins),
                "right_wins": int(right_wins),
                "ties": int(ties),
            }
        )
    return pd.DataFrame(rows)



def final_rows_by_seed(df: pd.DataFrame, sort_cols: List[str]) -> pd.DataFrame:
    return df.sort_values(sort_cols).groupby(["seed", "policy"], as_index=False).tail(1).copy()



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
    accepted_counts = df.groupby(["seed", "policy"], as_index=False)["accepted"].sum().rename(columns={"accepted": "accepted_count"})
    accepted_counts["accepted_count"] = accepted_counts["accepted_count"] - 1

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

    final_rows = final_rows_by_seed(df, ["seed", "policy", "proposal_step"]).merge(accepted_counts, on=["seed", "policy"], how="left")
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
        final_accepted_count_mean=("accepted_count", "mean"),
        final_accepted_count_std=("accepted_count", "std"),
        final_iid_acc_mean=("active_test_iid_acc", "mean"),
        final_iid_acc_std=("active_test_iid_acc", "std"),
        final_shift_acc_mean=("active_test_shift_acc", "mean"),
        final_shift_acc_std=("active_test_shift_acc", "std"),
        final_correction_train_gain_mean=("active_correction_train_gain", "mean"),
        final_correction_train_gain_std=("active_correction_train_gain", "std"),
        final_correction_eval_gain_mean=("active_correction_eval_gain", "mean"),
        final_correction_eval_gain_std=("active_correction_eval_gain", "std"),
        final_correction_param_shift_l2_mean=("active_correction_param_shift_l2", "mean"),
        final_correction_param_shift_l2_std=("active_correction_param_shift_l2", "std"),
    ).reset_index()
    table = add_mean_std_display(
        table,
        [
            "final_active_width",
            "final_accepted_count",
            "final_iid_acc",
            "final_shift_acc",
            "final_correction_train_gain",
            "final_correction_eval_gain",
            "final_correction_param_shift_l2",
        ],
    )
    table.to_csv(os.path.join(output_dir, "h_axis_neural_summary.csv"), index=False)
    table.to_csv(os.path.join(output_dir, "h_axis_rebuttal_table.csv"), index=False)

    paired_rows: List[pd.DataFrame] = []
    twogate_rows = final_rows[final_rows["policy"] == "twogate"][[
        "seed",
        "active_test_iid_acc",
        "active_test_shift_acc",
        "active_correction_train_gain",
        "active_correction_eval_gain",
    ]]
    for policy in [policy for policy in POLICY_ORDER_H if policy != "twogate"]:
        compare_rows = final_rows[final_rows["policy"] == policy][[
            "seed",
            "active_test_iid_acc",
            "active_test_shift_acc",
            "active_correction_train_gain",
            "active_correction_eval_gain",
        ]]
        paired_rows.append(
            make_paired_wins(
                left=twogate_rows,
                right=compare_rows,
                left_name="twogate",
                right_name=policy,
                metrics=[
                    ("active_test_iid_acc", True),
                    ("active_test_shift_acc", True),
                    ("active_correction_train_gain", True),
                    ("active_correction_eval_gain", True),
                ],
            )
        )
    pd.concat(paired_rows, ignore_index=True).to_csv(os.path.join(output_dir, "h_axis_paired_wins.csv"), index=False)



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

    final_rows = final_rows_by_seed(df, ["seed", "policy", "step_mass", "stage"])
    budget_rows = df[df["step_mass"] <= df["step_budget"] + 1e-12].sort_values(["seed", "policy", "step_mass", "stage"]).groupby(["seed", "policy"], as_index=False).tail(1)
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
        final_iid_acc_mean=("test_iid_acc", "mean"),
        final_iid_acc_std=("test_iid_acc", "std"),
        final_shift_acc_mean=("test_shift_acc", "mean"),
        final_shift_acc_std=("test_shift_acc", "std"),
        final_probe_train_gain_mean=("probe_train_gain", "mean"),
        final_probe_train_gain_std=("probe_train_gain", "std"),
        final_probe_eval_gain_mean=("probe_eval_gain", "mean"),
        final_probe_eval_gain_std=("probe_eval_gain", "std"),
        final_probe_param_shift_l2_mean=("probe_param_shift_l2", "mean"),
        final_probe_param_shift_l2_std=("probe_param_shift_l2", "std"),
        final_gen_gap_mean=("gen_gap", "mean"),
        final_gen_gap_std=("gen_gap", "std"),
    ).reset_index()
    table = add_mean_std_display(
        table,
        [
            "final_step_mass",
            "final_iid_acc",
            "final_shift_acc",
            "final_probe_train_gain",
            "final_probe_eval_gain",
            "final_probe_param_shift_l2",
            "final_gen_gap",
        ],
    )
    table.to_csv(os.path.join(output_dir, "a_axis_summary.csv"), index=False)

    cap_final = final_rows[final_rows["policy"] == "cap"][[
        "seed",
        "step_mass",
        "test_iid_acc",
        "test_shift_acc",
        "probe_train_gain",
        "probe_eval_gain",
        "probe_param_shift_l2",
        "gen_gap",
    ]]
    no_cap_at_budget = budget_rows[budget_rows["policy"] == "no_cap"][[
        "seed",
        "step_mass",
        "test_iid_acc",
        "test_shift_acc",
        "probe_train_gain",
        "probe_eval_gain",
        "probe_param_shift_l2",
        "gen_gap",
    ]]
    no_cap_final = final_rows[final_rows["policy"] == "no_cap"][[
        "seed",
        "step_mass",
        "test_iid_acc",
        "test_shift_acc",
        "probe_train_gain",
        "probe_eval_gain",
        "probe_param_shift_l2",
        "gen_gap",
    ]]

    rebuttal_rows = [
        cap_final.assign(summary_row="cap_at_budget"),
        no_cap_at_budget.assign(summary_row="no_cap_at_budget"),
        no_cap_final.assign(summary_row="no_cap_final"),
    ]
    rebuttal = pd.concat(rebuttal_rows, ignore_index=True)
    rebuttal_table = rebuttal.groupby("summary_row").agg(
        step_mass_mean=("step_mass", "mean"),
        step_mass_std=("step_mass", "std"),
        iid_acc_mean=("test_iid_acc", "mean"),
        iid_acc_std=("test_iid_acc", "std"),
        shift_acc_mean=("test_shift_acc", "mean"),
        shift_acc_std=("test_shift_acc", "std"),
        probe_train_gain_mean=("probe_train_gain", "mean"),
        probe_train_gain_std=("probe_train_gain", "std"),
        probe_eval_gain_mean=("probe_eval_gain", "mean"),
        probe_eval_gain_std=("probe_eval_gain", "std"),
        probe_param_shift_l2_mean=("probe_param_shift_l2", "mean"),
        probe_param_shift_l2_std=("probe_param_shift_l2", "std"),
        gen_gap_mean=("gen_gap", "mean"),
        gen_gap_std=("gen_gap", "std"),
    ).reset_index()
    rebuttal_table = add_mean_std_display(
        rebuttal_table,
        [
            "step_mass",
            "iid_acc",
            "shift_acc",
            "probe_train_gain",
            "probe_eval_gain",
            "probe_param_shift_l2",
            "gen_gap",
        ],
    )
    rebuttal_table.to_csv(os.path.join(output_dir, "a_axis_rebuttal_table.csv"), index=False)

    no_cap_budget_vs_final = no_cap_final.merge(no_cap_at_budget, on="seed", suffixes=("_final", "_at_budget"))
    excess_table = pd.DataFrame(
        [
            {
                "policy": "no_cap",
                "paired_seed_count": len(no_cap_budget_vs_final),
                "step_mass_at_budget_mean": no_cap_budget_vs_final["step_mass_at_budget"].mean(),
                "step_mass_final_mean": no_cap_budget_vs_final["step_mass_final"].mean(),
                "gen_gap_at_budget_mean": no_cap_budget_vs_final["gen_gap_at_budget"].mean(),
                "gen_gap_final_mean": no_cap_budget_vs_final["gen_gap_final"].mean(),
                "gen_gap_excess_mean": (no_cap_budget_vs_final["gen_gap_final"] - no_cap_budget_vs_final["gen_gap_at_budget"]).mean(),
                "shift_acc_at_budget_mean": no_cap_budget_vs_final["test_shift_acc_at_budget"].mean(),
                "shift_acc_final_mean": no_cap_budget_vs_final["test_shift_acc_final"].mean(),
                "shift_acc_excess_mean": (no_cap_budget_vs_final["test_shift_acc_final"] - no_cap_budget_vs_final["test_shift_acc_at_budget"]).mean(),
                "probe_eval_gain_at_budget_mean": no_cap_budget_vs_final["probe_eval_gain_at_budget"].mean(),
                "probe_eval_gain_final_mean": no_cap_budget_vs_final["probe_eval_gain_final"].mean(),
                "probe_eval_gain_excess_mean": (no_cap_budget_vs_final["probe_eval_gain_final"] - no_cap_budget_vs_final["probe_eval_gain_at_budget"]).mean(),
            }
        ]
    )
    excess_table.to_csv(os.path.join(output_dir, "a_axis_budget_excess.csv"), index=False)

    paired_budget = make_paired_wins(
        left=cap_final,
        right=no_cap_at_budget,
        left_name="cap",
        right_name="no_cap_at_budget",
        metrics=[
            ("test_iid_acc", True),
            ("test_shift_acc", True),
            ("probe_train_gain", True),
            ("probe_eval_gain", True),
            ("gen_gap", False),
        ],
    )
    paired_budget.to_csv(os.path.join(output_dir, "a_axis_paired_wins.csv"), index=False)



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

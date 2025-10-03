import argparse
import glob
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects


def plot_h_axis(input_dir: str):
    """Plot H-axis comparison of two-gate vs. destructive policies."""
    csv_files = glob.glob(os.path.join(input_dir, "h_axis_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No H-axis CSV files found in '{input_dir}'")

    df_list = []
    for f in csv_files:
        policy_name = os.path.basename(f).replace("h_axis_", "").replace(".csv", "")
        temp_df = pd.read_csv(f)
        temp_df['policy'] = policy_name
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Filter to only include twogate and dest_train
    df = df[df['policy'].isin(['twogate', 'dest_train'])]
    
    # Stop lines after last acceptance (no forward-fill)
    def keep_to_last_accept(group):
        acc = group['accepted'].to_numpy()
        if acc.any():
            last = np.where(acc == 1)[0].max()
            g = group.copy()
            g.loc[g.index > g.index[last], 'test_loss'] = np.nan
            return g
        return group
    
    df = df.groupby(['seed', 'policy'], group_keys=False).apply(keep_to_last_accept)
    df['current_test_loss'] = df['test_loss']

    # Use dest_val color for twogate, and dest_train color for dest_train
    colors = {'twogate': '#4facfe', 'dest_train': '#fa709a'}
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor='white')
    ax.set_facecolor('#fafbfc')
    
    # Add subtle grid
    ax.grid(True, alpha=0.15, linewidth=1.2, linestyle='-', color='#cbd5e0')
    ax.set_axisbelow(True)
    
    # Plot twogate policy first with solid line
    sns.lineplot(
        data=df[df['policy'] == 'twogate'],
        x='degree',
        y='current_test_loss',
        color=colors['twogate'],
        label='TwoGate',
        errorbar=None,  # Remove error bars
        ax=ax,
        linewidth=3.5,
        marker='o',
        markersize=10,
        alpha=0.8,
        zorder=10
    )
    
    # Plot dest_train policy
    sns.lineplot(
        data=df[df['policy'] == 'dest_train'],
        x='degree',
        y='current_test_loss',
        color=colors['dest_train'],
        label='Destructive',
        errorbar=None,  # Remove error bars
        ax=ax,
        linewidth=3.0,
        linestyle='-.',
        marker='^',
        markersize=8,
        alpha=0.8,
        zorder=5
    )
    
    # Enhanced title with custom styling
    ax.set_title(
        r'$\mathcal{M}_H$: Two-Gate vs. Destructive Policies',
        fontsize=20,
        fontweight='bold',
        pad=20,
        color='#2d3748'
    )
    
    ax.set_xlabel('Polynomial Degree (Complexity)', fontsize=14, fontweight='600', color='#4a5568', labelpad=12)
    ax.set_ylabel('Test 0-1 Loss', fontsize=14, fontweight='600', color='#4a5568', labelpad=12)
    
    # Enhance legend
    legend = ax.legend(
        title='Policy',
        loc='upper left',
        frameon=True,
        shadow=True,
        fancybox=True,
        fontsize=11,
        title_fontsize=12
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('#e2e8f0')
    legend.get_frame().set_linewidth(2)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#e2e8f0')
        spine.set_linewidth(2)
    
    # Enhance tick labels
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#4a5568', length=6, width=2)
    
    plt.tight_layout()
    plt.savefig("h_axis_comparison.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Saved H-axis plot to h_axis_comparison.png")
    plt.show()


def plot_a_axis(input_dir: str):
    """Plot A-axis generalization gap vs. step-mass."""
    csv_files = glob.glob(os.path.join(input_dir, "a_axis_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No A-axis CSV files found in '{input_dir}'")

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    # Map policy names to more readable format
    df['policy'] = df['policy'].map({'cap': 'TwoGate', 'no_cap': 'Destructive'}).fillna(df['policy'])
    
    # Vibrant gradient colors
    colors = ['#667eea', '#f093fb']
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor='white')
    ax.set_facecolor('#fafbfc')
    
    # Add subtle grid
    ax.grid(True, alpha=0.15, linewidth=1.2, linestyle='-', color='#cbd5e0')
    ax.set_axisbelow(True)
    
    sns.lineplot(
        data=df,
        x='step_mass',
        y='gen_gap',
        hue='policy',
        errorbar=('ci', 95),
        ax=ax,
        linewidth=3.5,
        palette=colors,
        alpha=0.9
    )
    
    # Enhanced title with custom styling
    ax.set_title(
        r'$\mathcal{M}_A$: TwoGate vs. Destructive Generalization',
        fontsize=20,
        fontweight='bold',
        pad=20,
        color='#2d3748'
    )
    
    ax.set_xlabel('Cumulative Step-Mass ($M_T$)', fontsize=14, fontweight='600', color='#4a5568', labelpad=12)
    ax.set_ylabel('Generalization Gap (Test Loss - Train Loss)', fontsize=13, fontweight='600', color='#4a5568', labelpad=12)
    
    # Enhance legend
    legend = ax.legend(
        title='Policy',
        loc='upper left',
        frameon=True,
        shadow=True,
        fancybox=True,
        fontsize=11,
        title_fontsize=12
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('#e2e8f0')
    legend.get_frame().set_linewidth(2)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#e2e8f0')
        spine.set_linewidth(2)
    
    # Enhance tick labels
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#4a5568', length=6, width=2)
    
    plt.tight_layout()
    plt.savefig("a_axis_comparison.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Saved A-axis plot to a_axis_comparison.png")
    plt.show()


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--axis",
        type=str,
        choices=['h', 'a'],
        required=True,
        help="Which axis to plot: 'h' for H-axis or 'a' for A-axis"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="experiments/outputs",
        help="Directory containing CSV output files"
    )
    args = parser.parse_args()

    if args.axis == 'h':
        plot_h_axis(args.input_dir)
    elif args.axis == 'a':
        plot_a_axis(args.input_dir)


if __name__ == "__main__":
    main()

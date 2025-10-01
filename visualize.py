import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse

# Premium styling with publication-quality aesthetics
sns.set_style("whitegrid", {
    'grid.linestyle': '-',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.2
})

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SF Pro Display', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial'],
    'font.size': 12,
    'figure.figsize': (12, 7),
    'figure.facecolor': '#FAFBFC',
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#FAFBFC',
    'axes.facecolor': 'white',
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'axes.labelweight': 600,
    'axes.titleweight': 700,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#D1D5DB',
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.alpha': 0.18,
    'grid.linewidth': 0.6,
    'grid.color': '#E5E7EB',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.color': '#6B7280',
    'ytick.color': '#6B7280',
    'legend.fontsize': 11,
    'legend.framealpha': 0.97,
    'legend.edgecolor': '#D1D5DB',
    'legend.fancybox': True,
    'legend.shadow': False,
    'lines.linewidth': 3,
    'lines.markersize': 9,
    'lines.solid_capstyle': 'round'
})

# Sophisticated color palette with depth
PALETTE = {
    "Twogate": "#2563EB",      # Vibrant blue with depth
    "Destructive": "#EF4444",  # Bold red
    "Cap": "#2563EB", 
    "No_cap": "#EF4444"
}

def plot_h_axis():
    """Plot representational self-modification (H-axis)"""
    
    def load_and_process(policy: str):
        df = pd.read_csv(f"experiments/outputs/h_axis_{policy}.csv")
        last = df.groupby('seed', as_index=False).last()
        last['policy'] = policy.capitalize()
        
        g = df.groupby('degree')
        stats_df = pd.DataFrame({
            'degree': g.size().index,
            'mean': g['test_loss'].mean().values,
            'sem': (g['test_loss'].std(ddof=1) / np.sqrt(g.size())).values,
            'n': g.size().values
        })
        stats_df['policy'] = policy.capitalize()
        return stats_df, last

    tg_stats, tg_last = load_and_process("twogate")
    ds_stats, ds_last = load_and_process("destructive")

    # Statistical comparison
    tg_final = tg_last['test_loss'].values
    ds_final = ds_last['test_loss'].values
    _, p_val = stats.ttest_ind(tg_final, ds_final, equal_var=False)
    mean_diff = ds_final.mean() - tg_final.mean()
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

    # Create figure with premium styling
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#FAFBFC')

    for stats_df, last_df, policy, color in [
        (tg_stats, tg_last, 'Two-Gate', PALETTE['Twogate']),
        (ds_stats, ds_last, 'Destructive', PALETTE['Destructive'])
    ]:
        # Gradient-like uncertainty band (double layer for depth)
        ax.fill_between(stats_df['degree'], 
                         stats_df['mean'] - stats_df['sem'],
                         stats_df['mean'] + stats_df['sem'],
                         alpha=0.12, color=color, linewidth=0, zorder=2)
        ax.fill_between(stats_df['degree'], 
                         stats_df['mean'] - 0.5*stats_df['sem'],
                         stats_df['mean'] + 0.5*stats_df['sem'],
                         alpha=0.15, color=color, linewidth=0, zorder=2)
        
        # Main trajectory line with premium styling
        ax.plot(stats_df['degree'], stats_df['mean'], 'o-', 
                color=color, lw=3.5, markersize=10, 
                label=f"{policy} (final: {last_df['test_loss'].mean():.3f})",
                markeredgecolor='white', markeredgewidth=2.5, zorder=6,
                alpha=1.0, solid_capstyle='round')
        
        # Individual seed outcomes with enhanced star markers
        ax.scatter(last_df['degree'], last_df['test_loss'],
                   s=180, marker='★', color=color, 
                   edgecolor='white', linewidth=1.5, alpha=0.5, zorder=4)
        
        # Elegant median stopping degree line
        median_deg = last_df['degree'].median()
        ax.axvline(median_deg, color=color, ls=':', 
                   lw=2.8, alpha=0.25, zorder=1)

    # Enhanced statistical annotation box
    ax.text(0.98, 0.03, f'Δ = {mean_diff:+.3f} ({sig})', 
            transform=ax.transAxes, fontsize=11.5, family='monospace',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.7', 
                      facecolor='#F8F9FA', 
                      edgecolor='#DEE2E6', 
                      linewidth=1.5,
                      alpha=0.98))

    # Enhanced labels and title
    ax.set_xlabel('Polynomial Degree', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Test Loss', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('$\mathcal{M}_H$: Capacity-Seeking vs. Bounded Growth', 
                 fontsize=15, fontweight='bold', pad=20)

    # Enhanced legend
    legend = ax.legend(loc='upper left', framealpha=0.98, 
                      edgecolor='#DEE2E6', fancybox=True,
                      shadow=False, fontsize=11)
    legend.get_frame().set_linewidth(1.5)

    # Refined grid
    ax.grid(True, alpha=0.25, linewidth=0.8, linestyle='-', color='#CED4DA')
    ax.set_axisbelow(True)

    # Set explicit limits with better spacing
    ax.set_xlim(-0.4, ax.get_xlim()[1] + 0.4)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.set_ylim(ax.get_ylim()[0] - 0.03*y_range, 
                ax.get_ylim()[1] + 0.03*y_range)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    # Save
    out_dir = Path('figures')
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / 'h_axis_comparison.png', 
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()

    print(f"Figure saved: {out_dir / 'h_axis_comparison.png'}")
    print(f"\nResults:")
    print(f"  Two-Gate:    {tg_final.mean():.4f} ± {tg_final.std():.4f}")
    print(f"  Destructive: {ds_final.mean():.4f} ± {ds_final.std():.4f}")
    print(f"  Difference:  {mean_diff:+.4f} (p={p_val:.4f})")


def plot_a_axis():
    """Plot algorithmic self-modification (A-axis): generalization gap vs step-mass"""
    
    cap_df = pd.read_csv("experiments/outputs/a_axis_cap.csv")
    nocap_df = pd.read_csv("experiments/outputs/a_axis_no_cap.csv")
    
    # Filter to iterations where we have data (every 10th)
    cap_df = cap_df[cap_df['iter'] > 0]
    nocap_df = nocap_df[nocap_df['iter'] > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('white')
    
    # --- LEFT PANEL: Generalization gap vs step-mass ---
    ax1 = axes[0]
    
    # Define a common step-mass range for both policies
    max_step_mass = max(cap_df['step_mass'].max(), nocap_df['step_mass'].max())
    min_step_mass = min(cap_df['step_mass'].min(), nocap_df['step_mass'].min())
    
    # Create bins spanning the full range
    bins = np.linspace(min_step_mass, max_step_mass, 30)
    
    for df, policy, color in [
        (cap_df, 'Cap', PALETTE['Cap']),
        (nocap_df, 'No-Cap', PALETTE['No_cap'])
    ]:
        # Bin using the common bins
        df['step_mass_bin'] = pd.cut(df['step_mass'], bins=bins)
        agg = df.groupby('step_mass_bin').agg({
            'gen_gap': ['mean', 'sem', 'count'],
            'step_mass': 'mean'
        }).reset_index(drop=True)
        
        step_mass = agg[('step_mass', 'mean')].values
        gap_mean = agg[('gen_gap', 'mean')].values
        gap_sem = agg[('gen_gap', 'sem')].values
        
        # Remove NaN rows
        valid = ~np.isnan(step_mass)
        step_mass = step_mass[valid]
        gap_mean = gap_mean[valid]
        gap_sem = gap_sem[valid]
        
        ax1.plot(step_mass, gap_mean, 'o-', 
                color=color, lw=3, markersize=8,
                label=f'{policy}', markeredgecolor='white', 
                markeredgewidth=1.5, zorder=5, alpha=0.95)
        
        ax1.fill_between(step_mass, 
                         gap_mean - gap_sem,
                         gap_mean + gap_sem,
                         alpha=0.15, color=color)
    
    # Explicitly set x-axis limits to show full range
    ax1.set_xlim(0, max_step_mass * 1.05)
    
    ax1.set_xlabel('Step-Mass $M_T = \sum_t \eta_t$', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Generalization Gap (test - train)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title('Gap scales with $M_T$ (Theorem 7)', fontsize=14, fontweight='bold', pad=15)
    legend1 = ax1.legend(loc='upper left', framealpha=0.98, edgecolor='#DEE2E6', fontsize=11)
    legend1.get_frame().set_linewidth(1.5)
    ax1.grid(True, alpha=0.25, linewidth=0.8, color='#CED4DA')
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    
    # --- RIGHT PANEL: Test loss vs iteration ---
    ax2 = axes[1]
    
    for df, policy, color in [
        (cap_df, 'Cap', PALETTE['Cap']),
        (nocap_df, 'No-Cap', PALETTE['No_cap'])
    ]:
        # Average over seeds at each iteration
        agg = df.groupby('iter').agg({
            'test_loss': ['mean', 'sem'],
            'step_mass': 'mean'
        }).reset_index()
        
        iters = agg['iter'].values
        test_mean = agg[('test_loss', 'mean')].values
        test_sem = agg[('test_loss', 'sem')].values
        final_mass = agg[('step_mass', 'mean')].values[-1]
        
        ax2.plot(iters, test_mean, '-', 
                color=color, lw=3,
                label=f'{policy} ($M_T$ = {final_mass:.1f})', zorder=5, alpha=0.95)
        
        ax2.fill_between(iters, 
                         test_mean - test_sem,
                         test_mean + test_sem,
                         alpha=0.15, color=color)
    
    ax2.set_xlabel('Training Iteration', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Test Loss', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_title('Convergence: Both policies reach similar loss', 
                  fontsize=14, fontweight='bold', pad=15)
    legend2 = ax2.legend(loc='upper right', framealpha=0.98, edgecolor='#DEE2E6', fontsize=11)
    legend2.get_frame().set_linewidth(1.5)
    ax2.grid(True, alpha=0.25, linewidth=0.8, color='#CED4DA')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout(pad=2.0)
    
    # Save
    out_dir = Path('figures')
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / 'a_axis_comparison.png', 
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Final statistics
    cap_final = cap_df.groupby('seed').last()
    nocap_final = nocap_df.groupby('seed').last()
    
    print(f"Figure saved: {out_dir / 'a_axis_comparison.png'}")
    print(f"\nResults:")
    print(f"  Cap policy:")
    print(f"    Final step-mass: {cap_final['step_mass'].mean():.2f} ± {cap_final['step_mass'].std():.2f}")
    print(f"    Final gap:       {cap_final['gen_gap'].mean():.4f} ± {cap_final['gen_gap'].std():.4f}")
    print(f"    Final test loss: {cap_final['test_loss'].mean():.4f} ± {cap_final['test_loss'].std():.4f}")
    print(f"  No-cap policy:")
    print(f"    Final step-mass: {nocap_final['step_mass'].mean():.2f} ± {nocap_final['step_mass'].std():.2f}")
    print(f"    Final gap:       {nocap_final['gen_gap'].mean():.4f} ± {nocap_final['gen_gap'].std():.4f}")
    print(f"    Final test loss: {nocap_final['test_loss'].mean():.4f} ± {nocap_final['test_loss'].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize self-modification experiments')
    parser.add_argument('--axis', choices=['h', 'a'], default='h',
                       help='Which axis to plot: h (representational) or a (algorithmic)')
    args = parser.parse_args()
    
    if args.axis == 'h':
        plot_h_axis()
    elif args.axis == 'a':
        plot_a_axis()


if __name__ == "__main__":
    main()
    
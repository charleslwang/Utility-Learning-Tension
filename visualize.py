import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy import stats

# Styling
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'figure.figsize': (12, 7),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

PALETTE = {"Twogate": "#2E86AB", "Destructive": "#E63946"}

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

# Create figure
fig, ax = plt.subplots()

for stats_df, last_df, policy, color in [
    (tg_stats, tg_last, 'Two-Gate', PALETTE['Twogate']),
    (ds_stats, ds_last, 'Destructive', PALETTE['Destructive'])
]:
    # Main trajectory line
    ax.plot(stats_df['degree'], stats_df['mean'], 'o-', 
            color=color, lw=2.8, markersize=8, 
            label=f"{policy} (final: {last_df['test_loss'].mean():.3f})",
            markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    
    # Uncertainty band
    ax.fill_between(stats_df['degree'], 
                     stats_df['mean'] - stats_df['sem'],
                     stats_df['mean'] + stats_df['sem'],
                     alpha=0.18, color=color, linewidth=0)
    
    # Individual seed outcomes
    ax.scatter(last_df['degree'], last_df['test_loss'],
               s=100, marker='*', color=color, 
               edgecolor='black', linewidth=0.6, alpha=0.5, zorder=3)
    
    # Median stopping degree
    median_deg = last_df['degree'].median()
    ax.axvline(median_deg, color=color, ls='--', 
               lw=2, alpha=0.35, zorder=1)

# Statistical annotation box
ax.text(0.98, 0.02, f'Δ = {mean_diff:+.3f} ({sig})', 
        transform=ax.transAxes, fontsize=12,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.6', 
                  facecolor='white', 
                  edgecolor='gray', 
                  linewidth=1.5,
                  alpha=0.95))

# Axis labels and title
ax.set_xlabel('Polynomial Degree', fontweight='semibold')
ax.set_ylabel('Test Loss', fontweight='semibold')
ax.set_title('$M_H$: Capacity-Seeking vs. Bounded Growth', 
             fontweight='bold', pad=18)

# Legend
ax.legend(loc='upper left', framealpha=0.98, 
          edgecolor='gray', fancybox=True)

# Grid
ax.grid(True, alpha=0.3, linewidth=0.8, linestyle='-')
ax.set_axisbelow(True)

# Set explicit limits to prevent clipping
ax.set_xlim(-0.3, ax.get_xlim()[1] + 0.3)
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
ax.set_ylim(ax.get_ylim()[0] - 0.02*y_range, 
            ax.get_ylim()[1] + 0.02*y_range)

# Save
out_dir = Path('figures')
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'policy_comparison_final.png', 
            facecolor='white', edgecolor='none')
plt.close()

print(f"Figure saved: {out_dir / 'policy_comparison_final.png'}")
print(f"\nResults:")
print(f"  Two-Gate:    {tg_final.mean():.4f} ± {tg_final.std():.4f}")
print(f"  Destructive: {ds_final.mean():.4f} ± {ds_final.std():.4f}")
print(f"  Difference:  {mean_diff:+.4f} (p={p_val:.4f})")

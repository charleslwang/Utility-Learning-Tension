import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.figsize': (10, 6),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3
})

def load_and_process(policy):
    """Load data and compute statistics."""
    df = pd.read_csv(f'experiments/outputs/h_axis_{policy}.csv')
    
    # Find where each seed stopped
    last_points = df.groupby('seed').last().reset_index()
    
    # Compute mean and std across seeds for each degree
    stats = df.groupby('degree').agg({
        'test_loss': ['mean', 'std', 'count'],
        'val_loss': 'mean'
    }).reset_index()
    
    # Add policy label
    stats['policy'] = policy.capitalize()
    last_points['policy'] = policy.capitalize()
    
    return stats, last_points

# Load both policies
twogate_stats, twogate_last = load_and_process('twogate')
destructive_stats, destructive_last = load_and_process('destructive')

# Combine data
all_stats = pd.concat([twogate_stats, destructive_stats])
all_last = pd.concat([twogate_last, destructive_last])

# Create plot
plt.figure(figsize=(12, 7))

# Plot mean test loss with confidence intervals
for policy in ['Twogate', 'Destructive']:
    data = all_stats[all_stats['policy'] == policy]
    plt.plot(
        data['degree'], 
        data[('test_loss', 'mean')],
        'o-', 
        label=policy,
        linewidth=2.5
    )
    
    # Add confidence intervals (mean ± std)
    plt.fill_between(
        data['degree'],
        data[('test_loss', 'mean')] - data[('test_loss', 'std')],
        data[('test_loss', 'mean')] + data[('test_loss', 'std')],
        alpha=0.2
    )
    
    # Mark stopping points
    last = all_last[all_last['policy'] == policy]
    plt.scatter(
        last['degree'], 
        last['test_loss'],
        s=100, 
        marker='*',
        edgecolor='black',
        zorder=10
    )

# Add labels and title
plt.xlabel('Polynomial Degree', fontsize=14, fontweight='bold')
plt.ylabel('Test Loss', fontsize=14, fontweight='bold')
plt.title('Model Selection Policy Comparison\n(Mean ± Std. Dev. across seeds)', 
          fontsize=16, pad=20)

# Add legend and grid
plt.legend(title='Policy', title_fontsize=12, fontsize=11)
plt.grid(True, alpha=0.3)

# Add text annotation
plt.annotate(
    'TwoGate stops earlier, preventing overfitting\nwhile maintaining good performance',
    xy=(2, 0.25), 
    xytext=(3, 0.3),
    arrowprops=dict(facecolor='black', shrink=0.05),
    fontsize=11,
    ha='center'
)

# Save high-quality figure
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'policy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved to {output_dir / 'policy_comparison.png'}")

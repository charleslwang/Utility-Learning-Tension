"""Metrics, breakpoints, and plotting utilities (polished)."""
import os
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up logging
logger = logging.getLogger(__name__)

# ===== Breakpoints on aggregated curves =====================================

def compute_breakpoints_on_group(df_group: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute k* (one-SE rule if SEM present) and knee (Kneedle) on aggregated mean curves.
    Expect columns: d, test_error_mean, optionally test_error_sem.
    """
    k_star, k_elbow = None, None
    try:
        dd = df_group.sort_values('d')
        if 'test_error_sem' in dd.columns and len(dd) > 1:
            idx_min = dd['test_error_mean'].idxmin()
            err_min = dd.loc[idx_min, 'test_error_mean']
            thr = err_min + dd.loc[idx_min, 'test_error_sem']
            post = dd[dd['d'] >= dd.loc[idx_min, 'd']]
            over = post[post['test_error_mean'] > thr]
            if len(over):
                k_star = float(over.iloc[0]['d'])
        # knee
        try:
            from kneed import KneeLocator
            kneedle = KneeLocator(
                dd['d'].values, dd['test_error_mean'].values,
                curve='convex', direction='increasing'
            )
            k_elbow = float(kneedle.elbow) if kneedle.elbow is not None else None
        except Exception:
            k_elbow = None
    except Exception:
        pass
    return k_star, k_elbow

# ===== Aggregated metrics ====================================================

def compute_metrics_aggregated(agg_df: pd.DataFrame, d_star: int) -> Dict[str, Dict]:
    """
    Metrics on aggregated (mean) curves.
    Expects: policy, variant, d, test_error_mean, (train_error_mean optional).
    """
    out = {}
    for (policy, variant), dfp in agg_df.groupby(['policy', 'variant']):
        dfp = dfp.sort_values('d')
        metrics = {}

        # AOM on mean curve
        if 'test_error_mean' in dfp.columns:
            min_err = dfp['test_error_mean'].min()
            metrics['aom_mean'] = float((dfp['test_error_mean'] - min_err).sum())

        # gen-gap slope (if train/test both present)
        if {'train_error_mean', 'test_error_mean'}.issubset(dfp.columns):
            mask = dfp['d'] > d_star
            if mask.sum() > 1:
                try:
                    slope, _ = np.polyfit(
                        dfp.loc[mask, 'd'],
                        (dfp.loc[mask, 'train_error_mean'] - dfp.loc[mask, 'test_error_mean']), 1
                    )
                    metrics['gen_gap_slope_mean'] = float(slope)
                except Exception:
                    metrics['gen_gap_slope_mean'] = np.nan

        # k* & knee
        k_star, k_elbow = compute_breakpoints_on_group(dfp)
        metrics['k_star'] = k_star
        metrics['k_elbow'] = k_elbow

        out[f'{policy}::{variant}'] = metrics
    return out

# ===== Styling helpers =======================================================

def _base_rcparams():
    # Reproducible, print-friendly defaults (no seaborn)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'IBM Plex Sans', 'DejaVu Sans', 'Helvetica', 'Arial'],
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.edgecolor': '#C9CED6',
        'axes.linewidth': 0.9,
        'grid.color': '#AEB6C2',
        'grid.alpha': 0.18,
        'grid.linestyle': '-',
        'axes.grid': True,
        'axes.axisbelow': True,
        'figure.constrained_layout.use': True,
    })

def get_color_and_style(policy: str, variant: str, variant_styles: dict) -> dict:
    """Consistent, print-friendly styling for each (policy, variant)."""
    key = f"{policy}:{variant}"
    if key not in variant_styles:
        # Tol colorblind-safe palette + alt set
        tol = [
            "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
            "#DDCC77", "#CC6677", "#882255", "#AA4499", "#6699CC",
        ]
        alt = [
            "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
            "#e6ab02", "#a6761d", "#666666", "#1f77b4", "#9467bd",
        ]
        base_palette = tol if policy == 'two_gate' else alt
        # Rotate per-policy for stable assignment
        idx = len([k for k in variant_styles if k.startswith(policy + ":")]) % len(base_palette)
        color = base_palette[idx]

        if policy == 'two_gate':
            style = dict(
                lw=2.3, alpha=0.96, marker='o', ms=5.6, linestyle='-',
                zorder=10, markeredgewidth=1.0, markerfacecolor='white',
                markeredgecolor=color,
            )
        else:  # destructive
            style = dict(
                lw=1.7, alpha=0.9, marker='s', ms=4.8, linestyle='--',
                zorder=6, dashes=(4, 2.4), markeredgewidth=0.9,
                markerfacecolor='white', markeredgecolor=color,
            )
        variant_styles[key] = {**style, 'color': color}
    return variant_styles[key]

def _stable_jitter(key, d_vals, width=0.12):
    # Gentle jitter for categorical-ish d to de-clutter markers
    rng = np.random.RandomState(abs(hash(key)) % 2**32)
    return d_vals + rng.uniform(-width, width, size=len(d_vals))

# ===== Plotting ==============================================================

def plot_learning_curves(
    results: pd.DataFrame,
    d_star: Optional[int],
    save_path: Optional[str] = None,
    show: bool = True,
    facet: bool = True,
    overlay_destructive_in_two_gate: bool = True,
) -> plt.Figure:
    """
    Publication-grade learning curves.

    - Faceted: left = Destructive, right = Two-Gate
    - Two-Gate panel overlays destructive (muted) for context (same y, clipped x)
    - **Two-Gate x-axis will be clipped to [d_min, min(d_max, d_star + 1)]** if d_star is provided
    """

    _base_rcparams()

    # Determine columns
    is_agg = 'test_error_mean' in results.columns
    ycol = 'test_error_mean' if is_agg else 'test_error'
    semcol = 'test_error_sem' if is_agg and 'test_error_sem' in results.columns else None

    # Global x-range (for destructive); two_gate will be clipped later if requested
    d_min, d_max = float(results['d'].min()), float(results['d'].max())
    x_pad = max(1.0, round(0.02 * (d_max - d_min), 2)) if d_max > d_min else 1.0
    global_xlim = (d_min - x_pad, d_max + x_pad)

    # Figure/axes
    if facet:
        # Increase figure height to accommodate legend
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.5), gridspec_kw={'wspace': 0.3}, constrained_layout=True)
        
        # Adjust subplot parameters to make room for legend
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.02, wspace=0.1)
        
        axes_map = {'destructive': axes[0], 'two_gate': axes[1]}
        axes[0].set_title('Destructive Policy', pad=10)
        axes[1].set_title('Two-Gate Policy', pad=10)
    else:
        fig, ax = plt.subplots(figsize=(9.0, 5.4))
        axes_map = {'destructive': ax, 'two_gate': ax}

    variant_styles = {}
    handles, labels = [], []

    # Primary plotting by (policy, variant)
    for (policy, variant), g in results.groupby(['policy', 'variant']):
        ax = axes_map.get(policy, list(axes_map.values())[0])
        g = g.sort_values('d').copy()

        # If we're in the two_gate panel and we have d_star, clip data & x-axis to d_star+1
        if facet and policy == 'two_gate' and d_star is not None:
            xmax_tg = min(d_max, float(d_star) + 1.0)
            g = g[g['d'] <= xmax_tg].copy()

        x = g['d'].values
        if len(x) == 0:
            continue

        xj = _stable_jitter((policy, variant), x)
        style = get_color_and_style(policy, variant, variant_styles)
        style_plot = {k: v for k, v in style.items() if k != 'zorder'}
        z = style.get('zorder', 7)

        h = ax.plot(xj, g[ycol].values, label=f"{policy}:{variant}", zorder=z, **style_plot)[0]
        if semcol:
            ax.fill_between(
                xj, (g[ycol] - g[semcol]).values, (g[ycol] + g[semcol]).values,
                color=style['color'], alpha=0.15, zorder=z-1
            )
        handles.append(h)
        labels.append(f"{policy}:{variant}")

    # Optional: overlay destructive in the two_gate panel (context view)
    if facet and overlay_destructive_in_two_gate and 'two_gate' in axes_map and 'destructive' in results['policy'].unique():
        ax_overlay = axes_map['two_gate']
        xmax_tg = min(d_max, float(d_star) + 1.0) if d_star is not None else d_max
        backdrop = results[(results['policy'] == 'destructive') & (results['d'] <= xmax_tg)]
        for (policy, variant), g in backdrop.groupby(['policy', 'variant']):
            g = g.sort_values('d').copy()
            x = g['d'].values
            if len(x) == 0:
                continue
            xj = _stable_jitter(("overlay", variant), x, width=0.06)  # smaller jitter for overlay
            style = get_color_and_style('destructive', variant, variant_styles)
            z = 3  # lower z for backdrop
            ax_overlay.plot(
                xj, g[ycol].values,
                linestyle='--', dashes=(4, 2.4),
                lw=1.2, alpha=0.35, marker='s', ms=3.8,
                color=style['color'], zorder=z,
            )
            if semcol:
                ax_overlay.fill_between(
                    xj, (g[ycol] - g[semcol]).values, (g[ycol] + g[semcol]).values,
                    color=style['color'], alpha=0.08, zorder=z-1
                )

    # Final axis polish
    for panel, ax in axes_map.items():
        # X-limits: destructive = global; two_gate = clipped to d_star+1 if provided
        if panel == 'two_gate' and d_star is not None:
            xmax_tg = min(d_max, float(d_star) + 1.0)
            ax.set_xlim(global_xlim[0], xmax_tg)
        else:
            ax.set_xlim(*global_xlim)

        # Vertical guide at d*
        if d_star is not None:
            ax.axvline(d_star, color='#707985', ls=(0, (3, 3)), lw=1.2, alpha=0.95, zorder=1)

        # Labels
        ax.set_xlabel('Polynomial degree (d)', labelpad=8)
        if panel == 'destructive' or not facet:
            ax.set_ylabel('Test error', labelpad=8)

        # Ticks & spines
        ax.tick_params(axis='both', which='both', direction='out', length=3.6, width=0.7)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', length=2, width=0.5)
        ax.tick_params(axis='y', which='minor', length=2, width=0.5)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)

    # Legend
    if facet and len(handles) > 0:
        # Place legend below the subplots with some padding
        legend = fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.05),  # Position below the subplots
            ncol=min(4, len(handles)),
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=1.0,
            edgecolor='#E0E0E0',
            bbox_transform=fig.transFigure  # Use figure coordinates
        )
        
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0.1, 1, 1])
    else:
        legend = list(axes_map.values())[0].legend(
            handles, labels,
            loc='upper left', bbox_to_anchor=(1.02, 1.0),
            frameon=True, framealpha=0.98, facecolor='white', edgecolor='#D8DCE3',
            fancybox=True, borderpad=0.8, handlelength=2.2, handletextpad=0.6
        )
        legend.get_frame().set_linewidth(0.6)

    if facet:
        fig.suptitle('Test Error vs. Model Capacity', y=1.02, fontsize=13.2, fontweight='semibold')

    # Save / show
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
            logging.info(f'Saved {save_path}')
        except Exception as e:
            logging.error(f'Error saving figure: {e}')
    if show:
        plt.show()

    return fig


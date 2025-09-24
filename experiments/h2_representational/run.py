#!/usr/bin/env python3
"""
Experiment H.2 runner — 1D Polynomial Thresholds.

- Loops over m_values × noise_levels × seeds
- Runs destructive & two-gate policies
- Aggregates per (policy, variant, d, m, noise)
- Computes breakpoints on aggregated curves
- Saves curves & metrics and generates plots

Usage:
  python run_h2.py --config configs/default.yaml --output-dir results --seeds 42 43 44
"""
import os, sys, yaml, argparse, logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from data import generate_polynomial_data, save_dataset, load_dataset
from models import PTFLearner
from policies import create_policy, PolicyOutput
from metrics import (
    plot_learning_curves,
    compute_breakpoints_on_group,
    compute_metrics_aggregated
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("H2")

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Ensure numeric values are properly typed
    if 'lambda_reg' in config:
        config['lambda_reg'] = float(config['lambda_reg'])
    return config

def run_single_experiment(
    config: Dict[str, Any],
    seed: int,
    output_dir: Path,
    save_data: bool = True,
    save_models: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / f'data_seed{seed}.pkl'
    if data_path.exists():
        dataset = load_dataset(data_path)
    else:
        dataset = generate_polynomial_data(
            d_star=config['d_star'],
            m=config['m'],
            val_ratio=config['val_ratio'],
            test_size=config['test_size'],
            noise_level=config['noise_level'],
            seed=seed
        )
        if save_data:
            save_dataset(dataset, str(data_path))

    results = []
    models = {}

    for policy_type in ['destructive', 'two_gate']:
        if policy_type == 'destructive' and 'destructive_lambdas' not in config:
            continue
        if policy_type == 'two_gate' and 'K_values' not in config:
            continue

        variants = (
            [{'lambda_dest': l} for l in config['destructive_lambdas']]
            if policy_type == 'destructive'
            else [{'K': K, 'tau': config.get('tau', 0.0), 'delta': config.get('delta', 0.05),
                   'c': config.get('c', 2.0)} for K in config['K_values']]
        )

        for variant in variants:
            policy = create_policy(policy_type=policy_type, max_degree=config['max_degree'], **variant)
            d = 1
            current_model = PTFLearner(d, lambda_reg=config['lambda_reg'])

            policy_str = policy_type
            variant_str = '_'.join(f'{k}_{v}' for k, v in variant.items())

            for step in range(config['max_steps']):
                current_model.fit(dataset.X_train, dataset.y_train)

                cur_train = 1 - current_model.score(dataset.X_train, dataset.y_train)
                cur_val   = 1 - current_model.score(dataset.X_val,   dataset.y_val)
                cur_test  = 1 - current_model.score(dataset.X_test,  dataset.y_test)

                next_d = d + 1
                if next_d > config['max_degree']:
                    decision = PolicyOutput(accept=False, info={'reason': 'max_degree_reached'})
                    cand = None
                else:
                    candidate = PTFLearner(next_d, lambda_reg=config['lambda_reg']).fit(
                        dataset.X_train, dataset.y_train
                    )
                    cand_train = 1 - candidate.score(dataset.X_train, dataset.y_train)
                    cand_val   = 1 - candidate.score(dataset.X_val,   dataset.y_val)
                    cand_test  = 1 - candidate.score(dataset.X_test,  dataset.y_test)

                    if policy_type == 'destructive':
                        decision = policy.decide(current_d=d, train_error_new=cand_train)
                    else:
                        decision = policy.decide(
                            current_d=d,
                            val_error_old=cur_val,
                            val_error_new=cand_val,
                            n_val=len(dataset.X_val)
                        )

                    cand = dict(train=cand_train, val=cand_val, test=cand_test)

                    if decision.accept:
                        current_model = candidate
                        d = next_d
                        cur_train, cur_val, cur_test = cand_train, cand_val, cand_test

                row = {
                    'seed': seed,
                    'policy': policy_str,
                    'variant': variant_str,
                    'policy_type': policy_type,
                    'step': step,
                    'd': d,
                    'train_error': cur_train,
                    'val_error': cur_val,
                    'test_error': cur_test,
                    'accepted': decision.accept,
                    'reason': decision.info.get('reason', 'unknown'),
                    'm': dataset.metadata['m'],
                    'noise': dataset.metadata['noise_level'],
                    **({} if policy_type != 'destructive' else {'lambda_dest': variant.get('lambda_dest', np.nan)}),
                    **({} if policy_type != 'two_gate' else {'K': variant.get('K', np.nan)})
                }

                if cand is not None:
                    row.update({
                        'cand_d': next_d,
                        'cand_train_error': cand['train'],
                        'cand_val_error':   cand['val'],
                        'cand_test_error':  cand['test']
                    })

                if policy_type == 'destructive':
                    row['utility'] = - (cand['train'] if cand is not None else cur_train) + \
                                     variant.get('lambda_dest', 0.0) * (next_d if cand is not None else d)

                results.append(row)

                if save_models and step % 5 == 0:
                    model_path = output_dir / f"model_{policy_str}_{variant_str}_d{d}_seed{seed}.pkl"
                    joblib.dump(current_model, model_path)

                if not decision.accept or d >= config['max_degree']:
                    break

            models[f"{policy_str}_{variant_str}"] = current_model

    df = pd.DataFrame(results)
    df.to_csv(output_dir / f'results_seed{seed}.csv', index=False)
    return df, {}

def aggregate_results(results_dir: Path, d_star: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Grab all results under nested (m_*/noise_*/seed_*)
    result_files = list(results_dir.glob('**/results_seed*.csv'))
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")

    frames = []
    for f in result_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")
    results_df = pd.concat(frames, ignore_index=True)

    # Ensure required columns exist
    for col in ['policy', 'variant', 'd', 'train_error', 'val_error', 'test_error']:
        if col not in results_df.columns:
            raise ValueError(f"Missing required column: {col}")

    if 'utility' not in results_df.columns: results_df['utility'] = np.nan
    if 'm' not in results_df.columns:       results_df['m'] = np.nan
    if 'noise' not in results_df.columns:   results_df['noise'] = np.nan

    # Aggregate: per (policy, variant, d, m, noise)
    group_cols = ['policy', 'variant', 'd', 'm', 'noise']
    agg = results_df.groupby(group_cols, dropna=False).agg({
        'train_error': ['mean', 'std', 'count'],
        'val_error':   ['mean', 'std'],
        'test_error':  ['mean', 'std'],
        'utility':     ['mean', 'std']
    }).reset_index()
    agg.columns = ['_'.join(c).strip('_') for c in agg.columns.values]

    # SEM for test error
    if {'test_error_std', 'train_error_count'}.issubset(set(agg.columns)):
        agg['test_error_sem'] = agg['test_error_std'] / np.sqrt(agg['train_error_count'].clip(lower=1))

    # Breakpoints per (policy, variant, m, noise)
    bp_rows = []
    for keys, g in agg.groupby(['policy', 'variant', 'm', 'noise'], dropna=False):
        k_star, k_elbow = compute_breakpoints_on_group(g)
        bp_rows.append({
            'policy': keys[0], 'variant': keys[1], 'm': keys[2], 'noise': keys[3],
            'k_star': k_star, 'k_elbow': k_elbow
        })
    bp = pd.DataFrame(bp_rows)
    agg = agg.merge(bp, on=['policy', 'variant', 'm', 'noise'], how='left')

    # Save
    agg.to_csv(results_dir / 'aggregated_results.csv', index=False)
    bp.to_csv(results_dir / 'aggregated_breakpoints.csv', index=False)

    # Aggregated curve metrics on mean curves
    agg_metrics = compute_metrics_aggregated(agg, d_star=d_star)
    with open(results_dir / 'aggregated_curve_metrics.yaml', 'w') as f:
        yaml.dump(agg_metrics, f)

    return agg, agg_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('--output-dir', type=str, default='results')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for provenance
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    if not args.aggregate_only:
        m_values = config.get('m_values', [config.get('m', 1000)])
        noise_levels = config.get('noise_levels', [config.get('noise_level', 0.0)])

        for m in m_values:
            for noise in noise_levels:
                logger.info(f"Grid run: m={m}, noise={noise}, seeds={args.seeds}")
                grid_dir = output_dir / f"m_{m}" / f"noise_{noise}"
                grid_dir.mkdir(parents=True, exist_ok=True)
                for seed in args.seeds:
                    try:
                        run_single_experiment(
                            config={**config, 'm': m, 'noise_level': noise},
                            seed=seed,
                            output_dir=grid_dir / f"seed_{seed}",
                            save_data=True,
                            save_models=False
                        )
                    except Exception as e:
                        logger.error(f"Run failed m={m} noise={noise} seed={seed}: {e}", exc_info=True)

    # Aggregate across everything under results_dir
    try:
        agg, agg_metrics = aggregate_results(output_dir, d_star=config['d_star'])
        logger.info("Aggregation complete.")

        # Plot aggregated curves (one line per (policy,variant), across all m/noise mixed).
        # If you want per-(m,noise) plots, loop over agg.groupby(['m','noise']).
        plot_learning_curves(
            results=agg,
            d_star=config['d_star'],
            save_path=output_dir / 'learning_curves.png',
            show=False
        )
        logger.info("Saved learning_curves.png")
        # Also save one plot per (m, noise) cell — cleaner figures for the paper
        try:
            # If your agg columns are named differently, adjust ['m','noise'] below
            for (m_val, noise_val), cell in agg.groupby(['m', 'noise'], dropna=False):
                if cell.empty:
                    continue
                out_path = output_dir / f'learning_curves_m{m_val}_noise{noise_val}.png'
                plot_learning_curves(
                    results=cell,              # works because cell still has *_mean / *_sem columns
                    d_star=config['d_star'],
                    save_path=out_path,
                    show=False
                )
                logger.info(f"Saved {out_path}")
        except Exception as e:
            logger.warning(f"Per-(m,noise) plotting skipped: {e}")

    except Exception as e:
        logger.error(f"Aggregation/plotting failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()

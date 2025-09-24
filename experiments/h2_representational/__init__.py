"""
Experiment H.2: Representational Axis - 1D Polynomial Thresholds

This module implements an experiment demonstrating the utility-learning tension
in the representational axis, where an agent can increase its capacity by adding
polynomial features.
"""

from .data import generate_polynomial_data, save_dataset, load_dataset
from .models import PTFLearn, compute_vc_bound
from .policies import create_policy, DestructivePolicy, TwoGatePolicy, PolicyOutput
from .metrics import compute_metrics, plot_learning_curves, plot_decision_boundaries
from .run import run_single_experiment, aggregate_results, main

__all__ = [
    'generate_polynomial_data',
    'save_dataset',
    'load_dataset',
    'PTFLearn',
    'compute_vc_bound',
    'create_policy',
    'DestructivePolicy',
    'TwoGatePolicy',
    'PolicyOutput',
    'compute_metrics',
    'plot_learning_curves',
    'plot_decision_boundaries',
    'run_single_experiment',
    'aggregate_results',
    'main'
]

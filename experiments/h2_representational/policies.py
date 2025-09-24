"""Policies (Destructive and Two-Gate)."""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from models import PTFLearner, compute_vc_bound

@dataclass
class PolicyOutput:
    accept: bool
    info: Dict[str, Any]

class Policy:
    def decide(self, *args, **kwargs) -> PolicyOutput:
        raise NotImplementedError

class DestructivePolicy(Policy):
    """Accepts capacity increases, utility = -train_error + lambda * d'."""
    def __init__(self, lambda_dest: float = 0.01, max_degree: int = 30):
        self.lambda_dest = lambda_dest
        self.max_degree = max_degree

    def decide(self, current_d: int, train_error_new: float, **_) -> PolicyOutput:
        next_d = current_d + 1
        if next_d > self.max_degree:
            return PolicyOutput(False, {'reason': 'max_degree_reached', 'utility': -np.inf})
        utility = -train_error_new + self.lambda_dest * next_d
        return PolicyOutput(True, {'reason': 'capacity_reward', 'utility': utility, 'train_error': train_error_new})

class TwoGatePolicy(Policy):
    """
    Accept if:
      (1) next_d <= K (capacity gate) and
      (2) val_new <= val_old - (2 * eps + tau) (validation gate),
    where eps uses a VC-style bound with d'.
    """
    def __init__(self, K: int, tau: float = 0.0, delta: float = 0.05, c: float = 2.0, max_degree: int = 30):
        self.K = K
        self.tau = tau
        self.delta = delta
        self.c = c
        self.max_degree = max_degree
        self.best_val_error = float('inf')

    def decide(self, current_d: int, val_error_old: float, val_error_new: float, n_val: int, **_) -> PolicyOutput:
        next_d = current_d + 1
        if next_d > self.K or next_d > self.max_degree:
            return PolicyOutput(False, {
                'reason': 'capacity_constraint', 'current_d': current_d,
                'next_d': next_d, 'K': self.K, 'max_degree': self.max_degree
            })

        epsilon = compute_vc_bound(next_d, n_val, self.delta, self.c)
        target = min(val_error_old, self.best_val_error)
        if val_error_new <= target - (epsilon + self.tau):
            self.best_val_error = min(self.best_val_error, val_error_new)
            return PolicyOutput(True, {
                'reason': 'validation_improvement',
                'val_error_old': val_error_old,
                'val_error_new': val_error_new,
                'best_val_error': self.best_val_error,
                'epsilon': epsilon,
                'improvement': target - val_error_new - (epsilon + self.tau)
            })
        else:
            return PolicyOutput(False, {
                'reason': 'validation_gate',
                'val_error_old': val_error_old,
                'val_error_new': val_error_new,
                'best_val_error': self.best_val_error,
                'epsilon': epsilon,
                'improvement': target - val_error_new - (epsilon + self.tau)
            })

def create_policy(policy_type: str, **kwargs) -> Policy:
    if policy_type == 'destructive':
        return DestructivePolicy(**kwargs)
    if policy_type == 'two_gate':
        return TwoGatePolicy(**kwargs)
    raise ValueError(f"Unknown policy type: {policy_type}")

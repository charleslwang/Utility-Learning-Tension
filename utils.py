# experiments/utils.py
from __future__ import annotations
import os, random
import numpy as np
from typing import Tuple

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def set_all_seeds(seed: int) -> None:
    import numpy as np, random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def split_indices(n_total: int, m: int, n_v: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    i_tr = idx[:m]
    i_va = idx[m:m+n_v]
    i_te = idx[m+n_v:]
    return i_tr, i_va, i_te

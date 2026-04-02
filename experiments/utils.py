import os
import random
from typing import List

import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)



def parse_int_list(text: str) -> List[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    parsed = [int(item) for item in values]
    if sorted(parsed) != parsed:
        raise ValueError("Integer list must be sorted in non-decreasing order.")
    return parsed

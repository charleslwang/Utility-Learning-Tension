from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class DataConfig:
    m: int = 1000          # train size
    n_v: int = 1000        # val size
    n_test: int = 50000    # test size
    sigma: float = 0.25
    flip_prob: float = 0.15
    dim: int = 1
    seed: int = 1337

@dataclass
class GateConfig:
    delta_v: float = 0.05
    c0: float = 2.0            # VC bound constant
    tau_mult: float = 0.5      # tau = tau_mult * epsV
    K_max: int = 50            # absolute cap (safety)
    K_mult: float = 0.5        # K(m) = K_mult * sqrt(m)

@dataclass
class RunConfig:
    # policy label for filenames; actual logic lives in run_h.py
    policy: Literal["twogate","dest_train","dest_val","dest_val_loose"] = "twogate"
    degree_start: int = 0
    degree_step: int = 1
    seeds: int = 20
    out_dir: str = "outputs"
    # K(m) schedule:  floor(K_mult * sqrt(m)); override with int to force.
    K_schedule_multiplier: float = 0.5
    K_override: Optional[int] = None

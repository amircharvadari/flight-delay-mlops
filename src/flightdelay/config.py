"""Load the YAML configuration and expose small helpers.

Kept deliberately simple: one function to load the config, one to set seeds,
and one guard that makes the 2025 holdout explicit in code (D-020).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read a YAML config file and return it as a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fix the random generators used by the project."""
    random.seed(seed)
    np.random.seed(seed)


def is_holdout_month(month: str, config: dict[str, Any]) -> bool:
    """Return True if a "YYYY-MM" month belongs to the isolated final holdout."""
    start, end = config["periods"]["final_holdout"]
    return start <= month <= end

import os
import random

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and, if available, PyTorch.

    Parameters
    ----------
    seed: int
        The random seed to use.
    deterministic: bool, optional
        Whether to set additional flags for deterministic behaviour when
        using PyTorch. Has no effect if PyTorch is not installed.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

from __future__ import annotations

import numpy as np


def mse(clean: np.ndarray, estimate: np.ndarray) -> float:
    diff = clean.astype(np.float64) - estimate.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(clean: np.ndarray, estimate: np.ndarray, max_value: float = 255.0) -> float:
    error = mse(clean, estimate)
    if error <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((max_value ** 2) / error))

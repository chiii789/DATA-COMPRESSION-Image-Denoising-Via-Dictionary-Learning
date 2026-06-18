from __future__ import annotations

import numpy as np


def omp_single(
    D: np.ndarray,
    y: np.ndarray,
    sparsity: int,
    error_tolerance: float | None = None,
) -> np.ndarray:
    _, n_atoms = D.shape
    x = np.zeros(n_atoms, dtype=np.float64)
    residual = y.astype(np.float64).copy()
    support: list[int] = []

    for _ in range(min(sparsity, n_atoms)):
        correlations = D.T @ residual
        if support:
            correlations[np.asarray(support)] = 0.0
        atom_idx = int(np.argmax(np.abs(correlations)))
        if atom_idx in support or abs(correlations[atom_idx]) < 1e-12:
            break
        support.append(atom_idx)

        D_support = D[:, support]
        coeffs, _, _, _ = np.linalg.lstsq(D_support, y, rcond=None)
        residual = y - D_support @ coeffs
        residual_norm = np.linalg.norm(residual)
        if residual_norm < 1e-6:
            break
        if error_tolerance is not None and residual_norm <= error_tolerance:
            break

    if support:
        x[np.asarray(support)] = coeffs
    return x


def omp_batch(
    D: np.ndarray,
    Y: np.ndarray,
    sparsity: int,
    error_tolerance: float | np.ndarray | None = None,
) -> np.ndarray:
    n_atoms = D.shape[1]
    n_samples = Y.shape[1]
    X = np.zeros((n_atoms, n_samples), dtype=np.float64)

    if error_tolerance is None or np.isscalar(error_tolerance):
        tolerances = [error_tolerance] * n_samples
    else:
        tolerances = np.asarray(error_tolerance, dtype=np.float64).reshape(-1)
        if tolerances.size != n_samples:
            raise ValueError("error_tolerance must be scalar or have one value per sample.")

    for idx in range(n_samples):
        X[:, idx] = omp_single(D, Y[:, idx], sparsity, tolerances[idx])
    return X

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from tqdm import tqdm

from src.omp import omp_batch
from src.patch_utils import (
    extract_overlapping_patches,
    matrix_to_patches,
    patches_to_matrix,
    reconstruct_from_overlapping_patches,
)


def normalise_columns(D: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return D / norms


def initialise_dictionary(Y_train: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    if Y_train.ndim != 2:
        raise ValueError(f"Y_train must be a 2D matrix, got shape {Y_train.shape}.")
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive.")
    rng = np.random.default_rng(seed)
    n_samples = Y_train.shape[1]
    replace = n_samples < n_atoms
    indices = rng.choice(n_samples, size=n_atoms, replace=replace)
    D = Y_train[:, indices].copy()
    return normalise_columns(D)


@lru_cache(maxsize=None)
def create_dct_dictionary(patch_size: int, n_atoms: int) -> np.ndarray:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive.")
    atoms_per_dim = math.ceil(math.sqrt(n_atoms))
    dct_1d = np.zeros((patch_size, atoms_per_dim), dtype=np.float64)
    grid = np.arange(patch_size, dtype=np.float64)

    for k in range(atoms_per_dim):
        basis = np.cos(np.pi * (2.0 * grid + 1.0) * k / (2.0 * atoms_per_dim))
        if k > 0:
            basis = basis - np.mean(basis)
        dct_1d[:, k] = basis / np.linalg.norm(basis)

    atoms: list[np.ndarray] = []
    for u in range(atoms_per_dim):
        for v in range(atoms_per_dim):
            atoms.append(np.kron(dct_1d[:, u], dct_1d[:, v]))
    D = np.column_stack(atoms[:n_atoms])
    return normalise_columns(D)


def ksvd(
    Y_train: np.ndarray,
    n_atoms: int,
    sparsity: int,
    n_iter: int,
    seed: int,
    init_method: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    if Y_train.ndim != 2:
        raise ValueError(f"Y_train must be a 2D matrix, got shape {Y_train.shape}.")
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive.")
    if sparsity <= 0:
        raise ValueError("sparsity must be positive.")
    if sparsity > n_atoms:
        raise ValueError(f"sparsity={sparsity} cannot exceed n_atoms={n_atoms}.")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")
    if init_method not in {"random", "dct"}:
        raise ValueError(f"Unsupported init_method '{init_method}'.")
    if init_method == "dct":
        patch_size = int(round(np.sqrt(Y_train.shape[0])))
        if patch_size * patch_size != Y_train.shape[0]:
            raise ValueError(
                "Y_train row dimension must be a perfect square when using DCT initialization."
            )
        D = create_dct_dictionary(patch_size, n_atoms)
    else:
        D = initialise_dictionary(Y_train, n_atoms, seed)

    rng = np.random.default_rng(seed)
    X = np.zeros((n_atoms, Y_train.shape[1]), dtype=np.float64)

    for _ in tqdm(range(n_iter), desc="K-SVD", leave=False):
        X = omp_batch(D, Y_train, sparsity)

        for atom_idx in range(n_atoms):
            usage = np.flatnonzero(np.abs(X[atom_idx]) > 1e-12)
            if usage.size == 0:
                replacement_idx = int(rng.integers(0, Y_train.shape[1]))
                D[:, atom_idx] = Y_train[:, replacement_idx]
                D[:, atom_idx] /= max(np.linalg.norm(D[:, atom_idx]), 1e-12)
                continue

            coeffs = X[atom_idx, usage].copy()
            X[atom_idx, usage] = 0.0
            residual = Y_train[:, usage] - D @ X[:, usage]

            U, S, Vt = np.linalg.svd(residual, full_matrices=False)
            D[:, atom_idx] = U[:, 0]
            X[atom_idx, usage] = S[0] * Vt[0, :]

        D = normalise_columns(D)

    X = omp_batch(D, Y_train, sparsity)
    return D, X


def denoise_patch_matrix(
    patch_matrix: np.ndarray,
    D: np.ndarray,
    patch_size: int,
    sparsity: int,
    sigma_noise: float | None = None,
) -> np.ndarray:
    if patch_matrix.ndim != 2:
        raise ValueError(f"patch_matrix must be 2D, got shape {patch_matrix.shape}.")
    if patch_matrix.shape[0] != patch_size * patch_size:
        raise ValueError(
            f"patch_matrix first dimension must equal patch_size^2={patch_size * patch_size}, "
            f"got {patch_matrix.shape[0]}."
        )
    if D.ndim != 2:
        raise ValueError(f"Dictionary D must be 2D, got shape {D.shape}.")
    if D.shape[0] != patch_matrix.shape[0]:
        raise ValueError(
            f"Dictionary row dimension {D.shape[0]} does not match patch dimension {patch_matrix.shape[0]}."
        )
    if sparsity <= 0:
        raise ValueError("sparsity must be positive.")
    if sparsity > D.shape[1]:
        raise ValueError(f"sparsity={sparsity} cannot exceed number of atoms={D.shape[1]}.")
    patch_means = np.mean(patch_matrix, axis=0, keepdims=True)
    centered = patch_matrix - patch_means

    error_tolerance = None
    if sigma_noise is not None:
        patch_dim = patch_size * patch_size
        error_tolerance = 1.15 * sigma_noise * np.sqrt(patch_dim)

    codes = omp_batch(D, centered, sparsity, error_tolerance=error_tolerance)
    return D @ codes + patch_means


def denoise_with_dictionary(
    noisy_image: np.ndarray,
    D: np.ndarray,
    patch_size: int,
    sparsity: int,
    sigma_noise: float | None = None,
) -> np.ndarray:
    patches = extract_overlapping_patches(noisy_image, patch_size)
    patch_matrix = patches_to_matrix(patches)
    reconstructed = denoise_patch_matrix(
        patch_matrix=patch_matrix,
        D=D,
        patch_size=patch_size,
        sparsity=sparsity,
        sigma_noise=sigma_noise,
    )
    denoised_patches = matrix_to_patches(reconstructed, patch_size)
    denoised = reconstruct_from_overlapping_patches(
        denoised_patches, noisy_image.shape, patch_size
    )
    return np.clip(denoised, 0.0, 255.0)

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def extract_overlapping_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}.")
    height, width = image.shape
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if patch_size > min(height, width):
        raise ValueError(
            f"patch_size={patch_size} is larger than image dimensions {image.shape}."
        )
    windows = sliding_window_view(image, (patch_size, patch_size))
    return windows.reshape(-1, patch_size, patch_size).astype(np.float64)


def reconstruct_from_overlapping_patches(
    patches: np.ndarray,
    image_shape: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    if len(image_shape) != 2:
        raise ValueError(f"image_shape must describe a 2D image, got {image_shape}.")
    height, width = image_shape
    expected = (height - patch_size + 1) * (width - patch_size + 1)
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if patch_size > min(height, width):
        raise ValueError(
            f"patch_size={patch_size} is larger than image dimensions {image_shape}."
        )
    if patches.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} patches for image shape {image_shape} and patch_size={patch_size}, "
            f"got {patches.shape[0]}."
        )
    out = np.zeros(image_shape, dtype=np.float64)
    weight = np.zeros(image_shape, dtype=np.float64)
    index = 0
    for row in range(height - patch_size + 1):
        for col in range(width - patch_size + 1):
            out[row : row + patch_size, col : col + patch_size] += patches[index]
            weight[row : row + patch_size, col : col + patch_size] += 1.0
            index += 1
    return out / np.maximum(weight, 1e-12)


def patches_to_matrix(patches: np.ndarray) -> np.ndarray:
    if patches.ndim != 3:
        raise ValueError(f"Expected patches with shape (n_patches, p, p), got {patches.shape}.")
    n_patches, patch_size, _ = patches.shape
    return patches.reshape(n_patches, patch_size * patch_size).T.astype(np.float64)


def matrix_to_patches(matrix: np.ndarray, patch_size: int) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D patch matrix, got shape {matrix.shape}.")
    if matrix.shape[0] != patch_size * patch_size:
        raise ValueError(
            f"Patch matrix first dimension must be {patch_size * patch_size}, got {matrix.shape[0]}."
        )
    return matrix.T.reshape(-1, patch_size, patch_size).astype(np.float64)


def sample_training_patches(
    patch_matrix: np.ndarray,
    n_train_patches: int,
    seed: int,
) -> np.ndarray:
    if patch_matrix.ndim != 2:
        raise ValueError(f"Expected a 2D patch matrix, got shape {patch_matrix.shape}.")
    if n_train_patches <= 0:
        raise ValueError("n_train_patches must be positive.")
    n_samples = patch_matrix.shape[1]
    if n_train_patches >= n_samples:
        return patch_matrix.copy()
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_samples, size=n_train_patches, replace=False)
    return patch_matrix[:, indices]

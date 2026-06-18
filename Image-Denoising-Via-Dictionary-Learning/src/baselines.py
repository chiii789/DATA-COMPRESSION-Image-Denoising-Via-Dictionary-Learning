from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from src.ksvd import create_dct_dictionary, denoise_patch_matrix
from src.patch_utils import (
    extract_overlapping_patches,
    matrix_to_patches,
    patches_to_matrix,
    reconstruct_from_overlapping_patches,
)


def gaussian_filter_denoise(noisy_image: np.ndarray, sigma_noise: float) -> np.ndarray:
    filter_sigma = max(0.3, sigma_noise / 18.0)
    denoised = gaussian_filter(noisy_image, sigma=filter_sigma)
    return np.clip(denoised, 0.0, 255.0)


def dct_omp_denoise(
    noisy_image: np.ndarray,
    patch_size: int,
    sparsity: int,
    n_atoms: int,
    sigma_noise: float | None = None,
    patch_matrix: np.ndarray | None = None,
) -> np.ndarray:
    D = create_dct_dictionary(patch_size, n_atoms)
    if patch_matrix is None:
        patch_matrix = patches_to_matrix(extract_overlapping_patches(noisy_image, patch_size))
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

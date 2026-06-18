from __future__ import annotations

from pathlib import Path

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_comparison_figure(
    clean: np.ndarray,
    noisy: np.ndarray,
    gaussian: np.ndarray,
    dct: np.ndarray,
    ksvd: np.ndarray,
    filename: str | Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    images = [clean, noisy, gaussian, dct, ksvd]
    names = ["Clean", "Noisy", "Gaussian", "DCT+OMP", "K-SVD+OMP"]

    for ax, image, name in zip(axes, images, names):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_noise_variance(metrics_df: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        metrics_df.groupby(["sigma", "noise_variance", "method"], as_index=False)[["mse", "psnr"]]
        .mean()
        .sort_values(["sigma", "method"])
    )

    for metric_name, ylabel in [("mse", "Average MSE"), ("psnr", "Average PSNR (dB)")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, group in summary.groupby("method"):
            ax.plot(
                group["noise_variance"],
                group[metric_name],
                marker="o",
                linewidth=2,
                label=method,
            )
        ax.set_xlabel("Noise variance ($\\sigma^2$)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Noise Variance")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"average_{metric_name}_vs_noise_variance.png", dpi=200)
        plt.close(fig)


def save_dictionary_atoms(D: np.ndarray, patch_size: int, filename: str | Path) -> None:
    n_atoms = D.shape[1]
    grid_size = math.ceil(math.sqrt(n_atoms))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = np.atleast_2d(axes)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= n_atoms:
            continue
        atom = D[:, idx].reshape(patch_size, patch_size)
        ax.imshow(atom, cmap="gray")

    fig.tight_layout(pad=0.2)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

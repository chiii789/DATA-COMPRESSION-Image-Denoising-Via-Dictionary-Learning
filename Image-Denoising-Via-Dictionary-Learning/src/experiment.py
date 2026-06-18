from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

from src.baselines import dct_omp_denoise, gaussian_filter_denoise
from src.data_utils import add_gaussian_noise, load_grayscale_image, select_random_images
from src.ksvd import denoise_patch_matrix, ksvd
from src.metrics import mse, psnr
from src.patch_utils import (
    extract_overlapping_patches,
    matrix_to_patches,
    patches_to_matrix,
    reconstruct_from_overlapping_patches,
    sample_training_patches,
)
from src.plotting import plot_metric_vs_noise_variance, save_comparison_figure, save_dictionary_atoms


NOISE_LEVELS = [5, 10, 15, 25]


def _validate_args(args) -> None:
    if args.patch_size > args.image_size:
        raise ValueError(
            f"patch_size={args.patch_size} cannot exceed image_size={args.image_size}."
        )
    if args.sparsity > args.n_atoms:
        raise ValueError(
            f"sparsity={args.sparsity} cannot exceed n_atoms={args.n_atoms}."
        )
    if args.n_atoms < args.patch_size * args.patch_size:
        raise ValueError(
            "n_atoms should be at least patch_size^2 for an overcomplete or complete patch dictionary. "
            f"Got n_atoms={args.n_atoms} and patch_size^2={args.patch_size * args.patch_size}."
        )
    if args.output_dir.strip() == "":
        raise ValueError("output_dir cannot be empty.")


def _prepare_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "images": root / "images",
        "figures": root / "figures",
        "tables": root / "tables",
        "dictionaries": root / "dictionaries",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_method_images(
    dirs: dict[str, Path],
    image_name: str,
    sigma: int,
    clean: np.ndarray,
    noisy: np.ndarray,
    gaussian: np.ndarray,
    dct: np.ndarray,
    learned: np.ndarray,
) -> None:
    arrays = {
        "clean": clean,
        "noisy": noisy,
        "gaussian": gaussian,
        "dct_omp": dct,
        "ksvd_omp": learned,
    }
    for method, image in arrays.items():
        filename = dirs["images"] / f"{image_name}_sigma{sigma}_{method}.png"
        io.imsave(filename, np.clip(image, 0, 255).astype(np.uint8), check_contrast=False)


def _build_wide_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    wide = (
        metrics_df.pivot_table(
            index=["image_name", "sigma", "noise_variance"],
            columns="method",
            values=["mse", "psnr"],
        )
        .sort_index()
    )
    wide.columns = [f"{metric}_{method}".lower().replace("+", "_plus_") for metric, method in wide.columns]
    return wide.reset_index()


def _write_run_config(args, dirs: dict[str, Path], selected_names: list[str]) -> None:
    config = {
        "data_dir": args.data_dir,
        "n_images": args.n_images,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "n_train_patches": args.n_train_patches,
        "n_atoms": args.n_atoms,
        "sparsity": args.sparsity,
        "ksvd_iter": args.ksvd_iter,
        "seed": args.seed,
        "output_dir": str(dirs["root"]),
        "fast_mode": bool(args.fast_mode),
        "noise_levels": NOISE_LEVELS,
        "selected_images": selected_names,
    }
    (dirs["root"] / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )


def run_experiment(args) -> None:
    if args.fast_mode:
        args.n_train_patches = 1000
        args.n_atoms = 64
        args.ksvd_iter = 3
        args.sparsity = 4

    _validate_args(args)
    dirs = _prepare_output_dirs(args.output_dir)
    selected_paths = select_random_images(args.data_dir, args.n_images, args.seed)
    selected_names = [path.name for path in selected_paths]
    (dirs["root"] / "selected_images.txt").write_text(
        "\n".join(selected_names) + "\n", encoding="utf-8"
    )
    _write_run_config(args, dirs, selected_names)

    print(f"Selected images: {', '.join(selected_names)}")
    print(
        "Settings: "
        f"patch_size={args.patch_size}, n_train_patches={args.n_train_patches}, "
        f"n_atoms={args.n_atoms}, sparsity={args.sparsity}, ksvd_iter={args.ksvd_iter}"
    )

    metric_rows: list[dict[str, float | int | str]] = []

    image_bar = tqdm(selected_paths, desc="Images")
    for image_idx, image_path in enumerate(image_bar):
        clean = load_grayscale_image(image_path, args.image_size)
        image_stem = image_path.stem

        for sigma in tqdm(NOISE_LEVELS, desc=f"{image_stem} sigmas", leave=False):
            noisy = add_gaussian_noise(clean, sigma, seed=args.seed + image_idx * 100 + sigma)
            noisy_patches = extract_overlapping_patches(noisy, args.patch_size)
            noisy_matrix = patches_to_matrix(noisy_patches)
            gaussian = gaussian_filter_denoise(noisy, sigma_noise=sigma)
            dct = dct_omp_denoise(
                noisy_image=noisy,
                patch_size=args.patch_size,
                sparsity=args.sparsity,
                n_atoms=args.n_atoms,
                sigma_noise=sigma,
                patch_matrix=noisy_matrix,
            )

            centered_train = noisy_matrix - np.mean(noisy_matrix, axis=0, keepdims=True)
            Y_train = sample_training_patches(
                centered_train,
                n_train_patches=args.n_train_patches,
                seed=args.seed + image_idx * 1000 + sigma,
            )

            D_learned, _ = ksvd(
                Y_train=Y_train,
                n_atoms=args.n_atoms,
                sparsity=args.sparsity,
                n_iter=args.ksvd_iter,
                seed=args.seed + image_idx * 1000 + sigma,
                init_method="dct",
            )
            learned_matrix = denoise_patch_matrix(
                patch_matrix=noisy_matrix,
                D=D_learned,
                patch_size=args.patch_size,
                sparsity=args.sparsity,
                sigma_noise=sigma,
            )
            learned_patches = matrix_to_patches(learned_matrix, args.patch_size)
            learned = reconstruct_from_overlapping_patches(
                learned_patches, noisy.shape, args.patch_size
            )
            learned = np.clip(learned, 0.0, 255.0)

            _save_method_images(
                dirs,
                image_stem,
                sigma,
                clean,
                noisy,
                gaussian,
                dct,
                learned,
            )

            figure_name = dirs["figures"] / f"{image_stem}_sigma{sigma}_comparison.png"
            save_comparison_figure(
                clean=clean,
                noisy=noisy,
                gaussian=gaussian,
                dct=dct,
                ksvd=learned,
                filename=figure_name,
                title=f"{image_path.name} | sigma={sigma}",
            )

            dict_name = dirs["dictionaries"] / f"{image_stem}_sigma{sigma}_dictionary.png"
            save_dictionary_atoms(D_learned, args.patch_size, dict_name)

            for method_name, estimate in [
                ("Noisy", noisy),
                ("Gaussian", gaussian),
                ("DCT+OMP", dct),
                ("K-SVD+OMP", learned),
            ]:
                metric_rows.append(
                    {
                        "image_name": image_path.name,
                        "sigma": sigma,
                        "noise_variance": sigma ** 2,
                        "method": method_name,
                        "mse": mse(clean, estimate),
                        "psnr": psnr(clean, estimate),
                    }
                )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = dirs["tables"] / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    wide_metrics_df = _build_wide_metrics_table(metrics_df)
    wide_metrics_path = dirs["tables"] / "metrics_wide.csv"
    wide_metrics_df.to_csv(wide_metrics_path, index=False)

    summary_df = (
        metrics_df.groupby(["sigma", "noise_variance", "method"], as_index=False)[["mse", "psnr"]]
        .mean()
        .sort_values(["sigma", "method"])
    )
    summary_path = dirs["tables"] / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_metric_vs_noise_variance(metrics_df, dirs["root"])

    avg_psnr_table = summary_df.pivot(index="sigma", columns="method", values="psnr")

    print("\nRun complete.")
    print(f"Selected image names: {', '.join(selected_names)}")
    print(f"Run config: {dirs['root'] / 'config.json'}")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Wide metrics CSV: {wide_metrics_path}")
    print(f"Generated figures: {dirs['figures']}")
    print("\nAverage PSNR table by method and sigma:")
    print(avg_psnr_table.round(3).to_string())

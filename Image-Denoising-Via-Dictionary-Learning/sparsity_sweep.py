from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.baselines import dct_omp_denoise
from src.data_utils import add_gaussian_noise, load_grayscale_image, resolve_image_dir
from src.ksvd import denoise_patch_matrix, ksvd
from src.metrics import mse, psnr
from src.patch_utils import (
    extract_overlapping_patches,
    matrix_to_patches,
    patches_to_matrix,
    reconstruct_from_overlapping_patches,
    sample_training_patches,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a compact sparsity sensitivity sweep for DCT+OMP and K-SVD+OMP."
    )
    parser.add_argument("--data_dir", default="BSDS300/images/train")
    parser.add_argument(
        "--selected_images_file",
        default="results/selected_images.txt",
        help="Text file listing one selected image filename per line.",
    )
    parser.add_argument("--image_size", type=_positive_int, default=256)
    parser.add_argument("--patch_size", type=_positive_int, default=8)
    parser.add_argument("--n_train_patches", type=_positive_int, default=6000)
    parser.add_argument("--n_atoms", type=_positive_int, default=256)
    parser.add_argument("--ksvd_iter", type=_positive_int, default=10)
    parser.add_argument("--sigma", type=_positive_int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sparsities",
        type=_positive_int,
        nargs="+",
        default=[4, 6, 8],
        help="List of sparsity levels T0 to evaluate.",
    )
    parser.add_argument("--output_csv", default="results/tables/sparsity_sweep_sigma15.csv")
    return parser


def _load_selected_image_paths(data_dir: str | Path, selected_images_file: str | Path) -> list[Path]:
    data_dir = resolve_image_dir(data_dir)
    selected_file = Path(selected_images_file)
    if not selected_file.exists():
        raise FileNotFoundError(
            f"Selected image list not found: {selected_file}. "
            "Run main.py first or pass --selected_images_file with an existing file."
        )
    names = [
        line.strip()
        for line in selected_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"No image names found in {selected_file}.")
    paths = [data_dir / name for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing selected images: {missing}")
    return paths


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_paths = _load_selected_image_paths(args.data_dir, args.selected_images_file)
    rows: list[dict[str, float | int | str]] = []

    for image_idx, image_path in enumerate(selected_paths):
        clean = load_grayscale_image(image_path, args.image_size)
        noisy = add_gaussian_noise(clean, args.sigma, seed=args.seed + image_idx * 100 + args.sigma)
        noisy_patches = extract_overlapping_patches(noisy, args.patch_size)
        noisy_matrix = patches_to_matrix(noisy_patches)
        centered_train = noisy_matrix - noisy_matrix.mean(axis=0, keepdims=True)

        for sparsity in args.sparsities:
            dct = dct_omp_denoise(
                noisy_image=noisy,
                patch_size=args.patch_size,
                sparsity=sparsity,
                n_atoms=args.n_atoms,
                sigma_noise=args.sigma,
                patch_matrix=noisy_matrix,
            )

            y_train = sample_training_patches(
                centered_train,
                n_train_patches=args.n_train_patches,
                seed=args.seed + image_idx * 1000 + args.sigma + sparsity,
            )
            dictionary, _ = ksvd(
                Y_train=y_train,
                n_atoms=args.n_atoms,
                sparsity=sparsity,
                n_iter=args.ksvd_iter,
                seed=args.seed + image_idx * 1000 + args.sigma + sparsity,
                init_method="dct",
            )
            learned_matrix = denoise_patch_matrix(
                patch_matrix=noisy_matrix,
                D=dictionary,
                patch_size=args.patch_size,
                sparsity=sparsity,
                sigma_noise=args.sigma,
            )
            learned_patches = matrix_to_patches(learned_matrix, args.patch_size)
            learned = reconstruct_from_overlapping_patches(
                learned_patches, noisy.shape, args.patch_size
            )

            for method_name, estimate in [("DCT+OMP", dct), ("K-SVD+OMP", learned)]:
                rows.append(
                    {
                        "image_name": image_path.name,
                        "sigma": args.sigma,
                        "sparsity": sparsity,
                        "method": method_name,
                        "mse": mse(clean, estimate),
                        "psnr": psnr(clean, estimate),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    summary = (
        df.groupby(["sparsity", "method"], as_index=False)[["mse", "psnr"]]
        .mean()
        .sort_values(["sparsity", "method"])
    )
    print(summary.to_string(index=False))
    print(f"\nSaved sweep results to {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import color, io, transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_DATA_DIR = Path("BSDS300") / "images" / "train"


def resolve_image_dir(data_dir: str | Path) -> Path:
    """Resolve the image directory, including the common parent-folder layout."""
    data_path = Path(data_dir).expanduser()
    if data_path.exists():
        return data_path

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / data_path,
        project_root.parent / data_path,
        project_root.parent / DEFAULT_DATA_DIR,
        project_root / "results" / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            print(
                f"Data directory '{data_path}' was not found; using '{candidate}' instead."
            )
            return candidate

    return data_path


def list_image_files(data_dir: str | Path) -> list[Path]:
    data_path = resolve_image_dir(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_path}. "
            "Expected something like BSDS300/images/train, ../BSDS300/images/train, "
            "or pass --data_dir with the folder that contains the images."
        )
    files = sorted(
        path for path in data_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {data_path}.")
    return files


def _center_crop_or_resize(image: np.ndarray, image_size: int) -> np.ndarray:
    height, width = image.shape
    if height >= image_size and width >= image_size:
        top = (height - image_size) // 2
        left = (width - image_size) // 2
        return image[top : top + image_size, left : left + image_size]

    resized = transform.resize(
        image,
        (image_size, image_size),
        anti_aliasing=True,
        preserve_range=True,
    )
    return resized.astype(np.float64)


def load_grayscale_image(path: str | Path, image_size: int) -> np.ndarray:
    image = io.imread(path)
    if image.ndim == 3:
        image = color.rgb2gray(image)
        image = image * 255.0
    image = image.astype(np.float64)
    if image.ndim != 2:
        raise ValueError(f"Unsupported image shape {image.shape} for file {path}.")
    image = _center_crop_or_resize(image, image_size)
    return np.clip(image, 0.0, 255.0)


def select_random_images(data_dir: str | Path, n_images: int, seed: int) -> list[Path]:
    files = list_image_files(data_dir)
    if n_images > len(files):
        raise ValueError(f"Requested {n_images} images, but only found {len(files)}.")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(files), size=n_images, replace=False)
    return [files[idx] for idx in sorted(indices)]


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    return image.astype(np.float64) + noise

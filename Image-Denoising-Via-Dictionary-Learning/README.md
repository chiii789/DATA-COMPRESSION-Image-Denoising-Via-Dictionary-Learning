# Image Denoising Via Dictionary Learning

This project implements a compact sparse-representation pipeline for grayscale image denoising. The repository includes three methods:

- Gaussian filtering baseline
- Fixed DCT dictionary with OMP
- Learned dictionary with K-SVD and OMP

The default setup uses three randomly selected BSDS300 images, additive Gaussian noise with standard deviations `5`, `10`, `15`, and `25`, overlapping `8 x 8` patches, and `6000` training patches for dictionary learning.

## Project Files

The repository contains the source code needed to run the experiment:

- `src/`: implementation of patch extraction, OMP, K-SVD, metrics, and plotting
- `main.py`: main experiment script
- `sparsity_sweep.py`: small parameter test for the sparsity level

## What The Code Does

- Selects `3` images uniformly at random from the chosen BSDS300 folder
- Converts images to grayscale if needed
- Uses overlapping `8 x 8` patches
- Learns one dictionary per image and per noise level
- Trains each dictionary from `6000` randomly sampled noisy patches from the same noisy image
- Reconstructs the image by averaging overlapping denoised patches
- Reports `MSE` and `PSNR`

One implementation choice not stated explicitly in the assignment is image-size normalization. Images are center-cropped to `256 x 256` when possible; otherwise they are resized to `256 x 256` so every run uses the same working size.

## Dataset Layout

The default command expects the Berkeley dataset images in this layout:

```text
BSDS300/
  images/
    train/
    test/
  iids_train.txt
  iids_test.txt
```

Only image files are used by this project.

If the BSDS300 archive is stored next to this project directory as `../BSDS300-images.tgz`, extract it so that `main.py` can access `BSDS300/images/train` relative to the project root.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run The Experiment

Quick debug run:

```bash
python main.py --fast_mode
```

Assignment-style run:

```bash
python main.py
```

After running the main experiment, the selected image names are saved in `results/selected_images.txt`. To run the sparsity-sensitivity extension experiment at `sigma = 15`, use:

```bash
python sparsity_sweep.py --data_dir BSDS300/images/train --selected_images_file results/selected_images.txt --sigma 15 --sparsities 4 6 8 --output_csv results/tables/sparsity_sweep_sigma15.csv
```

Example with explicit settings:

```bash
python main.py --data_dir BSDS300/images/train --n_images 3 --image_size 256 --patch_size 8 --n_train_patches 6000 --n_atoms 256 --sparsity 6 --ksvd_iter 10 --seed 42 --output_dir results
```

## Reproducibility Notes

- The image selection is random but reproducible through `--seed`
- The default seed is `42`
- The selected filenames are stored in `results/selected_images.txt`
- The full run configuration is stored in `results/config.json`

Because the dictionary is learned separately for each image and each noise level, the learned-model run is noticeably slower than the fixed baselines.

## Outputs

Each run creates:

- `results/images/`
- `results/figures/`
- `results/tables/`
- `results/dictionaries/`

Main files:

- `results/tables/metrics.csv`
- `results/tables/metrics_wide.csv`
- `results/tables/metrics_summary.csv`
- one comparison figure per image and noise level
- average `MSE` vs noise variance plot
- average `PSNR` vs noise variance plot
- one learned-dictionary visualization per image and noise level
- `results/selected_images.txt`

## Limitations

This is a small image-adaptive denoising experiment rather than a large benchmark. The learned dictionary is trained separately for each noisy image, so the results compare denoising methods under a shared setup but do not test cross-image generalization.

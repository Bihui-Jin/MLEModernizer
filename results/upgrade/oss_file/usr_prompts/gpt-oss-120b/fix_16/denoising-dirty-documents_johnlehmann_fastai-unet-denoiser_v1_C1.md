# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.8

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
sklearn-pandas==2.2.0
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.03227

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from pathlib import Path
import zipfile, csv, gc
import torch, torch.nn as nn
from fastai.vision.all import *
from fastai.metrics import rmse  # added import for the RMSE metric
from skimage.color import rgb2gray as _rgb2gray  # retained for fallback if needed
import numpy as np  # new import for efficient array handling



## === cell 1
path_root = Path("/kaggle/input/denoising-dirty-documents")
if not path_root.exists():
    path_root = Path(".")
candidate = path_root / "denoising-dirty-documents"
if (candidate / "train").exists():
    path_root = candidate

path_train = path_root / "train"
path_train_clean = path_root / "train_cleaned"
path_test = path_root / "test"

path_submission = Path("/kaggle/working")
path_submission.mkdir(parents=True, exist_ok=True)




## === cell 2
def get_target(fn: Path) -> Path:
    """Map a noisy training image to its cleaned counterpart."""
    return path_train_clean / fn.name


denoise_block = DataBlock(
    blocks=(ImageBlock, ImageBlock),  # input & target are RGB images
    get_items=get_image_files,
    get_y=get_target,
    splitter=RandomSplitter(seed=42),
    item_tfms=Resize(256),  # less aggressive down‑sampling
    batch_tfms=None,  # removed ImageNet normalization to keep pixel values in original [0,1] range
)

dls = denoise_block.dataloaders(path_train, bs=4, num_workers=0)



## === cell 3
learn = unet_learner(dls, resnet34, loss_func=nn.MSELoss(), metrics=rmse, n_out=3)
learn.model_dir = Path("models")
learn.freeze()
learn.fit_one_cycle(3, 1e-3)  # reduced primary training
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-5, 1e-3))  # reduced fine‑tuning




## === cell 4
def rgb2gray(tensor):
    """Convert a 3‑channel tensor to single‑channel grayscale safely."""
    tensor = tensor.detach().cpu().float()
    if tensor.shape[0] == 1:  # already 1‑channel
        return tensor.clamp(0, 1)
    gray = tensor.mean(dim=0, keepdim=True)
    return gray.clamp(0, 1)


submission_path = path_submission / "submission.csv"
with submission_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    img_files = sorted(path_test.rglob("*.png"), key=lambda p: int(p.stem))
    for img_path in img_files:
        img_id = int(img_path.stem)
        img = PILImage.create(img_path)
        pred = learn.predict(img)[0]  # (C, H, W) tensor
        pred = rgb2gray(pred)  # (1, H, W) grayscale in [0,1]
        pred_np = pred.squeeze(0).numpy()  # (H, W) numpy array
        h, w = pred_np.shape
        rows = np.arange(1, h + 1).reshape(-1, 1)
        cols = np.arange(1, w + 1).reshape(1, -1)
        ids = (
            np.char.add(
                np.char.add(np.char.add(str(img_id) + "_", rows.astype(str)), "_"),
                cols.astype(str),
            )
        ).ravel()
        values = pred_np.ravel().astype(float).tolist()
        writer.writerows(zip(ids, values))

print(f"Submission written to {submission_path}")

# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Detect the presence of ink from 3d x-ray scans of detached fragments of ancient papyrus scrolls.

## Metric
We evaluate how well your output image matches our reference image using a modified version of the [Sørensen--Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), where instead of using the F1 score, we are using the F0.5 score. The F0.5 score is given by:

$$
\frac{\left(1+\beta^2\right) p r}{\beta^2 p+r} \text { where } p=\frac{t p}{t p+f p}, r=\frac{t p}{t p+f n}, \beta=0.5
$$

The F0.5 score weights precision higher than recall, which improves the ability to form coherent characters out of detected ink areas.

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the output should be binary, with 0 indicating "no ink" and 1 indicating "ink".

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

Your output should be a single file, **submission.csv**, with this run-length encoded information. This should have a header with two columns, `Id` and `Predicted`, and with one row for every directory under **test/**. For example:

```
Id,Predicted
a,1 1 5 1 etc.
b,10 20 etc.
```

For a real-world example of what these files look like, see `inklabels_rce.csv` in the data directories, which have been generated with [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0).


## Data
- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray [surface volume](https://scrollprize.org/tutorial1#3-surface-volumes). Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us `width * height * 65` number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). This is the same format as you should make your submission in.
- **train/[fragment_id]/ir.png** --- the infrared photo on which the binary mask is based.
- **sample_submission.csv**, an example of a submission file in the correct format. You need to output the following file in the home directory: **submission.csv**.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        input/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        working/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
```

-> data/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> input/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.1280975430645314

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.21672) has done: 'I fix the reason you’re not getting a Kaggle score by ensuring the notebook actually sees the dataset directory (your current `COMPETITION_DATA_DIR` selection can point to a non-existent path and then `iterdir()` crashes). Then I make one minimal, metric-relevant improvement: choose the binarization threshold using the validation fragment by maximizing an F0.5-like score on masked pixels, instead of using a fixed 0.4 (this preserves your model/training logic and only changes post-processing toward the competition metric). I also make sure the submission rows exactly match `sample_submission.csv` Id order (prevents accidental Id mismatch in hidden test). All other core logic (UNet, Dice loss, training loop, sliding-window inference) stays unchanged.'
- What this solution (achieved 0.21684) has done: 'I make the notebook yield a valid Kaggle submission reliably by fixing the cell numbering (your script starts at cell 0, which breaks the required cell parser format) while keeping all paths and core modeling logic unchanged. Then, to move the score upward toward your target (higher-is-better) without altering the model/training approach, I make one metric-aligned, minimal post-processing change: tune the binarization threshold to maximize validation F0.5 (on masked pixels) rather than matching the target score value directly. Finally, I ensure the submission `Id` order exactly matches `sample_submission.csv` and that RLE encoding is compliant (sorted runs, 1-indexed, no duplicates), which prevents “no score” outcomes due to formatting/alignment issues.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"

_CANDIDATES = [
    INPUT_DIR / "vesuvius-challenge-ink-detection",
    INPUT_DIR / "vesuvius-challenge-ink-detection" / "vesuvius-challenge-ink-detection",
    KAGGLE_DIR / "data" / "vesuvius-challenge-ink-detection",
    KAGGLE_DIR / "data",
    INPUT_DIR,
]
COMPETITION_DATA_DIR = None
for c in _CANDIDATES:
    if (c / "train").exists() and (c / "test").exists():
        COMPETITION_DATA_DIR = c
        break
if COMPETITION_DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find dataset directory containing 'train/' and 'test/'. "
        f"Tried: {[str(x) for x in _CANDIDATES]}"
    )

print("COMPETITION_DATA_DIR:", COMPETITION_DATA_DIR)
print("Exists:", COMPETITION_DATA_DIR.exists())
print("Contents sample:", sorted([p.name for p in COMPETITION_DATA_DIR.iterdir()])[:10])



## === cell 1
import importlib


def _try_import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


_monai = _try_import("monai")
print(
    "monai importable:", _monai is not None, "(will use torch-only pipeline if False)"
)



## === cell 2
from typing import Tuple

TRAIN_DATA_CSV_PATH = Path("./train.csv")
TEST_DATA_CSV_PATH = Path("./test.csv")

DOWNSAMPLING = 1.0
NUM_Z_SLICES = 8

ACCELERATOR = "gpu"
BATCH_SIZE = 1
DEVICES = 1
DROPOUT = 0.0
ETA_MIN = 1e-6
FAST_DEV_RUN = False
LEARNING_RATE = 3e-4
LOSS = "Dice"
MODEL_NAME = "UNet"
MAX_EPOCHS = 5
NUM_WORKERS = 0
NUM_SAMPLES = 128
OPTIMIZER = "AdamW"
OVERFIT_BATCHES = 0
PATCH_SIZE = (512, 512)
PRECISION = 16
SCHEDULER = "CosineAnnealingLR"
SEED = 2023
SW_BATCH_SIZE = 16
VAL_FRAGMENT_ID = 3  # if fragment 3 not present, we will fallback to fragment 2 later
WEIGHT_DECAY = 1e-6

THRESHOLD = 0.4

TARGET_SCORE = 0.1280975430645314



## === cell 3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm




## === cell 4
def lovely_array(a):
    a = np.asarray(a)
    return (
        f"shape={a.shape}, dtype={a.dtype}, min={a.min() if a.size else 'NA'}, "
        f"max={a.max() if a.size else 'NA'}"
    )




## === cell 5
def create_df_from_mask_paths(stage: str, downsampling: float):
    mask_paths = sorted((COMPETITION_DATA_DIR / stage).glob("*/mask.png"))
    df = pd.DataFrame({"mask_png": [str(p) for p in mask_paths]})

    df["stage"] = stage
    df["fragment_id"] = df["mask_png"].str.split("/").str[-2]

    base = Path("working_npy") / f"{stage}_{downsampling}"
    df["mask_npy"] = df["fragment_id"].apply(lambda fid: str(base / fid / "mask.npy"))
    df["volumes_dir"] = df["mask_png"].str.replace(
        "mask.png", "surface_volume", regex=False
    )
    df["volume_npy"] = df["fragment_id"].apply(
        lambda fid: str(base / fid / "volume.npy")
    )

    if stage == "train":
        df["label_png"] = df["mask_png"].str.replace(
            "mask.png", "inklabels.png", regex=False
        )
        df["label_npy"] = df["fragment_id"].apply(
            lambda fid: str(base / fid / "inklabels.npy")
        )

    return df


train_df = create_df_from_mask_paths("train", DOWNSAMPLING)
test_df = create_df_from_mask_paths("test", DOWNSAMPLING)

available_train_ids = sorted(train_df["fragment_id"].unique().tolist())
if str(VAL_FRAGMENT_ID) not in available_train_ids:
    VAL_FRAGMENT_ID = (
        int(available_train_ids[-1]) if available_train_ids else VAL_FRAGMENT_ID
    )

print("Train fragments:", available_train_ids, "VAL_FRAGMENT_ID:", VAL_FRAGMENT_ID)
print("Num train rows:", len(train_df), "Num test rows:", len(test_df))

train_df.to_csv(TRAIN_DATA_CSV_PATH, index=False)
test_df.to_csv(TEST_DATA_CSV_PATH, index=False)




## === cell 6
def load_image(path):
    return Image.open(path)


def resize_image(image, downsampling: float):
    if downsampling == 1.0:
        return image
    size = int(image.size[0] * downsampling), int(image.size[1] * downsampling)
    return image.resize(size, resample=Image.Resampling.BILINEAR)


def load_and_resize_image(path, downsampling: float):
    image = load_image(path)
    return resize_image(image, downsampling)


def load_label_npy(path, downsampling: float):
    label = load_and_resize_image(path, downsampling)
    return (np.array(label) > 0).astype(np.uint8)


def load_mask_npy(path, downsampling: float):
    mask = load_and_resize_image(path, downsampling).convert("1")
    return (np.array(mask) > 0).astype(np.uint8)


def load_z_slice_npy(path, downsampling: float):
    z_slice = load_and_resize_image(path, downsampling)
    return np.array(z_slice, dtype=np.float32) / 65535.0


def load_volume_npy(volumes_dir, num_z_slices: int, downsampling: float):
    mid = 65 // 2
    start = mid - num_z_slices // 2
    end = start + num_z_slices
    z_slices_paths = sorted(Path(volumes_dir).glob("*.tif"))[start:end]
    if len(z_slices_paths) != num_z_slices:
        raise RuntimeError(
            f"Expected {num_z_slices} slices, got {len(z_slices_paths)} in {volumes_dir}"
        )

    z_slices = [load_z_slice_npy(p, downsampling) for p in z_slices_paths]
    volume = np.stack(z_slices, axis=0).astype(np.float32)  # (C,H,W)
    return volume


def save_data_as_npy(df: pd.DataFrame, downsampling: float, train: bool = True):
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc=f"Creating npy ({'train' if train else 'test'})",
    ):
        mask_npy_path = Path(row.mask_npy)
        vol_npy_path = Path(row.volume_npy)
        mask_npy_path.parent.mkdir(exist_ok=True, parents=True)
        vol_npy_path.parent.mkdir(exist_ok=True, parents=True)

        label_ok = True
        if train:
            label_npy_path = Path(row.label_npy)
            label_npy_path.parent.mkdir(exist_ok=True, parents=True)
            label_ok = label_npy_path.exists()

        if (
            mask_npy_path.exists()
            and vol_npy_path.exists()
            and ((not train) or label_ok)
        ):
            continue

        mask_npy = load_mask_npy(row.mask_png, downsampling)
        volume_npy = load_volume_npy(row.volumes_dir, NUM_Z_SLICES, downsampling)

        np.save(mask_npy_path, mask_npy)
        np.save(vol_npy_path, volume_npy)

        if train:
            label_npy = load_label_npy(row.label_png, downsampling)
            np.save(Path(row.label_npy), label_npy)


save_data_as_npy(train_df, DOWNSAMPLING, train=True)
save_data_as_npy(test_df, DOWNSAMPLING, train=False)




## === cell 7
def set_global_determinism(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VesuviusPatchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        patch_size: Tuple[int, int],
        num_samples: int,
        train: bool,
        has_labels: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.train = train
        self.has_labels = has_labels
        self._cache = {}  # row_idx -> (vol, mask, lab)

    def __len__(self):
        return len(self.df) * (self.num_samples if self.train else 1)

    def _load_row(self, row_idx: int):
        cached = self._cache.get(row_idx, None)
        if cached is not None:
            return cached

        row = self.df.iloc[row_idx]

        vol = np.load(row["volume_npy"], mmap_mode="r")  # (C,H,W) float32
        mask = np.load(row["mask_npy"], mmap_mode="r")  # (H,W) uint8

        lab = None
        if self.has_labels:
            lab = np.load(row["label_npy"], mmap_mode="r")  # (H,W) uint8

        self._cache[row_idx] = (vol, mask, lab)
        return vol, mask, lab

    def _random_crop_center_from_mask(self, mask: np.ndarray):
        h, w = mask.shape
        ph, pw = self.patch_size
        ph = min(ph, h)
        pw = min(pw, w)

        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            i = np.random.randint(0, len(ys))
            cy, cx = int(ys[i]), int(xs[i])
        else:
            cy, cx = np.random.randint(0, h), np.random.randint(0, w)

        y0 = np.clip(cy - ph // 2, 0, h - ph)
        x0 = np.clip(cx - pw // 2, 0, w - pw)
        return int(y0), int(x0), ph, pw

    def __getitem__(self, idx: int):
        row_idx = idx // self.num_samples if self.train else idx
        vol, mask, lab = self._load_row(row_idx)

        if self.train:
            y0, x0, ph, pw = self._random_crop_center_from_mask(mask)
            vol = np.asarray(vol[:, y0 : y0 + ph, x0 : x0 + pw], dtype=np.float32)
            mask = np.asarray(mask[y0 : y0 + ph, x0 : x0 + pw], dtype=np.uint8)
            if lab is not None:
                lab = np.asarray(lab[y0 : y0 + ph, x0 : x0 + pw], dtype=np.uint8)

            if np.random.rand() < 0.5:
                vol = vol[:, ::-1, :]
                mask = mask[::-1, :]
                if lab is not None:
                    lab = lab[::-1, :]
            if np.random.rand() < 0.5:
                vol = vol[:, :, ::-1]
                mask = mask[:, ::-1]
                if lab is not None:
                    lab = lab[:, ::-1]

            vol = np.ascontiguousarray(vol)
            mask = np.ascontiguousarray(mask)
            if lab is not None:
                lab = np.ascontiguousarray(lab)
        else:
            vol = np.asarray(vol, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.uint8)
            if lab is not None:
                lab = np.asarray(lab, dtype=np.uint8)

            vol = np.ascontiguousarray(vol)
            mask = np.ascontiguousarray(mask)
            if lab is not None:
                lab = np.ascontiguousarray(lab)

        vol_t = torch.from_numpy(vol).float()  # (C,H,W)
        mask_t = torch.from_numpy(mask[None, ...]).float()  # (1,H,W)

        out = {"volume_npy": vol_t, "mask_npy": mask_t}
        if lab is not None:
            out["label_npy"] = torch.from_numpy(lab[None, ...]).float()
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]

        self.enc1 = DoubleConv(in_channels, chs[0], dropout=dropout)
        self.enc2 = DoubleConv(chs[0], chs[1], dropout=dropout)
        self.enc3 = DoubleConv(chs[1], chs[2], dropout=dropout)
        self.enc4 = DoubleConv(chs[2], chs[3], dropout=dropout)
        self.bott = DoubleConv(chs[3], chs[4], dropout=dropout)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec4 = DoubleConv(chs[4], chs[3], dropout=dropout)
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = DoubleConv(chs[3], chs[2], dropout=dropout)
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = DoubleConv(chs[2], chs[1], dropout=dropout)
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = DoubleConv(chs[1], chs[0], dropout=dropout)

        self.out = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bott(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


def masked_dice_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
):
    probs = torch.sigmoid(logits)
    probs = probs * mask
    targets = targets * mask
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    denom = torch.sum(probs, dims) + torch.sum(targets, dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()




## === cell 8
class VesuvisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_csv_path: str,
        num_workers: int,
        num_samples: int,
        patch_size: Tuple[int, int],
        val_fragment_id: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.df = pd.read_csv(data_csv_path)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_val_df = self.df[self.df.stage == "train"].reset_index(drop=True)
            if len(train_val_df) == 0:
                raise RuntimeError(
                    "No train rows found in data_csv_path; expected stage=='train' rows."
                )

            train_df_ = train_val_df[
                train_val_df.fragment_id != str(self.hparams.val_fragment_id)
            ].reset_index(drop=True)
            val_df_ = train_val_df[
                train_val_df.fragment_id == str(self.hparams.val_fragment_id)
            ].reset_index(drop=True)

            if len(val_df_) == 0 and len(train_val_df) > 0:
                fallback_val_id = sorted(train_val_df.fragment_id.unique().tolist())[-1]
                val_df_ = train_val_df[
                    train_val_df.fragment_id == fallback_val_id
                ].reset_index(drop=True)
                train_df_ = train_val_df[
                    train_val_df.fragment_id != fallback_val_id
                ].reset_index(drop=True)
                print("VAL_FRAGMENT_ID not present; fallback_val_id:", fallback_val_id)

            self.train_dataset = VesuviusPatchDataset(
                train_df_,
                self.hparams.patch_size,
                self.hparams.num_samples,
                train=True,
                has_labels=True,
            )
            self.val_dataset = VesuviusPatchDataset(
                val_df_,
                self.hparams.patch_size,
                self.hparams.num_samples,
                train=False,
                has_labels=True,
            )

            print(f"# train: {len(self.train_dataset)}")
            print(f"# val: {len(self.val_dataset)}")

        if stage == "predict" or stage is None:
            if "stage" in self.df.columns:
                if (
                    Path(self.hparams.data_csv_path).name
                    == Path(TEST_DATA_CSV_PATH).name
                ):
                    predict_df = self.df[self.df.stage == "test"].reset_index(drop=True)
                else:
                    predict_df = self.df.reset_index(drop=True)
            else:
                predict_df = self.df.reset_index(drop=True)

            if len(predict_df) == 0:
                raise RuntimeError("No rows found in data_csv_path for predict.")
            self.predict_dataset = VesuviusPatchDataset(
                predict_df,
                self.hparams.patch_size,
                self.hparams.num_samples,
                train=False,
                has_labels=False,
            )
            print(f"# predict: {len(self.predict_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.hparams.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.hparams.num_workers > 0),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.hparams.num_workers > 0),
        )




## === cell 9
def visualize_dataloaders(dataloaders, train=True, max_batches=1):
    for stage, dataloader in dataloaders.items():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            volumes = batch["volume_npy"]
            masks = batch["mask_npy"]
            labels = batch["label_npy"] if train and "label_npy" in batch else masks

            for volume, mask, label in zip(volumes, masks, labels):
                vol_np = volume.detach().cpu().numpy()
                c = vol_np.shape[0] // 2
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                plt.suptitle(f"stage: {stage}, batch: {batch_idx}")
                axes[0].imshow(vol_np[c], cmap="gray")
                axes[0].set_title(f"volume[c={c}] {vol_np.shape}")
                axes[1].imshow(mask[0].detach().cpu().numpy(), cmap="gray")
                axes[1].set_title("mask")
                axes[2].imshow(label[0].detach().cpu().numpy(), cmap="gray")
                axes[2].set_title("label")
                for ax in axes:
                    ax.axis("off")
                plt.show()


try:
    dm_vis = VesuvisDataModule(
        batch_size=BATCH_SIZE,
        data_csv_path=str(TRAIN_DATA_CSV_PATH),
        num_workers=NUM_WORKERS,
        num_samples=2,
        patch_size=PATCH_SIZE,
        val_fragment_id=VAL_FRAGMENT_ID,
    )
    dm_vis.setup(stage="fit")
    visualize_dataloaders(
        {"train": dm_vis.train_dataloader(), "val": dm_vis.val_dataloader()},
        train=True,
        max_batches=1,
    )
except Exception as e:
    print("Visualization skipped due to:", repr(e))




## === cell 10
def _sliding_window_predict_proba(
    model: nn.Module,
    volume: torch.Tensor,  # (1,C,H,W)
    patch_size: Tuple[int, int],
    sw_batch_size: int = 8,
    device: torch.device | None = None,
) -> torch.Tensor:
    assert (
        volume.ndim == 4 and volume.shape[0] == 1
    ), f"Expected (1,C,H,W), got {tuple(volume.shape)}"
    ph, pw = patch_size
    _, _, H, W = volume.shape
    ph = min(ph, H)
    pw = min(pw, W)
    stride_h = max(1, ph // 2)
    stride_w = max(1, pw // 2)

    if device is None:
        device = next(model.parameters()).device

    vol = volume.to(device, non_blocking=True)

    pad_h = (ph - (H % stride_h)) % stride_h if H > ph else (ph - H)
    pad_w = (pw - (W % stride_w)) % stride_w if W > pw else (pw - W)
    pad_h = int(max(0, pad_h))
    pad_w = int(max(0, pad_w))
    vol = F.pad(vol, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = vol.shape

    ys = list(range(0, Hp - ph + 1, stride_h))
    xs = list(range(0, Wp - pw + 1, stride_w))
    if ys[-1] != Hp - ph:
        ys.append(Hp - ph)
    if xs[-1] != Wp - pw:
        xs.append(Wp - pw)

    out = torch.zeros((1, 1, Hp, Wp), device=device, dtype=torch.float32)
    wgt = torch.zeros((1, 1, Hp, Wp), device=device, dtype=torch.float32)

    yy = torch.tensor(ys, device=device, dtype=torch.long)
    xx = torch.tensor(xs, device=device, dtype=torch.long)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)  # (N,2)
    N = coords.shape[0]

    oy = torch.arange(ph, device=device, dtype=torch.long)
    ox = torch.arange(pw, device=device, dtype=torch.long)
    off_y, off_x = torch.meshgrid(oy, ox, indexing="ij")
    off_y = off_y.reshape(1, -1)  # (1,ph*pw)
    off_x = off_x.reshape(1, -1)

    model.eval()
    with torch.no_grad():
        for s in range(0, N, sw_batch_size):
            e = min(N, s + sw_batch_size)
            c = coords[s:e]  # (B,2)
            patches = [
                vol[:, :, int(y) : int(y) + ph, int(x) : int(x) + pw] for y, x in c
            ]
            batch = torch.cat(patches, dim=0)  # (B,C,ph,pw)

            probs = torch.sigmoid(model(batch)).float()  # (B,1,ph,pw)
            B = probs.shape[0]
            probs_flat = probs.reshape(B, -1)  # (B,ph*pw)

            base_y = c[:, 0].reshape(B, 1)
            base_x = c[:, 1].reshape(B, 1)
            iy = base_y + off_y  # (B,ph*pw)
            ix = base_x + off_x
            flat_idx = (iy * Wp + ix).reshape(-1)  # (B*ph*pw)

            out_flat = out.view(-1)
            wgt_flat = wgt.view(-1)
            out_flat.scatter_add_(0, flat_idx, probs_flat.reshape(-1))
            wgt_flat.scatter_add_(
                0,
                flat_idx,
                torch.ones_like(probs_flat, dtype=torch.float32).reshape(-1),
            )

    out = out / torch.clamp_min(wgt, 1.0)
    out = out[:, :, :H, :W]
    return out


class VesuvisModule(pl.LightningModule):
    def __init__(
        self,
        dropout: float,
        eta_min: float,
        learning_rate: float,
        loss: str,
        model_name: str,
        max_epochs: int,
        num_z_slices: int,
        optimizer: str,
        patch_size: Tuple[int, int],
        scheduler: str,
        sw_batch_size: int,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()

    def _init_model(self):
        if self.hparams.model_name == "UNet":
            return UNet2D(
                in_channels=self.hparams.num_z_slices,
                out_channels=1,
                base=16,
                dropout=self.hparams.dropout,
            )
        raise ValueError(f"{self.hparams.model_name} is not implemented")

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"{self.hparams.optimizer} is not implemented")

        if self.hparams.scheduler == "CosineAnnealingLR":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.hparams.max_epochs, eta_min=self.hparams.eta_min
            )
        else:
            raise ValueError(f"{self.hparams.scheduler} is not implemented")

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        logits = self(batch["volume_npy"])
        loss = masked_dice_loss_with_logits(
            logits, batch["label_npy"], batch["mask_npy"]
        )
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=logits.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        probs_full = _sliding_window_predict_proba(
            self.model,
            batch["volume_npy"],
            patch_size=self.hparams.patch_size,
            sw_batch_size=self.hparams.sw_batch_size,
            device=self.device,
        )  # (1,1,H,W)
        probs_full = probs_full.clamp(1e-6, 1 - 1e-6)
        logits_full = torch.log(probs_full / (1.0 - probs_full))
        loss = masked_dice_loss_with_logits(
            logits_full,
            batch["label_npy"].to(self.device),
            batch["mask_npy"].to(self.device),
        )
        self.log(
            "val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1
        )
        return loss

    def predict_step(self, batch, batch_idx):
        probs_full = _sliding_window_predict_proba(
            self.model,
            batch["volume_npy"],
            patch_size=self.hparams.patch_size,
            sw_batch_size=self.hparams.sw_batch_size,
            device=self.device,
        )  # (1,1,H,W)
        probs = probs_full[0, 0]  # (H,W)
        return probs.detach().cpu()




## === cell 11
def train(
    accelerator=ACCELERATOR,
    batch_size=BATCH_SIZE,
    data_csv_path=str(TRAIN_DATA_CSV_PATH),
    devices=DEVICES,
    dropout=DROPOUT,
    eta_min=ETA_MIN,
    fast_dev_run=FAST_DEV_RUN,
    learning_rate=LEARNING_RATE,
    loss=LOSS,
    model_name=MODEL_NAME,
    max_epochs=MAX_EPOCHS,
    num_workers=NUM_WORKERS,
    num_samples=NUM_SAMPLES,
    num_z_slices=NUM_Z_SLICES,
    optimizer=OPTIMIZER,
    overfit_batches=OVERFIT_BATCHES,
    patch_size=PATCH_SIZE,
    precision=PRECISION,
    scheduler=SCHEDULER,
    seed=SEED,
    sw_batch_size=SW_BATCH_SIZE,
    val_fragment_id=VAL_FRAGMENT_ID,
    weight_decay=WEIGHT_DECAY,
):
    set_global_determinism(seed)
    pl.seed_everything(seed, workers=True)

    data_module = VesuvisDataModule(
        batch_size=batch_size,
        data_csv_path=data_csv_path,
        num_workers=num_workers,
        num_samples=num_samples,
        patch_size=patch_size,
        val_fragment_id=val_fragment_id,
    )

    module = VesuvisModule(
        dropout=dropout,
        eta_min=eta_min,
        learning_rate=learning_rate,
        loss=loss,
        model_name=model_name,
        max_epochs=max_epochs,
        num_z_slices=num_z_slices,
        optimizer=optimizer,
        patch_size=patch_size,
        scheduler=scheduler,
        sw_batch_size=sw_batch_size,
        weight_decay=weight_decay,
    )

    precision_setting = (
        "16-mixed" if (torch.cuda.is_available() and precision == 16) else 32
    )

    trainer = pl.Trainer(
        accelerator=accelerator if torch.cuda.is_available() else "cpu",
        benchmark=False,
        check_val_every_n_epoch=max(1, max_epochs // 5),
        devices=devices if torch.cuda.is_available() else 1,
        fast_dev_run=fast_dev_run,
        logger=pl.loggers.CSVLogger(save_dir="logs/"),
        log_every_n_steps=1,
        max_epochs=max_epochs,
        overfit_batches=overfit_batches,
        precision=precision_setting,
        strategy="ddp" if (torch.cuda.is_available() and devices > 1) else "auto",
        enable_model_summary=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, datamodule=data_module)
    return module, trainer


module, trainer = train()



## === cell 12
try:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    cols = [
        c for c in ["epoch", "train_loss_epoch", "val_loss"] if c in metrics.columns
    ]
    if "epoch" in cols:
        plot_df = metrics[cols].groupby("epoch").mean(numeric_only=True)
        sns.relplot(
            data=plot_df.reset_index(),
            x="epoch",
            y=[c for c in plot_df.columns if c != "epoch"],
            kind="line",
            height=5,
            aspect=1.5,
        )
        plt.grid()
        plt.show()
except Exception as e:
    print("Metrics plot skipped due to:", repr(e))




## === cell 13
def predict(
    module,
    accelerator=ACCELERATOR,
    batch_size=BATCH_SIZE,
    data_csv_path=str(TEST_DATA_CSV_PATH),
    devices=DEVICES,
    num_workers=NUM_WORKERS,
    num_samples=NUM_SAMPLES,
    patch_size=PATCH_SIZE,
    precision=PRECISION,
    seed=SEED,
    val_fragment_id=VAL_FRAGMENT_ID,
):
    set_global_determinism(seed)
    pl.seed_everything(seed, workers=True)

    data_module = VesuvisDataModule(
        batch_size=1,
        data_csv_path=data_csv_path,
        num_workers=num_workers,
        num_samples=num_samples,
        patch_size=patch_size,
        val_fragment_id=val_fragment_id,
    )

    precision_setting = (
        "16-mixed" if (torch.cuda.is_available() and precision == 16) else 32
    )

    trainer = pl.Trainer(
        accelerator=accelerator if torch.cuda.is_available() else "cpu",
        devices=devices if torch.cuda.is_available() else 1,
        precision=precision_setting,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    predictions = trainer.predict(module, datamodule=data_module)
    return predictions


predictions = predict(module, data_csv_path=str(TEST_DATA_CSV_PATH))
print("Num predictions:", len(predictions))




## === cell 14
def rle(img: np.ndarray) -> str:
    """
    img: 2D numpy array, 1 - mask, 0 - background
    Returns run length as space-delimited string, 1-indexed.
    """
    img = (img > 0).astype(np.uint8)
    pixels = img.ravel(order="C")
    if pixels.size == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # starts (1-indexed)
    runs[1::2] -= runs[::2]  # lengths
    return " ".join(str(int(x)) for x in runs)


def plot_image(image, title):
    fig = plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()




## === cell 15
def fbeta_from_counts(tp, fp, fn, beta=0.5, eps=1e-9):
    beta2 = beta * beta
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    return (1 + beta2) * p * r / (beta2 * p + r + eps)


def tune_threshold_on_val_max_fbeta(module, val_fragment_id: int, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37, dtype=np.float32)

    val_df = train_df[train_df.fragment_id == str(val_fragment_id)].reset_index(
        drop=True
    )
    if len(val_df) == 0:
        print("No val fragment rows found; keeping default THRESHOLD:", THRESHOLD)
        return float(THRESHOLD)

    dm_val = VesuvisDataModule(
        batch_size=1,
        data_csv_path=str(TRAIN_DATA_CSV_PATH),
        num_workers=NUM_WORKERS,
        num_samples=1,
        patch_size=PATCH_SIZE,
        val_fragment_id=val_fragment_id,
    )
    dm_val.setup(stage="fit")

    trainer = pl.Trainer(
        accelerator=ACCELERATOR if torch.cuda.is_available() else "cpu",
        devices=DEVICES if torch.cuda.is_available() else 1,
        precision="16-mixed" if (torch.cuda.is_available() and PRECISION == 16) else 32,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    val_probs_list = trainer.predict(module, dataloaders=dm_val.val_dataloader())
    if len(val_probs_list) == 0:
        print("Val predict returned no batches; keeping default THRESHOLD:", THRESHOLD)
        return float(THRESHOLD)

    probs = val_probs_list[0].detach().cpu().numpy()

    row = val_df.iloc[0]
    label_np = (np.array(load_image(row.label_png)) > 0).astype(np.uint8)
    mask_np = (np.array(load_image(row.mask_png).convert("1")) > 0).astype(np.uint8)

    if probs.shape != label_np.shape:
        pr_img = Image.fromarray(
            (probs * 255.0).clip(0, 255).astype(np.uint8), mode="L"
        )
        pr_img = pr_img.resize(
            (label_np.shape[1], label_np.shape[0]),
            resample=Image.Resampling.BILINEAR,
        )
        probs = np.array(pr_img).astype(np.float32) / 255.0

    valid = mask_np > 0
    y = label_np[valid].astype(np.uint8)

    best_t = float(THRESHOLD)
    best_score = -1.0

    for t in thresholds:
        pred = (probs[valid] > float(t)).astype(np.uint8)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        s = float(fbeta_from_counts(tp, fp, fn, beta=0.5))
        if s > best_score:
            best_score = s
            best_t = float(t)

    print(
        f"Tuned THRESHOLD (maximize val F0.5) on fragment {val_fragment_id}: "
        f"{best_t:.3f} (val F0.5={best_score:.4f})"
    )
    return best_t


THRESHOLD = tune_threshold_on_val_max_fbeta(module, VAL_FRAGMENT_ID)



## === cell 16
sample_sub_path = COMPETITION_DATA_DIR / "sample_submission.csv"
if not sample_sub_path.exists():
    sample_sub_path = INPUT_DIR / "sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

discovered_test_ids = sorted(test_df["fragment_id"].astype(str).unique().tolist())
sample_ids = sample_sub["Id"].astype(str).tolist()

submission_ids = [i for i in sample_ids if i in set(discovered_test_ids)]
if len(submission_ids) == 0:
    submission_ids = discovered_test_ids

print("discovered_test_ids:", discovered_test_ids)
print("sample_ids:", sample_ids)
print("final submission_ids:", submission_ids)

test_df_local = test_df.copy()
test_df_local["Id"] = test_df_local["fragment_id"].astype(str)
test_df_local = test_df_local.drop_duplicates(subset=["Id"], keep="first").reset_index(
    drop=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
module = module.to(device)
module.eval()


def predict_single_fragment_proba(fragment_row: pd.Series) -> np.ndarray:
    vol = np.load(fragment_row["volume_npy"], mmap_mode="r")  # (C,H,W)
    vol_t = torch.from_numpy(np.ascontiguousarray(np.asarray(vol, dtype=np.float32)))[
        None, ...
    ].to(device)
    with torch.no_grad():
        probs = (
            _sliding_window_predict_proba(
                module.model,
                vol_t,
                patch_size=PATCH_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                device=device,
            )[0, 0]
            .detach()
            .cpu()
            .numpy()
        )
    return probs


pred_by_id = {}
for frag_id in submission_ids:
    rows = test_df_local.loc[test_df_local["Id"] == frag_id]
    if len(rows) == 0:
        raise RuntimeError(f"Id '{frag_id}' not found in discovered test_df.")
    row = rows.iloc[0]
    pred_by_id[frag_id] = predict_single_fragment_proba(row)

predictions_rle = []
for frag_id in submission_ids:
    row = test_df_local.loc[test_df_local["Id"] == frag_id].iloc[0]
    pred = np.asarray(pred_by_id[frag_id])
    if pred.ndim != 2:
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            raise RuntimeError(
                f"Unexpected prediction shape for {frag_id}: {pred.shape}"
            )

    mask_img = load_image(row.mask_png).convert("1")
    mask_np = (np.array(mask_img) > 0).astype(np.uint8)

    pred_img = Image.fromarray((pred * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    pred_resized = pred_img.resize(mask_img.size, resample=Image.Resampling.BILINEAR)
    pred_resized = np.array(pred_resized).astype(np.float32) / 255.0

    pred_masked = pred_resized * mask_np
    pred_bin = (pred_masked > float(THRESHOLD)).astype(np.uint8)
    predictions_rle.append(rle(pred_bin))

submission_out = pd.DataFrame({"Id": submission_ids, "Predicted": predictions_rle})
submission_out = submission_out[["Id", "Predicted"]].reset_index(drop=True)
submission_out.to_csv("submission.csv", index=False)

print("Using THRESHOLD:", THRESHOLD)
print("Wrote submission.csv with shape:", submission_out.shape)
print(submission_out.head())
print(
    "submission.csv exists:",
    Path("submission.csv").exists(),
    "size:",
    Path("submission.csv").stat().st_size,
)

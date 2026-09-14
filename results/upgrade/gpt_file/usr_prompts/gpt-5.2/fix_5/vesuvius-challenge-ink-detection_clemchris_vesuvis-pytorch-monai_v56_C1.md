# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.108536956034582

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")



## === cell 1
from typing import Tuple

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns



## === cell 2
KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"

COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"
PREPARED_DATA_DIR = INPUT_DIR / "vesuvis-data-preparation"

TRAIN_DATA_CSV_PATH = PREPARED_DATA_DIR / "data_0.5.csv"
TEST_DATA_CSV_PATH = "test.csv"

try:
    DOWNSAMPLING = float(TRAIN_DATA_CSV_PATH.name.split("_")[-1].replace(".csv", ""))
except Exception:
    DOWNSAMPLING = 0.5

Z_START = 27  # First slice in the z direction to use
Z_DIM = 16  # Number of slices in the z direction

ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
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
PRECISION = 16  # Lightning will effectively use fp32 on CPU
SCHEDULER = "CosineAnnealingLR"
SEED = 2023
SW_BATCH_SIZE = 16

VAL_FRAGMENT_ID = 2
WEIGHT_DECAY = 1e-6




## === cell 3
def visualize_dataloaders(dataloaders, train=True, max_batches=1):
    for stage, dataloader in dataloaders.items():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            volumes = batch["volume_npy"]
            masks = batch["mask_npy"]
            labels = batch["label_npy"] if train and "label_npy" in batch else masks

            for i in range(min(1, volumes.shape[0])):
                vol = volumes[i].detach().cpu().numpy()  # (Z,H,W)
                mid = vol.shape[0] // 2
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(vol[mid], cmap="gray")
                axes[0].set_title(f"volume z={mid}")
                axes[1].imshow(masks[i, 0].detach().cpu().numpy(), cmap="gray")
                axes[1].set_title("mask")
                axes[2].imshow(labels[i, 0].detach().cpu().numpy(), cmap="gray")
                axes[2].set_title("label")
                for ax in axes:
                    ax.axis("off")
                plt.suptitle(f"stage={stage}, batch={batch_idx}")
                plt.show()




## === cell 4
def create_df_from_mask_paths(mask_paths, train=True):
    df = pd.DataFrame({"mask_png": [str(p) for p in mask_paths]})
    df["stage"] = df["mask_png"].str.split("/").str[-3]
    df["fragmet_id"] = df["mask_png"].str.split("/").str[-2].astype(str)

    out_base = str(KAGGLE_DIR / "working")
    df["mask_npy"] = df["mask_png"].str.replace(str(INPUT_DIR), out_base, regex=False)
    df["mask_npy"] = df["mask_npy"].str.replace(
        "/test/", f"/test_{DOWNSAMPLING}/", regex=False
    )
    df["mask_npy"] = df["mask_npy"].str.replace(
        "/train/", f"/train_{DOWNSAMPLING}/", regex=False
    )
    df["mask_npy"] = df["mask_npy"].str.replace("mask.png", "mask.npy", regex=False)

    if train:
        df["label_png"] = df["mask_png"].str.replace(
            "mask.png", "inklabels.png", regex=False
        )
        df["label_npy"] = df["mask_npy"].str.replace(
            "mask.npy", "inklabels.npy", regex=False
        )

    df["volumes_dir"] = df["mask_png"].str.replace(
        "mask.png", "surface_volume", regex=False
    )
    df["volume_npy"] = df["mask_npy"].str.replace("mask.npy", "volume.npy", regex=False)

    return df




## === cell 5
def load_image(path):
    return Image.open(path)


def resize_image(image, downsampling):
    size = (int(image.size[0] * downsampling), int(image.size[1] * downsampling))
    size = (max(1, size[0]), max(1, size[1]))
    return image.resize(size)


def load_and_resize_image(path, downsampling):
    image = load_image(path)
    return resize_image(image, downsampling)


def load_mask_npy(path, downsampling):
    mask = load_and_resize_image(path, downsampling).convert("1")
    return np.array(mask, dtype=np.uint8)


def load_label_npy(path, downsampling):
    label = load_and_resize_image(path, downsampling).convert("L")
    arr = (np.array(label, dtype=np.uint8) > 0).astype(np.uint8)
    return arr


def load_z_slice_npy(path, downsampling):
    z_slice = load_and_resize_image(path, downsampling)
    return np.array(z_slice, dtype=np.float32) / 65535.0


def load_volume_npy(volumes_dir, downsampling):
    surface_volume_paths = sorted(Path(volumes_dir).glob("*.tif"))[
        Z_START : Z_START + Z_DIM
    ]
    if len(surface_volume_paths) != Z_DIM:
        raise RuntimeError(
            f"Expected {Z_DIM} slices, got {len(surface_volume_paths)} from {volumes_dir}"
        )

    batch_size = 8
    paths_batches = [
        surface_volume_paths[i : i + batch_size]
        for i in range(0, len(surface_volume_paths), batch_size)
    ]

    volumes = []
    for paths_batch in tqdm(
        paths_batches, leave=False, desc="Processing z-batches", position=1
    ):
        z_slices = [
            load_z_slice_npy(path, downsampling)
            for path in tqdm(
                paths_batch, leave=False, desc="Processing z-slices", position=2
            )
        ]
        volumes.append(np.stack(z_slices, axis=0))
        del z_slices

    volume = np.concatenate(volumes, axis=0)  # (Z, H, W)
    return volume




## === cell 6
def save_data_as_npy(df, train=True):
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Processing fragments",
        position=0,
    ):
        mask_npy = load_mask_npy(row.mask_png, DOWNSAMPLING)
        volume_npy = load_volume_npy(row.volumes_dir, DOWNSAMPLING)

        Path(row.mask_npy).parent.mkdir(exist_ok=True, parents=True)
        np.save(row.mask_npy, mask_npy)
        np.save(row.volume_npy, volume_npy)

        if train:
            label_npy = load_label_npy(row.label_png, DOWNSAMPLING)
            np.save(row.label_npy, label_npy)

        tqdm.write(
            f"Created {row.volume_npy} with shape {volume_npy.shape} and mask shape {mask_npy.shape}"
        )




## === cell 7
train_mask_paths = sorted((COMPETITION_DATA_DIR / "train").glob("*/mask.png"))
test_mask_paths = sorted((COMPETITION_DATA_DIR / "test").glob("*/mask.png"))

train_df = create_df_from_mask_paths(train_mask_paths, train=True)
test_df = create_df_from_mask_paths(test_mask_paths, train=False)

need_train = []
for r in train_df.itertuples(index=False):
    if not (
        Path(r.mask_npy).exists()
        and Path(r.volume_npy).exists()
        and Path(r.label_npy).exists()
    ):
        need_train.append(r)
if len(need_train) > 0:
    save_data_as_npy(train_df, train=True)

need_test = []
for r in test_df.itertuples(index=False):
    if not (Path(r.mask_npy).exists() and Path(r.volume_npy).exists()):
        need_test.append(r)
if len(need_test) > 0:
    save_data_as_npy(test_df, train=False)

all_df = pd.concat([train_df, test_df], ignore_index=True)
ALL_DATA_CSV_PATH = str(KAGGLE_DIR / "working" / "all_data.csv")
all_df.to_csv(ALL_DATA_CSV_PATH, index=False)

TEST_DATA_CSV_PATH = str(KAGGLE_DIR / "working" / "test.csv")
TRAIN_DATA_CSV_PATH = ALL_DATA_CSV_PATH
test_df.to_csv(TEST_DATA_CSV_PATH, index=False)

train_df.head(), test_df.head()




## === cell 8
def set_determinism(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VesuviusCSVDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        train: bool,
        patch_size: Tuple[int, int],
        num_samples: int,
    ):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.patch_size = patch_size
        self.num_samples = num_samples

    def __len__(self):
        return len(self.df)

    def _load_case(self, idx: int):
        row = self.df.iloc[idx]
        vol = np.load(row["volume_npy"]).astype(np.float32)  # (Z,H,W)
        msk = np.load(row["mask_npy"]).astype(np.float32)  # (H,W) 0/1
        out = {
            "volume_npy": torch.from_numpy(vol),
            "mask_npy": torch.from_numpy(msk)[None, ...],  # (1,H,W)
        }
        if self.train:
            lab = np.load(row["label_npy"]).astype(np.float32)  # (H,W) 0/1
            out["label_npy"] = torch.from_numpy(lab)[None, ...]  # (1,H,W)
        return out

    def _rand_crop_coords(self, H: int, W: int, ph: int, pw: int):
        if H == ph:
            y0 = 0
        else:
            y0 = np.random.randint(0, H - ph + 1)
        if W == pw:
            x0 = 0
        else:
            x0 = np.random.randint(0, W - pw + 1)
        return y0, x0

    def _weighted_crop(self, sample):
        vol = sample["volume_npy"]  # (Z,H,W)
        msk = sample["mask_npy"]  # (1,H,W)
        lab = sample.get("label_npy", None)

        Z, H, W = vol.shape
        ph, pw = self.patch_size
        ph = min(ph, H)
        pw = min(pw, W)

        best_yx = None
        for _ in range(16):
            y0, x0 = self._rand_crop_coords(H, W, ph, pw)
            m = msk[:, y0 : y0 + ph, x0 : x0 + pw]
            if m.sum().item() > 0:
                best_yx = (y0, x0)
                break
        if best_yx is None:
            best_yx = self._rand_crop_coords(H, W, ph, pw)

        y0, x0 = best_yx
        vol_c = vol[:, y0 : y0 + ph, x0 : x0 + pw]  # (Z,ph,pw)
        msk_c = msk[:, y0 : y0 + ph, x0 : x0 + pw]
        out = {"volume_npy": vol_c, "mask_npy": msk_c}
        if lab is not None:
            out["label_npy"] = lab[:, y0 : y0 + ph, x0 : x0 + pw]
        return out

    def __getitem__(self, idx: int):
        sample = self._load_case(idx)

        if self.train:
            crops = [self._weighted_crop(sample) for _ in range(self.num_samples)]
            if np.random.rand() < 0.5:
                for c in crops:
                    c["volume_npy"] = torch.flip(c["volume_npy"], dims=[1])
                    c["mask_npy"] = torch.flip(c["mask_npy"], dims=[1])
                    c["label_npy"] = torch.flip(c["label_npy"], dims=[1])
            if np.random.rand() < 0.5:
                for c in crops:
                    c["volume_npy"] = torch.flip(c["volume_npy"], dims=[2])
                    c["mask_npy"] = torch.flip(c["mask_npy"], dims=[2])
                    c["label_npy"] = torch.flip(c["label_npy"], dims=[2])

            vol = torch.stack([c["volume_npy"] for c in crops], dim=0)  # (S,Z,ph,pw)
            msk = torch.stack([c["mask_npy"] for c in crops], dim=0)  # (S,1,ph,pw)
            lab = torch.stack([c["label_npy"] for c in crops], dim=0)  # (S,1,ph,pw)
            return {"volume_npy": vol, "mask_npy": msk, "label_npy": lab}

        return sample


def dice_loss_with_mask(logits, targets, mask, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs * mask
    targets = targets * mask
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    dice = (num + eps) / den
    return 1 - dice.mean()


def sliding_window_inference_torch(
    inputs: torch.Tensor,  # (B,Z,H,W)
    roi_size: Tuple[int, int],
    sw_batch_size: int,
    predictor,
):
    B, Z, H, W = inputs.shape
    ph, pw = roi_size
    ph = min(ph, H)
    pw = min(pw, W)
    sh = max(1, ph // 2)
    sw = max(1, pw // 2)

    out = torch.zeros((B, 1, H, W), device=inputs.device, dtype=inputs.dtype)
    cnt = torch.zeros((B, 1, H, W), device=inputs.device, dtype=inputs.dtype)

    ys = list(range(0, max(1, H - ph + 1), sh))
    xs = list(range(0, max(1, W - pw + 1), sw))
    if ys[-1] != H - ph:
        ys.append(H - ph)
    if xs[-1] != W - pw:
        xs.append(W - pw)

    patches = []
    locs = []
    for y0 in ys:
        for x0 in xs:
            patch = inputs[:, :, y0 : y0 + ph, x0 : x0 + pw]  # (B,Z,ph,pw)
            patches.append(patch)
            locs.append((y0, x0))

            if len(patches) == sw_batch_size:
                batch_p = torch.cat(patches, dim=0)  # (B*nb,Z,ph,pw)
                pred = predictor(batch_p)  # (B*nb,1,ph,pw)
                nb = pred.shape[0] // B
                pred = pred.view(B, nb, 1, ph, pw)
                for i, (yy, xx) in enumerate(locs):
                    out[:, :, yy : yy + ph, xx : xx + pw] += pred[:, i]
                    cnt[:, :, yy : yy + ph, xx : xx + pw] += 1
                patches, locs = [], []

    if patches:
        batch_p = torch.cat(patches, dim=0)
        pred = predictor(batch_p)
        nb = pred.shape[0] // B
        pred = pred.view(B, nb, 1, ph, pw)
        for i, (yy, xx) in enumerate(locs):
            out[:, :, yy : yy + ph, xx : xx + pw] += pred[:, i]
            cnt[:, :, yy : yy + ph, xx : xx + pw] += 1

    out = out / cnt.clamp_min(1.0)
    return out




## === cell 9
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=16, out_channels=1, dropout=0.0, base=16):
        super().__init__()
        self.inc = DoubleConv(in_channels, base, dropout=dropout)
        self.down1 = Down(base, base * 2, dropout=dropout)
        self.down2 = Down(base * 2, base * 4, dropout=dropout)
        self.down3 = Down(base * 4, base * 8, dropout=dropout)
        self.down4 = Down(base * 8, base * 16, dropout=dropout)
        self.up1 = Up(base * 16, base * 8, dropout=dropout)
        self.up2 = Up(base * 8, base * 4, dropout=dropout)
        self.up3 = Up(base * 4, base * 2, dropout=dropout)
        self.up4 = Up(base * 2, base, dropout=dropout)
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)




## === cell 10
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

            train_df = train_val_df[
                train_val_df.fragmet_id != str(self.hparams.val_fragment_id)
            ].reset_index(drop=True)
            val_df = train_val_df[
                train_val_df.fragmet_id == str(self.hparams.val_fragment_id)
            ].reset_index(drop=True)

            if len(train_df) == 0 or len(val_df) == 0:
                raise RuntimeError(
                    f"Empty train/val split. Check VAL_FRAGMENT_ID={self.hparams.val_fragment_id} and fragment ids: "
                    f"{sorted(train_val_df.fragmet_id.unique().tolist())}"
                )

            self.train_dataset = VesuviusCSVDataset(
                train_df,
                train=True,
                patch_size=self.hparams.patch_size,
                num_samples=self.hparams.num_samples,
            )
            self.val_dataset = VesuviusCSVDataset(
                val_df, train=False, patch_size=self.hparams.patch_size, num_samples=1
            )
            print(
                f"# train fragments: {len(train_df)} (each yields {self.hparams.num_samples} crops/step)"
            )
            print(f"# val fragments: {len(val_df)}")

        if stage == "predict" or stage is None:
            predict_df = self.df[self.df.stage == "test"].reset_index(drop=True)
            self.predict_dataset = VesuviusCSVDataset(
                predict_df,
                train=False,
                patch_size=self.hparams.patch_size,
                num_samples=1,
            )
            print(f"# predict fragments: {len(predict_df)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )




## === cell 11
class VesuvisModule(pl.LightningModule):
    def __init__(
        self,
        dropout: float,
        eta_min: float,
        learning_rate: float,
        loss: str,
        model_name: str,
        max_epochs: int,
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
            return SimpleUNet(
                in_channels=Z_DIM,
                out_channels=1,
                dropout=self.hparams.dropout,
                base=16,
            )
        raise ValueError(f"{self.hparams.model_name} is not implemented")

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"{self.hparams.optimizer} is not implemented")

        if self.hparams.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.eta_min
            )
        else:
            raise ValueError(f"{self.hparams.scheduler} is not implemented")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        vol = batch["volume_npy"]
        msk = batch["mask_npy"]
        lab = batch["label_npy"]
        B, S, Z, H, W = vol.shape
        vol = vol.view(B * S, Z, H, W)
        msk = msk.view(B * S, 1, H, W)
        lab = lab.view(B * S, 1, H, W)

        logits = self(vol)
        loss = dice_loss_with_mask(logits, lab, msk)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B
        )
        return loss

    def validation_step(self, batch, batch_idx):
        vol = batch["volume_npy"]
        msk = batch["mask_npy"]
        lab = batch["label_npy"]

        if vol.ndim == 3:
            vol = vol.unsqueeze(0)
        if msk.ndim == 3:
            msk = msk.unsqueeze(0)
        if lab.ndim == 3:
            lab = lab.unsqueeze(0)

        logits = sliding_window_inference_torch(
            inputs=vol,
            roi_size=self.hparams.patch_size,
            sw_batch_size=self.hparams.sw_batch_size,
            predictor=self,
        )
        loss = dice_loss_with_mask(logits, lab, msk)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1
        )
        return loss

    def predict_step(self, batch, batch_idx):
        vol = batch["volume_npy"]
        if vol.ndim == 3:
            vol = vol.unsqueeze(0)
        logits = sliding_window_inference_torch(
            inputs=vol,
            roi_size=self.hparams.patch_size,
            sw_batch_size=self.hparams.sw_batch_size,
            predictor=self,
        )
        return torch.sigmoid(logits).squeeze(0).squeeze(0)  # (H,W)




## === cell 12
def train(
    accelerator=ACCELERATOR,
    batch_size=BATCH_SIZE,
    data_csv_path=TRAIN_DATA_CSV_PATH,
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
    set_determinism(seed)
    pl.seed_everything(seed, workers=True)

    data_module = VesuvisDataModule(
        batch_size=batch_size,
        data_csv_path=str(data_csv_path),
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
        optimizer=optimizer,
        patch_size=patch_size,
        scheduler=scheduler,
        sw_batch_size=sw_batch_size,
        weight_decay=weight_decay,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        benchmark=True,
        check_val_every_n_epoch=max(1, max_epochs // 5),
        devices=devices,
        fast_dev_run=fast_dev_run,
        logger=pl.loggers.CSVLogger(save_dir="logs/"),
        log_every_n_steps=1,
        max_epochs=max_epochs,
        overfit_batches=overfit_batches,
        precision=precision if accelerator != "cpu" else "32-true",
        strategy="ddp" if devices > 1 else "auto",
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    trainer.fit(module, datamodule=data_module)
    return module, trainer




## === cell 13
module, trainer = train()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/161691657.py in <cell line: 0>()
----> 1 module, trainer = train()
      2 

/tmp/ipykernel_55/2127639767.py in train(accelerator, batch_size, data_csv_path, devices, dropout, eta_min, fast_dev_run, learning_rate, loss, model_name, max_epochs, num_workers, num_samples, optimizer, overfit_batches, patch_size, precision, scheduler, seed, sw_batch_size, val_fragment_id, weight_decay)
     65     )
     66 
---> 67     trainer.fit(module, datamodule=data_module)
     68     return module, trainer
     69 

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in fit(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
    558         self.training = True
    559         self.should_stop = False
--> 560         call._call_and_handle_interrupt(
    561             self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
    562         )

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
     47         if trainer.strategy.launcher is not None:
     48             return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
---> 49         return trainer_fn(*args, **kwargs)
     50 
     51     except _TunerExitException:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _fit_impl(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
    596             model_connected=self.lightning_module is not None,
    597         )
--> 598         self._run(model, ckpt_path=ckpt_path)
    599 
    600         assert self.state.stopped

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run(self, model, ckpt_path)
   1009         # RUN THE TRAINER
   1010         # ----------------------------
-> 1011         results = self._run_stage()
   1012 
   1013         # ----------------------------

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run_stage(self)
   1051         if self.training:
   1052             with isolate_rng():
-> 1053                 self._run_sanity_check()
   1054             with torch.autograd.set_detect_anomaly(self._detect_anomaly):
   1055                 self.fit_loop.run()

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run_sanity_check(self)
   1080 
   1081             # run eval step
-> 1082             val_loop.run()
   1083 
   1084             call._call_callback_hooks(self, "on_sanity_check_end")

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/utilities.py in _decorator(self, *args, **kwargs)
    177             context_manager = torch.no_grad
    178         with context_manager():
--> 179             return loop_run(self, *args, **kwargs)
    180 
    181     return _decorator

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/evaluation_loop.py in run(self)
    143                 self.batch_progress.is_last_batch = data_fetcher.done
    144                 # run step hooks
--> 145                 self._evaluation_step(batch, batch_idx, dataloader_idx, dataloader_iter)
    146             except StopIteration:
    147                 # this needs to wrap the `*_step` call too (not just `next`) for `dataloader_iter` support

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/evaluation_loop.py in _evaluation_step(self, batch, batch_idx, dataloader_idx, dataloader_iter)
    435             else (dataloader_iter,)
    436         )
--> 437         output = call._call_strategy_hook(trainer, hook_name, *step_args)
    438 
    439         self.batch_progress.increment_processed()

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py in _call_strategy_hook(trainer, hook_name, *args, **kwargs)
    327 
    328     with trainer.profiler.profile(f"[Strategy]{trainer.strategy.__class__.__name__}.{hook_name}"):
--> 329         output = fn(*args, **kwargs)
    330 
    331     # restore current_fx when nested context

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/strategies/strategy.py in validation_step(self, *args, **kwargs)
    410             if self.model != self.lightning_module:
    411                 return self._forward_redirection(self.model, self.lightning_module, "validation_step", *args, **kwargs)
--> 412             return self.lightning_module.validation_step(*args, **kwargs)
    413 
    414     def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:

/tmp/ipykernel_55/3517730601.py in validation_step(self, batch, batch_idx)
     73         vol = batch["volume_npy"]
     74         msk = batch["mask_npy"]
---> 75         lab = batch["label_npy"]
     76 
     77         # Ensure shapes are (B,Z,H,W), (B,1,H,W), (B,1,H,W)

KeyError: 'label_npy'

## === cell 14
if "trainer" in globals() and getattr(trainer, "logger", None) is not None:
    metrics_path = Path(trainer.logger.log_dir) / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        keep = [c for c in ["epoch", "train_loss", "val_loss"] if c in metrics.columns]
        metrics = metrics[keep].dropna(subset=["epoch"]).copy()
        if "epoch" in metrics.columns:
            metrics.set_index("epoch", inplace=True)
        if len(metrics) > 0:
            sns.relplot(
                data=metrics.reset_index(),
                x="epoch",
                y=[c for c in keep if c != "epoch"],
                kind="line",
                height=5,
                aspect=1.5,
            )
            plt.grid()
            plt.show()




## === cell 15
def predict(
    module,
    accelerator=ACCELERATOR,
    batch_size=BATCH_SIZE,
    data_csv_path=TEST_DATA_CSV_PATH,
    devices=DEVICES,
    num_workers=NUM_WORKERS,
    num_samples=NUM_SAMPLES,
    patch_size=PATCH_SIZE,
    precision=PRECISION,
    seed=SEED,
    val_fragment_id=VAL_FRAGMENT_ID,
):
    set_determinism(seed)
    pl.seed_everything(seed, workers=True)

    data_module = VesuvisDataModule(
        batch_size=batch_size,
        data_csv_path=str(data_csv_path),
        num_workers=num_workers,
        num_samples=num_samples,
        patch_size=patch_size,
        val_fragment_id=val_fragment_id,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision if accelerator != "cpu" else "32-true",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    predictions = trainer.predict(module, datamodule=data_module)
    return predictions




## === cell 16
predictions = predict(module)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4174353991.py in <cell line: 0>()
----> 1 predictions = predict(module)
      2 
      3 

NameError: name 'module' is not defined

## === cell 17
def plot_image(image, title):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def rle(img: np.ndarray) -> str:
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted (1-indexed), sorted, no duplicates.
    Flatten is row-major (left->right, top->bottom), which matches competition expectation.
    """
    pixels = img.astype(np.uint8).flatten(order="C")
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    runs = np.stack([starts, lengths], axis=1).reshape(-1)
    return " ".join(str(int(x)) for x in runs)




## === cell 18
sample_sub_path = COMPETITION_DATA_DIR / "sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
required_ids = sample_sub["Id"].astype(str).tolist()

test_df_by_id = test_df.set_index(test_df["fragmet_id"].astype(str))
pred_by_id = {}
for frag_id, pred in zip(test_df["fragmet_id"].astype(str).tolist(), predictions):
    pred_by_id[frag_id] = pred

predictions_rle = []
for frag_id in required_ids:
    if frag_id not in test_df_by_id.index or frag_id not in pred_by_id:
        predictions_rle.append("")
        continue

    mask_png_path = test_df_by_id.loc[frag_id, "mask_png"]
    prediction = pred_by_id[frag_id]

    pred = prediction.detach().cpu().numpy().astype(np.float32)  # (H,W) downsampled

    mask_img = load_image(mask_png_path).convert("1")
    mask_arr = np.array(mask_img, dtype=np.uint8)  # full-res 0/1

    pred_img = Image.fromarray((pred * 255.0).clip(0, 255).astype(np.uint8))
    pred_resized = (
        np.array(
            pred_img.resize(mask_img.size, resample=Image.BILINEAR), dtype=np.float32
        )
        / 255.0
    )

    threshold = 0.8
    pred_masked = pred_resized * mask_arr
    pred_thresholded = (pred_masked > threshold).astype(np.uint8)

    predictions_rle.append(rle(pred_thresholded))

submission_df = pd.DataFrame({"Id": required_ids, "Predicted": predictions_rle})
submission_path = Path("submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Wrote {submission_path.resolve()} with shape {submission_df.shape}")
submission_df

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/639168451.py in <cell line: 0>()
      8 test_df_by_id = test_df.set_index(test_df["fragmet_id"].astype(str))
      9 pred_by_id = {}
---> 10 for frag_id, pred in zip(test_df["fragmet_id"].astype(str).tolist(), predictions):
     11     pred_by_id[frag_id] = pred
     12 

NameError: name 'predictions' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Expected 1 rows in the submission DataFrame, but got 3 rows.

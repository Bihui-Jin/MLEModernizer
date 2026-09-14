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

0.1342954358193677

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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")



## === cell 1
from typing import Tuple

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from torchvision.io import read_image, ImageReadMode



## === cell 2
try:
    import lovely_numpy as ln  # type: ignore

    _has_lovely = True
except Exception:
    _has_lovely = False

    class _LN:
        @staticmethod
        def lovely(x):
            try:
                return (
                    f"shape={getattr(x, 'shape', None)}, dtype={getattr(x, 'dtype', None)}, "
                    f"min={np.min(x):.4g}, max={np.max(x):.4g}"
                )
            except Exception:
                return str(x)

    ln = _LN()  # noqa: N816



## === cell 3
KAGGLE_DIR = Path("/") / "kaggle"

INPUT_DIR = KAGGLE_DIR / "input"
WORKING_DIR = KAGGLE_DIR / "working"

_CANDIDATES = [
    INPUT_DIR / "vesuvius-challenge-ink-detection",
    INPUT_DIR / "vesuvius-challenge-ink-detection" / "vesuvius-challenge-ink-detection",
    KAGGLE_DIR / "data" / "vesuvius-challenge-ink-detection",
    KAGGLE_DIR
    / "data"
    / "vesuvius-challenge-ink-detection"
    / "vesuvius-challenge-ink-detection",
    KAGGLE_DIR / "data",
    INPUT_DIR,
]
COMPETITION_DATA_DIR = None
for c in _CANDIDATES:
    if (c / "train").exists() and (c / "test").exists():
        COMPETITION_DATA_DIR = c
        break
if COMPETITION_DATA_DIR is None:
    COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"

PREPARED_DATA_DIR = WORKING_DIR / "vesuvius_prepared"

TRAIN_DATA_CSV_PATH = PREPARED_DATA_DIR / "data_0.1.csv"
TEST_DATA_CSV_PATH = "test.csv"
TRAIN_DATA_CSV_OUT = "train.csv"

if str(TRAIN_DATA_CSV_PATH).endswith(".csv"):
    try:
        DOWNSAMPLING = float(
            TRAIN_DATA_CSV_PATH.name.split("_")[-1].replace(".csv", "")
        )
    except Exception:
        DOWNSAMPLING = 0.1
else:
    DOWNSAMPLING = 0.1

Z_START = 27  # First slice in the z direction to use
Z_DIM = 16  # Number of slices in the z direction

ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
DEVICES = 1
DROPOUT = 0.0
ETA_MIN = 1e-6
FAST_DEV_RUN = False
LEARNING_RATE = 3e-4
LOSS = "Dice"  # kept for config compatibility (fallback uses BCEWithLogits)
MODEL_NAME = "UNet"  # kept for config compatibility (fallback uses small CNN)
MAX_EPOCHS = 5  # keep as-is
NUM_WORKERS = 0
NUM_SAMPLES = 64  # unused in fallback
OPTIMIZER = "AdamW"
OVERFIT_BATCHES = 0
PATCH_SIZE = (512, 512)  # unused in fallback
PRECISION = 16  # Lightning will handle on GPU; on CPU we override to 32
SCHEDULER = "CosineAnnealingLR"
SEED = 2023
SW_BATCH_SIZE = 16  # unused in fallback
VAL_FRAGMET_ID = 3
WEIGHT_DECAY = 1e-6

pl.seed_everything(SEED, workers=True)
torch.set_float32_matmul_precision("high")

print("Using COMPETITION_DATA_DIR:", str(COMPETITION_DATA_DIR))




## === cell 4
def create_df_from_mask_paths(mask_paths, train=True):
    df = pd.DataFrame({"mask_png": mask_paths})
    df["mask_png"] = df["mask_png"].astype(str)

    df["stage"] = df["mask_png"].str.split("/").str[-3]
    df["fragmet_id"] = df["mask_png"].str.split("/").str[-2]

    base_out = PREPARED_DATA_DIR / "prepared_npy" / f"down_{DOWNSAMPLING}"
    df["mask_npy"] = df.apply(
        lambda r: str(base_out / r["stage"] / str(r["fragmet_id"]) / "mask.npy"), axis=1
    )

    if train:
        df["label_png"] = df["mask_png"].str.replace(
            "mask.png", "inklabels.png", regex=False
        )
        df["label_npy"] = df.apply(
            lambda r: str(
                base_out / r["stage"] / str(r["fragmet_id"]) / "inklabels.npy"
            ),
            axis=1,
        )

    df["volumes_dir"] = df["mask_png"].str.replace(
        "mask.png", "surface_volume", regex=False
    )
    df["volume_npy"] = df.apply(
        lambda r: str(base_out / r["stage"] / str(r["fragmet_id"]) / "volume.npy"),
        axis=1,
    )
    return df




## === cell 5
def load_image(path):
    return Image.open(path)


def resize_image(image, downsampling, resample):
    size = int(image.size[0] * downsampling), int(image.size[1] * downsampling)
    return image.resize(size, resample=resample)


def load_and_resize_image(path, downsampling, resample):
    image = load_image(path)
    return resize_image(image, downsampling, resample=resample)


def load_mask_npy(path, downsampling):
    mask = load_and_resize_image(path, downsampling, resample=Image.NEAREST).convert(
        "1"
    )
    return np.array(mask, dtype=np.uint8)


def load_label_npy(path, downsampling):
    label = load_and_resize_image(path, downsampling, resample=Image.NEAREST).convert(
        "1"
    )
    return np.array(label, dtype=np.uint8)


def load_z_slice_npy(path, downsampling):
    img = read_image(str(path), mode=ImageReadMode.GRAY)  # uint8/uint16, shape (1,H,W)
    arr = img.squeeze(0).cpu().numpy()
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    denom = 65535.0 if arr.max() > 255.0 else 255.0
    arr = arr / denom
    if downsampling != 1.0:
        pil = Image.fromarray(arr, mode="F")
        size = (int(pil.size[0] * downsampling), int(pil.size[1] * downsampling))
        pil = pil.resize(size, resample=Image.BILINEAR)
        arr = np.array(pil, dtype=np.float32)
    return arr


def load_volume_npy(volumes_dir, downsampling):
    surface_volume_paths = sorted(Path(volumes_dir).glob("*.tif"))[
        Z_START : Z_START + Z_DIM
    ]

    batch_size = 8
    paths_batches = [
        surface_volume_paths[i : i + batch_size]
        for i in range(0, len(surface_volume_paths), batch_size)
    ]

    volumes = []
    for paths_batch in tqdm(
        paths_batches, leave=False, desc="Processing batches", position=1
    ):
        z_slices = [
            load_z_slice_npy(path, downsampling)
            for path in tqdm(
                paths_batch, leave=False, desc="Processing paths", position=2
            )
        ]
        volumes.append(np.stack(z_slices, axis=0))
        del z_slices

    volume = np.concatenate(volumes, axis=0)
    return volume




## === cell 6
def save_data_as_npy(df, train=True):
    for row in tqdm(
        df.itertuples(), total=len(df), desc="Processing fragments", position=0
    ):
        mask_npy = load_mask_npy(row.mask_png, DOWNSAMPLING)
        volume_npy = load_volume_npy(row.volumes_dir, DOWNSAMPLING)

        Path(row.mask_npy).parent.mkdir(exist_ok=True, parents=True)
        np.save(row.mask_npy, mask_npy)
        np.save(row.volume_npy, volume_npy)

        if train:
            label_npy = load_label_npy(row.label_png, DOWNSAMPLING)
            np.save(row.label_npy, label_npy)

        tqdm.write(f"Created {row.volume_npy} with shape {volume_npy.shape}")




## === cell 7
test_mask_paths = sorted((COMPETITION_DATA_DIR / "test").glob("*/mask.png"))
test_mask_paths = [p for p in test_mask_paths if p.name == "mask.png"]

test_df = create_df_from_mask_paths(test_mask_paths, train=False)
test_df.to_csv(TEST_DATA_CSV_PATH, index=False)
test_df



## === cell 8
train_mask_paths = sorted((COMPETITION_DATA_DIR / "train").glob("*/mask.png"))
train_mask_paths = [p for p in train_mask_paths if p.name == "mask.png"]
train_df = create_df_from_mask_paths(train_mask_paths, train=True)
train_df.to_csv(TRAIN_DATA_CSV_OUT, index=False)
train_df




## === cell 9
def ensure_prepared(df: pd.DataFrame, train: bool):
    needed_rows = []
    for idx, r in enumerate(df.itertuples(index=False)):
        if not Path(r.mask_npy).exists():
            needed_rows.append(idx)
            continue
        if not Path(r.volume_npy).exists():
            needed_rows.append(idx)
            continue
        if train and (not Path(r.label_npy).exists()):
            needed_rows.append(idx)
            continue
    if len(needed_rows) == 0:
        return
    subdf = df.iloc[needed_rows].copy()
    save_data_as_npy(subdf, train=train)


ensure_prepared(test_df, train=False)
ensure_prepared(train_df, train=True)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/327176117.py in <cell line: 0>()
     17 
     18 
---> 19 ensure_prepared(test_df, train=False)
     20 ensure_prepared(train_df, train=True)
     21 

/tmp/ipykernel_55/327176117.py in ensure_prepared(df, train)
     14         return
     15     subdf = df.iloc[needed_rows].copy()
---> 16     save_data_as_npy(subdf, train=train)
     17 
     18 

/tmp/ipykernel_55/244011798.py in save_data_as_npy(df, train)
      4     ):
      5         mask_npy = load_mask_npy(row.mask_png, DOWNSAMPLING)
----> 6         volume_npy = load_volume_npy(row.volumes_dir, DOWNSAMPLING)
      7 
      8         Path(row.mask_npy).parent.mkdir(exist_ok=True, parents=True)

/tmp/ipykernel_55/1781119962.py in load_volume_npy(volumes_dir, downsampling)
     59         paths_batches, leave=False, desc="Processing batches", position=1
     60     ):
---> 61         z_slices = [
     62             load_z_slice_npy(path, downsampling)
     63             for path in tqdm(

/tmp/ipykernel_55/1781119962.py in <listcomp>(.0)
     60     ):
     61         z_slices = [
---> 62             load_z_slice_npy(path, downsampling)
     63             for path in tqdm(
     64                 paths_batch, leave=False, desc="Processing paths", position=2

/tmp/ipykernel_55/1781119962.py in load_z_slice_npy(path, downsampling)
     29 # Change (score+runtime relevant): use torchvision TIFF reader (faster) and keep same normalization.
     30 def load_z_slice_npy(path, downsampling):
---> 31     img = read_image(str(path), mode=ImageReadMode.GRAY)  # uint8/uint16, shape (1,H,W)
     32     arr = img.squeeze(0).cpu().numpy()
     33     if arr.dtype != np.float32:

/usr/local/lib/python3.11/dist-packages/torchvision/io/image.py in read_image(path, mode, apply_exif_orientation)
    335         _log_api_usage_once(read_image)
    336     data = read_file(path)
--> 337     return decode_image(data, mode, apply_exif_orientation=apply_exif_orientation)
    338 
    339 

/usr/local/lib/python3.11/dist-packages/torchvision/io/image.py in decode_image(input, mode, apply_exif_orientation)
    322     if isinstance(mode, str):
    323         mode = ImageReadMode[mode.upper()]
--> 324     output = torch.ops.image.decode_image(input, mode.value, apply_exif_orientation)
    325     return output
    326 

/usr/local/lib/python3.11/dist-packages/torch/_ops.py in __call__(self, *args, **kwargs)
   1121         if self._has_torchbind_op_overload and _must_dispatch_in_python(args, kwargs):
   1122             return _call_overload_packet_from_python(self, args, kwargs)
-> 1123         return self._op(*args, **(kwargs or {}))
   1124 
   1125     # TODO: use this to make a __dir__

RuntimeError: Unsupported image file. Only jpeg, png, webp and gif are currently supported. For avif and heic format, please rely on `decode_avif` and `decode_heic` directly.

## === cell 10
class VesuvisTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        vol = np.load(r["volume_npy"]).astype(np.float32)  # (Z,H,W)
        m = np.load(r["mask_npy"]).astype(np.uint8)  # (H,W)

        x = torch.from_numpy(vol)  # (Z,H,W)
        mask = torch.from_numpy(m)  # (H,W)
        frag_id = str(r["fragmet_id"])
        return {"x": x, "mask": mask, "fragmet_id": frag_id}




## === cell 11
class VesuvisTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        vol = np.load(r["volume_npy"]).astype(np.float32)  # (Z,H,W)
        y = np.load(r["label_npy"]).astype(np.uint8)  # (H,W)
        m = np.load(r["mask_npy"]).astype(np.uint8)  # (H,W)

        y = (y * m).astype(np.uint8)

        x = torch.from_numpy(vol)  # (Z,H,W)
        y = torch.from_numpy(y)  # (H,W)
        return {"x": x, "y": y, "mask": torch.from_numpy(m)}




## === cell 12
class SmallSegNet(nn.Module):
    """
    Minimal 2D CNN that maps 16-channel input -> 1-channel logits.
    """

    def __init__(self, in_ch=16, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


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
        self.model = SmallSegNet(in_ch=Z_DIM, dropout=dropout)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["x"].float()
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1,Z,H,W)

        y = batch["y"].float()
        if y.ndim == 2:
            y = y.unsqueeze(0).unsqueeze(1)  # (1,1,H,W)
        elif y.ndim == 3:
            y = y.unsqueeze(1)  # (B,1,H,W)

        m = batch["mask"].float()
        if m.ndim == 2:
            m = m.unsqueeze(0).unsqueeze(1)  # (1,1,H,W)
        elif m.ndim == 3:
            m = m.unsqueeze(1)  # (B,1,H,W)

        logits = self(x)
        per_pixel = self.criterion(logits, y)  # (B,1,H,W)
        denom = torch.clamp(m.sum(), min=1.0)
        loss = (per_pixel * m).sum() / denom

        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch["x"].float()  # (Z,H,W) or (B,Z,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1,Z,H,W)
        logits = self(x)  # (B,1,H,W)
        probs = torch.sigmoid(logits)
        return probs.squeeze(1)  # (B,H,W)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs, eta_min=self.hparams.eta_min
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
        }




## === cell 13
def predict(
    module,
    accelerator=ACCELERATOR,
    batch_size=BATCH_SIZE,
    data_csv_path=TEST_DATA_CSV_PATH,
    devices=DEVICES,
    num_workers=NUM_WORKERS,
    precision=PRECISION,
):
    df = pd.read_csv(data_csv_path)
    ds = VesuvisTestDataset(df)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision if accelerator != "cpu" else "32-true",
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    preds = trainer.predict(module, dataloaders=dl)
    return preds, df




## === cell 14
def fbeta_score_numpy(
    y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5
) -> float:
    y_true = (y_true.astype(np.uint8).reshape(-1) > 0).astype(np.uint8)
    y_pred = (y_pred.astype(np.uint8).reshape(-1) > 0).astype(np.uint8)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0

    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    b2 = beta * beta
    return float((1 + b2) * p * r / (b2 * p + r + 1e-12))


@torch.no_grad()
def calibrate_global_threshold(
    module: pl.LightningModule, train_df: pd.DataFrame
) -> float:
    ds = VesuvisTrainDataset(train_df)
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION if ACCELERATOR != "cpu" else "32-true",
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    preds_list = trainer.predict(module, dataloaders=dl)

    preds_flat = []
    for p in preds_list:
        if isinstance(p, torch.Tensor):
            if p.ndim == 3:
                preds_flat.append(p[0])  # (H,W)
            elif p.ndim == 2:
                preds_flat.append(p)
            else:
                raise ValueError(
                    f"Unexpected pred tensor shape during calibration: {tuple(p.shape)}"
                )
        else:
            raise TypeError(f"Unexpected prediction type during calibration: {type(p)}")

    assert len(preds_flat) == len(train_df)

    gts = []
    msks = []
    for r in train_df.itertuples(index=False):
        y = np.load(r.label_npy).astype(np.uint8)
        m = np.load(r.mask_npy).astype(np.uint8)
        y = (y * m).astype(np.uint8)
        gts.append(y)
        msks.append(m)

    thr_grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)

    best_thr = 0.5
    best_score = -1.0
    for thr in thr_grid:
        scores = []
        for pred_t, y, m in zip(preds_flat, gts, msks):
            pr = pred_t.detach().float().cpu().numpy()
            m_bool = m > 0
            yp = (pr > float(thr)).astype(np.uint8)

            scores.append(fbeta_score_numpy(y[m_bool], yp[m_bool], beta=0.5))
        sc = float(np.mean(scores)) if len(scores) else 0.0
        if sc > best_score:
            best_score = sc
            best_thr = float(thr)

    print(f"[calibration] best_thr={best_thr:.3f}, mean_F0.5={best_score:.4f}")
    return best_thr




## === cell 15
train_ds = VesuvisTrainDataset(train_df)
train_dl = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

module = VesuvisModule(
    dropout=DROPOUT,
    eta_min=ETA_MIN,
    learning_rate=LEARNING_RATE,
    loss=LOSS,
    model_name=MODEL_NAME,
    max_epochs=MAX_EPOCHS,
    optimizer=OPTIMIZER,
    patch_size=PATCH_SIZE,
    scheduler=SCHEDULER,
    sw_batch_size=SW_BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
)

trainer = pl.Trainer(
    accelerator=ACCELERATOR,
    devices=DEVICES,
    precision=PRECISION if ACCELERATOR != "cpu" else "32-true",
    max_epochs=MAX_EPOCHS,
    logger=False,
    enable_checkpointing=False,
    fast_dev_run=FAST_DEV_RUN,
    deterministic=True,
)

trainer.fit(module, train_dataloaders=train_dl)

GLOBAL_THR = calibrate_global_threshold(module, train_df)

predictions, _pred_df = predict(module)

_flat_preds = []
for p in predictions:
    if isinstance(p, torch.Tensor):
        if p.ndim == 2:
            _flat_preds.append(p)
        elif p.ndim == 3:
            for i in range(p.shape[0]):
                _flat_preds.append(p[i])
        else:
            raise ValueError(f"Unexpected pred tensor shape: {tuple(p.shape)}")
    elif isinstance(p, (list, tuple)):
        for t in p:
            _flat_preds.append(t)
    else:
        raise TypeError(f"Unexpected prediction type: {type(p)}")
predictions = _flat_preds
assert len(predictions) == len(_pred_df), (len(predictions), len(_pred_df))




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1458038183.py in <cell line: 0>()
     33 )
     34 
---> 35 trainer.fit(module, train_dataloaders=train_dl)
     36 
     37 GLOBAL_THR = calibrate_global_threshold(module, train_df)

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
   1053                 self._run_sanity_check()
   1054             with torch.autograd.set_detect_anomaly(self._detect_anomaly):
-> 1055                 self.fit_loop.run()
   1056             return None
   1057         raise RuntimeError(f"Unexpected state {self.state}")

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py in run(self)
    214             try:
    215                 self.on_advance_start()
--> 216                 self.advance()
    217                 self.on_advance_end()
    218             except StopIteration:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py in advance(self)
    456         with self.trainer.profiler.profile("run_training_epoch"):
    457             assert self._data_fetcher is not None
--> 458             self.epoch_loop.run(self._data_fetcher)
    459 
    460     def on_advance_end(self) -> None:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py in run(self, data_fetcher)
    150         while not self.done:
    151             try:
--> 152                 self.advance(data_fetcher)
    153                 self.on_advance_end(data_fetcher)
    154             except StopIteration:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py in advance(self, data_fetcher)
    308         else:
    309             dataloader_iter = None
--> 310             batch, _, __ = next(data_fetcher)
    311             # TODO: we should instead use the batch_idx returned by the fetcher, however, that will require saving the
    312             # fetcher state so that the batch_idx is correct after restarting

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py in __next__(self)
    132         elif not self.done:
    133             # this will run only when no pre-fetching was done.
--> 134             batch = super().__next__()
    135         else:
    136             # the iterator is empty

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py in __next__(self)
     59         self._start_profiler()
     60         try:
---> 61             batch = next(self.iterator)
     62         except StopIteration:
     63             self.done = True

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py in __next__(self)
    339     def __next__(self) -> _ITERATOR_RETURN:
    340         assert self._iterator is not None
--> 341         out = next(self._iterator)
    342         if isinstance(self._iterator, _Sequential):
    343             return out

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py in __next__(self)
     76         for i in range(n):
     77             try:
---> 78                 out[i] = next(self.iterators[i])
     79             except StopIteration:
     80                 self._consumed[i] = True

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/1154500561.py in __getitem__(self, idx)
      8     def __getitem__(self, idx: int):
      9         r = self.df.iloc[idx]
---> 10         vol = np.load(r["volume_npy"]).astype(np.float32)  # (Z,H,W)
     11         y = np.load(r["label_npy"]).astype(np.uint8)  # (H,W)
     12         m = np.load(r["mask_npy"]).astype(np.uint8)  # (H,W)

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vesuvius_prepared/prepared_npy/down_0.1/train/2/volume.npy'

## === cell 16
def fast_rle(
    prediction_resized: np.ndarray, threshold: float, debug_print: bool = False
) -> str:
    flat = prediction_resized.astype(np.float32).reshape(-1, order="F")
    flat = (flat > threshold).astype(np.uint8)

    if flat.size == 0:
        return ""

    padded = np.concatenate([[0], flat, [0]])
    changes = np.diff(padded)

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    lengths = ends - starts

    if debug_print:
        print("prediction_resized", ln.lovely(prediction_resized))
        print("flat", ln.lovely(flat))
        print("starts0", ln.lovely(starts))
        print("lengths", ln.lovely(lengths))

    if len(starts) == 0:
        return ""

    starts_1 = starts + 1  # 1-indexed
    rle = np.column_stack([starts_1, lengths]).reshape(-1)
    return " ".join(map(str, rle.tolist()))




## === cell 17
sample_sub_path = COMPETITION_DATA_DIR / "sample_submission.csv"
if not sample_sub_path.exists():
    sample_sub_path = INPUT_DIR / "sample_submission.csv"
sample_submission = pd.read_csv(sample_sub_path)

submission_df = sample_submission.copy()
submission_df["Predicted"] = ""

pred_map = {}
for frag_id, prediction in zip(
    _pred_df["fragmet_id"].astype(str).tolist(), predictions
):
    pred_map[str(frag_id)] = prediction

predictions_rle = []
for frag_id in submission_df["Id"].astype(str).tolist():
    pred_tensor = pred_map.get(str(frag_id), None)
    if pred_tensor is None:
        predictions_rle.append("")
        continue

    mask_png_path = COMPETITION_DATA_DIR / "test" / str(frag_id) / "mask.png"
    if not mask_png_path.exists():
        alt = INPUT_DIR / "test" / str(frag_id) / "mask.png"
        mask_png_path = alt if alt.exists() else mask_png_path

    mask_img = Image.open(mask_png_path).convert("1")
    mask_full = np.array(mask_img, dtype=np.uint8)

    pred = pred_tensor.detach().float().cpu().numpy()  # (h,w) at downsampled resolution
    if pred.ndim != 2:
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            predictions_rle.append("")
            continue

    mask_small_img = mask_img.resize(
        (pred.shape[1], pred.shape[0]), resample=Image.NEAREST
    ).convert("1")
    mask_small = np.array(mask_small_img, dtype=np.uint8)

    pred = pred * mask_small.astype(np.float32)

    pred_img = Image.fromarray(pred.astype(np.float32), mode="F")
    pred_resized = np.array(
        pred_img.resize(mask_img.size, resample=Image.BILINEAR), dtype=np.float32
    )
    pred_resized = pred_resized * mask_full.astype(np.float32)

    thr = float(np.clip(GLOBAL_THR, 0.0, 1.0))

    try:
        prediction_rle = fast_rle(pred_resized, thr, debug_print=False)
    except Exception:
        prediction_rle = ""

    predictions_rle.append(prediction_rle)

submission_df["Predicted"] = predictions_rle

submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
submission_df

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2205284253.py in <cell line: 0>()
      9 pred_map = {}
     10 for frag_id, prediction in zip(
---> 11     _pred_df["fragmet_id"].astype(str).tolist(), predictions
     12 ):
     13     pred_map[str(frag_id)] = prediction

NameError: name '_pred_df' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Expected 2 indices in the submission DataFrame, but got 6 indices.

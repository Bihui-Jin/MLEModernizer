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

0.1280975430645314

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

KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"

COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"
if not COMPETITION_DATA_DIR.exists():
    alt = (
        INPUT_DIR
        / "vesuvius-challenge-ink-detection"
        / "vesuvius-challenge-ink-detection"
    )
    if alt.exists():
        COMPETITION_DATA_DIR = alt

print("COMPETITION_DATA_DIR:", COMPETITION_DATA_DIR)
print("Exists:", COMPETITION_DATA_DIR.exists())
print("Contents sample:", sorted([p.name for p in COMPETITION_DATA_DIR.iterdir()])[:10])



## === cell 1
import importlib


def _try_import(name):
    try:
        return importlib.import_module(name)
    except Exception as e:
        return None


assert (
    _try_import("monai") is not None
), "monai is required but not importable in this environment."



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/2222556937.py in <cell line: 0>()
     13 # monai is required and should be available (per installed packages list)
     14 assert (
---> 15     _try_import("monai") is not None
     16 ), "monai is required but not importable in this environment."
     17 

AssertionError: monai is required but not importable in this environment.

## === cell 2
from typing import Tuple

TRAIN_DATA_CSV_PATH = Path("train.csv")
TEST_DATA_CSV_PATH = Path("test.csv")

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

import monai
from monai.data import CSVDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.visualize import matshow3d
from torchmetrics import Dice, MetricCollection
from tqdm.auto import tqdm




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/2532934551.py in <cell line: 0>()
     11 import torch
     12 
---> 13 import monai
     14 from monai.data import CSVDataset, DataLoader
     15 from monai.inferers import sliding_window_inference

ModuleNotFoundError: No module named 'monai'

## === cell 4
def lovely_array(a):
    a = np.asarray(a)
    return f"shape={a.shape}, dtype={a.dtype}, min={a.min() if a.size else 'NA'}, max={a.max() if a.size else 'NA'}"




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

        if (
            mask_npy_path.exists()
            and vol_npy_path.exists()
            and ((not train) or Path(row.label_npy).exists())
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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1083073883.py in <cell line: 0>()
     78 
     79 
---> 80 save_data_as_npy(train_df, DOWNSAMPLING, train=True)
     81 save_data_as_npy(test_df, DOWNSAMPLING, train=False)
     82 

/tmp/ipykernel_55/1083073883.py in save_data_as_npy(df, downsampling, train)
     49 
     50 def save_data_as_npy(df: pd.DataFrame, downsampling: float, train: bool = True):
---> 51     for row in tqdm(
     52         df.itertuples(index=False),
     53         total=len(df),

NameError: name 'tqdm' is not defined

## === cell 7
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
        self.keys = ("volume_npy", "mask_npy", "label_npy")
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.predict_transform = self._init_predict_transform()

    def _init_train_transform(self):
        return monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys="volume_npy"),
                monai.transforms.LoadImaged(
                    keys=("mask_npy", "label_npy"), ensure_channel_first=True
                ),
                monai.transforms.RandWeightedCropd(
                    keys=self.keys,
                    spatial_size=self.hparams.patch_size,
                    num_samples=self.hparams.num_samples,
                    w_key="mask_npy",
                ),
                monai.transforms.RandFlipd(keys=self.keys, prob=0.5, spatial_axis=0),
                monai.transforms.RandFlipd(keys=self.keys, prob=0.5, spatial_axis=1),
            ]
        )

    def _init_val_transform(self):
        return monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys="volume_npy"),
                monai.transforms.LoadImaged(
                    keys=("mask_npy", "label_npy"), ensure_channel_first=True
                ),
            ]
        )

    def _init_predict_transform(self):
        return monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys="volume_npy"),
                monai.transforms.LoadImaged(keys="mask_npy", ensure_channel_first=True),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_val_df = self.df[self.df.stage == "train"].reset_index(drop=True)
            train_df = train_val_df[
                train_val_df.fragment_id != self.hparams.val_fragment_id
            ].reset_index(drop=True)
            val_df = train_val_df[
                train_val_df.fragment_id == self.hparams.val_fragment_id
            ].reset_index(drop=True)

            self.train_dataset = self._dataset(train_df, self.train_transform)
            self.val_dataset = self._dataset(val_df, self.val_transform)

            print(f"# train: {len(self.train_dataset)}")
            print(f"# val: {len(self.val_dataset)}")

        if stage == "predict" or stage is None:
            predict_df = self.df[self.df.stage == "test"].reset_index(drop=True)
            self.predict_dataset = self._dataset(predict_df, self.predict_transform)
            print(f"# predict: {len(self.predict_dataset)}")

    def _dataset(self, df, transform):
        return CSVDataset(src=df, transform=transform)

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def predict_dataloader(self):
        return self._dataloader(self.predict_dataset)

    def _dataloader(self, dataset, train=False):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
        )




## === cell 8
def visualize_dataloaders(dataloaders, train=True, max_batches=1):
    for stage, dataloader in dataloaders.items():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            volumes = batch["volume_npy"]
            masks = batch["mask_npy"]
            labels = batch["label_npy"] if train and "label_npy" in batch else masks

            for volume, mask, label in zip(volumes, masks, labels):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                plt.suptitle(f"stage: {stage}, batch: {batch_idx}")

                for idx, image in enumerate((volume, mask, label)):
                    matshow3d(
                        volume=image,
                        fig=axes[idx],
                        title=f"{list(image.shape)}, {float(image.min())}, {float(image.max())}",
                        vmin=0.0,
                        vmax=1.0,
                        every_n=2,
                        fill_value=1.0,
                        margin=4,
                        cmap="gray",
                    )




## === cell 9
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
        self.loss = self._init_loss()
        self.metrics = self._init_metrics()

    def _init_model(self):
        if self.hparams.model_name == "UNet":
            return monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=self.hparams.num_z_slices,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                dropout=self.hparams.dropout,
            )
        raise ValueError(f"{self.hparams.model_name} is not implemented")

    def _init_loss(self):
        if self.hparams.loss == "Dice":
            loss = monai.losses.DiceLoss(sigmoid=True)
        elif self.hparams.loss == "Jaccard":
            loss = monai.losses.DiceLoss(sigmoid=True, jaccard=True)
        elif self.hparams.loss == "DiceCELoss":
            loss = monai.losses.DiceCELoss(sigmoid=True)
        else:
            raise ValueError(f"{self.hparams.loss} is not implemented")
        return monai.losses.MaskedLoss(loss)

    def _init_metrics(self):
        metric_collection = MetricCollection({"dice": Dice()})
        return torch.nn.ModuleDict(
            {
                "train_metrics": metric_collection.clone(prefix="train_"),
                "val_metrics": metric_collection.clone(prefix="val_"),
            }
        )

    def configure_optimizers(self):
        optimizer = self._init_optimizer()
        scheduler = self._init_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def _init_optimizer(self):
        if self.hparams.optimizer == "AdamW":
            return torch.optim.AdamW(
                params=self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        raise ValueError(f"{self.hparams.optimizer} is not implemented")

    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=self.hparams.eta_min,
            )
        raise ValueError(f"{self.hparams.scheduler} is not implemented")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        outputs = self._forward_pass(batch, "predict")
        return outputs.sigmoid().squeeze(0).squeeze(0)  # (H,W)

    def _shared_step(self, batch, stage):
        outputs, labels, masks = self._forward_pass(batch, stage)
        loss = self.loss(outputs, labels, masks)
        self.metrics[f"{stage}_metrics"](outputs, labels)
        self._log(loss, stage, batch_size=len(outputs))
        return loss

    def _forward_pass(self, batch, stage):
        volumes = batch["volume_npy"].as_tensor()

        if stage == "train":
            outputs = self(volumes)
        else:
            outputs = sliding_window_inference(
                inputs=volumes,
                roi_size=self.hparams.patch_size,
                sw_batch_size=self.hparams.sw_batch_size,
                predictor=self,
            )

        if stage == "predict":
            return outputs
        labels = batch["label_npy"].long()
        masks = batch["mask_npy"]
        return outputs, labels, masks

    def _log(self, loss, stage, batch_size):
        self.log(f"{stage}_loss", loss, batch_size=batch_size, prog_bar=True)
        self.log_dict(
            self.metrics[f"{stage}_metrics"], batch_size=batch_size, prog_bar=True
        )




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
    monai.utils.set_determinism(seed)
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
        precision=precision,
        strategy="ddp" if devices > 1 else "auto",
        enable_model_summary=False,
    )

    trainer.fit(module, datamodule=data_module)
    return module, trainer


module, trainer = train()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1047880414.py in <cell line: 0>()
     70 
     71 
---> 72 module, trainer = train()
     73 

/tmp/ipykernel_55/1047880414.py in train(accelerator, batch_size, data_csv_path, devices, dropout, eta_min, fast_dev_run, learning_rate, loss, model_name, max_epochs, num_workers, num_samples, num_z_slices, optimizer, overfit_batches, patch_size, precision, scheduler, seed, sw_batch_size, val_fragment_id, weight_decay)
     24     weight_decay=WEIGHT_DECAY,
     25 ):
---> 26     monai.utils.set_determinism(seed)
     27     pl.seed_everything(seed, workers=True)
     28 

NameError: name 'monai' is not defined

## === cell 12
try:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    cols = [
        c
        for c in ["epoch", "train_loss", "val_loss", "train_dice", "val_dice"]
        if c in metrics.columns
    ]
    if "epoch" in cols:
        plot_df = metrics[cols].groupby("epoch").mean(numeric_only=True)
        sns.relplot(
            data=plot_df.reset_index(),
            x="epoch",
            y=[c for c in plot_df.columns],
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
    monai.utils.set_determinism(seed)
    pl.seed_everything(seed, workers=True)

    data_module = VesuvisDataModule(
        batch_size=batch_size,
        data_csv_path=data_csv_path,
        num_workers=num_workers,
        num_samples=num_samples,
        patch_size=patch_size,
        val_fragment_id=val_fragment_id,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    predictions = trainer.predict(module, datamodule=data_module)
    return predictions


predictions = predict(module)
print("Num predictions:", len(predictions))




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3534015089.py in <cell line: 0>()
     37 
     38 
---> 39 predictions = predict(module)
     40 print("Num predictions:", len(predictions))
     41 

NameError: name 'module' is not defined

## === cell 14
def rle(img: np.ndarray) -> str:
    """
    img: 2D numpy array, 1 - mask, 0 - background
    Returns run length as space-delimited string.
    """
    pixels = img.astype(np.uint8).ravel(order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def plot_image(image, title):
    fig = plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()




## === cell 15
submission_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")
assert {"Id", "Predicted"}.issubset(submission_df.columns)

test_df_local = test_df.copy()
test_df_local["Id"] = test_df_local["fragment_id"].astype(str)

pred_by_id = {
    row_id: pred.detach().cpu().numpy()
    for row_id, pred in zip(test_df_local["Id"].tolist(), predictions)
}

predictions_rle = []
for frag_id in submission_df["Id"].astype(str).tolist():
    row = test_df_local.loc[test_df_local["Id"] == frag_id].iloc[0]
    pred = pred_by_id[frag_id]  # expected shape (H,W) float in [0,1]
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
    pred_bin = (pred_masked > THRESHOLD).astype(np.uint8)

    predictions_rle.append(rle(pred_bin))

submission_df["Predicted"] = predictions_rle
submission_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
print(
    "submission.csv exists:",
    Path("submission.csv").exists(),
    "size:",
    Path("submission.csv").stat().st_size,
)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3467382815.py in <cell line: 0>()
     10 pred_by_id = {
     11     row_id: pred.detach().cpu().numpy()
---> 12     for row_id, pred in zip(test_df_local["Id"].tolist(), predictions)
     13 }
     14 

NameError: name 'predictions' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Expected 2 indices in the submission DataFrame, but got 6 indices.

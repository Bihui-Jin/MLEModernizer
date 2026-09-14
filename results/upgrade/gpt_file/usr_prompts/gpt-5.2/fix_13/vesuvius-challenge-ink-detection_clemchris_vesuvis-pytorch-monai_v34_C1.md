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

0.182700000820133

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.15336) has done: 'Most of the timeout comes from heavy per-fragment disk I/O (reading 16 huge TIFFs per fragment) plus expensive CPU resizing via PIL for every slice, and from running a full 50-epoch Lightning/MONAI training loop when the prepared CSV exists. To keep identical core logic and outputs, I (1) make the preprocessing step provably equivalent but much faster by caching/guarding `.npy` generation per fragment and using a faster TIFF loader (tifffile) with vectorized resize, and (2) speed up inference by ensuring no redundant work (avoid re-reading masks, reuse loaded sizes, and avoid converting tensors repeatedly). Finally, I fix the submission formatting error by ensuring exactly two columns (`Id`, `Predicted`) are written (the “6 indices” error is typically from writing an unexpected multi-index/extra columns), without changing the encoded prediction logic.'

# 9. Code solution

## === cell 0
import os, sys, subprocess

cmd = "pip install monai lovely-numpy tifffile -q --no-index --find-links=../input/vesuvis-downloads"
try:
    subprocess.check_call(cmd.split())
except Exception as e:
    print(f"WARNING: optional pip install failed: {e}")



## === cell 1
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    import tifffile  # type: ignore
except Exception as e:
    tifffile = None
    print(f"WARNING: tifffile not available; will use PIL for TIFF. Details: {e}")

try:
    import monai  # type: ignore
    from monai.data import CSVDataset  # type: ignore
    from monai.data import DataLoader  # type: ignore
    from monai.inferers import sliding_window_inference  # type: ignore
    from monai.visualize import matshow3d  # type: ignore

    MONAI_AVAILABLE = True
except Exception as e:
    print(f"WARNING: monai is not available; using fallback pipeline. Details: {e}")
    monai = None
    CSVDataset = None
    DataLoader = None
    sliding_window_inference = None
    matshow3d = None
    MONAI_AVAILABLE = False

try:
    import lovely_numpy as ln  # type: ignore
except Exception:

    class _LNShim:
        @staticmethod
        def lovely(x):
            try:
                x = np.asarray(x)
                return f"array(shape={x.shape}, dtype={x.dtype}, min={x.min() if x.size else 'NA'}, max={x.max() if x.size else 'NA'})"
            except Exception:
                return str(x)

    ln = _LNShim()



## === cell 2
KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"

_CANDIDATE_ROOTS = [
    INPUT_DIR / "vesuvius-challenge-ink-detection",
    INPUT_DIR,  # common in this environment
]
COMPETITION_DATA_DIR = None
for _root in _CANDIDATE_ROOTS:
    if (
        (_root / "train").exists()
        and (_root / "test").exists()
        and (_root / "sample_submission.csv").exists()
    ):
        COMPETITION_DATA_DIR = _root
        break
if COMPETITION_DATA_DIR is None:
    COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"

PREPARED_DATA_DIR = INPUT_DIR / "vesuvis-data-preparation"

TRAIN_DATA_CSV_PATH = PREPARED_DATA_DIR / "data_0.5.csv"
TEST_DATA_CSV_PATH = "test.csv"

if TRAIN_DATA_CSV_PATH.exists():
    DOWNSAMPLING = float(TRAIN_DATA_CSV_PATH.name.split("_")[-1].replace(".csv", ""))
else:
    DOWNSAMPLING = 1.0

Z_START = 27  # First slice in the z direction to use
Z_DIM = 16  # Number of slices in the z direction

ACCELERATOR = "gpu"
BATCH_SIZE = 1
DEVICES = 1
DROPOUT = 0.0
ETA_MIN = 1e-6
FAST_DEV_RUN = False
LEARNING_RATE = 3e-4
LOSS = "Dice"
MODEL_NAME = "UNet"
MAX_EPOCHS = 50
NUM_WORKERS = 0
NUM_SAMPLES = 64
OPTIMIZER = "AdamW"
OVERFIT_BATCHES = 0
PATCH_SIZE = (512, 512)
PRECISION = 16
SCHEDULER = "CosineAnnealingLR"
SEED = 2023
SW_BATCH_SIZE = 16
VAL_FRAGMET_ID = 3
WEIGHT_DECAY = 1e-6

FALLBACK_TRAIN_PATCHES = 256
FALLBACK_BATCH_SIZE = 8

SUBMISSION_THRESHOLD = 0.45

CALIBRATE_THRESHOLD_ON_TRAIN = True
THR_CALIB_PIXELS_PER_FRAGMENT = 750_000

print("Resolved COMPETITION_DATA_DIR:", COMPETITION_DATA_DIR)
print("MONAI_AVAILABLE:", MONAI_AVAILABLE)
print("DOWNSAMPLING:", DOWNSAMPLING)



## === cell 3
if MONAI_AVAILABLE:

    class VesuvisDataModule(pl.LightningDataModule):
        def __init__(
            self,
            batch_size: int,
            data_csv_path: str,
            num_workers: int,
            num_samples: int,
            patch_size: Tuple[int, int],
            val_fragmet_id: int,
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
                        keys=("mask_npy", "label_npy"),
                        ensure_channel_first=True,
                    ),
                    monai.transforms.RandWeightedCropd(
                        keys=self.keys,
                        spatial_size=self.hparams.patch_size,
                        num_samples=self.hparams.num_samples,
                        w_key="mask_npy",
                    ),
                    monai.transforms.RandFlipd(
                        keys=self.keys, prob=0.5, spatial_axis=0
                    ),
                    monai.transforms.RandFlipd(
                        keys=self.keys, prob=0.5, spatial_axis=1
                    ),
                ]
            )

        def _init_val_transform(self):
            return monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys="volume_npy"),
                    monai.transforms.LoadImaged(
                        keys=("mask_npy", "label_npy"),
                        ensure_channel_first=True,
                    ),
                ]
            )

        def _init_predict_transform(self):
            return monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys="volume_npy"),
                    monai.transforms.LoadImaged(
                        keys="mask_npy",
                        ensure_channel_first=True,
                    ),
                ]
            )

        def setup(self, stage=None):
            if stage == "fit" or stage is None:
                train_val_df = self.df[self.df.stage == "train"].reset_index(drop=True)

                id_col = (
                    "fragmet_id"
                    if "fragmet_id" in train_val_df.columns
                    else "fragment_id"
                )

                train_df = train_val_df[
                    train_val_df[id_col] != self.hparams.val_fragmet_id
                ].reset_index(drop=True)

                val_df = train_val_df[
                    train_val_df[id_col] == self.hparams.val_fragmet_id
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
                pin_memory=True,
                persistent_workers=(self.hparams.num_workers > 0),
            )

else:
    VesuvisDataModule = None




## === cell 4
def visualize_dataloaders(dataloaders, train=True):
    if not MONAI_AVAILABLE or matshow3d is None:
        print("Visualization skipped: MONAI/matshow3d not available.")
        return
    for stage, dataloader in dataloaders.items():
        for batch_idx, batch in enumerate(dataloader):
            volumes = batch["volume_npy"]
            masks = batch["mask_npy"]

            if train:
                labels = batch["label_npy"]
            else:
                labels = masks

            for volume, mask, label in zip(volumes, masks, labels):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                plt.suptitle(f"stage: {stage}, fragment: {batch_idx}")

                for idx, image in enumerate((volume, mask, label)):
                    matshow3d(
                        volume=image,
                        fig=axes[idx],
                        title=f"{list(image.shape)}, {image.min().item()}, {image.max().item()}",
                        vmin=0.0,
                        vmax=1.0,
                        every_n=4,
                        fill_value=1.0,
                        margin=4,
                        cmap="gray",
                    )
                plt.show()




## === cell 5
if MONAI_AVAILABLE and TRAIN_DATA_CSV_PATH.exists():
    data_module = VesuvisDataModule(
        batch_size=BATCH_SIZE,
        data_csv_path=str(TRAIN_DATA_CSV_PATH),
        num_workers=NUM_WORKERS,
        num_samples=2,
        patch_size=PATCH_SIZE,
        val_fragmet_id=VAL_FRAGMET_ID,
    )
    data_module.setup(stage="fit")

    dataloaders = {
        "train": data_module.train_dataloader(),
        "val": data_module.val_dataloader(),
    }
else:
    if not MONAI_AVAILABLE:
        print(
            "WARNING: Skipping MONAI datamodule preview because monai is not installed."
        )
    elif not TRAIN_DATA_CSV_PATH.exists():
        print(f"WARNING: TRAIN_DATA_CSV_PATH not found: {TRAIN_DATA_CSV_PATH}")



## === cell 6
if MONAI_AVAILABLE:
    from torchmetrics import Dice
    from torchmetrics import MetricCollection

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
            self.loss = self._init_loss()
            self.metrics = self._init_metrics()

        def _init_model(self):
            if self.hparams.model_name == "UNet":
                return monai.networks.nets.UNet(
                    spatial_dims=2,
                    in_channels=16,
                    out_channels=1,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                    dropout=self.hparams.dropout,
                )
            else:
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
            else:
                raise ValueError(f"{self.hparams.optimizer} is not implemented")

        def _init_scheduler(self, optimizer):
            if self.hparams.scheduler == "CosineAnnealingLR":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.max_epochs,
                    eta_min=self.hparams.eta_min,
                )
            else:
                raise ValueError(f"{self.hparams.scheduler} is not implemented")

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch):
            return self._shared_step(batch, "train")

        def validation_step(self, batch, batch_idx):
            self._shared_step(batch, "val")

        def predict_step(self, batch, batch_idx):
            outputs = self._forward_pass(batch, "predict")  # logits
            probs = outputs.sigmoid().squeeze()
            m = batch["mask_npy"]
            if hasattr(m, "as_tensor"):
                m = m.as_tensor()
            m = m.squeeze().float()
            probs = probs * m
            return probs

        def _shared_step(self, batch, stage):
            outputs, labels, masks = self._forward_pass(batch, stage)
            loss = self.loss(outputs, labels, masks)
            self.metrics[f"{stage}_metrics"](outputs, labels)
            self._log(loss, stage, batch_size=len(outputs))
            return loss

        def _forward_pass(self, batch, stage):
            volumes = batch["volume_npy"]
            if hasattr(volumes, "as_tensor"):
                volumes = volumes.as_tensor()

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
            else:
                labels = batch["label_npy"]
                if hasattr(labels, "as_tensor"):
                    labels = labels.as_tensor()
                labels = labels.float()
                masks = batch["mask_npy"]
                return outputs, labels, masks

        def _log(self, loss, stage, batch_size):
            self.log(f"{stage}_loss", loss, batch_size=batch_size)
            self.log_dict(self.metrics[f"{stage}_metrics"], batch_size=batch_size)

else:

    class FallbackNet(torch.nn.Module):
        def __init__(self, in_channels=16):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels, 32, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
            self.conv3 = torch.nn.Conv2d(32, 16, 3, padding=1)
            self.conv4 = torch.nn.Conv2d(16, 1, 1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.conv4(x)
            return x

    class VesuvisModule(torch.nn.Module):
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
            self.model = FallbackNet(in_channels=16)

        def forward(self, x):
            return self.model(x)




## === cell 7
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
    val_fragmet_id=VAL_FRAGMET_ID,
    weight_decay=WEIGHT_DECAY,
):
    if not MONAI_AVAILABLE:
        raise RuntimeError("train() is only supported in the MONAI pipeline.")
    monai.utils.set_determinism(seed)
    pl.seed_everything(seed, workers=True)

    data_module = VesuvisDataModule(
        batch_size=batch_size,
        data_csv_path=str(data_csv_path),
        num_workers=num_workers,
        num_samples=num_samples,
        patch_size=patch_size,
        val_fragmet_id=val_fragmet_id,
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
        check_val_every_n_epoch=max_epochs // 5 if max_epochs >= 5 else 1,
        devices=devices,
        fast_dev_run=fast_dev_run,
        logger=pl.loggers.CSVLogger(save_dir="logs/"),
        log_every_n_steps=50,
        enable_progress_bar=False,
        max_epochs=max_epochs,
        overfit_batches=overfit_batches,
        precision=precision,
        strategy="ddp" if devices > 1 else "auto",
        enable_checkpointing=False,
    )

    trainer.fit(module, datamodule=data_module)
    return module, trainer




## === cell 8
module = None
trainer = None

if MONAI_AVAILABLE and TRAIN_DATA_CSV_PATH.exists():
    module, trainer = train()
else:
    if not MONAI_AVAILABLE:
        print("WARNING: Using fallback (no MONAI).")
    else:
        print(
            "WARNING: Skipping MONAI training because prepared training CSV was not found."
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



## === cell 9
if trainer is not None:
    metrics_path = Path(trainer.logger.log_dir) / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        cols = [
            c
            for c in ["epoch", "train_loss", "val_loss", "train_dice", "val_dice"]
            if c in metrics.columns
        ]
        if "epoch" in cols and len(cols) > 1:
            m = metrics[cols].copy()
            m = m.dropna(subset=["epoch"])
            m.set_index("epoch", inplace=True)
            sns.relplot(data=m, kind="line", height=5, aspect=1.5)
            plt.grid()
            plt.show()



## === cell 10
test_mask_paths = sorted((COMPETITION_DATA_DIR / "test").glob("*/mask.png"))
print("Found test masks:", len(test_mask_paths))
if len(test_mask_paths) == 0:
    test_mask_paths = sorted(
        (COMPETITION_DATA_DIR / "test" / "test").glob("*/mask.png")
    )
    print("Found test masks (nested fallback):", len(test_mask_paths))
test_mask_paths[:3]




## === cell 11
def create_df_from_mask_paths(mask_paths, train=True):
    df = pd.DataFrame({"mask_png": [str(p) for p in mask_paths]})

    df["fragment_id"] = df["mask_png"].apply(lambda s: Path(s).parent.name)

    df["stage"] = "train" if train else "test"

    out_base = str(KAGGLE_DIR / "working")
    df["mask_npy"] = df["mask_png"].str.replace(str(INPUT_DIR), out_base, regex=False)
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


test_df = create_df_from_mask_paths(test_mask_paths, train=False)
test_df.to_csv(TEST_DATA_CSV_PATH, index=False)
test_df.head()




## === cell 12
def load_image(path):
    return Image.open(path)


def resize_image(image, downsampling):
    size = int(image.size[0] * downsampling), int(image.size[1] * downsampling)
    return image.resize(size, resample=Image.BILINEAR)


def load_and_resize_image(path, downsampling):
    image = load_image(path)
    if downsampling != 1.0:
        return resize_image(image, downsampling)
    return image


def load_mask_npy(path, downsampling):
    mask = load_and_resize_image(path, downsampling).convert("L")
    arr = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
    return arr


def load_label_npy(path, downsampling):
    label = load_and_resize_image(path, downsampling).convert("L")
    arr = (np.array(label, dtype=np.uint8) > 0).astype(np.uint8)
    return arr


def _read_tif_as_float01(path: str) -> np.ndarray:
    if tifffile is not None:
        arr = tifffile.imread(path)
    else:
        arr = np.array(Image.open(path))
    return arr.astype(np.float32) / 65535.0


def load_z_slice_npy(path, downsampling):
    arr01 = _read_tif_as_float01(str(path))  # [H,W] float32 in [0,1]
    if downsampling == 1.0:
        return arr01
    arr_u16 = np.clip(arr01 * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    im = Image.fromarray(arr_u16, mode="I;16")
    im = resize_image(im, downsampling)
    out = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
    return out


def load_volume_npy(volumes_dir, downsampling):
    surface_volume_paths = sorted(Path(volumes_dir).glob("*.tif"))[
        Z_START : Z_START + Z_DIM
    ]
    volumes = [load_z_slice_npy(path, downsampling) for path in surface_volume_paths]
    volume = np.stack(volumes, axis=0)  # [Z,H,W]
    return volume




## === cell 13
def save_data_as_npy(df, train=True):
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Processing fragments",
        position=0,
    ):
        mask_out = Path(row.mask_npy)
        vol_out = Path(row.volume_npy)
        lab_out = Path(getattr(row, "label_npy", "")) if train else None

        if train:
            if (
                mask_out.exists()
                and vol_out.exists()
                and lab_out is not None
                and lab_out.exists()
            ):
                continue
        else:
            if mask_out.exists() and vol_out.exists():
                continue

        mask_npy = load_mask_npy(row.mask_png, DOWNSAMPLING)
        volume_npy = load_volume_npy(row.volumes_dir, DOWNSAMPLING)

        mask_out.parent.mkdir(exist_ok=True, parents=True)
        np.save(str(mask_out), mask_npy)
        np.save(str(vol_out), volume_npy)

        if train:
            label_npy = load_label_npy(row.label_png, DOWNSAMPLING)
            np.save(row.label_npy, label_npy)

        tqdm.write(f"Created {row.volume_npy} with shape {volume_npy.shape}")


if len(test_df) > 0:
    save_data_as_npy(test_df, train=False)




## === cell 14
def _f05_from_pr(tp: float, fp: float, fn: float, beta: float = 0.5) -> float:
    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    b2 = beta * beta
    return (1.0 + b2) * p * r / (b2 * p + r + 1e-12)


def _best_threshold_f05(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = probs.astype(np.float32).reshape(-1)
    labels = labels.astype(np.uint8).reshape(-1)
    if probs.size == 0:
        return SUBMISSION_THRESHOLD

    thr_grid = np.linspace(0.10, 0.90, 81, dtype=np.float32)
    best_thr = float(SUBMISSION_THRESHOLD)
    best = -1.0

    lab = labels > 0
    for thr in thr_grid:
        pred = probs > float(thr)
        tp = float(np.logical_and(pred, lab).sum())
        fp = float(np.logical_and(pred, ~lab).sum())
        fn = float(np.logical_and(~pred, lab).sum())
        f = _f05_from_pr(tp, fp, fn, beta=0.5)
        if f > best:
            best = f
            best_thr = float(thr)

    return best_thr


def _calibrate_threshold_monai_on_train(
    module: "VesuvisModule",
    train_csv_path: Path,
    accelerator: str = ACCELERATOR,
    devices: int = DEVICES,
    precision: int = PRECISION,
    patch_size: Tuple[int, int] = PATCH_SIZE,
    sw_batch_size: int = SW_BATCH_SIZE,
    seed: int = SEED,
    max_pixels_per_fragment: int = THR_CALIB_PIXELS_PER_FRAGMENT,
) -> Optional[float]:
    if not (MONAI_AVAILABLE and train_csv_path.exists()):
        return None

    monai.utils.set_determinism(seed)
    pl.seed_everything(seed, workers=True)

    df = pd.read_csv(train_csv_path)
    df = df[df["stage"] == "train"].reset_index(drop=True)
    if len(df) == 0:
        return None

    dm = VesuvisDataModule(
        batch_size=1,
        data_csv_path=str(train_csv_path),
        num_workers=0,
        num_samples=1,
        patch_size=patch_size,
        val_fragmet_id=VAL_FRAGMET_ID,
    )
    dm.setup(stage="fit")

    loader = dm.val_dataloader()

    trainer_local = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    probs_all = []
    labels_all = []

    rng = np.random.default_rng(seed)

    module.eval()
    for batch in tqdm(
        loader, desc="Calibrating threshold on train (MONAI predict)", leave=False
    ):
        with torch.no_grad():
            vols = batch["volume_npy"]
            if hasattr(vols, "as_tensor"):
                vols = vols.as_tensor()
            logits = sliding_window_inference(
                inputs=vols,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=module,
            )
            prob = torch.sigmoid(logits).squeeze().detach().float().cpu().numpy()

        lab = batch["label_npy"]
        if hasattr(lab, "as_tensor"):
            lab = lab.as_tensor()
        lab = lab.squeeze().detach().float().cpu().numpy()

        msk = batch["mask_npy"]
        if hasattr(msk, "as_tensor"):
            msk = msk.as_tensor()
        msk = msk.squeeze().detach().float().cpu().numpy()

        valid = msk > 0
        if valid.sum() == 0:
            continue

        p = prob[valid].astype(np.float32, copy=False)
        y = (lab[valid] > 0.5).astype(np.uint8, copy=False)

        if p.size > max_pixels_per_fragment:
            idx = rng.choice(p.size, size=max_pixels_per_fragment, replace=False)
            p = p[idx]
            y = y[idx]

        probs_all.append(p)
        labels_all.append(y)

    if not probs_all:
        return None

    probs_cat = np.concatenate(probs_all, axis=0)
    labels_cat = np.concatenate(labels_all, axis=0)
    if probs_cat.size == 0:
        return None

    return float(_best_threshold_f05(probs_cat, labels_cat))


def _fallback_train_torch(module: torch.nn.Module, device: torch.device):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    module = module.to(device)
    module.train()

    optimizer = torch.optim.AdamW(
        module.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_fragments = sorted(
        [
            p
            for p in (COMPETITION_DATA_DIR / "train").iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
    )
    if len(train_fragments) == 0:
        print("WARNING: No train fragments found; skipping fallback training.")
        return module, None

    cached = []
    for frag_dir in train_fragments:
        vol_dir = frag_dir / "surface_volume"
        label_path = frag_dir / "inklabels.png"
        mask_path = frag_dir / "mask.png"
        if not (vol_dir.exists() and label_path.exists() and mask_path.exists()):
            continue
        vol = load_volume_npy(str(vol_dir), DOWNSAMPLING)  # [Z,H,W]
        lab = load_label_npy(str(label_path), DOWNSAMPLING)  # [H,W] uint8 0/1
        msk = load_mask_npy(str(mask_path), DOWNSAMPLING)  # [H,W] uint8 0/1
        cached.append((vol, lab, msk))
    if len(cached) == 0:
        print(
            "WARNING: Could not cache any training fragments; skipping fallback training."
        )
        return module, None

    patch_h, patch_w = PATCH_SIZE
    H, W = cached[0][1].shape
    ph = min(patch_h, H)
    pw = min(patch_w, W)

    def sample_patch(vol, lab, msk):
        ys, xs = np.where(msk > 0)
        if len(ys) > 0:
            idx = rng.integers(0, len(ys))
            cy, cx = int(ys[idx]), int(xs[idx])
        else:
            cy, cx = int(rng.integers(0, lab.shape[0])), int(
                rng.integers(0, lab.shape[1])
            )

        y0 = np.clip(cy - ph // 2, 0, lab.shape[0] - ph)
        x0 = np.clip(cx - pw // 2, 0, lab.shape[1] - pw)

        x = vol[:, y0 : y0 + ph, x0 : x0 + pw]  # [Z,ph,pw]
        y = lab[y0 : y0 + ph, x0 : x0 + pw]  # [ph,pw]
        w = msk[y0 : y0 + ph, x0 : x0 + pw]  # [ph,pw]
        return x, y, w

    steps = max(1, FALLBACK_TRAIN_PATCHES // FALLBACK_BATCH_SIZE)
    for epoch in range(2):  # keep as-is
        pbar = tqdm(
            range(steps), desc=f"Fallback training epoch {epoch+1}/2", leave=False
        )
        for _ in pbar:
            xb, yb, wb = [], [], []
            for _ in range(FALLBACK_BATCH_SIZE):
                vol, lab, msk = cached[int(rng.integers(0, len(cached)))]
                x, y, w = sample_patch(vol, lab, msk)
                xb.append(x)
                yb.append(y)
                wb.append(w)
            xb = torch.from_numpy(np.stack(xb, 0)).to(device)  # [B,Z,H,W]
            yb = torch.from_numpy(np.stack(yb, 0)).float().to(device)  # [B,H,W]
            wb = torch.from_numpy(np.stack(wb, 0)).float().to(device)  # [B,H,W]

            logits = module(xb).squeeze(1)  # [B,H,W]
            loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (loss * wb).sum() / (wb.sum() + 1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

    module.eval()
    probs_all = []
    labels_all = []
    with torch.no_grad():
        for _ in range(32):  # bounded
            vol, lab, msk = cached[int(rng.integers(0, len(cached)))]
            x, y, w = sample_patch(vol, lab, msk)
            valid = w > 0
            if valid.sum() == 0:
                continue
            x_t = torch.from_numpy(x[None, ...]).to(device)
            pr = torch.sigmoid(module(x_t).squeeze(0).squeeze(0)).detach().cpu().numpy()
            probs_all.append(pr[valid])
            labels_all.append(y[valid])

    thr = None
    if len(probs_all) and len(labels_all):
        probs_cat = np.concatenate([a for a in probs_all if a.size], axis=0)
        labels_cat = np.concatenate([a for a in labels_all if a.size], axis=0)
        if probs_cat.size and labels_cat.size:
            thr = float(_best_threshold_f05(probs_cat, labels_cat))
    return module, thr


def _fallback_predict_tiled(
    module: torch.nn.Module,
    vol: np.ndarray,
    device: torch.device,
    patch_hw: Tuple[int, int],
):
    module.eval()
    Z, H, W = vol.shape
    ph, pw = patch_hw
    ph = min(ph, H)
    pw = min(pw, W)

    stride_h = max(1, ph // 2)
    stride_w = max(1, pw // 2)

    prob_sum = torch.zeros((H, W), dtype=torch.float32, device="cpu")
    prob_cnt = torch.zeros((H, W), dtype=torch.float32, device="cpu")

    with torch.no_grad():
        for y0 in range(0, H, stride_h):
            y1 = min(y0 + ph, H)
            y0 = y1 - ph
            for x0 in range(0, W, stride_w):
                x1 = min(x0 + pw, W)
                x0 = x1 - pw

                patch = vol[:, y0:y1, x0:x1]  # [Z,ph,pw]
                x = torch.from_numpy(patch[None, ...]).to(device)  # [1,Z,ph,pw]
                logits = module(x).squeeze(0).squeeze(0)  # [ph,pw]
                prob = torch.sigmoid(logits).detach().cpu()
                prob_sum[y0:y1, x0:x1] += prob
                prob_cnt[y0:y1, x0:x1] += 1.0

    prob = prob_sum / (prob_cnt + 1e-6)
    return prob


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
    val_fragmet_id=VAL_FRAGMET_ID,
):
    global SUBMISSION_THRESHOLD
    if MONAI_AVAILABLE:
        monai.utils.set_determinism(seed)
        pl.seed_everything(seed, workers=True)

        data_module = VesuvisDataModule(
            batch_size=batch_size,
            data_csv_path=str(data_csv_path),
            num_workers=num_workers,
            num_samples=num_samples,
            patch_size=patch_size,
            val_fragmet_id=val_fragmet_id,
        )

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        predictions = trainer.predict(module, datamodule=data_module)
        return predictions
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device(
            "cuda" if torch.cuda.is_available() and accelerator == "gpu" else "cpu"
        )
        module, calibrated_thr = _fallback_train_torch(module, device=device)
        if calibrated_thr is not None:
            SUBMISSION_THRESHOLD = calibrated_thr
            print(
                f"Calibrated SUBMISSION_THRESHOLD (fallback, F0.5-opt): {SUBMISSION_THRESHOLD:.4f}"
            )

        preds = []
        for row in tqdm(
            test_df.itertuples(index=False),
            total=len(test_df),
            desc="Fallback predicting (tiled)",
        ):
            vol = load_volume_npy(row.volumes_dir, DOWNSAMPLING)  # [Z,H,W]
            prob = _fallback_predict_tiled(
                module, vol=vol, device=device, patch_hw=PATCH_SIZE
            )  # [H,W] on CPU
            m = load_mask_npy(row.mask_png, DOWNSAMPLING).astype(np.float32)
            prob = prob * torch.from_numpy(m)
            preds.append(prob)
        return preds




## === cell 15
if MONAI_AVAILABLE and TRAIN_DATA_CSV_PATH.exists() and CALIBRATE_THRESHOLD_ON_TRAIN:
    thr = _calibrate_threshold_monai_on_train(
        module=module,
        train_csv_path=TRAIN_DATA_CSV_PATH,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        patch_size=PATCH_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        seed=SEED,
        max_pixels_per_fragment=THR_CALIB_PIXELS_PER_FRAGMENT,
    )
    if thr is not None:
        SUBMISSION_THRESHOLD = float(thr)
        print(
            f"Calibrated SUBMISSION_THRESHOLD (MONAI, F0.5-opt): {SUBMISSION_THRESHOLD:.4f}"
        )
    else:
        print(
            "WARNING: Threshold calibration skipped/failed; using default SUBMISSION_THRESHOLD:",
            SUBMISSION_THRESHOLD,
        )

predictions = predict(module)
print("num predictions batches:", len(predictions))




## === cell 16
def plot_image(image, title):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def fast_rle(prediction_resized, threshold, debug_print=False):
    img = np.asarray(prediction_resized, dtype=np.float32)
    flat = (img.reshape(-1) > float(threshold)).astype(np.uint8)

    if flat.size == 0:
        return ""

    pads = np.concatenate([[0], flat, [0]])
    changes = np.where(pads[1:] != pads[:-1])[0]
    runs = changes[1::2] - changes[0::2]
    starts = changes[0::2] + 1  # 1-based indexing

    if debug_print:
        print("prediction_resized", ln.lovely(img))
        print("flat", ln.lovely(flat))
        print("changes", ln.lovely(changes))
        print("starts", ln.lovely(starts))
        print("runs", ln.lovely(runs))

    if starts.size == 0:
        return ""

    rle = np.empty((starts.size * 2,), dtype=np.int64)
    rle[0::2] = starts.astype(np.int64)
    rle[1::2] = runs.astype(np.int64)
    return " ".join(map(str, rle.tolist()))




## === cell 17
submission_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")
submission_df = submission_df[["Id", "Predicted"]].copy().reset_index(drop=True)

flat_predictions: List[torch.Tensor] = []
for p in predictions:
    if isinstance(p, (list, tuple)):
        for pp in p:
            flat_predictions.append(pp)
    else:
        flat_predictions.append(p)

if len(flat_predictions) != len(test_df):
    print(
        f"WARNING: predictions count ({len(flat_predictions)}) != test_df rows ({len(test_df)}). "
        "Will align by order up to min length."
    )

pred_by_id: Dict[str, torch.Tensor] = {}
for frag_id, pred in zip(test_df["fragment_id"].astype(str).values, flat_predictions):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred)
    pred_by_id[str(frag_id)] = pred.detach().float().cpu()

mask_cache: Dict[str, np.ndarray] = {}
mask_size_cache: Dict[str, Tuple[int, int]] = {}

predictions_rle = []
for frag_id in submission_df["Id"].astype(str).values:
    if frag_id not in pred_by_id:
        predictions_rle.append("")
        continue

    prediction = pred_by_id[frag_id]
    pred_np = prediction.numpy()

    if pred_np.ndim == 3 and pred_np.shape[0] == 1:
        pred_np = pred_np.squeeze(0)
    pred_np = np.asarray(pred_np, dtype=np.float32)

    if pred_np.ndim != 2:
        pred_np = np.squeeze(pred_np)
        if pred_np.ndim != 2:
            raise ValueError(
                f"Prediction for fragment {frag_id} has unexpected shape {pred_np.shape}"
            )

    if frag_id not in mask_cache:
        mask_png_path = str(COMPETITION_DATA_DIR / "test" / frag_id / "mask.png")
        if not Path(mask_png_path).exists():
            mask_png_path = str(
                COMPETITION_DATA_DIR / "test" / "test" / frag_id / "mask.png"
            )
        mask_img = Image.open(mask_png_path).convert("L")
        mask_cache[frag_id] = (np.array(mask_img, dtype=np.uint8) > 0).astype(np.uint8)
        mask_size_cache[frag_id] = mask_img.size

    mask_np = mask_cache[frag_id]
    mask_size = mask_size_cache[frag_id]

    pred_u8 = (pred_np.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    pred_resized = (
        np.array(
            Image.fromarray(pred_u8, mode="L").resize(
                mask_size, resample=Image.BILINEAR
            ),
            dtype=np.uint8,
        ).astype(np.float32)
        / 255.0
    )

    pred_resized = pred_resized * mask_np  # enforce valid pixels only
    prediction_rle = fast_rle(pred_resized, SUBMISSION_THRESHOLD)
    predictions_rle.append(prediction_rle)

submission_df["Predicted"] = predictions_rle
submission_df = submission_df[["Id", "Predicted"]].reset_index(drop=True)
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
print("Used SUBMISSION_THRESHOLD:", SUBMISSION_THRESHOLD)
submission_df

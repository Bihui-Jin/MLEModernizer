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

0.1290207398594625

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

CANDIDATES = [
    INPUT_DIR / "vesuvius-challenge-ink-detection",
    KAGGLE_DIR / "data" / "vesuvius-challenge-ink-detection",
    INPUT_DIR / "vesuvius-challenge-ink-detection" / "vesuvius-challenge-ink-detection",
    KAGGLE_DIR
    / "data"
    / "vesuvius-challenge-ink-detection"
    / "vesuvius-challenge-ink-detection",
]
COMPETITION_DATA_DIR = None
for p in CANDIDATES:
    if p.exists():
        COMPETITION_DATA_DIR = p
        break
if COMPETITION_DATA_DIR is None:
    for p in INPUT_DIR.glob("**/sample_submission.csv"):
        COMPETITION_DATA_DIR = p.parent
        break

if COMPETITION_DATA_DIR is None or not COMPETITION_DATA_DIR.exists():
    raise FileNotFoundError(
        "Could not locate competition data directory containing sample_submission.csv"
    )

print("Using COMPETITION_DATA_DIR:", COMPETITION_DATA_DIR)



## === cell 1
import subprocess, sys
from pathlib import Path

wheel_dir = Path("../input/vesuvis-downloads")
if wheel_dir.exists():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "monai",
            "-q",
            "--no-index",
            f"--find-links={wheel_dir}",
        ],
        check=False,
    )
else:
    pass



## === cell 2
from collections import defaultdict
from io import StringIO
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import seaborn as sns
import torch
import pytorch_lightning as pl

import monai
from monai.data import CSVDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.visualize import matshow3d
from torchmetrics import Dice, MetricCollection
from tqdm.auto import tqdm

try:
    import lovely_numpy as ln  # type: ignore
except Exception:
    ln = None



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/1568666208.py in <cell line: 0>()
     11 import pytorch_lightning as pl
     12 
---> 13 import monai
     14 from monai.data import CSVDataset, DataLoader
     15 from monai.inferers import sliding_window_inference

ModuleNotFoundError: No module named 'monai'

## === cell 3
PREPARED_DATA_DIR = INPUT_DIR / "vesuvis-data-preparation"

TRAIN_DATA_CSV_PATH = PREPARED_DATA_DIR / "data_0.5.csv"
TEST_DATA_CSV_PATH = "test.csv"

if TRAIN_DATA_CSV_PATH.exists():
    DOWNSAMPLING = float(TRAIN_DATA_CSV_PATH.name.split("_")[-1].replace(".csv", ""))
else:
    DOWNSAMPLING = 1.0

Z_START = 27
Z_DIM = 16

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
NUM_SAMPLES = 64
OPTIMIZER = "AdamW"
OVERFIT_BATCHES = 0
PATCH_SIZE = (512, 512)
PRECISION = 16 if torch.cuda.is_available() else 32
SCHEDULER = "CosineAnnealingLR"
SEED = 2023
SW_BATCH_SIZE = 16

VAL_FRAGMET_ID = "2"
WEIGHT_DECAY = 1e-6

print(
    "ACCELERATOR:", ACCELERATOR, "PRECISION:", PRECISION, "DOWNSAMPLING:", DOWNSAMPLING
)




## === cell 4
class VesuvisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_csv_path: str,
        num_workers: int,
        num_samples: int,
        patch_size: Tuple[int, int],
        val_fragmet_id: str,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.df = pd.read_csv(data_csv_path)

        if "fragmet_id" in self.df.columns:
            self.df["fragmet_id"] = self.df["fragmet_id"].astype(str)

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
                train_val_df.fragmet_id != str(self.hparams.val_fragmet_id)
            ].reset_index(drop=True)
            val_df = train_val_df[
                train_val_df.fragmet_id == str(self.hparams.val_fragmet_id)
            ].reset_index(drop=True)

            if len(val_df) == 0:
                last_id = sorted(train_val_df.fragmet_id.unique())[-1]
                train_df = train_val_df[train_val_df.fragmet_id != last_id].reset_index(
                    drop=True
                )
                val_df = train_val_df[train_val_df.fragmet_id == last_id].reset_index(
                    drop=True
                )

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




## === cell 5
def visualize_dataloaders(dataloaders, train=True, max_batches=1):
    for stage, dataloader in dataloaders.items():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            volumes = batch["volume_npy"]
            masks = batch["mask_npy"]

            if train:
                labels = batch["label_npy"]
            else:
                labels = masks

            for volume, mask, label in zip(volumes, masks, labels):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                plt.suptitle(f"stage: {stage}, batch: {batch_idx}")

                for idx, image in enumerate((volume, mask, label)):
                    matshow3d(
                        volume=image,
                        fig=axes[idx],
                        title=f"{list(image.shape)}, {float(image.min()):.4f}, {float(image.max()):.4f}",
                        vmin=0.0,
                        vmax=1.0,
                        every_n=4,
                        fill_value=1.0,
                        margin=4,
                        cmap="gray",
                    )
                plt.show()




## === cell 6
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

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        outputs = self._forward_pass(batch, "predict")
        return outputs.sigmoid().squeeze()

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
        self.log(f"{stage}_loss", loss, batch_size=batch_size)
        self.log_dict(self.metrics[f"{stage}_metrics"], batch_size=batch_size)




## === cell 7
CAN_TRAIN = TRAIN_DATA_CSV_PATH.exists()
print("TRAIN_DATA_CSV_PATH exists:", CAN_TRAIN, TRAIN_DATA_CSV_PATH)




## === cell 8
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
    monai.utils.set_determinism(seed)
    pl.seed_everything(seed, workers=True)

    data_module = VesuvisDataModule(
        batch_size=batch_size,
        data_csv_path=str(data_csv_path),
        num_workers=num_workers,
        num_samples=num_samples,
        patch_size=patch_size,
        val_fragmet_id=str(val_fragmet_id),
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
        precision=precision,
        strategy="ddp" if devices > 1 else None,
        enable_checkpointing=False,  # keep minimal I/O
    )

    trainer.fit(module, datamodule=data_module)
    return module, trainer




## === cell 9
if CAN_TRAIN:
    module, trainer = train()
else:
    module, trainer = None, None



## === cell 10
if trainer is not None:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    cols = [
        c
        for c in ["epoch", "train_loss", "val_loss", "train_dice", "val_dice"]
        if c in metrics.columns
    ]
    if cols:
        metrics = metrics[cols]
        if "epoch" in metrics.columns:
            metrics = metrics.set_index("epoch")
        sns.relplot(data=metrics, kind="line", height=5, aspect=1.5)
        plt.grid()
        plt.show()



## === cell 11
test_mask_paths = sorted((COMPETITION_DATA_DIR / "test").glob("*/*mask.png"))
if len(test_mask_paths) == 0:
    test_mask_paths = sorted((COMPETITION_DATA_DIR / "test").glob("*/mask.png"))
print("Found test fragments:", len(test_mask_paths))
test_mask_paths[:3]




## === cell 12
def create_df_from_mask_paths(mask_paths, train=True):
    df = pd.DataFrame({"mask_png": mask_paths})
    df["mask_png"] = df["mask_png"].astype(str)

    df["stage"] = df["mask_png"].str.split("/").str[-3]
    df["fragmet_id"] = df["mask_png"].str.split("/").str[-2].astype(str)

    out_root = Path("/kaggle/working") / f"prepared_{DOWNSAMPLING}"
    df["mask_npy"] = df["fragmet_id"].apply(
        lambda x: str(
            out_root / df.loc[df["fragmet_id"] == x, "stage"].iloc[0] / x / "mask.npy"
        )
    )
    df["volume_npy"] = df["fragmet_id"].apply(
        lambda x: str(
            out_root / df.loc[df["fragmet_id"] == x, "stage"].iloc[0] / x / "volume.npy"
        )
    )

    if train:
        df["label_png"] = df["mask_png"].str.replace(
            "mask.png", "inklabels.png", regex=False
        )
        df["label_npy"] = df["fragmet_id"].apply(
            lambda x: str(out_root / "train" / x / "inklabels.npy")
        )

    df["volumes_dir"] = df["mask_png"].str.replace(
        "mask.png", "surface_volume", regex=False
    )
    return df


test_df = create_df_from_mask_paths(test_mask_paths, train=False)
test_df.to_csv(TEST_DATA_CSV_PATH, index=False)
test_df.head()




## === cell 13
def load_image(path):
    return Image.open(path)


def resize_image(image, downsampling):
    if downsampling == 1.0:
        return image
    size = int(image.size[0] * downsampling), int(image.size[1] * downsampling)
    return image.resize(size)


def load_and_resize_image(path, downsampling):
    image = load_image(path)
    return resize_image(image, downsampling)


def load_mask_npy(path, downsampling):
    mask = load_and_resize_image(path, downsampling).convert("1")
    return np.array(mask, dtype=np.uint8)


def load_label_npy(path, downsampling):
    label = load_and_resize_image(path, downsampling).convert("1")
    return np.array(label, dtype=np.uint8)


def load_z_slice_npy(path, downsampling):
    z_slice = load_and_resize_image(path, downsampling)
    return np.array(z_slice, dtype=np.float32) / 65535.0


def load_volume_npy(volumes_dir, downsampling):
    surface_volume_paths = sorted(Path(volumes_dir).glob("*.tif"))[
        Z_START : Z_START + Z_DIM
    ]
    if len(surface_volume_paths) != Z_DIM:
        raise FileNotFoundError(
            f"Expected {Z_DIM} tif slices in {volumes_dir}, got {len(surface_volume_paths)}"
        )
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

    volume = np.concatenate(volumes, axis=0)  # (Z_DIM, H, W)
    return volume




## === cell 14
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

        tqdm.write(f"Created {row.volume_npy} with shape {volume_npy.shape}")


save_data_as_npy(test_df, train=False)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3763280152.py in <cell line: 0>()
     20 
     21 
---> 22 save_data_as_npy(test_df, train=False)
     23 
     24 

/tmp/ipykernel_55/3763280152.py in save_data_as_npy(df, train)
      1 def save_data_as_npy(df, train=True):
----> 2     for row in tqdm(
      3         df.itertuples(index=False),
      4         total=len(df),
      5         desc="Processing fragments",

NameError: name 'tqdm' is not defined

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
    val_fragmet_id=VAL_FRAGMET_ID,
):
    monai.utils.set_determinism(seed)
    pl.seed_everything(seed, workers=True)

    data_module = VesuvisDataModule(
        batch_size=batch_size,
        data_csv_path=str(data_csv_path),
        num_workers=num_workers,
        num_samples=num_samples,
        patch_size=patch_size,
        val_fragmet_id=str(val_fragmet_id),
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
    )

    predictions = trainer.predict(module, datamodule=data_module)
    return predictions


if module is not None:
    predictions = predict(module)
else:
    predictions = None




## === cell 16
def plot_image(image, title):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(image, cmap="gray")
    plt.show()


def fast_rle(prediction_resized, threshold):
    """
    Robust RLE for 2D array:
    - threshold to binary
    - flatten row-major
    - 1-indexed starts
    - returns "" if no positives
    """
    img = (prediction_resized > threshold).astype(np.uint8)
    pixels = img.flatten(order="C")
    if pixels.sum() == 0:
        return ""

    padded = np.pad(pixels, (1, 1), mode="constant", constant_values=0)
    changes = np.where(padded[1:] != padded[:-1])[0] + 1  # positions in padded
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts

    starts = starts  # since padded has leading 0, this already corresponds to original 1-indexing
    rle = np.vstack([starts, lengths]).T.reshape(-1)
    return " ".join(map(str, rle.tolist()))




## === cell 17
submission_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")

if predictions is None:
    submission_df["Predicted"] = [""] * len(submission_df)
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with empty masks (no training CSV/model available).")
else:
    pred_by_id = {}
    for frag_id, mask_png_path, prediction in zip(
        test_df["fragmet_id"].values, test_df["mask_png"].values, predictions
    ):
        mask = Image.open(mask_png_path)
        pred_np = prediction.detach().cpu().numpy()

        pred_resized = np.array(
            Image.fromarray(pred_np).resize(mask.size), dtype=np.float32
        )

        threshold = float(pred_resized.mean() + pred_resized.std())

        pred_rle = fast_rle(pred_resized, threshold)
        pred_by_id[str(frag_id)] = pred_rle

    submission_df["Id"] = submission_df["Id"].astype(str)
    submission_df["Predicted"] = submission_df["Id"].map(
        lambda x: pred_by_id.get(x, "")
    )
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")
submission_df

## --- ERROR in outputing the csv:
Invalid submission: Expected 2 indices in the submission DataFrame, but got 6 indices.

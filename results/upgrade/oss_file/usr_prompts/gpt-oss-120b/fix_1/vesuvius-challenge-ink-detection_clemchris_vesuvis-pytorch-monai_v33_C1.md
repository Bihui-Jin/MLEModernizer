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

## === cell 2
!pip install monai lovely-numpy -q --no-index --find-links=../input/vesuvis-downloads

## === cell 4
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Tuple

import lovely_numpy as ln
import monai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import pytorch_lightning as pl
import seaborn as sns
import torch
from monai.data import CSVDataset
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.visualize import matshow3d
from torchmetrics import Dice
from torchmetrics import MetricCollection
from tqdm.auto import tqdm

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1586708905.py in <cell line: 0>()
      4 from typing import Tuple
      5 
----> 6 import lovely_numpy as ln
      7 import monai
      8 import matplotlib.pyplot as plt

ModuleNotFoundError: No module named 'lovely_numpy'

## === cell 6
KAGGLE_DIR = Path("/") / "kaggle"

INPUT_DIR = KAGGLE_DIR / "input"

COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"
PREPARED_DATA_DIR = INPUT_DIR / "vesuvis-data-preparation"

TRAIN_DATA_CSV_PATH = PREPARED_DATA_DIR / "data_0.5.csv"
TEST_DATA_CSV_PATH = "test.csv"

DOWNSAMPLING = float(TRAIN_DATA_CSV_PATH.name.split("_")[-1].replace(".csv", ""))

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
MAX_EPOCHS = 5
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

## === cell 8
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
                monai.transforms.LoadImaged(
                    keys="volume_npy",
                ),
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
                    keys=self.keys,
                    prob=0.5,
                    spatial_axis=0,
                ),
                monai.transforms.RandFlipd(
                    keys=self.keys,
                    prob=0.5,
                    spatial_axis=1,
                ),
            ]
        )

    def _init_val_transform(self):
        return monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(
                    keys="volume_npy",
                ),
                monai.transforms.LoadImaged(
                    keys=("mask_npy", "label_npy"),
                    ensure_channel_first=True,
                ),
            ]
        )

    def _init_predict_transform(self):
        return monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(
                    keys="volume_npy",
                ),
                monai.transforms.LoadImaged(
                    keys="mask_npy",
                    ensure_channel_first=True,
                ),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_val_df = self.df[self.df.stage == "train"].reset_index(drop=True)

            train_df = train_val_df[
                train_val_df.fragmet_id != self.hparams.val_fragmet_id
            ].reset_index(drop=True)

            val_df = train_val_df[
                train_val_df.fragmet_id == self.hparams.val_fragmet_id
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
        return CSVDataset(
            src=df,
            transform=transform,
        )

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

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/786883231.py in <cell line: 0>()
----> 1 class VesuvisDataModule(pl.LightningDataModule):
      2     def __init__(
      3         self,
      4         batch_size: int,
      5         data_csv_path: str,

NameError: name 'pl' is not defined

## === cell 10
def visualize_dataloaders(dataloaders, train=True):
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

## === cell 11
data_module = VesuvisDataModule(
    batch_size=BATCH_SIZE,
    data_csv_path=TRAIN_DATA_CSV_PATH,
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

visualize_dataloaders(dataloaders)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2958430549.py in <cell line: 0>()
----> 1 data_module = VesuvisDataModule(
      2     batch_size=BATCH_SIZE,
      3     data_csv_path=TRAIN_DATA_CSV_PATH,
      4     num_workers=NUM_WORKERS,
      5     num_samples=2,

NameError: name 'VesuvisDataModule' is not defined

## === cell 13
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
            loss = monai.losses.DiceLoss(
                sigmoid=True,
                jaccard=True,
            )
        elif self.hparams.loss == "DiceCELoss":
            loss = monai.losses.DiceCELoss(sigmoid=True)
        else:
            raise ValueError(f"{self.hparams.loss} is not implemented")

        return monai.losses.MaskedLoss(loss)

    def _init_metrics(self):
        metric_collection = MetricCollection(
            {
                "dice": Dice(),
            }
        )

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
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
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
        else:
            labels = batch["label_npy"].long()
            masks = batch["mask_npy"]
            return outputs, labels, masks

    def _log(self, loss, stage, batch_size):
        self.log(f"{stage}_loss", loss, batch_size=batch_size)
        self.log_dict(self.metrics[f"{stage}_metrics"], batch_size=batch_size)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3395177093.py in <cell line: 0>()
----> 1 class VesuvisModule(pl.LightningModule):
      2     def __init__(
      3         self,
      4         dropout: float,
      5         eta_min: float,

NameError: name 'pl' is not defined

## === cell 15
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
        data_csv_path=data_csv_path,
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
        check_val_every_n_epoch=max_epochs // 5,
        devices=devices,
        fast_dev_run=fast_dev_run,
        logger=pl.loggers.CSVLogger(save_dir='logs/'),
        log_every_n_steps=1,
        max_epochs=max_epochs,
        overfit_batches=overfit_batches,
        precision=precision,
        strategy="ddp" if devices > 1 else None,
    )

    trainer.fit(module, datamodule=data_module)

    return module, trainer

## === cell 16
module, trainer = train()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/942244826.py in <cell line: 0>()
----> 1 module, trainer = train()

/tmp/ipykernel_11/1563446005.py in train(accelerator, batch_size, data_csv_path, devices, dropout, eta_min, fast_dev_run, learning_rate, loss, model_name, max_epochs, num_workers, num_samples, optimizer, overfit_batches, patch_size, precision, scheduler, seed, sw_batch_size, val_fragmet_id, weight_decay)
     23     weight_decay=WEIGHT_DECAY,
     24 ):
---> 25     monai.utils.set_determinism(seed)
     26     pl.seed_everything(seed, workers=True)
     27 

NameError: name 'monai' is not defined

## === cell 17
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
metrics = metrics[["epoch", "train_loss", "val_loss", "train_dice", "val_dice"]]
metrics.set_index("epoch", inplace=True)

sns.relplot(data=metrics, kind="line", height=5, aspect=1.5)
plt.grid()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1211374037.py in <cell line: 0>()
      1 # From https://www.kaggle.com/code/jirkaborovec?scriptVersionId=93358967&cellId=22
----> 2 metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
      3 metrics = metrics[["epoch", "train_loss", "val_loss", "train_dice", "val_dice"]]
      4 metrics.set_index("epoch", inplace=True)
      5 

NameError: name 'pd' is not defined

## === cell 20
test_mask_paths = sorted(COMPETITION_DATA_DIR.glob("test/*/mask.png"))
test_mask_paths

## === cell 21
def create_df_from_mask_paths(mask_paths, train=True):
    df = pd.DataFrame({"mask_png": mask_paths})

    df["mask_png"] = df["mask_png"].astype(str)

    df["stage"] = df["mask_png"].str.split("/").str[-3]
    df["fragmet_id"] = df["mask_png"].str.split("/").str[-2]

    df["mask_npy"] = df["mask_png"].str.replace(
        "train", f"train_{DOWNSAMPLING}", regex=False
    )
    df["mask_npy"] = df["mask_npy"].str.replace("input", "working", regex=False)
    df["mask_npy"] = df["mask_npy"].str.replace("png", "npy", regex=False)

    if train:
        df["label_png"] = df["mask_png"].str.replace("mask", "inklabels", regex=False)
        df["label_npy"] = df["mask_npy"].str.replace("mask", "inklabels", regex=False)

    df["volumes_dir"] = df["mask_png"].str.replace(
        "mask.png", "surface_volume", regex=False
    )
    df["volume_npy"] = df["mask_npy"].str.replace("mask", "volume", regex=False)
    
    return df

## === cell 22
test_df = create_df_from_mask_paths(test_mask_paths, train=False)

test_df.to_csv("test.csv")

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2793708118.py in <cell line: 0>()
----> 1 test_df = create_df_from_mask_paths(test_mask_paths, train=False)
      2 
      3 test_df.to_csv("test.csv")

/tmp/ipykernel_11/2062614379.py in create_df_from_mask_paths(mask_paths, train)
      1 def create_df_from_mask_paths(mask_paths, train=True):
----> 2     df = pd.DataFrame({"mask_png": mask_paths})
      3 
      4     df["mask_png"] = df["mask_png"].astype(str)
      5 

NameError: name 'pd' is not defined

## === cell 23
test_df

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1820894078.py in <cell line: 0>()
----> 1 test_df

NameError: name 'test_df' is not defined

## === cell 25
def load_image(path):
    return Image.open(path)


def resize_image(image, downsampling):
    size = int(image.size[0] * downsampling), int(image.size[1] * downsampling)
    return image.resize(size)


def load_and_resize_image(path, downsampling):
    image = load_image(path)
    return resize_image(image, downsampling)


def load_mask_npy(path, downsampling):
    mask = load_and_resize_image(path, downsampling).convert("1")
    return np.array(mask)


def load_z_slice_npy(path, downsampling):
    z_slice = load_and_resize_image(path, downsampling)
    return np.array(z_slice, dtype=np.float32) / 65535.0


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

## === cell 26
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

## === cell 27
save_data_as_npy(test_df, train=False)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4002959331.py in <cell line: 0>()
----> 1 save_data_as_npy(test_df, train=False)

NameError: name 'test_df' is not defined

## === cell 29
data_module = VesuvisDataModule(
    batch_size=BATCH_SIZE,
    data_csv_path=TEST_DATA_CSV_PATH,
    num_workers=NUM_WORKERS,
    num_samples=NUM_SAMPLES,
    patch_size=PATCH_SIZE,
    val_fragmet_id=VAL_FRAGMET_ID,
)
data_module.setup(stage="predict")

dataloaders = {
    "predict": data_module.predict_dataloader(),
}

visualize_dataloaders(dataloaders, train=False)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1934341999.py in <cell line: 0>()
----> 1 data_module = VesuvisDataModule(
      2     batch_size=BATCH_SIZE,
      3     data_csv_path=TEST_DATA_CSV_PATH,
      4     num_workers=NUM_WORKERS,
      5     num_samples=NUM_SAMPLES,

NameError: name 'VesuvisDataModule' is not defined

## === cell 31
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
        data_csv_path=data_csv_path,
        num_workers=num_workers,
        num_samples=num_samples,
        patch_size=patch_size,
        val_fragmet_id=val_fragmet_id,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
    )

    predictions = trainer.predict(module, datamodule=data_module)

    return predictions

## === cell 32
predictions = predict(module)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2733779795.py in <cell line: 0>()
----> 1 predictions = predict(module)

NameError: name 'module' is not defined

## === cell 34
def plot_image(image, title):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(image, cmap="gray")


def fast_rle(prediction_resized, threshold, debug_print=False):
    flat_img = prediction_resized.flatten()
    flat_img = np.where(flat_img > threshold, 1, 0).astype(np.uint8)
    
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))

    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2    
    
    if debug_print:
        print("prediction_resized", ln.lovely(prediction_resized))
        print("flat_img", ln.lovely(flat_img))
        print("starts_ix", ln.lovely(starts_ix))
        print("ends_ix", ln.lovely(ends_ix))

    lengths = ends_ix - starts_ix

    predicted_arr = np.stack([starts_ix, lengths]).T.flatten()

    f = StringIO()
    np.savetxt(f, predicted_arr.reshape(1, -1), delimiter=" ", fmt="%d")
    
    return f.getvalue().strip()

## === cell 35
submission_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")

predictions_rle = []
for mask_png_path, prediction in zip(test_df["mask_png"].values, predictions):
    mask = Image.open(mask_png_path)

    prediction_resized = np.array(
        Image.fromarray(prediction.numpy()).resize(mask.size)
    )

    threshold = np.mean(prediction_resized) + np.std(prediction_resized)

    print("threshold:", threshold)
    
    if 0:
        prediction_thresholded = np.where(prediction_resized > threshold, 1, 0).astype(np.uint8)
        plot_image(mask, f"mask {ln.lovely(np.array(mask))}")
        plot_image(prediction_resized, f"prediction resized {ln.lovely(prediction_resized)}")
        plot_image(
            prediction_thresholded,
            f"prediction thresholded {ln.lovely(prediction_thresholded)}",
        )

    try:
        prediction_rle = fast_rle(prediction_resized, threshold)
    except Exception as e:
        print(e)
        prediction_rle = "1 1 5 1"
    
    predictions_rle.append(prediction_rle)

submission_df["Predicted"] = predictions_rle
    
submission_df.to_csv("submission.csv", index=False)

submission_df

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2810934842.py in <cell line: 0>()
----> 1 submission_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")
      2 
      3 predictions_rle = []
      4 for mask_png_path, prediction in zip(test_df["mask_png"].values, predictions):
      5     mask = Image.open(mask_png_path)

NameError: name 'pd' is not defined

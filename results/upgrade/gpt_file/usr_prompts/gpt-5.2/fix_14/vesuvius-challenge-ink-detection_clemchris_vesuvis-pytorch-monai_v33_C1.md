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
    if (
        (p / "sample_submission.csv").exists()
        and (p / "test").exists()
        and (p / "train").exists()
    ):
        COMPETITION_DATA_DIR = p
        break

if COMPETITION_DATA_DIR is None:
    for p in INPUT_DIR.glob("**/sample_submission.csv"):
        root = p.parent
        if (root / "test").exists() and (root / "train").exists():
            COMPETITION_DATA_DIR = root
            break

if COMPETITION_DATA_DIR is None or not COMPETITION_DATA_DIR.exists():
    raise FileNotFoundError(
        "Could not locate competition data directory containing sample_submission.csv plus train/ and test/."
    )

print("Using COMPETITION_DATA_DIR:", COMPETITION_DATA_DIR)



## === cell 1
import subprocess, sys
from pathlib import Path


def _ensure_monai():
    try:
        import monai  # noqa: F401

        return True
    except Exception:
        pass

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
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "monai", "-q"], check=False
        )

    try:
        import monai  # noqa: F401

        return True
    except Exception as e:
        print("[WARN] Could not import MONAI after install attempt:", repr(e))
        return False


_MONAI_INSTALLED_OK = _ensure_monai()
print("MONAI install/import status:", _MONAI_INSTALLED_OK)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
from tqdm.auto import tqdm

try:
    import monai  # type: ignore
    from monai.data import CSVDataset, DataLoader  # type: ignore
    from monai.inferers import sliding_window_inference  # type: ignore
    from monai.visualize import matshow3d  # type: ignore

    MONAI_AVAILABLE = True
except Exception as e:
    monai = None
    CSVDataset = None
    DataLoader = None
    sliding_window_inference = None
    matshow3d = None
    MONAI_AVAILABLE = False
    print(
        "MONAI not available; will run in no-train fallback mode. Import error:",
        repr(e),
    )

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score

try:
    import lovely_numpy as ln  # type: ignore
except Exception:
    ln = None




## === cell 3
def _find_prepared_train_csv() -> Path | None:
    candidate_roots = [
        INPUT_DIR,
        KAGGLE_DIR / "data",
        KAGGLE_DIR / "working",
    ]
    patterns = [
        "**/vesuvius-data-preparation/data_*.csv",
        "**/vesuvis-data-preparation/data_*.csv",  # keep backward-compat in case it exists
        "**/vesuvius*prepar*/data_*.csv",
        "**/data_*.csv",
    ]
    found = []
    for root in candidate_roots:
        if not root.exists():
            continue
        for pat in patterns:
            found.extend(list(root.glob(pat)))
    found = [
        p
        for p in found
        if p.is_file() and p.name.startswith("data_") and p.suffix == ".csv"
    ]
    if not found:
        return None
    found_sorted = sorted(
        found,
        key=lambda p: (0 if "vesuvius-data-preparation" in str(p) else 1, len(str(p))),
    )
    return found_sorted[0]


TRAIN_DATA_CSV_PATH = _find_prepared_train_csv()
TEST_DATA_CSV_PATH = str((KAGGLE_DIR / "working" / "test.csv").resolve())

if TRAIN_DATA_CSV_PATH is not None and TRAIN_DATA_CSV_PATH.exists():
    try:
        DOWNSAMPLING = float(
            TRAIN_DATA_CSV_PATH.name.split("_")[-1].replace(".csv", "")
        )
    except Exception:
        DOWNSAMPLING = 1.0
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
print("TRAIN_DATA_CSV_PATH:", TRAIN_DATA_CSV_PATH)




## === cell 4
def _discover_train_fragment_dirs(root: Path) -> list[Path]:
    train_dir = root / "train"
    if not train_dir.exists():
        return []
    dirs = []
    for p in sorted(train_dir.iterdir()):
        if not p.is_dir():
            continue
        if (
            (p / "mask.png").exists()
            and (p / "inklabels.png").exists()
            and (p / "surface_volume").exists()
        ):
            dirs.append(p)
    return dirs


def create_train_csv_from_fragments(fragment_dirs: list[Path]) -> Path:
    out_root = Path("/kaggle/working") / f"prepared_{DOWNSAMPLING}"
    rows = []
    for frag_dir in fragment_dirs:
        frag_id = frag_dir.name
        rows.append(
            {
                "mask_png": str(frag_dir / "mask.png"),
                "label_png": str(frag_dir / "inklabels.png"),
                "stage": "train",
                "fragmet_id": str(frag_id),
                "mask_npy": str(out_root / "train" / frag_id / "mask.npy"),
                "label_npy": str(out_root / "train" / frag_id / "label.npy"),
                "volume_npy": str(out_root / "train" / frag_id / "volume.npy"),
                "volumes_dir": str(frag_dir / "surface_volume"),
                "ir_png": str(frag_dir / "ir.png"),
            }
        )
    df = pd.DataFrame(rows)
    train_csv_path = Path("/kaggle/working") / "train.csv"
    df.to_csv(train_csv_path, index=False)
    return train_csv_path


if TRAIN_DATA_CSV_PATH is None or not Path(TRAIN_DATA_CSV_PATH).exists():
    train_fragment_dirs = _discover_train_fragment_dirs(COMPETITION_DATA_DIR)
    if len(train_fragment_dirs) > 0:
        TRAIN_DATA_CSV_PATH = create_train_csv_from_fragments(train_fragment_dirs)
        print("[INFO] Created TRAIN_DATA_CSV_PATH:", TRAIN_DATA_CSV_PATH)
    else:
        print("[WARN] No train fragments found; training will be disabled.")



## === cell 5
if MONAI_AVAILABLE:

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
                        keys=("mask_npy", "label_npy"), ensure_channel_first=True
                    ),
                ]
            )

        def _init_predict_transform(self):
            return monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys="volume_npy"),
                    monai.transforms.LoadImaged(
                        keys="mask_npy", ensure_channel_first=True
                    ),
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

                if len(val_df) == 0 and len(train_val_df) > 0:
                    last_id = sorted(train_val_df.fragmet_id.unique())[-1]
                    train_df = train_val_df[
                        train_val_df.fragmet_id != last_id
                    ].reset_index(drop=True)
                    val_df = train_val_df[
                        train_val_df.fragmet_id == last_id
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




## === cell 6
def visualize_dataloaders(dataloaders, train=True, max_batches=1):
    if not MONAI_AVAILABLE or matshow3d is None:
        print("visualize_dataloaders skipped because MONAI is not available.")
        return
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




## === cell 7
if MONAI_AVAILABLE:

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
            metric_collection = MetricCollection({"dice": BinaryF1Score()})
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
            if volumes.ndim == 4:
                volumes = volumes.permute(0, 1, 2, 3).contiguous()
            if (
                volumes.ndim == 4
                and volumes.shape[1] != Z_DIM
                and volumes.shape[-1] == Z_DIM
            ):
                volumes = volumes.permute(0, 3, 1, 2).contiguous()

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

            labels = batch["label_npy"].float()
            masks = batch["mask_npy"]  # expects 1 where valid pixels exist
            return outputs, labels, masks

        def _log(self, loss, stage, batch_size):
            self.log(f"{stage}_loss", loss, batch_size=batch_size)
            self.log_dict(self.metrics[f"{stage}_metrics"], batch_size=batch_size)




## === cell 8
CAN_TRAIN = (
    MONAI_AVAILABLE
    and (TRAIN_DATA_CSV_PATH is not None)
    and Path(TRAIN_DATA_CSV_PATH).exists()
)
print("MONAI_AVAILABLE:", MONAI_AVAILABLE)
print(
    "TRAIN_DATA_CSV_PATH exists:",
    (TRAIN_DATA_CSV_PATH is not None) and Path(TRAIN_DATA_CSV_PATH).exists(),
    TRAIN_DATA_CSV_PATH,
)
print("CAN_TRAIN:", CAN_TRAIN)




## === cell 9
def train(
    accelerator=ACCELERATOR,
    batch_size=BATCH_SIZE,
    data_csv_path=None,
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
        raise RuntimeError("train() called but MONAI is not available.")
    if data_csv_path is None:
        raise RuntimeError("train() requires data_csv_path, but got None.")
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
        enable_checkpointing=False,
    )

    trainer.fit(module, datamodule=data_module)
    return module, trainer




## === cell 10
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
    return (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)


def load_label_npy(path, downsampling):
    lab = load_and_resize_image(path, downsampling).convert("1")
    return (np.array(lab, dtype=np.uint8) > 0).astype(np.uint8)


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


def save_data_as_npy(df, train=False):
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Processing fragments",
        position=0,
    ):
        try:
            mask_npy = load_mask_npy(row.mask_png, DOWNSAMPLING)
            volume_npy = load_volume_npy(row.volumes_dir, DOWNSAMPLING)

            Path(row.mask_npy).parent.mkdir(exist_ok=True, parents=True)
            np.save(row.mask_npy, mask_npy)
            np.save(row.volume_npy, volume_npy)

            if train and hasattr(row, "label_png") and isinstance(row.label_png, str):
                label_npy = load_label_npy(row.label_png, DOWNSAMPLING)
                np.save(row.label_npy, label_npy)
            else:
                if hasattr(row, "label_npy") and isinstance(row.label_npy, str):
                    np.save(row.label_npy, np.zeros((1, 1), dtype=np.uint8))

            tqdm.write(f"Created {row.volume_npy} with shape {volume_npy.shape}")
        except Exception as e:
            tqdm.write(
                f"[WARN] Failed to prepare npy for fragment {getattr(row, 'fragmet_id', 'unknown')}: {repr(e)}"
            )
            try:
                Path(row.mask_npy).parent.mkdir(exist_ok=True, parents=True)
                mask_npy = load_mask_npy(row.mask_png, DOWNSAMPLING)
                np.save(row.mask_npy, mask_npy)
            except Exception:
                pass


if CAN_TRAIN:
    train_df = pd.read_csv(TRAIN_DATA_CSV_PATH)
    need_prepare = False
    for col in ["mask_npy", "volume_npy", "label_npy"]:
        if col not in train_df.columns:
            need_prepare = True
            break
    if not need_prepare and len(train_df) > 0:
        r0 = train_df.iloc[0]
        need_prepare = not (
            Path(r0["mask_npy"]).exists()
            and Path(r0["volume_npy"]).exists()
            and Path(r0["label_npy"]).exists()
        )
    if need_prepare:
        print(
            "[INFO] Preparing missing train .npy artifacts (required for MONAI LoadImaged)."
        )
        save_data_as_npy(train_df, train=True)
    else:
        print("[INFO] Train .npy artifacts already exist; skipping preparation.")



## === cell 11
if CAN_TRAIN:
    module, trainer = train(data_csv_path=TRAIN_DATA_CSV_PATH)
else:
    module, trainer = None, None



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1163610581.py in <cell line: 0>()
      1 if CAN_TRAIN:
----> 2     module, trainer = train(data_csv_path=TRAIN_DATA_CSV_PATH)
      3 else:
      4     module, trainer = None, None
      5 

/tmp/ipykernel_55/1044576609.py in train(accelerator, batch_size, data_csv_path, devices, dropout, eta_min, fast_dev_run, learning_rate, loss, model_name, max_epochs, num_workers, num_samples, optimizer, overfit_batches, patch_size, precision, scheduler, seed, sw_batch_size, val_fragmet_id, weight_decay)
     53     )
     54 
---> 55     trainer = pl.Trainer(
     56         accelerator=accelerator,
     57         benchmark=True,

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/argparse.py in insert_env_defaults(self, *args, **kwargs)
     68 
     69         # all args were already moved to kwargs
---> 70         return fn(self, **kwargs)
     71 
     72     return cast(_T, insert_env_defaults)

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in __init__(self, accelerator, strategy, devices, num_nodes, precision, logger, callbacks, fast_dev_run, max_epochs, min_epochs, max_steps, min_steps, max_time, limit_train_batches, limit_val_batches, limit_test_batches, limit_predict_batches, overfit_batches, val_check_interval, check_val_every_n_epoch, num_sanity_val_steps, log_every_n_steps, enable_checkpointing, enable_progress_bar, enable_model_summary, accumulate_grad_batches, gradient_clip_val, gradient_clip_algorithm, deterministic, benchmark, inference_mode, use_distributed_sampler, profiler, detect_anomaly, barebones, plugins, sync_batchnorm, reload_dataloaders_every_n_epochs, default_root_dir, model_registry)
    402         self._data_connector = _DataConnector(self)
    403 
--> 404         self._accelerator_connector = _AcceleratorConnector(
    405             devices=devices,
    406             accelerator=accelerator,

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py in __init__(self, devices, num_nodes, accelerator, strategy, plugins, precision, sync_batchnorm, benchmark, use_distributed_sampler, deterministic)
    129         self.checkpoint_io: Optional[CheckpointIO] = None
    130 
--> 131         self._check_config_and_set_final_flags(
    132             strategy=strategy,
    133             accelerator=accelerator,

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py in _check_config_and_set_final_flags(self, strategy, accelerator, precision, plugins, sync_batchnorm)
    192 
    193         if strategy != "auto" and strategy not in self._registered_strategies and not isinstance(strategy, Strategy):
--> 194             raise ValueError(
    195                 f"You selected an invalid strategy name: `strategy={strategy!r}`."
    196                 " It must be either a string or an instance of `pytorch_lightning.strategies.Strategy`."

ValueError: You selected an invalid strategy name: `strategy=None`. It must be either a string or an instance of `pytorch_lightning.strategies.Strategy`. Example choices: auto, ddp, ddp_spawn, deepspeed, ... Find a complete list of options in our documentation at https://lightning.ai

## === cell 12
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



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3709089847.py in <cell line: 0>()
----> 1 if trainer is not None:
      2     metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
      3     cols = [
      4         c
      5         for c in ["epoch", "train_loss", "val_loss", "train_dice", "val_dice"]

NameError: name 'trainer' is not defined

## === cell 13
test_dir = COMPETITION_DATA_DIR / "test"
all_test_dirs = sorted([p for p in test_dir.iterdir() if p.is_dir()])
test_fragment_dirs = sorted(
    [
        p
        for p in all_test_dirs
        if (p / "mask.png").exists() and (p / "surface_volume").exists()
    ]
)
print(
    "Found test dirs:",
    len(all_test_dirs),
    "Filtered test fragments:",
    len(test_fragment_dirs),
)
print("Filtered fragments:", [p.name for p in test_fragment_dirs[:10]])




## === cell 14
def create_df_from_fragment_dirs(fragment_dirs, train=False):
    rows = []
    out_root = Path("/kaggle/working") / f"prepared_{DOWNSAMPLING}"
    for frag_dir in fragment_dirs:
        frag_id = frag_dir.name
        mask_png = frag_dir / "mask.png"
        volumes_dir = frag_dir / "surface_volume"
        if not mask_png.exists():
            raise FileNotFoundError(f"Missing mask.png for fragment: {frag_dir}")
        if not volumes_dir.exists():
            raise FileNotFoundError(f"Missing surface_volume for fragment: {frag_dir}")

        rows.append(
            {
                "mask_png": str(mask_png),
                "stage": "test",
                "fragmet_id": str(frag_id),
                "mask_npy": str(out_root / "test" / frag_id / "mask.npy"),
                "volume_npy": str(out_root / "test" / frag_id / "volume.npy"),
                "label_npy": str(out_root / "test" / frag_id / "label_dummy.npy"),
                "volumes_dir": str(volumes_dir),
                "ir_png": str(frag_dir / "ir.png"),
            }
        )
    df = pd.DataFrame(rows)
    return df


test_df = create_df_from_fragment_dirs(test_fragment_dirs, train=False)

submission_template_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")
submission_ids = submission_template_df["Id"].astype(str).tolist()
if len(submission_ids) == len(test_df):
    test_df = test_df.sort_values("fragmet_id").reset_index(drop=True)
    test_df["fragmet_id"] = submission_ids
else:
    print(
        "[WARN] sample_submission Id count does not match discovered test fragments:",
        len(submission_ids),
        "vs",
        len(test_df),
        "- keeping folder-based ids.",
    )

test_df.to_csv(TEST_DATA_CSV_PATH, index=False)
test_df.head()



## === cell 15
save_data_as_npy(test_df, train=False)




## === cell 16
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
    if not MONAI_AVAILABLE:
        raise RuntimeError("predict() called but MONAI is not available.")
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
    try:
        predictions = predict(module)
    except Exception as e:
        print(
            "[WARN] Model prediction failed; will use fallback submission. Error:",
            repr(e),
        )
        predictions = None
else:
    predictions = None




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1101239848.py in <cell line: 0>()
     38 
     39 
---> 40 if module is not None:
     41     try:
     42         predictions = predict(module)

NameError: name 'module' is not defined

## === cell 17
def plot_image(image, title):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(image, cmap="gray")
    plt.show()


def fast_rle(prediction_resized, threshold):
    """
    Competition-standard RLE for Vesuvius:
    - threshold to a binary mask
    - flatten in column-major order (Fortran order) WITHOUT transpose
    - 1-indexed starts
    """
    img = (prediction_resized > float(threshold)).astype(np.uint8)

    pixels = img.flatten(order="F")
    if int(pixels.sum()) == 0:
        return ""

    padded = np.pad(pixels, (1, 1), mode="constant", constant_values=0)
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts

    rle = np.vstack([starts, lengths]).T.reshape(-1)
    return " ".join(map(str, rle.tolist()))


def _to_2d_prediction(pred_tensor_or_array):
    if isinstance(pred_tensor_or_array, torch.Tensor):
        arr = pred_tensor_or_array.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(pred_tensor_or_array, dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Prediction must be 2D after squeeze, got shape={arr.shape}")
    return arr


def _load_mask_uint8(mask_png_path: str) -> np.ndarray:
    mask = Image.open(mask_png_path).convert("1")
    return (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)


def fallback_predict_mask_only(mask_png_path: str) -> np.ndarray:
    mask_np = _load_mask_uint8(mask_png_path)
    return np.zeros(mask_np.shape, dtype=np.float32)


def fallback_predict_from_ir(mask_png_path: str, ir_png_path: str) -> np.ndarray:
    mask_np = _load_mask_uint8(mask_png_path)

    ir_path = Path(ir_png_path)
    if not ir_path.exists():
        return np.zeros(mask_np.shape, dtype=np.float32)

    ir = Image.open(ir_png_path).convert("L")
    mask_img = Image.open(mask_png_path)
    if ir.size != mask_img.size:
        ir = ir.resize(mask_img.size, resample=Image.BILINEAR)
    ir_np = np.array(ir, dtype=np.float32) / 255.0

    pred = (1.0 - ir_np) * mask_np.astype(np.float32)
    return pred


def fbeta05_score_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.uint8).ravel()
    y_pred = y_pred.astype(np.uint8).ravel()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = 0.25
    denom = beta2 * p + r
    if denom <= 0:
        return 0.0
    return float((1 + beta2) * p * r / denom)


def choose_threshold_by_val_f05(
    module, train_csv_path: str, val_fragmet_id: str
) -> float | None:
    if module is None or (not MONAI_AVAILABLE):
        return None
    try:
        df = pd.read_csv(train_csv_path)
        if "stage" not in df.columns:
            return None
        val_df = df[
            (df["stage"] == "train")
            & (df["fragmet_id"].astype(str) == str(val_fragmet_id))
        ].reset_index(drop=True)
        if len(val_df) == 0:
            return None

        tmp_csv = Path("/kaggle/working") / "val_only.csv"
        val_df.to_csv(tmp_csv, index=False)

        dm = VesuvisDataModule(
            batch_size=1,
            data_csv_path=str(tmp_csv),
            num_workers=0,
            num_samples=NUM_SAMPLES,
            patch_size=PATCH_SIZE,
            val_fragmet_id=str(val_fragmet_id),
        )
        dm.setup("fit")

        tr = pl.Trainer(
            accelerator=ACCELERATOR,
            devices=DEVICES,
            precision=PRECISION,
            logger=False,
            enable_checkpointing=False,
        )
        preds = tr.predict(module, datamodule=dm)  # list of tensors

        mask_png_path = val_df.iloc[0]["mask_png"]
        label_png_path = val_df.iloc[0].get("label_png", None)
        if label_png_path is None or (not Path(str(label_png_path)).exists()):
            label_npy = val_df.iloc[0].get("label_npy", None)
            if label_npy is None or (not Path(str(label_npy)).exists()):
                return None
            y_true = np.load(str(label_npy)).astype(np.uint8)
        else:
            y_true = (
                np.array(Image.open(str(label_png_path)).convert("1"), dtype=np.uint8)
                > 0
            ).astype(np.uint8)

        mask = _load_mask_uint8(str(mask_png_path)).astype(np.uint8)
        y_true = (y_true > 0).astype(np.uint8) * mask

        h, w = y_true.shape
        pred_acc = np.zeros((h, w), dtype=np.float32)
        pred_cnt = 0
        for p in preds:
            p2d = _to_2d_prediction(p)
            pimg = Image.fromarray(p2d.astype(np.float32), mode="F")
            pres = np.array(
                pimg.resize((w, h), resample=Image.BILINEAR), dtype=np.float32
            )
            pred_acc += pres
            pred_cnt += 1
        if pred_cnt == 0:
            return None
        pred_mean = (pred_acc / float(pred_cnt)) * mask.astype(np.float32)

        thresholds = np.linspace(0.35, 0.90, 12, dtype=np.float32)
        best_t, best_s = None, -1.0
        for t in thresholds:
            y_pred = (pred_mean > float(t)).astype(np.uint8) * mask
            s = fbeta05_score_from_binary(y_true, y_pred)
            if s > best_s:
                best_s = s
                best_t = float(t)
        print(f"[INFO] Chosen threshold by val F0.5: t={best_t} (val_f05={best_s:.6f})")
        return best_t
    except Exception as e:
        print(
            "[WARN] Threshold calibration failed; using default threshold. Error:",
            repr(e),
        )
        return None




## === cell 18
CALIBRATED_THRESHOLD = None
if module is not None and CAN_TRAIN:
    CALIBRATED_THRESHOLD = choose_threshold_by_val_f05(
        module, str(TRAIN_DATA_CSV_PATH), str(VAL_FRAGMET_ID)
    )

DEFAULT_MODEL_THRESHOLD = 0.75
MODEL_THRESHOLD = (
    float(CALIBRATED_THRESHOLD)
    if CALIBRATED_THRESHOLD is not None
    else float(DEFAULT_MODEL_THRESHOLD)
)
print("MODEL_THRESHOLD:", MODEL_THRESHOLD)

submission_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")
submission_df = submission_df[["Id", "Predicted"]].copy()
submission_df["Id"] = submission_df["Id"].astype(str)

pred_by_id = {}

if predictions is None:
    for frag_id, mask_png_path, ir_png_path in zip(
        test_df["fragmet_id"].values,
        test_df["mask_png"].values,
        test_df["ir_png"].values,
    ):
        if Path(ir_png_path).exists():
            pred_resized = fallback_predict_from_ir(mask_png_path, ir_png_path)
            threshold = 0.85
        else:
            pred_resized = fallback_predict_mask_only(mask_png_path)
            threshold = 0.5

        mask_np = _load_mask_uint8(mask_png_path).astype(np.float32)
        pred_resized = pred_resized * mask_np

        pred_rle = fast_rle(pred_resized, threshold)
        pred_by_id[str(frag_id)] = pred_rle

    submission_df["Predicted"] = submission_df["Id"].map(
        lambda x: pred_by_id.get(x, "")
    )
    submission_df["Predicted"] = submission_df["Predicted"].fillna("").astype(str)
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv using fallback (IR if available else empty mask).")
else:
    if len(predictions) != len(test_df):
        raise RuntimeError(
            f"Predictions count mismatch: len(predictions)={len(predictions)} vs len(test_df)={len(test_df)}"
        )

    for frag_id, mask_png_path, prediction in zip(
        test_df["fragmet_id"].values, test_df["mask_png"].values, predictions
    ):
        mask_img = Image.open(mask_png_path)
        mask_np = _load_mask_uint8(mask_png_path).astype(np.float32)

        pred_np = _to_2d_prediction(prediction)

        pred_img = Image.fromarray(pred_np.astype(np.float32), mode="F")
        pred_resized = np.array(
            pred_img.resize(mask_img.size, resample=Image.BILINEAR),
            dtype=np.float32,
        )

        pred_resized = pred_resized * mask_np

        pred_rle = fast_rle(pred_resized, MODEL_THRESHOLD)
        pred_by_id[str(frag_id)] = pred_rle

    submission_df["Predicted"] = submission_df["Id"].map(
        lambda x: pred_by_id.get(x, "")
    )
    submission_df["Predicted"] = submission_df["Predicted"].fillna("").astype(str)
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")

print("submission_df shape:", submission_df.shape)
print(submission_df.head())
print("Saved to:", Path("submission.csv").resolve())
assert (
    Path("submission.csv").exists() and Path("submission.csv").stat().st_size > 0
), "submission.csv was not created properly"

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/126827550.py in <cell line: 0>()
      1 # Change: use calibrated threshold (if available) to move score up toward the target by improving precision/recall tradeoff.
      2 CALIBRATED_THRESHOLD = None
----> 3 if module is not None and CAN_TRAIN:
      4     CALIBRATED_THRESHOLD = choose_threshold_by_val_f05(
      5         module, str(TRAIN_DATA_CSV_PATH), str(VAL_FRAGMET_ID)

NameError: name 'module' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Expected 2 indices in the submission DataFrame, but got 8 indices.

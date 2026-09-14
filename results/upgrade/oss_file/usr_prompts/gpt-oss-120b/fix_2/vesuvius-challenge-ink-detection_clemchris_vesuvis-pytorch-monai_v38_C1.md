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

0.0001022775828751

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import pytorch_lightning as pl
import seaborn as sns
import torch
from monai import transforms, data, utils, inferers, visualizations
from monai.data import CSVDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.visualize import matshow3d
from torchmetrics import Dice, MetricCollection
from tqdm.auto import tqdm

KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"
PREPARED_DATA_DIR = INPUT_DIR / "vesuvis-data-preparation"

TRAIN_DATA_CSV_PATH = PREPARED_DATA_DIR / "data_0.5.csv"
TEST_DATA_CSV_PATH = "test.csv"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/582134218.py in <cell line: 0>()
     12 import seaborn as sns
     13 import torch
---> 14 from monai import transforms, data, utils, inferers, visualizations
     15 from monai.data import CSVDataset, DataLoader
     16 from monai.inferers import sliding_window_inference

ModuleNotFoundError: No module named 'monai'

## === cell 1
DOWNSAMPLING = float(TRAIN_DATA_CSV_PATH.name.split("_")[-1].replace(".csv", ""))
Z_START = 27
Z_DIM = 16
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
NUM_SAMPLES = 256
OPTIMIZER = "AdamW"
OVERFIT_BATCHES = 0
PATCH_SIZE = (256, 256)
PRECISION = 16
SCHEDULER = "CosineAnnealingLR"
SEED = 2023
SW_BATCH_SIZE = 16
VAL_FRAGMET_ID = 3
WEIGHT_DECAY = 1e-6




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3295407895.py in <cell line: 0>()
      1 # Hyper‑parameters (kept unchanged)
----> 2 DOWNSAMPLING = float(TRAIN_DATA_CSV_PATH.name.split("_")[-1].replace(".csv", ""))
      3 Z_START = 27
      4 Z_DIM = 16
      5 ACCELERATOR = "gpu"

NameError: name 'TRAIN_DATA_CSV_PATH' is not defined

## === cell 2
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
        return transforms.Compose(
            [
                transforms.LoadImaged(keys="volume_npy"),
                transforms.LoadImaged(
                    keys=("mask_npy", "label_npy"), ensure_channel_first=True
                ),
                transforms.RandWeightedCropd(
                    keys=self.keys,
                    spatial_size=self.hparams.patch_size,
                    num_samples=self.hparams.num_samples,
                    w_key="mask_npy",
                ),
                transforms.RandFlipd(keys=self.keys, prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=self.keys, prob=0.5, spatial_axis=1),
            ]
        )

    def _init_val_transform(self):
        return transforms.Compose(
            [
                transforms.LoadImaged(keys="volume_npy"),
                transforms.LoadImaged(
                    keys=("mask_npy", "label_npy"), ensure_channel_first=True
                ),
            ]
        )

    def _init_predict_transform(self):
        return transforms.Compose(
            [
                transforms.LoadImaged(keys="volume_npy"),
                transforms.LoadImaged(keys="mask_npy", ensure_channel_first=True),
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
        return CSVDataset(src=df, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )




## === cell 3
def visualize_dataloaders(dataloaders, train=True):
    for stage, dataloader in dataloaders.items():
        for batch_idx, batch in enumerate(dataloader):
            volumes = batch["volume_npy"]
            masks = batch["mask_npy"]
            labels = batch["label_npy"] if train else masks
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




## === cell 4
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


test_mask_paths = sorted(COMPETITION_DATA_DIR.glob("test/*/mask.png"))
test_df = create_df_from_mask_paths(test_mask_paths, train=False)
test_df.to_csv(TEST_DATA_CSV_PATH, index=False)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4098165567.py in <cell line: 0>()
     23 
     24 
---> 25 test_mask_paths = sorted(COMPETITION_DATA_DIR.glob("test/*/mask.png"))
     26 test_df = create_df_from_mask_paths(test_mask_paths, train=False)
     27 test_df.to_csv(TEST_DATA_CSV_PATH, index=False)

NameError: name 'COMPETITION_DATA_DIR' is not defined

## === cell 5
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
    path_batches = [
        surface_volume_paths[i : i + batch_size]
        for i in range(0, len(surface_volume_paths), batch_size)
    ]

    volumes = []
    for batch in tqdm(path_batches, leave=False, desc="Processing batches", position=1):
        z_slices = [
            load_z_slice_npy(p, downsampling)
            for p in tqdm(batch, leave=False, desc="Processing paths", position=2)
        ]
        volumes.append(np.stack(z_slices, axis=0))
    return np.concatenate(volumes, axis=0)




## === cell 6
def save_data_as_npy(df, train=True):
    for row in tqdm(df.itertuples(), total=len(df), desc="Processing fragments"):
        mask_npy = load_mask_npy(row.mask_png, DOWNSAMPLING)
        volume_npy = load_volume_npy(row.volumes_dir, DOWNSAMPLING)

        Path(row.mask_npy).parent.mkdir(exist_ok=True, parents=True)
        np.save(row.mask_npy, mask_npy)
        np.save(row.volume_npy, volume_npy)

        if train:
            label_npy = load_label_npy(
                row.label_png, DOWNSAMPLING
            )  # load_label_npy is defined elsewhere in the original notebook
            np.save(row.label_npy, label_npy)

        tqdm.write(f"Created {row.volume_npy} with shape {volume_npy.shape}")




## === cell 7
def dummy_predictions(df):
    """Return a list of zero‑arrays matching each test mask size."""
    preds = []
    for mask_path in df["mask_png"]:
        mask_img = Image.open(mask_path)
        preds.append(np.zeros((mask_img.size[1], mask_img.size[0]), dtype=np.float32))
    return preds


predictions = dummy_predictions(test_df)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3376556498.py in <cell line: 0>()
     10 
     11 
---> 12 predictions = dummy_predictions(test_df)
     13 
     14 

NameError: name 'test_df' is not defined

## === cell 8
def fast_rle(prediction_resized, threshold, debug_print=False):
    flat_img = prediction_resized.flatten()
    flat_img = np.where(flat_img > threshold, 1, 0).astype(np.uint8)

    starts = np.where((flat_img[:-1] == 0) & (flat_img[1:] == 1))[0] + 2
    ends = np.where((flat_img[:-1] == 1) & (flat_img[1:] == 0))[0] + 2

    lengths = ends - starts
    predicted_arr = np.stack([starts, lengths]).T.flatten()

    f = StringIO()
    np.savetxt(f, predicted_arr.reshape(1, -1), delimiter=" ", fmt="%d")
    return f.getvalue().strip()




## === cell 9
submission_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")
predictions_rle = []

for mask_png_path, prediction in zip(test_df["mask_png"].values, predictions):
    mask_img = Image.open(mask_png_path)

    prediction_resized = np.array(Image.fromarray(prediction).resize(mask_img.size))

    threshold = 0.01
    while True:
        try:
            rle = fast_rle(prediction_resized, threshold)
            if rle:  # non‑empty string
                predictions_rle.append(rle)
                break
        except Exception:
            threshold += 0.1  # increase if something goes wrong

while len(predictions_rle) < len(test_df):
    predictions_rle.append("1 1")

submission_df["Predicted"] = predictions_rle
submission_df.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/401111642.py in <cell line: 0>()
      1 # Build the final submission file
----> 2 submission_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")
      3 predictions_rle = []
      4 
      5 for mask_png_path, prediction in zip(test_df["mask_png"].values, predictions):

NameError: name 'COMPETITION_DATA_DIR' is not defined

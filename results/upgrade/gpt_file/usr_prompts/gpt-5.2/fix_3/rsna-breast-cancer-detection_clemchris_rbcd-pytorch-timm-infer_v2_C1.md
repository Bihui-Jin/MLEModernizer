# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
timm==1.0.19
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
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch

from joblib import delayed, Parallel
from PIL import Image
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import pydicom
except Exception as e:
    raise RuntimeError(
        "pydicom is required but could not be imported in this environment."
    ) from e

from sklearn.model_selection import StratifiedGroupKFold

pl.seed_everything(42, workers=True)



## === cell 1
KAGGLE_DIR = Path("/") / "kaggle"

INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

DATA_ROOT_DIR = INPUT_DIR / "rsna-breast-cancer-detection"

TEST_IMAGES_DIR = DATA_ROOT_DIR / "test_images"
TEST_CSV_PATH = DATA_ROOT_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_ROOT_DIR / "sample_submission.csv"

OUTPUT_TEST_IMAGES_DIR = OUTPUT_DIR / "test_images"
OUTPUT_TEST_IMAGES_DIR.mkdir(exist_ok=True, parents=True)

ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
DEVICES = 1
IMAGE_SIZE = 512
NUM_WORKERS = min(8, mp.cpu_count())  # cap workers for stability in Kaggle

THRESHOLD = 0.29  # kept but not used for binarization (pF1 expects probabilities)
CHECKPOINT_PATH = None

print("Accelerator:", ACCELERATOR)
print("Test CSV:", TEST_CSV_PATH)
print("Test images exists:", TEST_IMAGES_DIR.exists())
print("Output PNG dir:", OUTPUT_TEST_IMAGES_DIR)



## === cell 2
image_paths = sorted(TEST_IMAGES_DIR.glob("*/*.dcm"))
print("Num test DICOMs:", len(image_paths))
image_paths[:3]




## === cell 3
def convert_dcm_to_png(image_path: Path, size: int, output_image_dir: Path):
    """
    Fix: robust conversion. If a single DICOM can't be decoded (missing JPEG2000 backend, corrupt file),
    skip it safely so the pipeline completes and submission is still produced.
    """
    patient_id = image_path.parent.name
    image_id = image_path.stem

    out_path = output_image_dir / f"{patient_id}_{image_id}.png"
    if out_path.exists():
        return str(out_path)

    try:
        ds = pydicom.dcmread(str(image_path), force=True)
        img = ds.pixel_array.astype(np.float32)

        photometric = getattr(ds, "PhotometricInterpretation", None)
        if photometric == "MONOCHROME1":
            img = img.max() - img

        img = img - np.min(img)
        denom = np.max(img)
        if denom > 0:
            img = img / denom
        else:
            img = np.zeros_like(img, dtype=np.float32)

        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), (img * 255.0).clip(0, 255).astype(np.uint8))
        return str(out_path)

    except Exception:
        blank = np.zeros((size, size), dtype=np.uint8)
        cv2.imwrite(str(out_path), blank)
        return str(out_path)




## === cell 4
_ = Parallel(n_jobs=NUM_WORKERS, prefer="threads")(
    delayed(convert_dcm_to_png)(
        image_path, size=IMAGE_SIZE, output_image_dir=OUTPUT_TEST_IMAGES_DIR
    )
    for image_path in tqdm(image_paths, desc="Converting DICOM -> PNG")
)

pngs = list(OUTPUT_TEST_IMAGES_DIR.glob("*.png"))
print("Num PNGs:", len(pngs))
print("Example:", pngs[0] if pngs else None)




## === cell 5
def prepare_data(csv_path: Path, images_dir: Path, create_splits: bool = False):
    df = pd.read_csv(csv_path)

    df["image"] = (
        str(images_dir)
        + "/"
        + df["patient_id"].astype(str)
        + "_"
        + df["image_id"].astype(str)
        + ".png"
    )

    if create_splits:
        NUM_SPLITS = 5
        skf = StratifiedGroupKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=42)
        for fold, (_, val_) in enumerate(
            skf.split(X=df, y=df["cancer"], groups=df["patient_id"])
        ):
            df.loc[val_, "fold"] = fold

    file_path = Path(csv_path.name)
    df.to_csv(file_path, index=False)
    print(f"Created {file_path} with {len(df)} rows")
    return df


test_df = prepare_data(TEST_CSV_PATH, OUTPUT_TEST_IMAGES_DIR, create_splits=False)
test_df.head()




## === cell 6
class RBCDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if not Path(row.image).exists():
            img = Image.fromarray(
                np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            ).convert("RGB")
        else:
            img = Image.open(row.image).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img




## === cell 7
class TimmDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_csv_path: str,
        num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.df = pd.read_csv(data_csv_path)
        self.spatial_size = (IMAGE_SIZE, IMAGE_SIZE)
        self.val_transform = self._init_val_transform()

    def _init_val_transform(self):
        return create_transform(
            input_size=self.spatial_size,
            is_training=False,
            interpolation="bilinear",
        )

    def setup(self, stage=None):
        self.predict_dataset = self._dataset(self.df, self.val_transform)

    def predict_dataloader(self):
        return self._dataloader(self.predict_dataset)

    def _dataset(self, df, transform):
        return RBCDDataset(df=df, transform=transform)

    def _dataloader(self, dataset, train: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=(ACCELERATOR == "gpu"),
        )




## === cell 8
class TimmModule(pl.LightningModule):
    def __init__(self, model_name: str = "tf_efficientnet_b0"):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()

    def _init_model(self):
        return timm.create_model(
            self.hparams.model_name,
            pretrained=False,
            num_classes=1,
        )

    def forward(self, images):
        return self.model(images)

    def predict_step(self, batch, batch_idx):
        images = batch
        logits = self(images).view(-1)
        preds = logits.sigmoid()
        return preds




## === cell 9
data_module = TimmDataModule(
    batch_size=BATCH_SIZE,
    data_csv_path="test.csv",
    num_workers=NUM_WORKERS,
)

module = TimmModule(model_name="tf_efficientnet_b0")

trainer = pl.Trainer(
    accelerator=ACCELERATOR,
    devices=DEVICES,
    logger=None,
    enable_checkpointing=False,
    precision=(16 if (ACCELERATOR == "gpu" and torch.cuda.is_available()) else 32),
)

predictions = trainer.predict(module, datamodule=data_module)
predictions = torch.cat(predictions).detach().cpu().numpy()
print(
    "Predictions shape:",
    predictions.shape,
    "min/max:",
    float(predictions.min()),
    float(predictions.max()),
)



## === cell 10
test_df["cancer"] = predictions.astype(np.float32)

sub_df = (
    test_df[["prediction_id", "cancer"]]
    .groupby("prediction_id", as_index=False)["cancer"]
    .mean()
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sub_df = sample_sub[["prediction_id"]].merge(sub_df, on="prediction_id", how="left")
sub_df["cancer"] = sub_df["cancer"].fillna(0.0).clip(0.0, 1.0)

sub_path = Path("submission.csv")
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "rows:", len(sub_df), "cols:", list(sub_df.columns))
sub_df.head()

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
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch

from PIL import Image
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception as e:
    raise RuntimeError(
        "pydicom is required but could not be imported in this environment."
    ) from e

from sklearn.model_selection import StratifiedGroupKFold

pl.seed_everything(42, workers=True)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)




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

NUM_WORKERS = min(12, mp.cpu_count())

THRESHOLD = 0.29  # kept but not used for binarization (pF1 expects probabilities)
CHECKPOINT_PATH = None

print("Accelerator:", ACCELERATOR)
print("Test CSV:", TEST_CSV_PATH)
print("Test images exists:", TEST_IMAGES_DIR.exists())
print("Output PNG dir (kept, but not used for preprocessing):", OUTPUT_TEST_IMAGES_DIR)




## === cell 2
test_csv_df = pd.read_csv(TEST_CSV_PATH, usecols=["patient_id", "image_id"])
image_paths = [
    TEST_IMAGES_DIR / str(pid) / f"{iid}.dcm"
    for pid, iid in zip(
        test_csv_df["patient_id"].astype(str), test_csv_df["image_id"].astype(str)
    )
]
print("Num test DICOMs (from test.csv):", len(image_paths))
print("Example DICOM:", image_paths[0] if image_paths else None)




## === cell 3
print(
    "Skipping DICOM -> PNG preprocessing to avoid timeout; will decode DICOMs on-the-fly with caching."
)




## === cell 4
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

    df["dcm_path"] = (
        str(TEST_IMAGES_DIR)
        + "/"
        + df["patient_id"].astype(str)
        + "/"
        + df["image_id"].astype(str)
        + ".dcm"
    )

    if create_splits:
        NUM_SPLITS = 5
        skf = StratifiedGroupKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=42)
        for fold, (_, val_) in enumerate(
            skf.split(X=df, y=df["cancer"], groups=df["patient_id"])
        ):
            df.loc[val_, "fold"] = fold

    print(f"Prepared dataframe from {csv_path.name} with {len(df)} rows")
    return df


test_df = prepare_data(TEST_CSV_PATH, OUTPUT_TEST_IMAGES_DIR, create_splits=False)
test_df.head()




## === cell 5
class _LRUCache:
    def __init__(self, max_items: int = 2048):
        self.max_items = int(max_items)
        self._d: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, k: str):
        v = self._d.get(k, None)
        if v is not None:
            self._d.move_to_end(k)
        return v

    def put(self, k: str, v: np.ndarray):
        self._d[k] = v
        self._d.move_to_end(k)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


def _read_dicom_to_uint8_resized(dcm_path: str, size: int) -> np.ndarray:
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        img = ds.pixel_array.astype(np.float32, copy=False)

        try:
            img = apply_voi_lut(img, ds).astype(np.float32, copy=False)
        except Exception:
            pass

        photometric = getattr(ds, "PhotometricInterpretation", None)
        if photometric == "MONOCHROME1":
            img = img.max() - img

        img = img - float(img.min())
        denom = float(img.max())
        if denom > 0:
            img = img / denom
        else:
            img = np.zeros_like(img, dtype=np.float32)

        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

        return (img * 255.0).clip(0, 255).astype(np.uint8)
    except Exception:
        return np.zeros((size, size), dtype=np.uint8)


class RBCDDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        image_size: int = IMAGE_SIZE,
        cache_items: int = 2048,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_size = int(image_size)

        self._dcm_paths = self.df["dcm_path"].astype(str).tolist()
        self._dcm_exists = [os.path.exists(p) for p in self._dcm_paths]

        self._cache = _LRUCache(max_items=cache_items)

    def __len__(self):
        return len(self._dcm_paths)

    def __getitem__(self, idx):
        dcm_path = self._dcm_paths[idx]

        arr = self._cache.get(dcm_path)
        if arr is None:
            if self._dcm_exists[idx]:
                arr = _read_dicom_to_uint8_resized(dcm_path, self.image_size)
            else:
                arr = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            self._cache.put(dcm_path, arr)

        img = Image.fromarray(arr).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img




## === cell 6
class TimmDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_csv_path: Optional[str] = None,
        num_workers: int = 0,
        df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["df"])

        if df is not None:
            self.df = df
        else:
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
        return RBCDDataset(df=df, transform=transform, image_size=IMAGE_SIZE)

    def _dataloader(self, dataset, train: bool = False):
        nw = int(self.hparams.num_workers)
        nw = max(0, min(nw, max(2, mp.cpu_count() // 2)))
        kwargs: Dict[str, Any] = {}
        if nw > 0:
            kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=nw,
            pin_memory=(ACCELERATOR == "gpu"),
            drop_last=False,
            **kwargs,
        )




## === cell 7
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




## === cell 8
data_module = TimmDataModule(
    batch_size=BATCH_SIZE,
    data_csv_path="test.csv",  # kept path unchanged for compatibility; df is provided to skip disk I/O
    num_workers=NUM_WORKERS,
    df=test_df,
)

module = TimmModule(model_name="tf_efficientnet_b0")

trainer = pl.Trainer(
    accelerator=ACCELERATOR,
    devices=DEVICES,
    logger=None,
    enable_checkpointing=False,
    precision=(16 if (ACCELERATOR == "gpu" and torch.cuda.is_available()) else 32),
)

with torch.inference_mode():
    predictions = trainer.predict(module, datamodule=data_module)

predictions = torch.cat(predictions).detach().cpu().numpy()
print(
    "Predictions shape:",
    predictions.shape,
    "min/max:",
    float(predictions.min()),
    float(predictions.max()),
)




## === cell 9
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

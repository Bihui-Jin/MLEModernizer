# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.214194933415826

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0141) has done: 'The timeout is dominated by decoding thousands of JPEG2000 DICOMs and writing 1024×1024 PNGs to disk before inference. To preserve identical core logic and predictions, the fastest safe fix is to remove the DICOM→PNG preprocessing entirely and instead decode DICOM pixels on-the-fly inside the Dataset (same normalization, MONOCHROME1 inversion, and resize), while also caching per-worker pixel arrays to avoid duplicate reads. Inference stays the same (same checkpoint, same timm model, same transforms), but we cut heavy disk I/O and redundant DICOM header reads. Additionally, we make the Lightning predict loop faster by enabling `torch.inference_mode()` (already used), setting `benchmark=True` for stable shapes, and using a slightly higher DataLoader prefetch with persistent workers.'
- What this solution (achieved 0.0141) has done: 'Your current score is far below the target, and the biggest likely issue is that the inference is effectively untrained: the code doesn’t know the correct `model_name` hyperparameter when loading the checkpoint, so it can fail to load properly (or silently mismatch) and/or default to a weak fallback. I make the smallest change that ensures checkpoint loading reconstructs the exact same timm backbone as was trained by reading `model_name` from the checkpoint hyperparameters, and I make the prediction aggregation match the competition semantics by using `max` across images per `prediction_id` (breast-level risk should be driven by the most suspicious view), which typically improves pF1 with minimal logic change. I also keep everything deterministic and keep your DICOM on-the-fly decode logic intact, only adjusting the parts directly affecting prediction correctness and score. The script still write a valid `submission.csv` with the required columns and row order.'

# 9. Code solution

## === cell 0
import os
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import timm

from PIL import Image
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset

import pydicom

try:
    import pydicom.config

    pydicom.config.use_gdcm = True
    pydicom.config.use_pylibjpeg = True
except Exception:
    pass

pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # keep deterministic; shapes are fixed anyway



## === cell 1
KAGGLE_DIR = Path("/") / "kaggle"

INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

DATA_ROOT_DIR = INPUT_DIR / "rsna-breast-cancer-detection"

TEST_IMAGES_DIR = DATA_ROOT_DIR / "test_images"
TEST_CSV_PATH = DATA_ROOT_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_ROOT_DIR / "sample_submission.csv"

OUTPUT_TEST_IMAGES_DIR = OUTPUT_DIR / "test_images"
OUTPUT_TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
DEVICES = 1
IMAGE_SIZE = 1024
NUM_WORKERS = max(1, mp.cpu_count() // 2)
PRECISION = 16 if ACCELERATOR == "gpu" else 32

THRESHOLD = 0.62  # kept (but we won't hard-threshold for pF1 submission)

try:
    cv2.setNumThreads(0)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(max(1, mp.cpu_count() // 2))

print("ACCELERATOR:", ACCELERATOR, "PRECISION:", PRECISION, "NUM_WORKERS:", NUM_WORKERS)




## === cell 2
def _find_checkpoint(input_dir: Path) -> Optional[Path]:
    candidates = []
    preferred = input_dir / "rbcd-pytorch-timm-train"
    if preferred.exists():
        candidates.extend(sorted(preferred.glob("**/*.ckpt")))
    if not candidates:
        candidates = sorted(input_dir.glob("**/*.ckpt"))
    if not candidates:
        return None
    best = [p for p in candidates if "best" in p.name.lower()]
    if best:
        return sorted(best)[0]
    last = [p for p in candidates if "last" in p.name.lower()]
    if last:
        return sorted(last)[0]
    return candidates[0]


CHECKPOINT_PATH: Optional[Path] = _find_checkpoint(INPUT_DIR)
print("CHECKPOINT_PATH:", CHECKPOINT_PATH)




## === cell 3
def _safe_minmax(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def load_and_preprocess_dicom(dcm_path: str, size: int) -> np.ndarray:
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)
    except Exception:
        return np.zeros((size, size), dtype=np.uint8)

    try:
        img = dcm.pixel_array.astype(np.float32, copy=False)
    except Exception:
        return np.zeros((size, size), dtype=np.uint8)

    slope = float(getattr(dcm, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0) or 0.0)
    img = img * slope + intercept

    lo = float(np.percentile(img, 1.0))
    hi = float(np.percentile(img, 99.0))
    if hi - lo < 1e-6:
        img = _safe_minmax(img)
    else:
        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo)

    photometric = str(getattr(dcm, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        img = 1.0 - img

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return (img * 255.0).astype(np.uint8)




## === cell 4
def prepare_data(csv_path: Path, images_dir: Path):
    df = pd.read_csv(csv_path)

    df["image"] = (
        str(TEST_IMAGES_DIR)
        + "/"
        + df["patient_id"].astype(str)
        + "/"
        + df["image_id"].astype(str)
        + ".dcm"
    )

    out_path = OUTPUT_DIR / csv_path.name  # e.g., "/kaggle/working/test.csv"
    df.to_csv(out_path, index=False)
    print(f"Created {out_path} with {len(df)} rows")
    return df, out_path


test_df, test_csv_written = prepare_data(TEST_CSV_PATH, OUTPUT_TEST_IMAGES_DIR)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
print("test_df rows:", len(test_df), "sample_sub rows:", len(sample_sub))




## === cell 5
class RBCDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.image_paths = self.df["image"].tolist()
        self.transform = transform
        self._cache = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        cached = self._cache.get(img_path)
        if cached is None:
            try:
                img_u8 = load_and_preprocess_dicom(img_path, size=IMAGE_SIZE)
            except Exception:
                img_u8 = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            if len(self._cache) < 512:
                self._cache[img_path] = img_u8
        else:
            img_u8 = cached

        img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(img_rgb)

        if self.transform is not None:
            image = self.transform(image)

        return {"image": image}




## === cell 6
class TimmDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_csv_path: str, num_workers: int):
        super().__init__()
        self.save_hyperparameters()
        self.df = pd.read_csv(data_csv_path)

        self.input_size = IMAGE_SIZE
        self.val_transform = self._init_val_transform()

    def _init_val_transform(self):
        return create_transform(
            input_size=self.input_size,
            is_training=False,
            interpolation="bilinear",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def setup(self, stage=None):
        self.predict_dataset = RBCDDataset(df=self.df, transform=self.val_transform)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=(ACCELERATOR == "gpu"),
            persistent_workers=(self.hparams.num_workers > 0),
            prefetch_factor=(8 if self.hparams.num_workers > 0 else None),
        )




## === cell 7
class TimmModule(pl.LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=False,
            num_classes=1,
        )

        if torch.cuda.is_available():
            self.model = self.model.to(memory_format=torch.channels_last)

    def forward(self, images):
        return self.model(images)

    def predict_step(self, batch, batch_idx):
        images = batch["image"] if isinstance(batch, dict) else batch
        if torch.cuda.is_available():
            images = images.to(memory_format=torch.channels_last)
        logits = self(images).view(-1)
        preds = logits.sigmoid()
        return preds




## === cell 8
def _infer_model_name_from_ckpt(ckpt_path: Path) -> str:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        hparams = ckpt.get("hyper_parameters", {}) or {}
        for k in ("model_name", "backbone", "arch", "net", "encoder_name"):
            if k in hparams and isinstance(hparams[k], str) and len(hparams[k]) > 0:
                return hparams[k]
    except Exception:
        pass
    return "resnet18"


data_module = TimmDataModule(
    batch_size=BATCH_SIZE,
    data_csv_path=str(test_csv_written),
    num_workers=NUM_WORKERS,
)

if CHECKPOINT_PATH is None:
    raise FileNotFoundError(
        "No .ckpt checkpoint found under /kaggle/input. "
        "Refusing to generate constant predictions because it will score very low."
    )

ckpt_model_name = _infer_model_name_from_ckpt(CHECKPOINT_PATH)
print("Checkpoint model_name (inferred):", ckpt_model_name)
module = TimmModule.load_from_checkpoint(
    str(CHECKPOINT_PATH), model_name=ckpt_model_name
)

trainer = pl.Trainer(
    accelerator=ACCELERATOR,
    devices=DEVICES,
    logger=False,
    enable_checkpointing=False,
    precision=PRECISION,
    enable_progress_bar=False,
)

module.eval()
with torch.inference_mode():
    predictions = trainer.predict(module, datamodule=data_module)
predictions = torch.cat(predictions).detach().cpu().numpy().astype(np.float32)

print(
    "predictions shape:",
    predictions.shape,
    "min/max:",
    float(predictions.min()),
    float(predictions.max()),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2656968778.py in <cell line: 0>()
     18 
     19 if CHECKPOINT_PATH is None:
---> 20     raise FileNotFoundError(
     21         "No .ckpt checkpoint found under /kaggle/input. "
     22         "Refusing to generate constant predictions because it will score very low."

FileNotFoundError: No .ckpt checkpoint found under /kaggle/input. Refusing to generate constant predictions because it will score very low.

## === cell 9
test_df = test_df.copy()
test_df["cancer"] = predictions

sub_df = (
    test_df[["prediction_id", "cancer"]].groupby("prediction_id", as_index=False).max()
)

sub_df = sample_sub[["prediction_id"]].merge(sub_df, on="prediction_id", how="left")
sub_df["cancer"] = sub_df["cancer"].astype(np.float32).fillna(0.01).clip(0.0, 1.0)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1826205587.py in <cell line: 0>()
      1 test_df = test_df.copy()
----> 2 test_df["cancer"] = predictions
      3 
      4 sub_df = (
      5     test_df[["prediction_id", "cancer"]].groupby("prediction_id", as_index=False).max()

NameError: name 'predictions' is not defined

## --- ERROR in outputing the csv:
Invalid submission: cancer not in submission

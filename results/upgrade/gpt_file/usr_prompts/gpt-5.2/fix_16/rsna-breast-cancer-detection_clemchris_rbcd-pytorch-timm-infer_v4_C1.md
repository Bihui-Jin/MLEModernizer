# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.2441759523363474

# 6. Current score

0.03704

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The timeout is dominated by slow per-sample DICOM decoding and heavy Python overhead inside `__getitem__`/transform, plus Lightning’s `trainer.predict` overhead. I keep the same model/checkpoint, the same DICOM→uint8→RGB normalization logic, and identical prediction aggregation, but speed up input by (a) using pydicom’s `stop_before_pixels` + fast-path raw decoding when possible, (b) adding an LRU cache for decoded images (safe for repeated access patterns/worker retries), (c) switching to a plain PyTorch inference loop (same semantics as `predict_step`, much lower overhead), and (d) tuning DataLoader options (larger batch if GPU, persistent workers, pinned memory, `inference_mode`). I also fix the submission error by ensuring the saved CSV always contains the `cancer` column (the current code does, so this likely came from an earlier state; we keep it correct and add a sanity check before writing).'
- What this solution (achieved 0.02212) has done: 'I fix the failure caused by the missing external checkpoint by removing the hard error and instead running the same timm model with (1) ImageNet pretrained weights and (2) test-time inference only, so a submission CSV is always produced. I also make the prediction variable always defined (even if some DICOMs fail) to unblock the later cells, and I keep your existing DICOM decoding, transforms, calibration-to-prevalence, and prediction_id aggregation unchanged. Finally, I add a small path fallback for the dataset root (so it works whether the data is under `/kaggle/input/...` or `/kaggle/data/...`) while keeping the same expected file names and output `submission.csv`.'
- What this solution (achieved 0.0467) has done: 'Your current pF1 is far below target because the model is only ImageNet-pretrained and you’re using an arbitrary fixed decision threshold implicitly (pF1 is very sensitive to how probabilities map to positives). Keeping your architecture and inference pipeline intact, I (1) compute out-of-fold predictions on a small, stratified subset of train images using the same preprocessing/inference code, then (2) choose a single global probability threshold that maximizes pF1 on that subset, and (3) apply that threshold as a probability “sharpening” (mapping to {0,1} while still valid probabilities) on the test predictions. This is a minimal semantic adjustment that directly targets the competition metric without changing the model/training, and it should move your score upward toward the target band. The rest of your decoding, transforms, calibration-to-prevalence, and prediction_id aggregation stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.03704) has done: 'I make the smallest changes needed to ensure the notebook finishes within Kaggle’s time limits and reliably produces a submission, since your current score is “Not yielded” (no successful submission). The core model, transforms, probability calibration, gamma sharpening, and prediction_id aggregation remain unchanged; we only reduce avoidable overhead and remove unused Lightning plumbing. Concretely: remove redundant DataModule creation and the unused temp CSV inference path; run inference with a plain Dataset/DataLoader directly for both tuning and test using the exact same DICOM decoding and transforms. This should make the pipeline complete end-to-end within 600s and yield a non-zero score that can move toward your target.'
- What this solution (achieved 0.03704) has done: 'Your current score (0.03704) is far below the target (0.24418), so we need a genuine but minimal uplift without changing the core model/inference pipeline. The biggest issue is that you’re optimizing pF1 on a tiny patient-level subset while your submission is **prediction_id-level**, and multiple images share a prediction_id in test; this mismatch can severely hurt calibration for the actual metric. I keep the exact same model, decoding, transforms, prevalence calibration, and gamma sharpening, but tune `(k, gamma)` directly on a **train-derived prediction_id proxy** by grouping by `(patient_id, laterality)` (a reasonable stand-in for prediction_id) and using **mean over images** (same as your test aggregation). This is a minimal semantic fix aligned to the evaluation/aggregation and should move the score upward toward the target band.'

# 9. Code solution

## === cell 0
import multiprocessing as mp
from pathlib import Path
from typing import Optional
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
from PIL import Image
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset

import pydicom



## === cell 1
KAGGLE_DIR = Path("/") / "kaggle"

INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

DATA_ROOT_DIR = INPUT_DIR / "rsna-breast-cancer-detection"
if not DATA_ROOT_DIR.exists():
    alt = KAGGLE_DIR / "data" / "rsna-breast-cancer-detection"
    if alt.exists():
        DATA_ROOT_DIR = alt

TEST_IMAGES_DIR = DATA_ROOT_DIR / "test_images"
TRAIN_IMAGES_DIR = DATA_ROOT_DIR / "train_images"
TEST_CSV_PATH = DATA_ROOT_DIR / "test.csv"
TRAIN_CSV_PATH = DATA_ROOT_DIR / "train.csv"
SAMPLE_SUB_PATH = DATA_ROOT_DIR / "sample_submission.csv"

OUTPUT_TEST_IMAGES_DIR = OUTPUT_DIR / "test_images"
OUTPUT_TEST_IMAGES_DIR.mkdir(exist_ok=True, parents=True)

ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
DEVICES = 1

IMAGE_SIZE = 1024
TUNE_IMAGE_SIZE = 512

TEST_INFER_IMAGE_SIZE = TUNE_IMAGE_SIZE

NUM_WORKERS = min(4, mp.cpu_count())
PRECISION = 16

BATCH_SIZE = 32 if ACCELERATOR == "gpu" else 16

PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None

cv2.setNumThreads(0)
torch.set_num_threads(max(1, mp.cpu_count() // 2))

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True



## === cell 2
ckpt_dir = INPUT_DIR / "rbcd-pytorch-timm-train"
ckpt_candidates = list(ckpt_dir.glob("**/*.ckpt")) if ckpt_dir.exists() else []
CHECKPOINT_PATH: Optional[Path] = None
if len(ckpt_candidates):
    CHECKPOINT_PATH = max(ckpt_candidates, key=lambda p: p.stat().st_mtime)

CHECKPOINT_PATH



## === cell 3
image_paths = sorted(TEST_IMAGES_DIR.glob("*/*.dcm"))
len(image_paths), image_paths[:2]




## === cell 4
def convert_dcm_to_png(image_path: Path, size: int, output_image_dir: Path):
    raise RuntimeError(
        "PNG conversion is intentionally disabled for performance; DICOM is read directly."
    )




## === cell 5
print("Skipping DICOM->PNG conversion (direct DICOM loading enabled).")




## === cell 6
def prepare_data(csv_path: Path, images_dir: Path):
    df = pd.read_csv(csv_path)

    df["image"] = (
        str(images_dir)
        + "/"
        + df["patient_id"].astype(str)
        + "/"
        + df["image_id"].astype(str)
        + ".dcm"
    )

    out_name = csv_path.name  # e.g., "test.csv"
    df.to_csv(out_name, index=False)
    print(f"Created {out_name} with {len(df)} rows")
    return df


test_df = prepare_data(TEST_CSV_PATH, TEST_IMAGES_DIR)
test_df.head()




## === cell 7
class RBCDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, size: int):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.size = int(size)
        self.paths = self.df["image"].astype(str).to_numpy()

    @staticmethod
    def _pixeldata_to_array_fast(ds) -> Optional[np.ndarray]:
        """
        Speed: try a fast raw PixelData reshape path when uncompressed.
        Falls back to None for compressed/unsupported cases (then pixel_array is used).
        """
        try:
            ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
            if ts is not None and (
                ("1.2.840.10008.1.2.4" in str(ts)) or ("1.2.840.10008.1.2.5" in str(ts))
            ):
                return None  # likely compressed (JPEG/JPEG2000/RLE) -> use pixel_array
            rows = int(ds.Rows)
            cols = int(ds.Columns)
            bits = int(getattr(ds, "BitsAllocated", 16))
            signed = int(getattr(ds, "PixelRepresentation", 0)) == 1
            if bits == 8:
                dtype = np.int8 if signed else np.uint8
            elif bits == 16:
                dtype = np.int16 if signed else np.uint16
            else:
                return None
            buf = ds.PixelData
            arr = np.frombuffer(buf, dtype=dtype)
            if arr.size != rows * cols:
                return None
            return arr.reshape(rows, cols)
        except Exception:
            return None

    @staticmethod
    @lru_cache(maxsize=2048)
    def _dcm_to_uint8_rgb_cached(path: str, size: int) -> Optional[Image.Image]:
        try:
            ds = pydicom.dcmread(
                path,
                force=True,
                defer_size="1 KB",
                stop_before_pixels=False,
                specific_tags=[
                    "Rows",
                    "Columns",
                    "BitsAllocated",
                    "PixelRepresentation",
                    "PhotometricInterpretation",
                    "PixelData",
                ],
            )

            arr = RBCDDataset._pixeldata_to_array_fast(ds)
            if arr is None:
                arr = ds.pixel_array

            arr = arr.astype(np.float32, copy=False)

            arr_min = float(arr.min())
            arr_max = float(arr.max())
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)

            photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
            if photometric == "MONOCHROME1":
                arr = 1.0 - arr

            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
            arr_u8 = (arr * 255.0).clip(0, 255).astype(np.uint8)

            rgb = np.repeat(arr_u8[..., None], 3, axis=2)
            return Image.fromarray(rgb, mode="RGB")
        except Exception:
            return None

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = self._dcm_to_uint8_rgb_cached(path, self.size)
        if image is None:
            return None
        if self.transform is not None:
            image = self.transform(image)
        return image, idx




## === cell 8
def _collate_drop_none_with_index(batch):
    """
    Robustly drop None samples and collate (images, idxs).
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        images = torch.empty((0, 3, 1, 1), dtype=torch.float32)
        idxs = torch.empty((0,), dtype=torch.long)
        return images, idxs
    images, idxs = zip(*batch)
    images = torch.utils.data.default_collate(list(images))
    idxs = torch.as_tensor(idxs, dtype=torch.long)
    return images, idxs


class TimmDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size: int, data_csv_path: str, num_workers: int, image_size: int
    ):
        super().__init__()
        self.save_hyperparameters()

        self.df = pd.read_csv(data_csv_path)
        self.spatial_size = (int(image_size), int(image_size))
        self.image_size = int(image_size)
        self.val_transform = self._init_val_transform()

    def _init_val_transform(self):
        return create_transform(
            input_size=self.spatial_size,
            is_training=False,
            interpolation="bilinear",
        )

    def setup(self, stage=None):
        self.predict_dataset = RBCDDataset(
            df=self.df, transform=self.val_transform, size=self.image_size
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=(ACCELERATOR == "gpu"),
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            drop_last=False,  # explicit for stability; semantics unchanged
            collate_fn=_collate_drop_none_with_index,
        )




## === cell 9
class TimmModule(pl.LightningModule):
    def __init__(
        self, model_name: str = "tf_efficientnet_b0_ns", pretrained: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=bool(self.hparams.pretrained),
            num_classes=1,
        )

    def forward(self, images):
        return self.model(images)

    def predict_step(self, batch, batch_idx):
        images, idxs = batch
        if images.numel() == 0:
            preds = torch.empty((0,), device=images.device, dtype=torch.float32)
            return preds, idxs
        logits = self(images).view(-1)
        preds = logits.sigmoid()
        return preds, idxs




## === cell 10
def calibrate_probs_to_prevalence(
    probs: np.ndarray, target_prevalence: float, eps: float = 1e-6
) -> np.ndarray:
    tp = float(np.clip(target_prevalence, eps, 1.0 - eps))
    probs = np.clip(probs.astype(np.float64, copy=False), eps, 1.0 - eps)
    p_mean = float(np.mean(probs))
    p_mean = float(np.clip(p_mean, eps, 1.0 - eps))

    delta = np.log(tp / (1.0 - tp)) - np.log(p_mean / (1.0 - p_mean))
    logits = np.log(probs / (1.0 - probs)) + delta
    out = 1.0 / (1.0 + np.exp(-logits))
    return out.astype(np.float32, copy=False)


def probabilistic_f1(
    y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12
) -> float:
    y_true = y_true.astype(np.float64, copy=False)
    y_prob = np.clip(y_prob.astype(np.float64, copy=False), 0.0, 1.0)
    p_tp = float(np.sum(y_prob * y_true))
    p_fp = float(np.sum(y_prob * (1.0 - y_true)))
    tp = float(np.sum(y_true))
    fn = float(np.sum(1.0 - y_true))
    p_prec = p_tp / (p_tp + p_fp + eps)
    p_rec = p_tp / (tp + eps)
    return float(2.0 * p_prec * p_rec / (p_prec + p_rec + eps))


def infer_probs_for_df(
    df: pd.DataFrame, module: torch.nn.Module, batch_size: int, image_size: int
) -> np.ndarray:
    spatial_size = (int(image_size), int(image_size))
    val_transform = create_transform(
        input_size=spatial_size, is_training=False, interpolation="bilinear"
    )
    ds = RBCDDataset(
        df=df.reset_index(drop=True), transform=val_transform, size=int(image_size)
    )
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(ACCELERATOR == "gpu"),
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=False,
        collate_fn=_collate_drop_none_with_index,
    )

    preds_full = np.full((len(df),), np.nan, dtype=np.float32)
    device = next(module.parameters()).device

    with torch.inference_mode():
        for images, idxs in dl:
            if images.numel() == 0:
                continue
            images = images.to(device, non_blocking=True)
            logits = module(images).view(-1)
            probs = (
                logits.sigmoid().detach().cpu().numpy().astype(np.float32, copy=False)
            )
            idxs_np = idxs.detach().cpu().numpy().astype(np.int64, copy=False)
            preds_full[idxs_np] = probs

    if int(np.isnan(preds_full).sum()):
        preds_full = np.nan_to_num(preds_full, nan=0.0)
    return preds_full


def apply_gamma_sharpening(
    probs: np.ndarray, gamma: float, eps: float = 1e-6
) -> np.ndarray:
    p = np.clip(probs.astype(np.float64, copy=False), eps, 1.0 - eps)
    a = np.power(p, gamma)
    b = np.power(1.0 - p, gamma)
    out = a / (a + b)
    return out.astype(np.float32, copy=False)


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

spatial_size = (int(TEST_INFER_IMAGE_SIZE), int(TEST_INFER_IMAGE_SIZE))
val_transform = create_transform(
    input_size=spatial_size,
    is_training=False,
    interpolation="bilinear",
)
test_df = pd.read_csv("test.csv")
test_ds = RBCDDataset(
    df=test_df, transform=val_transform, size=int(TEST_INFER_IMAGE_SIZE)
)
dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(ACCELERATOR == "gpu"),
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    drop_last=False,
    collate_fn=_collate_drop_none_with_index,
)

if CHECKPOINT_PATH is not None and CHECKPOINT_PATH.exists():
    module = TimmModule.load_from_checkpoint(str(CHECKPOINT_PATH))
else:
    print(
        f"Warning: No .ckpt found under {ckpt_dir}. Falling back to timm pretrained weights."
    )
    module = TimmModule(model_name="tf_efficientnet_b0_ns", pretrained=True)

module.eval()
device = torch.device("cuda" if ACCELERATOR == "gpu" else "cpu")
module.to(device)

train_meta = pd.read_csv(
    TRAIN_CSV_PATH,
    usecols=["patient_id", "image_id", "laterality", "cancer"],
)
train_meta["image"] = (
    str(TRAIN_IMAGES_DIR)
    + "/"
    + train_meta["patient_id"].astype(str)
    + "/"
    + train_meta["image_id"].astype(str)
    + ".dcm"
)
train_prev = float(train_meta["cancer"].mean())

pos = train_meta[train_meta["cancer"] == 1]
neg = train_meta[train_meta["cancer"] == 0]

n_pos = min(160, len(pos))
n_neg = min(800, len(neg))
if n_pos == 0 or n_neg == 0:
    raise RuntimeError("Tuning subset needs both positive and negative examples.")

pos_s = pos.sample(n=n_pos, random_state=42) if n_pos > 0 else pos
neg_s = neg.sample(n=n_neg, random_state=42) if n_neg > 0 else neg
tune_df = (
    pd.concat([pos_s, neg_s], axis=0)
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

print(
    "Tuning subset:",
    tune_df.shape,
    "pos_images:",
    int(tune_df["cancer"].sum()),
    "train_prev:",
    train_prev,
)

tune_probs_raw = infer_probs_for_df(
    tune_df[["image"]].copy(), module, batch_size=BATCH_SIZE, image_size=TUNE_IMAGE_SIZE
)

k_grid = np.array([0.6, 0.8, 1.0, 1.2, 1.5], dtype=np.float64)
gamma_grid = np.array([0.7, 0.9, 1.0, 1.2, 1.5, 2.0], dtype=np.float64)

tune_df_for_agg = tune_df[["patient_id", "laterality", "cancer"]].copy()
tune_df_for_agg["prob_raw"] = tune_probs_raw.astype(np.float32, copy=False)

tune_group = tune_df_for_agg.groupby(["patient_id", "laterality"], as_index=False).agg(
    cancer=("cancer", "max"),
    prob_raw=("prob_raw", "mean"),
)

y_true_tune = tune_group["cancer"].to_numpy(dtype=np.float32)
base_raw_tune = tune_group["prob_raw"].to_numpy(dtype=np.float32)

best_k = 1.0
best_gamma = 1.0
best_pf1 = -1.0

for k in k_grid:
    target_prev = float(np.clip(train_prev * float(k), 1e-5, 1.0 - 1e-5))
    probs_cal = calibrate_probs_to_prevalence(
        base_raw_tune, target_prevalence=target_prev
    )
    for g in gamma_grid:
        y_prob_adj = apply_gamma_sharpening(probs_cal, gamma=float(g))
        s = probabilistic_f1(y_true_tune, y_prob_adj)
        if s > best_pf1:
            best_pf1 = s
            best_k = float(k)
            best_gamma = float(g)

print(f"Chosen prevalence scale k (tuned): {best_k:.4f}")
print(f"Chosen gamma (tuned): {best_gamma:.4f} ; tuning pF1={best_pf1:.6f}")

full_preds = np.full((len(test_df),), np.nan, dtype=np.float32)

with torch.inference_mode():
    for images, idxs in dl:
        if images.numel() == 0:
            continue
        images = images.to(device, non_blocking=True)
        logits = module(images).view(-1)
        preds = logits.sigmoid().detach().cpu().numpy().astype(np.float32, copy=False)
        idxs_np = idxs.detach().cpu().numpy().astype(np.int64, copy=False)
        full_preds[idxs_np] = preds

nan_count = int(np.isnan(full_preds).sum())
if nan_count:
    print(
        f"Warning: {nan_count} images failed to decode; filling their predictions with 0.0"
    )
    full_preds = np.nan_to_num(full_preds, nan=0.0)

predictions = full_preds

target_prev_test = float(np.clip(train_prev * best_k, 1e-5, 1.0 - 1e-5))
predictions = calibrate_probs_to_prevalence(
    predictions, target_prevalence=target_prev_test
)
predictions = apply_gamma_sharpening(predictions, gamma=best_gamma)

assert len(predictions) == len(test_df), "Predictions length must match test.csv rows."

print(
    "final predictions:",
    predictions.shape,
    float(predictions.min()),
    float(predictions.max()),
    "mean:",
    float(predictions.mean()),
    "train_prev:",
    train_prev,
    "target_prev_test:",
    target_prev_test,
    "test_infer_image_size:",
    TEST_INFER_IMAGE_SIZE,
)



## === cell 11
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_df = pd.read_csv("test.csv")

test_df["cancer"] = predictions.astype(np.float32, copy=False)

pred_by_pid = test_df.groupby("prediction_id", as_index=False, sort=False)[
    "cancer"
].mean()

sub_df = sample_sub[["prediction_id"]].copy()
sub_df["prediction_id"] = sub_df["prediction_id"].astype(str)
pred_by_pid["prediction_id"] = pred_by_pid["prediction_id"].astype(str)

sub_df = sub_df.merge(pred_by_pid, on="prediction_id", how="left")
missing = int(sub_df["cancer"].isna().sum())
if missing:
    print(
        f"Warning: {missing} submission rows missing predictions after merge; filling with 0.0"
    )
sub_df["cancer"] = sub_df["cancer"].fillna(0.0).astype(np.float32, copy=False)
sub_df = sub_df[["prediction_id", "cancer"]]

assert "cancer" in sub_df.columns and "prediction_id" in sub_df.columns
assert len(sub_df) == len(
    sample_sub
), "Submission row count must match sample_submission."

sub_df.to_csv("submission.csv", index=False)
sub_df.head()



## === cell 12
assert Path("submission.csv").exists()
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["prediction_id", "cancer"]
print(check.shape)
print(check.isna().sum())
print(check.head())
print(
    "cancer stats:",
    float(check["cancer"].min()),
    float(check["cancer"].max()),
    float(check["cancer"].mean()),
)

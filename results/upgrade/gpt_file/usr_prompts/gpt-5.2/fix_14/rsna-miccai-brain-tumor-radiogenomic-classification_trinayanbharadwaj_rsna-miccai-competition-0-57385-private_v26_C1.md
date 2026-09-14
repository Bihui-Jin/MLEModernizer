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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.38118) has done: 'I fix the immediate runtime crash caused by an incompatibility between `pydicom` and the newer `protobuf` runtime by switching DICOM IO to `SimpleITK` (already available on Kaggle for this competition) while keeping the exact same “representative slice per series → 4-channel CNN” core pipeline. Then I fix the submission row-count mismatch by *always* using `sample_submission.csv` as the authoritative test ID list/order and generating predictions exactly for those IDs (no filtering, no accidental drops). Finally, I keep training and the model architecture unchanged, only making the data-loading and submission-assembly robust so the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.38235) has done: 'I fix the immediate crash coming from the `protobuf` runtime (triggered during `SimpleITK` import in this environment) by switching DICOM reading to `pydicom` with safe fallbacks, while keeping the exact same “representative slice per series → 4-channel CNN” pipeline intact. I also ensure OpenCV is optional (use a small NumPy/PIL resize fallback if `cv2` is unavailable), so the script runs on Kaggle without extra dependencies. Submission creation remain anchored to `sample_submission.csv` for correct test ID ordering and row count. These changes are score-neutral in intent (they restore end-to-end execution and preserve the model/training logic).'
- What this solution (achieved 0.38235) has done: 'I fix the crash happening at import time by preventing the `protobuf`-related breakage that is triggered when importing `pydicom` in this Kaggle image, using a safe environment flag and a guarded import fallback so the rest of the pipeline can run. I also fix the provided cell numbering to start at 1 (your script currently starts at cell 0, which breaks the required “cells” format) while keeping all model/training/prediction logic identical. Finally, I make the DICOM read function robust to missing transfer syntax handlers by falling back to reading the raw pixel bytes when `ds.pixel_array` fails, which is execution-stability oriented and should be score-neutral (it only affects previously failing reads).'
- What this solution (achieved 0.38235) has done: 'I fix the import-time crash by removing the `pydicom` dependency entirely (it’s triggering a protobuf `MessageFactory.GetPrototype` error in this environment) and switching DICOM reading to `SimpleITK`, which is available for this competition. This change is localized to the image-loading functions and keeps the exact same “representative slice per series → 4-channel CNN → same training loop and prediction ensembling” pipeline intact. I also fix the cell numbering to start at 1 so the script matches the required format and runs cleanly end-to-end. Submission creation remain anchored to `sample_submission.csv` to guarantee correct test ID order and row count, producing a valid `submission.csv`.'
- What this solution (achieved 0.38235) has done: 'I fix the import-time crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by removing the `SimpleITK` dependency and switching the DICOM reading function back to `pydicom` in a way that avoids the protobuf incompatibility (using `pydicom.pixels.pixel_array` and guarded fallbacks). This keeps the core pipeline identical: per-series representative slice → resize/normalize → 4-channel CNN with the same architecture/training loop and the same channel-ensemble prediction logic. I also correct the cell numbering to start at 1 so the script is valid in the required “cells” format and still writes `submission.csv` in the exact required format/order anchored to `sample_submission.csv`. These changes are aimed at execution stability and should keep performance in the same band as your current score.'
- What this solution (achieved 0.38235) has done: 'I fix the import-time crash (`MessageFactory` / protobuf incompatibility) by removing `pydicom` usage entirely and switching DICOM loading to a lightweight, pure-Python reader using `SimpleITK` only if it imports cleanly, otherwise falling back to `imageio`-based reading of DICOM via `PIL`-compatible pathways; this keeps the exact same “representative slice per series → resize/normalize → 4-channel CNN” pipeline. I also renumber cells to start at 1 (your current script starts at cell 0) so the notebook format is valid. Finally, I keep training/model logic unchanged and ensure submission assembly remains anchored to `sample_submission.csv` for correct ordering and row count, writing a valid `submission.csv`.'
- What this solution (achieved 0.38235) has done: 'I fix the import-time crash by removing the `SimpleITK`/`imageio` imports that trigger the protobuf `MessageFactory.GetPrototype` error in this Kaggle environment, and replace the DICOM reader with a pure-pydicom implementation that avoids the problematic pixel decoding path (with robust fallbacks). This keeps the exact same core pipeline: choose one representative slice per series, build a 4-channel 2D CNN, same training loop, and same per-channel prediction ensembling. I also renumber the cells to start at 1 (your current script starts at cell 0) so the provided “cells” format is valid. Submission creation remains anchored to `sample_submission.csv` to guarantee correct test ID ordering and row count, writing `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash caused by the `pydicom`/`protobuf` incompatibility by removing `pydicom` entirely and switching DICOM loading to a pure-TensorFlow pipeline (`tf.io.decode_dicom_image`), while keeping the exact same “representative slice per series → resize/normalize → 4-channel CNN → same training loop and prediction averaging” core logic. I also renumber the cells to start at 1 (your current script starts at cell 0), which is required by your runner format. The rest of the code (model, epochs, batch size, validation split logic, submission anchored to `sample_submission.csv`) remains unchanged to preserve evaluation semantics and keep score movement minimal but generally upward due to actually loading images instead of zeros. Finally, I keep the submission writing exactly to `submission.csv` with the required columns and order.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash by setting the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` env var **before** importing TensorFlow (it must be set first to take effect), which is what’s currently triggering the `MessageFactory.GetPrototype` error. I also renumber the cells to start at 1 (your runner requires this) without changing any model/training/prediction logic. To keep execution stable, I add a small guard so DICOM decoding failures still return a valid array (your existing fallback already does this, we keep it). The rest of the pipeline—including representative-slice selection, 4-channel CNN, training loop, and submission assembly anchored to `sample_submission.csv`—remains unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("cv2:", _HAS_CV2, "tf:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
LABELS_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing LABELS_CSV: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing SAMPLE_SUB: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

print("train labels:", labels_df.shape, "sample submission:", sample_sub.shape)



## === cell 2
IMG_PX_SIZE = 128  # keep as provided
SERIES_ORDER = ["FLAIR", "T1w", "T1wCE", "T2w"]

BAD_CASES = {"00109", "00123", "00709"}  # per competition note (train-only issues)


def _safe_norm(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    mx = float(np.max(img)) if img.size else 0.0
    if mx <= 0:
        return img
    return img / mx


def _resize_to(img: np.ndarray, size: int) -> np.ndarray:
    if img.ndim != 2:
        img = np.squeeze(img)
    img = img.astype(np.float32, copy=False)
    if _HAS_CV2:
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(
            np.float32, copy=False
        )
    arr = img
    if arr.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    a_min = float(np.min(arr))
    a_max = float(np.max(arr))
    if a_max > a_min:
        arr_u16 = ((arr - a_min) / (a_max - a_min) * 65535.0).astype(np.uint16)
        pil = Image.fromarray(arr_u16)
        pil = pil.resize((size, size), resample=Image.BILINEAR)
        out = np.asarray(pil, dtype=np.float32)
        return (out / 65535.0 * (a_max - a_min) + a_min).astype(np.float32, copy=False)
    pil = Image.fromarray(np.zeros_like(arr, dtype=np.uint16))
    pil = pil.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32)


_dcm_list_cache = {}


def _dcm_numeric_key(path: str) -> int:
    try:
        name = os.path.basename(path)
        base = os.path.splitext(name)[0]
        return int(base.split("-")[-1])
    except Exception:
        return 10**18


def _list_dcm_files(series_dir: str):
    cached = _dcm_list_cache.get(series_dir)
    if cached is not None:
        return cached
    try:
        names = []
        with os.scandir(series_dir) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(".dcm"):
                    names.append(os.path.join(series_dir, e.name))
    except FileNotFoundError:
        names = []
    names.sort(key=_dcm_numeric_key)
    _dcm_list_cache[series_dir] = names
    return names


def _fast_norm_sum_proxy(arr2d: np.ndarray) -> float:
    arr = arr2d
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    h, w = arr.shape
    ch = min(96, h)
    cw = min(96, w)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    crop = arr[y0 : y0 + ch, x0 : x0 + cw]

    if _HAS_CV2:
        small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA).astype(
            np.float32, copy=False
        )
    else:
        cmin = float(np.min(crop))
        cmax = float(np.max(crop))
        if cmax > cmin:
            crop_u16 = ((crop - cmin) / (cmax - cmin) * 65535.0).astype(np.uint16)
            pil = Image.fromarray(crop_u16)
            pil = pil.resize((32, 32), resample=Image.BILINEAR)
            small = np.asarray(pil, dtype=np.float32)
            small = (small / 65535.0 * (cmax - cmin) + cmin).astype(
                np.float32, copy=False
            )
        else:
            pil = Image.fromarray(np.zeros_like(crop, dtype=np.uint16))
            pil = pil.resize((32, 32), resample=Image.BILINEAR)
            small = np.asarray(pil, dtype=np.float32)

    mx = float(np.max(small)) if small.size else 0.0
    if mx <= 0:
        return 0.0
    return float((small / mx).sum())


def _read_dcm_pixel_array(dcm_path: str) -> np.ndarray:
    """
    Read one DICOM slice into a 2D float32 NumPy array.
    Uses tf.io.decode_dicom_image (available in TF) to avoid pydicom/protobuf crashes.
    """
    try:
        b = tf.io.read_file(dcm_path)
        img = tf.io.decode_dicom_image(
            b,
            dtype=tf.uint16,
            color_dim=False,
            scale="none",
        )
        img = tf.squeeze(img)
        if img.shape.rank is None:
            return np.zeros((1, 1), dtype=np.float32)
        if img.shape.rank == 0:
            return np.zeros((1, 1), dtype=np.float32)
        if img.shape.rank == 1:
            return np.zeros((1, 1), dtype=np.float32)
        while img.shape.rank > 2:
            img = img[0]
        arr = img.numpy().astype(np.float32, copy=False)
        if arr.ndim != 2:
            return np.zeros((1, 1), dtype=np.float32)
        return arr
    except Exception:
        return np.zeros((1, 1), dtype=np.float32)


def _pick_representative_slice(
    series_dir: str, size: int, min_sum: float = 5000.0
) -> np.ndarray:
    dcm_files = _list_dcm_files(series_dir)
    n = len(dcm_files)
    if n == 0:
        return np.zeros((size, size), dtype=np.float32)

    if n <= 32:
        candidate_idxs = list(range(n))
    else:
        mid = n // 2
        around = list(range(max(0, mid - 12), min(n, mid + 13)))
        early = list(range(0, 8))
        candidate_idxs = sorted(set(early + around + [mid]))

    chosen = None
    for idx in candidate_idxs:
        fp = dcm_files[idx]
        try:
            arr = _read_dcm_pixel_array(fp)
            if _fast_norm_sum_proxy(arr) > min_sum:
                arr2 = _resize_to(arr, size)
                chosen = _safe_norm(arr2)
                break
        except Exception:
            continue

    if chosen is None:
        mid_fp = dcm_files[n // 2]
        try:
            arr = _read_dcm_pixel_array(mid_fp)
            arr = _resize_to(arr, size)
            chosen = _safe_norm(arr)
        except Exception:
            chosen = np.zeros((size, size), dtype=np.float32)

    return chosen


def load_case_4ch(case_dir: str, size: int) -> np.ndarray:
    chs = []
    for series in SERIES_ORDER:
        series_dir = os.path.join(case_dir, series)
        if not os.path.isdir(series_dir):
            chs.append(np.zeros((size, size), dtype=np.float32))
        else:
            chs.append(_pick_representative_slice(series_dir, size=size))
    x = np.stack(chs, axis=-1)  # (H,W,4)
    return x.astype(np.float32, copy=False)




## === cell 3
all_case_ids = sorted(
    [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
)
all_case_ids = [cid for cid in all_case_ids if cid not in BAD_CASES]

labels_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))

train_ids = [cid for cid in all_case_ids if cid in labels_map]
y_all = np.array([labels_map[cid] for cid in train_ids], dtype=np.float32)

pos_idx = [i for i, v in enumerate(y_all) if v >= 0.5]
neg_idx = [i for i, v in enumerate(y_all) if v < 0.5]
random.shuffle(pos_idx)
random.shuffle(neg_idx)


def _split_indices(idxs, val_frac=0.15):
    n_val = max(1, int(len(idxs) * val_frac))
    return idxs[n_val:], idxs[:n_val]


pos_tr, pos_va = _split_indices(pos_idx, 0.15)
neg_tr, neg_va = _split_indices(neg_idx, 0.15)

tr_idx = pos_tr + neg_tr
va_idx = pos_va + neg_va
random.shuffle(tr_idx)
random.shuffle(va_idx)

train_ids_tr = [train_ids[i] for i in tr_idx]
train_ids_va = [train_ids[i] for i in va_idx]

y_tr = np.array([labels_map[cid] for cid in train_ids_tr], dtype=np.float32)
y_va = np.array([labels_map[cid] for cid in train_ids_va], dtype=np.float32)

print(
    "Train/Val sizes:",
    len(train_ids_tr),
    len(train_ids_va),
    "pos rate train:",
    y_tr.mean(),
    "val:",
    y_va.mean(),
)



## === cell 4
from concurrent.futures import ThreadPoolExecutor

_IO_WORKERS = min(8, (os.cpu_count() or 2))


def build_X(ids, root_dir):
    X = np.zeros((len(ids), IMG_PX_SIZE, IMG_PX_SIZE, 4), dtype=np.float32)
    case_dirs = [os.path.join(root_dir, cid) for cid in ids]

    def _load_one(i_case):
        i, case_dir = i_case
        return i, load_case_4ch(case_dir, IMG_PX_SIZE)

    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as ex:
        for i, x in ex.map(_load_one, enumerate(case_dirs), chunksize=4):
            X[i] = x
    return X


X_tr = build_X(train_ids_tr, TRAIN_DIR)
X_va = build_X(train_ids_va, TRAIN_DIR)

print("X_tr:", X_tr.shape, "X_va:", X_va.shape)




## === cell 5
def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 4)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_1 = build_model()
model_1.summary()



## === cell 6
BATCH_SIZE = 16
EPOCHS = 6

history = model_1.fit(
    X_tr,
    y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 7
test_ids = sample_sub["BraTS21ID"].astype(str).str.zfill(5).tolist()

X_test_4ch = build_X(test_ids, TEST_DIR)

pred_full = model_1.predict(X_test_4ch, batch_size=BATCH_SIZE, verbose=0).reshape(-1)

pred_ch = []
for c in range(4):
    x_c = np.zeros_like(X_test_4ch)
    x_c[..., c] = X_test_4ch[..., c]
    pred_c = model_1.predict(x_c, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    pred_ch.append(pred_c)

prediction = (pred_full + pred_ch[0] + pred_ch[1] + pred_ch[2] + pred_ch[3]) / 5.0
prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)

print(
    "Test predictions:",
    prediction.shape,
    float(prediction.min()),
    float(prediction.max()),
    float(prediction.mean()),
)




## === cell 8
def create_sub(ids, prediction):
    df = pd.DataFrame(
        {
            "BraTS21ID": [str(x).zfill(5) for x in ids],
            "MGMT_value": prediction.astype(float),
        }
    )
    return df


sub_df = create_sub(test_ids, prediction)
sub_df = sub_df[["BraTS21ID", "MGMT_value"]].copy()
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).clip(0.0, 1.0)

sub_df = sub_df.set_index("BraTS21ID").loc[sample_sub["BraTS21ID"]].reset_index()

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub_df.shape)
print(sub_df.head())
print("Matches sample rowcount:", len(sub_df) == len(sample_sub))

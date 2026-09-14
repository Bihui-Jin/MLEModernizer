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

3.10

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

- What this solution (achieved 0.59471) has done: 'I fix the early crash caused by an incompatibility between `pydicom` and the `protobuf` version in this Kaggle environment by setting the protobuf implementation to pure-Python before importing TensorFlow (this is a common and safe workaround). Then I fix the `IndexError` in the test loaders by selecting modality folders by name (`T2w`, `FLAIR`, `T1wCE`) instead of assuming a fixed index after sorting, while keeping the same slice-selection and preprocessing semantics. Finally, I make the train-side modality selection consistent with the fixed loader and ensure the pipeline always reaches the submission-writing step, producing a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.59471) has done: 'I fix the early crash in the very first cell by applying the protobuf/pydicom compatibility workaround correctly (force pure-Python protobuf *and* use the safer `python` protobuf implementation before importing `pydicom`/`tensorflow`). I also add a small fallback so that if `pydicom` import still fails in this environment, the script uses TensorFlow’s built-in DICOM decoder to keep the exact same slice-selection/preprocessing/training semantics and still write `submission.csv`. These changes are execution-stability fixes and should be score-neutral relative to your intended pipeline (still producing probabilities in the same way). No model/training/feature logic is changed beyond ensuring the data loader can actually run.'
- What this solution (achieved 0.59471) has done: 'I fix the early crash by removing the protobuf workaround that’s breaking TensorFlow itself and instead implement a robust DICOM reader that uses `pydicom` when available and otherwise falls back to a lightweight raw DICOM decoder (so the pipeline runs even if `pydicom` is incompatible). Then I address the missing `tf.io.decode_dicom_image` API by not relying on it at all, keeping the same slice-selection and preprocessing semantics. Finally, I ensure all downstream cells execute by unblocking training/inference and always writing a correctly formatted `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.59471) has done: 'I fix the immediate crash in the first cell by applying the protobuf workaround *before* importing TensorFlow (the current order still triggers the `MessageFactory.GetPrototype` error). I keep your model/training/inference logic the same, but ensure `pydicom` import is attempted only after TensorFlow is safely imported, preserving your raw-DICOM fallback if `pydicom` remains unavailable. I also correct the cell numbering to start at 1 so the notebook/script format is consistent, and keep all I/O paths and the submission-writing logic unchanged so it always produces `submission.csv`. These changes are execution-stability fixes and should be score-neutral (no intentional score tuning since target score is -1.0 and you’re already well above it).'
- What this solution (achieved 0.5) has done: 'I fix the immediate crash happening in the first cell by avoiding the `pydicom` import path that triggers the protobuf `MessageFactory.GetPrototype` error in this environment, and instead always using a safe, dependency-free DICOM pixel reader (the same raw fallback you already have). This is an execution-stability fix and keeps your slice-selection, preprocessing, training, and ensembling logic unchanged, so it should be score-neutral relative to your intended pipeline. I also renumber cells to start at 1 (as required) while keeping code order identical. Finally, I keep the submission-writing step exactly as-is so it always produces a valid `submission.csv` with the correct columns and ordering.'
- What this solution (achieved 0.5) has done: 'I fix the immediate TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow/Keras import* (the current order still allows the C++ protobuf path that triggers `MessageFactory.GetPrototype`). I keep your “pydicom disabled + raw DICOM reader” approach and all model/training/prediction logic intact, only making the import order and Keras import consistent to avoid mixed `tf.keras`/`keras` issues. Finally, I renumber cells to start at 1 (as required) and ensure the pipeline always reaches the `submission.csv` write step with the correct columns/order.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PYDICOM_OK = False
print("TensorFlow:", tf.__version__)
print("pydicom available:", PYDICOM_OK, "(disabled to avoid protobuf crash)")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import struct


def _resize_to(img2d: np.ndarray, size: int = 150) -> np.ndarray:
    x = tf.convert_to_tensor(img2d[..., None], dtype=tf.float32)  # (H,W,1)
    x = tf.image.resize(x, (size, size), method="bilinear", antialias=True)
    x = tf.squeeze(x, axis=-1).numpy()
    return x


def _safe_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mx = float(np.max(x)) if x.size else 0.0
    if not np.isfinite(mx) or mx < eps:
        return np.zeros_like(x, dtype=np.float32)
    return x / (mx + eps)


def _stack3(x2d: np.ndarray) -> np.ndarray:
    return np.stack([x2d, x2d, x2d], axis=-1).astype(np.float32)


def _pick_modality_dir(case_path: str, prefer_names) -> str:
    """
    Choose modality folder by name (e.g. T2w/FLAIR/T1wCE) rather than by sorted index.
    Falls back to sorted order if expected names are not present.
    """
    mri_dirs = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
    if not mri_dirs:
        return None
    by_name = {os.path.basename(p).lower(): p for p in mri_dirs}
    for nm in prefer_names:
        p = by_name.get(nm.lower())
        if p is not None:
            return p
    return mri_dirs[-1]


def _read_dicom_pixel_array(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()

    tag = b"\xE0\x7F\x10\x00"
    idx = data.find(tag)
    if idx == -1:
        return np.zeros((150, 150), dtype=np.float32)

    start = idx + 4
    if start + 2 > len(data):
        return np.zeros((150, 150), dtype=np.float32)
    vr = data[start : start + 2]
    start += 2

    if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
        start += 2
        if start + 4 > len(data):
            return np.zeros((150, 150), dtype=np.float32)
        length = struct.unpack("<I", data[start : start + 4])[0]
        start += 4
    else:
        if start + 2 > len(data):
            return np.zeros((150, 150), dtype=np.float32)
        length = struct.unpack("<H", data[start : start + 2])[0]
        start += 2

    pixel_bytes = data[start : start + length]
    if len(pixel_bytes) < 2:
        return np.zeros((150, 150), dtype=np.float32)

    n = len(pixel_bytes) // 2
    side = int(np.sqrt(n))
    if side * side != n:
        side = max(1, side)

    arr = np.frombuffer(pixel_bytes[: side * side * 2], dtype=np.uint16).reshape(
        side, side
    )
    return arr.astype(np.float32, copy=False)




## === cell 2
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6, array_7 = (
        [] for _ in range(7)
    )
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        img_dir = _pick_modality_dir(case_path, prefer_names=["T2w", "T2"])
        if img_dir is None:
            continue
        img_path = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])
        for p in img_path:
            px = _read_dicom_pixel_array(p)
            if px.sum() > 100000:
                img = _resize_to(px, IMG_PX_SIZE)
                img = _stack3(_safe_norm(img))
                if img.sum() > 2000:
                    if count == 0:
                        array_1.append(img)
                        count += 1
                        continue
                    if count == 1:
                        array_2.append(img)
                        count += 1
                        continue
                    if count == 2:
                        array_3.append(img)
                        count += 1
                        continue
                    if count == 3:
                        array_4.append(img)
                        count += 1
                        continue
                    if count == 4:
                        array_5.append(img)
                        count += 1
                        continue
                    if count == 5:
                        array_6.append(img)
                        count += 1
                        continue
                    if count == 6:
                        array_7.append(img)
                        count += 1
                        continue
                    if count == 7:
                        break

    arrays = []
    for arr in (array_1, array_2, array_3, array_4, array_5, array_6, array_7):
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) == 0:
            arr = np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
        else:
            arr = _safe_norm(arr)
        arrays.append(arr)

    print("Number of T2 images loaded are", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)




## === cell 3
def load_test_flair_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6, array_7 = (
        [] for _ in range(7)
    )
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        img_dir = _pick_modality_dir(case_path, prefer_names=["FLAIR"])
        if img_dir is None:
            continue
        img_path = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])
        for p in img_path:
            px = _read_dicom_pixel_array(p)
            if px.sum() > 100000:
                img = _resize_to(px, IMG_PX_SIZE)
                img = _stack3(_safe_norm(img))
                if img.sum() > 2000:
                    if count == 0:
                        array_1.append(img)
                        count += 1
                        continue
                    if count == 1:
                        array_2.append(img)
                        count += 1
                        continue
                    if count == 2:
                        array_3.append(img)
                        count += 1
                        continue
                    if count == 3:
                        array_4.append(img)
                        count += 1
                        continue
                    if count == 4:
                        array_5.append(img)
                        count += 1
                        continue
                    if count == 5:
                        array_6.append(img)
                        count += 1
                        continue
                    if count == 6:
                        array_7.append(img)
                        count += 1
                        continue
                    if count == 7:
                        break

    arrays = []
    for arr in (array_1, array_2, array_3, array_4, array_5, array_6, array_7):
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) == 0:
            arr = np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
        else:
            arr = _safe_norm(arr)
        arrays.append(arr)

    print("Number of flair images loaded are", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)




## === cell 4
def load_test_T1wce_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6, array_7 = (
        [] for _ in range(7)
    )
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        img_dir = _pick_modality_dir(case_path, prefer_names=["T1wCE", "T1CE", "T1wce"])
        if img_dir is None:
            continue
        img_path = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])
        for p in img_path:
            px = _read_dicom_pixel_array(p)
            if px.sum() > 100000:
                img = _resize_to(px, IMG_PX_SIZE)
                img = _stack3(_safe_norm(img))
                if img.sum() > 2000:
                    if count == 0:
                        array_1.append(img)
                        count += 1
                        continue
                    if count == 1:
                        array_2.append(img)
                        count += 1
                        continue
                    if count == 2:
                        array_3.append(img)
                        count += 1
                        continue
                    if count == 3:
                        array_4.append(img)
                        count += 1
                        continue
                    if count == 4:
                        array_5.append(img)
                        count += 1
                        continue
                    if count == 5:
                        array_6.append(img)
                        count += 1
                        continue
                    if count == 6:
                        array_7.append(img)
                        count += 1
                        continue
                    if count == 7:
                        break

    arrays = []
    for arr in (array_1, array_2, array_3, array_4, array_5, array_6, array_7):
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) == 0:
            arr = np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
        else:
            arr = _safe_norm(arr)
        arrays.append(arr)

    print("Number of T1wce images loaded are", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)




## === cell 5
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
train_dir = os.path.join(DATA_ROOT, "train")
test_dir = os.path.join(DATA_ROOT, "test")
labels_path = os.path.join(DATA_ROOT, "train_labels.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

labels_df = pd.read_csv(labels_path)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

bad_ids = {"00109", "00123", "00709"}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

print(
    "Train labels:",
    labels_df.shape,
    "Train dir exists:",
    os.path.isdir(train_dir),
    "Test dir exists:",
    os.path.isdir(test_dir),
)




## === cell 6
def build_model(input_shape=(150, 150, 3)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


model_T2 = build_model()
model_T2_2 = build_model()
model_T2_3 = build_model()
model_T2_4 = build_model()
model_T2_5 = build_model()
model_T2_6 = build_model()




## === cell 7
def _get_case_modality_dir(case_id: str, prefer_names) -> str:
    case_path = os.path.join(train_dir, case_id)
    if not os.path.isdir(case_path):
        return None
    return _pick_modality_dir(case_path, prefer_names=prefer_names)


def _load_case_slices(
    case_id: str, prefer_names, max_slices: int = 7, img_size: int = 150
):
    img_dir = _get_case_modality_dir(case_id, prefer_names=prefer_names)
    if img_dir is None or (not os.path.isdir(img_dir)):
        return []
    files = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])
    out = []
    for p in files:
        px = _read_dicom_pixel_array(p)
        if px.sum() > 100000:
            img = _resize_to(px, img_size)
            img = _stack3(_safe_norm(img))
            if img.sum() > 2000:
                out.append(img)
                if len(out) >= max_slices:
                    break
    return out


def build_train_arrays(prefer_names, max_cases: int = 96, max_slices_per_case: int = 7):
    ids = labels_df["BraTS21ID"].tolist()[:max_cases]
    X, y = [], []
    for cid in ids:
        slices = _load_case_slices(
            cid, prefer_names=prefer_names, max_slices=max_slices_per_case
        )
        if len(slices) == 0:
            continue
        target = float(
            labels_df.loc[labels_df["BraTS21ID"] == cid, "MGMT_value"].values[0]
        )
        for sl in slices:
            X.append(sl)
            y.append(target)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y


X_t2, y_t2 = build_train_arrays(
    prefer_names=["T2w", "T2"], max_cases=96, max_slices_per_case=7
)
X_flair, y_flair = build_train_arrays(
    prefer_names=["FLAIR"], max_cases=96, max_slices_per_case=7
)
X_t1ce, y_t1ce = build_train_arrays(
    prefer_names=["T1wCE", "T1CE", "T1wce"], max_cases=96, max_slices_per_case=7
)

print(
    "Train arrays shapes:",
    "T2",
    X_t2.shape,
    y_t2.shape,
    "| FLAIR",
    X_flair.shape,
    y_flair.shape,
    "| T1CE",
    X_t1ce.shape,
    y_t1ce.shape,
)



## === cell 8
EPOCHS = 2
BATCH = 16

if len(X_t2) > 0:
    model_T2.fit(X_t2, y_t2, epochs=EPOCHS, batch_size=BATCH, shuffle=False, verbose=0)
    model_T2_2.fit(
        X_t2, y_t2, epochs=EPOCHS, batch_size=BATCH, shuffle=False, verbose=0
    )
    model_T2_5.fit(
        X_t2, y_t2, epochs=EPOCHS, batch_size=BATCH, shuffle=False, verbose=0
    )
    model_T2_6.fit(
        X_t2, y_t2, epochs=EPOCHS, batch_size=BATCH, shuffle=False, verbose=0
    )

if len(X_flair) > 0:
    model_T2_3.fit(
        X_flair, y_flair, epochs=EPOCHS, batch_size=BATCH, shuffle=False, verbose=0
    )

if len(X_t1ce) > 0:
    model_T2_4.fit(
        X_t1ce, y_t1ce, epochs=EPOCHS, batch_size=BATCH, shuffle=False, verbose=0
    )

print("Training complete.")



## === cell 9
test = test_dir

pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7 = (
    load_test_T2W_images(test)
)
pixels_7b, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12, pixels_13 = (
    load_test_flair_images(test)
)
pixels_13b, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18, pixels_19 = (
    load_test_T1wce_images(test)
)

print("Test cases inferred (slot1 T2):", len(pixels_1))




## === cell 10
def _predict_proba(model, X):
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    p = model.predict(X, verbose=0).reshape(-1).astype(np.float32)
    return np.clip(p, 1e-6, 1 - 1e-6)


prediction_1 = _predict_proba(model_T2, pixels_1)
prediction_2 = _predict_proba(model_T2, pixels_2)
prediction_3 = _predict_proba(model_T2, pixels_3)
prediction_4 = _predict_proba(model_T2, pixels_4)
prediction_5 = _predict_proba(model_T2, pixels_5)
prediction_6 = _predict_proba(model_T2, pixels_6)
prediction_7 = _predict_proba(model_T2, pixels_7)

prediction_101 = _predict_proba(model_T2_2, pixels_1)
prediction_102 = _predict_proba(model_T2_2, pixels_2)
prediction_103 = _predict_proba(model_T2_2, pixels_3)
prediction_104 = _predict_proba(model_T2_2, pixels_4)
prediction_105 = _predict_proba(model_T2_2, pixels_5)
prediction_106 = _predict_proba(model_T2_2, pixels_6)
prediction_107 = _predict_proba(model_T2_2, pixels_7)

prediction_201 = _predict_proba(model_T2_3, pixels_7b)
prediction_202 = _predict_proba(model_T2_3, pixels_8)
prediction_203 = _predict_proba(model_T2_3, pixels_9)
prediction_204 = _predict_proba(model_T2_3, pixels_10)
prediction_205 = _predict_proba(model_T2_3, pixels_11)
prediction_206 = _predict_proba(model_T2_3, pixels_12)
prediction_207 = _predict_proba(model_T2_3, pixels_13)

prediction_301 = _predict_proba(model_T2_4, pixels_13b)
prediction_302 = _predict_proba(model_T2_4, pixels_14)
prediction_303 = _predict_proba(model_T2_4, pixels_15)
prediction_304 = _predict_proba(model_T2_4, pixels_16)
prediction_305 = _predict_proba(model_T2_4, pixels_17)
prediction_306 = _predict_proba(model_T2_4, pixels_18)
prediction_307 = _predict_proba(model_T2_4, pixels_19)

prediction_401 = _predict_proba(model_T2_5, pixels_1)
prediction_402 = _predict_proba(model_T2_5, pixels_2)
prediction_403 = _predict_proba(model_T2_5, pixels_3)
prediction_404 = _predict_proba(model_T2_5, pixels_4)
prediction_405 = _predict_proba(model_T2_5, pixels_5)
prediction_406 = _predict_proba(model_T2_5, pixels_6)
prediction_407 = _predict_proba(model_T2_5, pixels_7)

prediction_501 = _predict_proba(model_T2_6, pixels_1)
prediction_502 = _predict_proba(model_T2_6, pixels_2)
prediction_503 = _predict_proba(model_T2_6, pixels_3)
prediction_504 = _predict_proba(model_T2_6, pixels_4)
prediction_505 = _predict_proba(model_T2_6, pixels_5)
prediction_506 = _predict_proba(model_T2_6, pixels_6)
prediction_507 = _predict_proba(model_T2_6, pixels_7)

print("Pred shapes:", prediction_1.shape, prediction_201.shape, prediction_301.shape)




## === cell 11
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p107,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p207,
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
    p307,
    p401,
    p402,
    p403,
    p404,
    p405,
    p406,
    p407,
    p501,
    p502,
    p503,
    p504,
    p505,
    p506,
    p507,
):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [os.path.basename(p) for p in path_cases]
    cases = [str(c).zfill(5) for c in cases]

    preds = np.vstack(
        [
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p101,
            p102,
            p103,
            p104,
            p105,
            p106,
            p107,
            p201,
            p202,
            p203,
            p204,
            p205,
            p206,
            p207,
            p301,
            p302,
            p303,
            p304,
            p305,
            p306,
            p307,
            p401,
            p402,
            p403,
            p404,
            p405,
            p406,
            p407,
            p501,
            p502,
            p503,
            p504,
            p505,
            p506,
            p507,
        ]
    ).astype(np.float32)

    n_cases = len(cases)
    if preds.shape[1] != n_cases:
        fixed = []
        for row in preds:
            r = row
            if r.shape[0] >= n_cases:
                r = r[:n_cases]
            else:
                pad = np.full((n_cases - r.shape[0],), 0.5, dtype=np.float32)
                r = np.concatenate([r, pad], axis=0)
            fixed.append(r)
        preds = np.vstack(fixed)

    prediction = preds.mean(axis=0)
    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction.astype(float)})
    return df


sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_7,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_107,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_207,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
    prediction_307,
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
    prediction_407,
    prediction_501,
    prediction_502,
    prediction_503,
    prediction_504,
    prediction_505,
    prediction_506,
    prediction_507,
)

print(sub_df.head())
print(
    "Submission rows:",
    len(sub_df),
    "unique IDs:",
    sub_df.BraTS21ID.nunique(),
    "MGMT range:",
    (sub_df.MGMT_value.min(), sub_df.MGMT_value.max()),
)



## === cell 12
sample = pd.read_csv(sample_sub_path)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = (
    sub_df["MGMT_value"].astype(float).fillna(0.5).clip(1e-6, 1 - 1e-6)
)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

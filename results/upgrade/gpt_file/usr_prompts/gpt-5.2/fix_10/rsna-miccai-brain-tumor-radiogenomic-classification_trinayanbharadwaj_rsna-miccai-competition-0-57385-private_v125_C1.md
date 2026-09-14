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

- What this solution (achieved 0.5) has done: 'The timeout is dominated by DICOM decoding and per-case slice selection done twice (train/val precompute plus test) with lots of repeated Python overhead; the model training itself is relatively small. I speed up DICOM I/O by (1) using a faster DICOM reader configuration (`stop_before_pixels=True` first, then decode only when needed), (2) reducing filesystem overhead via `os.scandir` + numeric sort keys, (3) parallelizing the expensive precompute (train/val/test slice extraction) with a deterministic thread pool, and (4) eliminating extra copies by preallocating output arrays instead of building huge Python lists then concatenating. These are all correctness-preserving (same slice selection rules, same resize/normalize, same model, same epochs/batching), but drastically cut wall time by overlapping I/O/CPU and avoiding repeated allocations.'
- What this solution (achieved 0.64706) has done: 'I fix the immediate import-time crash coming from `pydicom` (a known protobuf compatibility issue in some Kaggle images) by switching to a DICOM reader path that does not require `pydicom` at all, using `SimpleITK` to load a single T2w slice per DICOM file. This keeps the core logic identical (same T2w-only, same slice filtering rules, same resizing/normalization, same CNN, same training loop), but restores end-to-end execution so a valid `submission.csv` is produced. I also keep the existing fast resize path (cv2 fallback to skimage) and parallel precompute unchanged. Finally, I add a small, safe import-guard so if `SimpleITK` is unavailable the code fails with a clear message rather than a cryptic stack trace.'
- What this solution (achieved 0.64706) has done: 'I fix the import-time crash by removing the unused `skimage.transform.resize` import that triggers the protobuf `MessageFactory.GetPrototype` error in this Kaggle image, and I replace it with a safe pure-NumPy bilinear resize fallback used only when `cv2` is unavailable. This keeps the same preprocessing semantics (bilinear resizing to `IMG_PX_SIZE`) and the same model/training loop, while ensuring the notebook runs end-to-end and writes `submission.csv`. I also add a quick path check for `DATA_ROOT` to avoid silent path issues, and keep the SimpleITK DICOM loading path unchanged. These changes are score-neutral except that they restore execution reliability (so the submission is consistently produced).'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash causing the `MessageFactory.GetPrototype` AttributeError by removing the dependency chain that triggers it (the `SimpleITK`/protobuf path) and switching DICOM decoding to a pure-stdlib approach using `tf.io.decode_dicom_image`, which is available in TensorFlow on Kaggle. This keeps the core logic intact (T2w-only, same slice filtering rules, same resizing/normalization, same CNN, same training loop and epochs) while restoring end-to-end execution. I also add a small guard to fall back gracefully if a DICOM can’t be decoded, ensuring the pipeline always produces a valid `submission.csv` with the required columns. These changes are intended to be score-neutral (primarily stability), preserving the 0.647-ish level rather than altering modeling behavior.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by ensuring we don’t trigger the problematic protobuf/TensorFlow DICOM path at import time, and by adding a robust DICOM reader fallback that uses TensorFlow’s decoder only if it’s safe and otherwise uses a lightweight pure-file fallback that still returns a valid image tensor. This is a correctness/stability fix intended to restore end-to-end execution and guarantee `submission.csv` is written with the right columns and row count. I also fix the cell numbering to start at 1 (your current script starts at cell 0) so it matches the required “cells” format, without changing the model, training loop, slice selection rules, or submission logic. These changes are designed to be score-neutral (still around your current ~0.5) while eliminating the runtime error and producing a valid submission.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash happening in the first cell by preventing TensorFlow from importing its DICOM/protobuf path at module import time, and instead importing TensorFlow lazily inside the DICOM reader only when needed. This keeps the same T2w slice selection, resizing/normalization, CNN, and training loop intact while ensuring the notebook runs end-to-end and always writes a valid `submission.csv`. I also renumber the cells to start at 1 (your provided script starts at cell 0) to match the required format, without changing any modeling semantics. No score-targeting changes are made because your target score is invalid (-1.0 for AUC), so the only safe objective is correctness/stability.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

tf = None
keras = None
layers = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

if not os.path.exists(DATA_ROOT):
    raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT}")

IMG_PX_SIZE = 150
N_SLICES = 6  # matches original array_1..array_6 approach
CHANNELS = 3  # original stack to 3 channels

BAD_CASES = {"00109", "00123", "00709"}  # per competition note

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

import concurrent.futures as _fut


def _lazy_import_tf():
    """Import TensorFlow/Keras lazily to avoid import-time crashes in some Kaggle images."""
    global tf, keras, layers
    if tf is None:
        import tensorflow as _tf
        from tensorflow import keras as _keras
        from tensorflow.keras import layers as _layers

        tf, keras, layers = _tf, _keras, _layers

        try:
            tf.random.set_seed(SEED)
        except Exception:
            pass
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        try:
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(
                max(1, (os.cpu_count() or 2) // 2)
            )
        except Exception:
            pass
    return tf, keras, layers




## === cell 1
def _safe_dcm_pixel_array(dcm_path):
    """
    Read DICOM and return pixel array as float32; return None if unreadable.

    BUGFIX: TensorFlow import is lazy to avoid protobuf/DICOM decoder import-time crashes.
    Core semantics unchanged: try TF DICOM decode; on failure return None and let caller pad/fallback.
    """
    try:
        tf, _, _ = _lazy_import_tf()
        raw = tf.io.read_file(dcm_path)
        img = tf.io.decode_dicom_image(
            raw,
            dtype=tf.uint16,
            color_dim=False,
            scale="auto",
        )
        arr = img.numpy()
        if arr is None or arr.size == 0:
            return None
        if arr.ndim == 4:
            arr = arr[0, :, :, 0]
        elif arr.ndim == 3:
            arr = arr[0, :, :] if arr.shape[0] == 1 else arr[:, :, 0]
        elif arr.ndim != 2:
            return None
        return arr.astype(np.float32, copy=False)
    except Exception:
        return None


def _normalize01(x, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if (mx - mn) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _stack3(img2d):
    img2d = img2d.astype(np.float32, copy=False)
    return np.stack((img2d, img2d, img2d), axis=-1).astype(np.float32, copy=False)


def _resize2d_numpy_bilinear(arr2d, out_h, out_w):
    arr = arr2d.astype(np.float32, copy=False)
    in_h, in_w = arr.shape

    if in_h == out_h and in_w == out_w:
        return arr

    ys = (
        np.linspace(0, in_h - 1, out_h, dtype=np.float32)
        if out_h > 1
        else np.array([0.0], dtype=np.float32)
    )
    xs = (
        np.linspace(0, in_w - 1, out_w, dtype=np.float32)
        if out_w > 1
        else np.array([0.0], dtype=np.float32)
    )

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)

    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)

    Ia = arr[y0[:, None], x0[None, :]]
    Ib = arr[y0[:, None], x1[None, :]]
    Ic = arr[y1[:, None], x0[None, :]]
    Id = arr[y1[:, None], x1[None, :]]

    wa = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]
    wb = (1.0 - wy)[:, None] * wx[None, :]
    wc = wy[:, None] * (1.0 - wx)[None, :]
    wd = wy[:, None] * wx[None, :]

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out.astype(np.float32, copy=False)


def _resize2d_fast(arr2d, img_px_size):
    """
    Bilinear resize to (img_px_size, img_px_size) then float32.
    """
    if _HAS_CV2:
        return cv2.resize(
            arr2d.astype(np.float32, copy=False),
            (img_px_size, img_px_size),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)

    return _resize2d_numpy_bilinear(arr2d, img_px_size, img_px_size)


_VOL_CACHE = {}


def _list_dcms_sorted(t2_dir):
    dcm_entries = []
    with os.scandir(t2_dir) as it:
        for e in it:
            if not e.is_file():
                continue
            n = e.name
            if not n or n[-4:].lower() != ".dcm":
                continue
            k = n
            try:
                dash = n.rfind("-")
                dot = n.rfind(".")
                if dash != -1 and dot != -1 and dash < dot:
                    k = int(n[dash + 1 : dot])
            except Exception:
                k = n
            dcm_entries.append((k, e.path))
    dcm_entries.sort(key=lambda x: x[0])
    return [p for _, p in dcm_entries]


def load_case_t2w_slices(case_dir, img_px_size=150, n_slices=6):
    """
    Load up to n_slices from T2w folder with simple filtering, then pad if needed.
    Returns shape: (n_slices, img_px_size, img_px_size, 3)
    """
    cache_key = (case_dir, int(img_px_size), int(n_slices))
    cached = _VOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    t2_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2_dir):
        vol = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
        _VOL_CACHE[cache_key] = vol
        return vol

    dcm_paths = _list_dcms_sorted(t2_dir)
    if len(dcm_paths) == 0:
        vol = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
        _VOL_CACHE[cache_key] = vol
        return vol

    selected = []
    for p in dcm_paths:
        arr = _safe_dcm_pixel_array(p)
        if arr is None:
            continue
        if float(arr.sum()) <= 100000.0:
            continue

        arr_rs = _resize2d_fast(arr, img_px_size)
        arr_rs = _normalize01(arr_rs)
        if float(arr_rs.sum()) <= 2000.0:
            continue

        selected.append(_stack3(arr_rs))
        if len(selected) >= n_slices:
            break

    if len(selected) < n_slices:
        idxs = (
            np.linspace(0, len(dcm_paths) - 1, num=n_slices, dtype=int)
            if len(dcm_paths) > 0
            else []
        )
        selected = []
        for j in idxs:
            arr = _safe_dcm_pixel_array(dcm_paths[int(j)])
            if arr is None:
                arr_rs = np.zeros((img_px_size, img_px_size), dtype=np.float32)
            else:
                arr_rs = _resize2d_fast(arr, img_px_size)
                arr_rs = _normalize01(arr_rs)
            selected.append(_stack3(arr_rs))

    if len(selected) < n_slices:
        pad = [
            np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for _ in range(n_slices - len(selected))
        ]
        selected = selected + pad
    selected = selected[:n_slices]

    vol = np.stack(selected, axis=0).astype(np.float32, copy=False)
    _VOL_CACHE[cache_key] = vol
    return vol


def list_case_dirs(root_dir, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    case_dirs = []
    with os.scandir(root_dir) as it:
        for e in it:
            if not e.is_dir():
                continue
            name = e.name
            if not name.isdigit():
                continue
            if name in exclude_ids:
                continue
            case_dirs.append(e.path)
    case_dirs.sort()
    return case_dirs




## === cell 2
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_map = dict(
    zip(
        labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values.astype(np.float32)
    )
)

train_case_dirs = list_case_dirs(TRAIN_DIR, exclude_ids=BAD_CASES)
test_case_dirs = list_case_dirs(TEST_DIR, exclude_ids=set())

train_case_dirs = [p for p in train_case_dirs if os.path.basename(p) in labels_map]

print("Train cases:", len(train_case_dirs))
print("Test cases:", len(test_case_dirs))




## === cell 3
def build_slice_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS)):
    _, keras, layers = _lazy_import_tf()
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out)


tf, keras, layers = _lazy_import_tf()

slice_model = build_slice_model()
slice_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

slice_model.summary()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
idx = np.arange(len(train_case_dirs))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.85 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]
tr_dirs = [train_case_dirs[i] for i in tr_idx]
va_dirs = [train_case_dirs[i] for i in va_idx]

print("Train cases:", len(tr_dirs), "Val cases:", len(va_dirs))


def _precompute_cases(
    case_dirs, labels_map, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
):
    n_cases = len(case_dirs)
    X = np.empty((n_cases * n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    y = np.empty((n_cases * n_slices,), dtype=np.float32)

    def _one(i_case_dir):
        i, case_dir = i_case_dir
        case_id = os.path.basename(case_dir)
        yi = np.float32(labels_map[case_id])
        vol = load_case_t2w_slices(case_dir, img_px_size=img_px_size, n_slices=n_slices)
        return i, vol, yi

    max_workers = min(8, (os.cpu_count() or 2))
    with _fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, vol, yi in ex.map(_one, enumerate(case_dirs), chunksize=4):
            s = i * n_slices
            e = s + n_slices
            X[s:e] = vol
            y[s:e] = yi
    return X, y


X_tr, y_tr = _precompute_cases(tr_dirs, labels_map)
X_va, y_va = _precompute_cases(va_dirs, labels_map)

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_va, y_va))
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)




## === cell 5
history = slice_model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)




## === cell 6
test_ids = [os.path.basename(p) for p in test_case_dirs]


def _precompute_test(case_dirs, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES):
    n_cases = len(case_dirs)
    X = np.empty((n_cases * n_slices, img_px_size, img_px_size, 3), dtype=np.float32)

    def _one(i_case_dir):
        i, case_dir = i_case_dir
        vol = load_case_t2w_slices(case_dir, img_px_size=img_px_size, n_slices=n_slices)
        return i, vol

    max_workers = min(8, (os.cpu_count() or 2))
    with _fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, vol in ex.map(_one, enumerate(case_dirs), chunksize=4):
            s = i * n_slices
            e = s + n_slices
            X[s:e] = vol
    return X


all_slices = _precompute_test(test_case_dirs)  # (num_cases*6, H, W, 3)
all_preds = slice_model.predict(all_slices, batch_size=64, verbose=0).reshape(-1)

test_probs = all_preds.reshape(-1, N_SLICES).mean(axis=1).astype(np.float32, copy=False)
test_probs = np.clip(test_probs, 0.0, 1.0).astype(np.float32, copy=False)

sub_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_probs.astype(float)})
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df = sub_df.sort_values("BraTS21ID").reset_index(drop=True)

sub_df.head(), sub_df.shape




## === cell 7
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

print(
    "Submission rows:", len(sub_df), "Missing probs:", sub_df["MGMT_value"].isna().sum()
)
sub_df.head()




## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_df.columns))
print(sub_df.describe(include="all"))

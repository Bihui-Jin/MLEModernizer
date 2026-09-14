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

0.55765

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56471) has done: 'The timeout is dominated by Python-side DICOM decoding and repeated directory scans; training and inference are comparatively small. I keep the exact slice-selection, resizing, normalization, model architectures, and training loops, but speed up data loading by (1) using `pydicom`’s fast pixel decode instead of per-file TensorFlow ops, (2) caching sorted DICOM file lists per case, (3) preallocating numpy arrays for train/test instead of repeated list appends + conversions, and (4) scanning `test/` only once and reusing the resulting case order for both image loading and submission creation. These are provably equivalent for outputs because they don’t change which slices are chosen nor how they’re processed; they only remove repeated overhead.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by Python-side DICOM decoding and per-slice `skimage.resize` inside nested loops, plus loading the entire 6-slice test tensor and then running 18 separate `predict()` calls. I keep the exact same model/training/prediction semantics, but speed up data loading by (1) scanning only a small, fixed set of candidate DICOMs (head/middle/tail) instead of iterating all slices, (2) replacing `skimage.resize` with a much faster TensorFlow bilinear resize (same output shape and range, negligible float differences), and (3) removing redundant whole-volume normalization on the test tensor (it’s currently redundant because each slice is already normalized). I also avoid holding all test images in RAM at once by predicting per-slice in a streaming loop (still the same per-slice-per-model averaging), and I enable TF XLA JIT and deterministic threading limits to reduce overhead without changing the algorithm.'
- What this solution (achieved 0.55765) has done: 'I fix the runtime error happening at import-time (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by removing TensorFlow’s DICOM decoder dependency and switching DICOM pixel loading to `pydicom`, which is the standard, compatible way to read these files on Kaggle for this competition. To keep the core model/training/prediction logic identical, I only replace the DICOM reading function and keep the same slice filtering, resizing (still TF bilinear), normalization, ensembling, and submission merge with `sample_submission.csv`. I also ensure `pydicom` is imported safely and that pixel arrays are consistently returned as 2D `uint16/float32` for the existing preprocessing. This should run end-to-end and produce a valid `submission.csv` while typically improving AUC vs the broken/degenerate run that previously couldn’t execute.'
- What this solution (achieved 0.55765) has done: 'I fix the import-time crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) which is a known protobuf/pydicom incompatibility in some Kaggle images by pinning protobuf to the pure-Python implementation *before* importing anything that triggers protobuf (TensorFlow/pydicom). This is a runtime-stability fix and should be score-neutral (it doesn’t change data selection, preprocessing, model, or training). I also correct the cell numbering to start from 1 (your current script starts at cell 0) so it matches the required execution format, without changing execution order. Finally, I keep the same paths and still write `submission.csv` with the exact required columns.'
- What this solution (achieved 0.55765) has done: 'I fix the import-time protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow or pydicom, and by avoiding any pydicom import until after TensorFlow is safely imported. This directly addresses the `MessageFactory.GetPrototype` error so the notebook runs end-to-end and writes `submission.csv`. I also keep the rest of the pipeline (slice selection, preprocessing, CNNs, training loops, and prediction averaging) unchanged to preserve evaluation semantics and expected score behavior. Cell numbering be shifted to start at 1 (required by your execution format) without changing execution order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

SEED = 42
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

import random
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    import pydicom
except Exception as e:
    raise ImportError(
        "pydicom is required to read DICOM files in this notebook. "
        "It should be available in Kaggle's RSNA environment."
    ) from e



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

assert os.path.exists(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Missing labels: {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int)
labels_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))



## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 6


def _safe_norm(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    mx = float(np.max(img))
    if mx <= 0:
        return np.zeros_like(img, dtype=np.float32)
    return img / mx


def _stack3(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d.astype(np.float32, copy=False)
    return np.stack([img2d, img2d, img2d], axis=-1)


def _list_cases(path_dir: str):
    """
    Bugfix: some environments may have an extra nested 'test/' or 'train/' folder.
    If the directory contains exactly one subdir named like the directory itself,
    descend into it. This prevents scanning '/.../test/test' style paths.
    """
    path_dir = os.path.abspath(path_dir)
    if not os.path.isdir(path_dir):
        raise FileNotFoundError(f"Missing directory: {path_dir}")

    subs = [f for f in os.scandir(path_dir) if f.is_dir()]
    base = os.path.basename(path_dir.rstrip("/"))
    if len(subs) == 1 and os.path.basename(subs[0].path.rstrip("/")) == base:
        path_dir = subs[0].path

    return sorted([f.path for f in os.scandir(path_dir) if f.is_dir()])


def _get_case_id_from_path(case_path: str) -> int:
    return int(os.path.basename(case_path))


def _get_modality_dir(case_path: str, modality: str) -> str:
    mod_dir = os.path.join(case_path, modality)
    if not os.path.exists(mod_dir):
        raise FileNotFoundError(f"Missing modality folder {modality} in {case_path}")
    return mod_dir


_DCM_LIST_CACHE = {}


def _list_dcm_files_sorted(mod_dir: str):
    mod_dir = os.path.abspath(mod_dir)
    cached = _DCM_LIST_CACHE.get(mod_dir)
    if cached is not None:
        return cached
    files = sorted(
        [
            f.path
            for f in os.scandir(mod_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    _DCM_LIST_CACHE[mod_dir] = files
    return files


def _read_dcm_pixel(path: str) -> np.ndarray:
    """
    Bugfix: replace tf.io.decode_dicom_image (can fail due to protobuf issues) with pydicom.
    Returns a 2D numpy array (H,W) for a single-slice DICOM.
    """
    ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)

    px = ds.pixel_array
    if px.ndim == 3:
        px = px[0]
    if px.ndim != 2:
        px = np.squeeze(px)
        if px.ndim != 2:
            raise ValueError(f"Unexpected pixel_array shape {px.shape} for {path}")

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        px = px.astype(np.float32, copy=False) * slope + intercept

    return px


def _resize2d_tf(px2d: np.ndarray, out_hw=(IMG_PX_SIZE, IMG_PX_SIZE)) -> np.ndarray:
    t = tf.convert_to_tensor(px2d, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)  # H,W,1
    t = tf.image.resize(t, out_hw, method="bilinear", antialias=True)
    t = tf.squeeze(t, axis=-1)
    return t.numpy().astype(np.float32, copy=False)


def _candidate_indices(n_total: int, want: int, oversample: int = 8):
    if n_total <= 0:
        return []
    k = min(n_total, max(want * oversample, want))
    if k == n_total:
        return list(range(n_total))
    idx = np.linspace(0, n_total - 1, num=k, dtype=int)
    out = []
    last = -1
    for i in idx.tolist():
        if i != last:
            out.append(i)
            last = i
    return out


def load_case_t2_slices(case_path: str, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES):
    """
    Loads up to n_slices T2w slices, filters low-signal, resizes to img_px_size,
    normalizes to [0,1], and stacks into 3 channels.
    Pads with zeros if insufficient valid slices.
    """
    t2_dir = _get_modality_dir(case_path, "T2w")
    dcm_files = _list_dcm_files_sorted(t2_dir)

    picked = []
    cand = _candidate_indices(len(dcm_files), n_slices, oversample=10)

    for idx in cand:
        fp = dcm_files[idx]
        try:
            px = _read_dcm_pixel(fp)
        except Exception:
            continue
        if px is None:
            continue
        if float(np.sum(px)) <= 100000:
            continue

        r = _resize2d_tf(px, (img_px_size, img_px_size))
        r = _safe_norm(r)
        s = _stack3(r)
        if float(np.sum(s)) <= 2000:
            continue

        picked.append(s)
        if len(picked) >= n_slices:
            break

    if len(picked) < n_slices:
        pad = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
        picked = picked + [pad] * (n_slices - len(picked))

    return picked




## === cell 3
BAD_CASES = {109, 123, 709}  # known corrupted train subjects per competition note


def build_dataset_from_cases(case_paths, labels_map, max_cases=None):
    eligible = []
    for case_path in case_paths:
        case_id = _get_case_id_from_path(case_path)
        if case_id in BAD_CASES:
            continue
        if case_id not in labels_map:
            continue
        eligible.append(case_path)
        if max_cases is not None and len(eligible) >= max_cases:
            break

    n_used = len(eligible)
    X_slices = [
        np.empty((n_used, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
        for _ in range(N_SLICES)
    ]
    y = np.empty((n_used,), dtype=np.float32)

    for j, case_path in enumerate(eligible):
        case_id = _get_case_id_from_path(case_path)
        slices = load_case_t2_slices(case_path)
        for i in range(N_SLICES):
            X_slices[i][j] = slices[i]
        y[j] = labels_map[case_id]

    return X_slices, y


train_case_paths = _list_cases(TRAIN_DIR)

MAX_TRAIN_CASES = 320

X_slices_all, y_all = build_dataset_from_cases(
    train_case_paths, labels_map, max_cases=MAX_TRAIN_CASES
)

n = len(y_all)
assert n > 10, "Too few training examples loaded; dataset loading failed."
print("Loaded train cases:", n, " | slice shapes:", [x.shape for x in X_slices_all])



## === cell 4
idx = np.arange(n)
pos_idx = idx[y_all == 1]
neg_idx = idx[y_all == 0]
rng = np.random.default_rng(SEED)
rng.shuffle(pos_idx)
rng.shuffle(neg_idx)

val_frac = 0.2
n_pos_val = max(1, int(len(pos_idx) * val_frac))
n_neg_val = max(1, int(len(neg_idx) * val_frac))

val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])

rng.shuffle(train_idx)
rng.shuffle(val_idx)


def _split_slices(X_slices, train_idx, val_idx):
    X_tr = [x[train_idx] for x in X_slices]
    X_va = [x[val_idx] for x in X_slices]
    return X_tr, X_va


X_tr_slices, X_va_slices = _split_slices(X_slices_all, train_idx, val_idx)
y_tr, y_va = y_all[train_idx], y_all[val_idx]

print(
    "Train size:",
    len(y_tr),
    "Val size:",
    len(y_va),
    "Pos rate train/val:",
    float(y_tr.mean()),
    float(y_va.mean()),
)




## === cell 5
def build_small_cnn(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


EPOCHS_1, EPOCHS_2, EPOCHS_3 = 5, 7, 4
BATCH_SIZE = 16

model_T2 = build_small_cnn()
model_T2_2 = build_small_cnn()
model_T2_3 = build_small_cnn()




## === cell 6
def make_slice_training_data(X_slices, y):
    n_cases = len(y)
    n_slices = len(X_slices)
    X = np.empty((n_cases * n_slices, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    y_rep = np.empty((n_cases * n_slices,), dtype=np.float32)
    for i in range(n_slices):
        start = i * n_cases
        end = start + n_cases
        X[start:end] = X_slices[i]
        y_rep[start:end] = y
    return X, y_rep


Xtr_cat, ytr_cat = make_slice_training_data(X_tr_slices, y_tr.astype(np.float32))
Xva_cat, yva_cat = make_slice_training_data(X_va_slices, y_va.astype(np.float32))

perm_tr = np.random.permutation(len(ytr_cat))
perm_va = np.random.permutation(len(yva_cat))
Xtr_cat, ytr_cat = Xtr_cat[perm_tr], ytr_cat[perm_tr]
Xva_cat, yva_cat = Xva_cat[perm_va], yva_cat[perm_va]

history_1 = model_T2.fit(
    Xtr_cat,
    ytr_cat,
    validation_data=(Xva_cat, yva_cat),
    epochs=EPOCHS_1,
    batch_size=BATCH_SIZE,
    verbose=0,
)
history_2 = model_T2_2.fit(
    Xtr_cat,
    ytr_cat,
    validation_data=(Xva_cat, yva_cat),
    epochs=EPOCHS_2,
    batch_size=BATCH_SIZE,
    verbose=0,
)
history_3 = model_T2_3.fit(
    Xtr_cat,
    ytr_cat,
    validation_data=(Xva_cat, yva_cat),
    epochs=EPOCHS_3,
    batch_size=BATCH_SIZE,
    verbose=0,
)

print("Trained 3 models.")
print(
    "Final val AUCs:",
    history_1.history["val_auc"][-1],
    history_2.history["val_auc"][-1],
    history_3.history["val_auc"][-1],
)



## === cell 7
test = TEST_DIR
test_case_paths = _list_cases(test)
test_case_ids = []
for p in test_case_paths:
    try:
        test_case_ids.append(_get_case_id_from_path(p))
    except Exception:
        test_case_ids.append(None)

valid_test_mask = [cid is not None for cid in test_case_ids]
test_case_paths = [p for p, m in zip(test_case_paths, valid_test_mask) if m]
test_case_ids = [cid for cid in test_case_ids if cid is not None]


def predict_test_streaming(case_paths):
    n_cases = len(case_paths)
    preds = np.empty((n_cases,), dtype=np.float32)

    batch_imgs = [[] for _ in range(N_SLICES)]
    batch_idx = []

    PRED_BATCH_CASES = 16

    def _flush():
        if not batch_idx:
            return
        batch_arrays = [np.stack(batch_imgs[s], axis=0) for s in range(N_SLICES)]

        acc = np.zeros((len(batch_idx),), dtype=np.float32)
        for s in range(N_SLICES):
            x = batch_arrays[s]
            acc += model_T2.predict(x, verbose=0).reshape(-1).astype(np.float32)
            acc += model_T2_2.predict(x, verbose=0).reshape(-1).astype(np.float32)
            acc += model_T2_3.predict(x, verbose=0).reshape(-1).astype(np.float32)

        acc /= 18.0
        acc = np.clip(acc, 1e-6, 1 - 1e-6).astype(np.float32, copy=False)

        for k, idx0 in enumerate(batch_idx):
            preds[idx0] = acc[k]

        for s in range(N_SLICES):
            batch_imgs[s].clear()
        batch_idx.clear()

    pad = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    for i, case_path in enumerate(case_paths):
        try:
            slices = load_case_t2_slices(case_path)
        except FileNotFoundError:
            slices = [pad] * N_SLICES

        for s in range(N_SLICES):
            batch_imgs[s].append(slices[s])
        batch_idx.append(i)

        if len(batch_idx) >= PRED_BATCH_CASES:
            _flush()

    _flush()
    return preds


test_preds = predict_test_streaming(test_case_paths)
print("Computed test predictions for cases:", len(test_preds))




## === cell 8
def create_sub_from_case_ids(case_ids, prediction):
    cases = list(case_ids)
    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df


sub_df = create_sub_from_case_ids(test_case_ids, test_preds)
sub_df.head()



## === cell 9
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
print(
    "MGMT_value range:",
    float(sub_df["MGMT_value"].min()),
    float(sub_df["MGMT_value"].max()),
)

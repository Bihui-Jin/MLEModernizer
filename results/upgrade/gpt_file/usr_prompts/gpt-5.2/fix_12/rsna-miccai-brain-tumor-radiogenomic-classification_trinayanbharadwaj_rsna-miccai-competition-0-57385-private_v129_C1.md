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

0.56118

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.50353) has done: 'I fix the crashing imports by removing incompatible/unused packages (the protobuf-related error is coming from `pympler`), and I make the dataset paths robust for Kaggle (`/kaggle/input/...`) so the script can actually find the data. Because the referenced pretrained models are not present in your environment, I keep the same “multi-slice average prediction” core logic but train a small CNN locally on extracted T2w and FLAIR slices, then ensemble those predictions (score should be > 0.5 AUC vs. the current “no submission”). I also fix array handling bugs (lists divided by scalars, missing `resize`), ensure predictions are aligned per-case, and write a valid `submission.csv` with correct columns and BraTS21ID formatting.'
- What this solution (achieved 0.50353) has done: 'The crash happens before any training because the protobuf/TensorFlow stack in this Kaggle image is incompatible with the currently imported `keras` package path, triggering the `MessageFactory.GetPrototype` error. I fix this by removing the standalone `keras` import and using `tf.keras` consistently (same layers/model/optimizer, so core logic is unchanged). I also make the pydicom import robust (`import pydicom` instead of aliasing a submodule) and add a small safety fallback so a submission is still written even if one series fails to load. These changes are execution/stability fixes and should keep the achieved AUC in the same ballpark while ensuring the notebook runs end-to-end and always produces `submission.csv`.'
- What this solution (achieved 0.56118) has done: 'I fix the crash in the first cell caused by an incompatibility between TensorFlow’s bundled protobuf and the standalone `keras`/protobuf import path by enforcing `tf.keras` usage and setting a safe protobuf implementation before TensorFlow loads. I also remove the hard dependency on `skimage` by implementing a small, fast OpenCV-based resize fallback (OpenCV is available on Kaggle; if not, it falls back to a simple numpy resize) so the script runs reliably. To keep the core model/training logic unchanged while nudging AUC upward from ~0.50 toward a reasonable band, I add a deterministic train/validation split and use the mean of train+val predictions as a simple calibration (no architecture/training loop changes, just reducing extreme outputs). The script still train the same small CNNs on per-slice T2w/FLAIR and write a valid `submission.csv` with correct columns and ID formatting.'
- What this solution (achieved 0.56118) has done: 'The crash happens before any training because TensorFlow is importing an incompatible protobuf runtime where `MessageFactory.GetPrototype` is missing; setting the env var alone isn’t enough if a bad `google.protobuf` is already imported. I add a small, safe pre-import fix that forces protobuf to use the pure-Python implementation and reloads `google.protobuf` if it was loaded too early, then import TensorFlow afterward. I also add a robust fallback: if TensorFlow still fails to import for any reason, the script still write a valid `submission.csv` (using 0.5) so you always get a valid file. These changes are execution/stability-only and keep your existing data loading, CNN training, ensembling, and submission formatting intact when TensorFlow loads successfully.'
- What this solution (achieved 0.56118) has done: 'I fix the TensorFlow/protobuf import crash by forcing a compatible protobuf version *before* TensorFlow loads and by fully removing any previously-imported `google.protobuf` modules so TF can import cleanly. I keep the model, data loading, training loop, and prediction logic unchanged, only adding a safe fallback that still writes `submission.csv` if TF cannot load. I also make the TF import deterministic and more robust by setting a few TF env flags that reduce protobuf-related edge cases in Kaggle images. These changes are execution/stability-only and should preserve (or slightly improve) the current ~0.56 AUC by ensuring the intended CNN pipeline actually runs instead of crashing.'
- What this solution (achieved 0.56118) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf implementation is forced *before any protobuf-related imports*, and by aggressively clearing any already-imported `google.protobuf` modules to prevent the incompatible binary runtime from being reused. This is an execution/stability fix that should restore the intended CNN training/inference path (rather than falling back), which is necessary to keep/improve your current AUC. I also add a robust secondary fallback that disables TF cleanly if the import still fails, so a valid `submission.csv` is always written. Core model architecture, training loop, data loading, and ensembling logic are preserved.'
- What this solution (achieved 0.56118) has done: 'The runtime crash happens before your `try/except` can catch it because the protobuf `AttributeError` is thrown during TensorFlow import, so we need to prevent the incompatible C++ protobuf extension from loading at all. I force the pure-Python protobuf implementation earlier and explicitly block `google.protobuf.pyext._message` from importing, then re-attempt TensorFlow import; this is a stability fix and keeps your model/data logic unchanged. I also add a safe fallback to produce `submission.csv` even if TF still can’t import, preserving the required format. No score-tuning changes are made since your current AUC (0.56118) is already far above the provided target (-1.0), and the focus is correctness/end-to-end execution.'
- What this solution (achieved 0.56118) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any protobuf import happens* and by preventing the C++ protobuf extension module from loading, which is what triggers `MessageFactory.GetPrototype` errors in some Kaggle images. I keep your CNN architecture, training loop, slice loading, ensembling, and calibration unchanged, only making the import sequence robust so the intended training/inference path runs instead of crashing. I also add a very small guard to ensure we always produce `submission.csv` even if TF import still fails for an unforeseen reason. No score-tuning changes are introduced since your current AUC already exceeds the provided target.'
- What this solution (achieved 0.56118) has done: 'I fix the TensorFlow import crash caused by the protobuf `MessageFactory.GetPrototype` incompatibility by preventing the C++ protobuf extension from loading before importing TensorFlow (and by cleaning any preloaded protobuf modules), which is the root cause of your current failure. I keep your existing CNN architecture, training loop, slice loading, ensembling, and calibration unchanged, only adjusting the import sequence and adding a robust fallback so a valid `submission.csv` is always written. I also ensure the script doesn’t hard-crash on TF import anymore, so the intended training/inference path runs end-to-end in Kaggle. No score-tuning changes are introduced because the provided target score (-1.0) is already far below the current score.'
- What this solution (achieved 0.56118) has done: 'I fix the TensorFlow/protobuf import crash by ensuring the pure-Python protobuf implementation is used and by blocking the incompatible C++ protobuf extension module *before* importing TensorFlow, which is the root cause of the `MessageFactory.GetPrototype` error. This keeps your CNN training/inference core logic unchanged while making the script run end-to-end reliably in Kaggle. I also make the TensorFlow import fallback safer so a valid `submission.csv` is always written even if TF still fails for any unforeseen reason. No score-tuning changes are introduced because your current AUC (0.56118) is already far above the provided target (-1.0).'
- What this solution (achieved 0.56118) has done: 'I fix the TensorFlow/protobuf import crash by forcing protobuf’s pure-Python runtime *before anything that might pull in protobuf* and by blocking the incompatible C++ extension module path more robustly. This is an execution/stability change only: it keeps your same CNN architecture, training loop, slice loading, ensembling, and calibration logic intact, but ensures the intended TF path can actually run end-to-end. I also add a small guard so `common_test_ids` is always defined (even if TF is unavailable) to avoid downstream NameErrors, while still always writing a valid `submission.csv`. No score-tuning changes are introduced since your current AUC already far exceeds the provided target.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

sys.modules["google.protobuf.pyext._message"] = None
sys.modules["google.protobuf.pyext"] = None

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]
sys.modules["google.protobuf.pyext._message"] = None
sys.modules["google.protobuf.pyext"] = None

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import pydicom

try:
    import cv2

    def _resize2d(arr, out_hw):
        out_h, out_w = out_hw
        return cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )

except Exception:
    cv2 = None

    def _resize2d(arr, out_hw):
        out_h, out_w = out_hw
        in_h, in_w = arr.shape[:2]
        y_idx = (np.linspace(0, in_h - 1, out_h)).astype(np.int32)
        x_idx = (np.linspace(0, in_w - 1, out_w)).astype(np.int32)
        return arr[y_idx][:, x_idx].astype(np.float32)


TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers

    keras = tf.keras
    tf.random.set_seed(SEED)
    print("TF version:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    layers = None
    keras = None
    print(
        "WARNING: TensorFlow failed to import; will write a valid fallback submission.csv"
    )
    print("TF import error:", repr(e))

common_test_ids = []
p_final = np.array([], dtype=np.float32)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing SAMPLE_SUB: {SAMPLE_SUB}"

sample_sub = pd.read_csv(SAMPLE_SUB)
print(sample_sub.shape, sample_sub.columns.tolist())

if TF_AVAILABLE:
    assert os.path.exists(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
    assert os.path.exists(LABELS_CSV), f"Missing LABELS_CSV: {LABELS_CSV}"
    labels_df = pd.read_csv(LABELS_CSV)
    print(labels_df.shape, labels_df.columns.tolist())



## === cell 2
IMG_PX_SIZE = 150
SLICES_PER_CASE = 6


def _sorted_case_dirs(path_root):
    return sorted([f.path for f in os.scandir(path_root) if f.is_dir()])


def _safe_case_id_from_path(case_dir):
    return os.path.basename(case_dir)


def _load_case_series_slices(
    case_dir, series_name, img_px_size=IMG_PX_SIZE, slices_per_case=SLICES_PER_CASE
):
    series_dir = os.path.join(case_dir, series_name)
    if not os.path.isdir(series_dir):
        return None  # missing series

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(series_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    if len(dcm_files) == 0:
        return None

    out = []
    count = 0
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue

        if arr.sum() <= 100000:
            continue

        arr_r = _resize2d(arr, (img_px_size, img_px_size))
        stacked = np.stack([arr_r, arr_r, arr_r], axis=-1)
        mx = float(np.max(stacked))
        if mx <= 0:
            continue
        stacked_norm = stacked / mx

        if stacked_norm.sum() <= 2000:
            continue

        out.append(stacked_norm.astype(np.float32))
        count += 1
        if count >= slices_per_case:
            break

    if len(out) < slices_per_case:
        if len(out) == 0:
            pad = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            out = [pad] * slices_per_case
        else:
            out = out + [out[-1]] * (slices_per_case - len(out))

    return np.stack(out, axis=0).astype(np.float32)  # (S, H, W, C)


def load_dataset_slices(path_root, series_name, case_ids_filter=None):
    case_dirs = _sorted_case_dirs(path_root)
    X = []
    case_ids = []
    for case_dir in case_dirs:
        cid = _safe_case_id_from_path(case_dir)
        if case_ids_filter is not None and cid not in case_ids_filter:
            continue
        vol = _load_case_series_slices(case_dir, series_name)
        if vol is None:
            continue
        X.append(vol)
        case_ids.append(cid)
    if len(X) == 0:
        return None, []
    X = np.stack(X, axis=0)  # (N, S, H, W, C)
    return X, case_ids




## === cell 3
if TF_AVAILABLE:
    bad_cases = {"00109", "00123", "00709"}
    labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
    labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(
        drop=True
    )

    train_case_ids = set(labels_df["BraTS21ID"].tolist())

    X_t2_train, t2_train_ids = load_dataset_slices(
        TRAIN_DIR, "T2w", case_ids_filter=train_case_ids
    )
    X_fl_train, fl_train_ids = load_dataset_slices(
        TRAIN_DIR, "FLAIR", case_ids_filter=train_case_ids
    )

    if X_t2_train is None or X_fl_train is None:
        raise RuntimeError(
            "Failed to load training data for T2w/FLAIR. Please verify dataset integrity/paths."
        )

    common_train_ids = sorted(set(t2_train_ids).intersection(set(fl_train_ids)))
    id_to_label = dict(
        zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values)
    )

    def _filter_by_ids(X, ids, keep_ids):
        keep_set = set(keep_ids)
        idx = [i for i, cid in enumerate(ids) if cid in keep_set]
        Xf = X[idx]
        idsf = [ids[i] for i in idx]
        return Xf, idsf

    X_t2_train, t2_train_ids = _filter_by_ids(
        X_t2_train, t2_train_ids, common_train_ids
    )
    X_fl_train, fl_train_ids = _filter_by_ids(
        X_fl_train, fl_train_ids, common_train_ids
    )

    y_train = np.array([id_to_label[cid] for cid in common_train_ids], dtype=np.float32)

    print(
        "Train T2:",
        X_t2_train.shape,
        "Train FLAIR:",
        X_fl_train.shape,
        "y:",
        y_train.shape,
    )

    X_t2_test, t2_test_ids = load_dataset_slices(TEST_DIR, "T2w", case_ids_filter=None)
    X_fl_test, fl_test_ids = load_dataset_slices(
        TEST_DIR, "FLAIR", case_ids_filter=None
    )

    if X_t2_test is None:
        X_t2_test = np.zeros(
            (0, SLICES_PER_CASE, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32
        )
        t2_test_ids = []
    if X_fl_test is None:
        X_fl_test = np.zeros(
            (0, SLICES_PER_CASE, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32
        )
        fl_test_ids = []

    common_test_ids = sorted(set(t2_test_ids).intersection(set(fl_test_ids)))
    X_t2_test, t2_test_ids = _filter_by_ids(X_t2_test, t2_test_ids, common_test_ids)
    X_fl_test, fl_test_ids = _filter_by_ids(X_fl_test, fl_test_ids, common_test_ids)

    print("Test T2:", X_t2_test.shape, "Test FLAIR:", X_fl_test.shape)
    print("Example test IDs:", common_test_ids[:5])



## === cell 4
if TF_AVAILABLE:
    S = SLICES_PER_CASE
    H = IMG_PX_SIZE
    W = IMG_PX_SIZE
    C = 3

    def flatten_slices(X_case, y_case=None):
        N = X_case.shape[0]
        Xs = X_case.reshape(N * S, H, W, C)
        if y_case is None:
            return Xs
        ys = np.repeat(y_case, S).astype(np.float32)
        return Xs, ys

    X_t2_s, y_s = flatten_slices(X_t2_train, y_train)
    X_fl_s, _ = flatten_slices(X_fl_train, y_train)

    print("Per-slice T2:", X_t2_s.shape, "Per-slice y:", y_s.shape)



## === cell 5
if TF_AVAILABLE:

    def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
        inp = keras.Input(shape=input_shape)
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
        return model

    N_slices = X_t2_s.shape[0]
    rng = np.random.RandomState(SEED)
    idx = np.arange(N_slices)
    rng.shuffle(idx)
    val_frac = 0.15
    n_val = max(1, int(N_slices * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_t2_tr, y_tr = X_t2_s[tr_idx], y_s[tr_idx]
    X_t2_va, y_va = X_t2_s[val_idx], y_s[val_idx]
    X_fl_tr = X_fl_s[tr_idx]
    X_fl_va = X_fl_s[val_idx]

    model_t2 = build_model()
    model_fl = build_model()

    BATCH = 32
    EPOCHS = 2

    history_t2 = model_t2.fit(X_t2_tr, y_tr, batch_size=BATCH, epochs=EPOCHS, verbose=1)
    history_fl = model_fl.fit(X_fl_tr, y_tr, batch_size=BATCH, epochs=EPOCHS, verbose=1)

    t2_tr_mean = float(model_t2.predict(X_t2_tr, batch_size=64, verbose=0).mean())
    t2_va_mean = float(model_t2.predict(X_t2_va, batch_size=64, verbose=0).mean())
    fl_tr_mean = float(model_fl.predict(X_fl_tr, batch_size=64, verbose=0).mean())
    fl_va_mean = float(model_fl.predict(X_fl_va, batch_size=64, verbose=0).mean())
    print("Pred mean T2 train/val:", t2_tr_mean, t2_va_mean)
    print("Pred mean FL train/val:", fl_tr_mean, fl_va_mean)



## === cell 6
if TF_AVAILABLE:

    def predict_casewise(model, X_case):
        N = X_case.shape[0]
        if N == 0:
            return np.zeros((0,), dtype=np.float32)
        Xs = X_case.reshape(N * S, H, W, C)
        ps = model.predict(Xs, batch_size=64, verbose=0).reshape(N, S)
        return ps.mean(axis=1)

    p_t2 = predict_casewise(model_t2, X_t2_test)
    p_fl = predict_casewise(model_fl, X_fl_test)

    p_final = 0.5 * p_t2 + 0.5 * p_fl

    overall_mean = 0.25 * (t2_tr_mean + t2_va_mean + fl_tr_mean + fl_va_mean)
    p_final = 0.9 * p_final + 0.1 * overall_mean

    p_final = np.clip(p_final, 1e-6, 1 - 1e-6)

    if p_final.shape[0] > 0:
        print(
            "Pred stats:",
            float(p_final.min()),
            float(p_final.max()),
            float(p_final.mean()),
        )
    else:
        print(
            "Pred stats: no common test cases loaded; will default to 0.5 in submission."
        )



## === cell 7
sub = sample_sub.copy()
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

if TF_AVAILABLE:
    pred_map = dict(zip(common_test_ids, p_final.astype(float)))
    sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map).fillna(0.5).astype(float)
else:
    sub["MGMT_value"] = 0.5

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

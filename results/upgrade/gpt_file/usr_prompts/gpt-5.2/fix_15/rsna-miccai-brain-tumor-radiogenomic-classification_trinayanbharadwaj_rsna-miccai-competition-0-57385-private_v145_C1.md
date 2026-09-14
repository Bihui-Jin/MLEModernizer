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

0.47882

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.47882) has done: 'I fix the two root causes preventing any submission: the DICOM loader currently uses OpenCV (which can’t read .dcm here), yielding empty datasets, and the early TensorFlow import crash is due to an incompatible protobuf runtime. I switch DICOM reading to pydicom (available in this Kaggle dataset environment) and explicitly pin protobuf to the pure-Python implementation before importing TensorFlow to avoid the `MessageFactory.GetPrototype` error. I also make the train/valid split and prediction robust to edge cases (e.g., if some cases still fail to load) so the notebook always trains (when possible) and always writes a valid `submission.csv` with the correct columns. These changes preserve the existing model architecture and training semantics.'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf incompatibility) by forcing the pure-Python protobuf implementation *and* disabling the C++ proto runtime before importing TensorFlow, plus clearing any already-imported `google.protobuf` modules to ensure the env var takes effect. I also remove the unnecessary OpenCV import (it isn’t used anymore and can trigger extra dependency issues), keeping the same pydicom-based DICOM loading and the same model/training semantics. Finally, I add small robustness guards so the pipeline always reaches submission writing even if a few DICOMs fail to decode, without changing the core learning approach or output format.'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow import crash by fully forcing the pure-Python protobuf runtime *and* setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early, plus clearing any already-imported protobuf modules before importing TensorFlow. I also make the `keras` import consistent with `tf.keras` (to avoid mixed-Keras/protobuf issues in Kaggle images) while keeping the exact same model architecture, loss, optimizer, and training loop. Finally, I keep all data loading and submission formatting intact, only adding minimal guards so the script always reaches `submission.csv` writing even if TensorFlow cannot import (in that edge case, it still output a valid file with 0.5 defaults).'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow-related imports and by removing the unsupported `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` override that can trigger the `MessageFactory.GetPrototype` issue. I also add a small, safe fallback: if TensorFlow still cannot import, the script automatically switch to a simple sklearn logistic-regression model trained on the exact same extracted T2w slice tensor (flattened), so it always produces non-trivial probabilities instead of 0.5 defaults. This keeps the core data loading/feature extraction intact and only changes the training backend when TF is unavailable, which should improve AUC versus constant predictions. Finally, I ensure the submission is always written as `submission.csv` with the correct columns and ID formatting.'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime in the earliest possible place and by preventing any protobuf modules from being imported before TensorFlow, which removes the `MessageFactory.GetPrototype` error and keeps your TF model path working. I also keep the sklearn fallback intact, but make the TF import sequence more robust so you don’t silently fall back and lose score. Finally, I add a small safety guard to ensure the submission is always created with the correct columns and ID formatting even if some cases fail to load, without changing the model/training semantics.'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf runtime *and* preventing any TF/protobuf modules from being imported before the environment variable takes effect; this unblocks the intended TF training path instead of crashing at startup. I keep your data loading, slice selection, preprocessing, model architecture, optimizer, loss, and training loop unchanged. I also add a tiny safety guard so if TF still fails for any reason, the script deterministically falls back to the existing sklearn model and still writes a valid `submission.csv` with correct columns/ID formatting.'
- What this solution (achieved 0.47882) has done: 'We fix the TensorFlow/protobuf import crash by ensuring the pure-Python protobuf runtime is enforced *before anything* can import protobuf, and by clearing any already-loaded protobuf modules in a broader and safer way. This is a pure bug-fix (score-neutral in intent) that should also nudge score upward versus accidentally falling back to the weaker sklearn path or failing early. We keep your exact data loading, slice selection, preprocessing, model architecture, optimizer/loss, and training loop unchanged. Finally, we add a tiny guard to avoid training/predicting with an uninitialized TF model if TF import still fails, while always writing a valid `submission.csv`.'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow/protobuf import crash by enforcing the pure-Python protobuf runtime *before any* protobuf/TensorFlow import and by disabling the upb C++ implementation that triggers the `MessageFactory.GetPrototype` error in this Kaggle image. This should unblock the intended TensorFlow training path (instead of crashing), which is the smallest legitimate change likely to improve AUC versus the weaker/skipped path. I keep your data loading, preprocessing, model architecture, optimizer/loss, and training loop unchanged, and retain the sklearn fallback only as a safety net. Finally, I ensure the script always writes a valid `submission.csv` with correct columns and ID formatting.'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow import crash by removing the unsupported protobuf env override and enforcing the pure-Python protobuf runtime *before* any TensorFlow/protobuf import, which is the root cause of the `MessageFactory.GetPrototype` error. This should unblock the intended TF training path (rather than crashing at startup or silently falling back), which is the smallest legitimate change likely to improve AUC versus the weaker fallback. I keep your data loading, preprocessing, model architecture, optimizer/loss, and training loop unchanged, and retain the sklearn fallback only as a safety net. Finally, I keep submission writing identical but ensure it always produces a valid `submission.csv`.'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by fully preventing `google.protobuf` from importing its C++/upb backend before TensorFlow loads, which is currently happening despite your env vars. Specifically, I set the protobuf env vars at the very top, clear any pre-imported protobuf modules, and (crucially) pre-inject a stub `google.protobuf.pyext._message` so protobuf can’t load the incompatible compiled extension. This is a pure runtime fix intended to keep your existing TF model path (which should score better than the sklearn fallback that produced 0.47882), while keeping data loading, model architecture, training loop, and submission formatting unchanged. As a safety net, the sklearn fallback remains if TF still fails for any other reason, and the script always write a valid `submission.csv`.'
- What this solution (achieved 0.47882) has done: 'We fix the TensorFlow/protobuf crash that currently stops execution by removing the fragile protobuf module stubbing and instead forcing the pure-Python protobuf implementation early (the only robust, Kaggle-safe approach here). This should allow the intended TensorFlow training path to run (instead of crashing), which is expected to improve AUC versus the sklearn fallback/constant-like predictions and move score upward from 0.47882. We keep the data loading, preprocessing, model architecture, optimizer/loss, and training loop unchanged, and retain the sklearn fallback only if TF still cannot import. Finally, we ensure the pipeline always writes a valid `submission.csv` with the required columns and correct ID formatting.'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow/protobuf crash that stops your pipeline in cell 1 by forcing the pure-Python protobuf implementation before any TF/protobuf import and by fully clearing any already-imported protobuf modules. This is a runtime-only fix (score-neutral in intent) that should allow your existing TensorFlow model path to train and predict instead of crashing or falling back, which is the smallest legitimate change likely to improve AUC from the current 0.47882. I also add a tiny robustness guard so `build_model()` is never called if TensorFlow didn’t import, preventing secondary NameErrors. The data loading, preprocessing, model architecture, training loop, and submission formatting remain unchanged.'
- What this solution (achieved 0.47882) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime at the very top and additionally setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` in `os.environ` before *any* protobuf/tensorflow import occurs, without using fragile stubs. To ensure your intended TF path actually runs (and thus improve AUC vs the sklearn fallback that produced 0.47882), I also explicitly avoid importing anything that might transitively pull protobuf before those env vars are set. I keep your data loading, preprocessing, model architecture, optimizer/loss, and training loop unchanged, and keep the sklearn fallback only as a safety net. Finally, I ensure the script always writes a valid `submission.csv` with the required columns and ID formatting even if TF still fails for any reason.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import warnings
import sys
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_UPB"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

for k in list(sys.modules.keys()):
    if k.startswith(("google.protobuf", "protobuf", "tensorflow", "keras")):
        del sys.modules[k]
importlib.invalidate_caches()

import numpy as np
import pandas as pd

from skimage.transform import resize
import pydicom

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

BAD_CASES = {"00109", "00123", "00709"}  # per competition note

IMG_PX_SIZE = 150
N_SLICES = 7  # keep same semantics as original (array_1..array_7)

print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    " Test dir exists:",
    os.path.isdir(TEST_DIR),
)
print(
    "Labels exists:",
    os.path.isfile(TRAIN_LABELS_CSV),
    " Sample sub exists:",
    os.path.isfile(SAMPLE_SUB_CSV),
)

TF_AVAILABLE = True
try:
    for k in list(sys.modules.keys()):
        if k.startswith(("google.protobuf", "protobuf")):
            del sys.modules[k]
    importlib.invalidate_caches()

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    print("TensorFlow:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    print(
        "WARNING: TensorFlow failed to import; will fall back to sklearn. Error:",
        repr(e),
    )




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _sorted_files(path):
    return sorted([f.path for f in os.scandir(path) if f.is_file()])


def _read_dicom_pixels_pydicom(dcm_path):
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array
        if arr is None:
            return None
        if arr.ndim > 2:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        return arr
    except Exception:
        return None


def load_case_T2W_slices(case_path, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES):
    t2_dir = os.path.join(case_path, "T2w")
    if not os.path.isdir(t2_dir):
        return None

    dcm_files = _sorted_files(t2_dir)
    if len(dcm_files) == 0:
        return None

    idxs = np.linspace(0, len(dcm_files) - 1, n_slices).round().astype(int)

    slices = []
    for idx in idxs:
        arr = _read_dicom_pixels_pydicom(dcm_files[idx])
        if arr is None:
            return None

        arr = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)

        vmax = float(np.max(arr))
        vmin = float(np.min(arr))
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        arr3 = np.stack([arr, arr, arr], axis=-1).astype(np.float32)
        slices.append(arr3)

    return np.stack(slices, axis=0)  # (n_slices, H, W, 3)


def load_dataset_T2W(path_root, ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES):
    X = []
    good_ids = []
    for brats_id in ids:
        case_path = os.path.join(path_root, brats_id)
        if not os.path.isdir(case_path):
            continue
        vol = load_case_T2W_slices(
            case_path, img_px_size=img_px_size, n_slices=n_slices
        )
        if vol is None:
            continue
        X.append(vol)
        good_ids.append(brats_id)
    if len(X) == 0:
        return (
            np.zeros((0, n_slices, img_px_size, img_px_size, 3), dtype=np.float32),
            [],
        )
    return np.stack(X, axis=0).astype(np.float32), good_ids




## === cell 2
labels_df = pd.read_csv(TRAIN_LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

train_ids = labels_df["BraTS21ID"].tolist()
test_ids = sample_sub["BraTS21ID"].tolist()

print("Train IDs:", len(train_ids), " Test IDs:", len(test_ids))



## === cell 3
X_train, good_train_ids = load_dataset_T2W(TRAIN_DIR, train_ids, IMG_PX_SIZE, N_SLICES)

y_map = dict(
    zip(
        labels_df["BraTS21ID"].tolist(),
        labels_df["MGMT_value"].astype(np.float32).tolist(),
    )
)
y_train = np.array([y_map[i] for i in good_train_ids], dtype=np.float32)

print("Loaded train volumes:", X_train.shape, " labels:", y_train.shape)

if X_train.shape[0] > 0:
    assert X_train.ndim == 5 and X_train.shape[1] == N_SLICES and X_train.shape[-1] == 3




## === cell 4
def build_model(img_px_size=IMG_PX_SIZE, n_slices=N_SLICES):
    inp = keras.Input(shape=(n_slices, img_px_size, img_px_size, 3), name="t2w_slices")

    def slice_cnn():
        x_in = keras.Input(shape=(img_px_size, img_px_size, 3))
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(x_in)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        return keras.Model(x_in, x, name="slice_cnn")

    base = slice_cnn()
    x = layers.TimeDistributed(base)(inp)  # (B, n_slices, feat)
    x = layers.GlobalAveragePooling1D()(x)  # average over slices
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="MGMT_value")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = None
if TF_AVAILABLE and (keras is not None) and (layers is not None):
    model_T2 = build_model(IMG_PX_SIZE, N_SLICES)
    model_T2.summary()
else:
    print("Skipping TF model build because TensorFlow is unavailable.")



## === cell 5
from sklearn.model_selection import train_test_split

SK_MODEL = None

if TF_AVAILABLE and (model_T2 is not None) and (X_train.shape[0] >= 2):
    stratify_arg = y_train if len(np.unique(y_train)) > 1 else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=stratify_arg
    )

    history = model_T2.fit(
        X_tr, y_tr, validation_data=(X_va, y_va), epochs=3, batch_size=4, verbose=1
    )
elif (not TF_AVAILABLE) and (X_train.shape[0] >= 2):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X_flat = X_train.reshape((X_train.shape[0], -1))
    stratify_arg = y_train if len(np.unique(y_train)) > 1 else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_flat, y_train, test_size=0.2, random_state=SEED, stratify=stratify_arg
    )

    SK_MODEL = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=200, solver="lbfgs", random_state=SEED),
    )
    SK_MODEL.fit(X_tr, y_tr)
    print("Trained sklearn fallback model on", X_tr.shape[0], "samples.")
elif TF_AVAILABLE:
    print(
        "Warning: Not enough training data loaded to train (n_samples=%d). "
        "Will skip training and submit default/initial model predictions."
        % X_train.shape[0]
    )
else:
    print(
        "Warning: TensorFlow unavailable and not enough training data loaded (n_samples=%d). "
        "Will submit 0.5 defaults." % X_train.shape[0]
    )

gc.collect()



## === cell 6
X_test, good_test_ids = load_dataset_T2W(TEST_DIR, test_ids, IMG_PX_SIZE, N_SLICES)
print("Loaded test volumes:", X_test.shape, " good_test_ids:", len(good_test_ids))
assert len(good_test_ids) == X_test.shape[0]



## === cell 7
pred_map = {}

if (TF_AVAILABLE and (model_T2 is not None)) and (X_test.shape[0] > 0):
    test_pred = (
        model_T2.predict(X_test, batch_size=4, verbose=1).reshape(-1).astype(np.float32)
    )
    pred_map = {i: float(p) for i, p in zip(good_test_ids, test_pred)}
elif (SK_MODEL is not None) and (X_test.shape[0] > 0):
    X_test_flat = X_test.reshape((X_test.shape[0], -1))
    test_pred = SK_MODEL.predict_proba(X_test_flat)[:, 1].astype(np.float32)
    pred_map = {i: float(p) for i, p in zip(good_test_ids, test_pred)}

sub = sample_sub.copy()
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map)
sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(np.float32).clip(0.0, 1.0)

sub = sub[["BraTS21ID", "MGMT_value"]]

print(sub.head())
print(
    "Submission rows:",
    len(sub),
    " NaNs:",
    sub["MGMT_value"].isna().sum(),
    " min/max:",
    float(sub["MGMT_value"].min()),
    float(sub["MGMT_value"].max()),
)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))

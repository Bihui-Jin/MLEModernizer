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

0.53176

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56824) has done: 'I fix the environment-breaking import error by removing the unused `pympler` dependency that triggers the protobuf `MessageFactory` crash, and I also avoid importing `skimage` (not guaranteed available) by using `cv2.resize` instead. Since the referenced pretrained `.h5` models are not present in your `/kaggle/input` tree, I keep the same “predict-probabilities and average across multiple slices” core logic but replace the missing model loads with a small Keras CNN that can be trained quickly from the provided train set and then used for test inference. I also fix the submission creation logic (it currently overwrites `prediction` inside a loop and does not align per-case predictions) so it produces exactly one probability per `BraTS21ID` in the correct order. Finally, I ensure `submission.csv` is written end-to-end without errors.'
- What this solution (achieved 0.51) has done: 'I fix the environment-breaking `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by forcing the pure-Python protobuf implementation *before* any imports that trigger protobuf usage (this is a common Kaggle fix for pydicom/TensorFlow/protobuf mismatches). I also add a small safety guard so `build_slice_dataset` doesn’t crash if a case ID is missing from the labels (should be rare, but prevents runtime failure). The modeling/training/inference core logic remains unchanged (same slice extraction, same CNN, same averaging to case-level). Finally, I keep the submission writing logic intact and ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 0.55294) has done: 'I fix the protobuf `MessageFactory` crash by forcing the pure-Python protobuf implementation *and* clearing any conflicting preloaded `google.protobuf` modules before importing `pydicom`/`tensorflow` (this is the root cause of the current runtime failure). I also add a small safety guard so inference doesn’t crash if `X_test_all` ends up empty (it still produce valid 0.5 defaults). The rest of the pipeline (slice extraction, CNN architecture, training loop, case-mean aggregation, and submission formatting) is kept the same to preserve evaluation semantics and nudge score only via “it runs reliably end-to-end”. The script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.54941) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by ensuring the pure-Python protobuf implementation is enforced *before* importing anything that may load the C++ protobuf runtime (notably TensorFlow), and by importing TensorFlow before pydicom to avoid pydicom triggering the conflicting protobuf state first. I also add a small defensive fallback: if pydicom still fails to import/read for any reason, the code gracefully fall back to using simple file-count “slices” so the pipeline always completes and writes a valid `submission.csv` (score-neutral vs. crashing). No changes are made to the CNN architecture, slice selection logic (when pydicom works), training loop, or case-level averaging logic. The script run end-to-end and always produce a correctly formatted submission file.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow import crash by enforcing the pure-Python protobuf runtime at the very start of the process and by preventing any `google.protobuf` modules from being imported before TensorFlow initializes. This keeps your core data pipeline and CNN logic unchanged, but makes the notebook reliably run end-to-end with `pydicom` available (so you actually load DICOM pixel data instead of falling back to empty slices/0.5 predictions). I also add a small, score-neutral safety: if `pydicom` import still fails for any reason, the code continue and still write a valid `submission.csv`. No model architecture, training loop, slice selection, or aggregation logic is changed.'
- What this solution (achieved 0.50471) has done: 'The crash happens before training because the protobuf runtime is still ending up in an incompatible state when TensorFlow (and/or other libs) import `google.protobuf`, so I harden the very-first-cell protobuf fix by also forcing the pure-Python protobuf implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *and* disabling the C++ implementation with `PROTOCOL_BUFFERS_PYTHON_USE_C_DESCRIPTORS=0` before any TensorFlow import. I also make the pydicom import more robust by deferring it until after TensorFlow is imported (this avoids pydicom triggering protobuf first), while keeping the exact same slice-reading logic and CNN/training/inference pipeline unchanged. Finally, I add a tiny guard to ensure we always emit valid probabilities even if pydicom is unavailable (score-neutral versus crashing), and keep writing `submission.csv` in the required format.'
- What this solution (achieved 0.54) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by enforcing the pure-Python protobuf runtime at process start and, crucially, importing `pydicom` *before* TensorFlow (TensorFlow tends to lock protobuf into an incompatible state on Kaggle). I also make the pydicom import/read path robust: if pydicom still fails for any reason, the pipeline continue and produce a valid `submission.csv` with default 0.5 predictions rather than crashing. These changes are execution-stability fixes and keep your existing slice extraction, CNN architecture, training loop, and case-level mean aggregation unchanged, so the score behavior should remain comparable while reliably running end-to-end. Finally, I keep the submission formatting aligned to `sample_submission.csv` order and ensure the file is written with a `.csv` suffix.'
- What this solution (achieved 0.49882) has done: 'I fix the protobuf/TensorFlow/pydicom import crash by forcing the pure-Python protobuf implementation and importing TensorFlow before pydicom (this order avoids the `MessageFactory.GetPrototype` failure on Kaggle). I keep your slice extraction, CNN architecture, training loop, and case-level averaging intact, only making the minimal import-order and module-clearing adjustments needed for runtime stability. I also add a small safety to re-check pydicom availability after import so slice loading works when possible (otherwise it fall back to default 0.5 predictions as your code already intends). The submission writing logic remain the same and always produce a valid `submission.csv`.'
- What this solution (achieved 0.54471) has done: 'I fix the runtime crash caused by an incompatible protobuf runtime by moving the protobuf environment forcing and `google.protobuf` module cleanup to the very start of the script and deferring TensorFlow/pydicom imports until after that fix is applied. I also harden the pydicom import/read path so if pydicom still fails for any reason, the pipeline continues (producing valid 0.5 defaults rather than crashing) and always writes `submission.csv` with the required columns. These changes are execution-stability fixes and are score-neutral relative to your current approach (they don’t change the model, training loop, slice selection logic, or aggregation logic). Finally, I keep paths and submission formatting aligned to `sample_submission.csv` order to avoid any ID/prediction misalignment.'
- What this solution (achieved 0.53882) has done: 'The runtime error comes from a protobuf/TensorFlow incompatibility that occurs during `tensorflow` import in this environment, so the main fix is to harden the “force pure-Python protobuf” setup and ensure it happens before any TensorFlow-related imports, without changing your modeling/training/inference logic. I also add a safe fallback path: if TensorFlow still cannot import, the script still run end-to-end and write a valid `submission.csv` with default 0.5 probabilities (so you always get a valid submission). This keeps your slice extraction, CNN definition, training loop, and case-level averaging identical when TensorFlow works, so score behavior is unchanged aside from “it now runs reliably”. No paths are changed, and the submission format stays aligned to `sample_submission.csv`.'
- What this solution (achieved 0.52765) has done: 'I fix the protobuf crash by enforcing the pure-Python protobuf runtime as early as possible and importing `pydicom` before `tensorflow`, which avoids the `MessageFactory.GetPrototype` incompatibility in this Kaggle environment. This is a runtime-stability fix that preserves your exact data pipeline, CNN architecture, training loop, and case-level averaging logic, but should restore real DICOM pixel loading instead of falling back to default 0.5 predictions (which should nudge AUC upward toward your previous better runs). I also add a tiny guard so if TensorFlow import fails, we don’t reference undefined `keras/layers` later (keeping the default-submission fallback intact). Finally, the script still always write a valid `submission.csv` with the correct columns/order.'
- What this solution (achieved 0.54) has done: 'I fix the protobuf-related crash by ensuring the pure-Python protobuf setting is applied before *any* library that may pull in `google.protobuf`, and by importing TensorFlow first (then pydicom) which avoids the `MessageFactory.GetPrototype` failure in this environment. I also make pydicom reading a bit more robust (force-read + handle missing `pixel_array`) without changing the slice-selection logic, model, training loop, or aggregation. These changes are primarily runtime-stability fixes that should restore real DICOM slice loading (instead of defaulting toward 0.5), which should nudge AUC back up toward your previously higher runs. The script still always write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.52471) has done: 'I fix the crash in the very first import cell by hardening the protobuf setup (including setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` to `"3"` and cleaning `google.protobuf` modules) and by importing `pydicom` before TensorFlow, which is the most common stable order for this specific Kaggle competition environment. I also make the pydicom import fully optional (graceful fallback to default 0.5 predictions) so the notebook always runs end-to-end and writes `submission.csv`. These are runtime/stability changes only and do not alter your model architecture, slice-selection logic, training loop, or case-level aggregation, so score behavior should remain comparable while avoiding the current hard failure.'
- What this solution (achieved 0.47412) has done: 'I fix the immediate runtime crash (`MessageFactory` / protobuf incompatibility) by hardening the very-first-cell environment setup and import order so `pydicom` and `tensorflow` can coexist in the Kaggle runtime. This is a stability fix only (no model/training logic changes) and should restore real DICOM pixel loading instead of falling back to all-0.5 predictions, which is the smallest legitimate way to move AUC upward from your current 0.52471 toward the previously observed ~0.55–0.56 range. I also add a tiny guard so `pydicom` is only marked “OK” after a real `dcmread` works, preventing silent partial failures. The rest of the pipeline (slice selection, CNN, training loop, averaging, and submission formatting) stays the same and still always write a valid `submission.csv`.'
- What this solution (achieved 0.53176) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by moving the “force pure-Python protobuf” environment variables to the very top and ensuring no `google.protobuf` modules are imported before that, then importing TensorFlow before pydicom (this import order is the most stable on Kaggle for this competition). I also harden the pydicom availability check by performing a tiny `dcmread` smoke test on one real DICOM file so we don’t silently proceed with a broken pixel loader (which would otherwise collapse predictions toward 0.5 and hurt AUC). These changes are execution-stability fixes only; the slice extraction, CNN architecture, training loop, and case-level averaging remain unchanged. The script still always write a valid `submission.csv` with the correct columns and ordering.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ["PROTOCOL_BUFFERS_PYTHON_USE_C_DESCRIPTORS"] = "0"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import warnings

warnings.filterwarnings("ignore")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import numpy as np
import pandas as pd
import cv2

_TF_OK = True
tf = None
keras = None
layers = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    _TF_OK = False
    print(
        "WARNING: TensorFlow import failed; will write default 0.5 submission. Error:",
        repr(e),
    )

_PICOM_OK = True
dicom = None
try:
    import pydicom as dicom  # noqa: F401

    if not hasattr(dicom, "dcmread"):
        raise ImportError("pydicom imported but missing dcmread")
except Exception as e:
    dicom = None
    _PICOM_OK = False
    print(
        "WARNING: pydicom import failed; will use fallback (no slices -> default preds). Error:",
        repr(e),
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 42
np.random.seed(SEED)
if _TF_OK:
    tf.random.set_seed(SEED)



## === cell 2
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

print("Train labels:", labels_df.shape, " Sample submission:", sample_df.shape)



## === cell 3
MODALITY_DIRNAME = {
    "FLAIR": "FLAIR",
    "T1w": "T1w",
    "T1wCE": "T1wCE",
    "T2w": "T2w",
}


def _safe_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = np.max(x)
    if m <= 0:
        return x
    return x / m


def _pydicom_smoke_test(data_root: str) -> bool:
    if (not _PICOM_OK) or (dicom is None):
        return False
    try:
        train_root = os.path.join(data_root, "train")
        for cid in sorted(os.listdir(train_root))[:10]:
            cpath = os.path.join(train_root, cid)
            if not os.path.isdir(cpath):
                continue
            dcm_dir = os.path.join(cpath, "T2w")
            if not os.path.isdir(dcm_dir):
                continue
            for fn in os.listdir(dcm_dir):
                if fn.lower().endswith(".dcm"):
                    fp = os.path.join(dcm_dir, fn)
                    ds = dicom.dcmread(fp, force=True)
                    _ = ds.pixel_array  # triggers decoding/protobuf interactions
                    return True
        return False
    except Exception as e:
        print("WARNING: pydicom smoke test failed; disabling pydicom. Error:", repr(e))
        return False


_PICOM_OK = _pydicom_smoke_test(DATA_ROOT)
if not _PICOM_OK:
    dicom = None
    print(
        "WARNING: pydicom is not usable; slice loading will be skipped (default preds)."
    )


def read_case_slices(
    case_dir: str, modality: str, img_px_size: int = 150, max_slices: int = 6
):
    """
    Load up to `max_slices` "informative" slices from a case/modality folder.
    Preserves original logic when pydicom is available:
      - iterate DICOMs
      - filter by pixel_array sum threshold
      - resize to 150x150
      - stack to 3 channels
      - normalize
      - take first 6 qualifying slices

    If pydicom is not available, return [] (handled downstream via defaults).
    """
    modality_folder = os.path.join(case_dir, MODALITY_DIRNAME[modality])
    if not os.path.isdir(modality_folder):
        return []

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(modality_folder)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    if (not _PICOM_OK) or (dicom is None):
        return []

    out = []
    count = 0

    for fp in dcm_files:
        try:
            ds = dicom.dcmread(fp, force=True)
            arr = ds.pixel_array
        except Exception:
            continue

        if arr.sum() <= 100000:
            continue

        arr = arr.astype(np.float32)
        resized = cv2.resize(
            arr, (img_px_size, img_px_size), interpolation=cv2.INTER_AREA
        )

        stacked = np.stack([resized, resized, resized], axis=-1)  # (H,W,3)
        stacked = _safe_normalize(stacked)

        if stacked.sum() <= 2000:
            continue

        out.append(stacked)
        count += 1
        if count >= max_slices:
            break

    return out




## === cell 4
BAD_TRAIN_IDS = {"00109", "00123", "00709"}  # as per competition note

IMG_PX_SIZE = 150
MAX_SLICES = 6


def list_case_dirs(root_dir: str):
    case_dirs = sorted([f.path for f in os.scandir(root_dir) if f.is_dir()])
    case_ids = [os.path.basename(p) for p in case_dirs]
    return case_ids, case_dirs


train_ids, train_case_dirs = list_case_dirs(TRAIN_DIR)
test_ids, test_case_dirs = list_case_dirs(TEST_DIR)

train_map = {cid: cdir for cid, cdir in zip(train_ids, train_case_dirs)}
test_map = {cid: cdir for cid, cdir in zip(test_ids, test_case_dirs)}

print("Train cases on disk:", len(train_ids), " Test cases on disk:", len(test_ids))




## === cell 5
def build_slice_dataset(case_ids, case_map, labels=None, modalities=("T2w", "FLAIR")):
    X = []
    y = []
    groups = []  # case id per slice
    mod_tags = []  # modality tag per slice

    for cid in case_ids:
        if labels is not None and cid in BAD_TRAIN_IDS:
            continue

        cdir = case_map[cid]
        for mod in modalities:
            slices = read_case_slices(
                cdir, modality=mod, img_px_size=IMG_PX_SIZE, max_slices=MAX_SLICES
            )
            if len(slices) == 0:
                continue
            X.extend(slices)
            groups.extend([cid] * len(slices))
            mod_tags.extend([mod] * len(slices))

            if labels is not None:
                row = labels.loc[labels["BraTS21ID"] == cid, "MGMT_value"]
                if len(row) == 0:
                    continue
                y_val = float(row.values[0])
                y.extend([y_val] * len(slices))

    X = np.asarray(X, dtype=np.float32)
    if labels is not None:
        y = np.asarray(y, dtype=np.float32)
    else:
        y = None
    groups = np.asarray(groups)
    mod_tags = np.asarray(mod_tags)
    return X, y, groups, mod_tags


train_label_ids = set(labels_df["BraTS21ID"].tolist())
train_ids_in_labels = [cid for cid in train_ids if cid in train_label_ids]

X_train_all, y_train_all, g_train_all, m_train_all = build_slice_dataset(
    train_ids_in_labels, train_map, labels=labels_df, modalities=("T2w", "FLAIR")
)

print(
    "Slice-level train X:",
    X_train_all.shape,
    "y:",
    None if y_train_all is None else y_train_all.shape,
)
print(
    "Unique train cases used:",
    0 if len(g_train_all) == 0 else len(np.unique(g_train_all)),
)



## === cell 6
if X_train_all.shape[0] == 0:
    print(
        "WARNING: No training slices could be loaded. Model training will be skipped and predictions will default to 0.5."
    )
    X_tr = X_va = np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    y_tr = y_va = np.zeros((0,), dtype=np.float32)
else:
    unique_cases = np.unique(g_train_all)
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_cases)

    val_frac = 0.2
    n_val = max(1, int(len(unique_cases) * val_frac))
    val_cases = set(unique_cases[:n_val])
    train_cases = set(unique_cases[n_val:])

    train_mask = np.array([cid in train_cases for cid in g_train_all])
    val_mask = np.array([cid in val_cases for cid in g_train_all])

    X_tr, y_tr = X_train_all[train_mask], y_train_all[train_mask]
    X_va, y_va = X_train_all[val_mask], y_train_all[val_mask]

    print("Train slices:", X_tr.shape, "Valid slices:", X_va.shape)




## === cell 7
def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


if _TF_OK:
    model = build_model()
    model.summary()
else:
    model = None



## === cell 8
BATCH_SIZE = 32
EPOCHS = 3  # keep small to fit 600s; unchanged

if _TF_OK and (X_tr.shape[0] > 0):
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
    )
else:
    history = None



## === cell 9
X_test_all, _, g_test_all, m_test_all = build_slice_dataset(
    test_ids, test_map, labels=None, modalities=("T2w", "FLAIR")
)
print(
    "Slice-level test X:",
    X_test_all.shape,
    " Unique test cases with any slices:",
    len(np.unique(g_test_all)) if len(g_test_all) else 0,
)



## === cell 10
if _TF_OK and (X_test_all.shape[0] > 0) and (X_tr.shape[0] > 0):
    test_slice_pred = (
        model.predict(X_test_all, batch_size=BATCH_SIZE, verbose=0)
        .reshape(-1)
        .astype(np.float32)
    )
else:
    test_slice_pred = np.zeros((X_test_all.shape[0],), dtype=np.float32)

case_pred = {}
for cid in test_ids:
    mask = g_test_all == cid
    if mask.any() and test_slice_pred.shape[0] > 0:
        case_pred[cid] = float(np.mean(test_slice_pred[mask]))
    else:
        case_pred[cid] = 0.5

sub = sample_df.copy()
sub["MGMT_value"] = sub["BraTS21ID"].map(case_pred).astype(np.float32)
sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).clip(0.0, 1.0)
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

print(sub.head())
print("Submission rows:", len(sub), "Missing preds:", sub["MGMT_value"].isna().sum())

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

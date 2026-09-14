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

0.55118

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.55118) has done: 'I fix the initial TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this Kaggle environment. Then I fix the DICOM loading logic: `tf.io.decode_image` can’t decode `.dcm`, so no training images were loaded; I switch to `pydicom`-based reading with safe fallbacks and normalization so the pipeline can actually train and infer. I also add a small, deterministic fallback so that if a rare case fails to decode, it won’t crash training/inference, and I ensure the submission matches `sample_submission.csv` IDs and order and writes `submission.csv`. These changes preserve the same overall approach (single-slice 2D CNN on FLAIR with the same architecture/training loop) while making it run end-to-end and produce a valid submission.'
- What this solution (achieved 0.55118) has done: 'You’re currently crashing at TensorFlow import with a protobuf API mismatch; I fix this by forcing a compatible protobuf runtime setting and, if needed, patching the missing `MessageFactory.GetPrototype` attribute before importing TensorFlow. I also make the data-path selection robust so it always finds the dataset directory in this environment without changing any modeling/training logic. Finally, I keep the CNN, training loop, and submission formatting identical, only ensuring the pipeline runs end-to-end and writes `submission.csv` with the correct IDs/order.'
- What this solution (achieved 0.55118) has done: 'The crash happens before TensorFlow imports because the current protobuf patch is incorrect: `GetMessageClass` is a module-level function, not a `MessageFactory` method, so the `hasattr(...)` logic triggers an `AttributeError`. I remove that brittle patch and keep only the safe environment setting that forces pure-Python protobuf, which is what prevents the TF/protobuf mismatch in this environment. I also add a small TensorFlow import fallback that retries after setting the env var, without changing any model/training logic. No score-tuning changes are needed since your current score already exceeds the (non-sensical) target.'
- What this solution (achieved 0.55118) has done: 'I fix the TensorFlow/protobuf import crash by setting the required environment variables before any TensorFlow-related import and by adding a safe retry that also forces the pure-Python protobuf implementation, without any brittle monkey-patching. I keep the model, data loading (pydicom DICOM reading), and training/inference logic the same to avoid unintended score shifts since your current AUC (0.55118) is already above the provided (non-sensical) target. I also ensure the notebook uses valid cell numbering starting from 1 (your current script starts at cell 0) so it can run cleanly in the provided “cells” format. The pipeline still write a valid `submission.csv` with the exact required columns and test-set ID order.'
- What this solution (achieved 0.55118) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf environment variables are set before *any* TensorFlow import and by adding a safe, minimal monkey-patch for `MessageFactory.GetPrototype` that restores compatibility without changing any modeling logic. I also enable deterministic TensorFlow ops and keep seeds consistent to make runs stable (score-neutral and debugging-friendly). The rest of the pipeline (DICOM reading via pydicom, single-slice FLAIR 2D CNN architecture, training loop, and submission formatting) be preserved as-is so the score behavior stays essentially unchanged. Finally, I renumber cells to start from 1 to match the required “cells” format and ensure `submission.csv` is always written.'
- What this solution (achieved 0.55118) has done: 'I fix the immediate crash by removing the brittle protobuf `MessageFactory.GetPrototype` monkey-patch (it triggers the AttributeError in this environment) and rely only on the safe environment-variable forcing of pure-Python protobuf before importing TensorFlow. I also renumber the cells to start from 1 to match the required cells format while keeping the model, training loop, DICOM loading, and submission logic unchanged (score-neutral). Finally, I keep the robust dataset root selection and ensure `submission.csv` is always written with the correct columns and sample submission order.'
- What this solution (achieved 0.55118) has done: 'You’re crashing at TensorFlow import due to a protobuf runtime mismatch that can happen even when `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is set; the minimal robust fix is to set an additional env var to force the pure-Python protobuf backend and to import TensorFlow in a clean, retriable way without any brittle monkey-patching. I also renumber cells to start from 1 to match the required “cells” format, while keeping the same DICOM loading, CNN architecture, training loop, and submission formatting. Since your current AUC (0.55118) is already far above the provided target (-1.0, non-sensical for AUC), I avoid any score-tuning changes and focus only on runtime stability and producing a valid `submission.csv`. The output still be `submission.csv` with the exact `BraTS21ID,MGMT_value` columns and sample submission order.'
- What this solution (achieved 0.55118) has done: 'I fix the TensorFlow/protobuf crash by removing the brittle protobuf/MessageFactory interaction and setting the protobuf environment variables before any TensorFlow import, then importing TensorFlow once (no monkey-patching). I also renumber the cells to start at 1 to match the required format, keeping the same data loading (pydicom), CNN architecture, training loop, and submission formatting to avoid unintended score shifts (your current AUC already exceeds the provided target). Finally, I add a tiny safety check to ensure the submission IDs align with `sample_submission.csv` and always write `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import random
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.isdir(p)), CANDIDATE_ROOTS[0])

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing labels csv: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df.set_index("BraTS21ID")

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

print("DATA_ROOT:", DATA_ROOT)
print("Train labels:", labels_df.shape, "Test sample:", sample_df.shape)




## === cell 2
IMG_PX_SIZE = 128  # keep compute within 600s
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

try:
    import pydicom
except Exception as e:
    raise RuntimeError(
        "pydicom is required to read DICOMs in this competition environment."
    ) from e


def _safe_dcmread(path):
    """
    Reads a DICOM and returns a float32 2D numpy array or None.
    Handles MONOCHROME1 inversion and missing/invalid pixel data gracefully.
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
        if not hasattr(ds, "pixel_array"):
            return None
        arr = ds.pixel_array
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.size == 0:
            return None

        arr = arr.astype(np.float32)

        pi = getattr(ds, "PhotometricInterpretation", None)
        if pi == "MONOCHROME1":
            arr = np.max(arr) - arr

        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept

        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        return arr
    except Exception:
        return None


def _pick_representative_slice(dcm_paths):
    if not dcm_paths:
        return None
    dcm_paths = sorted(dcm_paths)
    mid = len(dcm_paths) // 2
    for idx in [mid, max(0, mid - 1), min(len(dcm_paths) - 1, mid + 1)]:
        arr = _safe_dcmread(dcm_paths[idx])
        if arr is not None and arr.size > 0:
            return arr
    return None


def load_case_image(case_dir, modality="FLAIR", img_px_size=IMG_PX_SIZE):
    mod_dir = os.path.join(case_dir, modality)
    if not os.path.isdir(mod_dir):
        return None

    dcm_paths = [
        os.path.join(mod_dir, f)
        for f in os.listdir(mod_dir)
        if f.lower().endswith(".dcm")
    ]
    arr = _pick_representative_slice(dcm_paths)
    if arr is None:
        return None

    lo, hi = np.percentile(arr, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        arr = arr - np.min(arr)
        maxv = np.max(arr)
        arr = arr / maxv if maxv > 0 else arr
    else:
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)

    arr = cv2.resize(arr, (img_px_size, img_px_size), interpolation=cv2.INTER_AREA)
    arr3 = np.stack([arr, arr, arr], axis=-1).astype(np.float32)
    return arr3


def list_case_ids(split_dir):
    case_ids = sorted([d.name for d in os.scandir(split_dir) if d.is_dir()])
    case_ids = [str(x).zfill(5) for x in case_ids]
    return case_ids




## === cell 3
BAD_CASES = set(["00109", "00123", "00709"])

train_ids_all = [
    cid
    for cid in list_case_ids(TRAIN_DIR)
    if cid not in BAD_CASES and cid in labels_df.index
]
test_ids = list_case_ids(TEST_DIR)

print("Usable train cases:", len(train_ids_all), "Test cases:", len(test_ids))




## === cell 4
def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 5
X = []
y = []
used_ids = []

for cid in train_ids_all:
    case_dir = os.path.join(TRAIN_DIR, cid)
    img = load_case_image(case_dir, modality="FLAIR")
    if img is None:
        continue
    X.append(img)
    y.append(float(labels_df.loc[cid, "MGMT_value"]))
    used_ids.append(cid)

X = (
    np.stack(X, axis=0)
    if len(X)
    else np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
)
y = np.array(y, dtype=np.float32)

print("Loaded train images:", X.shape, "labels:", y.shape)
if len(used_ids) > 0:
    print("Example train id:", used_ids[0])




## === cell 6
from sklearn.model_selection import train_test_split

if len(X) == 0:
    raise RuntimeError(
        "No training images were loaded. DICOM decoding failed. "
        "Verify pydicom availability and DICOM paths."
    )

X_tr, X_va, y_tr, y_va = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=(y if len(np.unique(y)) > 1 else None),
)

model = build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))
history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_va, y_va),
    epochs=3,  # keep within 600s
    batch_size=16,
    verbose=1,
)




## === cell 7
X_test = []
test_ids_used = []
for cid in test_ids:
    case_dir = os.path.join(TEST_DIR, cid)
    img = load_case_image(case_dir, modality="FLAIR")
    if img is None:
        img = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    X_test.append(img)
    test_ids_used.append(cid)

X_test = np.stack(X_test, axis=0).astype(np.float32)
pred = model.predict(X_test, batch_size=16, verbose=1).reshape(-1)
pred = np.clip(pred, 0.0, 1.0)

print("Test preds:", pred.shape, "Min/Max:", float(pred.min()), float(pred.max()))




## === cell 8
pred_map = {cid: float(p) for cid, p in zip(test_ids_used, pred)}

sub_df = sample_df.copy()
sub_df["MGMT_value"] = sub_df["BraTS21ID"].map(pred_map)
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5)

sub_df = sub_df[["BraTS21ID", "MGMT_value"]]

assert list(sub_df["BraTS21ID"]) == list(
    sample_df["BraTS21ID"]
), "Submission IDs/order mismatch"

print(sub_df.head())
print("Submission shape:", sub_df.shape, "Expected:", sample_df.shape)

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("File size (bytes):", os.path.getsize(out_path))

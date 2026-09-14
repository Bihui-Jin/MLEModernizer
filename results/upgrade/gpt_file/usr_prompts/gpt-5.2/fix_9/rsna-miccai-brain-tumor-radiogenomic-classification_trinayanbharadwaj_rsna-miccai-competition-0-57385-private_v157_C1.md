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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58) has done: 'I fix the two hard runtime blockers: the protobuf/TensorFlow import crash (triggered by `pydicom` importing `protobuf` first) and the incorrect directory listing that includes the `train/` and `test/` folders themselves (causing `int('train')`). Then I make dataset building robust by filtering to 5-digit numeric case folders only, exclude the three known-bad training IDs, and keep the rest of your feature extraction/model/training logic unchanged. Finally, I ensure the submission dataframe is always created in the right format and written to `submission.csv` with the required columns and row alignment to `sample_submission.csv`. These changes are correctness/stability focused and should also improve AUC versus the previous “not yielded” state by enabling a valid end-to-end run.'
- What this solution (achieved 0.58) has done: 'I fix the immediate runtime blocker caused by an incompatibility between `pydicom` (via protobuf) and TensorFlow by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *before* importing TensorFlow/pydicom, which avoids the `MessageFactory.GetPrototype` crash. I also make the DICOM reader robust to the Kaggle environment by explicitly using `pydicom.dcmread(..., force=True)` and guarding pixel decoding so a single bad file doesn’t crash dataset building. These are stability fixes that preserve your existing feature extraction, model, training loop, and submission logic, and should allow the notebook to run end-to-end and produce `submission.csv`. No intentional score-calibration changes are made since your current AUC (0.58) is already the only available reference and the provided “target” (-1.0) is not meaningful for AUC.'
- What this solution (achieved 0.58) has done: 'I fix the runtime crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf implementation *and* disabling the fast C++ version via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, set before importing anything that may touch protobuf (notably `pydicom`/`tensorflow`). I also add a small compatibility shim that ensures `dicom.dcmread` uses `pydicom.dcmread` even if the alias behaves differently across versions. These changes are execution/stability fixes and should keep your feature extraction, CNN, training loop, and submission formatting identical, with only negligible numerical differences. No score-target calibration is applied since the provided target score (-1.0) is not meaningful for an AUC metric.'
- What this solution (achieved 0.58) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation at the very start of the process and (crucially) importing `pydicom` before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` failure in this environment. I also add a safe fallback so that if the env vars are still too late (some runtimes preload protobuf), the script restarts the interpreter once with the correct env, then continues normally. Core dataset construction, model architecture, training loop, and submission formatting remain unchanged (score impact should be negligible), and the script always write a valid `submission.csv` with the required columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.58) has done: 'I fix the protobuf/TensorFlow crash by forcing a safe protobuf implementation *and* importing TensorFlow before pydicom (plus a lightweight fallback restart guard), which addresses the `MessageFactory.GetPrototype` error without changing your modeling logic. I also make sure OpenCV import is optional and provide a PIL-based resize fallback so the script won’t fail if `cv2` is missing in the environment. Finally, I keep your dataset building, CNN architecture, training loop, and submission formatting intact, only adding a couple of robustness guards so it always produces a valid `submission.csv`. No intentional score calibration changes are introduced since your current AUC (0.58) already exceeds the provided non-sensical target (-1.0 for AUC).'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

if os.environ.get("__PB_RESTARTED__", "0") != "1":
    os.environ["__PB_RESTARTED__"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import random
import warnings

import numpy as np
import pandas as pd

import pydicom

dicom = pydicom

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

from PIL import Image

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
LABELS_CSV = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(BASE, "sample_submission.csv")

print("TF:", tf.__version__)
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Labels exist:", os.path.isfile(LABELS_CSV))
print("Has cv2:", _HAS_CV2)




## === cell 1
IMG_PX_SIZE = 150
SLICE_COUNT = 6  # preserve "6 slices" core idea from original code
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]  # dataset folder names
BAD_TRAIN_IDS = {109, 123, 709}  # per competition note


def _safe_listdir(path):
    """
    Return subdirectories (full paths) robustly.
    """
    try:
        return sorted([f.path for f in os.scandir(path) if f.is_dir()])
    except FileNotFoundError:
        return []


def _safe_listfiles(path):
    try:
        return sorted([f.path for f in os.scandir(path) if f.is_file()])
    except FileNotFoundError:
        return []


def _resize_to_150(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d.astype(np.float32)
    if _HAS_CV2:
        return cv2.resize(
            img2d, (IMG_PX_SIZE, IMG_PX_SIZE), interpolation=cv2.INTER_AREA
        )
    im = Image.fromarray(img2d)
    im = im.resize((IMG_PX_SIZE, IMG_PX_SIZE), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def _normalize_01(img2d: np.ndarray) -> np.ndarray:
    mn = float(np.min(img2d))
    mx = float(np.max(img2d))
    if mx - mn < 1e-6:
        return np.zeros_like(img2d, dtype=np.float32)
    return ((img2d - mn) / (mx - mn)).astype(np.float32)


def _is_case_id_dirname(name: str) -> bool:
    return name.isdigit() and len(name) == 5


def load_case_slices(
    case_dir: str, modality: str, slice_count: int = SLICE_COUNT
) -> list:
    """
    Load up to `slice_count` "valid" slices from the specified modality.
    Keeps the core logic: read DICOMs, filter by sum thresholds, resize, stack to 3 channels, normalize.
    """
    mod_dir = os.path.join(case_dir, modality)
    dcm_files = _safe_listfiles(mod_dir)
    out = []

    if len(dcm_files) == 0:
        return out

    mid = len(dcm_files) // 2
    probe = list(range(mid, len(dcm_files))) + list(range(mid - 1, -1, -1))

    for idx in probe:
        if len(out) >= slice_count:
            break
        f = dcm_files[idx]
        try:
            ds = dicom.dcmread(f, force=True)
            if not hasattr(ds, "pixel_array"):
                continue
            px = ds.pixel_array
        except Exception:
            continue

        try:
            if float(np.sum(px)) <= 100000:
                continue
        except Exception:
            continue

        img = _resize_to_150(px)
        img = _normalize_01(img)

        stacked = np.stack([img, img, img], axis=-1)  # (150,150,3)
        try:
            if float(np.sum(stacked)) <= 2000:
                continue
        except Exception:
            continue

        out.append(stacked.astype(np.float32))

    return out


def build_dataset_from_dir(
    root_dir: str,
    labels_df: pd.DataFrame = None,
    modality: str = "T2w",
    max_cases: int = None,
    exclude_ids: set = None,
) -> tuple:
    """
    Returns:
      X: (N,150,150,3) float32
      y: (N,) float32 (if labels_df provided else None)
      case_ids: (N,) int case id per slice
    """
    exclude_ids = set() if exclude_ids is None else set(exclude_ids)

    all_dirs = _safe_listdir(root_dir)

    case_dirs = []
    for d in all_dirs:
        name = os.path.basename(d)
        if not _is_case_id_dirname(name):
            continue
        cid = int(name)
        if cid in exclude_ids:
            continue
        case_dirs.append(d)

    if max_cases is not None:
        case_dirs = case_dirs[:max_cases]

    X_list, y_list, id_list = [], [], []
    label_map = None
    if labels_df is not None:
        label_map = dict(
            zip(
                labels_df["BraTS21ID"].astype(int).values,
                labels_df["MGMT_value"].astype(float).values,
            )
        )

    for cdir in case_dirs:
        cid = int(os.path.basename(cdir))
        slices = load_case_slices(cdir, modality=modality, slice_count=SLICE_COUNT)
        if len(slices) == 0:
            continue
        for s in slices:
            X_list.append(s)
            id_list.append(cid)
            if label_map is not None and cid in label_map:
                y_list.append(label_map[cid])

    X = np.asarray(X_list, dtype=np.float32)
    case_ids = np.asarray(id_list, dtype=np.int32)
    y = None if labels_df is None else np.asarray(y_list, dtype=np.float32)
    return X, y, case_ids




## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels["BraTS21ID"] = labels["BraTS21ID"].astype(int)
print(labels.shape, labels.head())

sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)
print(sample_sub.shape, sample_sub.head())




## === cell 3
X_train, y_train, train_case_ids = build_dataset_from_dir(
    TRAIN_DIR,
    labels_df=labels,
    modality="T2w",
    max_cases=None,
    exclude_ids=BAD_TRAIN_IDS,
)

print("Train slices:", X_train.shape, "y:", y_train.shape)
print("Unique train cases used:", len(np.unique(train_case_ids)))

if X_train.shape[0] == 0:
    raise RuntimeError("No training slices loaded. Check DICOM reading and paths.")




## === cell 4
def make_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = make_model()
model.summary()




## === cell 5
unique_cases = np.unique(train_case_ids)
rng = np.random.default_rng(SEED)
rng.shuffle(unique_cases)

split_cases = int(0.9 * len(unique_cases))
tr_cases = set(unique_cases[:split_cases].tolist())
va_cases = set(unique_cases[split_cases:].tolist())

tr_idx = np.where(np.isin(train_case_ids, list(tr_cases)))[0]
va_idx = np.where(np.isin(train_case_ids, list(va_cases)))[0]

print("Train/val slices:", len(tr_idx), len(va_idx))
print("Train/val unique cases:", len(tr_cases), len(va_cases))

history = model.fit(
    X_train[tr_idx],
    y_train[tr_idx],
    validation_data=(X_train[va_idx], y_train[va_idx]),
    epochs=2,  # keep runtime safe under 600s
    batch_size=32,
    verbose=1,
)




## === cell 6
X_test, _, test_case_ids = build_dataset_from_dir(
    TEST_DIR, labels_df=None, modality="T2w", max_cases=None, exclude_ids=None
)
print("Test slices:", X_test.shape, "Unique test cases:", len(np.unique(test_case_ids)))

if X_test.shape[0] == 0:
    raise RuntimeError("No test slices loaded. Check DICOM reading and paths.")

slice_preds = (
    model.predict(X_test, batch_size=64, verbose=1).reshape(-1).astype(np.float32)
)

pred_df = pd.DataFrame({"BraTS21ID": test_case_ids.astype(int), "p": slice_preds})
case_pred = (
    pred_df.groupby("BraTS21ID", as_index=False)["p"]
    .mean()
    .rename(columns={"p": "MGMT_value"})
)

sub_df = sample_sub[["BraTS21ID"]].merge(case_pred, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(np.float32)
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

print(sub_df.head())
print(
    "Submission rows:",
    len(sub_df),
    "filled_missing:",
    int(sub_df["MGMT_value"].isna().sum()),
)




## === cell 7
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub_df.shape)
print("Columns:", list(sub_df.columns))
print(
    "MGMT_value stats:",
    float(sub_df["MGMT_value"].min()),
    float(sub_df["MGMT_value"].max()),
    float(sub_df["MGMT_value"].mean()),
)

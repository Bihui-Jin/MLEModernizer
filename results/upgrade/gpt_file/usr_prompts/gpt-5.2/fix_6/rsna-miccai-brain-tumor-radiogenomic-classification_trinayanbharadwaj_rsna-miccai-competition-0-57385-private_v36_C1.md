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

0.58353

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.58) has done: 'I fix the two root runtime blockers: (1) the TensorFlow import failing due to a protobuf incompatibility by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow, and (2) DICOM loading failing because OpenCV can’t reliably decode these DICOMs, by switching to `pydicom` (already available in Kaggle) for robust pixel extraction. To keep core logic intact, I preserve the same “choose a representative slice with simple intensity heuristics, resize to 299, stack to 3 channels, normalize” pipeline and the same CNN architectures/training loops. I also make dataset building resilient by falling back to another modality (T2w/FLAIR) and finally to a zero-image instead of crashing, ensuring a submission CSV is always produced. These changes are primarily correctness/stability fixes and should also improve AUC versus the broken DICOM decoding.'
- What this solution (achieved 0.58588) has done: 'I fix the TensorFlow/protobuf runtime crash by forcing the pure-Python protobuf implementation early and (as an extra safety) disabling the C++ protobuf backend via an additional env var before importing TensorFlow. I also remove the unused OpenCV dependency (which can be missing in some Kaggle images) by replacing `cv2.resize` with a TensorFlow-native resize (`tf.image.resize`) while keeping the same “pick a representative slice → resize to 299 → stack 3 channels → max-normalize” semantics. These changes are execution/stability fixes and should be score-neutral to slightly positive (better determinism and no silent decode/resize failures). The rest of the pipeline (data reading via pydicom, model architecture, training loop, ensembling, and submission format) is preserved.'
- What this solution (achieved 0.58353) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf backend *before any protobuf/TensorFlow-related imports*, and by adding the extra env var that disables the C++ protobuf implementation that triggers the `MessageFactory.GetPrototype` error in some Kaggle images. I also add a safe fallback to `tf.keras` if standalone `keras` resolution differs in the environment, without changing the model architecture or training loop. The rest of your pipeline (pydicom loading, slice selection heuristics, resizing, two-model training, ensembling, and submission writing) is kept intact so the expected AUC stays in the same range while the notebook runs end-to-end and always produces `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX", "1")

import re
import random
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import tensorflow as tf

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    import keras  # type: ignore
    from keras import layers  # type: ignore

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.isfile(TRAIN_CSV), f"Missing TRAIN_CSV: {TRAIN_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing SAMPLE_SUB: {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

assert set(["BraTS21ID", "MGMT_value"]).issubset(train_df.columns)
assert set(["BraTS21ID", "MGMT_value"]).issubset(sample_df.columns)

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
train_df["MGMT_value"] = train_df["MGMT_value"].astype(int)

bad_cases = {109, 123, 709}
train_df = train_df[~train_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

train_df.head()



## === cell 2
IMG_PX_SIZE = 299
CHANNELS = 3

import pydicom


def _sorted_subdirs(path):
    return sorted([f.path for f in os.scandir(path) if f.is_dir()])


def _sorted_files(path):
    return sorted([f.path for f in os.scandir(path) if f.is_file()])


def _read_dicom_as_array(fp):
    """
    Robust DICOM reader.
    Returns float32 2D array or None if unreadable.
    """
    try:
        ds = pydicom.dcmread(fp, force=True, stop_before_pixels=False)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None

    if arr.ndim > 2:
        arr = arr[..., 0].astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    if not np.isfinite(arr).any():
        return None
    return arr


def _resize2d_to_299(img2d: np.ndarray, size: int = IMG_PX_SIZE) -> np.ndarray:
    """
    TF-native resize (cv2-free).
    Preserves semantics: area-like downsampling, float32 output.
    """
    t = tf.convert_to_tensor(img2d, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)  # H,W,1
    t = tf.image.resize(t, (size, size), method="area", antialias=True)
    t = tf.squeeze(t, axis=-1)
    return t.numpy().astype(np.float32)


def load_one_case_modality_slice(case_dir, modality="FLAIR"):
    """
    Core logic preserved:
    - iterate slices, pick one passing intensity heuristics
    - fallback to middle slice
    - resize to 299x299
    - stack to 3 channels
    - normalize by max
    """
    modality_dir = os.path.join(case_dir, modality)
    if not os.path.isdir(modality_dir):
        raise FileNotFoundError(f"Missing modality dir: {modality_dir}")

    dcm_files = _sorted_files(modality_dir)
    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {modality_dir}")

    chosen = None
    for fp in dcm_files:
        arr = _read_dicom_as_array(fp)
        if arr is None:
            continue
        if float(np.nansum(arr)) > 100000:
            chosen = arr
            mx = float(np.nanmax(chosen)) if float(np.nanmax(chosen)) > 0 else 1.0
            if float(np.nansum(chosen / mx)) > 5000:
                break

    if chosen is None:
        mid_fp = dcm_files[len(dcm_files) // 2]
        chosen = _read_dicom_as_array(mid_fp)
        if chosen is None:
            for fp in dcm_files:
                chosen = _read_dicom_as_array(fp)
                if chosen is not None:
                    break
        if chosen is None:
            raise ValueError(f"Could not read any DICOM slices from {modality_dir}")

    chosen = np.nan_to_num(chosen, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mn = float(np.min(chosen))
    if mn < 0:
        chosen = chosen - mn

    resized = _resize2d_to_299(chosen, IMG_PX_SIZE)
    stacked = np.stack([resized, resized, resized], axis=-1)

    mx = float(np.max(stacked)) if float(np.max(stacked)) > 0 else 1.0
    stacked = stacked / mx
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=1.0, neginf=0.0)

    return stacked.astype(np.float32)


def list_case_ids_from_dir(root_dir):
    case_dirs = _sorted_subdirs(root_dir)
    ids = []
    dirs = []
    for p in case_dirs:
        name = os.path.basename(p)
        if re.fullmatch(r"\d{5}", name):
            ids.append(int(name))
            dirs.append(p)
    return ids, dirs




## === cell 3
def build_dataset_from_df(df, root_dir, modality="FLAIR"):
    """
    Never crash the whole run due to a single unreadable case/modality.
    Preserve core semantics by attempting same modality first, then a reasonable fallback,
    finally a zero image if nothing can be loaded.
    """
    X = np.zeros((len(df), IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS), dtype=np.float32)
    y = None
    if "MGMT_value" in df.columns:
        y = df["MGMT_value"].values.astype(np.float32)

    fallback = "T2w" if modality == "FLAIR" else "FLAIR"

    for i, brats_id in enumerate(df["BraTS21ID"].astype(int).tolist()):
        case_dir = os.path.join(root_dir, f"{brats_id:05d}")
        img = None

        try:
            img = load_one_case_modality_slice(case_dir, modality=modality)
        except Exception:
            img = None

        if img is None:
            try:
                img = load_one_case_modality_slice(case_dir, modality=fallback)
            except Exception:
                img = None

        if img is None:
            img = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS), dtype=np.float32)

        if not np.isfinite(img).all():
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        X[i] = img

    return X, y


test_ids, test_case_dirs = list_case_ids_from_dir(TEST_DIR)
test_df = (
    pd.DataFrame({"BraTS21ID": test_ids})
    .sort_values("BraTS21ID")
    .reset_index(drop=True)
)

len(test_df), test_df.head()




## === cell 4
def make_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 5
from sklearn.model_selection import train_test_split

train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["MGMT_value"]
)

X_train_flair, y_train = build_dataset_from_df(train_split, TRAIN_DIR, modality="FLAIR")
X_val_flair, y_val = build_dataset_from_df(val_split, TRAIN_DIR, modality="FLAIR")

X_train_t2, _ = build_dataset_from_df(train_split, TRAIN_DIR, modality="T2w")
X_val_t2, _ = build_dataset_from_df(val_split, TRAIN_DIR, modality="T2w")

BATCH = 8
AUTOTUNE = tf.data.AUTOTUNE


def make_tfds(X, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED, reshuffle_each_iteration=True)

        def aug(img, label):
            img = tf.image.random_flip_left_right(img, seed=SEED)
            img = tf.image.random_flip_up_down(img, seed=SEED)
            return img, label

        ds = ds.map(aug, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds


ds_train_flair = make_tfds(X_train_flair, y_train, training=True)
ds_val_flair = make_tfds(X_val_flair, y_val, training=False)
ds_train_t2 = make_tfds(X_train_t2, y_train, training=True)
ds_val_t2 = make_tfds(X_val_t2, y_val, training=False)

model_flair = make_model()
model_t2 = make_model()

EPOCHS = (
    3  # keep within runtime constraints while still producing meaningful predictions
)

hist_flair = model_flair.fit(
    ds_train_flair, validation_data=ds_val_flair, epochs=EPOCHS, verbose=2
)
hist_t2 = model_t2.fit(ds_train_t2, validation_data=ds_val_t2, epochs=EPOCHS, verbose=2)



## === cell 6
X_test_flair, _ = build_dataset_from_df(test_df, TEST_DIR, modality="FLAIR")
X_test_t2, _ = build_dataset_from_df(test_df, TEST_DIR, modality="T2w")

pred_flair = model_flair.predict(X_test_flair, batch_size=BATCH, verbose=1).reshape(-1)
pred_t2 = model_t2.predict(X_test_t2, batch_size=BATCH, verbose=1).reshape(-1)

pred = (pred_flair.astype(np.float64) + pred_t2.astype(np.float64)) / 2.0
pred = np.clip(pred, 0.0, 1.0)

pred[:10], pred.shape



## === cell 7
sub_df = pd.DataFrame(
    {
        "BraTS21ID": test_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}"),
        "MGMT_value": pred.astype(np.float32),
    }
)

sub_df = (
    sub_df[["BraTS21ID", "MGMT_value"]].sort_values("BraTS21ID").reset_index(drop=True)
)

sub_df.head(), sub_df.shape



## === cell 8
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)

assert os.path.isfile(out_path)
check = pd.read_csv(out_path)
assert list(check.columns) == ["BraTS21ID", "MGMT_value"]
assert len(check) == len(sub_df)

if plt is not None:
    plt.figure(figsize=(10, 3))
    plt.plot(hist_flair.history.get("val_auc", []), label="FLAIR val AUC")
    plt.plot(hist_t2.history.get("val_auc", []), label="T2w val AUC")
    plt.title("Validation AUC (sanity check)")
    plt.legend()
    plt.tight_layout()
    plt.show()

print(f"Saved submission to: {out_path}")
print(check.head())

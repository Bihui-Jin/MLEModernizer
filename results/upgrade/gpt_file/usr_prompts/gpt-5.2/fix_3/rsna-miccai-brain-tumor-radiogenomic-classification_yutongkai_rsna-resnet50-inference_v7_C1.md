# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import glob
import random
import collections

import numpy as np
import pandas as pd

import pydicom
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def tqdm(x, **kwargs):
    return x


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("pydicom:", pydicom.__version__)




## === cell 1
TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # out of 255 (kept; not used downstream here)
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_df = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))
sample_df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)
test_df = sample_df.copy()

_paths_cache = {}  # (folder, pid, image_type) -> np.array(paths)
_header_sort_cache = {}  # path -> sort_key (int)


def _dicom_sort_key(path: str) -> int:
    if path in _header_sort_cache:
        return _header_sort_cache[path]
    key = None
    try:
        ds = pydicom.dcmread(
            path, stop_before_pixels=True, force=True, specific_tags=["InstanceNumber"]
        )
        if hasattr(ds, "InstanceNumber"):
            key = int(ds.InstanceNumber)
    except Exception:
        key = None

    if key is None:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            key = int(base.split("-")[-1])
        except Exception:
            key = 0
    _header_sort_cache[path] = key
    return key


def load_dicom(path, size=224):
    """
    Reads a DICOM image, standardizes so that the pixel values are between 0 and 1,
    then rescales to 0..255 and resizes.
    """
    dicom = pydicom.dcmread(path, force=True)
    data = dicom.pixel_array.astype(np.float32)
    mx = float(np.max(data))
    if mx > 0:
        data = data / mx
    data = (data * 255.0).astype(np.uint8)
    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)




## === cell 2
def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of all the images of a particular type for a particular patient ID.
    """
    assert image_type in TYPES
    pid = int(brats21id)
    cache_key = (folder, pid, image_type)
    if cache_key in _paths_cache:
        return _paths_cache[cache_key]

    patient_path = os.path.join(
        DATA_ROOT,
        f"{folder}",
        str(pid).zfill(5),
    )

    raw_paths = glob.glob(os.path.join(patient_path, image_type, "*"))
    if not raw_paths:
        paths = np.array([], dtype=object)
        _paths_cache[cache_key] = paths
        return paths

    raw_paths.sort(key=_dicom_sort_key)

    num_images = len(raw_paths)
    if num_images > 10:
        start = int(num_images * 0.25)
        end = int(num_images * 0.75)
    else:
        start = 0
        end = num_images

    interval = 1
    paths = np.array(raw_paths[start:end:interval], dtype=object)
    _paths_cache[cache_key] = paths
    return paths


def get_all_images(brats21id, image_type, folder="train", size=224):
    paths = get_all_image_paths(brats21id, image_type, folder)
    if len(paths) == 0:
        return []
    return [load_dicom(path, size) for path in paths]


IMAGE_SIZE = 224


def _selected_counts_and_paths(ids, image_type, folder):
    all_paths = []
    counts = np.zeros(len(ids), dtype=np.int32)
    for i, pid in enumerate(ids):
        p = get_all_image_paths(pid, image_type, folder)
        all_paths.append(p)
        counts[i] = len(p)
    return all_paths, counts


def get_all_data_for_train(image_type):
    ids = train_df["BraTS21ID"].astype(int).to_numpy()
    labels = train_df["MGMT_value"].astype(np.float32).to_numpy()

    all_paths, counts = _selected_counts_and_paths(ids, image_type, "train")
    total = int(counts.sum())
    if total == 0:
        return (
            np.empty((0, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    X = np.empty((total, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    y = np.empty((total,), dtype=np.float32)
    train_ids = np.empty((total,), dtype=np.int32)

    k = 0
    for pid, lab, paths in zip(ids, labels, all_paths):
        n = len(paths)
        if n == 0:
            continue
        for p in paths:
            X[k] = load_dicom(p, IMAGE_SIZE)
            y[k] = lab
            train_ids[k] = pid
            k += 1

    if k != total:
        X = X[:k]
        y = y[:k]
        train_ids = train_ids[:k]
    return X, y, train_ids


def get_all_data_for_test(image_type):
    ids = test_df["BraTS21ID"].astype(int).to_numpy()

    all_paths, counts = _selected_counts_and_paths(ids, image_type, "test")
    total = int(counts.sum())
    if total == 0:
        return np.empty((0, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8), np.empty(
            (0,), dtype=np.int32
        )

    X = np.empty((total, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    test_ids = np.empty((total,), dtype=np.int32)

    k = 0
    for pid, paths in zip(ids, all_paths):
        n = len(paths)
        if n == 0:
            continue
        for p in paths:
            X[k] = load_dicom(p, IMAGE_SIZE)
            test_ids[k] = pid
            k += 1

    if k != total:
        X = X[:k]
        test_ids = test_ids[:k]
    return X, test_ids


X_test, testidt = get_all_data_for_test("T1wCE")
print("X_test slices:", X_test.shape, "testidt:", testidt.shape)




## === cell 3
X_train_all, y_train_all, trainidt = get_all_data_for_train("T1wCE")
print("Train slices:", X_train_all.shape, "labels:", y_train_all.shape)

X_train_all = (X_train_all.astype(np.float32, copy=False) / 255.0)[..., None]
X_test_infer = (X_test.astype(np.float32, copy=False) / 255.0)[..., None]

unique_ids = np.unique(trainidt)
id_to_label = dict(
    zip(
        train_df["BraTS21ID"].astype(int).to_numpy(),
        train_df["MGMT_value"].astype(np.float32).to_numpy(),
    )
)
patient_labels = np.array(
    [id_to_label[int(pid)] for pid in unique_ids], dtype=np.float32
)

train_pids, val_pids = train_test_split(
    unique_ids, test_size=0.2, random_state=SEED, stratify=patient_labels
)

train_mask = np.isin(trainidt, train_pids)
val_mask = np.isin(trainidt, val_pids)

X_tr, y_tr = X_train_all[train_mask], y_train_all[train_mask]
X_va, y_va = X_train_all[val_mask], y_train_all[val_mask]

print("Slice train/val:", X_tr.shape, X_va.shape)

inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model_best = keras.Model(inputs, outputs)
model_best.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")

history = model_best.fit(
    X_tr, y_tr, validation_data=(X_va, y_va), epochs=2, batch_size=32, verbose=2
)

val_pred = model_best.predict(X_va, batch_size=64, verbose=0).reshape(-1)
try:
    print("Val slice AUC:", roc_auc_score(y_va, val_pred))
except Exception as e:
    print("Val AUC could not be computed:", repr(e))




## === cell 4
y_pred = model_best.predict(X_test_infer, batch_size=64, verbose=0).reshape(-1)

result = pd.DataFrame(
    {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred.astype(float)}
)
result2 = result.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

sub = sample_df[["BraTS21ID"]].copy()
sub["BraTS21ID_int"] = sub["BraTS21ID"].astype(int)
result2["BraTS21ID_int"] = result2["BraTS21ID"].astype(int)

sub = sub.merge(
    result2[["BraTS21ID_int", "MGMT_value"]], on="BraTS21ID_int", how="left"
)
sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).clip(0.0, 1.0)
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
sub = sub[["BraTS21ID", "MGMT_value"]]

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

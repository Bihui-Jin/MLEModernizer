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

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

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

0.45412

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the import-time protobuf/TensorFlow crash by removing the forced pure-Python protobuf setting that is incompatible with the Kaggle environment’s protobuf version. Then I fix the dataset path/ID parsing so `test_df` is built correctly (your current code is accidentally seeing a non-numeric folder named `test`). Finally, I make the pipeline robust so the training/validation tensors are always created before fitting and so the best model is always available (fallback to the in-memory model if the checkpoint file is not written), ensuring a valid `submission.csv` is produced end-to-end without changing the modeling approach.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash by pinning TensorFlow to use the pure-Python protobuf implementation, which is the most reliable workaround for the `MessageFactory.GetPrototype` incompatibility in this environment. I keep the model/training logic unchanged, and only add deterministic seeding plus a small amount of defensive setup to ensure the pipeline always reaches the submission-writing step. I also ensure the dataset paths and ID parsing remain correct and that the submission aligns exactly to `sample_submission.csv` IDs with proper zero-padding. These changes are intended to be score-neutral aside from restoring the ability to train/infer end-to-end and generate a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I remove the forced pure-Python protobuf setting because it triggers the TensorFlow/protobuf `MessageFactory.GetPrototype` crash in this environment, and keep the rest of the pipeline unchanged. I also add a small compatibility shim that safely unsets that environment variable before importing TensorFlow to ensure imports succeed reliably. Finally, I keep the existing path/ID parsing and submission-writing logic intact so the notebook runs end-to-end and always produces a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the reliable workaround for the `MessageFactory.GetPrototype` issue in this Kaggle environment. I also make the `tqdm` import robust (notebook vs script) so progress bars don’t error out. These changes are execution/stability fixes and are intended to be score-neutral, keeping your data loading, split strategy, model, training loop, and submission logic unchanged. The pipeline run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf setting that is incompatible with this environment, and instead explicitly unset it before importing TensorFlow. I also fix a runtime shape/logic bug: the model is 2D ConvNet but `X_train/X_valid/X_test` were being expanded to 4D with an extra channel dimension, causing inconsistent input handling; I keep the core 2D model unchanged and simply ensure the tensors passed to it have the expected shape. Finally, I ensure test inference uses the same `X_test` tensor that matches training (not the incorrectly-expanded `X_test_tf`), while keeping the rest of the pipeline, training loop, and submission formatting identical.'
- What this solution (achieved 0.48353) has done: 'I fix the current runtime crash happening at TensorFlow import by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the known workaround for the `MessageFactory.GetPrototype` issue in this environment. I keep your model, training loop, and data pipeline intact, and only add minimal safety around DICOM sorting (some files can break the numeric “Image-xxx” assumption) so data loading doesn’t silently fail. I also ensure IDs are consistently treated as 5-digit strings when building `test_df` and when merging into the submission so every sample gets a prediction (or the intended 0.5 fallback) without misalignment. These changes are execution/stability fixes and should be score-neutral-to-slightly-positive, moving you safely toward the target band without changing core modeling.'
- What this solution (achieved 0.46941) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf environment override, which is triggering the `MessageFactory.GetPrototype` error in this Kaggle runtime. I keep your data loading, slice selection, model definition, training loop, and prediction/aggregation logic unchanged so the score behavior stays essentially the same (only execution stability is improved). I also add a tiny compatibility guard to ensure any inherited protobuf env vars are unset before importing TensorFlow. Finally, I keep the submission-writing logic intact so a valid `submission.csv` is always produced end-to-end.'
- What this solution (achieved 0.45412) has done: 'I fix the TensorFlow/protobuf import crash that currently stops execution by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow (this is the known workaround for the `MessageFactory.GetPrototype` mismatch). I keep your data loading, split strategy, model (the 2D CNN in `get_model02`), training loop, and aggregation/submission logic unchanged. I also add a small safety guard to ensure deterministic behavior and that the checkpoint load always succeeds (fallback to the in-memory model is kept). These changes are execution/stability fixes and should be score-neutral aside from enabling the pipeline to run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
from pathlib import Path
import random

import numpy as np
import pandas as pd

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

import cv2
import pydicom

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

print("TF:", tf.__version__)
print("pydicom:", pydicom.__version__)

np.random.seed(0)
random.seed(12)
tf.random.set_seed(12)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709]  # Bad images (as per competition note)

train_dir = data_dir / "train"
test_dir = data_dir / "test"

assert train_dir.exists(), f"train_dir not found: {train_dir}"
assert test_dir.exists(), f"test_dir not found: {test_dir}"



## === cell 2
train_df = pd.read_csv(data_dir / "train_labels.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

test_ids = sorted(
    [p.name for p in test_dir.iterdir() if p.is_dir() and p.name.isdigit()]
)
test_df = pd.DataFrame({"BraTS21ID": [int(x) for x in test_ids]})

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)].reset_index(drop=True)

print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")
print(f"test data: Rows={test_df.shape[0]}, Columns={test_df.shape[1]}")
train_df.head()




## === cell 3
def load_dicom(path, size=350):
    """Read a DICOM image and resize to (size,size) uint8."""
    dicom = pydicom.dcmread(path, force=True)
    data = dicom.pixel_array.astype(np.float32)

    mx = float(np.max(data)) if data.size else 0.0
    if mx > 0:
        data = data / mx
    data = (data * 255.0).clip(0, 255).astype(np.uint8)

    if data.ndim > 2:
        data = data[..., 0]
    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)




## === cell 4
def _safe_image_sort_key(path_str: str) -> int:
    """
    Some DICOMs may not follow Image-<number>.dcm naming strictly.
    Keep core logic (sorted slices), but make it robust to unexpected names.
    """
    stem = Path(path_str).stem  # e.g., "Image-387"
    try:
        return int(stem.split("-")[-1])
    except Exception:
        return 10**18  # push unknowns to the end


def get_all_image_paths(brats21id, image_type, folder="train"):
    """Return array of DICOM paths for a patient, selecting middle slices with a stride."""
    assert image_type in mri_types

    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder,
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_safe_image_sort_key,
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval], dtype=object)


def get_all_images(brats21id, image_type, folder="train", size=225):
    paths = get_all_image_paths(brats21id, image_type, folder)
    if len(paths) == 0:
        return []
    images = []
    for path in paths:
        try:
            images.append(load_dicom(path, size))
        except Exception:
            continue
    return images




## === cell 5
def get_all_data_for_train(image_type, image_size=32):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        pid = int(x["BraTS21ID"])
        images = get_all_images(pid, image_type, "train", image_size)
        label = int(x["MGMT_value"])

        if len(images) == 0:
            continue

        X += images
        y += [label] * len(images)
        train_ids += [pid] * len(images)

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int64)
    train_ids = np.array(train_ids, dtype=np.int64)
    assert len(X) == len(y) == len(train_ids)
    return X, y, train_ids


def get_all_data_for_test(image_type, image_size=32):
    global test_df

    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        x = test_df.loc[i]
        pid = int(x["BraTS21ID"])
        images = get_all_images(pid, image_type, "test", image_size)

        if len(images) == 0:
            continue

        X += images
        test_ids += [pid] * len(images)

    X = np.array(X, dtype=np.uint8)
    test_ids = np.array(test_ids, dtype=np.int64)
    assert len(X) == len(test_ids)
    return X, test_ids




## === cell 6
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

print("Train slice data:", X.shape, y.shape, trainidt.shape)
print("Test slice data:", X_test.shape, testidt.shape)

assert X.size > 0, "No training slices were loaded; check data paths."
assert X_test.size > 0, "No test slices were loaded; check data paths."



## === cell 7
unique_ids = np.unique(trainidt)
id_labels = train_df.set_index("BraTS21ID").loc[unique_ids, "MGMT_value"].values

ids_train, ids_valid = train_test_split(
    unique_ids, test_size=0.2, random_state=12, stratify=id_labels
)

train_mask = np.isin(trainidt, ids_train)
valid_mask = np.isin(trainidt, ids_valid)

X_train, y_train, trainidt_train = X[train_mask], y[train_mask], trainidt[train_mask]
X_valid, y_valid, trainidt_valid = X[valid_mask], y[valid_mask], trainidt[valid_mask]

print("Train slices:", X_train.shape, y_train.shape)
print("Valid slices:", X_valid.shape, y_valid.shape)



## === cell 8
X_train = np.expand_dims(X_train, axis=-1)
X_valid = np.expand_dims(X_valid, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

y_train_cat = to_categorical(y_train, num_classes=2)
y_valid_cat = to_categorical(y_valid, num_classes=2)

print("X_train:", X_train.shape, "y_train:", y_train_cat.shape)
print("X_test:", X_test.shape)




## === cell 9
def get_model01(width=128, height=128, depth=64, name="3dcnn"):
    """Build a 3D convolutional neural network model."""
    inputs = tf.keras.Input((width, height, depth, 1))

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name=name)

    initial_learning_rate = 0.00009
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )
    return model




## === cell 10
def get_model02():
    np.random.seed(0)
    random.seed(12)
    tf.random.set_seed(12)

    inpt = keras.Input(shape=tuple(X_train.shape[1:]))

    h = layers.Rescaling(1.0 / 255)(inpt)

    h = keras.layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
    h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

    h = keras.layers.Dropout(0.1)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(32, activation="relu")(h)

    output = keras.layers.Dense(2, activation="softmax")(h)

    model = keras.Model(inpt, output)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 11
checkpoint_filepath = "best_model.keras"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_freq="epoch",
    verbose=1,
)



## === cell 12
model = get_model02()
history = model.fit(
    x=X_train,
    y=y_train_cat,
    epochs=25,
    callbacks=[model_checkpoint_callback],
    validation_data=(X_valid, y_valid_cat),
    verbose=2,
)



## === cell 13
if Path(checkpoint_filepath).exists():
    model_best = tf.keras.models.load_model(filepath=checkpoint_filepath)
else:
    model_best = model



## === cell 14
y_pred_valid = model_best.predict(X_valid, verbose=0)[:, 1]

result = pd.DataFrame(
    {"BraTS21ID": trainidt_valid.astype(int), "MGMT_value": y_pred_valid}
)
result2 = result.groupby("BraTS21ID", as_index=False).mean()

result2 = result2.merge(
    train_df, on="BraTS21ID", how="left", suffixes=("_pred", "_true")
)
auc = roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])
print(f"Validation AUC={auc:.5f}")



## === cell 15
y_pred_test = model_best.predict(X_test, verbose=0)[:, 1]
test_result = pd.DataFrame(
    {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred_test}
)
test_result2 = test_result.groupby("BraTS21ID", as_index=False).mean()

sub = sample_submission.copy()
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
test_result2["BraTS21ID"] = (
    test_result2["BraTS21ID"].astype(int).astype(str).str.zfill(5)
)

sub = sub.drop(columns=["MGMT_value"])
sub = sub.merge(test_result2, on="BraTS21ID", how="left")

sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert sub.shape[1] == 2 and list(sub.columns) == ["BraTS21ID", "MGMT_value"]
assert str(Path("submission.csv")).endswith(".csv")

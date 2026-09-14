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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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

0.49412

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49412) has done: 'I fix the two runtime blockers coming from library API changes: `pydicom.read_file` was removed (use `pydicom.dcmread`), and the protobuf `MessageFactory.GetPrototype` crash is triggered by TensorFlow imports in this Kaggle image (pin protobuf to the pure-Python implementation via env vars before importing TF). Then I make the evaluation/prediction logic consistent with the ROC-AUC metric by using the model’s positive-class probability (not `argmax` hard labels) while keeping the same model and training loop. Finally, I ensure the submission is aligned to `sample_submission.csv` order and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.43647) has done: 'I fix the remaining protobuf/TensorFlow crash by forcing the pure-Python protobuf runtime via environment variables *and* ensuring TensorFlow is imported only after those variables are set (this avoids the `MessageFactory.GetPrototype` error in this Kaggle image). I also make the ModelCheckpoint compatible with TF/Keras 2.18 by switching to a `.keras` checkpoint path (TF 2.13+ discourages/limits `.h5` for full-model saving and can error in some environments). These are runtime/stability fixes and should be score-neutral (they don’t change the model, data, or training semantics), so your score should remain around the current level while the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.52824) has done: 'We fix the remaining protobuf/TensorFlow crash by setting the additional environment flag `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before *any* TensorFlow-related import and by forcing protobuf to use its pure-Python backend reliably in this Kaggle image. We also make the `ModelCheckpoint` monitor consistent with the compiled metric name (`val_auc`) and ensure the checkpoint file is always written/read safely. Finally, we keep the model and training loop intact, but ensure the submission is always aligned to `sample_submission.csv` and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.45882) has done: 'We fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by setting the necessary environment variables *and* importing TensorFlow only after those variables are in place (and forcing the pure-Python protobuf implementation reliably). This is a runtime-only change and should be score-neutral. We also make the training/validation split patient-safe (split by `BraTS21ID` rather than by slices) to avoid leakage, which should improve AUC toward a more realistic (and typically higher) level while keeping the same model, loss, and overall training approach. Finally, we keep the submission generation aligned to `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.51176) has done: 'You’re hitting the protobuf/TensorFlow `MessageFactory.GetPrototype` crash before any training starts, so the main fix is to force the pure-Python protobuf implementation *and* disable the C++ implementation via environment variables **before importing anything that may transitively import protobuf/TensorFlow**. I also make the TensorFlow import occur only after those env vars are set and clear the TF session deterministically; these are runtime-stability fixes and should be score-neutral. Finally, I keep your exact model/training logic and patient-level aggregation intact, only ensuring the submission is always written as `submission.csv` with the correct columns/order.'
- What this solution (achieved 0.52) has done: 'I fix the current runtime blocker (`MessageFactory.GetPrototype` from protobuf) by forcing the pure-Python protobuf implementation before any protobuf/TensorFlow import and by importing TensorFlow only after that is set, plus a small compatibility patch for newer protobuf versions. This is a stability-only change and should be score-neutral. I also make the DICOM slice sorting robust (some files are not consistently named) to prevent occasional crashes during path parsing without changing what data is used. Finally, I keep the same model/training logic and ensure the code always writes a valid `submission.csv` in the required format/order.'
- What this solution (achieved 0.52118) has done: 'I fix the protobuf `MessageFactory.GetPrototype` patch so it never crashes on import by patching the *class* method safely (and only if needed), which unblocks the whole pipeline. This change is runtime/stability-only and keeps the model/data/training logic identical. I also make the tqdm import resilient to the Kaggle script environment (falls back to standard `tqdm` if `tqdm.notebook` isn’t available) without changing any computations. The rest of the code (data loading, patient-safe split, model, training loop, and probability-based submission) is preserved so your score behavior should remain consistent while producing a valid `submission.csv`.'
- What this solution (achieved 0.50118) has done: 'The crash happens before training because the protobuf/TensorFlow import path still ends up calling `MessageFactory().GetPrototype(...)` even though newer protobuf versions removed that method on the instance. I fix this with a safe compatibility shim that adds an instance `GetPrototype` method which forwards to `GetMessageClass`, and I apply it before importing TensorFlow so the runtime doesn’t crash. This is a stability-only change (no model/data/training semantics change), so it should keep your score behavior similar while ensuring the notebook runs end-to-end and writes `submission.csv` in the required format. Everything else (patient-safe split, model, training loop, probability-based submission) remains unchanged.'
- What this solution (achieved 0.49412) has done: 'You’re still crashing in cell 1 due to protobuf: the failing call is on a *MessageFactory instance* (`MessageFactory().GetPrototype`), but the current shim only adds `GetPrototype` to the class in a way that doesn’t cover all cases in this Kaggle image. I patch protobuf more robustly by (a) forcing the pure-Python protobuf runtime early, and (b) adding a safe instance-level fallback for `GetPrototype` that forwards to `GetMessageClass` (or to the classmethod if present) before importing TensorFlow. This is a runtime/stability-only fix and should be score-neutral (so your ~0.50 AUC behavior should remain similar), and the rest of the pipeline (patient-safe split, probability predictions, and submission alignment) stays unchanged and still write `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION"] = "1"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from google.protobuf.message_factory import (
        MessageFactory as _MessageFactory,
    )  # noqa: N812

    if not hasattr(_MessageFactory, "GetPrototype") and hasattr(
        _MessageFactory, "GetMessageClass"
    ):

        def _mf_getprototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        setattr(_MessageFactory, "GetPrototype", _mf_getprototype)

    _mf_inst = _MessageFactory()
    if not hasattr(_mf_inst, "GetPrototype") and hasattr(_mf_inst, "GetMessageClass"):

        def _inst_getprototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        setattr(_MessageFactory, "GetPrototype", _inst_getprototype)

except Exception:
    pass

import json
import glob
import random
import collections

import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # out of 255
EXCLUDE = [109, 123, 709]

BASE_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_df = pd.read_csv(os.path.join(BASE_PATH, "train_labels.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)

print("Train rows:", len(train_df), "Test rows:", len(test_df))
print(train_df.head())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_dicom(path, size=224):
    """Read a DICOM image and return a resized uint8 image."""
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    maxv = np.max(data)
    if maxv > 0:
        data = data / maxv
    data = (data * 255.0).clip(0, 255).astype(np.uint8)

    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)


def _safe_slice_key(p):
    base = os.path.splitext(os.path.basename(p))[0]
    try:
        return int(base.split("-")[-1])
    except Exception:
        return base


def get_all_image_paths(brats21id, image_type, folder="train"):
    """Return a subset of slice paths for a patient and series type."""
    assert image_type in TYPES

    patient_path = os.path.join(
        BASE_PATH,
        folder,
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_safe_slice_key,
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([])

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval])


def get_all_images(brats21id, image_type, folder="train", size=224):
    paths = get_all_image_paths(brats21id, image_type, folder)
    if len(paths) == 0:
        return []
    return [load_dicom(path, size) for path in paths]




## === cell 2
IMAGE_SIZE = 32


def get_all_data_for_train(image_type):
    X, y, train_ids = [], [], []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        pid = int(x["BraTS21ID"])
        images = get_all_images(pid, image_type, "train", IMAGE_SIZE)
        label = int(x["MGMT_value"])

        if len(images) == 0:
            continue

        X += images
        y += [label] * len(images)
        train_ids += [pid] * len(images)

    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type):
    X, test_ids = [], []

    for i in tqdm(test_df.index):
        x = test_df.loc[i]
        pid = int(x["BraTS21ID"])
        images = get_all_images(pid, image_type, "test", IMAGE_SIZE)

        if len(images) == 0:
            continue

        X += images
        test_ids += [pid] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 3
X, y, trainidt = get_all_data_for_train("T1wCE")
X_test, testidt = get_all_data_for_test("T1wCE")
print("Train slices:", X.shape, y.shape, trainidt.shape)
print("Test slices:", X_test.shape, testidt.shape)



## === cell 4
unique_ids = np.unique(trainidt)
id_to_label = train_df.set_index("BraTS21ID")["MGMT_value"].to_dict()
patient_labels = np.array([int(id_to_label[int(pid)]) for pid in unique_ids])

train_ids_u, valid_ids_u = train_test_split(
    unique_ids, test_size=0.2, random_state=42, stratify=patient_labels
)

train_mask = np.isin(trainidt, train_ids_u)
valid_mask = np.isin(trainidt, valid_ids_u)

X_train = X[train_mask]
y_train = y[train_mask]
trainidt_train = trainidt[train_mask]

X_valid = X[valid_mask]
y_valid = y[valid_mask]
trainidt_valid = trainidt[valid_mask]

X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)

y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
print(
    "Unique train patients:",
    len(np.unique(trainidt_train)),
    "Unique valid patients:",
    len(np.unique(trainidt_valid)),
)



## === cell 5
tf.keras.backend.clear_session()
inputs = keras.Input(shape=X_train.shape[1:])

h = keras.layers.Rescaling(1.0 / 255.0)(inputs)

h = keras.layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(h)
h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

h = keras.layers.Dropout(0.3)(h)

h = keras.layers.Flatten()(h)
h = keras.layers.Dropout(0.2)(h)
h = keras.layers.Dense(32, activation="relu")(h)

output = keras.layers.Dense(2, activation="softmax")(h)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=20, restore_best_weights=False
)

model = keras.Model(inputs, output)
print(model.summary())

checkpoint_filepath = "best_model.keras"
if os.path.exists(checkpoint_filepath):
    try:
        os.remove(checkpoint_filepath)
    except OSError:
        pass

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_freq="epoch",
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
)

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    callbacks=[model_checkpoint_callback, early_stopping],
    validation_data=(X_valid, y_valid),
    verbose=2,
)



## === cell 6
if os.path.exists(checkpoint_filepath):
    model_best = tf.keras.models.load_model(filepath=checkpoint_filepath)
else:
    model_best = model



## === cell 7
y_pred_valid = model_best.predict(X_valid, verbose=0)[:, 1]

result = pd.DataFrame({"BraTS21ID": trainidt_valid, "MGMT_value": y_pred_valid})
result2 = result.groupby("BraTS21ID", as_index=False).mean()

result2 = result2.merge(
    train_df, on="BraTS21ID", how="inner", suffixes=("_pred", "_true")
)
auc = roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])
print("Validation patient-level AUC:", auc, "Patients:", len(result2))



## === cell 8
len(result2)



## === cell 9
sample = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

if len(X_test) > 0:
    X_test_t = tf.expand_dims(X_test, axis=-1)
    y_pred_test = model_best.predict(X_test_t, verbose=0)[:, 1]
    test_pred_df = pd.DataFrame({"BraTS21ID": testidt, "MGMT_value": y_pred_test})
    test_pred_df = test_pred_df.groupby("BraTS21ID", as_index=False).mean()
else:
    test_pred_df = pd.DataFrame({"BraTS21ID": [], "MGMT_value": []})

sub = sample[["BraTS21ID"]].merge(test_pred_df, on="BraTS21ID", how="left")
sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(float)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

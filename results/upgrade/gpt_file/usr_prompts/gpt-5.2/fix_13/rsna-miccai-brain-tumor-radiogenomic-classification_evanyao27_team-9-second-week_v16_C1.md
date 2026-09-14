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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
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

0.48471

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59176) has done: 'I fix the two runtime blockers coming from library/API changes: the protobuf incompatibility that breaks TensorFlow import, and the pydicom API change (`read_file` → `dcmread`). Then I make the submission generation consistent with the AUC metric by outputting the positive-class probability (not `argmax` class labels) while keeping the exact same CNN/training loop. Finally, I ensure IDs are formatted and aligned to the provided `sample_submission.csv` so the written `submission.csv` is valid and ordered correctly.'
- What this solution (achieved 0.55882) has done: 'We fix the runtime blocker in cell 1 caused by an incompatibility between TensorFlow 2.18 and protobuf 6.x by forcing TensorFlow to use the pure-Python protobuf implementation and by importing TensorFlow only after that env var is set (and enabling safe Keras/TensorFlow integration). Then we keep your exact DICOM loading, sampling, CNN, training loop, and probability-based submission logic unchanged, only adding small robustness guards to avoid empty-image edge cases that can crash the pipeline. These changes are score-neutral in intent (the target score given is invalid for AUC), and the primary goal is to ensure the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.50588) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x runtime API mismatch by forcing the pure-Python protobuf implementation early and (critically) disabling the compiled protobuf fast-path. Then I keep your exact data loading, model, training loop, and probability-based submission logic unchanged, only adding a small robustness fallback so DICOM read/pixel decode errors don’t crash the run. Finally, I keep the submission formatting/alignment to `sample_submission.csv` to guarantee a valid `submission.csv` is always written. These changes are intended to be score-neutral (mainly to unblock execution and ensure the CSV is produced).'
- What this solution (achieved 0.52588) has done: 'You’re hitting a TensorFlow/protobuf runtime incompatibility that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during the TF import, so the first fix is to force the pure-Python protobuf implementation *and* disable the compiled fast-path before importing TensorFlow. I also keep your exact data loading/training/inference logic, but add a small safety guard so the model always gets at least one test image (otherwise `predict()` can crash or produce an empty merge). Finally, I keep the submission generation identical (positive-class probability, grouped by ID, aligned to `sample_submission.csv`) and ensure it always writes `submission.csv`.'
- What this solution (achieved 0.56235) has done: 'We fix the TensorFlow import crash by moving the protobuf environment variables to the very top and additionally setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before *any* TensorFlow/keras import (this is the root cause of the `MessageFactory.GetPrototype` error in TF 2.18 + protobuf 6.x). We keep your data loading, CNN, training loop, and probability-based submission logic unchanged, only making the import ordering and a couple of robustness guards deterministic so the pipeline always runs end-to-end. Since your provided target score (-1.0) is not a valid AUC target, we not change modeling choices for score tuning; the changes are intended to be score-neutral and stability-focused. The script always write a valid `submission.csv` with the exact required columns and ID formatting aligned to `sample_submission.csv`.'
- What this solution (achieved 0.52588) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow/Keras import and by additionally blocking the C++ protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` with an early import order. I also make the train/test ID handling robust by ensuring `BraTS21ID` is read/kept as a zero-padded string only at merge/submission time (avoids mismatches and missing predictions). Finally, I keep your exact data loading, CNN, training loop, and probability-based submission logic unchanged, only adding small guards so that (a) AUC monitoring works reliably and (b) a valid `submission.csv` is always produced end-to-end.'
- What this solution (achieved 0.55882) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf environment variables before any TensorFlow/Keras import and by avoiding `tqdm.notebook` (which can pull in Jupyter widgets and indirectly import things too early). I keep your exact DICOM loading, sampling, CNN architecture, training loop, and probability-based submission logic unchanged, only adding minimal determinism and a small robustness guard so a valid `submission.csv` is always produced. Your target score `-1.0` is not meaningful for AUC (AUC ∈ [0,1]), so I won’t make score-tuning changes beyond ensuring the pipeline runs and predicts valid probabilities. The resulting script run end-to-end and write `submission.csv` with the required columns and ID formatting aligned to `sample_submission.csv`.'
- What this solution (achieved 0.53059) has done: 'We fix the TensorFlow/protobuf crash by moving the protobuf environment variables to the absolute top of the script and importing `google.protobuf` (and `pydicom`) only after that, which prevents the `MessageFactory.GetPrototype` failure in TF 2.18 + protobuf 6.x. We also keep your exact model/training/prediction logic, but make one minimal robustness fix: use a safer DICOM filename sort that won’t crash if filenames don’t match the expected pattern. Finally, we keep the submission logic identical (mean probability per patient, aligned to `sample_submission.csv`) and ensure `submission.csv` is always written with correct columns and ID formatting.'
- What this solution (achieved 0.55294) has done: 'I fix the runtime blocker happening at import time by ensuring the protobuf “pure python” environment variables are set before *any* protobuf/TensorFlow-related import, and by avoiding an early explicit `import google.protobuf` which can trigger the incompatible C++ path. This is a stability fix only and should not change your model/training logic or submission semantics. I also keep your current data loading, CNN, training loop, and probability-based submission generation identical, only reordering imports safely. Finally, I ensure the script still writes a valid `submission.csv` with the required columns and correct zero-padded IDs aligned to `sample_submission.csv`.'
- What this solution (achieved 0.55412) has done: 'We fix the TensorFlow import crash caused by the TF 2.18 + protobuf 6.x incompatibility by moving the protobuf environment variables to the absolute top of the script and preventing any early protobuf-triggering imports before TensorFlow loads. This is a runtime/stability fix only and keeps your data loading, CNN architecture, training loop, and probability-based submission semantics unchanged. We also make the import order explicitly safe (TensorFlow before pydicom) and keep ID formatting/merging exactly aligned to `sample_submission.csv` so the output `submission.csv` is always valid. Since the provided target score (-1.0) is not meaningful for AUC, no score-tuning changes are made beyond ensuring the pipeline runs deterministically.'
- What this solution (achieved 0.55176) has done: 'The crash happens before training because TensorFlow 2.18 + protobuf 6.x can still load the incompatible compiled protobuf backend unless the env vars are set *and* TensorFlow is imported before any protobuf/pydicom-related imports. I move the protobuf environment variable setup to the absolute top and delay importing `pydicom` until after `tensorflow` is successfully imported, which resolves the `MessageFactory.GetPrototype` error. I also fix a small checkpoint monitor mismatch (`val_auc` vs `val_auc` name) by ensuring the monitored metric name exists (keeping the same metric) and keep the rest of your pipeline identical so score behavior is unchanged while the notebook runs end-to-end. The script still write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.48471) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by moving the protobuf environment variables to the absolute top of the script and ensuring no protobuf-triggering imports (like `pydicom`) happen before importing TensorFlow. I keep your exact data loading, DICOM sampling, CNN architecture, training loop, and probability-based submission logic unchanged, only adjusting import order and adding a small guard to ensure the checkpoint monitor metric name always matches what Keras logs. These changes are intended to be score-neutral (your current score is already valid and the provided target score -1.0 isn’t meaningful for AUC), while making the notebook run end-to-end reliably and always write a valid `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_FAST_CPP"] = "1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import re
import glob
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import cv2

import pydicom

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # out of 255
EXCLUDE = [109, 123, 709]

DATA_DIR = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(int)
train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_dicom(path, size=224):
    """
    Reads a DICOM image, standardizes so that the pixel values are between 0 and 1,
    then rescales to 0..255 and resizes.
    """
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array
    except Exception:
        return None

    if data is None:
        return None

    data = np.asarray(data)
    if data.size == 0:
        return None

    mx = np.max(data)
    if mx != 0:
        data = data / mx
    data = (data * 255).astype(np.uint8)

    try:
        return cv2.resize(data, (size, size))
    except Exception:
        return None


def _safe_image_sort_key(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"(\d+)(?:\.\w+)?$", base)
    return int(m.group(1)) if m else 0


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of all the images of a particular type for a particular patient ID.
    """
    assert image_type in TYPES

    patient_path = os.path.join(
        f"{DATA_DIR}/{folder}/",
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_safe_image_sort_key,
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([])

    start = int(num_images * 0.2)
    end = int(num_images * 0.8)

    interval = 3
    if num_images < 10:
        interval = 1

    sel = paths[start:end:interval]
    if len(sel) == 0 and num_images > 0:
        sel = [paths[num_images // 2]]

    return np.array(sel)


def get_all_images(brats21id, image_type, folder="train", size=225):
    paths = get_all_image_paths(brats21id, image_type, folder)
    if len(paths) == 0:
        return []
    imgs = []
    for path in paths:
        im = load_dicom(path, size)
        if im is not None:
            imgs.append(im)
    return imgs




## === cell 2
def get_all_data_for_train(image_type):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index, total=len(train_df)):
        x = train_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "train", 32)
        label = x["MGMT_value"]

        if len(images) == 0:
            continue

        X += images
        y += [label] * len(images)
        train_ids += [int(x["BraTS21ID"])] * len(images)
        assert len(X) == len(y)
    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type):
    global test_df

    X = []
    test_ids = []

    for i in tqdm(test_df.index, total=len(test_df)):
        x = test_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "test", 32)

        if len(images) == 0:
            continue

        X += images
        test_ids += [int(x["BraTS21ID"])] * len(images)

    if len(X) == 0:
        X = [np.zeros((32, 32), dtype=np.uint8)]
        test_ids = [int(test_df.loc[0, "BraTS21ID"])]

    return np.array(X), np.array(test_ids)




## === cell 3
X, y, trainidt = get_all_data_for_train("T1wCE")
X_test, testidt = get_all_data_for_test("T1wCE")
X.shape, y.shape, trainidt.shape, X_test.shape, testidt.shape



## === cell 4
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X, y, trainidt, test_size=0.2, random_state=40
)

X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, trainidt_train.shape, trainidt_valid.shape



## === cell 5
np.random.seed(0)
random.seed(12)
tf.random.set_seed(12)

inpt = keras.Input(shape=X_train.shape[1:])

h = keras.layers.Rescaling(1.0 / 255)(inpt)

h = keras.layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(h)
h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

h = keras.layers.Dropout(0.3)(h)

h = keras.layers.Flatten()(h)
h = keras.layers.Dense(32, activation="relu")(h)
h = keras.layers.Dense(8, activation="relu")(h)

output = keras.layers.Dense(2, activation="softmax")(h)

model = keras.Model(inpt, output)

checkpoint_filepath = "best_model.keras"

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
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=20,
    callbacks=[model_checkpoint_callback],
    validation_data=(X_valid, y_valid),
)



## === cell 6
model_best = tf.keras.models.load_model(filepath=checkpoint_filepath)



## === cell 7
y_pred = model_best.predict(X_valid, verbose=0)
pred_prob = y_pred[:, 1]

result = pd.DataFrame({"BraTS21ID": trainidt_valid, "MGMT_value": pred_prob})
result2 = result.groupby("BraTS21ID", as_index=False).mean()
result2 = result2.merge(
    train_df, on="BraTS21ID", how="inner", suffixes=("_pred", "_true")
)

roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])



## === cell 8
len(result2), result2.head()



## === cell 9
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

y_pred_test = model_best.predict(tf.expand_dims(X_test, axis=-1), verbose=0)
pred_prob_test = y_pred_test[:, 1]

test_result = pd.DataFrame({"BraTS21ID": testidt, "MGMT_value": pred_prob_test})
test_result2 = test_result.groupby("BraTS21ID", as_index=False).mean()

test_result2["BraTS21ID"] = (
    test_result2["BraTS21ID"].astype(int).astype(str).str.zfill(5)
)

sub = sample[["BraTS21ID"]].copy()
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
sub = sub.merge(test_result2, on="BraTS21ID", how="left")

sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
sub

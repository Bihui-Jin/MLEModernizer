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

0.50118

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.46941) has done: 'I fix the two root runtime issues preventing any training/inference: the protobuf/pydicom compatibility error and the removed `pydicom.read_file` API, replacing it with `pydicom.dcmread` and importing tensorflow before pydicom to avoid the protobuf crash. I also update the deprecated `keras.layers.experimental.preprocessing.Rescaling` to the supported `keras.layers.Rescaling` so the model builds under TF/Keras 2.18/3.x. Finally, I correct the prediction-to-probability logic so validation AUC and the submission use the positive-class probability (not argmax/rounded labels), while keeping the same CNN and training loop; this should both produce a valid `submission.csv` and improve ROC-AUC versus the current broken pipeline.'
- What this solution (achieved 0.46471) has done: 'I fix the protobuf/pydicom crash causing the very first cell to fail by forcing the Python protobuf implementation before importing TensorFlow and pydicom (this is the known safe workaround in Kaggle for this dataset). I also make the imports compatible with the environment by avoiding `tqdm.notebook` (can break outside notebooks) and using standard `tqdm`. Finally, I keep the exact same data pipeline/model/training logic and only add a small safety wrapper around DICOM reads so a single corrupted slice can’t crash the full run, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 0.54588) has done: 'I fix the protobuf/TensorFlow/pydicom import crash that stops the notebook at cell 0 by pinning protobuf to the pure-Python implementation and also forcing the pydicom VR to a safe setting before importing TensorFlow/pydicom. I keep your exact data pipeline/model/training loop intact, but correct the checkpoint monitor name (`val_auc` doesn’t exist because the metric is named `auc`), which currently prevents saving/loading the best model reliably and can hurt score. I also ensure the pipeline always writes a valid `submission.csv` with the required columns, even if some patients yield no readable slices.'
- What this solution (achieved 0.51294) has done: 'I fix the crash in the first cell caused by an incompatible protobuf runtime by ensuring TensorFlow is imported before pydicom (and keeping the pure-Python protobuf setting), which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle environment. I keep your data pipeline, model, and training loop the same, but correct the checkpoint callback monitor from the nonexistent `val_auc` to `val_loss` so the best model can be saved/loaded reliably under Keras 3 (this is score-stabilizing and often slightly improves). I also add a small safety guard when sorting DICOM filenames so unexpected names don’t crash path collection, without changing which slices are used in normal cases. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.50471) has done: 'I fix the protobuf/pydicom crash in the first cell by forcing the pure-Python protobuf implementation and importing `pydicom` before `tensorflow` (this avoids the `MessageFactory.GetPrototype` error seen in Kaggle for this dataset). I keep your data pipeline, model architecture, and training loop unchanged, only making this import-order/environment fix plus a tiny robustness tweak to ensure IDs are consistently `int` for merges/groupbys. Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns (`BraTS21ID`, `MGMT_value`) and correct probability usage.'
- What this solution (achieved 0.50118) has done: 'I fix the immediate runtime crash by changing the import order so TensorFlow loads before pydicom while keeping your existing protobuf environment workaround; this resolves the `MessageFactory.GetPrototype` issue in this Kaggle environment. I also add a small safety guard so patients with zero readable slices don’t break `train_test_split`/training (they be skipped in train and get default 0.5 in test), which is correctness/stability and should be score-neutral. Finally, I keep the same model, training loop, and probability aggregation, and ensure `submission.csv` is always written with the required columns and ID formatting.'
- What this solution (achieved 0.48941) has done: 'I fix the protobuf/TensorFlow/pydicom incompatibility that causes the `MessageFactory.GetPrototype` crash by applying the known-safe workaround: force the pure-Python protobuf runtime *and* import `pydicom` before importing `tensorflow`. I keep your data pipeline/model/training exactly the same, but move the `pydicom` config lines to occur immediately after importing pydicom (before TensorFlow touches protobuf internals). This should unblock end-to-end execution so training/inference completes and a valid `submission.csv` is always written. No score-tuning changes are introduced since your current score is already far above the provided (odd) target of `-1.0`.'
- What this solution (achieved 0.56706) has done: 'I fix the protobuf/TensorFlow/pydicom crash that prevents the very first cell from running by adjusting the environment variables and import order to the known-stable configuration for this Kaggle dataset. I keep the data pipeline, CNN architecture, training loop, and probability aggregation logic identical, only adding minimal robustness so a single bad DICOM slice can’t abort execution. Since your target score is given as `-1.0` (which is not meaningful for ROC-AUC and your current score is already far above it), I not introduce any score-tuning changes beyond ensuring the model actually trains/inferences and outputs valid probabilities. The script run end-to-end and always write a correctly formatted `submission.csv`.'
- What this solution (achieved 0.47176) has done: 'I fix the protobuf/TensorFlow/pydicom crash by switching to the stable import order and environment settings used in Kaggle for this dataset (import TensorFlow before pydicom, and force the pure-Python protobuf runtime). I keep your data loading, slice selection, CNN architecture, training loop, and probability aggregation unchanged, only adding a small DICOM-read safety fallback so a single bad file can’t abort the run. I also ensure the submission is always written as `submission.csv` with the exact required columns and correct ID typing/formatting. Since your provided target score is `-1.0` (not meaningful for ROC-AUC) and current score is already strong, I won’t introduce score-tuning changes beyond restoring correct execution.'
- What this solution (achieved 0.50118) has done: 'I fix the protobuf/TensorFlow/pydicom crash by applying the known-stable Kaggle workaround: force the pure-Python protobuf implementation and import `pydicom` (with a safe config) before importing TensorFlow/Keras. I keep your data pipeline, model architecture, training loop, and probability aggregation identical, only making these import/config changes plus a tiny robustness guard so DICOM reads won’t abort the run. Since your target score is `-1.0` (not meaningful for ROC-AUC) and your current score is already valid, I won’t introduce any score-tuning changes beyond restoring stable execution and ensuring `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import json
import glob
import random
import collections

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut  # kept as in original

try:
    pydicom.config.settings.reading_validation_mode = "IGNORE"
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # out of 255
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_df = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(int)

train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print(train_df.head())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_dicom(path, size=224):
    """Reads a DICOM image, scales to [0,255] uint8, resizes to (size,size)."""
    try:
        dicom = pydicom.dcmread(path, force=True)
        data = dicom.pixel_array
    except Exception:
        return np.zeros((size, size), dtype=np.uint8)

    maxv = np.max(data)
    if maxv != 0:
        data = data / maxv
    data = (data * 255).astype(np.uint8)

    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)


def _dicom_sort_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(base.split("-")[-1])
    except Exception:
        return base


def get_all_image_paths(brats21id, image_type, folder="train"):
    """Returns an array of all image paths of a type for a patient."""
    assert image_type in TYPES

    patient_path = os.path.join(
        DATA_ROOT,
        folder,
        str(int(brats21id)).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_dicom_sort_key,
    )

    num_images = len(paths)
    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval])


def get_all_images(brats21id, image_type, folder="train", size=225):
    return [
        load_dicom(path, size)
        for path in get_all_image_paths(brats21id, image_type, folder)
    ]




## === cell 2
IMAGE_SIZE = 64


def get_all_data_for_train(image_type):
    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index, desc=f"Train {image_type}"):
        x = train_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "train", IMAGE_SIZE)
        label = int(x["MGMT_value"])

        if len(images) == 0:
            continue

        X += images
        y += [label] * len(images)
        train_ids += [int(x["BraTS21ID"])] * len(images)
        assert len(X) == len(y) == len(train_ids)

    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type):
    X = []
    test_ids = []

    for i in tqdm(test_df.index, desc=f"Test {image_type}"):
        x = test_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "test", IMAGE_SIZE)

        if len(images) == 0:
            continue

        X += images
        test_ids += [int(x["BraTS21ID"])] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 3
X, y, trainidt = get_all_data_for_train("T1wCE")
X_test, testidt = get_all_data_for_test("T1wCE")
print("X:", X.shape, "y:", y.shape, "trainidt:", trainidt.shape)
print("X_test:", X_test.shape, "testidt:", testidt.shape)



## === cell 4
can_train = X.shape[0] > 0 and len(np.unique(y)) > 1
print("can_train:", can_train, "| unique y:", np.unique(y) if X.shape[0] else "N/A")

if can_train:
    X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = (
        train_test_split(X, y, trainidt, test_size=0.2, random_state=40, stratify=y)
    )

    X_train = tf.expand_dims(X_train, axis=-1)
    X_valid = tf.expand_dims(X_valid, axis=-1)
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_valid_cat = to_categorical(y_valid, num_classes=2)

    print(
        X_train.shape,
        y_train_cat.shape,
        X_valid.shape,
        y_valid_cat.shape,
        trainidt_train.shape,
        trainidt_valid.shape,
    )
else:
    X_train = X_valid = y_train_cat = y_valid_cat = trainidt_valid = None

X_test_tf = (
    tf.expand_dims(X_test, axis=-1)
    if X_test.shape[0]
    else tf.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=tf.uint8)
)



## === cell 5
np.random.seed(0)
random.seed(12)
tf.random.set_seed(12)

inpt = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

h = keras.layers.Rescaling(1.0 / 255.0)(inpt)

h = keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", name="Conv_1")(h)
h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

h = keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", name="Conv_3")(h)
h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

h = keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", name="Conv_5")(h)
h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

h = keras.layers.Flatten()(h)
h = keras.layers.Dense(256, activation="relu")(h)
output = keras.layers.Dense(2, activation="softmax")(h)

model = keras.Model(inpt, output)

checkpoint_filepath = "best_model.keras"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_freq="epoch",
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

history = None
if can_train:
    history = model.fit(
        x=X_train,
        y=y_train_cat,
        epochs=50,
        callbacks=[model_checkpoint_callback],
        validation_data=(X_valid, y_valid_cat),
        verbose=2,
    )
else:
    print(
        "Skipping training due to insufficient loaded training data; will write baseline submission."
    )



## === cell 6
if os.path.exists(checkpoint_filepath):
    model_best = tf.keras.models.load_model(checkpoint_filepath)
else:
    model_best = model
print(
    "Using model:", "checkpoint" if os.path.exists(checkpoint_filepath) else "in-memory"
)



## === cell 7
if can_train:
    y_pred_valid = model_best.predict(X_valid, verbose=0)
    pos_prob_valid = y_pred_valid[:, 1]

    result = pd.DataFrame(
        {"BraTS21ID": trainidt_valid.astype(int), "MGMT_value": pos_prob_valid}
    )
    result2 = result.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

    result2 = result2.merge(train_df, on="BraTS21ID", suffixes=("_pred", "_true"))
    auc = roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])
    print("Patient-level valid AUC:", auc)
else:
    result2 = pd.DataFrame()
    print("No validation AUC computed (training skipped).")



## === cell 8
_ = len(result2)
print("Validation patients:", len(result2) if isinstance(result2, pd.DataFrame) else 0)



## === cell 9
sample = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

if X_test_tf.shape[0] == 0:
    submission = sample.copy()
    submission["MGMT_value"] = 0.5
else:
    y_pred_test = model_best.predict(X_test_tf, verbose=0)
    pos_prob_test = y_pred_test[:, 1]

    result = pd.DataFrame(
        {"BraTS21ID": testidt.astype(int), "MGMT_value": pos_prob_test}
    )
    result2 = result.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

    submission = sample[["BraTS21ID"]].merge(result2, on="BraTS21ID", how="left")
    submission["MGMT_value"] = submission["MGMT_value"].fillna(0.5).astype(float)

submission = submission[["BraTS21ID", "MGMT_value"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv columns:", submission.columns.tolist())

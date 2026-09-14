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
keras-tuner==1.4.7
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

0.54471

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.45059) has done: 'I fix the immediate runtime blockers caused by library/API changes: TensorFlow/Protobuf incompatibility (triggering the `MessageFactory.GetPrototype` error), the `pydicom.read_file` deprecation, and the removal of `layers.experimental.preprocessing` in TF/Keras 2.18. I also correct the early-stopping monitor name so training doesn’t crash, and ensure predictions use probabilities (not argmax classes) since the competition metric is ROC-AUC. Finally, I make sure the submission file is written as `submission.csv` with the exact required columns and IDs aligned to `sample_submission.csv`.'
- What this solution (achieved 0.48706) has done: 'I fix the immediate runtime blocker in TensorFlow/Protobuf by avoiding the incompatible “python” protobuf implementation and instead forcing the compatible “upb” implementation before importing TensorFlow. I also fix the model saving error by giving `model.save()` a valid extension (`.keras`) without changing the model/training logic. Finally, I make the early-stopping callback robust by restoring weights and ensuring the monitored metric exists, while keeping the tuner objective and training setup the same so score behavior stays comparable. The pipeline run end-to-end and always write a valid `submission.csv` with the required columns and ID alignment.'
- What this solution (achieved 0.50412) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation (the “upb” setting is what triggers the `MessageFactory.GetPrototype` AttributeError with this environment), and I do it before any TensorFlow import. I also make the file paths robust to both `/kaggle/input/...` and the relative `../input/...` layouts so the notebook runs reliably. To move AUC upward toward a better score without changing the model/training core, I ensure we actually train the tuned augmented architecture by rebuilding the best model with `make_model_augmented(best_hp)` (previously it tuned augmented but trained the non-augmented model). Finally, I keep the submission formatting strict and always write `submission.csv` with the correct columns and aligned IDs.'
- What this solution (achieved 0.50941) has done: 'I fix the TensorFlow/Protobuf crash by switching the protobuf implementation from `"python"` (which is triggering the `MessageFactory.GetPrototype` error in this environment) to the compatible default `"upb"` and doing it before any TensorFlow import. I also add a small guard to ensure the DICOM slice sorting never crashes on unexpected filenames, without changing which images are selected. Finally, I keep the current tuned augmented-model training and probability-based submission logic intact, but ensure the script always reaches the `submission.csv` write step successfully.'
- What this solution (achieved 0.45412) has done: 'I fix the immediate runtime crash happening before any model code runs: the TensorFlow/Protobuf `MessageFactory.GetPrototype` error, by forcing the stable pure-Python protobuf implementation *before* importing TensorFlow (this is the most reliable workaround in Kaggle TF 2.18 environments). I also correct a small bug in `SineConvLayer` where `kernel_size` was ignored in the non-first conv (doesn’t change your currently-used model, but prevents future runtime/logic issues if you switch tuners). Finally, I keep your training/tuning and submission logic unchanged, ensuring the pipeline completes and always writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.54471) has done: 'I fix the TensorFlow/Protobuf crash by switching the protobuf implementation to the compatible default (`upb`) and doing it before importing TensorFlow, which unblocks the whole pipeline. I also make the early-stopping callback consistent with the tuner objective (`val_loss`) so tuning doesn’t get silently misdirected or unstable, while keeping the same model architecture and training approach. Finally, I make the ID exclusion robust by matching the zero-padded ID format used in `train_labels.csv`, and keep the submission writing logic unchanged so it always produces a valid `submission.csv` with correct columns and alignment.'

# 9. Code solution

## === cell 0
import os
import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "upb"

import pandas as pd
import numpy as np
from pathlib import Path

import random
from tqdm.notebook import tqdm
import pydicom  # Handle MRI images

import cv2  # OpenCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomUniform

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")
if not data_dir.exists():
    data_dir = Path(
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
    )

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709]  # Bad images (as per competition note)

train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(str).str.zfill(5)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(str).str.zfill(5)
sample_submission["BraTS21ID"] = sample_submission["BraTS21ID"].astype(str).str.zfill(5)

excluded_ids = [str(x).zfill(5) for x in excluded_images]
train_df = train_df[~train_df.BraTS21ID.isin(excluded_ids)].reset_index(drop=True)




## === cell 2
def create_folds(data, num_splits):
    data = data.copy()
    data["kfold"] = -1
    kf = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=42)
    for f, (t, v) in enumerate(kf.split(X=data)):
        data.loc[v, "kfold"] = f
    return data




## === cell 3
k = 5
train_df = create_folds(train_df, k)



## === cell 4
train_df.head()




## === cell 5
def load_dicom(path, size=224):
    """Read a DICOM image, normalize to [0, 1], then scale to [0, 255] uint8 and resize."""
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)
    mx = np.max(data)
    if mx != 0:
        data = data / mx
    data = (data * 255.0).astype(np.uint8)
    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)


def _slice_sort_key(p):
    stem = os.path.splitext(os.path.basename(p))[0]
    try:
        return int(stem.split("-")[-1])
    except Exception:
        return stem


def get_all_image_paths(brats21id, image_type, folder="train"):
    """Return all DICOM paths for a given patient & modality, selecting mid-slices and skipping some for speed."""
    assert image_type in mri_types

    patient_path = os.path.join(
        str(data_dir / folder),
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_slice_sort_key,
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


def get_all_data_for_train(image_type, image_size=32):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "train", image_size)
        label = x["MGMT_value"]

        X += images
        y += [label] * len(images)
        train_ids += [int(x["BraTS21ID"])] * len(images)
        assert len(X) == len(y)
    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type, image_size=32):
    global test_df

    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        x = test_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "test", image_size)
        X += images
        test_ids += [int(x["BraTS21ID"])] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 6
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)



## === cell 7
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X, y, trainidt, test_size=0.2, random_state=42, stratify=y
)



## === cell 8
X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)
X_test_tf = tf.expand_dims(X_test, axis=-1)
X_train.shape



## === cell 9
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)




## === cell 10
class SineDenseLayer(keras.layers.Layer):
    def __init__(self, features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.features = features

        if self.is_first:
            initializer = RandomUniform(-1 / self.features, 1 / self.features)
            self.linear = keras.layers.Dense(features, kernel_initializer=initializer)
        else:
            initializer = RandomUniform(
                -np.sqrt(6 / self.features) / self.omega_0,
                np.sqrt(6 / self.features) / self.omega_0,
            )
            self.linear = keras.layers.Dense(features, kernel_initializer=initializer)

    def call(self, input):
        return tf.math.sin(self.omega_0 * self.linear(input))


class SineConvLayer(keras.layers.Layer):
    def __init__(self, features, kernel_size, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.features = features

        if self.is_first:
            initializer = RandomUniform(-1 / self.features, 1 / self.features)
            self.conv = keras.layers.Conv2D(
                features, kernel_size, kernel_initializer=initializer
            )
        else:
            initializer = RandomUniform(
                -np.sqrt(6 / self.features) / self.omega_0,
                np.sqrt(6 / self.features) / self.omega_0,
            )
            self.conv = keras.layers.Conv2D(
                features, kernel_size, kernel_initializer=initializer
            )

    def call(self, input):
        return tf.math.sin(self.omega_0 * self.conv(input))




## === cell 11
import keras_tuner as kt


def make_model(hp):
    inputs = keras.Input(shape=X_train.shape[1:])

    x = keras.layers.Rescaling(1.0 / 255)(inputs)

    x = keras.layers.Conv2D(
        filters=hp.Int("units_Conv_1_" + str(0), min_value=64, max_value=256, step=32),
        kernel_size=(4, 4),
        activation="relu",
        name="Conv_1",
    )(x)

    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = keras.layers.Conv2D(
        filters=hp.Int("units_conv2_" + str(1), min_value=16, max_value=128, step=16),
        kernel_size=(2, 2),
        activation="relu",
        name="Conv_2",
    )(x)

    x = keras.layers.MaxPool2D(pool_size=(1, 1))(x)

    x = layers.Dropout(hp.Float("dense_dropout", min_value=0.0, max_value=0.7))(x)
    x = keras.layers.Flatten()(x)

    x = layers.Dense(
        units=hp.Int("num_dense_units", min_value=16, max_value=64, step=8),
        activation="relu",
    )(x)

    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    roc_auc = tf.keras.metrics.AUC(name="roc_auc", curve="ROC")

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[roc_auc])
    return model




## === cell 12
def make_model_augmented(hp):
    input_shape = (32, 32, 1)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )

    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)

    x = keras.layers.Rescaling(1.0 / 255)(x)

    x = keras.layers.Conv2D(
        filters=hp.Int("units_Conv_1_" + str(0), min_value=64, max_value=256, step=32),
        kernel_size=(4, 4),
        activation="relu",
        name="Conv_1",
    )(x)

    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = keras.layers.Conv2D(
        filters=hp.Int("units_conv2_" + str(1), min_value=16, max_value=128, step=16),
        kernel_size=(2, 2),
        activation="relu",
        name="Conv_2",
    )(x)

    x = keras.layers.MaxPool2D(pool_size=(1, 1))(x)

    x = layers.Dropout(hp.Float("dense_dropout", min_value=0.0, max_value=0.7))(x)
    x = keras.layers.Flatten()(x)

    x = layers.Dense(
        units=hp.Int("num_dense_units", min_value=16, max_value=64, step=8),
        activation="relu",
    )(x)

    outputs = keras.layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    roc_auc = tf.keras.metrics.AUC(name="roc_auc", curve="ROC")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[roc_auc])
    return model




## === cell 13
import keras_tuner as kt


def make_model_siren(hp):
    inputs = keras.Input(shape=X_train.shape[1:])

    x = keras.layers.Rescaling(1.0 / 255)(inputs)

    x = SineConvLayer(
        features=hp.Int("features_conv_1", min_value=64, max_value=256, step=32),
        kernel_size=hp.Int("kernel_conv_1", min_value=2, max_value=7, step=1),
        is_first=True,
        omega_0=hp.Int("omega_0_conv_1", min_value=10, max_value=50, step=5),
    )(x)

    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = SineConvLayer(
        features=hp.Int("features_conv_2", min_value=16, max_value=128, step=16),
        kernel_size=hp.Int("kernel_conv_2", min_value=2, max_value=7, step=1),
        is_first=False,
        omega_0=hp.Int("omega_0_conv_2", min_value=10, max_value=50, step=5),
    )(x)

    x = keras.layers.MaxPool2D(pool_size=(1, 1))(x)

    x = layers.Dropout(hp.Float("dense_dropout", min_value=0.0, max_value=0.7))(x)
    x = keras.layers.Flatten()(x)

    x = SineDenseLayer(
        features=hp.Int("features_dense_1", min_value=64, max_value=256, step=32),
        is_first=False,
        omega_0=hp.Int("omega_0_dense_1", min_value=10, max_value=50, step=5),
    )(x)

    outputs = keras.layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    roc_auc = tf.keras.metrics.AUC(name="roc_auc", curve="ROC")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[roc_auc])
    return model




## === cell 14
tuner = kt.tuners.BayesianOptimization(
    make_model_augmented,
    objective="val_loss",
    max_trials=5,  # keep as original intent (quick run)
    overwrite=True,
    seed=SEED,
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=3, restore_best_weights=True
    )
]

tuner.search(
    X_train, y_train, validation_split=0.2, callbacks=callbacks, verbose=1, epochs=20
)



## === cell 15
best_hp = tuner.get_best_hyperparameters()[0]

best_model = make_model_augmented(best_hp)



## === cell 16
best_model.save("best_model.keras")



## === cell 17
history = best_model.fit(X_train, y_train, validation_split=0.2, epochs=50)



## === cell 18
y_pred_valid = best_model.predict(X_valid, verbose=0)
pred_prob_valid = y_pred_valid[:, 1]

result = pd.DataFrame(
    {"BraTS21ID": trainidt_valid.astype(int), "MGMT_value": pred_prob_valid}
)
result2 = result.groupby("BraTS21ID", as_index=False).mean()
result2



## === cell 19
result2 = result2.merge(
    train_df[["BraTS21ID", "MGMT_value"]],
    left_on="BraTS21ID",
    right_on=train_df["BraTS21ID"].astype(int),
    how="left",
    suffixes=("_pred", "_true"),
)

if "key_0" in result2.columns:
    result2 = result2.drop(columns=["key_0"])
if "BraTS21ID_true" in result2.columns and "BraTS21ID" in result2.columns:
    result2 = result2.drop(columns=["BraTS21ID_true"])

result2



## === cell 20
auc = roc_auc_score(
    result2["MGMT_value_true"],
    result2["MGMT_value_pred"],
)
print(f"Validation AUC={auc}")



## === cell 21
y_pred_test = best_model.predict(X_test_tf, verbose=0)
pred_prob_test = y_pred_test[:, 1]

test_result = pd.DataFrame(
    {"BraTS21ID": testidt.astype(int), "MGMT_value": pred_prob_test}
)
test_result2 = test_result.groupby("BraTS21ID", as_index=False).mean()

sub_ids = sample_submission[["BraTS21ID"]].copy()
sub_ids["BraTS21ID"] = sub_ids["BraTS21ID"].astype(int)

sub = sub_ids.merge(test_result2, on="BraTS21ID", how="left")

sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

sub["BraTS21ID"] = sub["BraTS21ID"].astype(int).astype(str).str.zfill(5)

sub.head()



## === cell 22
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub

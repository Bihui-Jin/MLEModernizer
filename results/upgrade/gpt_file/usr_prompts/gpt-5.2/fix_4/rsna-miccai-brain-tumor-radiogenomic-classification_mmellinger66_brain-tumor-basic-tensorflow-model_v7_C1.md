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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the two blocking runtime issues first: the TensorFlow import crash caused by an incompatible `protobuf` version, and the `pydicom.read_file` API removal (use `pydicom.dcmread`). Then I make the training/inference path run end-to-end by ensuring the model checkpoint path is valid in TF/Keras 2.18 (use `.keras`), and by aligning predictions with the ROC-AUC metric (use class-1 probability rather than `argmax`, and remove rounding that destroys calibration). Finally, I keep your core data-loading and CNN architecture intact while making the output submission match the sample format and ordering exactly, writing `submission.csv` in the working directory.'
- What this solution (achieved 0.5) has done: 'The immediate blocker is the TensorFlow import crash caused by an incompatible `protobuf` version in this environment; the only safe fix is to force protobuf’s pure-Python implementation before *any* TensorFlow/protobuf-related imports, and if needed pin the python implementation more explicitly. After that, the rest of your pipeline should run, but I also make sure the DICOM reader uses `pydicom.dcmread` (already correct) and that the submission IDs are consistently zero-padded to match the sample submission merge (avoids silent NaNs that would hurt AUC). These changes are runtime/stability and alignment fixes and should keep your core modeling/training logic intact while nudging score upward by preventing ID-mismatch dilution. The script write a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow import crash by pinning protobuf to the Python implementation *and* applying the compatibility shim for TF/TFP that avoids the `MessageFactory.GetPrototype` failure in this Kaggle image. Then I make the DICOM path sorting robust (some filenames may not follow the assumed pattern), which prevents occasional empty/failed loads that silently degrade training/inference. Finally, I keep your model/training/prediction logic unchanged and ensure the submission IDs/predictions align exactly to the sample submission and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if (
        hasattr(_message_factory, "MessageFactory")
        and not hasattr(_message_factory.MessageFactory, "GetPrototype")
        and hasattr(_message_factory.MessageFactory, "GetMessageClass")
    ):
        _message_factory.MessageFactory.GetPrototype = (
            _message_factory.MessageFactory.GetMessageClass
        )
except Exception:
    pass

import glob
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
import cv2
import pydicom

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("pydicom:", pydicom.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709]  # Bad images

np.random.seed(0)
random.seed(12)
tf.random.set_seed(12)



## === cell 2
train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)].reset_index(drop=True)

print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")
print(f"test data: Rows={test_df.shape[0]}, Columns={test_df.shape[1]}")




## === cell 3
def load_dicom(path, size=224):
    """
    Reads a DICOM image, standardizes pixel values to [0,1], rescales to [0,255],
    and resizes to (size, size).
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    mx = np.max(data)
    if mx != 0:
        data = data / mx

    data = (data * 255.0).astype(np.uint8)
    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)




## === cell 4
def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of all the images of a particular type for a particular patient ID.
    """
    assert image_type in mri_types

    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder,
        str(brats21id).zfill(5),
    )

    def _slice_key(p):
        base = os.path.basename(p)
        try:
            stem = os.path.splitext(base)[0]
            return int(stem.split("-")[-1])
        except Exception:
            return 10**9

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")), key=_slice_key
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


def get_all_images(brats21id, image_type, folder="train", size=225):
    paths = get_all_image_paths(brats21id, image_type, folder)
    return [load_dicom(path, size) for path in paths]




## === cell 5
def get_all_data_for_train(image_type, image_size=32):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index, desc=f"Loading train {image_type}"):
        x = train_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "train", image_size)
        label = int(x["MGMT_value"])

        if len(images) == 0:
            continue

        X += images
        y += [label] * len(images)
        train_ids += [int(x["BraTS21ID"])] * len(images)

        assert len(X) == len(y)

    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type, image_size=32):
    global test_df

    X = []
    test_ids = []

    for i in tqdm(test_df.index, desc=f"Loading test {image_type}"):
        x = test_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "test", image_size)

        if len(images) == 0:
            continue

        X += images
        test_ids += [int(x["BraTS21ID"])] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 6
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

print("Train image tensor:", X.shape, "labels:", y.shape, "train ids:", trainidt.shape)
print("Test image tensor:", X_test.shape, "test ids:", testidt.shape)



## === cell 7
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X, y, trainidt, test_size=0.2, random_state=42, stratify=y
)

X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)
X_test_tf = tf.expand_dims(X_test, axis=-1)

y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)
print("X_test:", X_test_tf.shape)




## === cell 8
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

    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )
    return model


def get_model02():
    inpt = keras.Input(shape=X_train.shape[1:])
    h = keras.layers.Rescaling(1.0 / 255)(inpt)

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


def get_model03():
    inpt = keras.Input(shape=X_train.shape[1:])
    h = keras.layers.Rescaling(1.0 / 255)(inpt)

    h = keras.layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
    h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

    h = keras.layers.Dropout(0.1)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(32, activation="relu")(h)

    output = keras.layers.Dense(2, activation="softmax")(h)

    model = keras.Model(inpt, output)

    initial_learning_rate = 0.1  # kept as-is to preserve core logic
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 9
checkpoint_filepath = "best_model.keras"

model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_freq="epoch",
    verbose=1,
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)



## === cell 10
model = get_model03()
model.summary()



## === cell 11
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=20,
    callbacks=[model_checkpoint_cb],
    validation_data=(X_valid, y_valid),
    verbose=2,
)



## === cell 12
model_best = tf.keras.models.load_model(checkpoint_filepath)

y_pred_valid = model_best.predict(X_valid, verbose=0)[:, 1]

result = pd.DataFrame({"BraTS21ID": trainidt_valid, "pred": y_pred_valid})
result2 = result.groupby("BraTS21ID", as_index=False)["pred"].mean()
result2 = result2.merge(train_df, on="BraTS21ID", how="left")

auc = roc_auc_score(result2["MGMT_value"], result2["pred"])
print(f"Validation AUC={auc:.5f}")



## === cell 13
y_pred_test = model_best.predict(X_test_tf, verbose=0)[:, 1]

test_pred_df = pd.DataFrame({"BraTS21ID": testidt, "MGMT_value": y_pred_test})
test_pred_df = test_pred_df.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

submission = sample_submission[["BraTS21ID"]].copy()
submission["BraTS21ID"] = submission["BraTS21ID"].astype(str).str.zfill(5)

test_pred_df["BraTS21ID"] = (
    test_pred_df["BraTS21ID"].astype(int).astype(str).str.zfill(5)
)

submission = submission.merge(test_pred_df, on="BraTS21ID", how="left")
submission["MGMT_value"] = submission["MGMT_value"].fillna(0.5).astype(float)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

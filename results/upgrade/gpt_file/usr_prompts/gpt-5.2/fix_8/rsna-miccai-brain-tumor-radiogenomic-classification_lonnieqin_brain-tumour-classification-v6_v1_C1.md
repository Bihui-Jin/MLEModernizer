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

0.51176

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.40353) has done: 'I fix the runtime crashes caused by package/API changes: the protobuf `MessageFactory.GetPrototype` issue (by safely pinning protobuf to the pure-Python implementation via environment variables before importing TensorFlow) and the `pydicom.read_file` removal (switching to `pydicom.dcmread`). I also fix a Keras 2.18 incompatibility by replacing the deprecated `keras.layers.experimental.preprocessing.Rescaling` with `keras.layers.Rescaling`, and ensure the checkpoint monitors the correct metric name (`val_auc`). Finally, I keep your training/inference logic the same but correct the submission to use predicted probabilities (not argmax class labels) so it matches ROC-AUC evaluation and produces a valid `submission.csv`.'
- What this solution (achieved 0.49176) has done: 'I fix the protobuf/TensorFlow crash by moving the environment-variable pinning to the very top (before any TensorFlow/protobuf-related imports) and by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle image. I also make the DICOM slice sorting more robust (handle unexpected filenames safely) and guard against rare empty-slice cases so the pipeline completes and always writes a valid `submission.csv`. Core model/training logic and inference averaging remain unchanged to preserve evaluation semantics and keep the score behavior aligned with your current 0.40353. The script run end-to-end and produce `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.49059) has done: 'I fix the protobuf/TensorFlow `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation **before any protobuf/TensorFlow import** and also by disabling the C++ implementation explicitly, which is the common Kaggle TF2.18 failure mode. I keep your model/training/inference logic unchanged, only making the import order/environment pinning robust and adding a lightweight fallback so the notebook always completes and writes `submission.csv`. I also ensure ID types are consistent (string zfill) at read time to avoid any accidental merge/reindex mismatches that can silently hurt AUC. These changes are score-neutral in intent, but the ID consistency can slightly improve effective scoring by preventing misalignment.'
- What this solution (achieved 0.45647) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import* and by removing the conflicting disable flag that can still trigger the C++ path in this Kaggle image. I also make the model checkpoint filename compatible with TF/Keras 2.18 (use a `.keras` model file to avoid HDF5 save/load edge cases) while keeping the same monitored metric (`val_auc`) and training loop unchanged. Finally, I add a small safety fallback so that if the checkpoint isn’t written (e.g., metric name mismatch), inference still run using the in-memory model and a valid `submission.csv` always be produced. These are stability fixes intended to be score-neutral (your ROC-AUC alignment via probability outputs is preserved).'
- What this solution (achieved 0.53412) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation at the very top of the script (before any TF/protobuf-related import) and by also disabling the C++ implementation explicitly, which is the common cause of `MessageFactory.GetPrototype` errors in this environment. I keep your data loading, model, training loop, and inference logic unchanged, only adjusting import order and environment pinning so it runs end-to-end. I also add a tiny safety check to ensure the `submission.csv` is always written with the exact required columns and correct ID formatting even if something unexpected happens during prediction alignment. These changes are intended to be score-neutral (or slightly positive only by preventing crashes/misalignment), keeping your current modeling semantics intact.'
- What this solution (achieved 0.52588) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow-related import* and by also setting the standard `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which avoids the `MessageFactory.GetPrototype` error in this Kaggle image. I keep your model, training loop, and inference logic intact, only making import ordering and a couple of safety checks more robust so the pipeline always completes. I also ensure IDs are consistently zero-padded strings at submission time to avoid any silent misalignment. These changes are intended to be score-neutral while guaranteeing end-to-end execution and a valid `submission.csv`.'
- What this solution (achieved 0.51176) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before any TensorFlow/protobuf-related import, which is the root cause in this Kaggle image. I also remove the unintended performance limiter by disabling early stopping (it was monitoring `val_accuracy` while the checkpoint monitors `val_auc`, so training could stop early and hurt AUC), while keeping the same model architecture, optimizer, loss, epochs, and checkpoint behavior. Finally, I keep your existing submission alignment logic but add small safety checks so prediction arrays and IDs always align and a valid `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION", "1")

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
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # out of 255
EXCLUDE = [109, 123, 709]

DATA_DIR = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_df = pd.read_csv(
    os.path.join(DATA_DIR, "train_labels.csv"), dtype={"BraTS21ID": str}
)
test_df = pd.read_csv(
    os.path.join(DATA_DIR, "sample_submission.csv"), dtype={"BraTS21ID": str}
)

exclude_str = {str(x).zfill(5) for x in EXCLUDE}
train_df = train_df[~train_df.BraTS21ID.isin(exclude_str)].reset_index(drop=True)

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print(train_df.head())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_dicom(path, size=224):
    """
    Reads a DICOM image, standardizes so pixel values are between 0 and 1, then rescales to 0..255.
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    mx = np.max(data)
    if mx != 0:
        data = data / mx

    data = (data * 255.0).astype(np.uint8)
    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)


def _safe_slice_index_from_path(p):
    base = os.path.splitext(os.path.basename(p))[0]
    try:
        return int(base.split("-")[-1])
    except Exception:
        return 10**18  # push unparseable to the end


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of all image file paths of a given type for a given patient ID.
    Uses mid-slices (25%..75%) with step interval to reduce redundancy.
    """
    assert image_type in TYPES

    patient_path = os.path.join(DATA_DIR, folder, str(int(brats21id)).zfill(5))
    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_safe_slice_index_from_path,
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([])

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)
    if end <= start:
        start = 0
        end = num_images

    interval = 3
    if num_images < 10:
        interval = 1

    sel = paths[start:end:interval]
    if len(sel) == 0:
        sel = paths[::interval] if interval > 0 else paths
    return np.array(sel)


def get_all_images(brats21id, image_type, folder="train", size=225):
    paths = get_all_image_paths(brats21id, image_type, folder)
    if len(paths) == 0:
        return []
    return [load_dicom(path, size) for path in paths]




## === cell 2
IMAGE_SIZE = 32


def get_all_data_for_train(image_type):
    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        brats_id = int(x["BraTS21ID"])
        images = get_all_images(brats_id, image_type, "train", IMAGE_SIZE)
        label = int(x["MGMT_value"])

        if len(images) == 0:
            continue

        X += images
        y += [label] * len(images)
        train_ids += [brats_id] * len(images)
        assert len(X) == len(y)

    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type):
    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        x = test_df.loc[i]
        brats_id = int(x["BraTS21ID"])
        images = get_all_images(brats_id, image_type, "test", IMAGE_SIZE)

        if len(images) == 0:
            continue

        X += images
        test_ids += [brats_id] * len(images)

    return np.array(X), np.array(test_ids)




## === cell 3
X, y, trainidt = get_all_data_for_train("T1wCE")
X_test, testidt = get_all_data_for_test("T1wCE")
print("X:", X.shape, "y:", y.shape, "trainidt:", trainidt.shape)
print("X_test:", X_test.shape, "testidt:", testidt.shape)



## === cell 4
if len(X) == 0 or len(X_test) == 0:
    sample = pd.read_csv(
        os.path.join(DATA_DIR, "sample_submission.csv"), dtype={"BraTS21ID": str}
    )
    sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)
    sample["MGMT_value"] = 0.5
    sample.to_csv("submission.csv", index=False)
    print("No images found; wrote default submission.csv with shape:", sample.shape)
    raise SystemExit(0)

X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X, y, trainidt, test_size=0.2, random_state=42, stratify=y
)

X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)
X_test_infer = tf.expand_dims(X_test, axis=-1)

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

print(
    X_train.shape,
    y_train.shape,
    X_valid.shape,
    y_valid.shape,
    trainidt_train.shape,
    trainidt_valid.shape,
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
h = keras.layers.Dense(32, activation="relu")(h)

output = keras.layers.Dense(2, activation="softmax")(h)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=20, restore_best_weights=False
)

model = keras.Model(inputs, output)
print(model.summary())

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
    metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
)

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    callbacks=[model_checkpoint_callback],  # intentionally omit early_stopping
    validation_data=(X_valid, y_valid),
    verbose=2,
)



## === cell 6
if os.path.exists(checkpoint_filepath):
    model_best = tf.keras.models.load_model(checkpoint_filepath)
else:
    model_best = model
    print(
        f"Warning: checkpoint {checkpoint_filepath} not found; using in-memory model for inference."
    )



## === cell 7
y_pred_valid = model_best.predict(X_valid, verbose=0)
if y_pred_valid.ndim != 2 or y_pred_valid.shape[1] != 2:
    raise ValueError(f"Unexpected valid prediction shape: {y_pred_valid.shape}")
prob_valid = y_pred_valid[:, 1]

result = pd.DataFrame({"BraTS21ID": trainidt_valid, "MGMT_value": prob_valid})
result2 = result.groupby("BraTS21ID", as_index=False).mean()
result2 = result2.merge(
    train_df.assign(BraTS21ID=train_df["BraTS21ID"].astype(int)),
    on="BraTS21ID",
    suffixes=("_pred", "_true"),
)

auc = roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])
print("Patient-level valid AUC:", auc)
print("Patients in valid:", len(result2))



## === cell 8
sample = pd.read_csv(
    os.path.join(DATA_DIR, "sample_submission.csv"), dtype={"BraTS21ID": str}
)

y_pred_test = model_best.predict(X_test_infer, verbose=0)
if y_pred_test.ndim != 2 or y_pred_test.shape[1] != 2:
    raise ValueError(f"Unexpected test prediction shape: {y_pred_test.shape}")
prob_test = y_pred_test[:, 1]

result = pd.DataFrame({"BraTS21ID": testidt, "MGMT_value": prob_test})
result2 = result.groupby("BraTS21ID", as_index=False).mean()

result2["BraTS21ID"] = result2["BraTS21ID"].astype(str).str.zfill(5)
sample_ids = sample["BraTS21ID"].astype(str).str.zfill(5)

result2 = result2.set_index("BraTS21ID").reindex(sample_ids).reset_index()
result2.columns = ["BraTS21ID", "MGMT_value"]

result2["MGMT_value"] = result2["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

result2 = result2[["BraTS21ID", "MGMT_value"]]
result2.to_csv("submission.csv", index=False)

print(result2.head())
print("Wrote submission.csv with shape:", result2.shape)

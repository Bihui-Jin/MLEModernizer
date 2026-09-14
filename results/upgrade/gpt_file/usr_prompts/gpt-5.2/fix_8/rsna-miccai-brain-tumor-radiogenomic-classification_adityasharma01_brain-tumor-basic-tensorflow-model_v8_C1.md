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

0.46941

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50118) has done: 'I fix the environment-breaking import error by ensuring `protobuf` uses the pure-Python implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` crash. Then I fix the DICOM loader for pydicom v3 by replacing the removed `pydicom.read_file` with `pydicom.dcmread`, and add small guards so missing/failed slices don’t crash the pipeline. Finally, I correct the prediction-to-probability logic (use the softmax probability for class 1, not `argmax`) and ensure the submission aligns with `sample_submission.csv` and writes a valid `submission.csv`. These changes are minimal, keep the same CNN training loop/architecture, and should improve AUC because the competition expects probabilities.'
- What this solution (achieved 0.50235) has done: 'I fix the TensorFlow/protobuf crash by setting the required environment variables before importing TensorFlow and by switching the TF-Probability backend to TensorFlow (to avoid the `MessageFactory.GetPrototype` path). I keep the model/training logic unchanged, but I correct the input pipeline bug where the CNN expects 2D images yet the data is expanded to 3D (extra channel dim), which can silently break training or crash depending on TF version. I also make test/train ID typing consistent with the sample submission (string zero-padded IDs) to prevent merge/alignment issues that can harm score or yield NaNs. These are stability/correctness fixes; they should keep semantics the same while preventing runtime errors and improving the validity of the produced probabilities.'
- What this solution (achieved 0.50941) has done: 'I fix the TensorFlow/protobuf import crash that currently stops execution by forcing the pure-Python protobuf implementation and ensuring TensorFlow Probability isn’t imported/used in a way that triggers the broken `MessageFactory.GetPrototype` path. I also make the “excluded images” filter actually work (it currently compares ints to zero-padded strings, so nothing is excluded) and keep IDs consistently typed to avoid silent merge/alignment issues. These changes are stability/correctness-focused and keep the same model/training/prediction logic, while slightly improving training data quality by correctly dropping the known-bad cases. The script still train the same 2D CNN and write a valid `submission.csv` in the required format.'
- What this solution (achieved 0.46706) has done: 'I fix the TensorFlow/protobuf crash by setting the additional environment variable that forces the pure-Python protobuf implementation before TensorFlow is imported (this is the root cause of the `MessageFactory.GetPrototype` failure in this environment). I also fix the DICOM path sorting logic so it robustly parses slice numbers from filenames like `Image-387.dcm` (the current key function can fail or mis-sort), without changing what images are selected. Finally, I fix the submission ID formatting check by reading `BraTS21ID` as a zero-padded string (pandas otherwise infer it as int and drop leading zeros, causing the current assertion to fail), while keeping the submission file content/format unchanged.'
- What this solution (achieved 0.61765) has done: 'The run currently fails immediately due to a TensorFlow/protobuf incompatibility that your existing environment-variable workaround is not fully preventing. I fix this by pinning protobuf to its pure-Python runtime *and* forcing TF to avoid the C++ protobuf path before TensorFlow (and any transitive protobuf users) import, which resolves the `MessageFactory.GetPrototype` crash in this Kaggle image. I keep the model, training loop, and preprocessing logic unchanged, only adding a safe fallback import path and a small GPU-memory-growth guard to improve stability without affecting evaluation semantics. The submission-writing logic remains the same and still produce a valid `submission.csv` in the required format.'
- What this solution (achieved 0.47765) has done: 'I fix the immediate runtime crash caused by a TensorFlow/protobuf incompatibility by explicitly selecting the pure-Python protobuf implementation *before* any TensorFlow-related imports and by importing TensorFlow through a guarded path that avoids the failing `MessageFactory.GetPrototype` codepath in this environment. Then I ensure ID typing/zero-padding stays consistent end-to-end so merges and submission alignment cannot silently drop rows. These changes are stability/correctness focused and keep the same data loading, 2D CNN model, training loop, and probability extraction logic, so they should preserve semantics while letting the notebook run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.46941) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf runtime is selected as early as possible and by avoiding the incompatible protobuf 6.x version in this Kaggle image (TensorFlow 2.18 expects protobuf < 6). Then I keep your training/inference logic unchanged, but make the imports robust so the notebook runs end-to-end and always writes `submission.csv`. These changes are score-neutral (they only unblock execution); your existing probability extraction and submission alignment remain as-is. I also add a small defensive check around the protobuf downgrade so it won’t fail if the environment already has a compatible version.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_C_DESCRIPTORS"] = "0"
os.environ.setdefault(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C_DESCRIPTORS", "1"
)

os.environ.setdefault("TFP_BACKEND", "tensorflow")

import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version

        v = version("protobuf")
        major = int(v.split(".")[0])
        if major >= 6:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.25.3,<6"]
            )
            import importlib
            import google.protobuf

            importlib.reload(google.protobuf)
    except Exception as e:
        print(
            "Warning: protobuf compatibility check/install did not complete:", repr(e)
        )


_ensure_compatible_protobuf()

import glob
import pandas as pd
import numpy as np
from pathlib import Path
import random
from tqdm.notebook import tqdm
import pydicom  # Handle MRI images
import cv2  # OpenCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

np.random.seed(0)
random.seed(12)
tf.random.set_seed(12)

print("TF version:", tf.__version__)
print("Protobuf impl:", os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"))
print(
    "Use C descriptors:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_C_DESCRIPTORS"),
)



## === cell 1
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709]  # Bad images



## === cell 2
train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df["_id_int"] = train_df["BraTS21ID"].astype(int)
train_df = train_df[~train_df["_id_int"].isin(excluded_images)].reset_index(drop=True)
train_df = train_df.drop(columns=["_id_int"])

print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")
print(f"test data: Rows={test_df.shape[0]}, Columns={test_df.shape[1]}")




## === cell 3
def load_dicom(path, size=350):
    """
    Reads a DICOM image, standardizes pixel values to [0,1], then rescales to [0,255] uint8.
    Fix: pydicom v3 removed read_file; use dcmread.
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    mx = float(np.max(data)) if data.size else 0.0
    if mx > 0:
        data = data / mx
    data = (data * 255.0).clip(0, 255).astype(np.uint8)

    if data.ndim != 2:
        data = data.squeeze()
        if data.ndim != 2:
            data = data.reshape((data.shape[0], data.shape[1]))
    return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)




## === cell 4
def _slice_number_from_path(p: str) -> int:
    """
    Fix: robustly extract slice number from filenames like 'Image-387.dcm'.
    """
    base = os.path.basename(p)
    stem, _ext = os.path.splitext(base)
    if "-" in stem:
        tail = stem.split("-")[-1]
        if tail.isdigit():
            return int(tail)
    return 0


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of all the images of a particular type for a particular patient ID.
    """
    assert image_type in mri_types

    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder,
        str(int(brats21id)).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_slice_number_from_path,
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    sliced = paths[start:end:interval]
    return np.array(sliced, dtype=object)


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
def get_all_data_for_train(image_type, image_size=430):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        pid = int(x["BraTS21ID"])
        images = get_all_images(pid, image_type, "train", image_size)
        if len(images) == 0:
            continue
        label = int(x["MGMT_value"])

        X += images
        y += [label] * len(images)
        train_ids += [pid] * len(images)

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int64)
    train_ids = np.array(train_ids, dtype=np.int64)

    assert len(X) == len(y) == len(train_ids)
    return X, y, train_ids




## === cell 6
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




## === cell 7
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

print("Train arrays:", X.shape, y.shape, trainidt.shape)
print("Test arrays:", X_test.shape, testidt.shape)



## === cell 8
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X,
    y,
    trainidt,
    random_state=12,
    test_size=0.25,
    stratify=y if len(np.unique(y)) > 1 else None,
)

X_train = np.expand_dims(X_train, axis=-1)
X_valid = np.expand_dims(X_valid, axis=-1)
X_test_tf = np.expand_dims(X_test, axis=-1)

y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)




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
def get_model02(input_shape):
    inpt = keras.Input(shape=input_shape)

    h = keras.layers.Rescaling(1.0 / 255.0)(inpt)

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
model = get_model02(input_shape=tuple(X_train.shape[1:]))

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=25,
    callbacks=[model_checkpoint_callback],
    validation_data=(X_valid, y_valid),
    verbose=2,
)



## === cell 13
model_best = tf.keras.models.load_model(filepath=checkpoint_filepath)



## === cell 14
y_pred_valid = model_best.predict(X_valid, verbose=0)
pred_prob_valid = y_pred_valid[:, 1].astype(np.float32)

result = pd.DataFrame(
    {"BraTS21ID": trainidt_valid.astype(int), "MGMT_value": pred_prob_valid}
)
result2 = result.groupby("BraTS21ID", as_index=False).mean()

train_df_merge = train_df.copy()
train_df_merge["BraTS21ID"] = train_df_merge["BraTS21ID"].astype(int)

result2 = result2.merge(
    train_df_merge, on="BraTS21ID", how="inner", suffixes=("_pred", "_true")
)
auc = roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])
print(f"Validation AUC={auc:.6f}")



## === cell 15
y_pred_test = model_best.predict(X_test_tf, verbose=0)
pred_prob_test = y_pred_test[:, 1].astype(np.float32)

result = pd.DataFrame({"BraTS21ID": testidt.astype(int), "MGMT_value": pred_prob_test})
result2 = result.groupby("BraTS21ID", as_index=False).mean()

sub = sample_submission[["BraTS21ID"]].copy()
sub["BraTS21ID"] = sub["BraTS21ID"].astype(int)

result2["BraTS21ID"] = result2["BraTS21ID"].astype(int)
sub = sub.merge(result2, on="BraTS21ID", how="left")

sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).clip(0.0, 1.0)
sub["BraTS21ID"] = sub["BraTS21ID"].apply(lambda x: str(int(x)).zfill(5))

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 16
assert os.path.exists("submission.csv")

check = pd.read_csv("submission.csv", dtype={"BraTS21ID": str})
assert list(check.columns) == ["BraTS21ID", "MGMT_value"]
assert check["MGMT_value"].between(0, 1).all()
assert check["BraTS21ID"].astype(str).str.len().eq(5).all()
print("submission.csv is valid.")

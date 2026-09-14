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

3.9

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

0.43647

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.41765) has done: 'I fix the environment/import crash caused by an incompatible `protobuf`/TF stack by pinning `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow. Then I fix the DICOM loader for pydicom v3 (`dcmread` instead of removed `read_file`) and correct the excluded IDs filtering (IDs are zero-padded strings in the CSV). Finally, I make the training/inference pipeline run end-to-end with minimal semantic changes: ensure model preprocessing uses the non-deprecated `layers.Rescaling`, and output a valid submission with probabilities (use the softmax class-1 probability) without rounding so AUC isn’t harmed.'
- What this solution (achieved 0.48118) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation env var earlier and, crucially, forcing TensorFlow to use the pure-Python protobuf backend even if the C++ one is already selected. Then I add a small compatibility fallback to avoid hard failure if TF still triggers the `MessageFactory.GetPrototype` issue in this environment. Finally, I keep the model/training/inference logic the same, only ensuring the pipeline always reaches the submission write step and produces a valid `submission.csv` with the required columns and probability values.'
- What this solution (achieved 0.48235) has done: 'I fix the TensorFlow/protobuf crash that’s preventing the notebook from running by applying a safe, Kaggle-compatible monkey-patch for the missing `MessageFactory.GetPrototype` (the root cause of your current runtime error), while keeping the rest of your pipeline unchanged. I also ensure the protobuf environment variables are set before any protobuf/TensorFlow import and add deterministic seeds without altering training semantics. Finally, I keep your exact data loading, model, aggregation, and submission formatting logic intact so the score behavior is driven by the same approach, just unblocked and stable end-to-end.'
- What this solution (achieved 0.49882) has done: 'I fix the TensorFlow/protobuf crash by applying a more robust monkey-patch that also covers the instance-method call site (so `MessageFactory().GetPrototype(...)` won’t raise), while keeping your training/inference logic unchanged. I also force the protobuf pure-Python backend *before any protobuf import* and add a safe fallback if protobuf internals differ, so the notebook consistently runs end-to-end. Since your current score (0.48235) is already far above the given target (-1.0), I not make any score-improving changes; the rest stays identical except for stability fixes and ensuring `submission.csv` is written.'
- What this solution (achieved 0.37176) has done: 'I fix the crash in the protobuf/TensorFlow compatibility patch by monkey‑patching both the `MessageFactory` class and the default factory *instance* so `GetPrototype` always exists before TensorFlow imports. This is a pure runtime-stability fix and does not change your model, training loop, data pipeline, or prediction logic, so your score behavior should remain essentially the same while the notebook runs end-to-end. I also move the patch to run unconditionally and safely even if protobuf internals differ, so it won’t intermittently fail depending on import order. Finally, I keep the submission writing exactly as required and ensure the CSV is always produced.'
- What this solution (achieved 0.44824) has done: 'I fix the crash happening before TensorFlow imports by making the protobuf `MessageFactory.GetPrototype` monkey‑patch robust for both the class and the default factory instance, without calling any missing attribute. I also ensure `MessageFactory` (from either `google.protobuf.message_factory` or `google.protobuf.message_factory` internals) is patched in a way that matches protobuf v6 behavior so TensorFlow can import cleanly. These changes are runtime-stability only and don’t alter your data pipeline, model, training loop, or prediction logic, so the score should remain essentially unchanged (your current score is already well above the given target). Finally, I keep the exact submission formatting and guarantee `submission.csv` is written.'
- What this solution (achieved 0.50235) has done: 'Your current score (0.44824 AUC) is already far above the provided target score (-1.0), so the only way to move “toward the target” is to *decrease* performance, but producing deliberately worse predictions would be artificial and not a legitimate improvement. Instead, I keep your core training/inference pipeline identical and focus on stability + correctness that won’t intentionally optimize AUC: ensure IDs are read from the real `test/` folder (not `sample_submission.csv` reused as a “test dataframe”), force consistent float32 tensors for TF, and add a safety fallback so every test subject gets a prediction even if a subject has zero slices. This preserves semantics (same model, same training loop, same aggregation) while reducing the risk of accidental leakage/format mismatch and ensuring a valid submission is always written.'
- What this solution (achieved 0.43647) has done: 'Your target score (-1.0) is unreachable for AUC (it’s bounded in [0, 1]), so the closest achievable value to -1.0 is actually 0.0; since your current score is 0.50235 (already far from 0.0), the only way to move “toward the target” would be to intentionally make predictions worse, which is not a legitimate improvement. Instead, I keep your modeling/training logic intact and make only correctness/stability fixes that can prevent accidental score loss: ensure deterministic test ID ordering matches the sample submission, avoid NaNs by guaranteeing every test ID gets a prediction (even if no slices), and fix a subtle size inconsistency (32 vs 225 default) by aligning the default `get_all_images` size to the caller-provided `image_size`. These changes don’t optimize AUC; they mainly reduce risk of misalignment/empty-case artifacts and ensure the submission is consistently valid.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
from pathlib import Path
import random

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pydicom  # Handle MRI images
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    from google.protobuf import message_factory as _message_factory

    MF = getattr(_message_factory, "MessageFactory", None)

    def _safe_add_getprototype_to_factory_class(MF_cls):
        if MF_cls is None:
            return
        if hasattr(MF_cls, "GetPrototype"):
            return
        if hasattr(MF_cls, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            MF_cls.GetPrototype = _GetPrototype
            return

        def _GetPrototype_stub(self, descriptor):
            raise AttributeError(
                "MessageFactory has neither GetPrototype nor GetMessageClass; cannot patch."
            )

        MF_cls.GetPrototype = _GetPrototype_stub

    _safe_add_getprototype_to_factory_class(MF)

    default_factory = getattr(_message_factory, "message_factory", None)
    if default_factory is not None and not hasattr(default_factory, "GetPrototype"):
        if hasattr(default_factory, "GetMessageClass"):

            def _inst_GetPrototype(descriptor, _df=default_factory):
                return _df.GetMessageClass(descriptor)

            default_factory.GetPrototype = _inst_GetPrototype

except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

np.random.seed(0)
random.seed(12)
tf.random.set_seed(12)



## === cell 1
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709]  # Bad images (as ints)



## === cell 2
train_df = pd.read_csv(data_dir / "train_labels.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

sample_submission["BraTS21ID"] = sample_submission["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sample_submission["BraTS21ID"].tolist()
test_df = pd.DataFrame({"BraTS21ID": test_ids})

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(str).str.zfill(5)

excluded_ids = {str(x).zfill(5) for x in excluded_images}
train_df = train_df[~train_df.BraTS21ID.isin(excluded_ids)].reset_index(drop=True)

print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")
print(train_df.head())
print(f"test ids: Rows={test_df.shape[0]}, Columns={test_df.shape[1]}")
print(test_df.head())




## === cell 3
def load_dicom(path, size=224):
    """
    Reads a DICOM image, standardizes pixel values to [0, 1], rescales to [0, 255], resizes.
    Compatibility fix: pydicom.read_file was removed in pydicom v3; use pydicom.dcmread.
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    mx = float(np.max(data)) if data.size else 0.0
    if mx > 0:
        data = data / mx
    data = (data * 255.0).clip(0, 255).astype(np.uint8)

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

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=lambda x: int(os.path.splitext(x)[0].split("-")[-1]),
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


def get_all_images(brats21id, image_type, folder="train", size=32):
    paths = get_all_image_paths(brats21id, image_type, folder)
    return [load_dicom(path, size) for path in paths]




## === cell 5
def get_all_data_for_train(image_type, image_size=32):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(range(len(train_df))):
        x = train_df.loc[i]
        pid = x["BraTS21ID"]
        images = get_all_images(pid, image_type, "train", image_size)
        label = int(x["MGMT_value"])

        X += images
        y += [label] * len(images)
        train_ids += [pid] * len(images)

    return np.array(X), np.array(y), np.array(train_ids)




## === cell 6
def get_all_data_for_test(image_type, image_size=32):
    global test_df

    X = []
    test_ids_local = []

    for i in tqdm(range(len(test_df))):
        x = test_df.loc[i]
        pid = x["BraTS21ID"]
        images = get_all_images(pid, image_type, "test", image_size)
        X += images
        test_ids_local += [pid] * len(images)

    return np.array(X), np.array(test_ids_local)




## === cell 7
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)

print("Loaded shapes:", X.shape, y.shape, trainidt.shape, X_test.shape, testidt.shape)



## === cell 8
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X,
    y,
    trainidt,
    test_size=0.2,
    random_state=42,
    stratify=y if len(np.unique(y)) > 1 else None,
)



## === cell 9
X_train = tf.cast(tf.expand_dims(X_train, axis=-1), tf.float32)
X_valid = tf.cast(tf.expand_dims(X_valid, axis=-1), tf.float32)
X_test_tf = tf.cast(tf.expand_dims(X_test, axis=-1), tf.float32)

print("Tensor shapes:", X_train.shape, X_valid.shape, X_test_tf.shape)



## === cell 10
y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

print("y shapes:", y_train.shape, y_valid.shape)




## === cell 11
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




## === cell 12
def get_model02(input_shape):
    np.random.seed(0)
    random.seed(12)
    tf.random.set_seed(12)

    inpt = keras.Input(shape=input_shape)

    h = layers.Rescaling(1.0 / 255.0)(inpt)

    h = layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(h)
    h = layers.MaxPool2D(pool_size=(2, 2))(h)

    h = layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
    h = layers.MaxPool2D(pool_size=(1, 1))(h)

    h = layers.Dropout(0.1)(h)

    h = layers.Flatten()(h)
    h = layers.Dense(32, activation="relu")(h)

    output = layers.Dense(2, activation="softmax")(h)

    model = keras.Model(inpt, output)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 13
checkpoint_filepath = "best_model.keras"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1,
)



## === cell 14
model = get_model02(input_shape=tuple(X_train.shape[1:]))

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=20,
    callbacks=[model_checkpoint_callback],
    validation_data=(X_valid, y_valid),
    verbose=2,
)



## === cell 15
if os.path.exists(checkpoint_filepath):
    model_best = tf.keras.models.load_model(checkpoint_filepath)
else:
    model_best = model



## === cell 16
y_pred_valid = model_best.predict(X_valid, verbose=0)
pred_valid_prob = y_pred_valid[:, 1]

result = pd.DataFrame({"BraTS21ID": trainidt_valid, "MGMT_value": pred_valid_prob})
result2 = result.groupby("BraTS21ID", as_index=False).mean()

result2 = result2.merge(
    train_df, on="BraTS21ID", how="left", suffixes=("_pred", "_true")
)
auc = roc_auc_score(result2["MGMT_value_true"], result2["MGMT_value_pred"])
print(f"Validation AUC={auc:.5f}")



## === cell 17
if X_test_tf.shape[0] > 0:
    y_pred_test = model_best.predict(X_test_tf, verbose=0)
    pred_test_prob = y_pred_test[:, 1]

    result_test = pd.DataFrame({"BraTS21ID": testidt, "MGMT_value": pred_test_prob})
    result_test_agg = result_test.groupby("BraTS21ID", as_index=False).mean()
else:
    result_test_agg = pd.DataFrame(columns=["BraTS21ID", "MGMT_value"])

result_test_agg = test_df[["BraTS21ID"]].merge(
    result_test_agg, on="BraTS21ID", how="left"
)
result_test_agg["MGMT_value"] = result_test_agg["MGMT_value"].fillna(0.5).astype(float)



## === cell 18
sub = sample_submission[["BraTS21ID"]].merge(
    result_test_agg, on="BraTS21ID", how="left"
)

sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(float)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

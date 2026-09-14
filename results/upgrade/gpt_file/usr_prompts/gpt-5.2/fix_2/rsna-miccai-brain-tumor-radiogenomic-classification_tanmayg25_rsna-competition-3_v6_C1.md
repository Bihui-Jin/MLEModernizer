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
import math
import warnings

import numpy as np
import pandas as pd
import cv2
import pydicom

warnings.filterwarnings("ignore")

BASE_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_CSV = os.path.join(BASE_DIR, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

BAD_IDS = {"00109", "00123", "00709"}  # per competition note

train_patients = sorted(
    [p for p in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, p))]
)
test_patients = sorted(
    [p for p in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, p))]
)

print("n_train_folders:", len(train_patients), "n_test_folders:", len(test_patients))




## === cell 1
def average(l):
    return sum(l) / len(l)


def layers(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


m2 = [
    22,
    23,
    24,
    36,
    37,
    38,
    39,
    40,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    65,
    66,
    67,
    68,
    69,
    70,
    81,
    82,
    83,
    84,
    97,
    98,
]
m3 = [19, 20, 21, 25, 26, 33, 34, 35, 41, 42, 49]
m4 = [18, 27, 28]
m5 = [17]


def adjuster(file):
    if len(file) in m5:
        n = 5
    elif len(file) in m4:
        n = 4
    elif len(file) in m3:
        n = 3
    elif len(file) in m2:
        n = 2
    else:
        n = 1
    new_file = []
    for i in range(len(file)):
        for _ in range(n):
            new_file.append(file[i])
    return new_file


def decider(length, layer_number):
    return math.ceil(length / layer_number)


def adjuster2(layers_list, layer_number):
    if len(layers_list) == layer_number - 1:
        layers_list.append(layers_list[-1])


def _dcm_sort_key(dcm_path):
    """
    Bugfix: pydicom.read_file is deprecated/removed; use dcmread.
    Robust sorting: prefer ImagePositionPatient[2], fallback to InstanceNumber.
    """
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        try:
            return float(ipp[2])
        except Exception:
            pass
    inst = getattr(ds, "InstanceNumber", None)
    if inst is not None:
        try:
            return int(inst)
        except Exception:
            pass
    base = os.path.basename(dcm_path)
    nums = "".join([c if c.isdigit() else " " for c in base]).split()
    return int(nums[-1]) if nums else 0


def load_series_resized(series_dir, size=(64, 64)):
    files = [f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    full_paths = [os.path.join(series_dir, f) for f in files]
    full_paths.sort(key=_dcm_sort_key)

    imgs = []
    for fp in full_paths:
        ds = pydicom.dcmread(fp, force=True)
        arr = ds.pixel_array.astype(np.float32)
        arr = cv2.resize(arr, size)
        imgs.append(arr)
    return imgs


def build_patient_scan(root_dir, patient_id, layer_number=16):
    """
    Re-implements your commented training extraction verbatim in a function,
    preserving the same averaging/chunking/stacking logic.
    Output shape: (layer_number, 64, 64, 4)
    """
    scan = []
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
    series = {}

    for mod in modalities:
        series_dir = os.path.join(root_dir, patient_id, mod)
        imgs = load_series_resized(series_dir, size=(64, 64))

        new_imgs = []
        imgs = adjuster(imgs)
        layer_size = decider(len(imgs), layer_number)

        for layer_chunk in layers(imgs, layer_size):
            layer_chunk = list(map(average, zip(*layer_chunk)))
            new_imgs.append(layer_chunk)

        adjuster2(new_imgs, layer_number)
        if len(new_imgs) < layer_number:
            new_imgs += [new_imgs[-1]] * (layer_number - len(new_imgs))
        if len(new_imgs) > layer_number:
            new_imgs = new_imgs[:layer_number]

        series[mod] = new_imgs

    for i in range(layer_number):
        x = []
        for j in range(64):
            y = []
            for k in range(64):
                channel = [
                    series["FLAIR"][i][j][k],
                    series["T1w"][i][j][k],
                    series["T1wCE"][i][j][k],
                    series["T2w"][i][j][k],
                ]
                y.append(channel)
            x.append(y)
        scan.append(np.array(x, dtype=np.float32))

    return np.array(scan, dtype=np.float32)




## === cell 2
labels_df = pd.read_csv(LABELS_CSV, dtype={"BraTS21ID": str})
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].str.zfill(5)

labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_IDS)].reset_index(drop=True)

available_train = set(train_patients)
labels_df = labels_df[labels_df["BraTS21ID"].isin(available_train)].reset_index(
    drop=True
)

train_ids = labels_df["BraTS21ID"].tolist()
Output_Values = labels_df["MGMT_value"].astype(np.float32).values.reshape(-1, 1)

print("n_train_used:", len(train_ids), "Output_Values shape:", Output_Values.shape)



## === cell 3
CACHE_X = "/kaggle/working/Input_Values.npy"
CACHE_Y = "/kaggle/working/Output_Values.npy"
CACHE_IDS = "/kaggle/working/train_ids.npy"

layer_number = 16

if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y) and os.path.exists(CACHE_IDS):
    Input_Values = np.load(CACHE_X, allow_pickle=False)
    Output_Values = np.load(CACHE_Y, allow_pickle=False)
    cached_ids = np.load(CACHE_IDS, allow_pickle=True).tolist()
    if cached_ids != train_ids or len(Input_Values) != len(Output_Values):
        print("Cache mismatch; rebuilding Input_Values...")
        os.remove(CACHE_X)
        os.remove(CACHE_Y)
        os.remove(CACHE_IDS)
        Input_Values = None
else:
    Input_Values = None

if Input_Values is None:
    Input_Values_list = []
    for idx, pid in enumerate(train_ids, 1):
        scan = build_patient_scan(TRAIN_DIR, pid, layer_number=layer_number)
        Input_Values_list.append(scan)
        if idx % 25 == 0 or idx == len(train_ids):
            print(f"built {idx}/{len(train_ids)}")

    Input_Values = np.array(Input_Values_list, dtype=np.float32)
    np.save(CACHE_X, Input_Values)
    np.save(CACHE_Y, Output_Values.astype(np.float32))
    np.save(CACHE_IDS, np.array(train_ids, dtype=object))

print("Input_Values shape:", Input_Values.shape, "dtype:", Input_Values.dtype)
print("Output_Values shape:", Output_Values.shape, "dtype:", Output_Values.dtype)



## === cell 4
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv3D,
    MaxPooling3D,
    BatchNormalization,
    Dropout,
    Dense,
    GlobalAveragePooling3D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

print("TensorFlow version:", tf.__version__)



## === cell 5
model = Sequential()
model.add(
    Conv3D(
        32,
        (3, 3, 3),
        activation="relu",
        input_shape=(16, 64, 64, 4),
        kernel_regularizer=l2(0.001),
        bias_regularizer=l2(0.001),
    )
)
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(
    Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_regularizer=l2(0.001),
        bias_regularizer=l2(0.001),
    )
)
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(GlobalAveragePooling3D())
model.add(
    Dense(
        64, activation="relu", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)
    )
)
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.summary()



## === cell 6
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=opt)



## === cell 7
assert Input_Values is not None and len(Input_Values) > 0, "Input_Values is empty."
assert Output_Values is not None and len(Output_Values) == len(
    Input_Values
), "Mismatched X/Y lengths."

history = model.fit(Input_Values, Output_Values, epochs=30, batch_size=32, shuffle=True)

print(history.history.keys())



## === cell 8
pass



## === cell 9
test_dir = TEST_DIR
test = test_patients
print("n_test:", len(test))



## === cell 10
CACHE_TEST = "/kaggle/working/Test_Values.npy"

if os.path.exists(CACHE_TEST):
    Test_Values = np.load(CACHE_TEST, allow_pickle=False)
else:
    Test_Values_list = []
    for idx, pid in enumerate(test, 1):
        scan = build_patient_scan(TEST_DIR, pid, layer_number=layer_number)
        Test_Values_list.append(scan)
        if idx % 10 == 0 or idx == len(test):
            print(f"built test {idx}/{len(test)}")
    Test_Values = np.array(Test_Values_list, dtype=np.float32)
    np.save(CACHE_TEST, Test_Values)

print("Test_Values shape:", Test_Values.shape, "dtype:", Test_Values.dtype)



## === cell 11
assert Test_Values is not None and len(Test_Values) == len(
    test
), "Test_Values mismatch with test IDs."



## === cell 12
Results = model.predict(Test_Values, batch_size=16).reshape(-1)
print(
    "Results shape:",
    Results.shape,
    "min/max:",
    float(np.min(Results)),
    float(np.max(Results)),
)



## === cell 13
sub = pd.DataFrame({"BraTS21ID": test, "MGMT_value": Results.astype(float)})
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")

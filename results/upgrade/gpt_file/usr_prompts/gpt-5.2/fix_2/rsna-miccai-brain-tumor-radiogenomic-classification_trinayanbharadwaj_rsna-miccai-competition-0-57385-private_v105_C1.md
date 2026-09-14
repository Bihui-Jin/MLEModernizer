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

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

labels_df = pd.read_csv(LABELS_CSV)
sample_sub_df = pd.read_csv(SAMPLE_SUB)

print("Train dir exists:", os.path.isdir(TRAIN_DIR), "n=", len(os.listdir(TRAIN_DIR)))
print("Test dir exists:", os.path.isdir(TEST_DIR), "n=", len(os.listdir(TEST_DIR)))
print("labels_df:", labels_df.shape, "sample_sub_df:", sample_sub_df.shape)




## === cell 2
def _read_dicom_pixel_array(dcm_path: str):
    """Read a DICOM slice and return a float32 2D array; return None if unreadable."""
    try:
        dcm = dicom.dcmread(dcm_path)
        arr = dcm.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def load_T2W_six_slices_for_cases(
    path_cases, img_px_size=150, min_sum=100000, min_norm_sum=1800
):
    """
    Core logic preserved:
    - Use T2w folder (index 3 in sorted modality list like original code)
    - Scan slices in order and keep first 6 that pass thresholds
    - Convert to 3-channel and normalize per-slice by max
    Returns: arrays_list = [arr1,...,arr6], each shape (n_cases, H, W, 3)
    """
    arrays = [[] for _ in range(6)]
    for case_path in path_cases:
        count = 0
        try:
            mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
            t2w_dir = mri_type[3]
            img_paths = sorted([f.path for f in os.scandir(t2w_dir) if f.is_file()])
        except Exception:
            continue

        for p in img_paths:
            if count >= 6:
                break
            arr = _read_dicom_pixel_array(p)
            if arr is None:
                continue
            if arr.sum() <= min_sum:
                continue

            resized_img = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked = np.stack((resized_img,) * 3, axis=-1)

            mx = float(np.max(stacked))
            if mx <= 0:
                continue
            stacked_norm = stacked / mx

            if stacked_norm.sum() <= min_norm_sum:
                continue

            arrays[count].append(stacked_norm)
            count += 1

        if count == 0:
            pad = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for j in range(6):
                arrays[j].append(pad)
        elif count < 6:
            last = arrays[count - 1][-1]
            for j in range(count, 6):
                arrays[j].append(last)

    out = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float32)
        mx = float(np.max(a)) if a.size else 1.0
        if mx > 0:
            a = a / mx
        out.append(a)
    print("Loaded T2W arrays lengths:", [len(x) for x in out])
    return out  # list of 6 arrays




## === cell 3
BAD_CASES = {"00109", "00123", "00709"}

labels_df["BraTS21ID_str"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID_str"].isin(BAD_CASES)].reset_index(
    drop=True
)

train_case_paths = [
    os.path.join(TRAIN_DIR, cid) for cid in labels_df["BraTS21ID_str"].tolist()
]
train_case_paths = [p for p in train_case_paths if os.path.isdir(p)]

existing_ids = set(os.path.basename(p) for p in train_case_paths)
labels_df = labels_df[labels_df["BraTS21ID_str"].isin(existing_ids)].reset_index(
    drop=True
)
train_case_paths = [
    os.path.join(TRAIN_DIR, cid) for cid in labels_df["BraTS21ID_str"].tolist()
]

y = labels_df["MGMT_value"].astype(np.float32).values

print(
    "Usable train cases:",
    len(train_case_paths),
    "labels:",
    len(y),
    "pos rate:",
    float(y.mean()),
)



## === cell 4

N_TRAIN_CASES = min(
    240, len(train_case_paths)
)  # small but non-trivial for AUC; keeps runtime manageable
idx = np.random.RandomState(SEED).choice(
    len(train_case_paths), size=N_TRAIN_CASES, replace=False
)
train_subset_paths = [train_case_paths[i] for i in idx]
y_subset = y[idx]

train_arrays_6 = load_T2W_six_slices_for_cases(train_subset_paths, img_px_size=150)

X_train = np.concatenate(train_arrays_6, axis=0)  # shape (N*6,150,150,3)
y_train = np.repeat(y_subset, 6)

perm = np.random.RandomState(SEED).permutation(len(X_train))
X_train = X_train[perm]
y_train = y_train[perm]

print("X_train:", X_train.shape, "y_train:", y_train.shape)



## === cell 5


def build_model(input_shape=(150, 150, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = build_model((150, 150, 3))
model_T2.summary()



## === cell 6
BATCH_SIZE = 16
EPOCHS = 3  # small to meet 600s runtime constraint

history = model_T2.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.15,
    verbose=2,
)



## === cell 7
test_case_paths = sorted([f.path for f in os.scandir(TEST_DIR) if f.is_dir()])
test_arrays_6 = load_T2W_six_slices_for_cases(test_case_paths, img_px_size=150)

pred_slices = []
for k in range(6):
    pk = model_T2.predict(test_arrays_6[k], batch_size=16, verbose=0).reshape(-1)
    pred_slices.append(pk.astype(np.float32))

pred_matrix = np.stack(pred_slices, axis=1)  # (n_cases, 6)
prediction = pred_matrix.mean(axis=1)

print(
    "Test prediction shape:",
    prediction.shape,
    "min/max:",
    float(prediction.min()),
    float(prediction.max()),
)




## === cell 8
def create_sub(path_cases, prediction):
    cases = [int(os.path.basename(p)) for p in path_cases]
    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction.astype(float)})
    df = df.sort_values("BraTS21ID").reset_index(drop=True)
    return df


sub_df = create_sub(test_case_paths, prediction)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sample_ids = sample_sub_df["BraTS21ID"].astype(int).values
sub_df = sub_df.set_index("BraTS21ID").reindex(sample_ids).reset_index()

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5)

print(sub_df.head())
print(
    "Submission shape:",
    sub_df.shape,
    "missing preds:",
    int(sub_df["MGMT_value"].isna().sum()),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2911472299.py in <cell line: 0>()
      7 
      8 
----> 9 sub_df = create_sub(test_case_paths, prediction)
     10 
     11 # Ensure submission has all expected rows; align to sample_submission order if needed.

/tmp/ipykernel_11/2911472299.py in create_sub(path_cases, prediction)
      1 def create_sub(path_cases, prediction):
      2     # Build BraTS21ID from folder names to ensure correct alignment and formatting.
----> 3     cases = [int(os.path.basename(p)) for p in path_cases]
      4     df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction.astype(float)})
      5     df = df.sort_values("BraTS21ID").reset_index(drop=True)

/tmp/ipykernel_11/2911472299.py in <listcomp>(.0)
      1 def create_sub(path_cases, prediction):
      2     # Build BraTS21ID from folder names to ensure correct alignment and formatting.
----> 3     cases = [int(os.path.basename(p)) for p in path_cases]
      4     df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction.astype(float)})
      5     df = df.sort_values("BraTS21ID").reset_index(drop=True)

ValueError: invalid literal for int() with base 10: 'test'

## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv; size:", os.path.getsize("submission.csv"), "bytes")
print("Columns:", list(sub_df.columns))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1404964936.py in <cell line: 0>()
      1 # Write submission
----> 2 sub_df.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv; size:", os.path.getsize("submission.csv"), "bytes")
      4 print("Columns:", list(sub_df.columns))

NameError: name 'sub_df' is not defined

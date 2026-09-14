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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists :", os.path.isdir(TEST_DIR))
print("Labels exists   :", os.path.isfile(LABELS_CSV))

labels_df = pd.read_csv(LABELS_CSV, dtype={"BraTS21ID": str})
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].str.zfill(5)

bad_ids = {"00109", "00123", "00709"}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

labels_df.head()




## === cell 2
def _safe_dcm_read(path):
    """Read DICOM robustly; return None on failure."""
    try:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def load_case_slices(case_dir, modality="T2w", img_px_size=150, num_slices=6):
    """
    Load up to `num_slices` informative slices from a case directory for a given modality.
    Returns: np.ndarray shape (num_slices, img_px_size, img_px_size, 3)
    """
    mod_dir = os.path.join(case_dir, modality)
    if not os.path.isdir(mod_dir):
        return None

    dcm_paths = sorted(
        [
            os.path.join(mod_dir, f)
            for f in os.listdir(mod_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    if len(dcm_paths) == 0:
        return None

    selected = []
    for p in dcm_paths:
        arr = _safe_dcm_read(p)
        if arr is None:
            continue

        if float(arr.sum()) <= 100000.0:
            continue

        arr_rs = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        mx = float(np.max(arr_rs))
        if mx <= 0:
            continue
        arr_norm = (
            arr_rs / mx
        )  # per-slice normalization (keeps logic close to original)

        if float(arr_norm.sum()) <= 2500.0:
            continue

        rgb = np.stack([arr_norm, arr_norm, arr_norm], axis=-1)  # (H,W,3)
        selected.append(rgb)

        if len(selected) >= num_slices:
            break

    if len(selected) == 0:
        return None

    while len(selected) < num_slices:
        selected.append(selected[-1])

    return np.stack(selected, axis=0).astype(np.float32)


def build_dataset_from_ids(
    ids, base_dir, modality="T2w", img_px_size=150, num_slices=6, verbose=1
):
    """
    Build X as flattened slices: (n_cases*num_slices, H, W, 3)
    and map each slice to its case label (repeated).
    """
    X_list = []
    case_index = []  # maps each slice row to case index in ids

    for ci, case_id in enumerate(ids):
        case_dir = os.path.join(base_dir, case_id)
        arr = load_case_slices(
            case_dir, modality=modality, img_px_size=img_px_size, num_slices=num_slices
        )
        if arr is None:
            if verbose:
                print(f"Warning: no slices loaded for case {case_id}; skipping case.")
            continue
        for s in range(num_slices):
            X_list.append(arr[s])
            case_index.append(ci)

    if len(X_list) == 0:
        raise RuntimeError("No images were loaded. Check paths/modalities.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    case_index = np.array(case_index, dtype=np.int32)
    return X, case_index




## === cell 3
IMG_PX_SIZE = 150
NUM_SLICES = 6
MODALITY = "T2w"

train_ids = labels_df["BraTS21ID"].tolist()
y_by_case = labels_df.set_index("BraTS21ID")["MGMT_value"].astype(np.float32)

X_train_slices, train_case_index = build_dataset_from_ids(
    train_ids,
    TRAIN_DIR,
    modality=MODALITY,
    img_px_size=IMG_PX_SIZE,
    num_slices=NUM_SLICES,
    verbose=0,
)

train_ids_arr = np.array(train_ids)
y_train_slices = np.array(
    [y_by_case[train_ids_arr[i]] for i in train_case_index], dtype=np.float32
)

print("Loaded slice-level train X:", X_train_slices.shape, "y:", y_train_slices.shape)
print("Positive rate (slice-level):", float(y_train_slices.mean()))




## === cell 4
def make_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


model_T2 = make_model()
model_T2.summary()



## === cell 5
from sklearn.model_selection import train_test_split

X_tr, X_va, y_tr, y_va = train_test_split(
    X_train_slices,
    y_train_slices,
    test_size=0.2,
    random_state=SEED,
    stratify=(y_train_slices > 0.5),
)

BATCH_SIZE = 32
EPOCHS = (
    2  # keep runtime within 600s in Kaggle CPU; minimal change to ensure completion
)

history = model_T2.fit(
    X_tr,
    y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 6
sample_df = pd.read_csv(SAMPLE_SUB, dtype={"BraTS21ID": str})
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].str.zfill(5)
test_ids = sample_df["BraTS21ID"].tolist()

X_test_slices, test_case_index = build_dataset_from_ids(
    test_ids,
    TEST_DIR,
    modality=MODALITY,
    img_px_size=IMG_PX_SIZE,
    num_slices=NUM_SLICES,
    verbose=0,
)

print("Loaded slice-level test X:", X_test_slices.shape)



## === cell 7
test_slice_pred = (
    model_T2.predict(X_test_slices, batch_size=BATCH_SIZE, verbose=0)
    .reshape(-1)
    .astype(np.float32)
)

n_cases = len(test_ids)
sum_pred = np.zeros(n_cases, dtype=np.float32)
cnt_pred = np.zeros(n_cases, dtype=np.int32)

for pred, ci in zip(test_slice_pred, test_case_index):
    sum_pred[ci] += pred
    cnt_pred[ci] += 1

case_pred = np.where(cnt_pred > 0, sum_pred / np.maximum(cnt_pred, 1), 0.5).astype(
    np.float32
)

case_pred = np.clip(case_pred, 0.0, 1.0)

sub_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": case_pred})
sub_df.head()



## === cell 8
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df = sub_df.merge(sample_df[["BraTS21ID"]], on="BraTS21ID", how="right")
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(np.float32).fillna(0.5)

print(sub_df.shape)
print(sub_df.isna().sum())
sub_df.head()



## === cell 9
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub_df.head(10).to_string(index=False))

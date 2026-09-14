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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    from skimage.transform import resize as sk_resize
except Exception as e:
    sk_resize = None
    print(
        "WARNING: skimage.transform.resize import failed; will fallback to tf.image.resize. Error:",
        repr(e),
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

BAD_CASES = {109, 123, 709}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

print("Train labels:", labels_df.shape, "Sample submission:", sample_sub.shape)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)




## === cell 2
def _safe_resize(img2d: np.ndarray, out_hw: int) -> np.ndarray:
    """Resize a 2D image to (out_hw, out_hw) float32."""
    img2d = img2d.astype(np.float32)
    if sk_resize is not None:
        return sk_resize(
            img2d, (out_hw, out_hw), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
    x = tf.convert_to_tensor(img2d[None, ..., None], dtype=tf.float32)
    x = tf.image.resize(x, (out_hw, out_hw), method="bilinear")
    return x[0, ..., 0].numpy().astype(np.float32)


def _read_dicom_pixel_array(dcm_path: str) -> np.ndarray:
    """Read dicom and return pixel_array as float32; handle occasional read issues."""
    try:
        ds = dicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def load_subject_t2w_stack(
    subject_path: str, img_px_size: int = 150, n_slices: int = 6
) -> np.ndarray:
    """
    Core logic preserved: pick up to 6 informative T2w slices (same spirit as original code),
    resize to IMG_PX_SIZE, stack to 3 channels, normalize.
    Returns: (n_slices, img_px_size, img_px_size, 3)
    """
    t2w_dir = os.path.join(subject_path, "T2w")
    if not os.path.isdir(t2w_dir):
        return np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)

    dcm_files = sorted(
        [
            os.path.join(t2w_dir, f)
            for f in os.listdir(t2w_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    if len(dcm_files) == 0:
        return np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)

    selected = []
    for p in dcm_files:
        arr = _read_dicom_pixel_array(p)
        if arr is None:
            continue

        if arr.sum() <= 100000:
            continue

        img = _safe_resize(arr, img_px_size)
        stacked = np.stack((img,) * 3, axis=-1)

        mx = np.max(stacked)
        if mx > 0:
            stacked = stacked / mx

        if stacked.sum() > 2000:
            selected.append(stacked.astype(np.float32))
            if len(selected) >= n_slices:
                break

    if len(selected) == 0:
        out = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    else:
        out = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
        out[: len(selected)] = np.stack(selected, axis=0)

    mx = np.max(out)
    if mx > 0:
        out = out / mx

    return out




## === cell 3
def load_dataset_from_dir(
    base_dir: str, ids: list[int], img_px_size: int = 150, n_slices: int = 6
):
    """
    Load stacks for a list of subject IDs.
    Returns X shape: (len(ids)*n_slices, H, W, 3) and a mapping subject_index for aggregation.
    """
    X_list = []
    subj_index = []
    for idx, brats_id in enumerate(ids):
        subj_folder = os.path.join(base_dir, f"{brats_id:05d}")
        stack = load_subject_t2w_stack(
            subj_folder, img_px_size=img_px_size, n_slices=n_slices
        )
        X_list.append(stack)
        subj_index.extend([idx] * n_slices)
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    subj_index = np.array(subj_index, dtype=np.int32)
    return X, subj_index


all_train_ids = labels_df["BraTS21ID"].astype(int).tolist()
all_y = labels_df["MGMT_value"].astype(np.float32).values

from sklearn.model_selection import train_test_split

train_ids, val_ids, y_train, y_val = train_test_split(
    all_train_ids, all_y, test_size=0.15, random_state=SEED, stratify=all_y
)

IMG_PX_SIZE = 150
N_SLICES = 6

print("Loading training images...")
X_train, train_subj_idx = load_dataset_from_dir(
    TRAIN_DIR, train_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
print("Loading validation images...")
X_val, val_subj_idx = load_dataset_from_dir(
    TRAIN_DIR, val_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)

y_train_rep = np.repeat(y_train, N_SLICES).astype(np.float32)
y_val_rep = np.repeat(y_val, N_SLICES).astype(np.float32)

print("X_train:", X_train.shape, "y_train:", y_train_rep.shape)
print("X_val:", X_val.shape, "y_val:", y_val_rep.shape)




## === cell 4
def build_model(input_shape=(150, 150, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_model((IMG_PX_SIZE, IMG_PX_SIZE, 3))
model.summary()

history = model.fit(
    X_train,
    y_train_rep,
    validation_data=(X_val, y_val_rep),
    epochs=2,
    batch_size=32,
    verbose=2,
)




## === cell 5
def predict_subject_probs(model, X: np.ndarray, subj_index: np.ndarray) -> np.ndarray:
    """
    Predict per-slice probabilities then aggregate per subject by mean (as original code averaged slices/models).
    Returns probs per subject ordered by subject index (0..n_subjects-1).
    """
    p = model.predict(X, batch_size=64, verbose=0).reshape(-1).astype(np.float32)
    n_subjects = int(subj_index.max()) + 1 if subj_index.size else 0
    out = np.zeros(n_subjects, dtype=np.float32)
    for i in range(n_subjects):
        out[i] = float(p[subj_index == i].mean()) if np.any(subj_index == i) else 0.5
    return out


from sklearn.metrics import roc_auc_score

val_probs = predict_subject_probs(model, X_val, val_subj_idx)
val_auc = roc_auc_score(y_val, val_probs)
print("Validation subject-level AUC:", val_auc)



## === cell 6
test_ids = sample_sub["BraTS21ID"].astype(int).tolist()
print("Loading test images...")
X_test, test_subj_idx = load_dataset_from_dir(
    TEST_DIR, test_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)

test_probs = predict_subject_probs(model, X_test, test_subj_idx)
print(
    "Test probs:",
    test_probs.shape,
    "min/max:",
    float(test_probs.min()),
    float(test_probs.max()),
)



## === cell 7
sub_df = pd.DataFrame(
    {
        "BraTS21ID": [f"{i:05d}" for i in test_ids],
        "MGMT_value": test_probs.astype(float),
    }
)

sub_df = (
    sub_df.set_index("BraTS21ID")
    .loc[sample_sub["BraTS21ID"].astype(str).values]
    .reset_index()
)
assert sub_df.shape[0] == sample_sub.shape[0]
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2131921375.py in <cell line: 0>()
      9 # Ensure correct row count and ordering exactly matching sample_submission
     10 sub_df = (
---> 11     sub_df.set_index("BraTS21ID")
     12     .loc[sample_sub["BraTS21ID"].astype(str).values]
     13     .reset_index()

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['356', '140', '757', '275', '570', '155', '740', '21', '84', '729',\n       '1009', '556', '735', '587', '214', '732', '254', '1001', '31', '823',\n       '459', '496', '622', '512', '263', '511', '591', '269', '525', '661',\n       '332', '650', '532', '656', '339', '718', '245', '737', '177', '657',\n       '59', '236', '623', '54', '241', '399', '413', '2', '613', '601', '74',\n       '132', '803', '499', '495', '539', '470', '185', '19'],\n      dtype='object', name='BraTS21ID')] are in the [index]"

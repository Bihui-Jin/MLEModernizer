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
import random
import numpy as np
import pandas as pd

import pydicom
from skimage.transform import (
    resize,
)  # kept imported to preserve environment parity if referenced elsewhere

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

IMG_PX_SIZE = 150
N_SLICES = 6  # matches original array_1..array_6 approach
CHANNELS = 3  # original stack to 3 channels

BAD_CASES = {"00109", "00123", "00709"}  # per competition note



## === cell 1


def _safe_dcm_pixel_array(dcm_path):
    """Read DICOM and return pixel array as float32; return None if unreadable."""
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)
        arr = dcm.pixel_array
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _normalize01(x, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if (mx - mn) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _stack3(img2d):
    img2d = img2d.astype(np.float32, copy=False)
    return np.stack((img2d, img2d, img2d), axis=-1).astype(np.float32, copy=False)


def _resize2d_tf(arr2d, img_px_size):
    t = tf.convert_to_tensor(arr2d, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)  # (H,W,1)
    t = tf.image.resize(
        t, (img_px_size, img_px_size), method="bilinear", antialias=True
    )
    t = tf.squeeze(t, axis=-1)
    return t.numpy().astype(np.float32, copy=False)


_VOL_CACHE = {}


def load_case_t2w_slices(case_dir, img_px_size=150, n_slices=6):
    """
    Load up to n_slices from T2w folder with simple filtering, then pad if needed.
    Returns shape: (n_slices, img_px_size, img_px_size, 3)
    """
    cache_key = (case_dir, int(img_px_size), int(n_slices))
    cached = _VOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    t2_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2_dir):
        vol = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
        _VOL_CACHE[cache_key] = vol
        return vol

    dcm_paths = []
    with os.scandir(t2_dir) as it:
        for e in it:
            if e.is_file():
                name = e.name
                if name and name[-4:].lower() == ".dcm":
                    dcm_paths.append(e.path)
    dcm_paths.sort()

    if len(dcm_paths) == 0:
        vol = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
        _VOL_CACHE[cache_key] = vol
        return vol

    selected = []
    for p in dcm_paths:
        arr = _safe_dcm_pixel_array(p)
        if arr is None:
            continue
        if float(arr.sum()) <= 100000.0:
            continue

        arr_rs = _resize2d_tf(arr, img_px_size)
        arr_rs = _normalize01(arr_rs)
        if float(arr_rs.sum()) <= 2000.0:
            continue

        selected.append(_stack3(arr_rs))
        if len(selected) >= n_slices:
            break

    if len(selected) < n_slices:
        idxs = (
            np.linspace(0, len(dcm_paths) - 1, num=n_slices, dtype=int)
            if len(dcm_paths) > 0
            else []
        )
        selected = []
        for j in idxs:
            arr = _safe_dcm_pixel_array(dcm_paths[int(j)])
            if arr is None:
                arr_rs = np.zeros((img_px_size, img_px_size), dtype=np.float32)
            else:
                arr_rs = _resize2d_tf(arr, img_px_size)
                arr_rs = _normalize01(arr_rs)
            selected.append(_stack3(arr_rs))

    if len(selected) < n_slices:
        pad = [
            np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for _ in range(n_slices - len(selected))
        ]
        selected = selected + pad
    selected = selected[:n_slices]

    vol = np.stack(selected, axis=0).astype(np.float32, copy=False)
    _VOL_CACHE[cache_key] = vol
    return vol


def list_case_dirs(root_dir, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    case_dirs = []
    with os.scandir(root_dir) as it:
        for e in it:
            if not e.is_dir():
                continue
            name = e.name
            if not name.isdigit():
                continue
            if name in exclude_ids:
                continue
            case_dirs.append(e.path)
    case_dirs.sort()
    return case_dirs




## === cell 2
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_map = dict(
    zip(
        labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values.astype(np.float32)
    )
)

train_case_dirs = list_case_dirs(TRAIN_DIR, exclude_ids=BAD_CASES)
test_case_dirs = list_case_dirs(TEST_DIR, exclude_ids=set())

train_case_dirs = [p for p in train_case_dirs if os.path.basename(p) in labels_map]

print("Train cases:", len(train_case_dirs))
print("Test cases:", len(test_case_dirs))




## === cell 3
def build_slice_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out)


slice_model = build_slice_model()
slice_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

slice_model.summary()



## === cell 4


def gen_train_slices(case_dirs, labels_map, n_slices=N_SLICES):
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        y = np.float32(labels_map[case_id])
        vol = load_case_t2w_slices(
            case_dir, img_px_size=IMG_PX_SIZE, n_slices=n_slices
        )  # cached
        for i in range(n_slices):
            yield vol[i], y


idx = np.arange(len(train_case_dirs))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.85 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]
tr_dirs = [train_case_dirs[i] for i in tr_idx]
va_dirs = [train_case_dirs[i] for i in va_idx]

output_signature = (
    tf.TensorSpec(shape=(IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
)

train_ds = (
    tf.data.Dataset.from_generator(
        lambda: gen_train_slices(tr_dirs, labels_map, N_SLICES),
        output_signature=output_signature,
    )
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_generator(
        lambda: gen_train_slices(va_dirs, labels_map, N_SLICES),
        output_signature=output_signature,
    )
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)

print("Train cases:", len(tr_dirs), "Val cases:", len(va_dirs))



## === cell 5
history = slice_model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)



## === cell 6


def predict_case_probability(case_dir, model, n_slices=N_SLICES):
    vol = load_case_t2w_slices(case_dir, img_px_size=IMG_PX_SIZE, n_slices=n_slices)
    preds = model.predict(vol, verbose=0).reshape(-1)
    p = float(np.mean(preds))
    return float(np.clip(p, 0.0, 1.0))


test_ids = [os.path.basename(p) for p in test_case_dirs]

test_vols = [
    load_case_t2w_slices(p, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES)
    for p in test_case_dirs
]
all_slices = np.concatenate(test_vols, axis=0)  # (num_cases*6, H, W, 3)
all_preds = slice_model.predict(all_slices, batch_size=64, verbose=0).reshape(
    -1
)  # (num_cases*6,)

test_probs = []
for i in range(len(test_case_dirs)):
    s = i * N_SLICES
    e = s + N_SLICES
    p = float(np.mean(all_preds[s:e]))
    test_probs.append(float(np.clip(p, 0.0, 1.0)))

sub_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_probs})
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df = sub_df.sort_values("BraTS21ID").reset_index(drop=True)

sub_df.head(), sub_df.shape



## === cell 7
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

print(
    "Submission rows:", len(sub_df), "Missing probs:", sub_df["MGMT_value"].isna().sum()
)
sub_df.head()



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_df.columns))
print(sub_df.describe(include="all"))

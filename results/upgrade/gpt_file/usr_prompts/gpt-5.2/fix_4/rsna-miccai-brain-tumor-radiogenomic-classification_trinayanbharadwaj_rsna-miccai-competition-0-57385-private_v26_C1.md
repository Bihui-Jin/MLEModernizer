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

import pydicom as dicom
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
LABELS_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing LABELS_CSV: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing SAMPLE_SUB: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

print("train labels:", labels_df.shape, "sample submission:", sample_sub.shape)



## === cell 2
IMG_PX_SIZE = 128  # keep as provided
SERIES_ORDER = ["FLAIR", "T1w", "T1wCE", "T2w"]

BAD_CASES = {"00109", "00123", "00709"}  # per competition note


def _safe_norm(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    mx = float(np.max(img))
    if mx <= 0:
        return img
    return img / mx


def _resize_to(img: np.ndarray, size: int) -> np.ndarray:
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32, copy=False
    )


_dcm_list_cache = {}


def _dcm_numeric_key(name: str) -> int:
    try:
        base = os.path.splitext(name)[0]
        return int(base.split("-")[-1])
    except Exception:
        return 10**18


def _list_dcm_files(series_dir: str):
    cached = _dcm_list_cache.get(series_dir)
    if cached is not None:
        return cached
    try:
        names = [f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    except FileNotFoundError:
        names = []
    names.sort(key=_dcm_numeric_key)
    files = [os.path.join(series_dir, f) for f in names]
    _dcm_list_cache[series_dir] = files
    return files


def _fast_norm_sum_proxy(pixel_array: np.ndarray) -> float:
    arr = pixel_array
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    h, w = arr.shape
    ch = min(96, h)
    cw = min(96, w)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    crop = arr[y0 : y0 + ch, x0 : x0 + cw]
    small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA).astype(
        np.float32, copy=False
    )
    mx = float(np.max(small))
    if mx <= 0:
        return 0.0
    return float((small / mx).sum())


_series_instance_cache = {}


def _instance_number_key(ds, fallback: int):
    try:
        v = getattr(ds, "InstanceNumber", None)
        if v is None:
            return fallback
        return int(v)
    except Exception:
        return fallback


def _get_series_files_sorted_by_instance(series_dir: str):
    cached = _series_instance_cache.get(series_dir)
    if cached is not None:
        return cached

    files = _list_dcm_files(series_dir)
    if not files:
        _series_instance_cache[series_dir] = files
        return files

    items = []
    for i, fp in enumerate(files):
        try:
            ds = dicom.dcmread(
                fp,
                stop_before_pixels=True,
                specific_tags=["InstanceNumber"],
                force=True,
            )
            inst = _instance_number_key(ds, i)
        except Exception:
            inst = i
        items.append((inst, i, fp))

    items.sort(key=lambda x: (x[0], x[1]))
    sorted_files = [fp for _, _, fp in items]
    _series_instance_cache[series_dir] = sorted_files
    return sorted_files


def _pick_representative_slice(
    series_dir: str, size: int, min_sum: float = 5000.0
) -> np.ndarray:
    """
    Same high-level logic: iterate slices in order and pick first with normalized sum > threshold;
    else fall back to middle slice. Optimized by ordering via InstanceNumber header read, and
    avoiding pixel decode for non-selected slices as much as possible.
    """
    dcm_files = _get_series_files_sorted_by_instance(series_dir)
    if len(dcm_files) == 0:
        return np.zeros((size, size), dtype=np.float32)

    chosen = None

    for fp in dcm_files:
        try:
            dcm = dicom.dcmread(fp, stop_before_pixels=False, force=True)
            arr = dcm.pixel_array
            if _fast_norm_sum_proxy(arr) > min_sum:
                arr2 = _resize_to(arr, size)
                chosen = _safe_norm(arr2)
                break
        except Exception:
            continue

    if chosen is None:
        mid_fp = dcm_files[len(dcm_files) // 2]
        try:
            dcm = dicom.dcmread(mid_fp, stop_before_pixels=False, force=True)
            arr = dcm.pixel_array
            arr = _resize_to(arr, size)
            chosen = _safe_norm(arr)
        except Exception:
            chosen = np.zeros((size, size), dtype=np.float32)

    return chosen


def load_case_4ch(case_dir: str, size: int) -> np.ndarray:
    """
    Returns HxWx4 float32 tensor for a case (one slice per series).
    """
    chs = []
    for series in SERIES_ORDER:
        series_dir = os.path.join(case_dir, series)
        if not os.path.isdir(series_dir):
            chs.append(np.zeros((size, size), dtype=np.float32))
        else:
            chs.append(_pick_representative_slice(series_dir, size=size))
    x = np.stack(chs, axis=-1)  # (H,W,4)
    return x.astype(np.float32, copy=False)




## === cell 3
all_case_ids = sorted(
    [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
)
all_case_ids = [cid for cid in all_case_ids if cid not in BAD_CASES]

labels_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))

train_ids = [cid for cid in all_case_ids if cid in labels_map]
y_all = np.array([labels_map[cid] for cid in train_ids], dtype=np.float32)

pos_idx = [i for i, v in enumerate(y_all) if v >= 0.5]
neg_idx = [i for i, v in enumerate(y_all) if v < 0.5]
random.shuffle(pos_idx)
random.shuffle(neg_idx)


def _split_indices(idxs, val_frac=0.15):
    n_val = max(1, int(len(idxs) * val_frac))
    return idxs[n_val:], idxs[:n_val]


pos_tr, pos_va = _split_indices(pos_idx, 0.15)
neg_tr, neg_va = _split_indices(neg_idx, 0.15)

tr_idx = pos_tr + neg_tr
va_idx = pos_va + neg_va
random.shuffle(tr_idx)
random.shuffle(va_idx)

train_ids_tr = [train_ids[i] for i in tr_idx]
train_ids_va = [train_ids[i] for i in va_idx]

y_tr = np.array([labels_map[cid] for cid in train_ids_tr], dtype=np.float32)
y_va = np.array([labels_map[cid] for cid in train_ids_va], dtype=np.float32)

print(
    "Train/Val sizes:",
    len(train_ids_tr),
    len(train_ids_va),
    "pos rate train:",
    y_tr.mean(),
    "val:",
    y_va.mean(),
)




## === cell 4
def build_X(ids, root_dir):
    X = np.zeros((len(ids), IMG_PX_SIZE, IMG_PX_SIZE, 4), dtype=np.float32)
    case_dirs = [os.path.join(root_dir, cid) for cid in ids]
    for i, case_dir in enumerate(case_dirs):
        X[i] = load_case_4ch(case_dir, IMG_PX_SIZE)
    return X


X_tr = build_X(train_ids_tr, TRAIN_DIR)
X_va = build_X(train_ids_va, TRAIN_DIR)

print("X_tr:", X_tr.shape, "X_va:", X_va.shape)




## === cell 5
def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 4)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_1 = build_model()
model_1.summary()



## === cell 6
BATCH_SIZE = 16
EPOCHS = 6

history = model_1.fit(
    X_tr,
    y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 7
test_ids = sorted(
    [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
)
test_ids = [tid for tid in test_ids if tid not in BAD_CASES]
test_ids = [str(tid).zfill(5) for tid in test_ids]

X_test_4ch = build_X(test_ids, TEST_DIR)

n = X_test_4ch.shape[0]
X_big = np.zeros((n * 5, IMG_PX_SIZE, IMG_PX_SIZE, 4), dtype=np.float32)

X_big[0:n] = X_test_4ch
for c in range(4):
    start = (c + 1) * n
    end = start + n
    X_big[start:end, ..., c] = X_test_4ch[..., c]

pred_big = model_1.predict(X_big, batch_size=BATCH_SIZE, verbose=0).reshape(-1, 1)
pred_big = pred_big.reshape(5, n).T  # (n,5) with columns: [full, ch0, ch1, ch2, ch3]

prediction = pred_big.mean(axis=1)
prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)

print(
    "Test predictions:",
    prediction.shape,
    float(prediction.min()),
    float(prediction.max()),
    float(prediction.mean()),
)




## === cell 8
def create_sub(path_test, ids, prediction):
    df = pd.DataFrame(
        {
            "BraTS21ID": [str(x).zfill(5) for x in ids],
            "MGMT_value": prediction.astype(float),
        }
    )
    return df


sub_df = create_sub(TEST_DIR, test_ids, prediction)
sub_df.head()



## === cell 9
sub_df = sub_df[["BraTS21ID", "MGMT_value"]].copy()
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).clip(0.0, 1.0)

sample_ids = set(sample_sub["BraTS21ID"].tolist())
if set(sub_df["BraTS21ID"]) == sample_ids:
    sub_df = sub_df.set_index("BraTS21ID").loc[sample_sub["BraTS21ID"]].reset_index()

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub_df.shape)
print(sub_df.head())

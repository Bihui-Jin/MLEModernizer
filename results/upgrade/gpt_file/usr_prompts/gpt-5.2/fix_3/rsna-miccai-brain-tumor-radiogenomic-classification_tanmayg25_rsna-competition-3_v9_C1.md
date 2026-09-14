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
import numpy as np
import pandas as pd
import pydicom
import cv2

RNG_SEED = 42
np.random.seed(RNG_SEED)

BASE_DIR = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

patients = sorted(os.listdir(TRAIN_DIR))
test = sorted(os.listdir(TEST_DIR))




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


def _dcmread_header(path):
    return pydicom.dcmread(path, force=True, stop_before_pixels=True)


def _dcmread_pixels(path):
    return pydicom.dcmread(path, force=True)


def _slice_sort_key_from_header(ds):
    if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            pass
    if hasattr(ds, "InstanceNumber"):
        try:
            return int(ds.InstanceNumber)
        except Exception:
            pass
    return 0.0


def _list_dcm_paths(series_dir):
    try:
        with os.scandir(series_dir) as it:
            return [
                e.path for e in it if e.is_file() and e.name.lower().endswith(".dcm")
            ]
    except FileNotFoundError:
        return []


def load_series_mid_slices(series_dir, layer_number=16, size=64):
    """
    Minimal-runtime feature construction (same semantics as original):
    - Read and sort DICOMs in a series.
    - Select layer_number approximately evenly spaced slices.
    - Resize each slice to (size, size).
    Returns: np.ndarray of shape (layer_number, size, size)
    """
    fps = _list_dcm_paths(series_dir)
    n = len(fps)
    if n == 0:
        return np.zeros((layer_number, size, size), dtype=np.float32)

    keys = []
    for fp in fps:
        try:
            ds = _dcmread_header(fp)
            keys.append(_slice_sort_key_from_header(ds))
        except Exception:
            keys.append(0.0)
    order = np.argsort(np.asarray(keys, dtype=np.float64), kind="mergesort")
    fps_sorted = [fps[i] for i in order.tolist()]

    idxs = np.linspace(0, n - 1, num=layer_number, dtype=np.int32)

    out = np.zeros((layer_number, size, size), dtype=np.float32)
    for i, idx in enumerate(idxs.tolist()):
        fp = fps_sorted[int(idx)]
        try:
            ds = _dcmread_pixels(fp)
            img = ds.pixel_array.astype(np.float32, copy=False)
            mn = float(img.min())
            mx = float(img.max())
            if mx > mn:
                img = (img - mn) / (mx - mn)
            else:
                img = img * 0.0
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            out[i] = img
        except Exception:
            out[i] = 0.0
    return out


def build_4ch_volume(subject_dir, layer_number=16, size=64):
    """
    Returns a (layer_number, size, size, 4) float32 tensor: [FLAIR, T1w, T1wCE, T2w]
    """
    flair = load_series_mid_slices(
        os.path.join(subject_dir, "FLAIR"), layer_number, size
    )
    t1w = load_series_mid_slices(os.path.join(subject_dir, "T1w"), layer_number, size)
    t1wce = load_series_mid_slices(
        os.path.join(subject_dir, "T1wCE"), layer_number, size
    )
    t2w = load_series_mid_slices(os.path.join(subject_dir, "T2w"), layer_number, size)
    vol = np.stack([flair, t1w, t1wce, t2w], axis=-1).astype(np.float32, copy=False)
    return vol




## === cell 2
df = pd.read_csv(LABELS_PATH)

bad_ids = {"00109", "00123", "00709"}
df["BraTS21ID"] = df["BraTS21ID"].astype(str).str.zfill(5)
df_train = df[~df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

y = df_train["MGMT_value"].astype(np.float32).to_numpy().reshape(-1, 1)
train_ids = df_train["BraTS21ID"].tolist()

print("Train rows after exclusion:", len(train_ids), "Labels shape:", y.shape)



## === cell 3
layer_number = 16
size = 64

from concurrent.futures import ThreadPoolExecutor


def _build_one_train(idx_pid):
    idx, pid = idx_pid
    subj_dir = os.path.join(TRAIN_DIR, pid)
    if not os.path.isdir(subj_dir):
        return idx, pid, None
    vol = build_4ch_volume(subj_dir, layer_number=layer_number, size=size)
    return idx, pid, vol


max_workers = min(8, (os.cpu_count() or 2))
vols = [None] * len(train_ids)

kept = [False] * len(train_ids)
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for j, (idx, pid, vol) in enumerate(
        ex.map(_build_one_train, list(enumerate(train_ids))), start=1
    ):
        if vol is not None:
            vols[idx] = vol
            kept[idx] = True
        if j % 50 == 0:
            print(f"Built train volumes: {j}/{len(train_ids)}")

kept_ids = [pid for pid, k in zip(train_ids, kept) if k]
Input_Values = np.stack([v for v in vols if v is not None], axis=0)

keep_mask = df_train["BraTS21ID"].isin(set(kept_ids)).to_numpy()
Output_Values = (
    df_train.loc[keep_mask, "MGMT_value"].astype(np.float32).to_numpy().reshape(-1, 1)
)

print("Input_Values shape:", Input_Values.shape)
print("Output_Values shape:", Output_Values.shape)



## === cell 4
use_tf = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import (
        Conv3D,
        MaxPooling3D,
        BatchNormalization,
        Dropout,
        Dense,
        GlobalAveragePooling3D,
        LeakyReLU,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.regularizers import l2

    try:
        tf.random.set_seed(RNG_SEED)
    except Exception:
        pass
except Exception as e:
    use_tf = False
    tf_import_error = repr(e)

print("TensorFlow available:", use_tf)
if not use_tf:
    print("TensorFlow import error:", tf_import_error)



## === cell 5
if use_tf:
    model = Sequential()
    model.add(
        Conv3D(
            32,
            (3, 3, 3),
            activation="relu",
            input_shape=(16, 64, 64, 4),
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
        )
    )
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling3D(pool_size=(2, 3, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(
        Conv3D(
            64,
            (3, 3, 3),
            activation="relu",
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
        )
    )
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(GlobalAveragePooling3D())
    model.add(
        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()



## === cell 6
if use_tf:
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=opt)



## === cell 7
if use_tf:
    history = model.fit(
        Input_Values, Output_Values, epochs=45, batch_size=32, shuffle=True, verbose=2
    )



## === cell 8
if not use_tf:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = Input_Values.reshape(Input_Values.shape[0], -1, 4)
    feat_mean = X.mean(axis=1)
    feat_std = X.std(axis=1)
    X_feat = np.concatenate([feat_mean, feat_std], axis=1).astype(np.float32)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=RNG_SEED)),
        ]
    )
    clf.fit(X_feat, Output_Values.ravel())
    print("Trained sklearn LogisticRegression on features:", X_feat.shape)



## === cell 9
test_dir = TEST_DIR



## === cell 10
from concurrent.futures import ThreadPoolExecutor


def _build_one_test(idx_pid):
    idx, pid = idx_pid
    subj_dir = os.path.join(test_dir, pid)
    vol = build_4ch_volume(subj_dir, layer_number=layer_number, size=size)
    return idx, vol


max_workers = min(8, (os.cpu_count() or 2))
test_vols = [None] * len(test)
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for j, (idx, vol) in enumerate(
        ex.map(_build_one_test, list(enumerate(test))), start=1
    ):
        test_vols[idx] = vol
        if j % 20 == 0:
            print(f"Built test volumes: {j}/{len(test)}")

Test_Values = np.stack(test_vols, axis=0)
print("Test_Values shape:", Test_Values.shape)



## === cell 11
if use_tf:
    Results = model.predict(Test_Values, batch_size=8, verbose=0).reshape(-1)
else:
    Xt = Test_Values.reshape(Test_Values.shape[0], -1, 4)
    feat_mean_t = Xt.mean(axis=1)
    feat_std_t = Xt.std(axis=1)
    X_feat_t = np.concatenate([feat_mean_t, feat_std_t], axis=1).astype(np.float32)
    Results = clf.predict_proba(X_feat_t)[:, 1].reshape(-1)

Results = np.clip(Results, 0.0, 1.0)
print("Preds shape:", Results.shape, "min/max:", Results.min(), Results.max())



## === cell 12
sub = pd.read_csv(SAMPLE_SUB_PATH)
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

pred_map = {pid: float(p) for pid, p in zip(test, Results)}
sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map).astype(np.float32)

sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5)

sub.to_csv("./submission.csv", index=False)
print(sub.head())
print("Wrote ./submission.csv with shape:", sub.shape)

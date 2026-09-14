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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import sys
import json
import glob
import random
import collections
import time
import re
import math
import warnings

import numpy as np
import pandas as pd

import pydicom

from sklearn import model_selection as sk_model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("pydicom:", pydicom.__version__)



## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

pytorch3dpath = "../input/efficientnetpyttorch3d/EfficientNet-PyTorch-3D"

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 64

if not os.path.exists(data_directory):
    alt = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
    if os.path.exists(alt):
        data_directory = alt

print("Using data_directory:", data_directory)
print("Exists?", os.path.exists(data_directory))



## === cell 2
sample_submission_path = os.path.join(data_directory, "sample_submission.csv")
train_labels_path = os.path.join(data_directory, "train_labels.csv")

sample_submission = pd.read_csv(sample_submission_path)
train_labels = pd.read_csv(train_labels_path)

sample_submission["BraTS21ID"] = sample_submission["BraTS21ID"].astype(int)
train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(int)

sample_submission["BraTS21ID5"] = sample_submission["BraTS21ID"].apply(
    lambda x: f"{x:05d}"
)
train_labels["BraTS21ID5"] = train_labels["BraTS21ID"].apply(lambda x: f"{x:05d}")

print(sample_submission.head(3))
print(train_labels.head(3), train_labels.shape)



## === cell 3
import cv2

_DCM_LIST_CACHE = {}  # (split, id5, mri_type) -> tuple(paths_sorted)

_PIXEL_LRU = collections.OrderedDict()
_PIXEL_LRU_MAX = 2048  # bounded to avoid RAM blowups

_VOL3D_CACHE = (
    {}
)  # in-memory cache: (split, id5, mri_type, img_size, num_imgs, rotate) -> (1,H,W,D)

WORK_CACHE_DIR = "/kaggle/working/vol_cache_v1"
os.makedirs(WORK_CACHE_DIR, exist_ok=True)

DCM_INDEX_PATH = os.path.join(WORK_CACHE_DIR, "dcm_index_v1.json")

IO_PARALLEL_CALLS = 4


def _lru_get(cache, key):
    try:
        val = cache.pop(key)
        cache[key] = val
        return val
    except KeyError:
        return None


def _lru_put(cache, key, val, maxsize):
    if key in cache:
        cache.pop(key, None)
    cache[key] = val
    if len(cache) > maxsize:
        cache.popitem(last=False)


_img_num_re = re.compile(r"(\d+)")


def _extract_num(path):
    base = os.path.basename(path)
    m = _img_num_re.findall(base)
    return int(m[-1]) if m else 0


def _fast_list_dcm(dir_path):
    try:
        with os.scandir(dir_path) as it:
            return [e.path for e in it if e.is_file() and e.name.endswith(".dcm")]
    except FileNotFoundError:
        return []
    except Exception:
        return glob.glob(os.path.join(dir_path, "*.dcm"))


def _build_or_load_dcm_index():
    if os.path.exists(DCM_INDEX_PATH):
        try:
            with open(DCM_INDEX_PATH, "r") as f:
                raw = json.load(f)
            for k, v in raw.items():
                _DCM_LIST_CACHE[tuple(k.split("|", 2))] = tuple(v)
            print("Loaded DICOM index entries:", len(_DCM_LIST_CACHE))
            return
        except Exception as e:
            print("Failed to load DICOM index, rebuilding. Error:", repr(e))
            _DCM_LIST_CACHE.clear()

    t0 = time.time()
    tmp = {}
    for split in ("train", "test"):
        split_dir = os.path.join(data_directory, split)
        if not os.path.exists(split_dir):
            continue
        try:
            with os.scandir(split_dir) as it:
                ids = [e.name for e in it if e.is_dir()]
        except Exception:
            ids = [
                os.path.basename(p)
                for p in glob.glob(os.path.join(split_dir, "*"))
                if os.path.isdir(p)
            ]
        for sid in ids:
            for mt in mri_types:
                dcm_dir = os.path.join(split_dir, sid, mt)
                files = _fast_list_dcm(dcm_dir)
                if files:
                    files.sort(key=_extract_num)
                tmp[f"{split}|{sid}|{mt}"] = files
    try:
        with open(DCM_INDEX_PATH, "w") as f:
            json.dump(tmp, f)
        print(
            "Built+saved DICOM index entries:",
            len(tmp),
            "in",
            round(time.time() - t0, 2),
            "s",
        )
    except Exception as e:
        print("Failed to save DICOM index:", repr(e))
    for k, v in tmp.items():
        _DCM_LIST_CACHE[tuple(k.split("|", 2))] = tuple(v)


_build_or_load_dcm_index()


def _get_sorted_dcm_files(scan_id, mri_type="FLAIR", split="test"):
    key = (split, str(scan_id), mri_type)
    files = _DCM_LIST_CACHE.get(key)
    if files is None:
        dcm_dir = os.path.join(data_directory, split, str(scan_id), mri_type)
        unsorted = _fast_list_dcm(dcm_dir)
        if not unsorted:
            files = tuple()
        else:
            unsorted.sort(key=_extract_num)
            files = tuple(unsorted)
        _DCM_LIST_CACHE[key] = files
    return files


def _resize2d_area(arr, img_size):
    if arr.shape[0] == img_size and arr.shape[1] == img_size:
        return arr.astype(np.float32, copy=False)
    return cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(
        np.float32, copy=False
    )


def load_dicom_image(path, img_size=IMAGE_SIZE, rotate=0):
    base = _lru_get(_PIXEL_LRU, path)
    if base is None:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32, copy=False)

        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept

        _lru_put(_PIXEL_LRU, path, arr, _PIXEL_LRU_MAX)
        base = arr

    arr = base
    if rotate:
        k = int(rotate) % 4
        if k:
            arr = np.rot90(arr, k=k)

    arr = _resize2d_area(arr, img_size)
    return arr


def _vol_cache_path(split, scan_id, mri_type, img_size, num_imgs, rotate):
    fname = f"{split}_{scan_id}_{mri_type}_{img_size}_{num_imgs}_r{int(rotate)}.npy"
    return os.path.join(WORK_CACHE_DIR, fname)


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    cache_key = (
        split,
        str(scan_id),
        mri_type,
        int(img_size),
        int(num_imgs),
        int(rotate),
    )
    cached = _VOL3D_CACHE.get(cache_key)
    if cached is not None:
        return cached

    npy_path = _vol_cache_path(
        split, str(scan_id), mri_type, img_size, num_imgs, rotate
    )
    if os.path.exists(npy_path):
        out = np.load(npy_path, allow_pickle=False)
        _VOL3D_CACHE[cache_key] = out
        return out

    files = _get_sorted_dcm_files(scan_id, mri_type=mri_type, split=split)

    if len(files) == 0:
        out = np.zeros((1, img_size, img_size, num_imgs), dtype=np.float32)
        _VOL3D_CACHE[cache_key] = out
        try:
            np.save(npy_path, out, allow_pickle=False)
        except Exception:
            pass
        return out

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    chosen = files[p1:p2] if (p2 > p1) else files

    depth = min(len(chosen), num_imgs)
    img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
    for i in range(depth):
        img3d[:, :, i] = load_dicom_image(chosen[i], img_size=img_size, rotate=rotate)

    mn, mx = float(np.min(img3d)), float(np.max(img3d))
    if mn < mx:
        img3d = (img3d - mn) / (mx - mn)
    else:
        img3d[:] = 0.0

    out = np.expand_dims(img3d.astype(np.float32, copy=False), 0)  # (1,H,W,D)
    _VOL3D_CACHE[cache_key] = out
    try:
        np.save(npy_path, out, allow_pickle=False)
    except Exception:
        pass
    return out




## === cell 4
print("Skipping pre-training DICOM sanity load to save time.")



## === cell 5
bad_ids = {109, 123, 709}
train_df = (
    train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].copy().reset_index(drop=True)
)

trn_df, val_df = sk_model_selection.train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["MGMT_value"]
)

print("Train/Val:", trn_df.shape, val_df.shape)


def make_tf_dataset(
    df, is_train=True, batch_size=1, shuffle=True, split="train", mri_type="FLAIR"
):
    id5 = df["BraTS21ID5"].astype(str).values
    if is_train and "MGMT_value" in df.columns:
        y = df["MGMT_value"].astype(np.float32).values
    else:
        y = None

    if y is not None:
        ds = tf.data.Dataset.from_tensor_slices((id5, y))
    else:
        ds = tf.data.Dataset.from_tensor_slices(id5)

    if is_train and shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _py_load(id5_t):
        sid = id5_t.numpy().decode("utf-8")
        npy_path = _vol_cache_path(split, sid, mri_type, IMAGE_SIZE, NUM_IMAGES, 0)
        if os.path.exists(npy_path):
            vol = np.load(npy_path, allow_pickle=False)
        else:
            vol = load_dicom_images_3d(sid, split=split, mri_type=mri_type)
        x = vol[0][..., None].astype(np.float32, copy=False)  # (H,W,D,1)
        return x

    def _load_x(id5_tensor, y_tensor=None):
        x = tf.py_function(_py_load, [id5_tensor], Tout=tf.float32)
        x.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1))
        if y_tensor is None:
            return x
        return x, y_tensor

    if y is not None:
        ds = ds.map(
            lambda a, b: _load_x(a, b),
            num_parallel_calls=IO_PARALLEL_CALLS,
            deterministic=True,
        )
    else:
        ds = ds.map(
            lambda a: _load_x(a, None),
            num_parallel_calls=IO_PARALLEL_CALLS,
            deterministic=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = make_tf_dataset(
    trn_df, is_train=True, batch_size=1, shuffle=True, split="train", mri_type="FLAIR"
)
val_dataset = make_tf_dataset(
    val_df, is_train=True, batch_size=1, shuffle=False, split="train", mri_type="FLAIR"
)
test_dataset = make_tf_dataset(
    sample_submission,
    is_train=False,
    batch_size=1,
    shuffle=False,
    split="test",
    mri_type="FLAIR",
)

for xb, yb in train_dataset.take(1):
    print("Batch X shape:", xb.shape, "Batch y shape:", yb.shape)




## === cell 6
def build_3d_cnn(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv3D(8, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(16, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model


model = build_3d_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)
model.summary()



## === cell 7
EPOCHS = 2

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=2,
)



## === cell 8
preds = (
    model.predict(
        test_dataset,
        verbose=1,
    )
    .reshape(-1)
    .astype(np.float32)
)

preds = np.clip(preds, 0.0, 1.0)

print(
    "Preds:",
    preds[:10],
    "min/max:",
    float(preds.min()),
    float(preds.max()),
    "len:",
    len(preds),
    "expected:",
    len(sample_submission),
)



## === cell 9
submission = pd.DataFrame(
    {
        "BraTS21ID": sample_submission["BraTS21ID"].astype(int).values,
        "MGMT_value": preds,
    }
)

assert submission.shape[0] == sample_submission.shape[0]
assert list(submission.columns) == ["BraTS21ID", "MGMT_value"]
assert submission["MGMT_value"].notna().all()

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")

print("Done.")

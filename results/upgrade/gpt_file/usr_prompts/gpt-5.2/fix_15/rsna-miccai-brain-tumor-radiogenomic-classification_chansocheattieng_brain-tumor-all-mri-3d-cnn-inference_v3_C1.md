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
import glob
import re
import math
import numpy as np
import pandas as pd
import cv2

try:
    cv2.setNumThreads(0)
except Exception:
    pass

pydicom = None
_HAVE_PYDICOM = False

from sklearn import model_selection as sk_model_selection

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC

tf.random.set_seed(0)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Keras module:", keras.__name__)
print("CPU count:", os.cpu_count())


## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types_orig = ["FLAIR", "T1w", "T1wCE", "T2w"]
mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

IMAGE_SIZE = 128
NUM_IMAGES_PER_TYPE = 32
NUM_IMAGES = NUM_IMAGES_PER_TYPE * len(mri_types)
BATCH_SIZE = 2  # keep as-is

train_df = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)

to_exclude = [109, 123, 709]
train_df = train_df[~train_df["BraTS21ID"].isin(to_exclude)].reset_index(drop=True)

train_df["BraTS21ID5"] = [format(x, "05d") for x in train_df.BraTS21ID]
print("Train rows:", len(train_df))
train_df.head(3)


## === cell 2
sample_submission = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(x, "05d") for x in test.BraTS21ID]
test.head(3)


## === cell 3
pass


## === cell 4
_DICOM_NUM_RE = re.compile(r"(\d+)")


def _dicom_num_key_from_name(name: str) -> int:
    m = _DICOM_NUM_RE.search(name)
    return int(m.group(1)) if m else 0


_DCM_LIST_CACHE = {}
DCM_INDEX_DIR = "/kaggle/working/dcm_index_v1"
os.makedirs(DCM_INDEX_DIR, exist_ok=True)


def _dcm_index_path(folder: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", folder.strip("/"))[-180:]
    return os.path.join(DCM_INDEX_DIR, safe + ".npy")


def _list_dcm_sorted_fast(folder: str):
    v = _DCM_LIST_CACHE.get(folder, None)
    if v is not None:
        return v

    ipath = _dcm_index_path(folder)
    if os.path.exists(ipath):
        arr = np.load(ipath, allow_pickle=False)
        files = arr.tolist()
        _DCM_LIST_CACHE[folder] = files
        return files

    try:
        with os.scandir(folder) as it:
            names = [e.name for e in it if e.is_file() and e.name.endswith(".dcm")]
    except FileNotFoundError:
        names = []

    if names:
        names.sort(key=_dicom_num_key_from_name)
        files = [os.path.join(folder, n) for n in names]
    else:
        files = []

    tmp = ipath + ".tmp.npy"
    np.save(tmp, np.asarray(files, dtype=np.str_), allow_pickle=False)
    os.replace(tmp, ipath)

    _DCM_LIST_CACHE[folder] = files
    return files


CACHE_DIR = "/kaggle/working/vol_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(split: str, sid: str) -> str:
    return os.path.join(
        CACHE_DIR,
        f"{split}_{sid}_sz{IMAGE_SIZE}_n{NUM_IMAGES_PER_TYPE}_mt{len(mri_types)}.npy",
    )


def _read_dicom_pixels_fallback(path: str) -> np.ndarray:
    try:
        with open(path, "rb") as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    except Exception:
        img = None

    if img is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32, copy=False)


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    data = _read_dicom_pixels_fallback(path)

    if rotate > 0:
        rot_choices = [
            0,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])

    data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return data


def _pick_slice_paths_centered(files, num_imgs):
    n = len(files)
    if n == 0:
        return []
    middle = n // 2
    p1 = max(0, middle - num_imgs)
    p2 = min(n, middle + num_imgs)
    pick = files[p1:p2:2]
    if len(pick) < (num_imgs // 2):
        half = num_imgs // 2
        p1 = max(0, middle - half)
        p2 = min(n, middle + half)
        pick = files[p1:p2]
        if len(pick) == 0:
            pick = [files[middle]]
    return pick


from concurrent.futures import ThreadPoolExecutor

_SLICE_THREADPOOL = None
_SLICE_NTHREADS = max(1, min(8, (os.cpu_count() or 2)))


def _get_slice_pool():
    global _SLICE_THREADPOOL
    if _SLICE_THREADPOOL is None:
        _SLICE_THREADPOOL = ThreadPoolExecutor(max_workers=_SLICE_NTHREADS)
    return _SLICE_THREADPOOL


def load_dicom_images_3d(
    scan_id,
    split="train",
    mri_type="FLAIR",
    num_imgs=NUM_IMAGES_PER_TYPE,
    img_size=IMAGE_SIZE,
    rotate=0,
    files_sorted=None,
):
    if files_sorted is None:
        folder = f"{data_directory}/{split}/{scan_id}/{mri_type}"
        files = _list_dcm_sorted_fast(folder)
    else:
        files = files_sorted

    pick = _pick_slice_paths_centered(files, num_imgs)

    if len(pick) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    pool = _get_slice_pool()
    slices = list(
        pool.map(lambda p: load_dicom_image(p, img_size=img_size, rotate=rotate), pick)
    )

    d = len(slices)
    img3d = np.empty((img_size, img_size, d), dtype=np.float32)
    for k, sl in enumerate(slices):
        img3d[:, :, k] = sl

    if img3d.shape[-1] < num_imgs:
        n_front = (num_imgs - img3d.shape[-1]) // 2
        n_back = num_imgs - img3d.shape[-1] - n_front
        if n_front:
            img3d = np.concatenate(
                (np.zeros((img_size, img_size, n_front), dtype=img3d.dtype), img3d),
                axis=-1,
            )
        if n_back:
            img3d = np.concatenate(
                (img3d, np.zeros((img_size, img_size, n_back), dtype=img3d.dtype)),
                axis=-1,
            )
    elif img3d.shape[-1] > num_imgs:
        img3d = img3d[:, :, :num_imgs]

    img3d = img3d.astype(np.float32, copy=False)
    mn, mx = float(np.min(img3d)), float(np.max(img3d))
    if mn < mx:
        img3d = (img3d - mn) / (mx - mn)

    return np.expand_dims(img3d, 0)


_MEM_VOL_CACHE = {}
_MEM_VOL_CACHE_MAX = 64


def _mem_cache_get(key):
    return _MEM_VOL_CACHE.get(key, None)


def _mem_cache_put(key, value):
    if key in _MEM_VOL_CACHE:
        _MEM_VOL_CACHE[key] = value
        return
    if len(_MEM_VOL_CACHE) >= _MEM_VOL_CACHE_MAX:
        _MEM_VOL_CACHE.pop(next(iter(_MEM_VOL_CACHE)))
    _MEM_VOL_CACHE[key] = value


def load_dicom_images_3d_all(scan_id, split="train", dicom_index=None):
    cpath = _cache_path(split, scan_id)

    mem_key = (split, scan_id, "vol_v2")
    v = _mem_cache_get(mem_key)
    if v is not None:
        return v

    if os.path.exists(cpath):
        v = np.load(cpath, mmap_mode="r")
        _mem_cache_put(mem_key, v)
        return v

    img3d_all = np.concatenate(
        [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types],
        axis=-1,
    )

    tmp = cpath + ".tmp.npy"
    np.save(tmp, img3d_all)
    os.replace(tmp, cpath)

    v = np.load(cpath, mmap_mode="r")
    _mem_cache_put(mem_key, v)
    return v


_sanity_id = test["BraTS21ID5"].iloc[0]
print("Sanity id:", _sanity_id, "expected cache path:", _cache_path("test", _sanity_id))


## === cell 5
pass


## === cell 6
df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

print("Train/valid:", len(df_train), len(df_valid))


## === cell 7
df_train.head()


## === cell 8
del train_df


## === cell 9
pass


## === cell 10
from tensorflow.keras.utils import Sequence
from collections import OrderedDict


class _LRUCache:
    def __init__(self, max_items=64):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, key):
        v = self._d.get(key, None)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


class Dataset(Sequence):
    def __init__(
        self,
        df,
        split="train",
        is_train=True,
        batch_size=BATCH_SIZE,
        shuffle=True,
        dicom_index=None,
        cache_max_items=64,
    ):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["BraTS21ID5"].values
        self.y = (
            self.df["MGMT_value"].values
            if ("MGMT_value" in self.df.columns and is_train)
            else None
        )
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.indices = np.arange(len(self.paths))
        self.dicom_index = dicom_index
        self.cache = _LRUCache(max_items=cache_max_items)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        batch_ids = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_paths = self.paths[batch_ids]

        X = np.empty(
            (len(batch_paths), IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=np.float32
        )

        for i, sid in enumerate(batch_paths):
            key = (self.split, sid)
            vol = self.cache.get(key)
            if vol is None:
                vol = load_dicom_images_3d_all(
                    sid, self.split, dicom_index=self.dicom_index
                )  # (1,H,W,D)
                vol = np.expand_dims(np.asarray(vol), axis=-1)[0].astype(
                    np.float32, copy=False
                )  # (H,W,D,1)
                self.cache.put(key, vol)
            X[i] = vol

        if self.is_train:
            y = self.y[batch_ids].astype(np.float32).reshape(-1, 1)  # (B, 1)
            return X, y
        return X

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            np.random.shuffle(self.indices)




## === cell 11
import multiprocessing as mp


def _cache_worker(args):
    sid, split = args
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    global _SLICE_THREADPOOL
    _SLICE_THREADPOOL = None
    _ = load_dicom_images_3d_all(sid, split=split, dicom_index=None)
    return sid


def _ensure_cached_volumes(ids5, split):
    ids5 = list(map(str, ids5))
    to_make = []
    for sid in ids5:
        cpath = _cache_path(split, sid)
        if not os.path.exists(cpath):
            to_make.append(sid)

    if not to_make:
        print(f"Cache warm: {split} ({len(ids5)} ids), nothing to do.")
        return

    ncpu = os.cpu_count() or 2
    nproc = min(6, max(2, ncpu - 1))
    chunksize = max(4, len(to_make) // (nproc * 2))
    print(
        f"Building cache for {split}: {len(to_make)}/{len(ids5)} volumes using {nproc} proc (chunksize={chunksize})..."
    )

    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    work_items = [(sid, split) for sid in to_make]
    with ctx.Pool(processes=nproc, maxtasksperchild=64) as pool:
        for k, sid in enumerate(
            pool.imap_unordered(_cache_worker, work_items, chunksize=chunksize), 1
        ):
            if k % 25 == 0 or k == len(to_make):
                print(f"  cached {k}/{len(to_make)} ({split})")


def _make_tf_dataset(
    ids5, y=None, split="train", dicom_index=None, batch_size=1, shuffle=False
):
    ids5 = np.asarray(ids5, dtype=str)
    cache_paths = np.asarray([_cache_path(split, sid) for sid in ids5], dtype="S")

    def _py_load_npy(path_bytes):
        path = path_bytes.decode("utf-8")
        vol = np.load(path, mmap_mode="r")  # (1,H,W,D)
        x = np.asarray(vol)[0][..., None].astype(np.float32, copy=False)  # (H,W,D,1)
        return x

    def _tf_load(path):
        x = tf.numpy_function(_py_load_npy, [path], Tout=tf.float32)
        x.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1))
        return x

    ds_paths = tf.data.Dataset.from_tensor_slices(cache_paths)
    if shuffle:
        ds_paths = ds_paths.shuffle(
            buffer_size=len(cache_paths), seed=0, reshuffle_each_iteration=True
        )

    opts = tf.data.Options()
    opts.deterministic = True
    ds_paths = ds_paths.with_options(opts)

    ds_x = ds_paths.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)

    if y is not None:
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        ds_y = tf.data.Dataset.from_tensor_slices(y)
        ds = tf.data.Dataset.zip((ds_x, ds_y))
    else:
        ds = ds_x

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


train_ids = df_train["BraTS21ID5"].values
valid_ids = df_valid["BraTS21ID5"].values
test_ids = test["BraTS21ID5"].values

all_train_needed = np.unique(np.concatenate([train_ids, valid_ids]))

_ensure_cached_volumes(all_train_needed, "train")
_ensure_cached_volumes(test_ids, "test")

train_dataset = _make_tf_dataset(
    train_ids,
    y=df_train["MGMT_value"].values,
    split="train",
    dicom_index=None,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
valid_dataset = _make_tf_dataset(
    valid_ids,
    y=df_valid["MGMT_value"].values,
    split="train",
    dicom_index=None,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_dataset = _make_tf_dataset(
    test_ids,
    y=None,
    split="test",
    dicom_index=None,
    batch_size=4,
    shuffle=False,
)


## === cell 12
pass


## === cell 13
pass


## === cell 14
pass


## === cell 15
pass


## === cell 16
pass


## === cell 17
pass


## === cell 18
pass


## === cell 19
pass


## === cell 20
pass


## === cell 21
pass


## === cell 22
pass


## === cell 23
pass


## === cell 24
pass


## === cell 25
pass


## === cell 26
pass


## === cell 27
pass


## === cell 28
pass


## === cell 29
pass


## === cell 30
pass


## === cell 31
pass


## === cell 32
pass


## === cell 33
pass


## === cell 34
pass


## === cell 35
pass


## === cell 36
pass




## === cell 37
def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3D_CNN")
    return model




## === cell 38
model = get_model()
model.summary()


## === cell 39
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[AUC(name="auc")],
)


## === cell 40
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=2,
    verbose=1,
)


## === cell 41
pass


## === cell 42
pass


## === cell 43
pass


## === cell 44
pass


## === cell 45
pass


## === cell 46
pass


## === cell 47
test_dataset


## === cell 48
predictions = model.predict(
    test_dataset,
    verbose=1,
).reshape(-1)

predictions = np.clip(predictions, 0.0, 1.0)
print(
    "Predictions:",
    predictions.shape,
    "min/max/mean:",
    float(predictions.min()),
    float(predictions.max()),
    float(predictions.mean()),
)


## === cell 49
submission = pd.DataFrame(
    {
        "BraTS21ID": sample_submission["BraTS21ID"].values,
        "MGMT_value": predictions,
    }
)


## === cell 50
submission.head()


## === cell 51
assert len(submission) == len(sample_submission)
assert (submission["BraTS21ID"].values == sample_submission["BraTS21ID"].values).all()


## === cell 52
submission.describe()


## === cell 53
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

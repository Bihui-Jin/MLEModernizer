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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import glob
import re
import math
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

from random import shuffle
from sklearn import model_selection as sk_model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC

SEED = 12
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    ncpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(6, max(2, ncpu)))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)



## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
if not os.path.exists(data_directory):
    data_directory = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"

assert os.path.exists(data_directory), f"Dataset directory not found: {data_directory}"



## === cell 2
mri_types_orig = ["FLAIR", "T1w", "T1wCE", "T2w"]
mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

IMAGE_SIZE = 128
NUM_IMAGES_PER_TYPE = 32
NUM_IMAGES = NUM_IMAGES_PER_TYPE * len(mri_types)
BATCH_SIZE = 4

train_df = pd.read_csv(f"{data_directory}/train_labels.csv")

to_exclude = [109, 123, 709]
train_df = train_df[~train_df["BraTS21ID"].isin(to_exclude)].copy()

train_df["BraTS21ID5"] = [format(x, "05d") for x in train_df.BraTS21ID]
print(len(train_df))
train_df.head(3)



## === cell 3
sample_submission = pd.read_csv(f"{data_directory}/sample_submission.csv")
test = sample_submission.copy()
test["BraTS21ID5"] = [format(x, "05d") for x in test.BraTS21ID]
test.head(3)



## === cell 4
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import multiprocessing as mp

try:
    import pydicom  # kept for compatibility

    _HAVE_PYDICOM = True
except Exception:
    pydicom = None
    _HAVE_PYDICOM = False

_num_tail_re = re.compile(r"(\d+)(?!.*\d)")


def _fast_dcm_sort_key(path: str):
    b = os.path.basename(path)
    m = _num_tail_re.search(b)
    return int(m.group(1)) if m else b


_DCM_INDEX = {}  # key: (split, scan_id, mri_type) -> tuple(filepaths)


def _build_dcm_index_for_split(split: str):
    root = os.path.join(data_directory, split)
    idx = {}
    with os.scandir(root) as it_scans:
        for e_scan in it_scans:
            if not e_scan.is_dir():
                continue
            scan_id = e_scan.name
            scan_path = e_scan.path
            for mt in mri_types:
                mt_path = os.path.join(scan_path, mt)
                if not os.path.isdir(mt_path):
                    idx[(split, scan_id, mt)] = tuple()
                    continue
                files = []
                with os.scandir(mt_path) as it_files:
                    for ef in it_files:
                        if ef.is_file() and ef.name.endswith(".dcm"):
                            files.append(ef.path)
                files.sort(key=_fast_dcm_sort_key)
                idx[(split, scan_id, mt)] = tuple(files)
    return idx


def _load_or_build_dcm_index():
    cache_dir = "/kaggle/working"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "dcm_index_v2.npy")
    if os.path.exists(cache_path):
        try:
            obj = np.load(cache_path, allow_pickle=True).item()
            if isinstance(obj, dict) and len(obj) > 0:
                return obj
        except Exception:
            pass
    idx = {}
    idx.update(_build_dcm_index_for_split("train"))
    idx.update(_build_dcm_index_for_split("test"))
    try:
        np.save(cache_path, idx, allow_pickle=True)
    except Exception:
        pass
    return idx


_DCM_INDEX = _load_or_build_dcm_index()


@lru_cache(maxsize=32768)
def _list_dcm_files(scan_id, split, mri_type):
    return _DCM_INDEX.get((split, scan_id, mri_type), tuple())


@lru_cache(maxsize=32768)
def _selected_slice_files(scan_id, split, mri_type, num_imgs):
    files = _list_dcm_files(scan_id, split, mri_type)
    if len(files) == 0:
        return tuple()

    n = len(files)
    if n <= num_imgs:
        return tuple(files)

    mid = n // 2
    half = num_imgs // 2
    start = max(0, mid - half)
    end = start + num_imgs
    if end > n:
        end = n
        start = end - num_imgs
    return tuple(files[start:end])


def _tf_decode_dicom_to_float32(path: str) -> np.ndarray:
    b = tf.io.read_file(path)
    img = tf.io.decode_dicom_image(
        b,
        color_dim=False,
        on_error="skip",
        scale="auto",
        dtype=tf.uint16,
    )
    img = tf.cast(img, tf.float32)
    img = tf.squeeze(img)
    return img.numpy()


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    img = None
    try:
        img = _tf_decode_dicom_to_float32(path)
    except Exception:
        img = None

    if img is None and _HAVE_PYDICOM:
        try:
            ds = pydicom.dcmread(path, force=True)
            img = ds.pixel_array.astype(np.float32, copy=False)
        except Exception:
            img = None

    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.zeros((img_size, img_size), dtype=np.float32)
        img = img.astype(np.float32, copy=False)

    if rotate > 0:
        rot_choices = [
            0,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        img = cv2.rotate(img, rot_choices[rotate])

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


class _LRUCache:
    def __init__(self, max_items=1024):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, key):
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


_VOL_CACHE = _LRUCache(max_items=1536)


def load_dicom_images_3d(
    scan_id,
    split="train",
    mri_type="FLAIR",
    num_imgs=NUM_IMAGES_PER_TYPE,
    img_size=IMAGE_SIZE,
    rotate=0,
):
    cache_key = (scan_id, split, mri_type, num_imgs, img_size, rotate)
    cached = _VOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    files_sel = _selected_slice_files(scan_id, split, mri_type, num_imgs)

    if len(files_sel) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        out = np.expand_dims(img3d, 0)
        _VOL_CACHE.put(cache_key, out)
        return out

    imgs = [load_dicom_image(p, img_size=img_size, rotate=rotate) for p in files_sel]
    img3d = np.stack(imgs).T  # (H,W,N)

    if img3d.shape[-1] < num_imgs:
        n_front = (num_imgs - img3d.shape[-1]) // 2
        n_back = num_imgs - img3d.shape[-1] - n_front
        if n_front > 0 or n_back > 0:
            n_zero_front = np.zeros((img_size, img_size, n_front), dtype=np.float32)
            n_zero_back = np.zeros((img_size, img_size, n_back), dtype=np.float32)
            img3d = np.concatenate((n_zero_front, img3d, n_zero_back), axis=-1)
    elif img3d.shape[-1] > num_imgs:
        start = (img3d.shape[-1] - num_imgs) // 2
        img3d = img3d[..., start : start + num_imgs]

    mn, mx = float(np.min(img3d)), float(np.max(img3d))
    if mn < mx:
        img3d = (img3d - mn) / (mx - mn)
    else:
        img3d = np.zeros_like(img3d, dtype=np.float32)

    out = np.expand_dims(img3d.astype(np.float32), 0)
    _VOL_CACHE.put(cache_key, out)
    return out


@lru_cache(maxsize=4096)
def _load_dicom_images_3d_all_cached(scan_id, split):
    img3d_all = np.concatenate(
        [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types],
        axis=-1,
    )
    return img3d_all.astype(np.float32, copy=False)


def load_dicom_images_3d_all(scan_id, split="train"):
    return _load_dicom_images_3d_all_cached(scan_id, split)




## === cell 5
a = load_dicom_images_3d_all("00002", "test")
print(a.shape, np.min(a), np.max(a), np.mean(a))



## === cell 6
from tensorflow.keras.utils import Sequence

_MEMMAP_WORKER_SPLIT = None


def _memmap_worker_init(split):
    global _MEMMAP_WORKER_SPLIT
    _MEMMAP_WORKER_SPLIT = split


def _memmap_worker_build_one(args):
    i, sid = args
    split = _MEMMAP_WORKER_SPLIT
    vol = load_dicom_images_3d_all(str(sid), split=split)[0]  # (H,W,NUM_IMAGES)
    return i, vol


def _ensure_memmap_cache(ids5, split, cache_basename, force_rebuild=False):
    cache_dir = "/kaggle/working"
    os.makedirs(cache_dir, exist_ok=True)
    path_dat = os.path.join(cache_dir, f"{cache_basename}.{split}.dat")
    path_ids = os.path.join(cache_dir, f"{cache_basename}.{split}.ids.npy")

    ids5 = np.asarray(ids5, dtype=object)
    n = len(ids5)
    shape = (n, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1)

    if (not force_rebuild) and os.path.exists(path_dat) and os.path.exists(path_ids):
        prev_ids = np.load(path_ids, allow_pickle=True)
        if len(prev_ids) == n and np.all(prev_ids == ids5):
            mm = np.memmap(path_dat, dtype="float32", mode="r", shape=shape)
            return mm, ids5

    mm_w = np.memmap(path_dat, dtype="float32", mode="w+", shape=shape)

    max_workers = min(6, max(1, (os.cpu_count() or 4) // 2))
    ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp

    with ctx.Pool(
        processes=max_workers, initializer=_memmap_worker_init, initargs=(split,)
    ) as pool:
        for k, (i, vol) in enumerate(
            pool.imap(_memmap_worker_build_one, enumerate(ids5), chunksize=8), start=1
        ):
            mm_w[i, ..., 0] = vol.astype(np.float32, copy=False)
            if k % 32 == 0:
                mm_w.flush()

    mm_w.flush()
    np.save(path_ids, ids5, allow_pickle=True)
    mm_r = np.memmap(path_dat, dtype="float32", mode="r", shape=shape)
    return mm_r, ids5


class Dataset(Sequence):
    def __init__(
        self, df, split="train", is_train=True, batch_size=BATCH_SIZE, shuffle=True
    ):
        self.df = df.reset_index(drop=True).copy()
        self.idx = self.df["BraTS21ID"].values
        self.paths = self.df["BraTS21ID5"].values

        self.y = (
            self.df["MGMT_value"].values if "MGMT_value" in self.df.columns else None
        )

        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split

        self._mm, self._mm_ids = _ensure_memmap_cache(
            self.paths,
            split=self.split,
            cache_basename=f"volcache_v2_{self.split}_{len(self.paths)}",
            force_rebuild=False,
        )

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            perm = np.random.permutation(len(self.idx))
            self.idx = self.idx[perm]
            self.paths = self.paths[perm]
            if self.y is not None:
                self.y = self.y[perm]

        if not hasattr(self, "_cache_pos"):
            self._cache_pos = {sid: i for i, sid in enumerate(self._mm_ids.tolist())}
        self._row_indices = np.fromiter(
            (self._cache_pos[sid] for sid in self.paths),
            count=len(self.paths),
            dtype=np.int32,
        )

    def __getitem__(self, ids):
        s = ids * self.batch_size
        e = (ids + 1) * self.batch_size
        rows = self._row_indices[s:e]
        batch_X = np.asarray(self._mm[rows], dtype=np.float32)
        if self.is_train and self.y is not None:
            batch_y = self.y[s:e].astype(np.float32, copy=False)
            return batch_X, batch_y
        else:
            return batch_X


def make_tf_dataset(ds: Dataset):
    rows = ds._row_indices.copy()
    mm = ds._mm
    if ds.y is None:

        def gen():
            for r in rows:
                yield np.asarray(mm[r], dtype=np.float32)

        out_sig = tf.TensorSpec(
            shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=tf.float32
        )
        tds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
        tds = tds.batch(ds.batch_size, drop_remainder=False)
    else:
        y = ds.y.astype(np.float32, copy=False)

        def gen():
            for i, r in enumerate(rows):
                yield np.asarray(mm[r], dtype=np.float32), y[i]

        out_sig = (
            tf.TensorSpec(
                shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
        tds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
        tds = tds.batch(ds.batch_size, drop_remainder=False)

    return tds.prefetch(tf.data.AUTOTUNE)




## === cell 7
df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["MGMT_value"],
)

train_dataset = Dataset(
    df_train, split="train", is_train=True, batch_size=BATCH_SIZE, shuffle=True
)

valid_dataset = Dataset(
    df_valid, split="train", is_train=False, batch_size=BATCH_SIZE, shuffle=False
)

test_dataset = Dataset(test, split="test", is_train=False, batch_size=1, shuffle=False)

train_tds = make_tf_dataset(train_dataset)
valid_tds = make_tf_dataset(valid_dataset)
test_tds = make_tf_dataset(test_dataset)

del train_df




## === cell 8
def plot_sample_all(images, label, j):
    plt.figure(figsize=(35, 35))
    for i in range(NUM_IMAGES):
        plt.subplot(16, 16, (i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[0, :, :, i, j], cmap="gray")
    plt.show()




## === cell 9
try:
    i = 0
    images, label = train_dataset[i]
    print("Train batch shape:", images.shape, "labels:", label[: min(4, len(label))])
except Exception as e:
    print("Skipping visualization sanity check due to:", repr(e))




## === cell 10
def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.1)(x)

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
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3D_CNN")
    return model




## === cell 11
model = get_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[AUC(name="auc")],
)
model.summary()



## === cell 12
checkpoint_path = "best_model.weights.h5"
callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
]

history = model.fit(
    train_tds,
    validation_data=valid_tds,
    epochs=2,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)



## === cell 13
predictions = (
    model.predict(
        test_tds,
        verbose=1,
    )
    .reshape(-1)
    .astype(np.float32)
)

predictions = np.clip(predictions, 0.0, 1.0)
print("Preds:", predictions[:5], "n=", len(predictions))

if len(predictions) != len(sample_submission):
    raise ValueError(
        f"Prediction length mismatch: preds={len(predictions)} vs sample_submission={len(sample_submission)}"
    )



## === cell 14
submission = pd.DataFrame(
    {
        "BraTS21ID": sample_submission["BraTS21ID"].values,
        "MGMT_value": predictions,
    }
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))

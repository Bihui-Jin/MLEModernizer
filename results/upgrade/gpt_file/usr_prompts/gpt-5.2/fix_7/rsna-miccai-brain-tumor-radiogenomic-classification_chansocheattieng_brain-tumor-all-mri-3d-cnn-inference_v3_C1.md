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
    import pydicom  # keep import for metadata fallback if needed

    _HAVE_PYDICOM = True
except Exception as e:
    print(
        "Warning: pydicom import failed, falling back to OpenCV DICOM reader. Error:",
        repr(e),
    )
    pydicom = None
    _HAVE_PYDICOM = False

from sklearn import model_selection as sk_model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
tf.random.set_seed(0)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)



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


def _dicom_num_key(path: str) -> int:
    base = os.path.basename(path)
    m = _DICOM_NUM_RE.search(base)
    return int(m.group(1)) if m else 0


def _build_dicom_index(data_dir, split, ids5, mri_types):
    idx = {}
    for sid in ids5:
        per_type = {}
        for mt in mri_types:
            dcm_dir = f"{data_dir}/{split}/{sid}/{mt}/*.dcm"
            files = glob.glob(dcm_dir)
            if files:
                files.sort(key=_dicom_num_key)
            per_type[mt] = files
        idx[sid] = per_type
    return idx


CACHE_DIR = "/kaggle/working/vol_cache_v1"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(split: str, sid: str) -> str:
    return os.path.join(
        CACHE_DIR,
        f"{split}_{sid}_sz{IMAGE_SIZE}_n{NUM_IMAGES_PER_TYPE}_mt{len(mri_types)}.npy",
    )


def _read_dicom_pixels_fallback(path: str) -> np.ndarray:
    """
    Fallback DICOM reader that does not require pydicom.
    Uses OpenCV's DICOM support if available. Returns float32 array.
    """
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32, copy=False)


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    if _HAVE_PYDICOM:
        ds = pydicom.dcmread(path, force=True)

        data = None
        try:
            if hasattr(ds, "pixel_array"):
                data = ds.pixel_array
        except Exception:
            data = None

        if data is None:
            rows = int(getattr(ds, "Rows", 0) or 0)
            cols = int(getattr(ds, "Columns", 0) or 0)
            bits_alloc = int(getattr(ds, "BitsAllocated", 16) or 16)
            pixel_repr = int(getattr(ds, "PixelRepresentation", 0) or 0)

            if not hasattr(ds, "PixelData"):
                data = np.zeros((img_size, img_size), dtype=np.uint16)
            else:
                raw = ds.PixelData
                if bits_alloc <= 8:
                    dtype = np.int8 if pixel_repr == 1 else np.uint8
                else:
                    dtype = np.int16 if pixel_repr == 1 else np.uint16

                arr = np.frombuffer(raw, dtype=dtype)
                if rows > 0 and cols > 0 and arr.size >= rows * cols:
                    data = arr[: rows * cols].reshape(rows, cols)
                else:
                    side = int(np.sqrt(arr.size))
                    side = max(side, 1)
                    data = arr[: side * side].reshape(side, side)

        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        data = data.astype(np.float32, copy=False) * slope + intercept
    else:
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
        dcm_dir = f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"
        files = glob.glob(dcm_dir)
        files.sort(key=_dicom_num_key)
    else:
        files = files_sorted

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    middle = len(files) // 2
    num_imgs2 = num_imgs
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    pick = files[p1:p2:2]
    if len(pick) == 0:
        pick = files[max(0, middle - 1) : min(len(files), middle + 1)]

    img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in pick]).T

    if img3d.shape[-1] < num_imgs // 2:
        num_imgs2 = num_imgs // 2
        p1 = max(0, middle - num_imgs2)
        p2 = min(len(files), middle + num_imgs2)
        pick = files[p1:p2]
        if len(pick) == 0:
            pick = [files[middle]]
        img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in pick]).T

    if img3d.shape[-1] < num_imgs:
        n_front = (num_imgs - img3d.shape[-1]) // 2
        n_back = num_imgs - img3d.shape[-1] - n_front
        n_zero_front = np.zeros((img_size, img_size, n_front), dtype=img3d.dtype)
        n_zero_back = np.zeros((img_size, img_size, n_back), dtype=img3d.dtype)
        img3d = np.concatenate((n_zero_front, img3d, n_zero_back), axis=-1)
    elif img3d.shape[-1] > num_imgs:
        img3d = img3d[:, :, :num_imgs]

    img3d = img3d.astype(np.float32, copy=False)
    mn, mx = float(np.min(img3d)), float(np.max(img3d))
    if mn < mx:
        img3d = (img3d - mn) / (mx - mn)

    return np.expand_dims(img3d, 0)


_MEM_VOL_CACHE = {}
_MEM_VOL_CACHE_MAX = 128


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

    mem_key = (split, scan_id, "vol_v1")
    v = _mem_cache_get(mem_key)
    if v is not None:
        return v

    if os.path.exists(cpath):
        v = np.load(cpath, mmap_mode=None)
        _mem_cache_put(mem_key, v)
        return v

    if dicom_index is None:
        img3d_all = np.concatenate(
            [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types],
            axis=-1,
        )
    else:
        per_type = dicom_index.get(scan_id, None)
        if per_type is None:
            img3d_all = np.concatenate(
                [
                    load_dicom_images_3d(scan_id, split, mri_type)
                    for mri_type in mri_types
                ],
                axis=-1,
            )
        else:
            parts = []
            for mri_type in mri_types:
                parts.append(
                    load_dicom_images_3d(
                        scan_id,
                        split=split,
                        mri_type=mri_type,
                        files_sorted=per_type.get(mri_type, []),
                    )
                )
            img3d_all = np.concatenate(parts, axis=-1)

    tmp = cpath + ".tmp.npy"
    np.save(tmp, img3d_all)
    os.replace(tmp, cpath)

    _mem_cache_put(mem_key, img3d_all)
    return img3d_all


_sanity_id = test["BraTS21ID5"].iloc[0]
a = load_dicom_images_3d_all(_sanity_id, "test")
print("Sanity volume shape:", a.shape, "min/max:", float(np.min(a)), float(np.max(a)))



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
                vol = np.expand_dims(vol, axis=-1)[0].astype(
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
def _ensure_cached_volumes(ids5, split, dicom_index):
    missing = []
    for sid in ids5:
        if not os.path.exists(_cache_path(split, sid)):
            missing.append(sid)
    if missing:
        for sid in missing:
            _ = load_dicom_images_3d_all(sid, split, dicom_index=dicom_index)


def _make_tf_dataset(
    ids5, y=None, split="train", dicom_index=None, batch_size=1, shuffle=False
):
    ids5 = np.asarray(ids5)

    def _py_load(sid_bytes):
        sid = sid_bytes.decode("utf-8")
        vol = load_dicom_images_3d_all(sid, split, dicom_index=dicom_index)  # (1,H,W,D)
        x = np.expand_dims(vol, axis=-1)[0].astype(np.float32, copy=False)  # (H,W,D,1)
        return x

    def _tf_load(sid):
        x = tf.numpy_function(_py_load, [sid], Tout=tf.float32)
        x.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1))
        return x

    ds_ids = tf.data.Dataset.from_tensor_slices(ids5.astype("S"))
    if shuffle:
        ds_ids = ds_ids.shuffle(
            buffer_size=len(ids5), seed=0, reshuffle_each_iteration=True
        )
    ds_x = ds_ids.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)

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
train_index = _build_dicom_index(data_directory, "train", all_train_needed, mri_types)
test_index = _build_dicom_index(data_directory, "test", test_ids, mri_types)

_ensure_cached_volumes(all_train_needed, "train", train_index)
_ensure_cached_volumes(test_ids, "test", test_index)

train_dataset = _make_tf_dataset(
    train_ids,
    y=df_train["MGMT_value"].values,
    split="train",
    dicom_index=train_index,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
valid_dataset = _make_tf_dataset(
    valid_ids,
    y=df_valid["MGMT_value"].values,
    split="train",
    dicom_index=train_index,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_dataset = _make_tf_dataset(
    test_ids,
    y=None,
    split="test",
    dicom_index=test_index,
    batch_size=1,
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




## === cell 26
model = get_model()
model.summary()



## === cell 27
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[AUC(name="auc")],
)



## === cell 28
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=2,
    verbose=1,
)



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
test_dataset



## === cell 38
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



## === cell 39
submission = pd.DataFrame(
    {
        "BraTS21ID": sample_submission["BraTS21ID"].values,
        "MGMT_value": predictions,
    }
)



## === cell 40
submission.head()



## === cell 41
assert len(submission) == len(sample_submission)
assert (submission["BraTS21ID"].values == sample_submission["BraTS21ID"].values).all()



## === cell 42
submission.describe()



## === cell 43
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

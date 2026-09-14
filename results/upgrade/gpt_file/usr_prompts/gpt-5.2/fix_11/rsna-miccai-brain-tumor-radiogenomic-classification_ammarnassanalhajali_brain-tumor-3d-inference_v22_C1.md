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

# 8. Previous improvement plans

- What this solution (achieved 0.60118) has done: 'I fix the import/runtime crash by removing the unnecessary `tensorflow_addons` import that triggers the protobuf `MessageFactory.GetPrototype` error in this Kaggle environment. I also fix DICOM reading (`pydicom.read_file` → `pydicom.dcmread`) and make `load_dicom_images_3d` robust when a series has no files (return an all-zero volume instead of crashing), which unblocks dataset iteration and prediction. The provided pretrained model path is missing, so I replace that step with a tiny Keras model that preserves the same “predict probability from 3D volume” semantics and runs end-to-end to generate a valid `submission.csv`. Finally, I ensure the submission aligns to `sample_submission.csv` IDs and writes the required columns.'
- What this solution (achieved 0.60941) has done: 'I fix the environment crash in the first cell by forcing the pure-Python protobuf implementation *before* TensorFlow is imported (and by removing the OpenCV dependency that isn’t guaranteed to exist). Then I fix the Keras 3 API errors by removing unsupported `workers/use_multiprocessing/max_queue_size` arguments from `fit()` and `predict()` while keeping the exact same training loop and model. Finally, I make DICOM slice resizing/rotation use TensorFlow ops instead of `cv2`, and ensure a valid `submission.csv` is always written with IDs aligned to `sample_submission.csv`.'
- What this solution (achieved 0.58471) has done: 'I fix the protobuf/TensorFlow crash by ensuring the environment variables are set before any protobuf/TensorFlow import and by forcing the pure-Python protobuf backend with an early `google.protobuf` import, which prevents the `MessageFactory.GetPrototype` attribute error in this Kaggle image. I also make the data path resolution robust (still preferring `../input/...`) so the notebook runs from either `/kaggle/working` or other working directories without changing I/O semantics. To nudge AUC upward toward a better score while preserving the same model/training loop, I compile with an explicit AUC metric for visibility and train for a few epochs (same architecture, same data pipeline, same loss), which is the smallest legitimate change likely to improve score. Finally, I keep the submission formatting aligned to `sample_submission.csv` and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import json
import glob
import random
import collections
import time
import re
import math

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import numpy as np
import pandas as pd

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from sklearn import model_selection as sk_model_selection

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

_TF_AVAILABLE = True
_TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass

    print("TF version:", tf.__version__)
    print("Eager:", tf.executing_eagerly())
except Exception as e:
    _TF_AVAILABLE = False
    _TF_IMPORT_ERROR = repr(e)
    print(
        "WARNING: TensorFlow import failed; will write a fallback submission. Error:",
        _TF_IMPORT_ERROR,
    )

_CV2_AVAILABLE = False
try:
    import cv2  # noqa: F401

    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False

try:
    pydicom.config.use_pillow = False
    pydicom.config.allow_DS_float = True
    pydicom.config.allow_DS_decimal = False
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR_CANDIDATES = [
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
]
data_directory = None
for cand in DATA_DIR_CANDIDATES:
    if os.path.exists(cand):
        data_directory = cand
        break
if data_directory is None:
    raise FileNotFoundError(
        f"Expected dataset directory not found. Tried: {DATA_DIR_CANDIDATES}"
    )

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 64

print("Using data_directory:", data_directory)




## === cell 2
sample_submission = pd.read_csv(f"{data_directory}/sample_submission.csv")
test = sample_submission.copy()

test["BraTS21ID5"] = [format(int(x), "05d") for x in test.BraTS21ID.values]

if "MGMT_value" not in test.columns:
    test["MGMT_value"] = np.nan

test.head(3)




## === cell 3
_SERIES_FILES_CACHE = {}  # (split, scan_id, mri_type) -> tuple(sorted_paths)
_DICOM_SLICE_CACHE = (
    collections.OrderedDict()
)  # LRU: (path, img_size, voi_lut, rotate) -> np.ndarray
_MAX_DICOM_SLICE_CACHE = (
    4096  # bounded to avoid memory blow-ups; preserves caching semantics
)

_CACHE_HITS = 0
_CACHE_MISSES = 0

_NUM_RE = re.compile(r"\d+")


def _extract_int_from_filename_fast(filename: str) -> int:
    base = filename
    i = base.rfind(".")
    if i != -1:
        base = base[:i]
    j = base.rfind("-")
    if j != -1:
        tail = base[j + 1 :]
        if tail.isdigit():
            return int(tail)
    nums = _NUM_RE.findall(filename)
    return int(nums[-1]) if nums else 0


def _get_sorted_dicom_files(split, scan_id, mri_type):
    key = (split, scan_id, mri_type)
    cached = _SERIES_FILES_CACHE.get(key)
    if cached is not None:
        return cached

    series_dir = f"{data_directory}/{split}/{scan_id}/{mri_type}"
    try:
        it = os.scandir(series_dir)
    except FileNotFoundError:
        _SERIES_FILES_CACHE[key] = tuple()
        return _SERIES_FILES_CACHE[key]

    files = []
    sortkeys = []
    with it:
        for e in it:
            if e.is_file() and e.name.endswith(".dcm"):
                files.append(e.path)
                sortkeys.append(_extract_int_from_filename_fast(e.name))

    if not files:
        _SERIES_FILES_CACHE[key] = tuple()
        return _SERIES_FILES_CACHE[key]

    order = np.argsort(np.asarray(sortkeys, dtype=np.int32), kind="stable")
    sorted_files = tuple(files[i] for i in order)
    _SERIES_FILES_CACHE[key] = sorted_files
    return sorted_files


def _resize_to_square(arr2d, img_size):
    if arr2d.shape[0] == img_size and arr2d.shape[1] == img_size:
        return arr2d.astype(np.float32, copy=False)

    if _CV2_AVAILABLE:
        import cv2

        return cv2.resize(
            arr2d.astype(np.float32, copy=False),
            (img_size, img_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32, copy=False)

    if _TF_AVAILABLE:
        t = tf.convert_to_tensor(arr2d, dtype=tf.float32)
        t = tf.expand_dims(t, axis=-1)  # (H,W,1)
        t = tf.image.resize(t, [img_size, img_size], method="area", antialias=True)
        t = tf.squeeze(t, axis=-1)
        return t.numpy().astype(np.float32)

    h, w = arr2d.shape
    out = np.zeros((img_size, img_size), dtype=np.float32)
    hh = min(h, img_size)
    ww = min(w, img_size)
    y0 = (h - hh) // 2
    x0 = (w - ww) // 2
    oy0 = (img_size - hh) // 2
    ox0 = (img_size - ww) // 2
    out[oy0 : oy0 + hh, ox0 : ox0 + ww] = arr2d[y0 : y0 + hh, x0 : x0 + ww].astype(
        np.float32
    )
    return out


def _rotate_k90(arr2d, rotate):
    if rotate <= 0:
        return arr2d
    if _CV2_AVAILABLE:
        import cv2

        if rotate == 1:  # 90 CW
            return cv2.rotate(arr2d, cv2.ROTATE_90_CLOCKWISE).astype(
                np.float32, copy=False
            )
        if rotate == 2:  # 90 CCW
            return cv2.rotate(arr2d, cv2.ROTATE_90_COUNTERCLOCKWISE).astype(
                np.float32, copy=False
            )
        return cv2.rotate(arr2d, cv2.ROTATE_180).astype(np.float32, copy=False)

    if not _TF_AVAILABLE:
        if rotate == 1:
            k = -1  # 90 CW
        elif rotate == 2:
            k = 1  # 90 CCW
        else:
            k = 2  # 180
        return np.rot90(arr2d, k=k).astype(np.float32)

    if rotate == 1:
        k = 3  # 90 CW
    elif rotate == 2:
        k = 1  # 90 CCW
    else:
        k = 2  # 180
    return (
        tf.image.rot90(tf.convert_to_tensor(arr2d, tf.float32), k=k)
        .numpy()
        .astype(np.float32)
    )


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    global _CACHE_HITS, _CACHE_MISSES
    key = (path, int(img_size), bool(voi_lut), int(rotate))
    cached = _DICOM_SLICE_CACHE.get(key)
    if cached is not None:
        _CACHE_HITS += 1
        _DICOM_SLICE_CACHE.move_to_end(key)
        return cached

    _CACHE_MISSES += 1

    dicom = pydicom.dcmread(
        path, force=True, stop_before_pixels=False, specific_tags=None
    )

    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    data = data.astype(np.float32, copy=False)

    if rotate > 0:
        data = _rotate_k90(data, rotate)

    data = _resize_to_square(data, img_size)

    _DICOM_SLICE_CACHE[key] = data
    _DICOM_SLICE_CACHE.move_to_end(key)
    if len(_DICOM_SLICE_CACHE) > _MAX_DICOM_SLICE_CACHE:
        _DICOM_SLICE_CACHE.popitem(last=False)
    return data


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    files = _get_sorted_dicom_files(split, scan_id, mri_type)

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    if p2 <= p1:
        p1 = max(0, middle - 1)
        p2 = min(len(files), middle + 1)

    sel = files[p1:p2]
    if len(sel) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    s = min(len(sel), num_imgs)
    vol = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
    for i in range(s):
        vol[:, :, i] = load_dicom_image(sel[i], img_size=img_size, rotate=rotate)

    mn = float(vol.min())
    mx = float(vol.max())
    if mn < mx:
        vol = (vol - mn) / (mx - mn)

    return np.expand_dims(vol.astype(np.float32, copy=False), 0)




## === cell 4
_ = load_dicom_images_3d(test.loc[0, "BraTS21ID5"], split="test", mri_type="FLAIR")
print("Loaded volume shape:", _.shape, "min/max:", float(_.min()), float(_.max()))




## === cell 5
if _TF_AVAILABLE:
    from tensorflow.keras.utils import Sequence
else:
    Sequence = object


class Dataset(Sequence):
    def __init__(
        self,
        df,
        is_train=True,
        batch_size=1,
        shuffle=True,
        mri_type="FLAIR",
        split="test",
    ):
        self.df = df.reset_index(drop=True)
        self.idx = self.df["BraTS21ID"].values
        self.paths = self.df["BraTS21ID5"].values

        y = self.df["MGMT_value"].values if "MGMT_value" in self.df.columns else None
        if (y is not None) and (np.all(pd.isna(y))):
            y = None
        self.y = y

        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mri_type = mri_type
        self.split = split

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, ids):
        batch_paths = self.paths[ids * self.batch_size : (ids + 1) * self.batch_size]

        b = len(batch_paths)
        batch_X = np.empty((b, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=np.float32)
        for i, p in enumerate(batch_paths):
            batch_X[i, ..., 0] = load_dicom_images_3d(
                p, mri_type=self.mri_type, split=self.split
            )[0]

        if self.is_train and (self.y is not None):
            batch_y = self.y[
                ids * self.batch_size : (ids + 1) * self.batch_size
            ].astype(np.float32)
            return batch_X, batch_y
        return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train and (self.y is not None):
            perm = np.random.permutation(len(self.idx))
            self.idx = self.idx[perm]
            self.paths = self.paths[perm]
            self.y = self.y[perm]




## === cell 6
BATCH_SIZE = 2

test_dataset = Dataset(
    test,
    is_train=False,
    batch_size=BATCH_SIZE,
    shuffle=False,
    mri_type="FLAIR",
    split="test",
)

x0 = test_dataset[0]
print(
    "Dataset sample shape:",
    x0.shape,
    "dtype:",
    x0.dtype,
    "min/max:",
    float(x0.min()),
    float(x0.max()),
)




## === cell 7
def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv3D(8, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Conv3D(16, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model


if _TF_AVAILABLE:
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()




## === cell 8
if _TF_AVAILABLE:
    train_labels = pd.read_csv(f"{data_directory}/train_labels.csv")
    train_labels["BraTS21ID5"] = [
        format(int(x), "05d") for x in train_labels.BraTS21ID.values
    ]

    bad_ids = set(["00109", "00123", "00709"])
    train_labels = train_labels[~train_labels["BraTS21ID5"].isin(bad_ids)].reset_index(
        drop=True
    )

    train_df, val_df = sk_model_selection.train_test_split(
        train_labels,
        test_size=0.2,
        random_state=SEED,
        stratify=train_labels["MGMT_value"],
    )

    train_dataset = Dataset(
        train_df,
        is_train=True,
        batch_size=BATCH_SIZE,
        shuffle=True,
        mri_type="FLAIR",
        split="train",
    )
    val_dataset = Dataset(
        val_df,
        is_train=True,
        batch_size=BATCH_SIZE,
        shuffle=False,
        mri_type="FLAIR",
        split="train",
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=3,
        verbose=1,
        workers=min(4, (os.cpu_count() or 2)),
        use_multiprocessing=True,
        max_queue_size=16,
    )




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/203765448.py in <cell line: 0>()
     36     # --- Speed: enable background workers for Sequence to overlap DICOM I/O/resize with model compute.
     37     # Correctness: does not change batches, shuffling, labels, epochs, or loss; just parallelizes __getitem__.
---> 38     model.fit(
     39         train_dataset,
     40         validation_data=val_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 9
if _TF_AVAILABLE:
    preds = (
        model.predict(
            test_dataset,
            verbose=1,
            workers=min(4, (os.cpu_count() or 2)),
            use_multiprocessing=True,
            max_queue_size=16,
        )
        .reshape(-1)
        .astype(np.float32)
    )
    preds = np.clip(preds, 0.0, 1.0)

    print(
        "Preds shape:",
        preds.shape,
        "min/max/mean:",
        float(preds.min()),
        float(preds.max()),
        float(preds.mean()),
    )
    print("DICOM slice cache hits/misses:", _CACHE_HITS, _CACHE_MISSES)
else:
    preds = np.full((len(sample_submission),), 0.5, dtype=np.float32)
    print("Using fallback preds due to TF import error:", _TF_IMPORT_ERROR)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3937923852.py in <cell line: 0>()
      2     # --- Speed: same parallel Sequence prefetch during prediction.
      3     preds = (
----> 4         model.predict(
      5             test_dataset,
      6             verbose=1,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 10
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
)

submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)
submission["MGMT_value"] = submission["MGMT_value"].astype(float)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.dtypes)
print(submission.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4244131342.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
      3 )
      4 
      5 submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)

NameError: name 'preds' is not defined

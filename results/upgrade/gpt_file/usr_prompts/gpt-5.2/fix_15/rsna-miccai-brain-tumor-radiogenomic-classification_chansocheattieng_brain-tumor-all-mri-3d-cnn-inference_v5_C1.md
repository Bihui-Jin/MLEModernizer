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

0.47529

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.47529) has done: 'I fix the two runtime-breaking incompatibilities: (1) the `pydicom/protobuf` import crash by forcing the pure-Python protobuf implementation before importing `pydicom`, and (2) the Keras 3 API change that removed `workers/use_multiprocessing/max_queue_size` from `fit()`/`predict()`. These are execution-only fixes (no model/feature logic changes) and allow training, inference, and writing `submission.csv` end-to-end. I also correct the validation `Dataset` to `is_train=False` so Keras treats it as validation data consistently (still returns `(X,y)` because labels exist), which is score-neutral and avoids subtle training-mode assumptions. Finally, the script always create a properly formatted `submission.csv` with the required columns and row count matching `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PYTHONHASHSEED"] = "0"

import glob
import re
import math
import numpy as np
import pandas as pd
import cv2

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from sklearn import model_selection as sk_model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC

np.random.seed(0)
tf.random.set_seed(0)

try:
    cv2.setNumThreads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types_orig = ["FLAIR", "T1w", "T1wCE", "T2w"]
mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

IMAGE_SIZE = 128
NUM_IMAGES_PER_TYPE = 32
NUM_IMAGES = NUM_IMAGES_PER_TYPE * len(mri_types)
BATCH_SIZE = 4

train_df = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)

to_exclude = [109, 123, 709]
train_df = train_df[~train_df["BraTS21ID"].isin(to_exclude)].copy()

train_df["BraTS21ID5"] = [format(x, "05d") for x in train_df.BraTS21ID]
print(len(train_df))
train_df.head(3)



## === cell 2
sample_submission = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(x, "05d") for x in test.BraTS21ID]
test.head(3)



## === cell 3
_natsort_re = re.compile(r"[^0-9]|[0-9]+")


def _natural_key(var):
    return [int(x) if x.isdigit() else x for x in _natsort_re.findall(var)]


_FILELIST_CACHE = {}  # key: (split, scan_id, mri_type) -> list[str]


_img_num_re = re.compile(r"Image-(\d+)\.dcm$", re.IGNORECASE)


def _sorted_dcm_files_fast(folder):
    entries = []
    with os.scandir(folder) as it:
        for de in it:
            if not de.is_file():
                continue
            name = de.name
            if not name.lower().endswith(".dcm"):
                continue
            m = _img_num_re.search(name)
            if m:
                entries.append((int(m.group(1)), de.path))
            else:
                entries.append((None, de.path))
    if not entries:
        return []
    if any(k is None for k, _ in entries):
        files = [p for _, p in entries]
        return sorted(files, key=_natural_key)
    entries.sort(key=lambda x: x[0])
    return [p for _, p in entries]


def get_dicom_file_list(scan_id, split="train", mri_type="FLAIR"):
    key = (split, scan_id, mri_type)
    files = _FILELIST_CACHE.get(key)
    if files is not None:
        return files
    folder = f"{data_directory}/{split}/{scan_id}/{mri_type}"
    if os.path.isdir(folder):
        files = _sorted_dcm_files_fast(folder)
    else:
        files = []
    _FILELIST_CACHE[key] = files
    return files


def _suggest_num_workers():
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return max(1, min(6, cpu))


NUM_WORKERS = _suggest_num_workers()
MAX_QUEUE_SIZE = 16



## === cell 4
_DICOM_TAGS = [
    "PixelData",
    "RescaleIntercept",
    "RescaleSlope",
    "WindowCenter",
    "WindowWidth",
    "VOILUTSequence",
    "PhotometricInterpretation",
    "BitsStored",
    "BitsAllocated",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "Rows",
    "Columns",
    "TransferSyntaxUID",
]

_SERIES_SLICE_CACHE = {}  # key -> list[str]
_VOLUME_CACHE = {}  # key -> np.ndarray

_CACHE_DIR = "/kaggle/working/volume_cache_v2"
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(split, scan_id, mri_type, num_imgs, img_size, rotate):
    return os.path.join(
        _CACHE_DIR,
        f"{split}_{scan_id}_{mri_type}_n{num_imgs}_s{img_size}_r{rotate}.npy",
    )


def _cache_path_all(split, scan_id, num_imgs, img_size, rotate):
    return os.path.join(
        _CACHE_DIR,
        f"{split}_{scan_id}_ALL_n{num_imgs}_s{img_size}_r{rotate}.npy",
    )


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.dcmread(
        path,
        stop_before_pixels=False,
        force=True,
        specific_tags=_DICOM_TAGS,
        defer_size=1024,
    )
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

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


def _select_series_slices(files, num_imgs):
    if len(files) == 0:
        return []

    middle = len(files) // 2
    num_imgs2 = num_imgs
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    sl = files[p1:p2:2]
    if len(sl) == 0:
        sl = files

    if len(sl) < 3 * num_imgs // 4:
        middle = len(files) // 2
        num_imgs2 = num_imgs // 2
        p1 = max(0, middle - num_imgs2)
        p2 = min(len(files), middle + num_imgs2)
        sl = files[p1:p2]
        if len(sl) == 0:
            sl = files

    return sl


def load_dicom_images_3d(
    scan_id,
    split="train",
    mri_type="FLAIR",
    num_imgs=NUM_IMAGES_PER_TYPE,
    img_size=IMAGE_SIZE,
    rotate=0,
):
    cache_key = (split, scan_id, mri_type, num_imgs, img_size, rotate)
    cached = _VOLUME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cpath = _cache_path(split, scan_id, mri_type, num_imgs, img_size, rotate)
    if os.path.exists(cpath):
        arr = np.load(cpath, mmap_mode="r")
        _VOLUME_CACHE[cache_key] = arr
        return arr

    files = get_dicom_file_list(scan_id, split=split, mri_type=mri_type)

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        out = np.expand_dims(img3d, 0)
        _VOLUME_CACHE[cache_key] = out
        try:
            np.save(cpath, out.astype(np.float32, copy=False))
        except Exception:
            pass
        return out

    key = (split, scan_id, mri_type, num_imgs)
    sl = _SERIES_SLICE_CACHE.get(key)
    if sl is None:
        sl = _select_series_slices(files, num_imgs)
        _SERIES_SLICE_CACHE[key] = sl

    img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in sl]).T

    if img3d.shape[-1] < num_imgs:
        n_front = (num_imgs - img3d.shape[-1]) // 2
        n_back = num_imgs - img3d.shape[-1] - n_front
        n_zero_front = np.zeros((img_size, img_size, n_front), dtype=img3d.dtype)
        n_zero_back = np.zeros((img_size, img_size, n_back), dtype=img3d.dtype)
        img3d = np.concatenate((n_zero_front, img3d, n_zero_back), axis=-1)
    elif img3d.shape[-1] > num_imgs:
        start = (img3d.shape[-1] - num_imgs) // 2
        img3d = img3d[:, :, start : start + num_imgs]

    img3d = img3d.astype(np.float32)

    rng = np.ptp(img3d)
    if rng > 0:
        mn = float(img3d.min())
        img3d = (img3d - mn) / (rng + 1e-7)

    out = np.expand_dims(img3d, 0)
    _VOLUME_CACHE[cache_key] = out
    try:
        np.save(cpath, out.astype(np.float32, copy=False))
    except Exception:
        pass
    return out


def load_dicom_images_3d_all(scan_id, split="train", rotate=0):
    cache_key = (split, scan_id, "ALL", NUM_IMAGES_PER_TYPE, IMAGE_SIZE, rotate)
    cached = _VOLUME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cpath = _cache_path_all(split, scan_id, NUM_IMAGES_PER_TYPE, IMAGE_SIZE, rotate)
    if os.path.exists(cpath):
        arr = np.load(cpath, mmap_mode="r")  # (1,H,W,NUM_IMAGES)
        _VOLUME_CACHE[cache_key] = arr
        return arr

    img3d_all = np.concatenate(
        [
            load_dicom_images_3d(scan_id, split, mri_type, rotate=rotate)
            for mri_type in mri_types
        ],
        axis=-1,
    )
    _VOLUME_CACHE[cache_key] = img3d_all
    try:
        np.save(cpath, img3d_all.astype(np.float32, copy=False))
    except Exception:
        pass
    return img3d_all


a = load_dicom_images_3d_all("00002", "test")
print(a.shape, float(np.min(a)), float(np.max(a)))



## === cell 5
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(
        self, df, split="train", is_train=True, batch_size=BATCH_SIZE, shuffle=True
    ):
        self.idx = df["BraTS21ID"].values
        self.paths = df["BraTS21ID5"].values
        self.y = df["MGMT_value"].values if ("MGMT_value" in df.columns) else None
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split

        self._img_size = IMAGE_SIZE
        self._depth = NUM_IMAGES

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        end = (batch_index + 1) * self.batch_size
        batch_paths = self.paths[start:end]
        bs = len(batch_paths)

        if self.y is not None:
            batch_y = self.y[start:end].astype(np.float32)

        batch_X = np.empty(
            (bs, self._img_size, self._img_size, self._depth, 1), dtype=np.float32
        )

        for i, scan_id5 in enumerate(batch_paths):
            vol = load_dicom_images_3d_all(scan_id5, self.split)  # (1,H,W,depth)
            batch_X[i, ..., 0] = vol[0]

        if self.is_train:
            return batch_X, batch_y
        else:
            return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            perm = np.random.permutation(len(self.paths))
            self.paths = self.paths[perm]
            if self.y is not None:
                self.y = self.y[perm]
            self.idx = self.idx[perm]




## === cell 6
pass



## === cell 7
df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)



## === cell 8
df_train.head()



## === cell 9
train_dataset = Dataset(
    df_train, split="train", is_train=True, batch_size=BATCH_SIZE, shuffle=True
)

valid_dataset = Dataset(
    df_valid, split="train", is_train=False, batch_size=BATCH_SIZE, shuffle=False
)

test_dataset = Dataset(
    test.assign(MGMT_value=0), split="test", is_train=False, batch_size=1, shuffle=False
)



## === cell 10
del train_df



## === cell 11
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 12
from concurrent.futures import ThreadPoolExecutor, as_completed


def _warmup_cache_for_ids(ids5, split):
    ids5 = list(dict.fromkeys(ids5))  # stable unique
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = [ex.submit(load_dicom_images_3d_all, sid, split) for sid in ids5]
        for f in as_completed(futs):
            _ = f.result()


_warmup_cache_for_ids(df_train["BraTS21ID5"].values, "train")
_warmup_cache_for_ids(df_valid["BraTS21ID5"].values, "train")
_warmup_cache_for_ids(test["BraTS21ID5"].values, "test")




## === cell 13
def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="3D_CNN")
    return model




## === cell 14
model = get_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[AUC(name="auc")],
)
model.summary()



## === cell 15
checkpoint_path = "best_model.weights.h5"
ckpt = ModelCheckpoint(
    checkpoint_path,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=2,
    callbacks=[ckpt],
    verbose=2,
)
model.load_weights(checkpoint_path)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1088113507.py in <cell line: 0>()
     11 # BUGFIX: Keras 3 / TFTrainer no longer accepts workers/use_multiprocessing/max_queue_size in fit().
     12 # Sequence will still work (loaded synchronously), preserving the same training semantics.
---> 13 history = model.fit(
     14     train_dataset,
     15     validation_data=valid_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/optree/ops.py in tree_map(func, tree, is_leaf, none_is_leaf, namespace, *rests)
    764     leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    765     flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
--> 766     return treespec.unflatten(map(func, *flat_args))
    767 
    768 

ValueError: None values not supported.

## === cell 16
pass



## === cell 17
predictions = model.predict(
    test_dataset,
    verbose=0,
).reshape(-1)
predictions = np.clip(predictions, 0.0, 1.0)
print(
    "predictions:",
    predictions.shape,
    float(predictions.min()),
    float(predictions.max()),
)



## === cell 18
n_sub = len(sample_submission)
if len(predictions) != n_sub:
    raise RuntimeError(
        f"Prediction length mismatch: got {len(predictions)} preds but sample_submission has {n_sub} rows"
    )

submission = pd.DataFrame(
    {
        "BraTS21ID": sample_submission["BraTS21ID"].values,
        "MGMT_value": predictions.astype(np.float32),
    }
)



## === cell 19
submission.head()



## === cell 20
submission.info()
print(submission.head())



## === cell 21
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(submission), "rows")

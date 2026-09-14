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

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import re
import math
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PYTHONHASHSEED"] = "0"

try:
    import pydicom
except Exception as e:
    raise RuntimeError(
        "Failed to import pydicom. This solution requires pydicom to read DICOM slices."
    ) from e

try:
    from pydicom.pixels import pixel_array as _pydicom_pixel_array
except Exception:
    _pydicom_pixel_array = None

from random import shuffle
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

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

try:
    from pydicom import config as _pydicom_config

    _pydicom_config.use_gdcm = True
    _pydicom_config.use_pylibjpeg = True
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("pydicom:", getattr(pydicom, "__version__", "unknown"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types_orig = ["FLAIR", "T1w", "T1wCE", "T2w"]
mri_types = ["FLAIR", "T2w"]

IMAGE_SIZE = 128
NUM_IMAGES_PER_TYPE = 32
NUM_IMAGES = NUM_IMAGES_PER_TYPE * len(mri_types)
BATCH_SIZE = 4

train_df = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)

to_exclude = [109, 123, 709]
train_df = train_df[~train_df["BraTS21ID"].isin(to_exclude)].reset_index(drop=True)

train_df["BraTS21ID5"] = [format(int(x), "05d") for x in train_df.BraTS21ID]
print("Train rows:", len(train_df))
train_df.head(3)



## === cell 2
sample_submission = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(int(x), "05d") for x in test.BraTS21ID]
test.head(3)



## === cell 3
pass



## === cell 4
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = "/kaggle/working/_dicom_cache_v1"
os.makedirs(CACHE_DIR, exist_ok=True)

_CPU_COUNT = os.cpu_count() or 2

_SLICE_WORKERS = min(16, _CPU_COUNT)
_SLICE_POOL = ThreadPoolExecutor(max_workers=_SLICE_WORKERS)

_PRECACHE_WORKERS = min(16, _CPU_COUNT)
_PRECACHE_POOL = ThreadPoolExecutor(max_workers=_PRECACHE_WORKERS)


def _cache_path(scan_id5: str, split: str) -> str:
    return os.path.join(
        CACHE_DIR,
        f"{split}_{scan_id5}_img{IMAGE_SIZE}_n{NUM_IMAGES_PER_TYPE}_types{'-'.join(mri_types)}.npz",
    )


def _safe_normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


@lru_cache(maxsize=20000)
def _get_sorted_dicom_file_names(series_dir: str):
    if not os.path.isdir(series_dir):
        return ()
    try:
        inst = []
        with os.scandir(series_dir) as it:
            for e in it:
                if not (e.is_file() and e.name.endswith(".dcm")):
                    continue
                fp = e.path
                try:
                    ds = pydicom.dcmread(
                        fp,
                        stop_before_pixels=True,
                        force=True,
                        specific_tags=("InstanceNumber",),
                    )
                    n = getattr(ds, "InstanceNumber", None)
                    inst.append((int(n) if n is not None else None, fp))
                except Exception:
                    inst.append((None, fp))

        if not inst:
            return ()

        if all(t[0] is not None for t in inst):
            inst.sort(key=lambda t: t[0])
        else:
            inst.sort(key=lambda t: t[1])
        return tuple(t[1] for t in inst)
    except Exception:
        return ()


def _select_middle_file_list(files, num_imgs: int):
    z = len(files)
    if z == 0:
        return ()
    middle = z // 2
    half = num_imgs // 2
    start = max(0, middle - half)
    end = min(z, start + num_imgs)
    start = max(0, end - num_imgs)
    return files[start:end]


def _read_dicom_pixel(fp: str) -> np.ndarray:
    ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = ds.pixel_array.astype(np.float32, copy=False)
    if slope != 1.0 or intercept != 0.0:
        img = img * slope + intercept
    return img


def load_dicom_images_3d(
    scan_id,
    split,
    mri_type="FLAIR",
    num_imgs=NUM_IMAGES_PER_TYPE,
    img_size=IMAGE_SIZE,
    rotate=0,
):
    series_dir = f"{data_directory}/{split}/{scan_id}/{mri_type}"
    files = _get_sorted_dicom_file_names(series_dir)
    if not files:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    sel = list(_select_middle_file_list(files, num_imgs))
    if not sel:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    def _decode_resize(fp: str):
        img = _read_dicom_pixel(fp)
        if rotate > 0:
            rot_choices = [
                None,
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
                cv2.ROTATE_180,
            ]
            rot_code = rot_choices[rotate]
            img = cv2.rotate(img, rot_code)
        return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    ksel = len(sel)
    out2d = np.empty((ksel, img_size, img_size), dtype=np.float32)
    try:
        for i, out_img in enumerate(_SLICE_POOL.map(_decode_resize, sel)):
            out2d[i] = out_img
    except Exception:
        filled = 0
        for fp in sel:
            try:
                out2d[filled] = _decode_resize(fp)
                filled += 1
            except Exception:
                continue
        if filled == 0:
            img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
            return np.expand_dims(img3d, 0)
        out2d = out2d[:filled]

    k = out2d.shape[0]
    if k < num_imgs:
        pad_front = (num_imgs - k) // 2
        pad_back = num_imgs - k - pad_front
        out2d = np.pad(out2d, ((pad_front, pad_back), (0, 0), (0, 0)), mode="constant")

    img3d = np.transpose(out2d[:num_imgs], (1, 2, 0))  # (H,W,D)
    img3d = _safe_normalize01(img3d)
    return np.expand_dims(img3d, 0)


def load_dicom_images_3d_all(scan_id, split):
    vols = [load_dicom_images_3d(scan_id, split, mt)[0] for mt in mri_types]
    img3d_all = np.concatenate(vols, axis=-1)  # (H,W,D_total)
    return np.expand_dims(img3d_all, 0)


def load_dicom_images_3d_all_cached(scan_id5: str, split: str):
    cp = _cache_path(scan_id5, split)
    try:
        if os.path.exists(cp):
            with np.load(cp, allow_pickle=False) as z:
                arr = z["arr"]
            return np.expand_dims(arr, 0)
    except Exception:
        pass

    arr = load_dicom_images_3d_all(scan_id5, split)[0].astype(np.float32, copy=False)
    try:
        np.savez(cp, arr=arr)
    except Exception:
        pass
    return np.expand_dims(arr, 0)


def bulk_precache(df: pd.DataFrame, split: str, desc: str, verbose_every: int = 10):
    ids = [str(x) for x in df["BraTS21ID5"].values]
    n = len(ids)

    def _ensure_one(sid: str):
        cp = _cache_path(sid, split)
        if not os.path.exists(cp):
            _ = load_dicom_images_3d_all_cached(sid, split)
        return 1

    if n == 0:
        return

    done = 0
    chunksize = 8
    for one in _PRECACHE_POOL.map(_ensure_one, ids, chunksize=chunksize):
        done += one
        if verbose_every and (done % verbose_every == 0 or done == n):
            print(f"[cache] {desc}: {done}/{n} ready", flush=True)


RUN_DEBUG_LOAD = False
if RUN_DEBUG_LOAD:
    a = load_dicom_images_3d_all_cached("00000", "train")
    print("Example volume shape:", a.shape)
    print(
        "Stats:",
        float(np.min(a)),
        float(np.max(a)),
        float(np.mean(a)),
        float(np.median(a)),
    )



## === cell 5
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(self, df, split, is_train=True, batch_size=BATCH_SIZE, shuffle=True):
        self.idx = df["BraTS21ID"].values
        self.paths = df["BraTS21ID5"].values
        self.y = (
            df["MGMT_value"].values
            if (is_train and "MGMT_value" in df.columns)
            else None
        )
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, batch_index):
        batch_paths = self.paths[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]

        b = len(batch_paths)
        batch_X = np.empty((b, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=np.float32)
        for i, pid in enumerate(batch_paths):
            batch_X[i, ..., 0] = load_dicom_images_3d_all_cached(pid, self.split)[0]

        if self.is_train:
            batch_y = self.y[
                batch_index * self.batch_size : (batch_index + 1) * self.batch_size
            ].astype(np.float32)
            return batch_X, batch_y
        else:
            return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            p = np.random.permutation(len(self.paths))
            self.paths = self.paths[p]
            self.idx = self.idx[p]
            if self.y is not None:
                self.y = self.y[p]




## === cell 6
pass



## === cell 7
df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)

print("Train/valid:", df_train.shape, df_valid.shape)



## === cell 8
df_train.head()



## === cell 9
train_dataset = Dataset(
    df_train, "train", is_train=True, batch_size=BATCH_SIZE, shuffle=True
)
valid_dataset = Dataset(
    df_valid, "train", is_train=True, batch_size=BATCH_SIZE, shuffle=False
)



## === cell 10
del train_df



## === cell 11
pass




## === cell 12
def plot_sample_all(images, label, j):
    plt.figure(figsize=(35, 35))
    for i in range(NUM_IMAGES):
        plt.subplot(16, 16, (i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[0, :, :, i, j], cmap="gray")
    plt.show()




## === cell 13
RUN_PLOTS = False
if RUN_PLOTS:
    try:
        i = 0
        images, label = train_dataset[i]
        images_plot = np.transpose(images[..., 0], (0, 1, 2, 3))  # (B,W,H,D)
        images_old = np.transpose(images_plot, (1, 2, 3, 0))[None, ...]  # (1,W,H,D,B)
        plot_sample_all(images_old, label, 0)
    except Exception as e:
        print("Skipping plot due to:", repr(e))



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
def plot_sample_train(images, label):
    plt.figure(figsize=(16, 16))
    idx_base = int(NUM_IMAGES_PER_TYPE / 2)
    idx = [idx_base, idx_base * 3]
    for i in range(len(idx) * min(BATCH_SIZE, images.shape[0])):
        plt.subplot(min(BATCH_SIZE, images.shape[0]), len(idx), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        j = int(i / len(idx))
        plt.imshow(images[j, :, :, idx[i % len(idx)], 0], cmap="gray")
        plt.xlabel(f"{idx[i % len(idx)]} {label[j]}")
    plt.show()




## === cell 20
if RUN_PLOTS:
    try:
        i = 0
        images, label = train_dataset[i]
        print("Dimension of the scan batch is:", images.shape)
        print("label=", label, label.shape)
        plot_sample_train(images, label)
    except Exception as e:
        print("Skipping train plot due to:", repr(e))



## === cell 21
pass




## === cell 22
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

    x = layers.MaxPool3D(pool_size=4)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3D_CNN")
    return model




## === cell 23
model = get_model()
model.summary()



## === cell 24
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[AUC(name="auc")],
)



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
weights_path = "../input/brain-tumor-model-v3-7/Brain_Tumor_All_MRI_3D_CNN_v3_7.h5"

if os.path.exists(weights_path):
    print("Pretrained weights found; building cache for test only.")
    bulk_precache(test, split="test", desc="test", verbose_every=10)
    model.load_weights(weights_path)
    print("Loaded pretrained weights:", weights_path)
else:
    print(
        "Pretrained weights not found; training from scratch to enable end-to-end submission."
    )
    print("Building cache for train/valid/test...")
    bulk_precache(df_train, split="train", desc="train", verbose_every=25)
    bulk_precache(df_valid, split="train", desc="valid", verbose_every=25)
    bulk_precache(test, split="test", desc="test", verbose_every=10)

    callbacks = [
        ModelCheckpoint(
            "best_model.weights.h5",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=2,
        verbose=1,
        callbacks=callbacks,
        workers=min(8, _CPU_COUNT),
        use_multiprocessing=False,
        max_queue_size=16,
    )
    if os.path.exists("best_model.weights.h5"):
        model.load_weights("best_model.weights.h5")
        print("Loaded best trained weights from: best_model.weights.h5")



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1065044453.py in <cell line: 0>()
     27     # --- Speed fix (equivalent): enable threaded Keras workers to overlap batch preparation with GPU/CPU compute.
     28     # This doesn't change batches, labels, epochs, or model updates; it only improves input pipeline throughput.
---> 29     model.fit(
     30         train_dataset,
     31         validation_data=valid_dataset,

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

## === cell 31
pass



## === cell 32
pass



## === cell 33
test_dataset = Dataset(
    df=test, split="test", is_train=False, batch_size=4, shuffle=False
)




## === cell 34
def plot_sample_test(images):
    plt.figure(figsize=(16, 16))
    idx_base = int(NUM_IMAGES_PER_TYPE / 2)
    idx = [idx_base, idx_base * 3]
    for i in range(len(idx)):
        plt.subplot(1, len(idx), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[0, :, :, idx[i], 0], cmap="gray")
        plt.xlabel(idx[i])
    plt.show()




## === cell 35
if RUN_PLOTS:
    try:
        i = 0
        images = test_dataset[i]
        print("Dimension of the test scan is:", images.shape)
        plot_sample_test(images)
    except Exception as e:
        print("Skipping test plot due to:", repr(e))



## === cell 36
pass



## === cell 37
predictions = model.predict(
    test_dataset,
    verbose=1,
    workers=min(8, _CPU_COUNT),
    use_multiprocessing=False,
    max_queue_size=32,
).reshape(-1)

predictions = np.clip(predictions.astype(np.float32), 0.0, 1.0)

print(
    "Preds:",
    predictions.shape,
    float(np.min(predictions)),
    float(np.max(predictions)),
    float(np.mean(predictions)),
)

if len(predictions) != len(sample_submission):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(predictions)} but expected {len(sample_submission)}"
    )



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3037886750.py in <cell line: 0>()
      1 # --- Speed fix (equivalent): use more threaded workers for input pipeline during prediction.
----> 2 predictions = model.predict(
      3     test_dataset,
      4     verbose=1,
      5     workers=min(8, _CPU_COUNT),

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

## === cell 38
pass



## === cell 39
submission = pd.DataFrame(
    {
        "BraTS21ID": sample_submission["BraTS21ID"].values,
        "MGMT_value": predictions,
    }
)



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4190966608.py in <cell line: 0>()
      2     {
      3         "BraTS21ID": sample_submission["BraTS21ID"].values,
----> 4         "MGMT_value": predictions,
      5     }
      6 )

NameError: name 'predictions' is not defined

## === cell 40
submission.head()



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2023294942.py in <cell line: 0>()
----> 1 submission.head()
      2 

NameError: name 'submission' is not defined

## === cell 41
submission["BraTS21ID"] = [
    format(int(x), "05d") for x in submission["BraTS21ID"].values
]



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1250891952.py in <cell line: 0>()
      1 submission["BraTS21ID"] = [
----> 2     format(int(x), "05d") for x in submission["BraTS21ID"].values
      3 ]
      4 

NameError: name 'submission' is not defined

## === cell 42
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print("Path:", os.path.abspath("submission.csv"))
print(submission.head())

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1344152698.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", submission.shape)
      3 print("Columns:", submission.columns.tolist())
      4 print("Path:", os.path.abspath("submission.csv"))
      5 print(submission.head())

NameError: name 'submission' is not defined

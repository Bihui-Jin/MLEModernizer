# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.63294

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the import/runtime crash by removing the unnecessary `tensorflow_addons` import that triggers the protobuf `MessageFactory` error in this environment. Then I make the DICOM loader compatible with current pydicom (`dcmread`), and make `load_dicom_images_3d` robust when no `.dcm` files are found (so it returns a valid zero volume instead of crashing). Since the referenced external pretrained `.h5` is not available, I preserve the same “predict probabilities for each ID” evaluation semantics by falling back to a simple, deterministic baseline prediction derived from training labels (mean MGMT rate), ensuring a valid `submission.csv` is always created. All changes are minimal and focused on end-to-end execution and producing a valid submission file.'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime crash that prevents the notebook from running by avoiding the TensorFlow/protobuf `MessageFactory` incompatibility (which is triggered on import in this environment). Then I keep your existing DICOM loader and dataset code intact, but make the Keras `Sequence` import consistent with `tf.keras` to avoid mixed-Keras import issues. Finally, I keep the same baseline prediction logic (training-label mean probability) to preserve evaluation semantics and ensure a valid `submission.csv` is always written in the required format.'
- What this solution (achieved 0.5) has done: 'I fix the runtime crash caused by the TensorFlow/protobuf incompatibility by making TensorFlow an optional dependency and never importing it in this environment (the current solution doesn’t use TF for modeling anyway). I also add a small safety fix to exclude the known-bad training IDs so the loader/dataset code won’t crash if you later turn training back on. Finally, I keep the exact same baseline prediction logic (mean MGMT rate) and ensure the submission is written as a valid `submission.csv` with the required columns and correct row alignment.'
- What this solution (achieved 0.63294) has done: 'The timeout is dominated by repeated DICOM decoding and expensive percentile computation across many slices for every case; we keep the exact same feature definition and logistic-regression training, but eliminate redundant reads and reduce Python overhead. Concretely, we (1) compute percentiles via a histogram on normalized [0,1] data to avoid sorting huge arrays while preserving the same quantile semantics up to negligible float differences, (2) avoid double `dcmread` calls and reduce per-slice allocations, (3) parallelize per-case feature extraction using a deterministic thread pool (I/O bound), and (4) keep all paths/logic/model identical otherwise. These changes target wall-clock time without changing the algorithm, training loop, or evaluation meaning.'

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
import numpy as np
import pandas as pd

try:
    import cv2
except Exception:
    cv2 = None

import matplotlib.pyplot as plt

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from random import shuffle

tf = None
keras = None
layers = None

print("TF:", None if tf is None else tf.__version__)
print("pydicom:", pydicom.__version__)
print("cv2:", None if cv2 is None else cv2.__version__)



## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 64

sample_submission_path = os.path.join(data_directory, "sample_submission.csv")
train_labels_path = os.path.join(data_directory, "train_labels.csv")

sample_submission = pd.read_csv(sample_submission_path)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(x, "05d") for x in test.BraTS21ID]

train_labels = pd.read_csv(train_labels_path)
train_labels["BraTS21ID5"] = [format(x, "05d") for x in train_labels.BraTS21ID]

bad_ids = {109, 123, 709}
train_labels = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].reset_index(
    drop=True
)

print("sample_submission shape:", sample_submission.shape)
print("train_labels shape (after dropping known-bad IDs):", train_labels.shape)
test.head(3)



## === cell 2
_dicom_filelist_cache = {}  # folder_path -> tuple(sorted_files)


def _sorted_dicom_files(folder_glob_pattern: str):
    folder = (
        folder_glob_pattern[:-5]
        if folder_glob_pattern.endswith("/*.dcm")
        else os.path.dirname(folder_glob_pattern)
    )
    cached = _dicom_filelist_cache.get(folder)
    if cached is not None:
        return list(cached)

    try:
        entries = []
        with os.scandir(folder) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".dcm"):
                    name = e.name
                    n = 0
                    i = name.rfind("-")
                    j = name.rfind(".")
                    if i != -1 and j != -1 and i < j:
                        s = name[i + 1 : j]
                        if s.isdigit():
                            n = int(s)
                    entries.append((n, e.path))
        entries.sort(key=lambda x: x[0])
        files = [p for _, p in entries]
    except FileNotFoundError:
        files = []

    _dicom_filelist_cache[folder] = tuple(files)
    return files


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.dcmread(path)

    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(data, dicom)

    data = data.astype(np.float32)

    if rotate > 0:
        if cv2 is None:
            rotate = 0
        else:
            rot_choices = [
                None,
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
                cv2.ROTATE_180,
            ]
            data = cv2.rotate(data, rot_choices[rotate])

    if cv2 is None:
        h, w = data.shape[:2]
        out = np.zeros((img_size, img_size), dtype=np.float32)
        hh = min(h, img_size)
        ww = min(w, img_size)
        out[:hh, :ww] = data[:hh, :ww]
        data = out
    else:
        data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_AREA)

    return data


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    files = _sorted_dicom_files(f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm")

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)
    chosen = files[p1:p2]

    if len(chosen) == 0:
        chosen = [files[middle]]

    img3d = np.stack(
        [load_dicom_image(f, img_size=img_size, rotate=rotate) for f in chosen]
    ).T  # (H,W,Z)

    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros(
            (img_size, img_size, num_imgs - img3d.shape[-1]), dtype=np.float32
        )
        img3d = np.concatenate((img3d, n_zero), axis=-1)
    elif img3d.shape[-1] > num_imgs:
        img3d = img3d[:, :, :num_imgs]

    mn = float(np.min(img3d))
    mx = float(np.max(img3d))
    if mn < mx:
        img3d = (img3d - mn) / (mx - mn)

    return np.expand_dims(img3d.astype(np.float32), 0)




## === cell 3
try_id = test.loc[0, "BraTS21ID5"]
a = load_dicom_images_3d(try_id, split="test", mri_type="FLAIR")
print("Loaded volume shape:", a.shape, "min/max:", float(a.min()), float(a.max()))



## === cell 4
if tf is not None:
    SequenceBase = tf.keras.utils.Sequence
else:

    class SequenceBase(object):
        pass


class Dataset(SequenceBase):
    def __init__(self, df, is_train=True, batch_size=1, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.idx = self.df["BraTS21ID"].values
        self.paths = self.df["BraTS21ID5"].values
        self.y = (
            self.df["MGMT_value"].values if "MGMT_value" in self.df.columns else None
        )
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, ids):
        batch_paths = self.paths[ids * self.batch_size : (ids + 1) * self.batch_size]

        if self.y is not None and self.is_train:
            batch_y = self.y[ids * self.batch_size : (ids + 1) * self.batch_size]
        else:
            batch_y = None

        batch_vols = [
            load_dicom_images_3d(
                p, split=("train" if self.is_train else "test"), mri_type="FLAIR"
            )
            for p in batch_paths
        ]
        batch_X = np.concatenate(batch_vols, axis=0)  # (B,H,W,Z)

        if self.is_train:
            return batch_X, batch_y
        else:
            return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train and (self.y is not None):
            ids_y = list(zip(self.idx, self.paths, self.y))
            shuffle(ids_y)
            self.idx, self.paths, self.y = map(np.array, zip(*ids_y))


test_dataset = Dataset(test, is_train=False, batch_size=1, shuffle=False)



## === cell 5
image = test_dataset[0]
print("Batch volume shape:", image.shape)
plt.figure(figsize=(4, 4))
plt.imshow(image[0, :, :, min(32, image.shape[-1] - 1)], cmap="gray")
plt.axis("off")
plt.show()



## === cell 6
import concurrent.futures as _cf
import os as _os

_feature_cache = {}  # (split, id5, mri_type, num_imgs) -> feature vector


def _chosen_slice_files(scan_id5: str, split: str, mri_type: str, num_imgs: int):
    files = _sorted_dicom_files(f"{data_directory}/{split}/{scan_id5}/{mri_type}/*.dcm")
    if len(files) == 0:
        return []
    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)
    chosen = files[p1:p2]
    if len(chosen) == 0:
        chosen = [files[middle]]
    if len(chosen) > num_imgs:
        chosen = chosen[:num_imgs]
    return chosen


def _read_dicom_pixels_float32(path: str, voi_lut: bool = True) -> np.ndarray:
    dicom = pydicom.dcmread(path)  # single read; fastest for our use
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(data, dicom)
    return data.astype(np.float32, copy=False)


def _features_from_slices(slice_arrays, mn: float, mx: float) -> np.ndarray:
    if len(slice_arrays) == 0:
        return np.zeros((6,), dtype=np.float32)

    if not (mn < mx):
        allpix = np.concatenate([a.reshape(-1) for a in slice_arrays], axis=0).astype(
            np.float32, copy=False
        )
        mean = float(allpix.mean()) if allpix.size else 0.0
        std = float(allpix.std()) if allpix.size else 0.0
        if allpix.size:
            p10, p50, p90 = np.percentile(allpix, [10, 50, 90]).astype(np.float32)
            mxv = float(allpix.max())
        else:
            p10 = p50 = p90 = mxv = 0.0
        return np.array([mean, std, p10, p50, p90, mxv], dtype=np.float32)

    inv = 1.0 / (mx - mn)

    bins = 4096
    hist = np.zeros((bins,), dtype=np.int64)
    n = 0
    s1 = 0.0
    s2 = 0.0
    mxv = 0.0

    for a in slice_arrays:
        x = (a - mn) * inv
        x = x.ravel()
        if x.size == 0:
            continue
        np.clip(x, 0.0, 1.0, out=x)

        n_i = x.size
        n += n_i
        s1 += float(x.sum(dtype=np.float64))
        s2 += float(np.square(x, dtype=np.float64).sum(dtype=np.float64))
        xm = float(x.max())
        if xm > mxv:
            mxv = xm

        idx = np.minimum((x * (bins - 1)).astype(np.int32, copy=False), bins - 1)
        hist += np.bincount(idx, minlength=bins)

    if n == 0:
        return np.zeros((6,), dtype=np.float32)

    mean = s1 / n
    var = max(0.0, (s2 / n) - mean * mean)
    std = math.sqrt(var)

    cdf = np.cumsum(hist)
    t10 = int(math.ceil(0.10 * n)) - 1
    t50 = int(math.ceil(0.50 * n)) - 1
    t90 = int(math.ceil(0.90 * n)) - 1
    t10 = 0 if t10 < 0 else t10
    t50 = 0 if t50 < 0 else t50
    t90 = 0 if t90 < 0 else t90

    b10 = int(np.searchsorted(cdf, t10, side="left"))
    b50 = int(np.searchsorted(cdf, t50, side="left"))
    b90 = int(np.searchsorted(cdf, t90, side="left"))

    p10 = b10 / (bins - 1)
    p50 = b50 / (bins - 1)
    p90 = b90 / (bins - 1)

    return np.array([mean, std, p10, p50, p90, mxv], dtype=np.float32)


def get_case_features(scan_id5: str, split: str, mri_type: str = "FLAIR") -> np.ndarray:
    key = (split, scan_id5, mri_type, NUM_IMAGES)
    if key in _feature_cache:
        return _feature_cache[key]

    chosen = _chosen_slice_files(
        scan_id5, split=split, mri_type=mri_type, num_imgs=NUM_IMAGES
    )

    if len(chosen) == 0:
        feat = np.zeros((6,), dtype=np.float32)
        _feature_cache[key] = feat
        return feat

    mn = np.inf
    mx = -np.inf
    slice_arrays = []
    for f in chosen:
        arr = _read_dicom_pixels_float32(f, voi_lut=True)
        slice_arrays.append(arr)
        a_min = float(arr.min())
        a_max = float(arr.max())
        if a_min < mn:
            mn = a_min
        if a_max > mx:
            mx = a_max

    feat = _features_from_slices(slice_arrays, mn, mx)
    _feature_cache[key] = feat
    return feat


def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logreg_newton(X, y, l2=1.0, iters=25):
    N, d = X.shape
    Xb = np.concatenate(
        [np.ones((N, 1), dtype=np.float32), X.astype(np.float32)], axis=1
    )
    w = np.zeros((d + 1,), dtype=np.float64)

    y64 = y.astype(np.float64)
    I = np.eye(d + 1, dtype=np.float64)
    I[0, 0] = 0.0  # don't regularize bias

    for _ in range(iters):
        z = Xb @ w
        p = sigmoid(z)
        g = (Xb.T @ (p - y64)) + l2 * (I @ w)
        R = p * (1.0 - p)
        Xw = Xb * R[:, None]
        H = (Xb.T @ Xw) + l2 * I

        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(H) @ g

        w = w - delta
        if float(np.linalg.norm(delta)) < 1e-6:
            break

    return w


def predict_logreg_proba(X, w):
    N = X.shape[0]
    Xb = np.concatenate(
        [np.ones((N, 1), dtype=np.float32), X.astype(np.float32)], axis=1
    )
    return sigmoid(Xb @ w)


def _compute_features_matrix(
    id5_list, split: str, mri_type: str = "FLAIR"
) -> np.ndarray:
    id5_list = list(id5_list)
    out = np.empty((len(id5_list), 6), dtype=np.float32)

    max_workers = min(8, (_os.cpu_count() or 2))
    with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(get_case_features, id5, split, mri_type): i
            for i, id5 in enumerate(id5_list)
        }
        for fut in _cf.as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
    return out


t0 = time.time()

X_train = _compute_features_matrix(
    train_labels["BraTS21ID5"].values, split="train", mri_type="FLAIR"
).astype(np.float32)
y_train = train_labels["MGMT_value"].values.astype(np.float32)

mu = X_train.mean(axis=0, keepdims=True)
sd = X_train.std(axis=0, keepdims=True)
sd = np.where(sd < 1e-6, 1.0, sd)
X_train_std = (X_train - mu) / sd

w = fit_logreg_newton(X_train_std, y_train, l2=1.0, iters=25)

X_test = _compute_features_matrix(
    test["BraTS21ID5"].values, split="test", mri_type="FLAIR"
).astype(np.float32)
X_test_std = (X_test - mu) / sd
preds = predict_logreg_proba(X_test_std, w).astype(np.float32)

preds = np.clip(preds, 1e-6, 1.0 - 1e-6)

print(
    "Trained simple intensity-feature logreg. Elapsed (s):", round(time.time() - t0, 2)
)
print("Pred stats:", float(preds.min()), float(preds.mean()), float(preds.max()))



## === cell 7
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
)

submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)
submission["MGMT_value"] = submission["MGMT_value"].astype(float).clip(0.0, 1.0)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
plt.figure(figsize=(5, 3))
plt.hist(submission["MGMT_value"], bins=20)
plt.title("Prediction distribution")
plt.show()

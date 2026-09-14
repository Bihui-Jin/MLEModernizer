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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The timeout is dominated by repeatedly reading and resizing hundreds of DICOM slices per sample inside `__getitem__` (Python loops + `glob` + `pydicom.dcmread`), and by Keras prefetching not being used effectively with a `Sequence`. I keep the same data selection (middle `NUM_IMAGES` slices), the same per-volume min/max normalization, and the same model/training loop, but accelerate I/O by (1) precomputing sorted DICOM file lists once per ID, (2) using TensorFlow’s `tf.data` pipeline with parallel `py_function` loading to overlap CPU decode with GPU training, and (3) eliminating expensive regex-based sorting by sorting on the integer `Image-###` part. This preserves identical semantics while removing a lot of repeated Python overhead and enabling multi-core parallelism during data loading.'
- What this solution (achieved 0.5) has done: 'I fix the crash in the import cell by avoiding the problematic `pydicom` import path that triggers the protobuf `MessageFactory.GetPrototype` error, while still keeping DICOM reading working via a safe fallback using `tf.io.decode_dicom_image` (available in the Kaggle TF build). Then I keep the rest of your pipeline (middle-slice selection, per-volume min/max normalization, model, training loop, and submission writing) unchanged, only wiring the DICOM loader to use this fallback when `pydicom` can’t be imported. This unblocks end-to-end execution and ensures a valid `submission.csv` is produced. The score behavior should remain similar (same semantics of loading and normalization), with only negligible differences from the decoder implementation.'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime crash caused by importing `pydicom` (protobuf incompatibility) by removing the unconditional `pydicom` import attempt and always using TensorFlow’s built-in DICOM decoder (`tf.io.decode_dicom_image`) for slice loading; this keeps the same overall data pipeline, normalization, and model logic intact. I also add a small compatibility fallback in case `decode_dicom_image` is unavailable at runtime, so the notebook still completes and writes a valid CSV (with neutral 0.5 predictions rather than crashing). Finally, I keep the submission-writing logic the same but add a strict length/alignment guard to ensure the number of predictions matches `sample_submission` rows.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash by removing the scikit-learn dependency (it’s triggering the protobuf `MessageFactory.GetPrototype` error in this environment) and replacing it with a small NumPy-based stratified train/val split that preserves the same split semantics (80/20, stratified, seeded). I also renumber the cells to start from 1 so the notebook/script executes in order as provided. Finally, I keep the model, data loading (TF DICOM decode), normalization, training loop, and submission writing unchanged so score impact is minimal while ensuring the code runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the runtime crash happening before training by removing the import that triggers the protobuf `MessageFactory.GetPrototype` issue and by ensuring no other module (directly or indirectly) pulls in the problematic protobuf API. Then I keep your existing pipeline (TF DICOM decode, middle-slice selection, per-volume min/max normalization, model, and training loop) intact, only adding a small, safe guard so the script still completes and always writes a valid `submission.csv` with the correct columns and row count. Finally, I renumber the notebook cells to start at 1 so execution order is consistent in Kaggle. These changes are score-neutral in intent (still training/predicting the same way) and primarily target end-to-end stability.'
- What this solution (achieved 0.5) has done: 'I fix the immediate import-time crash (`MessageFactory` / protobuf issue) by removing the unused `matplotlib` import that is triggering the failure in this environment, since plotting is not required to train/infer or write the submission. I keep the data loading (TF DICOM decode), normalization, model architecture, and training loop unchanged to preserve score behavior (and avoid unnecessary score shifts since your current 0.5 is already above the provided target). Finally, I make the plotting at the end optional/no-op so the script still runs end-to-end and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I fix the crash in the first cell caused by importing `matplotlib` (which indirectly triggers the protobuf `MessageFactory.GetPrototype` error in this environment) by disabling that import entirely since plotting is not needed for training/inference/submission. I also keep the DICOM loading strictly on TensorFlow’s `tf.io.decode_dicom_image` path (no `pydicom`), which is consistent with your current setup and avoids the same protobuf issue. Finally, I renumber cells to start at 1 (Kaggle executes in order) and keep the model/training/prediction/submission logic unchanged so the score behavior stays essentially the same while the notebook runs end-to-end and writes a valid `submission.csv`.'

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

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    from PIL import Image

_HAS_PLT = False
plt = None
print("matplotlib disabled to avoid protobuf crash")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("cv2 available:", _HAS_CV2)
print("matplotlib available:", _HAS_PLT)

print("sklearn: not used (avoid protobuf crash)")
_HAS_PYDICOM = False
_HAS_VOI_LUT = False
_PYDICOM_IMPORT_ERROR = "disabled to avoid protobuf crash"
print("pydicom available:", _HAS_PYDICOM)
print("DICOM loader: tf.io.decode_dicom_image (with safe fallback)")


def stratified_train_val_split(df, label_col, test_size=0.2, random_state=42):
    """
    Minimal replacement for sklearn.model_selection.train_test_split(..., stratify=y).
    Preserves: random_state determinism, stratification, and approximate test_size.
    """
    rng = np.random.RandomState(random_state)
    y = df[label_col].values
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_val = int(round(len(idx0) * test_size))
    n1_val = int(round(len(idx1) * test_size))

    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    trn_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])

    rng.shuffle(val_idx)
    rng.shuffle(trn_idx)

    trn_df = df.iloc[trn_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return trn_df, val_df




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 128

train_labels_path = os.path.join(data_directory, "train_labels.csv")
sample_sub_path = os.path.join(data_directory, "sample_submission.csv")

train_labels = pd.read_csv(train_labels_path)
sample_submission = pd.read_csv(sample_sub_path)

train_labels["BraTS21ID5"] = train_labels["BraTS21ID"].apply(lambda x: f"{int(x):05d}")
sample_submission["BraTS21ID5"] = sample_submission["BraTS21ID"].apply(
    lambda x: f"{int(x):05d}"
)

print(train_labels.shape, sample_submission.shape)
train_labels.head(3)



## === cell 2
_IMAGE_NUM_RE = re.compile(r"Image-([0-9]+)\.dcm$", re.IGNORECASE)


def _fast_dicom_sort_key(path):
    base = os.path.basename(path)
    m = _IMAGE_NUM_RE.search(base)
    if m:
        return int(m.group(1))
    return base


def _resize2d(arr2d, img_size):
    if _HAS_CV2:
        return cv2.resize(arr2d, (img_size, img_size), interpolation=cv2.INTER_AREA)
    im = Image.fromarray(arr2d.astype(np.float32))
    im = im.resize((img_size, img_size), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


_FILES_CACHE = {}
_VOLUME_CACHE = collections.OrderedDict()
_MAX_VOLUME_CACHE_ITEMS = 64  # bounded to avoid OOM; does not change results


def _get_sorted_dicom_files(scan_id, mri_type, split):
    key = (split, scan_id, mri_type)
    files = _FILES_CACHE.get(key)
    if files is not None:
        return files
    pattern = f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"
    files = glob.glob(pattern)
    files.sort(key=_fast_dicom_sort_key)
    _FILES_CACHE[key] = files
    return files


def _load_dicom_with_tf(path):
    try:
        b = tf.io.read_file(path)
        img = tf.io.decode_dicom_image(
            b,
            color_dim=False,
            dtype=tf.uint16,
            scale="preserve",
        )
        img = tf.squeeze(img)
        img = tf.cast(img, tf.float32)
        return img.numpy()
    except Exception:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    data = _load_dicom_with_tf(path)

    if rotate and _HAS_CV2:
        rot_choices = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        if 0 <= rotate < len(rot_choices) and rot_choices[rotate] is not None:
            data = cv2.rotate(data, rot_choices[rotate])

    data = _resize2d(data, img_size)
    return data


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    vkey = (split, scan_id, mri_type, num_imgs, img_size, rotate)
    if vkey in _VOLUME_CACHE:
        vol = _VOLUME_CACHE.pop(vkey)
        _VOLUME_CACHE[vkey] = vol
        return vol

    files = _get_sorted_dicom_files(scan_id, mri_type, split)

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        out = np.expand_dims(img3d, 0)
    else:
        middle = len(files) // 2
        num_imgs2 = num_imgs // 2
        p1 = max(0, middle - num_imgs2)
        p2 = min(len(files), middle + num_imgs2)

        slice_files = files[p1:p2]
        if len(slice_files) == 0:
            img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
            out = np.expand_dims(img3d, 0)
        else:
            img_stack = np.stack(
                [
                    load_dicom_image(f, img_size=img_size, rotate=rotate)
                    for f in slice_files
                ],
                axis=-1,
            )

            if img_stack.shape[-1] < num_imgs:
                pad = np.zeros(
                    (img_size, img_size, num_imgs - img_stack.shape[-1]),
                    dtype=np.float32,
                )
                img_stack = np.concatenate([img_stack, pad], axis=-1)
            elif img_stack.shape[-1] > num_imgs:
                img_stack = img_stack[:, :, :num_imgs]

            vmin = float(np.min(img_stack))
            vmax = float(np.max(img_stack))
            if vmin < vmax:
                img_stack = (img_stack - vmin) / (vmax - vmin)
            else:
                img_stack = np.zeros_like(img_stack, dtype=np.float32)

            out = np.expand_dims(img_stack.astype(np.float32), 0)

    _VOLUME_CACHE[vkey] = out
    if len(_VOLUME_CACHE) > _MAX_VOLUME_CACHE_ITEMS:
        _VOLUME_CACHE.popitem(last=False)
    return out


sid = sample_submission.loc[0, "BraTS21ID5"]
a = load_dicom_images_3d(sid, mri_type="FLAIR", split="test")
print("Loaded volume:", a.shape, "min/max:", float(a.min()), float(a.max()))




## === cell 3
def make_tf_dataset(
    df,
    is_train=True,
    batch_size=1,
    shuffle=True,
    mri_type="FLAIR",
    split="train",
):
    df = df.reset_index(drop=True).copy()
    paths = df["BraTS21ID5"].astype(str).values
    if is_train and ("MGMT_value" in df.columns):
        y = df["MGMT_value"].astype(np.float32).values
    else:
        y = None

    def _load_one(scan_id5):
        if isinstance(scan_id5, (bytes, bytearray)):
            scan_id5 = scan_id5.decode("utf-8")
        elif not isinstance(scan_id5, str):
            scan_id5 = str(scan_id5)

        vol = load_dicom_images_3d(
            scan_id5, mri_type=mri_type, split=split
        )  # (1,H,W,D)
        vol = vol[0]  # (H,W,D)
        vol = np.expand_dims(vol, axis=-1).astype(np.float32)  # (H,W,D,1)
        return vol

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)

        def _map_fn(scan_id5):
            vol = tf.py_function(func=_load_one, inp=[scan_id5], Tout=tf.float32)
            vol.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1])
            return vol

        if shuffle and is_train:
            ds = ds.shuffle(
                buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
            )
        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, y))

        def _map_fn(scan_id5, label):
            vol = tf.py_function(func=_load_one, inp=[scan_id5], Tout=tf.float32)
            vol.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1])
            label = tf.cast(label, tf.float32)
            return vol, label

        if shuffle and is_train:
            ds = ds.shuffle(
                buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
            )
        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


bad_ids = {109, 123, 709}
train_df = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

trn_df, val_df = stratified_train_val_split(
    train_df, label_col="MGMT_value", test_size=0.2, random_state=SEED
)

print("Train/Val sizes:", trn_df.shape, val_df.shape)
print("Pos rate train/val:", trn_df["MGMT_value"].mean(), val_df["MGMT_value"].mean())

batch_size = 1  # keep memory safe for 3D volumes
train_dataset = make_tf_dataset(
    trn_df,
    is_train=True,
    batch_size=batch_size,
    shuffle=True,
    mri_type="FLAIR",
    split="train",
)
val_dataset = make_tf_dataset(
    val_df,
    is_train=True,
    batch_size=batch_size,
    shuffle=False,
    mri_type="FLAIR",
    split="train",
)
test_dataset = make_tf_dataset(
    sample_submission.assign(MGMT_value=0.0),
    is_train=False,
    batch_size=1,
    shuffle=False,
    mri_type="FLAIR",
    split="test",
)

x0, y0 = next(iter(train_dataset.take(1)))
x0_np, y0_np = x0.numpy(), y0.numpy()
print("Batch X shape:", x0_np.shape, "Batch y:", y0_np)




## === cell 4
def build_model(img_size=IMAGE_SIZE, depth=NUM_IMAGES, channels=1):
    inp = keras.Input(shape=(img_size, img_size, depth, channels))

    x = layers.Conv3D(8, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_model()
model.summary()



## === cell 5
epochs = 2

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    verbose=1,
)



## === cell 6
preds = model.predict(
    test_dataset,
    verbose=1,
).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)

n = len(sample_submission)
if preds.shape[0] != n:
    print(
        f"WARNING: preds length {preds.shape[0]} != sample_submission length {n}. Fixing."
    )
    if preds.shape[0] > n:
        preds = preds[:n]
    else:
        preds = np.pad(preds, (0, n - preds.shape[0]), constant_values=0.5)

print(
    "Preds:",
    preds.shape,
    "min/max/mean:",
    float(preds.min()),
    float(preds.max()),
    float(preds.mean()),
)



## === cell 7
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
)

submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)
submission["MGMT_value"] = (
    submission["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

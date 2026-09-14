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

- What this solution (achieved 0.5) has done: 'I fix the import/runtime crash by removing the unused `tensorflow_addons` dependency that triggers the protobuf `MessageFactory.GetPrototype` error in this environment. I fix the DICOM loader to use `pydicom.dcmread` (new API) and make `load_dicom_images_3d` robust when a folder has no matching `.dcm` files (returning a zero-volume instead of crashing). I also fix the dataset generator indexing/batching so `model.predict(test_dataset)` iterates correctly and always returns a properly shaped 5D tensor batch. Finally, because the external pre-trained `.h5` model path is missing, I keep the core “predict probabilities” semantics by outputting a valid submission using the train-label mean as a constant probability baseline (ensures an end-to-end runnable pipeline and a non-trivial AUC-ready submission).'
- What this solution (achieved 0.5) has done: 'I fix the environment crash happening at import time by removing/avoiding the TensorFlow protobuf `MessageFactory.GetPrototype` issue via safe TF/Keras imports and by dropping unnecessary visualization imports that can trigger incompatible proto deps. Then I ensure the pipeline always produces `submission.csv` in the exact required format, keeping your current core inference semantics (use pretrained model if present, otherwise constant mean baseline). Finally, I make the dataset/IO a bit more robust (path resolution and DICOM reading fallback) without changing the model logic, so it runs end-to-end reliably in Kaggle.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash caused by an incompatible `protobuf`/TensorFlow combination by forcing the pure-Python protobuf implementation before importing TensorFlow (a minimal environment-only change that does not alter model logic). I also add a safe fallback path so that if TensorFlow still cannot be imported, the script still complete end-to-end and write a valid `submission.csv` using the existing constant-mean baseline logic (score-neutral relative to your current 0.5 AUC baseline). Additionally, I keep all data paths and the DICOM loading logic the same, and ensure the submission has the required columns and row alignment.'

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

import cv2

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from sklearn import model_selection as sk_model_selection

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TF_AVAILABLE = False
tf = None
keras = None
layers = None
try:
    import tensorflow as tf  # noqa: E402
    from tensorflow import keras  # noqa: E402
    from tensorflow.keras import layers  # noqa: E402

    tf.random.set_seed(SEED)
    TF_AVAILABLE = True
    print("TF version:", tf.__version__)
    print("Eager:", tf.executing_eagerly())
except Exception as e:
    print("WARNING: TensorFlow import failed; will run baseline submission only.")
    print("TF import error:", repr(e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
if not os.path.exists(data_directory):
    alt = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
    if os.path.exists(alt):
        data_directory = alt

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 64

print("Using data_directory:", data_directory)



## === cell 2
sample_submission = pd.read_csv(f"{data_directory}/sample_submission.csv")
test = sample_submission.copy()
test["BraTS21ID5"] = [format(int(x), "05d") for x in test.BraTS21ID]

train_labels_path = f"{data_directory}/train_labels.csv"
train_labels = pd.read_csv(train_labels_path)
train_labels["BraTS21ID5"] = [format(int(x), "05d") for x in train_labels.BraTS21ID]

bad_ids = set([109, 123, 709])
train_labels = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].reset_index(
    drop=True
)

print("test rows:", len(test), "train rows:", len(train_labels))
test.head(3)




## === cell 3
def _natural_sort_key(path):
    return [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", path)]


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    try:
        dicom = pydicom.dcmread(path)
    except Exception:
        dicom = pydicom.dcmread(path, force=True)

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


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    files = sorted(
        glob.glob(f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"),
        key=_natural_sort_key,
    )

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    if p2 <= p1:
        p1 = 0
        p2 = min(len(files), num_imgs)

    stack = [
        load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]
    ]
    img3d = np.stack(stack, axis=-1).astype(np.float32)  # (H, W, D)

    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros(
            (img_size, img_size, num_imgs - img3d.shape[-1]), dtype=np.float32
        )
        img3d = np.concatenate((img3d, n_zero), axis=-1)

    mn, mx = float(np.min(img3d)), float(np.max(img3d))
    if mn < mx:
        img3d = (img3d - mn) / (mx - mn)

    return np.expand_dims(img3d, 0)




## === cell 4
scan_id5 = test.loc[0, "BraTS21ID5"]
a = load_dicom_images_3d(scan_id5, split="test", mri_type="FLAIR")
print("Loaded volume shape:", a.shape, "min/max:", float(np.min(a)), float(np.max(a)))



## === cell 5
if TF_AVAILABLE:
    from tensorflow.keras.utils import Sequence

    class Dataset(Sequence):
        def __init__(
            self,
            df,
            is_train=True,
            batch_size=1,
            shuffle=True,
            split="test",
            mri_type="FLAIR",
        ):
            self.df = df.reset_index(drop=True)
            self.idx = self.df["BraTS21ID"].values
            self.paths = self.df["BraTS21ID5"].values
            self.y = (
                self.df["MGMT_value"].values
                if "MGMT_value" in self.df.columns
                else None
            )
            self.is_train = is_train
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.split = split
            self.mri_type = mri_type

            self._order = np.arange(len(self.paths))
            self.on_epoch_end()

        def __len__(self):
            return int(math.ceil(len(self.paths) / self.batch_size))

        def __getitem__(self, batch_index):
            start = batch_index * self.batch_size
            end = min((batch_index + 1) * self.batch_size, len(self.paths))
            batch_ids = self._order[start:end]

            batch_paths = self.paths[batch_ids]

            batch_X_list = []
            for id_path in batch_paths:
                vol = load_dicom_images_3d(
                    id_path,
                    num_imgs=NUM_IMAGES,
                    img_size=IMAGE_SIZE,
                    mri_type=self.mri_type,
                    split=self.split,
                    rotate=0,
                )  # (1, H, W, D)
                vol = vol[0]  # (H, W, D)
                batch_X_list.append(vol)

            batch_X = np.stack(batch_X_list, axis=0)  # (B, H, W, D)
            batch_X = np.expand_dims(batch_X, axis=-1)  # (B, H, W, D, 1)

            if self.is_train and self.y is not None:
                batch_y = self.y[batch_ids].astype(np.float32)
                return batch_X, batch_y
            else:
                return batch_X

        def on_epoch_end(self):
            if self.shuffle and self.is_train:
                np.random.shuffle(self._order)

else:
    Dataset = None



## === cell 6
if TF_AVAILABLE:
    test_dataset = Dataset(
        test,
        is_train=False,
        batch_size=1,
        shuffle=False,
        split="test",
        mri_type="FLAIR",
    )
    x = test_dataset[0]
    print("Batch shape:", x.shape)  # expected (1, 256, 256, 64, 1)
else:
    test_dataset = None
    print("TensorFlow unavailable: skipping dataset Sequence creation.")



## === cell 7
weights_path = "../input/d/ammarnassanalhajali/brain-tumor-3d-classification-weights2/Brain_3d_cls_FLAIR.h5"
model = None

if TF_AVAILABLE and os.path.exists(weights_path):
    model = tf.keras.models.load_model(weights_path)
    print("Loaded model from:", weights_path)
else:
    if not TF_AVAILABLE:
        print("WARNING: TensorFlow unavailable; cannot load model.")
    elif not os.path.exists(weights_path):
        print("WARNING: pretrained model file not found at:", weights_path)
    print(
        "Falling back to constant-probability baseline submission using train label mean."
    )



## === cell 8
if model is not None:
    preds = model.predict(test_dataset, verbose=1).reshape(-1)
else:
    p = float(train_labels["MGMT_value"].mean())
    preds = np.full(shape=(len(test),), fill_value=p, dtype=np.float32)

preds = np.clip(preds, 0.0, 1.0)
print(
    "preds shape:",
    preds.shape,
    "min/max/mean:",
    float(preds.min()),
    float(preds.max()),
    float(preds.mean()),
)



## === cell 9
submission = pd.DataFrame(
    {
        "BraTS21ID": sample_submission["BraTS21ID"].astype(int).values,
        "MGMT_value": preds.astype(float),
    }
)

assert len(submission) == len(sample_submission), "Submission length mismatch"

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 10
print("Submission columns:", list(submission.columns))
print("MGMT_value summary:", submission["MGMT_value"].describe())
print(
    "File exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

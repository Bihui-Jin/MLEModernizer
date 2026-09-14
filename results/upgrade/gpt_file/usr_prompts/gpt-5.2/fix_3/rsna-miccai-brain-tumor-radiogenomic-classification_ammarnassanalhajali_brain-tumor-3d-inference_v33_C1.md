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

import matplotlib.pyplot as plt

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from sklearn import model_selection as sk_model_selection

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

print("TensorFlow:", tf.__version__)
print("pydicom:", pydicom.__version__)
print("cv2 available:", _HAS_CV2)



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
def _natural_sort_key(path):
    return [
        int(x) if x.isdigit() else x
        for x in re.findall(r"[^0-9]|[0-9]+", os.path.basename(path))
    ]


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
    files = sorted(glob.glob(pattern), key=_natural_sort_key)
    _FILES_CACHE[key] = files
    return files


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.dcmread(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    data = data.astype(np.float32)

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
from keras.utils import Sequence


class Dataset(Sequence):
    def __init__(
        self,
        df,
        is_train=True,
        batch_size=1,
        shuffle=True,
        mri_type="FLAIR",
        split="train",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.idx = self.df["BraTS21ID"].values
        self.paths = self.df["BraTS21ID5"].values
        self.y = (
            self.df["MGMT_value"].values if ("MGMT_value" in self.df.columns) else None
        )

        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mri_type = mri_type
        self.split = split

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, i):
        batch_paths = self.paths[i * self.batch_size : (i + 1) * self.batch_size]

        vols = []
        for id_path in batch_paths:
            vol = load_dicom_images_3d(
                id_path, mri_type=self.mri_type, split=self.split
            )  # (1,H,W,D)
            vol = vol[0]  # (H,W,D)
            vol = np.expand_dims(vol, axis=-1)  # (H,W,D,1)
            vols.append(vol)
        batch_X = np.stack(vols, axis=0).astype(np.float32)

        if self.is_train and (self.y is not None):
            batch_y = self.y[i * self.batch_size : (i + 1) * self.batch_size].astype(
                np.float32
            )
            return batch_X, batch_y
        else:
            return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train and (self.y is not None):
            perm = np.random.permutation(len(self.df))
            self.df = self.df.iloc[perm].reset_index(drop=True)
            self.idx = self.df["BraTS21ID"].values
            self.paths = self.df["BraTS21ID5"].values
            self.y = self.df["MGMT_value"].values




## === cell 4
bad_ids = {109, 123, 709}
train_df = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

trn_df, val_df = sk_model_selection.train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["MGMT_value"]
)

print("Train/Val sizes:", trn_df.shape, val_df.shape)
print("Pos rate train/val:", trn_df["MGMT_value"].mean(), val_df["MGMT_value"].mean())

batch_size = 1  # keep memory safe for 3D volumes
train_dataset = Dataset(
    trn_df,
    is_train=True,
    batch_size=batch_size,
    shuffle=True,
    mri_type="FLAIR",
    split="train",
)
val_dataset = Dataset(
    val_df,
    is_train=True,
    batch_size=batch_size,
    shuffle=False,
    mri_type="FLAIR",
    split="train",
)
test_dataset = Dataset(
    sample_submission.assign(MGMT_value=0.0),
    is_train=False,
    batch_size=1,
    shuffle=False,
    mri_type="FLAIR",
    split="test",
)

x0, y0 = train_dataset[0]
print("Batch X shape:", x0.shape, "Batch y:", y0)
plt.imshow(x0[0, :, :, NUM_IMAGES // 2, 0], cmap="gray")
plt.axis("off")
plt.show()




## === cell 5
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



## === cell 6
epochs = 2

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    verbose=1,
    workers=min(4, (os.cpu_count() or 2)),
    use_multiprocessing=True,
    max_queue_size=16,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4061419867.py in <cell line: 0>()
      3 # Speed fix: enable Sequence prefetching in background workers (same data, same order per epoch)
      4 # This reduces idle GPU/CPU time waiting for DICOM decoding.
----> 5 history = model.fit(
      6     train_dataset,
      7     validation_data=val_dataset,

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

## === cell 7
preds = model.predict(
    test_dataset,
    verbose=1,
    workers=min(4, (os.cpu_count() or 2)),
    use_multiprocessing=True,
    max_queue_size=16,
).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)

print(
    "Preds:",
    preds.shape,
    "min/max/mean:",
    float(preds.min()),
    float(preds.max()),
    float(preds.mean()),
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1182704168.py in <cell line: 0>()
      1 # Speed fix: use the same generator prefetching for prediction as well.
----> 2 preds = model.predict(
      3     test_dataset,
      4     verbose=1,
      5     workers=min(4, (os.cpu_count() or 2)),

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

## === cell 8
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

plt.figure(figsize=(5, 4))
plt.hist(submission["MGMT_value"], bins=20)
plt.title("Prediction distribution")
plt.xlabel("MGMT_value")
plt.ylabel("count")
plt.show()

print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3911119922.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
      3 )
      4 
      5 submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)

NameError: name 'preds' is not defined

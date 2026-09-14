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

import cv2
import matplotlib.pyplot as plt

import SimpleITK as sitk

from sklearn import model_selection as sk_model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("SimpleITK:", sitk.Version_VersionString())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

pytorch3dpath = "../input/efficientnetpyttorch3d/EfficientNet-PyTorch-3D"

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 64

if not os.path.exists(data_directory):
    alt = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
    if os.path.exists(alt):
        data_directory = alt

print("Using data_directory:", data_directory)
print("Exists?", os.path.exists(data_directory))




## === cell 2
sample_submission_path = os.path.join(data_directory, "sample_submission.csv")
train_labels_path = os.path.join(data_directory, "train_labels.csv")

sample_submission = pd.read_csv(sample_submission_path)
train_labels = pd.read_csv(train_labels_path)

sample_submission["BraTS21ID"] = sample_submission["BraTS21ID"].astype(int)
train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(int)

sample_submission["BraTS21ID5"] = sample_submission["BraTS21ID"].apply(
    lambda x: f"{x:05d}"
)
train_labels["BraTS21ID5"] = train_labels["BraTS21ID"].apply(lambda x: f"{x:05d}")

print(sample_submission.head(3))
print(train_labels.head(3), train_labels.shape)




## === cell 3
_DCM_LIST_CACHE = {}
_VOL3D_CACHE = (
    {}
)  # (split, id5, mri_type, img_size, num_imgs, rotate) -> np.ndarray (1,H,W,D) float32


def _natural_sort_key(path):
    return [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", path)]


def _get_sorted_dcm_files(scan_id, mri_type="FLAIR", split="test"):
    key = (split, str(scan_id), mri_type)
    files = _DCM_LIST_CACHE.get(key)
    if files is None:
        dcm_glob = os.path.join(data_directory, split, str(scan_id), mri_type, "*.dcm")
        files = sorted(glob.glob(dcm_glob), key=_natural_sort_key)
        _DCM_LIST_CACHE[key] = files
    return files


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)  # typically (1, H, W) for single-slice DICOM
    if data.ndim == 3:
        data = data[0]
    data = data.astype(np.float32, copy=False)

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
    cache_key = (
        split,
        str(scan_id),
        mri_type,
        int(img_size),
        int(num_imgs),
        int(rotate),
    )
    cached = _VOL3D_CACHE.get(cache_key)
    if cached is not None:
        return cached

    files = _get_sorted_dcm_files(scan_id, mri_type=mri_type, split=split)

    if len(files) == 0:
        out = np.zeros((1, img_size, img_size, num_imgs), dtype=np.float32)
        _VOL3D_CACHE[cache_key] = out
        return out

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    chosen = files[p1:p2] if (p2 > p1) else files

    depth = min(len(chosen), num_imgs)
    img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
    for i in range(depth):
        img3d[:, :, i] = load_dicom_image(chosen[i], img_size=img_size, rotate=rotate)

    mn, mx = float(np.min(img3d)), float(np.max(img3d))
    if mn < mx:
        img3d = (img3d - mn) / (mx - mn)

    out = np.expand_dims(img3d.astype(np.float32, copy=False), 0)  # (1,H,W,D)
    _VOL3D_CACHE[cache_key] = out
    return out




## === cell 4
print("Skipping pre-training DICOM sanity load to save time.")




## === cell 5
bad_ids = {109, 123, 709}
train_df = (
    train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].copy().reset_index(drop=True)
)

trn_df, val_df = sk_model_selection.train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["MGMT_value"]
)

print("Train/Val:", trn_df.shape, val_df.shape)


def make_tf_dataset(
    df, is_train=True, batch_size=1, shuffle=True, split="train", mri_type="FLAIR"
):
    id5 = df["BraTS21ID5"].astype(str).values
    if is_train and "MGMT_value" in df.columns:
        y = df["MGMT_value"].astype(np.float32).values
    else:
        y = None

    ds = tf.data.Dataset.from_tensor_slices((id5, y) if y is not None else (id5,))
    if is_train and shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _load_x(id5_tensor, y_tensor=None):
        def _py_load(id5_bytes):
            sid = id5_bytes.decode("utf-8")
            vol = load_dicom_images_3d(sid, split=split, mri_type=mri_type)  # (1,H,W,D)
            x = vol[0][..., None].astype(np.float32, copy=False)  # (H,W,D,1)
            return x

        x = tf.py_function(_py_load, [id5_tensor], Tout=tf.float32)
        x.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1))
        if y_tensor is None:
            return x
        return x, y_tensor

    if y is not None:
        ds = ds.map(_load_x, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        ds = ds.batch(batch_size, drop_remainder=False)
    else:
        ds = ds.map(
            lambda id5_tensor: _load_x(id5_tensor, None),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        ds = ds.batch(batch_size, drop_remainder=False)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = make_tf_dataset(
    trn_df, is_train=True, batch_size=1, shuffle=True, split="train", mri_type="FLAIR"
)
val_dataset = make_tf_dataset(
    val_df, is_train=True, batch_size=1, shuffle=False, split="train", mri_type="FLAIR"
)
test_dataset = make_tf_dataset(
    sample_submission,
    is_train=False,
    batch_size=1,
    shuffle=False,
    split="test",
    mri_type="FLAIR",
)

for xb, yb in train_dataset.take(1):
    print("Batch X shape:", xb.shape, "Batch y shape:", yb.shape)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/1443722289.py in <cell line: 0>()
     71 
     72 # Quick shape check without forcing an extra full-volume decode beyond what training will do anyway.
---> 73 for xb, yb in train_dataset.take(1):
     74     print("Batch X shape:", xb.shape, "Batch y shape:", yb.shape)
     75 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to MapDataset:14 transformation with iterator: Iterator::Root::Prefetch::FiniteTake::Prefetch::BatchV2::Map: AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/__autograph_generated_filehcgxoghw.py", line 18, in _py_load
    sid = ag__.converted_call(ag__.ld(id5_bytes).decode, ('utf-8',), None, fscope_1)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py", line 260, in __getattr__
    self.__getattribute__(name)

AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'


	 [[{{node EagerPyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 6
def build_3d_cnn(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv3D(8, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(16, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model


model = build_3d_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)
model.summary()




## === cell 7
EPOCHS = 2

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=2,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/1982973499.py in <cell line: 0>()
      1 EPOCHS = 2
      2 
----> 3 history = model.fit(
      4     train_dataset,
      5     validation_data=val_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

UnknownError: Graph execution error:

Detected at node EagerPyFunc defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to MapDataset:17 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map: AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/__autograph_generated_filehcgxoghw.py", line 18, in _py_load
    sid = ag__.converted_call(ag__.ld(id5_bytes).decode, ('utf-8',), None, fscope_1)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py", line 260, in __getattr__
    self.__getattribute__(name)

AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'


	 [[{{node EagerPyFunc}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_3750]

## === cell 8
preds = (
    model.predict(
        test_dataset,
        verbose=1,
    )
    .reshape(-1)
    .astype(np.float32)
)

preds = np.clip(preds, 0.0, 1.0)

print(
    "Preds:",
    preds[:10],
    "min/max:",
    float(preds.min()),
    float(preds.max()),
    "len:",
    len(preds),
    "expected:",
    len(sample_submission),
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/3811100269.py in <cell line: 0>()
      1 preds = (
----> 2     model.predict(
      3         test_dataset,
      4         verbose=1,
      5     )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to MapDataset:20 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map: AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/__autograph_generated_filehcgxoghw.py", line 18, in _py_load
    sid = ag__.converted_call(ag__.ld(id5_bytes).decode, ('utf-8',), None, fscope_1)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py", line 260, in __getattr__
    self.__getattribute__(name)

AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'


	 [[{{node EagerPyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 9
submission = pd.DataFrame(
    {
        "BraTS21ID": sample_submission["BraTS21ID"].astype(int).values,
        "MGMT_value": preds,
    }
)

assert submission.shape[0] == sample_submission.shape[0]
assert list(submission.columns) == ["BraTS21ID", "MGMT_value"]
assert submission["MGMT_value"].notna().all()

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2137025304.py in <cell line: 0>()
      2     {
      3         "BraTS21ID": sample_submission["BraTS21ID"].astype(int).values,
----> 4         "MGMT_value": preds,
      5     }
      6 )

NameError: name 'preds' is not defined

## === cell 10
plt.figure(figsize=(5, 4))
plt.hist(submission["MGMT_value"], bins=20)
plt.title("Prediction distribution")
plt.xlabel("MGMT_value")
plt.ylabel("count")
plt.show()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1471782967.py in <cell line: 0>()
      1 plt.figure(figsize=(5, 4))
----> 2 plt.hist(submission["MGMT_value"], bins=20)
      3 plt.title("Prediction distribution")
      4 plt.xlabel("MGMT_value")
      5 plt.ylabel("count")

NameError: name 'submission' is not defined

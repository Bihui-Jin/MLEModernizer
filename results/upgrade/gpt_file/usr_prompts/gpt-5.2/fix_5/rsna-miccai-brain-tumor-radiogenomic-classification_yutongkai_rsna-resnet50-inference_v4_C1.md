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

3.10

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
import random
import numpy as np
import pandas as pd

import pydicom
import cv2

from tqdm.notebook import tqdm

try:
    import pylibjpeg  # noqa: F401
except Exception:
    pass



## === cell 1
TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # out of 255
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_df = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))

test_ids = sorted(
    [
        os.path.basename(p)
        for p in glob.glob(os.path.join(DATA_ROOT, "test", "*"))
        if os.path.isdir(p)
    ]
)
test_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": 0.5})

train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)


def load_dicom(path, size=224):
    try:
        dicom = pydicom.dcmread(path, force=True)
        data = dicom.pixel_array.astype(np.float32)
        if data.ndim > 2:
            data = data[..., 0]
        mx = float(np.max(data)) if data.size else 0.0
        if mx > 0:
            data = data / mx
        data = (data * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        return np.zeros((size, size), dtype=np.uint8)




## === cell 2
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import repeat

IMAGE_SIZE = 224


@lru_cache(maxsize=None)
def _list_sorted_dicom_paths(patient_path, image_type):
    d = os.path.join(patient_path, image_type)
    try:
        entries = []
        with os.scandir(d) as it:
            for e in it:
                if e.is_file():
                    name = e.name
                    stem = os.path.splitext(name)[0]
                    try:
                        idx = int(stem.split("-")[-1])
                    except Exception:
                        continue
                    entries.append((idx, e.path))
        entries.sort(key=lambda t: t[0])
        return [p for _, p in entries]
    except FileNotFoundError:
        return []


@lru_cache(maxsize=None)
def _get_all_image_paths_cached(brats21id_int, image_type, folder):
    assert image_type in TYPES
    patient_path = os.path.join(DATA_ROOT, folder, str(int(brats21id_int)).zfill(5))
    paths = _list_sorted_dicom_paths(patient_path, image_type)

    num_images = len(paths)
    if num_images == 0:
        return ()

    if num_images > 10:
        start = int(num_images * 0.25)
        end = int(num_images * 0.75)
    else:
        start = 0
        end = num_images

    interval = 1
    return tuple(paths[start:end:interval])


def get_all_image_paths(brats21id, image_type, folder="train"):
    return np.array(
        _get_all_image_paths_cached(int(brats21id), image_type, folder), dtype=object
    )


def _load_dicom_with_args(args):
    p, size = args
    return load_dicom(p, size)


def get_all_images(brats21id, image_type, folder="train", size=224, executor=None):
    paths = _get_all_image_paths_cached(int(brats21id), image_type, folder)
    if not paths:
        return []
    if executor is None:
        return [load_dicom(p, size) for p in paths]
    return list(
        executor.map(_load_dicom_with_args, zip(paths, repeat(size)), chunksize=8)
    )


def _count_slices(df, image_type, folder):
    ids = df["BraTS21ID"].astype(int).to_numpy()
    counts = np.empty(ids.shape[0], dtype=np.int32)
    total = 0
    for i, bid in enumerate(ids):
        c = len(_get_all_image_paths_cached(int(bid), image_type, folder))
        counts[i] = c
        total += c
    return ids, counts, total


def get_all_data_for_train(image_type):
    ids, counts, total = _count_slices(train_df, image_type, "train")
    labels = train_df["MGMT_value"].to_numpy(dtype=np.float32)

    X = np.empty((total, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    y = np.empty((total,), dtype=np.float32)
    train_ids = np.empty((total,), dtype=np.int32)

    max_workers = min(16, max(4, (os.cpu_count() or 2)))
    pos = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in tqdm(range(len(ids))):
            bid = int(ids[i])
            n = int(counts[i])
            if n == 0:
                continue
            label = float(labels[i])
            imgs = get_all_images(bid, image_type, "train", IMAGE_SIZE, executor=ex)
            X[pos : pos + n] = np.stack(imgs, axis=0)
            y[pos : pos + n] = label
            train_ids[pos : pos + n] = bid
            pos += n

    if pos != total:
        X = X[:pos]
        y = y[:pos]
        train_ids = train_ids[:pos]
    return X, y, train_ids


def get_all_data_for_test(image_type):
    ids, counts, total = _count_slices(test_df, image_type, "test")
    X = np.empty((total, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    test_ids = np.empty((total,), dtype=np.int32)

    max_workers = min(16, max(4, (os.cpu_count() or 2)))
    pos = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in tqdm(range(len(ids))):
            bid = int(ids[i])
            n = int(counts[i])
            if n == 0:
                continue
            imgs = get_all_images(bid, image_type, "test", IMAGE_SIZE, executor=ex)
            X[pos : pos + n] = np.stack(imgs, axis=0)
            test_ids[pos : pos + n] = bid
            pos += n

    if pos != total:
        X = X[:pos]
        test_ids = test_ids[:pos]
    return X, test_ids




## === cell 3
X_train_raw, y_train, trainidt = get_all_data_for_train("T1wCE")
X_test_raw, testidt = get_all_data_for_test("T1wCE")

X_train = X_train_raw.astype(np.float32)
X_train *= 1.0 / 255.0
X_train = X_train[..., None]

X_test = X_test_raw.astype(np.float32)
X_test *= 1.0 / 255.0
X_test = X_test[..., None]

y_train = y_train.astype(np.float32)

print("Train slices:", X_train.shape, "Train labels:", y_train.shape)
print("Test slices:", X_test.shape, "Test ids:", testidt.shape)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3472179158.py in <cell line: 0>()
      1 X_train_raw, y_train, trainidt = get_all_data_for_train("T1wCE")
----> 2 X_test_raw, testidt = get_all_data_for_test("T1wCE")
      3 
      4 X_train = X_train_raw.astype(np.float32)
      5 X_train *= 1.0 / 255.0

/tmp/ipykernel_11/2476828713.py in get_all_data_for_test(image_type)
    112 
    113 def get_all_data_for_test(image_type):
--> 114     ids, counts, total = _count_slices(test_df, image_type, "test")
    115     X = np.empty((total, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    116     test_ids = np.empty((total,), dtype=np.int32)

/tmp/ipykernel_11/2476828713.py in _count_slices(df, image_type, folder)
     71 
     72 def _count_slices(df, image_type, folder):
---> 73     ids = df["BraTS21ID"].astype(int).to_numpy()
     74     counts = np.empty(ids.shape[0], dtype=np.int32)
     75     total = 0

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    131     if copy or arr.dtype == object or dtype == object:
    132         # Explicit copy, or required since NumPy can't view from / to object.
--> 133         return arr.astype(dtype, copy=True)
    134 
    135     return arr.astype(dtype, copy=copy)

ValueError: invalid literal for int() with base 10: 'test'

## === cell 4
import keras
from keras import layers

seed = 42
random.seed(seed)
np.random.seed(seed)

try:
    import tensorflow as tf  # optional backend control if present/working

    tf.random.set_seed(seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            min(8, os.cpu_count() or 2)
        )
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass
except Exception:
    tf = None




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # probability for ROC-AUC
    model = keras.Model(inputs, outputs)
    return model


model_best = build_model()

model_best.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

history = model_best.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=3,
    batch_size=32,
    verbose=1,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1114965329.py in <cell line: 0>()
     24 
     25 history = model_best.fit(
---> 26     X_train,
     27     y_train,
     28     validation_split=0.1,

NameError: name 'X_train' is not defined

## === cell 6
y_pred = model_best.predict(X_test, batch_size=64, verbose=1).reshape(-1)

pred_df = pd.DataFrame(
    {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred.astype(float)}
)
pred_df = pred_df.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

sample = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
sample["BraTS21ID_int"] = sample["BraTS21ID"].astype(int)

pred_df_map = pred_df.set_index("BraTS21ID")["MGMT_value"]
sample["MGMT_value"] = sample["BraTS21ID_int"].map(pred_df_map)

sample["MGMT_value"] = sample["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

submission = sample[["BraTS21ID", "MGMT_value"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3245217653.py in <cell line: 0>()
----> 1 y_pred = model_best.predict(X_test, batch_size=64, verbose=1).reshape(-1)
      2 
      3 pred_df = pd.DataFrame(
      4     {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred.astype(float)}
      5 )

NameError: name 'X_test' is not defined

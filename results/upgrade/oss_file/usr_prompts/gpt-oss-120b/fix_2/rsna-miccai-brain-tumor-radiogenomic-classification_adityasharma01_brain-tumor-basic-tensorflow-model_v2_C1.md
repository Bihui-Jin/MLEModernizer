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

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2  # image resizing
import pydicom

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = Path("kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/")

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = {"00109", "00123", "00709"}



## === cell 2
train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = test_df.copy()

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)].reset_index(drop=True)

print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3178142221.py in <cell line: 0>()
----> 1 train_df = pd.read_csv(data_dir / "train_labels.csv")
      2 test_df = pd.read_csv(data_dir / "sample_submission.csv")
      3 sample_submission = test_df.copy()
      4 
      5 # filter out the known bad training cases

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'

## === cell 3
def load_dicom(path, size=388):
    """
    Load a DICOM file, normalize to [0,255] uint8 and resize to (size, size).
    """
    dicom = pydicom.dcmread(path)  # newer API
    data = dicom.pixel_array.astype(np.float32)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.resize(data, (size, size))




## === cell 4
def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Return an array with the selected slice paths for a given patient and image type.
    """
    assert image_type in mri_types
    patient_path = os.path.join(data_dir / folder, str(brats21id).zfill(5))
    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]),
    )
    num_images = len(paths)
    start = int(num_images * 0.25)
    end = int(num_images * 0.75)
    interval = 3 if num_images >= 10 else 1
    return np.array(paths[start:end:interval])


def get_all_images(brats21id, image_type, folder="train", size=225):
    paths = get_all_image_paths(brats21id, image_type, folder)
    return [load_dicom(p, size) for p in paths]




## === cell 5
def get_all_data_for_train(image_type, image_size=32):
    X, y, ids = [], [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        images = get_all_images(row["BraTS21ID"], image_type, "train", image_size)
        if not images:
            continue
        X.extend(images)
        y.extend([row["MGMT_value"]] * len(images))
        ids.extend([row["BraTS21ID"]] * len(images))
    return np.array(X), np.array(y), np.array(ids)




## === cell 6
def get_all_data_for_test(image_type, image_size=32):
    X, ids = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        images = get_all_images(row["BraTS21ID"], image_type, "test", image_size)
        if not images:
            continue
        X.extend(images)
        ids.extend([row["BraTS21ID"]] * len(images))
    return np.array(X), np.array(ids)




## === cell 7
X, y, train_ids = get_all_data_for_train("T1wCE", image_size=64)
X_test, test_ids = get_all_data_for_test("T1wCE", image_size=64)

print("Training data shape:", X.shape, y.shape)
print("Test data shape:", X_test.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1154304647.py in <cell line: 0>()
      1 # Load a modest image size to keep memory usage reasonable
----> 2 X, y, train_ids = get_all_data_for_train("T1wCE", image_size=64)
      3 X_test, test_ids = get_all_data_for_test("T1wCE", image_size=64)
      4 
      5 print("Training data shape:", X.shape, y.shape)

/tmp/ipykernel_55/1345551420.py in get_all_data_for_train(image_type, image_size)
      1 def get_all_data_for_train(image_type, image_size=32):
      2     X, y, ids = [], [], []
----> 3     for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
      4         images = get_all_images(row["BraTS21ID"], image_type, "train", image_size)
      5         if not images:

NameError: name 'train_df' is not defined

## === cell 8
X_train, X_valid, y_train, y_valid, train_ids_train, train_ids_valid = train_test_split(
    X, y, train_ids, test_size=0.2, random_state=42, stratify=y
)

X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)

y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3085606097.py in <cell line: 0>()
      1 # Train/validation split
      2 X_train, X_valid, y_train, y_valid, train_ids_train, train_ids_valid = train_test_split(
----> 3     X, y, train_ids, test_size=0.2, random_state=42, stratify=y
      4 )
      5 

NameError: name 'X' is not defined

## === cell 9
def get_model02():
    tf.random.set_seed(12)
    inpt = keras.Input(shape=X_train.shape[1:])  # (H, W, 1)

    x = keras.layers.Rescaling(1.0 / 255)(inpt)

    x = keras.layers.Conv2D(32, kernel_size=3, activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = keras.layers.Conv2D(64, kernel_size=3, activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    output = keras.layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inpt, output)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 10
checkpoint_filepath = "best_model.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)



## === cell 11
model = get_model02()
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_valid, y_valid),
    callbacks=[model_checkpoint_callback],
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1409051499.py in <cell line: 0>()
----> 1 model = get_model02()
      2 history = model.fit(
      3     X_train,
      4     y_train,
      5     epochs=10,

/tmp/ipykernel_55/950204397.py in get_model02()
      2     # Simple 2‑D CNN suitable for the small grayscale patches
      3     tf.random.set_seed(12)
----> 4     inpt = keras.Input(shape=X_train.shape[1:])  # (H, W, 1)
      5 
      6     x = keras.layers.Rescaling(1.0 / 255)(inpt)

NameError: name 'X_train' is not defined

## === cell 12
model_best = tf.keras.models.load_model(checkpoint_filepath)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1235837018.py in <cell line: 0>()
----> 1 model_best = tf.keras.models.load_model(checkpoint_filepath)
      2 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'best_model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 13
y_pred_valid = model_best.predict(X_valid)
prob_valid = y_pred_valid[:, 1]

valid_df = pd.DataFrame({"BraTS21ID": train_ids_valid, "MGMT_value": prob_valid})
valid_score = valid_df.groupby("BraTS21ID", as_index=False).mean()
merged = valid_score.merge(train_df, on="BraTS21ID", suffixes=("_pred", "_true"))
auc = roc_auc_score(merged["MGMT_value_true"], merged["MGMT_value_pred"])
print(f"Validation AUC = {auc:.5f}")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2968461118.py in <cell line: 0>()
----> 1 y_pred_valid = model_best.predict(X_valid)
      2 # probability of the positive class (label 1)
      3 prob_valid = y_pred_valid[:, 1]
      4 
      5 # aggregate by patient ID (average of slice probabilities)

NameError: name 'model_best' is not defined

## === cell 14
y_pred_test = model_best.predict(tf.expand_dims(X_test, -1))
prob_test = y_pred_test[:, 1]

test_pred_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": prob_test})
submission = test_pred_df.groupby("BraTS21ID", as_index=False).mean()
submission = sample_submission[["BraTS21ID"]].merge(
    submission, on="BraTS21ID", how="left"
)
submission["MGMT_value"].fillna(submission["MGMT_value"].mean(), inplace=True)

submission.to_csv("submission.csv", index=False)
print("submission.csv written:", submission.shape)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4253520145.py in <cell line: 0>()
----> 1 y_pred_test = model_best.predict(tf.expand_dims(X_test, -1))
      2 prob_test = y_pred_test[:, 1]
      3 
      4 test_pred_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": prob_test})
      5 # Average probabilities per patient

NameError: name 'model_best' is not defined

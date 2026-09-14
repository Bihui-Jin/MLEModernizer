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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9961

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from zipfile import ZipFile
import os
import shutil

WORK_DIR = "/kaggle/working"
INPUT_DIR = "/kaggle/input/aerial-cactus-identification"

os.makedirs(WORK_DIR, exist_ok=True)


def _safe_rmtree(path: str):
    if not os.path.exists(path):
        return
    try:
        shutil.rmtree(path)
    except OSError:
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
                for d in dirs:
                    dp = os.path.join(root, d)
                    try:
                        os.rmdir(dp)
                    except OSError:
                        pass
            try:
                os.rmdir(path)
            except OSError:
                pass
        except Exception:
            pass


for d in ["train", "test", "aerial-cactus-identification"]:
    _safe_rmtree(os.path.join(WORK_DIR, d))

test_zip = os.path.join(INPUT_DIR, "test.zip")
train_zip = os.path.join(INPUT_DIR, "train.zip")

if os.path.isfile(test_zip):
    with ZipFile(test_zip, "r") as z:
        z.extractall(WORK_DIR)
if os.path.isfile(train_zip):
    with ZipFile(train_zip, "r") as z:
        z.extractall(WORK_DIR)

nested_base = os.path.join(WORK_DIR, "aerial-cactus-identification")
if os.path.isdir(nested_base):
    for sub in ["train", "test"]:
        src = os.path.join(nested_base, sub)
        dst = os.path.join(WORK_DIR, sub)
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(src):
                s = os.path.join(src, fn)
                t = os.path.join(dst, fn)
                if not os.path.exists(t):
                    try:
                        shutil.copy2(s, t)
                    except Exception:
                        pass

print("Working dir contents:", os.listdir(WORK_DIR))
print(
    "Train images:",
    (
        len(os.listdir(os.path.join(WORK_DIR, "train")))
        if os.path.isdir(os.path.join(WORK_DIR, "train"))
        else 0
    ),
)
print(
    "Test images:",
    (
        len(os.listdir(os.path.join(WORK_DIR, "test")))
        if os.path.isdir(os.path.join(WORK_DIR, "test"))
        else 0
    ),
)

for dirname, _, filenames in os.walk(INPUT_DIR):
    for filename in filenames:
        if filename.endswith((".csv", ".zip")):
            print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import optimizers, models, layers

trainDir = "/kaggle/working/train"
testDir = "/kaggle/working/test"


def _resolve_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


trainCsvDir = _resolve_existing_path(
    [
        "/kaggle/input/aerial-cactus-identification/train.csv",
        "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification/train.csv",
        "/kaggle/data/aerial-cactus-identification/train.csv",
        "/kaggle/data/train.csv",
    ]
)
sampleSubPath = _resolve_existing_path(
    [
        "/kaggle/input/aerial-cactus-identification/sample_submission.csv",
        "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification/sample_submission.csv",
        "/kaggle/data/aerial-cactus-identification/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

assert os.path.isdir(trainDir), f"Missing train dir: {trainDir}"
assert os.path.isdir(testDir), f"Missing test dir: {testDir}"
assert trainCsvDir is not None and os.path.isfile(
    trainCsvDir
), f"Missing train.csv (checked common paths)"
assert sampleSubPath is not None and os.path.isfile(
    sampleSubPath
), f"Missing sample_submission.csv (checked common paths)"

print("Using train.csv:", trainCsvDir)
print("Using sample_submission.csv:", sampleSubPath)

trainDataFrame = pd.read_csv(trainCsvDir)
trainDataFrame.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (32, 32)


def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_ds(df, directory, batch_size, shuffle, repeat=False):
    paths = tf.constant([os.path.join(directory, fname) for fname in df["id"].tolist()])
    labels = tf.constant(df["has_cactus"].astype(np.float32).values)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    def _map_fn(p, y):
        return _load_image(p), tf.reshape(y, (1,))

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


split_idx = min(15000, len(trainDataFrame))
train_df = trainDataFrame.iloc[:split_idx].copy()
val_df = trainDataFrame.iloc[split_idx:].copy()

if len(val_df) == 0:
    val_df = trainDataFrame.iloc[-500:].copy()
    train_df = trainDataFrame.iloc[:-500].copy()

batch_train = 150
batch_val = 50

train_ds = make_ds(
    train_df, trainDir, batch_size=batch_train, shuffle=True, repeat=True
)
val_ds = make_ds(val_df, trainDir, batch_size=batch_val, shuffle=False, repeat=True)

steps_per_epoch = 100
validation_steps = 50
epochs = 8

print("Train rows:", len(train_df), "Val rows:", len(val_df))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/365117162.py in <cell line: 0>()
     32 
     33 
---> 34 split_idx = min(15000, len(trainDataFrame))
     35 train_df = trainDataFrame.iloc[:split_idx].copy()
     36 val_df = trainDataFrame.iloc[split_idx:].copy()

NameError: name 'trainDataFrame' is not defined

## === cell 3
model = models.Sequential()
model.add(
    layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same"
    )
)
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()



## === cell 4
model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(), metrics=["acc"])



## === cell 5
history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_ds,
    validation_steps=validation_steps,
    verbose=2,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3633801559.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_ds,
      3     steps_per_epoch=steps_per_epoch,
      4     epochs=epochs,
      5     validation_data=val_ds,

NameError: name 'train_ds' is not defined

## === cell 6
epochs_ = range(0, epochs)
acc_key = "acc" if "acc" in history.history else "accuracy"
val_acc_key = "val_acc" if "val_acc" in history.history else "val_accuracy"

acc = history.history[acc_key]
acc_val = history.history[val_acc_key]

plt.plot(epochs_, acc, label="training accuracy")
plt.scatter(epochs_, acc_val, label="validation accuracy")
plt.xlabel("no of epochs")
plt.ylabel("accuracy")
plt.title("no of epochs vs accuracy")
plt.legend()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1523037647.py in <cell line: 0>()
----> 1 epochs_ = range(0, epochs)
      2 acc_key = "acc" if "acc" in history.history else "accuracy"
      3 val_acc_key = "val_acc" if "val_acc" in history.history else "val_accuracy"
      4 
      5 acc = history.history[acc_key]

NameError: name 'epochs' is not defined

## === cell 7
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.plot(epochs_, loss, label="training loss")
plt.scatter(epochs_, val_loss, label="validation loss")
plt.xlabel("No of epochs")
plt.ylabel("loss")
plt.title("no of epochs vs loss")
plt.legend()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2137891448.py in <cell line: 0>()
----> 1 loss = history.history["loss"]
      2 val_loss = history.history["val_loss"]
      3 
      4 plt.plot(epochs_, loss, label="training loss")
      5 plt.scatter(epochs_, val_loss, label="validation loss")

NameError: name 'history' is not defined

## === cell 8
sample_sub = pd.read_csv(sampleSubPath)


def make_test_ds(id_list, directory, batch_size):
    paths = tf.constant([os.path.join(directory, fname) for fname in id_list])
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(p):
        return _load_image(p)

    ds = (
        ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return ds


missing = []
for _id in sample_sub["id"].iloc[:20].tolist():
    if not os.path.isfile(os.path.join(testDir, _id)):
        missing.append(_id)

if missing:
    alt_testDir = "/kaggle/working/aerial-cactus-identification/test"
    if os.path.isdir(alt_testDir):
        testDir = alt_testDir

for _id in sample_sub["id"].iloc[:5].tolist():
    p = os.path.join(testDir, _id)
    assert os.path.isfile(p), f"Test image not found at expected path: {p}"

test_ds = make_test_ds(sample_sub["id"].tolist(), testDir, batch_size=50)

predictions = model.predict(test_ds, verbose=0).reshape(-1)

submission = pd.DataFrame(
    {"id": sample_sub["id"].values, "has_cactus": predictions.astype(np.float32)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2412207692.py in <cell line: 0>()
----> 1 sample_sub = pd.read_csv(sampleSubPath)
      2 
      3 
      4 def make_test_ds(id_list, directory, batch_size):
      5     paths = tf.constant([os.path.join(directory, fname) for fname in id_list])

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
    726 
    727     # open URLs
--> 728     ioargs = _get_filepath_or_buffer(
    729         path_or_buf,
    730         encoding=encoding,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in _get_filepath_or_buffer(filepath_or_buffer, encoding, compression, mode, storage_options)
    470     ):
    471         msg = f"Invalid file path or buffer object type: {type(filepath_or_buffer)}"
--> 472         raise ValueError(msg)
    473 
    474     return IOArgs(

ValueError: Invalid file path or buffer object type: <class 'NoneType'>

## === cell 9
print("Submission file exists:", os.path.isfile("submission.csv"))
print("Working dir final contents:", os.listdir("/kaggle/working"))

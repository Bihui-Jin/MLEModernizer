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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8115744938047749

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.56614) has done: 'I fix the TensorFlow/protobuf import crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. Then I remove the dependency on the missing external `.h5` weight file (which is why `my_model` is undefined) by training the same kind of Keras image classifier end-to-end using the provided `train.csv` + `train_images`, and then predicting on `test_images`. Finally, I ensure the submission is written exactly as `submission.csv` with the required `image_id,label` columns and correct row alignment with `sample_submission.csv`. These changes are necessary to unblock execution and produce a valid submission; they should also yield a reasonable accuracy (though exact score depend on training time/compute).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/403585703.py in <cell line: 0>()
      9 import numpy as np
     10 import pandas as pd
---> 11 import tensorflow as tf
     12 
     13 from tensorflow.keras.preprocessing.image import ImageDataGenerator

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
DATA_CANDIDATES = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]

DATA_ROOT = None
for p in DATA_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations: "
        + ", ".join(DATA_CANDIDATES)
    )

TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

for req in [TRAIN_CSV_PATH, SAMPLE_SUB_PATH, TRAIN_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(req):
        raise FileNotFoundError(f"Missing required path: {req}")

print("DATA_ROOT:", DATA_ROOT)
print("Train images:", len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))))
print("Test images:", len(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))))



## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df["path"] = train_df["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

missing_train = train_df.loc[
    ~train_df["path"].apply(os.path.exists), "image_id"
].tolist()
if missing_train:
    raise FileNotFoundError(
        f"Some train images are missing. Example: {missing_train[:5]}"
    )

y = train_df["label"].astype(int).values
num_classes = int(train_df["label"].nunique())
if num_classes != 5:
    print("Warning: expected 5 classes, got:", num_classes)

rng = np.random.RandomState(SEED)
val_mask = np.zeros(len(train_df), dtype=bool)
val_frac = 0.15

for c in np.unique(y):
    idx = np.where(y == c)[0]
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    val_mask[idx[:n_val]] = True

df_val = train_df[val_mask].copy().reset_index(drop=True)
df_train = train_df[~val_mask].copy().reset_index(drop=True)

print("Train size:", len(df_train), "Val size:", len(df_val))
print("Train label counts:\n", df_train["label"].value_counts().sort_index())
print("Val label counts:\n", df_val["label"].value_counts().sort_index())



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/615702147.py in <cell line: 0>()
     15     print("Warning: expected 5 classes, got:", num_classes)
     16 
---> 17 rng = np.random.RandomState(SEED)
     18 val_mask = np.zeros(len(train_df), dtype=bool)
     19 val_frac = 0.15

NameError: name 'SEED' is not defined

## === cell 3
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # conservative to avoid OOM at 512x512

df_train_gen = df_train.copy()
df_val_gen = df_val.copy()
df_train_gen["label"] = df_train_gen["label"].astype(str)
df_val_gen["label"] = df_val_gen["label"].astype(str)

train_idg = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.1,
    horizontal_flip=True,
)

val_idg = ImageDataGenerator(rescale=1.0 / 255.0)

classes = [str(i) for i in range(5)]

train_gen = train_idg.flow_from_dataframe(
    dataframe=df_train_gen,
    x_col="path",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode="categorical",
    classes=classes,
)

val_gen = val_idg.flow_from_dataframe(
    dataframe=df_val_gen,
    x_col="path",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False,
    class_mode="categorical",
    classes=classes,
)

print("Class indices (generator):", train_gen.class_indices)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2804514415.py in <cell line: 0>()
      4 # Bugfix: Keras legacy ImageDataGenerator with class_mode="categorical"
      5 # requires y_col values to be string/list/tuple (not int). Keep semantics identical.
----> 6 df_train_gen = df_train.copy()
      7 df_val_gen = df_val.copy()
      8 df_train_gen["label"] = df_train_gen["label"].astype(str)

NameError: name 'df_train' is not defined

## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

my_model = tf.keras.Model(inputs=inputs, outputs=outputs)
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3044352351.py in <cell line: 0>()
----> 1 inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
      2 x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
      3 x = tf.keras.layers.MaxPooling2D()(x)
      4 x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
      5 x = tf.keras.layers.MaxPooling2D()(x)

NameError: name 'tf' is not defined

## === cell 5
EPOCHS = 3 if not DEBUG else 1

steps_per_epoch = max(1, len(train_gen))
validation_steps = max(1, len(val_gen))

history = my_model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    verbose=1,
)

val_metrics = my_model.evaluate(val_gen, verbose=0)
print("Validation metrics:", dict(zip(my_model.metrics_names, val_metrics)))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3885900858.py in <cell line: 0>()
----> 1 EPOCHS = 3 if not DEBUG else 1
      2 
      3 steps_per_epoch = max(1, len(train_gen))
      4 validation_steps = max(1, len(val_gen))
      5 

NameError: name 'DEBUG' is not defined

## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["path"] = sample_sub["image_id"].apply(
    lambda x: os.path.join(TEST_IMG_DIR, x)
)

missing = sample_sub.loc[~sample_sub["path"].apply(os.path.exists), "image_id"].tolist()
if len(missing) > 0:
    raise FileNotFoundError(
        f"Some test images listed in sample_submission are missing. Example: {missing[:5]}"
    )

df_test = sample_sub[["path"]].copy()


def make_test_gen(batch_size=64):
    my_test_idg = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=IMG_SIZE,
    )
    return test_gen




## === cell 7
pred_list = []
for _ in range(1):
    test_gen = make_test_gen(batch_size=64)
    pred_test = my_model.predict(test_gen, verbose=1)
    pred_list.append(pred_test)

pred_test = np.mean(pred_list, axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = sample_sub[["image_id"]].copy()
final_csv["label"] = pred_test_labels.astype(int)

if len(final_csv) != len(sample_sub):
    raise RuntimeError("Submission length mismatch with sample_submission.csv")

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1705210375.py in <cell line: 0>()
      1 pred_list = []
      2 for _ in range(1):
----> 3     test_gen = make_test_gen(batch_size=64)
      4     pred_test = my_model.predict(test_gen, verbose=1)
      5     pred_list.append(pred_test)

/tmp/ipykernel_11/466808797.py in make_test_gen(batch_size)
     14 
     15 def make_test_gen(batch_size=64):
---> 16     my_test_idg = ImageDataGenerator(rescale=1.0 / 255.0)
     17     test_gen = my_test_idg.flow_from_dataframe(
     18         dataframe=df_test,

NameError: name 'ImageDataGenerator' is not defined

## === cell 8
_check = pd.read_csv("submission.csv")
print(_check.head())
print(_check.dtypes)
assert list(_check.columns) == ["image_id", "label"]
assert _check["label"].dtype in [np.int64, np.int32, int]
assert len(_check) == len(pd.read_csv(SAMPLE_SUB_PATH))
print("submission.csv OK")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1716022173.py in <cell line: 0>()
----> 1 _check = pd.read_csv("submission.csv")
      2 print(_check.head())
      3 print(_check.dtypes)
      4 assert list(_check.columns) == ["image_id", "label"]
      5 assert _check["label"].dtype in [np.int64, np.int32, int]

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

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

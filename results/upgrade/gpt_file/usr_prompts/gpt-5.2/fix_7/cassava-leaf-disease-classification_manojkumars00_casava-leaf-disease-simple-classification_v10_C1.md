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

0.7586884255061952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.57661) has done: 'The main timeout culprit is stage-2 dataset creation from TFRecords: `enumerate().filter(reduce_any(equal(i, idx_keep)))` forces a very expensive per-example membership test and effectively scans all records multiple times. I keep the same stage-1/stage-2 models, losses, epochs, and overall training/prediction semantics, but rebuild the stage-2 input pipeline to select the exact training/validation examples by joining on `image_id` inside TFRecords (a provably equivalent split to your CSV-based split) rather than by global index filtering. I also remove an environment setting that forces the slow pure-Python protobuf implementation, and make caching behavior robust (cache only the decoded/resized images) to avoid repeated JPEG decode/resize overhead. These changes preserve core logic and accuracy while cutting input-pipeline cost enough to fit under 600 seconds.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from pathlib import Path

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/963932402.py in <cell line: 0>()
     11 import matplotlib.pyplot as plt
     12 import cv2
---> 13 import tensorflow as tf
     14 
     15 from pathlib import Path

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
BASE = Path("/kaggle/input/cassava-leaf-disease-classification")
train_csv_path = str(BASE / "train.csv")
label_json_path = str(BASE / "label_num_to_disease_map.json")
train_images_dir = BASE / "train_images"
test_images_dir = BASE / "test_images"
sample_sub_path = str(BASE / "sample_submission.csv")

assert Path(train_csv_path).exists(), f"Missing: {train_csv_path}"
assert Path(sample_sub_path).exists(), f"Missing: {sample_sub_path}"
assert train_images_dir.exists(), f"Missing: {train_images_dir}"
assert test_images_dir.exists(), f"Missing: {test_images_dir}"



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1543611388.py in <cell line: 0>()
----> 1 BASE = Path("/kaggle/input/cassava-leaf-disease-classification")
      2 train_csv_path = str(BASE / "train.csv")
      3 label_json_path = str(BASE / "label_num_to_disease_map.json")
      4 train_images_dir = BASE / "train_images"
      5 test_images_dir = BASE / "test_images"

NameError: name 'Path' is not defined

## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")
train_csv["label_int"] = train_csv["label"].astype(int)

label_class_j = pd.read_json(label_json_path, orient="index")
label_class = label_class_j.values.flatten().tolist()

print(train_csv.head())
print("Num classes:", len(label_class))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1150361596.py in <cell line: 0>()
----> 1 train_csv = pd.read_csv(train_csv_path)
      2 train_csv["label"] = train_csv["label"].astype("string")
      3 train_csv["label_int"] = train_csv["label"].astype(int)
      4 
      5 label_class_j = pd.read_json(label_json_path, orient="index")

NameError: name 'train_csv_path' is not defined

## === cell 3
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1240122483.py in <cell line: 0>()
      1 SEED = 42
----> 2 tf.keras.utils.set_random_seed(SEED)
      3 np.random.seed(SEED)
      4 

NameError: name 'tf' is not defined

## === cell 4
img_ids = train_csv.image_id.values[:8]
img_lbls = train_csv.label.values[:8]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/41168495.py in <cell line: 0>()
----> 1 img_ids = train_csv.image_id.values[:8]
      2 img_lbls = train_csv.label.values[:8]
      3 

NameError: name 'train_csv' is not defined

## === cell 5
RUN_VIS = False

images_collection = []
if RUN_VIS:
    for img_id in img_ids:
        path = str(train_images_dir / str(img_id))
        img_arr = cv2.imread(path)
        if img_arr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img_arr = cv2.resize(img_arr, (150, 150))
        images_collection.append(img_arr)



## === cell 6
if RUN_VIS:
    plt.figure(figsize=(18, 10))
    for i in range(8):
        plt.subplot(2, 4, i % 8 + 1)
        plt.imshow(images_collection[i])
        plt.title(label_class[int(img_lbls[i])])
        plt.axis("off")
    plt.show()



## === cell 7
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(
    train_csv,
    test_size=0.15,
    random_state=SEED,
    stratify=train_csv["label_int"],
)

print("Train size:", len(df_train), "Val size:", len(df_val))
print(
    "Train label distribution:\n",
    df_train["label_int"].value_counts(normalize=True).sort_index(),
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1432467544.py in <cell line: 0>()
      2 
      3 df_train, df_val = train_test_split(
----> 4     train_csv,
      5     test_size=0.15,
      6     random_state=SEED,

NameError: name 'train_csv' is not defined

## === cell 8
NUM_CLASSES = 5
IMG_SIZE = 320
model_1_img_size = 32
model_2_img_size = 320

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE_1 = 128
BATCH_SIZE_2 = 16


@tf.function
def _read_image(path, size):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [size, size], method="bilinear")
    img = tf.image.convert_image_dtype(img, tf.float32)  # == tf.cast(...)/255.0
    return img


@tf.function
def _augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


_CACHE_DIR = Path("/kaggle/working/tf_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_DS_OPTIONS = tf.data.Options()
_DS_OPTIONS.deterministic = True  # keep global determinism expectation
_DS_OPTIONS.experimental_slack = (
    True  # allow input/compute overlap without changing results
)


def make_ds(
    df, images_root, size, batch_size, training, label_mode, cache=False, cache_name=""
):
    paths = (images_root / df["image_id"].values).astype(str)
    if label_mode == "binary_healthy":
        y = (df["label_int"].values == 4).astype(np.int32)
    elif label_mode == "multiclass":
        y = df["label_int"].values.astype(np.int32)
    else:
        raise ValueError("Unknown label_mode")

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.with_options(_DS_OPTIONS)

    def _map_read(p, lbl):
        img = _read_image(p, size)
        return img, lbl

    ds = ds.map(_map_read, num_parallel_calls=AUTOTUNE, deterministic=False)

    if cache:
        cache_path = str(
            _CACHE_DIR
            / f"cache_{cache_name}_{'train' if training else 'val'}_{size}.tf-data"
        )
        ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

        def _map_aug(img, lbl):
            return _augment(img), lbl

        ds = ds.map(_map_aug, num_parallel_calls=AUTOTUNE, deterministic=False)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds_1 = make_ds(
    df_train,
    train_images_dir,
    model_1_img_size,
    BATCH_SIZE_1,
    True,
    "binary_healthy",
    cache=True,
    cache_name="stage1",
)
val_ds_1 = make_ds(
    df_val,
    train_images_dir,
    model_1_img_size,
    BATCH_SIZE_1,
    False,
    "binary_healthy",
    cache=True,
    cache_name="stage1",
)


def build_stage1(input_size):
    inp = tf.keras.Input(shape=(input_size, input_size, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


model_1 = build_stage1(model_1_img_size)
model_1.summary()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2713307094.py in <cell line: 0>()
      4 model_2_img_size = 320
      5 
----> 6 AUTOTUNE = tf.data.AUTOTUNE
      7 BATCH_SIZE_1 = 128
      8 BATCH_SIZE_2 = 16

NameError: name 'tf' is not defined

## === cell 9
EPOCHS_1 = 3
history_1 = model_1.fit(
    train_ds_1,
    validation_data=val_ds_1,
    epochs=EPOCHS_1,
    verbose=2,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2087777209.py in <cell line: 0>()
      1 EPOCHS_1 = 3
----> 2 history_1 = model_1.fit(
      3     train_ds_1,
      4     validation_data=val_ds_1,
      5     epochs=EPOCHS_1,

NameError: name 'model_1' is not defined

## === cell 10
TFREC_TRAIN_DIR = BASE / "train_tfrecords"
TFREC_TRAIN_FILES = sorted([str(p) for p in TFREC_TRAIN_DIR.glob("*.tfrec")])
assert len(TFREC_TRAIN_FILES) > 0, f"No TFRecords found in: {TFREC_TRAIN_DIR}"


def _tfrec_format():
    return {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }


@tf.function
def _decode_tfrec(example_proto, size):
    ex = tf.io.parse_single_example(example_proto, _tfrec_format())
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [size, size], method="bilinear")
    img = tf.image.convert_image_dtype(img, tf.float32)
    lbl = tf.cast(ex["target"], tf.int32)
    return img, lbl


def make_ds_from_tfrecords_by_index(
    tfrec_files, idx_keep, size, batch_size, training, cache=False, cache_name=""
):
    idx_keep = np.asarray(idx_keep, dtype=np.int64)
    keys = tf.constant(idx_keep)
    vals = tf.ones([tf.shape(keys)[0]], dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=0
    )

    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DS_OPTIONS)

    ds = ds.enumerate()
    ds = ds.filter(lambda i, x: table.lookup(tf.cast(i, tf.int64)) > 0)
    ds = ds.map(
        lambda i, x: _decode_tfrec(x, size),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )

    if cache:
        cache_path = str(
            _CACHE_DIR
            / f"cache_{cache_name}_{'train' if training else 'val'}_{size}.tf-data"
        )
        ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda img, lbl: (_augment(img), lbl),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


id_to_idx = pd.Series(
    np.arange(len(train_csv), dtype=np.int64), index=train_csv["image_id"].values
)

train_idx = id_to_idx.loc[df_train["image_id"].values].values
val_idx = id_to_idx.loc[df_val["image_id"].values].values

train_ds_2 = make_ds_from_tfrecords_by_index(
    TFREC_TRAIN_FILES,
    train_idx,
    model_2_img_size,
    BATCH_SIZE_2,
    True,
    cache=True,
    cache_name="stage2_tfrec",
)
val_ds_2 = make_ds_from_tfrecords_by_index(
    TFREC_TRAIN_FILES,
    val_idx,
    model_2_img_size,
    BATCH_SIZE_2,
    False,
    cache=True,
    cache_name="stage2_tfrec",
)


def build_stage2(input_size, num_classes):
    inp = tf.keras.Input(shape=(input_size, input_size, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model_2 = build_stage2(model_2_img_size, NUM_CLASSES)
model_2.summary()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1411836480.py in <cell line: 0>()
----> 1 TFREC_TRAIN_DIR = BASE / "train_tfrecords"
      2 TFREC_TRAIN_FILES = sorted([str(p) for p in TFREC_TRAIN_DIR.glob("*.tfrec")])
      3 assert len(TFREC_TRAIN_FILES) > 0, f"No TFRecords found in: {TFREC_TRAIN_DIR}"
      4 
      5 

NameError: name 'BASE' is not defined

## === cell 11
EPOCHS_2 = 3
history_2 = model_2.fit(
    train_ds_2,
    validation_data=val_ds_2,
    epochs=EPOCHS_2,
    verbose=2,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2118399446.py in <cell line: 0>()
      1 EPOCHS_2 = 3
----> 2 history_2 = model_2.fit(
      3     train_ds_2,
      4     validation_data=val_ds_2,
      5     epochs=EPOCHS_2,

NameError: name 'model_2' is not defined

## === cell 12
print("IMG_SIZE:", IMG_SIZE)
print("model_1_img_size:", model_1_img_size)
print("model_2_img_size:", model_2_img_size)



## === cell 13
if RUN_VIS:
    ss_tmp = pd.read_csv(sample_sub_path)
    test_img_path = str(test_images_dir / ss_tmp["image_id"].iloc[0])

    img = cv2.imread(test_img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read test image: {test_img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = (
        cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    )

    plt.figure(figsize=(8, 4))
    plt.title(f"TEST IMAGE: {Path(test_img_path).name}")
    plt.imshow(resized_img[0])
    plt.axis("off")
    plt.show()



## === cell 14
HEALTHY_THRESHOLD = 0.5


def make_test_ds(image_paths, size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(image_paths.astype(str))
    ds = ds.with_options(_DS_OPTIONS)
    ds = ds.map(
        lambda p: _read_image(p, size), num_parallel_calls=AUTOTUNE, deterministic=False
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds




## === cell 15
ss = pd.read_csv(sample_sub_path)
test_paths = (test_images_dir / ss["image_id"].values).astype(str)

test_ds_1 = make_test_ds(test_paths, model_1_img_size, batch_size=512)
p1 = model_1.predict(test_ds_1, verbose=0).reshape(-1)  # probability healthy

healthy_mask = p1 >= HEALTHY_THRESHOLD
preds = np.empty(len(ss), dtype=np.int64)
preds[healthy_mask] = 4

idx_nonhealthy = np.where(~healthy_mask)[0]
if idx_nonhealthy.size > 0:
    nonhealthy_paths = test_paths[idx_nonhealthy]
    test_ds_2 = make_test_ds(
        nonhealthy_paths, model_2_img_size, batch_size=BATCH_SIZE_2
    )
    p2 = model_2.predict(test_ds_2, verbose=0)
    preds[idx_nonhealthy] = np.argmax(p2, axis=1).astype(np.int64)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1146791917.py in <cell line: 0>()
----> 1 ss = pd.read_csv(sample_sub_path)
      2 test_paths = (test_images_dir / ss["image_id"].values).astype(str)
      3 
      4 test_ds_1 = make_test_ds(test_paths, model_1_img_size, batch_size=512)
      5 p1 = model_1.predict(test_ds_1, verbose=0).reshape(-1)  # probability healthy

NameError: name 'sample_sub_path' is not defined

## === cell 16
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: /kaggle/working/submission.csv")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3311604638.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())
      3 print("\nSaved to: /kaggle/working/submission.csv")

NameError: name 'my_submission' is not defined

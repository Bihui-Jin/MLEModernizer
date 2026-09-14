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

0.6639468117255969

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    ncpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(ncpu)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"

print("TF version:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2223860471.py in <cell line: 0>()
     12 import numpy as np
     13 import pandas as pd
---> 14 import tensorflow as tf
     15 
     16 from tensorflow.keras.models import Model

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
from sklearn.model_selection import train_test_split

IMG_SIZE = (300, 300)
BATCH_SIZE = 16 if not DEBUG else 8
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)

train_df["path"] = (TRAIN_IMG_DIR + "/") + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(str)  # keep identical label semantics

if DEBUG:
    train_df = train_df.sample(2000, random_state=SEED).reset_index(drop=True)

tr_df, va_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

AUTOTUNE = tf.data.AUTOTUNE

IMG_H = tf.constant(IMG_SIZE[0], dtype=tf.int32)
IMG_W = tf.constant(IMG_SIZE[1], dtype=tf.int32)
INV_255 = tf.constant(1.0 / 255.0, dtype=tf.float32)
PI_OVER_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)
SEED_T = tf.constant(SEED, dtype=tf.int32)


@tf.function(jit_compile=False)
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (IMG_H, IMG_W), method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * INV_255
    return img


@tf.function(jit_compile=False)
def _one_hot_from_string_label(label_str):
    label_int = tf.strings.to_number(label_str, out_type=tf.int32)
    return tf.one_hot(label_int, depth=NUM_CLASSES, dtype=tf.float32)


@tf.function(jit_compile=False)
def _augment(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)

    angle = (
        tf.random.stateless_uniform(
            [], seed=seed2 + tf.constant([1, 0], tf.int32), minval=-10.0, maxval=10.0
        )
        * PI_OVER_180
    )
    if hasattr(tf.image, "rotate"):
        img = tf.image.rotate(img, angles=angle, fill_mode="nearest")

    dx = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([2, 0], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_W, tf.float32)
    dy = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([3, 0], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_H, tf.float32)

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(
            tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0]), 0
        ),
        output_shape=tf.stack([IMG_H, IMG_W]),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]

    zoom = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([4, 0], tf.int32), minval=0.9, maxval=1.1
    )
    new_h = tf.cast(tf.round(tf.cast(IMG_H, tf.float32) * zoom), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_W, tf.float32) * zoom), tf.int32)
    img2 = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, IMG_H, IMG_W)
    img = img2

    return img


def make_train_ds(df, batch_size, training: bool, cache_name: str):
    paths = df["path"].values.astype(str)
    labels = df["label"].values.astype(str)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_slack = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    ds = ds.with_options(options)

    @tf.function(jit_compile=False)
    def _decode_only(path, label_str):
        return _decode_resize(path), label_str, path

    ds = ds.map(_decode_only, num_parallel_calls=AUTOTUNE, deterministic=True)

    @tf.function(jit_compile=False)
    def _map_after_decode(img, label_str, path_for_seed):
        h = tf.strings.to_hash_bucket_fast(path_for_seed, 2**31 - 1)
        seed2 = tf.stack([SEED_T, tf.cast(h, tf.int32)], axis=0)
        img = _augment(img, seed2)
        y = _one_hot_from_string_label(label_str)
        return img, y

    @tf.function(jit_compile=False)
    def _map_no_aug(img, label_str, path_for_seed):
        y = _one_hot_from_string_label(label_str)
        return img, y

    if training:
        ds = ds.map(_map_after_decode, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        ds = ds.map(_map_no_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(
    tr_df, BATCH_SIZE, training=True, cache_name="train_decode_cache"
)
valid_ds = make_train_ds(
    va_df, BATCH_SIZE, training=False, cache_name="valid_decode_cache"
)

inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=inp, outputs=out)

base.trainable = False
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_HEAD = 3 if not DEBUG else 1
my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_HEAD,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_FT = 2 if not DEBUG else 1
my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_FT,
    verbose=1,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2417191934.py in <cell line: 0>()
      2 
      3 IMG_SIZE = (300, 300)
----> 4 BATCH_SIZE = 16 if not DEBUG else 8
      5 NUM_CLASSES = 5
      6 

NameError: name 'DEBUG' is not defined

## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB)

sample_sub["path"] = (TEST_IMG_DIR + "/") + sample_sub["image_id"].astype(str)
df_test = sample_sub[["image_id", "path"]].copy()


def make_test_ds(batch_size=64):
    paths = df_test["path"].values.astype(str)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_slack = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/685618676.py in <cell line: 0>()
----> 1 sample_sub = pd.read_csv(SAMPLE_SUB)
      2 
      3 sample_sub["path"] = (TEST_IMG_DIR + "/") + sample_sub["image_id"].astype(str)
      4 df_test = sample_sub[["image_id", "path"]].copy()
      5 

NameError: name 'SAMPLE_SUB' is not defined

## === cell 3
test_ds = make_test_ds(batch_size=32 if not DEBUG else 16)

pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = pd.DataFrame(
    {
        "image_id": df_test["image_id"].values,
        "label": pred_test_labels,
    }
)

assert (
    final_csv.shape[0] == sample_sub.shape[0]
), "Submission row count must match sample_submission."
assert list(final_csv.columns) == ["image_id", "label"]

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2136937158.py in <cell line: 0>()
----> 1 test_ds = make_test_ds(batch_size=32 if not DEBUG else 16)
      2 
      3 pred_test = my_model.predict(test_ds, verbose=1)
      4 pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)
      5 

NameError: name 'make_test_ds' is not defined

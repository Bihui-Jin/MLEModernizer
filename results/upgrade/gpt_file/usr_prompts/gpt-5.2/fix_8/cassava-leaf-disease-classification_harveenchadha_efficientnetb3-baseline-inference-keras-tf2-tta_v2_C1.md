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

0.7892112420670897

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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2587979880.py in <cell line: 0>()
     11 import numpy as np
     12 import pandas as pd
---> 13 import tensorflow as tf
     14 
     15 from tensorflow.keras.models import Model, load_model

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
BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"

train_csv_path = os.path.join(BASE_INPUT, "train.csv")
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")

train_img_dir = os.path.join(BASE_INPUT, "train_images")
test_img_dir = os.path.join(BASE_INPUT, "test_images")

train_tfrec_dir = os.path.join(BASE_INPUT, "train_tfrecords")
test_tfrec_dir = os.path.join(BASE_INPUT, "test_tfrecords")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_img_dir), f"Missing dir: {train_img_dir}"
assert os.path.isdir(test_img_dir), f"Missing dir: {test_img_dir}"
assert os.path.isdir(train_tfrec_dir), f"Missing dir: {train_tfrec_dir}"
assert os.path.isdir(test_tfrec_dir), f"Missing dir: {test_tfrec_dir}"

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

assert {"image_id", "label"}.issubset(train_df.columns)
assert {"image_id", "label"}.issubset(sample_sub.columns)

train_df["path"] = train_img_dir + "/" + train_df["image_id"].astype(str)

num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

if DEBUG:
    train_df = train_df.sample(1024, random_state=SEED).reset_index(drop=True)

train_df.head()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/500696207.py in <cell line: 0>()
     28 assert num_classes == 5, f"Expected 5 classes, got {num_classes}"
     29 
---> 30 if DEBUG:
     31     train_df = train_df.sample(1024, random_state=SEED).reset_index(drop=True)
     32 

NameError: name 'DEBUG' is not defined

## === cell 2
idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.15
val_size = int(len(idx) * val_frac)

val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

trn_df.shape, val_df.shape




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4174200429.py in <cell line: 0>()
      1 idx = np.arange(len(train_df))
----> 2 rng = np.random.RandomState(SEED)
      3 rng.shuffle(idx)
      4 
      5 val_frac = 0.15

NameError: name 'SEED' is not defined

## === cell 3
IMG_SIZE = (300, 300)
BATCH_SIZE = 32 if not DEBUG else 16
AUTO = tf.data.AUTOTUNE

train_tfrec_files = sorted(glob.glob(os.path.join(train_tfrec_dir, "*.tfrec")))
test_tfrec_files = sorted(glob.glob(os.path.join(test_tfrec_dir, "*.tfrec")))
assert len(train_tfrec_files) > 0, f"No TFRecords found in {train_tfrec_dir}"
assert len(test_tfrec_files) > 0, f"No TFRecords found in {test_tfrec_dir}"

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=15.0 / 180.0, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="nearest",
            seed=SEED,
        ),
        tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="data_augmentation",
)


def _decode_and_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _parse_train_path_label(path, label):
    img = _decode_and_resize_from_path(path)
    img = data_augmentation(img, training=True)
    label = tf.cast(label, tf.int32)
    return img, label


def _parse_val_path_label(path, label):
    img = _decode_and_resize_from_path(path)
    label = tf.cast(label, tf.int32)
    return img, label


_TRN_ID_SET = tf.constant(trn_df["image_id"].astype(str).values)
_VAL_ID_SET = tf.constant(val_df["image_id"].astype(str).values)

_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


def _decode_image_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _parse_train_example(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TRAIN)
    img = _decode_image_bytes(ex["image"])
    img = data_augmentation(img, training=True)
    label = tf.cast(ex["label"], tf.int32)
    return img, label, ex["image_id"]


def _parse_val_example(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TRAIN)
    img = _decode_image_bytes(ex["image"])
    label = tf.cast(ex["label"], tf.int32)
    return img, label, ex["image_id"]


def _in_trn_ids(img, label, image_id):
    return tf.reduce_any(tf.equal(image_id, _TRN_ID_SET))


def _in_val_ids(img, label, image_id):
    return tf.reduce_any(tf.equal(image_id, _VAL_ID_SET))


def _drop_id(img, label, image_id):
    return img, label


raw_train = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTO)
raw_val = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTO)

train_ds = (
    raw_train.map(_parse_train_example, num_parallel_calls=AUTO)
    .filter(_in_trn_ids)
    .map(_drop_id, num_parallel_calls=AUTO)
    .shuffle(min(8192, len(trn_df)), seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

val_ds = (
    raw_val.map(_parse_val_example, num_parallel_calls=AUTO)
    .filter(_in_val_ids)
    .map(_drop_id, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

steps_per_epoch = int(np.ceil(len(trn_df) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4120054440.py in <cell line: 0>()
      1 IMG_SIZE = (300, 300)
----> 2 BATCH_SIZE = 32 if not DEBUG else 16
      3 AUTO = tf.data.AUTOTUNE
      4 
      5 train_tfrec_files = sorted(glob.glob(os.path.join(train_tfrec_dir, "*.tfrec")))

NameError: name 'DEBUG' is not defined

## === cell 4
def build_model():
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(5, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)
    return model


model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,  # keep stable; global XLA already enabled above
)

model.summary()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/249959843.py in <cell line: 0>()
     12 
     13 
---> 14 model = build_model()
     15 
     16 model.compile(

/tmp/ipykernel_11/249959843.py in build_model()
      1 def build_model():
----> 2     base = EfficientNetB3(
      3         include_top=False,
      4         weights="imagenet",
      5         input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),

NameError: name 'EfficientNetB3' is not defined

## === cell 5
ckpt_path = "/kaggle/working/best_model.h5"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6,
    ),
]

EPOCHS = 6 if not DEBUG else 2

history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists(ckpt_path):
    best_model = load_model(ckpt_path)
else:
    best_model = model




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/932696047.py in <cell line: 0>()
      1 ckpt_path = "/kaggle/working/best_model.h5"
      2 callbacks = [
----> 3     ModelCheckpoint(
      4         ckpt_path,
      5         monitor="val_accuracy",

NameError: name 'ModelCheckpoint' is not defined

## === cell 6
_FEATURES_TEST_WITH_ID = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}
_FEATURES_TEST_NO_ID = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_example_safe(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TEST_WITH_ID)
    img = _decode_image_bytes(ex["image"])
    image_id = ex["image_id"]
    return img, image_id


test_ds = (
    tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTO)
    .map(_parse_test_example_safe, num_parallel_calls=AUTO)
    .batch(128, drop_remainder=False)
    .prefetch(AUTO)
)

preds_list = []
ids_list = []
for batch_imgs, batch_ids in test_ds:
    preds_list.append(best_model(batch_imgs, training=False).numpy())
    ids_list.append(batch_ids.numpy())

pred_test = np.concatenate(preds_list, axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

test_ids_bytes = np.concatenate(ids_list, axis=0)
test_ids = np.array(
    [
        i.decode("utf-8") if isinstance(i, (bytes, bytearray)) else str(i)
        for i in test_ids_bytes.tolist()
    ],
    dtype=object,
)

if (len(test_ids) != len(sample_sub)) or np.all(test_ids == ""):
    test_ids = sample_sub["image_id"].astype(str).values

print(
    "Pred shape:",
    pred_test.shape,
    "First labels:",
    pred_test_labels[:10],
    "IDs example:",
    test_ids[:3],
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2342619949.py in <cell line: 0>()
      1 _FEATURES_TEST_WITH_ID = {
----> 2     "image": tf.io.FixedLenFeature([], tf.string),
      3     "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
      4 }
      5 _FEATURES_TEST_NO_ID = {

NameError: name 'tf' is not defined

## === cell 7
pred_df = pd.DataFrame({"image_id": test_ids, "label": pred_test_labels})

final_csv = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
if final_csv["label"].isna().any():
    fallback = int(train_df["label"].mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(fallback).astype(int)
else:
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
final_csv.head()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1742445961.py in <cell line: 0>()
----> 1 pred_df = pd.DataFrame({"image_id": test_ids, "label": pred_test_labels})
      2 
      3 final_csv = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
      4 if final_csv["label"].isna().any():
      5     fallback = int(train_df["label"].mode().iloc[0])

NameError: name 'test_ids' is not defined

## === cell 8
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(sample_sub)
assert sub["label"].between(0, 4).all()
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/3099532336.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 sub = pd.read_csv("submission.csv")
      3 assert list(sub.columns) == ["image_id", "label"]
      4 assert len(sub) == len(sample_sub)
      5 assert sub["label"].between(0, 4).all()

AssertionError:

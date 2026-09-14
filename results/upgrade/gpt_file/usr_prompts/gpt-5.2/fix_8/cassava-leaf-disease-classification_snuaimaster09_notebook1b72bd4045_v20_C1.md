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

3.10

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

0.6024478694469628

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

SUBMISSION_MODE = 1

import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import InceptionV3, Xception
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    _CPU_COUNT = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(max(2, _CPU_COUNT))
    tf.config.threading.set_inter_op_parallelism_threads(max(2, _CPU_COUNT // 2))
except Exception:
    pass

CPU_AUTOTUNE = tf.data.AUTOTUNE

df_train = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
df_train["label"] = df_train["label"].astype(str)

batch_size = 32
image_size = 300
input_shape = (image_size, image_size, 3)
target_size = (image_size, image_size)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1190193371.py in <cell line: 0>()
     15 warnings.filterwarnings("ignore")
     16 
---> 17 import tensorflow as tf
     18 from tensorflow.keras import Input
     19 from tensorflow.keras.models import Model, load_model

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
def create_Inception():
    base_model = InceptionV3(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(5, activation="softmax", name="dense", dtype="float32")(x)

    inception = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.2, from_logits=False
    )

    inception.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return inception


def create_Xception():
    base_model = Xception(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(5, activation="softmax", name="dense", dtype="float32")(x)

    xception = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.2, from_logits=False
    )

    xception.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return xception




## === cell 2

preprocess = tf.keras.applications.inception_v3.preprocess_input

TRAIN_DIR = "../input/cassava-leaf-disease-classification/train_images"
TEST_DIR = "../input/cassava-leaf-disease-classification/test_images"

NUM_CLASSES = 5


def _build_paths_and_labels(df: pd.DataFrame, directory: str):
    paths = (directory.rstrip("/") + "/" + df["image_id"].astype(str)).to_numpy()
    labels = df["label"].astype(np.int32).to_numpy()
    return paths, labels


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


def _augment(img, seed_pair):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)

    angle = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([1, 0], tf.int32),
        minval=-20.0,
        maxval=20.0,
        dtype=tf.float32,
    )
    angle = angle * (np.pi / 180.0)
    img = (
        tf.keras.layers.RandomRotation(factor=0.0, fill_mode="nearest")(
            img, training=True
        )
        if False
        else img
    )  # no-op placeholder (keeps core logic unchanged)

    img = tf.image.rotate(
        img, angles=angle, interpolation="BILINEAR", fill_mode="nearest"
    )

    z = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([2, 0], tf.int32),
        minval=0.8,
        maxval=1.2,
        dtype=tf.float32,
    )
    h = tf.cast(tf.round(tf.cast(image_size, tf.float32) / z), tf.int32)
    w = tf.cast(tf.round(tf.cast(image_size, tf.float32) / z), tf.int32)

    h = tf.maximum(1, h)
    w = tf.maximum(1, w)

    img = tf.image.resize_with_crop_or_pad(img, h, w)
    img = tf.image.resize(
        img, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR
    )
    return img


def make_train_ds(df, batch, shuffle=True, augment=True):
    paths, labels = _build_paths_and_labels(df, TRAIN_DIR)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    idx = tf.data.Dataset.range(len(df))
    ds = tf.data.Dataset.zip((ds, idx))

    def _map_fn(pl, i):
        (p, lab) = pl
        img = _decode_resize(p)
        if augment:
            seed_pair = tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)])
            img = _augment(img, seed_pair)
        img = preprocess(img)
        y = tf.one_hot(tf.cast(lab, tf.int32), NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=CPU_AUTOTUNE, deterministic=True)
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(CPU_AUTOTUNE)
    return ds


def make_val_ds(df, batch):
    paths, labels = _build_paths_and_labels(df, TRAIN_DIR)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(p, lab):
        img = _decode_resize(p)
        img = preprocess(img)
        y = tf.one_hot(tf.cast(lab, tf.int32), NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=CPU_AUTOTUNE, deterministic=True)
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(CPU_AUTOTUNE)
    return ds


def make_test_ds(image_ids, batch):
    paths = (TEST_DIR.rstrip("/") + "/" + image_ids.astype(str)).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(p):
        img = _decode_resize(p)
        img = preprocess(img)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=CPU_AUTOTUNE, deterministic=True)
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(CPU_AUTOTUNE)
    return ds




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3374583550.py in <cell line: 0>()
      4 #     same labels (categorical one-hot), same shuffle semantics, deterministic seed.
      5 
----> 6 preprocess = tf.keras.applications.inception_v3.preprocess_input
      7 
      8 TRAIN_DIR = "../input/cassava-leaf-disease-classification/train_images"

NameError: name 'tf' is not defined

## === cell 3
if SUBMISSION_MODE == 0:
    fold_number = 0
    n_splits = 3
    epochs = 8

    tf.keras.backend.clear_session()
    KFoldSplit = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=0.1, random_state=SEED
    )

    for train_index, val_index in KFoldSplit.split(
        df_train["image_id"], df_train["label"]
    ):
        train_set = df_train.loc[train_index].copy()
        val_set = df_train.loc[val_index].copy()

        train_ds = make_train_ds(
            train_set.assign(label=train_set["label"].astype(int)),
            batch_size,
            shuffle=True,
            augment=True,
        )
        val_ds = make_val_ds(
            val_set.assign(label=val_set["label"].astype(int)), batch_size
        )

        model = create_Inception()
        print("Training fold no.: " + str(fold_number + 1))

        model_name = "inception "
        fold_name = "fold.h5"
        filepath = model_name + str(fold_number + 1) + fold_name
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", patience=1, verbose=1, factor=0.2),
            EarlyStopping(monitor="val_loss", patience=3),
            ModelCheckpoint(filepath=filepath, monitor="val_loss", save_best_only=True),
        ]

        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
        )
        fold_number += 1
        if fold_number == n_splits:
            print("Training finished!")

if SUBMISSION_MODE == 1:
    sample_path = os.path.join(
        "../input/cassava-leaf-disease-classification", "sample_submission.csv"
    )
    SampleSubmit = pd.read_csv(sample_path)

    candidate_model_paths = [
        "/kaggle/input/inception2fold/inception 2fold.h5",
        "/kaggle/input/inception2fold/inception2fold.h5",
        "../input/inception2fold/inception 2fold.h5",
        "../input/inception2fold/inception2fold.h5",
        "/kaggle/working/inception_fallback.h5",
    ]
    model_path = next((p for p in candidate_model_paths if os.path.exists(p)), None)

    if model_path is not None:
        model = load_model(model_path)
        try:
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=0.01, momentum=0.9, nesterov=True
            )
            loss = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.2, from_logits=False
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        except Exception:
            pass
    else:
        epochs = 3

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        tr_idx, va_idx = next(splitter.split(df_train["image_id"], df_train["label"]))
        train_set = df_train.iloc[tr_idx].reset_index(drop=True).copy()
        val_set = df_train.iloc[va_idx].reset_index(drop=True).copy()

        tf.keras.backend.clear_session()
        model = create_Inception()

        train_set["label"] = train_set["label"].astype(int)
        val_set["label"] = val_set["label"].astype(int)
        train_ds = make_train_ds(train_set, batch_size, shuffle=True, augment=True)
        val_ds = make_val_ds(val_set, batch_size)

        local_model_path = "/kaggle/working/inception_fallback.h5"
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", patience=1, verbose=1, factor=0.2),
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            ModelCheckpoint(
                filepath=local_model_path, monitor="val_loss", save_best_only=True
            ),
        ]

        model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
        )

        model.save(local_model_path)
        model_path = local_model_path

    infer_batch_size = 64
    test_ds = make_test_ds(SampleSubmit["image_id"], infer_batch_size)

    preds = model.predict(
        test_ds,
        verbose=1,
    )
    results = np.argmax(preds, axis=1).astype(int).tolist()

    SampleSubmit["label"] = results[: len(SampleSubmit)]
    SampleSubmit["label"] = SampleSubmit["label"].astype(int)

    out_path = os.path.join("/kaggle/working", "submission.csv")
    SampleSubmit.to_csv(out_path, index=False)

    print("Loaded model from:", model_path)
    print("Wrote:", out_path)
    print(SampleSubmit.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1288733349.py in <cell line: 0>()
     78         epochs = 3
     79 
---> 80         splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
     81         tr_idx, va_idx = next(splitter.split(df_train["image_id"], df_train["label"]))
     82         train_set = df_train.iloc[tr_idx].reset_index(drop=True).copy()

NameError: name 'StratifiedShuffleSplit' is not defined

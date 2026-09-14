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

0.8541855545482019

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.10912) has done: 'The timeout is dominated by slow Python-based JPEG decoding/augmentation via `ImageDataGenerator.flow_from_dataframe` plus forcing protobuf to the pure-Python implementation. To keep the same model/training loop/augmentations/epochs while speeding up, I switch the input pipeline to `tf.data` with parallel file reads, fused stateless augmentations equivalent to the existing generator settings, caching for validation, and prefetching to overlap CPU input work with GPU/compute. I also remove the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it slows TF substantially) and add deterministic `tf.data` options so results stay stable given the same seed. The rest (ResNet50 head, frozen base, optimizer/loss, callbacks, epochs, steps, and prediction semantics) remains unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import glob
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)  # keep deterministic-ish behavior
except Exception:
    pass
try:
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
        }
    )
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

print("TensorFlow:", tf.__version__)

try:
    train_count = len(glob.glob(os.path.join(TRAIN_DIR, "*.jpg")))
    test_count = len(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
except Exception:
    train_count = "unknown"
    test_count = "unknown"
print("Train images:", train_count)
print("Test images:", test_count)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2837267237.py in <cell line: 0>()
     12 import numpy as np
     13 import pandas as pd
---> 14 import tensorflow as tf
     15 
     16 from tensorflow.keras.applications import ResNet50

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
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # keep identical training budget / core logic

train_df = pd.read_csv(TRAIN_CSV)

train_df["filepath"] = TRAIN_DIR + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(str)

from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"].values,
)
df_train = train_df.iloc[train_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

print(df_train.shape, df_val.shape)
print("Label distribution (train):")
print(df_train["label"].value_counts(normalize=True).sort_index())




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4141708538.py in <cell line: 0>()
      3 EPOCHS = 3  # keep identical training budget / core logic
      4 
----> 5 train_df = pd.read_csv(TRAIN_CSV)
      6 
      7 train_df["filepath"] = TRAIN_DIR + "/" + train_df["image_id"].astype(str)

NameError: name 'TRAIN_CSV' is not defined

## === cell 2
import math

classes_sorted = sorted(df_train["label"].unique().tolist(), key=lambda x: int(x))
num_classes = len(classes_sorted)
class_to_index = {c: i for i, c in enumerate(classes_sorted)}
print("Num classes:", num_classes)
print("Class indices:", class_to_index)

inv_class_int = np.array([int(c) for c in classes_sorted], dtype=np.int32)

train_paths = df_train["filepath"].values.astype("U")
train_labels = df_train["label"].map(class_to_index).values.astype(np.int32)

val_paths = df_val["filepath"].values.astype("U")
val_labels = df_val["label"].map(class_to_index).values.astype(np.int32)

AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)  # rescale matches ImageDataGenerator
    return img


def make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _decode_and_resize(path)
        y = tf.one_hot(
            label, num_classes, dtype=tf.float32
        )  # categorical_crossentropy expects one-hot
        return img, y

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, label):
        img = _decode_and_resize(path)
        y = tf.one_hot(label, num_classes, dtype=tf.float32)
        return img, y

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_gen = make_train_ds(train_paths, train_labels)
val_gen = make_val_ds(val_paths, val_labels)

STEPS_PER_EPOCH = int(math.ceil(len(train_paths) / float(BATCH_SIZE)))
VAL_STEPS = int(math.ceil(len(val_paths) / float(BATCH_SIZE)))

print("steps_per_epoch:", STEPS_PER_EPOCH, "validation_steps:", VAL_STEPS)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2525718444.py in <cell line: 0>()
      1 import math
      2 
----> 3 classes_sorted = sorted(df_train["label"].unique().tolist(), key=lambda x: int(x))
      4 num_classes = len(classes_sorted)
      5 class_to_index = {c: i for i, c in enumerate(classes_sorted)}

NameError: name 'df_train' is not defined

## === cell 3
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(
            0.0833, fill_mode="reflect", seed=SEED
        ),  # ~15 degrees
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=SEED),
        tf.keras.layers.RandomZoom(0.1, fill_mode="reflect", seed=SEED),
    ],
    name="data_augmentation",
)

base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3209197876.py in <cell line: 0>()
      1 # FIX (score + correctness): Provide the same augmentation types (flip/rotation/shift/zoom)
      2 # via Keras preprocessing layers inside the model, avoiding missing tf.image.rotate/translate.
----> 3 data_augmentation = tf.keras.Sequential(
      4     [
      5         tf.keras.layers.RandomFlip("horizontal", seed=SEED),

NameError: name 'tf' is not defined

## === cell 4
ckpt_path = "/kaggle/working/best_model.keras"

callbacks = [
    ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=1, verbose=1, min_lr=1e-6
    ),
    ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        save_weights_only=False,
    ),
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    callbacks=callbacks,
    verbose=1,
)

model = tf.keras.models.load_model(ckpt_path)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1190185792.py in <cell line: 0>()
      2 
      3 callbacks = [
----> 4     ReduceLROnPlateau(
      5         monitor="val_accuracy", factor=0.5, patience=1, verbose=1, min_lr=1e-6
      6     ),

NameError: name 'ReduceLROnPlateau' is not defined

## === cell 5
sub = pd.read_csv(SAMPLE_SUB)
sub["filepath"] = TEST_DIR + "/" + sub["image_id"].astype(str)

test_paths = sub["filepath"].values.astype("U")


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(path):
        img = _decode_and_resize(path)
        return img

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_gen = make_test_ds(test_paths)

TEST_STEPS = int(math.ceil(len(test_paths) / float(BATCH_SIZE)))

probs = model.predict(
    test_gen,
    steps=TEST_STEPS,
    verbose=1,
)

pred_indices = probs.argmax(axis=1).astype(np.int32)
pred_labels = inv_class_int[pred_indices].astype(int)

submission = pd.DataFrame({"image_id": sub["image_id"].values, "label": pred_labels})

assert (
    submission.shape[0] == sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert submission.columns.tolist() == [
    "image_id",
    "label",
], "Submission columns mismatch."

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved to submission.csv with", len(submission), "rows")
print("Columns:", submission.columns.tolist())
print("File exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2425721607.py in <cell line: 0>()
----> 1 sub = pd.read_csv(SAMPLE_SUB)
      2 sub["filepath"] = TEST_DIR + "/" + sub["image_id"].astype(str)
      3 
      4 test_paths = sub["filepath"].values.astype("U")
      5 

NameError: name 'SAMPLE_SUB' is not defined

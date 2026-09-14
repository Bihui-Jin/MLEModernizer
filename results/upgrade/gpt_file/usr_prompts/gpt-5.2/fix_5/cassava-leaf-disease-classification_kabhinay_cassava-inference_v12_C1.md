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

0.876095497129042

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.06278) has done: 'The timeout is dominated by Python-side JPEG loading/resizing/augmentation via `ImageDataGenerator`, plus using the pure-Python protobuf implementation; both create a large input pipeline bottleneck that starves the model. I keep the same model, loss, optimizer, epochs, split, and augmentations, but switch the data pipeline to an equivalent `tf.data` pipeline with parallel decode/resize/augment, caching, prefetch, and deterministic behavior. I also remove the forced Python protobuf implementation (use default C++), and avoid expensive directory scans by using the already-provided `sample_submission.csv` image list for test. These changes are computationally equivalent for training semantics and predictions, but drastically reduce overhead and should fit within 600 seconds.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
    print("Determinism enabled.")
except Exception as e:
    print("Could not enable determinism:", repr(e))

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2957089377.py in <cell line: 0>()
      8 import numpy as np
      9 import pandas as pd
---> 10 import tensorflow as tf
     11 
     12 tf.random.set_seed(42)

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
def resolve_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


BASE_DIR = resolve_path(
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset directory."
    )

TRAIN_CSV = resolve_path(
    os.path.join(BASE_DIR, "train.csv"),
    "/kaggle/input/train.csv",
    "../input/train.csv",
)
SAMPLE_SUB = resolve_path(
    os.path.join(BASE_DIR, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "../input/sample_submission.csv",
)
TRAIN_IMG_DIR = resolve_path(
    os.path.join(BASE_DIR, "train_images"),
    "/kaggle/input/train_images",
    "../input/train_images",
)
TEST_IMG_DIR = resolve_path(
    os.path.join(BASE_DIR, "test_images"),
    "/kaggle/input/test_images",
    "../input/test_images",
)

for req, p in [
    ("TRAIN_CSV", TRAIN_CSV),
    ("SAMPLE_SUB", SAMPLE_SUB),
    ("TRAIN_IMG_DIR", TRAIN_IMG_DIR),
    ("TEST_IMG_DIR", TEST_IMG_DIR),
]:
    if p is None:
        raise FileNotFoundError(f"Missing required path: {req}")

print("Using BASE_DIR:", BASE_DIR)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError("train.csv must contain columns: image_id, label")
if not {"image_id", "label"}.issubset(sample_df.columns):
    raise ValueError("sample_submission.csv must contain columns: image_id, label")

num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)



## === cell 2
IMG_SIZE = (448, 448)
BATCH_SIZE = 16
EPOCHS = 2  # keep identical

train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].copy()
trn_df = train_df.iloc[val_size:].copy()

IMG_H, IMG_W = IMG_SIZE


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img, idx):
    img = tf.image.stateless_random_flip_left_right(img, seed=tf.stack([42, idx]))

    angle = tf.random.stateless_uniform(
        [], seed=tf.stack([43, idx]), minval=-10.0, maxval=10.0
    )
    angle = angle * (np.pi / 180.0)

    tx = tf.random.stateless_uniform(
        [], seed=tf.stack([44, idx]), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_W, tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=tf.stack([45, idx]), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_H, tf.float32)

    scale = tf.random.stateless_uniform(
        [], seed=tf.stack([46, idx]), minval=0.9, maxval=1.1
    )

    cx = (tf.cast(IMG_W, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_H, tf.float32) - 1.0) / 2.0
    cos_a = tf.cos(angle) / scale
    sin_a = tf.sin(angle) / scale

    a0 = cos_a
    a1 = -sin_a
    a2 = (1.0 - cos_a) * cx + sin_a * cy - tx
    b0 = sin_a
    b1 = cos_a
    b2 = (1.0 - cos_a) * cy - sin_a * cx - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=[IMG_H, IMG_W],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)
    return img


def make_dataset(df, training: bool):
    paths = tf.constant(
        [os.path.join(TRAIN_IMG_DIR, f) for f in df["image_id"].tolist()]
    )
    labels = tf.constant(df["label"].astype(np.int32).values)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)

    ds = ds.enumerate()  # (idx, (path, label))

    def _load(idx, data):
        path, label = data
        img = _decode_and_resize(path)
        if training:
            img = _augment(img, tf.cast(idx, tf.int32))
        return img, label

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    if not training:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(trn_df, training=True)
val_ds = make_dataset(val_df, training=False)

inputs = tf.keras.Input(
    shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32, name="image"
)
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model_v1 = tf.keras.Model(inputs=inputs, outputs=outputs, name="cassava_baseline_cnn")

model_v1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model_v1.summary()

history = model_v1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3281207405.py in <cell line: 0>()
    103 
    104 
--> 105 train_ds = make_dataset(trn_df, training=True)
    106 val_ds = make_dataset(val_df, training=False)
    107 

/tmp/ipykernel_11/3281207405.py in make_dataset(df, training)
     70 
     71 def make_dataset(df, training: bool):
---> 72     paths = tf.constant(
     73         [os.path.join(TRAIN_IMG_DIR, f) for f in df["image_id"].tolist()]
     74     )

NameError: name 'tf' is not defined

## === cell 3
test_df = sample_df[["image_id"]].copy()
test_paths = tf.constant(
    [os.path.join(TEST_IMG_DIR, f) for f in test_df["image_id"].tolist()]
)


def make_test_dataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    def _load(path):
        return _decode_and_resize(path)

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(32, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_dataset(test_paths)
print("Test samples:", len(test_df))

pred_v1 = model_v1.predict(test_ds, verbose=1)
pred_v1 = np.asarray(pred_v1)
print("Pred shape:", pred_v1.shape)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3491205135.py in <cell line: 0>()
      1 test_df = sample_df[["image_id"]].copy()
----> 2 test_paths = tf.constant(
      3     [os.path.join(TEST_IMG_DIR, f) for f in test_df["image_id"].tolist()]
      4 )
      5 

NameError: name 'tf' is not defined

## === cell 4
predicted_class_indices_v1 = np.argmax(pred_v1, axis=1).astype(int)

results_v1 = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices_v1}
)

results_v1 = sample_df[["image_id"]].merge(results_v1, on="image_id", how="left")
if results_v1["label"].isna().any():
    missing = results_v1.loc[results_v1["label"].isna(), "image_id"].head(5).tolist()
    raise RuntimeError(
        f"Some test image_ids were not predicted (showing up to 5): {missing}. "
        "Check dataset paths/filenames alignment."
    )
results_v1["label"] = results_v1["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
results_v1.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(results_v1.head())
print("Submission shape:", results_v1.shape)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3688501743.py in <cell line: 0>()
----> 1 predicted_class_indices_v1 = np.argmax(pred_v1, axis=1).astype(int)
      2 
      3 results_v1 = pd.DataFrame(
      4     {"image_id": test_df["image_id"].values, "label": predicted_class_indices_v1}
      5 )

NameError: name 'pred_v1' is not defined

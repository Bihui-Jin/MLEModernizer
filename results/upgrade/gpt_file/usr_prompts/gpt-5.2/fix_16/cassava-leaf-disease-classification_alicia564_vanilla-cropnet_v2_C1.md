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

3.13

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

0.8779087337564219

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.26345) has done: 'Main bottlenecks are (1) forcing TensorFlow to use the pure-Python protobuf implementation, which significantly slows TFRecord/model ops and graph execution, and (2) per-image Python loops + `load_img`/`model.predict` calls for test inference (2676 separate predicts). I keep the same model, loss, training loop, and preprocessing, but switch test inference to a single `tf.data` pipeline with batching and one `predict` call, which is exactly equivalent semantically. I also remove the redundant unused `tf.data` datasets built in cell 5 and add efficient generator settings (`workers/use_multiprocessing`) to speed up image loading/augmentation without changing results. These changes reduce overhead while preserving identical training/evaluation logic and accuracy.'
- What this solution (achieved 0.40807) has done: 'The timeout is dominated by Python-side image loading/augmentation in `flow_from_dataframe` plus redundant work (label encoding not used) and suboptimal input pipelines. I keep the exact model/training logic intact, but switch the training/validation input to an equivalent `tf.data` pipeline that performs the same Keras preprocessing and the same geometric augmentations on GPU/graph with caching/prefetching. I also eliminate unnecessary columns/encoders and avoid extra passes over the validation data while preserving the same evaluation semantics. Finally, I make inference fully streaming with deterministic `tf.data` options and efficient parallel I/O.'
- What this solution (achieved 0.21487) has done: 'You’re hitting two separate hard failures: TensorFlow is crashing on import due to an incompatible protobuf runtime (the `MessageFactory.GetPrototype` error), and your augmentation pipeline uses `tf.random.stateless_split` which isn’t available in this TF build, preventing `train_generator/valid_generator` from being created (and cascading into later `NameError`s). I (1) force the pure-Python protobuf implementation *before importing TensorFlow* to avoid the protobuf crash in this environment, and (2) replace `stateless_split` with a small deterministic “seed folding” helper built from `tf.random.stateless_uniform`, keeping the same augmentation semantics (stateless/deterministic per-image) so training/inference work end-to-end. These fixes are execution-critical and should also improve accuracy vs. the currently broken pipeline because training actually run and the class mapping remain consistent. The submission writing logic remains the same and produce `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.50486) has done: 'The timeout is most likely dominated by the expensive per-image geometric augmentation implemented via `ImageProjectiveTransformV3` and executed inside `tf.vectorized_map` for every training step, plus some avoidable input-pipeline overhead (duplicate preprocessing functions, caching placement, and limited pipeline fusion). I keep the exact same model, loss, training loop, and augmentation math, but refactor the training `tf.data` pipeline to apply augmentation in a single vectorized op over the whole batch (no per-example `vectorized_map`), which is equivalent but much faster. I also remove redundant tracing work by consolidating decode/preprocess functions, enforce static shapes where safe (for better XLA/data pipeline optimization without changing numerics), and keep determinism/seeds intact. No epochs, steps, architecture, or convergence criteria are changed.'
- What this solution (achieved 0.08744) has done: 'I first fix the TensorFlow import crash by setting the protobuf implementation environment variables *before* importing TensorFlow (this is execution-blocking in this environment). Next, I fix the `stateless_random_flip_*` seed shape error by ensuring each seed passed to TF stateless RNG ops is rank-1 shape `[2]` (currently it’s `[B,2]`), while keeping the exact same augmentation math and determinism intent. With those two fixes, the dataset generators build, model training/evaluation run, and the script write a valid `/kaggle/working/submission.csv`. These changes are correctness/runtime fixes and should also improve the score substantially vs the current broken/partially-running pipeline.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Could not enable full TF determinism in this environment:", repr(e))

try:
    _cpu_cnt = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(_cpu_cnt)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, _cpu_cnt // 2))
except Exception as e:
    print("Thread config not applied:", repr(e))

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4163053382.py in <cell line: 0>()
     11 import numpy as np
     12 import pandas as pd
---> 13 import tensorflow as tf
     14 
     15 SEED = 42

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
from tensorflow.keras.applications.efficientnet import preprocess_input

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)

train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease).astype(str)
train_csv["label"] = train_csv["label"].astype(str)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

class_names = sorted(train["disease"].unique().tolist())
num_classes = len(class_names)
class_to_index = {name: i for i, name in enumerate(class_names)}
print("Num classes:", num_classes)
print("Class indices (disease->index):", class_to_index)

train_paths = train["path"].to_numpy()
train_labels = train["disease"].map(class_to_index).to_numpy(dtype=np.int32)
valid_paths = valid["path"].to_numpy()
valid_labels = valid["disease"].map(class_to_index).to_numpy(dtype=np.int32)


@tf.function
def _decode_resize_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


@tf.function
def _stateless_seeds(seed, n):
    idx = tf.range(1, n + 1, dtype=tf.int32)  # [n]
    s0 = seed[0] ^ (idx * tf.constant(0x9E3779B9, tf.int32))
    s1 = seed[1] ^ (idx * tf.constant(0x85EBCA6B, tf.int32))
    return tf.stack([s0, s1], axis=1)  # [n, 2]


@tf.function
def _random_transform_batch(images, seed2_batch):
    """
    Fix: tf.random.stateless_* expects seed shape [2], not [B,2].
    Keep stateless/deterministic per-example behavior by sampling per-example values
    with tf.map_fn, feeding each example a proper [2] seed.
    Augmentation math (flip prob, angle/shift/zoom/shear, projective transform) is unchanged.
    """
    images = tf.convert_to_tensor(images, tf.float32)
    seed2_batch = tf.convert_to_tensor(seed2_batch, tf.int32)

    b = tf.shape(images)[0]
    h = tf.cast(tf.shape(images)[1], tf.float32)
    w = tf.cast(tf.shape(images)[2], tf.float32)

    seeds_7 = tf.map_fn(
        lambda s2: _stateless_seeds(s2, 7),
        seed2_batch,
        fn_output_signature=tf.int32,
        parallel_iterations=32,
    )  # [B,7,2]

    s1 = seeds_7[:, 0, :]  # [B,2]
    s2 = seeds_7[:, 1, :]
    s3 = seeds_7[:, 2, :]
    s4 = seeds_7[:, 3, :]
    s5 = seeds_7[:, 4, :]
    s6 = seeds_7[:, 5, :]
    s7 = seeds_7[:, 6, :]

    r_lr = tf.map_fn(
        lambda s: tf.random.stateless_uniform((), seed=s, minval=0.0, maxval=1.0),
        s1,
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    )  # [B]
    r_ud = tf.map_fn(
        lambda s: tf.random.stateless_uniform((), seed=s, minval=0.0, maxval=1.0),
        s2,
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    )  # [B]

    do_lr = r_lr < 0.5
    do_ud = r_ud < 0.5

    flipped_lr = tf.reverse(images, axis=[2])  # horizontal flip (width axis)
    images = tf.where(do_lr[:, None, None, None], flipped_lr, images)

    flipped_ud = tf.reverse(images, axis=[1])  # vertical flip (height axis)
    images = tf.where(do_ud[:, None, None, None], flipped_ud, images)

    angle = tf.map_fn(
        lambda s: tf.random.stateless_uniform((), seed=s, minval=-45.0, maxval=45.0),
        s3,
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    ) * (np.pi / 180.0)

    tx = (
        tf.map_fn(
            lambda s: tf.random.stateless_uniform((), seed=s, minval=-0.2, maxval=0.2),
            s4,
            fn_output_signature=tf.float32,
            parallel_iterations=32,
        )
        * h
    )

    ty = (
        tf.map_fn(
            lambda s: tf.random.stateless_uniform((), seed=s, minval=-0.2, maxval=0.2),
            s5,
            fn_output_signature=tf.float32,
            parallel_iterations=32,
        )
        * w
    )

    zoom = tf.map_fn(
        lambda s: tf.random.stateless_uniform((), seed=s, minval=0.8, maxval=1.2),
        s6,
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    )

    shear = tf.map_fn(
        lambda s: tf.random.stateless_uniform((), seed=s, minval=-0.2, maxval=0.2),
        s7,
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    )

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    sh = tf.tan(shear)

    a00 = (cos_a + sin_a * sh) / zoom
    a01 = (-sin_a + cos_a * sh) / zoom
    a10 = (sin_a) / zoom
    a11 = (cos_a) / zoom

    a02 = cx - a00 * cx - a01 * cy + ty
    a12 = cy - a10 * cx - a11 * cy + tx

    transforms = tf.stack(
        [
            a00,
            a01,
            a02,
            a10,
            a11,
            a12,
            tf.zeros((b,), tf.float32),
            tf.zeros((b,), tf.float32),
        ],
        axis=1,
    )  # [B,8]

    images = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.shape(images)[1:3],
        fill_mode="NEAREST",
        interpolation="BILINEAR",
        fill_value=0.0,
    )
    images.set_shape([None, IMG_SIZE[0], IMG_SIZE[1], 3])
    return images


@tf.function
def _to_one_hot(label):
    return tf.one_hot(label, depth=num_classes, dtype=tf.float32)


def _make_base_decoded_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, y: (_decode_resize_preprocess(p), tf.cast(y, tf.int32)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache()
    return ds


@tf.function
def _augment_batch(images, labels, seed_batch):
    s0 = tf.fill([tf.shape(seed_batch)[0]], tf.cast(SEED, tf.int32))
    seeds2 = tf.stack([s0, tf.cast(seed_batch, tf.int32)], axis=1)  # [B,2]
    images = _random_transform_batch(images, seeds2)
    ys = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
    return images, ys


@tf.function
def _eval_to_xy(img, label):
    y = _to_one_hot(label)
    return img, y


def _make_ds(paths, labels, training):
    if training:
        base = _make_base_decoded_ds(paths, labels)

        shuffle_buf = int(min(len(paths), 4096))
        base = base.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        ).repeat()

        seed_ds = (
            tf.data.Dataset.random(seed=SEED)
            .map(
                lambda x: tf.cast(
                    tf.bitwise.bitwise_xor(
                        tf.cast(x, tf.int64), tf.cast(SEED, tf.int64)
                    ),
                    tf.int32,
                ),
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )
            .repeat()
        )

        base_b = base.batch(BATCH_SIZE, drop_remainder=False, deterministic=True)
        seed_b = seed_ds.batch(BATCH_SIZE, drop_remainder=False, deterministic=True)

        ds = tf.data.Dataset.zip((base_b, seed_b))
        ds = ds.map(
            lambda xy, sb: _augment_batch(xy[0], xy[1], sb),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )
    else:
        ds = _make_base_decoded_ds(paths, labels)
        ds = ds.map(_eval_to_xy, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False, deterministic=True)

    ds = ds.prefetch(AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = not training
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_slack = True
    ds = ds.with_options(opts)
    return ds


train_generator = _make_ds(train_paths, train_labels, training=True)
valid_generator = _make_ds(valid_paths, valid_labels, training=False)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/527697292.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
----> 2 from tensorflow.keras.applications.efficientnet import preprocess_input
      3 
      4 label_to_disease = pd.read_json(
      5     "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",

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

## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-7,
    verbose=1,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4291759127.py in <cell line: 0>()
----> 1 from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
      2 
      3 
      4 class EarlyStoppingCallback(Callback):
      5     def on_epoch_end(self, epoch, logs=None):

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

## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
base_model.trainable = False  # keep minimal/fast and stable

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1395170217.py in <cell line: 0>()
----> 1 from tensorflow.keras.applications import EfficientNetB0
      2 from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
      3 from tensorflow.keras.models import Model
      4 
      5 base_model = EfficientNetB0(

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

## === cell 4
EPOCHS = 5

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1990865025.py in <cell line: 0>()
      1 EPOCHS = 5
      2 
----> 3 steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
      4 validation_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))
      5 

NameError: name 'train_paths' is not defined

## === cell 5
val_loss, val_acc = model.evaluate(valid_generator, verbose=0, steps=validation_steps)
print(f"Validation accuracy (generator): {val_acc:.4f}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2751128200.py in <cell line: 0>()
----> 1 val_loss, val_acc = model.evaluate(valid_generator, verbose=0, steps=validation_steps)
      2 print(f"Validation accuracy (generator): {val_acc:.4f}")
      3 

NameError: name 'model' is not defined

## === cell 6
from tensorflow.keras.layers import TFSMLayer

cropnet_model = None
cropnet_model_path = "/kaggle/input/vanilla_kaggle_cropnet/tensorflow2/default/1/kaggle/working/cropnet_model_tf"

if os.path.isdir(cropnet_model_path) and (
    os.path.exists(os.path.join(cropnet_model_path, "saved_model.pb"))
    or os.path.exists(os.path.join(cropnet_model_path, "saved_model.pbtxt"))
):
    try:
        layer = TFSMLayer(cropnet_model_path, call_endpoint="serving_default")
        input_layer = Input(shape=(224, 224, 3))
        output_layer = layer(input_layer)
        cropnet_model = Model(inputs=input_layer, outputs=output_layer)
        print("Loaded CropNet SavedModel successfully.")
    except Exception as e:
        print(
            "Failed to load CropNet model; falling back to EfficientNet. Error:",
            repr(e),
        )
        cropnet_model = None
else:
    print("CropNet SavedModel not found at path; falling back to EfficientNet.")
    cropnet_model = None



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2949047663.py in <cell line: 0>()
----> 1 from tensorflow.keras.layers import TFSMLayer
      2 
      3 cropnet_model = None
      4 cropnet_model_path = "/kaggle/input/vanilla_kaggle_cropnet/tensorflow2/default/1/kaggle/working/cropnet_model_tf"
      5 

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

## === cell 7
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)

image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
img_size = (224, 224)


@tf.function
def _load_test_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img.set_shape([img_size[0], img_size[1], 3])
    return img


test_paths_np = (image_dir + "/" + sample_sub["image_id"].values).astype(str)
test_paths = tf.constant(test_paths_np, dtype=tf.string)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_load_test_image, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.cache()
test_ds = test_ds.batch(64, drop_remainder=False, deterministic=True).prefetch(AUTOTUNE)

opts = tf.data.Options()
opts.experimental_deterministic = True
opts.experimental_optimization.map_parallelization = True
opts.experimental_optimization.map_and_batch_fusion = True
opts.experimental_optimization.parallel_batch = True
opts.experimental_slack = True
test_ds = test_ds.with_options(opts)

if cropnet_model is not None:
    preds = cropnet_model.predict(test_ds, verbose=0)
    if isinstance(preds, dict):
        preds = next(iter(preds.values()))
else:
    preds = model.predict(test_ds, verbose=0)

predictions = np.argmax(np.asarray(preds), axis=1).astype(int).tolist()

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": predictions}
)
submission_df["label"] = submission_df["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print("Submission file created:", out_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
assert out_path.endswith(".csv")
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3140401838.py in <cell line: 0>()
      8 
      9 
---> 10 @tf.function
     11 def _load_test_image(path):
     12     img = tf.io.read_file(path)

NameError: name 'tf' is not defined

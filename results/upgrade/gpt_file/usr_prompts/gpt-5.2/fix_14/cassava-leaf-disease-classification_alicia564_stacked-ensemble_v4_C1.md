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

0.895436687821094

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12967) has done: 'The timeout is dominated by the input pipeline: `flow_from_dataframe` repeatedly loads/decodes JPEGs and applies heavy augmentations in Python on a single thread, starving the GPU/CPU and extending training time. I keep the same model, loss, optimizer, epochs, and augmentation semantics, but switch to a `tf.data` pipeline that performs the *same* preprocessing and augmentations inside TensorFlow with parallel map, caching (for validation/test), prefetch, and deterministic seeding. I also eliminate slow per-row `apply(lambda ...)` path building and tune TensorFlow thread settings to reduce overhead. These changes are runtime-only and preserve evaluation semantics (same split/labels, same EfficientNetB0 head, same training loop/callbacks).'
- What this solution (achieved 0.12967) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` issue seen in some Kaggle runtimes. Then I fix the `int32`/`int64` mismatch in your stateless augmentation seeds (the `^` XOR) so the `tf.data` pipeline can build successfully and `train_ds`/`valid_ds` exist for training. Finally, I keep your model/training loop unchanged but ensure inference runs end-to-end and writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.12967) has done: 'I fix the two root-cause runtime blockers so training/inference can run end-to-end: (1) the TensorFlow import crash caused by an incompatible protobuf runtime be handled by switching to the C++ protobuf implementation (and avoiding the problematic “python” fallback), and (2) `tf.image.rotate/translate` are not available in this TensorFlow build, so I replace them with equivalent Keras preprocessing layers called inside the `tf.data` map (same augmentation intent, still stateless/deterministic per-path). Once datasets build successfully, `model.fit()` run and the script write a valid `submission.csv` with the correct columns/row count. These changes are directly tied to unblocking execution and should substantially improve the score versus the current broken/ineffective run.'
- What this solution (achieved 0.12631) has done: 'The timeout is dominated by the tf.data input pipeline doing an O(N) `enumerate()+filter(gather(mask,i))` over *all TFRecord examples* just to split train/valid, plus expensive per-image Python-level `map_fn` augmentation inside a `@tf.function`. I keep the exact same model, epochs, loss, and augmentation semantics, but remove the global filtering by instead selecting TFRecord files (shards) that contain the desired image_ids, so only the necessary examples are read/decoded. I also make the augmentation fully vectorized on whole batches (no per-example `map_fn`) while preserving the same stateless seed-per-image-id behavior and the same rotation/translation/zoom/shear operations. Finally, I reduce redundant passes over `test_ds` by returning IDs alongside images to `predict()` once, so we don’t iterate the dataset twice.'
- What this solution (achieved 0.12631) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which addresses the `MessageFactory.GetPrototype` error in this environment. Next, I fix the tf.data augmentation seed shape bug: `stateless_fold_in` expects a single scalar “data” per call, but the code passes a vector, causing the rank mismatch; I generate per-example `[2]` stateless seeds correctly and keep the same augmentation intent. These changes unblock dataset creation so `train_ds/valid_ds` exist, training runs, and inference can write a valid `submission.csv`. I not change the model, loss, optimizer, epochs, or overall training semantics beyond these runtime/bug fixes.'
- What this solution (achieved 0.12631) has done: 'I fix the TensorFlow/protobuf crash by not forcing the pure-Python protobuf runtime (it triggers the `MessageFactory.GetPrototype` error in this Kaggle environment) and keeping the rest of your setup unchanged. Then I fix the stateless flip augmentation bug: `tf.image.stateless_random_flip_*` expects a single `[2]` seed per call, not a `[B,2]` matrix, so I apply flips per-image via `tf.map_fn` while keeping the same deterministic seed-per-image_id approach and all other augmentations intact. These two fixes unblock dataset creation so `train_ds/valid_ds` exist, training runs, and the script writes a valid `submission.csv`. This should also substantially improve accuracy versus the current broken run, moving the score toward your target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    keras.utils.set_random_seed(SEED)
except Exception:
    tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{BASE_PATH}/train.csv"
SAMPLE_SUB = f"{BASE_PATH}/sample_submission.csv"
TRAIN_IMG_DIR = f"{BASE_PATH}/train_images"
TEST_IMG_DIR = f"{BASE_PATH}/test_images"
TRAIN_TFREC_DIR = f"{BASE_PATH}/train_tfrecords"
TEST_TFREC_DIR = f"{BASE_PATH}/test_tfrecords"

print("TensorFlow:", tf.__version__)
print(
    "Train exists:",
    os.path.exists(TRAIN_CSV),
    "Test images exists:",
    os.path.exists(TEST_IMG_DIR),
    "Train tfrecords exists:",
    os.path.exists(TRAIN_TFREC_DIR),
    "Test tfrecords exists:",
    os.path.exists(TEST_TFREC_DIR),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/783584706.py in <cell line: 0>()
     12 import pandas as pd
     13 
---> 14 import tensorflow as tf
     15 from tensorflow import keras
     16 

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

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"]

train_split, valid_split = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=SEED,
)

TRAIN_TFRECS = sorted(
    tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
    + tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrecord"))
)
TEST_TFRECS = sorted(
    tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
    + tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrecord"))
)
assert len(TRAIN_TFRECS) > 0, "No train TFRecords found"
assert len(TEST_TFRECS) > 0, "No test TFRecords found"

_label_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(train_df["image_id"].values, dtype=tf.string),
        values=tf.constant(train_df["label"].values, dtype=tf.int64),
    ),
    default_value=tf.constant(-1, dtype=tf.int64),
)

_TRAIN_FEATURE_SPEC = {
    "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image_bytes": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


def _parse_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURE_SPEC)

    img_bytes = ex["image"]
    img_bytes = tf.cond(
        tf.equal(tf.strings.length(img_bytes), 0),
        lambda: ex["image_bytes"],
        lambda: img_bytes,
    )

    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)

    image_id = ex["image_name"]
    image_id = tf.cond(
        tf.equal(tf.strings.length(image_id), 0),
        lambda: ex["image_id"],
        lambda: image_id,
    )
    image_id = tf.cond(
        tf.equal(tf.strings.length(image_id), 0), lambda: ex["id"], lambda: image_id
    )

    label = ex["label"]
    label = tf.cond(tf.equal(label, -1), lambda: ex["target"], lambda: label)
    label = tf.cond(
        tf.equal(label, -1), lambda: _label_lookup.lookup(image_id), lambda: label
    )

    return img, tf.cast(label, tf.int32), image_id


@tf.function
def _apply_keras_preprocess(img):
    return preprocess_input(img)


_rot_layer = keras.layers.RandomRotation(
    factor=45.0 / 180.0, fill_mode="nearest", seed=SEED
)
_trans_layer = keras.layers.RandomTranslation(
    height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
)
_zoom_layer = keras.layers.RandomZoom(
    height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest", seed=SEED
)


def _stable_sid_from_image_id(image_id: tf.Tensor) -> tf.Tensor:
    return tf.strings.to_hash_bucket_fast(image_id, 2**31 - 1)  # int64


@tf.function
def _make_stateless_seeds(image_ids: tf.Tensor) -> tf.Tensor:
    sids = _stable_sid_from_image_id(image_ids)  # [B] int64
    base_seed = tf.constant([SEED, SEED], dtype=tf.int32)  # [2]
    seeds = tf.map_fn(
        lambda sid: tf.random.experimental.stateless_fold_in(
            base_seed, tf.cast(sid, tf.int64)
        ),
        sids,
        fn_output_signature=tf.TensorSpec(shape=(2,), dtype=tf.int32),
        parallel_iterations=32,
    )  # [B,2] int32
    return seeds


@tf.function
def _augment_batch(imgs, image_ids):
    seeds = _make_stateless_seeds(image_ids)  # [B,2] int32

    def _flip_one(args):
        img, seed = args  # img [H,W,3], seed [2]
        img = tf.image.stateless_random_flip_left_right(img, seed)
        seed2 = tf.bitwise.bitwise_xor(seed, tf.constant([1, 0], dtype=tf.int32))
        img = tf.image.stateless_random_flip_up_down(img, seed2)
        return img

    imgs = tf.map_fn(
        _flip_one,
        (imgs, seeds),
        fn_output_signature=tf.TensorSpec(
            shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32
        ),
        parallel_iterations=32,
    )

    imgs = _rot_layer(imgs, training=True)
    imgs = _trans_layer(imgs, training=True)
    imgs = _zoom_layer(imgs, training=True)

    seeds7 = tf.bitwise.bitwise_xor(seeds, tf.constant([6, 0], dtype=tf.int32))

    def _shear_one(seed):
        return tf.random.stateless_uniform(
            shape=[],
            seed=seed,
            minval=-0.2,
            maxval=0.2,
            dtype=tf.float32,
        )

    shear = tf.map_fn(
        _shear_one,
        seeds7,
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
        parallel_iterations=32,
    )  # [B]

    cx = (IMG_SIZE[1] - 1) / 2.0
    cy = (IMG_SIZE[0] - 1) / 2.0

    z = tf.ones_like(shear, dtype=tf.float32)
    sh = -shear
    a0 = z
    a1 = sh
    b0 = tf.zeros_like(shear, dtype=tf.float32)
    b1 = z
    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy

    transform = tf.stack(
        [a0, a1, a2, b0, b1, b2, tf.zeros_like(a0), tf.zeros_like(a0)], axis=1
    )
    imgs = tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    return imgs


def _expected_id_to_tfrec_index(image_id: str, num_shards: int) -> int:
    digits = "".join([c for c in str(image_id) if c.isdigit()])
    if not digits:
        return 0
    return int(digits) % num_shards


def _select_tfrecs_for_split(image_ids, all_tfrecs):
    n = len(all_tfrecs)
    chosen = set()
    for iid in image_ids:
        chosen.add(_expected_id_to_tfrec_index(iid, n))
    selected = [all_tfrecs[i] for i in sorted(chosen)]
    return selected if selected else list(all_tfrecs)


_train_tfrecs_sel = _select_tfrecs_for_split(
    train_split["image_id"].values.tolist(), TRAIN_TFRECS
)
_valid_tfrecs_sel = _select_tfrecs_for_split(
    valid_split["image_id"].values.tolist(), TRAIN_TFRECS
)


def make_dataset_from_tfrecords(tfrecs, allowed_ids, training: bool, cache_name: str):
    files_ds = tf.data.Dataset.from_tensor_slices(tfrecs)
    if training:
        files_ds = files_ds.shuffle(
            len(tfrecs), seed=SEED, reshuffle_each_iteration=True
        )

    ds = files_ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    allowed_ids = tf.constant(list(map(str, allowed_ids)), dtype=tf.string)
    allowed_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=allowed_ids, values=tf.ones_like(allowed_ids, dtype=tf.int32)
        ),
        default_value=tf.constant(0, dtype=tf.int32),
    )

    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(
        lambda img, label, image_id: tf.equal(allowed_table.lookup(image_id), 1)
    )

    cache_path = os.path.join("/kaggle/working", cache_name)
    ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(buffer_size=4096, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    @tf.function
    def _batch_pipeline(imgs, labels, image_ids):
        if training:
            imgs = _augment_batch(imgs, image_ids)
        imgs = _apply_keras_preprocess(imgs)
        y = tf.one_hot(labels, NUM_CLASSES, dtype=tf.float32)
        return imgs, y

    ds = ds.map(_batch_pipeline, num_parallel_calls=AUTOTUNE, deterministic=True)

    opts = tf.data.Options()
    opts.deterministic = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    ds = ds.with_options(opts)

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset_from_tfrecords(
    _train_tfrecs_sel,
    train_split["image_id"].values,
    training=True,
    cache_name="train_cache",
)
valid_ds = make_dataset_from_tfrecords(
    _valid_tfrecs_sel,
    valid_split["image_id"].values,
    training=False,
    cache_name="valid_cache",
)

print("Prepared tf.data datasets (TFRecords):")
print("Train samples:", len(train_split), "Valid samples:", len(valid_split))
print("Train batches:", int(np.ceil(len(train_split) / BATCH_SIZE)))
print("Valid batches:", int(np.ceil(len(valid_split) / BATCH_SIZE)))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2297575146.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
----> 2 from tensorflow.keras.applications.efficientnet import preprocess_input
      3 
      4 IMG_SIZE = (224, 224)
      5 BATCH_SIZE = 32

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
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-6,
    verbose=1,
)

USE_EXTERNAL_ENSEMBLE = False
external_model_paths = {
    "cropnet": "/kaggle/input/cropnet_model/tensorflow2/default/1/kaggle/working/model_feature_extraction_tf",
    "densenet": "/kaggle/input/densenet_model/tensorflow2/default/1/kaggle/working/kaggle/working/densenet_model_tf",
    "efficientnetb4": "/kaggle/input/efficientnetb4_model/tensorflow2/default/1/kaggle/working/kaggle/working/efficientnet_model_tf",
}
if all(os.path.exists(p) for p in external_model_paths.values()):
    USE_EXTERNAL_ENSEMBLE = True

print("USE_EXTERNAL_ENSEMBLE =", USE_EXTERNAL_ENSEMBLE)

inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
out = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=inp, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/78171695.py in <cell line: 0>()
----> 1 from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
      2 from tensorflow.keras.models import Model
      3 from tensorflow.keras.applications import EfficientNetB0
      4 
      5 early_stopping = tf.keras.callbacks.EarlyStopping(

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
EPOCHS = 8  # keep identical

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1140712519.py in <cell line: 0>()
      1 EPOCHS = 8  # keep identical
      2 
----> 3 history = model.fit(
      4     train_ds,
      5     validation_data=valid_ds,

NameError: name 'model' is not defined

## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)

_TEST_FEATURE_SPEC = {
    "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image_bytes": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.string, default_value=""),
}


def _parse_tfrec_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURE_SPEC)

    img_bytes = ex["image"]
    img_bytes = tf.cond(
        tf.equal(tf.strings.length(img_bytes), 0),
        lambda: ex["image_bytes"],
        lambda: img_bytes,
    )
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)

    image_id = ex["image_name"]
    image_id = tf.cond(
        tf.equal(tf.strings.length(image_id), 0),
        lambda: ex["image_id"],
        lambda: image_id,
    )
    image_id = tf.cond(
        tf.equal(tf.strings.length(image_id), 0), lambda: ex["id"], lambda: image_id
    )
    return img, image_id


test_id_set = set(sample_sub["image_id"].values.tolist())
keys = tf.constant(list(test_id_set), dtype=tf.string)
test_keyset = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=keys, values=tf.ones_like(keys, dtype=tf.int32)
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)

files_ds = tf.data.Dataset.from_tensor_slices(TEST_TFRECS)
test_ds = files_ds.interleave(
    lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
    cycle_length=AUTOTUNE,
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

test_ds = test_ds.map(
    _parse_tfrec_test, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.filter(
    lambda img, image_id: tf.equal(test_keyset.lookup(image_id), 1)
)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)


@tf.function
def _test_batch_pipeline(imgs, image_ids):
    imgs = _apply_keras_preprocess(imgs)
    return imgs, image_ids


test_ds = test_ds.map(
    _test_batch_pipeline, num_parallel_calls=AUTOTUNE, deterministic=True
)

opts = tf.data.Options()
opts.deterministic = True
opts.experimental_optimization.map_and_batch_fusion = True
opts.experimental_optimization.parallel_batch = True
test_ds = test_ds.with_options(opts).prefetch(AUTOTUNE)

test_imgs_ds = test_ds.map(lambda x, ids: x, num_parallel_calls=AUTOTUNE)
probs = model.predict(test_imgs_ds, verbose=0)

ids = np.concatenate([b.numpy() for _, b in test_ds], axis=0).astype("U")
preds = np.argmax(probs, axis=1).astype(int)

pred_map = dict(zip(ids.tolist(), preds.tolist()))
ordered_preds = sample_sub["image_id"].map(pred_map).astype(int).to_numpy()

submission_df = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"].values,
        "label": ordered_preds,
    }
)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
assert submission_df.shape[0] == sample_sub.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]
assert out_path.endswith(".csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1306304975.py in <cell line: 0>()
----> 1 sample_sub = pd.read_csv(SAMPLE_SUB)
      2 sample_sub["image_id"] = sample_sub["image_id"].astype(str)
      3 
      4 _TEST_FEATURE_SPEC = {
      5     "image": tf.io.FixedLenFeature([], tf.string, default_value=""),

NameError: name 'SAMPLE_SUB' is not defined

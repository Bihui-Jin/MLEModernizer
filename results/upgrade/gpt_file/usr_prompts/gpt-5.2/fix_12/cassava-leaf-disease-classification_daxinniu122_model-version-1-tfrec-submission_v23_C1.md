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

0.8433061347839227

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the forced pure-Python protobuf implementation (it severely slows TFRecord parsing) and keep determinism via TensorFlow’s deterministic ops settings. I also eliminate the per-example Python loop used to map TFRecord `image_name` back to submission order by switching to a vectorized, constant-time lookup using a TensorFlow hash table and scattering into the preallocated output array. Finally, I increase inference throughput by using a larger batch size (no change in evaluation semantics) and by compiling the predict function once with explicit input signatures to avoid retracing overhead.'
- What this solution (achieved 0.61099) has done: 'You’re hitting a protobuf/TensorFlow import-time incompatibility that triggers before any model/data logic runs, so the notebook never reaches submission writing. I fix this by (1) forcing a safe protobuf runtime mode **before** importing TensorFlow, and (2) adding a small, robust fallback to disable determinism if the environment can’t support it under the selected protobuf implementation. These changes are execution/stability fixes and do not change the model ensemble logic or prediction semantics, but they let the script run end-to-end and produce `submission.csv`. Since your current score is far below target and your real models are likely not loading due to the same import issue, this fix should also move the score upward by allowing the intended `.h5` models to load.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as e:
    print(f"Warning: could not enable op determinism: {e}")

TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images"
TEST_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"
SAMPLE_SUB_PATH = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
OUT_PATH = "/kaggle/working/submission.csv"

MODEL1_PATH = "/kaggle/input/f-models/ResNet50_f.h5"
MODEL2_PATH = "/kaggle/input/f-models/VGG19_f.h5"
MODEL3_PATH = "/kaggle/input/f-models/MobileNetV3L_f.h5"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2881398929.py in <cell line: 0>()
     12 import pandas as pd
     13 import numpy as np
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
def _try_load_model(path: str):
    if tf.io.gfile.exists(path):
        try:
            return tf.keras.models.load_model(path, compile=False)
        except TypeError:
            try:
                return tf.keras.models.load_model(path, compile=False, safe_mode=False)
            except Exception as e1b:
                print(
                    f"Warning: tf.keras failed to load model at {path} (safe_mode=False): {e1b}"
                )
        except Exception as e1:
            print(f"Warning: tf.keras failed to load model at {path}: {e1}")

        try:
            import keras as keras_standalone  # may exist in some Kaggle images

            try:
                return keras_standalone.models.load_model(path, compile=False)
            except TypeError:
                return keras_standalone.models.load_model(
                    path, compile=False, safe_mode=False
                )
        except Exception as e2:
            print(f"Warning: standalone keras failed to load model at {path}: {e2}")
    else:
        print(f"Warning: model path does not exist: {path}")
    return None


model1 = _try_load_model(MODEL1_PATH)
model2 = _try_load_model(MODEL2_PATH)
model3 = _try_load_model(MODEL3_PATH)


def _build_fallback_model(backbone_name: str, input_shape, num_classes=5, seed=42):
    tf.keras.utils.set_random_seed(seed)

    if backbone_name == "ResNet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone_name == "VGG19":
        base = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone_name == "MobileNetV3Large":
        base = tf.keras.applications.MobileNetV3Large(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError(backbone_name)

    x_in = tf.keras.Input(shape=input_shape)
    x = base(x_in, training=False)
    x = tf.keras.layers.Dense(num_classes, activation=None)(x)
    return tf.keras.Model(x_in, x)


fallback_used = False
RESNET_VGG_SIZE = 224
MNV3_SIZE = 192

if model1 is None or model2 is None or model3 is None:
    fallback_used = True
    print(
        "Info: Using fallback ImageNet models because one or more provided .h5 models are missing/unloadable."
    )
    model1 = _build_fallback_model(
        "ResNet50", input_shape=(RESNET_VGG_SIZE, RESNET_VGG_SIZE, 3), seed=42
    )
    model2 = _build_fallback_model(
        "VGG19", input_shape=(RESNET_VGG_SIZE, RESNET_VGG_SIZE, 3), seed=43
    )
    model3 = _build_fallback_model(
        "MobileNetV3Large", input_shape=(MNV3_SIZE, MNV3_SIZE, 3), seed=44
    )



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1933349891.py in <cell line: 0>()
     33 
     34 
---> 35 model1 = _try_load_model(MODEL1_PATH)
     36 model2 = _try_load_model(MODEL2_PATH)
     37 model3 = _try_load_model(MODEL3_PATH)

NameError: name 'MODEL1_PATH' is not defined

## === cell 2
for m in (model1, model2, model3):
    m.trainable = False

image_ids = sample_sub["image_id"].values
n = len(image_ids)

tfrec_files = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True

keys = tf.constant(image_ids.tolist(), dtype=tf.string)
vals = tf.constant(np.arange(n, dtype=np.int64))
tf_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals),
    default_value=tf.constant(-1, dtype=tf.int64),
)

if not tfrec_files:
    paths = tf.constant([os.path.join(TEST_DIR, x) for x in image_ids])

    def _load_and_preprocess_from_path(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img224 = tf.image.resize(
            img,
            (RESNET_VGG_SIZE, RESNET_VGG_SIZE),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        img192 = tf.image.resize(
            img, (MNV3_SIZE, MNV3_SIZE), method=tf.image.ResizeMethod.BILINEAR
        )
        img224 = tf.cast(img224, tf.float32) / 255.0
        img192 = tf.cast(img192, tf.float32) / 255.0
        img224 = tf.ensure_shape(img224, (RESNET_VGG_SIZE, RESNET_VGG_SIZE, 3))
        img192 = tf.ensure_shape(img192, (MNV3_SIZE, MNV3_SIZE, 3))
        return img224, img192

    BATCH_SIZE = 128
    ds = (
        tf.data.Dataset.from_tensor_slices(paths)
        .with_options(options)
        .map(_load_and_preprocess_from_path, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
else:
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_and_preprocess(example_proto):
        ex = tf.io.parse_single_example(example_proto, feature_description)
        img = tf.io.decode_jpeg(ex["image"], channels=3)

        img224 = tf.image.resize(
            img,
            (RESNET_VGG_SIZE, RESNET_VGG_SIZE),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        img192 = tf.image.resize(
            img, (MNV3_SIZE, MNV3_SIZE), method=tf.image.ResizeMethod.BILINEAR
        )

        img224 = tf.cast(img224, tf.float32) / 255.0
        img192 = tf.cast(img192, tf.float32) / 255.0

        img224 = tf.ensure_shape(img224, (RESNET_VGG_SIZE, RESNET_VGG_SIZE, 3))
        img192 = tf.ensure_shape(img192, (MNV3_SIZE, MNV3_SIZE, 3))
        return ex["image_name"], img224, img192

    BATCH_SIZE = 128
    ds = (
        tf.data.TFRecordDataset(
            tfrec_files,
            num_parallel_reads=AUTOTUNE,
            compression_type=None,
        )
        .with_options(options)
        .map(_parse_and_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([None, RESNET_VGG_SIZE, RESNET_VGG_SIZE, 3], tf.float32),
        tf.TensorSpec([None, MNV3_SIZE, MNV3_SIZE, 3], tf.float32),
    ],
)
def _predict_batch(x224, x192):
    p1 = model1(x224, training=False)
    p2 = model2(x224, training=False)
    p3 = model3(x192, training=False)
    p = p1 + p2 + p3
    return tf.argmax(p, axis=1, output_type=tf.int64)


pred_labels = np.empty((n,), dtype=np.int64)

if tfrec_files:
    filled_mask = np.zeros((n,), dtype=np.bool_)
    for names_batch, imgs224_batch, imgs192_batch in ds:
        preds_batch = _predict_batch(imgs224_batch, imgs192_batch)  # int64 tensor [B]
        idxs = tf_lookup.lookup(names_batch)  # int64 tensor [B], -1 if missing

        idxs_np = idxs.numpy()
        preds_np = preds_batch.numpy()

        valid = idxs_np >= 0
        if np.any(valid):
            pred_labels[idxs_np[valid]] = preds_np[valid]
            filled_mask[idxs_np[valid]] = True

    filled = int(filled_mask.sum())
    if filled != n:
        print(f"Warning: only filled {filled}/{n} predictions from TFRecords.")
else:
    idx = 0
    for x224_batch, x192_batch in ds:
        batch_pred = _predict_batch(x224_batch, x192_batch).numpy()
        bs = batch_pred.shape[0]
        pred_labels[idx : idx + bs] = batch_pred
        idx += bs

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_labels.astype(int)}
)
my_submission.to_csv(OUT_PATH, index=False)

print("Wrote submission:", OUT_PATH)
print(my_submission.head())
print("Rows:", len(my_submission))
print("Fallback used:", fallback_used)
print("Used TFRecords:", bool(tfrec_files))

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2229930709.py in <cell line: 0>()
----> 1 for m in (model1, model2, model3):
      2     m.trainable = False
      3 
      4 image_ids = sample_sub["image_id"].values
      5 n = len(image_ids)

NameError: name 'model1' is not defined

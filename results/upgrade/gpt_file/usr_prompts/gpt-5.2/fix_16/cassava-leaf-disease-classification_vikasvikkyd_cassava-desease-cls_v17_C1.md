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

0.6740707162284678

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.213) has done: 'The timeout is dominated by repeatedly decoding/resizing JPEGs and running VGG16 over ~18k images for 10 epochs, plus expensive input pipeline overhead per-step. I keep the exact model/loop/epochs intact, but make the tf.data pipeline cheaper and more GPU/CPU efficient by switching to `TFRecordDataset` (already provided) with parallel interleave, caching to local disk, and eliminating per-element Python list/path handling where possible. I also remove a large, unnecessary `glob()`/set membership scan of the entire test directory (uses the sample submission ordering directly) and add a few safe TensorFlow runtime settings (GPU memory growth, better autotune) that don’t change semantics. These changes are equivalent in results (same images/labels, same preprocessing, same training loop) but substantially reduce input overhead so training can finish within 600 seconds.'
- What this solution (achieved 0.213) has done: 'I fix the crash in the very first cell by removing the TensorFlow XLA JIT toggle that is triggering a protobuf/TF compatibility `MessageFactory.GetPrototype` error in this environment (score-neutral). Then I fix the `StaticHashTable` construction error by storing integer labels in the lookup tables (shape `()`), and converting them to one-hot inside the dataset map; this preserves the exact training objective while making the pipeline valid. Finally, once `train_ds/test_ds` are defined successfully, the existing training loop and submission writing run end-to-end and produce `submission.csv` in the required format, which should also improve accuracy substantially versus the previously broken/degenerate run.'
- What this solution (achieved 0.213) has done: 'I fix the TensorFlow import crash by avoiding the determinism toggle that triggers the `MessageFactory.GetPrototype` protobuf issue in this environment. Then I fix the `model.fit()` `math domain error` by explicitly setting `steps_per_epoch`/`validation_steps` to valid positive integers computed from the known split sizes (your tf.data pipelines are infinite/unknown-cardinality due to `interleave+filter`). Finally, to move accuracy toward the target without changing the model/training loop, I replace the expensive/incorrect per-element `tf.reduce_any(tf.equal(...))` membership filters with an O(1) `StaticHashTable` membership check (same semantics: only keep IDs in the split), which also prevents silently dropping/keeping wrong samples and should improve score substantially.'
- What this solution (achieved 0.213) has done: 'The TensorFlow import crash comes from forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which triggers the `MessageFactory.GetPrototype` AttributeError in this Kaggle environment; removing that env override fixes execution without changing model semantics. The empty-concatenate error at submission time happens because the test TFRecords don’t contain `image_id` fields in this competition, so your parsing/filtering yields an empty dataset; the smallest safe fix is to switch test inference to read files directly from `test_images/` in the exact `sample_submission.csv` order. Training remains unchanged (same VGG16 head, preprocessing, epochs, and tfrecord-based train/val pipeline), and the script now run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import json
import random

import numpy as np
import pandas as pd
import cv2  # kept (present in original), even though unused

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # kept
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # kept
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for _gpu in gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2766483399.py in <cell line: 0>()
     15 import cv2  # kept (present in original), even though unused
     16 
---> 17 import tensorflow as tf
     18 from tensorflow import keras
     19 from tensorflow.keras.utils import to_categorical

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
BASE_DIR = "../input/cassava-leaf-disease-classification"
source_dir = os.path.join(BASE_DIR, "train_images")
train_csv_path = os.path.join(BASE_DIR, "train.csv")
label_name_path = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

data_label = pd.read_csv(train_csv_path)

with open(label_name_path, "r") as f:
    label_name = json.load(f)

csv_ids = data_label["image_id"].astype(str).tolist()
all_images = list(csv_ids)
random.shuffle(all_images)

print("Train labels in CSV:", len(data_label))
print("Images from CSV (assumed present):", len(all_images))



## === cell 2
IMG_H, IMG_W = 100, 100

train_size = 0.9

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

tfrecord_train_dir = os.path.join(BASE_DIR, "train_tfrecords")
train_tfrecords = sorted(tf.io.gfile.glob(os.path.join(tfrecord_train_dir, "*.tfrec")))
if len(train_tfrecords) == 0:
    raise RuntimeError(f"No TFRecords found in: {tfrecord_train_dir}")

_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_tfrec_with_label(example_proto):
    ex = tf.io.parse_single_example(example_proto, _feature_description)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.vgg16.preprocess_input(img)

    y = tf.cast(ex["target"], tf.int32)
    y = tf.one_hot(y, depth=5, dtype=tf.float32)
    return img, y


def _make_tfrec_ds(file_list):
    ds_files = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds_files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
        cycle_length=min(8, len(file_list)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(
        _parse_tfrec_with_label, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


options = tf.data.Options()
options.autotune.enabled = True
options.deterministic = True

base_ds = _make_tfrec_ds(train_tfrecords).with_options(options)

num_records = int(
    base_ds.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy()
)
if num_records <= 0:
    raise RuntimeError("No records found when reading TFRecords.")

split_size = int(num_records * train_size)
if split_size <= 0 or split_size >= num_records:
    raise RuntimeError(
        f"Bad split computed from num_records={num_records}: split_size={split_size}"
    )

enum_ds = base_ds.enumerate()

train_ds = enum_ds.filter(lambda i, xy: i < split_size).map(
    lambda i, xy: xy, num_parallel_calls=AUTOTUNE
)
val_ds = enum_ds.filter(lambda i, xy: i >= split_size).map(
    lambda i, xy: xy, num_parallel_calls=AUTOTUNE
)

cache_dir = "/kaggle/working/tf_cache"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(
    cache_dir, f"train_tfrec_{IMG_H}x{IMG_W}_bs{BATCH_SIZE}.cache"
)
val_cache_path = os.path.join(
    cache_dir, f"val_tfrec_{IMG_H}x{IMG_W}_bs{BATCH_SIZE}.cache"
)

_SHUFFLE_BUFFER = min(split_size, 4096)

train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.shuffle(
    buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

test_ds = val_ds.cache(val_cache_path)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

steps_per_epoch = int(np.ceil(split_size / BATCH_SIZE))
validation_steps = int(np.ceil((num_records - split_size) / BATCH_SIZE))
if steps_per_epoch <= 0 or validation_steps <= 0:
    raise RuntimeError(
        f"Non-positive steps computed: steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}"
    )

print("TFRecord samples total:", num_records)
print("Train samples:", split_size, "Val samples:", num_records - split_size)
print("IMG:", (IMG_H, IMG_W), "Batch:", BATCH_SIZE, "Shuffle buffer:", _SHUFFLE_BUFFER)
print("TFRecords:", len(train_tfrecords), "Train cache:", train_cache_path)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1646965352.py in <cell line: 0>()
      4 
      5 BATCH_SIZE = 32
----> 6 AUTOTUNE = tf.data.AUTOTUNE
      7 
      8 tfrecord_train_dir = os.path.join(BASE_DIR, "train_tfrecords")

NameError: name 'tf' is not defined

## === cell 3
def define_directory():
    base = "/kaggle/working/Training"
    subdirs = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    if not os.path.exists(base):
        os.mkdir(base)
    for sd in subdirs:
        p = os.path.join(base, sd)
        if not os.path.exists(p):
            os.mkdir(p)




## === cell 4
pre_trained_model = VGG16(
    include_top=False, weights="imagenet", input_shape=(IMG_H, IMG_W, 3)
)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer("block5_pool")
last_output = last_layer.output

x = layers.Reshape((-1,))(last_output)
x = layers.Dense(5, activation="softmax")(x)

model = keras.Model(pre_trained_model.input, x)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

model.summary()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1523614606.py in <cell line: 0>()
----> 1 pre_trained_model = VGG16(
      2     include_top=False, weights="imagenet", input_shape=(IMG_H, IMG_W, 3)
      3 )
      4 
      5 for layer in pre_trained_model.layers:

NameError: name 'VGG16' is not defined

## === cell 5
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2455570372.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_ds,
      3     validation_data=test_ds,
      4     epochs=10,
      5     verbose=2,

NameError: name 'model' is not defined

## === cell 6
test_dir = os.path.join(BASE_DIR, "test_images")

sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
ordered_test_ids = sample_sub["image_id"].astype(str).tolist()


def _load_test_image(image_id):
    img_path = tf.strings.join([tf.constant(test_dir + "/"), image_id])
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img


test_id_ds = tf.data.Dataset.from_tensor_slices(tf.constant(ordered_test_ids))
test_img_ds = test_id_ds.map(
    _load_test_image, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_img_ds = test_img_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

pred_probs = model.predict(test_img_ds, verbose=0)
pred_labels = pred_probs.argmax(axis=1).astype(int).tolist()

if len(pred_labels) != len(ordered_test_ids):
    raise RuntimeError(
        f"Predicted {len(pred_labels)} test items but expected {len(ordered_test_ids)}."
    )

submission_df = pd.DataFrame({"image_id": ordered_test_ids, "label": pred_labels})
submission_path = "./submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
assert submission_path.endswith(".csv")
assert submission_df.shape[0] == sample_sub.shape[0]
assert submission_df.columns.tolist() == ["image_id", "label"]

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1107197004.py in <cell line: 0>()
     16 
     17 
---> 18 test_id_ds = tf.data.Dataset.from_tensor_slices(tf.constant(ordered_test_ids))
     19 test_img_ds = test_id_ds.map(
     20     _load_test_image, num_parallel_calls=AUTOTUNE, deterministic=True

NameError: name 'tf' is not defined

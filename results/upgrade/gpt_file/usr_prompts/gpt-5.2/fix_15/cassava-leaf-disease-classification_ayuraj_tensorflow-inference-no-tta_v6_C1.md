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

0.8164097914777878

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05531) has done: 'The main timeout driver is the input pipeline: you currently (1) shuffle TFRecord shards for train/val (which breaks the `steps_per_epoch` math based on `train.csv` split), forcing extra filtering/reading work, and (2) iterate the entire test dataset once just to collect IDs, then iterate again for prediction. I make the train/val TFRecord split deterministic and consistent with the `train.csv` split by filtering TFRecords with a precomputed “train/val membership” lookup table (same semantics as your existing split, just avoids wasted reads and fixes step alignment). I also make test inference single-pass by emitting `(pos, probs)` per batch and scattering into the final array, eliminating the extra full dataset traversal and large host-side concatenation. All changes preserve model architecture/training loops/loss and keep determinism.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism enable failed (non-fatal):", e)

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print("GPU mem growth set failed (non-fatal):", e)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT enable failed (non-fatal):", e)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing {TEST_TFREC_DIR}"

IMAGE_SIZE = 380
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
CLASS_NUMS = 5

train_df = pd.read_csv(TRAIN_CSV)
print("Train rows:", len(train_df), "columns:", train_df.columns.tolist())
print("Label distribution:\n", train_df["label"].value_counts().sort_index())




## === cell 2
tf.keras.utils.set_random_seed(42)

train_image_ids = train_df["image_id"].astype(str).to_numpy()
train_labels = train_df["label"].astype("int32").to_numpy()

idx = np.arange(len(train_image_ids))
rng = np.random.default_rng(42)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_image_ids, va_image_ids = train_image_ids[tr_idx], train_image_ids[va_idx]
tr_labels, va_labels = train_labels[tr_idx], train_labels[va_idx]

label_map = dict(zip(train_image_ids.tolist(), train_labels.tolist()))
label_keys = tf.constant(list(label_map.keys()), dtype=tf.string)
label_vals = tf.constant(list(label_map.values()), dtype=tf.int32)
label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(label_keys, label_vals),
    default_value=tf.constant(-1, dtype=tf.int32),
)

tr_keys = tf.constant(tr_image_ids.tolist(), dtype=tf.string)
va_keys = tf.constant(va_image_ids.tolist(), dtype=tf.string)
ones_tr = tf.ones([tf.shape(tr_keys)[0]], dtype=tf.int8)
ones_va = tf.ones([tf.shape(va_keys)[0]], dtype=tf.int8)
train_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tr_keys, ones_tr),
    default_value=tf.constant(0, dtype=tf.int8),
)
val_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(va_keys, ones_va),
    default_value=tf.constant(0, dtype=tf.int8),
)

print("Train/Val sizes:", len(tr_image_ids), len(va_image_ids))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/1992028994.py in <cell line: 0>()
     31 ones_tr = tf.ones([tf.shape(tr_keys)[0]], dtype=tf.int8)
     32 ones_va = tf.ones([tf.shape(va_keys)[0]], dtype=tf.int8)
---> 33 train_id_table = tf.lookup.StaticHashTable(
     34     tf.lookup.KeyValueTensorInitializer(tr_keys, ones_tr),
     35     default_value=tf.constant(0, dtype=tf.int8),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in __call__(cls, *args, **kwargs)
    101       previous_getter = _make_getter(getter, previous_getter)
    102 
--> 103     return previous_getter(*args, **kwargs)
    104 
    105 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in <lambda>(*a, **kw)
     96       return obj
     97 
---> 98     previous_getter = lambda *a, **kw: default_resource_creator(None, *a, **kw)
     99     resource_creator_stack = ops.get_default_graph()._resource_creator_stack
    100     for getter in resource_creator_stack[cls._resource_type()]:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in default_resource_creator(next_creator, *a, **kw)
     93       assert next_creator is None
     94       obj = cls.__new__(cls, *a, **kw)
---> 95       obj.__init__(*a, **kw)
     96       return obj
     97 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in __init__(self, initializer, default_value, name, experimental_is_anonymous)
    351     self._name = name or "hash_table"
    352     self._table_name = None
--> 353     super(StaticHashTable, self).__init__(default_value, initializer)
    354     self._value_shape = self._default_value.get_shape()
    355 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in __init__(self, default_value, initializer)
    199       self._initializer = self._track_trackable(initializer, "_initializer")
    200     with ops.init_scope():
--> 201       self._resource_handle = self._create_resource()
    202     if (not context.executing_eagerly() and
    203         ops.get_default_graph()._get_control_flow_context() is not None):  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in _create_resource(self)
    361           name=self._name)
    362     else:
--> 363       table_ref = gen_lookup_ops.hash_table_v2(
    364           shared_name=self._shared_name,
    365           key_dtype=self._initializer.key_dtype,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_lookup_ops.py in hash_table_v2(key_dtype, value_dtype, container, shared_name, use_node_name_sharing, name)
    483       return _result
    484     except _core._NotOkStatusException as e:
--> 485       _ops.raise_from_not_ok_status(e, name)
    486     except _core._FallbackException:
    487       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

NotFoundError: Could not find device for node: {{node HashTableV2}} = HashTableV2[container="", key_dtype=DT_STRING, shared_name="27", use_node_name_sharing=false, value_dtype=DT_INT8]
All kernels registered for op HashTableV2:
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT64]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_DOUBLE]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_BOOL]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_INT64]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_DOUBLE]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_DOUBLE]
 [Op:HashTableV2] name: hash_table

## === cell 3
TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


@tf.function
def _parse_id_and_label(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    image_id = ex["image_name"]
    y = label_table.lookup(image_id)
    return image_id, tf.cast(y, tf.int32), ex["image"]


@tf.function
def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(
        img, (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    return img


@tf.function
def _to_img_label(image_id, y, image_bytes):
    img = _decode_and_resize(image_bytes)
    return img, y


@tf.function
def _parse_tfrecord_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = tf.io.decode_jpeg(ex["image"], channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(
        img, (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    image_id = ex["image_name"]
    return img, image_id


@tf.function
def augment_if_training(img, label=None, training=False):
    if training:
        img = tf.image.random_flip_left_right(img)
    if label is None:
        return img
    return img, label


def _list_tfrecs(tfrecs_dir):
    files = tf.io.gfile.glob(os.path.join(tfrecs_dir, "*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .tfrec files found in {tfrecs_dir}")
    return files


def _make_ds_from_tfrecs(files, training, membership_table):
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        if hasattr(options.experimental_optimization, "autotune_buffers"):
            options.experimental_optimization.autotune_buffers = True
        if hasattr(options.experimental_optimization, "apply_default_optimizations"):
            options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    files = list(files)
    ds_files = tf.data.Dataset.from_tensor_slices(files).with_options(options)

    ds = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=1),
        cycle_length=min(len(files), 8) if len(files) > 0 else 1,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.map(_parse_id_and_label, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.filter(lambda image_id, y, image_bytes: y >= 0)
    ds = ds.filter(
        lambda image_id, y, image_bytes: membership_table.lookup(image_id) > 0
    )

    ds = ds.map(_to_img_label, num_parallel_calls=AUTOTUNE, deterministic=True)

    if training:
        ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda img, y: augment_if_training(img, y, training=True),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds.with_options(options)


all_train_tfrecs = _list_tfrecs(TRAIN_TFREC_DIR)

train_ds = _make_ds_from_tfrecs(
    all_train_tfrecs, training=True, membership_table=train_id_table
)
val_ds = _make_ds_from_tfrecs(
    all_train_tfrecs, training=False, membership_table=val_id_table
)

steps_per_epoch = int(np.ceil(len(tr_image_ids) / BATCH_SIZE))
validation_steps = int(np.ceil(len(va_image_ids) / BATCH_SIZE))

print(
    "TFRecord shards:",
    len(all_train_tfrecs),
    "-> using all shards with ID-based filtering",
)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3218879545.py in <cell line: 0>()
    115 # and avoids decoding images that don't belong to the split.
    116 train_ds = _make_ds_from_tfrecs(
--> 117     all_train_tfrecs, training=True, membership_table=train_id_table
    118 )
    119 val_ds = _make_ds_from_tfrecs(

NameError: name 'train_id_table' is not defined

## === cell 4
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling="avg",
)
base.trainable = False  # minimal training time and stable behavior

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = inputs
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(CLASS_NUMS, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()




## === cell 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/703762443.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_ds,
      3     validation_data=val_ds,
      4     epochs=2,
      5     steps_per_epoch=steps_per_epoch,

NameError: name 'train_ds' is not defined

## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
test_images = sample_sub["image_id"].tolist()  # canonical ordering

assert os.path.isdir(TEST_IMG_DIR), f"Missing test image directory: {TEST_IMG_DIR}"


def make_test_ds_from_tfrecs():
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        if hasattr(options.experimental_optimization, "autotune_buffers"):
            options.experimental_optimization.autotune_buffers = True
        if hasattr(options.experimental_optimization, "apply_default_optimizations"):
            options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    files = _list_tfrecs(TEST_TFREC_DIR)
    ds_files = tf.data.Dataset.from_tensor_slices(files).with_options(options)

    ds = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=1),
        cycle_length=min(len(files), 8) if len(files) > 0 else 1,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(_parse_tfrecord_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds.with_options(options)


test_ds_with_ids = make_test_ds_from_tfrecs()

test_ids_tf = tf.constant(test_images, dtype=tf.string)
test_pos_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        test_ids_tf, tf.range(tf.shape(test_ids_tf)[0], dtype=tf.int32)
    ),
    default_value=tf.constant(-1, dtype=tf.int32),
)

reordered_probs = tf.Variable(
    tf.zeros([len(test_images), CLASS_NUMS], dtype=tf.float32), trainable=False
)


@tf.function
def _infer_step(imgs, ids):
    probs = model(imgs, training=False)
    pos = test_pos_table.lookup(ids)
    return probs, pos


seen_total = 0
for batch_imgs, batch_ids in test_ds_with_ids:
    batch_probs, batch_pos = _infer_step(batch_imgs, batch_ids)
    valid = batch_pos >= 0
    batch_pos = tf.boolean_mask(batch_pos, valid)
    batch_probs = tf.boolean_mask(batch_probs, valid)
    reordered_probs.scatter_nd_update(tf.expand_dims(batch_pos, 1), batch_probs)
    seen_total += int(batch_pos.shape[0])

n_test = len(test_images)
assert (
    seen_total == n_test
), f"Failed to align all test ids: matched {seen_total}/{n_test}"

predictions = (
    tf.argmax(reordered_probs, axis=1, output_type=tf.int32)
    .numpy()
    .astype(int)
    .tolist()
)

print("Num test images:", len(test_images))
print("Num predictions:", len(predictions))
assert len(test_images) == len(
    predictions
), "Prediction count must match test image count"




## === cell 7
predictions[:20], test_images[:5]




## === cell 8
sub = pd.DataFrame({"image_id": test_images, "label": predictions})
print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))

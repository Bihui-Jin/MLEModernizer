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

0.6140828044726504

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")

print("TensorFlow:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train images dir exists:", os.path.exists(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.exists(TEST_IMG_DIR))

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

train_df["filepath"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)
sub_df["filepath"] = TEST_IMG_DIR + "/" + sub_df["image_id"].astype(str)

print("Train rows:", len(train_df))
print("Test rows:", len(sub_df))

num_classes = int(train_df["label"].nunique())
print("Num classes:", num_classes)
print(train_df.head())




## === cell 2
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5  # keep modest for runtime; no early stopping introduced

from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_df["filepath"].values,
    train_df["label"].values,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"].values,
)

print("Train/Val sizes:", len(train_paths), len(val_paths))


def _read_bytes(path):
    return tf.io.read_file(path)


def _decode_and_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet.preprocess_input(img)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    return _decode_and_resize_from_bytes(img_bytes)


def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED)
    return img


def _decode_with_label(path, label):
    return _decode_and_resize(path), tf.cast(label, tf.int32)


def _decode_only(path):
    return _decode_and_resize(path)


@tf.function
def _augment_with_label(img, label):
    img = _augment(img)
    return img, tf.cast(label, tf.int32)


def _bytes_with_label(path, label):
    return _read_bytes(path), tf.cast(label, tf.int32)


def _bytes_only(path):
    return _read_bytes(path)


def _decode_bytes_with_label(img_bytes, label):
    return _decode_and_resize_from_bytes(img_bytes), tf.cast(label, tf.int32)


def _decode_bytes_only(img_bytes):
    return _decode_and_resize_from_bytes(img_bytes)


DS_OPTIONS = tf.data.Options()
DS_OPTIONS.experimental_deterministic = False
try:
    DS_OPTIONS.experimental_optimization.map_parallelization = True
    DS_OPTIONS.experimental_optimization.parallel_batch = True
    DS_OPTIONS.experimental_optimization.autotune_buffers = True
except Exception:
    pass


def make_ds(paths, labels=None, training=False, cache=False, cache_name=""):
    """
    cache:
      - If True: uses in-memory cache unless cache_name is given, then file cache.
    Note: We cache JPEG bytes (small) rather than decoded tensors (huge), for speed.
    """
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(DS_OPTIONS)
        ds = ds.map(
            _bytes_only, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )
        ds = ds.apply(tf.data.experimental.ignore_errors())
        if cache:
            ds = ds.cache(cache_name if cache_name else None)
        ds = ds.map(
            _decode_bytes_only, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        try:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        except Exception:
            ds = ds.prefetch(1)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(DS_OPTIONS)
    ds = ds.map(
        _bytes_with_label, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache:
        ds = ds.cache(cache_name if cache_name else None)

    ds = ds.map(
        _decode_bytes_with_label,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            _augment_with_label,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    try:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    except Exception:
        ds = ds.prefetch(1)
    return ds


train_ds = make_ds(train_paths, train_labels, training=True, cache=True, cache_name="")
val_ds = make_ds(val_paths, val_labels, training=False, cache=True, cache_name="")

base = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # keep stable and fast

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

model.summary()

TRAIN_STEPS = int(np.ceil(len(train_paths) / BATCH_SIZE))
VAL_STEPS = int(np.ceil(len(val_paths) / BATCH_SIZE))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/760705809.py in <cell line: 0>()
    144 # --- Speed fix: in-memory byte cache is typically faster than file cache and avoids disk bottlenecks.
    145 # Correctness preserved (cache only stores file contents, not transformed images).
--> 146 train_ds = make_ds(train_paths, train_labels, training=True, cache=True, cache_name="")
    147 val_ds = make_ds(val_paths, val_labels, training=False, cache=True, cache_name="")
    148 

/tmp/ipykernel_11/760705809.py in make_ds(paths, labels, training, cache, cache_name)
    117 
    118     if cache:
--> 119         ds = ds.cache(cache_name if cache_name else None)
    120 
    121     # Decode after caching bytes (small), preserving the exact same decode+resize+preprocess ops.

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in cache(self, filename, name)
   1566     # pylint: disable=g-import-not-at-top,protected-access
   1567     from tensorflow.python.data.ops import cache_op
-> 1568     return cache_op._cache(self, filename, name)
   1569     # pylint: enable=g-import-not-at-top,protected-access
   1570 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/cache_op.py in _cache(input_dataset, filename, name)
     24 
     25 def _cache(input_dataset, filename, name):  # pylint: disable=unused-private-name
---> 26   return CacheDataset(input_dataset, filename, name)
     27 
     28 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/cache_op.py in __init__(self, input_dataset, filename, name)
     33     """See `Dataset.cache()` for details."""
     34     self._input_dataset = input_dataset
---> 35     self._filename = ops.convert_to_tensor(
     36         filename, dtype=dtypes.string, name="filename")
     37     self._name = name

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    232 
    233     if ret is None:
--> 234       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
    235 
    236     if ret is NotImplemented:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_tensor_conversion.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
     27 
     28   _ = as_ref
---> 29   return constant_op.constant(v, dtype=dtype, name=name)
     30 
     31 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
    140   def wrapper(*args, **kwargs):
    141     if not ops.is_auto_dtype_conversion_enabled():
--> 142       return op(*args, **kwargs)
    143     bound_arguments = signature.bind(*args, **kwargs)
    144     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
    274     ValueError: if called on a symbolic tensor.
    275   """
--> 276   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    277                         allow_broadcast=True)
    278 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
    287       with trace.Trace("tf.constant"):
    288         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
--> 289     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    290 
    291   const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    299 ) -> ops._EagerTensorBase:
    300   """Creates a constant on the current device."""
--> 301   t = convert_to_eager_tensor(value, ctx, dtype)
    302   if shape is None:
    303     return t

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
    106       dtype = dtypes.as_dtype(dtype).as_datatype_enum
    107   ctx.ensure_initialized()
--> 108   return ops.EagerTensor(value, ctx.device_name, dtype)
    109 
    110 

ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.

## === cell 3
history = model.fit(
    train_ds.repeat(),
    validation_data=val_ds.repeat(),
    epochs=EPOCHS,
    steps_per_epoch=TRAIN_STEPS,
    validation_steps=VAL_STEPS,
    verbose=2,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

history_ft = model.fit(
    train_ds.repeat(),
    validation_data=val_ds.repeat(),
    epochs=2,
    steps_per_epoch=TRAIN_STEPS,
    validation_steps=VAL_STEPS,
    verbose=2,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/822205642.py in <cell line: 0>()
      1 # --- Speed fix: repeat with explicit step counts (same samples per epoch as before).
----> 2 history = model.fit(
      3     train_ds.repeat(),
      4     validation_data=val_ds.repeat(),
      5     epochs=EPOCHS,

NameError: name 'model' is not defined

## === cell 4
test_paths = sub_df["filepath"].values

test_ds = make_ds(test_paths, labels=None, training=False, cache=False)

probs = model.predict(test_ds, verbose=0)
preds = np.argmax(probs, axis=1).astype(int)

submission = sub_df[["image_id"]].copy()
submission["label"] = preds

print(submission.head())
print("Submission shape:", submission.shape)
print("Unique predicted labels:", np.unique(submission["label"].values))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2376755266.py in <cell line: 0>()
      4 test_ds = make_ds(test_paths, labels=None, training=False, cache=False)
      5 
----> 6 probs = model.predict(test_ds, verbose=0)
      7 preds = np.argmax(probs, axis=1).astype(int)
      8 

NameError: name 'model' is not defined

## === cell 5
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(submission), "rows")
print("File exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/245161775.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with", len(submission), "rows")
      3 print("File exists:", os.path.exists("submission.csv"))

NameError: name 'submission' is not defined

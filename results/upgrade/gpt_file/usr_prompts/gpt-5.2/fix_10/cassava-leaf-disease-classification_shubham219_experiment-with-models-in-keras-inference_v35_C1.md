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

0.5666364460562103

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

SEED = 42
DEBUG = False

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"




## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import train_test_split

tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)

if DEBUG:
    train_df = train_df[train_df["path"].map(os.path.exists)].reset_index(drop=True)

num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

trn_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"],
)

if DEBUG:
    trn_df = trn_df.sample(512, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(256, random_state=SEED).reset_index(drop=True)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.deterministic = True
DATASET_OPTIONS.experimental_optimization.map_parallelization = True
DATASET_OPTIONS.experimental_optimization.map_and_batch_fusion = True
DATASET_OPTIONS.experimental_optimization.parallel_batch = True
DATASET_OPTIONS.experimental_optimization.autotune_buffers = True

IMG_SIZE_T = tf.constant(IMG_SIZE, dtype=tf.int32)
IMG_H_F = tf.cast(IMG_SIZE_T[0], tf.float32)
IMG_W_F = tf.cast(IMG_SIZE_T[1], tf.float32)

_ROT_RAD = 10.0 * np.pi / 180.0
_SHIFT_FRAC = 0.05
_ZOOM_RANGE = 0.1

_ROT_RAD_T = tf.constant(_ROT_RAD, tf.float32)
_SHIFT_FRAC_T = tf.constant(_SHIFT_FRAC, tf.float32)
_ZOOM_RANGE_T = tf.constant(_ZOOM_RANGE, tf.float32)
SEED_T = tf.constant(SEED, tf.int32)


@tf.function
def _read_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE_T, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1./255
    img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


@tf.function
def _apply_affine(img, rot, tx, ty, zoom):
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_t = tf.cos(rot)
    sin_t = tf.sin(rot)
    s = zoom

    inv00 = cos_t / s
    inv01 = sin_t / s
    inv10 = -sin_t / s
    inv11 = cos_t / s

    a0 = inv00
    a1 = inv01
    a2 = cx - inv00 * (cx + tx) - inv01 * (cy + ty)
    b0 = inv10
    b1 = inv11
    b2 = cy - inv10 * (cx + tx) - inv11 * (cy + ty)

    transforms = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transforms,
        output_shape=IMG_SIZE_T,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    out = tf.ensure_shape(out, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return out


@tf.function
def _augment(img, seed_pair):
    flip = (
        tf.random.stateless_uniform([], seed=seed_pair + tf.constant([1, 0], tf.int32))
        < 0.5
    )
    img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)

    rot = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([2, 0], tf.int32),
        minval=-_ROT_RAD_T,
        maxval=_ROT_RAD_T,
        dtype=tf.float32,
    )

    tx = (
        tf.random.stateless_uniform(
            [],
            seed=seed_pair + tf.constant([3, 0], tf.int32),
            minval=-_SHIFT_FRAC_T,
            maxval=_SHIFT_FRAC_T,
            dtype=tf.float32,
        )
        * IMG_W_F
    )
    ty = (
        tf.random.stateless_uniform(
            [],
            seed=seed_pair + tf.constant([4, 0], tf.int32),
            minval=-_SHIFT_FRAC_T,
            maxval=_SHIFT_FRAC_T,
            dtype=tf.float32,
        )
        * IMG_H_F
    )

    zoom = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([5, 0], tf.int32),
        minval=1.0 - _ZOOM_RANGE_T,
        maxval=1.0 + _ZOOM_RANGE_T,
        dtype=tf.float32,
    )

    img = _apply_affine(img, rot=rot, tx=tx, ty=ty, zoom=zoom)
    return img


def _make_train_ds(paths, labels, batch_size):
    paths_np = np.asarray(paths)
    labels_np = np.asarray(labels).astype(np.int32)

    keys = tf.constant(paths_np)
    vals = tf.constant(labels_np, dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=tf.constant(-1, tf.int32),
    )

    ds = tf.data.Dataset.from_tensor_slices(keys)
    ds = ds.with_options(DATASET_OPTIONS)
    ds = ds.shuffle(buffer_size=len(paths_np), seed=SEED, reshuffle_each_iteration=True)

    @tf.function
    def _map_fn(path):
        label = table.lookup(path)
        img = _read_and_resize(path)
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed_pair = tf.stack([tf.cast(h, tf.int32), SEED_T])
        img = _augment(img, seed_pair)
        return img, tf.cast(label, tf.float32)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(paths, labels, batch_size):
    paths = tf.convert_to_tensor(np.asarray(paths))
    labels = tf.convert_to_tensor(np.asarray(labels).astype(np.int32), dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(DATASET_OPTIONS)

    @tf.function
    def _map_fn(path, label):
        img = _read_and_resize(path)
        return img, tf.cast(label, tf.float32)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


steps_per_epoch = int(np.ceil(len(trn_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

train_ds = _make_train_ds(trn_df["path"].values, trn_df["label"].values, BATCH_SIZE)
val_ds = _make_val_ds(val_df["path"].values, val_df["label"].values, BATCH_SIZE)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3862176596.py in <cell line: 0>()
     32 # Removing it is score-neutral (it is only an optimization knob).
     33 # DATASET_OPTIONS.experimental_optimization.parallel_filter = True
---> 34 DATASET_OPTIONS.experimental_optimization.autotune_buffers = True
     35 
     36 IMG_SIZE_T = tf.constant(IMG_SIZE, dtype=tf.int32)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 3
base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # keep fast and stable within time limits

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(5, activation="softmax")(x)
my_model = keras.Model(inputs, outputs)

my_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 2 if not DEBUG else 1

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1910001710.py in <cell line: 0>()
     22 
     23 history = my_model.fit(
---> 24     train_ds,
     25     validation_data=val_ds,
     26     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["path"] = TEST_IMG_DIR.rstrip("/") + "/" + sample_sub["image_id"].astype(str)

if DEBUG:
    missing = sample_sub[~sample_sub["path"].map(os.path.exists)]
    if len(missing) > 0:
        raise FileNotFoundError(
            f"Some test images referenced by sample_submission are missing. Example:\n{missing.head()}"
        )

TEST_BATCH = 128


def _make_test_ds(paths, batch_size):
    paths = tf.convert_to_tensor(np.asarray(paths))

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(DATASET_OPTIONS)

    @tf.function
    def _map_fn(path):
        img = _read_and_resize(path)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_steps = int(np.ceil(len(sample_sub) / TEST_BATCH))
test_ds = _make_test_ds(sample_sub["path"].values, TEST_BATCH)

pred = my_model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
pred_labels = np.argmax(pred, axis=-1).astype(int)

submission = sample_sub[["image_id"]].copy()
submission["label"] = pred_labels
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]
assert os.path.exists("submission.csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/67875204.py in <cell line: 0>()
     31 
     32 test_steps = int(np.ceil(len(sample_sub) / TEST_BATCH))
---> 33 test_ds = _make_test_ds(sample_sub["path"].values, TEST_BATCH)
     34 
     35 pred = my_model.predict(

/tmp/ipykernel_11/67875204.py in _make_test_ds(paths, batch_size)
     23         return img
     24 
---> 25     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
     26     ds = ds.cache()
     27     ds = ds.batch(batch_size, drop_remainder=False)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_fileyh3lv63y.py in tf___map_fn(path)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(_read_and_resize), (ag__.ld(path),), None, fscope)
     11                 try:
     12                     do_return = True

NameError: in user code:

    File "/tmp/ipykernel_11/67875204.py", line 22, in _map_fn  *
        img = _read_and_resize(path)

    NameError: name '_read_and_resize' is not defined

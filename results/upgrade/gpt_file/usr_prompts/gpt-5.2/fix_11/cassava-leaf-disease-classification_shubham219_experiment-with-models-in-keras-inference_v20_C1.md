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

0.710486551828347

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")

print("TF version:", tf.__version__)
print("Base path exists:", os.path.exists(BASE_PATH))
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train img dir exists:", os.path.exists(TRAIN_IMG_DIR))
print("Test img dir exists:", os.path.exists(TEST_IMG_DIR))
print("Train tfrec dir exists:", os.path.exists(TRAIN_TFREC_DIR))
print("Test tfrec dir exists:", os.path.exists(TEST_TFREC_DIR))

assert os.path.isdir(TRAIN_IMG_DIR), "TRAIN_IMG_DIR not found"
assert os.path.isdir(TEST_IMG_DIR), "TEST_IMG_DIR not found"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = (300, 300)
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 2  # keep as-is per original core logic

df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(str)
df_train["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + df_train["image_id"].astype(str)
df_train = df_train.reset_index(drop=True)

idx = np.arange(len(df_train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]
train_df = df_train.iloc[train_idx].reset_index(drop=True)
val_df = df_train.iloc[val_idx].reset_index(drop=True)

print("Train/Val sizes:", len(train_df), len(val_df))
print("Train label sample:", train_df["label"].value_counts().sort_index().to_dict())

AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_resize_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    return _decode_and_resize_bytes(img)


_ROT_MAX = 10.0 * np.pi / 180.0
_WSHIFT = 0.05
_HSHIFT = 0.05
_ZOOM = 0.1


def _augment_fast(img, seed):
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)  # shape [2]

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    s0, s1, s2, s3 = tf.unstack(
        tf.random.experimental.stateless_split(seed, num=4), axis=0
    )

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    dx = tf.random.stateless_uniform((), s0, -_WSHIFT, _WSHIFT, dtype=tf.float32) * w
    dy = tf.random.stateless_uniform((), s1, -_HSHIFT, _HSHIFT, dtype=tf.float32) * h
    z = tf.random.stateless_uniform((), s2, 1.0 - _ZOOM, 1.0 + _ZOOM, dtype=tf.float32)
    ang = tf.random.stateless_uniform((), s3, -_ROT_MAX, _ROT_MAX, dtype=tf.float32)

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.cos(ang)
    sin_a = tf.sin(ang)
    inv_z = 1.0 / z
    a0 = cos_a * inv_z
    a1 = -sin_a * inv_z
    a3 = sin_a * inv_z
    a4 = cos_a * inv_z

    tx = cx - (a0 * cx + a1 * cy) - dx
    ty = cy - (a3 * cx + a4 * cy) - dy

    transform = tf.stack([a0, a1, tx, a3, a4, ty, 0.0, 0.0])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform[None, ...],
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


_data_opts = tf.data.Options()
_data_opts.experimental_deterministic = True
try:
    _data_opts.experimental_slack = True
except Exception:
    pass
try:
    _data_opts.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
except Exception:
    pass


_TFREC_IMG_KEY = "image"
_TFREC_LABEL_KEY = "target"


def _parse_train_tfrecord(example_proto):
    feat = {
        _TFREC_IMG_KEY: tf.io.FixedLenFeature([], tf.string),
        _TFREC_LABEL_KEY: tf.io.FixedLenFeature([], tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, feat)
    img = _decode_and_resize_bytes(x[_TFREC_IMG_KEY])
    label = tf.cast(x[_TFREC_LABEL_KEY], tf.int32)
    return img, label


def _parse_test_tfrecord(example_proto):
    feat = {_TFREC_IMG_KEY: tf.io.FixedLenFeature([], tf.string)}
    x = tf.io.parse_single_example(example_proto, feat)
    img = _decode_and_resize_bytes(x[_TFREC_IMG_KEY])
    return img


def _tfrecord_files(folder, pattern="*.tfrec"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "*.tfrecord")))
    return files


def _count_tfrecord_examples(files):
    total = 0
    for f in files:
        total += sum(1 for _ in tf.data.TFRecordDataset(f))
    return total


def make_train_val_datasets(train_df, val_df, batch_size):
    train_files = _tfrecord_files(TRAIN_TFREC_DIR)
    assert train_files, "No TFRecord files found in train_tfrecords."
    n_files = len(train_files)
    split_files = int(0.9 * n_files)
    tr_files = train_files[:split_files]
    va_files = train_files[split_files:] if split_files < n_files else train_files[-1:]

    n_train = _count_tfrecord_examples(tr_files)
    n_val = _count_tfrecord_examples(va_files)

    ds_train = tf.data.TFRecordDataset(tr_files, num_parallel_reads=AUTOTUNE)
    ds_train = ds_train.map(
        _parse_train_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds_train = ds_train.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    rng = tf.random.Generator.from_seed(SEED)
    ds_seed = (
        tf.data.Dataset.from_tensors(0)
        .repeat()
        .map(lambda _: rng.make_seeds(2)[0], num_parallel_calls=1, deterministic=True)
    )
    ds_train = tf.data.Dataset.zip((ds_train, ds_seed))
    ds_train = ds_train.map(
        lambda xy, sd: (_augment_fast(xy[0], sd), xy[1]),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds_train = ds_train.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    ds_train = ds_train.with_options(_data_opts)

    ds_val = tf.data.TFRecordDataset(va_files, num_parallel_reads=AUTOTUNE)
    ds_val = ds_val.map(
        _parse_train_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds_val = ds_val.with_options(_data_opts)

    steps_per_epoch = n_train // batch_size
    validation_steps = int(np.ceil(n_val / batch_size))

    return ds_train, ds_val, steps_per_epoch, validation_steps


train_ds, val_ds, STEPS_PER_EPOCH, VAL_STEPS = make_train_val_datasets(
    train_df, val_df, BATCH_SIZE
)

unique_labels = set(train_df["label"].unique().tolist())
assert unique_labels == set(
    map(str, range(NUM_CLASSES))
), "Unexpected number of classes parsed."

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    verbose=1,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2535754631.py in <cell line: 0>()
    194 
    195 
--> 196 train_ds, val_ds, STEPS_PER_EPOCH, VAL_STEPS = make_train_val_datasets(
    197     train_df, val_df, BATCH_SIZE
    198 )

/tmp/ipykernel_11/2535754631.py in make_train_val_datasets(train_df, val_df, batch_size)
    169     )
    170     ds_train = tf.data.Dataset.zip((ds_train, ds_seed))
--> 171     ds_train = ds_train.map(
    172         lambda xy, sd: (_augment_fast(xy[0], sd), xy[1]),
    173         num_parallel_calls=AUTOTUNE,

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filegyaz8vr0.py in <lambda>(xy, sd)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda xy, sd: ag__.with_function_scope(lambda lscope: (ag__.converted_call(_augment_fast, (xy[0], sd), None, lscope), xy[1]), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filegyaz8vr0.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda xy, sd: ag__.with_function_scope(lambda lscope: (ag__.converted_call(_augment_fast, (xy[0], sd), None, lscope), xy[1]), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_file8xx5bv79.py in tf___augment_fast(img, seed)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 seed = ag__.converted_call(ag__.ld(tf).convert_to_tensor, (ag__.ld(seed),), dict(dtype=ag__.ld(tf).int32), fscope)
     11                 img = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(img),), dict(seed=ag__.ld(seed)), fscope)
     12                 s0, s1, s2, s3 = ag__.converted_call(ag__.ld(tf).unstack, (ag__.converted_call(ag__.ld(tf).random.experimental.stateless_split, (ag__.ld(seed),), dict(num=4), fscope),), dict(axis=0), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    457 
    458   if kwargs is not None:
--> 459     return f(*args, **kwargs)
    460   return f(*args)
    461 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __tf_tensor__(self, dtype, name)
    759       ) -> "Tensor":
    760     if dtype is not None and not dtype.is_compatible_with(self.dtype):
--> 761       raise ValueError(
    762           _add_error_prefix(
    763               f"Tensor conversion requested dtype {dtype.name} "

ValueError: in user code:

    File "/tmp/ipykernel_11/2535754631.py", line 172, in None  *
        lambda xy, sd: (_augment_fast(xy[0], sd), xy[1])
    File "/tmp/ipykernel_11/2535754631.py", line 46, in _augment_fast  *
        seed = tf.convert_to_tensor(seed, dtype=tf.int32)  # shape [2]

    ValueError: Tensor conversion requested dtype int32 for Tensor with dtype int64: <tf.Tensor 'args_2:0' shape=(2,) dtype=int64>


## === cell 2
df_sub = pd.read_csv(SAMPLE_SUB)
df_test = df_sub.copy()
df_test["path"] = TEST_IMG_DIR.rstrip("/") + "/" + df_test["image_id"].astype(str)

print("Test rows:", len(df_test))


def make_test_gen(batch_size=64):
    test_files = _tfrecord_files(TEST_TFREC_DIR)
    assert test_files, "No TFRecord files found in test_tfrecords."
    ds_test = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE)
    ds_test = ds_test.map(
        _parse_test_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds_test = ds_test.with_options(_data_opts)

    n_test = _count_tfrecord_examples(test_files)
    test_steps = int(np.ceil(n_test / batch_size))
    return ds_test, test_steps




## === cell 3
test_gen, TEST_STEPS = make_test_gen(batch_size=128)

pred_test = my_model.predict(
    test_gen,
    steps=TEST_STEPS,
    verbose=1,
)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_test_labels

assert len(final_csv) == len(df_sub), "Submission row count mismatch."
assert list(final_csv.columns) == ["image_id", "label"], "Submission columns mismatch."
assert final_csv["label"].between(0, 4).all(), "Predicted labels out of range [0,4]."

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/711650386.py in <cell line: 0>()
      2 
      3 # CHANGE (timeout fix): provide explicit steps to prevent any extra input pipeline bookkeeping.
----> 4 pred_test = my_model.predict(
      5     test_gen,
      6     steps=TEST_STEPS,

NameError: name 'my_model' is not defined

## === cell 4
print(pd.read_csv("submission.csv").head())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/5545704.py in <cell line: 0>()
----> 1 print(pd.read_csv("submission.csv").head())
      2 print(
      3     "submission.csv exists:",
      4     os.path.exists("submission.csv"),
      5     "size:",

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

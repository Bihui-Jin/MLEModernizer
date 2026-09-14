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

0.5752493200362647

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification",
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset directory."
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

for p in [TRAIN_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR, SAMPLE_SUB]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected path: {p}")

print("Using DATA_ROOT:", DATA_ROOT)
print("Python:", os.sys.version)
print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
WEIGHT_PATH_CANDIDATES = [
    "../input/model-v13/clf_new_26 (1).h5",
    "/kaggle/input/model-v13/clf_new_26 (1).h5",
]
WEIGHT_PATH = next((p for p in WEIGHT_PATH_CANDIDATES if os.path.exists(p)), None)

my_model = None
if WEIGHT_PATH is not None:
    try:
        from tensorflow.keras.models import load_model

        my_model = load_model(WEIGHT_PATH, compile=False)
        print("Loaded pretrained model from:", WEIGHT_PATH)
    except Exception as e:
        print(
            "WARNING: Could not load model weights; will train a fresh model instead."
        )
        print("Load error:", repr(e))
        my_model = None
else:
    print("No pretrained weight file found; will train a fresh model.")



## === cell 2
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 3  # keep short to fit runtime; no early stopping used
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = (
    TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
).astype(str)

if DEBUG:
    train_df = train_df.sample(n=2000, random_state=SEED).reset_index(drop=True)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]
df_tr = train_df.iloc[train_idx].reset_index(drop=True)
df_va = train_df.iloc[val_idx].reset_index(drop=True)

y_tr = df_tr["label"].astype(np.int32).values
y_va = df_va["label"].astype(np.int32).values
x_tr = df_tr["path"].astype(str).values
x_va = df_va["path"].astype(str).values


@tf.function
def _read_decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _random_rotate_reflect(img, seed_pair):
    angle = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([3, 7], tf.int32),
        minval=-15.0 / 180.0 * np.pi,
        maxval=15.0 / 180.0 * np.pi,
        dtype=tf.float32,
    )
    return tf.image.rotate(
        img,
        angles=angle,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )


@tf.function
def _augment(img, seed_pair):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)

    img = _random_rotate_reflect(img, seed_pair)

    dx = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([0, 1], tf.int32),
        minval=-0.1,
        maxval=0.1,
        dtype=tf.float32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([1, 1], tf.int32),
        minval=-0.1,
        maxval=0.1,
        dtype=tf.float32,
    )

    z = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([2, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    new_h = tf.cast(tf.cast(h, tf.float32) / z, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) / z, tf.int32)
    new_h = tf.clip_by_value(new_h, 1, h)
    new_w = tf.clip_by_value(new_w, 1, w)
    img = tf.image.resize_with_crop_or_pad(img, new_h, new_w)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    pad_h = tf.cast(tf.round(tf.cast(h, tf.float32) * 0.1), tf.int32) + 2
    pad_w = tf.cast(tf.round(tf.cast(w, tf.float32) * 0.1), tf.int32) + 2
    img_pad = tf.pad(img, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="REFLECT")
    off_y = tf.cast(tf.round(dy * tf.cast(h, tf.float32)), tf.int32) + pad_h
    off_x = tf.cast(tf.round(dx * tf.cast(w, tf.float32)), tf.int32) + pad_w
    off_y = tf.clip_by_value(off_y, 0, tf.shape(img_pad)[0] - h)
    off_x = tf.clip_by_value(off_x, 0, tf.shape(img_pad)[1] - w)
    img = tf.image.crop_to_bounding_box(img_pad, off_y, off_x, h, w)
    return img


def _base_dataset_options(deterministic: bool):
    opts = tf.data.Options()
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_deterministic = deterministic
    return opts


@tf.function
def _path_to_seedpair(path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    return tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)])


def make_train_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    shuffle_buf = int(min(len(paths), 8192))
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.repeat()

    ds = ds.with_options(_base_dataset_options(deterministic=False))

    def _decode_label_seed(p, y):
        img = _read_decode_resize_rescale(p)
        seed_pair = _path_to_seedpair(p)
        return img, y, seed_pair

    ds = ds.map(_decode_label_seed, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()

    def _aug_map(img, y, seed_pair):
        img = _augment(img, seed_pair)
        return img, y

    ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_base_dataset_options(deterministic=True))

    ds = ds.map(
        lambda p, y: (_read_decode_resize_rescale(p), y),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.cache()
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(x_tr, y_tr, BATCH_SIZE)
val_ds = make_val_ds(x_va, y_va, BATCH_SIZE)

steps_per_epoch = len(x_tr) // BATCH_SIZE
if steps_per_epoch == 0:
    steps_per_epoch = 1
validation_steps = int(np.ceil(len(x_va) / BATCH_SIZE))

if my_model is None:
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    base.trainable = False  # keep training stable/fast

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    my_model = tf.keras.Model(inputs, outputs)

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    my_model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
    )



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3446142981.py in <cell line: 0>()
    167 
    168 
--> 169 train_ds = make_train_ds(x_tr, y_tr, BATCH_SIZE)
    170 val_ds = make_val_ds(x_va, y_va, BATCH_SIZE)
    171 

/tmp/ipykernel_11/3446142981.py in make_train_ds(paths, labels, batch_size)
    144         return img, y
    145 
--> 146     ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)
    147     ds = ds.apply(tf.data.experimental.ignore_errors())
    148     ds = ds.batch(batch_size, drop_remainder=True)

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

/tmp/__autograph_generated_fileioo0dycv.py in tf___aug_map(img, y, seed_pair)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(seed_pair)), None, fscope)
     11                 try:
     12                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_file7j_ku6mg.py in tf___augment(img, seed_pair)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(img),), dict(seed=ag__.ld(seed_pair)), fscope)
---> 11                 img = ag__.converted_call(ag__.ld(_random_rotate_reflect), (ag__.ld(img), ag__.ld(seed_pair)), None, fscope)
     12                 dx = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed_pair) + ag__.converted_call(ag__.ld(tf).constant, ([0, 1], ag__.ld(tf).int32), None, fscope), minval=-0.1, maxval=0.1, dtype=ag__.ld(tf).float32), fscope)
     13                 dy = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed_pair) + ag__.converted_call(ag__.ld(tf).constant, ([1, 1], ag__.ld(tf).int32), None, fscope), minval=-0.1, maxval=0.1, dtype=ag__.ld(tf).float32), fscope)

/tmp/__autograph_generated_fileb1tkxtir.py in tf___random_rotate_reflect(img, seed_pair)
     14                 except:
     15                     do_return = False
---> 16                     raise
     17                 return fscope.ret(retval_, do_return)
     18         return tf___random_rotate_reflect

AttributeError: in user code:

    File "/tmp/ipykernel_11/3446142981.py", line 143, in _aug_map  *
        img = _augment(img, seed_pair)
    File "/tmp/ipykernel_11/3446142981.py", line 62, in _augment  *
        img = _random_rotate_reflect(img, seed_pair)
    File "/tmp/ipykernel_11/3446142981.py", line 54, in _random_rotate_reflect  *
        fill_mode="REFLECT",

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 3
test_images = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
if len(test_images) == 0:
    raise FileNotFoundError(f"No .jpg files found in TEST_IMG_DIR: {TEST_IMG_DIR}")
test_images = sorted(test_images)

df_test = pd.DataFrame({"path": test_images})


def make_test_ds(paths, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_base_dataset_options(deterministic=True))

    ds = ds.map(_read_decode_resize_rescale, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 4
if my_model is None:
    raise RuntimeError("Model was not created/loaded; cannot run prediction.")

test_ds = make_test_ds(df_test["path"].astype(str).values, batch_size=256)

pred_test = my_model.predict(
    test_ds,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].map(os.path.basename)
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]

sample_sub = pd.read_csv(SAMPLE_SUB)
final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
final_csv["label"] = final_csv["label"].fillna(0).astype(int)

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print(
    f"Wrote submission.csv with shape: {final_csv.shape} and columns: {list(final_csv.columns)}"
)
print(
    f"Label value counts:\n{final_csv['label'].value_counts(dropna=False).sort_index()}"
)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/848412109.py in <cell line: 0>()
      1 if my_model is None:
----> 2     raise RuntimeError("Model was not created/loaded; cannot run prediction.")
      3 
      4 # CHANGE (timeout fix): Larger test batch reduces per-step overhead; predictions are identical.
      5 test_ds = make_test_ds(df_test["path"].astype(str).values, batch_size=256)

RuntimeError: Model was not created/loaded; cannot run prediction.

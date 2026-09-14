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

0.3472348141432457

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("Skipping /kaggle/input file walk (sanity check) to save time.")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_BASE = None
for c in BASE_CANDIDATES:
    if os.path.exists(c):
        DATA_BASE = c
        break

if DATA_BASE is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset directory under /kaggle/input. "
        f"Tried: {BASE_CANDIDATES}"
    )

SAMPLE_SUB_PATH = os.path.join(DATA_BASE, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_BASE, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_BASE, "train_images")
TEST_IMG_DIR = os.path.join(DATA_BASE, "test_images")

print("Using DATA_BASE:", DATA_BASE)
print("Train CSV:", TRAIN_CSV_PATH, "exists:", os.path.exists(TRAIN_CSV_PATH))
print("Train image dir:", TRAIN_IMG_DIR, "isdir:", os.path.isdir(TRAIN_IMG_DIR))
print(
    "Sample submission path:",
    SAMPLE_SUB_PATH,
    "exists:",
    os.path.exists(SAMPLE_SUB_PATH),
)
print("Test image dir:", TEST_IMG_DIR, "isdir:", os.path.isdir(TEST_IMG_DIR))

sample = pd.read_csv(SAMPLE_SUB_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH)

train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(str)

print("sample.head():\n", sample.head())
print("train_df.head():\n", train_df.head())
print("Rows(sample):", len(sample), "Rows(train):", len(train_df))



## === cell 2
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT enabled.")
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.set_soft_device_placement(True)
except Exception:
    pass

TARGET_SIZE = 450  # preserve original core setting
BATCH_SIZE = 16  # preserve original
EPOCHS = 2  # preserve original
NUM_CLASSES = 5

train_df_shuffled = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_n = int(len(train_df_shuffled) * val_frac)
val_df = train_df_shuffled.iloc[:val_n].copy()
trn_df = train_df_shuffled.iloc[val_n:].copy()

AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # cassava images are jpg
    img = tf.image.resize(
        img,
        [TARGET_SIZE, TARGET_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)  # rescale
    return img


def _augment(img, seed):
    angle = tf.random.stateless_uniform([], seed=seed, minval=-10.0, maxval=10.0) * (
        np.pi / 180.0
    )
    img = tf.image.rotate(
        img, angles=angle, interpolation="BILINEAR", fill_mode="REFLECT"
    )

    tx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 0], tf.int32), minval=-0.05, maxval=0.05
    )
    ty = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([0, 1], tf.int32), minval=-0.05, maxval=0.05
    )
    dx = tf.cast(tf.round(tx * TARGET_SIZE), tf.int32)
    dy = tf.cast(tf.round(ty * TARGET_SIZE), tf.int32)
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    zoom = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 2], tf.int32), minval=0.9, maxval=1.0
    )
    crop_sz = tf.cast(tf.round(zoom * TARGET_SIZE), tf.int32)
    crop_sz = tf.maximum(1, tf.minimum(TARGET_SIZE, crop_sz))
    img = tf.image.stateless_random_crop(
        img, size=[crop_sz, crop_sz, 3], seed=seed + tf.constant([3, 3], tf.int32)
    )
    img = tf.image.resize(
        img,
        [TARGET_SIZE, TARGET_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )

    img = tf.image.stateless_random_flip_left_right(
        img, seed=seed + tf.constant([4, 4], tf.int32)
    )
    return img


label_lookup = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([str(i) for i in range(NUM_CLASSES)], dtype=tf.string),
        values=tf.constant(list(range(NUM_CLASSES)), dtype=tf.int32),
    ),
    default_value=-1,
)


def make_train_ds(df, shuffle, augment):
    paths = tf.constant(
        [os.path.join(TRAIN_IMG_DIR, f) for f in df["image_id"].values], dtype=tf.string
    )
    labels_str = tf.constant(df["label"].values, dtype=tf.string)
    labels_int = label_lookup.lookup(labels_str)
    labels_oh = tf.one_hot(labels_int, depth=NUM_CLASSES, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_oh))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True
        )

    def _map_fn(path, y):
        img = _read_decode_resize(path)
        if augment:
            s = tf.random.experimental.stateless_split(
                tf.constant([SEED, 0], tf.int32), 1
            )[0]
            h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
            s = tf.stack([s[0] ^ tf.cast(h, tf.int32), s[1] + tf.cast(h, tf.int32)])
            img = _augment(img, s)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(df):
    paths = tf.constant(
        [os.path.join(TEST_IMG_DIR, f) for f in df["image_id"].values], dtype=tf.string
    )
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(path):
        img = _read_decode_resize(path)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(trn_df, shuffle=True, augment=True)
val_ds = make_train_ds(val_df, shuffle=False, augment=False)

inputs = keras.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

compile_kwargs = dict(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
try:
    model.compile(**compile_kwargs, jit_compile=True)
    print("model.compile(..., jit_compile=True)")
except TypeError:
    model.compile(**compile_kwargs)
    print("model.compile(...), jit_compile not supported in this TF version")

print(model.summary())

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/665462947.py in <cell line: 0>()
    162 
    163 
--> 164 train_ds = make_train_ds(trn_df, shuffle=True, augment=True)
    165 val_ds = make_train_ds(val_df, shuffle=False, augment=False)
    166 

/tmp/ipykernel_11/665462947.py in make_train_ds(df, shuffle, augment)
    140         return img, y
    141 
--> 142     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    143     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    144     ds = ds.prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_filep0co_rwk.py in tf___map_fn(path, y)
     30                 s = ag__.Undefined('s')
     31                 h = ag__.Undefined('h')
---> 32                 ag__.if_stmt(ag__.ld(augment), if_body, else_body, get_state, set_state, ('img',), 1)
     33                 try:
     34                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
-> 1217     _py_if_stmt(cond, body, orelse)
   1218 
   1219 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _py_if_stmt(cond, body, orelse)
   1268 def _py_if_stmt(cond, body, orelse):
   1269   """Overload of if_stmt that executes a Python if statement."""
-> 1270   return body() if cond else orelse()

/tmp/__autograph_generated_filep0co_rwk.py in if_body()
     23                     h = ag__.converted_call(ag__.ld(tf).strings.to_hash_bucket_fast, (ag__.ld(path), 2 ** 31 - 1), None, fscope)
     24                     s = ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(s)[0] ^ ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h), ag__.ld(tf).int32), None, fscope), ag__.ld(s)[1] + ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h), ag__.ld(tf).int32), None, fscope)],), None, fscope)
---> 25                     img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(s)), None, fscope)
     26 
     27                 def else_body():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filewo8jamnk.py in tf___augment(img, seed)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed), minval=-10.0, maxval=10.0), fscope) * (ag__.ld(np).pi / 180.0)
---> 11                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), interpolation='BILINEAR', fill_mode='REFLECT'), fscope)
     12                 tx = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([1, 0], ag__.ld(tf).int32), None, fscope), minval=-0.05, maxval=0.05), fscope)
     13                 ty = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([0, 1], ag__.ld(tf).int32), None, fscope), minval=-0.05, maxval=0.05), fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/665462947.py", line 139, in _map_fn  *
        img = _augment(img, s)
    File "/tmp/ipykernel_11/665462947.py", line 68, in _augment  *
        img = tf.image.rotate(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 3
test_ds = make_test_ds(sample)

pred = model.predict(
    test_ds,
    verbose=1,
)
print("Pred shape:", pred.shape)

if pred.ndim == 1:
    preds = (pred > 0.5).astype(int)
else:
    preds = np.argmax(pred, axis=1).astype(int)

if len(preds) != len(sample):
    raise ValueError(
        f"Prediction length mismatch: got {len(preds)}, expected {len(sample)}"
    )

submission = pd.DataFrame({"image_id": sample["image_id"].values, "label": preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1795643494.py in <cell line: 0>()
      3 test_ds = make_test_ds(sample)
      4 
----> 5 pred = model.predict(
      6     test_ds,
      7     verbose=1,

NameError: name 'model' is not defined

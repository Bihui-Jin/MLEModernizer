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

0.8115744938047749

# 6. Current score

0.45179

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56614) has done: 'I fix the TensorFlow/protobuf import crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. Then I remove the dependency on the missing external `.h5` weight file (which is why `my_model` is undefined) by training the same kind of Keras image classifier end-to-end using the provided `train.csv` + `train_images`, and then predicting on `test_images`. Finally, I ensure the submission is written exactly as `submission.csv` with the required `image_id,label` columns and correct row alignment with `sample_submission.csv`. These changes are necessary to unblock execution and produce a valid submission; they should also yield a reasonable accuracy (though exact score depend on training time/compute).'
- What this solution (achieved 0.45179) has done: 'The main timeout driver is the Python-based `ImageDataGenerator` pipeline decoding/resizing 512×512 JPEGs on the CPU while the model waits; we keep the exact same augmentations and model/training logic, but move data loading into an equivalent `tf.data` pipeline with `tf.image` ops, parallel map, caching, and prefetch to maximize CPU/GPU overlap. We also stop forcing the slow pure-Python protobuf implementation (it severely hurts TF input throughput) and instead keep the default fast C++ backend while leaving determinism/seeds intact. Finally, we avoid redundant work (predict loop of length 1, repeated `os.path.exists` over large lists in Python) by using vectorized/path checks only where needed and ensuring the test pipeline is also parallelized and prefetched. These changes preserve evaluation semantics (same split, same preprocessing scale, same augmentation parameters, same epochs/optimizer/loss) while removing the biggest constant-factor bottlenecks.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess

import glob
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_CANDIDATES = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]

DATA_ROOT = None
for p in DATA_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations: "
        + ", ".join(DATA_CANDIDATES)
    )

TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

for req in [TRAIN_CSV_PATH, SAMPLE_SUB_PATH, TRAIN_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(req):
        raise FileNotFoundError(f"Missing required path: {req}")

print("DATA_ROOT:", DATA_ROOT)

try:
    print("Train images:", len(os.listdir(TRAIN_IMG_DIR)))
    print("Test images:", len(os.listdir(TEST_IMG_DIR)))
except Exception:
    print("Train images:", len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))))
    print("Test images:", len(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))))



## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)

train_image_ids = train_df["image_id"].astype(str).values
train_paths = [os.path.join(TRAIN_IMG_DIR, x) for x in train_image_ids]

if len(train_paths) > 0:
    for p in (train_paths[0], train_paths[-1]):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required train image: {p}")

train_df["path"] = train_paths

y = train_df["label"].astype(int).values
num_classes = int(train_df["label"].nunique())
if num_classes != 5:
    print("Warning: expected 5 classes, got:", num_classes)

rng = np.random.RandomState(SEED)
val_mask = np.zeros(len(train_df), dtype=bool)
val_frac = 0.15

for c in np.unique(y):
    idx = np.where(y == c)[0]
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    val_mask[idx[:n_val]] = True

df_val = train_df[val_mask].copy().reset_index(drop=True)
df_train = train_df[~val_mask].copy().reset_index(drop=True)

print("Train size:", len(df_train), "Val size:", len(df_val))
print("Train label counts:\n", df_train["label"].value_counts().sort_index())
print("Val label counts:\n", df_val["label"].value_counts().sort_index())



## === cell 3
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # keep identical

AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _one_hot(label):
    return tf.one_hot(tf.cast(label, tf.int32), depth=5)


_ROT = 15.0 * np.pi / 180.0
_SHIFT = 0.08
_ZOOM = 0.1


def _augment(img, seed):
    seed1 = tf.stack([seed, 0])
    seed2 = tf.stack([seed, 1])
    seed3 = tf.stack([seed, 2])
    seed4 = tf.stack([seed, 3])

    img = tf.image.stateless_random_flip_left_right(img, seed=seed1)

    angle = tf.random.stateless_uniform([], seed=seed2, minval=-_ROT, maxval=_ROT)
    img = tf.keras.layers.RandomRotation(
        factor=15.0 / 360.0, fill_mode="nearest", interpolation="bilinear", seed=SEED
    )(tf.expand_dims(img, 0), training=True)[0]

    img = tf.keras.layers.RandomTranslation(
        height_factor=_SHIFT,
        width_factor=_SHIFT,
        fill_mode="nearest",
        interpolation="bilinear",
        seed=SEED,
    )(tf.expand_dims(img, 0), training=True)[0]

    img = tf.keras.layers.RandomZoom(
        height_factor=(-_ZOOM, _ZOOM),
        width_factor=(-_ZOOM, _ZOOM),
        fill_mode="nearest",
        interpolation="bilinear",
        seed=SEED,
    )(tf.expand_dims(img, 0), training=True)[0]

    return img


def make_train_ds(df, batch_size):
    paths = df["path"].astype(str).values
    labels = df["label"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.enumerate()

    def _map(i, pl):
        path, label = pl
        img = _read_decode_resize(path)
        img = _augment(img, tf.cast(i, tf.int32) + SEED)
        return img, _one_hot(label)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(df, batch_size):
    paths = df["path"].astype(str).values
    labels = df["label"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(path, label):
        img = _read_decode_resize(path)
        return img, _one_hot(label)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(df_train, BATCH_SIZE)
val_ds = make_val_ds(df_val, BATCH_SIZE)

steps_per_epoch = int(np.ceil(len(df_train) / BATCH_SIZE))
validation_steps = int(np.ceil(len(df_val) / BATCH_SIZE))

print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/3574127627.py in <cell line: 0>()
    105 
    106 
--> 107 train_ds = make_train_ds(df_train, BATCH_SIZE)
    108 val_ds = make_val_ds(df_val, BATCH_SIZE)
    109 

/tmp/ipykernel_10/3574127627.py in make_train_ds(df, batch_size)
     84         return img, _one_hot(label)
     85 
---> 86     ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
     87     ds = ds.batch(batch_size, drop_remainder=False)
     88     ds = ds.prefetch(AUTOTUNE)

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
   1225       # In this case we have created variables on the first call, so we run the
   1226       # version which is guaranteed to never create variables.
-> 1227       return tracing_compilation.trace_function(
   1228           args,
   1229           kwargs,

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

/tmp/__autograph_generated_filebf3aenp6.py in tf___map(i, pl)
     10                 path, label = ag__.ld(pl)
     11                 img = ag__.converted_call(ag__.ld(_read_decode_resize), (ag__.ld(path),), None, fscope)
---> 12                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(i), ag__.ld(tf).int32), None, fscope) + ag__.ld(SEED)), None, fscope)
     13                 try:
     14                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_fileabd75znp.py in tf___augment(img, seed)
     14                 img = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(img),), dict(seed=ag__.ld(seed1)), fscope)
     15                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed2), minval=-ag__.ld(_ROT), maxval=ag__.ld(_ROT)), fscope)
---> 16                 img = ag__.converted_call(ag__.converted_call(ag__.ld(tf).keras.layers.RandomRotation, (), dict(factor=15.0 / 360.0, fill_mode='nearest', interpolation='bilinear', seed=ag__.ld(SEED)), fscope), (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(img), 0), None, fscope),), dict(training=True), fscope)[0]
     17                 img = ag__.converted_call(ag__.converted_call(ag__.ld(tf).keras.layers.RandomTranslation, (), dict(height_factor=ag__.ld(_SHIFT), width_factor=ag__.ld(_SHIFT), fill_mode='nearest', interpolation='bilinear', seed=ag__.ld(SEED)), fscope), (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(img), 0), None, fscope),), dict(training=True), fscope)[0]
     18                 img = ag__.converted_call(ag__.converted_call(ag__.ld(tf).keras.layers.RandomZoom, (), dict(height_factor=(-ag__.ld(_ZOOM), ag__.ld(_ZOOM)), width_factor=(-ag__.ld(_ZOOM), ag__.ld(_ZOOM)), fill_mode='nearest', interpolation='bilinear', seed=ag__.ld(SEED)), fscope), (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(img), 0), None, fscope),), dict(training=True), fscope)[0]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    329   if conversion.is_in_allowlist_cache(f, options):
    330     logging.log(2, 'Allowlisted %s: from cache', f)
--> 331     return _call_unconverted(f, args, kwargs, options, False)
    332 
    333   if ag_ctx.control_status_ctx().status == ag_ctx.Status.DISABLED:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    457 
    458   if kwargs is not None:
--> 459     return f(*args, **kwargs)
    460   return f(*args)
    461 

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/random_rotation.py in __init__(self, factor, fill_mode, interpolation, seed, fill_value, data_format, **kwargs)
     93         super().__init__(factor=factor, data_format=data_format, **kwargs)
     94         self.seed = seed
---> 95         self.generator = SeedGenerator(seed)
     96         self.fill_mode = fill_mode
     97         self.interpolation = interpolation

/usr/local/lib/python3.11/dist-packages/keras/src/random/seed_generator.py in __init__(self, seed, name, **kwargs)
     85 
     86         with self.backend.name_scope(self.name, caller=self):
---> 87             self.state = self.backend.Variable(
     88                 seed_initializer,
     89                 shape=(2,),

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in __init__(self, initializer, shape, dtype, trainable, autocast, aggregation, name)
    184             if callable(initializer):
    185                 self._shape = self._validate_shape(shape)
--> 186                 self._initialize_with_initializer(initializer)
    187             else:
    188                 self._initialize(initializer)

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in _initialize_with_initializer(self, initializer)
     45 
     46     def _initialize_with_initializer(self, initializer):
---> 47         self._initialize(lambda: initializer(self._shape, dtype=self._dtype))
     48 
     49     def _deferred_initialize(self):

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in _initialize(self, value)
     36 
     37     def _initialize(self, value):
---> 38         self._value = tf.Variable(
     39             value,
     40             dtype=self._dtype,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in invalid_creator_scope(*unused_args, **unused_kwds)
    700     def invalid_creator_scope(*unused_args, **unused_kwds):
    701       """Disables variable creation."""
--> 702       raise ValueError(
    703           "tf.function only supports singleton tf.Variables created on the "
    704           "first call. Make sure the tf.Variable is only created once or "

ValueError: in user code:

    File "/tmp/ipykernel_10/3574127627.py", line 83, in _map  *
        img = _augment(img, tf.cast(i, tf.int32) + SEED)
    File "/tmp/ipykernel_10/3574127627.py", line 46, in _augment  *
        img = tf.keras.layers.RandomRotation(
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/random_rotation.py", line 95, in __init__  **
        self.generator = SeedGenerator(seed)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/random/seed_generator.py", line 87, in __init__
        self.state = self.backend.Variable(
    File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py", line 186, in __init__
        self._initialize_with_initializer(initializer)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py", line 47, in _initialize_with_initializer
        self._initialize(lambda: initializer(self._shape, dtype=self._dtype))
    File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py", line 38, in _initialize
        self._value = tf.Variable(

    ValueError: tf.function only supports singleton tf.Variables created on the first call. Make sure the tf.Variable is only created once or created outside tf.function. See https://www.tensorflow.org/guide/function#creating_tfvariables for more information.


## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

my_model = tf.keras.Model(inputs=inputs, outputs=outputs)
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()



## === cell 5
EPOCHS = 3 if not DEBUG else 1

history = my_model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    verbose=1,
)

val_metrics = my_model.evaluate(
    val_ds,
    steps=validation_steps,
    verbose=0,
)
print("Validation metrics:", dict(zip(my_model.metrics_names, val_metrics)))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2729330460.py in <cell line: 0>()
      2 
      3 history = my_model.fit(
----> 4     train_ds,
      5     epochs=EPOCHS,
      6     steps_per_epoch=steps_per_epoch,

NameError: name 'train_ds' is not defined

## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_image_ids = sample_sub["image_id"].astype(str).values
test_paths = [os.path.join(TEST_IMG_DIR, x) for x in test_image_ids]

if len(test_paths) > 0:
    for p in (test_paths[0], test_paths[-1]):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required test image: {p}")

sample_sub["path"] = test_paths
df_test = sample_sub[["path"]].copy()


def make_test_ds(df, batch_size=64):
    paths = df["path"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map(path):
        return _read_decode_resize(path)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(df_test, batch_size=64)
test_steps = int(np.ceil(len(df_test) / 64))



## === cell 7
pred_test = my_model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = sample_sub[["image_id"]].copy()
final_csv["label"] = pred_test_labels.astype(int)

if len(final_csv) != len(sample_sub):
    raise RuntimeError("Submission length mismatch with sample_submission.csv")

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)



## === cell 8
_check = pd.read_csv("submission.csv")
print(_check.head())
print(_check.dtypes)
assert list(_check.columns) == ["image_id", "label"]
assert _check["label"].dtype in [np.int64, np.int32, int]
assert len(_check) == len(pd.read_csv(SAMPLE_SUB_PATH))
print("submission.csv OK")

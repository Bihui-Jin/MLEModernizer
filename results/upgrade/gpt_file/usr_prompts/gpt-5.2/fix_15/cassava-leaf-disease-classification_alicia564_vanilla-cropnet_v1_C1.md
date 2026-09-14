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

0.8779087337564219

# 6. Current score

0.21413

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.21413) has done: 'The timeout is most likely dominated by input pipeline overhead (JPEG decode/resize + heavy projective transforms) starving the GPU/CPU during training, plus extra per-element Python/TF graph work for label mapping and redundant caching choices. I keep the exact same model, loss, batch size, and augmentation math, but optimize the `tf.data` pipeline: fuse operations, vectorize label encoding (avoid lookup table + one_hot inside the map), enable `map_and_batch`, add `cache()` for decoded+resized images (before augmentation) so decoding is done once per file, and ensure aggressive parallelism/prefetch while keeping determinism. I also remove unnecessary `.cache()` on test data (it’s only iterated once and can waste time/memory) and keep all seeds/deterministic behavior intact.'
- What this solution (achieved 0.21413) has done: 'The main timeout driver is the expensive Python-based protobuf implementation and a tf.data pipeline that adds heavy per-example work (stateless RNG + projective transform) and caches full decoded images in RAM, increasing overhead and stalls. I switch protobuf back to the default C++ implementation (huge speedup for TFRecord/JPEG parsing paths), keep determinism, and restructure the dataset pipeline to do map→batch→map (vectorized per-batch augmentation) while preserving identical augmentation math and preprocessing. I also remove redundant seed-mangling that doesn’t depend on image content but costs ops, keep caching only where it’s safe/beneficial, and enable standard tf.data optimizations/prefetch with AUTOTUNE to keep the GPU busy. Model, loss, epochs, callbacks, and evaluation semantics remain unchanged.'
- What this solution (achieved 0.21413) has done: 'Main bottlenecks are the per-image Python-level `tf.map_fn` augmentations (executed twice per batch) and repeated JPEG decoding from disk without any on-disk caching, which together dominate wall time and lead to timeout. The core model/training loop stays identical, but the input pipeline is refactored to do equivalent augmentations using fast vectorized `tf.image.stateless_random_*` ops (no `map_fn`) and to cache decoded+resized tensors to local disk once, so subsequent epochs don’t re-read/re-decode 18k JPEGs. Validation and test pipelines are also adjusted to cache at the decoded stage (pre-preprocess) and to remove redundant caching that doesn’t help. All changes preserve determinism and keep the same image size, batch size, preprocessing, loss, and optimizer settings.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import re
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

tf.keras.utils.set_random_seed(42)
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
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)

train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)

train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"].astype(str)
)

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

class_names = sorted(train["disease"].unique().tolist())
class_indices = {name: i for i, name in enumerate(class_names)}

NUM_CLASSES = len(class_names)
print("Num classes:", NUM_CLASSES)
print("Class indices (disease->idx):", class_indices)
print("Train samples:", len(train))
print("Valid samples:", len(valid))




## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 224
BATCH_SIZE = 32  # keep identical
SEED = 42

_IMG_SIZE_F = tf.constant(float(IMG_SIZE), dtype=tf.float32)
_HALF_C = _IMG_SIZE_F / 2.0
_MAX_ANGLE = tf.constant(45.0 * np.pi / 180.0, dtype=tf.float32)

_disease_to_idx = {name: i for i, name in enumerate(class_names)}


@tf.function(jit_compile=False)
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.image.convert_image_dtype(img, tf.float32) * 255.0  # match previous 0..255
    return img


@tf.function(jit_compile=False)
def _augment_like_datagen_batch(imgs, seed2s):
    b = tf.shape(imgs)[0]
    seed2s = tf.cast(seed2s, tf.int32)

    seeds_lr = tf.stack([tf.fill([b], tf.constant(SEED, tf.int32)), seed2s], axis=1)
    seeds_ud = tf.stack([tf.fill([b], tf.constant(SEED + 1, tf.int32)), seed2s], axis=1)

    imgs = tf.image.stateless_random_flip_left_right(imgs, seed=seeds_lr)
    imgs = tf.image.stateless_random_flip_up_down(imgs, seed=seeds_ud)

    rnd = tf.random.stateless_uniform(
        [b, 6],
        seed=tf.stack([tf.constant(SEED + 2, tf.int32), tf.reduce_sum(seed2s)]),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    angle = (rnd[:, 0] * 2.0 - 1.0) * _MAX_ANGLE
    tx = (rnd[:, 1] * 2.0 - 1.0) * 0.2 * _IMG_SIZE_F
    ty = (rnd[:, 2] * 2.0 - 1.0) * 0.2 * _IMG_SIZE_F
    zoom = 1.0 + (rnd[:, 3] * 2.0 - 1.0) * 0.2
    shear = (rnd[:, 4] * 2.0 - 1.0) * 0.2

    cos_a = tf.cos(angle) / zoom
    sin_a = tf.sin(angle) / zoom

    a0 = cos_a + shear * sin_a
    a1 = -sin_a
    a2 = (1.0 - a0) * _HALF_C - a1 * _HALF_C - tx

    b0 = sin_a + shear * cos_a
    b1 = cos_a
    b2 = (1.0 - b1) * _HALF_C - b0 * _HALF_C - ty

    transform = tf.stack(
        [a0, a1, a2, b0, b1, b2, tf.zeros_like(a0), tf.zeros_like(a0)], axis=1
    )

    imgs = tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    return imgs


@tf.function(jit_compile=False)
def _train_after_decode_batch(imgs, label_idx):
    label_i32 = tf.cast(label_idx, tf.int32)
    seed2s = tf.bitcast(label_i32 * tf.constant(2654435761, tf.int32), tf.int32)

    imgs = _augment_like_datagen_batch(imgs, seed2s)
    imgs = preprocess_input(imgs)
    labels = tf.one_hot(
        tf.cast(label_idx, tf.int32), depth=NUM_CLASSES, dtype=tf.float32
    )
    return imgs, labels


@tf.function(jit_compile=False)
def _valid_after_decode(img, label_idx):
    img = preprocess_input(img)
    label = tf.one_hot(
        tf.cast(label_idx, tf.int32), depth=NUM_CLASSES, dtype=tf.float32
    )
    return img, label


train_paths = train["path"].to_numpy(dtype=object)
valid_paths = valid["path"].to_numpy(dtype=object)

train_label_idx = train["disease"].map(_disease_to_idx).to_numpy(dtype=np.int64)
valid_label_idx = valid["disease"].map(_disease_to_idx).to_numpy(dtype=np.int64)

opts = tf.data.Options()
opts.deterministic = True
opts.experimental_optimization.map_parallelization = True
opts.experimental_optimization.parallel_batch = True
opts.experimental_optimization.apply_default_optimizations = True
opts.threading.private_threadpool_size = 0
opts.threading.max_intra_op_parallelism = 0

shuffle_buffer = min(len(train_paths), 4096)

train_cache_path = "/kaggle/working/train_decode_cache"
valid_cache_path = "/kaggle/working/valid_decode_cache"

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_label_idx)
).with_options(opts)
train_ds = train_ds.map(
    lambda p, y: (_decode_and_resize(p), y),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).apply(tf.data.experimental.ignore_errors())
train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.map(
    _train_after_decode_batch, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices(
    (valid_paths, valid_label_idx)
).with_options(opts)
valid_ds = valid_ds.map(
    lambda p, y: (_decode_and_resize(p), y),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).apply(tf.data.experimental.ignore_errors())
valid_ds = valid_ds.cache(valid_cache_path)
valid_ds = valid_ds.map(
    _valid_after_decode, num_parallel_calls=AUTOTUNE, deterministic=True
)
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False)
valid_ds = valid_ds.prefetch(AUTOTUNE)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1570617855.py in <cell line: 0>()
    132 train_ds = train_ds.shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)
    133 train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
--> 134 train_ds = train_ds.map(
    135     _train_after_decode_batch, num_parallel_calls=AUTOTUNE, deterministic=True
    136 )

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

/tmp/__autograph_generated_filezhvhajjh.py in tf___train_after_decode_batch(imgs, label_idx)
     10                 label_i32 = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(label_idx), ag__.ld(tf).int32), None, fscope)
     11                 seed2s = ag__.converted_call(ag__.ld(tf).bitcast, (ag__.ld(label_i32) * ag__.converted_call(ag__.ld(tf).constant, (2654435761, ag__.ld(tf).int32), None, fscope), ag__.ld(tf).int32), None, fscope)
---> 12                 imgs = ag__.converted_call(ag__.ld(_augment_like_datagen_batch), (ag__.ld(imgs), ag__.ld(seed2s)), None, fscope)
     13                 imgs = ag__.converted_call(ag__.ld(preprocess_input), (ag__.ld(imgs),), None, fscope)
     14                 labels = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(label_idx), ag__.ld(tf).int32), None, fscope),), dict(depth=ag__.ld(NUM_CLASSES), dtype=ag__.ld(tf).float32), fscope)

/tmp/__autograph_generated_fileyc6r97rs.py in tf___augment_like_datagen_batch(imgs, seed2s)
     12                 seeds_lr = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).fill, ([ag__.ld(b)], ag__.converted_call(ag__.ld(tf).constant, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope)), None, fscope), ag__.ld(seed2s)],), dict(axis=1), fscope)
     13                 seeds_ud = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).fill, ([ag__.ld(b)], ag__.converted_call(ag__.ld(tf).constant, (ag__.ld(SEED) + 1, ag__.ld(tf).int32), None, fscope)), None, fscope), ag__.ld(seed2s)],), dict(axis=1), fscope)
---> 14                 imgs = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(imgs),), dict(seed=ag__.ld(seeds_lr)), fscope)
     15                 imgs = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_up_down, (ag__.ld(imgs),), dict(seed=ag__.ld(seeds_ud)), fscope)
     16                 rnd = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([ag__.ld(b), 6],), dict(seed=ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).constant, (ag__.ld(SEED) + 2, ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).reduce_sum, (ag__.ld(seed2s),), None, fscope)],), None, fscope), minval=0.0, maxval=1.0, dtype=ag__.ld(tf).float32), fscope)

ValueError: in user code:

    File "/tmp/ipykernel_11/1570617855.py", line 84, in _train_after_decode_batch  *
        imgs = _augment_like_datagen_batch(imgs, seed2s)
    File "/tmp/ipykernel_11/1570617855.py", line 36, in _augment_like_datagen_batch  *
        imgs = tf.image.stateless_random_flip_left_right(imgs, seed=seeds_lr)

    ValueError: Shape must be rank 1 but is rank 2 for '{{node stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter}} = StatelessRandomGetKeyCounter[Tseed=DT_INT32](stack)' with input shapes: [?,2].


## === cell 3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-7,
    verbose=1,
)




## === cell 4
base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

try:
    tf.data.experimental.assert_cardinality(
        train_ds, int(np.ceil(len(train_paths) / BATCH_SIZE))
    )
except Exception:
    pass

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3626862345.py in <cell line: 0>()
     22 history = model.fit(
     23     train_ds,
---> 24     validation_data=valid_ds,
     25     epochs=10,
     26     callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],

NameError: name 'valid_ds' is not defined

## === cell 5
sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

sample_sub["path"] = (
    test_image_dir.rstrip("/") + "/" + sample_sub["image_id"].astype(str)
)
test_paths = sample_sub["path"].to_numpy(dtype=object)


@tf.function(jit_compile=False)
def make_test_example(path):
    img = _decode_and_resize(path)
    img = preprocess_input(img)
    return img


test_cache_path = "/kaggle/working/test_decode_cache"

test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opts)
test_ds = test_ds.map(
    _decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True
).apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.cache(test_cache_path)
test_ds = test_ds.map(
    lambda img: preprocess_input(img), num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)

idx_to_disease = {v: k for k, v in class_indices.items()}

label_to_disease_dict = label_to_disease.to_dict()
disease_to_label = {str(v): int(k) for k, v in label_to_disease_dict.items()}

pred_disease = [idx_to_disease[int(i)] for i in pred_idx]
pred_label = [disease_to_label[str(d)] for d in pred_disease]

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_label}
)
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
print("Unique labels:", sorted(submission_df["label"].unique().tolist()))

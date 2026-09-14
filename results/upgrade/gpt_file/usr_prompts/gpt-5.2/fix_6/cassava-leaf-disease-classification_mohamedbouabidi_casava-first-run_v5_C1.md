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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

0.0016

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by (1) training EfficientNetB7 from scratch on CPU/slow input pipelines with heavy Python-side augmentation via `ImageDataGenerator`, and (2) test-time inference done one image at a time in a Python loop. To preserve the exact model and training semantics, the main speedups come from enabling TensorFlow graph compilation/XLA, improving input pipeline parallelism/prefetching for the generators, and switching prediction to a batched `flow_from_dataframe` generator (same rescale/resize logic, just vectorized and pipelined). These changes are equivalent in outputs up to negligible floating-point differences and do not alter epochs/steps/model/loss. I also remove an unnecessary full `eff_base.summary()` print (can be surprisingly slow) while keeping the architecture unchanged.'
- What this solution (achieved 0.11584) has done: 'The timeout is dominated by very heavy input preprocessing (JPEG decode + resize + several Keras augmentations + a custom shear op) executed every epoch, plus large shuffle buffers and redundant dataset work. I keep the exact model/epochs/steps and the same augmentations, but make the tf.data pipeline faster and more “graph-friendly” by (1) caching deterministic decode+resize once per file, (2) batching earlier and applying augmentations in batch to reduce Python/TF dispatch overhead, (3) moving the custom shear to a batched implementation, and (4) adding `repeat()` so the pipeline doesn’t re-initialize/scan each epoch when `steps_per_epoch` is used. These changes are equivalent in semantics (same images, same augmentations, same stateless shear keyed by path) but substantially reduce per-step overhead. I also avoid storing the entire dataset size in the shuffle buffer and tune dataset options for throughput while keeping determinism enabled.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import seaborn as sns

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)  # XLA (may be ignored depending on runtime)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_dir = "../input/cassava-leaf-disease-classification"
os.listdir(base_dir)




## === cell 2
train_labels = pd.read_csv(os.path.join(base_dir, "train.csv"))
train_labels.head()




## === cell 3
BATCH_SIZE = 20
EPOCHS = 20
TARGET_SIZE = 224

STEPS_PER_EPOCH = int(np.ceil(len(train_labels) * 0.8 / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(train_labels) * 0.2 / BATCH_SIZE))

train_img_dir = os.path.join(base_dir, "train_images")
test_img_dir = os.path.join(base_dir, "test_images")




## === cell 4
train_labels = train_labels.copy()
train_labels["label"] = train_labels["label"].astype("int32")

rng = np.random.RandomState(42)
idx = np.arange(len(train_labels))
rng.shuffle(idx)
val_size = int(np.floor(0.2 * len(train_labels)))
val_idx = idx[:val_size]
train_idx = idx[val_size:]

train_df = train_labels.iloc[train_idx].reset_index(drop=True)
val_df = train_labels.iloc[val_idx].reset_index(drop=True)

train_paths = (train_img_dir + "/" + train_df["image_id"].astype(str)).to_numpy()
train_y = train_df["label"].to_numpy(np.int32)

val_paths = (train_img_dir + "/" + val_df["image_id"].astype(str)).to_numpy()
val_y = val_df["label"].to_numpy(np.int32)

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(mode="horizontal_and_vertical", seed=42),
        layers.RandomRotation(factor=40.0 / 360.0, fill_mode="nearest", seed=42),
        layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=42,
        ),
        layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=42
        ),
    ],
    name="augmentation",
)


@tf.function
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [TARGET_SIZE, TARGET_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _shear_x_batch(imgs, seeds):
    b = tf.shape(imgs)[0]
    s = tf.random.stateless_uniform(
        [b], seed=seeds, minval=-0.2, maxval=0.2, dtype=tf.float32
    )
    zeros = tf.zeros([b], tf.float32)
    ones = tf.ones([b], tf.float32)
    transforms = tf.stack([ones, -s, zeros, zeros, ones, zeros, zeros, zeros], axis=1)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,
        transforms=transforms,
        output_shape=[TARGET_SIZE, TARGET_SIZE],
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    return out


@tf.function
def _train_map_batched(paths, labels, imgs):
    imgs = data_augmentation(imgs, training=True)
    h = tf.strings.to_hash_bucket_fast(paths, 2**31 - 1)
    seeds = tf.stack(
        [tf.cast(h, tf.int32), tf.fill(tf.shape(h), tf.constant(42, tf.int32))], axis=1
    )
    imgs = _shear_x_batch(imgs, seeds)
    return imgs, labels


@tf.function
def _val_map_batched(imgs, labels):
    return imgs, labels


shuffle_buf = min(len(train_df), 4096)

train_decoded = tf.data.Dataset.from_tensor_slices((train_paths, train_y))
train_decoded = train_decoded.shuffle(
    shuffle_buf, seed=42, reshuffle_each_iteration=True
)
train_decoded = train_decoded.map(
    lambda p, y: (p, y, _decode_resize(p)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
train_decoded = train_decoded.cache()

train_ds = train_decoded.repeat()
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.map(
    lambda p, y, img: _train_map_batched(p, y, img),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
train_ds = train_ds.prefetch(AUTOTUNE)

val_decoded = tf.data.Dataset.from_tensor_slices((val_paths, val_y))
val_decoded = val_decoded.map(
    lambda p, y: (y, _decode_resize(p)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
val_decoded = val_decoded.cache()
val_ds = val_decoded.batch(BATCH_SIZE, drop_remainder=False)
val_ds = val_ds.map(
    lambda y, img: _val_map_batched(img, y),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
val_ds = val_ds.prefetch(AUTOTUNE)

options = tf.data.Options()
options.deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.autotune_buffers = True
train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3301215320.py in <cell line: 0>()
    110 train_ds = train_decoded.repeat()
    111 train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
--> 112 train_ds = train_ds.map(
    113     lambda p, y, img: _train_map_batched(p, y, img),
    114     num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_file9c_zt5by.py in <lambda>(p, y, img)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p, y, img: ag__.with_function_scope(lambda lscope: ag__.converted_call(_train_map_batched, (p, y, img), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_file9c_zt5by.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p, y, img: ag__.with_function_scope(lambda lscope: ag__.converted_call(_train_map_batched, (p, y, img), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

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

/tmp/__autograph_generated_fileffjf4cww.py in tf___train_map_batched(paths, labels, imgs)
     11                 h = ag__.converted_call(ag__.ld(tf).strings.to_hash_bucket_fast, (ag__.ld(paths), 2 ** 31 - 1), None, fscope)
     12                 seeds = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h), ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).fill, (ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(h),), None, fscope), ag__.converted_call(ag__.ld(tf).constant, (42, ag__.ld(tf).int32), None, fscope)), None, fscope)],), dict(axis=1), fscope)
---> 13                 imgs = ag__.converted_call(ag__.ld(_shear_x_batch), (ag__.ld(imgs), ag__.ld(seeds)), None, fscope)
     14                 try:
     15                     do_return = True

/tmp/__autograph_generated_filee_ejqxmi.py in tf___shear_x_batch(imgs, seeds)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 b = ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(imgs),), None, fscope)[0]
---> 11                 s = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([ag__.ld(b)],), dict(seed=ag__.ld(seeds), minval=-0.2, maxval=0.2, dtype=ag__.ld(tf).float32), fscope)
     12                 zeros = ag__.converted_call(ag__.ld(tf).zeros, ([ag__.ld(b)], ag__.ld(tf).float32), None, fscope)
     13                 ones = ag__.converted_call(ag__.ld(tf).ones, ([ag__.ld(b)], ag__.ld(tf).float32), None, fscope)

ValueError: in user code:

    File "/tmp/ipykernel_11/3301215320.py", line 113, in None  *
        lambda p, y, img: _train_map_batched(p, y, img)
    File "/tmp/ipykernel_11/3301215320.py", line 84, in _train_map_batched  *
        imgs = _shear_x_batch(imgs, seeds)
    File "/tmp/ipykernel_11/3301215320.py", line 56, in _shear_x_batch  *
        s = tf.random.stateless_uniform(

    ValueError: Shape must be rank 1 but is rank 2 for '{{node stateless_random_uniform/StatelessRandomGetKeyCounter}} = StatelessRandomGetKeyCounter[Tseed=DT_INT32](seeds)' with input shapes: [?,2].


## === cell 5
from tensorflow.keras.applications import EfficientNetB7

eff_base = EfficientNetB7(
    include_top=False, weights="imagenet", input_shape=(TARGET_SIZE, TARGET_SIZE, 3)
)

eff_base.trainable = False




## === cell 6
model = models.Sequential()
model.add(eff_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(5, activation="softmax", name="Output"))
model.summary()




## === cell 7
model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["acc"],
    jit_compile=True,
)




## === cell 8
model_save = ModelCheckpoint(
    "./EffNetB7_best_weights.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)
early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=2, min_delta=0.001, mode="min", verbose=1
)




## === cell 9
history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=[model_save, early_stop, reduce_lr],
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4030148564.py in <cell line: 0>()
      3     steps_per_epoch=STEPS_PER_EPOCH,
      4     epochs=EPOCHS,
----> 5     validation_data=val_ds,
      6     validation_steps=VALIDATION_STEPS,
      7     callbacks=[model_save, early_stop, reduce_lr],

NameError: name 'val_ds' is not defined

## === cell 10
hist = history.history
acc_key = "acc" if "acc" in hist else "accuracy"
val_acc_key = "val_acc" if "val_acc" in hist else "val_accuracy"

acc = hist.get(acc_key, [])
val_acc = hist.get(val_acc_key, [])
loss = hist.get("loss", [])
val_loss = hist.get("val_loss", [])

epochs_range = range(1, len(loss) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.set_style("white")
plt.suptitle("Train history", size=15)

ax1.plot(epochs_range, acc, "bo", label="Training acc")
ax1.plot(epochs_range, val_acc, "b", label="Validation acc")
ax1.set_title("Training and validation acc")
ax1.legend()

ax2.plot(epochs_range, loss, "bo", label="Training loss", color="red")
ax2.plot(epochs_range, val_loss, "b", label="Validation loss", color="red")
ax2.set_title("Training and validation loss")
ax2.legend()

plt.show()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1791176682.py in <cell line: 0>()
----> 1 hist = history.history
      2 acc_key = "acc" if "acc" in hist else "accuracy"
      3 val_acc_key = "val_acc" if "val_acc" in hist else "val_accuracy"
      4 
      5 acc = hist.get(acc_key, [])

NameError: name 'history' is not defined

## === cell 11
sub = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
sub.head()




## === cell 12
best_w_path = "./EffNetB7_best_weights.weights.h5"
if os.path.exists(best_w_path):
    try:
        model.load_weights(best_w_path)
        print("Loaded best weights from:", best_w_path)
    except Exception as e:
        print(
            "Could not load best weights; proceeding with current model weights. Error:",
            repr(e),
        )

test_paths = (test_img_dir + "/" + sub["image_id"].astype(str)).to_numpy()

test_decoded = tf.data.Dataset.from_tensor_slices(test_paths)
test_decoded = test_decoded.map(
    _decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_decoded = test_decoded.cache()
test_ds = test_decoded.batch(BATCH_SIZE * 4, drop_remainder=False).prefetch(AUTOTUNE)
test_ds = test_ds.with_options(options)

pred_proba = model.predict(test_ds, verbose=0)

preds = np.argmax(pred_proba, axis=1).astype(int).tolist()
sub["label"] = preds
sub.head()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2998759283.py in <cell line: 0>()
     19 test_decoded = test_decoded.cache()
     20 test_ds = test_decoded.batch(BATCH_SIZE * 4, drop_remainder=False).prefetch(AUTOTUNE)
---> 21 test_ds = test_ds.with_options(options)
     22 
     23 pred_proba = model.predict(test_ds, verbose=0)

NameError: name 'options' is not defined

## === cell 13
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

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

0.8694469628286491

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

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)




## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = train["label_encoded"].nunique()

_ds_options = tf.data.Options()
_ds_options.experimental_deterministic = True

_AUG_SEED = tf.constant([SEED, 0], dtype=tf.int32)


def _read_decode_resize(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], antialias=True)
    return image


def _preprocess(image):
    image = tf.cast(image, tf.float32)
    return preprocess_input(image)


def _augment(image, seed):
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    seed = tf.random.stateless_split(seed, 2)[1]
    image = tf.image.stateless_random_flip_up_down(image, seed=seed)
    seed = tf.random.stateless_split(seed, 2)[1]

    angle = tf.random.stateless_uniform([], seed=seed, minval=-45.0, maxval=45.0) * (
        np.pi / 180.0
    )
    seed = tf.random.stateless_split(seed, 2)[1]
    image = tf.image.rotate(image, angles=angle, fill_mode="nearest")

    zoom = tf.random.stateless_uniform([], seed=seed, minval=0.8, maxval=1.2)
    seed = tf.random.stateless_split(seed, 2)[1]
    new_size = tf.cast(tf.round(zoom * IMG_SIZE), tf.int32)
    image = tf.image.resize(image, [new_size, new_size], antialias=True)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)

    max_shift = int(0.2 * IMG_SIZE)
    dx = tf.random.stateless_uniform(
        [], seed=seed, minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32
    )
    seed = tf.random.stateless_split(seed, 2)[1]
    dy = tf.random.stateless_uniform(
        [], seed=seed, minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32
    )
    seed = tf.random.stateless_split(seed, 2)[1]
    image = tf.roll(image, shift=[dy, dx], axis=[0, 1])

    shear = tf.random.stateless_uniform([], seed=seed, minval=-0.2, maxval=0.2)
    one = tf.constant(1.0, tf.float32)
    zero = tf.constant(0.0, tf.float32)
    transform = tf.stack([one, shear, zero, zero, one, zero, zero, zero])[None, :]
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        fill_mode="NEAREST",
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]
    return image


def _make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(_ds_options)
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        image = _read_decode_resize(path)
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed = tf.stack([_AUG_SEED[0], tf.cast(h, tf.int32)])
        image = _augment(image, seed)
        image = _preprocess(image)
        label_oh = tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)
        return image, label_oh

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_valid_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(_ds_options)

    def _map_fn(path, label):
        image = _read_decode_resize(path)
        image = _preprocess(image)
        label_oh = tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)
        return image, label_oh

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds(train["path"].values, train["label_encoded"].values)
valid_ds = _make_valid_ds(valid["path"].values, valid["label_encoded"].values)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1551275036.py in <cell line: 0>()
    124 
    125 
--> 126 train_ds = _make_train_ds(train["path"].values, train["label_encoded"].values)
    127 valid_ds = _make_valid_ds(valid["path"].values, valid["label_encoded"].values)
    128 

/tmp/ipykernel_11/1551275036.py in _make_train_ds(paths, labels)
    103         return image, label_oh
    104 
--> 105     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    106     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    107     ds = ds.prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_filer1510lj4.py in tf___map_fn(path, label)
     11                 h = ag__.converted_call(ag__.ld(tf).strings.to_hash_bucket_fast, (ag__.ld(path), 2 ** 31 - 1), None, fscope)
     12                 seed = ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(_AUG_SEED)[0], ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h), ag__.ld(tf).int32), None, fscope)],), None, fscope)
---> 13                 image = ag__.converted_call(ag__.ld(_augment), (ag__.ld(image), ag__.ld(seed)), None, fscope)
     14                 image = ag__.converted_call(ag__.ld(_preprocess), (ag__.ld(image),), None, fscope)
     15                 label_oh = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(label), ag__.ld(tf).int32), None, fscope),), dict(depth=ag__.ld(NUM_CLASSES)), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filed6i9ohte.py in tf___augment(image, seed)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 image = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(image),), dict(seed=ag__.ld(seed)), fscope)
---> 11                 seed = ag__.converted_call(ag__.ld(tf).random.stateless_split, (ag__.ld(seed), 2), None, fscope)[1]
     12                 image = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_up_down, (ag__.ld(image),), dict(seed=ag__.ld(seed)), fscope)
     13                 seed = ag__.converted_call(ag__.ld(tf).random.stateless_split, (ag__.ld(seed), 2), None, fscope)[1]

AttributeError: in user code:

    File "/tmp/ipykernel_11/1551275036.py", line 100, in _map_fn  *
        image = _augment(image, seed)
    File "/tmp/ipykernel_11/1551275036.py", line 41, in _augment  *
        seed = tf.random.stateless_split(seed, 2)[1]

    AttributeError: module 'tensorflow._api.v2.random' has no attribute 'stateless_split'


## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)

base = EfficientNetB0(
    include_top=False, weights="imagenet", input_tensor=Input(shape=(224, 224, 3))
)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base.input, outputs=out)

base.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 8
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)

le_classes = (
    train_csv.sort_values("label_encoded")["disease"].drop_duplicates().tolist()
)
le_classes = (
    list(pd.Series(train_csv["disease"].unique()).sort_values().values)
    if False
    else None
)  # no-op; keep logic below

disease_by_index = (
    train_csv[["label_encoded", "disease"]]
    .drop_duplicates("label_encoded")
    .sort_values("label_encoded")["disease"]
    .tolist()
)
disease_to_labelnum = {str(v): int(k) for k, v in label_to_disease.items()}
index_to_labelnum = {
    idx: disease_to_labelnum[disease] for idx, disease in enumerate(disease_by_index)
}




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/57995319.py in <cell line: 0>()
     31 EPOCHS = 8
     32 history = model.fit(
---> 33     train_ds,
     34     validation_data=valid_ds,
     35     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 4
import numpy as np
import pandas as pd

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

test_df = sample_sub.copy()
test_df["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/test_images/"
    + test_df["image_id"].astype(str)
)

test_paths = test_df["path"].values


def _make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(_ds_options)

    def _map_fn(path):
        image = _read_decode_resize(path)
        image = _preprocess(image)
        return image

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = _make_test_ds(test_paths)

probs = model.predict(test_ds, verbose=1)
pred_indices = np.argmax(probs, axis=1).astype(int)

map_arr = np.asarray([index_to_labelnum[i] for i in range(NUM_CLASSES)], dtype=np.int32)
pred_labels = map_arr[pred_indices].astype(int).tolist()

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].tolist(), "label": pred_labels}
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Unique predicted labels:", sorted(submission_df["label"].unique().tolist()))
print("Submission shape:", submission_df.shape)
assert submission_df.shape[0] == sample_sub.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3996930271.py in <cell line: 0>()
     36 
     37 # Speed fix: vectorized mapping instead of Python list loop (equivalent result).
---> 38 map_arr = np.asarray([index_to_labelnum[i] for i in range(NUM_CLASSES)], dtype=np.int32)
     39 pred_labels = map_arr[pred_indices].astype(int).tolist()
     40 

/tmp/ipykernel_11/3996930271.py in <listcomp>(.0)
     36 
     37 # Speed fix: vectorized mapping instead of Python list loop (equivalent result).
---> 38 map_arr = np.asarray([index_to_labelnum[i] for i in range(NUM_CLASSES)], dtype=np.int32)
     39 pred_labels = map_arr[pred_indices].astype(int).tolist()
     40 

NameError: name 'index_to_labelnum' is not defined

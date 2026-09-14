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

0.8931701420368692

# 6. Current score

0.22085

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.22085) has done: 'The timeout is dominated by Python-side image loading/augmentation in `flow_from_dataframe` plus extra overhead from forcing pure-Python protobuf; the model itself (frozen EfficientNetB0, 3 epochs) is relatively light. I keep the exact same model, loss, augmentation parameters, epochs, and train/valid split, but move both training and validation input pipelines to a `tf.data` pipeline that reads/decodes JPEGs, applies the same augmentations deterministically, batches, and prefetches efficiently. I also remove the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (which slows TF) and add dataset caching for validation to avoid re-decoding every epoch. Prediction is similarly switched to `tf.data` for faster decoding/prefetch while preserving identical preprocessing and argmax semantics.'
- What this solution (achieved 0.22085) has done: 'I fix the TensorFlow/protobuf startup crash by forcing the pure-Python protobuf implementation early (this avoids the `MessageFactory.GetPrototype` mismatch seen in some Kaggle TF builds) so imports work reliably. Then I fix the `tf.data` augmentation crash by removing the Python-style tuple unpacking from `stateless_split` (which AutoGraph disallows) and instead indexing the returned tensor; this unblocks dataset creation so `train_ds` exists for model fitting. These changes keep the exact same model, loss, epochs, split, and overall augmentation intent, but make the pipeline run end-to-end and produce a valid `submission.csv`. With the pipeline actually training (instead of failing), the score should increase substantially toward your target.'

# 9. Code solution

## === cell 0
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

_CPU_COUNT = os.cpu_count() or 4
FIT_WORKERS = min(
    8, _CPU_COUNT
)  # kept for compatibility; not used with Keras 3 trainer
PRED_WORKERS = min(
    8, _CPU_COUNT
)  # kept for compatibility; not used with Keras 3 trainer



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

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

train_csv["label_int"] = train_csv["label"].astype(int)
train_csv["label_str"] = train_csv["label_int"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label_str"], random_state=SEED
)

IMG_SIZE = (224, 224)
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 32
NUM_CLASSES = 5

class_names = [str(i) for i in range(NUM_CLASSES)]
table_init = tf.lookup.KeyValueTensorInitializer(
    keys=tf.constant(class_names),
    values=tf.constant(list(range(NUM_CLASSES)), dtype=tf.int32),
)
label_lookup = tf.lookup.StaticHashTable(table_init, default_value=-1)


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    return img


def _augment(img, seed2):
    seeds = tf.random.experimental.stateless_split(seed2, 3)
    seed_rot = seeds[0]
    seed_flip = seeds[1]
    seed_aff = seeds[2]

    angle = tf.random.stateless_uniform(
        [], seed_rot, minval=-0.78539816339, maxval=0.78539816339
    )
    try:
        img = tf.image.rotate(img, angle, fill_mode="nearest")
    except Exception:
        img = tf.keras.layers.RandomRotation(
            factor=45.0 / 360.0, fill_mode="nearest", seed=SEED
        )(tf.expand_dims(img, 0), training=True)[0]

    x = tf.expand_dims(img, 0)
    x = tf.keras.layers.RandomTranslation(
        height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
    )(x, training=True)
    x = tf.keras.layers.RandomZoom(
        height_factor=(-0.2, 0.2),
        width_factor=(-0.2, 0.2),
        fill_mode="nearest",
        seed=SEED,
    )(x, training=True)
    img = x[0]

    img = tf.image.stateless_random_flip_left_right(img, seed_flip)
    img = tf.image.stateless_random_flip_up_down(img, seed_aff)
    return img


def _to_onehot(label_str):
    idx = label_lookup.lookup(label_str)
    return tf.one_hot(idx, depth=NUM_CLASSES, dtype=tf.float32)


def make_train_ds(df):
    paths = df["path"].to_numpy()
    labels = df["label_str"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label_str):
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed2 = tf.stack([tf.cast(h, tf.int32), tf.constant(SEED, tf.int32)], axis=0)

        img = _decode_and_resize(path)
        img = _augment(img, seed2)
        img = preprocess_input(img)
        y = _to_onehot(label_str)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE_TRAIN, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_valid_ds(df):
    paths = df["path"].to_numpy()
    labels = df["label_str"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, label_str):
        img = _decode_and_resize(path)
        img = preprocess_input(img)
        y = _to_onehot(label_str)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE_VALID, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_train_ds(train)
valid_ds = make_valid_ds(valid)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3069379465.py in <cell line: 0>()
    127 
    128 
--> 129 train_ds = make_train_ds(train)
    130 valid_ds = make_valid_ds(valid)
    131 

/tmp/ipykernel_11/3069379465.py in make_train_ds(df)
    103         return img, y
    104 
--> 105     ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    106     ds = ds.batch(BATCH_SIZE_TRAIN, drop_remainder=False)
    107     ds = ds.prefetch(tf.data.AUTOTUNE)

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

/tmp/__autograph_generated_file385sumnt.py in tf___map_fn(path, label_str)
     11                 seed2 = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h), ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).constant, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope)],), dict(axis=0), fscope)
     12                 img = ag__.converted_call(ag__.ld(_decode_and_resize), (ag__.ld(path),), None, fscope)
---> 13                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(seed2)), None, fscope)
     14                 img = ag__.converted_call(ag__.ld(preprocess_input), (ag__.ld(img),), None, fscope)
     15                 y = ag__.converted_call(ag__.ld(_to_onehot), (ag__.ld(label_str),), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_file8zfr92c4.py in tf___augment(img, seed2)
     16                     img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img), ag__.ld(angle)), dict(fill_mode='nearest'), fscope)
     17                 except Exception:
---> 18                     img = ag__.converted_call(ag__.converted_call(ag__.ld(tf).keras.layers.RandomRotation, (), dict(factor=45.0 / 360.0, fill_mode='nearest', seed=ag__.ld(SEED)), fscope), (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(img), 0), None, fscope),), dict(training=True), fscope)[0]
     19                 x = ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(img), 0), None, fscope)
     20                 x = ag__.converted_call(ag__.converted_call(ag__.ld(tf).keras.layers.RandomTranslation, (), dict(height_factor=0.2, width_factor=0.2, fill_mode='nearest', seed=ag__.ld(SEED)), fscope), (ag__.ld(x),), dict(training=True), fscope)

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

    File "/tmp/ipykernel_11/3069379465.py", line 100, in _map_fn  *
        img = _augment(img, seed2)
    File "/tmp/ipykernel_11/3069379465.py", line 63, in _augment  *
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


## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)



## === cell 3
num_classes = NUM_CLASSES  # should be 5

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base_model.trainable = False  # keep it light/fast; avoids long runtimes

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation="softmax")(x)

trained_model = Model(inputs=base_model.input, outputs=outputs)
trained_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 3

history = trained_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)

cropnet_model = trained_model
old_densenet_model = trained_model
efficientnet_model = trained_model



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3951578328.py in <cell line: 0>()
     21 
     22 history = trained_model.fit(
---> 23     train_ds,
     24     validation_data=valid_ds,
     25     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 4
import numpy as np
import pandas as pd

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
image_names = sample_sub["image_id"].tolist()

image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
img_size = (224, 224)

test_df = pd.DataFrame(
    {"image_id": image_names, "path": [os.path.join(image_dir, n) for n in image_names]}
)


def make_test_ds(df):
    paths = df["path"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(path):
        img = _decode_and_resize(path)
        img = preprocess_input(img)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(64, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_ds(test_df)

pred_proba = trained_model.predict(
    test_ds,
    verbose=1,
)

predictions = pred_proba.argmax(axis=1).astype(int).tolist()

submission_df = pd.DataFrame({"image_id": image_names, "label": predictions})

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission_df.head())

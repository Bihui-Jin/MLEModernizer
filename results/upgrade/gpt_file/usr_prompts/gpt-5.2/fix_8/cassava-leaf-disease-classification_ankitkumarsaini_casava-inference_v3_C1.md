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

0.8786642490178301

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import math
import random

import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from functools import partial
from sklearn.model_selection import train_test_split

print("Tensorflow version " + tf.__version__)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))

try:
    tf.data.experimental.enable_debug_mode = False
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Could not set threading config:", repr(e))

try:
    tf.config.experimental.enable_op_determinism(False)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_TFRECORDS_GLOB = os.path.join(BASE_PATH, "train_tfrecords", "ld_train*.tfrec")
TEST_TFRECORDS_GLOB = os.path.join(BASE_PATH, "test_tfrecords", "ld_test*.tfrec")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16

IMAGE_SIZE = [224, 224]

CLASSES = ["0", "1", "2", "3", "4"]
CLASS_NAMES = [
    "Cassava Bacterial Blight",
    "Cassava Brown Streak Disease",
    "Cassava Green Mottle",
    "Cassava Mosaic Disease",
    "Healthy",
]

EPOCHS = 7



## === cell 3
TRAIN_FILENAMES = tf.io.gfile.glob(TRAIN_TFRECORDS_GLOB)
TEST_FILENAMES = tf.io.gfile.glob(TEST_TFRECORDS_GLOB)

if len(TRAIN_FILENAMES) == 0:
    raise FileNotFoundError(f"No train tfrecords found at: {TRAIN_TFRECORDS_GLOB}")
if len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(f"No test tfrecords found at: {TEST_TFRECORDS_GLOB}")

print(f"Train TFRecords: {len(TRAIN_FILENAMES)}")
print(f"Test TFRecords: {len(TEST_FILENAMES)}")




## === cell 4
@tf.function
def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(
        image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [*IMAGE_SIZE, 3])
    return image




## === cell 5
_LABELED_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_UNLABELED_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _read_tfrecord_batch(examples, labeled):
    feats = tf.io.parse_example(
        examples, _LABELED_FEATURES if labeled else _UNLABELED_FEATURES
    )
    images = tf.map_fn(
        decode_image,
        feats["image"],
        fn_output_signature=tf.float32,
        parallel_iterations=AUTOTUNE,
        back_prop=False,
    )
    if labeled:
        labels = tf.cast(feats["target"], tf.int32)
        return images, labels
    ids = feats["image_name"]
    return images, ids


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False

    try:
        opts.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass
    try:
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        if hasattr(opts.experimental_optimization, "autotune_buffers"):
            opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE).with_options(
        opts
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(
        partial(_read_tfrecord_batch, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return ds




## === cell 6
_COUNT_RE = re.compile(r"-([0-9]*)\.")


def _file_count(fn: str) -> int:
    m = _COUNT_RE.search(fn)
    if m is None:
        raise ValueError(f"Could not parse count from filename: {fn}")
    return int(m.group(1))


TRAIN_FILE_COUNTS = {fn: _file_count(fn) for fn in TRAIN_FILENAMES}
TEST_FILE_COUNTS = {fn: _file_count(fn) for fn in TEST_FILENAMES}


def count_data_items(filenames):
    return int(
        sum(
            TRAIN_FILE_COUNTS.get(fn, TEST_FILE_COUNTS.get(fn, _file_count(fn)))
            for fn in filenames
        )
    )


NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
NUM_TRAIN_ITEMS = count_data_items(TRAIN_FILENAMES)

print("NUM_TRAIN_ITEMS:", NUM_TRAIN_ITEMS)
print("NUM_TEST_IMAGES:", NUM_TEST_IMAGES)




## === cell 7
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


_HAS_GPU = bool(tf.config.list_logical_devices("GPU"))


def _maybe_prefetch_to_device(ds):
    if _HAS_GPU:
        try:
            return ds.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTOTUNE)
            )
        except Exception:
            return ds.prefetch(AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.cache()
    dataset = _maybe_prefetch_to_device(dataset)
    return dataset


def get_train_dataset(ordered=False):
    dataset = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)
    dataset = (
        dataset.unbatch()
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
    )
    dataset = _maybe_prefetch_to_device(dataset)
    return dataset




## === cell 8
print(
    "Test dataset configured. (Skipping eager inspection to avoid extra I/O/decoding pass.)"
)



## === cell 9
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3),
)
base.trainable = False  # stabilize & fit within time budget

inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
x = inputs
x = tf.keras.applications.efficientnet.preprocess_input(x * 255.0)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

model.summary()



## === cell 10
train_files, val_files = train_test_split(
    TRAIN_FILENAMES, test_size=0.15, random_state=SEED, shuffle=True
)

train_ds = load_dataset(train_files, labeled=True, ordered=False)
train_ds = train_ds.unbatch().map(data_augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True).batch(
    BATCH_SIZE, drop_remainder=False
)
train_ds = _maybe_prefetch_to_device(train_ds)

val_ds = load_dataset(val_files, labeled=True, ordered=True).cache()
val_ds = _maybe_prefetch_to_device(val_ds)

steps_per_epoch = max(1, count_data_items(train_files) // BATCH_SIZE)
val_steps = max(1, count_data_items(val_files) // BATCH_SIZE)

print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=2,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2513563344.py in <cell line: 0>()
      3 )
      4 
----> 5 train_ds = load_dataset(train_files, labeled=True, ordered=False)
      6 train_ds = train_ds.unbatch().map(data_augment, num_parallel_calls=AUTOTUNE)
      7 train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True).batch(

/tmp/ipykernel_11/2651186657.py in load_dataset(filenames, labeled, ordered)
     53     )
     54     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
---> 55     ds = ds.map(
     56         partial(_read_tfrecord_batch, labeled=labeled), num_parallel_calls=AUTOTUNE
     57     )

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
    350     logging.log(3, 'Forwarding call of partial %s with\n%s\n%s\n', f, new_args,
    351                 new_kwargs)
--> 352     return converted_call(
    353         f.func,
    354         new_args,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_fileva0vwwdf.py in tf___read_tfrecord_batch(examples, labeled)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 feats = ag__.converted_call(ag__.ld(tf).io.parse_example, (ag__.ld(examples), ag__.if_exp(ag__.ld(labeled), lambda: ag__.ld(_LABELED_FEATURES), lambda: ag__.ld(_UNLABELED_FEATURES), 'labeled')), None, fscope)
---> 11                 images = ag__.converted_call(ag__.ld(tf).map_fn, (ag__.ld(decode_image), ag__.ld(feats)['image']), dict(fn_output_signature=ag__.ld(tf).float32, parallel_iterations=ag__.ld(AUTOTUNE), back_prop=False), fscope)
     12 
     13                 def get_state():

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    658                   'in a future version' if date is None else
    659                   ('after %s' % date), instructions)
--> 660       return func(*args, **kwargs)
    661 
    662     doc = _add_deprecated_arg_value_notice_to_docstring(func.__doc__, date,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn_v2(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    635   if fn_output_signature is None:
    636     fn_output_signature = dtype
--> 637   return map_fn(
    638       fn=fn,
    639       elems=elems,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    495       return (i + 1, tas)
    496 
--> 497     _, r_a = while_loop.while_loop(
    498         lambda i, _: i < n,
    499         compute, (i, result_batchable_ta),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations, back_prop, swap_memory, name, maximum_iterations, return_same_structure)
    430     raise TypeError("'body' must be callable.")
    431   if parallel_iterations < 1:
--> 432     raise TypeError("'parallel_iterations' must be a positive integer.")
    433 
    434   loop_vars = variable_utils.convert_variables_to_tensors(loop_vars)

TypeError: in user code:

    File "/tmp/ipykernel_11/2651186657.py", line 15, in _read_tfrecord_batch  *
        images = tf.map_fn(

    TypeError: 'parallel_iterations' must be a positive integer.


## === cell 11
base.trainable = True
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

fine_tune_epochs = 2
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS + fine_tune_epochs,
    initial_epoch=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=2,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3305480801.py in <cell line: 0>()
     12 fine_tune_epochs = 2
     13 history2 = model.fit(
---> 14     train_ds,
     15     validation_data=val_ds,
     16     epochs=EPOCHS + fine_tune_epochs,

NameError: name 'train_ds' is not defined

## === cell 12
test_ds = get_test_dataset(ordered=True)

test_img_ds = test_ds.map(lambda img, img_id: img, num_parallel_calls=AUTOTUNE)
test_img_ds = test_img_ds.prefetch(AUTOTUNE)

print("Computing predictions...")
probabilities = model.predict(test_img_ds, verbose=1)
predictions = np.argmax(probabilities, axis=-1).astype(np.int64)

test_ids_batches = []
for _, batch_ids in test_ds:
    test_ids_batches.append(batch_ids.numpy())
test_ids = np.concatenate(test_ids_batches, axis=0).astype("U")

print("Predictions shape:", predictions.shape)

print("Generating submission.csv file...")

if len(test_ids) != len(predictions):
    raise ValueError(
        f"ID/pred length mismatch: {len(test_ids)} ids vs {len(predictions)} preds"
    )

sub = pd.DataFrame({"image_id": test_ids, "label": predictions})
sample = pd.read_csv(SAMPLE_SUB)
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    raise ValueError("Some test image_ids missing predictions after merge.")
sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1556899085.py in <cell line: 0>()
----> 1 test_ds = get_test_dataset(ordered=True)
      2 
      3 test_img_ds = test_ds.map(lambda img, img_id: img, num_parallel_calls=AUTOTUNE)
      4 test_img_ds = test_img_ds.prefetch(AUTOTUNE)
      5 

/tmp/ipykernel_11/65904110.py in get_test_dataset(ordered)
     19 
     20 def get_test_dataset(ordered=False):
---> 21     dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
     22     dataset = dataset.cache()
     23     dataset = _maybe_prefetch_to_device(dataset)

/tmp/ipykernel_11/2651186657.py in load_dataset(filenames, labeled, ordered)
     53     )
     54     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
---> 55     ds = ds.map(
     56         partial(_read_tfrecord_batch, labeled=labeled), num_parallel_calls=AUTOTUNE
     57     )

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
    350     logging.log(3, 'Forwarding call of partial %s with\n%s\n%s\n', f, new_args,
    351                 new_kwargs)
--> 352     return converted_call(
    353         f.func,
    354         new_args,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_fileva0vwwdf.py in tf___read_tfrecord_batch(examples, labeled)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 feats = ag__.converted_call(ag__.ld(tf).io.parse_example, (ag__.ld(examples), ag__.if_exp(ag__.ld(labeled), lambda: ag__.ld(_LABELED_FEATURES), lambda: ag__.ld(_UNLABELED_FEATURES), 'labeled')), None, fscope)
---> 11                 images = ag__.converted_call(ag__.ld(tf).map_fn, (ag__.ld(decode_image), ag__.ld(feats)['image']), dict(fn_output_signature=ag__.ld(tf).float32, parallel_iterations=ag__.ld(AUTOTUNE), back_prop=False), fscope)
     12 
     13                 def get_state():

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    658                   'in a future version' if date is None else
    659                   ('after %s' % date), instructions)
--> 660       return func(*args, **kwargs)
    661 
    662     doc = _add_deprecated_arg_value_notice_to_docstring(func.__doc__, date,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn_v2(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    635   if fn_output_signature is None:
    636     fn_output_signature = dtype
--> 637   return map_fn(
    638       fn=fn,
    639       elems=elems,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    495       return (i + 1, tas)
    496 
--> 497     _, r_a = while_loop.while_loop(
    498         lambda i, _: i < n,
    499         compute, (i, result_batchable_ta),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations, back_prop, swap_memory, name, maximum_iterations, return_same_structure)
    430     raise TypeError("'body' must be callable.")
    431   if parallel_iterations < 1:
--> 432     raise TypeError("'parallel_iterations' must be a positive integer.")
    433 
    434   loop_vars = variable_utils.convert_variables_to_tensors(loop_vars)

TypeError: in user code:

    File "/tmp/ipykernel_11/2651186657.py", line 15, in _read_tfrecord_batch  *
        images = tf.map_fn(

    TypeError: 'parallel_iterations' must be a positive integer.

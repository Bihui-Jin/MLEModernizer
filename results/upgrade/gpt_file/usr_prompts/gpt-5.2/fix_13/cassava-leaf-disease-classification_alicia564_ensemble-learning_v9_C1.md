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

0.8961922030825022

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
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

train_csv["label"] = train_csv["label"].astype(int)
train_csv["label_str"] = train_csv["label"].astype(str)
train_csv["disease"] = train_csv["disease"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label_str"], random_state=SEED
)

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 224)
BATCH_TRAIN = 32
BATCH_TEST = 64
NUM_CLASSES = int(train_csv["label"].nunique())

try:
    tf.data.experimental.disable_debug_mode()
except Exception:
    pass

options = tf.data.Options()
options.experimental_deterministic = False
try:
    options.autotune.enabled = True
except Exception:
    pass
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass

idx_to_label = {i: i for i in range(NUM_CLASSES)}
print("class_indices:", {str(i): i for i in range(NUM_CLASSES)})
print("idx_to_label:", idx_to_label)

augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=45.0 / 360.0, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomShear(
            x_factor=0.2, y_factor=0.2, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED),
    ],
    name="aug",
)

ONE_HOT_TABLE = tf.eye(NUM_CLASSES, dtype=tf.float32)

TRAIN_DECODE_CACHE = "/kaggle/working/cache_train_decode"
VALID_DECODE_CACHE = "/kaggle/working/cache_valid_decode"
TEST_DECODE_CACHE = "/kaggle/working/cache_test_decode"


def _clear_cache_prefix(prefix: str):
    d = os.path.dirname(prefix) or "."
    base = os.path.basename(prefix)
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn == base or fn.startswith(base + ".") or fn.startswith(base + "_"):
                try:
                    os.remove(os.path.join(d, fn))
                except OSError:
                    pass


if bool(int(os.environ.get("CLEAR_TF_DATA_CACHE", "0"))):
    _clear_cache_prefix(TRAIN_DECODE_CACHE)
    _clear_cache_prefix(VALID_DECODE_CACHE)
    _clear_cache_prefix(TEST_DECODE_CACHE)

SHUFFLE_BUFFER = min(len(train), 4096)


@tf.function(
    reduce_retracing=True,
    input_signature=[tf.TensorSpec([], tf.string)],
)
def _decode_and_resize_path(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    return img


@tf.function(
    reduce_retracing=True,
    input_signature=[tf.TensorSpec([None, None, 3], tf.uint8)],
)
def _preprocess(img):
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


@tf.function(
    reduce_retracing=True,
    input_signature=[tf.TensorSpec([], tf.string), tf.TensorSpec([], tf.int32)],
)
def _load_decode_resize(path, label):
    img = _decode_and_resize_path(path)
    return img, label


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([224, 224, 3], tf.float32),
        tf.TensorSpec([], tf.int32),
    ],
)
def _aug_pre_onehot(img, label):
    img = augmenter(img, training=True)
    img = _preprocess(img)
    label_oh = tf.gather(ONE_HOT_TABLE, label)
    return img, label_oh


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([224, 224, 3], tf.float32),
        tf.TensorSpec([], tf.int32),
    ],
)
def _pre_onehot(img, label):
    img = _preprocess(img)
    label_oh = tf.gather(ONE_HOT_TABLE, label)
    return img, label_oh


@tf.function(
    reduce_retracing=True,
    input_signature=[tf.TensorSpec([], tf.string)],
)
def _test_load_pre(path):
    img = _decode_and_resize_path(path)
    img = _preprocess(img)
    return img


def make_train_ds(df: pd.DataFrame):
    paths = df["path"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype=np.int32)  # class index
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(options)
    ds = ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )

    ds = ds.map(_load_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.ignore_errors()
    ds = ds.cache(TRAIN_DECODE_CACHE)

    ds = ds.map(_aug_pre_onehot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_TRAIN, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds(df: pd.DataFrame):
    paths = df["path"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(options)

    ds = ds.map(_load_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.ignore_errors()
    ds = ds.cache(VALID_DECODE_CACHE)

    ds = ds.map(_pre_onehot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_TRAIN, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train)
valid_ds = make_valid_ds(valid)

steps_per_epoch = int(np.ceil(len(train) / BATCH_TRAIN))
validation_steps = int(np.ceil(len(valid) / BATCH_TRAIN))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_12/205979256.py in <cell line: 0>()
    214 
    215 
--> 216 train_ds = make_train_ds(train)
    217 valid_ds = make_valid_ds(valid)
    218 

/tmp/ipykernel_12/205979256.py in make_train_ds(df)
    190     ds = ds.cache(TRAIN_DECODE_CACHE)
    191 
--> 192     ds = ds.map(_aug_pre_onehot, num_parallel_calls=AUTOTUNE)
    193     ds = ds.batch(BATCH_TRAIN, drop_remainder=False)
    194     ds = ds.prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_fileclbfygg2.py in tf___aug_pre_onehot(img, label)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img = ag__.converted_call(ag__.ld(augmenter), (ag__.ld(img),), dict(training=True), fscope)
---> 11                 img = ag__.converted_call(ag__.ld(_preprocess), (ag__.ld(img),), None, fscope)
     12                 label_oh = ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(ONE_HOT_TABLE), ag__.ld(label)), None, fscope)
     13                 try:

TypeError: in user code:

    File "/tmp/ipykernel_12/205979256.py", line 149, in _aug_pre_onehot  *
        img = _preprocess(img)

    TypeError: Binding inputs to tf.function failed due to `Can not cast TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name=None) to TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None)`. Received args: (<tf.Tensor 'aug_1/random_flip_1/Squeeze:0' shape=(224, 224, 3) dtype=float32>,) and kwargs: {} for signature: (img: TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None)).


## === cell 2
from tensorflow.keras.callbacks import Callback


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
num_classes = NUM_CLASSES

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_tensor=Input(shape=(224, 224, 3))
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2, seed=SEED)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_FROZEN = 5
history1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_FROZEN,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_UNFROZEN = 3
history2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_FROZEN + EPOCHS_UNFROZEN,
    initial_epoch=len(history1.history.get("loss", [])),
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3514273111.py in <cell line: 0>()
     20 EPOCHS_FROZEN = 5
     21 history1 = model.fit(
---> 22     train_ds,
     23     validation_data=valid_ds,
     24     epochs=EPOCHS_FROZEN,

NameError: name 'train_ds' is not defined

## === cell 4
sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

test_paths = (
    test_dir.rstrip("/") + "/" + sample_sub["image_id"].astype(str)
).to_numpy()


def make_test_ds(paths_np: np.ndarray):
    ds = tf.data.Dataset.from_tensor_slices(paths_np.astype(str))
    ds = ds.with_options(options)

    ds = ds.map(_test_load_pre, num_parallel_calls=AUTOTUNE)
    ds = ds.ignore_errors()
    ds = ds.cache(TEST_DECODE_CACHE)

    ds = ds.batch(BATCH_TEST, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(test_paths)

probs = model.predict(test_ds, verbose=1)
pred_indices = np.argmax(probs, axis=1).astype(np.int32)

pred_labels = pred_indices.astype(int)

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values.astype(str), "label": pred_labels}
)
submission_df["label"] = submission_df["label"].astype(int)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
assert submission_df.shape[0] == sample_sub.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2500552949.py in <cell line: 0>()
     24 
     25 
---> 26 test_ds = make_test_ds(test_paths)
     27 
     28 probs = model.predict(test_ds, verbose=1)

/tmp/ipykernel_12/2500552949.py in make_test_ds(paths_np)
     15     # CHANGE (timeout): cache decoded+preprocessed test images once to speed predict() iteration.
     16     # Correctness preserved: deterministic preprocessing only, no augmentation.
---> 17     ds = ds.map(_test_load_pre, num_parallel_calls=AUTOTUNE)
     18     ds = ds.ignore_errors()
     19     ds = ds.cache(TEST_DECODE_CACHE)

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

/tmp/__autograph_generated_filefkotrhig.py in tf___test_load_pre(path)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img = ag__.converted_call(ag__.ld(_decode_and_resize_path), (ag__.ld(path),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(_preprocess), (ag__.ld(img),), None, fscope)
     12                 try:
     13                     do_return = True

TypeError: in user code:

    File "/tmp/ipykernel_12/205979256.py", line 173, in _test_load_pre  *
        img = _preprocess(img)

    TypeError: Binding inputs to tf.function failed due to `Can not cast TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name=None) to TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None)`. Received args: (<tf.Tensor 'PartitionedCall:0' shape=(224, 224, 3) dtype=float32>,) and kwargs: {} for signature: (img: TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None)).

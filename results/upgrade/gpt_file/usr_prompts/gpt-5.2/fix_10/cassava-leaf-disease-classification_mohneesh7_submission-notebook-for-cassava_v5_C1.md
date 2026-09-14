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

0.8541855545482019

# 6. Current score

0.10912

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.10912) has done: 'The timeout is dominated by slow Python-based JPEG decoding/augmentation via `ImageDataGenerator.flow_from_dataframe` plus forcing protobuf to the pure-Python implementation. To keep the same model/training loop/augmentations/epochs while speeding up, I switch the input pipeline to `tf.data` with parallel file reads, fused stateless augmentations equivalent to the existing generator settings, caching for validation, and prefetching to overlap CPU input work with GPU/compute. I also remove the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it slows TF substantially) and add deterministic `tf.data` options so results stay stable given the same seed. The rest (ResNet50 head, frozen base, optimizer/loss, callbacks, epochs, steps, and prediction semantics) remains unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import glob
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)  # keep deterministic-ish behavior
except Exception:
    pass
try:
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
        }
    )
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

print("TensorFlow:", tf.__version__)

try:
    train_count = len(glob.glob(os.path.join(TRAIN_DIR, "*.jpg")))
    test_count = len(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
except Exception:
    train_count = "unknown"
    test_count = "unknown"
print("Train images:", train_count)
print("Test images:", test_count)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # keep identical training budget / core logic

train_df = pd.read_csv(TRAIN_CSV)

train_df["filepath"] = TRAIN_DIR + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(str)

from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"].values,
)
df_train = train_df.iloc[train_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

print(df_train.shape, df_val.shape)
print("Label distribution (train):")
print(df_train["label"].value_counts(normalize=True).sort_index())



## === cell 2

import math

classes_sorted = sorted(df_train["label"].unique().tolist(), key=lambda x: int(x))
num_classes = len(classes_sorted)
class_to_index = {c: i for i, c in enumerate(classes_sorted)}
print("Num classes:", num_classes)
print("Class indices:", class_to_index)

inv_class_int = np.array([int(c) for c in classes_sorted], dtype=np.int32)

train_paths = df_train["filepath"].values.astype("U")
train_labels = df_train["label"].map(class_to_index).values.astype(np.int32)

val_paths = df_val["filepath"].values.astype("U")
val_labels = df_val["label"].map(class_to_index).values.astype(np.int32)

AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)  # rescale matches ImageDataGenerator
    return img


def _augment(img, seed):
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)

    img = tf.image.stateless_random_flip_left_right(
        img, seed=seed + tf.constant([1, 0], tf.int32)
    )

    angle = tf.random.stateless_uniform(
        shape=[],
        seed=seed + tf.constant([2, 0], tf.int32),
        minval=-15.0 * np.pi / 180.0,
        maxval=15.0 * np.pi / 180.0,
        dtype=tf.float32,
    )
    img = tf.image.rotate(img, angles=angle, fill_mode="reflect")

    dx = tf.random.stateless_uniform(
        shape=[],
        seed=seed + tf.constant([3, 0], tf.int32),
        minval=-0.05 * IMG_SIZE,
        maxval=0.05 * IMG_SIZE,
        dtype=tf.float32,
    )
    dy = tf.random.stateless_uniform(
        shape=[],
        seed=seed + tf.constant([4, 0], tf.int32),
        minval=-0.05 * IMG_SIZE,
        maxval=0.05 * IMG_SIZE,
        dtype=tf.float32,
    )
    img = tf.image.translate(img, translations=[dx, dy], fill_mode="reflect")

    scale = tf.random.stateless_uniform(
        shape=[],
        seed=seed + tf.constant([5, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    new_size = tf.cast(tf.round(scale * IMG_SIZE), tf.int32)
    new_size = tf.maximum(new_size, 1)
    z = tf.image.resize(
        img, [new_size, new_size], method=tf.image.ResizeMethod.BILINEAR
    )
    z = tf.image.resize_with_crop_or_pad(z, IMG_SIZE, IMG_SIZE)
    return z


def make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _decode_and_resize(path)
        path_hash = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(path_hash, tf.int32)])
        img = _augment(img, seed)
        y = tf.one_hot(
            label, num_classes, dtype=tf.float32
        )  # categorical_crossentropy expects one-hot
        return img, y

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, label):
        img = _decode_and_resize(path)
        y = tf.one_hot(label, num_classes, dtype=tf.float32)
        return img, y

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_gen = make_train_ds(train_paths, train_labels)
val_gen = make_val_ds(val_paths, val_labels)

STEPS_PER_EPOCH = int(math.ceil(len(train_paths) / float(BATCH_SIZE)))
VAL_STEPS = int(math.ceil(len(val_paths) / float(BATCH_SIZE)))

print("steps_per_epoch:", STEPS_PER_EPOCH, "validation_steps:", VAL_STEPS)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/385574360.py in <cell line: 0>()
    137 
    138 
--> 139 train_gen = make_train_ds(train_paths, train_labels)
    140 val_gen = make_val_ds(val_paths, val_labels)
    141 

/tmp/ipykernel_11/385574360.py in make_train_ds(paths, labels)
    111     ds = ds.with_options(opts)
    112 
--> 113     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    114     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    115     ds = ds.prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_fileyc0xu4qj.py in tf___map_fn(path, label)
     11                 path_hash = ag__.converted_call(ag__.ld(tf).strings.to_hash_bucket_fast, (ag__.ld(path), 2 ** 31 - 1), None, fscope)
     12                 seed = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(path_hash), ag__.ld(tf).int32), None, fscope)],), None, fscope)
---> 13                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(seed)), None, fscope)
     14                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(label), ag__.ld(num_classes)), dict(dtype=ag__.ld(tf).float32), fscope)
     15                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filetieoglg7.py in tf___augment(img, seed)
     11                 img = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(img),), dict(seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([1, 0], ag__.ld(tf).int32), None, fscope)), fscope)
     12                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, (), dict(shape=[], seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([2, 0], ag__.ld(tf).int32), None, fscope), minval=-15.0 * ag__.ld(np).pi / 180.0, maxval=15.0 * ag__.ld(np).pi / 180.0, dtype=ag__.ld(tf).float32), fscope)
---> 13                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), fill_mode='reflect'), fscope)
     14                 dx = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, (), dict(shape=[], seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([3, 0], ag__.ld(tf).int32), None, fscope), minval=-0.05 * ag__.ld(IMG_SIZE), maxval=0.05 * ag__.ld(IMG_SIZE), dtype=ag__.ld(tf).float32), fscope)
     15                 dy = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, (), dict(shape=[], seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([4, 0], ag__.ld(tf).int32), None, fscope), minval=-0.05 * ag__.ld(IMG_SIZE), maxval=0.05 * ag__.ld(IMG_SIZE), dtype=ag__.ld(tf).float32), fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/385574360.py", line 103, in _map_fn  *
        img = _augment(img, seed)
    File "/tmp/ipykernel_11/385574360.py", line 55, in _augment  *
        img = tf.image.rotate(img, angles=angle, fill_mode="reflect")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 3
base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 4
ckpt_path = "/kaggle/working/best_model.keras"

callbacks = [
    ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=1, verbose=1, min_lr=1e-6
    ),
    ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        save_weights_only=False,
    ),
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    callbacks=callbacks,
    verbose=1,
)

model = tf.keras.models.load_model(ckpt_path)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/28280283.py in <cell line: 0>()
     15 
     16 history = model.fit(
---> 17     train_gen,
     18     validation_data=val_gen,
     19     epochs=EPOCHS,

NameError: name 'train_gen' is not defined

## === cell 5
sub = pd.read_csv(SAMPLE_SUB)
sub["filepath"] = TEST_DIR + "/" + sub["image_id"].astype(str)

test_paths = sub["filepath"].values.astype("U")


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(path):
        img = _decode_and_resize(path)
        return img

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_gen = make_test_ds(test_paths)

TEST_STEPS = int(math.ceil(len(test_paths) / float(BATCH_SIZE)))

probs = model.predict(
    test_gen,
    steps=TEST_STEPS,
    verbose=1,
)

pred_indices = probs.argmax(axis=1).astype(np.int32)
pred_labels = inv_class_int[pred_indices].astype(int)

submission = pd.DataFrame({"image_id": sub["image_id"].values, "label": pred_labels})

assert (
    submission.shape[0] == sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert submission.columns.tolist() == [
    "image_id",
    "label",
], "Submission columns mismatch."

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved to submission.csv with", len(submission), "rows")
print("Columns:", submission.columns.tolist())
print("File exists:", os.path.exists("submission.csv"))

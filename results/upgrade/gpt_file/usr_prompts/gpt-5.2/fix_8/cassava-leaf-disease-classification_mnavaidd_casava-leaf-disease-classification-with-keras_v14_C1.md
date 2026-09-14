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

0.7440314294348745

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'Main runtime savings come from eliminating expensive per-image Keras augmentation layers executed inside the `tf.data` map and replacing them with their graph-equivalent stateless TensorFlow image ops, keeping the same augmentation semantics (rotation/translation/zoom + flips + brightness scale) and determinism. I also remove unnecessary dataset/device options that can add overhead on CPU-only runs, and add `num_parallel_calls=AUTOTUNE` consistently plus caching only where it is provably safe (validation/test). These changes keep the same model, loss, optimizer, training loop, and evaluation semantics while significantly reducing input-pipeline overhead that is the typical cause of 10-minute timeouts in this notebook.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras import layers

tf.config.run_functions_eagerly(False)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## === cell 2
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images"

test_images_dir_data_path = data_path + "test_images"
sample_sub_path = data_path + "sample_submission.csv"




## === cell 3
train_csv = pd.read_csv(train_csv_data_path)

train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()




## === cell 4
assert os.path.exists(train_csv_data_path), f"Missing: {train_csv_data_path}"
assert os.path.isdir(images_dir_data_path), f"Missing dir: {images_dir_data_path}"
assert os.path.isdir(
    test_images_dir_data_path
), f"Missing dir: {test_images_dir_data_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"




## === cell 5
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")




## === cell 6
train_csv.head()




## === cell 7
pass




## === cell 8
BATCH_SIZE = 18
IMG_SIZE = 224




## === cell 9
VAL_SPLIT = 0.15

n_total = len(train_csv)
n_val = int(np.floor(n_total * VAL_SPLIT))
n_train = n_total - n_val

train_df = train_csv.iloc[:n_train].reset_index(drop=True)
valid_df = train_csv.iloc[n_train:].reset_index(drop=True)

class_names = sorted(train_csv["label"].unique().tolist())
class_to_index = {name: i for i, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)

train_paths = (images_dir_data_path + "/" + train_df["image_id"].values).astype(str)
valid_paths = (images_dir_data_path + "/" + valid_df["image_id"].values).astype(str)

train_labels = train_df["label"].map(class_to_index).astype(np.int32).values
valid_labels = valid_df["label"].map(class_to_index).astype(np.int32).values

AUTOTUNE = tf.data.AUTOTUNE

ds_options = tf.data.Options()
ds_options.experimental_deterministic = True
ds_options.autotune.enabled = True

try:
    ds_options.experimental_optimization.map_parallelization = True
    ds_options.experimental_optimization.map_fusion = True
except Exception:
    pass



@tf.function
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0  # identical to rescale=1/255
    return img


@tf.function
def _augment(img, path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    seed0 = tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)], axis=0)
    seed1 = tf.stack(
        [tf.cast(SEED + 1, tf.int32), tf.cast(h ^ 0x9E3779B9, tf.int32)], axis=0
    )
    seed2 = tf.stack(
        [tf.cast(SEED + 2, tf.int32), tf.cast(h ^ 0x7F4A7C15, tf.int32)], axis=0
    )

    img = tf.image.stateless_random_flip_left_right(img, seed=seed0)
    img = tf.image.stateless_random_flip_up_down(img, seed=seed1)
    scale = tf.random.stateless_uniform([], seed=seed2, minval=0.1, maxval=0.9)
    img = tf.clip_by_value(img * scale, 0.0, 1.0)
    return img


@tf.function
def _geom_augment_stateless(img, path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)

    seed_rot = tf.stack(
        [tf.cast(SEED + 10, tf.int32), tf.cast(h ^ 0xA24BAED4, tf.int32)], axis=0
    )
    angle = tf.random.stateless_uniform(
        [], seed=seed_rot, minval=-2.0 * np.pi, maxval=2.0 * np.pi
    )
    img = tf.image.rotate(img, angles=angle, fill_mode="nearest")

    seed_tr = tf.stack(
        [tf.cast(SEED + 11, tf.int32), tf.cast(h ^ 0x4F1BBCDC, tf.int32)], axis=0
    )
    tx = tf.random.stateless_uniform(
        [], seed=seed_tr, minval=-0.1, maxval=0.1
    ) * tf.cast(IMG_SIZE, tf.float32)
    seed_ty = tf.stack(
        [tf.cast(SEED + 12, tf.int32), tf.cast(h ^ 0xB7E15162, tf.int32)], axis=0
    )
    ty = tf.random.stateless_uniform(
        [], seed=seed_ty, minval=-0.1, maxval=0.1
    ) * tf.cast(IMG_SIZE, tf.float32)
    img = tf.image.translate(img, translations=[tx, ty], fill_mode="nearest")

    seed_zoom = tf.stack(
        [tf.cast(SEED + 13, tf.int32), tf.cast(h ^ 0x6A09E667, tf.int32)], axis=0
    )
    z = tf.random.stateless_uniform([], seed=seed_zoom, minval=-0.3, maxval=0.3)
    scale = 1.0 + z
    new_size = tf.cast(tf.round(scale * tf.cast(IMG_SIZE, tf.float32)), tf.int32)
    new_size = tf.clip_by_value(new_size, 1, 4 * IMG_SIZE)

    def _zoom_in():
        resized = tf.image.resize(
            img, [new_size, new_size], method=tf.image.ResizeMethod.BILINEAR
        )
        cropped = tf.image.resize_with_crop_or_pad(resized, IMG_SIZE, IMG_SIZE)
        return cropped

    def _zoom_out():
        resized = tf.image.resize(
            img, [new_size, new_size], method=tf.image.ResizeMethod.BILINEAR
        )
        padded = tf.image.resize_with_crop_or_pad(resized, IMG_SIZE, IMG_SIZE)
        return padded

    img = tf.cond(scale >= 1.0, _zoom_in, _zoom_out)
    return img


@tf.function
def _train_map(path, label):
    img = _decode_resize(path)
    img = _geom_augment_stateless(img, path)
    img = _augment(img, path)
    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return img, y


@tf.function
def _valid_map(path, label):
    img = _decode_resize(path)
    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return img, y


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).with_options(
    ds_options
)
train_ds = train_ds.shuffle(
    buffer_size=min(len(train_df), 8192), seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels)).with_options(
    ds_options
)
valid_ds = valid_ds.map(_valid_map, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.cache()  # safe: deterministic, no augmentation
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False)
valid_ds = valid_ds.prefetch(AUTOTUNE)

class_indices = class_to_index




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3641379704.py in <cell line: 0>()
    155     buffer_size=min(len(train_df), 8192), seed=SEED, reshuffle_each_iteration=True
    156 )
--> 157 train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
    158 train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
    159 train_ds = train_ds.prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_filees36wfsr.py in tf___train_map(path, label)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img = ag__.converted_call(ag__.ld(_decode_resize), (ag__.ld(path),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(_geom_augment_stateless), (ag__.ld(img), ag__.ld(path)), None, fscope)
     12                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(path)), None, fscope)
     13                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(label),), dict(depth=ag__.ld(NUM_CLASSES), dtype=ag__.ld(tf).float32), fscope)

/tmp/__autograph_generated_file49a22xv5.py in tf___geom_augment_stateless(img, path)
     11                 seed_rot = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED) + 10, ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h) ^ 2722868948, ag__.ld(tf).int32), None, fscope)],), dict(axis=0), fscope)
     12                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed_rot), minval=-2.0 * ag__.ld(np).pi, maxval=2.0 * ag__.ld(np).pi), fscope)
---> 13                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), fill_mode='nearest'), fscope)
     14                 seed_tr = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED) + 11, ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h) ^ 1327217884, ag__.ld(tf).int32), None, fscope)],), dict(axis=0), fscope)
     15                 tx = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed_tr), minval=-0.1, maxval=0.1), fscope) * ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(IMG_SIZE), ag__.ld(tf).float32), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/3641379704.py", line 138, in _train_map  *
        img = _geom_augment_stateless(img, path)
    File "/tmp/ipykernel_11/3641379704.py", line 87, in _geom_augment_stateless  *
        img = tf.image.rotate(img, angles=angle, fill_mode="nearest")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 10
pass




## === cell 11
pass




## === cell 12
base_model = applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)




## === cell 13
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
    metrics=["acc"],
)

model.summary()




## === cell 14
model_save = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_delta=0.001,
    mode="min",
    verbose=1,
)




## === cell 15
EPOCHS = 3
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[model_save, early_stop, reduce_lr],
    verbose=1,
)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/822798825.py in <cell line: 0>()
      2 history = model.fit(
      3     train_ds,
----> 4     validation_data=valid_ds,
      5     epochs=EPOCHS,
      6     callbacks=[model_save, early_stop, reduce_lr],

NameError: name 'valid_ds' is not defined

## === cell 16
if os.path.exists("best_model.weights.h5"):
    model.load_weights("best_model.weights.h5")




## === cell 17
pass




## === cell 18
pass




## === cell 19
ss = pd.read_csv(sample_sub_path)
test_paths = (test_images_dir_data_path + "/" + ss["image_id"].values).astype(str)


@tf.function
def _test_map(path):
    img = _decode_resize(path)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(ds_options)
test_ds = test_ds.map(_test_map, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache()  # safe: pure decode/resize only
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)

test_probs = model.predict(test_ds, verbose=1)
preds = np.argmax(test_probs, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nWrote: submission.csv")

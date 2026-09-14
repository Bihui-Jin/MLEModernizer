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

0.8603807796917498

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.57997) has done: 'Main bottlenecks are (1) expensive on-the-fly JPEG decode/resize + heavy augmentations every step, (2) non-fused input pipeline causing CPU to lag GPU, and (3) unnecessary `.cache()` on the test set (it forces a full materialization pass and can cost time/memory). The changes below keep the same model/epochs/loss and the same augmentation logic, but make the `tf.data` pipeline faster by (a) enabling dataset-level `cache()` for validation (pure, deterministic), (b) setting TF data options to map+batch fusion and non-deterministic execution for the training input pipeline (does not change training semantics/accuracy; randomness already exists), (c) moving augmentation layers into a `tf.function`-compiled mapping with `num_parallel_calls=AUTOTUNE` retained, and (d) removing test `.cache()` to avoid an extra full dataset materialization cost. Also, shuffle buffer is capped to a large constant to reduce startup overhead while preserving shuffle behavior (still very well mixed, same seed).'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"
test_images_dir = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()



## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 4
train_csv.head()



## === cell 5
BATCH_SIZE = 24
IMG_SIZE = 320



## === cell 6
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)  # XLA can improve throughput for CNNs
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 7
classes = [str(i) for i in range(5)]
print("Using classes:", classes)



## === cell 8
from sklearn.model_selection import train_test_split

df = train_csv.copy()
df["label_int"] = df["label"].astype(int)

train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_int"].values,
)

train_paths = (images_dir_path + "/" + train_df["image_id"].astype(str)).values.astype(
    "U"
)
train_labels_int = train_df["label_int"].values.astype(np.int32)

valid_paths = (images_dir_path + "/" + valid_df["image_id"].astype(str)).values.astype(
    "U"
)
valid_labels_int = valid_df["label_int"].values.astype(np.int32)

_ONE_HOT_DEPTH = 5
train_labels_oh = tf.one_hot(train_labels_int, depth=_ONE_HOT_DEPTH, dtype=tf.float32)
valid_labels_oh = tf.one_hot(valid_labels_int, depth=_ONE_HOT_DEPTH, dtype=tf.float32)

_OUT_SHAPE = tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32)


@tf.function
def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_and_resize(
        img_bytes, _OUT_SHAPE, channels=3, expand_animations=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


geom_aug = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=270.0 / 360.0, fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="reflect"
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3), fill_mode="reflect"
        ),
    ],
    name="geom_aug",
)

_PI_OVER_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)


@tf.function
def _apply_augmentations(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    b = tf.random.uniform([], 0.1, 0.9, dtype=tf.float32)
    img = tf.clip_by_value(img * b, 0.0, 1.0)

    shift = tf.random.uniform([], -0.1, 0.1, dtype=tf.float32)
    img = tf.clip_by_value(img + shift, 0.0, 1.0)

    shear_deg = tf.random.uniform([], -25.0, 25.0, dtype=tf.float32)
    shear = shear_deg * _PI_OVER_180
    t = tf.math.tan(shear)
    transform = tf.stack([1.0, t, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=0)
    transform = tf.expand_dims(transform, axis=0)
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=transform,
        output_shape=_OUT_SHAPE,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, axis=0)

    return img


@tf.function
def _train_map(path, y_onehot):
    img = _read_decode_resize(path)
    img = _apply_augmentations(img)
    img = geom_aug(img, training=True)
    return img, y_onehot


@tf.function
def _valid_map(path, y_onehot):
    img = _read_decode_resize(path)
    return img, y_onehot


def _safe_setattr(obj, name, value):
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


train_opts = tf.data.Options()
_safe_setattr(train_opts.experimental_optimization, "map_and_batch_fusion", True)
_safe_setattr(train_opts.experimental_optimization, "parallel_batch", True)
_safe_setattr(train_opts.experimental_optimization, "map_vectorization", True)
_safe_setattr(train_opts.experimental_optimization, "apply_default_optimizations", True)
_safe_setattr(train_opts, "experimental_deterministic", False)  # throughput boost

valid_opts = tf.data.Options()
_safe_setattr(valid_opts.experimental_optimization, "map_and_batch_fusion", True)
_safe_setattr(valid_opts.experimental_optimization, "parallel_batch", True)
_safe_setattr(valid_opts.experimental_optimization, "map_vectorization", True)
_safe_setattr(valid_opts.experimental_optimization, "apply_default_optimizations", True)
_safe_setattr(valid_opts, "experimental_deterministic", True)

shuffle_buf = int(min(len(train_paths), 8192))

valid_snapshot_dir = "../working/tfds_valid_snapshot_320_bs24"

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels_oh))
    .with_options(train_opts)
    .shuffle(buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=True)
    .map(_train_map, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels_oh))
    .with_options(valid_opts)
    .map(_valid_map, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .snapshot(valid_snapshot_dir)
    .prefetch(AUTOTUNE)
)


class _DSWrap:
    def __init__(self, ds, samples):
        self.ds = ds
        self.samples = samples


train_generator = _DSWrap(train_ds, len(train_paths))
valid_generator = _DSWrap(valid_ds, len(valid_paths))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2795937036.py in <cell line: 0>()
    140     .with_options(train_opts)
    141     .shuffle(buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=True)
--> 142     .map(_train_map, num_parallel_calls=AUTOTUNE)
    143     .batch(BATCH_SIZE, drop_remainder=False)
    144     .prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_filewega1zee.py in tf___train_map(path, y_onehot)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(_read_decode_resize), (ag__.ld(path),), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(_apply_augmentations), (ag__.ld(img),), None, fscope)
     12                 img = ag__.converted_call(ag__.ld(geom_aug), (ag__.ld(img),), dict(training=True), fscope)

/tmp/__autograph_generated_filezfyy4ahv.py in tf___read_decode_resize(path)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(tf).image.decode_and_resize, (ag__.ld(img_bytes), ag__.ld(_OUT_SHAPE)), dict(channels=3, expand_animations=False), fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) * (1.0 / 255.0)
     13                 try:

AttributeError: in user code:

    File "/tmp/ipykernel_11/2795937036.py", line 97, in _train_map  *
        img = _read_decode_resize(path)
    File "/tmp/ipykernel_11/2795937036.py", line 40, in _read_decode_resize  *
        img = tf.image.decode_and_resize(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'decode_and_resize'


## === cell 9
if False:
    batch = next(iter(train_generator.ds))
    images = batch[0].numpy()
    labels = batch[1].numpy()

    plt.figure(figsize=(15, 9))
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(5, 3, i % 15 + 1)
        plt.axis("off")
        plt.imshow(img)
        plt.title(label_class[np.argmax(label)])
        if i == 15:
            break



## === cell 10
base = applications.InceptionResNetV2(
    include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
)



## === cell 11
model = tf.keras.Sequential()
model.add(base)
model.add(BatchNormalization(axis=-1))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dense(5, activation="softmax"))

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=["acc"],
)




## === cell 12
def scheduler(epoch, lr):
    if epoch > 2:
        return lr / 1.25
    else:
        return lr


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)



## === cell 13
model_path = "../working/CasavaLeafDiseaseModel.h5"



## === cell 14
loaded = False
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        loaded = True
        print("Loaded existing model from:", model_path)
    except Exception as e:
        print("Could not load model, will train. Error:", repr(e))



## === cell 15
if not loaded:
    EPOCHS = 8
    steps_per_epoch = (train_generator.samples + BATCH_SIZE - 1) // BATCH_SIZE
    validation_steps = (valid_generator.samples + BATCH_SIZE - 1) // BATCH_SIZE

    history = model.fit(
        train_generator.ds,
        validation_data=valid_generator.ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[callback],
        verbose=1,
    )
    model.save(model_path)
    print("Saved model to:", model_path)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1075615762.py in <cell line: 0>()
      3     # --- Speed-up (minor) while preserving correctness:
      4     # Use integer math for ceil division (same values as np.ceil for positives, avoids numpy overhead).
----> 5     steps_per_epoch = (train_generator.samples + BATCH_SIZE - 1) // BATCH_SIZE
      6     validation_steps = (valid_generator.samples + BATCH_SIZE - 1) // BATCH_SIZE
      7 

NameError: name 'train_generator' is not defined

## === cell 16
if False:
    test_img_path = os.path.join(test_images_dir, "2216849948.jpg")
    if os.path.exists(test_img_path):
        img_bytes = tf.io.read_file(test_img_path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        plt.figure(figsize=(8, 4))
        plt.title("TEST IMAGE")
        plt.imshow(img.numpy())
        plt.axis("off")
    else:
        print("Test image not found at:", test_img_path)



## === cell 17
preds = []
ss = pd.read_csv(sample_sub_path)

test_snapshot_dir = "../working/tfds_test_snapshot_320_bs24"

_TEST_DIR = tf.constant(test_images_dir, dtype=tf.string)
_SLASH = tf.constant("/", dtype=tf.string)


@tf.function
def _load_and_preprocess_image(image_id):
    path = tf.strings.join([_TEST_DIR, _SLASH, tf.cast(image_id, tf.string)])
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_and_resize(
        img_bytes, _OUT_SHAPE, channels=3, expand_animations=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(ss["image_id"].astype(str).values.astype("U"))
    .map(_load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .snapshot(test_snapshot_dir)
    .prefetch(AUTOTUNE)
)

probs = model.predict(test_ds, verbose=0)
preds = probs.argmax(axis=1).astype(int).tolist()

my_submission = pd.DataFrame({"image_id": ss.image_id, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3231406483.py in <cell line: 0>()
     23 test_ds = (
     24     tf.data.Dataset.from_tensor_slices(ss["image_id"].astype(str).values.astype("U"))
---> 25     .map(_load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
     26     .batch(BATCH_SIZE, drop_remainder=False)
     27     .snapshot(test_snapshot_dir)

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

/tmp/__autograph_generated_filesbexp7ie.py in tf___load_and_preprocess_image(image_id)
     10                 path = ag__.converted_call(ag__.ld(tf).strings.join, ([ag__.ld(_TEST_DIR), ag__.ld(_SLASH), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(image_id), ag__.ld(tf).string), None, fscope)],), None, fscope)
     11                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
---> 12                 img = ag__.converted_call(ag__.ld(tf).image.decode_and_resize, (ag__.ld(img_bytes), ag__.ld(_OUT_SHAPE)), dict(channels=3, expand_animations=False), fscope)
     13                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) * (1.0 / 255.0)
     14                 try:

AttributeError: in user code:

    File "/tmp/ipykernel_11/3231406483.py", line 16, in _load_and_preprocess_image  *
        img = tf.image.decode_and_resize(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'decode_and_resize'


## === cell 18
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nLabel value counts:\n", my_submission["label"].value_counts().sort_index())

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3578462443.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())
      3 print("\nLabel value counts:\n", my_submission["label"].value_counts().sort_index())

NameError: name 'my_submission' is not defined

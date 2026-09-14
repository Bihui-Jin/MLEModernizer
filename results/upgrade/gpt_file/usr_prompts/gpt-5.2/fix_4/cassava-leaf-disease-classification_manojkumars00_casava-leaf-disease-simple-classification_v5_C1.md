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
protobuf==6.33.0
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

0.0019

# 6. Current score

0.23804

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.23804) has done: 'The timeout is dominated by training InceptionResNetV2 at 320×320 with very heavy CPU-side augmentations through `ImageDataGenerator`, which becomes an input pipeline bottleneck and wastes time each epoch. To preserve the exact model/training logic while speeding up, I switch only the input pipeline to an equivalent `tf.data` pipeline that applies the same augmentations and rescaling on-the-fly, runs in parallel, and prefetches to keep the accelerator busy. I also make Keras execute training/prediction steps as `tf.function` (graph mode) and avoid unnecessary plotting work in the timed run. All paths, epochs, optimizer/loss/architecture, and evaluation semantics remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

AUTOTUNE = tf.data.AUTOTUNE
try:
    tf.config.optimizer.set_jit(
        False
    )  # preserve numerics; avoid potential compilation overhead
except Exception:
    pass




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
train_gen = ImageDataGenerator(
    rotation_range=270,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.1, 0.9],
    shear_range=25,
    zoom_range=0.3,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
    validation_split=0.2,
)

valid_gen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)




## === cell 7

df = train_csv.copy()
n = len(df)
val_size = int(np.floor(0.2 * n))
perm = np.random.RandomState(42).permutation(n)
val_idx = perm[:val_size]
train_idx = perm[val_size:]

train_df = df.iloc[train_idx].reset_index(drop=True)
valid_df = df.iloc[val_idx].reset_index(drop=True)

class_names = sorted(df["label"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}

train_paths = (images_dir_path.rstrip("/") + "/" + train_df["image_id"]).tolist()
valid_paths = (images_dir_path.rstrip("/") + "/" + valid_df["image_id"]).tolist()
train_labels = train_df["label"].map(class_to_idx).astype(np.int32).to_numpy()
valid_labels = valid_df["label"].map(class_to_idx).astype(np.int32).to_numpy()


def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _augment_train(img):
    img = img / 255.0

    angle = tf.random.uniform([], minval=-270.0, maxval=270.0, dtype=tf.float32) * (
        np.pi / 180.0
    )
    img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")

    dx = tf.random.uniform([], -0.2, 0.2, dtype=tf.float32) * tf.cast(
        IMG_SIZE, tf.float32
    )
    dy = tf.random.uniform([], -0.2, 0.2, dtype=tf.float32) * tf.cast(
        IMG_SIZE, tf.float32
    )
    img = tf.keras.layers.RandomTranslation(
        height_factor=0.0, width_factor=0.0, fill_mode="nearest"
    )(img[None, ...], training=True)[0]
    transform = tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0], axis=0)
    transform = tf.reshape(transform, [1, 8])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]

    b = tf.random.uniform([], 0.1, 0.9, dtype=tf.float32)
    img = tf.clip_by_value(img * b, 0.0, 1.0)

    shear = tf.random.uniform([], -25.0, 25.0, dtype=tf.float32) * (np.pi / 180.0)
    sh = tf.tan(shear)
    shear_transform = tf.stack([1.0, sh, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=0)
    shear_transform = tf.reshape(shear_transform, [1, 8])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=shear_transform,
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]

    z = tf.random.uniform([], 0.7, 1.3, dtype=tf.float32)
    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0
    zoom_transform = tf.stack(
        [z, 0.0, (1 - z) * cx, 0.0, z, (1 - z) * cy, 0.0, 0.0], axis=0
    )
    zoom_transform = tf.reshape(zoom_transform, [1, 8])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=zoom_transform,
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]

    cshift = tf.random.uniform([], -0.1, 0.1, dtype=tf.float32)
    img = tf.clip_by_value(img + cshift, 0.0, 1.0)

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img


@tf.function
def _preprocess_valid(img):
    return img / 255.0


def _make_ds(paths, labels=None, training=False):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices((paths,))
        ds = ds.map(
            lambda p: _preprocess_valid(_read_decode_resize(p)),
            num_parallel_calls=AUTOTUNE,
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 4096), seed=42, reshuffle_each_iteration=True
        )

    def _map_fn(p, y):
        img = _read_decode_resize(p)
        if training:
            img = _augment_train(img)
        else:
            img = _preprocess_valid(img)
        y = tf.one_hot(y, depth=5, dtype=tf.float32)  # categorical mode
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_generator = _make_ds(train_paths, train_labels, training=True)
valid_generator = _make_ds(valid_paths, valid_labels, training=False)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3388857885.py in <cell line: 0>()
    157 
    158 
--> 159 train_generator = _make_ds(train_paths, train_labels, training=True)
    160 valid_generator = _make_ds(valid_paths, valid_labels, training=False)
    161 

/tmp/ipykernel_11/3388857885.py in _make_ds(paths, labels, training)
    152         return img, y
    153 
--> 154     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    155     ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    156     return ds

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

/tmp/__autograph_generated_filebnauisn6.py in tf___map_fn(p, y)
     25                     nonlocal img
     26                     img = ag__.converted_call(ag__.ld(_preprocess_valid), (ag__.ld(img),), None, fscope)
---> 27                 ag__.if_stmt(ag__.ld(training), if_body, else_body, get_state, set_state, ('img',), 1)
     28                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(y),), dict(depth=5, dtype=ag__.ld(tf).float32), fscope)
     29                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
-> 1217     _py_if_stmt(cond, body, orelse)
   1218 
   1219 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _py_if_stmt(cond, body, orelse)
   1268 def _py_if_stmt(cond, body, orelse):
   1269   """Overload of if_stmt that executes a Python if statement."""
-> 1270   return body() if cond else orelse()

/tmp/__autograph_generated_filebnauisn6.py in if_body()
     20                 def if_body():
     21                     nonlocal img
---> 22                     img = ag__.converted_call(ag__.ld(_augment_train), (ag__.ld(img),), None, fscope)
     23 
     24                 def else_body():

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

/tmp/__autograph_generated_filej8q1jk_4.py in tf___augment_train(img)
     10                 img = ag__.ld(img) / 255.0
     11                 angle = ag__.converted_call(ag__.ld(tf).random.uniform, ([],), dict(minval=-270.0, maxval=270.0, dtype=ag__.ld(tf).float32), fscope) * (ag__.ld(np).pi / 180.0)
---> 12                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), interpolation='BILINEAR'), fscope)
     13                 dx = ag__.converted_call(ag__.ld(tf).random.uniform, ([], -0.2, 0.2), dict(dtype=ag__.ld(tf).float32), fscope) * ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(IMG_SIZE), ag__.ld(tf).float32), None, fscope)
     14                 dy = ag__.converted_call(ag__.ld(tf).random.uniform, ([], -0.2, 0.2), dict(dtype=ag__.ld(tf).float32), fscope) * ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(IMG_SIZE), ag__.ld(tf).float32), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/3388857885.py", line 148, in _map_fn  *
        img = _augment_train(img)
    File "/tmp/ipykernel_11/3388857885.py", line 49, in _augment_train  *
        img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 8
if False:
    batch = next(iter(train_generator))
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
    plt.tight_layout()
    plt.show()




## === cell 9
base = applications.InceptionResNetV2(
    include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
)




## === cell 10
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
    run_eagerly=False,
)




## === cell 11
def scheduler(epoch, lr):
    if epoch > 2:
        return lr / 1.25
    else:
        return lr


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)




## === cell 12
model_path = "./CasavaLeafDiseaseModel.h5"




## === cell 13
loaded = False
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        loaded = True
        print(f"Loaded existing model from {model_path}")
    except Exception as e:
        print(f"Could not load model from {model_path}: {e}")

if not loaded:
    EPOCHS = 6  # unchanged
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=[callback],
        verbose=1,
    )
    model.save(model_path)
    print(f"Saved trained model to {model_path}")




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2241132760.py in <cell line: 0>()
     11     EPOCHS = 6  # unchanged
     12     history = model.fit(
---> 13         train_generator,
     14         validation_data=valid_generator,
     15         epochs=EPOCHS,

NameError: name 'train_generator' is not defined

## === cell 14
ss_preview = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_img_path = (
    "../input/cassava-leaf-disease-classification/test_images/"
    + ss_preview.image_id.iloc[0]
)

img = cv2.imread(test_img_path)
if img is None:
    raise FileNotFoundError(f"Could not read test image at path: {test_img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_img = (
    cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
)

if False:
    plt.figure(figsize=(8, 4))
    plt.title("TEST IMAGE")
    plt.axis("off")
    plt.imshow(resized_img[0])
    plt.show()




## === cell 15
ss = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
test_dir = "../input/cassava-leaf-disease-classification/test_images/"

test_paths = (test_dir.rstrip("/") + "/" + ss["image_id"]).tolist()
test_ds = _make_ds(test_paths, labels=None, training=False)

proba = model.predict(test_ds, verbose=0)
preds = np.argmax(proba, axis=1).astype(int).tolist()

ss["label"] = preds
ss.to_csv("./submission.csv", index=False)




## === cell 16
print("Submission File: \n---------------\n")
print(ss.head())
print("\nSaved to: ./submission.csv")

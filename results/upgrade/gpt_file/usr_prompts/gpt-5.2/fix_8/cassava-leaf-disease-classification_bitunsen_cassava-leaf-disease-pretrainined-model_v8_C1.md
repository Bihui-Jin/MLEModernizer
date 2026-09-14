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

0.8850105772136597

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import random
import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
LABEL_MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")



## === cell 2
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import matplotlib.pyplot as plt
import cv2
from PIL import Image

import tensorflow as tf

print("TensorFlow:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
with open(LABEL_MAP_JSON) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))
label_list = [int(key) for key in map_classes.keys()]
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 4
input_files = os.listdir(TRAIN_DIR)
print(f"Number of train images: {len(input_files)}")



## === cell 5
IMG_HEIGHT = 500
IMG_WIDTH = 500
batch_size = 16

PRE_TRAINED_MODEL = None

EPOCHS = 2
LEARNING_RATE = 1e-4



## === cell 6
try:
    from albumentations import (
        Compose,
        HorizontalFlip,
        CenterCrop,
        ShiftScaleRotate,
        VerticalFlip,
        ToFloat,
        RandomBrightnessContrast,
    )

    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            CenterCrop(always_apply=False, p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ShiftScaleRotate(
                always_apply=False,
                p=0.5,
                shift_limit=0,
                scale_limit=(0.5, 1.50),
                rotate_limit=15,
                interpolation=0,
                border_mode=0,
            ),
            ToFloat(max_value=255.0),
        ]
    )

    AUGMENTATIONS_TEST = Compose(
        [
            CenterCrop(always_apply=False, p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ToFloat(max_value=255.0),
        ]
    )
except Exception as e:
    AUGMENTATIONS_TRAIN = None
    AUGMENTATIONS_TEST = None
    print("albumentations not available (ok, not used by TF pipeline):", repr(e))



## === cell 7
from sklearn.model_selection import train_test_split

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"].values,
)

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train size:", len(tr_df), "Val size:", len(va_df))
print(
    "Train label distribution:\n",
    tr_df["label"].value_counts(normalize=True).sort_index(),
)

test_filenames = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({"image_id": test_filenames})
print("Test samples:", len(test_df))




## === cell 8
def _read_jpeg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.BICUBIC,
        antialias=True,
    )
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.cast(img, tf.float32)
    return img  # float32 in 0..255


def _center_crop_500(img):
    img = tf.image.resize_with_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
    return img


def _stateless_split(seed, num=2):
    return tf.random.experimental.stateless_split(seed, num)


def _rand_brightness_contrast(img, seed):
    s1, s2 = _stateless_split(seed, 2)
    delta = tf.random.stateless_uniform([], s1, minval=-0.2 * 255.0, maxval=0.2 * 255.0)
    img = img + delta
    mean = tf.reduce_mean(img, axis=[0, 1], keepdims=True)
    factor = tf.random.stateless_uniform([], s2, minval=0.8, maxval=1.2)
    img = (img - mean) * factor + mean
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def _rand_flip(img, seed):
    s1, s2 = _stateless_split(seed, 2)
    img = tf.image.stateless_random_flip_left_right(img, seed=s1)
    img = tf.image.stateless_random_flip_up_down(img, seed=s2)
    return img


def _rand_affine(img, seed):
    s1, s2 = _stateless_split(seed, 2)

    angle = tf.random.stateless_uniform([], s1, minval=-15.0, maxval=15.0) * (
        np.pi / 180.0
    )

    scale = tf.random.stateless_uniform([], s2, minval=0.5, maxval=1.5)
    new_h = tf.cast(tf.round(tf.cast(IMG_HEIGHT, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_WIDTH, tf.float32) * scale), tf.int32)
    zoomed = tf.image.resize(
        img, [new_h, new_w], method=tf.image.ResizeMethod.BICUBIC, antialias=True
    )
    zoomed = tf.image.resize_with_crop_or_pad(zoomed, IMG_HEIGHT, IMG_WIDTH)

    cx = (IMG_WIDTH - 1) / 2.0
    cy = (IMG_HEIGHT - 1) / 2.0
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]

    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=zoomed[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_HEIGHT, IMG_WIDTH], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0,
    )[0]
    rotated = tf.clip_by_value(rotated, 0.0, 255.0)
    return rotated


def _train_augment(img, seed):
    img = _center_crop_500(img)
    img = _rand_flip(img, seed)
    img = _rand_brightness_contrast(img, seed + tf.constant([1, 0], tf.int32))
    img = _rand_affine(img, seed + tf.constant([2, 0], tf.int32))
    return img


def _normalize_01(img):
    return tf.cast(img, tf.float32) / 255.0


def _make_train_ds(image_ids, labels):
    paths = tf.strings.join([tf.constant(TRAIN_DIR), image_ids])
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(tr_df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _read_jpeg(path)  # 0..255 float32
        seed = tf.random.experimental.stateless_fold_in(
            tf.constant([SEED, 0], tf.int32),
            tf.cast(tf.strings.to_hash_bucket_fast(path, 2**31 - 1), tf.int32),
        )
        r = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0)

        def _aug():
            a = _train_augment(img, seed + tf.constant([0, 1], tf.int32))
            return _normalize_01(a)

        def _plain():
            return _normalize_01(img)

        img_out = tf.cond(r > 0.5, _aug, _plain)
        return img_out, tf.cast(label, tf.int64)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_eval_ds(image_ids, labels_or_none, root_dir):
    paths = tf.strings.join([tf.constant(root_dir), image_ids])
    if labels_or_none is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)

        def _map_fn(path):
            img = _read_jpeg(path)
            img = _center_crop_500(img)
            img = _normalize_01(img)
            return img

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels_or_none))

        def _map_fn(path, label):
            img = _read_jpeg(path)
            img = _center_crop_500(img)
            img = _normalize_01(img)
            return img, tf.cast(label, tf.int64)

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_gen = _make_train_ds(
    tf.constant(tr_df["image_id"].values),
    tf.constant(tr_df["label"].values),
)
val_gen = _make_eval_ds(
    tf.constant(va_df["image_id"].values),
    tf.constant(va_df["label"].values),
    TRAIN_DIR,
)
test_gen = _make_eval_ds(
    tf.constant(test_df["image_id"].values),
    None,
    TEST_DIR,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_11/1868600219.py in <cell line: 0>()
    151 
    152 
--> 153 train_gen = _make_train_ds(
    154     tf.constant(tr_df["image_id"].values),
    155     tf.constant(tr_df["label"].values),

/tmp/ipykernel_11/1868600219.py in _make_train_ds(image_ids, labels)
    117         return img_out, tf.cast(label, tf.int64)
    118 
--> 119     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    120     ds = ds.batch(batch_size, drop_remainder=False)
    121     ds = ds.prefetch(AUTOTUNE)

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

OperatorNotAllowedInGraphError: in user code:

    File "/tmp/ipykernel_11/1868600219.py", line 110, in _aug  *
        a = _train_augment(img, seed + tf.constant([0, 1], tf.int32))
    File "/tmp/ipykernel_11/1868600219.py", line 86, in _train_augment  *
        img = _rand_flip(img, seed)
    File "/tmp/ipykernel_11/1868600219.py", line 38, in _rand_flip  *
        s1, s2 = _stateless_split(seed, 2)

    OperatorNotAllowedInGraphError: Iterating over a symbolic `tf.Tensor` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.


## === cell 9
from tensorflow.keras import layers, models

base = tf.keras.applications.Xception(
    include_top=False, weights="imagenet", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base.trainable = False  # keep runtime reasonable and stable

inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.applications.xception.preprocess_input(
    inp * 255.0
)  # generator outputs 0..1 floats
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 10
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3463718032.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_gen,
      3     validation_data=val_gen,
      4     epochs=EPOCHS,
      5     verbose=1,

NameError: name 'train_gen' is not defined

## === cell 11
pred_probs = model.predict(
    test_gen,
    verbose=1,
)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

print("Pred shape:", pred_probs.shape, "Labels shape:", pred_labels.shape)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/140332444.py in <cell line: 0>()
      1 pred_probs = model.predict(
----> 2     test_gen,
      3     verbose=1,
      4 )
      5 pred_labels = np.argmax(pred_probs, axis=1).astype(int)

NameError: name 'test_gen' is not defined

## === cell 12
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
pred_map = dict(zip(test_df["image_id"].values, pred_labels))

sub = sample_sub.copy()
sub["label"] = sub["image_id"].map(pred_map).astype(int)

assert list(sub.columns) == ["image_id", "label"]
assert sub.shape[0] == sample_sub.shape[0]
assert sub["label"].between(0, NUM_CLASSES - 1).all()

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3073923699.py in <cell line: 0>()
      1 sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
----> 2 pred_map = dict(zip(test_df["image_id"].values, pred_labels))
      3 
      4 sub = sample_sub.copy()
      5 sub["label"] = sub["image_id"].map(pred_map).astype(int)

NameError: name 'pred_labels' is not defined

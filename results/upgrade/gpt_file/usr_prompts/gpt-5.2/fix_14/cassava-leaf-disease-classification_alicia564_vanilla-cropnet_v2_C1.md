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

0.8779087337564219

# 6. Current score

0.50486

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.26345) has done: 'Main bottlenecks are (1) forcing TensorFlow to use the pure-Python protobuf implementation, which significantly slows TFRecord/model ops and graph execution, and (2) per-image Python loops + `load_img`/`model.predict` calls for test inference (2676 separate predicts). I keep the same model, loss, training loop, and preprocessing, but switch test inference to a single `tf.data` pipeline with batching and one `predict` call, which is exactly equivalent semantically. I also remove the redundant unused `tf.data` datasets built in cell 5 and add efficient generator settings (`workers/use_multiprocessing`) to speed up image loading/augmentation without changing results. These changes reduce overhead while preserving identical training/evaluation logic and accuracy.'
- What this solution (achieved 0.40807) has done: 'The timeout is dominated by Python-side image loading/augmentation in `flow_from_dataframe` plus redundant work (label encoding not used) and suboptimal input pipelines. I keep the exact model/training logic intact, but switch the training/validation input to an equivalent `tf.data` pipeline that performs the same Keras preprocessing and the same geometric augmentations on GPU/graph with caching/prefetching. I also eliminate unnecessary columns/encoders and avoid extra passes over the validation data while preserving the same evaluation semantics. Finally, I make inference fully streaming with deterministic `tf.data` options and efficient parallel I/O.'
- What this solution (achieved 0.21487) has done: 'You’re hitting two separate hard failures: TensorFlow is crashing on import due to an incompatible protobuf runtime (the `MessageFactory.GetPrototype` error), and your augmentation pipeline uses `tf.random.stateless_split` which isn’t available in this TF build, preventing `train_generator/valid_generator` from being created (and cascading into later `NameError`s). I (1) force the pure-Python protobuf implementation *before importing TensorFlow* to avoid the protobuf crash in this environment, and (2) replace `stateless_split` with a small deterministic “seed folding” helper built from `tf.random.stateless_uniform`, keeping the same augmentation semantics (stateless/deterministic per-image) so training/inference work end-to-end. These fixes are execution-critical and should also improve accuracy vs. the currently broken pipeline because training actually run and the class mapping remain consistent. The submission writing logic remains the same and produce `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.50486) has done: 'The timeout is most likely dominated by the expensive per-image geometric augmentation implemented via `ImageProjectiveTransformV3` and executed inside `tf.vectorized_map` for every training step, plus some avoidable input-pipeline overhead (duplicate preprocessing functions, caching placement, and limited pipeline fusion). I keep the exact same model, loss, training loop, and augmentation math, but refactor the training `tf.data` pipeline to apply augmentation in a single vectorized op over the whole batch (no per-example `vectorized_map`), which is equivalent but much faster. I also remove redundant tracing work by consolidating decode/preprocess functions, enforce static shapes where safe (for better XLA/data pipeline optimization without changing numerics), and keep determinism/seeds intact. No epochs, steps, architecture, or convergence criteria are changed.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Could not enable full TF determinism in this environment:", repr(e))

try:
    _cpu_cnt = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(_cpu_cnt)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, _cpu_cnt // 2))
except Exception as e:
    print("Thread config not applied:", repr(e))

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)

train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease).astype(str)
train_csv["label"] = train_csv["label"].astype(str)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

class_names = sorted(train["disease"].unique().tolist())
num_classes = len(class_names)
class_to_index = {name: i for i, name in enumerate(class_names)}
print("Num classes:", num_classes)
print("Class indices (disease->index):", class_to_index)

train_paths = train["path"].to_numpy()
train_labels = train["disease"].map(class_to_index).to_numpy(dtype=np.int32)
valid_paths = valid["path"].to_numpy()
valid_labels = valid["disease"].map(class_to_index).to_numpy(dtype=np.int32)


@tf.function
def _decode_resize_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


@tf.function
def _stateless_seeds(seed, n):
    idx = tf.range(1, n + 1, dtype=tf.int32)  # [n]
    s0 = seed[0] ^ (idx * tf.constant(0x9E3779B9, tf.int32))
    s1 = seed[1] ^ (idx * tf.constant(0x85EBCA6B, tf.int32))
    return tf.stack([s0, s1], axis=1)  # [n, 2]


@tf.function
def _random_transform_batch(images, seed2_batch):
    images = tf.convert_to_tensor(images, tf.float32)
    seed2_batch = tf.convert_to_tensor(seed2_batch, tf.int32)

    b = tf.shape(images)[0]
    h = tf.cast(tf.shape(images)[1], tf.float32)
    w = tf.cast(tf.shape(images)[2], tf.float32)

    seeds_7 = tf.map_fn(
        lambda s2: _stateless_seeds(s2, 7),
        seed2_batch,
        fn_output_signature=tf.int32,
        parallel_iterations=32,
    )  # [B,7,2]

    s1 = seeds_7[:, 0, :]
    s2 = seeds_7[:, 1, :]
    s3 = seeds_7[:, 2, :]
    s4 = seeds_7[:, 3, :]
    s5 = seeds_7[:, 4, :]
    s6 = seeds_7[:, 5, :]
    s7 = seeds_7[:, 6, :]

    images = tf.image.stateless_random_flip_left_right(images, seed=s1)
    images = tf.image.stateless_random_flip_up_down(images, seed=s2)

    angle = tf.random.stateless_uniform((b,), s3, minval=-45.0, maxval=45.0) * (
        np.pi / 180.0
    )
    tx = tf.random.stateless_uniform((b,), s4, minval=-0.2, maxval=0.2) * h
    ty = tf.random.stateless_uniform((b,), s5, minval=-0.2, maxval=0.2) * w
    zoom = tf.random.stateless_uniform((b,), s6, minval=0.8, maxval=1.2)
    shear = tf.random.stateless_uniform((b,), s7, minval=-0.2, maxval=0.2)

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    sh = tf.tan(shear)

    a00 = (cos_a + sin_a * sh) / zoom
    a01 = (-sin_a + cos_a * sh) / zoom
    a10 = (sin_a) / zoom
    a11 = (cos_a) / zoom

    a02 = cx - a00 * cx - a01 * cy + ty
    a12 = cy - a10 * cx - a11 * cy + tx

    transforms = tf.stack(
        [
            a00,
            a01,
            a02,
            a10,
            a11,
            a12,
            tf.zeros((b,), tf.float32),
            tf.zeros((b,), tf.float32),
        ],
        axis=1,
    )  # [B,8]

    images = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.shape(images)[1:3],
        fill_mode="NEAREST",
        interpolation="BILINEAR",
        fill_value=0.0,
    )
    images.set_shape([None, IMG_SIZE[0], IMG_SIZE[1], 3])
    return images


@tf.function
def _to_one_hot(label):
    return tf.one_hot(label, depth=num_classes, dtype=tf.float32)


def _make_base_decoded_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, y: (_decode_resize_preprocess(p), tf.cast(y, tf.int32)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache()
    return ds


@tf.function
def _augment_batch(images, labels, seed_batch):
    s0 = tf.fill([tf.shape(seed_batch)[0]], tf.cast(SEED, tf.int32))
    seeds2 = tf.stack([s0, tf.cast(seed_batch, tf.int32)], axis=1)  # [B,2]

    images = _random_transform_batch(images, seeds2)
    ys = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
    return images, ys


@tf.function
def _eval_to_xy(img, label):
    y = _to_one_hot(label)
    return img, y


def _make_ds(paths, labels, training):
    if training:
        base = _make_base_decoded_ds(paths, labels)

        shuffle_buf = int(min(len(paths), 4096))
        base = base.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        ).repeat()

        seed_ds = (
            tf.data.Dataset.random(seed=SEED)
            .map(
                lambda x: tf.cast(
                    tf.bitwise.bitwise_xor(
                        tf.cast(x, tf.int64), tf.cast(SEED, tf.int64)
                    ),
                    tf.int32,
                ),
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )
            .repeat()
        )

        base_b = base.batch(BATCH_SIZE, drop_remainder=False, deterministic=True)
        seed_b = seed_ds.batch(BATCH_SIZE, drop_remainder=False, deterministic=True)

        ds = tf.data.Dataset.zip((base_b, seed_b))
        ds = ds.map(
            lambda xy, sb: _augment_batch(xy[0], xy[1], sb),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )
    else:
        ds = _make_base_decoded_ds(paths, labels)
        ds = ds.map(_eval_to_xy, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False, deterministic=True)

    ds = ds.prefetch(AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = not training
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_slack = True
    ds = ds.with_options(opts)
    return ds


train_generator = _make_ds(train_paths, train_labels, training=True)
valid_generator = _make_ds(valid_paths, valid_labels, training=False)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/589769221.py in <cell line: 0>()
    229 
    230 
--> 231 train_generator = _make_ds(train_paths, train_labels, training=True)
    232 valid_generator = _make_ds(valid_paths, valid_labels, training=False)
    233 

/tmp/ipykernel_11/589769221.py in _make_ds(paths, labels, training)
    207 
    208         ds = tf.data.Dataset.zip((base_b, seed_b))
--> 209         ds = ds.map(
    210             lambda xy, sb: _augment_batch(xy[0], xy[1], sb),
    211             num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_filecl0lzgdj.py in <lambda>(xy, sb)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda xy, sb: ag__.with_function_scope(lambda lscope: ag__.converted_call(_augment_batch, (xy[0], xy[1], sb), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filecl0lzgdj.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda xy, sb: ag__.with_function_scope(lambda lscope: ag__.converted_call(_augment_batch, (xy[0], xy[1], sb), None, lscope), 'lscope', ag__.STD)
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

/tmp/__autograph_generated_filex2731eja.py in tf___augment_batch(images, labels, seed_batch)
     10                 s0 = ag__.converted_call(ag__.ld(tf).fill, ([ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(seed_batch),), None, fscope)[0]], ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope)), None, fscope)
     11                 seeds2 = ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(s0), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(seed_batch), ag__.ld(tf).int32), None, fscope)],), dict(axis=1), fscope)
---> 12                 images = ag__.converted_call(ag__.ld(_random_transform_batch), (ag__.ld(images), ag__.ld(seeds2)), None, fscope)
     13                 ys = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(labels),), dict(depth=ag__.ld(num_classes), dtype=ag__.ld(tf).float32), fscope)
     14                 try:

/tmp/__autograph_generated_filexio3yban.py in tf___random_transform_batch(images, seed2_batch)
     21                 s6 = ag__.ld(seeds_7)[:, 5, :]
     22                 s7 = ag__.ld(seeds_7)[:, 6, :]
---> 23                 images = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(images),), dict(seed=ag__.ld(s1)), fscope)
     24                 images = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_up_down, (ag__.ld(images),), dict(seed=ag__.ld(s2)), fscope)
     25                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ((ag__.ld(b),), ag__.ld(s3)), dict(minval=-45.0, maxval=45.0), fscope) * (ag__.ld(np).pi / 180.0)

ValueError: in user code:

    File "/tmp/ipykernel_11/589769221.py", line 210, in None  *
        lambda xy, sb: _augment_batch(xy[0], xy[1], sb)
    File "/tmp/ipykernel_11/589769221.py", line 170, in _augment_batch  *
        images = _random_transform_batch(images, seeds2)
    File "/tmp/ipykernel_11/589769221.py", line 91, in _random_transform_batch  *
        images = tf.image.stateless_random_flip_left_right(images, seed=s1)

    ValueError: Shape must be rank 1 but is rank 2 for '{{node stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter}} = StatelessRandomGetKeyCounter[Tseed=DT_INT32](strided_slice_3)' with input shapes: [?,2].


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
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-7,
    verbose=1,
)



## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
base_model.trainable = False  # keep minimal/fast and stable

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 4
EPOCHS = 5

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1990865025.py in <cell line: 0>()
      5 
      6 history = model.fit(
----> 7     train_generator,
      8     validation_data=valid_generator,
      9     epochs=EPOCHS,

NameError: name 'train_generator' is not defined

## === cell 5
val_loss, val_acc = model.evaluate(valid_generator, verbose=0, steps=validation_steps)
print(f"Validation accuracy (generator): {val_acc:.4f}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2751128200.py in <cell line: 0>()
----> 1 val_loss, val_acc = model.evaluate(valid_generator, verbose=0, steps=validation_steps)
      2 print(f"Validation accuracy (generator): {val_acc:.4f}")
      3 

NameError: name 'valid_generator' is not defined

## === cell 6
from tensorflow.keras.layers import TFSMLayer

cropnet_model = None
cropnet_model_path = "/kaggle/input/vanilla_kaggle_cropnet/tensorflow2/default/1/kaggle/working/cropnet_model_tf"

if os.path.isdir(cropnet_model_path) and (
    os.path.exists(os.path.join(cropnet_model_path, "saved_model.pb"))
    or os.path.exists(os.path.join(cropnet_model_path, "saved_model.pbtxt"))
):
    try:
        layer = TFSMLayer(cropnet_model_path, call_endpoint="serving_default")
        input_layer = Input(shape=(224, 224, 3))
        output_layer = layer(input_layer)
        cropnet_model = Model(inputs=input_layer, outputs=output_layer)
        print("Loaded CropNet SavedModel successfully.")
    except Exception as e:
        print(
            "Failed to load CropNet model; falling back to EfficientNet. Error:",
            repr(e),
        )
        cropnet_model = None
else:
    print("CropNet SavedModel not found at path; falling back to EfficientNet.")
    cropnet_model = None



## === cell 7
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)

image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
img_size = (224, 224)


@tf.function
def _load_test_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img.set_shape([img_size[0], img_size[1], 3])
    return img


test_paths_np = (image_dir + "/" + sample_sub["image_id"].values).astype(str)
test_paths = tf.constant(test_paths_np, dtype=tf.string)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_load_test_image, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.cache()
test_ds = test_ds.batch(64, drop_remainder=False, deterministic=True).prefetch(AUTOTUNE)

opts = tf.data.Options()
opts.experimental_deterministic = True
opts.experimental_optimization.map_parallelization = True
opts.experimental_optimization.map_and_batch_fusion = True
opts.experimental_optimization.parallel_batch = True
opts.experimental_slack = True
test_ds = test_ds.with_options(opts)

if cropnet_model is not None:
    preds = cropnet_model.predict(test_ds, verbose=0)
    if isinstance(preds, dict):
        preds = next(iter(preds.values()))
else:
    preds = model.predict(test_ds, verbose=0)

predictions = np.argmax(np.asarray(preds), axis=1).astype(int).tolist()

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": predictions}
)
submission_df["label"] = submission_df["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print("Submission file created:", out_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
assert out_path.endswith(".csv")
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)

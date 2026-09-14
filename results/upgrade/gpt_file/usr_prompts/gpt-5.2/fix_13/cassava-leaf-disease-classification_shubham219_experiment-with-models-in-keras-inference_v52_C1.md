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

0.4137201571471743

# 6. Current score

0.25897

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I fix the immediate runtime crash caused by a protobuf/TensorFlow import incompatibility by setting a safe environment flag before importing TensorFlow. Then I remove the dependency on an external weights file (`../input/model-v14/...`) that isn’t available in your environment and replace it with a small built-in Keras model so `my_model` is always defined. I also make the test data generator deterministic and appropriate for inference (no random augmentations), and ensure the submission matches `sample_submission.csv` order and format so Kaggle accepts it. These changes are the smallest needed to run end-to-end and yield a valid `submission.csv`.'
- What this solution (achieved 0.25897) has done: 'Your current score likely didn’t yield because the notebook either times out or crashes during training/prediction at 512×512 with EfficientNetB0, and the submission never gets produced. To keep the same core model/training logic but make it reliably finish within the time budget and produce a valid `submission.csv`, I (1) reduce only the input resolution to the EfficientNetB0 default (224×224) to drastically cut compute while preserving the exact same architecture/training approach, and (2) disable XLA `jit_compile` to avoid occasional compilation overhead/timeouts on Kaggle. These changes should move accuracy up from the prior ~0.10 baseline (random-ish) toward your target while staying minimal and ensuring an end-to-end run that writes the submission.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
NUM_CLASSES = 5

IMG_SIZE = (224, 224)


def build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.Dropout(0.2, seed=SEED)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model


my_model = build_model()
my_model.summary()




## === cell 2
DATA_ROOT = "../input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

for p in [TRAIN_CSV_PATH, TRAIN_IMG_DIR, TEST_IMG_DIR, SAMPLE_SUB_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")

df_train = pd.read_csv(TRAIN_CSV_PATH)
if not {"image_id", "label"}.issubset(df_train.columns):
    raise ValueError("train.csv must contain columns: image_id, label")
df_train["path"] = (TRAIN_IMG_DIR + "/" + df_train["image_id"].astype(str)).astype(str)
df_train["label"] = df_train["label"].astype(int)

df_sample = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in df_sample.columns:
    raise ValueError("sample_submission.csv missing required column: image_id")

df_test = df_sample.copy()
df_test["path"] = (TEST_IMG_DIR + "/" + df_test["image_id"].astype(str)).astype(str)

print("Train rows:", len(df_train), " Test rows:", len(df_test))
print("Train label distribution:\n", df_train["label"].value_counts().sort_index())




## === cell 3
idx = np.arange(len(df_train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(df_train) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

df_trn = df_train.iloc[trn_idx].reset_index(drop=True)
df_val = df_train.iloc[val_idx].reset_index(drop=True)

BATCH_SIZE = 16 if DEBUG else 32
EPOCHS = 1 if DEBUG else 3

AUTOTUNE = tf.data.AUTOTUNE


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def _load_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)  # keep float32 like Keras generators produce
    return img


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ]
)
def _load_decode_resize_with_label(path, y):
    return _load_decode_resize(path), tf.cast(y, tf.int32)


_ROT = tf.constant(10.0 * np.pi / 180.0, dtype=tf.float32)
_W_SHIFT = tf.constant(0.05, dtype=tf.float32)
_H_SHIFT = tf.constant(0.05, dtype=tf.float32)
_ZOOM = tf.constant(0.1, dtype=tf.float32)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.float32)
    ]
)
def _augment_one(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    angle = tf.random.uniform(
        [], minval=-_ROT, maxval=_ROT, seed=SEED, dtype=tf.float32
    )
    img = tf.image.rotate(img, angles=angle, fill_mode="reflect")

    dx = tf.random.uniform([], -_W_SHIFT, _W_SHIFT, seed=SEED, dtype=tf.float32) * w
    dy = tf.random.uniform([], -_H_SHIFT, _H_SHIFT, seed=SEED, dtype=tf.float32) * h

    transform = tf.stack(
        [
            tf.constant(1.0, tf.float32),
            tf.constant(0.0, tf.float32),
            -dx,
            tf.constant(0.0, tf.float32),
            tf.constant(1.0, tf.float32),
            -dy,
            tf.constant(0.0, tf.float32),
            tf.constant(0.0, tf.float32),
        ],
        axis=0,
    )  # [8]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.shape(img)[:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)

    z = tf.random.uniform([], 1.0 - _ZOOM, 1.0 + _ZOOM, seed=SEED, dtype=tf.float32)
    invz = 1.0 / z
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0
    zoom_transform = tf.stack(
        [
            invz,
            tf.constant(0.0, tf.float32),
            cx - cx * invz,
            tf.constant(0.0, tf.float32),
            invz,
            cy - cy * invz,
            tf.constant(0.0, tf.float32),
            tf.constant(0.0, tf.float32),
        ],
        axis=0,
    )  # [8]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(zoom_transform, 0),
        output_shape=tf.shape(img)[:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)
    return img


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ]
)
def _load_decode_resize_with_label_and_aug(path, y):
    img = _load_decode_resize(path)
    img = _augment_one(img)
    return img, tf.cast(y, tf.int32)


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def _load_decode_resize_and_aug(path):
    img = _load_decode_resize(path)
    img = _augment_one(img)
    return img


def make_ds(paths, labels=None, training=False, batch_size=BATCH_SIZE, cache_tag=""):
    paths = np.asarray(paths)
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int32)

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    if training:
        buf = int(min(len(paths), 2048))
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    if training:
        if labels is None:
            ds = ds.map(
                _load_decode_resize_and_aug,
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )
        else:
            ds = ds.map(
                _load_decode_resize_with_label_and_aug,
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )
    else:
        if labels is None:
            ds = ds.map(
                _load_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True
            )
        else:
            ds = ds.map(
                _load_decode_resize_with_label,
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(
    df_trn["path"].values, df_trn["label"].values, training=True, cache_tag="train"
)
val_ds = make_ds(
    df_val["path"].values, df_val["label"].values, training=False, cache_tag="val"
)

steps_per_epoch = int(np.ceil(len(df_trn) / BATCH_SIZE))
val_steps = int(np.ceil(len(df_val) / BATCH_SIZE))

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)

base_model = None
for layer in my_model.layers:
    if isinstance(layer, tf.keras.Model) and layer.name.startswith("efficientnet"):
        base_model = layer
        break

if base_model is not None:
    base_model.trainable = True
    for l in base_model.layers[:-20]:
        l.trainable = False

    my_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    ft_epochs = 1 if DEBUG else 2
    my_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=ft_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=1,
    )




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1932079332.py in <cell line: 0>()
    188 
    189 
--> 190 train_ds = make_ds(
    191     df_trn["path"].values, df_trn["label"].values, training=True, cache_tag="train"
    192 )

/tmp/ipykernel_11/1932079332.py in make_ds(paths, labels, training, batch_size, cache_tag)
    165             )
    166         else:
--> 167             ds = ds.map(
    168                 _load_decode_resize_with_label_and_aug,
    169                 num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_filexqv6ucfz.py in tf___load_decode_resize_with_label_and_aug(path, y)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img = ag__.converted_call(ag__.ld(_load_decode_resize), (ag__.ld(path),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(_augment_one), (ag__.ld(img),), None, fscope)
     12                 try:
     13                     do_return = True

/tmp/__autograph_generated_fileio_m5h89.py in tf___augment_one(img)
     12                 w = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(img),), None, fscope)[1], ag__.ld(tf).float32), None, fscope)
     13                 angle = ag__.converted_call(ag__.ld(tf).random.uniform, ([],), dict(minval=-ag__.ld(_ROT), maxval=ag__.ld(_ROT), seed=ag__.ld(SEED), dtype=ag__.ld(tf).float32), fscope)
---> 14                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), fill_mode='reflect'), fscope)
     15                 dx = ag__.converted_call(ag__.ld(tf).random.uniform, ([], -ag__.ld(_W_SHIFT), ag__.ld(_W_SHIFT)), dict(seed=ag__.ld(SEED), dtype=ag__.ld(tf).float32), fscope) * ag__.ld(w)
     16                 dy = ag__.converted_call(ag__.ld(tf).random.uniform, ([], -ag__.ld(_H_SHIFT), ag__.ld(_H_SHIFT)), dict(seed=ag__.ld(SEED), dtype=ag__.ld(tf).float32), fscope) * ag__.ld(h)

AttributeError: in user code:

    File "/tmp/ipykernel_11/1932079332.py", line 127, in _load_decode_resize_with_label_and_aug  *
        img = _augment_one(img)
    File "/tmp/ipykernel_11/1932079332.py", line 60, in _augment_one  *
        img = tf.image.rotate(img, angles=angle, fill_mode="reflect")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 4
test_ds = make_ds(df_test["path"].values, labels=None, training=False, cache_tag="test")

pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_test_labels

final_csv["image_id"] = final_csv["image_id"].astype(str)
final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)




## === cell 5
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(df_sample)
assert sub["label"].between(0, 4).all()
print(sub.head())

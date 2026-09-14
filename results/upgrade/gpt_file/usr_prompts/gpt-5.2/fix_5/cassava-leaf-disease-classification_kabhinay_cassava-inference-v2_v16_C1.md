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

2.7

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

0.8828951344817165

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from __future__ import print_function
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
try:
    tf.random.set_seed(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
try:
    print("Eager:", tf.executing_eagerly())
except Exception:
    print("Eager: unknown")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _find_existing_dir(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def _find_existing_file(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


DATA_DIR = _find_existing_dir(
    [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
)
if DATA_DIR is None:
    raise OSError(
        "Could not locate cassava-leaf-disease-classification input directory."
    )

TRAIN_DIR = _find_existing_dir(
    [
        os.path.join(DATA_DIR, "train_images"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
    ]
)
TEST_DIR = _find_existing_dir(
    [
        os.path.join(DATA_DIR, "test_images"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    ]
)
train_csv_path = _find_existing_file(
    [
        os.path.join(DATA_DIR, "train.csv"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
    ]
)
sample_sub_path = _find_existing_file(
    [
        os.path.join(DATA_DIR, "sample_submission.csv"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    ]
)

if TRAIN_DIR is None or TEST_DIR is None:
    raise OSError(
        "train_images/test_images directories not found in expected locations."
    )
if train_csv_path is None or sample_sub_path is None:
    raise OSError("train.csv/sample_submission.csv not found in expected locations.")

print("DATA_DIR:", DATA_DIR)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("train.csv:", train_csv_path)
print("sample_submission.csv:", sample_sub_path)

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)
test_df = sample_sub[["image_id"]].copy()

assert "image_id" in train_df.columns and "label" in train_df.columns
assert "image_id" in test_df.columns
print("Train rows:", len(train_df), "Test rows:", len(test_df))
print("Train label distribution:\n", train_df["label"].value_counts().sort_index())




## === cell 2
def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros(
            (batch_x.shape[0], crop_length, crop_length, 3), dtype=batch_x.dtype
        )
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 5

from sklearn.model_selection import train_test_split

train_df["label"] = train_df["label"].astype(str)

tr_df, va_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=0,
    stratify=train_df["label"].values,
)

AUTOTUNE = getattr(tf.data, "AUTOTUNE", None)
if AUTOTUNE is None:
    AUTOTUNE = tf.data.experimental.AUTOTUNE

_classes = sorted(tr_df["label"].unique().tolist())
class_indices = dict((c, i) for i, c in enumerate(_classes))
print("class_indices:", class_indices)

tr_image_ids = tr_df["image_id"].values
tr_labels_str = tr_df["label"].values
va_image_ids = va_df["image_id"].values
va_labels_str = va_df["label"].values
te_image_ids = test_df["image_id"].values

tr_paths = np.array([os.path.join(TRAIN_DIR, x) for x in tr_image_ids])
va_paths = np.array([os.path.join(TRAIN_DIR, x) for x in va_image_ids])
te_paths = np.array([os.path.join(TEST_DIR, x) for x in te_image_ids])

tr_labels_int = np.array([class_indices[s] for s in tr_labels_str], dtype=np.int32)
va_labels_int = np.array([class_indices[s] for s in va_labels_str], dtype=np.int32)


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


BASE_SEED = tf.constant([0, 0], dtype=tf.int32)


def _augment(img, seed):
    angle = tf.random.stateless_uniform(
        [], seed=seed, minval=-15.0, maxval=15.0, dtype=tf.float32
    )
    angle = angle * (3.141592653589793 / 180.0)
    img = tf.keras.layers.RandomRotation(factor=0.0, fill_mode="reflect")(
        img, training=True
    )  # placeholder to keep layer imported; we will do explicit rotate below if available

    try:
        img = tf.image.rotate(img, angles=angle, fill_mode="reflect")
    except Exception:
        factor = 15.0 / 360.0
        rr = tf.keras.layers.RandomRotation(factor=factor, fill_mode="reflect", seed=0)
        img = rr(img, training=True)

    tx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 0], tf.int32), minval=-0.10, maxval=0.10
    )
    ty = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([0, 1], tf.int32), minval=-0.10, maxval=0.10
    )
    rt = tf.keras.layers.RandomTranslation(
        height_factor=0.10, width_factor=0.10, fill_mode="reflect", seed=0
    )
    img = rt(tf.expand_dims(img, 0), training=True)[0]

    rz = tf.keras.layers.RandomZoom(
        height_factor=(-0.10, 0.10),
        width_factor=(-0.10, 0.10),
        fill_mode="reflect",
        seed=0,
    )
    img = rz(tf.expand_dims(img, 0), training=True)[0]

    img = tf.image.stateless_random_flip_left_right(
        img, seed=seed + tf.constant([2, 2], tf.int32)
    )
    return img


def _make_train_ds(paths, labels_int):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_int))
    ds = ds.shuffle(buffer_size=len(paths), seed=0, reshuffle_each_iteration=True)

    ds = ds.enumerate()

    def _load_aug(idx, data):
        path, y = data
        img = _decode_resize(path)
        seed = BASE_SEED + tf.cast(tf.stack([idx, idx]), tf.int32)
        img = _augment(img, seed)
        y = tf.one_hot(tf.cast(y, tf.int32), depth=NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_load_aug, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(paths, labels_int):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_int))

    def _load(path, y):
        img = _decode_resize(path)
        y = tf.one_hot(tf.cast(y, tf.int32), depth=NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        img = _decode_resize(path)
        return img

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_dataset = _make_train_ds(tr_paths, tr_labels_int)
val_dataset = _make_val_ds(va_paths, va_labels_int)
test_dataset = _make_test_ds(te_paths)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

steps_per_epoch = int(np.ceil(float(len(tr_df)) / float(BATCH_SIZE)))
val_steps = int(np.ceil(float(len(va_df)) / float(BATCH_SIZE)))

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=val_steps,
    verbose=1,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_12/1531018542.py in <cell line: 0>()
    164 
    165 
--> 166 train_dataset = _make_train_ds(tr_paths, tr_labels_int)
    167 val_dataset = _make_val_ds(va_paths, va_labels_int)
    168 test_dataset = _make_test_ds(te_paths)

/tmp/ipykernel_12/1531018542.py in _make_train_ds(paths, labels_int)
    128         return img, y
    129 
--> 130     ds = ds.map(_load_aug, num_parallel_calls=AUTOTUNE)
    131     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    132     ds = ds.prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_filem33t2t8c.py in tf___load_aug(idx, data)
     12                 img = ag__.converted_call(ag__.ld(_decode_resize), (ag__.ld(path),), None, fscope)
     13                 seed = ag__.ld(BASE_SEED) + ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(idx), ag__.ld(idx)],), None, fscope), ag__.ld(tf).int32), None, fscope)
---> 14                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(seed)), None, fscope)
     15                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(y), ag__.ld(tf).int32), None, fscope),), dict(depth=ag__.ld(NUM_CLASSES), dtype=ag__.ld(tf).float32), fscope)
     16                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_fileo5mm3v6w.py in tf___augment(img, seed)
     11                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed), minval=-15.0, maxval=15.0, dtype=ag__.ld(tf).float32), fscope)
     12                 angle = ag__.ld(angle) * (3.141592653589793 / 180.0)
---> 13                 img = ag__.converted_call(ag__.converted_call(ag__.ld(tf).keras.layers.RandomRotation, (), dict(factor=0.0, fill_mode='reflect'), fscope), (ag__.ld(img),), dict(training=True), fscope)
     14                 try:
     15                     img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), fill_mode='reflect'), fscope)

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

    File "/tmp/ipykernel_12/1531018542.py", line 126, in _load_aug  *
        img = _augment(img, seed)
    File "/tmp/ipykernel_12/1531018542.py", line 72, in _augment  *
        img = tf.keras.layers.RandomRotation(factor=0.0, fill_mode="reflect")(
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


## === cell 4
test_steps = int(np.ceil(float(len(test_df)) / float(BATCH_SIZE)))
pred = model.predict(test_dataset, verbose=1, steps=test_steps)
pred = np.asarray(pred)

if pred.ndim != 2 or pred.shape[1] != NUM_CLASSES:
    raise ValueError("Unexpected prediction shape: {}".format(pred.shape))

predicted_class_indices = np.argmax(pred, axis=1).astype(int)
predicted_class_indices = predicted_class_indices[: len(test_df)]

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices}
)
submission["label"] = submission["label"].astype(int)
submission = submission[["image_id", "label"]]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Cols:", submission.columns.tolist())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2756306655.py in <cell line: 0>()
      1 # PERF: test inference via tf.data (prefetch/cache) to remove Python generator overhead.
      2 test_steps = int(np.ceil(float(len(test_df)) / float(BATCH_SIZE)))
----> 3 pred = model.predict(test_dataset, verbose=1, steps=test_steps)
      4 pred = np.asarray(pred)
      5 

NameError: name 'model' is not defined

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

0.6406769416742218

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

import pandas as pd
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train_images at {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test_images at {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"

print("Using BASE_INPUT:", BASE_INPUT)
print("TensorFlow:", tf.__version__)

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

train_df["path"] = (TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)).astype(str)

train_df["label"] = train_df["label"].astype(str)

tr_df, va_df = train_test_split(
    train_df, test_size=0.10, random_state=SEED, stratify=train_df["label"]
)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32 if not DEBUG else 8

AUTOTUNE = tf.data.AUTOTUNE
IMG_H, IMG_W = IMG_SIZE
NUM_CLASSES = train_df["label"].nunique()
print("NUM_CLASSES:", NUM_CLASSES)

class_names = sorted(train_df["label"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}
print("class_indices:", class_to_idx)

tr_paths = tr_df["path"].to_numpy()
va_paths = va_df["path"].to_numpy()
tr_labels = tr_df["label"].map(class_to_idx).to_numpy(dtype=np.int32)
va_labels = va_df["label"].map(class_to_idx).to_numpy(dtype=np.int32)

tr_steps = int(np.ceil(len(tr_paths) / BATCH_SIZE))
va_steps = int(np.ceil(len(va_paths) / BATCH_SIZE))


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 0], tf.int32), minval=-10.0, maxval=10.0
    ) * (np.pi / 180.0)
    try:
        img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")
    except Exception:
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img, 0),
            transforms=tf.expand_dims(
                tf.stack(
                    [
                        tf.cos(angle),
                        -tf.sin(angle),
                        0.0,
                        tf.sin(angle),
                        tf.cos(angle),
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                0,
            ),
            output_shape=[IMG_H, IMG_W],
            interpolation="BILINEAR",
            fill_mode="REFLECT",
        )[0]

    max_dx = 0.05 * float(IMG_W)
    max_dy = 0.05 * float(IMG_H)
    dx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 0], tf.int32), minval=-max_dx, maxval=max_dx
    )
    dy = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([3, 0], tf.int32), minval=-max_dy, maxval=max_dy
    )

    scale = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 0], tf.int32), minval=0.9, maxval=1.1
    )

    cx = (IMG_W - 1) / 2.0
    cy = (IMG_H - 1) / 2.0
    a0 = 1.0 / scale
    a1 = 0.0
    a2 = (dx - cx * (scale - 1.0)) / scale
    b0 = 0.0
    b1 = 1.0 / scale
    b2 = (dy - cy * (scale - 1.0)) / scale

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0]), 0),
        output_shape=[IMG_H, IMG_W],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )[0]
    return img


def _make_train_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _decode_and_resize(path)
        s = tf.random.experimental.stateless_fold_in(
            tf.constant([SEED, 0], tf.int32), tf.cast(label, tf.int32)
        )
        img = _augment(img, s)
        y = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_valid_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, label):
        img = _decode_and_resize(path)
        y = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds(tr_paths, tr_labels, BATCH_SIZE)
valid_ds = _make_valid_ds(va_paths, va_labels, BATCH_SIZE)

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
backbone = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs, outputs)

backbone.trainable = False
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_1 = 1 if DEBUG else 2
my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_1,
    steps_per_epoch=tr_steps,
    validation_steps=va_steps,
    verbose=1,
)

backbone.trainable = True
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_2 = 1 if DEBUG else 1
my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_1 + EPOCHS_2,
    initial_epoch=EPOCHS_1,
    steps_per_epoch=tr_steps,
    validation_steps=va_steps,
    verbose=1,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1867305426.py in <cell line: 0>()
    156 
    157 
--> 158 train_ds = _make_train_ds(tr_paths, tr_labels, BATCH_SIZE)
    159 valid_ds = _make_valid_ds(va_paths, va_labels, BATCH_SIZE)
    160 

/tmp/ipykernel_11/1867305426.py in _make_train_ds(paths, labels, batch_size)
    136         return img, y
    137 
--> 138     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    139     ds = ds.batch(batch_size, drop_remainder=False)
    140     ds = ds.prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_filef3q_ojjt.py in tf___map_fn(path, label)
     10                 img = ag__.converted_call(ag__.ld(_decode_and_resize), (ag__.ld(path),), None, fscope)
     11                 s = ag__.converted_call(ag__.ld(tf).random.experimental.stateless_fold_in, (ag__.converted_call(ag__.ld(tf).constant, ([ag__.ld(SEED), 0], ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(label), ag__.ld(tf).int32), None, fscope)), None, fscope)
---> 12                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(s)), None, fscope)
     13                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(label), ag__.ld(NUM_CLASSES)), dict(dtype=ag__.ld(tf).float32), fscope)
     14                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filer7z843nv.py in tf___augment(img, seed)
     13                     img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), interpolation='BILINEAR'), fscope)
     14                 except Exception:
---> 15                     img = ag__.converted_call(ag__.ld(tf).raw_ops.ImageProjectiveTransformV3, (), dict(images=ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(img), 0), None, fscope), transforms=ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cos, (ag__.ld(angle),), None, fscope), -ag__.converted_call(ag__.ld(tf).sin, (ag__.ld(angle),), None, fscope), 0.0, ag__.converted_call(ag__.ld(tf).sin, (ag__.ld(angle),), None, fscope), ag__.converted_call(ag__.ld(tf).cos, (ag__.ld(angle),), None, fscope), 0.0, 0.0, 0.0],), None, fscope), 0), None, fscope), output_shape=[ag__.ld(IMG_H), ag__.ld(IMG_W)], interpolation='BILINEAR', fill_mode='REFLECT'), fscope)[0]
     16                 max_dx = 0.05 * ag__.converted_call(ag__.ld(float), (ag__.ld(IMG_W),), None, fscope)
     17                 max_dy = 0.05 * ag__.converted_call(ag__.ld(float), (ag__.ld(IMG_H),), None, fscope)

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/tf_export.py in wrapper(*args, **kwargs)
    375           )
    376       )
--> 377     return f(**kwargs)
    378 
    379   return tf_decorator.make_decorator(f, wrapper, decorator_argspec=f_argspec)

TypeError: in user code:

    File "/tmp/ipykernel_11/1867305426.py", line 134, in _map_fn  *
        img = _augment(img, s)
    File "/tmp/ipykernel_11/1867305426.py", line 62, in _augment  *
        img = tf.raw_ops.ImageProjectiveTransformV3(

    TypeError: image_projective_transform_v3() missing 1 required positional argument: 'fill_value'


## === cell 2
test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))

assert len(test_images) > 0, f"No test images found in {TEST_IMG_DIR}"


def make_test_ds(batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices(test_images)

    def _map_fn(path):
        img = _decode_and_resize(path)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 3
test_ds = make_test_ds(batch_size=128 if not DEBUG else 16)
test_steps = int(np.ceil(len(test_images) / (128 if not DEBUG else 16)))

pred_test = my_model.predict(test_ds, steps=test_steps, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

image_ids = np.fromiter(
    (os.path.basename(p) for p in test_images), dtype=object, count=len(test_images)
)
final_submission = pd.DataFrame({"image_id": image_ids, "label": pred_test_labels})

sample_sub = pd.read_csv(SAMPLE_SUB)
final_csv = sample_sub[["image_id"]].merge(
    final_submission,
    on="image_id",
    how="left",
    validate="one_to_one",
)

if final_csv["label"].isna().any():
    final_csv["label"] = final_csv["label"].fillna(0).astype(int)

final_csv["label"] = final_csv["label"].astype(int)
final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2266885385.py in <cell line: 0>()
      3 
      4 # Speed: predict on tf.data pipeline with prefetch; identical model inference semantics.
----> 5 pred_test = my_model.predict(test_ds, steps=test_steps, verbose=1)
      6 pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)
      7 

NameError: name 'my_model' is not defined

## === cell 4
final_csv.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1842027079.py in <cell line: 0>()
----> 1 final_csv.head()

NameError: name 'final_csv' is not defined

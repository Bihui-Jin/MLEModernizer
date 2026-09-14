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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8856149894227864

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import sys
import random
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf
import matplotlib.pyplot as plt

from functools import partial
from sklearn.model_selection import train_test_split

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.optimizer.set_jit(True)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

print("TF version:", tf.__version__)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_TFRECS_GLOB = os.path.join(DATA_DIR, "train_tfrecords", "ld_train*.tfrec")
TEST_TFRECS_GLOB = os.path.join(DATA_DIR, "test_tfrecords", "ld_test*.tfrec")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMAGE_SIZE = [512, 512]
NUM_CLASSES = 5

BATCH_SIZE = 16  # keep as provided
EPOCHS = 5  # unchanged

base = tf.keras.applications.EfficientNetB5(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3),
)
inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model_28_2 = tf.keras.Model(inputs, outputs)

model_28_2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

model_28_2.summary()



## === cell 2
test_df = pd.read_csv(SAMPLE_SUB)
print(test_df.head())
print("Sample submission rows:", len(test_df))

AUTOTUNE = tf.data.AUTOTUNE
GCS_PATH = DATA_DIR

_size_re = re.compile(r"-([0-9]*)\.")


def dataset_sizes(filenames):
    n = [int(_size_re.search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


TRAIN_FILENAMES = tf.io.gfile.glob(TRAIN_TFRECS_GLOB)
TEST_FILENAMES = tf.io.gfile.glob(TEST_TFRECS_GLOB)

NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

print("Train TFRecords:", len(TRAIN_FILENAMES), "images:", NUM_TRAIN_IMAGES)
print("Test  TFRecords:", len(TEST_FILENAMES), "images:", NUM_TEST_IMAGES)



## === cell 3
TFREC_FORMAT_LABELED = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
TFREC_FORMAT_UNLABELED = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_and_resize(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return img


@tf.function
def read_tfrecord_labeled(example):
    ex = tf.io.parse_single_example(example, TFREC_FORMAT_LABELED)
    img = _decode_and_resize(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


@tf.function
def read_tfrecord_unlabeled(example):
    ex = tf.io.parse_single_example(example, TFREC_FORMAT_UNLABELED)
    img = _decode_and_resize(ex["image"])
    image_name = ex["image_name"]
    return img, image_name


def _dataset_options(ordered: bool):
    opts = tf.data.Options()
    opts.experimental_optimization.apply_default_optimizations = True
    try:
        opts.experimental_optimization.map_fusion = True
    except Exception:
        pass
    try:
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    opts.experimental_deterministic = bool(ordered)
    try:
        opts.experimental_slack = True
    except Exception:
        pass
    return opts


def load_raw_dataset(filenames, ordered=False):
    ds = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=AUTOTUNE,
        buffer_size=32 * 1024 * 1024,
    )
    ds = ds.with_options(_dataset_options(ordered))
    return ds


def load_dataset(filenames, labeled=True, ordered=False):
    ds = load_raw_dataset(filenames, ordered=ordered)
    if labeled:
        ds = ds.map(
            read_tfrecord_labeled, num_parallel_calls=AUTOTUNE, deterministic=ordered
        )
    else:
        ds = ds.map(
            read_tfrecord_unlabeled, num_parallel_calls=AUTOTUNE, deterministic=ordered
        )
    return ds


def _maybe_copy_to_device(ds):
    gpus = tf.config.list_logical_devices("GPU")
    if gpus:
        try:
            ds = ds.apply(tf.data.experimental.copy_to_device("/GPU:0"))
            ds = ds.prefetch(AUTOTUNE)
        except Exception:
            pass
    return ds


def get_test_data(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_copy_to_device(ds)
    return ds




## === cell 4
VAL_FRAC = 0.15

train_steps = int(np.ceil((NUM_TRAIN_IMAGES * (1.0 - VAL_FRAC)) / BATCH_SIZE))
val_steps = int(np.ceil((NUM_TRAIN_IMAGES * VAL_FRAC) / BATCH_SIZE))

SPLIT_MOD = 20
VAL_SHARDS = int(np.round(SPLIT_MOD * VAL_FRAC))
VAL_SHARDS = max(
    1, min(SPLIT_MOD - 1, VAL_SHARDS)
)  # safety; keeps intent for 0<VAL_FRAC<1

raw_all = load_raw_dataset(TRAIN_FILENAMES, ordered=False)


def _select_shards(ds, indices, mod):
    indices = list(indices)
    ids_ds = tf.data.Dataset.from_tensor_slices(tf.constant(indices, dtype=tf.int64))
    return ids_ds.interleave(
        lambda i: ds.shard(mod, tf.cast(i, tf.int32)),
        cycle_length=len(indices),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,  # preserve deterministic shard concatenation order
    )


val_raw = _select_shards(raw_all, range(0, VAL_SHARDS), SPLIT_MOD)
train_raw = _select_shards(raw_all, range(VAL_SHARDS, SPLIT_MOD), SPLIT_MOD)

train_ds = train_raw.map(
    read_tfrecord_labeled, num_parallel_calls=AUTOTUNE, deterministic=False
)

train_ds = (
    train_ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
train_ds = _maybe_copy_to_device(train_ds)

val_ds = val_raw.map(
    read_tfrecord_labeled, num_parallel_calls=AUTOTUNE, deterministic=True
)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
val_ds = _maybe_copy_to_device(val_ds)

print("Fitting model...")
history = model_28_2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1,
)

print("Computing predictions...")

test_ds = get_test_data(ordered=True)

probs = model_28_2.predict(test_ds.map(lambda x, y: x), verbose=0)
predictions = np.argmax(probs, axis=-1).astype(np.int32)

test_ids_bytes = np.concatenate([ids.numpy() for _, ids in test_ds], axis=0)
test_ids = np.char.decode(test_ids_bytes.astype("S"), "utf-8").astype(object)

print("Predictions shape:", predictions.shape, "unique:", np.unique(predictions))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1264234407.py in <cell line: 0>()
     27 
     28 
---> 29 val_raw = _select_shards(raw_all, range(0, VAL_SHARDS), SPLIT_MOD)
     30 train_raw = _select_shards(raw_all, range(VAL_SHARDS, SPLIT_MOD), SPLIT_MOD)
     31 

/tmp/ipykernel_55/1264234407.py in _select_shards(ds, indices, mod)
     19     indices = list(indices)
     20     ids_ds = tf.data.Dataset.from_tensor_slices(tf.constant(indices, dtype=tf.int64))
---> 21     return ids_ds.interleave(
     22         lambda i: ds.shard(mod, tf.cast(i, tf.int32)),
     23         cycle_length=len(indices),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in interleave(self, map_func, cycle_length, block_length, num_parallel_calls, deterministic, name)
   2532     # pylint: disable=g-import-not-at-top,protected-access
   2533     from tensorflow.python.data.ops import interleave_op
-> 2534     return interleave_op._interleave(self, map_func, cycle_length, block_length,
   2535                                      num_parallel_calls, deterministic, name)
   2536     # pylint: enable=g-import-not-at-top,protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/interleave_op.py in _interleave(input_dataset, map_func, cycle_length, block_length, num_parallel_calls, deterministic, name)
     47         input_dataset, map_func, cycle_length, block_length, name=name)
     48   else:
---> 49     return _ParallelInterleaveDataset(
     50         input_dataset,
     51         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/interleave_op.py in __init__(self, input_dataset, map_func, cycle_length, block_length, num_parallel_calls, buffer_output_elements, prefetch_input_elements, deterministic, name)
    117     """See `Dataset.interleave()` for details."""
    118     self._input_dataset = input_dataset
--> 119     self._map_func = structured_function.StructuredFunctionWrapper(
    120         map_func, self._transformation_name(), dataset=input_dataset)
    121     if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):

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

/tmp/__autograph_generated_fileu3923owl.py in <lambda>(i)
      5 
      6     def inner_factory(ag__):
----> 7         tf__lam = lambda i: ag__.with_function_scope(lambda lscope: ag__.converted_call(ds.shard, (mod, ag__.converted_call(tf.cast, (i, tf.int32), None, lscope)), None, lscope), 'lscope', ag__.STD)
      8         return tf__lam
      9     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_fileu3923owl.py in <lambda>(lscope)
      5 
      6     def inner_factory(ag__):
----> 7         tf__lam = lambda i: ag__.with_function_scope(lambda lscope: ag__.converted_call(ds.shard, (mod, ag__.converted_call(tf.cast, (i, tf.int32), None, lscope)), None, lscope), 'lscope', ag__.STD)
      8         return tf__lam
      9     return inner_factory

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in shard(self, num_shards, index, name)
   1688     # pylint: disable=g-import-not-at-top,protected-access
   1689     from tensorflow.python.data.ops import shard_op
-> 1690     return shard_op._shard(self, num_shards, index, name=name)
   1691     # pylint: enable=g-import-not-at-top,protected-access
   1692 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/shard_op.py in _shard(input_dataset, num_shards, index, name)
     23 def _shard(input_dataset, num_shards, index, name):  # pylint: disable=unused-private-name
     24   """See `Dataset.shard()` for details."""
---> 25   return _ShardDataset(input_dataset, num_shards, index, name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/shard_op.py in __init__(self, input_dataset, num_shards, index, name)
     34     self._num_shards = ops.convert_to_tensor(
     35         num_shards, dtype=dtypes.int64, name="num_shards")
---> 36     self._index = ops.convert_to_tensor(index, dtype=dtypes.int64, name="index")
     37     self._name = name
     38     variant_tensor = gen_dataset_ops.shard_dataset(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    207   overload = getattr(value, "__tf_tensor__", None)
    208   if overload is not None:
--> 209     return overload(dtype, name)  #  pylint: disable=not-callable
    210 
    211   for base_type, conversion_func in get(type(value)):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __tf_tensor__(self, dtype, name)
    759       ) -> "Tensor":
    760     if dtype is not None and not dtype.is_compatible_with(self.dtype):
--> 761       raise ValueError(
    762           _add_error_prefix(
    763               f"Tensor conversion requested dtype {dtype.name} "

ValueError: in user code:

    File "/tmp/ipykernel_55/1264234407.py", line 22, in None  *
        lambda i: ds.shard(mod, tf.cast(i, tf.int32))

    ValueError: index: Tensor conversion requested dtype int64 for Tensor with dtype int32: <tf.Tensor 'Cast:0' shape=() dtype=int32>


## === cell 5
print("Generating submission.csv file...")

if len(test_ids) != len(predictions):
    raise RuntimeError(
        f"Mismatch: got {len(test_ids)} test_ids but {len(predictions)} predictions"
    )

sub = pd.DataFrame({"image_id": test_ids, "label": predictions})

sub = test_df[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    fill_val = int(pd.Series(predictions).mode().iloc[0])
    sub["label"] = sub["label"].fillna(fill_val).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
print("Submission columns:", list(sub.columns))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2011101404.py in <cell line: 0>()
      1 print("Generating submission.csv file...")
      2 
----> 3 if len(test_ids) != len(predictions):
      4     raise RuntimeError(
      5         f"Mismatch: got {len(test_ids)} test_ids but {len(predictions)} predictions"

NameError: name 'test_ids' is not defined

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

0.6130250831066788

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

ROOT_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(ROOT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(ROOT_DIR, "test_images")

TRAIN_TFREC_DIR = os.path.join(ROOT_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(ROOT_DIR, "test_tfrecords")

print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("train.csv exists:", os.path.exists(TRAIN_CSV))
print("sample_submission.csv exists:", os.path.exists(SAMPLE_SUB))
print("train_images exists:", os.path.exists(TRAIN_IMG_DIR))
print("test_images exists:", os.path.exists(TEST_IMG_DIR))
print("train_tfrecords exists:", os.path.exists(TRAIN_TFREC_DIR))
print("test_tfrecords exists:", os.path.exists(TEST_TFREC_DIR))



## === cell 1
import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Could not enable op determinism:", repr(e))

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception as e:
    print("Could not set memory growth:", repr(e))

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))

tf.config.run_functions_eagerly(False)

print("TF version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

assert set(train_df.columns) == {"image_id", "label"}
assert set(sample_df.columns) == {"image_id", "label"}

num_classes = train_df["label"].nunique()
print("Train rows:", len(train_df), "Classes:", num_classes)
print("Test rows:", len(sample_df))



## === cell 3
from sklearn.model_selection import train_test_split

train_df2, val_df2 = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label"]
)
train_df2 = train_df2.reset_index(drop=True)
val_df2 = val_df2.reset_index(drop=True)

print("Train split:", len(train_df2), "Val split:", len(val_df2))



## === cell 4
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

AUG = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name="aug",
)

_FEATURE_DESC_U8 = {
    "image": tf.io.FixedLenFeature([IMG_SIZE * IMG_SIZE * 3], tf.uint8),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}
_FEATURE_DESC_JPEG = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


@tf.function
def _decode_and_resize_img_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _u8_flat_to_float_img(u8_flat):
    img = tf.reshape(u8_flat, [IMG_SIZE, IMG_SIZE, 3])
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def decode_and_resize(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = _decode_and_resize_img_bytes(img_bytes)
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


@tf.function
def _parse_tfrecord_train(example):
    ex = tf.io.parse_single_example(example, _FEATURE_DESC_U8)
    img = _u8_flat_to_float_img(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return img, y


@tf.function
def _parse_tfrecord_test(example):
    ex = tf.io.parse_single_example(example, _FEATURE_DESC_U8)
    img = _u8_flat_to_float_img(ex["image"])
    return img, ex["image_name"]


def _dataset_options(training: bool):
    opt = tf.data.Options()
    opt.experimental_deterministic = False if training else True
    opt.experimental_optimization.apply_default_optimizations = True
    opt.experimental_optimization.map_and_batch_fusion = True
    opt.experimental_optimization.parallel_batch = True
    opt.threading.private_threadpool_size = 0
    opt.threading.max_intra_op_parallelism = 0
    return opt


def _dataset_options_deterministic():
    opt = tf.data.Options()
    opt.experimental_deterministic = True
    opt.experimental_optimization.apply_default_optimizations = True
    opt.experimental_optimization.map_and_batch_fusion = True
    opt.experimental_optimization.parallel_batch = True
    opt.threading.private_threadpool_size = 0
    opt.threading.max_intra_op_parallelism = 0
    return opt


def _list_tfrecords(tfrecord_dir):
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No TFRecords found in: {tfrecord_dir}")
    return files


def make_ds_from_images(df, image_dir, training=False, cache_key=""):
    paths = tf.constant([os.path.join(image_dir, x) for x in df["image_id"].values])
    labels = tf.constant(df["label"].values, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_dataset_options(training))

    if training:
        ds = ds.shuffle(min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: decode_and_resize(p, y),
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )

    if not training:
        if cache_key:
            ds = ds.cache(os.path.join(CACHE_DIR, f"{cache_key}.cache"))
        else:
            ds = ds.cache()

    if training:
        ds = ds.map(
            lambda x, y: (AUG(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def _split_tfrecord_files(files, val_frac=0.1, seed=SEED):
    files = list(files)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(files) * val_frac)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    return sorted(train_files), sorted(val_files)


def make_ds_from_tfrecords(tfrecord_files, training=False, cache_key=""):
    ds = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=AUTOTUNE,
        compression_type=None,
    )
    ds = ds.with_options(_dataset_options(training))

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        _parse_tfrecord_train,
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )

    if cache_key:
        ds = ds.cache(os.path.join(CACHE_DIR, f"{cache_key}.cache"))

    if training:
        ds = ds.map(
            lambda x, y: (AUG(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


if os.path.exists(TRAIN_TFREC_DIR):
    train_tfrec_files_all = _list_tfrecords(TRAIN_TFREC_DIR)
    train_tfrec_files, val_tfrec_files = _split_tfrecord_files(
        train_tfrec_files_all, val_frac=0.1, seed=SEED
    )

    train_ds = make_ds_from_tfrecords(
        train_tfrec_files, training=True, cache_key="train_tfrec_decoded_224"
    )
    val_ds = make_ds_from_tfrecords(
        val_tfrec_files, training=False, cache_key="val_tfrec_decoded_224"
    )
else:
    train_ds = make_ds_from_images(
        train_df2, TRAIN_IMG_DIR, training=True, cache_key=""
    )
    val_ds = make_ds_from_images(
        val_df2, TRAIN_IMG_DIR, training=False, cache_key="val_images_decoded_224"
    )

train_steps = int(np.ceil(len(train_df2) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_df2) / BATCH_SIZE))
train_ds_fit = train_ds.repeat()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1028264022.py in <cell line: 0>()
    189     )
    190 
--> 191     train_ds = make_ds_from_tfrecords(
    192         train_tfrec_files, training=True, cache_key="train_tfrec_decoded_224"
    193     )

/tmp/ipykernel_11/1028264022.py in make_ds_from_tfrecords(tfrecord_files, training, cache_key)
    160         ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    161 
--> 162     ds = ds.map(
    163         _parse_tfrecord_train,
    164         num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_file353c495m.py in tf___parse_tfrecord_train(example)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 ex = ag__.converted_call(ag__.ld(tf).io.parse_single_example, (ag__.ld(example), ag__.ld(_FEATURE_DESC_U8)), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(_u8_flat_to_float_img), (ag__.ld(ex)['image'],), None, fscope)
     12                 y = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(ex)['target'], ag__.ld(tf).int32), None, fscope)

TypeError: in user code:

    File "/tmp/ipykernel_11/1028264022.py", line 63, in _parse_tfrecord_train  *
        ex = tf.io.parse_single_example(example, _FEATURE_DESC_U8)

    TypeError: Value passed to parameter 'dense_defaults' has DataType uint8 not in list of allowed values: float32, int64, string


## === cell 5
model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 6
EPOCHS = 4

history = model.fit(
    train_ds_fit,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1152966239.py in <cell line: 0>()
      2 
      3 history = model.fit(
----> 4     train_ds_fit,
      5     validation_data=val_ds,
      6     epochs=EPOCHS,

NameError: name 'train_ds_fit' is not defined

## === cell 7
if os.path.exists(TEST_TFREC_DIR):
    test_tfrec_files = _list_tfrecords(TEST_TFREC_DIR)

    test_ds = tf.data.TFRecordDataset(
        test_tfrec_files, num_parallel_reads=AUTOTUNE, compression_type=None
    )
    test_ds = test_ds.with_options(_dataset_options_deterministic())
    test_ds = test_ds.map(
        _parse_tfrecord_test, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    names_bytes = np.concatenate([n.numpy() for _, n in test_ds], axis=0)
    names = np.char.decode(names_bytes.astype("S"), "utf-8")

    test_img_ds = test_ds.map(lambda x, n: x, num_parallel_calls=AUTOTUNE).prefetch(
        AUTOTUNE
    )
    probs = model.predict(test_img_ds, verbose=0)
    preds = probs.argmax(axis=1).astype(np.int32)

    assert len(preds) == len(names)

    pred_df = pd.DataFrame({"image_id": names, "label": preds.astype(int)})
    submission = sample_df[["image_id"]].merge(pred_df, on="image_id", how="left")
    assert submission["label"].isna().sum() == 0
    submission["label"] = submission["label"].astype(int)

    print("Preds length:", len(submission), "Expected:", len(sample_df))
    assert len(submission) == len(sample_df)

    submission.to_csv("submission.csv", index=False)
    print(submission.head())
    print("Wrote submission.csv with rows:", len(submission))
else:
    test_paths = [os.path.join(TEST_IMG_DIR, x) for x in sample_df["image_id"].values]
    test_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_paths))
    test_ds = test_ds.map(
        lambda p: decode_and_resize(p, label=None),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    test_ds = test_ds.with_options(_dataset_options_deterministic())
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    probs = model.predict(test_ds, verbose=0)
    preds = probs.argmax(axis=1).astype(int)

    print("Preds length:", len(preds), "Expected:", len(sample_df))
    assert len(preds) == len(sample_df)

    submission = pd.DataFrame(
        {"image_id": sample_df["image_id"].values, "label": preds}
    )
    submission.to_csv("submission.csv", index=False)

    print(submission.head())
    print("Wrote submission.csv with rows:", len(submission))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/878472034.py in <cell line: 0>()
      8     )
      9     test_ds = test_ds.with_options(_dataset_options_deterministic())
---> 10     test_ds = test_ds.map(
     11         _parse_tfrecord_test, num_parallel_calls=AUTOTUNE, deterministic=True
     12     )

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

/tmp/__autograph_generated_filey3h8719s.py in tf___parse_tfrecord_test(example)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 ex = ag__.converted_call(ag__.ld(tf).io.parse_single_example, (ag__.ld(example), ag__.ld(_FEATURE_DESC_U8)), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(_u8_flat_to_float_img), (ag__.ld(ex)['image'],), None, fscope)
     12                 try:

TypeError: in user code:

    File "/tmp/ipykernel_11/1028264022.py", line 71, in _parse_tfrecord_test  *
        ex = tf.io.parse_single_example(example, _FEATURE_DESC_U8)

    TypeError: Value passed to parameter 'dense_defaults' has DataType uint8 not in list of allowed values: float32, int64, string

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

0.7334542157751587

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12182) has done: 'I fix the import/runtime crash by removing the incompatible standalone `keras` import and consistently using `tf.keras`, which avoids the protobuf `MessageFactory.GetPrototype` error and restores `ImageDataGenerator`. Then I fix the missing external model file issue by replacing the unavailable `../input/resnet-model/model.json/.h5` load with a small, standard `tf.keras.applications.ResNet50` inference model (no training), so the notebook runs end-to-end. Finally, I ensure the submission is aligned exactly to `sample_submission.csv` ordering and writes a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.13042) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory` missing `GetPrototype`) by removing the unused `model_from_json` import (it triggers the incompatible protobuf path in this environment) and keeping everything on `tf.keras`. Then I fix a logic issue that makes the score extremely low: the current model uses an ImageNet head with 5 random-output classes, so predictions are essentially meaningless; instead, I keep the same ResNet50 backbone but train only the final Dense(5) head on `train.csv` images (backbone frozen) and then predict test labels. Finally, I ensure the submission is aligned exactly to `sample_submission.csv` order and always writes a valid `submission.csv` with `image_id,label`.'

# 9. Code solution

## === cell 0
import os, glob

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

INPUT_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_CSV))
print(
    "Train images dir exists:", os.path.isdir(os.path.join(INPUT_DIR, "train_images"))
)
print("Test images dir exists:", os.path.isdir(os.path.join(INPUT_DIR, "test_images")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(int)

label_int = df_train["label"].to_numpy()

rng = np.random.RandomState(SEED)
idx = np.arange(len(df_train))
rng.shuffle(idx)

val_size = int(round(0.1 * len(idx)))
val_idx = idx[:val_size]
train_idx = idx[val_size:]

train_image_ids = df_train["image_id"].iloc[train_idx].to_numpy()
train_labels = label_int[train_idx]
val_image_ids = df_train["image_id"].iloc[val_idx].to_numpy()
val_labels = label_int[val_idx]

print("Split sizes:", len(train_image_ids), len(val_image_ids))

TRAIN_TFREC_GLOB = os.path.join(INPUT_DIR, "train_tfrecords", "*.tfrec")
TEST_TFREC_GLOB = os.path.join(INPUT_DIR, "test_tfrecords", "*.tfrec")
train_tfrecord_files = sorted(glob.glob(TRAIN_TFREC_GLOB))
test_tfrecord_files = sorted(glob.glob(TEST_TFREC_GLOB))

print("Num train tfrecords:", len(train_tfrecord_files))
print("Num test tfrecords:", len(test_tfrecord_files))
assert len(train_tfrecord_files) > 0, "No train TFRecords found."
assert len(test_tfrecord_files) > 0, "No test TFRecords found."

AUTOTUNE = tf.data.AUTOTUNE

opts = tf.data.Options()
opts.experimental_deterministic = True
try:
    opts.threading.private_threadpool_size = 0
    opts.threading.max_intra_op_parallelism = 0
except Exception:
    pass

train_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
test_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

is_train_mask = np.zeros(len(df_train), dtype=np.bool_)
is_train_mask[train_idx] = True
is_val_mask = np.zeros(len(df_train), dtype=np.bool_)
is_val_mask[val_idx] = True

labels_all = label_int

train_tables = []
val_tables = []
for c in range(NUM_CLASSES):
    m = labels_all == c
    train_tables.append(is_train_mask[m].astype(np.int8))
    val_tables.append(is_val_mask[m].astype(np.int8))

train_tables_tf = [tf.constant(t, dtype=tf.int8) for t in train_tables]
val_tables_tf = [tf.constant(t, dtype=tf.int8) for t in val_tables]


@tf.function
def _decode_resize_preprocess_from_parts(image_bytes, label):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return img, label_oh


@tf.function
def _parse_and_decode_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, train_feature_description)
    img, y = _decode_resize_preprocess_from_parts(ex["image"], ex["target"])
    return ex["target"], img, y


def _make_train_val_datasets_from_tfrecords(files):
    ds_raw = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).with_options(
        opts
    )
    ds_raw = ds_raw.apply(tf.data.experimental.ignore_errors())

    counters = tf.Variable(tf.zeros([NUM_CLASSES], dtype=tf.int32), trainable=False)

    @tf.function
    def _assign_split_and_strip(target, img, y):
        c = tf.cast(target, tf.int32)
        i = counters[c].read_value()
        counters[c].assign_add(1)

        in_train = tf.gather(train_tables_tf[c], i) > 0
        in_val = tf.gather(val_tables_tf[c], i) > 0
        tag = tf.where(in_train, 0, tf.where(in_val, 1, -1))
        return img, y, tag

    ds = ds_raw.map(_parse_and_decode_train, num_parallel_calls=AUTOTUNE)
    ds = ds.map(
        _assign_split_and_strip, num_parallel_calls=1
    )  # keep 1 for counter order
    ds = ds.filter(lambda img, y, tag: tag >= 0)

    train_ds_local = ds.filter(lambda img, y, tag: tag == 0).map(
        lambda img, y, tag: (img, y), num_parallel_calls=AUTOTUNE
    )
    val_ds_local = ds.filter(lambda img, y, tag: tag == 1).map(
        lambda img, y, tag: (img, y), num_parallel_calls=AUTOTUNE
    )

    SHUFFLE_BUFFER = min(len(train_image_ids), 8192)
    train_ds_local = train_ds_local.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )

    train_ds_local = train_ds_local.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        AUTOTUNE
    )
    val_ds_local = (
        val_ds_local.cache().batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    )
    return train_ds_local, val_ds_local


train_ds, val_ds = _make_train_val_datasets_from_tfrecords(train_tfrecord_files)

class_indices = {str(i): i for i in range(NUM_CLASSES)}
print("class_indices:", class_indices)
assert len(class_indices) == NUM_CLASSES, "Expected 5 classes from labels 0-4."



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/248175816.py in <cell line: 0>()
    137 
    138 
--> 139 train_ds, val_ds = _make_train_val_datasets_from_tfrecords(train_tfrecord_files)
    140 
    141 class_indices = {str(i): i for i in range(NUM_CLASSES)}

/tmp/ipykernel_11/248175816.py in _make_train_val_datasets_from_tfrecords(files)
    111 
    112     ds = ds_raw.map(_parse_and_decode_train, num_parallel_calls=AUTOTUNE)
--> 113     ds = ds.map(
    114         _assign_split_and_strip, num_parallel_calls=1
    115     )  # keep 1 for counter order

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

/tmp/__autograph_generated_filepautea4e.py in tf___assign_split_and_strip(target, img, y)
     10                 retval_ = ag__.UndefinedReturnValue()
     11                 c = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(target), ag__.ld(tf).int32), None, fscope)
---> 12                 i = ag__.converted_call(ag__.ld(counters)[ag__.ld(c)].read_value, (), None, fscope)
     13                 ag__.converted_call(ag__.ld(counters)[ag__.ld(c)].assign_add, (1,), None, fscope)
     14                 in_train = ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(train_tables_tf)[ag__.ld(c)], ag__.ld(i)), None, fscope) > 0

AttributeError: in user code:

    File "/tmp/ipykernel_11/248175816.py", line 104, in _assign_split_and_strip  *
        i = counters[c].read_value()

    AttributeError: 'SymbolicTensor' object has no attribute 'read_value'


## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

test_raw_bytes = (
    tf.data.TFRecordDataset(test_tfrecord_files, num_parallel_reads=AUTOTUNE)
    .with_options(opts)
    .apply(tf.data.experimental.ignore_errors())
)


@tf.function
def _parse_decode_test_with_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, test_feature_description)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, ex["image_name"]


test_ds_with_name = test_raw_bytes.map(
    _parse_decode_test_with_name, num_parallel_calls=AUTOTUNE
)

test_image_ids = []
for name_batch in test_ds_with_name.map(
    lambda img, name: name, num_parallel_calls=AUTOTUNE
).batch(4096):
    test_image_ids.extend(name_batch.numpy().astype(str).tolist())
test_image_ids = np.array(test_image_ids, dtype=object)

test_imgs_ds = test_ds_with_name.map(lambda img, name: img, num_parallel_calls=AUTOTUNE)
test_imgs_ds = test_imgs_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

print("Test TFRecord examples:", len(test_image_ids))
print("Sample submission rows:", len(sample_sub))
assert len(test_image_ids) == len(
    sample_sub
), "Test TFRecords count must match sample_submission."



## === cell 3
from tensorflow.keras import layers, models

inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")

base = tf.keras.applications.ResNet50(
    include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
)
base.trainable = False

out = layers.Dense(NUM_CLASSES, activation="softmax", name="cassava_head")(base.output)
model = models.Model(inputs=inp, outputs=out)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()

EPOCHS = 3  # keep as-is to preserve runtime/approach
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

pred_test = model.predict(
    test_imgs_ds,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

order = {k: i for i, k in enumerate(test_image_ids.tolist())}
idx_map = np.fromiter(
    (order[i] for i in sample_sub["image_id"].values),
    dtype=np.int64,
    count=len(sample_sub),
)
pred_test_labels_ordered = pred_test_labels[idx_map]

sub = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_test_labels_ordered}
)

assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(sample_sub)
assert sub["label"].between(0, NUM_CLASSES - 1).all()

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/29258956.py in <cell line: 0>()
     21 EPOCHS = 3  # keep as-is to preserve runtime/approach
     22 history = model.fit(
---> 23     train_ds,
     24     validation_data=val_ds,
     25     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

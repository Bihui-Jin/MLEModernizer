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

0.8851616802659413

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from google.protobuf.message_factory import MessageFactory  # type: ignore

    if not hasattr(MessageFactory, "GetPrototype") and hasattr(
        MessageFactory, "GetMessageClass"
    ):
        MessageFactory.GetPrototype = MessageFactory.GetMessageClass  # type: ignore[attr-defined]
except Exception:
    pass

import tensorflow as tf
import numpy as np
import pandas as pd
import random

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def get_strategy():
    """
    Detects and returns the best TensorFlow distribution strategy:
    - TPUStrategy for TPU(s)
    - MirroredStrategy for GPU(s)
    - Default strategy for CPU
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect TPU
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Using TPU strategy:", type(strategy).__name__)
    except (ValueError, tf.errors.NotFoundError):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            print("Using GPU strategy:", type(strategy).__name__)
        else:
            strategy = tf.distribute.get_strategy()
            print("No TPU/GPU found. Using default strategy:", type(strategy).__name__)
    print("REPLICAS:", strategy.num_replicas_in_sync)
    return strategy


strategy = get_strategy()



## === cell 2
AUTO = tf.data.AUTOTUNE
IMAGE_SIZE = (512, 512)
BATCH_SIZE_PER_REPLICA = 8
NUM_CLASSES = 5
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
print(f"Global Batch size: {BATCH_SIZE}")



## === cell 3
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

print("Data dir exists:", tf.io.gfile.exists(DATA_DIR))
print("Train CSV exists:", tf.io.gfile.exists(train_csv_path))
print("Sample sub exists:", tf.io.gfile.exists(sample_sub_path))



## === cell 4
MODEL_DIR = (
    "/kaggle/input/cassava-leaf-model/tensorflow2/default/1/final_model_cassava.keras"
)




## === cell 5
def seed_everthing(SEED=28):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    print(f"Global seed set to {SEED}")


seed_everthing(28)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.config.optimizer.set_jit(False)  # keep conservative; model already optimized enough




## === cell 6
def decode_train_example(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    example = tf.io.parse_single_example(example, feature_description)

    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    label = tf.cast(example["label"], tf.int32)
    return image, label


def decode_test_example(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)

    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)

    return image, example["image_name"]




## === cell 7
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess(image, label):
    image = preprocess_input(image)
    return image, label




## === cell 8
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        tf.keras.layers.RandomRotation(40 / 360),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomZoom(0.2, 0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
    ],
    name="data_augmentation",
)



## === cell 9
train_files = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
test_files = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))

if len(train_files) == 0:
    raise FileNotFoundError(f"No train TFRecords found in: {TRAIN_TFREC_DIR}")
if len(test_files) == 0:
    raise FileNotFoundError(f"No test TFRecords found in: {TEST_TFREC_DIR}")

train_files = sorted(train_files)
test_files = sorted(test_files)

raw_train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=AUTO).map(
    decode_train_example, num_parallel_calls=AUTO
)
raw_test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTO).map(
    decode_test_example, num_parallel_calls=AUTO
)

print("Train TFRecord files:", len(train_files))
print("Test TFRecord files:", len(test_files))



## === cell 10
train_count = int(pd.read_csv(train_csv_path).shape[0])
val_fraction = 0.1
val_count = int(train_count * val_fraction)
train_take = train_count - val_count

shuffled = raw_train_ds.shuffle(
    buffer_size=4096, seed=28, reshuffle_each_iteration=True
)

shuffled = shuffled.filter(lambda x, y: tf.greater_equal(y, 0))

train_ds = (
    shuffled.take(train_take)
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTO)
    .map(preprocess, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

val_ds = (
    shuffled.skip(train_take)
    .take(val_count)
    .map(preprocess, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_ds = raw_test_ds.batch(BATCH_SIZE).prefetch(AUTO)

steps_per_epoch = max(1, int(np.ceil(train_take / BATCH_SIZE)))
validation_steps = max(1, int(np.ceil(val_count / BATCH_SIZE)))

print("Datasets ready. Train/Val split:", train_take, "/", val_count)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1445911804.py in <cell line: 0>()
     14 train_ds = (
     15     shuffled.take(train_take)
---> 16     .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTO)
     17     .map(preprocess, num_parallel_calls=AUTO)
     18     .batch(BATCH_SIZE, drop_remainder=False)

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

/tmp/__autograph_generated_file8nbte5im.py in <lambda>(x, y)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda x, y: ag__.with_function_scope(lambda lscope: (ag__.converted_call(data_augmentation, (x,), dict(training=True), lscope), y), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_file8nbte5im.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda x, y: ag__.with_function_scope(lambda lscope: (ag__.converted_call(data_augmentation, (x,), dict(training=True), lscope), y), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

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

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in _adjust_input_rank(self, flat_inputs)
    270                     adjusted.append(ops.expand_dims(x, axis=-1))
    271                     continue
--> 272             raise ValueError(
    273                 f"Invalid input shape for input {x}. Expected shape "
    274                 f"{ref_shape}, but input has incompatible shape {x.shape}"

ValueError: in user code:

    File "/tmp/ipykernel_11/1445911804.py", line 16, in None  *
        lambda x, y: (data_augmentation(x, training=True), y)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler  **
        raise e.with_traceback(filtered_tb) from None
    File "/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py", line 272, in _adjust_input_rank
        raise ValueError(

    ValueError: Exception encountered when calling Sequential.call().
    
    Invalid input shape for input Tensor("args_0:0", shape=(512, 512, 3), dtype=float32). Expected shape (None, 512, 512, 3), but input has incompatible shape (512, 512, 3)
    
    Arguments received by Sequential.call():
      • inputs=tf.Tensor(shape=(512, 512, 3), dtype=float32)
      • training=True
      • mask=None


## === cell 11
def build_fallback_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = inputs
    x = preprocess_input(x)
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
    )
    base.trainable = False  # fast + stable within 600s
    x = base.output
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


with strategy.scope():
    if tf.io.gfile.exists(MODEL_DIR):
        model = tf.keras.models.load_model(MODEL_DIR)
        print("Loaded external model:", MODEL_DIR)
    else:
        print("External model not found, training fallback model instead:", MODEL_DIR)
        model = build_fallback_model()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=2,
        )




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1126143688.py in <cell line: 0>()
     26         model = build_fallback_model()
     27         model.fit(
---> 28             train_ds,
     29             validation_data=val_ds,
     30             epochs=3,

NameError: name 'train_ds' is not defined

## === cell 12
@tf.function(reduce_retracing=True)
def tta_predict_batch(model, images, num_tta: int):
    images = tf.cast(images, tf.float32)
    num_tta = tf.cast(num_tta, tf.int32)

    b = tf.shape(images)[0]
    images_rep = tf.repeat(images, repeats=num_tta, axis=0)

    aug = data_augmentation(images_rep, training=True)
    aug = preprocess_input(aug)

    preds_rep = tf.cast(model(aug, training=False), tf.float32)  # [b*num_tta, C]
    preds_rep = tf.reshape(preds_rep, [b, num_tta, NUM_CLASSES])
    preds = tf.reduce_mean(preds_rep, axis=1)
    return preds




## === cell 13
tta_num_augmentations = 10
all_probs = []
all_ids_tensors = []

for images, ids in test_ds:
    probs = tta_predict_batch(model, images, tta_num_augmentations).numpy()
    all_probs.append(probs)
    all_ids_tensors.append(ids)

all_probs = np.concatenate(all_probs, axis=0)  # [N,5]
pred_labels = np.argmax(all_probs, axis=1).astype(int)

all_ids_concat = tf.concat(all_ids_tensors, axis=0).numpy().tolist()
all_image_ids = [x.decode("utf-8") for x in all_ids_concat]

print(
    "Predictions done. N:", len(pred_labels), "Unique labels:", np.unique(pred_labels)
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2145719802.py in <cell line: 0>()
      3 all_ids_tensors = []
      4 
----> 5 for images, ids in test_ds:
      6     probs = tta_predict_batch(model, images, tta_num_augmentations).numpy()
      7     all_probs.append(probs)

NameError: name 'test_ds' is not defined

## === cell 14
sample_sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"image_id": all_image_ids, "label": pred_labels})

submission_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    fallback_label = int(np.bincount(pred_labels, minlength=NUM_CLASSES).argmax())
    submission_df["label"] = submission_df["label"].fillna(fallback_label).astype(int)

submission_df["label"] = submission_df["label"].astype(int)
submission_df.to_csv("submission.csv", index=False)

print("Submission file created successfully! -> submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
print("Any NA labels:", submission_df["label"].isna().any())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1721683119.py in <cell line: 0>()
      1 sample_sub = pd.read_csv(sample_sub_path)
----> 2 pred_df = pd.DataFrame({"image_id": all_image_ids, "label": pred_labels})
      3 
      4 submission_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
      5 

NameError: name 'all_image_ids' is not defined

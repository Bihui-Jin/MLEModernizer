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

0.8848594741613781

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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
import tensorflow as tf



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTO = tf.data.AUTOTUNE
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_CLASSES = 5

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 2
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

train_files = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_files = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

assert len(train_files) > 0, f"No train tfrecords found in {TRAIN_TFREC_DIR}"
assert len(test_files) > 0, f"No test tfrecords found in {TEST_TFREC_DIR}"

print(f"Found {len(train_files)} train tfrecords, {len(test_files)} test tfrecords")



## === cell 3
_TRAIN_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _maybe_resize_512(image):
    shp = tf.shape(image)
    h, w = shp[0], shp[1]
    return tf.cond(
        tf.logical_and(tf.equal(h, IMAGE_SIZE[0]), tf.equal(w, IMAGE_SIZE[1])),
        lambda: image,
        lambda: tf.image.resize(image, IMAGE_SIZE),
    )


def decode_train_example(example):
    example = tf.io.parse_single_example(example, _TRAIN_FEATURE_DESC)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = _maybe_resize_512(image)
    label = tf.cast(example["target"], tf.int32)
    return image, label


def decode_test_example(example):
    example = tf.io.parse_single_example(example, _TEST_FEATURE_DESC)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = _maybe_resize_512(image)
    return image, example["image_name"]




## === cell 4
from tensorflow.keras.applications.efficientnet import preprocess_input


def decode_and_preprocess_train(example):
    example = tf.io.parse_single_example(example, _TRAIN_FEATURE_DESC)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = _maybe_resize_512(image)
    image = preprocess_input(image)
    label = tf.cast(example["target"], tf.int32)
    return image, label


def decode_and_preprocess_test(example):
    example = tf.io.parse_single_example(example, _TEST_FEATURE_DESC)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = _maybe_resize_512(image)
    image = preprocess_input(image)
    return image, example["image_name"]




## === cell 5
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(40 / 360),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomZoom(0.2, 0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
    ],
    name="data_augmentation",
)



## === cell 6
options = tf.data.Options()
options.experimental_deterministic = True  # preserve deterministic semantics

try:
    options.threading.max_intra_op_parallelism = 0
except Exception:
    pass

try:
    rd_opts = tf.data.TFRecordDatasetOptions(compression_type=None)
except Exception:
    rd_opts = None


def _tfrecord_dataset(files):
    if rd_opts is not None:
        return tf.data.TFRecordDataset(files, num_parallel_reads=AUTO, options=rd_opts)
    return tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)


VAL_FRAC = 0.1
SHUFFLE_BUFFER = 8192

train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)
n_total = len(train_df)
n_val = max(1, int(n_total * VAL_FRAC))
n_train = n_total - n_val

base_all = _tfrecord_dataset(train_files).with_options(options)
base_all = base_all.map(
    decode_and_preprocess_train, num_parallel_calls=AUTO, deterministic=True
)

train_base = base_all.take(n_train)
val_base = base_all.skip(n_train).take(n_val)

TRAIN_CACHE_PATH = "/kaggle/working/train_cache"
VAL_CACHE_PATH = "/kaggle/working/val_cache"
TEST_CACHE_PATH = "/kaggle/working/test_cache"

train_base = train_base.cache(TRAIN_CACHE_PATH)
val_base = val_base.cache(VAL_CACHE_PATH)

train_ds = (
    train_base.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

val_ds = val_base.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

test_ds = _tfrecord_dataset(test_files).with_options(options)
test_ds = (
    test_ds.map(decode_and_preprocess_test, num_parallel_calls=AUTO, deterministic=True)
    .cache(TEST_CACHE_PATH)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

steps_per_epoch = n_train // BATCH_SIZE
validation_steps = int(np.ceil(n_val / BATCH_SIZE))

print("Datasets created successfully!")
print("n_total:", n_total, "n_train:", n_train, "n_val:", n_val)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1327146902.py in <cell line: 0>()
     33 # Speedup: single fused map (decode+preprocess) and cache after that.
     34 base_all = _tfrecord_dataset(train_files).with_options(options)
---> 35 base_all = base_all.map(
     36     decode_and_preprocess_train, num_parallel_calls=AUTO, deterministic=True
     37 )

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

/tmp/__autograph_generated_filenwt82gqo.py in tf__decode_and_preprocess_train(example)
     10                 example = ag__.converted_call(ag__.ld(tf).io.parse_single_example, (ag__.ld(example), ag__.ld(_TRAIN_FEATURE_DESC)), None, fscope)
     11                 image = ag__.converted_call(ag__.ld(tf).image.decode_jpeg, (ag__.ld(example)['image'],), dict(channels=3), fscope)
---> 12                 image = ag__.converted_call(ag__.ld(_maybe_resize_512), (ag__.ld(image),), None, fscope)
     13                 image = ag__.converted_call(ag__.ld(preprocess_input), (ag__.ld(image),), None, fscope)
     14                 label = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(example)['target'], ag__.ld(tf).int32), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filem7_u03iy.py in tf___maybe_resize(image)
     12                 try:
     13                     do_return = True
---> 14                     retval_ = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).logical_and, (ag__.converted_call(ag__.ld(tf).equal, (ag__.ld(h), ag__.ld(IMAGE_SIZE)[0]), None, fscope), ag__.converted_call(ag__.ld(tf).equal, (ag__.ld(w), ag__.ld(IMAGE_SIZE)[1]), None, fscope)), None, fscope), ag__.autograph_artifact(lambda: ag__.ld(image)), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(image), ag__.ld(IMAGE_SIZE)), None, fscope))), None, fscope)
     15                 except:
     16                     do_return = False

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/cond_v2.py in error(branch_idx, error_detail)
    878 
    879   def error(branch_idx, error_detail):
--> 880     raise TypeError(
    881         "{b0_name} and {bn_name} arguments to {op_name} must have the same "
    882         "number, type, and overall structure of return values.\n"

TypeError: in user code:

    File "/tmp/ipykernel_11/1229607729.py", line 9, in decode_and_preprocess_train  *
        image = _maybe_resize_512(image)
    File "/tmp/ipykernel_11/3812835827.py", line 21, in _maybe_resize_512  *
        lambda: tf.image.resize(image, IMAGE_SIZE),

    TypeError: true_fn and false_fn arguments to tf.cond must have the same number, type, and overall structure of return values.
    
    true_fn output: Tensor("cond/Identity:0", shape=(None, None, 3), dtype=uint8)
    false_fn output: Tensor("cond/Identity:0", shape=(512, 512, 3), dtype=float32)
    
    Error details:
    Tensor("cond/Identity:0", shape=(None, None, 3), dtype=uint8) and Tensor("cond/Identity:0", shape=(512, 512, 3), dtype=float32) have different types


## === cell 7
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False  # keep identical training approach

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = inputs
x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 8
EPOCHS = 3
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3117764597.py in <cell line: 0>()
      1 EPOCHS = 3
      2 history = model.fit(
----> 3     train_ds,
      4     validation_data=val_ds,
      5     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 9
@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec(shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    ],
)
def _tta_mean_predict(images, num_tta):
    preds = tf.zeros((tf.shape(images)[0], NUM_CLASSES), dtype=tf.float32)
    i = tf.constant(0, dtype=tf.int32)

    def cond(i, preds):
        return i < num_tta

    def body(i, preds):
        augmented = data_augmentation(images, training=True)
        p = model(augmented, training=False)
        preds = preds + tf.cast(p, tf.float32)
        return i + 1, preds

    _, preds_sum = tf.while_loop(
        cond, body, loop_vars=[i, preds], parallel_iterations=1
    )
    return preds_sum / tf.cast(num_tta, tf.float32)


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec(shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.string),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    ],
)
def _tta_predict_step(images, ids, num_tta):
    return _tta_mean_predict(images, num_tta), ids


@tf.function(reduce_retracing=True)
def _predict_testset_tta(ds, num_tta):
    preds_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
    ids_ta = tf.TensorArray(tf.string, size=0, dynamic_size=True, infer_shape=False)

    def _reduce_fn(state, elem):
        preds_ta, ids_ta = state
        idx, (batch_images, batch_ids) = elem
        mean_preds, out_ids = _tta_predict_step(batch_images, batch_ids, num_tta)
        preds_ta = preds_ta.write(idx, mean_preds)
        ids_ta = ids_ta.write(idx, out_ids)
        return (preds_ta, ids_ta)

    preds_ta, ids_ta = ds.enumerate().reduce((preds_ta, ids_ta), _reduce_fn)
    return preds_ta.concat(), ids_ta.concat()


tta_num_augmentations = 10
num_tta_tensor = tf.constant(tta_num_augmentations, dtype=tf.int32)

tta_predictions_tf, tta_image_names_tf = _predict_testset_tta(test_ds, num_tta_tensor)

tta_predictions = tta_predictions_tf.numpy()
tta_image_names_bytes = tta_image_names_tf.numpy()

if tta_predictions.size == 0:
    raise RuntimeError("No predictions were generated from test_ds (empty dataset).")

tta_predictions = np.asarray(tta_predictions)
if tta_predictions.ndim != 2 or tta_predictions.shape[1] != NUM_CLASSES:
    raise RuntimeError(f"Unexpected prediction shape: {tta_predictions.shape}")

tta_image_names = np.char.decode(tta_image_names_bytes.astype("S"), "utf-8").tolist()

print("TTA predictions shape:", tta_predictions.shape)
print("Collected image names:", len(tta_image_names))

pred_labels = np.argmax(tta_predictions, axis=1).astype(int)
print("Pred labels shape:", pred_labels.shape)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

pred_df = pd.DataFrame({"image_id": tta_image_names, "label": pred_labels})

submission_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(0).astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created successfully: {submission_path}")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())

assert submission_path.endswith(".csv")
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)
assert submission_df["label"].between(0, NUM_CLASSES - 1).all()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2284626962.py in <cell line: 0>()
     60 num_tta_tensor = tf.constant(tta_num_augmentations, dtype=tf.int32)
     61 
---> 62 tta_predictions_tf, tta_image_names_tf = _predict_testset_tta(test_ds, num_tta_tensor)
     63 
     64 tta_predictions = tta_predictions_tf.numpy()

NameError: name 'test_ds' is not defined

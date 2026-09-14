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

0.8635539437896645

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

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras:", keras.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB
assert os.path.exists(TRAIN_DIR), TRAIN_DIR
assert os.path.exists(TEST_DIR), TEST_DIR
assert os.path.exists(TRAIN_TFREC_DIR), TRAIN_TFREC_DIR
assert os.path.exists(TEST_TFREC_DIR), TEST_TFREC_DIR




## === cell 2
with open(MAP_JSON, "r") as f:
    map_classes = json.load(f)
print(json.dumps(map_classes, indent=2))
label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES, "labels:", label_list)




## === cell 3
IMG_HEIGHT = 500
IMG_WIDTH = 500
batch_size = 16

PRE_TRAINED_MODEL = "../input/unitedmodelv01/Cassava_Best_UnitedModel_V01.hdf5"
print("Pretrained exists?", os.path.exists(PRE_TRAINED_MODEL), PRE_TRAINED_MODEL)




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
sample_sub_df = pd.read_csv(SAMPLE_SUB)

assert set(train_df.columns) == {"image_id", "label"}
assert set(sample_sub_df.columns) == {"image_id", "label"}

print("Train rows:", len(train_df), "Test rows:", len(sample_sub_df))

missing = 0
for fn in train_df["image_id"].head(50):
    if not os.path.exists(os.path.join(TRAIN_DIR, fn)):
        missing += 1
print("Missing in first 50 train images:", missing)




## === cell 5
_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), _RESAMPLE)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # float32 0..1
    return arr




## === cell 6
AUTOTUNE = tf.data.AUTOTUNE


def _tf_load_image_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    return img


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_train_tfrecord(ex):
    x = tf.io.parse_single_example(ex, _TRAIN_FEATURES)
    img = _tf_load_image_from_bytes(x["image"])
    y = tf.cast(x["target"], tf.int32)
    y.set_shape([])
    return img, y


def _parse_test_tfrecord(ex):
    x = tf.io.parse_single_example(ex, _TEST_FEATURES)
    img = _tf_load_image_from_bytes(x["image"])
    image_name = x["image_name"]  # bytes
    return img, image_name


@tf.function
def _augment(x, y):
    x = tf.image.random_flip_left_right(x, seed=SEED)
    x = tf.image.random_flip_up_down(x, seed=SEED)
    x = tf.image.random_brightness(x, max_delta=0.15, seed=SEED)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y


train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df_shuf) * val_frac)
val_df = train_df_shuf.iloc[:val_size].reset_index(drop=True)
trn_df = train_df_shuf.iloc[val_size:].reset_index(drop=True)
print("Train split:", len(trn_df), "Val split:", len(val_df))

id_to_label = dict(
    zip(
        train_df["image_id"].astype(str).values,
        train_df["label"].astype(np.int32).values,
    )
)
trn_ids = set(trn_df["image_id"].astype(str).values.tolist())
val_ids = set(val_df["image_id"].astype(str).values.tolist())

trn_keys_tf = tf.constant(list(trn_ids), dtype=tf.string)
val_keys_tf = tf.constant(list(val_ids), dtype=tf.string)
dummy_vals = tf.zeros([tf.shape(trn_keys_tf)[0]], dtype=tf.int32)
dummy_vals2 = tf.zeros([tf.shape(val_keys_tf)[0]], dtype=tf.int32)
trn_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(trn_keys_tf, dummy_vals), default_value=-1
)
val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(val_keys_tf, dummy_vals2), default_value=-1
)


def _is_in_trn(img, name):
    name = tf.strings.strip(name)
    return trn_table.lookup(name) >= 0


def _is_in_val(img, name):
    name = tf.strings.strip(name)
    return val_table.lookup(name) >= 0


train_tfrecs = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in tf.io.gfile.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrecs = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in tf.io.gfile.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
print("Train tfrecs:", len(train_tfrecs), "Test tfrecs:", len(test_tfrecs))

opts = tf.data.Options()
opts.experimental_deterministic = True  # keep stable results
opts.threading.private_threadpool_size = 0

TRN_CACHE = "/kaggle/working/trn_cache.tfcache"
VAL_CACHE = "/kaggle/working/val_cache.tfcache"

raw_train = tf.data.TFRecordDataset(
    train_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(opts)
raw_train = raw_train.map(_parse_test_tfrecord, num_parallel_calls=AUTOTUNE)

trn_ds = raw_train.filter(_is_in_trn).map(
    lambda img, name: (
        img,
        tf.cast(id_to_label[name.numpy().decode("utf-8")], tf.int32),
    ),
    num_parallel_calls=1,
)

label_keys_tf = tf.constant(list(id_to_label.keys()), dtype=tf.string)
label_vals_tf = tf.constant(list(id_to_label.values()), dtype=tf.int32)
label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(label_keys_tf, label_vals_tf), default_value=-1
)


def _attach_label(img, name):
    name = tf.strings.strip(name)
    y = label_table.lookup(name)
    y.set_shape([])
    return img, y


trn_ds = (
    raw_train.filter(_is_in_trn)
    .map(_attach_label, num_parallel_calls=AUTOTUNE)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .cache(TRN_CACHE)
    .map(_augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = (
    raw_train.filter(_is_in_val)
    .map(_attach_label, num_parallel_calls=AUTOTUNE)
    .cache(VAL_CACHE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/567591771.py in <cell line: 0>()
    127 raw_train = raw_train.map(_parse_test_tfrecord, num_parallel_calls=AUTOTUNE)
    128 
--> 129 trn_ds = raw_train.filter(_is_in_trn).map(
    130     lambda img, name: (
    131         img,

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

/tmp/__autograph_generated_file2chwxpwp.py in <lambda>(img, name)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda img, name: ag__.with_function_scope(lambda lscope: (img, ag__.converted_call(tf.cast, (id_to_label[ag__.converted_call(ag__.converted_call(name.numpy, (), None, lscope).decode, ('utf-8',), None, lscope)], tf.int32), None, lscope)), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_file2chwxpwp.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda img, name: ag__.with_function_scope(lambda lscope: (img, ag__.converted_call(tf.cast, (id_to_label[ag__.converted_call(ag__.converted_call(name.numpy, (), None, lscope).decode, ('utf-8',), None, lscope)], tf.int32), None, lscope)), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __getattr__(self, name)
    258         tf.experimental.numpy.experimental_enable_numpy_behavior()
    259       """)
--> 260     self.__getattribute__(name)
    261 
    262   @property

AttributeError: in user code:

    File "/tmp/ipykernel_11/567591771.py", line 130, in None  *
        )

    AttributeError: 'SymbolicTensor' object has no attribute 'numpy'


## === cell 7
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()




## === cell 8
EPOCHS = 3
history = model.fit(trn_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/515619935.py in <cell line: 0>()
      1 EPOCHS = 3
----> 2 history = model.fit(trn_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      3 
      4 

NameError: name 'trn_ds' is not defined

## === cell 9
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # kept to preserve core logic imports

test_datagen = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
)


def get_augmented_images(image_id, n_aug=5):
    x = load_single_image("TEST_DATA", image_id)
    x = np.expand_dims(x, axis=0)  # (1,H,W,3)

    it = test_datagen.flow(x, batch_size=1, shuffle=False, seed=SEED)
    image_list = [x]
    for _ in range(n_aug):
        batch = next(it)
        batch = np.clip(batch.astype(np.float32), 0.0, 1.0)
        image_list.append(batch)
    return image_list


def _build_tta_model(seed):
    return keras.Sequential(
        [
            layers.RandomFlip(mode="horizontal", seed=seed),
            layers.RandomFlip(mode="vertical", seed=seed),
            layers.RandomRotation(factor=45.0 / 360.0, fill_mode="nearest", seed=seed),
            layers.RandomZoom(
                height_factor=(-0.3, 0.3),
                width_factor=(-0.3, 0.3),
                fill_mode="nearest",
                seed=seed,
            ),
            layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="nearest",
                seed=seed,
            ),
            layers.RandomShear(
                x_factor=0.1,
                y_factor=0.1,
                fill_mode="nearest",
                seed=seed,
            ),
        ],
        name=f"tta_{seed}",
    )


test_ids = sample_sub_df["image_id"].astype(str).values
n_test = len(test_ids)
n_aug = 5
tta_total = 1 + n_aug

test_opts = tf.data.Options()
test_opts.experimental_deterministic = True

raw_test = tf.data.TFRecordDataset(
    test_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(test_opts)
test_ds = (
    raw_test.map(_parse_test_tfrecord, num_parallel_calls=AUTOTUNE)
    .cache("/kaggle/working/test_cache.tfcache")
    .batch(32, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

tta_layers = [_build_tta_model(SEED + k) for k in range(n_aug)]


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
    ],
    jit_compile=True,
)
def _predict_with_tta(batch_x):
    probs = model(batch_x, training=False)
    for layer in tta_layers:
        ax = layer(batch_x, training=True)
        ax = tf.clip_by_value(ax, 0.0, 1.0)
        probs += model(ax, training=False)
    return probs


pred_map = {}
for batch_x, batch_name in test_ds:
    batch_probs_sum = _predict_with_tta(batch_x).numpy().astype(np.float32)
    batch_probs_mean = batch_probs_sum / float(tta_total)
    batch_pred = batch_probs_mean.argmax(axis=1).astype(int)
    for nm, pr in zip(batch_name.numpy().tolist(), batch_pred.tolist()):
        pred_map[nm.decode("utf-8")] = pr

test_results = [int(pred_map[iid]) for iid in test_ids]
submission = pd.DataFrame({"image_id": test_ids, "label": test_results})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3045239305.py in <cell line: 0>()
    102 pred_map = {}
    103 for batch_x, batch_name in test_ds:
--> 104     batch_probs_sum = _predict_with_tta(batch_x).numpy().astype(np.float32)
    105     batch_probs_mean = batch_probs_sum / float(tta_total)
    106     batch_pred = batch_probs_mean.argmax(axis=1).astype(int)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Detected unsupported operations when trying to compile graph __inference__predict_with_tta_3477[_XlaMustCompile=true,config_proto=13561319589895757934,executor_type=11160318154034397263] on XLA_CPU_JIT: ImageProjectiveTransformV3 (No registered 'ImageProjectiveTransformV3' OpKernel for XLA_CPU_JIT devices compatible with node {{node tta_42_1/random_rotation_1/ImageProjectiveTransformV3}}){{node tta_42_1/random_rotation_1/ImageProjectiveTransformV3}}
The op is created at: 
File "<frozen runpy>", line 198, in _run_module_as_main
File "<frozen runpy>", line 88, in _run_code
File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>
File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start
File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start
File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever
File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once
File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request
File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute
File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell
File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
File "/tmp/ipykernel_11/3045239305.py", line 104, in <cell line: 0>
File "/tmp/ipykernel_11/3045239305.py", line 94, in _predict_with_tta
File "/tmp/ipykernel_11/3045239305.py", line 95, in _predict_with_tta
File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 908, in __call__
File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py", line 46, in __call__
File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 156, in error_handler
File "/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py", line 213, in call
File "/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py", line 182, in call
File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/function.py", line 171, in _run_through_graph
File "/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py", line 637, in call
File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/tf_data_layer.py", line 43, in __call__
File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 908, in __call__
File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py", line 46, in __call__
File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 156, in error_handler
File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/base_image_preprocessing_layer.py", line 208, in call
File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/random_rotation.py", line 115, in transform_images
File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/image.py", line 360, in affine_transform
	tf2xla conversion failed while converting __inference__predict_with_tta_3477[_XlaMustCompile=true,config_proto=13561319589895757934,executor_type=11160318154034397263]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions. [Op:__inference__predict_with_tta_3477]

## === cell 10
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["image_id", "label"]
assert len(sub_check) == len(sample_sub_df)
assert sub_check["label"].between(0, NUM_CLASSES - 1).all()
print("submission.csv OK")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3061447740.py in <cell line: 0>()
----> 1 sub_check = pd.read_csv("submission.csv")
      2 assert list(sub_check.columns) == ["image_id", "label"]
      3 assert len(sub_check) == len(sample_sub_df)
      4 assert sub_check["label"].between(0, NUM_CLASSES - 1).all()
      5 print("submission.csv OK")

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

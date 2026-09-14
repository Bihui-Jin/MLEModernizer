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

0.895436687821094

# 6. Current score

0.12967

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12967) has done: 'The timeout is dominated by the input pipeline: `flow_from_dataframe` repeatedly loads/decodes JPEGs and applies heavy augmentations in Python on a single thread, starving the GPU/CPU and extending training time. I keep the same model, loss, optimizer, epochs, and augmentation semantics, but switch to a `tf.data` pipeline that performs the *same* preprocessing and augmentations inside TensorFlow with parallel map, caching (for validation/test), prefetch, and deterministic seeding. I also eliminate slow per-row `apply(lambda ...)` path building and tune TensorFlow thread settings to reduce overhead. These changes are runtime-only and preserve evaluation semantics (same split/labels, same EfficientNetB0 head, same training loop/callbacks).'
- What this solution (achieved 0.12967) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` issue seen in some Kaggle runtimes. Then I fix the `int32`/`int64` mismatch in your stateless augmentation seeds (the `^` XOR) so the `tf.data` pipeline can build successfully and `train_ds`/`valid_ds` exist for training. Finally, I keep your model/training loop unchanged but ensure inference runs end-to-end and writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.12967) has done: 'I fix the two root-cause runtime blockers so training/inference can run end-to-end: (1) the TensorFlow import crash caused by an incompatible protobuf runtime be handled by switching to the C++ protobuf implementation (and avoiding the problematic “python” fallback), and (2) `tf.image.rotate/translate` are not available in this TensorFlow build, so I replace them with equivalent Keras preprocessing layers called inside the `tf.data` map (same augmentation intent, still stateless/deterministic per-path). Once datasets build successfully, `model.fit()` run and the script write a valid `submission.csv` with the correct columns/row count. These changes are directly tied to unblocking execution and should substantially improve the score versus the current broken/ineffective run.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{BASE_PATH}/train.csv"
SAMPLE_SUB = f"{BASE_PATH}/sample_submission.csv"
TRAIN_IMG_DIR = f"{BASE_PATH}/train_images"
TEST_IMG_DIR = f"{BASE_PATH}/test_images"

print("TensorFlow:", tf.__version__)
print(
    "Train exists:",
    os.path.exists(TRAIN_CSV),
    "Test images exists:",
    os.path.exists(TEST_IMG_DIR),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)
train_df["image_id"] = train_df["image_id"].astype(str)

train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"]

train_split, valid_split = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=SEED,
)


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


def _apply_keras_preprocess(img):
    return preprocess_input(img)


_rot_layer = keras.layers.RandomRotation(
    factor=45.0 / 180.0, fill_mode="nearest", seed=SEED
)
_trans_layer = keras.layers.RandomTranslation(
    height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
)
_zoom_layer = keras.layers.RandomZoom(
    height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest", seed=SEED
)


@tf.function
def _augment_image(img, seed_pair):
    seed_pair = tf.cast(seed_pair, tf.int32)
    img = tf.image.stateless_random_flip_left_right(img, seed_pair)

    seed2 = tf.bitwise.bitwise_xor(seed_pair, tf.constant([1, 0], dtype=tf.int32))
    img = tf.image.stateless_random_flip_up_down(img, seed2)

    s = tf.cast(seed_pair[0], tf.int32) * tf.constant(1000003, tf.int32) + tf.cast(
        seed_pair[1], tf.int32
    )
    s = tf.math.floormod(s, tf.constant(2**31 - 1, tf.int32))

    img = _rot_layer(img, training=True, seed=s)
    img = _trans_layer(img, training=True, seed=s + 1)
    img = _zoom_layer(img, training=True, seed=s + 2)

    seed7 = tf.bitwise.bitwise_xor(seed_pair, tf.constant([6, 0], dtype=tf.int32))
    shear = tf.random.stateless_uniform(
        [],
        seed7,
        minval=-0.2,
        maxval=0.2,
        dtype=tf.float32,
    )

    cx = (IMG_SIZE[1] - 1) / 2.0
    cy = (IMG_SIZE[0] - 1) / 2.0

    z = 1.0
    sh = -shear  # inverse approx for output->input
    a0 = z
    a1 = sh
    b0 = 0.0
    b1 = z

    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    return img


def make_dataset(df, training: bool):
    paths = df["path"].values.astype(str)
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _decode_resize(path)
        if training:
            base_seed = tf.constant([SEED, SEED], dtype=tf.int32)
            bucket = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
            bucket = tf.cast(bucket, tf.int64)  # fold_in expects int64 for 'data'
            seed_pair = tf.random.experimental.stateless_fold_in(base_seed, bucket)
            seed_pair = tf.cast(seed_pair, tf.int32)
            img = _augment_image(img, seed_pair)
        img = _apply_keras_preprocess(img)
        y = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    if training:
        ds = ds.prefetch(AUTOTUNE)
    else:
        ds = ds.cache().prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_split, training=True)
valid_ds = make_dataset(valid_split, training=False)

print("Prepared tf.data datasets:")
print("Train batches:", int(np.ceil(len(train_split) / BATCH_SIZE)))
print("Valid batches:", int(np.ceil(len(valid_split) / BATCH_SIZE)))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3718853889.py in <cell line: 0>()
    135 
    136 
--> 137 train_ds = make_dataset(train_split, training=True)
    138 valid_ds = make_dataset(valid_split, training=False)
    139 

/tmp/ipykernel_11/3718853889.py in make_dataset(df, training)
    126         return img, y
    127 
--> 128     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    129     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    130     if training:

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

/tmp/__autograph_generated_filezze3zqou.py in tf___map_fn(path, label)
     33                 bucket = ag__.Undefined('bucket')
     34                 seed_pair = ag__.Undefined('seed_pair')
---> 35                 ag__.if_stmt(ag__.ld(training), if_body, else_body, get_state, set_state, ('img',), 1)
     36                 img = ag__.converted_call(ag__.ld(_apply_keras_preprocess), (ag__.ld(img),), None, fscope)
     37                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(label), ag__.ld(NUM_CLASSES)), dict(dtype=ag__.ld(tf).float32), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
-> 1217     _py_if_stmt(cond, body, orelse)
   1218 
   1219 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _py_if_stmt(cond, body, orelse)
   1268 def _py_if_stmt(cond, body, orelse):
   1269   """Overload of if_stmt that executes a Python if statement."""
-> 1270   return body() if cond else orelse()

/tmp/__autograph_generated_filezze3zqou.py in if_body()
     25                     seed_pair = ag__.converted_call(ag__.ld(tf).random.experimental.stateless_fold_in, (ag__.ld(base_seed), ag__.ld(bucket)), None, fscope)
     26                     seed_pair = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(seed_pair), ag__.ld(tf).int32), None, fscope)
---> 27                     img = ag__.converted_call(ag__.ld(_augment_image), (ag__.ld(img), ag__.ld(seed_pair)), None, fscope)
     28 
     29                 def else_body():

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

/tmp/__autograph_generated_filef7swdgyk.py in tf___augment_image(img, seed_pair)
     14                 s = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(seed_pair)[0], ag__.ld(tf).int32), None, fscope) * ag__.converted_call(ag__.ld(tf).constant, (1000003, ag__.ld(tf).int32), None, fscope) + ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(seed_pair)[1], ag__.ld(tf).int32), None, fscope)
     15                 s = ag__.converted_call(ag__.ld(tf).math.floormod, (ag__.ld(s), ag__.converted_call(ag__.ld(tf).constant, (2 ** 31 - 1, ag__.ld(tf).int32), None, fscope)), None, fscope)
---> 16                 img = ag__.converted_call(ag__.ld(_rot_layer), (ag__.ld(img),), dict(training=True, seed=ag__.ld(s)), fscope)
     17                 img = ag__.converted_call(ag__.ld(_trans_layer), (ag__.ld(img),), dict(training=True, seed=ag__.ld(s) + 1), fscope)
     18                 img = ag__.converted_call(ag__.ld(_zoom_layer), (ag__.ld(img),), dict(training=True, seed=ag__.ld(s) + 2), fscope)

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/tf_data_layer.py in __call__(self, inputs, **kwargs)
     45                 self.backend.reset()
     46                 if switch_convert_input_args:
---> 47                     self._convert_input_args = True
     48             return outputs
     49         return super().__call__(inputs, **kwargs)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

/usr/lib/python3.11/inspect.py in bind(self, *args, **kwargs)
   3193         if the passed arguments can not be bound.
   3194         """
-> 3195         return self._bind(args, kwargs)
   3196 
   3197     def bind_partial(self, /, *args, **kwargs):

/usr/lib/python3.11/inspect.py in _bind(self, args, kwargs, partial)
   3182                 arguments[kwargs_param.name] = kwargs
   3183             else:
-> 3184                 raise TypeError(
   3185                     'got an unexpected keyword argument {arg!r}'.format(
   3186                         arg=next(iter(kwargs))))

TypeError: in user code:

    File "/tmp/ipykernel_11/3718853889.py", line 123, in _map_fn  *
        img = _augment_image(img, seed_pair)
    File "/tmp/ipykernel_11/3718853889.py", line 66, in _augment_image  *
        img = _rot_layer(img, training=True, seed=s)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/tf_data_layer.py", line 43, in __call__  **
        outputs = super().__call__(inputs, **kwargs)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
        raise e.with_traceback(filtered_tb) from None
    File "/usr/lib/python3.11/inspect.py", line 3195, in bind
        return self._bind(args, kwargs)
    File "/usr/lib/python3.11/inspect.py", line 3184, in _bind
        raise TypeError(

    TypeError: got an unexpected keyword argument 'seed'


## === cell 2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-6,
    verbose=1,
)

USE_EXTERNAL_ENSEMBLE = False
external_model_paths = {
    "cropnet": "/kaggle/input/cropnet_model/tensorflow2/default/1/kaggle/working/model_feature_extraction_tf",
    "densenet": "/kaggle/input/densenet_model/tensorflow2/default/1/kaggle/working/kaggle/working/densenet_model_tf",
    "efficientnetb4": "/kaggle/input/efficientnetb4_model/tensorflow2/default/1/kaggle/working/kaggle/working/efficientnet_model_tf",
}
if all(os.path.exists(p) for p in external_model_paths.values()):
    USE_EXTERNAL_ENSEMBLE = True

print("USE_EXTERNAL_ENSEMBLE =", USE_EXTERNAL_ENSEMBLE)

inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
out = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=inp, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 3
EPOCHS = 8  # keep identical

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1140712519.py in <cell line: 0>()
      2 
      3 history = model.fit(
----> 4     train_ds,
      5     validation_data=valid_ds,
      6     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)

sample_sub["path"] = TEST_IMG_DIR + "/" + sample_sub["image_id"]
test_paths = sample_sub["path"].values.astype(str)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _test_map_fn(path):
    img = _decode_resize(path)
    img = _apply_keras_preprocess(img)
    return img


test_ds = test_ds.map(_test_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

submission_df = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"].values,
        "label": preds,
    }
)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
assert submission_df.shape[0] == sample_sub.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]
assert out_path.endswith(".csv")

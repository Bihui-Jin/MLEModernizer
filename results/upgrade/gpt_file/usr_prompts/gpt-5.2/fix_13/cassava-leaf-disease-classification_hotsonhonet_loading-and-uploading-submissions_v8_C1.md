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

0.8433061347839227

# 6. Current score

0.18348

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.18348) has done: 'The timeout is most likely coming from building the train/val split by running two `Dataset.filter(...)` passes over the full TFRecord stream (forcing extra decode/resize work), plus a slow Python loop to collect test `image_id`s by iterating the whole test dataset once before predicting. I keep the same model, augmentation, epochs, and losses, but change the input pipeline to (1) create a single parsed dataset and split it in one pass using `group_by_window` (so each example is decoded only once), and (2) extract test `image_id`s via a batched `dataset.map(...).as_numpy_iterator()` path while predicting from the same underlying batched stream to avoid an extra full dataset pass. I also enable `drop_remainder=True` for the training batches (same data/epochs, faster kernels) and keep determinism/seeds unchanged. These changes are runtime-only and preserve evaluation semantics.'

# 9. Code solution

## === cell 0
TRAIN_IMG_LOC = "../input/cassava-leaf-disease-classification/train_images"
TEST_IMG_LOC = "../input/cassava-leaf-disease-classification/test_images"
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "../input/cassava-leaf-disease-classification/sample_submission.csv"

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("ALL Modules are successfully loaded")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (300, 300)  # keep same as original inference smart_resize target
BATCH_SIZE = 16
NUM_CLASSES = 5
EPOCHS = 5  # keep as-is to preserve training semantics

STEPS_PER_EXECUTION = 32


def build_model():
    base = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model, base


model, base_model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
    steps_per_execution=STEPS_PER_EXECUTION,  # runtime-only optimization
)
print("Model build/compile complete")




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = TRAIN_IMG_LOC + "/" + train_df["image_id"].astype(str)

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
train_idx, val_idx = next(sss.split(train_df["filepath"], train_df["label"]))
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train size:", len(tr_df), "Val size:", len(va_df))
print("Label distribution (train):")
print(tr_df["label"].value_counts().sort_index())

aug = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.1, seed=SEED),
    ],
    name="aug",
)

ds_options = tf.data.Options()
ds_options.experimental_deterministic = True
try:
    ds_options.experimental_optimization.apply_default_optimizations = True
    ds_options.experimental_optimization.autotune_buffers = True
    ds_options.experimental_optimization.map_fusion = True
    ds_options.experimental_optimization.parallel_batch = True
except Exception:
    pass

TRAIN_TFREC_GLOB = (
    "../input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
)
TEST_TFREC_GLOB = "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"
train_tfrec_files = tf.io.gfile.glob(TRAIN_TFREC_GLOB)
test_tfrec_files = tf.io.gfile.glob(TEST_TFREC_GLOB)
train_tfrec_files = sorted(train_tfrec_files)
test_tfrec_files = sorted(test_tfrec_files)

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


@tf.function
def _parse_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img, label, image_id


@tf.function
def _apply_aug(img, label, image_id):
    img = aug(img, training=True)
    return img, label, image_id


@tf.function
def _drop_id(img, label, image_id):
    return img, label


@tf.function
def _keep_img(img, label, image_id):
    return img


val_ids = va_df["image_id"].astype(str).tolist()
val_ids_tf = tf.constant(val_ids, dtype=tf.string)
val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=val_ids_tf,
        values=tf.ones([len(val_ids)], dtype=tf.int32),
        key_dtype=tf.string,
        value_dtype=tf.int32,
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)


@tf.function
def _is_val_fast(img, label, image_id):
    return tf.not_equal(val_table.lookup(image_id), 0)


def make_split_ds_from_tfrecords(tfrec_files):
    base_ds = tf.data.TFRecordDataset(
        tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(ds_options)
    base_ds = base_ds.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    base_ds = base_ds.apply(tf.data.experimental.ignore_errors())

    def key_func(img, label, image_id):
        return tf.cast(_is_val_fast(img, label, image_id), tf.int64)

    def reduce_func(key, ds):
        return ds

    grouped = base_ds.apply(
        tf.data.experimental.group_by_window(
            key_func=key_func,
            reduce_func=reduce_func,
            window_size=10**9,  # effectively "all elements per key"
        )
    )

    def _take_key(ds_of_ds, desired_key):
        def _is_desired(subds):
            one = subds.take(1).map(
                lambda img, label, image_id: key_func(img, label, image_id)
            )
            return tf.equal(
                tf.cast(tf.data.experimental.get_single_element(one), tf.int64),
                tf.cast(desired_key, tf.int64),
            )

        first = ds_of_ds.take(1)
        second = ds_of_ds.skip(1).take(1)

        def _select_first():
            return first.flat_map(lambda x: x)

        def _select_second():
            return second.flat_map(lambda x: x)

        first_ok = tf.data.experimental.get_single_element(
            first.map(_is_desired).take(1)
        )
        return tf.cond(first_ok, _select_first, _select_second)

    train_stream = _take_key(grouped, desired_key=0)
    val_stream = _take_key(grouped, desired_key=1)

    train_stream = train_stream.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    train_stream = train_stream.map(_apply_aug, num_parallel_calls=tf.data.AUTOTUNE)
    train_stream = train_stream.map(_drop_id, num_parallel_calls=tf.data.AUTOTUNE)
    train_stream = train_stream.batch(BATCH_SIZE, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE
    )

    val_stream = val_stream.map(_drop_id, num_parallel_calls=tf.data.AUTOTUNE)
    val_stream = val_stream.cache()
    val_stream = val_stream.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        tf.data.AUTOTUNE
    )

    return train_stream, val_stream


train_ds, val_ds = make_split_ds_from_tfrecords(train_tfrec_files)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1299202665.py in <cell line: 0>()
    180 
    181 
--> 182 train_ds, val_ds = make_split_ds_from_tfrecords(train_tfrec_files)
    183 
    184 

/tmp/ipykernel_11/1299202665.py in make_split_ds_from_tfrecords(tfrec_files)
    157         return tf.cond(first_ok, _select_first, _select_second)
    158 
--> 159     train_stream = _take_key(grouped, desired_key=0)
    160     val_stream = _take_key(grouped, desired_key=1)
    161 

/tmp/ipykernel_11/1299202665.py in _take_key(ds_of_ds, desired_key)
    153 
    154         first_ok = tf.data.experimental.get_single_element(
--> 155             first.map(_is_desired).take(1)
    156         )
    157         return tf.cond(first_ok, _select_first, _select_second)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     41           "`num_parallel_calls` argument is specified."
     42       )
---> 43     return _MapDataset(
     44         input_dataset,
     45         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, force_synchronous, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, name)
    155     self._use_inter_op_parallelism = use_inter_op_parallelism
    156     self._preserve_cardinality = preserve_cardinality
--> 157     self._map_func = structured_function.StructuredFunctionWrapper(
    158         map_func,
    159         self._transformation_name(),

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

TypeError: in user code:


    TypeError: outer_factory.<locals>.inner_factory.<locals>.tf___is_desired() takes 1 positional argument but 3 were given


## === cell 3
base_model.trainable = False
history1 = model.fit(
    train_ds, validation_data=val_ds, epochs=max(1, EPOCHS // 2), verbose=2
)

base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
    steps_per_execution=STEPS_PER_EXECUTION,  # runtime-only optimization
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=max(1, EPOCHS // 2),
    verbose=2,
)

print("Training complete")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2680516508.py in <cell line: 0>()
      1 base_model.trainable = False
      2 history1 = model.fit(
----> 3     train_ds, validation_data=val_ds, epochs=max(1, EPOCHS // 2), verbose=2
      4 )
      5 

NameError: name 'train_ds' is not defined

## === cell 4
ss = pd.read_csv(SAMPLE_CSV)


def make_test_ds_with_ids(tfrec_files):
    ds = tf.data.TFRecordDataset(
        tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(ds_options)
    ds = ds.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds_full = make_test_ds_with_ids(test_tfrec_files)

test_image_ids = np.concatenate(
    [
        batch_ids.astype("U")
        for batch_ids in test_ds_full.map(
            lambda img, label, image_id: image_id
        ).as_numpy_iterator()
    ],
    axis=0,
).tolist()

test_ds_imgs = test_ds_full.map(
    lambda img, label, image_id: img, num_parallel_calls=tf.data.AUTOTUNE
)
probs = model.predict(test_ds_imgs, verbose=1, steps=None)
preds = np.argmax(probs, axis=1).astype(int)

pred_df = pd.DataFrame(
    {"image_id": np.array(test_image_ids, dtype=object), "label": preds}
)

my_submission = ss[["image_id"]].merge(pred_df, on="image_id", how="left")
my_submission["label"] = my_submission["label"].astype(int)

my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())

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

0.547899667573285

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

print("TensorFlow:", tf.__version__)
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        tf.config.threading.get_intra_op_parallelism_threads() or 0
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        tf.config.threading.get_inter_op_parallelism_threads() or 0
    )
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

DATA_OPTS = tf.data.Options()
DATA_OPTS.experimental_deterministic = True
try:
    DATA_OPTS.experimental_optimization.apply_default_optimizations = True
    DATA_OPTS.experimental_optimization.map_and_batch_fusion = True
    DATA_OPTS.experimental_optimization.parallel_batch = True
    DATA_OPTS.experimental_optimization.autotune_buffers = True
except Exception:
    pass

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64
NUM_CLASSES = 5

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


DATA_ROOT = first_existing(DATA_ROOT_CANDIDATES)
if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find dataset root in any of: {DATA_ROOT_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

for p in [TRAIN_CSV, SAMPLE_SUB, TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required path: {p}")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

print("Train:", train_df.shape, "Sample:", sample_df.shape)
print("Train label counts:\n", train_df["label"].value_counts().sort_index())

HAS_TFRECORDS = os.path.isdir(TRAIN_TFREC_DIR) and (
    len(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))) > 0
)
HAS_TEST_TFRECORDS = os.path.isdir(TEST_TFREC_DIR) and (
    len(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))) > 0
)
print("TFRecords available - train:", HAS_TFRECORDS, "test:", HAS_TEST_TFRECORDS)




## === cell 2
def build_model(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = keras.layers.Lambda(lambda t: tf.cast(t, tf.float32) / 255.0)(inputs)

    backbone = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    backbone.trainable = False

    x = backbone.output
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model()




## === cell 3
from sklearn.model_selection import train_test_split

train_paths = [os.path.join(TRAIN_DIR, fn) for fn in train_df["image_id"].values]
train_labels = train_df["label"].astype(np.int32).values

X_train, X_val, y_train, y_val = train_test_split(
    train_paths, train_labels, test_size=0.1, random_state=SEED, stratify=train_labels
)


@tf.function(reduce_retracing=True)
def load_image(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # forces RGB
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    img = tf.cast(img, tf.uint8)  # keep uint8; normalization happens in model
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


_TRAIN_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}
_TEST_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function(reduce_retracing=True)
def _tfrecord_parse_train_with_name(example):
    ex = tf.io.parse_single_example(example, _TRAIN_FEATURE_DESC)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    img = tf.cast(img, tf.uint8)
    lbl = tf.cast(ex["target"], tf.int32)
    name = ex["image_name"]
    return img, lbl, name


@tf.function(reduce_retracing=True)
def _tfrecord_parse_test_with_name(example):
    ex = tf.io.parse_single_example(example, _TEST_FEATURE_DESC)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    img = tf.cast(img, tf.uint8)
    return img, ex["image_name"]


def _build_train_val_from_tfrecords():
    tfrec_files = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
    if not tfrec_files:
        raise FileNotFoundError(f"No TFRecord files found in {TRAIN_TFREC_DIR}")

    train_names = np.array([os.path.basename(p) for p in X_train], dtype=object)
    val_names = np.array([os.path.basename(p) for p in X_val], dtype=object)

    keys = tf.constant(
        np.concatenate([train_names, val_names]).tolist(), dtype=tf.string
    )
    vals = tf.concat(
        [
            tf.ones([len(train_names)], dtype=tf.int32),
            tf.fill([len(val_names)], tf.constant(2, dtype=tf.int32)),
        ],
        axis=0,
    )
    split_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys=keys, values=vals),
        default_value=tf.constant(0, dtype=tf.int32),
    )

    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE).with_options(
        DATA_OPTS
    )
    ds = ds.map(_tfrecord_parse_train_with_name, num_parallel_calls=AUTOTUNE)

    @tf.function(reduce_retracing=True)
    def _attach_split(img, lbl, name):
        sid = split_table.lookup(name)
        return img, lbl, sid

    ds = ds.map(_attach_split, num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda img, lbl, sid: sid > 0)

    ds = ds.cache(os.path.join(CACHE_DIR, "train_tfrec.cache"))

    def key_func(img, lbl, sid):
        return sid  # 1 or 2

    def reduce_func(key, dataset):
        dataset = dataset.map(
            lambda img, lbl, sid: (img, lbl), num_parallel_calls=AUTOTUNE
        )
        return dataset

    grouped = ds.apply(
        tf.data.experimental.group_by_window(
            key_func=key_func,
            reduce_func=reduce_func,
            window_size=10**9,  # effectively "all elements for each key"
        )
    )

    ds_train_local = grouped.filter(
        lambda img, lbl: True
    )  # placeholder; replaced below

    ds_train_local = ds.filter(lambda img, lbl, sid: tf.equal(sid, 1)).map(
        lambda img, lbl, sid: (img, lbl), num_parallel_calls=AUTOTUNE
    )
    ds_val_local = ds.filter(lambda img, lbl, sid: tf.equal(sid, 2)).map(
        lambda img, lbl, sid: (img, lbl), num_parallel_calls=AUTOTUNE
    )

    ds_train_local = ds_train_local.shuffle(
        4096, seed=SEED, reshuffle_each_iteration=True
    )
    ds_train_local = ds_train_local.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        AUTOTUNE
    )
    ds_val_local = ds_val_local.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        AUTOTUNE
    )
    return ds_train_local, ds_val_local


if HAS_TFRECORDS:
    ds_train, ds_val = _build_train_val_from_tfrecords()
else:
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).with_options(
        DATA_OPTS
    )
    ds_train = ds_train.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)

    ds_train = ds_train.map(load_image, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.cache(os.path.join(CACHE_DIR, "train_files.cache"))
    ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).with_options(DATA_OPTS)
    ds_val = ds_val.map(load_image, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.cache(os.path.join(CACHE_DIR, "val_files.cache"))
    ds_val = ds_val.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2110543015.py in <cell line: 0>()
    137 
    138 if HAS_TFRECORDS:
--> 139     ds_train, ds_val = _build_train_val_from_tfrecords()
    140 else:
    141     ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).with_options(

/tmp/ipykernel_11/2110543015.py in _build_train_val_from_tfrecords()
    102         return dataset
    103 
--> 104     grouped = ds.apply(
    105         tf.data.experimental.group_by_window(
    106             key_func=key_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in apply(self, transformation_func)
   2585       A new `Dataset` with the transformation applied as described above.
   2586     """
-> 2587     dataset = transformation_func(self)
   2588     if not isinstance(dataset, data_types.DatasetV2):
   2589       raise TypeError(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/experimental/ops/grouping.py in _apply_fn(dataset)
     99   def _apply_fn(dataset):
    100     """Function from `Dataset` to `Dataset` that applies the transformation."""
--> 101     return dataset.group_by_window(
    102         key_func=key_func,
    103         reduce_func=reduce_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in group_by_window(self, key_func, reduce_func, window_size, window_size_func, name)
   3099     # pylint: disable=g-import-not-at-top,protected-access
   3100     from tensorflow.python.data.ops import group_by_window_op
-> 3101     return group_by_window_op._group_by_window(
   3102         self, key_func, reduce_func, window_size, window_size_func, name=name)
   3103     # pylint: enable=g-import-not-at-top,protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/group_by_window_op.py in _group_by_window(input_dataset, key_func, reduce_func, window_size, window_size_func, name)
     45   assert window_size_func is not None
     46 
---> 47   return _GroupByWindowDataset(
     48       input_dataset, key_func, reduce_func, window_size_func, name=name)
     49 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/group_by_window_op.py in __init__(self, input_dataset, key_func, reduce_func, window_size_func, name)
     60     """See `group_by_window()` for details."""
     61     self._input_dataset = input_dataset
---> 62     self._make_key_func(key_func, input_dataset)
     63     self._make_reduce_func(reduce_func, input_dataset)
     64     self._make_window_size_func(window_size_func)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/group_by_window_op.py in _make_key_func(self, key_func, input_dataset)
     98       return ops.convert_to_tensor(key_func(*args), dtype=dtypes.int64)
     99 
--> 100     self._key_func = structured_function.StructuredFunctionWrapper(
    101         key_func_wrapper, self._transformation_name(), dataset=input_dataset)
    102     if not self._key_func.output_structure.is_compatible_with(

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
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/group_by_window_op.py in key_func_wrapper(*args)
     96 
     97     def key_func_wrapper(*args):
---> 98       return ops.convert_to_tensor(key_func(*args), dtype=dtypes.int64)
     99 
    100     self._key_func = structured_function.StructuredFunctionWrapper(

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

ValueError: Tensor conversion requested dtype int64 for Tensor with dtype int32: <tf.Tensor 'args_2:0' shape=() dtype=int32>

## === cell 4
history1 = model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=2)

backbone = None
for layer in model.layers:
    if isinstance(layer, keras.Model) and layer.name.startswith("resnet"):
        backbone = layer
        break
if backbone is None:
    for layer in model.layers:
        if "resnet50" in layer.name.lower():
            backbone = layer
            break

if backbone is not None:
    backbone.trainable = True
    for l in backbone.layers:
        if isinstance(l, keras.layers.BatchNormalization):
            l.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history2 = model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=2)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2834305064.py in <cell line: 0>()
----> 1 history1 = model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=2)
      2 
      3 backbone = None
      4 for layer in model.layers:
      5     if isinstance(layer, keras.Model) and layer.name.startswith("resnet"):

NameError: name 'ds_train' is not defined

## === cell 5
def get_preds_from_sample(sample_df, image_dir, model_obj):
    img_ids = sample_df["image_id"].tolist()

    if HAS_TEST_TFRECORDS:
        tfrec_files = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
        if not tfrec_files:
            raise FileNotFoundError(f"No TFRecord files found in {TEST_TFREC_DIR}")

        ds_test = tf.data.TFRecordDataset(
            tfrec_files, num_parallel_reads=AUTOTUNE
        ).with_options(DATA_OPTS)
        ds_test = ds_test.map(
            _tfrecord_parse_test_with_name, num_parallel_calls=AUTOTUNE
        )

        keys = tf.constant(img_ids, dtype=tf.string)
        vals = tf.range(len(img_ids), dtype=tf.int32)
        index_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys=keys, values=vals),
            default_value=-1,
        )

        ds_test = ds_test.map(
            lambda img, name: (img, index_table.lookup(name)),
            num_parallel_calls=AUTOTUNE,
        )
        ds_test = ds_test.filter(lambda img, idx: idx >= 0)
        ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

        out = np.empty((len(img_ids),), dtype=np.int64)
        for imgs, idxs in ds_test:
            probs = model_obj.predict_on_batch(imgs).numpy()
            preds = np.argmax(probs, axis=1).astype(np.int64)
            out[idxs.numpy().astype(np.int64)] = preds

        return pd.DataFrame({"image_id": img_ids, "label": out.astype(int)})

    paths = [os.path.join(image_dir, x) for x in img_ids]
    for p in paths[:5]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Test image not found: {p}")

    ds_test = tf.data.Dataset.from_tensor_slices(paths).with_options(DATA_OPTS)
    ds_test = ds_test.map(lambda p: load_image(p, None), num_parallel_calls=AUTOTUNE)

    ds_test = ds_test.cache(os.path.join(CACHE_DIR, "test_files.cache"))

    ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    probs = model_obj.predict(ds_test, verbose=0)
    preds = np.argmax(probs, axis=1).astype(int)

    return pd.DataFrame({"image_id": img_ids, "label": preds})


predict_df = get_preds_from_sample(sample_df, TEST_DIR, model)
predict_df.to_csv("submission.csv", index=False)

print(predict_df.head())
print("Wrote submission.csv with shape:", predict_df.shape)
assert list(predict_df.columns) == ["image_id", "label"]
assert len(predict_df) == len(sample_df)
assert predict_df["label"].between(0, NUM_CLASSES - 1).all()
assert os.path.exists("submission.csv")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3370828408.py in <cell line: 0>()
     56 
     57 
---> 58 predict_df = get_preds_from_sample(sample_df, TEST_DIR, model)
     59 predict_df.to_csv("submission.csv", index=False)
     60 

/tmp/ipykernel_11/3370828408.py in get_preds_from_sample(sample_df, image_dir, model_obj)
     31         out = np.empty((len(img_ids),), dtype=np.int64)
     32         for imgs, idxs in ds_test:
---> 33             probs = model_obj.predict_on_batch(imgs).numpy()
     34             preds = np.argmax(probs, axis=1).astype(np.int64)
     35             out[idxs.numpy().astype(np.int64)] = preds

AttributeError: 'numpy.ndarray' object has no attribute 'numpy'

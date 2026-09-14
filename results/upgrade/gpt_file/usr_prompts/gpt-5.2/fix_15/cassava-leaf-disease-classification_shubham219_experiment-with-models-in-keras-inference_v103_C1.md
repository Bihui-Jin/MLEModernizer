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

3.11

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

0.572529465095195

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"

AUTOTUNE = tf.data.AUTOTUNE
preprocess = tf.keras.applications.resnet50.preprocess_input

NUM_CLASSES = 5
IMG_SIZE = (256, 256)

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # keep core logic

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tfl.Lambda(preprocess, name="preprocess")(inputs)
x = base(x, training=False)
x = tfl.GlobalAveragePooling2D()(x)
x = tfl.Dropout(0.2, seed=SEED)(x)
outputs = tfl.Dense(NUM_CLASSES, activation="softmax")(x)

my_model = Model(inputs, outputs)
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_CSV)

paths = df["image_id"].map(lambda x: os.path.join(TRAIN_IMG_DIR, x))
exists_mask = paths.map(os.path.exists).values
df = df.loc[exists_mask].reset_index(drop=True)
assert len(df) > 0, "No training images found after filtering by file existence."
df["path"] = paths.loc[exists_mask].values

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=SEED,
    stratify=df["label"],
)

BATCH_SIZE = 16

TRAIN_TFREC_GLOB = os.path.join(DATA_DIR, "train_tfrecords", "*.tfrec")
TEST_TFREC_GLOB = os.path.join(DATA_DIR, "test_tfrecords", "*.tfrec")
train_tfrecs = sorted(glob.glob(TRAIN_TFREC_GLOB))
test_tfrecs = sorted(glob.glob(TEST_TFREC_GLOB))
assert len(train_tfrecs) > 0, f"No train tfrecords found at {TRAIN_TFREC_GLOB}"
assert len(test_tfrecs) > 0, f"No test tfrecords found at {TEST_TFREC_GLOB}"

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


def _projective(img, transform):
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform[None, ...],
        output_shape=[IMG_SIZE[0], IMG_SIZE[1]],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]


@tf.function
def _augment_projective(img, seed):
    seed = tf.cast(seed, tf.int32)

    seed_flip = seed + tf.constant([1, 0], tf.int32)
    seed_tx = seed + tf.constant([2, 0], tf.int32)
    seed_ty = seed + tf.constant([3, 0], tf.int32)
    seed_zoom = seed + tf.constant([4, 0], tf.int32)
    seed_ang = seed + tf.constant([5, 0], tf.int32)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed_flip)

    tx = tf.random.stateless_uniform(
        [], minval=-0.05, maxval=0.05, seed=seed_tx
    ) * tf.cast(IMG_SIZE[1], tf.float32)
    ty = tf.random.stateless_uniform(
        [], minval=-0.05, maxval=0.05, seed=seed_ty
    ) * tf.cast(IMG_SIZE[0], tf.float32)

    zoom = tf.random.stateless_uniform([], minval=0.9, maxval=1.1, seed=seed_zoom)

    ang = tf.random.stateless_uniform([], minval=-15.0, maxval=15.0, seed=seed_ang) * (
        np.pi / 180.0
    )

    cx = (tf.cast(IMG_SIZE[1], tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_SIZE[0], tf.float32) - 1.0) / 2.0

    cos_a = tf.math.cos(ang)
    sin_a = tf.math.sin(ang)

    inv_scale = 1.0 / zoom
    a0 = inv_scale * cos_a
    a1 = inv_scale * sin_a
    b0 = inv_scale * -sin_a
    b1 = inv_scale * cos_a

    a2 = cx - a0 * cx - a1 * cy - tx
    b2 = cy - b0 * cx - b1 * cy - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)
    img = _projective(img, transform)
    return img


@tf.function
def _prep_train_from_img(img, label, seed2):
    img = _augment_projective(img, seed2)
    label = tf.cast(label, tf.int32)
    return img, label


@tf.function
def _prep_val_from_img(img, label):
    label = tf.cast(label, tf.int32)
    return img, label


train_names = train_df["image_id"].astype(str).tolist()
val_names = val_df["image_id"].astype(str).tolist()

train_keys_tf = tf.constant(train_names, dtype=tf.string)
val_keys_tf = tf.constant(val_names, dtype=tf.string)

with tf.device("/CPU:0"):
    _train_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=train_keys_tf,
            values=tf.ones([tf.shape(train_keys_tf)[0]], dtype=tf.int32),
        ),
        default_value=tf.constant(0, dtype=tf.int32),
    )
    _val_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=val_keys_tf,
            values=tf.ones([tf.shape(val_keys_tf)[0]], dtype=tf.int32),
        ),
        default_value=tf.constant(0, dtype=tf.int32),
    )


def _tfrec_dataset_options_deterministic():
    options = tf.data.Options()
    options.experimental_deterministic = True
    return options


@tf.function
def _parse_base(raw, idx):
    ex = tf.io.parse_single_example(raw, _TFREC_FEATURES)
    name = ex["image_name"]
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label, name, tf.cast(idx, tf.int32)


@tf.function
def _route_and_prep(img, label, name, idx, epoch_seed):
    in_train = tf.equal(_train_table.lookup(name), 1)
    in_val = tf.equal(_val_table.lookup(name), 1)

    name_hash = tf.cast(tf.strings.to_hash_bucket_fast(name, 2**31 - 1), tf.int32)
    seed2 = tf.stack([name_hash ^ idx, tf.cast(epoch_seed, tf.int32)], axis=0)

    img_train, lab_train = _prep_train_from_img(img, label, seed2)
    img_val, lab_val = _prep_val_from_img(img, label)

    route = tf.where(in_train, 0, tf.where(in_val, 1, 2))
    out_img = tf.cond(tf.equal(route, 0), lambda: img_train, lambda: img_val)
    out_lab = tf.cond(tf.equal(route, 0), lambda: lab_train, lambda: lab_val)
    return out_img, out_lab, route


def _make_shared_parsed_ds(tfrecs):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, buffer_size=32 * 1024 * 1024
    )
    ds = ds.with_options(_tfrec_dataset_options_deterministic())
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.enumerate()
    ds = ds.map(_parse_base, num_parallel_calls=AUTOTUNE, deterministic=True)
    return ds


def make_train_val_ds_from_tfrecords(tfrecs, batch_size):
    shared = _make_shared_parsed_ds(tfrecs)

    shared_train = shared.shuffle(
        buffer_size=4096, seed=SEED, reshuffle_each_iteration=True
    )

    epoch_seed = tf.constant(SEED, tf.int32)
    shared_train = shared_train.map(
        lambda img, label, name, idx: _route_and_prep(
            img, label, name, idx, epoch_seed
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    train_ds = (
        shared_train.filter(lambda img, label, route: tf.equal(route, 0))
        .map(
            lambda img, label, route: (img, label),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    shared_val = shared.cache()
    shared_val = shared_val.map(
        lambda img, label, name, idx: _route_and_prep(
            img, label, name, idx, epoch_seed
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    val_ds = (
        shared_val.filter(lambda img, label, route: tf.equal(route, 1))
        .map(
            lambda img, label, route: (img, label),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
    return train_ds, val_ds


train_ds, val_ds = make_train_val_ds_from_tfrecords(train_tfrecs, BATCH_SIZE)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1646221550.py in <cell line: 0>()
    231 
    232 
--> 233 train_ds, val_ds = make_train_val_ds_from_tfrecords(train_tfrecs, BATCH_SIZE)
    234 

/tmp/ipykernel_11/1646221550.py in make_train_val_ds_from_tfrecords(tfrecs, batch_size)
    181 
    182 def make_train_val_ds_from_tfrecords(tfrecs, batch_size):
--> 183     shared = _make_shared_parsed_ds(tfrecs)
    184 
    185     # Keep original training shuffle behavior (only affects training).

/tmp/ipykernel_11/1646221550.py in _make_shared_parsed_ds(tfrecs)
    176     ds = ds.apply(tf.data.experimental.ignore_errors())
    177     ds = ds.enumerate()
--> 178     ds = ds.map(_parse_base, num_parallel_calls=AUTOTUNE, deterministic=True)
    179     return ds
    180 

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

/tmp/__autograph_generated_filedc4ekxfq.py in tf___parse_base(raw, idx)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 ex = ag__.converted_call(ag__.ld(tf).io.parse_single_example, (ag__.ld(raw), ag__.ld(_TFREC_FEATURES)), None, fscope)
     11                 name = ag__.ld(ex)['image_name']
     12                 img = ag__.converted_call(ag__.ld(_decode_resize_from_bytes), (ag__.ld(ex)['image'],), None, fscope)

TypeError: in user code:

    File "/tmp/ipykernel_11/1646221550.py", line 146, in _parse_base  *
        ex = tf.io.parse_single_example(raw, _TFREC_FEATURES)

    TypeError: Input 'serialized' of 'ParseExampleV2' Op has type int64 that does not match expected type of string.


## === cell 2
EPOCHS = 2
_ = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_resize_from_bytes(ex["image"])
    return img, ex["image_name"]


def make_test_ds_from_tfrecords(tfrecs, batch_size):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, buffer_size=32 * 1024 * 1024
    )
    ds = ds.with_options(_tfrec_dataset_options_deterministic())
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds_from_tfrecords(test_tfrecs, BATCH_SIZE)


@tf.function
def _predict_batch(images, names):
    probs = my_model(images, training=False)
    preds = tf.argmax(probs, axis=-1, output_type=tf.int64)
    return names, preds


pred_ds = test_ds.map(_predict_batch, num_parallel_calls=AUTOTUNE, deterministic=True)

test_names_parts = []
pred_parts = []
for n_batch, p_batch in pred_ds:
    test_names_parts.append(n_batch.numpy())
    pred_parts.append(p_batch.numpy())

test_names = np.concatenate(test_names_parts, axis=0).astype("U")
pred_test_labels = np.concatenate(pred_parts, axis=0).astype(int)

final_csv = pd.DataFrame({"image_id": test_names, "label": pred_test_labels})
final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1874721857.py in <cell line: 0>()
      1 EPOCHS = 2
      2 _ = my_model.fit(
----> 3     train_ds,
      4     validation_data=val_ds,
      5     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 3
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.columns.tolist(), "rows:", len(sub))
assert sub.columns.tolist() == ["image_id", "label"]
assert sub["label"].between(0, 4).all()
assert sub["image_id"].str.endswith(".jpg").all()
print("submission.csv looks valid.")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3137129875.py in <cell line: 0>()
----> 1 sub = pd.read_csv("submission.csv")
      2 print(sub.head())
      3 print(sub.columns.tolist(), "rows:", len(sub))
      4 assert sub.columns.tolist() == ["image_id", "label"]
      5 assert sub["label"].between(0, 4).all()

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

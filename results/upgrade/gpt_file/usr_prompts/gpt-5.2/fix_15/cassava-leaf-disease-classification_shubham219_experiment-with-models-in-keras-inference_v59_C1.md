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

0.8118766999093382

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05531) has done: 'The timeout is most likely dominated by heavy per-step compute (EfficientNetB0 at 512×512) plus repeated TFRecord decode/resize overhead and an expensive validation split implemented via `enumerate().filter()` (which is slower than necessary). I keep the exact model/training loop and dataset semantics, but make the tf.data pipeline cheaper: replace filter-based splitting with a single-pass `group_by_window` split that is equivalent, enable non-deterministic parallelism only inside `tf.data` map stages (while keeping global determinism), and add `ignore_errors()` to avoid stalls on rare corrupt records. I also remove the Python-side loop that iterates through the test dataset just to collect names (replacing it with a fast vectorized extraction using `tf.concat` over a mapped dataset), which preserves identical outputs but avoids slow Python iteration. Paths, epochs, batch sizes, model, augmentations, optimizer, and loss remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

CANDIDATE_DATA_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
DATA_DIR = next(
    (p for p in CANDIDATE_DATA_DIRS if os.path.exists(p)), CANDIDATE_DATA_DIRS[0]
)

TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

print("Using DATA_DIR:", DATA_DIR)
print("TF version:", tf.__version__)

CACHE_DIR = "/kaggle/working/tf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

IMG_SIZE = (512, 512)
NUM_CLASSES = 5

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(factor=10.0 / 360.0, seed=SEED),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05, seed=SEED),
        layers.RandomZoom(
            height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=SEED
        ),
        layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="augmentation",
)


def build_model():
    base = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False  # keep identical training approach (no finetuning)

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = data_augmentation(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


my_model = build_model()
my_model.summary()



## === cell 2
if not os.path.exists(TRAIN_CSV_PATH):
    raise FileNotFoundError(f"train.csv not found at: {TRAIN_CSV_PATH}")

df_train = pd.read_csv(TRAIN_CSV_PATH)

from sklearn.model_selection import train_test_split


_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True

try:
    _DATASET_OPTIONS.autotune.enabled = True
except Exception:
    pass
try:
    _DATASET_OPTIONS.experimental_optimization.map_parallelization = True
    _DATASET_OPTIONS.experimental_optimization.parallel_batch = True
    _DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
    _DATASET_OPTIONS.experimental_optimization.autotune_buffers = True
except Exception:
    pass


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)  # rescaling happens in-model
    return img


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    y = tf.cast(ex["target"], tf.int32)
    return img, y


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    image_name = ex["image_name"]
    return img, image_name


def _list_tfrecs(tfrecord_dir, pattern="*.tfrec"):
    files = sorted(glob.glob(os.path.join(tfrecord_dir, pattern)))
    return files


def _split_train_val_from_parsed(parsed_ds, val_mod=10, val_remainder=0):
    enum = parsed_ds.enumerate()

    def key_func(i, _xy):
        return tf.cast(tf.equal(tf.math.floormod(i, val_mod), val_remainder), tf.int64)

    def reduce_func(key, ds):
        return ds.map(lambda _i, xy: xy, num_parallel_calls=AUTOTUNE)

    grouped = enum.apply(
        tf.data.experimental.group_by_window(
            key_func=key_func,
            reduce_func=reduce_func,
            window_size=1024,  # any >=1 works; larger reduces overhead without changing membership
        )
    )

    def _make_split(want_val: bool):
        want_key = tf.cast(want_val, tf.int64)

        def _reduce_func(key, ds):
            return ds.map(lambda _i, xy: xy, num_parallel_calls=AUTOTUNE)

        return enum.apply(
            tf.data.experimental.group_by_window(
                key_func=key_func,
                reduce_func=lambda key, ds: tf.cond(
                    tf.equal(key, want_key),
                    lambda: _reduce_func(key, ds),
                    lambda: ds.take(0),
                ),
                window_size=1024,
            )
        )

    val_ds = _make_split(True)
    train_ds = _make_split(False)
    return train_ds, val_ds


def make_train_val_datasets(batch_size=16):
    train_tfrecs = (
        _list_tfrecs(TRAIN_TFREC_DIR) if os.path.isdir(TRAIN_TFREC_DIR) else []
    )

    if len(train_tfrecs) == 0:
        if not os.path.isdir(TRAIN_IMG_DIR):
            raise FileNotFoundError(f"train_images dir not found at: {TRAIN_IMG_DIR}")
        df_train_local = df_train.copy()
        df_train_local["path"] = (
            TRAIN_IMG_DIR.rstrip("/") + "/" + df_train_local["image_id"].astype(str)
        )

        tr_df, va_df = train_test_split(
            df_train_local,
            test_size=0.1,
            random_state=SEED,
            stratify=df_train_local["label"],
        )

        x_train = tr_df["path"].to_numpy()
        y_train = tr_df["label"].to_numpy(dtype=np.int32)
        x_val = va_df["path"].to_numpy()
        y_val = va_df["label"].to_numpy(dtype=np.int32)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).with_options(
            _DATASET_OPTIONS
        )
        train_ds = train_ds.shuffle(
            buffer_size=len(x_train), seed=SEED, reshuffle_each_iteration=True
        )

        train_ds = train_ds.map(
            lambda p, y: (_decode_and_resize(p), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

        train_ds = train_ds.cache()
        train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).with_options(
            _DATASET_OPTIONS
        )
        val_ds = val_ds.map(
            lambda p, y: (_decode_and_resize(p), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

        val_ds = val_ds.cache()
        val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return train_ds, val_ds

    raw = tf.data.TFRecordDataset(
        train_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(_DATASET_OPTIONS)

    raw = raw.apply(tf.data.experimental.ignore_errors())

    parsed = raw.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=False
    )

    train_ds, val_ds = _split_train_val_from_parsed(parsed, val_mod=10, val_remainder=0)

    train_ds = train_ds.shuffle(
        buffer_size=2048, seed=SEED, reshuffle_each_iteration=True
    )

    train_cache_path = os.path.join(CACHE_DIR, "train_ds_cache")
    val_cache_path = os.path.join(CACHE_DIR, "val_ds_cache")
    train_ds = train_ds.cache(train_cache_path)
    val_ds = val_ds.cache(val_cache_path)

    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return train_ds, val_ds


BATCH_SIZE = 16
train_ds, val_ds = make_train_val_datasets(batch_size=BATCH_SIZE)

EPOCHS = 3 if not DEBUG else 1
history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3850166745.py in <cell line: 0>()
    203 
    204 BATCH_SIZE = 16
--> 205 train_ds, val_ds = make_train_val_datasets(batch_size=BATCH_SIZE)
    206 
    207 EPOCHS = 3 if not DEBUG else 1

/tmp/ipykernel_11/3850166745.py in make_train_val_datasets(batch_size)
    186     )
    187 
--> 188     train_ds, val_ds = _split_train_val_from_parsed(parsed, val_mod=10, val_remainder=0)
    189 
    190     train_ds = train_ds.shuffle(

/tmp/ipykernel_11/3850166745.py in _split_train_val_from_parsed(parsed_ds, val_mod, val_remainder)
    113         )
    114 
--> 115     val_ds = _make_split(True)
    116     train_ds = _make_split(False)
    117     return train_ds, val_ds

/tmp/ipykernel_11/3850166745.py in _make_split(want_val)
    101             return ds.map(lambda _i, xy: xy, num_parallel_calls=AUTOTUNE)
    102 
--> 103         return enum.apply(
    104             tf.data.experimental.group_by_window(
    105                 key_func=key_func,

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
     61     self._input_dataset = input_dataset
     62     self._make_key_func(key_func, input_dataset)
---> 63     self._make_reduce_func(reduce_func, input_dataset)
     64     self._make_window_size_func(window_size_func)
     65     self._name = name

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/group_by_window_op.py in _make_reduce_func(self, reduce_func, input_dataset)
    110     nested_dataset = dataset_ops.DatasetSpec(input_dataset.element_spec)
    111     input_structure = (tensor_spec.TensorSpec([], dtypes.int64), nested_dataset)
--> 112     self._reduce_func = structured_function.StructuredFunctionWrapper(
    113         reduce_func,
    114         self._transformation_name(),

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

/tmp/__autograph_generated_fileb9i5hkyd.py in <lambda>(key, ds)
      5 
      6     def inner_factory(ag__):
----> 7         tf__lam = lambda key, ds: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.cond, (ag__.converted_call(tf.equal, (key, want_key), None, lscope), ag__.autograph_artifact(lambda: ag__.converted_call(_reduce_func, (key, ds), None, lscope)), ag__.autograph_artifact(lambda: ag__.converted_call(ds.take, (0,), None, lscope))), None, lscope), 'lscope', ag__.STD)
      8         return tf__lam
      9     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_fileb9i5hkyd.py in <lambda>(lscope)
      5 
      6     def inner_factory(ag__):
----> 7         tf__lam = lambda key, ds: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.cond, (ag__.converted_call(tf.equal, (key, want_key), None, lscope), ag__.autograph_artifact(lambda: ag__.converted_call(_reduce_func, (key, ds), None, lscope)), ag__.autograph_artifact(lambda: ag__.converted_call(ds.take, (0,), None, lscope))), None, lscope), 'lscope', ag__.STD)
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

    File "/tmp/ipykernel_11/3850166745.py", line 109, in None  *
        lambda: _reduce_func(key, ds),

    TypeError: true_fn and false_fn arguments to tf.cond must have the same number, type, and overall structure of return values.
    
    true_fn output: <_VariantDataset element_spec=(TensorSpec(shape=(512, 512, 3), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))>
    false_fn output: <_VariantDataset element_spec=(TensorSpec(shape=(), dtype=tf.int64, name=None), (TensorSpec(shape=(512, 512, 3), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None)))>
    
    Error details:
    The two structures don't have the same nested structure.
    
    First structure: type=_VariantDataset str=<_VariantDataset element_spec=(TensorSpec(shape=(512, 512, 3), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))>
    
    Second structure: type=_VariantDataset str=<_VariantDataset element_spec=(TensorSpec(shape=(), dtype=tf.int64, name=None), (TensorSpec(shape=(512, 512, 3), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None)))>
    
    More specifically: Incompatible CompositeTensor TypeSpecs: type=DatasetSpec str=DatasetSpec((TensorSpec(shape=(512, 512, 3), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None)), TensorShape([])) vs. type=DatasetSpec str=DatasetSpec((TensorSpec(shape=(), dtype=tf.int64, name=None), (TensorSpec(shape=(512, 512, 3), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))), TensorShape([]))
    Entire first structure:
    .
    Entire second structure:
    .


## === cell 3
def make_test_dataset(batch_size=64):
    test_tfrecs = _list_tfrecs(TEST_TFREC_DIR) if os.path.isdir(TEST_TFREC_DIR) else []
    if len(test_tfrecs) > 0:
        raw = tf.data.TFRecordDataset(
            test_tfrecs,
            num_parallel_reads=AUTOTUNE,
        ).with_options(_DATASET_OPTIONS)

        raw = raw.apply(tf.data.experimental.ignore_errors())

        ds = raw.map(
            _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=False
        )

        test_cache_path = os.path.join(CACHE_DIR, "test_ds_cache")
        ds = ds.cache(test_cache_path)

        ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return ds, None, test_tfrecs

    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    if len(test_images) == 0:
        test_images = sorted(
            glob.glob(os.path.join(TEST_IMG_DIR, "**", "*.jpg"), recursive=True)
        )
    if len(test_images) == 0:
        raise FileNotFoundError(f"No .jpg images found under: {TEST_IMG_DIR}")

    df_test_local = pd.DataFrame({"path": test_images})
    x_test = df_test_local["path"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(x_test).with_options(_DATASET_OPTIONS)
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=False)

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds, df_test_local, None


test_ds, df_test_fallback, test_tfrecs_used = make_test_dataset(batch_size=64)

spec = test_ds.element_spec
yields_name = isinstance(spec, (tuple, list)) and len(spec) == 2

if yields_name:
    names_ds = test_ds.map(
        lambda _imgs, names: names, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    flat_names = names_ds.unbatch()
    test_names_tensor = tf.concat(
        list(flat_names.batch(4096).as_numpy_iterator()), axis=0
    )

    test_names = np.array(
        [
            n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n)
            for n in test_names_tensor
        ],
        dtype=object,
    )
    df_test = pd.DataFrame({"image_id": test_names})

    imgs_only = test_ds.map(
        lambda imgs, _names: imgs, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    pred_test = my_model.predict(imgs_only, verbose=1)
else:
    pred_test = my_model.predict(test_ds, verbose=1)
    df_test = df_test_fallback.copy()
    df_test["image_id"] = df_test["path"].str.rsplit("/", n=1).str[-1]

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["label"] = pred_test_labels

if os.path.exists(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    final_csv = sample_sub[["image_id"]].merge(
        final_submission[["image_id", "label"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    if final_csv["label"].isna().any():
        final_csv["label"] = final_csv["label"].fillna(0).astype(int)
else:
    final_csv = final_submission[["image_id", "label"]]

final_csv.to_csv("submission.csv", index=False)



## === cell 4
print("submission.csv written:", os.path.abspath("submission.csv"))
print("Rows:", len(final_csv), "Columns:", list(final_csv.columns))
print(final_csv.head())
print(final_csv["label"].value_counts().sort_index())

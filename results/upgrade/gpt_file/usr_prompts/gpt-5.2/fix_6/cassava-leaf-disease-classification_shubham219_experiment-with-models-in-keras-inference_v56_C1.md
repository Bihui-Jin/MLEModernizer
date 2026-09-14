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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.794

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.10127) has done: 'The timeout is dominated by slow Python-side image loading/augmentation in `ImageDataGenerator.flow_from_dataframe` plus extra overhead from forcing the pure-Python protobuf implementation. I keep the same model/training schedule/augmentations/loss, but switch the input pipeline to an equivalent `tf.data` pipeline (same MobileNetV2 preprocessing and same augmentation parameters) with parallel decode/resize, caching (val/test), prefetching, and deterministic seeding so results stay stable. I also remove the protobuf env override so TensorFlow can use the faster C++ protobuf runtime, and avoid per-row `apply(os.path.isfile)` and `apply(os.path.basename)` loops by using vectorized path operations. These changes reduce input overhead significantly while preserving the algorithm and evaluation semantics.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import math
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
DEBUG = False

BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TRAIN_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

print("TensorFlow:", tf.__version__)
print("Train dir exists:", os.path.isdir(TRAIN_DIR), TRAIN_DIR)
print("Test dir exists:", os.path.isdir(TEST_DIR), TEST_DIR)
print("Train CSV exists:", os.path.isfile(TRAIN_CSV_PATH), TRAIN_CSV_PATH)
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH), SAMPLE_SUB_PATH)

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df["path"] = TRAIN_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)

paths = train_df["path"].to_numpy()
missing_train = (~pd.Index(paths).map(os.path.isfile)).sum()
if missing_train:
    raise FileNotFoundError(
        f"Missing {missing_train} train image files referenced by train.csv"
    )

trn_df, val_df = train_test_split(
    train_df[["path", "label"]].copy(),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"],
)

print("Train/Val sizes:", len(trn_df), len(val_df))
print(
    "Train label distribution:\n",
    trn_df["label"].value_counts().sort_index().to_string(),
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
num_classes = train_df["label"].nunique()
print("num_classes:", num_classes)

_ROT = 15.0 * math.pi / 180.0
_WSHIFT = 0.05
_HSHIFT = 0.05
_ZOOM = 0.10


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


def _apply_preprocess(img):
    return preprocess_input(img)


def _random_rot90_stateless(img, seed):
    k = tf.random.stateless_uniform([], seed=seed, minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(img, k=k)


def _augment_with_path(path, label):
    img = _read_decode_resize(path)
    img = _apply_preprocess(img)

    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)], axis=0)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    seed_r = seed + tf.constant([1, 1], tf.int32)
    img = _random_rot90_stateless(img, seed_r)

    seed_s = seed + tf.constant([2, 2], tf.int32)
    dx = tf.random.stateless_uniform(
        [], seed=seed_s, minval=-_WSHIFT, maxval=_WSHIFT, dtype=tf.float32
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed_s + tf.constant([3, 3], tf.int32),
        minval=-_HSHIFT,
        maxval=_HSHIFT,
        dtype=tf.float32,
    )
    tx = dx * tf.cast(IMG_SIZE[1], tf.float32)
    ty = dy * tf.cast(IMG_SIZE[0], tf.float32)
    img = tf.roll(
        img, shift=[tf.cast(ty, tf.int32), tf.cast(tx, tf.int32)], axis=[0, 1]
    )

    seed_z = seed + tf.constant([4, 4], tf.int32)
    zoom = tf.random.stateless_uniform(
        [], seed=seed_z, minval=1.0 - _ZOOM, maxval=1.0 + _ZOOM, dtype=tf.float32
    )
    new_h = tf.cast(tf.round(zoom * tf.cast(IMG_SIZE[0], tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(zoom * tf.cast(IMG_SIZE[1], tf.float32)), tf.int32)
    img_zoom = tf.image.resize(
        img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.image.resize_with_crop_or_pad(img_zoom, IMG_SIZE[0], IMG_SIZE[1])

    return img, tf.cast(label, tf.int32)


def _val_map(path, label):
    img = _read_decode_resize(path)
    img = _apply_preprocess(img)
    return img, tf.cast(label, tf.int32)


def make_train_ds(df, batch_size):
    paths_ = df["path"].to_numpy()
    labels_ = df["label"].to_numpy().astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths_, labels_))
    ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_augment_with_path, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(df, batch_size):
    paths_ = df["path"].to_numpy()
    labels_ = df["label"].to_numpy().astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths_, labels_))
    ds = ds.map(_val_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(trn_df, BATCH_SIZE)
val_ds = make_val_ds(val_df, BATCH_SIZE)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/821063259.py in <cell line: 0>()
      7 # Keep the original "verify files exist" logic, but avoid slow per-row os.path.isfile loops.
      8 paths = train_df["path"].to_numpy()
----> 9 missing_train = (~pd.Index(paths).map(os.path.isfile)).sum()
     10 if missing_train:
     11     raise FileNotFoundError(

AttributeError: 'Index' object has no attribute 'sum'

## === cell 2
inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
base = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs,
    pooling="avg",
)
base.trainable = False

x = tf.keras.layers.Dropout(0.2)(base.output)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
my_model = tf.keras.Model(inputs=inputs, outputs=outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 3 if not DEBUG else 1
history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

FT_EPOCHS = 2 if not DEBUG else 1
history_ft = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FT_EPOCHS,
    verbose=1,
)

print("Model ready. Output shape:", my_model.output_shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3265788527.py in <cell line: 0>()
----> 1 inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
      2 base = tf.keras.applications.MobileNetV2(
      3     include_top=False,
      4     weights="imagenet",
      5     input_tensor=inputs,

NameError: name 'IMG_SIZE' is not defined

## === cell 3
test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_DIR}")

df_test = pd.DataFrame({"path": test_images})
df_test = df_test.sort_values("path").reset_index(drop=True)

print("Found test images:", len(df_test))


def _test_map(path):
    img = _read_decode_resize(path)
    img = _apply_preprocess(img)
    return img


def make_test_ds(batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices(df_test["path"].to_numpy())
    ds = ds.map(_test_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 4
pred_list = []
for _ in range(1):  # keep original core logic (single pass / "TTA" loop of length 1)
    test_ds = make_test_ds(batch_size=128)
    pred = my_model.predict(test_ds, verbose=1)
    pred_list.append(pred)

pred_test = np.mean(np.stack(pred_list, axis=0), axis=0)

if pred_test.shape[0] != len(df_test):
    raise ValueError(
        f"Prediction rows ({pred_test.shape[0]}) != test rows ({len(df_test)})"
    )

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.rsplit("/", n=1).str[-1]
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]].copy()

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
if final_csv["label"].isna().any():
    missing = final_csv[final_csv["label"].isna()]["image_id"].head(5).tolist()
    raise ValueError(f"Some test image_ids were not predicted (examples): {missing}")
final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1891182677.py in <cell line: 0>()
      1 pred_list = []
      2 for _ in range(1):  # keep original core logic (single pass / "TTA" loop of length 1)
----> 3     test_ds = make_test_ds(batch_size=128)
      4     pred = my_model.predict(test_ds, verbose=1)
      5     pred_list.append(pred)

/tmp/ipykernel_11/3394615638.py in make_test_ds(batch_size)
     17 def make_test_ds(batch_size=128):
     18     ds = tf.data.Dataset.from_tensor_slices(df_test["path"].to_numpy())
---> 19     ds = ds.map(_test_map, num_parallel_calls=AUTOTUNE, deterministic=True)
     20     ds = ds.batch(batch_size, drop_remainder=False)
     21     ds = ds.cache()

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

/tmp/__autograph_generated_filee71_hvtm.py in tf___test_map(path)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(_read_decode_resize), (ag__.ld(path),), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(_apply_preprocess), (ag__.ld(img),), None, fscope)
     12                 try:

NameError: in user code:

    File "/tmp/ipykernel_11/3394615638.py", line 12, in _test_map  *
        img = _read_decode_resize(path)

    NameError: name '_read_decode_resize' is not defined


## === cell 5
print(final_csv.head())
print(final_csv["label"].value_counts().sort_index())
print("submission.csv exists:", os.path.isfile("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.isfile("submission.csv") else None,
)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/899280751.py in <cell line: 0>()
----> 1 print(final_csv.head())
      2 print(final_csv["label"].value_counts().sort_index())
      3 print("submission.csv exists:", os.path.isfile("submission.csv"))
      4 print(
      5     "submission.csv size (bytes):",

NameError: name 'final_csv' is not defined

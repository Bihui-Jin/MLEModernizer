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

0.752190994258084

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

import glob, io, json, shutil
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

BASE = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(BASE, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing dir: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing dir: {TEST_TFREC_DIR}"

print("TensorFlow:", tf.__version__)
print("Train images:", len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))))
print("Test images:", len(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))))
print("Train tfrecords:", len(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))))
print("Test tfrecords:", len(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))))

try:
    tf.config.threading.set_inter_op_parallelism_threads(
        max(2, min(8, os.cpu_count() or 2))
    )
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() or 2))
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "remapping": True,
            "constant_folding": True,
            "shape_optimization": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
        }
    )
except Exception:
    pass

tf.config.run_functions_eagerly(False)
try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.data.experimental.AUTOTUNE  # trigger attribute existence
except Exception:
    pass

df_train = pd.read_csv(TRAIN_CSV)
df_train["path"] = (TRAIN_IMG_DIR.rstrip("/") + "/" + df_train["image_id"]).astype(str)
df_train["label"] = df_train["label"].astype(str)

num_classes = df_train["label"].nunique()
print("num_classes (from train.csv):", num_classes)

sample = pd.read_csv(SAMPLE_SUB)
sample["path"] = (TEST_IMG_DIR.rstrip("/") + "/" + sample["image_id"]).astype(str)

print(df_train.head())
print(sample.head())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32

from sklearn.model_selection import train_test_split

df_tr, df_val = train_test_split(
    df_train,
    test_size=0.15,
    random_state=SEED,
    stratify=df_train["label"],
)

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = False
try:
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
except Exception:
    pass
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_parallelization = True
except Exception:
    pass

class_names = sorted(df_tr["label"].unique().tolist())
num_classes = len(class_names)
label_to_index = {name: i for i, name in enumerate(class_names)}
print("num_classes (from mapping):", num_classes)
print("class_indices:", label_to_index)

train_paths = df_tr["path"].values.astype(str)
train_labels = df_tr["label"].map(label_to_index).values.astype(np.int32)

val_paths = df_val["path"].values.astype(str)
val_labels = df_val["label"].map(label_to_index).values.astype(np.int32)

test_paths = sample["path"].values.astype(str)

SHUFFLE_BUFFER = int(min(len(train_paths), 4096))

train_path_set = tf.constant(train_paths)
val_path_set = tf.constant(val_paths)
train_path_hash = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        train_path_set, tf.ones_like(train_path_set, dtype=tf.int32)
    ),
    default_value=0,
)
val_path_hash = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        val_path_set, tf.ones_like(val_path_set, dtype=tf.int32)
    ),
    default_value=0,
)

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


@tf.function(reduce_retracing=True)
def _path_to_seed_int(path):
    h = tf.strings.to_hash_bucket_strong(
        path, num_buckets=2**31 - 1, key=[12345, 67890]
    )
    return tf.cast(h, tf.int32)


@tf.function(reduce_retracing=True)
def _augment(img, seed_int):
    flip_r = tf.random.stateless_uniform(
        [], seed=[SEED, seed_int], minval=0.0, maxval=1.0
    )
    img = tf.cond(flip_r < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

    tx = tf.random.stateless_uniform(
        [], seed=[SEED + 1, seed_int], minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_SIZE, tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=[SEED + 2, seed_int], minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_SIZE, tf.float32)

    z = tf.random.stateless_uniform(
        [], seed=[SEED + 3, seed_int], minval=0.9, maxval=1.1
    )

    ang = tf.random.stateless_uniform(
        [], seed=[SEED + 4, seed_int], minval=-15.0, maxval=15.0
    ) * (np.pi / 180.0)

    c = tf.cast(IMG_SIZE, tf.float32) / 2.0
    cos_a = tf.math.cos(ang) / z
    sin_a = tf.math.sin(ang) / z

    a0 = cos_a
    a1 = -sin_a
    a2 = c - cos_a * c + sin_a * c - tx
    b0 = sin_a
    b1 = cos_a
    b2 = c - sin_a * c - cos_a * c - ty
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    return img


@tf.function(reduce_retracing=True)
def _to_onehot(label):
    return tf.one_hot(label, depth=num_classes, dtype=tf.float32)


feature_desc = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function(reduce_retracing=True)
def _decode_resize_rescale_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _tfrecord_ds(pattern):
    files = tf.io.gfile.glob(pattern)
    files = tf.sort(tf.constant(files))  # deterministic file order
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).with_options(
        options
    )
    return ds


@tf.function(reduce_retracing=True)
def _parse_tfrec(serialized):
    ex = tf.io.parse_single_example(serialized, feature_desc)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    image_id = ex["image_id"]
    path = tf.strings.join([TRAIN_IMG_DIR, "/", image_id])
    label = tf.cast(ex["label"], tf.int32)
    return path, img, label


@tf.function(reduce_retracing=True)
def _parse_tfrec_test(serialized):
    ex = tf.io.parse_single_example(
        serialized,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_id": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_resize_rescale_from_bytes(ex["image"])
    image_id = ex["image_id"]
    path = tf.strings.join([TEST_IMG_DIR, "/", image_id])
    return path, img


def make_train_ds_from_tfrecords():
    ds = _tfrecord_ds(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.filter(lambda p, img, y: train_path_hash.lookup(p) > 0)

    ds = ds.cache(
        os.path.join(CACHE_DIR, f"train_decoded_{IMG_SIZE}_{len(train_paths)}.cache")
    )

    ds = ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )

    @tf.function(reduce_retracing=True)
    def _pack(path, img, label):
        seed_int = _path_to_seed_int(path)
        img = _augment(img, seed_int)
        y = _to_onehot(label)
        return img, y

    ds = ds.map(_pack, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds_from_tfrecords():
    ds = _tfrecord_ds(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.filter(lambda p, img, y: val_path_hash.lookup(p) > 0)

    ds = ds.cache(
        os.path.join(CACHE_DIR, f"val_decoded_{IMG_SIZE}_{len(val_paths)}.cache")
    )

    @tf.function(reduce_retracing=True)
    def _pack_eval(path, img, y):
        return img, _to_onehot(y)

    ds = ds.map(_pack_eval, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_tfrecords():
    ds = _tfrecord_ds(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
    ds = ds.map(_parse_tfrec_test, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.cache(
        os.path.join(CACHE_DIR, f"test_decoded_{IMG_SIZE}_{len(test_paths)}.cache")
    )

    ds = ds.map(lambda p, img: img, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_gen = make_train_ds_from_tfrecords()
val_gen = make_val_ds_from_tfrecords()
test_gen = make_test_ds_from_tfrecords()

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/284804990.py in <cell line: 0>()
    251 
    252 
--> 253 train_gen = make_train_ds_from_tfrecords()
    254 val_gen = make_val_ds_from_tfrecords()
    255 test_gen = make_test_ds_from_tfrecords()

/tmp/ipykernel_11/284804990.py in make_train_ds_from_tfrecords()
    183 
    184 def make_train_ds_from_tfrecords():
--> 185     ds = _tfrecord_ds(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
    186     ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
    187     ds = ds.apply(tf.data.experimental.ignore_errors())

/tmp/ipykernel_11/284804990.py in _tfrecord_ds(pattern)
    149 def _tfrecord_ds(pattern):
    150     files = tf.io.gfile.glob(pattern)
--> 151     files = tf.sort(tf.constant(files))  # deterministic file order
    152     ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).with_options(
    153         options

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: Value for attr 'T' of string is not in the list of allowed values: bfloat16, half, float, double, int8, int16, int32, int64, complex64, complex128
	; NodeDef: {{node Neg}}; Op<name=Neg; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128]> [Op:Neg] name: 

## === cell 2
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()




## === cell 3
EPOCHS = 6  # keep as-is per original intent

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    verbose=1,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3138136223.py in <cell line: 0>()
      2 
      3 history = model.fit(
----> 4     train_gen,
      5     epochs=EPOCHS,
      6     steps_per_epoch=steps_per_epoch,

NameError: name 'train_gen' is not defined

## === cell 4
pred_test = model.predict(test_gen, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

submission = pd.DataFrame(
    {"image_id": sample["image_id"].values, "label": pred_test_labels}
)

assert submission.shape[0] == sample.shape[0]
assert list(submission.columns) == ["image_id", "label"]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3513613888.py in <cell line: 0>()
----> 1 pred_test = model.predict(test_gen, verbose=1)
      2 pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)
      3 
      4 submission = pd.DataFrame(
      5     {"image_id": sample["image_id"].values, "label": pred_test_labels}

NameError: name 'test_gen' is not defined

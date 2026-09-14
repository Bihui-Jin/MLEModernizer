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

0.876095497129042

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.06278) has done: 'The timeout is dominated by Python-side JPEG loading/resizing/augmentation via `ImageDataGenerator`, plus using the pure-Python protobuf implementation; both create a large input pipeline bottleneck that starves the model. I keep the same model, loss, optimizer, epochs, split, and augmentations, but switch the data pipeline to an equivalent `tf.data` pipeline with parallel decode/resize/augment, caching, prefetch, and deterministic behavior. I also remove the forced Python protobuf implementation (use default C++), and avoid expensive directory scans by using the already-provided `sample_submission.csv` image list for test. These changes are computationally equivalent for training semantics and predictions, but drastically reduce overhead and should fit within 600 seconds.'
- What this solution (achieved 0.05531) has done: 'The timeout is dominated by JPEG decode + resize + expensive projective-transform augmentation on 18.7k large (448×448) images, repeated every epoch. To keep the same model and training semantics while cutting wall time, I (a) avoid redundant work by caching the deterministic “decode+resize+normalize” stage in RAM and (b) fuse augmentation into a single XLA-compiled `tf.function` so the heavy math/transform runs as a compiled graph instead of Python-driven ops per element. I also tighten the `tf.data` pipeline (set `drop_remainder=True` for training, and add `experimental_slack`) to reduce per-step overhead without changing what data is seen. Prediction gets the same decode/resize cache benefit on test.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
    print("Determinism enabled.")
except Exception as e:
    print("Could not enable determinism:", repr(e))

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT enabled.")
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def resolve_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


BASE_DIR = resolve_path(
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset directory."
    )

TRAIN_CSV = resolve_path(
    os.path.join(BASE_DIR, "train.csv"),
    "/kaggle/input/train.csv",
    "../input/train.csv",
)
SAMPLE_SUB = resolve_path(
    os.path.join(BASE_DIR, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "../input/sample_submission.csv",
)
TRAIN_IMG_DIR = resolve_path(
    os.path.join(BASE_DIR, "train_images"),
    "/kaggle/input/train_images",
    "../input/train_images",
)
TEST_IMG_DIR = resolve_path(
    os.path.join(BASE_DIR, "test_images"),
    "/kaggle/input/test_images",
    "../input/test_images",
)

TRAIN_TFREC_DIR = resolve_path(
    os.path.join(BASE_DIR, "train_tfrecords"),
    "/kaggle/input/train_tfrecords",
    "../input/train_tfrecords",
)
TEST_TFREC_DIR = resolve_path(
    os.path.join(BASE_DIR, "test_tfrecords"),
    "/kaggle/input/test_tfrecords",
    "../input/test_tfrecords",
)

for req, p in [
    ("TRAIN_CSV", TRAIN_CSV),
    ("SAMPLE_SUB", SAMPLE_SUB),
    ("TRAIN_IMG_DIR", TRAIN_IMG_DIR),
    ("TEST_IMG_DIR", TEST_IMG_DIR),
    ("TRAIN_TFREC_DIR", TRAIN_TFREC_DIR),
    ("TEST_TFREC_DIR", TEST_TFREC_DIR),
]:
    if p is None:
        raise FileNotFoundError(f"Missing required path: {req}")

print("Using BASE_DIR:", BASE_DIR)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)
print("TRAIN_TFREC_DIR:", TRAIN_TFREC_DIR)
print("TEST_TFREC_DIR:", TEST_TFREC_DIR)

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError("train.csv must contain columns: image_id, label")
if not {"image_id", "label"}.issubset(sample_df.columns):
    raise ValueError("sample_submission.csv must contain columns: image_id, label")

num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)




## === cell 2
IMG_SIZE = (448, 448)
BATCH_SIZE = 16
EPOCHS = 2  # keep identical

train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].copy()
trn_df = train_df.iloc[val_size:].copy()

IMG_H, IMG_W = IMG_SIZE

_PI_OVER_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)
_SEED42 = tf.constant(42, dtype=tf.int32)
_SEED43 = tf.constant(43, dtype=tf.int32)
_SEED44 = tf.constant(44, dtype=tf.int32)
_SEED45 = tf.constant(45, dtype=tf.int32)
_SEED46 = tf.constant(46, dtype=tf.int32)
_IMG_W_F = tf.constant(float(IMG_W), dtype=tf.float32)
_IMG_H_F = tf.constant(float(IMG_H), dtype=tf.float32)
_CX = (_IMG_W_F - 1.0) / 2.0
_CY = (_IMG_H_F - 1.0) / 2.0
_TRAIN_DIR_PREFIX = tf.constant(TRAIN_IMG_DIR + os.sep, dtype=tf.string)
_TEST_DIR_PREFIX = tf.constant(TEST_IMG_DIR + os.sep, dtype=tf.string)

_trn_keys = tf.constant(trn_df["image_id"].astype(str).values)
_val_keys = tf.constant(val_df["image_id"].astype(str).values)
_trn_vals = tf.ones([tf.shape(_trn_keys)[0]], dtype=tf.bool)
_val_vals = tf.ones([tf.shape(_val_keys)[0]], dtype=tf.bool)

_trn_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(_trn_keys, _trn_vals), default_value=False
)
_val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(_val_keys, _val_vals), default_value=False
)

_FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}
_FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


def _list_tfrecs(tfrecord_dir: str):
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .tfrec files found in: {tfrecord_dir}")
    return files


train_tfrecs = _list_tfrecs(TRAIN_TFREC_DIR)
test_tfrecs = _list_tfrecs(TEST_TFREC_DIR)


def _resize_and_norm(img_uint8):
    img = tf.image.resize(
        img_uint8, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def _augment(img, idx):
    img = tf.image.stateless_random_flip_left_right(img, seed=tf.stack([_SEED42, idx]))

    angle = tf.random.stateless_uniform(
        [], seed=tf.stack([_SEED43, idx]), minval=-10.0, maxval=10.0, dtype=tf.float32
    )
    angle = angle * _PI_OVER_180

    tx = (
        tf.random.stateless_uniform(
            [],
            seed=tf.stack([_SEED44, idx]),
            minval=-0.05,
            maxval=0.05,
            dtype=tf.float32,
        )
        * _IMG_W_F
    )
    ty = (
        tf.random.stateless_uniform(
            [],
            seed=tf.stack([_SEED45, idx]),
            minval=-0.05,
            maxval=0.05,
            dtype=tf.float32,
        )
        * _IMG_H_F
    )

    scale = tf.random.stateless_uniform(
        [], seed=tf.stack([_SEED46, idx]), minval=0.9, maxval=1.1, dtype=tf.float32
    )

    cos_a = tf.cos(angle) / scale
    sin_a = tf.sin(angle) / scale

    a0 = cos_a
    a1 = -sin_a
    a2 = (1.0 - cos_a) * _CX + sin_a * _CY - tx
    b0 = sin_a
    b1 = cos_a
    b2 = (1.0 - cos_a) * _CY - sin_a * _CX - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=[IMG_H, IMG_W],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)
    return img


@tf.function(input_signature=(tf.TensorSpec([], tf.string),))
def _parse_train_minimal(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESC_TRAIN)
    return ex["image_id"], serialized


@tf.function(
    input_signature=(
        tf.TensorSpec([], tf.string),
        tf.TensorSpec([], tf.string),
        tf.TensorSpec([], tf.int64),
    )
)
def _parse_decode_resize_train_from_ex(serialized, image_id, idx):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESC_TRAIN)
    img_uint8 = tf.image.decode_jpeg(ex["image"], channels=3)  # uint8
    img = _resize_and_norm(img_uint8)
    img = _augment(img, tf.cast(idx, tf.int32))
    y = tf.cast(ex["target"], tf.int32)
    return img, y


@tf.function(input_signature=(tf.TensorSpec([], tf.string),))
def _parse_decode_resize_eval(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESC_TRAIN)
    img_uint8 = tf.image.decode_jpeg(ex["image"], channels=3)  # uint8
    img = _resize_and_norm(img_uint8)
    y = tf.cast(ex["target"], tf.int32)
    return img, y


@tf.function(input_signature=(tf.TensorSpec([], tf.string),))
def _parse_decode_resize_test(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESC_TEST)
    img_uint8 = tf.image.decode_jpeg(ex["image"], channels=3)  # uint8
    img = _resize_and_norm(img_uint8)
    return img


def _dataset_options():
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_fusion = True
    options.experimental_slack = True
    cpu = os.cpu_count() or 2
    options.threading.private_threadpool_size = max(4, min(16, cpu))
    return options


_CPU = os.cpu_count() or 2
_CYCLE = int(min(len(train_tfrecs), max(4, min(16, _CPU))))
_BLOCK = 16


def make_train_dataset_from_tfrecs(tfrecs):
    ds_files = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecs))
    ds_files = ds_files.shuffle(len(tfrecs), seed=42, reshuffle_each_iteration=True)

    ds = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=1),
        cycle_length=_CYCLE,
        block_length=_BLOCK,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.shuffle(buffer_size=4096, seed=42, reshuffle_each_iteration=True)

    ds = ds.with_options(_dataset_options())
    ds = ds.map(_parse_train_minimal, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(lambda image_id, serialized: _trn_table.lookup(image_id))

    ds = ds.enumerate(start=0)  # (idx, (image_id, serialized))
    ds = ds.map(
        lambda idx, pair: _parse_decode_resize_train_from_ex(
            pair[1], pair[0], tf.cast(idx, tf.int64)
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_dataset_from_tfrecs(tfrecs):
    ds_files = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecs))
    ds = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=1),
        cycle_length=_CYCLE,
        block_length=_BLOCK,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.with_options(_dataset_options())

    ds = ds.map(_parse_train_minimal, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(lambda image_id, serialized: _val_table.lookup(image_id))
    ds = ds.map(
        lambda image_id, serialized: _parse_decode_resize_eval(serialized),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_dataset_from_tfrecs(train_tfrecs)
val_ds = make_val_dataset_from_tfrecs(train_tfrecs).cache()

steps_per_epoch = int(np.ceil(len(trn_df) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

inputs = tf.keras.Input(
    shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32, name="image"
)
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model_v1 = tf.keras.Model(inputs=inputs, outputs=outputs, name="cassava_baseline_cnn")

model_v1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model_v1.summary()

history = model_v1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1864914597.py in <cell line: 0>()
    273 model_v1.summary()
    274 
--> 275 history = model_v1.fit(
    276     train_ds,
    277     validation_data=val_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:5 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map::Zip[1]::Filter::Map: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_2382]

## === cell 3
test_df = sample_df[["image_id"]].copy()


def make_test_dataset_from_tfrecs(tfrecs):
    ds_files = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecs))
    ds = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=1),
        cycle_length=int(min(len(tfrecs), max(4, min(16, (os.cpu_count() or 2))))),
        block_length=_BLOCK,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.with_options(_dataset_options())
    ds = ds.map(
        _parse_decode_resize_test, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(64, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_dataset_from_tfrecs(test_tfrecs)
print("Test samples (from sample_submission.csv):", len(test_df))

pred_v1 = model_v1.predict(test_ds, verbose=1)
pred_v1 = np.asarray(pred_v1)
print("Pred shape:", pred_v1.shape)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/669400.py in <cell line: 0>()
     23 print("Test samples (from sample_submission.csv):", len(test_df))
     24 
---> 25 pred_v1 = model_v1.predict(test_ds, verbose=1)
     26 pred_v1 = np.asarray(pred_v1)
     27 print("Pred shape:", pred_v1.shape)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Feature: image_id (data type: string) is required but could not be found.
	 [[{{function_node __inference__parse_decode_resize_test_2431}}{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 4
predicted_class_indices_v1 = np.argmax(pred_v1, axis=1).astype(int)

if len(predicted_class_indices_v1) != len(test_df):
    raise RuntimeError(
        f"Predictions length ({len(predicted_class_indices_v1)}) does not match "
        f"sample_submission length ({len(test_df)})."
    )

results_v1 = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices_v1}
)

results_v1 = sample_df[["image_id"]].merge(results_v1, on="image_id", how="left")
if results_v1["label"].isna().any():
    missing = results_v1.loc[results_v1["label"].isna(), "image_id"].head(5).tolist()
    raise RuntimeError(
        f"Some test image_ids were not predicted (showing up to 5): {missing}. "
        "Check dataset paths/filenames alignment."
    )
results_v1["label"] = results_v1["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
results_v1.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(results_v1.head())
print("Submission shape:", results_v1.shape)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3920657717.py in <cell line: 0>()
----> 1 predicted_class_indices_v1 = np.argmax(pred_v1, axis=1).astype(int)
      2 
      3 if len(predicted_class_indices_v1) != len(test_df):
      4     raise RuntimeError(
      5         f"Predictions length ({len(predicted_class_indices_v1)}) does not match "

NameError: name 'pred_v1' is not defined

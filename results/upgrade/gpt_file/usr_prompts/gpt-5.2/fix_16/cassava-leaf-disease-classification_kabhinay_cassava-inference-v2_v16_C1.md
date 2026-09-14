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

2.7

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

0.8828951344817165

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from __future__ import print_function
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
try:
    tf.random.set_seed(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow:", getattr(tf, "__version__", "unknown"))
try:
    print("Eager:", tf.executing_eagerly())
except Exception:
    print("Eager: unknown")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _find_existing_dir(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def _find_existing_file(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


DATA_DIR = _find_existing_dir(
    [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
)
if DATA_DIR is None:
    raise OSError(
        "Could not locate cassava-leaf-disease-classification input directory."
    )

TRAIN_DIR = _find_existing_dir(
    [
        os.path.join(DATA_DIR, "train_images"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
    ]
)
TEST_DIR = _find_existing_dir(
    [
        os.path.join(DATA_DIR, "test_images"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    ]
)

TRAIN_TFREC_DIR = _find_existing_dir(
    [
        os.path.join(DATA_DIR, "train_tfrecords"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_tfrecords",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_tfrecords",
    ]
)
TEST_TFREC_DIR = _find_existing_dir(
    [
        os.path.join(DATA_DIR, "test_tfrecords"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_tfrecords",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_tfrecords",
    ]
)

train_csv_path = _find_existing_file(
    [
        os.path.join(DATA_DIR, "train.csv"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
    ]
)
sample_sub_path = _find_existing_file(
    [
        os.path.join(DATA_DIR, "sample_submission.csv"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    ]
)

if TRAIN_DIR is None or TEST_DIR is None:
    raise OSError(
        "train_images/test_images directories not found in expected locations."
    )
if train_csv_path is None or sample_sub_path is None:
    raise OSError("train.csv/sample_submission.csv not found in expected locations.")

print("DATA_DIR:", DATA_DIR)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("TRAIN_TFREC_DIR:", TRAIN_TFREC_DIR)
print("TEST_TFREC_DIR:", TEST_TFREC_DIR)
print("train.csv:", train_csv_path)
print("sample_submission.csv:", sample_sub_path)

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)
test_df = sample_sub[["image_id"]].copy()

assert "image_id" in train_df.columns and "label" in train_df.columns
assert "image_id" in test_df.columns
print("Train rows:", len(train_df), "Test rows:", len(test_df))
print("Train label distribution:\n", train_df["label"].value_counts().sort_index())



## === cell 2
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 5

from sklearn.model_selection import train_test_split

train_df["label"] = train_df["label"].astype(str)

tr_df, va_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=0,
    stratify=train_df["label"].values,
)

AUTOTUNE = getattr(tf.data, "AUTOTUNE", None)
if AUTOTUNE is None:
    AUTOTUNE = tf.data.experimental.AUTOTUNE

_classes = sorted(tr_df["label"].unique().tolist())
class_indices = dict((c, i) for i, c in enumerate(_classes))
print("class_indices:", class_indices)

tr_labels_str = tr_df["label"].values
va_labels_str = va_df["label"].values

tr_labels_int = pd.Categorical(
    tr_labels_str, categories=_classes, ordered=True
).codes.astype(np.int32)
va_labels_int = pd.Categorical(
    va_labels_str, categories=_classes, ordered=True
).codes.astype(np.int32)

BASE_SEED = tf.constant([0, 0], dtype=tf.int32)
_ROT_FACTOR = 15.0 / 360.0

_aug_layers = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=_ROT_FACTOR, fill_mode="reflect", seed=0),
        tf.keras.layers.RandomTranslation(
            height_factor=0.10, width_factor=0.10, fill_mode="reflect", seed=0
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10),
            fill_mode="reflect",
            seed=0,
        ),
    ],
    name="aug",
)


@tf.function(reduce_retracing=True)
def _center_crop_resize_from_jpeg_bytes(img_bytes):
    shape = tf.image.extract_jpeg_shape(img_bytes)  # [h, w, 3]
    h = tf.cast(shape[0], tf.int32)
    w = tf.cast(shape[1], tf.int32)

    crop_h = tf.minimum(h, tf.cast(IMG_SIZE * 5 // 4, tf.int32))
    crop_w = tf.minimum(w, tf.cast(IMG_SIZE * 5 // 4, tf.int32))
    offset_y = tf.maximum((h - crop_h) // 2, 0)
    offset_x = tf.maximum((w - crop_w) // 2, 0)
    crop_window = tf.stack([offset_y, offset_x, crop_h, crop_w])

    img = tf.image.decode_and_crop_jpeg(img_bytes, crop_window=crop_window, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES_IMAGE_ONLY = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


def _list_tfrec_files(tfrec_dir, prefix):
    if tfrec_dir is None:
        return []
    files = tf.io.gfile.glob(os.path.join(tfrec_dir, prefix + "*.tfrec"))
    return sorted(files)


train_tfrec_files = _list_tfrec_files(TRAIN_TFREC_DIR, "ld_train")
test_tfrec_files = _list_tfrec_files(TEST_TFREC_DIR, "ld_test")

if not train_tfrec_files or not test_tfrec_files:
    raise OSError("TFRecord files not found; expected train_tfrecords/test_tfrecords.")

_ds_options = tf.data.Options()
try:
    _ds_options.experimental_deterministic = True
except Exception:
    pass
try:
    _ds_options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    _ds_options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass
try:
    _ds_options.experimental_optimization.parallel_batch = True
except Exception:
    pass

SHUFFLE_BUFFER = int(min(len(tr_df), 4096))


def _make_train_ds_from_tfrec(files):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_ds_options)

    def _parse(example):
        ex = tf.io.parse_single_example(example, _TRAIN_FEATURES)
        img = _center_crop_resize_from_jpeg_bytes(ex["image"])
        return img, ex["image_id"], tf.cast(ex["label"], tf.int32)

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)

    tr_ids = tf.constant(tr_df["image_id"].values.astype("S"))
    tr_set = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tr_ids, tf.ones([tf.shape(tr_ids)[0]], tf.int32)
        ),
        default_value=0,
    )

    def _is_in_train(img, image_id, y):
        return tr_set.lookup(image_id) > 0

    ds = ds.filter(_is_in_train)

    ds = ds.map(
        lambda img, image_id, y: (
            img,
            tf.one_hot(tf.cast(y, tf.int32), depth=NUM_CLASSES, dtype=tf.float32),
        ),
        num_parallel_calls=AUTOTUNE,
    )

    ds = ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=0, reshuffle_each_iteration=True
    ).repeat()

    ds = ds.enumerate(start=0)

    def _attach_seed(idx, img_y):
        img, y = img_y
        idx32 = tf.cast(idx, tf.int32)
        seed = BASE_SEED + tf.stack([idx32, idx32])
        return (img, seed), y

    ds = ds.map(_attach_seed, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds_from_tfrec(files):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_ds_options)

    def _parse(example):
        ex = tf.io.parse_single_example(example, _TRAIN_FEATURES)
        img = _center_crop_resize_from_jpeg_bytes(ex["image"])
        y = tf.one_hot(
            tf.cast(ex["label"], tf.int32), depth=NUM_CLASSES, dtype=tf.float32
        )
        return img, ex["image_id"], y

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)

    va_ids = tf.constant(va_df["image_id"].values.astype("S"))
    va_set = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            va_ids, tf.ones([tf.shape(va_ids)[0]], tf.int32)
        ),
        default_value=0,
    )

    def _is_in_val(img, image_id, y):
        return va_set.lookup(image_id) > 0

    ds = ds.filter(_is_in_val)
    ds = ds.map(
        lambda img, image_id, y: ((img, tf.zeros([2], tf.int32)), y),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_test_ds_from_tfrec(files, image_id_series):
    image_ids = tf.constant(image_id_series.values.astype("S"))

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_ds_options)

    def _parse(example):
        ex = tf.io.parse_single_example(example, _TEST_FEATURES_IMAGE_ONLY)
        img = _center_crop_resize_from_jpeg_bytes(ex["image"])
        return img

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)
    ds = ds.enumerate(start=0)

    def _attach_id(i, img):
        image_id = image_ids[tf.cast(i, tf.int32)]
        return img, image_id

    ds = ds.map(_attach_id, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_dataset = _make_train_ds_from_tfrec(train_tfrec_files)
val_dataset = _make_val_ds_from_tfrec(train_tfrec_files)
test_dataset = _make_test_ds_from_tfrec(test_tfrec_files, test_df["image_id"])

img_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img")
seed_in = keras.Input(shape=(2,), dtype=tf.int32, name="seed")


def _apply_aug_in_model(img, seed):
    x = _aug_layers(img, training=True)

    def _flip_one(args):
        xi, si = args
        return tf.image.stateless_random_flip_left_right(
            xi, seed=si + tf.constant([2, 2], tf.int32)
        )

    x = tf.map_fn(
        _flip_one, (x, seed), fn_output_signature=tf.float32, parallel_iterations=32
    )
    return x


augmented = keras.layers.Lambda(
    lambda t: _apply_aug_in_model(t[0], t[1]),
    name="augment",
    output_shape=(IMG_SIZE, IMG_SIZE, 3),
)([img_in, seed_in])

x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(augmented)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model([img_in, seed_in], outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

steps_per_epoch = int(np.ceil(float(len(tr_df)) / float(BATCH_SIZE)))
val_steps = int(np.ceil(float(len(va_df)) / float(BATCH_SIZE)))

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=val_steps,
    verbose=1,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3099969376.py in <cell line: 0>()
    280 val_steps = int(np.ceil(float(len(va_df)) / float(BATCH_SIZE)))
    281 
--> 282 history = model.fit(
    283     train_dataset,
    284     epochs=EPOCHS,

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::Zip[1]::ShuffleAndRepeat::ParallelMapV2::Filter::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_3000]

## === cell 3
zeros_seed = tf.constant([0, 0], tf.int32)

test_dataset_for_pred = test_dataset.map(
    lambda img, image_id: (
        img,
        tf.repeat(tf.expand_dims(zeros_seed, 0), tf.shape(img)[0], axis=0),
    ),
    num_parallel_calls=AUTOTUNE,
)

pred = model.predict(test_dataset_for_pred, verbose=1)
pred = np.asarray(pred)

if pred.ndim != 2 or pred.shape[1] != NUM_CLASSES:
    raise ValueError("Unexpected prediction shape: {}".format(pred.shape))

predicted_class_indices = np.argmax(pred, axis=1).astype(int)
predicted_class_indices = predicted_class_indices[: len(test_df)]

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices}
)
submission["label"] = submission["label"].astype(int)
submission = submission[["image_id", "label"]]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Cols:", submission.columns.tolist())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1405941425.py in <cell line: 0>()
      9 )
     10 
---> 11 pred = model.predict(test_dataset_for_pred, verbose=1)
     12 pred = np.asarray(pred)
     13 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    158     inputs = tree.flatten(inputs)
    159     if len(inputs) != len(input_spec):
--> 160         raise ValueError(
    161             f'Layer "{layer_name}" expects {len(input_spec)} input(s),'
    162             f" but it received {len(inputs)} input tensors. "

ValueError: Layer "functional" expects 2 input(s), but it received 1 input tensors. Inputs received: [<tf.Tensor 'data:0' shape=(32, 224, 224, 3) dtype=float32>]

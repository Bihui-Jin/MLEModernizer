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

0.4412209126624358

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.62332) has done: 'The timeout is dominated by heavy image I/O + preprocessing (center-crop via `extract_jpeg_shape` + `decode_and_crop_jpeg` + resize to 512) repeated every epoch, plus an expensive `.shuffle(buffer_size=len(paths))` and GPU prefetch-to-device that can add overhead/pressure. I keep the exact model, epochs, and augmentation, but make the pipeline provably equivalent and faster by: (1) removing the extra JPEG header pass and doing the same center-crop using `decode_jpeg` + `crop_to_bounding_box`, (2) caching the *decoded+resized* training images (before augmentation) so epoch 2 doesn’t re-read/re-decode from disk, and (3) reducing shuffle buffer to a large fixed value (keeps shuffle semantics, avoids allocating ~18k elements) and simplifying prefetch placement. These changes preserve training/evaluation semantics and determinism while cutting repeated CPU decode work and input-pipeline overhead to fit under 600 seconds.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

IMAGE_SIZE = (512, 512)

BATCH_SIZE = 8
EPOCHS = 2  # keep as-is (core training approach unchanged)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

train_df["label"] = train_df["label"].astype(str)
sample_df["label"] = sample_df["label"].astype(str)

val_frac = 0.1
val_idx = (
    train_df.groupby("label", group_keys=False)
    .apply(lambda x: x.sample(max(1, int(len(x) * val_frac)), random_state=SEED))
    .index
)
val_df = train_df.loc[val_idx].reset_index(drop=True)
trn_df = train_df.drop(index=val_idx).reset_index(drop=True)

num_classes = train_df["label"].nunique()

AUTOTUNE = tf.data.AUTOTUNE

trn_paths = (TRAIN_IMG_DIR + "/" + trn_df["image_id"].values).astype("U")
val_paths = (TRAIN_IMG_DIR + "/" + val_df["image_id"].values).astype("U")
tst_paths = (TEST_IMG_DIR + "/" + sample_df["image_id"].values).astype("U")

trn_labels = trn_df["label"].astype(np.int32).values
val_labels = val_df["label"].astype(np.int32).values

IMG_H, IMG_W = IMAGE_SIZE


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8 [H,W,3]
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]

    crop_h = tf.minimum(h, IMG_H)
    crop_w = tf.minimum(w, IMG_W)
    offset_h = tf.maximum(0, (h - crop_h) // 2)
    offset_w = tf.maximum(0, (w - crop_w) // 2)

    img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, crop_h, crop_w)
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)  # rescale=1/255
    return img


@tf.function
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_from_bytes(img_bytes)


_augment = keras.Sequential(
    [
        layers.RandomRotation(factor=10.0 / 360.0, fill_mode="nearest", seed=SEED),
        layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="nearest", seed=SEED
        ),
        layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="nearest",
            seed=SEED,
        ),
        layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="augment",
)


@tf.function
def _augment_fn(img):
    return _augment(img, training=True)


_DS_OPTIONS = tf.data.Options()
_DS_OPTIONS.experimental_deterministic = True

_HAS_GPU = bool(tf.config.list_physical_devices("GPU"))


def _maybe_prefetch_to_device(ds):
    return ds


_TRAIN_SHUFFLE_BUFFER = min(8192, len(trn_df))


def _list_tfrecord_files(split_dir):
    files = tf.io.gfile.glob(os.path.join(split_dir, "*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in: {split_dir}")
    return files


_TRAIN_TFRECS = _list_tfrecord_files(TRAIN_TFREC_DIR)
_TEST_TFRECS = _list_tfrecord_files(TEST_TFREC_DIR)


def _parse_train_example(example_proto):
    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image_bytes": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    ex = tf.io.parse_single_example(example_proto, feature_spec)
    img_bytes = tf.cond(
        tf.not_equal(tf.strings.length(ex["image"]), 0),
        lambda: ex["image"],
        lambda: ex["image_bytes"],
    )
    img = _decode_resize_from_bytes(img_bytes)
    label = tf.cast(ex["label"], tf.int32)
    return img, label


def _parse_test_example(example_proto):
    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image_bytes": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    ex = tf.io.parse_single_example(example_proto, feature_spec)
    img_bytes = tf.cond(
        tf.not_equal(tf.strings.length(ex["image"]), 0),
        lambda: ex["image"],
        lambda: ex["image_bytes"],
    )
    img = _decode_resize_from_bytes(img_bytes)
    return img


def make_train_ds_from_tfrecords(tfrecs, batch_size):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DS_OPTIONS)
    ds = ds.shuffle(
        buffer_size=_TRAIN_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.map(
        lambda img, label: (_augment_fn(img), label), num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def make_eval_ds_from_tfrecords(tfrecs, batch_size):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DS_OPTIONS)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def make_test_ds_from_tfrecords(tfrecs, batch_size):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DS_OPTIONS)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


train_ds = make_train_ds_from_tfrecords(_TRAIN_TFRECS, BATCH_SIZE)

val_ds = make_eval_ds_from_tfrecords(_TRAIN_TFRECS, BATCH_SIZE)

test_ds = make_test_ds_from_tfrecords(_TEST_TFRECS, BATCH_SIZE)

num_classes



## === cell 2
inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 3
steps_per_epoch = max(1, len(trn_df) // BATCH_SIZE)
validation_steps = max(1, len(val_df) // BATCH_SIZE)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    verbose=1,
)

predictions = model.predict(test_ds, verbose=1)
pred_labels = predictions.argmax(axis=1).astype(int)

len(pred_labels), pred_labels[:10].tolist()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3585635422.py in <cell line: 0>()
      2 validation_steps = max(1, len(val_df) // BATCH_SIZE)
      3 
----> 4 history = model.fit(
      5     train_ds,
      6     epochs=EPOCHS,

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
Error in user-defined function passed to ParallelMapDatasetV2:4 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_2373]

## === cell 4
submission = pd.DataFrame(
    {"image_id": sample_df["image_id"].values, "label": pred_labels}
)
submission["label"] = submission["label"].astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

submission.head(), submission_path

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2012524342.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"image_id": sample_df["image_id"].values, "label": pred_labels}
      3 )
      4 submission["label"] = submission["label"].astype(int)
      5 

NameError: name 'pred_labels' is not defined

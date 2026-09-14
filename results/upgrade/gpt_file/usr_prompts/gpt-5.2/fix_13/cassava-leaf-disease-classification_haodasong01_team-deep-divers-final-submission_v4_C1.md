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

0.7985796313085525

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.keras.backend.clear_session()
tf.config.run_functions_eagerly(False)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

print("TF version:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train image dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))

from tensorflow.keras import layers, models



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

assert {"image_id", "label"}.issubset(train_df.columns)
assert {"image_id", "label"}.issubset(sample_sub.columns)

train_split, val_split = tf.keras.utils.split_dataset(
    np.arange(len(train_df)),
    left_size=0.9,
    shuffle=True,
    seed=SEED,
)

train_idx = np.array(list(train_split))
val_idx = np.array(list(val_split))
train_df_split = train_df.iloc[train_idx].copy()
val_df_split = train_df.iloc[val_idx].copy()

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 3  # unchanged

AUTOTUNE = tf.data.AUTOTUNE


@tf.function(jit_compile=True)
def _read_jpeg_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1./255
    img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


@tf.function(jit_compile=True)
def _rotate_img(img, angle_rad):
    angle_rad = tf.cast(angle_rad, tf.float32)

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.cos(angle_rad)
    sin_a = tf.sin(angle_rad)

    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = cy - sin_a * cx - cos_a * cy

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)
    transform = tf.reshape(transform, [1, 8])

    img4 = tf.expand_dims(img, axis=0)

    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    rotated = tf.squeeze(rotated, axis=0)
    rotated = tf.ensure_shape(rotated, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return rotated


@tf.function(jit_compile=True)
def _augment(img, seed_pair):
    seed_pair = tf.cast(seed_pair, tf.int32)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)

    angle = tf.random.stateless_uniform(
        [], seed=seed_pair + tf.constant([1, 0], tf.int32), minval=-10.0, maxval=10.0
    ) * (np.pi / 180.0)
    img = _rotate_img(img, angle)

    max_dx = tf.cast(tf.round(0.05 * tf.cast(IMG_SIZE[1], tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(IMG_SIZE[0], tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([2, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([3, 0], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    scale = tf.random.stateless_uniform(
        [], seed=seed_pair + tf.constant([4, 0], tf.int32), minval=0.9, maxval=1.1
    )
    h = tf.cast(IMG_SIZE[0], tf.float32)
    w = tf.cast(IMG_SIZE[1], tf.float32)
    new_h = tf.cast(tf.round(h / scale), tf.int32)
    new_w = tf.cast(tf.round(w / scale), tf.int32)

    def zoom_in():
        crop = tf.image.stateless_random_crop(
            img,
            size=[new_h, new_w, 3],
            seed=seed_pair + tf.constant([5, 0], tf.int32),
        )
        out = tf.image.resize(
            crop, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        return tf.ensure_shape(out, (IMG_SIZE[0], IMG_SIZE[1], 3))

    def zoom_out():
        pad_h = tf.maximum(new_h - IMG_SIZE[0], 0)
        pad_w = tf.maximum(new_w - IMG_SIZE[1], 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = tf.pad(
            img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="REFLECT"
        )
        crop = tf.image.stateless_random_crop(
            padded,
            size=[new_h, new_w, 3],
            seed=seed_pair + tf.constant([6, 0], tf.int32),
        )
        out = tf.image.resize(
            crop, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        return tf.ensure_shape(out, (IMG_SIZE[0], IMG_SIZE[1], 3))

    img = tf.cond(scale >= 1.0, zoom_in, zoom_out)
    return img


_SHUFFLE_BUFFER = min(len(train_df_split), 4096)

_BASE_DS_OPTS = tf.data.Options()
_BASE_DS_OPTS.experimental_optimization.map_parallelization = True
_BASE_DS_OPTS.experimental_optimization.parallel_batch = True
_BASE_DS_OPTS.experimental_optimization.map_fusion = True
_BASE_DS_OPTS.experimental_optimization.apply_default_optimizations = True
_BASE_DS_OPTS.deterministic = True

_SEED_TENSOR = tf.constant(SEED, dtype=tf.int32)


@tf.function(jit_compile=True)
def _decode_label(path, label):
    img = _read_jpeg_resize(path)
    return img, tf.cast(label, tf.int32)


@tf.function(jit_compile=True)
def _one_hot(label):
    return tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)


def _make_train_ds(df_split):
    paths = (TRAIN_IMG_DIR + "/" + df_split["image_id"].astype(str)).to_numpy()
    labels = df_split["label"].astype(np.int32).to_numpy()
    idx = np.arange(len(paths), dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((idx, paths, labels))
    ds = ds.shuffle(
        buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.with_options(_BASE_DS_OPTS)

    @tf.function(jit_compile=True)
    def _decode_with_idx(i, p, y):
        img, y2 = _decode_label(p, y)
        return i, img, y2

    ds = ds.map(_decode_with_idx, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()

    @tf.function(jit_compile=True)
    def _aug_then_onehot(i, img, label):
        seed_pair = tf.stack([_SEED_TENSOR, tf.cast(i, tf.int32)], axis=0)
        img2 = _augment(img, seed_pair)
        return img2, _one_hot(label)

    ds = ds.map(_aug_then_onehot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


def _make_val_ds(df_split):
    paths = (TRAIN_IMG_DIR + "/" + df_split["image_id"].astype(str)).to_numpy()
    labels = df_split["label"].astype(np.int32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_BASE_DS_OPTS)
    ds = ds.map(_decode_label, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()

    @tf.function(jit_compile=True)
    def _to_onehot(img, label):
        return img, _one_hot(label)

    ds = ds.map(_to_onehot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


train_ds = _make_train_ds(train_df_split)
val_ds = _make_val_ds(val_df_split)

model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

steps_per_epoch = int(np.ceil(len(train_df_split) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_df_split) / BATCH_SIZE))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 2
test_df = sample_sub[["image_id"]].copy()

test_paths = (TEST_IMG_DIR + "/" + test_df["image_id"].astype(str)).to_numpy()
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


@tf.function(jit_compile=True)
def _map_test(path):
    img = _read_jpeg_resize(path)
    return img


test_ds = test_ds.with_options(_BASE_DS_OPTS)
test_ds = test_ds.map(_map_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())

predict_steps = int(np.ceil(len(test_df) / BATCH_SIZE))

probs = model.predict(test_ds, steps=predict_steps, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

if len(preds) != len(test_df):
    preds = preds[: len(test_df)]

my_submission = pd.DataFrame({"image_id": test_df["image_id"].values, "label": preds})
out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(my_submission.head())
print("Rows:", len(my_submission), "Expected:", len(sample_sub))
assert len(my_submission) == len(sample_sub)
assert list(my_submission.columns) == ["image_id", "label"]
assert out_path.endswith(".csv")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
/tmp/ipykernel_11/1181603487.py in <cell line: 0>()
     20 predict_steps = int(np.ceil(len(test_df) / BATCH_SIZE))
     21 
---> 22 probs = model.predict(test_ds, steps=predict_steps, verbose=1)
     23 preds = np.argmax(probs, axis=1).astype(int)
     24 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py in predict(self, x, batch_size, verbose, steps, callbacks)
    567         callbacks.on_predict_end()
    568         outputs = tree.map_structure_up_to(
--> 569             batch_outputs, potentially_ragged_concat, outputs
    570         )
    571         return tree.map_structure(convert_to_np_if_not_ragged, outputs)

UnboundLocalError: cannot access local variable 'batch_outputs' where it is not associated with a value

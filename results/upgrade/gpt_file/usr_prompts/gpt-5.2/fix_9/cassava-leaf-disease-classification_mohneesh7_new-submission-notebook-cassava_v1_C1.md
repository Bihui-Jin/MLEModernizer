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

0.6128739800543971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import warnings

warnings.filterwarnings("ignore")

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick good defaults
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

print("TensorFlow:", tf.__version__)
print("Train CSV rows:", sum(1 for _ in open(TRAIN_CSV)) - 1)
print("Train images:", len(os.listdir(TRAIN_IMG_DIR)))
print("Test images:", len(os.listdir(TEST_IMG_DIR)))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

train_df["image_path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)

train_df["label"] = train_df["label"].astype(str)

image_size = 224
batch_size = 32
num_classes = train_df["label"].nunique()

AUTOTUNE = tf.data.AUTOTUNE
IMG_SHAPE = (image_size, image_size)

val_frac = 0.1
n_total = len(train_df)
n_val = int(round(n_total * val_frac))
n_train = n_total - n_val

train_df_train = train_df.iloc[:n_train].reset_index(drop=True)
train_df_val = train_df.iloc[n_train:].reset_index(drop=True)

class_names = sorted(train_df["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(class_names)}
idx_to_label = {i: int(c) for c, i in class_to_index.items()}
print("class_indices:", class_to_index)


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR)
    return img


def _translate(img, dx, dy):
    transform = tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0])
    transform = tf.expand_dims(transform, axis=0)
    img = tf.expand_dims(img, axis=0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img,
        transforms=transform,
        output_shape=tf.constant([image_size, image_size], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return out[0]


def _rotate(img, angle_rad):
    angle = tf.cast(angle_rad, tf.float32)
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)

    cx = (tf.cast(image_size, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(image_size, tf.float32) - 1.0) / 2.0

    a0 = c
    a1 = -s
    a3 = s
    a4 = c
    a2 = cx - a0 * cx - a1 * cy
    a5 = cy - a3 * cx - a4 * cy

    transform = tf.stack([a0, a1, a2, a3, a4, a5, 0.0, 0.0])
    transform = tf.expand_dims(transform, axis=0)

    img_b = tf.expand_dims(img, axis=0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img_b,
        transforms=transform,
        output_shape=tf.constant([image_size, image_size], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return out[0]


def _random_zoom(img, zoom_range=0.1):
    z = tf.random.uniform([], 1.0 - zoom_range, 1.0 + zoom_range, seed=SEED)
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    new_h = tf.cast(tf.round(h / z), tf.int32)
    new_w = tf.cast(tf.round(w / z), tf.int32)

    def zoom_in():
        offset_h = tf.random.uniform(
            [],
            0,
            tf.maximum(1, tf.shape(img)[0] - new_h + 1),
            dtype=tf.int32,
            seed=SEED,
        )
        offset_w = tf.random.uniform(
            [],
            0,
            tf.maximum(1, tf.shape(img)[1] - new_w + 1),
            dtype=tf.int32,
            seed=SEED,
        )
        cropped = tf.image.crop_to_bounding_box(img, offset_h, offset_w, new_h, new_w)
        return tf.image.resize(
            cropped, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR
        )

    def zoom_out():
        pad_h = tf.maximum(0, new_h - tf.shape(img)[0])
        pad_w = tf.maximum(0, new_w - tf.shape(img)[1])
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = tf.pad(
            img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="REFLECT"
        )
        padded = tf.image.resize(
            padded, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR
        )
        return padded

    return tf.cond(z >= 1.0, zoom_in, zoom_out)


def _augment(img):
    angle = tf.random.uniform([], -10.0, 10.0, seed=SEED) * (np.pi / 180.0)
    img = _rotate(img, angle)

    max_dx = 0.05 * image_size
    max_dy = 0.05 * image_size
    dx = tf.random.uniform([], -max_dx, max_dx, seed=SEED)
    dy = tf.random.uniform([], -max_dy, max_dy, seed=SEED)
    img = _translate(img, dx, dy)

    img = _random_zoom(img, zoom_range=0.1)

    img = tf.image.random_flip_left_right(img, seed=SEED)
    return img


def _preprocess(img):
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def _labels_to_indices(labels_str_arr):
    cat = pd.Categorical(labels_str_arr, categories=class_names, ordered=True)
    return cat.codes.astype(np.int32)


def _make_ds(df, training):
    paths = df["image_path"].astype(str).values
    labels_str = df["label"].values.astype(str)
    labels = _labels_to_indices(labels_str)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.autotune_buffers = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True
        )

    def _load(path, label):
        img = _read_decode_resize(path)
        if training:
            img = _augment(img)
        img = _preprocess(img)
        y = tf.one_hot(label, num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    if not training:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_ds(train_df_train, training=True)
val_ds = _make_ds(train_df_val, training=False)

base = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
out = Dense(num_classes, activation="softmax", dtype="float32")(
    x
)  # keep output float32
model = Model(inputs=base.input, outputs=out)

base.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

epochs = 3
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2335325611.py in <cell line: 0>()
    195 
    196 
--> 197 train_ds = _make_ds(train_df_train, training=True)
    198 val_ds = _make_ds(train_df_val, training=False)
    199 

/tmp/ipykernel_11/2335325611.py in _make_ds(df, training)
    165     options.deterministic = True
    166     options.experimental_optimization.apply_default_optimizations = True
--> 167     options.experimental_optimization.autotune_buffers = True
    168     ds = ds.with_options(options)
    169 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 2
sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sub["image_id"].tolist()
test_paths = (TEST_IMG_DIR + "/" + sub["image_id"].astype(str)).values.astype(str)


def _make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.autotune_buffers = True
    ds = ds.with_options(options)

    def _load(path):
        img = _read_decode_resize(path)
        img = _preprocess(img)
        return img

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = _make_test_ds(test_paths)

probs = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)
pred_labels = [idx_to_label[int(i)] for i in pred_idx]

submission = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
assert submission.shape[0] == sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]
assert os.path.exists("submission.csv")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/216833153.py in <cell line: 0>()
     25 
     26 
---> 27 test_ds = _make_test_ds(test_paths)
     28 
     29 probs = model.predict(test_ds, verbose=1)

/tmp/ipykernel_11/216833153.py in _make_test_ds(paths)
     11     options.deterministic = True
     12     options.experimental_optimization.apply_default_optimizations = True
---> 13     options.experimental_optimization.autotune_buffers = True
     14     ds = ds.with_options(options)
     15 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

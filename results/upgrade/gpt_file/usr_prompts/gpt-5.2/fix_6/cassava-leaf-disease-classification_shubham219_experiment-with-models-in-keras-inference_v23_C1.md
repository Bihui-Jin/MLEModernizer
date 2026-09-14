# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TRAIN_TFREC_DIR), f"Missing {TRAIN_TFREC_DIR}"
assert os.path.exists(TEST_TFREC_DIR), f"Missing {TEST_TFREC_DIR}"

IMG_SIZE = (300, 300)
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(str)  # kept for stratify parity

tr_df, va_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["label"]
)

BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE

TRAIN_TFRECS = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
TEST_TFRECS = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

VAL_FRACTION = 0.15
VAL_MOD = 10000
VAL_THRESHOLD = int(round(VAL_FRACTION * VAL_MOD))

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0  # matches rescale=1/255
    return img


def _augment(img, key):
    k1 = tf.stack([key[0], key[1] + 1])
    k2 = tf.stack([key[0], key[1] + 2])
    k3 = tf.stack([key[0], key[1] + 3])
    k4 = tf.stack([key[0], key[1] + 4])

    img = tf.image.stateless_random_flip_left_right(img, seed=k1)

    angle = tf.random.stateless_uniform([], seed=k2, minval=-10.0, maxval=10.0) * (
        np.pi / 180.0
    )
    try:
        img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")
    except Exception:
        img = img

    h = IMG_SIZE[0]
    w = IMG_SIZE[1]
    max_dx = tf.cast(tf.round(0.05 * tf.cast(w, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(h, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [], seed=k3, minval=-max_dx, maxval=max_dx + 1, dtype=tf.int32
    )
    dy = tf.random.stateless_uniform(
        [], seed=k4, minval=-max_dy, maxval=max_dy + 1, dtype=tf.int32
    )

    pad_x = max_dx
    pad_y = max_dy
    img_pad = tf.pad(img, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT")
    img = tf.image.crop_to_bounding_box(img_pad, pad_y + dy, pad_x + dx, h, w)

    kz = tf.stack([key[0], key[1] + 5])
    scale = tf.random.stateless_uniform([], seed=kz, minval=0.9, maxval=1.0)
    new_h = tf.cast(tf.round(scale * tf.cast(h, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(scale * tf.cast(w, tf.float32)), tf.int32)
    kc = tf.stack([key[0], key[1] + 6])
    offset_y = tf.random.stateless_uniform(
        [], seed=kc, minval=0, maxval=h - new_h + 1, dtype=tf.int32
    )
    kc2 = tf.stack([key[0], key[1] + 7])
    offset_x = tf.random.stateless_uniform(
        [], seed=kc2, minval=0, maxval=w - new_w + 1, dtype=tf.int32
    )
    img = tf.image.crop_to_bounding_box(img, offset_y, offset_x, new_h, new_w)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    return img


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_and_resize(ex["image"])
    label = tf.one_hot(tf.cast(ex["target"], tf.int32), NUM_CLASSES)
    name = ex["image_name"]
    hid = tf.strings.to_hash_bucket_fast(name, 2**31 - 1)
    key = tf.stack([tf.constant(SEED, tf.int32), tf.cast(hid, tf.int32)])
    img = _augment(img, key)
    return img, label, name


def _parse_valid_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_and_resize(ex["image"])
    label = tf.one_hot(tf.cast(ex["target"], tf.int32), NUM_CLASSES)
    name = ex["image_name"]
    return img, label, name


@tf.function
def _is_valid_name(name):
    salted = tf.strings.join([tf.as_string(tf.constant(SEED)), name], separator="|")
    bucket = tf.math.floormod(
        tf.strings.to_hash_bucket_fast(salted, 2**31 - 1), VAL_MOD
    )
    return bucket < VAL_THRESHOLD


def _attach_is_valid(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    name = ex["image_name"]
    is_val = _is_valid_name(name)
    return example_proto, is_val


def make_train_valid_ds(batch_size=BATCH_SIZE):
    raw = tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=AUTO)
    tagged = raw.map(_attach_is_valid, num_parallel_calls=AUTO)

    train_tagged = tagged.filter(lambda ex, is_val: tf.logical_not(is_val)).map(
        lambda ex, is_val: ex, num_parallel_calls=AUTO
    )
    train_ds = (
        train_tagged.map(_parse_train_example, num_parallel_calls=AUTO)
        .map(lambda img, label, name: (img, label), num_parallel_calls=AUTO)
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTO)
    )

    valid_tagged = tagged.filter(lambda ex, is_val: is_val).map(
        lambda ex, is_val: ex, num_parallel_calls=AUTO
    )
    valid_ds = (
        valid_tagged.map(_parse_valid_example, num_parallel_calls=AUTO)
        .map(lambda img, label, name: (img, label), num_parallel_calls=AUTO)
        .cache()
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTO)
    )

    return train_ds, valid_ds


train_gen, valid_gen = make_train_valid_ds()

inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=inp, outputs=out)

base.trainable = False
my_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_HEAD = 2 if not DEBUG else 1
my_model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS_HEAD, verbose=1)

base.trainable = True
my_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_FT = 1 if not DEBUG else 1
my_model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS_FT, verbose=1)



## === cell 1
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
    img = _decode_and_resize(ex["image"])
    name = ex["image_name"]
    return img, name


def make_test_ds(batch_size=128):
    ds = tf.data.TFRecordDataset(TEST_TFRECS, num_parallel_reads=AUTO)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTO)
    return ds


test_ds = make_test_ds(batch_size=128)



## === cell 2
all_names = []
all_pred_labels = []

for batch_imgs, batch_names in test_ds:
    probs = my_model(batch_imgs, training=False)
    batch_labels = tf.argmax(probs, axis=-1, output_type=tf.int32)

    all_names.append(batch_names.numpy())
    all_pred_labels.append(batch_labels.numpy())

test_names = np.concatenate(all_names, axis=0).astype("U")  # bytes->str
pred_test_labels = np.concatenate(all_pred_labels, axis=0).astype(int)

name_to_label = dict(zip(test_names.tolist(), pred_test_labels.tolist()))

sample = pd.read_csv(SAMPLE_SUB)
labels = sample["image_id"].map(name_to_label)

if labels.isna().any():
    fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
    labels = labels.fillna(fill_label).astype(int)
else:
    labels = labels.astype(int)

sub = pd.DataFrame({"image_id": sample["image_id"].values, "label": labels.values})
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

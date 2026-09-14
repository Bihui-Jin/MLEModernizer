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
import random
import glob
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)  # let TF choose
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{BASE_PATH}/train.csv"
SAMPLE_SUB = f"{BASE_PATH}/sample_submission.csv"
TRAIN_IMG_DIR = f"{BASE_PATH}/train_images"
TEST_IMG_DIR = f"{BASE_PATH}/test_images"
TRAIN_TFREC_DIR = f"{BASE_PATH}/train_tfrecords"
TEST_TFREC_DIR = f"{BASE_PATH}/test_tfrecords"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing {TEST_TFREC_DIR}"

IMG_SIZE = (300, 300)
BATCH_SIZE = 32 if not DEBUG else 8
EPOCHS = 3 if not DEBUG else 1
NUM_CLASSES = 5

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

AUTO = tf.data.AUTOTUNE


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # dataset are JPEGs
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0  # match ImageDataGenerator(rescale=1/255)
    return img


try:
    import tensorflow_addons as tfa  # may not be present; fallback below if missing

    _HAS_TFA = True
except Exception:
    tfa = None
    _HAS_TFA = False


def _augment(img, seed):
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)
    s = tf.random.experimental.stateless_split(seed, 8)

    img = tf.image.stateless_random_flip_left_right(img, seed=s[0])

    z = tf.random.stateless_uniform(
        [], seed=s[1], minval=0.9, maxval=1.1, dtype=tf.float32
    )
    new_h = tf.cast(tf.round(tf.cast(IMG_SIZE[0], tf.float32) * z), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_SIZE[1], tf.float32) * z), tf.int32)
    img2 = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, IMG_SIZE[0], IMG_SIZE[1])

    dx = tf.random.stateless_uniform(
        [], seed=s[2], minval=-0.05, maxval=0.05, dtype=tf.float32
    )
    dy = tf.random.stateless_uniform(
        [], seed=s[3], minval=-0.05, maxval=0.05, dtype=tf.float32
    )
    tx = tf.cast(tf.round(dx * tf.cast(IMG_SIZE[1], tf.float32)), tf.int32)
    ty = tf.cast(tf.round(dy * tf.cast(IMG_SIZE[0], tf.float32)), tf.int32)
    pad_x = tf.abs(tx)
    pad_y = tf.abs(ty)
    img3 = tf.pad(img2, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT")
    start_y = pad_y + tf.maximum(-ty, 0)
    start_x = pad_x + tf.maximum(-tx, 0)
    img3 = tf.image.crop_to_bounding_box(
        img3, start_y, start_x, IMG_SIZE[0], IMG_SIZE[1]
    )

    if _HAS_TFA:
        angle = tf.random.stateless_uniform(
            [], seed=s[4], minval=-15.0, maxval=15.0, dtype=tf.float32
        )
        angle = angle * (np.pi / 180.0)
        img3 = tfa.image.rotate(
            img3, angles=angle, interpolation="BILINEAR", fill_mode="nearest"
        )

    return img3


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_tfrecord(example_proto, labeled=True):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    if labeled:
        label = tf.cast(ex["target"], tf.int32)
        y = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, y, ex["image_name"]
    else:
        return img, ex["image_name"]


def _ds_options(deterministic=True):
    opt = tf.data.Options()
    opt.experimental_deterministic = deterministic
    opt.experimental_optimization.apply_default_optimizations = True
    try:
        opt.experimental_optimization.map_fusion = True
        opt.experimental_optimization.map_and_batch_fusion = True
        opt.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    return opt


def _list_tfrecs(dir_path, pattern="*.tfrec"):
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    if not files:
        raise FileNotFoundError(f"No TFRecords found in: {dir_path}")
    return files


train_tfrecs = _list_tfrecs(TRAIN_TFREC_DIR)
test_tfrecs = _list_tfrecs(TEST_TFREC_DIR)



## === cell 1
df = pd.read_csv(TRAIN_CSV)
df["label"] = df["label"].astype(np.int32)

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label"]
)

val_names = tf.constant(val_df["image_id"].astype(str).values)
val_name_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        val_names, tf.ones_like(val_names, dtype=tf.int32)
    ),
    default_value=0,
)


def _seed_from_name(name):
    h = tf.strings.to_hash_bucket_fast(name, 2**31 - 1)
    return tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)], axis=0)


def _make_train_val_ds_from_tfrecs(tfrecs):
    raw = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTO, compression_type=None
    )
    raw = raw.with_options(_ds_options(deterministic=True))

    def _parse_and_tag(x):
        img, y, name = _parse_tfrecord(x, labeled=True)
        is_val = tf.equal(val_name_table.lookup(name), 1)
        return img, y, name, is_val

    parsed = raw.map(_parse_and_tag, num_parallel_calls=AUTO)
    parsed = (
        parsed.cache()
    )  # cache after decode/resize: identical data, large speedup across epochs

    train = parsed.filter(lambda img, y, name, is_val: tf.logical_not(is_val))
    val = parsed.filter(lambda img, y, name, is_val: is_val)

    train = train.shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)

    def _train_map(img, y, name, is_val):
        img = _augment(img, _seed_from_name(name))
        return img, y

    def _val_map(img, y, name, is_val):
        return img, y

    train = train.map(_train_map, num_parallel_calls=AUTO)
    val = val.map(_val_map, num_parallel_calls=AUTO)

    train = train.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    val = val.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    return train, val


train_ds, val_ds = _make_train_val_ds_from_tfrecs(train_tfrecs)



## === cell 2
base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)

base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=1, verbose=1, min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists("best_model.keras"):
    model = keras.models.load_model("best_model.keras")



## === cell 3
sub = pd.read_csv(SAMPLE_SUB)
sub["path"] = TEST_IMG_DIR.rstrip("/") + "/" + sub["image_id"].astype(str)
test_names = sub["image_id"].astype(str).values


def _make_test_ds_from_tfrecs(tfrecs, batch_size=128):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTO, compression_type=None)
    ds = ds.with_options(_ds_options(deterministic=True))
    ds = ds.map(lambda x: _parse_tfrecord(x, labeled=False), num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


test_ds = _make_test_ds_from_tfrecs(test_tfrecs, batch_size=128)

name_ds = test_ds.map(lambda imgs, names: names, num_parallel_calls=AUTO).unbatch()
tfrecord_names = np.array(list(name_ds.as_numpy_iterator())).astype("U").tolist()

pred = model.predict(test_ds, verbose=1)
pred_labels = np.argmax(pred, axis=1).astype(int)

name_to_pred = dict(zip(tfrecord_names, pred_labels))
ordered_pred = np.array([name_to_pred[n] for n in test_names], dtype=int)

submission = sub[["image_id"]].copy()
submission["label"] = ordered_pred
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[0] == sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

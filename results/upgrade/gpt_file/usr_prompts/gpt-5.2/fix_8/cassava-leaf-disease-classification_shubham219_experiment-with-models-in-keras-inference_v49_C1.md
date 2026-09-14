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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification",
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset directory."
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

for p in [TRAIN_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR, SAMPLE_SUB]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected path: {p}")

print("Using DATA_ROOT:", DATA_ROOT)
print("Python:", os.sys.version)
print("TensorFlow:", tf.__version__)



## === cell 1
WEIGHT_PATH_CANDIDATES = [
    "../input/model-v13/clf_new_26 (1).h5",
    "/kaggle/input/model-v13/clf_new_26 (1).h5",
]
WEIGHT_PATH = next((p for p in WEIGHT_PATH_CANDIDATES if os.path.exists(p)), None)

my_model = None
if WEIGHT_PATH is not None:
    try:
        from tensorflow.keras.models import load_model

        my_model = load_model(WEIGHT_PATH, compile=False)
        print("Loaded pretrained model from:", WEIGHT_PATH)
    except Exception as e:
        print(
            "WARNING: Could not load model weights; will train a fresh model instead."
        )
        print("Load error:", repr(e))
        my_model = None
else:
    print("No pretrained weight file found; will train a fresh model.")



## === cell 2
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 3  # keep short to fit runtime; no early stopping used
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = (
    TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
).astype(str)

if DEBUG:
    train_df = train_df.sample(n=2000, random_state=SEED).reset_index(drop=True)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]
df_tr = train_df.iloc[train_idx].reset_index(drop=True)
df_va = train_df.iloc[val_idx].reset_index(drop=True)

y_tr = df_tr["label"].astype(np.int32).values
y_va = df_va["label"].astype(np.int32).values
x_tr = df_tr["path"].astype(str).values
x_va = df_va["path"].astype(str).values


def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


_rot_layer = tf.keras.layers.RandomRotation(
    factor=15.0 / 180.0,  # +/- 15 degrees
    fill_mode="reflect",
    interpolation="bilinear",
    seed=SEED,
)


def _augment(img, seed_pair):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)

    img = _rot_layer(img, training=True)

    dx = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([0, 1], tf.int32),
        minval=-0.1,
        maxval=0.1,
        dtype=tf.float32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([1, 1], tf.int32),
        minval=-0.1,
        maxval=0.1,
        dtype=tf.float32,
    )

    z = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([2, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    new_h = tf.cast(tf.cast(h, tf.float32) / z, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) / z, tf.int32)
    new_h = tf.clip_by_value(new_h, 1, h)
    new_w = tf.clip_by_value(new_w, 1, w)
    img = tf.image.resize_with_crop_or_pad(img, new_h, new_w)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    pad_h = tf.cast(tf.round(tf.cast(h, tf.float32) * 0.1), tf.int32) + 2
    pad_w = tf.cast(tf.round(tf.cast(w, tf.float32) * 0.1), tf.int32) + 2
    img_pad = tf.pad(img, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="REFLECT")
    off_y = tf.cast(tf.round(dy * tf.cast(h, tf.float32)), tf.int32) + pad_h
    off_x = tf.cast(tf.round(dx * tf.cast(w, tf.float32)), tf.int32) + pad_w
    off_y = tf.clip_by_value(off_y, 0, tf.shape(img_pad)[0] - h)
    off_x = tf.clip_by_value(off_x, 0, tf.shape(img_pad)[1] - w)
    img = tf.image.crop_to_bounding_box(img_pad, off_y, off_x, h, w)
    return img


def make_train_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    cache_path = os.path.join("/kaggle/working", "cache_train_decode_resize")
    ds = ds.map(
        lambda p, y: (_decode_resize_rescale(p), y), num_parallel_calls=AUTOTUNE
    )
    ds = ds.cache(cache_path)

    ds = ds.enumerate()

    def _map(i, il):
        img, label = il
        seed_pair = tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)])
        img = _augment(img, seed_pair)
        return img, label

    opts = tf.data.Options()
    opts.experimental_deterministic = False
    ds = ds.with_options(opts)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(
        lambda p, y: (_decode_resize_rescale(p), y), num_parallel_calls=AUTOTUNE
    )
    cache_path = os.path.join("/kaggle/working", "cache_val_decode_resize")
    ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(x_tr, y_tr, BATCH_SIZE)
val_ds = make_val_ds(x_va, y_va, BATCH_SIZE)

steps_per_epoch = len(x_tr) // BATCH_SIZE
if steps_per_epoch == 0:
    steps_per_epoch = 1
validation_steps = int(np.ceil(len(x_va) / BATCH_SIZE))

if my_model is None:
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    base.trainable = False  # keep training stable/fast

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    my_model = tf.keras.Model(inputs, outputs)

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    my_model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
    )



## === cell 3
test_images = sorted(tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No .jpg files found in TEST_IMG_DIR: {TEST_IMG_DIR}")

df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_ds(paths, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_decode_resize_rescale, num_parallel_calls=AUTOTUNE)
    cache_path = os.path.join("/kaggle/working", "cache_test_decode_resize")
    ds = ds.cache(cache_path)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 4
if my_model is None:
    raise RuntimeError("Model was not created/loaded; cannot run prediction.")

test_ds = make_test_ds(df_test["path"].astype(str).values, batch_size=128)

pred_test = my_model.predict(
    test_ds,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].map(os.path.basename)
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]

sample_sub = pd.read_csv(SAMPLE_SUB)
final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
final_csv["label"] = final_csv["label"].fillna(0).astype(int)

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print(
    f"Wrote submission.csv with shape: {final_csv.shape} and columns: {list(final_csv.columns)}"
)
print(
    f"Label value counts:\n{final_csv['label'].value_counts(dropna=False).sort_index()}"
)

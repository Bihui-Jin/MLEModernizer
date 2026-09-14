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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

BASE_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

print("TensorFlow:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train image dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))



## === cell 1
IMG_SIZE = (300, 300)  # matches original target_size
NUM_CLASSES = 5

df = pd.read_csv(TRAIN_CSV)

train_prefix = TRAIN_IMG_DIR.rstrip("/") + "/"
df["path"] = train_prefix + df["image_id"].astype(str)
df["label"] = df["label"].astype(np.int32)

if DEBUG:
    df = df.sample(2000, random_state=SEED).reset_index(drop=True)

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label"]
)

BATCH_SIZE = 32
EPOCHS = 2  # unchanged

AUTOTUNE = tf.data.AUTOTUNE

_DATA_OPTS = tf.data.Options()
_DATA_OPTS.experimental_optimization.apply_default_optimizations = True
_DATA_OPTS.experimental_optimization.map_parallelization = True
_DATA_OPTS.experimental_optimization.parallel_batch = True
_DATA_OPTS.experimental_deterministic = (
    True  # keep determinism consistent with seeds/op determinism
)


def _decode_jpeg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


def _augment(img, seed_vec):
    img = tf.image.stateless_random_flip_left_right(
        img, seed=seed_vec + tf.constant([2, 0], tf.int32)
    )
    img = tf.image.stateless_random_flip_up_down(
        img, seed=seed_vec + tf.constant([5, 0], tf.int32)
    )

    k = tf.random.stateless_uniform(
        [],
        seed=seed_vec + tf.constant([6, 0], tf.int32),
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    img = tf.image.rot90(img, k=k)

    crop_frac = tf.random.stateless_uniform(
        [], seed=seed_vec + tf.constant([3, 0], tf.int32), minval=0.9, maxval=1.0
    )
    crop_size = tf.cast(
        tf.round(crop_frac * tf.cast(tf.shape(img)[:2], tf.float32)), tf.int32
    )
    crop_size = tf.maximum(crop_size, 1)

    pad_y = tf.cast(tf.round(0.05 * tf.cast(tf.shape(img)[0], tf.float32)), tf.int32)
    pad_x = tf.cast(tf.round(0.05 * tf.cast(tf.shape(img)[1], tf.float32)), tf.int32)
    img_pad = tf.pad(img, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT")

    img_crop = tf.image.stateless_random_crop(
        img_pad,
        size=tf.stack([crop_size[0], crop_size[1], 3]),
        seed=seed_vec + tf.constant([4, 0], tf.int32),
    )
    img = tf.image.resize(img_crop, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return img


def _make_dataset(paths, labels=None, training=False, batch_size=32, seed=SEED):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    if labels is not None:
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    ds = ds.with_options(_DATA_OPTS)

    if training:
        ds = ds.shuffle(
            buffer_size=min(int(paths.shape[0]), 8192),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _map_with_label(path, y):
        img = _decode_jpeg(path)
        if training:
            h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
            seed_vec = tf.stack([tf.cast(h, tf.int32), tf.cast(seed, tf.int32)])
            img = _augment(img, seed_vec)
        y = tf.one_hot(y, NUM_CLASSES, dtype=tf.float32)  # class_mode="categorical"
        return img, y

    def _map_no_label(path):
        img = _decode_jpeg(path)
        return img

    if labels is not None:
        ds = ds.map(_map_with_label, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        ds = ds.map(_map_no_label, num_parallel_calls=AUTOTUNE, deterministic=True)

    if not training:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_dataset(
    train_df["path"].values,
    train_df["label"].values,
    training=True,
    batch_size=BATCH_SIZE,
    seed=SEED,
)
val_ds = _make_dataset(
    val_df["path"].values,
    val_df["label"].values,
    training=False,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=inp, outputs=out)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 2
test_images = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
test_images.sort()

df_test = pd.DataFrame({"path": test_images})


def make_test_ds(batch_size=64):
    return _make_dataset(
        df_test["path"].values,
        labels=None,
        training=False,
        batch_size=batch_size,
        seed=SEED,
    )




## === cell 3
test_ds = make_test_ds(batch_size=128)

pred_test = my_model.predict(
    test_ds,
    verbose=1,
)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = [
    os.path.basename(p) for p in final_submission["path"].astype(str).tolist()
]
final_submission["label"] = pred_test_labels
final_csv = final_submission[["image_id", "label"]]

sample = pd.read_csv(SAMPLE_SUB)
final_csv = sample[["image_id"]].merge(final_csv, on="image_id", how="left")

if final_csv["label"].isna().any():
    fill_label = (
        int(pd.Series(pred_test_labels).mode().iloc[0]) if len(pred_test_labels) else 0
    )
    final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)
else:
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## === cell 4
final_csv.head()

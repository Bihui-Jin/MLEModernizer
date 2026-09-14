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
import os, math, random, glob, json
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_TFREC = os.path.join(BASE_PATH, "train_tfrecords", "*.tfrec")
TEST_TFREC = os.path.join(BASE_PATH, "test_tfrecords", "*.tfrec")

print("TensorFlow:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train tfrecs:", len(glob.glob(TRAIN_TFREC)))
print("Test tfrecs:", len(glob.glob(TEST_TFREC)))




## === cell 1
AUTO = tf.data.AUTOTUNE
NUM_CLASSES = 5
IMG_SIZE = 300
BATCH_SIZE = 32  # keep unchanged
EPOCHS = 5  # keep unchanged

TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def decode_image(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def parse_train_tfrecord(example_proto):
    x = tf.io.parse_single_example(example_proto, TRAIN_FEATURES)
    img = decode_image(x["image"])
    label = tf.cast(x["target"], tf.int32)
    return img, label


@tf.function
def parse_test_tfrecord(example_proto):
    x = tf.io.parse_single_example(example_proto, TEST_FEATURES)
    img = decode_image(x["image"])
    return img, x["image_name"]


@tf.function
def augment(img, label):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED)
    img = tf.image.random_brightness(img, max_delta=0.1, seed=SEED)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_train_val_datasets(train_files, val_files):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True

    train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=AUTO)
    train_ds = train_ds.with_options(options)
    train_ds = train_ds.map(parse_train_tfrecord, num_parallel_calls=AUTO)
    train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.map(augment, num_parallel_calls=AUTO)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

    val_ds = tf.data.TFRecordDataset(val_files, num_parallel_reads=AUTO)
    val_ds = val_ds.with_options(options)
    val_ds = val_ds.map(parse_train_tfrecord, num_parallel_calls=AUTO)
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    return train_ds, val_ds


def make_test_dataset(test_files):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True

    test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTO)
    test_ds = test_ds.with_options(options)
    test_ds = test_ds.map(parse_test_tfrecord, num_parallel_calls=AUTO)
    test_ds = test_ds.cache()
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    return test_ds




## === cell 2
train_files = sorted(glob.glob(TRAIN_TFREC))
test_files = sorted(glob.glob(TEST_TFREC))

if len(train_files) == 0 or len(test_files) == 0:
    raise FileNotFoundError("TFRecord files not found. Check dataset paths.")

rng = np.random.default_rng(SEED)
idx = np.arange(len(train_files))
rng.shuffle(idx)

split = int(0.9 * len(train_files))
tr_idx, va_idx = idx[:split], idx[split:]
tr_files = [train_files[i] for i in tr_idx]
va_files = [train_files[i] for i in va_idx]

print("Train shards:", len(tr_files), "Val shards:", len(va_files))

train_ds, val_ds = make_train_val_datasets(tr_files, va_files)
test_ds = make_test_dataset(test_files)




## === cell 3
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = True

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=True)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()




## === cell 4
history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 5
test_image_ids_chunks = []
for _, batch_ids in test_ds:
    test_image_ids_chunks.append(batch_ids.numpy())
test_image_ids = np.concatenate(test_image_ids_chunks, axis=0)
test_image_ids = np.array([x.decode("utf-8") for x in test_image_ids])

test_probs = my_model.predict(
    test_ds.map(lambda x, y: x, num_parallel_calls=AUTO), verbose=0
)
pred_test_labels = np.argmax(test_probs, axis=-1).astype(int)

print("Preds:", pred_test_labels.shape, "IDs:", test_image_ids.shape)
print("Unique labels predicted:", np.unique(pred_test_labels))




## === cell 6
sample = pd.read_csv(SAMPLE_SUB)
pred_df = pd.DataFrame({"image_id": test_image_ids, "label": pred_test_labels})

sub = sample[["image_id"]].merge(pred_df, on="image_id", how="left")

if sub["label"].isna().any():
    fill_value = int(pd.Series(pred_test_labels).mode().iloc[0])
    sub["label"] = sub["label"].fillna(fill_value).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

print(sub.head())
print("Submission rows:", len(sub), "Missing labels:", sub["label"].isna().sum())

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Saved at:", os.path.abspath("submission.csv"))

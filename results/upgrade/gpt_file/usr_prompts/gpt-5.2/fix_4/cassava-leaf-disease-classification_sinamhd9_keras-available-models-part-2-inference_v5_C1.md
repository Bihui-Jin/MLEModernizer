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
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing dir: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing dir: {TEST_TFREC_DIR}"

print("TensorFlow:", tf.__version__)
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))
print("Train tfrecords dir exists:", os.path.isdir(TRAIN_TFREC_DIR))
print("Test tfrecords dir exists:", os.path.isdir(TEST_TFREC_DIR))



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

assert {"image_id", "label"}.issubset(train_df.columns)
assert {"image_id", "label"}.issubset(sub_df.columns)

train_df["filepath"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x)
)

exists_mask = train_df["filepath"].apply(tf.io.gfile.exists).to_numpy()
if exists_mask.mean() < 1.0:
    missing = train_df.loc[~exists_mask, "image_id"].head(10).tolist()
    print(
        f"Warning: missing {len(train_df) - int(exists_mask.sum())} train files; examples: {missing}"
    )
train_df = train_df.loc[exists_mask].reset_index(drop=True)

num_classes = int(train_df["label"].nunique())
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"
print("Train rows:", len(train_df), "Classes:", num_classes)
print("Sample submission rows:", len(sub_df))



## === cell 2
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

tfdata_opts = tf.data.Options()
tfdata_opts.experimental_deterministic = True


def decode_and_resize_from_jpeg_bytes(img_bytes, label=None, training=False):
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # force RGB
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    if training:
        img = tf.image.random_flip_left_right(img, seed=SEED)
        img = tf.image.random_flip_up_down(img, seed=SEED)
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


def decode_and_resize(path, label=None, training=False):
    img_bytes = tf.io.read_file(path)
    return decode_and_resize_from_jpeg_bytes(img_bytes, label=label, training=training)


TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_train_example(example_proto, training=False):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img_bytes = ex["image"]
    label = ex["target"]
    return decode_and_resize_from_jpeg_bytes(img_bytes, label=label, training=training)


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img_bytes = ex["image"]
    image_name = ex["image_name"]  # bytes
    img = decode_and_resize_from_jpeg_bytes(img_bytes, label=None, training=False)
    return image_name, img


val_frac = 0.10
train_idx = []
val_idx = []
for cls, grp in train_df.groupby("label"):
    idx = grp.index.to_numpy()
    rng = np.random.RandomState(SEED + int(cls))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val_idx.extend(idx[:n_val].tolist())
    train_idx.extend(idx[n_val:].tolist())

train_split = (
    train_df.loc[train_idx].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
)
val_split = train_df.loc[val_idx].reset_index(drop=True)
print("Train split:", len(train_split), "Val split:", len(val_split))

train_paths_set = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(train_split["image_id"].values, dtype=tf.string),
        values=tf.ones([len(train_split)], dtype=tf.int64),
    ),
    default_value=tf.constant(0, dtype=tf.int64),
)
val_paths_set = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(val_split["image_id"].values, dtype=tf.string),
        values=tf.ones([len(val_split)], dtype=tf.int64),
    ),
    default_value=tf.constant(0, dtype=tf.int64),
)

train_tfrecs = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrecs = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(train_tfrecs) > 0, "No train tfrecords found"
assert len(test_tfrecs) > 0, "No test tfrecords found"


def _is_in_train(example_proto):
    ex = tf.io.parse_single_example(
        example_proto, {"image_name": TFREC_FEATURES["image_name"]}
    )
    return tf.equal(train_paths_set.lookup(ex["image_name"]), 1)


def _is_in_val(example_proto):
    ex = tf.io.parse_single_example(
        example_proto, {"image_name": TFREC_FEATURES["image_name"]}
    )
    return tf.equal(val_paths_set.lookup(ex["image_name"]), 1)


raw_train = tf.data.TFRecordDataset(
    train_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(tfdata_opts)

train_ds = raw_train.filter(_is_in_train)
train_ds = train_ds.map(
    lambda x: _parse_train_example(x, training=True),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

val_ds = raw_train.filter(_is_in_val)
val_ds = val_ds.map(
    lambda x: _parse_train_example(x, training=False),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)




## === cell 3
def build_model(img_size=IMG_SIZE, n_classes=num_classes):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25, seed=SEED)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 4
EPOCHS = 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
)



## === cell 5
test_image_ids = sub_df["image_id"].tolist()

missing = [
    img_id
    for img_id in test_image_ids
    if not tf.io.gfile.exists(os.path.join(TEST_IMG_DIR, img_id))
]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images. Example: {missing[0]}"
    )

raw_test = tf.data.TFRecordDataset(
    test_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(tfdata_opts)
test_ds = raw_test.map(
    _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 6
name_batches = []
prob_batches = []
for names, imgs in test_ds:
    probs_batch = model(imgs, training=False).numpy()
    name_batches.append(names.numpy())
    prob_batches.append(probs_batch)

test_names = np.concatenate(name_batches, axis=0)
probs = np.concatenate(prob_batches, axis=0)

test_names = np.array([n.decode("utf-8") for n in test_names], dtype=object)

pred_labels = np.argmax(probs, axis=1).astype(int)
name_to_label = dict(zip(test_names.tolist(), pred_labels.tolist()))

y_preds = np.array([name_to_label[img_id] for img_id in test_image_ids], dtype=int)

assert len(y_preds) == len(test_image_ids), (len(y_preds), len(test_image_ids))
print("Predictions shape:", probs.shape)



## === cell 7
df_sub = pd.DataFrame({"image_id": test_image_ids, "label": y_preds})
df_sub["label"] = df_sub["label"].astype(int)

assert df_sub.shape[0] == sub_df.shape[0]
assert list(df_sub.columns) == ["image_id", "label"]
print(df_sub.head())

out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df_sub))

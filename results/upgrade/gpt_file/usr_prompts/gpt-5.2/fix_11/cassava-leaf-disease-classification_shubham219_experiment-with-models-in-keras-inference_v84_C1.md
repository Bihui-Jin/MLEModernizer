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
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("Imports OK. TF:", tf.__version__)


def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None




## === cell 1
USING_FALLBACK = True  # will become False once we build a proper 5-class model

train_csv_path = first_existing(
    [
        "/kaggle/input/cassava-leaf-disease-classification/train.csv",
        "../input/cassava-leaf-disease-classification/train.csv",
        "/kaggle/data/cassava-leaf-disease-classification/train.csv",
        "../data/cassava-leaf-disease-classification/train.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
    ]
)
train_img_dir = first_existing(
    [
        "/kaggle/input/cassava-leaf-disease-classification/train_images",
        "../input/cassava-leaf-disease-classification/train_images",
        "/kaggle/data/cassava-leaf-disease-classification/train_images",
        "../data/cassava-leaf-disease-classification/train_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
    ]
)

if train_csv_path is None or train_img_dir is None:
    raise FileNotFoundError(
        "Could not find train.csv and/or train_images in expected locations."
    )

df_train = pd.read_csv(train_csv_path)
if "image_id" not in df_train.columns or "label" not in df_train.columns:
    raise ValueError("train.csv must contain image_id and label columns.")

df_train["path"] = train_img_dir.rstrip("/") + "/" + df_train["image_id"].astype(str)

if DEBUG:
    missing_paths = (~df_train["path"].map(os.path.exists)).sum()
    if missing_paths > 0:
        raise FileNotFoundError(
            f"Found {missing_paths} missing training image files. Check train_images path."
        )

NUM_CLASSES = int(df_train["label"].nunique())
if NUM_CLASSES != 5:
    raise ValueError(f"Expected 5 classes for Cassava, got {NUM_CLASSES}.")

IMG_SIZE = (224, 224)

base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False  # minimal/fast: train head first; keeps runtime < 600s

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base(x, training=False)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = tf.keras.Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

USING_FALLBACK = False
print(
    "Built Cassava model: EfficientNetB0 backbone (frozen) + Dense(5) head, input=224"
)

df_train = df_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_n = int(len(df_train) * val_frac)
df_val = df_train.iloc[:val_n].copy()
df_tr = df_train.iloc[val_n:].copy()

AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # matches .jpg input
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


def make_train_ds(df, batch_size=32, training=True):
    paths = df["path"].to_numpy()
    labels = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    opts = tf.data.Options()
    opts.experimental_deterministic = True  # preserve stable semantics
    opts.autotune.enabled = True
    ds = ds.with_options(opts)

    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # cache decoded+resized tensors once

    if training:
        buf = min(len(df), 4096)
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


BATCH_SIZE = 32
train_ds = make_train_ds(df_tr, batch_size=BATCH_SIZE, training=True)
val_ds = make_train_ds(df_val, batch_size=BATCH_SIZE, training=False)

EPOCHS_HEAD = 3
history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
EPOCHS_FT = 1
history2 = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    verbose=1,
)



## === cell 2
test_img_dir = first_existing(
    [
        "/kaggle/input/cassava-leaf-disease-classification/test_images",
        "../input/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/cassava-leaf-disease-classification/test_images",
        "../data/cassava-leaf-disease-classification/test_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    ]
)

if test_img_dir is None:
    raise FileNotFoundError(
        "Could not find test_images directory in expected locations."
    )

test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
if len(test_images) == 0:
    raise RuntimeError(f"No .jpg files found under: {test_img_dir}")

df_test = pd.DataFrame({"path": test_images})

MODEL_IMAGE_SIZE = IMG_SIZE  # for our trained model


def _decode_resize_test(path):
    return _decode_resize(path, None)


def make_test_ds(batch_size=32):
    paths = df_test["path"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(paths)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.autotune.enabled = True
    ds = ds.with_options(opts)

    ds = ds.map(_decode_resize_test, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


print("Test images:", len(df_test))
print("Using fallback:", USING_FALLBACK)
print("Model image size:", MODEL_IMAGE_SIZE)



## === cell 3
test_ds = make_test_ds(batch_size=32)
pred_test = my_model.predict(test_ds, verbose=1)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].map(os.path.basename)
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]].copy()

sample_path = first_existing(
    [
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "../input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "../data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    ]
)

if sample_path is not None:
    sample = pd.read_csv(sample_path)
    final_csv = sample[["image_id"]].merge(final_csv, on="image_id", how="left")
    if final_csv["label"].isna().any():
        missing = final_csv[final_csv["label"].isna()]["image_id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions for some test ids, e.g.: {missing}")
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())

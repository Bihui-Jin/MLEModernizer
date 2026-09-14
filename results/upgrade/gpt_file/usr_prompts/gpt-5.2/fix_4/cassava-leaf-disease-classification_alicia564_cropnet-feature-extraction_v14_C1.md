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

3.13

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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMAGE_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
OUT_PATH = "/kaggle/working/submission.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3  # small but non-trivial; keep runtime under 600s
NUM_CLASSES = 5

for p in [TRAIN_CSV, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, SAMPLE_SUB_PATH]:
    if not tf.io.gfile.exists(p):
        raise FileNotFoundError(f"Missing required path: {p}")




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError(f"Unexpected train.csv columns: {train_df.columns.tolist()}")

train_paths = (TRAIN_IMAGE_DIR + "/" + train_df["image_id"].astype(str)).to_numpy(
    dtype=object
)
train_labels = train_df["label"].astype(np.int32).to_numpy()

if len(train_paths) != len(train_labels) or len(train_paths) == 0:
    raise RuntimeError("Train data not loaded correctly.")

missing = [p for p in train_paths[:200] if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Some train images are missing (showing up to 3): {missing[:3]}"
    )

idx = np.arange(len(train_paths))
rng = np.random.default_rng(42)
rng.shuffle(idx)

val_frac = 0.1
val_n = int(len(idx) * val_frac)

val_idx = idx[:val_n]
trn_idx = idx[val_n:]

trn_paths = train_paths[trn_idx]
trn_labels = train_labels[trn_idx]

val_paths = train_paths[val_idx]
val_labels = train_labels[val_idx]

print("Train size:", len(trn_paths), "Val size:", len(val_paths))
print("Train label dist:", np.bincount(trn_labels, minlength=NUM_CLASSES))




## === cell 3
def decode_resize_norm(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True

trn_ds = tf.data.Dataset.from_tensor_slices((trn_paths, trn_labels)).with_options(
    options
)
trn_ds = trn_ds.shuffle(4096, seed=42, reshuffle_each_iteration=True)
trn_ds = trn_ds.map(decode_resize_norm, num_parallel_calls=AUTOTUNE, deterministic=True)
trn_ds = trn_ds.cache()
trn_ds = trn_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).with_options(
    options
)
val_ds = val_ds.map(decode_resize_norm, num_parallel_calls=AUTOTUNE, deterministic=True)
val_ds = val_ds.cache()
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)




## === cell 4
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # stability + speed in Kaggle

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    jit_compile=True,
)

model.summary()




## === cell 5
history = model.fit(
    trn_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    jit_compile=True,
)

history_ft = model.fit(
    trn_ds,
    validation_data=val_ds,
    epochs=1,
    verbose=2,
)




## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if not {"image_id", "label"}.issubset(sample_sub.columns):
    raise ValueError(
        f"Unexpected sample_submission columns: {sample_sub.columns.tolist()}"
    )

image_ids = sample_sub["image_id"].astype(str).to_numpy()
test_paths = (TEST_IMAGE_DIR + "/" + image_ids).astype(object)

missing_test = [p for p in test_paths[:200] if not os.path.exists(p)]
if missing_test:
    raise FileNotFoundError(
        f"Some test images are missing (showing up to 3): {missing_test[:3]}"
    )

test_options = tf.data.Options()
test_options.experimental_deterministic = True

test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(test_options)
test_ds = test_ds.map(
    lambda p: decode_resize_norm(p, None),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=0)
if probs.shape[0] != len(image_ids) or probs.shape[1] != NUM_CLASSES:
    raise RuntimeError(f"Unexpected prediction shape: {probs.shape}")

pred_labels = np.argmax(probs, axis=1).astype(int)

submission_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
if submission_df["label"].min() < 0 or submission_df["label"].max() >= NUM_CLASSES:
    raise ValueError(
        f"Predicted labels out of range: min={submission_df['label'].min()}, max={submission_df['label'].max()}, classes={NUM_CLASSES}"
    )

submission_df.to_csv(OUT_PATH, index=False)
print(f"Submission file created: {OUT_PATH}")
print(submission_df.head())
print("Label counts:\n", submission_df["label"].value_counts().sort_index())

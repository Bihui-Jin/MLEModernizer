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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
import tensorflow as tf



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTO = tf.data.AUTOTUNE
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_CLASSES = 5

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 2
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

train_files = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_files = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

assert len(train_files) > 0, f"No train tfrecords found in {TRAIN_TFREC_DIR}"
assert len(test_files) > 0, f"No test tfrecords found in {TEST_TFREC_DIR}"

print(f"Found {len(train_files)} train tfrecords, {len(test_files)} test tfrecords")



## === cell 3
_TRAIN_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function(reduce_retracing=True)
def _decode_jpeg_resize(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(
        image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image = tf.cast(image, tf.float32)
    image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return image




## === cell 4
from tensorflow.keras.applications.efficientnet import preprocess_input


@tf.function(reduce_retracing=True)
def decode_and_preprocess_train(example):
    example = tf.io.parse_single_example(example, _TRAIN_FEATURE_DESC)
    image = _decode_jpeg_resize(example["image"])
    image = preprocess_input(image)
    label = tf.cast(example["target"], tf.int32)
    return image, label


@tf.function(reduce_retracing=True)
def decode_and_preprocess_test(example):
    example = tf.io.parse_single_example(example, _TEST_FEATURE_DESC)
    image = _decode_jpeg_resize(example["image"])
    image = preprocess_input(image)
    return image, example["image_name"]




## === cell 5
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(40 / 360),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomZoom(0.2, 0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
    ],
    name="data_augmentation",
)



## === cell 6
options = tf.data.Options()
options.experimental_deterministic = True  # preserve deterministic semantics

try:
    options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    options.experimental_optimization.autotune_buffers = True
except Exception:
    pass
try:
    options.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    options.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
except Exception:
    pass


def _tfrecord_dataset(files):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    return ds.with_options(options)


VAL_FRAC = 0.1
SHUFFLE_BUFFER = 8192

train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)
n_total = len(train_df)
n_val = max(1, int(n_total * VAL_FRAC))
n_train = n_total - n_val

n_train_files = max(1, int(round(len(train_files) * (1.0 - VAL_FRAC))))
train_files_split = train_files[:n_train_files]
val_files_split = (
    train_files[n_train_files:]
    if n_train_files < len(train_files)
    else train_files[-1:]
)

raw_train = _tfrecord_dataset(train_files_split)
raw_val = _tfrecord_dataset(val_files_split)

train_base = raw_train.map(
    decode_and_preprocess_train, num_parallel_calls=AUTO, deterministic=True
)
val_base = raw_val.map(
    decode_and_preprocess_train, num_parallel_calls=AUTO, deterministic=True
)


@tf.function(reduce_retracing=True)
def _augment(x, y):
    return data_augmentation(x, training=True), y


train_ds = (
    train_base.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .map(_augment, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

val_ds = val_base.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

raw_test = _tfrecord_dataset(test_files)
test_ds = (
    raw_test.map(
        decode_and_preprocess_test, num_parallel_calls=AUTO, deterministic=True
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

steps_per_epoch = max(1, n_train // BATCH_SIZE)
validation_steps = max(1, int(np.ceil(n_val / BATCH_SIZE)))

print("Datasets created successfully!")
print("n_total:", n_total, "n_train:", n_train, "n_val:", n_val)
print(
    "train tfrec files:",
    len(train_files_split),
    "val tfrec files:",
    len(val_files_split),
)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 7
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False  # keep identical training approach

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = inputs
x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 8
EPOCHS = 3
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)




## === cell 9
@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec(shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    ],
)
def _tta_mean_predict(images, num_tta):
    b = tf.shape(images)[0]
    images_t = tf.tile(tf.expand_dims(images, axis=0), [num_tta, 1, 1, 1, 1])
    flat = tf.reshape(images_t, [num_tta * b, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    aug = data_augmentation(flat, training=True)
    p = model(aug, training=False)  # [T*B, C]
    p = tf.reshape(p, [num_tta, b, NUM_CLASSES])
    return tf.reduce_mean(tf.cast(p, tf.float32), axis=0)


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec(shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.string),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    ],
)
def _tta_predict_step(images, ids, num_tta):
    return _tta_mean_predict(images, num_tta), ids


tta_num_augmentations = 10
num_tta_tensor = tf.constant(tta_num_augmentations, dtype=tf.int32)

pred_ds = test_ds.map(
    lambda imgs, ids: _tta_predict_step(imgs, ids, num_tta_tensor),
    num_parallel_calls=AUTO,
    deterministic=True,
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
n_test = len(sample_sub)

tta_predictions = np.empty((n_test, NUM_CLASSES), dtype=np.float32)
tta_image_names = [None] * n_test

offset = 0
for batch_preds, batch_ids in pred_ds:
    bp = batch_preds.numpy()
    bi = batch_ids.numpy()
    bs = bp.shape[0]
    tta_predictions[offset : offset + bs] = bp
    tta_image_names[offset : offset + bs] = np.char.decode(
        bi.astype("S"), "utf-8"
    ).tolist()
    offset += bs

if offset != n_test:
    raise RuntimeError(f"Expected {n_test} test predictions, got {offset}")

print("TTA predictions shape:", tta_predictions.shape)
print("Collected image names:", len(tta_image_names))

pred_labels = np.argmax(tta_predictions, axis=1).astype(int)
print("Pred labels shape:", pred_labels.shape)

pred_df = pd.DataFrame({"image_id": tta_image_names, "label": pred_labels})

submission_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(0).astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created successfully: {submission_path}")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())

assert submission_path.endswith(".csv")
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)
assert submission_df["label"].between(0, NUM_CLASSES - 1).all()

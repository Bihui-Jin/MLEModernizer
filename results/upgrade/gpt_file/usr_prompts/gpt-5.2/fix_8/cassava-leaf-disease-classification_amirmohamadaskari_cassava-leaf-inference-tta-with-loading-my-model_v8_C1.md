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


def decode_train_example(example):
    example = tf.io.parse_single_example(example, _TRAIN_FEATURE_DESC)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    label = tf.cast(example["target"], tf.int32)
    return image, label


def decode_test_example(example):
    example = tf.io.parse_single_example(example, _TEST_FEATURE_DESC)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image, example["image_name"]




## === cell 4
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess_train(image, label):
    image = preprocess_input(image)
    return image, label


def preprocess_test(image, image_id):
    image = preprocess_input(image)
    return image, image_id




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
options.experimental_deterministic = True

try:
    options.threading.max_intra_op_parallelism = 0
except Exception:
    pass

try:
    rd_opts = tf.data.TFRecordDatasetOptions(compression_type=None)
except Exception:
    rd_opts = None


def _tfrecord_dataset(files):
    if rd_opts is not None:
        return tf.data.TFRecordDataset(files, num_parallel_reads=AUTO, options=rd_opts)
    return tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)


VAL_FRAC = 0.1
SHUFFLE_BUFFER = 8192

train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)
n_total = len(train_df)
n_val = max(1, int(n_total * VAL_FRAC))
n_train = n_total - n_val

base_all = _tfrecord_dataset(train_files).with_options(options)
base_all = base_all.map(decode_train_example, num_parallel_calls=AUTO)
base_all = base_all.map(preprocess_train, num_parallel_calls=AUTO)

train_base = base_all.take(n_train)
val_base = base_all.skip(n_train).take(n_val)

TRAIN_CACHE_PATH = "/kaggle/working/train_cache"
VAL_CACHE_PATH = "/kaggle/working/val_cache"
TEST_CACHE_PATH = "/kaggle/working/test_cache"

train_base = train_base.cache(TRAIN_CACHE_PATH)
val_base = val_base.cache(VAL_CACHE_PATH)

train_ds = (
    train_base.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

val_ds = val_base.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

test_ds = _tfrecord_dataset(test_files).with_options(options)
test_ds = (
    test_ds.map(decode_test_example, num_parallel_calls=AUTO)
    .map(preprocess_test, num_parallel_calls=AUTO)
    .cache(TEST_CACHE_PATH)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

steps_per_epoch = n_train // BATCH_SIZE
validation_steps = int(np.ceil(n_val / BATCH_SIZE))

print("Datasets created successfully!")
print("n_total:", n_total, "n_train:", n_train, "n_val:", n_val)
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
    preds = tf.zeros((tf.shape(images)[0], NUM_CLASSES), dtype=tf.float32)
    i = tf.constant(0, dtype=tf.int32)

    def cond(i, preds):
        return i < num_tta

    def body(i, preds):
        augmented = data_augmentation(images, training=True)
        p = model(augmented, training=False)
        preds = preds + tf.cast(p, tf.float32)
        return i + 1, preds

    _, preds_sum = tf.while_loop(
        cond, body, loop_vars=[i, preds], parallel_iterations=1
    )
    return preds_sum / tf.cast(num_tta, tf.float32)


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


@tf.function(reduce_retracing=True)
def _predict_testset_tta(ds, num_tta):
    preds_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
    ids_ta = tf.TensorArray(tf.string, size=0, dynamic_size=True, infer_shape=False)

    i = tf.constant(0, tf.int32)
    for batch_images, batch_ids in ds:
        mean_preds, out_ids = _tta_predict_step(batch_images, batch_ids, num_tta)
        preds_ta = preds_ta.write(i, mean_preds)
        ids_ta = ids_ta.write(i, out_ids)
        i += 1

    preds = preds_ta.concat()
    ids = ids_ta.concat()
    return preds, ids


tta_num_augmentations = 10
num_tta_tensor = tf.constant(tta_num_augmentations, dtype=tf.int32)

tta_predictions_tf, tta_image_names_tf = _predict_testset_tta(test_ds, num_tta_tensor)

tta_predictions = tta_predictions_tf.numpy()
tta_image_names_bytes = tta_image_names_tf.numpy()

if tta_predictions.size == 0:
    raise RuntimeError("No predictions were generated from test_ds (empty dataset).")

tta_predictions = np.asarray(tta_predictions)
if tta_predictions.ndim != 2 or tta_predictions.shape[1] != NUM_CLASSES:
    raise RuntimeError(f"Unexpected prediction shape: {tta_predictions.shape}")

tta_image_names = [b.decode("utf-8") for b in tta_image_names_bytes.tolist()]

print("TTA predictions shape:", tta_predictions.shape)
print("Collected image names:", len(tta_image_names))

pred_labels = np.argmax(tta_predictions, axis=1).astype(int)
print("Pred labels shape:", pred_labels.shape)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

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

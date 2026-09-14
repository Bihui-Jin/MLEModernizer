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

import tensorflow as tf
import numpy as np
import pandas as pd
import random

print("TF version:", tf.__version__)




## === cell 1
def get_strategy():
    """
    Detects and returns the best TensorFlow distribution strategy:
    - TPUStrategy for TPU(s)
    - MirroredStrategy for GPU(s)
    - Default strategy for CPU
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect TPU
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Using TPU strategy:", type(strategy).__name__)
    except (ValueError, tf.errors.NotFoundError):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            print("Using GPU strategy:", type(strategy).__name__)
        else:
            strategy = tf.distribute.get_strategy()
            print("No TPU/GPU found. Using default strategy:", type(strategy).__name__)
    print("REPLICAS:", strategy.num_replicas_in_sync)
    return strategy


strategy = get_strategy()



## === cell 2
AUTO = tf.data.AUTOTUNE
IMAGE_SIZE = (512, 512)
BATCH_SIZE_PER_REPLICA = 8
NUM_CLASSES = 5
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
print(f"Global Batch size: {BATCH_SIZE}")



## === cell 3
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

print("Data dir exists:", tf.io.gfile.exists(DATA_DIR))
print("Train CSV exists:", tf.io.gfile.exists(train_csv_path))
print("Sample sub exists:", tf.io.gfile.exists(sample_sub_path))



## === cell 4
MODEL_DIR = (
    "/kaggle/input/cassava-leaf-model/tensorflow2/default/1/final_model_cassava.keras"
)




## === cell 5
def seed_everthing(SEED=28):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    print(f"Global seed set to {SEED}")


seed_everthing(28)




## === cell 6
def decode_train_example(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    example = tf.io.parse_single_example(example, feature_description)

    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    label = tf.cast(example["label"], tf.int32)
    return image, label


def decode_test_example(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)

    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)

    return image, example["image_name"]




## === cell 7
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess(image, label):
    image = preprocess_input(image)
    return image, label




## === cell 8
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



## === cell 9
train_files = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
test_files = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))

if len(train_files) == 0:
    raise FileNotFoundError(f"No train TFRecords found in: {TRAIN_TFREC_DIR}")
if len(test_files) == 0:
    raise FileNotFoundError(f"No test TFRecords found in: {TEST_TFREC_DIR}")

train_files = sorted(train_files)
test_files = sorted(test_files)

raw_train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=AUTO).map(
    decode_train_example, num_parallel_calls=AUTO
)
raw_test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTO).map(
    decode_test_example, num_parallel_calls=AUTO
)

print("Train TFRecord files:", len(train_files))
print("Test TFRecord files:", len(test_files))



## === cell 10
train_count = int(pd.read_csv(train_csv_path).shape[0])
val_fraction = 0.1
val_count = int(train_count * val_fraction)
train_take = train_count - val_count

shuffled = raw_train_ds.shuffle(
    buffer_size=4096, seed=28, reshuffle_each_iteration=True
)

shuffled = shuffled.filter(lambda x, y: tf.greater_equal(y, 0))

train_ds = (
    shuffled.take(train_take)
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTO)
    .map(preprocess, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

val_ds = (
    shuffled.skip(train_take)
    .map(preprocess, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_ds = raw_test_ds.batch(BATCH_SIZE).prefetch(AUTO)

print("Datasets ready. Train/Val split:", train_take, "/", val_count)




## === cell 11
def build_fallback_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = inputs
    x = preprocess_input(x)
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
    )
    base.trainable = False  # fast + stable within 600s
    x = base.output
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


with strategy.scope():
    if tf.io.gfile.exists(MODEL_DIR):
        model = tf.keras.models.load_model(MODEL_DIR)
        print("Loaded external model:", MODEL_DIR)
    else:
        print("External model not found, training fallback model instead:", MODEL_DIR)
        model = build_fallback_model()
        model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)




## === cell 12
def apply_test_augmentation_batch(images):
    images = tf.cast(images, tf.float32)
    return tf.map_fn(
        lambda img: data_augmentation(img, training=True),
        images,
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    )


@tf.function
def tta_predict_batch(model, images, num_tta: int):
    preds = tf.zeros((tf.shape(images)[0], NUM_CLASSES), dtype=tf.float32)
    num_tta = tf.cast(num_tta, tf.int32)
    for _ in tf.range(num_tta):
        aug = apply_test_augmentation_batch(images)
        aug = preprocess_input(aug)
        preds += tf.cast(model(aug, training=False), tf.float32)
    return preds / tf.cast(num_tta, tf.float32)




## === cell 13
tta_num_augmentations = 10
all_image_ids = []
all_probs = []

for images, ids in test_ds:
    batch_ids = [x.decode("utf-8") for x in ids.numpy().tolist()]
    probs = tta_predict_batch(model, images, tta_num_augmentations).numpy()
    all_image_ids.extend(batch_ids)
    all_probs.append(probs)

all_probs = np.concatenate(all_probs, axis=0)  # [N,5]
pred_labels = np.argmax(all_probs, axis=1).astype(int)

print(
    "Predictions done. N:", len(pred_labels), "Unique labels:", np.unique(pred_labels)
)



## === cell 14
sample_sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"image_id": all_image_ids, "label": pred_labels})

submission_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    fallback_label = int(np.bincount(pred_labels, minlength=NUM_CLASSES).argmax())
    submission_df["label"] = submission_df["label"].fillna(fallback_label).astype(int)

submission_df["label"] = submission_df["label"].astype(int)
submission_df.to_csv("submission.csv", index=False)

print("Submission file created successfully! -> submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
print("Any NA labels:", submission_df["label"].isna().any())

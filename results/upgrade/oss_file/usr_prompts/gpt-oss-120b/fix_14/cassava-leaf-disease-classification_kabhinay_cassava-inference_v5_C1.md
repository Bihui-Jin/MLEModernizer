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
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)




## === cell 1
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_IMAGES_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_IMAGES_DIR = os.path.join(BASE_PATH, "train_images")




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)




## === cell 3
IMG_SIZE = (224, 224)
BATCH_SIZE = 128
NUM_CLASSES = 5
AUTOTUNE = tf.data.AUTOTUNE


def load_and_preprocess(path):
    """Read image file, decode JPEG, resize to IMG_SIZE."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img


np.random.seed(42)
perm = np.random.permutation(len(train_df))
split_idx = int(0.9 * len(train_df))
train_idx, val_idx = perm[:split_idx], perm[split_idx:]

train_paths = (
    train_df.loc[train_idx, "image_id"]
    .apply(lambda x: os.path.join(TRAIN_IMAGES_DIR, x))
    .values
)
train_labels = train_df.loc[train_idx, "label"].values.astype(np.int32)

val_paths = (
    train_df.loc[val_idx, "image_id"]
    .apply(lambda x: os.path.join(TRAIN_IMAGES_DIR, x))
    .values
)
val_labels = train_df.loc[val_idx, "label"].values.astype(np.int32)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(
    lambda p, l: (load_and_preprocess(p), l), num_parallel_calls=AUTOTUNE
)
train_ds = (
    train_ds.shuffle(buffer_size=1000, seed=42).batch(BATCH_SIZE).prefetch(AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(
    lambda p, l: (load_and_preprocess(p), l), num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)




## === cell 5
inputs = layers.Input(shape=IMG_SIZE + (3,))

x = layers.Rescaling(1.0 / 255)(inputs)

x = layers.RandomFlip("horizontal")(x)
x = layers.RandomRotation(0.1111)(x)  # ~20 degrees
x = layers.RandomTranslation(0.1, 0.1)(x)  # width/height shift
x = layers.RandomZoom(0.1)(x)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)




## === cell 6
model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds,
    callbacks=[callbacks.TerminateOnNaN()],
    verbose=2,
)




## === cell 7
test_df = pd.DataFrame()
test_df["image_id"] = os.listdir(TEST_IMAGES_DIR)

test_paths = (
    test_df["image_id"].apply(lambda x: os.path.join(TEST_IMAGES_DIR, x)).values
)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(lambda p: load_and_preprocess(p), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)




## === cell 8
preds = model.predict(test_ds, verbose=1)
predicted_class_indices = np.argmax(preds, axis=1)
predictions = predicted_class_indices.astype(int).tolist()




## === cell 9
submission = pd.DataFrame(
    {
        "image_id": (
            test_generator.filenames
            if "test_generator" in globals()
            else test_df["image_id"]
        ),
        "label": predictions,
    }
)
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

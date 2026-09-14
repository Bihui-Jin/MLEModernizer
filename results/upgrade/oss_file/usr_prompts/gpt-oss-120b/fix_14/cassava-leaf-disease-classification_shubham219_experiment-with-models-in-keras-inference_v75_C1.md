# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.3624962224236929

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.22123) has done: 'I remove the problematic imports and pip‑install commands, replace the missing EfficientNet package with the version bundled in TensorFlow, build and briefly train a small EfficientNetB0 model on the training images, and then generate predictions for the test set and write a proper `submission.csv`. This fixes the runtime errors, ensures a valid CSV is produced, and adds a lightweight model that should reach the required accuracy without altering the core competition logic.'
- What this solution (achieved 0.15321) has done: 'Optimized the data pipeline by increasing the number of loading workers and enabling multiprocessing for faster image preprocessing, and raised the batch size for training and validation (which keeps the number of epochs unchanged). These changes speed up I/O and reduce the number of gradient steps per epoch without altering the model architecture, loss, or evaluation logic, preserving result accuracy.'
- What this solution (achieved 0.10688) has done: 'The update replaces the slow `ImageDataGenerator` pipelines with efficient `tf.data` pipelines that decode and preprocess images in parallel, use `prefetch`, and compute exact step counts, eliminating unnecessary Python‑level loops while keeping the model architecture and training schedule unchanged. This speeds up data loading dramatically, allowing the full 6‑epoch training and prediction to finish well within the 600 s limit.'

# 9. Code solution

## === cell 0
import os
import tempfile

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import glob

SEED = 42
DEBUG = False
tf.random.set_seed(SEED)

tf.config.threading.set_inter_op_parallelism_threads(tf.data.AUTOTUNE)
tf.config.threading.set_intra_op_parallelism_threads(tf.data.AUTOTUNE)




## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_img_dir = "../input/cassava-leaf-disease-classification/train_images/"

df_train = pd.read_csv(train_csv_path)
df_train["label"] = df_train["label"].astype(int)
df_train["path"] = train_img_dir + df_train["image_id"]

train_df, val_df = train_test_split(
    df_train,
    test_size=0.2,
    stratify=df_train["label"],
    random_state=SEED,
)


def _parse_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)
    return img, tf.cast(label, tf.int32)


def make_dataset(df, batch_size=256, shuffle=False, cache_path=None):
    """
    Build a tf.data pipeline.
    Disk caching is optional; we skip it to avoid massive write‑read overhead.
    """
    paths = df["path"].values
    labels = df["label"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    if cache_path is not None:
        ds = ds.cache(cache_path)  # used only if explicitly requested
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, batch_size=256, shuffle=True)
val_ds = make_dataset(val_df, batch_size=256, shuffle=False)




## === cell 2
tf.keras.backend.clear_session()  # ensure a fresh graph

base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False  # stage‑1: freeze backbone

x = GlobalAveragePooling2D()(base.output)
output = Dense(5, activation="softmax")(x)
model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

stage1_epochs = 5 if not DEBUG else 1
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=stage1_epochs,
    verbose=1,
)

base.trainable = True
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

stage2_epochs = 5 if not DEBUG else 1
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=stage2_epochs,
    verbose=1,
)




## === cell 3
test_img_dir = "../input/cassava-leaf-disease-classification/test_images/"
test_images = glob.glob(os.path.join(test_img_dir, "*.jpg"))
df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_dataset(df, batch_size=256):
    paths = df["path"].values
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda p: preprocess_input(
            tf.image.resize(
                tf.image.decode_jpeg(tf.io.read_file(p), channels=3), [224, 224]
            )
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_dataset(df_test, batch_size=256)




## === cell 4
pred_probs = model.predict(test_ds, verbose=1)
pred_labels = np.argmax(pred_probs, axis=-1)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split("/").str[-1]
final_submission["label"] = pred_labels
submission_csv = final_submission[["image_id", "label"]]

submission_path = "submission.csv"
submission_csv.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

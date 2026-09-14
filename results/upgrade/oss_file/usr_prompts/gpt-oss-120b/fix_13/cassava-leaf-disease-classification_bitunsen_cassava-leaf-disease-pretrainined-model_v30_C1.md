# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8730734360834089

# 6. Current score

0.4361

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'The script is re‑structured to replace the Python‑level `ImageDataGenerator` pipeline with a TensorFlow `tf.data` pipeline that performs decoding, resizing, scaling and the same augmentations using TF’s fast, parallel ops. This removes the heavy per‑image Python processing and multiprocessing overhead, keeping the same model architecture, training epochs, and loss, while preserving deterministic seeds. The test set is also read via `tf.data` for fast batch prediction.'
- What this solution (achieved 0.4361) has done: 'I fixed the protobuf import error, corrected the data‑augmentation layers so they are instantiated once (avoiding variable‑creation inside `tf.data` maps), and updated the cell ordering to start at 1. These changes let the dataset pipeline build correctly, run the training loop, and write a proper `submission.csv` file, moving the model’s performance far closer to the target score.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

SEED = 42
NUM_WORKERS = min(4, max(1, os.cpu_count() - 1))
tf.config.threading.set_intra_op_parallelism_threads(NUM_WORKERS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_WORKERS)

np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")




## === cell 2
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as f:
    label_map = json.load(f)




## === cell 3
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(str)




## === cell 4
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 256
NUM_CLASSES = 5

train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
split_idx = int(0.9 * len(train_df))
train_split = train_df.iloc[:split_idx]
val_split = train_df.iloc[split_idx:]

AUTOTUNE = tf.data.AUTOTUNE


def path_label_from_df(df, directory):
    paths = df["image_id"].apply(lambda x: os.path.join(directory, x)).values
    labels = df["label"].astype(int).values
    return paths, labels


train_paths, train_labels = path_label_from_df(train_split, TRAIN_DIR)
val_paths, val_labels = path_label_from_df(val_split, TRAIN_DIR)

flip_layer = RandomFlip(mode="horizontal_and_vertical")
rotation_layer = RandomRotation(0.125)
zoom_layer = RandomZoom(0.4)
translation_layer = RandomTranslation(0.1, 0.1)


def decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    return img


def preprocess_train(path, label):
    img = decode_resize(path)
    img = flip_layer(img)
    img = rotation_layer(img)
    img = zoom_layer(img)
    img = translation_layer(img)
    return img, tf.one_hot(label, NUM_CLASSES)


def preprocess_val(path, label):
    img = decode_resize(path)
    return img, tf.one_hot(label, NUM_CLASSES)


train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer=1024, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2386968905.py in <cell line: 0>()
     54     tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
     55     .map(preprocess_train, num_parallel_calls=AUTOTUNE)
---> 56     .shuffle(buffer=1024, seed=SEED)
     57     .batch(BATCH_SIZE)
     58     .prefetch(AUTOTUNE)

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 5
base_model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)




## === cell 6
EPOCHS = 3
model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    verbose=2,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1311508042.py in <cell line: 0>()
      1 EPOCHS = 3
      2 model.fit(
----> 3     train_dataset,
      4     epochs=EPOCHS,
      5     validation_data=val_dataset,

NameError: name 'train_dataset' is not defined

## === cell 7
test_filenames = [
    f
    for f in sorted(os.listdir(TEST_DIR))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
test_df = pd.DataFrame({"image_id": test_filenames})
test_paths = test_df["image_id"].apply(lambda x: os.path.join(TEST_DIR, x)).values


def preprocess_test(path):
    img = decode_resize(path)
    return img


test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(preprocess_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

probs = model.predict(test_dataset, verbose=0)
predictions = np.argmax(probs, axis=1)
test_df["label"] = predictions
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)




## === cell 8
submission = pd.read_csv(submission_path)
submission.head(3)

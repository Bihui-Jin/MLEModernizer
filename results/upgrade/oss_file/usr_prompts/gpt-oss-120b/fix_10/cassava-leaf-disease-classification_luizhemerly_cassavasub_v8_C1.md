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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.8176186158960411

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Implemented a full end‑to‑end pipeline that avoids the broken TFRecord loading and legacy SavedModel issues.  
* Reads the CSV metadata and image files directly from the provided directories.  
* Builds a lightweight EfficientNet‑B0 model, trains it on 90 % of the data, validates on 10 %.  
* Generates predictions for the test set, preserves the original image order, and writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I patch the protobuf incompatibility by adding a fallback `GetPrototype` method before importing TensorFlow, then fine‑tune the EfficientNetB0 backbone (unfreeze its last few layers) and train for more epochs to lift the validation accuracy toward the target. The rest of the pipeline and submission format stay unchanged.'
- What this solution (achieved 0.62182) has done: 'The changes add mixed‑precision and XLA to let TensorFlow use the GPU more efficiently, and cache the pre‑processed images on disk so they are read only once instead of every epoch. Both optimizations keep the exact model, loss, epochs and data‑splits, merely accelerating I/O and computation while preserving numerical results.'
- What this solution (achieved 0.11024) has done: 'I add a ModelCheckpoint callback to keep the best‐validation‑accuracy weights and load them before predicting, and extend training to 30 epochs so the model has more opportunity to improve. These changes preserve the exact architecture, loss, optimizer and augmentation while giving a higher‑accuracy model that should move the validation score closer to the target. No other logic is altered, and the submission file is still written in the same format.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_ROOT = os.path.join("data", "cassava-leaf-disease-classification")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_df = pd.read_csv(TRAIN_CSV)
train_paths = [os.path.join(TRAIN_IMG_DIR, img_id) for img_id in train_df["image_id"]]
train_labels = train_df["label"].values


def _process_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


train_paths_split, val_paths_split, train_labels_split, val_labels_split = (
    train_test_split(
        train_paths,
        train_labels,
        test_size=0.1,
        stratify=train_labels,
        random_state=42,
    )
)

train_cache_path = "/tmp/train_cache"
val_cache_path = "/tmp/val_cache"
os.makedirs(train_cache_path, exist_ok=True)
os.makedirs(val_cache_path, exist_ok=True)

options = tf.data.Options()
options.experimental_deterministic = False

train_ds = tf.data.Dataset.from_tensor_slices((train_paths_split, train_labels_split))
train_ds = (
    train_ds.map(_process_image, num_parallel_calls=AUTOTUNE, deterministic=False)
    .cache(train_cache_path)
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
    .with_options(options)
)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths_split, val_labels_split))
val_ds = (
    val_ds.map(_process_image, num_parallel_calls=AUTOTUNE, deterministic=False)
    .cache(val_cache_path)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
    .with_options(options)
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)
)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)
x = tf.keras.layers.RandomContrast(0.1)(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=3, verbose=1, min_lr=1e-6
)

checkpoint_path = "/tmp/best_model.weights.h5"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,  # a bit longer training for higher accuracy
    verbose=2,
    callbacks=[lr_callback, checkpoint_cb],
)

model.load_weights(checkpoint_path)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3646447475.py in <cell line: 0>()
     41 
     42 model.fit(
---> 43     train_ds,
     44     validation_data=val_ds,
     45     epochs=40,  # a bit longer training for higher accuracy

NameError: name 'train_ds' is not defined

## === cell 2
test_filenames = sorted(
    [f for f in tf.io.gfile.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
)
test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_filenames]


def _load_test_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    _load_test_image, num_parallel_calls=AUTOTUNE, deterministic=False
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

probabilities = model.predict(test_ds, verbose=0)
predictions = np.argmax(probabilities, axis=1).astype(int)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_55/463276704.py in <cell line: 0>()
      1 test_filenames = sorted(
----> 2     [f for f in tf.io.gfile.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
      3 )
      4 test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_filenames]
      5 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/lib/io/file_io.py in list_directory_v2(path)
    766   """
    767   if not is_directory(path):
--> 768     raise errors.NotFoundError(
    769         node_def=None,
    770         op=None,

NotFoundError: Could not find directory data/cassava-leaf-disease-classification/test_images

## === cell 3
submission = pd.DataFrame({"image_id": test_filenames, "label": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(submission.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3640045183.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": test_filenames, "label": predictions})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 
      5 print(submission.head())

NameError: name 'test_filenames' is not defined

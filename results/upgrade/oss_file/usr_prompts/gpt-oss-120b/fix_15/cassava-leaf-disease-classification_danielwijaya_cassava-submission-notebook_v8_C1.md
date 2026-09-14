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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8898458748866727

# 6. Current score

0.45291

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the broken imports, stop trying to load missing pre‑trained .h5 models, and instead build a lightweight EfficientNet‑B0 model directly in the notebook. The script read the provided TFRecord files, create a train/validation split, fine‑tune the model for a few epochs (enough to reach ~0.80 accuracy, which is within the required tolerance), predict the test set, and finally write a correctly‑formatted `submission.csv` file.'
- What this solution (achieved 0.11883) has done: 'I set the protobuf implementation to pure‑Python before importing TensorFlow to avoid the `MessageFactory` error, remove the unnecessary filesystem walk, unfreeze the EfficientNet backbone for better fine‑tuning, and increase training epochs to improve validation accuracy toward the target score. These changes fix the runtime crash and modestly boost model performance while preserving the original workflow.'
- What this solution (achieved 0.22085) has done: 'I rearrange the imports so the protobuf environment variable is set *before* any library (including pandas) can load protobuf, eliminating the `MessageFactory` error. Additionally, I run the EfficientNet backbone in training mode (`training=True`) when building the model so its batch‑norm layers update during fine‑tuning, which should raise validation accuracy toward the target without altering the core architecture. The rest of the workflow remains unchanged.'
- What this solution (achieved 0.11584) has done: 'I fixed the protobuf import error by setting both required environment variables before any imports, corrected the TFRecord parsing key from a non‑existent “target” to the actual “label”, added a shuffle before the train/validation split to ensure a representative validation set, and raised the number of training epochs modestly to improve accuracy while keeping the original model architecture untouched. These minimal changes resolve the runtime crash and should move the validation accuracy much closer to the target score, and the script now writes a proper `submission.csv`.'
- What this solution (achieved 0.15359) has done: 'Optimized the data pipeline by caching decoded images to disk, enabled mixed‑precision training and XLA JIT compilation to accelerate model fitting, and kept all original model architecture and training epochs unchanged. These changes reduce expensive I/O and compute overhead while preserving exact semantics of the original solution.'
- What this solution (achieved 0.47982) has done: 'Optimized the input pipeline by casting images to float16 early and caching the decoded images in memory, eliminating repeated JPEG reads for every epoch while keeping the same model architecture and training regimen. This reduces I/O overhead dramatically and fits within the 600 s limit without altering any learning logic.'
- What this solution (achieved 0.45291) has done: 'The fix sets the protobuf environment variable correctly (removing the version flag that caused the import error) and adds EfficientNet‑specific preprocessing to the image pipeline, which improves model accuracy without altering the core architecture. The rest of the workflow stays the same, and the script now writes a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras import mixed_precision

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.data.Options().experimental_deterministic = False

mixed_precision.set_global_policy("mixed_float16")

import numpy as np
import pandas as pd
import re, random

print("TensorFlow version:", tf.__version__)
print("Data root exists:", tf.io.gfile.isdir("/kaggle/input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = [224, 224]  # EfficientNetB0 default
NUM_CLASSES = 5
GCS_PATH = "/kaggle/input/cassava-leaf-disease-classification"

from tensorflow.keras.applications.efficientnet import preprocess_input


def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0‑1 float
    img = tf.image.resize(img, IMAGE_SIZE)
    img = preprocess_input(img)
    return img


def load_image(path, label):
    img_bytes = tf.io.read_file(path)
    img = decode_img(img_bytes)
    return img, label


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label


train_csv_path = os.path.join(GCS_PATH, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_image_paths = (
    train_df["image_id"]
    .apply(lambda x: os.path.join(GCS_PATH, "train_images", x))
    .values
)
train_labels = train_df["label"].astype(np.int32).values

full_train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
full_train_ds = full_train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
full_train_ds = full_train_ds.map(
    lambda img, lbl: (tf.cast(img, tf.float16), lbl), num_parallel_calls=AUTOTUNE
)
full_train_ds = full_train_ds.cache()  # eliminates repeated disk reads

options = tf.data.Options()
options.experimental_deterministic = False
full_train_ds = full_train_ds.with_options(options)

full_train_ds = full_train_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=False)

train_split = int(0.95 * len(train_image_paths))
train_ds = (
    full_train_ds.take(train_split)
    .map(augment, num_parallel_calls=AUTOTUNE)  # augmentation per epoch
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
val_ds = full_train_ds.skip(train_split).batch(BATCH_SIZE).prefetch(AUTOTUNE)




## === cell 2
tf.config.optimizer.set_jit(True)

strategy = (
    tf.distribute.OneDeviceStrategy(device="/GPU:0")
    if gpus
    else tf.distribute.get_strategy()
)

with strategy.scope():
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = True

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

model.summary()




## === cell 3
EPOCHS = 30  # increased epochs for better learning
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)




## === cell 4
sample_sub_path = os.path.join(GCS_PATH, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_image_paths = (
    sample_sub["image_id"]
    .apply(lambda x: os.path.join(GCS_PATH, "test_images", x))
    .values
)

test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_ds = (
    test_ds.map(lambda p: decode_img(tf.io.read_file(p)), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)  # removed in‑memory cache

probabilities = model.predict(test_ds)
predictions = np.argmax(probabilities, axis=-1).astype(
    int
)  # integer labels as required

submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": predictions})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path}")
print(submission.head())

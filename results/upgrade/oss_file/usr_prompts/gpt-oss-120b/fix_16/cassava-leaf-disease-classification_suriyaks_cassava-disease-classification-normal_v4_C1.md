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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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
tf_keras==2.18.0

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

0.6128

# 6. Current score

0.73057

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.7216) has done: 'I fixed the indexing error, corrected the import of `ImageDataGenerator`, replaced the broken custom CNN with a small pretrained MobileNetV2 model (keeps the overall training flow unchanged), updated the early‑stopping monitor name, and rewrote the inference loop so that a proper `submission.csv` is written to the working directory.'
- What this solution (achieved 0.72608) has done: 'I fix the import error by using `tensorflow.keras.preprocessing.image.ImageDataGenerator` instead of the incompatible standalone Keras version, and renumber the notebook cells to start at 1 as required. This resolves the `NameError` for `ImageDataGenerator`, allowing the data pipelines, model training, and inference to run end‑to‑end and produce a valid `submission.csv` file.'
- What this solution (achieved 0.73281) has done: 'I replace the problematic standalone `keras` import with the TensorFlow‑integrated version (`tensorflow.keras`). This eliminates the protobuf conflict that caused the `AttributeError` while keeping all subsequent Keras calls unchanged, so the model, training, and submission logic remain the same and the script produce a valid `submission.csv`.'
- What this solution (achieved 0.72534) has done: 'I fixed the protobuf import error by setting the environment variable before loading TensorFlow and replaced all standalone‑Keras imports with the TensorFlow‑integrated `tf.keras` equivalents, preserving the original model and training flow. The cells are renumbered to start at 1, and the script now runs end‑to‑end and writes a correct `submission.csv` while keeping the achieved score above the target.'
- What this solution (achieved 0.72534) has done: 'I reordered the imports in the first cell so that the protobuf‑related environment variable is set **before any library (especially TensorFlow) that may load protobuf**. This eliminates the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` and lets the script run end‑to‑end, producing a valid `submission.csv` while preserving the existing model and training logic.'
- What this solution (achieved 0.72534) has done: 'I moved the protobuf‑environment setting to the very first lines and ensured it runs before any TensorFlow import, adding a secondary legacy flag for extra safety. The rest of the pipeline stays unchanged, so the model, training, and submission logic are preserved and the score (already above the target) remains stable while the script now completes without the protobuf AttributeError and writes a proper `submission.csv`.'
- What this solution (achieved 0.72571) has done: 'I merge the environment‑variable setup with the imports and renumber the notebook cells so they start at 1, which resolves the protobuf import error and ensures the script runs end‑to‑end. No changes are made to the model or training logic, so the achieved score (already above the target) remains unchanged while a proper `submission.csv` is written.'
- What this solution (achieved 0.72496) has done: 'I moved the protobuf‑compatibility environment variables to the very top of the script and ensured they are set **before any TensorFlow import**. All imports are now ordered so that `tensorflow` (and `ImageDataGenerator`) are loaded after the variables, eliminating the `MessageFactory` AttributeError. The rest of the pipeline—including data generators, the MobileNetV2 model, training, prediction, and CSV writing—remains unchanged, preserving the existing score while guaranteeing a valid `submission.csv` is produced.'
- What this solution (achieved 0.72085) has done: 'Implemented a robust data pipeline that avoids the protobuf import error by replacing `ImageDataGenerator` with a `tf.data` pipeline, while preserving the original model architecture and training setup. Added proper environment variable handling, deterministic train/validation split, and streamlined test inference to ensure a correct `submission.csv` is produced.'
- What this solution (achieved 0.71861) has done: 'The fix moves the protobuf‑compatibility environment variables to the very top of the script, before *any* imports, ensuring TensorFlow loads without the `MessageFactory` error. No changes are made to the model or training logic, preserving the existing high score while guaranteeing a valid `submission.csv` is written.'
- What this solution (achieved 0.72347) has done: 'I moved the protobuf‑compatibility environment variables to the very top of the script—before any other imports—so TensorFlow loads without the `MessageFactory` AttributeError. I also renumbered the notebook cells to start at 1 as required. No other logic changes are made, preserving the existing model and score (which already exceeds the target).'
- What this solution (achieved 0.71338) has done: 'I moved the protobuf‑compatibility environment variables to the very top of the script and imported TensorFlow immediately afterwards, before any other libraries that might trigger protobuf loading. This prevents the `MessageFactory` attribute error while leaving the model, data pipeline, training, and submission logic unchanged, so the existing high score is preserved and a valid `submission.csv` is written.'
- What this solution (achieved 0.72347) has done: 'I renumber the notebook so the first cell (with the protobuf‑compatibility environment variables) is executed before any TensorFlow import, which prevents the `MessageFactory` error. I also write the submission to the Kaggle working directory (`/kaggle/working/submission.csv`) to guarantee the file is saved where Kaggle expects it. No other logic is changed, preserving the model and training pipeline while ensuring a valid CSV is produced.'
- What this solution (achieved 0.73057) has done: 'I renumber the cells so they start at 1 (Kaggle notebooks expect the first executable cell to be 1) and keep the protobuf‑compatibility environment variables at the very top before any imports. No other logic is changed, preserving the model, training, and inference pipeline while still producing a valid `submission.csv`. This minor edit resolves the cell‑indexing issue without affecting the already high score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_LEGACY"] = "1"

import tensorflow as tf
import json
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

print("Environment ready")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
train_path = os.path.join(BASE, "train_images")
test_path = os.path.join(BASE, "test_images")
train_csv = os.path.join(BASE, "train.csv")
sample_sub = os.path.join(BASE, "sample_submission.csv")



## === cell 2
train_df = pd.read_csv(train_csv)
print("train shape:", train_df.shape)



## === cell 3
train_df["label"] = train_df["label"].astype(int)

train_df["file_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(train_path, x)
)

train_files, val_files, train_labels, val_labels = train_test_split(
    train_df["file_path"].values,
    train_df["label"].values,
    test_size=0.25,
    random_state=42,
    stratify=train_df["label"].values,
)


def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return img


def _preprocess(path, label):
    return _load_image(path), tf.one_hot(label, depth=5)


AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    .map(_preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(1000, seed=42)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    .map(_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 4
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False

model = Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(5, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 5
early_stop = EarlyStopping(
    monitor="val_accuracy", patience=2, restore_best_weights=True
)

model.fit(
    train_ds,
    epochs=8,
    validation_data=val_ds,
    callbacks=[early_stop],
    verbose=1,
)



## === cell 6
test_df = pd.read_csv(sample_sub)[["image_id"]]
test_df["file_path"] = test_df["image_id"].apply(lambda x: os.path.join(test_path, x))

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_df["file_path"].values)
    .map(lambda p: _load_image(p), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 7
preds = model.predict(test_ds, verbose=1)
pred_labels = np.argmax(preds, axis=1)



## === cell 8
submission = pd.DataFrame(
    {"image_id": test_df["image_id"], "label": pred_labels.astype(int)}
)
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(f"Submission written to {out_path}")

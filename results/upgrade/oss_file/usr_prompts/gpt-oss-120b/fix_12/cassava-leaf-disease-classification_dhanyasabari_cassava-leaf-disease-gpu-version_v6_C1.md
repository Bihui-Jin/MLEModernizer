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

0.6900876397703234

# 6. Current score

0.76756

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10164) has done: 'The fix sets the protobuf implementation to the pure‑Python version before importing TensorFlow (avoiding the `MessageFactory` error) and replaces the manual reshape in `decode_image` with a proper resize to the target image size, eliminating the tensor‑size mismatch during training and inference. No other logic is changed, preserving the original model and workflow while ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.11584) has done: 'The fix moves the protobuf environment setting to the very top of the script (before any imports) to prevent the `MessageFactory` error, and modestly increases training epochs so the model can achieve a higher validation accuracy, moving the score toward the target. No core logic or model architecture is changed.'
- What this solution (achieved 0.76943) has done: 'I add the missing TensorFlow import (setting the protobuf implementation beforehand) and complete the pipeline: load the CSVs, build TF‑datasets with the provided decode_image, create a simple EfficientNet‑B0 classifier, train it, run inference on the test images, and finally write a correctly‑named submission.csv with the required columns. These fixes remove the NameError, prevent the protobuf MessageFactory crash, and ensure a valid submission file is produced, moving the score toward the target.'
- What this solution (achieved 0.7728) has done: 'The fix moves the protobuf environment flag to the absolute first lines (before any import) to guarantee the pure‑Python protobuf implementation is selected and eliminates the `MessageFactory` import error. The rest of the pipeline is unchanged; it still builds the EfficientNet‑B0 model, trains, predicts, and writes a correctly‑named `submission.csv` with the required columns, preserving the achieved score (which already exceeds the target).'
- What this solution (achieved 0.77055) has done: 'The fix moves the protobuf environment flag to the very first lines (before any imports) so TensorFlow loads without the `MessageFactory` error, and the rest of the pipeline remains unchanged, guaranteeing a valid `submission.csv` and preserving the score that already exceeds the target.'
- What this solution (achieved 0.76981) has done: 'The fix adds the missing protobuf version environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3` right before any TensorFlow import. This prevents the `MessageFactory` attribute error while keeping the original EfficientNet‑B0 pipeline unchanged, allowing the script to run end‑to‑end and produce a valid `submission.csv` with the same high score.'
- What this solution (achieved 0.77205) has done: 'The fix moves the protobuf environment configuration to the very first lines (before any imports) and renumbers the notebook cells to start at 1, ensuring the script runs without the `MessageFactory` error and still produces a correct `submission.csv` while keeping the high score already above the target.'
- What this solution (achieved 0.77093) has done: 'The script failed during TensorFlow import due to a protobuf incompatibility. We move the protobuf‑environment configuration to the very top and, to guarantee it takes effect even if TensorFlow was pre‑loaded, we explicitly reload the TensorFlow module after setting the flags. No other logic is changed, preserving the existing model and training pipeline while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.76756) has done: 'We move the protobuf environment configuration to the very first lines (before any imports) and drop the reload logic, ensuring TensorFlow loads without the `MessageFactory` error. The rest of the pipeline remains unchanged, preserving the high validation accuracy (0.77 > target) and guaranteeing a correctly‑named `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "/kaggle/input/cassava-leaf-disease-classification"
BATCH_SIZE = 16 * 4  # 64 – fits comfortably on most kernels
IMAGE_SIZE = [224, 224]  # smaller size speeds up training and inference
CLASSES = ["0", "1", "2", "3", "4"]
EPOCHS = 20  # increased epochs for better accuracy




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def decode_image(image):
    """Decode JPEG bytes, resize, and apply EfficientNet preprocessing."""
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image




## === cell 2
train_csv_path = os.path.join(GCS_PATH, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(GCS_PATH, "train_images", x)
)

train_df["label_int"] = train_df["label"].astype(int)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_df["image_path"].values,
    train_df["label_int"].values,
    test_size=0.2,
    stratify=train_df["label_int"],
    random_state=42,
)


def make_dataset(image_paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(
        lambda path, label: (decode_image(tf.io.read_file(path)), label),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_paths, train_labels)
val_ds = make_dataset(val_paths, val_labels)



## === cell 3
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=IMAGE_SIZE + [3],
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze base

inputs = tf.keras.Input(shape=IMAGE_SIZE + [3])
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)



## === cell 4
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
)



## === cell 5
sample_sub_path = os.path.join(GCS_PATH, "sample_submission.csv")
test_df = pd.read_csv(sample_sub_path)

test_df["image_path"] = test_df["image_id"].apply(
    lambda x: os.path.join(GCS_PATH, "test_images", x)
)

test_ds = tf.data.Dataset.from_tensor_slices(test_df["image_path"].values)
test_ds = test_ds.map(
    lambda path: decode_image(tf.io.read_file(path)),
    num_parallel_calls=AUTOTUNE,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

pred_probs = model.predict(test_ds, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1).astype(str)

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

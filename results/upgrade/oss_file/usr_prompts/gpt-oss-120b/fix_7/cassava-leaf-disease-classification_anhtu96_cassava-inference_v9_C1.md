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

0.6334239951647024

# 6. Current score

0.78326

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.13789) has done: 'I fix the protobuf import error by setting the appropriate environment variable before importing TensorFlow, add a safe fallback that builds a simple EfficientNet‑B1 model when the expected *.h5 file is missing, and adjust the inference loop so that image IDs and predictions are collected correctly. These changes unblock the script, ensure a valid `submission.csv` is written, and keep the core modeling approach unchanged.'
- What this solution (achieved 0.82025) has done: 'I replace the TFRecord‑based pipeline with a direct image‑loading pipeline that avoids the protobuf incompatibility, add a short training loop for the EfficientNet‑B1 fallback model (so predictions are no longer random), and keep the rest of the logic unchanged. This fixes the runtime error, produces a valid `submission.csv`, and improves accuracy toward the target score.'
- What this solution (achieved 0.81652) has done: 'The changes add on‑disk caching to the training and test pipelines so images are decoded and resized only once, and batch the test dataset to reduce the number of model calls.  Cache files are written to `/tmp` to avoid RAM overflow, and the inference loop is updated to handle batched predictions while preserving the original ordering.  These optimizations keep the model architecture and training exactly the same, but dramatically cut I/O and per‑epoch preprocessing time, keeping total runtime under the 600 s limit.'
- What this solution (achieved 0.78326) has done: 'I added a small compatibility patch that supplies the missing `GetPrototype` method to `google.protobuf.message_factory.MessageFactory` before TensorFlow is imported. This resolves the `AttributeError` shown in cell 0, allowing the rest of the notebook to run unchanged and produce a valid `submission.csv`. No core modeling logic was altered, and the existing score (which is already above the target) is retained.'

# 9. Code solution

## === cell 0
import os

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):
        MessageFactory.GetPrototype = MessageFactory.GetMessageClass
except Exception:
    pass

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = "/kaggle/input/"
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = (224, 224)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _load_and_preprocess(image_path, label=None):
    """Read an image file, decode, resize and apply EfficientNet preprocessing."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    if label is None:
        return image
    return image, label


def get_train_dataset(csv_path, img_dir, batch_size=BATCH_SIZE):
    """Create a tf.data.Dataset for training from the CSV and image directory.
    Adds on‑disk caching to avoid re‑reading images each epoch."""
    df = pd.read_csv(csv_path)
    img_paths = tf.strings.join([img_dir, df["image_id"]], separator="/")
    labels = df["label"].values
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(filename="/tmp/train_cache")
    ds = ds.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)
    return ds, len(df)


def get_test_dataset(img_dir, batch_size=1):
    """Create a tf.data.Dataset for inference from the test image directory.
    Uses on‑disk caching and allows configurable batch size."""
    img_files = tf.io.gfile.glob(os.path.join(img_dir, "*.jpg"))
    img_files = sorted(img_files)  # deterministic order
    ds = tf.data.Dataset.from_tensor_slices(img_files)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(filename="/tmp/test_cache")
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, img_files




## === cell 2
train_csv_path = os.path.join(ROOT_DIR, "cassava-leaf-disease-classification/train.csv")
train_img_dir = os.path.join(
    ROOT_DIR, "cassava-leaf-disease-classification/train_images"
)
test_img_dir = os.path.join(ROOT_DIR, "cassava-leaf-disease-classification/test_images")

train_dataset, train_len = get_train_dataset(train_csv_path, train_img_dir)
TEST_BATCH_SIZE = 32
test_dataset, test_file_paths = get_test_dataset(
    test_img_dir, batch_size=TEST_BATCH_SIZE
)




## === cell 3
def build_model():
    """Build the EfficientNet‑B1 fallback model (no external weights)."""
    base = tf.keras.applications.EfficientNetB1(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    output = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model()




## === cell 4
EPOCHS = 3
model.fit(train_dataset, epochs=EPOCHS, verbose=1)




## === cell 5
test_IDs = []
predictions = []

for batch_index, batch in enumerate(test_dataset):
    start = batch_index * TEST_BATCH_SIZE
    end = start + tf.shape(batch)[0]
    batch_paths = test_file_paths[start:end]

    probs = model(batch, training=False)  # direct call, returns a tensor
    pred_labels = tf.argmax(probs, axis=1).numpy()
    predictions.extend(pred_labels.tolist())
    test_IDs.extend([os.path.basename(p) for p in batch_paths])




## === cell 6
submission_df = pd.DataFrame({"image_id": test_IDs, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(submission_df)} rows.")

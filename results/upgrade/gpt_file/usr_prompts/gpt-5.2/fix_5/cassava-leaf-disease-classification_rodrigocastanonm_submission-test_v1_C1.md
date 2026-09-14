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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
pillow==11.3.0
protobuf==6.33.0
seaborn==0.12.2
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

0.6161982472045935

# 6. Current score

0.74178

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08782) has done: 'I remove the failing `keras` import that’s triggering the protobuf `MessageFactory.GetPrototype` crash by switching to `tf.keras` for all model operations (compatible with TF 2.18 in this environment). Since the referenced external weights file doesn’t exist, I replace that load step with a minimal, standard `tf.keras.applications.ResNet50` classifier head so the notebook can run end-to-end without extra inputs. I also fix preprocessing (RGB conversion + ResNet50 `preprocess_input`) and make sure predictions are generated for every test image with consistent ordering to avoid the length-mismatch error when building the submission. Finally, I write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.78662) has done: 'I fix the protobuf crash by removing the standalone `keras` import/usage and consistently using `tf.keras` (which is compatible with TF 2.18 here). To move accuracy up toward your target with minimal core-logic change, I add a small training step on `train.csv` images (same ResNet50 backbone + same softmax head), then use the trained model for test predictions. I also keep preprocessing aligned with ResNet50 (`preprocess_input`) and speed up inference by batching via a `tf.data` pipeline while preserving the sample submission ordering. The script still write a valid `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.80157) has done: 'The protobuf `MessageFactory.GetPrototype` crash is happening at import-time in this environment due to an incompatible protobuf runtime, so the main fix is to pin protobuf’s Python implementation *before* importing TensorFlow. After that, the rest of your pipeline (ResNet50 backbone, preprocessing, 2-epoch train, batched inference, submission writing) can remain unchanged, which should keep score behavior essentially the same (no intentional score-tuning since you’re already above the target band). I also add a small safety check to ensure all referenced image paths exist to prevent silent empty batches, and keep the output as a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.74178) has done: 'The import-time crash happens before your training code runs because TF 2.18 is hitting an incompatible protobuf symbol (`MessageFactory.GetPrototype`) in this environment. I fix this by forcing the pure-Python protobuf implementation and pinning a compatible protobuf version at runtime *before* importing TensorFlow, which is the minimal change that unblocks execution while keeping your model/training logic identical. I also renumber the cells to start at 1 (your current notebook starts at cell 0) to match the required format, but won’t change any training/inference semantics so your score behavior should remain essentially the same. Finally, I keep the same submission creation path and ensure `submission.csv` is always written with the correct columns.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as pb_ver
except Exception:
    pb_ver = None


def _major(v):
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return None


if _major(pb_ver) is None or _major(pb_ver) >= 5:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )

import numpy as np
import pandas as pd
import tensorflow as tf

WORK_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(WORK_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(WORK_DIR, "train_images")
TEST_IMG_DIR = os.path.join(WORK_DIR, "test_images")
SAMPLE_SUB = os.path.join(WORK_DIR, "sample_submission.csv")

assert os.path.exists(WORK_DIR), f"Missing dataset at {WORK_DIR}"
assert os.path.exists(TRAIN_CSV), f"Missing train csv at {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train images dir at {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test images dir at {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample submission at {SAMPLE_SUB}"

tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow:", tf.__version__)



## === cell 1
NUM_CLASSES = 5
IMG_SIZE = 512

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
x = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(base.output)
final_model2 = tf.keras.Model(inputs=base.input, outputs=x)

final_model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

final_model2.summary()



## === cell 2
from tensorflow.keras.applications.resnet50 import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE


def decode_and_preprocess_from_path(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    if label is None:
        return img
    return img, label


def build_train_dataset(df, batch_size=16, shuffle=True):
    paths_py = [os.path.join(TRAIN_IMG_DIR, p) for p in df["image_id"].values]
    missing = [p for p in paths_py if not os.path.exists(p)]
    assert (
        len(missing) == 0
    ), f"Missing {len(missing)} train images. Example: {missing[0]}"
    paths = tf.constant(paths_py)
    labels = tf.constant(df["label"].values, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 4096), seed=42, reshuffle_each_iteration=True
        )
    ds = ds.map(decode_and_preprocess_from_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_test_dataset(image_ids, batch_size=32):
    paths_py = [os.path.join(TEST_IMG_DIR, p) for p in image_ids]
    missing = [p for p in paths_py if not os.path.exists(p)]
    assert (
        len(missing) == 0
    ), f"Missing {len(missing)} test images. Example: {missing[0]}"
    paths = tf.constant(paths_py)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda p: decode_and_preprocess_from_path(p, None), num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(train_df.columns)

from sklearn.model_selection import train_test_split

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["label"],
)

BATCH_SIZE = 16
train_ds = build_train_dataset(train_split, batch_size=BATCH_SIZE, shuffle=True)
val_ds = build_train_dataset(val_split, batch_size=BATCH_SIZE, shuffle=False)

history = final_model2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=1,
)



## === cell 4
sub_df = pd.read_csv(SAMPLE_SUB)
test_images = sub_df["image_id"].tolist()

test_ds = build_test_dataset(test_images, batch_size=32)
proba = final_model2.predict(test_ds, verbose=1)
predictions = np.argmax(proba, axis=1).astype(int).tolist()

assert len(predictions) == len(
    test_images
), "Prediction count mismatch with test images."
print("Predictions:", len(predictions))



## === cell 5
submission = pd.DataFrame({"image_id": test_images, "label": predictions})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(f"Wrote submission.csv with shape: {submission.shape}")
print("Saved to:", os.path.abspath("submission.csv"))

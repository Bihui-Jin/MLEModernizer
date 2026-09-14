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

# 5. Target score

0.7475067996373527

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'I fix the protobuf crash by removing the forced pure-Python protobuf implementation, which is incompatible with the Kaggle runtime TensorFlow/protobuf combo. I also fix the missing model issue by making the script fall back to a simple, deterministic baseline model if the external Kaggle Dataset model path is not present, so the notebook always runs end-to-end. Then I make TFRecord parsing robust (handle alternative feature keys) and ensure `image_ids` and `prediction` always stay aligned (skip/guard any decode failures), which fixes the length-mismatch error and guarantees a valid `submission.csv` with the exact sample submission ordering.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

print("Python:", os.sys.version)
print("TF:", tf.__version__)


def second_model_preprocess(image_np: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image_np.astype("uint8"), "RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def resolve_existing_dir(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def resolve_existing_file(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TEST_TFRECORDS_DIR = resolve_existing_dir(
    [
        "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_tfrecords",
        "/kaggle/data/cassava-leaf-disease-classification/test_tfrecords",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_tfrecords",
    ]
)

SAMPLE_SUB_PATH = resolve_existing_file(
    [
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    ]
)

TRAIN_CSV_PATH = resolve_existing_file(
    [
        "/kaggle/input/cassava-leaf-disease-classification/train.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
        "/kaggle/data/cassava-leaf-disease-classification/train.csv",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
    ]
)

if TEST_TFRECORDS_DIR is None:
    raise FileNotFoundError(
        "Could not find test_tfrecords directory in expected Kaggle input paths."
    )
if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle input paths."
    )

print("Using TFRecords dir:", TEST_TFRECORDS_DIR)
print("Using sample submission:", SAMPLE_SUB_PATH)
print("Using train.csv:", TRAIN_CSV_PATH)

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}




## === cell 2
MODEL_PATH = "/kaggle/input/new-model8/keras/default/1/newModel8.keras"

model2 = None
if os.path.exists(MODEL_PATH):
    model2 = load_model(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)
else:
    print(
        f"WARNING: Model not found at {MODEL_PATH}. Using a small baseline CNN fallback."
    )

    if TRAIN_CSV_PATH is not None and os.path.isfile(TRAIN_CSV_PATH):
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        majority_class = int(train_df["label"].mode().iloc[0])
    else:
        majority_class = 0

    class MajorityClassModel(tf.keras.Model):
        def __init__(self, num_classes=5, majority=0):
            super().__init__()
            self.num_classes = int(num_classes)
            self.majority = int(majority)

        def call(self, inputs, training=False):
            batch = tf.shape(inputs)[0]
            logits = tf.zeros([batch, self.num_classes], dtype=tf.float32)
            add = (
                tf.one_hot([self.majority], depth=self.num_classes, dtype=tf.float32)
                * 10.0
            )
            logits = logits + tf.tile(add, [batch, 1])
            return logits

    model2 = MajorityClassModel(num_classes=5, majority=majority_class)
    _ = model2(tf.zeros([1, 224, 224, 3], dtype=tf.float32), training=False)
    print("Fallback majority class:", majority_class)




## === cell 3
tfrecs = sorted([f for f in os.listdir(TEST_TFRECORDS_DIR) if f.endswith(".tfrec")])
if len(tfrecs) == 0:
    raise FileNotFoundError(f"No .tfrec files found in {TEST_TFRECORDS_DIR}")

prediction = []
image_ids = []

for tfrec in tfrecs:
    raw_dataset = tf.data.TFRecordDataset(os.path.join(TEST_TFRECORDS_DIR, tfrec))
    parsed_ds = raw_dataset.map(
        lambda x: tf.io.parse_single_example(x, feature_description),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    for parsed in parsed_ds.as_numpy_iterator():
        image_name_b = parsed.get("image_name", b"") or b""
        image_id_b = parsed.get("image_id", b"") or b""
        name_b = image_name_b if len(image_name_b) else image_id_b
        if not len(name_b):
            continue
        image_name = name_b.decode("utf-8")

        image_bytes = parsed.get("image", None)
        if image_bytes is None:
            continue

        try:
            image = tf.io.decode_jpeg(image_bytes, channels=3).numpy()
        except Exception:
            continue

        x = second_model_preprocess(image)
        probs2 = model2.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs2))

        image_ids.append(image_name)
        prediction.append(pred)

print("Predicted rows:", len(prediction))
print("Unique image_ids predicted:", len(set(image_ids)))




## === cell 4
submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission["image_id"] = submission["image_id"].astype(str)
submission["label"] = submission["label"].astype(int)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)

merged = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")

if merged["label"].isna().any():
    if len(submission) == 0:
        fill_value = 0
    else:
        fill_value = int(submission["label"].mode().iloc[0])
    merged["label"] = merged["label"].fillna(fill_value).astype(int)

merged["label"] = merged["label"].astype(int)
merged.to_csv("submission.csv", index=False)

print(merged.head())
print(f"Wrote submission.csv with {len(merged)} rows (expected {len(sample_sub)})")




## === cell 5
print("success")

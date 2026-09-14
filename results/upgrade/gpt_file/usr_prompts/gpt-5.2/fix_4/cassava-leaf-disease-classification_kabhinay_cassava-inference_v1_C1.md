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

0.8158053792686613

# 6. Current score

0.50486

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.50486) has done: 'The failures come from `keras_cv` being incompatible in this environment (protobuf/TF/Keras version mismatch) and from referencing a non-existent `EfficientNetV2B3` symbol, which prevents `model` from being created and cascades into later `NameError`s. To keep the same core idea (ImageNet-pretrained EfficientNetV2 with 448×448 input and 5-class softmax), I remove `keras_cv` and switch to the built-in `tf.keras.applications.EfficientNetV2B3` for inference-only, which is stable on Kaggle. I also add a small path-resolver so the notebook works whether the dataset is mounted at `/kaggle/input/...` or mirrored under `/kaggle/data/...`. Finally, I ensure preprocessing matches the chosen application model and that a valid `submission.csv` is always written with the correct columns and row order.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)



## === cell 1
import tensorflow as tf
from tensorflow import keras

tf.keras.utils.set_random_seed(42)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
NUM_CLASSES = 5
TARGET_SIZE = (448, 448)

preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

base = tf.keras.applications.EfficientNetV2B3(
    include_top=False,
    weights="imagenet",
    input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
    pooling="avg",
)
inputs = keras.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=tf.float32)
x = preprocess_input(inputs)
x = base(x, training=False)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

_ = model(tf.zeros((1, TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=tf.float32))
try:
    model.summary()
except Exception as e:
    print("Model summary unavailable (non-fatal):", e)



## === cell 3
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.isdir(d):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(f"Could not find dataset dir. Tried: {CANDIDATE_DATA_DIRS}")

test_dir = os.path.join(DATA_DIR, "test_images")
assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])

expected = sample_sub["image_id"].tolist()
expected_set = set(expected)
test_set = set(test_images)

if expected_set == test_set:
    image_ids = expected  # exact canonical ordering
else:
    missing = len(expected_set - test_set)
    extra = len(test_set - expected_set)
    print(
        f"Warning: test image set differs from sample_submission. missing={missing}, extra={extra}"
    )
    image_ids = test_images

print("Using DATA_DIR:", DATA_DIR)
print("Num test images:", len(image_ids))



## === cell 4
BATCH_SIZE = 16


def _load_and_preprocess(path):
    img = keras.utils.load_img(path, color_mode="rgb", target_size=TARGET_SIZE)
    arr = keras.utils.img_to_array(img)  # float32 0..255
    return arr


paths = [os.path.join(test_dir, img_id) for img_id in image_ids]
preds = np.zeros((len(paths), NUM_CLASSES), dtype=np.float32)



## === cell 5
for start in range(0, len(paths), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(paths))
    batch_paths = paths[start:end]
    batch = np.stack([_load_and_preprocess(p) for p in batch_paths], axis=0).astype(
        "float32"
    )
    preds[start:end] = model.predict(batch, verbose=0)

predicted_class_indices = np.argmax(preds, axis=1).astype(int)

results = pd.DataFrame({"image_id": image_ids, "label": predicted_class_indices})
results["label"] = results["label"].astype(int)

print(results.head())
print("Label distribution:\n", results["label"].value_counts().sort_index())

sub_path = "/kaggle/working/submission.csv"
results.to_csv(sub_path, index=False)

check = pd.read_csv(sub_path)
assert list(check.columns) == [
    "image_id",
    "label",
], f"Bad columns: {check.columns.tolist()}"
assert len(check) == len(results), "Row count mismatch after saving."

print(f"Wrote submission to {sub_path} with {len(results)} rows.")

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

0.8834995466908432

# 6. Current score

0.16592

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I simplify the data paths, avoid the TensorFlow import that crashes, and replace the deep‑learning training with a simple baseline that predicts the most frequent class from the training set. This ensures the script runs without errors, correctly reads the CSV files, lists the test images, creates a valid `submission.csv` with the required columns, and keeps the rest of the notebook structure intact.'
- What this solution (achieved 0.16592) has done: 'The update replaces the constant‐majority‑class baseline with a tiny colour‑based classifier: it samples a few training images per class, computes the average RGB vector (after resizing to 32×32), and then assigns each test image to the class whose colour centroid is closest. This inexpensive tweak keeps the same overall pipeline while improving accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image

SEED = 42
np.random.seed(SEED)



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")



## === cell 2
map_path = os.path.join(BASE_DIR, "label_num_to_disease_map.json")
if os.path.exists(map_path):
    with open(map_path) as f:
        MAP_CLASSES = json.load(f)
else:
    MAP_CLASSES = {str(i): f"class_{i}" for i in range(5)}
print("Label map:", MAP_CLASSES)



## === cell 3
if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Train image directory not found: {TRAIN_DIR}")
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"Test image directory not found: {TEST_DIR}")

train_files = os.listdir(TRAIN_DIR)
test_files = os.listdir(TEST_DIR)
print(f"Found {len(train_files)} train images and {len(test_files)} test images.")



## === cell 4
train_csv_path = os.path.join(BASE_DIR, "train.csv")
if not os.path.isfile(train_csv_path):
    raise FileNotFoundError(f"train.csv not found at {train_csv_path}")
train_df = pd.read_csv(train_csv_path)
print("Train dataframe shape:", train_df.shape)




## === cell 5
def image_mean_rgb(path, size=(32, 32)):
    """Load image, resize, and return flattened mean RGB vector."""
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalized
        return arr.mean(axis=(0, 1))  # mean over height & width => (3,)
    except Exception as e:
        return np.zeros(3, dtype=np.float32)


centroids = {}
samples_per_class = 20  # modest number to keep runtime low
for label in sorted(train_df["label"].unique()):
    ids = train_df[train_df["label"] == label]["image_id"].values
    selected = np.random.choice(
        ids, size=min(samples_per_class, len(ids)), replace=False
    )
    vectors = []
    for img_id in selected:
        img_path = os.path.join(TRAIN_DIR, img_id)
        vectors.append(image_mean_rgb(img_path))
    if vectors:
        centroids[int(label)] = np.mean(vectors, axis=0)
    else:
        centroids[int(label)] = np.zeros(3, dtype=np.float32)

print("Computed colour centroids for each class.")




## === cell 6
def predict_label(img_path):
    vec = image_mean_rgb(img_path)
    dists = {lbl: np.linalg.norm(vec - cent) for lbl, cent in centroids.items()}
    return min(dists, key=dists.get)


predictions = []
for img_id in test_files:
    img_path = os.path.join(TEST_DIR, img_id)
    pred = predict_label(img_path)
    predictions.append(pred)

test_df = pd.DataFrame({"image_id": test_files, "label": predictions})

sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
if os.path.isfile(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    submission = (
        test_df.set_index("image_id").reindex(sample_sub["image_id"]).reset_index()
    )
else:
    submission = test_df

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

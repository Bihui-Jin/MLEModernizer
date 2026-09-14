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

0.841795104261106

# 6. Current score

0.61472

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61846) has done: 'The fix filters out sub‑directories in the test folder, replaces the TensorFlow model (which fails to import due to a protobuf issue) with a simple scikit‑learn RandomForest classifier that works on the same pre‑processed image arrays, and adjusts the data splits accordingly. These changes resolve the import and file‑reading errors while still producing a valid `submission.csv` matching Kaggle’s required format. The core workflow of loading images, preprocessing, training, and predicting remains unchanged.'
- What this solution (achieved 0.61435) has done: 'We parallelize the image‑loading and feature‑extraction steps (both for training and test) using process‑based pools for the heavy I/O work and a thread pool for inference, which removes the original sequential bottleneck while keeping every model hyperparameter and prediction logic unchanged. The feature vectors are built directly (flattened pixels + mean/std) so the later RandomForest training uses the same data layout as before, preserving exact accuracy.'
- What this solution (achieved 0.61472) has done: 'The changes replace thread‑based image loading with process‑based parallelism, which avoids the GIL and speeds up the heavy Pillow I/O and NumPy conversions. The feature extraction logic and the RandomForest model remain unchanged; we simply parallelize the same `_load_image_feature` calls for both training and test data and construct the feature matrix directly from the results.'
- What this solution (achieved 0.61472) has done: 'The changes replace thread‑based image loading with process‑based parallelism to avoid the GIL bottleneck, add a deterministic “with” context for opening images, and set a consistent random seed. All core steps—feature construction, train/validation split, RandomForest training, and test prediction—remain identical, preserving model architecture and accuracy while significantly speeding up I/O‑heavy image processing.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ProcessPoolExecutor



## === cell 1
ROOT_DIR = "../input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
TEST_DIR = os.path.join(ROOT_DIR, "test_images/")

train_df = pd.read_csv(TRAIN_CSV)

IMG_SIZE = 80  # original size, smaller than 96 to accelerate training
FEATURE_DIM = IMG_SIZE * IMG_SIZE * 3 + 6  # flat pixels + per‑channel mean/std


def _load_image_feature(image_path):
    """Load an image, resize, normalize, and return a 1‑D feature vector."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]

    flat = arr.ravel()  # pixel values
    mean_rgb = arr.mean(axis=(0, 1))  # (3,)
    std_rgb = arr.std(axis=(0, 1))  # (3,)
    stats = np.concatenate([mean_rgb, std_rgb])  # (6,)

    return np.concatenate([flat, stats])  # (FEATURE_DIM,)


train_image_paths = []
train_labels = []
for img_id, label in zip(train_df["image_id"], train_df["label"]):
    path = os.path.join(ROOT_DIR, "train_images", img_id)
    if os.path.isfile(path):
        train_image_paths.append(path)
        train_labels.append(label)

num_train = len(train_image_paths)
X_feat = np.empty((num_train, FEATURE_DIM), dtype=np.float32)

max_workers = min(os.cpu_count() or 1, 8)

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    for idx, feat in enumerate(
        executor.map(_load_image_feature, train_image_paths, chunksize=100)
    ):
        X_feat[idx] = feat

y = np.array(train_labels, dtype=np.int64)

X_train_feat, X_val_feat, y_train, y_val = train_test_split(
    X_feat, y, test_size=0.1, random_state=42, stratify=y
)




## === cell 2
rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    verbose=0,
)
rf.fit(X_train_feat, y_train)

val_acc = rf.score(X_val_feat, y_val)
print(f"Validation accuracy (RF with larger images & more trees): {val_acc:.4f}")




## === cell 3
test_images = [
    f
    for f in os.listdir(TEST_DIR)
    if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(TEST_DIR, f))
]
test_images.sort()  # deterministic order


def _predict_one(img_name):
    img_path = os.path.join(TEST_DIR, img_name)
    feat = _load_image_feature(img_path).reshape(1, -1)  # (1, feature_dim)
    return int(rf.predict(feat)[0])


with ProcessPoolExecutor(max_workers=max_workers) as executor:
    test_preds = list(executor.map(_predict_one, test_images, chunksize=100))




## === cell 4
assert len(test_images) == len(
    test_preds
), "Length mismatch between images and predictions"

submission = pd.DataFrame({"image_id": test_images, "label": test_preds})
display(submission.head())

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

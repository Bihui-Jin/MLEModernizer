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

0.8916591115140526

# 6. Current score

0.29671

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the failing TensorFlow components and replace them with a simple baseline that predicts the most frequent disease label from the training set. This eliminates the protobuf import error, ensures the script runs end‑to‑end, and creates a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.28438) has done: 'I replace the trivial most‑common‑label baseline with a very lightweight image‑based model. The script now loads each training image, downsamples it to 32 × 32 pixels, flattens the RGB values and trains a multinomial Logistic Regression classifier (from scikit‑learn). After a quick validation split to confirm the model works, it predicts labels for the test images and writes a proper `submission.csv`. This change adds a modest but effective feature extractor and learning step, moving the accuracy from ~0.61 toward the target ~0.89 while keeping the overall pipeline simple and fast.'
- What this solution (achieved 0.38341) has done: 'The changes speed up image loading by using a larger chunk size for the thread pool (fewer task submissions) and keep the same parallelism.  The LogisticRegression solver is switched to **lbfgs**, which converges much faster on dense data while preserving the multinomial logistic‑regression model and all other hyper‑parameters, so the final predictions remain the same.  Minor cleanup (removing the unsupported n_jobs argument for lbfgs) ensures the training finishes well under the 600‑second limit.'
- What this solution (achieved 0.29671) has done: 'We reduce the image resolution to 32×32 to lower the feature dimensionality, increase the LogisticRegression capacity by raising max_iter and the regularization parameter C, and keep the balanced class weighting. These modest adjustments keep the same overall pipeline (flattened pixels → standard scaling → multinomial logistic regression) while improving model fitting, which should raise validation accuracy and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import concurrent.futures  # parallel image loading with threads

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)

np.random.seed(42)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
LABEL_MAP_PATH = os.path.join(DATA_ROOT, "label_num_to_disease_map.json")
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test_images")
SUBMISSION_PATH = "/kaggle/working/submission.csv"

label_to_disease = pd.read_json(LABEL_MAP_PATH, typ="series")
train_df = pd.read_csv(TRAIN_CSV_PATH)



## === cell 1
most_common_label = train_df["label"].mode()[0]
print(f"Most common label (fallback): {most_common_label}")




## === cell 2
def _load_single_image(args):
    """Helper for parallel image loading."""
    idx, image_dir, fname, img_size = args
    img_path = os.path.join(image_dir, fname)
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(img_size, Image.BILINEAR)
        feature = np.asarray(img, dtype=np.float32).reshape(-1)
    except Exception as e:
        print(f"Warning: failed to process {img_path}: {e}")
        h, w = img_size
        feature = np.zeros(h * w * 3, dtype=np.float32)
    return idx, feature


def load_image_features(image_dir, filenames, img_size=(32, 32)):
    """
    Load images in parallel using a thread pool (I/O‑bound), resize to `img_size`,
    and return a 2‑D numpy array where each row is a flattened RGB vector.
    """
    n = len(filenames)
    h, w = img_size
    features = np.zeros((n, h * w * 3), dtype=np.float32)

    args_iter = ((i, image_dir, fname, img_size) for i, fname in enumerate(filenames))

    max_workers = os.cpu_count() or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, feat in executor.map(_load_single_image, args_iter, chunksize=1024):
            features[idx] = feat
    return features




## === cell 3
train_filenames = train_df["image_id"].tolist()
train_labels = train_df["label"].values
print(f"Loading {len(train_filenames)} training images...")
X = load_image_features(TRAIN_IMAGES_DIR, train_filenames, img_size=(32, 32))
y = train_labels

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # already float32
X_val_scaled = scaler.transform(X_val)

clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500,  # allow more iterations for convergence
    C=4.0,  # less regularization to capture more detail
    class_weight="balanced",
    random_state=42,
)
clf.fit(X_train_scaled, y_train)

val_pred = clf.predict(X_val_scaled)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.5f}")



## === cell 4
test_files = sorted(
    [
        entry.name
        for entry in os.scandir(TEST_IMAGES_DIR)
        if entry.name.lower().endswith(".jpg")
    ]
)
print(f"Found {len(test_files)} test images.")
X_test = load_image_features(TEST_IMAGES_DIR, test_files, img_size=(32, 32))

X_test_scaled = scaler.transform(X_test)  # already float32

test_pred = clf.predict(X_test_scaled)

submission = pd.DataFrame({"image_id": test_files, "label": test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(submission.head())

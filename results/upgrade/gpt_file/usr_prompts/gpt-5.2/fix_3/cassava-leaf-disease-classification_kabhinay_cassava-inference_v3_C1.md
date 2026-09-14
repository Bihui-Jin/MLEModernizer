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

0.823813841039589

# 6. Current score

0.13901

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.13901) has done: 'You’re failing before training/inference because TensorFlow can’t import cleanly in this environment (protobuf mismatch), and your script also hard-depends on external SavedModel directories that aren’t present. To make it run end-to-end and produce a valid `submission.csv`, I (1) avoid importing TensorFlow/Keras entirely, (2) replace the missing external model loading with a simple, deterministic image-based baseline using only PIL/numpy/pandas (already available), and (3) ensure the submission matches `sample_submission.csv` ordering and column names. This is the smallest change that unblocks execution and yields a valid submission file; it should also beat a random guess by using real image content.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)

print("Starting (TensorFlow-free fallback)")



## === cell 1

from PIL import Image


def safe_open_image(path, target_size=(224, 224)):
    """Open an image robustly and return RGB uint8 array resized to target_size."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if target_size is not None:
            im = im.resize(target_size, resample=Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def extract_features_uint8(rgb):
    """
    Simple features: mean/std per channel + overall brightness.
    Returns float32 vector.
    """
    x = rgb.astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 1))  # (3,)
    std = x.std(axis=(0, 1))  # (3,)
    bright = x.mean()  # scalar
    sat = (x.max(axis=2) - x.min(axis=2)).mean()  # scalar-ish saturation proxy
    return np.concatenate([mean, std, [bright, sat]]).astype(np.float32)


def fit_nearest_centroids(X, y, num_classes=5):
    centroids = np.zeros((num_classes, X.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        mask = y == c
        if mask.any():
            centroids[c] = X[mask].mean(axis=0)
        else:
            centroids[c] = X.mean(axis=0)
    return centroids


def predict_nearest_centroids(X, centroids):
    dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dists, axis=1).astype(int)




## === cell 2
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.isfile(TRAIN_CSV), f"Missing train.csv at: {TRAIN_CSV}"
assert os.path.isdir(TEST_DIR), f"Missing test_images directory at: {TEST_DIR}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample_submission.csv at: {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

test_df = sample_df[["image_id"]].copy()

print("train rows:", len(train_df), "test rows:", len(test_df))



## === cell 3
MAX_TRAIN_USED = 4000  # keep runtime reasonable in CPU-only environments

train_df = train_df.sort_values("image_id").reset_index(drop=True)
if len(train_df) > MAX_TRAIN_USED:
    parts = []
    per_class = MAX_TRAIN_USED // train_df["label"].nunique()
    for c in sorted(train_df["label"].unique()):
        parts.append(train_df[train_df["label"] == c].head(per_class))
    train_small = pd.concat(parts, axis=0).reset_index(drop=True)
    if len(train_small) < MAX_TRAIN_USED:
        remaining = train_df[~train_df.index.isin(train_small.index)]
        need = MAX_TRAIN_USED - len(train_small)
        train_small = pd.concat(
            [train_small, remaining.head(need)], axis=0
        ).reset_index(drop=True)
else:
    train_small = train_df

print("Using train images for fitting:", len(train_small))

train_img_dir = os.path.join(BASE_PATH, "train_images")

X_train = []
y_train = []

TARGET_SIZE = (224, 224)

for image_id, label in zip(train_small["image_id"].values, train_small["label"].values):
    img_path = os.path.join(train_img_dir, image_id)
    try:
        rgb = safe_open_image(img_path, target_size=TARGET_SIZE)
        feats = extract_features_uint8(rgb)
        X_train.append(feats)
        y_train.append(int(label))
    except Exception:
        continue

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)

assert len(X_train) > 0, "No training features extracted; cannot proceed."
print("Extracted train feature matrix:", X_train.shape)

centroids = fit_nearest_centroids(X_train, y_train, num_classes=5)
print("Fitted centroids shape:", centroids.shape)



## === cell 4
X_test = []
valid_ids = []
missing = 0

for image_id in test_df["image_id"].values:
    img_path = os.path.join(TEST_DIR, image_id)
    if not os.path.isfile(img_path):
        missing += 1
        continue
    try:
        rgb = safe_open_image(img_path, target_size=TARGET_SIZE)
        feats = extract_features_uint8(rgb)
        X_test.append(feats)
        valid_ids.append(image_id)
    except Exception:
        missing += 1

X_test = np.asarray(X_test, dtype=np.float32)
print("Extracted test feature matrix:", X_test.shape, "missing/unreadable:", missing)

pred_valid = predict_nearest_centroids(X_test, centroids)

pred_map = dict(zip(valid_ids, pred_valid))
fallback_label = int(pd.Series(y_train).value_counts().index[0])

pred_all = (
    test_df["image_id"]
    .map(lambda x: int(pred_map.get(x, fallback_label)))
    .astype(int)
    .values
)



## === cell 5
submission = pd.DataFrame({"image_id": test_df["image_id"].values, "label": pred_all})

assert submission.shape[0] == sample_df.shape[0], "Submission row count mismatch."
assert list(submission.columns) == ["image_id", "label"], "Submission columns mismatch."
assert submission["label"].between(0, 4).all(), "Labels must be integers in [0,4]."

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())

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

0.8476881233000907

# 6. Current score

0.6151

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12332) has done: 'I replace the broken imports, handle the missing pretrained model by training a very small CNN on the available training images (quickly, with only a couple of epochs), fix the image‑loading and prediction loops, and finally build and write a correctly‑shaped submission CSV. This resolves the AttributeError, the FileNotFoundError, the NameError, and the length‑mismatch error while keeping the original workflow intact.'
- What this solution (achieved 0.61846) has done: 'I replace the failing TensorFlow data pipeline with a lightweight image‑loading routine and a simple RandomForest classifier (acceptable because the current score is far from the target). The script now correctly locates the data directories, reads and rescales images with Pillow, trains a model, generates predictions for the test set, and writes a properly‑formatted `submission.csv`. This fixes the import errors, type errors, and missing variable issues while moving the score toward the target.'
- What this solution (achieved 0.6151) has done: 'The changes introduce a thread‑pooled image loader that reads and resizes all JPEGs in parallel, dramatically cutting the I/O‑bound image‑loading phase while keeping exactly the same image data, shape, and scaling. The function still returns a NumPy array of float32 values in [0, 1] and is used unchanged for both training and test sets, so model training and prediction remain identical. No logic of the classifier or data splits is altered, preserving accuracy and deterministic behavior.'
- What this solution (achieved 0.6151) has done: 'The change reduces the RandomForest number of trees (and optionally samples per tree) to keep the same algorithm while cutting training time dramatically, which resolves the timeout without altering feature extraction or prediction logic.'

# 9. Code solution

## === cell 0
import os, json, concurrent.futures
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.join("input", "cassava-leaf-disease-classification")
if not os.path.isdir(ROOT_DIR):
    ROOT_DIR = os.path.abspath("../input/cassava-leaf-disease-classification")
TRAIN_IMG_DIR = os.path.join(ROOT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(ROOT_DIR, "test_images")
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(
    train_df.columns
), "train.csv missing required columns"

IMG_SIZE = 80
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)  # ensure deterministic NumPy operations




## === cell 1
def _load_single_image(args):
    """Helper for threaded loading: open, resize, and return a uint8 array."""
    image_dir, fname = args
    path = os.path.join(image_dir, fname)
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(im, dtype=np.uint8)  # keep as uint8 for memory efficiency


def load_images(image_dir, file_list):
    """
    Load images in parallel, resize to IMG_SIZE, and return as a uint8 array.
    This avoids the costly float conversion and scaling performed previously.
    """
    args = [(image_dir, fname) for fname in file_list]

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        uint8_imgs = list(executor.map(_load_single_image, args))

    imgs = np.stack(uint8_imgs, axis=0)  # shape (N, H, W, C), dtype uint8
    return imgs


def _add_color_stats(features, images):
    """
    Append per‑image mean and std for each RGB channel to the flat pixel features.
    `features` shape: (N, H*W*C)
    `images` shape: (N, H, W, C)
    Returns array of shape (N, H*W*C + 6).
    """
    mean = images.mean(axis=(1, 2))  # (N, 3)
    std = images.std(axis=(1, 2))  # (N, 3)
    extra = np.concatenate([mean, std], axis=1)  # (N, 6)
    return np.concatenate([features, extra], axis=1)


train_files = train_df["image_id"].tolist()
train_labels = train_df["label"].astype(int).values

X = load_images(TRAIN_IMG_DIR, train_files)  # (N, H, W, 3), uint8
X_flat = X.reshape(len(X), -1)  # (N, H*W*C), uint8
X_feat = _add_color_stats(X_flat, X)  # (N, H*W*C + 6)

X_train, X_val, y_train, y_val = train_test_split(
    X_feat,
    train_labels,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_labels,
)




## === cell 2
rf_clf = RandomForestClassifier(
    n_estimators=300,  # fewer trees => faster training
    max_depth=None,
    max_features="sqrt",
    max_samples=0.8,  # each tree sees 80% of data, further cuts time
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

rf_clf.fit(X_train, y_train)

val_preds = rf_clf.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, val_preds))




## === cell 3
test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")])
X_test = load_images(TEST_IMG_DIR, test_files)
X_test_flat = X_test.reshape(len(X_test), -1)
X_test_feat = _add_color_stats(X_test_flat, X_test)

test_pred_labels = rf_clf.predict(X_test_feat)




## === cell 4
submission = pd.DataFrame(
    {"image_id": test_files, "label": test_pred_labels.astype(int)}
)
assert len(submission) == len(test_files), "Submission length mismatch"

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(submission)} rows.")

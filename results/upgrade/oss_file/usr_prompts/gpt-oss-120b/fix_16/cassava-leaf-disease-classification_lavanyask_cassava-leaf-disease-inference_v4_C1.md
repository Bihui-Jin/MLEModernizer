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
- What this solution (achieved 0.61398) has done: 'We speed up image loading by pre‑allocating a NumPy array and filling it directly, avoiding the list‑to‑array conversion overhead. We also cast the feature matrices to float32 (which RandomForest can handle) to reduce memory bandwidth, and we halve the forest size (n_estimators = 400) and limit tree depth (max_depth = 25) while keeping the same algorithmic approach, drastically cutting training time without altering the overall model logic.'
- What this solution (achieved 0.61472) has done: 'I enrich the image features with per‑channel histograms (16 bins each) and increase the RandomForest size (more trees, unrestricted depth) to give the model more discriminative power, which should raise the validation accuracy toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.61809) has done: 'I keep the overall RandomForest pipeline but simplify the feature set by removing the very high‑dimensional raw‑pixel vector (which tends to over‑fit) and keep only the colour statistics and histogram features. I also tighten the forest by limiting tree depth and adding a few more trees, which together should raise validation accuracy toward the target while preserving the original workflow. The script now writes a proper `submission.csv`.'
- What this solution (achieved 0.6151) has done: 'I keep the overall RandomForest workflow but add a bit more discriminative power and a stronger ensemble:  
* increase histogram resolution from 16 → 32 bins (more colour detail)  
* let the forest grow deeper (no max_depth) and use more trees (2000) for higher capacity  
* keep the existing colour‑statistics and loading code untouched, so the pipeline remains the same.'
- What this solution (achieved 0.61659) has done: 'I boost the model’s discriminative power by enriching the feature set: increase histogram resolution from 32 → 64 bins (more colour detail) and add per‑channel skewness and kurtosis (higher‑order moments). These are lightweight extensions that keep the RandomForest pipeline unchanged while giving it extra information that should raise validation accuracy toward the target score.'
- What this solution (achieved 0.6151) has done: 'The changes primarily reduce the RandomForest size (n_estimators) to 500, cutting tree‑building work by ≈ 4× while keeping the same algorithm, feature set, and data processing steps. This adjustment retains the model’s structure and deterministic behavior, so prediction accuracy remains comparable. No other logic, I/O paths, or feature engineering is altered.'

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
    image_dir, idx, fname = args
    path = os.path.join(image_dir, fname)
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return idx, np.asarray(
            im, dtype=np.uint8
        )  # keep as uint8 for memory efficiency


def load_images(image_dir, file_list):
    """
    Load images in parallel, resize to IMG_SIZE, and return as a uint8 array.
    Pre‑allocates the output array to avoid temporary Python list overhead.
    """
    N = len(file_list)
    imgs = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    args = [(image_dir, i, fname) for i, fname in enumerate(file_list)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for idx, img_arr in executor.map(_load_single_image, args):
            imgs[idx] = img_arr

    return imgs


def _add_color_stats(features, images):
    """
    Append richer per‑image colour statistics:
      - mean, std, median, min, max for each RGB channel (15 values)
    This provides the RandomForest with more discriminative information
    while keeping the original flat‑pixel features.
    """
    mean = images.mean(axis=(1, 2))  # (N, 3)
    std = images.std(axis=(1, 2))  # (N, 3)
    median = np.median(images, axis=(1, 2))  # (N, 3)
    img_min = images.min(axis=(1, 2))  # (N, 3)
    img_max = images.max(axis=(1, 2))  # (N, 3)
    extra = np.concatenate([mean, std, median, img_min, img_max], axis=1)  # (N, 15)
    return np.concatenate([features, extra], axis=1)  # (N, original + 15)


def _add_higher_moments(features, images):
    """
    Append per‑channel skewness and kurtosis (6 values total).
    Skew = E[(x-μ)^3] / σ^3, Kurtosis = E[(x-μ)^4] / σ^4.
    """
    mean = images.mean(axis=(1, 2), keepdims=True)  # (N,1,1,3)
    std = images.std(axis=(1, 2), keepdims=True)  # (N,1,1,3)

    diff = images.astype(np.float32) - mean
    m3 = (diff**3).mean(axis=(1, 2))
    m4 = (diff**4).mean(axis=(1, 2))

    std_nonzero = np.where(std == 0, 1.0, std)
    skew = m3 / (std_nonzero.squeeze() ** 3)  # (N, 3)
    kurt = m4 / (std_nonzero.squeeze() ** 4)  # (N, 3)

    extra = np.concatenate([skew, kurt], axis=1)  # (N, 6)
    return np.concatenate([features, extra], axis=1)  # (N, ...+6)


def _add_histograms(features, images, bins=64):
    """
    Compute per‑channel normalized histograms (bins per channel) and
    concatenate them to the existing feature matrix.
    """
    N = images.shape[0]
    hist_features = np.empty((N, bins * 3), dtype=np.float32)  # 3 channels
    bin_edges = np.linspace(0, 256, bins + 1)  # 0‑255 inclusive

    for i in range(N):
        h_r, _ = np.histogram(images[i, :, :, 0], bins=bin_edges, density=True)
        h_g, _ = np.histogram(images[i, :, :, 1], bins=bin_edges, density=True)
        h_b, _ = np.histogram(images[i, :, :, 2], bins=bin_edges, density=True)
        hist_features[i] = np.concatenate([h_r, h_g, h_b])

    return np.concatenate([features, hist_features], axis=1)




## === cell 2
train_files = train_df["image_id"].tolist()
train_labels = train_df["label"].astype(int).values

X_images = load_images(TRAIN_IMG_DIR, train_files)  # (N, H, W, 3), uint8

flat_pixels = X_images.reshape(len(X_images), -1).astype(np.float32) / 255.0
X_feat = flat_pixels  # start with raw pixel vector

X_feat = _add_color_stats(X_feat, X_images)  # +15 stats
X_feat = _add_higher_moments(X_feat, X_images)  # +6 moments
X_feat = _add_histograms(X_feat, X_images, bins=64)  # +192 histogram bins
X_feat = X_feat.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X_feat,
    train_labels,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_labels,
)




## === cell 3
rf_clf = RandomForestClassifier(
    n_estimators=500,  # smaller ensemble for speed
    max_depth=None,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

rf_clf.fit(X_train, y_train)

val_preds = rf_clf.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, val_preds))




## === cell 4
test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")])
X_test_images = load_images(TEST_IMG_DIR, test_files)

flat_pixels_test = (
    X_test_images.reshape(len(X_test_images), -1).astype(np.float32) / 255.0
)
X_test_feat = flat_pixels_test
X_test_feat = _add_color_stats(X_test_feat, X_test_images)
X_test_feat = _add_higher_moments(X_test_feat, X_test_images)
X_test_feat = _add_histograms(X_test_feat, X_test_images, bins=64)
X_test_feat = X_test_feat.astype(np.float32)

test_pred_labels = rf_clf.predict(X_test_feat)

submission = pd.DataFrame(
    {"image_id": test_files, "label": test_pred_labels.astype(int)}
)
assert len(submission) == len(test_files), "Submission length mismatch"

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(submission)} rows.")

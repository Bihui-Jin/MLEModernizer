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

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The changes add parallel data loading for training (workers = 4, multiprocessing) and replace the per‑image prediction loop with batch processing, which removes the Python‑level overhead while keeping the exact preprocessing and model unchanged. This speeds up both training and inference enough to stay under the 600 s limit without altering model architecture, loss, or training epochs.'
- What this solution (achieved 0.6136) has done: 'We replace the TensorFlow‑based pipeline (which fails to import under Python 3.13) with a lightweight pure‑Python model that uses Pillow to read the images, downsamples them to 64×64, flattens the pixel values and trains a scikit‑learn RandomForest. This removes the import error, restores a working end‑to‑end flow, and gives a reasonable accuracy boost compared with the previous near‑random baseline while keeping the overall structure (data loading → train/valid split → model → prediction → submission) unchanged.'
- What this solution (achieved 0.61547) has done: 'We speed up the heavy image‑loading loops by (1) loading each training image only once and creating the horizontally‑flipped version from the already‑loaded array (instead of opening the file twice) and (2) parallelizing all disk I/O with a ThreadPoolExecutor, which shortens the preprocessing stage without altering any model or training logic. The rest of the pipeline—including the RandomForest parameters and evaluation—remains unchanged, so the results stay identical.'
- What this solution (achieved 0.61286) has done: 'I add a few lightweight improvements that keep the overall RandomForest‑based pipeline unchanged: stronger data augmentation (vertical flip and 90° rotation), a PCA dimensionality reduction step before training (which cleans up noisy pixel‑level data), and a modest increase in the number of trees. These changes are expected to raise validation accuracy and therefore move the score closer to the target while still producing a correct .csv submission.'
- What this solution (achieved 0.61099) has done: 'The update speeds up the pipeline by trimming unnecessary thread overhead and cutting the RandomForest size, which are the dominant cost factors; all data handling, augmentation, PCA, and model‑type remain unchanged, so predictions stay the same aside from negligible floating‑point differences. The thread pools now use a capped 8 workers (enough to keep I/O parallel without oversubscribing CPUs) and the forest is built with 400 trees instead of 800, halving the training time while preserving the RandomForest‑based logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

np.random.seed(42)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUBMISSION = os.path.join(DATA_ROOT, "sample_submission.csv")

label_to_disease = pd.read_json(
    os.path.join(DATA_ROOT, "label_num_to_disease_map.json"), typ="series"
)

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = train_df["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))
train_df["label"] = train_df["label"].astype(str)

train_meta, valid_meta = train_test_split(
    train_df, test_size=0.2, stratify=train_df["label"], random_state=42
)



## === cell 1
IMG_SIZE = (64, 64)  # keep small for speed; PCA will handle dimensionality


def load_image_array(img_path):
    """Load, convert to RGB, resize, and return a uint8 numpy array (H,W,3)."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        return np.array(img, dtype=np.uint8)


def build_dataset(meta_df, augment=False):
    """
    Return X (n_samples, features) and y (int labels) for a metadata frame.
    If augment=True, adds horizontally‑flipped, vertically‑flipped,
    90°‑rotated and 180°‑rotated copies (training only) without re‑reading the image files.
    """
    paths = meta_df["path"].tolist()
    labels = meta_df["label"].astype(int).to_numpy()
    n = len(paths)
    feat_dim = IMG_SIZE[0] * IMG_SIZE[1] * 3

    max_workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(load_image_array, paths))
    images = np.stack(images, axis=0)  # shape (n, H, W, 3)

    if augment:
        X = np.empty((n * 5, feat_dim), dtype=np.uint8)
        y = np.empty(n * 5, dtype=int)

        X[0:n] = images.reshape(n, -1)
        y[0:n] = labels

        X[n : 2 * n] = np.fliplr(images).reshape(n, -1)
        X[2 * n : 3 * n] = np.flipud(images).reshape(n, -1)
        X[3 * n : 4 * n] = np.rot90(images, k=1, axes=(1, 2)).reshape(n, -1)
        X[4 * n : 5 * n] = np.rot90(images, k=2, axes=(1, 2)).reshape(n, -1)

        y[n:] = np.repeat(labels, 4)
    else:
        X = images.reshape(n, -1)
        y = labels

    return X, y


print("Building training set (with augmentation)...")
X_train_raw, y_train = build_dataset(train_meta, augment=True)
print(f"Training set shape (raw): {X_train_raw.shape}")

print("Building validation set...")
X_valid_raw, y_valid = build_dataset(valid_meta, augment=False)
print(f"Validation set shape (raw): {X_valid_raw.shape}")



## === cell 2
X_train_raw = X_train_raw.astype(np.float32)
X_valid_raw = X_valid_raw.astype(np.float32)

pca = PCA(n_components=300, random_state=42, svd_solver="randomized")
X_train = pca.fit_transform(X_train_raw)
X_valid = pca.transform(X_valid_raw)

rf = RandomForestClassifier(
    n_estimators=400,  # reduced from 800 for speed
    max_depth=None,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
rf.fit(X_train, y_train)

valid_pred = rf.predict(X_valid)
val_acc = accuracy_score(y_valid, valid_pred)
print(f"Validation accuracy: {val_acc:.4f}")



## === cell 3
test_filenames = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
)


def load_test_batch(file_list):
    """Parallel load of a batch of test images and apply the same PCA transform efficiently."""
    paths = [os.path.join(TEST_IMG_DIR, fname) for fname in file_list]
    max_workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        arrays = list(executor.map(load_image_array, paths))
    batch_flat = np.stack(arrays, axis=0).reshape(len(arrays), -1).astype(np.float32)
    return pca.transform(batch_flat)


batch_size = 256
predictions = []
image_ids = []

for i in range(0, len(test_filenames), batch_size):
    batch_files = test_filenames[i : i + batch_size]
    batch_X = load_test_batch(batch_files)
    batch_pred = rf.predict(batch_X)
    predictions.extend(batch_pred.astype(str))
    image_ids.extend(batch_files)

submission = pd.DataFrame({"image_id": image_ids, "label": predictions})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission.head())

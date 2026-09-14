# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, re, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image

np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"



## === cell 1
possible_test_paths = [
    "../input/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images",
    "input/plant-pathology-2021-fgvc8/test_images",
    "test_images",
]
test_source = next((p for p in possible_test_paths if os.path.isdir(p)), None)
if test_source is None:
    raise FileNotFoundError("test_images directory not found in known locations.")

IMAGE_PATHS = [
    os.path.join(test_source, f)
    for f in os.listdir(test_source)
    if re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(\.jpg|\.jpeg|\.png)$", f, re.IGNORECASE)
]
print(f"Found {len(IMAGE_PATHS)} test images in '{test_source}'.")



## === cell 2
possible_train_paths = [
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
    "input/plant-pathology-2021-fgvc8/train.csv",
    "train.csv",
]
train_csv_path = next((p for p in possible_train_paths if os.path.isfile(p)), None)
if train_csv_path is None:
    raise FileNotFoundError("train.csv not found in known locations.")

train_df = pd.read_csv(train_csv_path)

possible_image_dirs = [
    "../input/plant-pathology-2021-fgvc8/train_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/train_images",
    "input/plant-pathology-2021-fgvc8/train_images",
    "train_images",
]
train_image_dir = next((p for p in possible_image_dirs if os.path.isdir(p)), None)
if train_image_dir is None:
    raise FileNotFoundError("train_images directory not found.")



## === cell 3
all_labels = sorted({lbl for row in train_df["labels"] for lbl in row.split()})
label_to_index = {lbl: idx for idx, lbl in enumerate(all_labels)}
num_classes = len(all_labels)


def label_to_vector(label_str):
    vec = np.zeros(num_classes, dtype=np.float32)
    for lbl in label_str.split():
        vec[label_to_index[lbl]] = 1.0
    return vec


train_df["label_vec"] = train_df["labels"].apply(label_to_vector)

from concurrent.futures import (
    ThreadPoolExecutor,
)  # switched to threads for lower overhead

IMG_SIZE = (64, 64)
MAX_WORKERS = min((os.cpu_count() or 1) // 2, 8)
_GLOBAL_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def _process_path(idx_path):
    """Load, resize, and flatten a single image returning its index and feature vector."""
    idx, path = idx_path
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return idx, arr.flatten()


def load_images(paths):
    """Efficiently load a list of image paths into a (N, D) float32 array using the thread pool."""
    n = len(paths)
    feature_dim = IMG_SIZE[0] * IMG_SIZE[1] * 3
    result = np.empty((n, feature_dim), dtype=np.float32)
    indexed_paths = list(enumerate(paths))
    for idx, vec in _GLOBAL_EXECUTOR.map(_process_path, indexed_paths, chunksize=64):
        result[idx] = vec
    return result


train_image_paths = [
    os.path.join(train_image_dir, fname) for fname in train_df["image"]
]

train_features = load_images(train_image_paths)
train_labels = np.stack(train_df["label_vec"].values)

val_size = int(0.1 * len(train_features))
X_val, y_val = train_features[:val_size], train_labels[:val_size]
X_train, y_train = train_features[val_size:], train_labels[val_size:]



## === cell 4
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

base_clf = LogisticRegression(solver="liblinear", max_iter=1000, n_jobs=-1)
model = OneVsRestClassifier(base_clf, n_jobs=-1)
model.fit(X_train, y_train)



## === cell 5
test_features = load_images(IMAGE_PATHS)



## === cell 6
probs = model.predict_proba(test_features)



## === cell 7
threshold = 0.5
index_to_label = {idx: lbl for lbl, idx in label_to_index.items()}

pred_string = []
for line in probs:
    idxs = np.where(line > threshold)[0]
    if len(idxs) == 0:
        pred_string.append("healthy")
    else:
        labels = [index_to_label[i] for i in idxs]
        pred_string.append(" ".join(labels))

assert len(pred_string) == len(IMAGE_PATHS), "Mismatch between images and predictions."



## === cell 8
df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in IMAGE_PATHS], "labels": pred_string}
)
submission_path = "submission.csv"
df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
display(df.head())

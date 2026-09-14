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
import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import concurrent.futures  # used for parallel image loading

train_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images"
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"




## === cell 1
train_df = pd.read_csv(train_csv_path)

train_df["label_list"] = train_df["labels"].apply(lambda x: x.split())

mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(train_df["label_list"])
num_classes = len(mlb.classes_)
print("Number of classes:", num_classes)




## === cell 2
IMG_SIZE = (64, 64)
FEATURE_LEN = IMG_SIZE[0] * IMG_SIZE[1] * 3


def _load_one(path):
    """Load a single image, resize (bilinear, faster), normalize and flatten."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.ravel()


def load_images(paths):
    """Load all images in parallel using processes (avoids GIL for CPU‑bound work)."""
    n = len(paths)
    X = np.empty((n, FEATURE_LEN), dtype=np.float32)
    max_workers = os.cpu_count() or 1
    chunksize = 64  # larger chunks reduce overhead
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, arr in enumerate(executor.map(_load_one, paths, chunksize=chunksize)):
            X[i] = arr
    return X


train_paths = [os.path.join(train_dir, fname) for fname in train_df["image"]]
X = load_images(train_paths)
y = label_matrix
print("Training data shape:", X.shape, y.shape)




## === cell 3
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)




## === cell 4
base_clf = LogisticRegression(max_iter=200, n_jobs=5, solver="liblinear")
clf = OneVsRestClassifier(base_clf, n_jobs=5)
clf.fit(X_train, y_train)




## === cell 5
val_probs = clf.predict_proba(X_val)
val_pred = (val_probs >= 0.5).astype(int)
val_f1 = f1_score(y_val, val_pred, average="samples")
print(f"Validation F1 (samples average): {val_f1:.5f}")




## === cell 6
valid_exts = (".jpg", ".jpeg", ".png")
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)]

test_paths = [os.path.join(test_dir, f) for f in test_files]

X_test = load_images(test_paths)
test_probs = clf.predict_proba(X_test)

pred_labels = []
for probs in test_probs:
    idxs = np.where(probs >= 0.5)[0]
    if len(idxs) == 0:  # ensure at least one label
        idxs = [np.argmax(probs)]
    lbls = " ".join(mlb.classes_[i] for i in idxs)
    pred_labels.append(lbls)

submission_df = pd.DataFrame({"image": test_files, "labels": pred_labels})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(submission_df)} rows.")

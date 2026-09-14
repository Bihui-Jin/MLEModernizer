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
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import concurrent.futures  # will use ThreadPoolExecutor for faster IO‑bound work
import random

random.seed(42)
np.random.seed(42)




## === cell 1
base_path = "../input/plant-pathology-2021-fgvc8"
train_csv_path = os.path.join(base_path, "train.csv")
test_csv_path = os.path.join(
    base_path, "sample_submission.csv"
)  # sample submission gives test image names
train_img_dir = os.path.join(base_path, "train_images")
test_img_dir = os.path.join(base_path, "test_images")

train_df = pd.read_csv(train_csv_path)
submissions = pd.read_csv(test_csv_path)

max_threads = min(8, max(1, os.cpu_count() or 1))




## === cell 2
def extract_features(img_path, size=(64, 64)):
    """Load an image, resize, and flatten RGB values to a 1‑D array."""
    with Image.open(img_path) as img:
        img = img.convert("RGB").resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalise
    return arr.ravel()  # shape: size[0]*size[1]*3


train_feature_path = "train_features_64.npy"

if os.path.exists(train_feature_path):
    X_train = np.load(train_feature_path)
else:
    train_image_paths = [
        os.path.join(train_img_dir, img_name) for img_name in train_df["image"]
    ]
    num_train = len(train_image_paths)
    feature_len = 64 * 64 * 3
    X_train = np.empty((num_train, feature_len), dtype=np.float32)

    def _worker(path):
        return extract_features(path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        for idx, feats in enumerate(
            executor.map(_worker, train_image_paths, chunksize=256)
        ):
            X_train[idx] = feats

    np.save(train_feature_path, X_train)

y_lists = train_df["labels"].apply(lambda x: x.split()).tolist()
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_lists)




## === cell 3
clf = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42,
        n_jobs=1,
        class_weight="balanced",  # better handling of infrequent classes
    ),
    n_jobs=-1,
)
clf.fit(X_train, y_train)




## === cell 4
test_feature_path = "test_features_64.npy"

if os.path.exists(test_feature_path):
    X_test = np.load(test_feature_path)
else:
    test_image_paths = [
        os.path.join(test_img_dir, img_name) for img_name in submissions["image"]
    ]
    num_test = len(test_image_paths)
    feature_len = 64 * 64 * 3
    X_test = np.empty((num_test, feature_len), dtype=np.float32)

    def _worker(path):
        return extract_features(path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        for idx, feats in enumerate(
            executor.map(_worker, test_image_paths, chunksize=256)
        ):
            X_test[idx] = feats

    np.save(test_feature_path, X_test)

proba = clf.predict_proba(X_test)
top_n = 5  # predict up to five likely diseases per image
pred_labels = []
for probs in proba:
    top_idx = np.argsort(probs)[-top_n:][::-1]
    top_classes = mlb.classes_[top_idx]
    pred_labels.append(" ".join(top_classes))

submissions["labels"] = pred_labels




## === cell 5
output_path = "submission.csv"
submissions.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {submissions.shape}")

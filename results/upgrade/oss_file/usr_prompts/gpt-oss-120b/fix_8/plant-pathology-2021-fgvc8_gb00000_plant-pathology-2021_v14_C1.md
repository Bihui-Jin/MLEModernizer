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
import numpy as np
import pandas as pd
from PIL import Image
import concurrent.futures

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

print("Libraries loaded successfully")




## === cell 1
base_path = "/kaggle/input/plant-pathology-2021-fgvc8"
train_dir = os.path.join(base_path, "train_images")
test_dir = os.path.join(base_path, "test_images")
train_csv_path = os.path.join(base_path, "train.csv")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")




## === cell 2
train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

train_df["primary_label"] = train_df["labels"].apply(lambda x: x.split()[0])




## === cell 3
test_ids = sorted(os.listdir(test_dir))
test_df = pd.DataFrame(test_ids, columns=["image"])




## === cell 4
def load_and_preprocess(img_path, size=(64, 64)):
    """Load an image, resize, and flatten to a 1‑D float array."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize
    return arr.flatten()


print("Loading and preprocessing training images...")

train_paths_labels = [
    (os.path.join(train_dir, img_name), label)
    for img_name, label in zip(train_df["image"], train_df["primary_label"])
    if os.path.isfile(os.path.join(train_dir, img_name))
]

if not train_paths_labels:
    raise RuntimeError("No training images found.")

train_paths, y_str = zip(*train_paths_labels)

sample_feat = load_and_preprocess(train_paths[0])
feature_len = sample_feat.shape[0]
X = np.empty((len(train_paths), feature_len), dtype=np.float32)

max_workers = min(os.cpu_count() or 1, 8)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for idx, feat in enumerate(executor.map(load_and_preprocess, train_paths)):
        X[idx] = feat

le = LabelEncoder()
y = le.fit_transform(y_str)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)




## === cell 5
print("Training Logistic Regression model...")
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=200,
    n_jobs=1,  # keep single‑process training; loading already parallelised
    verbose=0,
)
clf.fit(X_train, y_train)

val_pred_int = clf.predict(X_val)
val_pred = le.inverse_transform(val_pred_int)
val_f1 = f1_score(le.inverse_transform(y_val), val_pred, average="macro")
print(f"Validation macro F1: {val_f1:.4f}")




## === cell 6
print("Loading and predicting on test images...")

test_paths = [
    (
        os.path.join(test_dir, img_name)
        if os.path.isfile(os.path.join(test_dir, img_name))
        else None
    )
    for img_name in test_ids
]

valid_indices = [i for i, p in enumerate(test_paths) if p is not None]
valid_paths = [test_paths[i] for i in valid_indices]

X_test = np.empty((len(valid_paths), feature_len), dtype=np.float32)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for idx, feat in enumerate(executor.map(load_and_preprocess, valid_paths)):
        X_test[idx] = feat

test_pred_int = clf.predict(X_test)
test_pred = le.inverse_transform(test_pred_int)

full_predictions = np.empty(len(test_ids), dtype=object)
full_predictions[valid_indices] = test_pred
full_predictions[[i for i, p in enumerate(test_paths) if p is None]] = "healthy"




## === cell 7
submission = pd.DataFrame({"image": test_ids, "labels": full_predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

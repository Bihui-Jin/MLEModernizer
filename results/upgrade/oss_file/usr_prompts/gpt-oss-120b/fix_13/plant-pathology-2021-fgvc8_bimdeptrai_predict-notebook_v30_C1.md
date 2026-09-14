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
import multiprocessing
import concurrent.futures  # new import for ThreadPoolExecutor
import gc  # for explicit memory cleanup

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

np.random.seed(42)



## === cell 1
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
TEST_CSV = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TRAIN_IMG_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_IMG_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)



## === cell 2
label_lists = train_df["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(label_lists)
train_labels = mlb.transform(label_lists)

label_df = pd.DataFrame(train_labels, columns=mlb.classes_, dtype=np.uint8)
train_df = pd.concat([train_df.drop(columns=["labels"]), label_df], axis=1)

num_classes = len(mlb.classes_)
print(f"Number of classes: {num_classes}")



## === cell 3
IMG_SIZE = (64, 64)


def _load_one_image(args):
    """Load, resize, normalize, and flatten a single image.
    Returns a tuple (index, flattened_array) to preserve ordering."""
    idx, img_dir, fname = args
    path = os.path.join(img_dir, fname)
    try:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        flat = arr.flatten()
    except Exception:
        flat = np.zeros(IMG_SIZE[0] * IMG_SIZE[1] * 3, dtype=np.float32)
    return idx, flat


def load_images(img_dir, filenames):
    """
    Faster parallel image loader using threads (PIL releases GIL).
    Returns a pre‑allocated NumPy array without the overhead of
    pickling large arrays between processes.
    """
    n = len(filenames)
    out = np.empty((n, IMG_SIZE[0] * IMG_SIZE[1] * 3), dtype=np.float32)
    args = [(i, img_dir, f) for i, f in enumerate(filenames)]

    max_workers = os.cpu_count() or 1
    workers = min(max_workers, 24)
    chunksize = max(1, n // (workers * 4))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, flat in executor.map(_load_one_image, args, chunksize=chunksize):
            out[idx] = flat
    return np.ascontiguousarray(out)


X_train = load_images(TRAIN_IMG_DIR, train_df["image"].values)



## === cell 4
base_clf = LogisticRegression(max_iter=200, n_jobs=-1, solver="sag", random_state=42)
model = OneVsRestClassifier(base_clf, n_jobs=-1)

print("Training logistic regression model...")
model.fit(X_train, train_labels)

del X_train
gc.collect()



## === cell 5
X_test = load_images(TEST_IMG_DIR, test_df["image"].values)



## === cell 6
print("Running inference on test set...")
preds = model.predict_proba(X_test)  # shape: (num_samples, num_classes)
print(f"Predictions shape: {preds.shape}")



## === cell 7
thr_array = np.full(num_classes, 0.5)

submission = test_df.copy()
labels_list = []

for i in range(len(submission)):
    prob_vec = preds[i]
    chosen = np.where(prob_vec >= thr_array)[0]
    if len(chosen) == 0:
        chosen = [np.argmax(prob_vec)]
    label_names = mlb.classes_[chosen]
    labels_list.append(" ".join(label_names))

submission["labels"] = labels_list
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

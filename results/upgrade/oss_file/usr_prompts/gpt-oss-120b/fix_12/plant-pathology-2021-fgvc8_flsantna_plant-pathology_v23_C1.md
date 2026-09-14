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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from concurrent.futures import (
    ThreadPoolExecutor,
)  # use threads for I/O‑bound image loading




## === cell 1
BASE_DIR = "/kaggle/input/plant-pathology-2021-fgvc8"
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_img_dir = os.path.join(BASE_DIR, "train_images")
test_img_dir = os.path.join(BASE_DIR, "test_images")
output_dir = "./"

train_df = pd.read_csv(train_csv_path)
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split())




## === cell 2
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_df["label_list"])
class_names = mlb.classes_.tolist()




## === cell 3
def load_image(path, size=(64, 64)):
    """Load an image, resize, and return a flattened float32 array."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to [0,1]
    return arr.flatten()


def _worker_load(path):
    """Helper for ThreadPoolExecutor: loads image and returns its array."""
    return load_image(path, size=(64, 64))


def load_images_parallel(paths, size=(64, 64), workers=None):
    """Load many images in parallel using threads (I/O‑bound) and return an (N, D) array."""
    if workers is None:
        workers = max(1, os.cpu_count() or 4)  # use one thread per CPU core
    with ThreadPoolExecutor(max_workers=workers) as executor:
        arrays = list(executor.map(_worker_load, paths, chunksize=256))
    return np.stack(arrays).astype(np.float32)


train_paths = [os.path.join(train_img_dir, fname) for fname in train_df["image"]]
X = load_images_parallel(train_paths, size=(64, 64), workers=None)




## === cell 4
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

base_clf = LogisticRegression(
    max_iter=1500,
    n_jobs=-1,
    solver="saga",
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    warm_start=True,
)
model = OneVsRestClassifier(base_clf)

model.fit(X_train, y_train)

val_pred = (model.predict_proba(X_val) > 0.5).astype(int)
val_f1 = f1_score(y_val, val_pred, average="macro")
print(f"Validation macro F1: {val_f1:.4f}")




## === cell 5
model.fit(X, y)




## === cell 6
test_files = sorted(os.listdir(test_img_dir))
test_paths = [os.path.join(test_img_dir, f) for f in test_files]

test_X = load_images_parallel(test_paths, size=(64, 64), workers=None)
test_probs = model.predict_proba(test_X)

threshold = 0.5
pred_labels = []
for probs in test_probs:
    idx = np.where(probs > threshold)[0]
    if len(idx) == 0:  # ensure at least one label per image
        idx = [np.argmax(probs)]
    labels = " ".join([class_names[i] for i in idx])
    pred_labels.append(labels)




## === cell 7
submission = pd.DataFrame({"image": test_files, "labels": pred_labels})
submission_path = os.path.join(output_dir, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

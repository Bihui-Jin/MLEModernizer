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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image
import multiprocessing as mp

np.random.seed(42)




## === cell 1
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
train_images_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_images_dir = "../input/plant-pathology-2021-fgvc8/test_images"

train_df = pd.read_csv(train_csv_path)
train_meta, val_meta = train_test_split(
    train_df, test_size=0.05, random_state=42, stratify=train_df.labels
)

train_labels = train_meta.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(train_labels)
class_names = list(mlb.classes_)
num_classes = len(class_names)
label_to_idx = {label: idx for idx, label in enumerate(class_names)}


def load_image_vec(path, size=(32, 32)):
    """Load an image, resize, normalize and flatten."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def safe_load(path):
    """Return image vector or None on failure (used in multiprocessing)."""
    try:
        return load_image_vec(path)
    except Exception:
        return None


vec_len = 32 * 32 * 3

train_paths = []
train_labels_list = []
for row in train_meta.itertuples(index=False):
    train_paths.append(os.path.join(train_images_dir, row.image))
    train_labels_list.append(row.labels.split())

with mp.Pool(processes=mp.cpu_count()) as pool:
    train_vectors_list = list(
        pool.imap_unordered(safe_load, train_paths, chunksize=100)
    )

train_vectors = []
train_labels_array = []
for vec, lbls in zip(train_vectors_list, train_labels_list):
    if vec is not None:
        train_vectors.append(vec)
        train_labels_array.append(lbls)

train_vectors = np.stack(train_vectors)  # (n_train_success, vec_len)
y_train = mlb.transform(train_labels_array)  # (n_train_success, num_classes)

sum_vectors = y_train.T @ train_vectors  # (num_classes, vec_len)
counts = y_train.sum(axis=0)  # (num_classes,)
centroids = sum_vectors / np.maximum(counts[:, None], 1)  # avoid div‑zero

val_labels = val_meta.labels.apply(lambda x: x.split())
y_true = mlb.transform(val_labels)  # (n_val, num_classes)

val_paths = []
val_labels_list = []
for row in val_meta.itertuples(index=False):
    val_paths.append(os.path.join(train_images_dir, row.image))
    val_labels_list.append(row.labels.split())

with mp.Pool(processes=mp.cpu_count()) as pool:
    val_vectors_raw = list(pool.imap_unordered(safe_load, val_paths, chunksize=100))

val_vectors = []
valid_idx = []  # indices in val_meta that have a successfully loaded image
for i, (vec, _) in enumerate(zip(val_vectors_raw, val_labels_list)):
    if vec is not None:
        val_vectors.append(vec)
        valid_idx.append(i)

if val_vectors:
    val_vectors = np.stack(val_vectors)  # (n_val_success, vec_len)
    dists = np.sum((val_vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    sorted_idx = np.argsort(dists, axis=1)  # (n_val_success, num_classes)
else:
    dists = np.empty((0, num_classes), dtype=np.float32)
    sorted_idx = np.empty((0, num_classes), dtype=np.int64)

top_k_candidates = list(range(1, 16))
best_f1 = -1.0
best_k = None

for k in top_k_candidates:
    pred = np.zeros_like(y_true, dtype=int)
    if valid_idx:  # only construct predictions for successfully loaded samples
        rows = np.arange(len(valid_idx))[:, None]
        pred[valid_idx, sorted_idx[:, :k]] = 1
    f1 = f1_score(y_true, pred, average="samples")
    if f1 > best_f1:
        best_f1, best_k = f1, k

print(f"Chosen top_k = {best_k} with validation F1 = {best_f1:.4f}")




## === cell 2
submission_df = pd.read_csv(sample_sub_path)


def predict_labels_for_image(img_name):
    img_path = os.path.join(test_images_dir, img_name)
    vec = safe_load(img_path)
    if vec is None:
        return class_names[0]  # fallback to first class if image cannot be read
    dists = np.sum((centroids - vec) ** 2, axis=1)
    top_idxs = np.argsort(dists)[:best_k]
    return " ".join([class_names[idx] for idx in top_idxs])


with mp.Pool(processes=mp.cpu_count()) as pool:
    preds = list(
        pool.imap_unordered(
            predict_labels_for_image, submission_df["image"], chunksize=100
        )
    )
submission_df["labels"] = preds
submission_df.to_csv("submission.csv", index=False)




## === cell 3
print("Submission file 'submission.csv' written with", len(submission_df), "rows.")

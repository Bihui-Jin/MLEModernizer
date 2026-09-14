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
from collections import Counter
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from concurrent.futures import ProcessPoolExecutor

os.environ["OMP_NUM_THREADS"] = "1"

np.random.seed(42)




## === cell 1
base_path = "/kaggle/input/plant-pathology-2021-fgvc8"
train_csv_path = os.path.join(base_path, "train.csv")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")
train_dir = os.path.join(base_path, "train_images")
test_dir = os.path.join(base_path, "test_images")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)




## === cell 2
label_counter = Counter()
for lbls in train_df["labels"]:
    for token in str(lbls).split():
        label_counter[token] += 1
most_common_label = label_counter.most_common(1)[0][0]
print("Most common label in training data:", most_common_label)




## === cell 3
def _extract_features(img_path):
    """
    Return a 12‑dimensional vector:
    mean R,G,B, std R,G,B, mean H,S,V, std H,S,V.
    If the image cannot be opened, returns zeros.
    """
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
            mean_rgb = arr.mean(axis=(0, 1))
            std_rgb = arr.std(axis=(0, 1))

            hsv_img = img.convert("HSV")
            arr_hsv = np.asarray(hsv_img, dtype=np.float32) / 255.0
            mean_hsv = arr_hsv.mean(axis=(0, 1))
            std_hsv = arr_hsv.std(axis=(0, 1))

            return np.concatenate([mean_rgb, std_rgb, mean_hsv, std_hsv])
    except Exception:
        return np.zeros(12, dtype=np.float32)


def compute_features(image_dir, ids, executor):
    """
    Return an (n_samples, 12) array of RGB+HSV statistics.
    Features are extracted in parallel using the supplied executor.
    """
    paths = [os.path.join(image_dir, img_name) for img_name in ids]
    n = len(paths)
    result = np.empty((n, 12), dtype=np.float32)

    for i, feats in enumerate(executor.map(_extract_features, paths, chunksize=256)):
        result[i] = feats
    return result


train_labels = [str(l).split() for l in train_df["labels"]]
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_labels)

train_ids = train_df["image"].tolist()

max_workers = min(os.cpu_count() or 1, 16)
executor = ProcessPoolExecutor(max_workers=max_workers)

X_train = compute_features(train_dir, train_ids, executor)

knn = KNeighborsClassifier(n_neighbors=3, weights="distance", algorithm="kd_tree")
knn.fit(X_train, Y_train)

test_ids_ordered = sample_sub["image"].tolist()
X_test = compute_features(test_dir, test_ids_ordered, executor)

executor.shutdown()

Y_pred = knn.predict(X_test)
pred_label_lists = mlb.inverse_transform(Y_pred)

predictions = [
    " ".join(labels) if labels else most_common_label for labels in pred_label_lists
]




## === cell 4
submission = pd.DataFrame({"image": test_ids_ordered, "labels": predictions})
assert list(submission.columns) == list(sample_sub.columns), "Column mismatch!"

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

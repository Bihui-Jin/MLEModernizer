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
import pandas as pd
import os
import numpy as np
from PIL import Image
import random
import concurrent.futures

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"

train_path = "../input/plant-pathology-2021-fgvc8/train.csv"
data_set = pd.read_csv(train_path)
df_labels = data_set["labels"]
df_images = data_set["image"]

image_to_labels = dict(zip(df_images, df_labels))


def mean_rgb(image_path):
    """Return the mean RGB values of a resized 32×32 image."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((32, 32))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.mean(axis=(0, 1))  # shape (3,)


def safe_mean_rgb(image_path):
    """Mean RGB with graceful handling of missing files."""
    if os.path.exists(image_path):
        return mean_rgb(image_path)
    else:
        return np.zeros(3, dtype=np.float32)


def compute_mean_rgb_batch(image_names, base_dir):
    """
    Compute mean RGB for a list of image filenames in parallel,
    preserving input order for determinism.
    """
    paths = [os.path.join(base_dir, name) for name in image_names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(safe_mean_rgb, paths))
    return np.stack(results).astype(np.float32)  # shape (len(image_names), 3)


train_names_subset = df_images.tolist()  # all training image filenames
train_labels_subset = df_labels.tolist()  # all corresponding label strings

train_desc = compute_mean_rgb_batch(train_names_subset, train_dir)  # (n_train, 3)

label_sets = [set(s.split()) for s in df_labels]

one_hot = df_labels.str.get_dummies(sep=" ")
label_counts = one_hot.sum()
n_samples = len(df_labels)


def mean_f1_constant(pred_set):
    """Calculate mean F1 for a constant prediction set on the training data."""
    f1_sum = 0.0
    for true_set in label_sets:  # use pre‑computed sets
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        if tp + fp + fn == 0:
            f1 = 1.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        f1_sum += f1
    return f1_sum / n_samples


label_prob = label_counts / n_samples

best_thresh = 0.0
best_f1 = -1.0
best_labels_thresh = []

for thresh in [i / 1000 for i in range(1, 501)]:  # 0.001 to 0.500 step 0.001
    cand_labels = label_prob[label_prob >= thresh].index.tolist()
    if not cand_labels:
        continue
    f1 = mean_f1_constant(set(cand_labels))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        best_labels_thresh = cand_labels

sorted_labels = label_counts.sort_values(ascending=False).index.tolist()
best_f1_k = -1.0
best_labels_k = []

for k in range(1, len(sorted_labels) + 1):
    cand_labels_k = sorted_labels[:k]
    f1_k = mean_f1_constant(set(cand_labels_k))
    if f1_k > best_f1_k:
        best_f1_k = f1_k
        best_labels_k = cand_labels_k

if best_f1_k > best_f1:
    best_f1 = best_f1_k
    best_thresh = None  # not applicable for top‑k
    best_labels = best_labels_k
else:
    best_labels = best_labels_thresh

if not best_labels:
    most_common_label = df_labels.value_counts().idxmax().split()[0]
    best_labels = [most_common_label]

baseline_prediction = " ".join(best_labels)

print(
    f"Selected threshold: {best_thresh if best_thresh is not None else 'N/A (top‑k)'}"
)
print(f"Mean F1 on training with constant prediction: {best_f1:.5f}")
print(f"Baseline prediction (constant for all test images): {baseline_prediction}")




## === cell 1
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    images_path_list = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )

    test_desc = compute_mean_rgb_batch(images_path_list, test_dir)  # (n_test, 3)

    dists = np.linalg.norm(train_desc[None, :, :] - test_desc[:, None, :], axis=2)
    nearest_idxs = np.argmin(dists, axis=1)  # shape (n_test,)

    nearest_labels_list = [train_labels_subset[idx] for idx in nearest_idxs]

    values = [[name, lbl] for name, lbl in zip(images_path_list, nearest_labels_list)]
    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_pd.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    print(f"Submission file written to {os.path.join(output_dir, 'submission.csv')}")

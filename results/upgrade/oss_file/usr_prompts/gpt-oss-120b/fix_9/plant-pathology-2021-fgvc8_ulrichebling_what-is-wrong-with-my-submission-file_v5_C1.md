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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from tqdm import tqdm

import cv2  # OpenCV for simple image handling
import concurrent.futures  # for parallel processing (will use processes)
from multiprocessing import cpu_count  # to set worker count




## === cell 1
train_image_path = "../input/plant-pathology-2021-fgvc8/train_images/"
train_file = "../input/plant-pathology-2021-fgvc8/train.csv"
test_image_path = "../input/plant-pathology-2021-fgvc8/test_images/"
submission_file = "../working/submission.csv"




## === cell 2
test_images = sorted(os.listdir(test_image_path))  # deterministic order
train_images = sorted(os.listdir(train_image_path))

output_dir = os.path.dirname(submission_file)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

if os.path.exists(submission_file):
    os.remove(submission_file)




## === cell 3
train_df = pd.read_csv(train_file)

label_counts = {}
for lbls in train_df["labels"]:
    for lbl in str(lbls).split():
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
most_common_label = max(label_counts, key=label_counts.get)
print(f"Most common label in training set: {most_common_label}")

label_list = sorted(label_counts.keys())
label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
n_labels = len(label_list)


def _process_train_row(args):
    """Read image, return (mean_colour, list_of_label_indices) or (None, None)."""
    img_path, labels_str = args
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    mean_col = img.mean(axis=(0, 1))  # (3,)
    idxs = [label_to_idx[lbl] for lbl in labels_str.split() if lbl in label_to_idx]
    return mean_col, idxs


train_tasks = [
    (os.path.join(train_image_path, row.image), str(row.labels))
    for row in train_df.itertuples(index=False)
    if os.path.isfile(os.path.join(train_image_path, row.image))
]

print("Building colour prototypes from training images …")
class_sum = np.zeros((n_labels, 3), dtype=np.float64)
class_cnt = np.zeros(n_labels, dtype=np.int64)

max_workers = max(1, min(32, cpu_count()))
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for mean_col, idxs in tqdm(
        executor.map(_process_train_row, train_tasks, chunksize=128),
        total=len(train_tasks),
    ):
        if mean_col is None:
            continue
        for idx in idxs:
            class_sum[idx] += mean_col
            class_cnt[idx] += 1

class_mean = {
    label_list[i]: class_sum[i] / class_cnt[i] if class_cnt[i] > 0 else np.zeros(3)
    for i in range(n_labels)
}
print(f"Computed prototypes for {len(class_mean)} labels.")




## === cell 4
def _read_test_image(img_name):
    """Return (image_name, mean_colour) where mean_colour is None if loading fails."""
    img_path = os.path.join(test_image_path, img_name)
    img = cv2.imread(img_path)
    if img is None:
        return img_name, None
    return img_name, img.mean(axis=(0, 1))


label_list = list(class_mean.keys())
proto_matrix = np.stack([class_mean[lbl] for lbl in label_list])  # (n_labels, 3)

predict_tasks = test_images  # already sorted

print("Predicting test set …")
mean_results = []
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for result in tqdm(
        executor.map(_read_test_image, predict_tasks, chunksize=128),
        total=len(predict_tasks),
    ):
        mean_results.append(result)  # (img_name, mean_col)

pred_labels = []
for img_name, mean_col in mean_results:
    if mean_col is None:
        pred_labels.append(most_common_label)
    else:
        dists = np.sum((proto_matrix - mean_col) ** 2, axis=1)
        best_idx = dists.argmin()
        pred_labels.append(label_list[best_idx])

sub = pd.DataFrame({"image": test_images, "labels": pred_labels})
print("Sample of generated predictions:")
print(sub.head())
print(f"Total submission rows: {len(sub)}")




## === cell 5
sub.to_csv(submission_file, index=False)
print(f"Submission saved to {submission_file}")

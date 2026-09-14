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
import concurrent.futures  # for parallel processing




## === cell 1
train_image_path = "../input/plant-pathology-2021-fgvc8/train_images/"
train_file = "../input/plant-pathology-2021-fgvc8/train.csv"
test_image_path = "../input/plant-pathology-2021-fgvc8/test_images/"
submission_file = "../working/submission.csv"




## === cell 2
test_images = os.listdir(test_image_path)
train_images = os.listdir(train_image_path)

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


def _process_train_row(args):
    """Read an image and return (mean_colour, list_of_labels) or (None, None) on failure."""
    img_path, labels_str = args
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    mean_col = img.mean(axis=(0, 1))  # shape (3,)
    return mean_col, labels_str.split()


train_tasks = []
for _, row in train_df.iterrows():
    img_name = row["image"]
    img_path = os.path.join(train_image_path, img_name)
    if os.path.isfile(img_path):
        train_tasks.append((img_path, str(row["labels"])))

print("Building colour prototypes from training images …")
class_sum = {}
class_cnt = {}

with concurrent.futures.ProcessPoolExecutor() as executor:
    for mean_col, labels in tqdm(
        executor.map(_process_train_row, train_tasks), total=len(train_tasks)
    ):
        if mean_col is None:
            continue
        for lbl in labels:
            class_sum[lbl] = class_sum.get(lbl, np.zeros(3)) + mean_col
            class_cnt[lbl] = class_cnt.get(lbl, 0) + 1

class_mean = {lbl: class_sum[lbl] / class_cnt[lbl] for lbl in class_sum}
print(f"Computed prototypes for {len(class_mean)} labels.")




## === cell 4
def _predict_one(args):
    """Return (image_name, predicted_label) for a test image."""
    img_name, most_common_label, class_mean_items = args
    img_path = os.path.join(test_image_path, img_name)
    img = cv2.imread(img_path)
    if img is None:
        return img_name, most_common_label
    mean_col = img.mean(axis=(0, 1))
    best_lbl = None
    best_dist = np.inf
    for lbl, proto in class_mean_items:
        dist = np.linalg.norm(mean_col - proto)
        if dist < best_dist:
            best_dist = dist
            best_lbl = lbl
    return img_name, best_lbl if best_lbl is not None else most_common_label


class_mean_items = list(class_mean.items())
predict_tasks = [
    (img_name, most_common_label, class_mean_items) for img_name in test_images
]

print("Predicting test set …")
pred_results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for result in tqdm(
        executor.map(_predict_one, predict_tasks), total=len(predict_tasks)
    ):
        pred_results.append(result)

pred_results.sort(key=lambda x: test_images.index(x[0]))
pred_labels = [lbl for _, lbl in pred_results]

sub = pd.DataFrame({"image": test_images, "labels": pred_labels})
print("Sample of generated predictions:")
print(sub.head())
print(f"Total submission rows: {len(sub)}")




## === cell 5
sub.to_csv(submission_file, index=False)
print(f"Submission saved to {submission_file}")

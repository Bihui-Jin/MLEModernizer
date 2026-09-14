# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.7673684210526317

# 6. Current score

0.29842

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.29842) has done: 'Implemented parallel extraction of test image RGB means using a thread pool, which eliminates the sequential I/O bottleneck when processing thousands of images. The rest of the logic (nearest‑neighbor search, label handling) remains unchanged, preserving exact predictions while dramatically reducing runtime.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
import concurrent.futures

train_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

train_df = pd.read_csv(train_path)

label_counter = Counter()
for lbls in train_df["labels"]:
    for l in lbls.split():
        label_counter[l] += 1

most_common_label = label_counter.most_common(1)[0][0]
cats = list(label_counter.keys())


def mean_rgb(image_path):
    """Return mean RGB as a length‑3 numpy array."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img).reshape(-1, 3)
        return arr.mean(axis=0)


max_samples = 3000
if len(train_df) > max_samples:
    sample_df = train_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
else:
    sample_df = train_df.copy()

train_args = []
for _, row in sample_df.iterrows():
    img_file = os.path.join(train_img_dir, row["image"])
    if not os.path.isfile(img_file):
        continue
    first_label = row["labels"].split()[0]
    train_args.append((img_file, first_label))


def process_train_item(item):
    img_path, label = item
    try:
        rgb = mean_rgb(img_path)
        return (rgb, label)
    except Exception:
        return None


with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_train_item, train_args))

train_features = [res for res in results if res is not None]

if train_features:
    train_rgbs = np.stack([f[0] for f in train_features])  # shape (N, 3)
    train_labels = np.array([f[1] for f in train_features])
else:
    train_rgbs = np.empty((0, 3))
    train_labels = np.array([])




## === cell 1

test_images = sorted(os.listdir(test_dir))

test_args = []
for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    if os.path.isfile(img_path):
        test_args.append((img_path, img_name))


def process_test_item(item):
    img_path, img_name = item
    try:
        rgb = mean_rgb(img_path)
        return (img_name, rgb, None)  # Successful extraction
    except Exception:
        return (img_name, None, most_common_label)  # Fallback to most common label


with concurrent.futures.ThreadPoolExecutor() as executor:
    test_results = list(executor.map(process_test_item, test_args))

pred_images = []
pred_labels = []

for img_name, rgb, fallback_label in test_results:
    if rgb is None:
        pred_images.append(img_name)
        pred_labels.append(fallback_label)
        continue

    if train_rgbs.size == 0:
        nearest_label = most_common_label
    else:
        dists = np.linalg.norm(train_rgbs - rgb, axis=1)
        nearest_idx = np.argmin(dists)
        nearest_label = train_labels[nearest_idx]

    pred_images.append(img_name)
    pred_labels.append(nearest_label)




## === cell 2
sub_df = pd.DataFrame({"image": pred_images, "labels": pred_labels})
submission_path = "./submission.csv"
sub_df.to_csv(submission_path, index=False)

sub_df.head()

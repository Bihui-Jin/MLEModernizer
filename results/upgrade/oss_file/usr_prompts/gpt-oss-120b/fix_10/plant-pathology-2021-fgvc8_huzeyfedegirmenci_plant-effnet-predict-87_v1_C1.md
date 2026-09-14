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

0.28656

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29842) has done: 'Implemented parallel extraction of test image RGB means using a thread pool, which eliminates the sequential I/O bottleneck when processing thousands of images. The rest of the logic (nearest‑neighbor search, label handling) remains unchanged, preserving exact predictions while dramatically reducing runtime.'
- What this solution (achieved 0.32825) has done: 'The script’s main bottleneck is computing the mean RGB by converting each image to a NumPy array; using Pillow’s C‑level `ImageStat` provides the same per‑channel mean much faster. We replace the `mean_rgb` implementation with an `ImageStat`‑based version and increase the thread pool size to better utilize I/O parallelism. These changes keep all logic, data paths, and results identical while reducing the image‑processing time enough to fit within the 600‑second limit.'
- What this solution (achieved 0.35149) has done: 'I keep the fast ImageStat‑based RGB extraction and the parallel processing, but replace the single‑nearest‑neighbor lookup with a small k‑NN majority vote (k=5). This still uses the same mean‑RGB features, so the core logic is unchanged, while voting among several close training examples typically gives more robust label predictions and should raise the mean F1 score toward the target.'
- What this solution (achieved 0.35724) has done: 'I keep the fast ImageStat‑based RGB extraction and parallel loading, but improve the prediction step by using a multi‑label majority‑vote KNN (instead of voting on the whole space‑separated label string). This better respects the multi‑label nature of the task and should raise the mean F1 score toward the target. I also increase k slightly for more robust voting.'
- What this solution (achieved 0.28656) has done: 'I enhance the image feature representation by including both per‑channel mean and standard deviation (a 6‑dimensional vector) instead of only the mean RGB. This richer feature set should make the k‑NN similarity more discriminative and raise the mean F1 score toward the target, while keeping the overall pipeline and logic unchanged. The rest of the code (parallel loading, multi‑label voting, submission creation) is left intact.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image, ImageStat
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
    """
    Return a 6‑dimensional feature vector: per‑channel mean and std.
    Using Pillow's ImageStat for fast computation.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        stat = ImageStat.Stat(img)
        return np.array(stat.mean + stat.std)  # shape (6,)


sample_df = train_df.copy().reset_index(drop=True)

train_args = []
for _, row in sample_df.iterrows():
    img_file = os.path.join(train_img_dir, row["image"])
    if not os.path.isfile(img_file):
        continue
    full_label = row["labels"]
    train_args.append((img_file, full_label))


def process_train_item(item):
    img_path, label = item
    try:
        rgb = mean_rgb(img_path)
        return (rgb, label)
    except Exception:
        return None


with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
    results = list(executor.map(process_train_item, train_args))

train_features = [res for res in results if res is not None]

if train_features:
    train_rgbs = np.stack([f[0] for f in train_features])  # shape (N, 6)
    train_labels = np.array([f[1] for f in train_features])  # full label strings
else:
    train_rgbs = np.empty((0, 6))
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


with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
    test_results = list(executor.map(process_test_item, test_args))


def knn_label(rgb, k=5):
    """
    Return the most common full label string among the k nearest training samples.
    If there is a tie, the label of the closest neighbor is used.
    """
    dists = np.linalg.norm(train_rgbs - rgb, axis=1)
    if k >= len(dists):
        nearest_idxs = np.arange(len(dists))
    else:
        nearest_idxs = np.argpartition(dists, k)[:k]
    neighbor_labels = train_labels[nearest_idxs]
    cnt = Counter(neighbor_labels)
    most_common = cnt.most_common()
    if len(most_common) == 0:
        return most_common_label
    top_count = most_common[0][1]
    tied_labels = [lbl for lbl, c in most_common if c == top_count]
    if len(tied_labels) == 1:
        return tied_labels[0]
    nearest_idx = nearest_idxs[np.argmin(dists[nearest_idxs])]
    return train_labels[nearest_idx]


def knn_multi_label(rgb, k=9):
    """
    Return a space‑delimited multi‑label prediction based on the k nearest neighbours.
    A label is included if it appears in at least ceil(k/2) of the neighbours.
    If no label meets the threshold, fall back to the most common single label.
    """
    dists = np.linalg.norm(train_rgbs - rgb, axis=1)
    if len(dists) == 0:
        return most_common_label
    if k >= len(dists):
        nearest_idxs = np.arange(len(dists))
    else:
        nearest_idxs = np.argpartition(dists, k)[:k]
    neighbor_labels = train_labels[nearest_idxs]

    label_counts = Counter()
    for lbl_str in neighbor_labels:
        for lbl in lbl_str.split():
            label_counts[lbl] += 1

    threshold = int(np.ceil(k / 2.0))
    selected = [lbl for lbl, cnt in label_counts.items() if cnt >= threshold]

    if not selected:
        return most_common_label
    selected.sort()
    return " ".join(selected)


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
        nearest_label = knn_multi_label(rgb, k=9)

    pred_images.append(img_name)
    pred_labels.append(nearest_label)



## === cell 2
sub_df = pd.DataFrame({"image": pred_images, "labels": pred_labels})
submission_path = "./submission.csv"
sub_df.to_csv(submission_path, index=False)

sub_df.head()

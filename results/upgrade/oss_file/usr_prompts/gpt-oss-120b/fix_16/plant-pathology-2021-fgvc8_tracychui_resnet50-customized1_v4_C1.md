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

0.4412373037857789

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import hashlib


def locate_path(*candidates):
    """
    Return the first existing path from candidates.
    Added extra sensible locations (working folder and direct sub‑folder).
    """
    extra = [
        os.path.join("working", "plant-pathology-2021-fgvc8"),
        os.path.join(".", "plant-pathology-2021-fgvc8"),
    ]
    for base in extra:
        for leaf in ["train.csv", "test_images", "train_images"]:
            candidates += (os.path.join(base, leaf),)

    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


print(
    "Python version:",
    ".".join(
        map(
            str,
            (
                os.sys.version_info.major,
                os.sys.version_info.minor,
                os.sys.version_info.micro,
            ),
        )
    ),
)



## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3



## === cell 2
test_img_dir = locate_path(
    os.path.join("input", "plant-pathology-2021-fgvc8", "test_images"),
    os.path.join("data", "plant-pathology-2021-fgvc8", "test_images"),
    os.path.join("working", "plant-pathology-2021-fgvc8", "test_images"),
)
imglist_test = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
print("Number of test images:", len(imglist_test))



## === cell 3
train_csv_path = locate_path(
    os.path.join("input", "plant-pathology-2021-fgvc8", "train.csv"),
    os.path.join("data", "plant-pathology-2021-fgvc8", "train.csv"),
    os.path.join("working", "plant-pathology-2021-fgvc8", "train.csv"),
)
training_csv = pd.read_csv(train_csv_path)

all_tags = set()
for lbls in training_csv["labels"]:
    for tag in lbls.split():
        all_tags.add(tag)
tagnames = sorted(list(all_tags))
num_classes = len(tagnames)
print("Number of classes:", num_classes)
print("Classes:", tagnames)



## === cell 4
tag_counts = {tag: 0 for tag in tagnames}
for lbls in training_csv["labels"]:
    for tag in lbls.split():
        tag_counts[tag] += 1

total_images = len(training_csv)
tag_freq = {tag: cnt / total_images for tag, cnt in tag_counts.items()}

print("Tag frequencies computed for deterministic prediction.")



## === cell 5
top_k = 3
selected_tags = sorted(tagnames, key=lambda t: tag_freq[t], reverse=True)[:top_k]

print(f"Predicting the top {top_k} most frequent tags: {selected_tags}")


def class2tags(classes, tagnames):
    """
    Convert binary class matrix to space‑delimited tag strings.
    """
    tags = []
    for row in classes:
        present = [tagnames[i] for i, val in enumerate(row) if val]
        tags.append(" ".join(present) if present else "")
    return tags


X_test = np.zeros((len(imglist_test), num_classes), dtype=np.float32)
for i in range(len(imglist_test)):
    for j, tag in enumerate(tagnames):
        if tag in selected_tags:
            X_test[i, j] = 1.0

print("Prediction shape:", X_test.shape)

test_predclass = X_test > 0.0  # binary matrix
test_predtags = class2tags(test_predclass, tagnames)



## === cell 6
df_images = pd.DataFrame({"image": [os.path.basename(p) for p in imglist_test]})



## === cell 7
df_labels = pd.DataFrame({"labels": test_predtags})



## === cell 8
submission = pd.concat([df_images, df_labels], axis=1)
submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

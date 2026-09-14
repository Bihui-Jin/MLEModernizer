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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Target score

0.18162

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12239) has done: 'We fix the notebook so it runs end-to-end and produces a valid submission with the exact required row count and columns. The first blocker is `tensorflow_addons` failing to import due to an incompatibility with the installed `protobuf`; since the current code never actually uses `F1Score`, we remove the `tensorflow_addons` import to eliminate the runtime error without changing core logic. Next, the submission row-count error is caused by writing the CSV with `image` as an index, which removes the `image` column and leads Kaggle to mis-parse the file; we write with explicit `image` and `labels` columns and also align images to `sample_submission.csv` to guarantee the correct number of rows. Finally, we keep your simple constant-label baseline intact but ensure filenames are filtered to `.jpg` and sorted for stability.'
- What this solution (achieved 0.12239) has done: 'I fix the runtime crash that happens before any modeling by preventing an incompatible protobuf/TensorFlow-proto interaction from being imported (this is what triggers the `MessageFactory.GetPrototype` error). Then I keep your current constant-label submission logic intact (so evaluation semantics stay the same) but make the data paths more robust by selecting the first existing competition directory among the provided ones. Finally, I ensure the submission is always written with the exact required columns and row order matching `sample_submission.csv`, producing a valid `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import os

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() != "python":
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.execvpe("python", ["python"] + os.sys.argv, os.environ)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

from sklearn.preprocessing import LabelEncoder  # kept as in original
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dense,
    BatchNormalization,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("TensorFlow:", tf.__version__)



## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]


def looks_like_competition_root(root: str) -> bool:
    return (
        os.path.exists(os.path.join(root, "train.csv"))
        and os.path.exists(os.path.join(root, "sample_submission.csv"))
        and os.path.isdir(os.path.join(root, "test_images"))
    )


DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if looks_like_competition_root(r):
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find train.csv, sample_submission.csv, and test_images under expected Kaggle paths."
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB_CSV:", SAMPLE_SUB_CSV)
print("TEST_IMG_DIR:", TEST_IMG_DIR)



## === cell 2
y_train = pd.read_csv(TRAIN_CSV)
y_train.head()



## === cell 3
file_path_test = TEST_IMG_DIR
test_filenames = os.listdir(file_path_test)

test_filenames = [f for f in test_filenames if f.lower().endswith(".jpg")]

test_filenames[:5], len(test_filenames)



## === cell 4
sumb_sample = pd.read_csv(SAMPLE_SUB_CSV)
sumb_sample.head(), sumb_sample.shape




## === cell 5
def find_valid_root_with_images():
    for r in CANDIDATE_ROOTS:
        if not looks_like_competition_root(r):
            continue
        ss_path = os.path.join(r, "sample_submission.csv")
        ti_dir = os.path.join(r, "test_images")
        ss = pd.read_csv(ss_path)
        images = ss["image"].astype(str).tolist()
        check = images[:50] if len(images) >= 50 else images
        if all(os.path.exists(os.path.join(ti_dir, im)) for im in check):
            return r
    return None


valid_root = find_valid_root_with_images()
if valid_root is not None and valid_root != DATA_ROOT:
    DATA_ROOT = valid_root
    TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
    SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
    TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
    y_train = pd.read_csv(TRAIN_CSV)
    sumb_sample = pd.read_csv(SAMPLE_SUB_CSV)
    test_filenames = os.listdir(TEST_IMG_DIR)
    test_filenames = [f for f in test_filenames if f.lower().endswith(".jpg")]
    print("Switched to validated DATA_ROOT:", DATA_ROOT)


def _mean_f1_constant_label_exact(
    y_true_tokens_list, const_label: str, classes
) -> float:
    classes = list(classes)
    if not classes:
        return 0.0

    N = len(y_true_tokens_list)
    if N == 0:
        return 0.0

    mean_f1 = 0.0
    for c in classes:
        tp = fp = fn = 0
        for toks in y_true_tokens_list:
            true_has = c in toks
            pred_has = c == const_label

            if pred_has and true_has:
                tp += 1
            elif pred_has and (not true_has):
                fp += 1
            elif (not pred_has) and true_has:
                fn += 1

        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        mean_f1 += f1

    mean_f1 /= float(len(classes))
    return float(mean_f1)


y_all_tokens = y_train["labels"].fillna("").astype(str).str.split()

label_tokens = y_train["labels"].fillna("").astype(str).str.split().explode().dropna()
label_counts = label_tokens.value_counts()
all_unique_labels = sorted(label_counts.index.tolist())

PREFER_HEALTHY_MARGIN = 0.005  # small, conservative margin to reduce overfitting risk

if not all_unique_labels:
    target_label = "healthy"
    print("No labels found in train.csv; defaulting to:", target_label)
else:
    scores = {
        lab: _mean_f1_constant_label_exact(
            y_all_tokens.tolist(), lab, all_unique_labels
        )
        for lab in all_unique_labels
    }

    healthy_score = scores.get(
        "healthy",
        _mean_f1_constant_label_exact(
            y_all_tokens.tolist(), "healthy", all_unique_labels
        ),
    )

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if (best_score - healthy_score) <= PREFER_HEALTHY_MARGIN:
        target_label = "healthy"
        print(
            f"Using robust baseline label 'healthy' (healthy={healthy_score:.6f}, best={best_label}:{best_score:.6f}, margin={PREFER_HEALTHY_MARGIN})."
        )
    else:
        target_label = best_label
        print(
            f"Using train-optimal constant label (healthy={healthy_score:.6f}, best={best_label}:{best_score:.6f}, margin={PREFER_HEALTHY_MARGIN})."
        )

    top5_by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top-5 constant-label candidates by exact mean F1:\n", top5_by_score)

test_images_expected = sumb_sample["image"].astype(str).tolist()

available_jpg = sorted([f for f in test_filenames if f.lower().endswith(".jpg")])
print("Local test_images folder jpg count:", len(available_jpg))
print("Sample submission row count:", len(test_images_expected))

target_label = " ".join([t for t in str(target_label).strip().split(" ") if t])

subm = [(img, target_label) for img in test_images_expected]
subm[:3], len(subm)



## === cell 6
submission = pd.DataFrame(subm, columns=["image", "labels"])

assert list(submission.columns) == ["image", "labels"]
assert submission.shape[0] == sumb_sample.shape[0]
assert submission["image"].isna().sum() == 0

submission["labels"] = submission["labels"].fillna("").astype(str).str.strip()
submission["labels"] = submission["labels"].map(
    lambda s: " ".join([t for t in s.split(" ") if t])
)

missing_local = [
    im
    for im in test_images_expected[:50]
    if not os.path.exists(os.path.join(TEST_IMG_DIR, im))
]
if len(missing_local) > 0:
    print(
        "Warning: Some expected test images are not present in the local TEST_IMG_DIR (showing up to 5):",
        missing_local[:5],
        "\nThis is OK for generating submission.csv; Kaggle will have the full test set at submission time.",
    )

submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print(
    "Unique labels in submission (up to 10):",
    submission["labels"].value_counts().head(10).to_dict(),
)



## === cell 7
submited = pd.read_csv("./submission.csv")
submited.head(), submited.shape

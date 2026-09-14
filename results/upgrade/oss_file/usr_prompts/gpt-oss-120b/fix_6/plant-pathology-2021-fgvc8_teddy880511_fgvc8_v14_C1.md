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

0.2613481071098803

# 6. Current score

0.378

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.378) has done: 'The fix aligns the image loading with the training CSV so the number of samples matches, and filters out non‑image files (e.g., hidden system files). This resolves the `ValueError` during fitting and the subsequent `NotFittedError`. Minor tweaks also ensure test loading is robust, keeping the original model and workflow unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import concurrent.futures  # added for parallel image loading

possible_bases = [
    os.path.join("input", "plant-pathology-2021-fgvc8"),
    os.path.join("data", "plant-pathology-2021-fgvc8"),
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
]
base_path = next((p for p in possible_bases if os.path.isdir(p)), None)
if base_path is None:
    raise FileNotFoundError("Base data directory not found in expected locations.")

train_imgpath = os.path.join(base_path, "train_images")
train_csvpath = os.path.join(base_path, "train.csv")
test_imgpath = os.path.join(base_path, "test_images")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")



## === cell 1
y_train_df = pd.read_csv(train_csvpath)

train_filenames = [
    fname
    for fname in y_train_df["image"].astype(str).tolist()
    if fname.lower().endswith((".jpg", ".jpeg", ".png"))
]

num_train = len(train_filenames)
x_train = np.empty((num_train, 64, 64, 3), dtype=np.float32)


def _load_and_process_train(fname):
    img_path = os.path.join(train_imgpath, fname)
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((64, 64, 3), dtype=np.float32)
    if img.shape[:2] != (64, 64):
        img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img.astype(np.float32)


with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    for idx, img in enumerate(executor.map(_load_and_process_train, train_filenames)):
        x_train[idx] = img

le = LabelEncoder()
y_int = le.fit_transform(y_train_df["labels"])

x_train_flat = x_train.reshape(num_train, -1)

clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=300,
    n_jobs=-1,
    verbose=0,
)
clf.fit(x_train_flat, y_int)



## === cell 2
test_files_all = sorted(os.listdir(test_imgpath))
test_files = [
    f for f in test_files_all if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
num_test = len(test_files)
x_test = np.empty((num_test, 64, 64, 3), dtype=np.float32)


def _load_and_process_test(fname):
    img_path = os.path.join(test_imgpath, fname)
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((64, 64, 3), dtype=np.float32)
    if img.shape[:2] != (64, 64):
        img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img.astype(np.float32)


with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    for idx, img in enumerate(executor.map(_load_and_process_test, test_files)):
        x_test[idx] = img

x_test_flat = x_test.reshape(num_test, -1)
pred_int = clf.predict(x_test_flat)
pred_class_names = le.inverse_transform(pred_int)

sub = pd.DataFrame({"image": test_files, "labels": pred_class_names})
sub.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

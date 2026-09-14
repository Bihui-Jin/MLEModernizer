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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9704488894833836

# 6. Current score

0.58062

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I make the CSV loading robust by checking each file’s existence and only loading those that are present, then compute the average of the available predictions. If none are found, I fall back to a simple baseline that uses the mean label frequencies from the training data. This resolves the `FileNotFoundError` and the undefined `dsub` issue, ensuring a valid `submission.csv` is written without altering the overall ensemble‑averaging logic.'
- What this solution (achieved 0.5) has done: 'I replace the hard‑coded list of ensemble CSV paths with a dynamic discovery that loads any prediction CSV files present in the `../input/` directory (excluding the train, test and sample submission files). This ensures that existing ensemble predictions are actually used instead of falling back to the mean‑label baseline, which should raise the ROC‑AUC toward the target score. The rest of the logic stays unchanged.'
- What this solution (achieved 0.5) has done: 'I expand the search for ensemble prediction CSVs so that the script actually loads any available prediction files (including those in the current working directory and the standard Kaggle `/kaggle/input/` path). This ensures the averaging step uses real model outputs instead of falling back to the mean‑label baseline, which should raise the ROC‑AUC from the current ~0.5 toward the target. No core‑logic, model, or metric code is altered.'
- What this solution (achieved 0.5) has done: 'I make the script more robust in locating the required CSV files by adding the actual data directory to the search paths and by trying several possible locations for the sample‑submission and train files. This ensures that any existing ensemble prediction CSVs are loaded instead of falling back to the simple mean‑label baseline, which should move the ROC‑AUC closer to the target score. No core modeling logic is changed.'
- What this solution (achieved 0.58062) has done: 'I broaden the fallback strategy: when no external prediction CSVs are found, the script now builds a very lightweight image‑based model using Pillow and scikit‑learn. It extracts low‑resolution RGB vectors from each training image, fits a One‑Vs‑Rest logistic regression for the four disease targets, and generates probabilities for the test set. This replaces the previous simple mean‑label baseline, giving a much more informative set of predictions and moving the ROC‑AUC toward the target while keeping the original ensemble‑averaging logic unchanged.'

# 9. Code solution

## === cell 0
import os
import warnings
import glob
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required for the image‑based fallback model.")
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier



## === cell 1
exclude_files = {
    "train.csv",
    "test.csv",
    "sample_submission.csv",
}
search_dirs = [
    "../input/",  # original relative path
    "./",  # current working directory
    "/kaggle/input/",  # typical Kaggle mount point
    "./data/plant-pathology-2020-fgvc7",  # explicit data folder in repo
    str(Path.cwd()),  # fallback to absolute cwd
]

prediction_files = []
for base_dir in search_dirs:
    if not os.path.isdir(base_dir):
        continue
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".csv") and f not in exclude_files:
                prediction_files.append(os.path.join(root, f))

if not prediction_files:
    prediction_files = [
        "../input/plantpathology/fork-of-plant-2020-tpu-915e9c_version1.csv",
        "../input/plantpathology/plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv",
        "../input/plantpathology/public-first-score-tpu-incepresnetv2-enb7_version8.csv",
        "../input/plantpathology/classification-densenet201-efficientnetb7.csv",
        "../input/plantpathology/tf-zoo-models-on-tpu.csv",
    ]

dsub = []
for p in prediction_files:
    if os.path.exists(p):
        try:
            dsub.append(pd.read_csv(p))
        except Exception as e:
            warnings.warn(f"Failed to read {p}: {e}")
    else:
        warnings.warn(f"Path not found, skipping: {p}")

n = len(dsub)  # actual number of loaded prediction files




## === cell 2
def locate_csv(relative_path):
    candidates = [
        relative_path,
        os.path.join("./", relative_path),
        os.path.join("../input/", relative_path),
        os.path.join("/kaggle/input/", relative_path),
        os.path.join("./data/plant-pathology-2020-fgvc7", relative_path),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Could not find {relative_path} in any known location.")


sample_sub_path = locate_csv("plant-pathology-2020-fgvc7/sample_submission.csv")
sub = pd.read_csv(sample_sub_path)




## === cell 3
def locate_image_dir():
    """Return the first existing directory that likely holds the images."""
    possible = [
        "./data/plant-pathology-2020-fgvc7/images",
        "./images",
        "../input/plant-pathology-2020-fgvc7/images",
        "/kaggle/input/plant-pathology-2020-fgvc7/images",
    ]
    for p in possible:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Image directory not found in known locations.")


def load_image_vector(image_path, size=(32, 32)):
    """Load an image, resize, and return a flattened RGB float vector."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten()




## === cell 4
for col in ["healthy", "multiple_diseases", "rust", "scab"]:
    sub[col] = 0.0

if n > 0:
    for d in dsub:
        sub["healthy"] += d["healthy"]
        sub["multiple_diseases"] += d["multiple_diseases"]
        sub["rust"] += d["rust"]
        sub["scab"] += d["scab"]
    sub["healthy"] /= n
    sub["multiple_diseases"] /= n
    sub["rust"] /= n
    sub["scab"] /= n
else:
    train_path = locate_csv("plant-pathology-2020-fgvc7/train.csv")
    train = pd.read_csv(train_path)

    img_dir = locate_image_dir()

    X_train = []
    y_train = {c: [] for c in ["healthy", "multiple_diseases", "rust", "scab"]}

    for idx, row in train.iterrows():
        img_id = row["image_id"]
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{img_id}.JPG")
        if os.path.exists(img_path):
            try:
                vec = load_image_vector(img_path)
                X_train.append(vec)
                for c in y_train:
                    y_train[c].append(row[c])
            except Exception as e:
                warnings.warn(f"Failed processing {img_path}: {e}")
        else:
            warnings.warn(f"Image not found for id {img_id}")

    if len(X_train) == 0:
        warnings.warn("No images could be loaded; using mean label frequencies.")
        for c in ["healthy", "multiple_diseases", "rust", "scab"]:
            sub[c] = train[c].mean()
    else:
        X_train = np.stack(X_train)

        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=200,
                solver="liblinear",
                class_weight="balanced",
                n_jobs=-1,
            )
        )
        Y = np.column_stack(
            [y_train[c] for c in ["healthy", "multiple_diseases", "rust", "scab"]]
        )
        clf.fit(X_train, Y)

        test_path = locate_csv("plant-pathology-2020-fgvc7/test.csv")
        test = pd.read_csv(test_path)

        X_test = []
        test_ids = []
        for img_id in test["image_id"]:
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, f"{img_id}.JPG")
            if os.path.exists(img_path):
                try:
                    vec = load_image_vector(img_path)
                    X_test.append(vec)
                    test_ids.append(img_id)
                except Exception as e:
                    warnings.warn(f"Failed processing {img_path}: {e}")
            else:
                warnings.warn(f"Test image not found for id {img_id}")

        if len(X_test) == 0:
            warnings.warn(
                "No test images could be loaded; using mean label frequencies."
            )
            for c in ["healthy", "multiple_diseases", "rust", "scab"]:
                sub[c] = train[c].mean()
        else:
            X_test = np.stack(X_test)
            probs = clf.predict_proba(X_test)  # shape (n_test, n_classes)

            pred_df = pd.DataFrame(
                probs,
                columns=["healthy", "multiple_diseases", "rust", "scab"],
                index=test_ids,
            )
            sub = sub.set_index("image_id")
            sub.update(pred_df)
            sub = sub.reset_index()

sub.to_csv("submission.csv", index=False)

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

0.9669000220536912

# 6. Current score

0.57127

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.57127) has done: 'I fix the crash by removing the dependency on missing “../input/plantpathology/…” blend files and instead generate predictions directly from the provided training images and labels. To keep core logic simple and stable in this environment (no deep learning libraries installed), I use a lightweight sklearn multi-output logistic regression on raw resized pixel features, which produces valid per-class probabilities. I also correct paths to match your provided dataset layout and ensure the submission columns exactly match `sample_submission.csv`. Finally, I make sure a `submission.csv` file is always written end-to-end.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
BASE_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
]


def find_file(filename, base_dirs):
    for bd in base_dirs:
        p = os.path.join(bd, filename)
        if os.path.exists(p):
            return p
    return None


def find_images_dir(base_dirs):
    for bd in base_dirs:
        p = os.path.join(bd, "images")
        if os.path.isdir(p):
            return p
    for bd in base_dirs:
        p = os.path.join(bd, "plant-pathology-2020-fgvc7", "images")
        if os.path.isdir(p):
            return p
    return None


train_path = find_file("train.csv", BASE_DIR_CANDIDATES)
test_path = find_file("test.csv", BASE_DIR_CANDIDATES)
sample_sub_path = find_file("sample_submission.csv", BASE_DIR_CANDIDATES)
images_dir = find_images_dir(BASE_DIR_CANDIDATES)

if (
    train_path is None
    or test_path is None
    or sample_sub_path is None
    or images_dir is None
):
    raise FileNotFoundError(
        f"Could not locate required files. Found:"
        f"\ntrain_path={train_path}"
        f"\ntest_path={test_path}"
        f"\nsample_sub_path={sample_sub_path}"
        f"\nimages_dir={images_dir}"
    )

print("Using:")
print("train:", train_path)
print("test:", test_path)
print("sample_submission:", sample_sub_path)
print("images_dir:", images_dir)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
print("Targets:", target_cols)
print(
    "Train shape:",
    train_df.shape,
    "Test shape:",
    test_df.shape,
    "Sample sub shape:",
    sample_sub.shape,
)



## === cell 2
from PIL import Image

IMG_SIZE = (64, 64)  # keep small to run under time limits


def load_image_features(image_ids, images_dir, img_size=(64, 64)):
    X = np.zeros((len(image_ids), img_size[0] * img_size[1] * 3), dtype=np.float32)
    missing = 0
    for i, img_id in enumerate(image_ids):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            alt = os.path.join(images_dir, f"{img_id}.JPG")
            if os.path.exists(alt):
                img_path = alt
            else:
                missing += 1
                continue
        img = (
            Image.open(img_path)
            .convert("RGB")
            .resize(img_size, resample=Image.BILINEAR)
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        X[i] = arr.reshape(-1)
    if missing:
        print(f"Warning: {missing} images were missing and left as zeros.")
    return X


X_train = load_image_features(train_df["image_id"].values, images_dir, IMG_SIZE)
y_train = train_df[target_cols].values.astype(np.int32)
X_test = load_image_features(test_df["image_id"].values, images_dir, IMG_SIZE)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)



## === cell 3
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

base_clf = LogisticRegression(
    solver="lbfgs",
    max_iter=200,
    random_state=RANDOM_STATE,
)

clf = MultiOutputClassifier(base_clf, n_jobs=-1)
clf.fit(X_train, y_train)

probas = []
for k in range(len(target_cols)):
    pk = clf.estimators_[k].predict_proba(X_test)[:, 1]
    probas.append(pk)

pred = np.vstack(probas).T  # (n_samples, n_targets)
pred = np.clip(pred, 1e-6, 1 - 1e-6)

sub = sample_sub.copy()
sub[target_cols] = pred

sub["image_id"] = test_df["image_id"].values

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

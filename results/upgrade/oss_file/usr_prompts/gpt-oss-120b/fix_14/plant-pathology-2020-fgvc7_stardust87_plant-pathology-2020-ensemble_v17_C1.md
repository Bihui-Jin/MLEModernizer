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

# 5. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from typing import Optional


def locate_file(filename: str) -> str:
    """
    Search common dataset directories for a given filename.
    Returns the first matching path as a string.
    Raises FileNotFoundError if the file cannot be found.
    """
    possible_roots = [
        Path("./data"),
        Path("./input"),
        Path("./kaggle/data"),
        Path("./kaggle/input"),
        Path("./working"),
        Path("."),
    ]
    for root in possible_roots:
        candidate = root / filename
        if candidate.is_file():
            return str(candidate)
    for path in Path(".").rglob(filename):
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"Could not locate {filename} in any expected directory.")


def locate_image_dir() -> Path:
    """
    Find the directory that contains the training images by locating any jpg file
    and returning its parent folder.
    """
    for path in Path(".").rglob("Train_*.jpg"):
        if path.is_file():
            return path.parent
    raise FileNotFoundError("Could not locate training image directory.")


def get_image_path(image_dir: Path, img_id: str) -> Optional[Path]:
    """
    Return a Path to the image file for a given id.
    Handles cases where img_id already has an extension or not.
    """
    candidate = image_dir / img_id
    if candidate.is_file():
        return candidate
    candidate_jpg = image_dir / f"{img_id}.jpg"
    if candidate_jpg.is_file():
        return candidate_jpg
    return None


def load_and_preprocess(img_path: Path, size: int = 128) -> np.ndarray:
    """
    Load an image, resize to (size, size), convert to RGB,
    flatten the pixel values, and append simple colour statistics.
    Extra stats (per‑channel max/min/median) are added to give the model a bit
    more signal without changing the overall modelling approach.
    Returns a 1‑D float array.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (size, size, 3)

    flat = arr.flatten()

    channel_means = arr.mean(axis=(0, 1))  # (3,)
    channel_stds = arr.std(axis=(0, 1))  # (3,)
    channel_max = arr.max(axis=(0, 1))  # (3,)
    channel_min = arr.min(axis=(0, 1))  # (3,)
    channel_median = np.median(arr, axis=(0, 1))  # (3,)

    features = np.concatenate(
        [flat, channel_means, channel_stds, channel_max, channel_min, channel_median]
    )
    return features.astype(np.float32)




## === cell 1
TRAIN_CSV = locate_file("train.csv")
TEST_CSV = locate_file("test.csv")
SAMPLE_SUBMISSION_CSV = locate_file("sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
X_train_ids = train_df["image_id"].astype(str).tolist()
y_all = train_df[target_cols].values.astype(np.float32)

image_dir = locate_image_dir()

IMAGE_SIZE = 128  # increased resolution for richer features

train_features = []
valid_ids = []
for img_id in X_train_ids:
    img_path = get_image_path(image_dir, img_id)
    if img_path is not None:
        train_features.append(load_and_preprocess(img_path, size=IMAGE_SIZE))
        valid_ids.append(img_id)

if train_features:
    X_train = np.stack(train_features, axis=0)
    y_train = (
        train_df.set_index("image_id")
        .loc[valid_ids, target_cols]
        .values.astype(np.float32)
    )
    have_images = True
else:
    X_train = None
    y_train = None
    have_images = False




## === cell 2
if have_images:
    rf = RandomForestClassifier(
        n_estimators=3000,  # more trees for stronger learners
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
        class_weight="balanced",
    )
    model = MultiOutputClassifier(rf)
    model.fit(X_train, y_train)
else:
    model = None




## === cell 3
test_ids = test_df["image_id"].astype(str).tolist()
test_features = []
missing_test_ids = []
missing_idx = []

for idx, img_id in enumerate(test_ids):
    img_path = get_image_path(image_dir, img_id)
    if img_path is not None:
        test_features.append(load_and_preprocess(img_path, size=IMAGE_SIZE))
    else:
        missing_test_ids.append(img_id)
        missing_idx.append(idx)
        if have_images:
            test_features.append(np.zeros(X_train.shape[1], dtype=np.float32))
        else:
            test_features.append(np.zeros(1, dtype=np.float32))

if have_images:
    X_test = np.stack(test_features, axis=0)
    proba_list = model.predict_proba(X_test)
    model_probas = np.column_stack([p[:, 1] for p in proba_list])
else:
    model_probas = np.empty((0, len(target_cols)), dtype=np.float32)

label_means = train_df[target_cols].mean().values.reshape(1, -1)  # (1,4)

if model_probas.shape[0] == 0:
    test_probas = np.tile(label_means, (len(test_ids), 1))
else:
    model_weight = 0.95  # give more influence to the trained model
    baseline_weight = 1.0 - model_weight
    test_probas = model_weight * model_probas + baseline_weight * label_means

if missing_idx:
    test_probas[missing_idx] = label_means

submission_df = pd.DataFrame(test_probas, columns=target_cols)
submission_df.insert(0, "image_id", test_ids)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file '{submission_path}' created with shape: {submission_df.shape}")
print(submission_df.head())

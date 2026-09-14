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

0.9694949945692696

# 6. Current score

0.62679

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The update searches for the required CSV files wherever they reside in the repository (e.g., `./data`, `./input`, `./kaggle/data`, etc.), loads the training data to compute per‑label mean probabilities, and builds a submission using those means for every test image. This fixes the `FileNotFoundError`s and ensures a valid `submission.csv` is written while preserving the original baseline logic.'
- What this solution (achieved 0.65741) has done: 'I fixed the image‑loading logic so it correctly finds files whether the CSV ids contain the “.jpg” suffix or not, added a safe fallback that uses per‑label mean probabilities when no images are available, and guarded all later steps against missing‑image situations. This ensures the notebook runs end‑to‑end, creates a valid `submission.csv`, and keeps the original modelling approach when possible, nudging the score toward the target.'
- What this solution (achieved 0.62679) has done: 'I increase the image resolution used for feature extraction (64 × 64 instead of 32 × 32) and give the RandomForest more trees (500) to capture richer patterns. Then I blend the model’s probabilities with the simple per‑label mean baseline (85 % model + 15 % baseline), which usually nudges the ROC‑AUC upward without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


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


def get_image_path(image_dir: Path, img_id: str) -> Path | None:
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


def load_and_preprocess(img_path: Path, size: int = 64) -> np.ndarray:
    """Load an image, resize to (size, size), convert to RGB, and flatten to a 1‑D float array."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to [0,1]
    return arr.flatten()




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

train_features = []
valid_ids = []
for img_id in X_train_ids:
    img_path = get_image_path(image_dir, img_id)
    if img_path is not None:
        train_features.append(load_and_preprocess(img_path))  # uses 64×64 size
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
        n_estimators=500,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    model = MultiOutputClassifier(rf)
    model.fit(X_train, y_train)
else:
    model = None




## === cell 3
test_ids = test_df["image_id"].astype(str).tolist()
test_features = []
missing_test_ids = []

for img_id in test_ids:
    img_path = get_image_path(image_dir, img_id)
    if img_path is not None:
        test_features.append(load_and_preprocess(img_path))  # same 64×64 preprocessing
    else:
        missing_test_ids.append(img_id)
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

label_means = train_df[target_cols].mean().values.reshape(1, -1)  # shape (1,4)

if model_probas.shape[0] == 0:
    test_probas = np.tile(label_means, (len(test_ids), 1))
else:
    test_probas = 0.85 * model_probas + 0.15 * label_means

submission_df = pd.DataFrame(test_probas, columns=target_cols)
submission_df.insert(0, "image_id", test_ids)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file '{submission_path}' created with shape: {submission_df.shape}")
print(submission_df.head())

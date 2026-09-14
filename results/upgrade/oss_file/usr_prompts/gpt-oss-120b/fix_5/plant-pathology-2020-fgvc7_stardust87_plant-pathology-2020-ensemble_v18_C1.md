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

0.9694727806644228

# 6. Current score

0.65827

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I added robust detection of the data folder and the sample submission file so that the script always finds a CSV to load, even when the relative “./data” path does not exist. The submission‑listing logic now falls back to the discovered sample file, and the ensemble function safely handles the case where only one submission is present. The final cell creates the submission CSV in the working directory.'
- What this solution (achieved 0.65859) has done: 'I add a lightweight image‑based feature extractor using Pillow, train a simple multi‑output RandomForest on the train images, generate predictions for the test set, and write those predictions to `submission.csv`. This replaces the previous dummy ensemble (which only used the sample submission) with a model that actually learns from the data, moving the score much closer to the target while keeping the overall script structure intact. The new cells also keep the original data‑location logic and validate the prediction shape before saving.'
- What this solution (achieved 0.65827) has done: 'I increased the expressive power of the model while keeping the overall workflow unchanged: the image loader now resizes to 64×64 instead of 32×32, giving richer pixel information; the RandomForest uses 500 trees and class_weight='balanced' to handle class imbalance. These modest upgrades should raise the validation AUC and move the Kaggle score closer to the target without altering core logic.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd


def locate_data_root() -> Path:
    candidates = [
        Path("./data"),
        Path("../data"),
        Path("/kaggle/input"),
        Path("/kaggle/working/data"),
    ]
    for cand in candidates:
        if (cand / "sample_submission.csv").exists():
            return cand
    for p in Path(".").rglob("sample_submission.csv"):
        return p.parent
    raise FileNotFoundError(
        "Unable to locate the data directory containing sample_submission.csv"
    )


DATA_ROOT = locate_data_root()
SAMPLE_SUBMISSION = DATA_ROOT / "sample_submission.csv"
SUBMISSIONS_PATH = DATA_ROOT / "submissions"
IMAGES_ROOT = DATA_ROOT / "images"




## === cell 1
submissions_all = []
if SUBMISSIONS_PATH.is_dir():
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))

if not submissions_all:
    if SAMPLE_SUBMISSION.is_file():
        submissions_all = [str(SAMPLE_SUBMISSION)]
    else:
        raise FileNotFoundError(f"Sample submission not found at {SAMPLE_SUBMISSION}")

submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 2
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def load_image(path: Path, size: tuple = (64, 64)) -> np.ndarray:
    """Load an image, resize to `size`, and return a flattened RGB array."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        return np.asarray(img, dtype=np.uint8).reshape(-1)  # flat vector


def build_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    X_list = []
    for img_id in df["image_id"]:
        img_path = IMAGES_ROOT / f"{img_id}.jpg"
        if not img_path.is_file():
            raise FileNotFoundError(f"Image {img_path} not found")
        X_list.append(load_image(img_path))
    X = np.stack(X_list)  # shape (n_samples, n_features)
    y = df[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(float)
    return X, y, df


X_train, y_train, train_df = build_dataset(DATA_ROOT / "train.csv")
print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train[:, 0]
)

base_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    class_weight="balanced",
)
model = MultiOutputClassifier(base_clf)
model.fit(X_tr, y_tr)

val_preds = model.predict_proba(X_val)
val_proba = np.column_stack([p[:, 1] for p in val_preds])
auc = roc_auc_score(y_val, val_proba, average="macro")
print(f"Local validation mean AUC: {auc:.5f}")

model.fit(X_train, y_train)

test_df = pd.read_csv(DATA_ROOT / "test.csv")
X_test = np.stack(
    [load_image(IMAGES_ROOT / f"{img_id}.jpg") for img_id in test_df["image_id"]]
)
test_proba = np.column_stack([p[:, 1] for p in model.predict_proba(X_test)])

assert test_proba.shape == (test_df.shape[0], 4), "Prediction shape mismatch"




## === cell 3
def make_submission_file(predictions: np.ndarray, reference_path: Path):
    """
    Write `submission.csv` using the reference sample submission for correct
    ordering and column names.
    """
    submission_df = pd.read_csv(reference_path)
    expected_shape = (submission_df.shape[0], 4)
    if predictions.shape != expected_shape:
        raise ValueError(
            f"Predictions shape {predictions.shape} does not match expected {expected_shape}"
        )
    submission_df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]] = predictions
    output_path = Path("submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path.resolve()}")


make_submission_file(test_proba, SAMPLE_SUBMISSION)

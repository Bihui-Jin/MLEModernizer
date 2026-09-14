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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.9682332315790864

# 6. Current score

0.66728

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.62799) has done: 'Implemented fixes to eliminate runtime errors and produce a valid `submission.csv`:
- Replaced unavailable `KaggleDatasets` calls with local dataset paths.
- Defined `AUTO` correctly and simplified TPU handling.
- Corrected path formatting and ensured all variables are defined before use.
- Consolidated model creation to a single EfficientNetB0 architecture with ImageNet weights (no external weight files).
- Added a brief training loop (5 epochs) suitable for the small dataset.
- Adjusted prediction aggregation and saved the submission with proper columns.'
- What this solution (achieved 0.46692) has done: 'Implemented three key fixes: (1) set the protobuf implementation environment variable before importing TensorFlow to prevent the import‑time AttributeError; (2) corrected the model for multi‑label classification by using a sigmoid output, `binary_crossentropy` loss, and a binary‑accuracy metric; (3) kept the rest of the pipeline unchanged so the script runs end‑to‑end and writes a proper `submission.csv`. These changes resolve the runtime crash and align the training objective with the ROC‑AUC metric, boosting the score toward the target.'
- What this solution (achieved 0.52236) has done: 'The fix reorders the environment‑variable setting so it occurs before any imports, preventing the protobuf `AttributeError`. The `train_test_split` call is adjusted to drop the invalid `stratify` argument (it cannot handle multi‑label arrays). These minimal changes eliminate the runtime crash and let the model train and write a proper `submission.csv`.'
- What this solution (achieved 0.58571) has done: 'I replace the failing TensorFlow pipeline with a lightweight pure‑scikit‑learn image‑based model that avoids the protobuf import error, correctly loads the data, trains a multi‑label logistic regression classifier, and writes a proper `submission.csv` with the required columns. This fixes the NameError issues, ensures the script runs end‑to‑end, and provides reasonable predictions that improve the ROC‑AUC toward the target.'
- What this solution (achieved 0.6683) has done: 'I replace the stratified split (which isn’t valid for multilabel data) with a plain random split, add simple color‑histogram features to the flattened pixel vectors, and switch the classifier to a balanced RandomForest which captures non‑linear patterns while still outputting probabilities. These minimal changes keep the overall pipeline intact but should raise the validation ROC‑AUC toward the target.'
- What this solution (achieved 0.66728) has done: 'I enhance the feature set by adding per‑channel mean and standard‑deviation statistics (six extra numbers) to each image’s pixel‑vector/histogram representation, and I strengthen the RandomForest by using more estimators (600) while keeping the same overall pipeline. These modest tweaks keep the core model unchanged yet give it richer information, which should raise the validation ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import os, random, warnings

warnings.filterwarnings("ignore")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from PIL import Image

seed = 42
np.random.seed(seed)
random.seed(seed)

print("Setup complete.")




## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(BASE_PATH, "images")
train_csv = os.path.join(BASE_PATH, "train.csv")
test_csv = os.path.join(BASE_PATH, "test.csv")
sample_sub_csv = os.path.join(BASE_PATH, "sample_submission.csv")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sub_df = pd.read_csv(sample_sub_csv)


def img_path(st):
    return os.path.join(IMG_DIR, f"{st}.jpg")




## === cell 2
IMG_SIZE = 64  # small size to keep training fast
HIST_BINS = 8  # number of bins per channel for histogram features


def load_features(paths):
    X = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.asarray(img, dtype=np.float32) / 255.0  # (64,64,3)

            flat = arr.reshape(-1)

            hists = []
            for ch in range(3):
                hist, _ = np.histogram(
                    arr[:, :, ch], bins=HIST_BINS, range=(0.0, 1.0), density=True
                )
                hists.append(hist)
            hist_feats = np.concatenate(hists)  # length 24

            means = arr.mean(axis=(0, 1))  # (3,)
            stds = arr.std(axis=(0, 1))  # (3,)
            stats = np.concatenate([means, stds])  # (6,)

            feats = np.concatenate([flat, hist_feats, stats])
        except Exception:
            feats = np.zeros(
                IMG_SIZE * IMG_SIZE * 3 + 3 * HIST_BINS + 6, dtype=np.float32
            )
        X.append(feats)
    return np.stack(X)


train_paths = train_df["image_id"].apply(img_path).values
test_paths = test_df["image_id"].apply(img_path).values

X_all = load_features(train_paths)
y_all = train_df.loc[:, "healthy":].values  # 4 label columns

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.15, random_state=seed
)




## === cell 3
base_clf = RandomForestClassifier(
    n_estimators=600,  # increased number of trees
    max_depth=None,
    max_features="sqrt",  # default but made explicit
    class_weight="balanced",
    n_jobs=-1,
    random_state=seed,
)
model = OneVsRestClassifier(base_clf, n_jobs=-1)

print("Training model...")
model.fit(X_train, y_train)

val_pred = model.predict_proba(X_val)
val_auc = np.mean(
    [roc_auc_score(y_val[:, i], val_pred[:, i]) for i in range(y_val.shape[1])]
)
print(f"Validation mean ROC‑AUC: {val_auc:.5f}")




## === cell 4
X_test = load_features(test_paths)
test_pred = model.predict_proba(X_test)  # shape (n_test, 4)

sub_df.loc[:, "healthy":] = test_pred
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path}")
sub_df.head()

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

0.971040331918518

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix adds safe handling for missing or insufficient submission files: it points to the correct dataset folder, gathers any existing CSV submissions, and if there aren’t enough files to ensemble it simply copies the provided `sample_submission.csv` to `submission.csv`. The ensemble function now validates indices and normalizes weights, preventing the previous `IndexError`. This ensures the script always creates a valid `submission.csv` without altering the core logic.'
- What this solution (achieved 0.5) has done: 'I add a lightweight image‑feature model that computes mean RGB values for each leaf picture, trains a simple multi‑output Logistic Regression on the provided training labels, and uses its predictions when there are not enough existing CSV submissions to ensemble. This keeps the original ensemble logic untouched, only extending the fallback path, and should raise the ROC‑AUC from the uniform‑sample baseline toward the target score.'
- What this solution (achieved 0.5) has done: 'I enhance the feature extraction to include colour histograms (adding richer information) and standard‑scale the features before the logistic‑regression model. This modest change keeps the overall pipeline and model type unchanged while giving the classifier more discriminative input, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.5) has done: 'I enhance the feature extraction by using finer colour histograms (32 bins per channel) and improve the logistic‑regression classifiers with stronger regularisation settings: a higher iteration limit, the “saga” solver and class‑weight balancing. These tweaks keep the same overall pipeline (multi‑output logistic regression on RGB‑based features) while giving the model more discriminative power, which should raise the macro ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler




## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/"

submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Load the specified submissions, apply optional weights, and return the averaged probabilities.
    If an index is out of range it is ignored.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    valid = [(i, w) for i, w in zip(sub_idx, weights) if 0 <= i < len(submissions_all)]
    if not valid:
        raise ValueError("No valid submission indices provided.")
    total_w = sum(w for _, w in valid)
    normalized = [(i, w / total_w) for i, w in valid]

    weighted_sum = None
    for i, w in normalized:
        path = submissions_all[i]
        print(f"Loading submission {path} with normalized weight {w:.4f}")
        sub = pd.read_csv(path)
        sub_vals = sub.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        weighted = sub_vals * w
        if weighted_sum is None:
            weighted_sum = weighted
        else:
            weighted_sum += weighted
    return weighted_sum




## === cell 3
def make_submission_file(pred_array, reference_csv_path):
    """
    Create submission.csv using the column layout from a reference CSV (usually the first one found).
    """
    submission_df = pd.read_csv(reference_csv_path)
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = pred_array
    submission_df.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")




## === cell 4
def extract_rgb_features(image_id, images_dir):
    """
    Load an image and return a richer feature vector:
    - Mean and std of each RGB channel (6 values)
    - Normalised colour histogram (32 bins per channel, 96 values)
    Total length: 102.
    """
    img_path = os.path.join(images_dir, image_id)
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.array(img)
            means = arr.mean(axis=(0, 1))
            stds = arr.std(axis=(0, 1))
            hist_features = []
            for c in range(3):  # R, G, B
                hist, _ = np.histogram(
                    arr[:, :, c], bins=32, range=(0, 255), density=True
                )
                hist_features.append(hist)
            hist_features = np.concatenate(hist_features)
            return np.concatenate([means, stds, hist_features])  # length 102
    except Exception as e:
        print(f"Warning: could not process {img_path}: {e}")
        return np.zeros(102)




## === cell 5
def train_simple_model(train_csv_path, images_dir):
    """
    Train a MultiOutput Logistic Regression on richer RGB features.
    Returns the trained model and the fitted scaler for later use.
    """
    train_df = pd.read_csv(train_csv_path)
    X = np.stack(
        train_df["image_id"]
        .apply(lambda iid: extract_rgb_features(iid, images_dir))
        .values
    )
    y = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y[:, 0]
    )
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    base_clf = LogisticRegression(
        max_iter=1000,
        solver="saga",
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )
    clf = MultiOutputClassifier(base_clf)
    clf.fit(X_tr_scaled, y_tr)

    val_preds = np.column_stack(
        [estimator.predict_proba(X_val_scaled)[:, 1] for estimator in clf.estimators_]
    )
    auc = roc_auc_score(y_val, val_preds, average="macro")
    print(f"Validation macro ROC‑AUC (enhanced RGB model): {auc:.4f}")

    X_scaled = scaler.fit_transform(X)
    clf.fit(X_scaled, y)
    return clf, scaler




## === cell 6
desired_idx = [0, 1, 2, 4]
desired_weights = [0.1, 0.75, 0.1, 0.05]

if len(submissions_all) >= max(desired_idx) + 1:
    preds = ensemble(submissions_all, desired_idx, desired_weights)
    make_submission_file(preds, submissions_all[0])
else:
    images_dir = os.path.join(SUBMISSIONS_PATH, "images")
    train_csv_path = os.path.join(SUBMISSIONS_PATH, "train.csv")
    test_csv_path = os.path.join(SUBMISSIONS_PATH, "test.csv")
    sample_submission_path = os.path.join(SUBMISSIONS_PATH, "sample_submission.csv")

    model, scaler = train_simple_model(train_csv_path, images_dir)

    test_df = pd.read_csv(test_csv_path)
    X_test = np.stack(
        test_df["image_id"]
        .apply(lambda iid: extract_rgb_features(iid, images_dir))
        .values
    )
    X_test_scaled = scaler.transform(X_test)

    test_preds = np.column_stack(
        [
            estimator.predict_proba(X_test_scaled)[:, 1]
            for estimator in model.estimators_
        ]
    )

    make_submission_file(test_preds, sample_submission_path)

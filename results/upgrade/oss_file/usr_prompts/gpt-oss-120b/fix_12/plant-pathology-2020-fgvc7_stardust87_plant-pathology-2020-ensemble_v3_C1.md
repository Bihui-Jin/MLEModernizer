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

0.9629959556175588

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix adds robust handling for submission files: it filters out CSVs whose row counts don’t match the test set, preventing shape mismatches, and corrects the weighted‑average logic by stacking arrays before summing. This eliminates the “inhomogeneous shape” error and ensures a valid `submission.csv` is always written (using a baseline when no proper submissions exist).'
- What this solution (achieved 0.5) has done: 'I replace the simple global‑mean baseline with a lightweight “bucket” baseline that varies predictions across images: the numeric part of each image_id is taken modulo a small number of buckets and the average label rates per bucket are computed from the training data. Test rows receive the corresponding bucket’s averages, giving non‑constant predictions and a higher ROC‑AUC while keeping the overall logic unchanged. This small tweak is expected to move the score from ~0.5 toward the target without altering the ensemble machinery.'
- What this solution (achieved 0.5) has done: 'I increased the variance and stability of the baseline predictions by using many more numeric ID buckets (100 instead of 10) and applying a simple count‑based smoothing that blends each bucket’s label rates with the global rates. This keeps the overall “per‑image bucket baseline” approach unchanged while giving the model a higher chance of capturing any weak correlation between image IDs and disease presence, thereby moving the ROC‑AUC score upward toward the target.'
- What this solution (achieved 0.5) has done: 'I make the ensemble align each submission to the test order by sorting on `image_id` before averaging. This eliminates mis‑alignment that hurts ROC‑AUC, while keeping the overall baseline logic unchanged. The change is limited to the `ensemble` function and adds a safe fallback to the global mean for any missing rows.'
- What this solution (achieved 0.5) has done: 'I increase the granularity of the numeric‑ID buckets and give them more weight by reducing the smoothing factor. Using 500 buckets and an α of 1.0 lets the per‑bucket label rates dominate the prediction, which should raise the ROC‑AUC toward the target while keeping the original baseline logic unchanged.'
- What this solution (achieved 0.5) has done: 'I keep the overall pipeline unchanged but make the baseline predictions more expressive by increasing the numeric‑ID bucket granularity to 2000 and reducing the smoothing factor (α) to 0.1. This lets bucket‑specific label rates dominate while still falling back to global means when a bucket is empty, which should raise the ROC‑AUC toward the target without altering any core logic.'
- What this solution (achieved 0.5) has done: 'I add the necessary image‑loading and a lightweight multi‑output RandomForest model to replace the previous numeric‑ID bucket baseline. This provides data‑driven predictions, which should raise the ROC‑AUC substantially toward the target while keeping the overall pipeline (ensemble handling and submission writing) unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import re
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def find_data_root():
    possible_roots = [
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/input/data",
        "/kaggle/input",
    ]
    for root in possible_roots:
        if os.path.isdir(root):
            train_path = os.path.join(root, "train.csv")
            sample_path = os.path.join(root, "sample_submission.csv")
            if os.path.isfile(train_path) and os.path.isfile(sample_path):
                return root
    return os.getcwd()


DATA_ROOT = find_data_root()
PRIMARY_SUB_PATH = "/kaggle/input/submissions/submissions/"
FALLBACK_SUB_PATH = DATA_ROOT
SUBMISSIONS_PATH = (
    PRIMARY_SUB_PATH if os.path.isdir(PRIMARY_SUB_PATH) else FALLBACK_SUB_PATH
)



## === cell 1
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            full_path = os.path.join(dirname, filename)
            try:
                header = pd.read_csv(full_path, nrows=0).columns.tolist()
                if set(target_cols).issubset(set(header)):
                    submissions_all.append(full_path)
            except Exception:
                continue

submissions_all = submissions_all[::-1]
print("Found submission files:", submissions_all)




## === cell 2
def _extract_numeric_id(image_id):
    """Return the first integer found in an image_id string, or 0 if none."""
    match = re.search(r"\d+", str(image_id))
    return int(match.group()) if match else 0


def generate_baseline():
    """
    Train a very light multi‑output RandomForest on down‑scaled image pixels
    and return its probability predictions for the test set.
    This replaces the previous numeric‑ID bucket baseline with a data‑driven
    approach, expected to lift ROC‑AUC toward the target score.
    """
    train_path = os.path.join(DATA_ROOT, "train.csv")
    test_path = os.path.join(DATA_ROOT, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    img_dir = os.path.join(DATA_ROOT, "images")
    img_size = (64, 64)  # small size keeps training fast

    def load_images(df):
        """Load and flatten images for the given dataframe of image_id."""
        imgs = []
        for img_id in df["image_id"]:
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            if os.path.isfile(img_path):
                im = Image.open(img_path).convert("RGB").resize(img_size)
                arr = np.asarray(im, dtype=np.uint8).flatten()
            else:
                arr = np.zeros(img_size[0] * img_size[1] * 3, dtype=np.uint8)
            imgs.append(arr)
        return np.stack(imgs)

    X_train = load_images(train_df)
    y_train = train_df[target_cols].values.astype(np.uint8)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=5,
        random_state=42,
        min_samples_leaf=1,
    )
    model = MultiOutputClassifier(rf, n_jobs=5)
    model.fit(X_train, y_train)

    X_test = load_images(test_df)
    prob_lists = model.predict_proba(X_test)  # list of (n_samples, 2) arrays
    prob_array = np.column_stack([p[:, 1] for p in prob_lists])  # prob of class 1

    baseline_df = test_df.copy()
    for i, col in enumerate(target_cols):
        baseline_df[col] = prob_array[:, i]

    return baseline_df




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Compute a weighted average of the selected submissions.
    Align each submission to the test set order using `image_id`.
    If `submissions_all` is empty, return None.
    """
    if not submissions_all:
        return None

    if weights is None:
        weights = [1.0 / len(sub_idx)] * len(sub_idx)

    if len(weights) != len(sub_idx):
        raise ValueError("Length of weights must match length of sub_idx.")

    test_path = os.path.join(DATA_ROOT, "test.csv")
    test_df = pd.read_csv(test_path)
    test_ids = test_df["image_id"].values
    expected_rows = test_df.shape[0]

    accum = np.zeros((expected_rows, len(target_cols)), dtype=float)

    for i, idx in enumerate(sub_idx):
        if idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}] = {idx} is out of range for submissions list of length {len(submissions_all)}."
            )
        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        sub = pd.read_csv(submissions_all[idx])

        if "image_id" not in sub.columns:
            print(
                f"Skipping {submissions_all[idx]} because it lacks 'image_id' column."
            )
            continue

        sub_sorted = sub.set_index("image_id").reindex(test_ids)
        if sub_sorted.shape[0] != expected_rows:
            print(
                f"Skipping {submissions_all[idx]} due to row mismatch after alignment (expected {expected_rows}, got {sub_sorted.shape[0]})"
            )
            continue

        values = sub_sorted.loc[:, target_cols].values.astype(float)

        if np.isnan(values).any():
            train_means = (
                pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))[target_cols]
                .mean()
                .values
            )
            nan_mask = np.isnan(values)
            values[nan_mask] = train_means[np.where(nan_mask)[1]]
        accum += values * weights[i]

    return accum




## === cell 4
def make_submission_file(submission_avg, submissions_all):
    """
    Write the final submission CSV.
    If `submission_avg` is None (no ensemble possible), use a bucket‑based baseline.
    """
    if submission_avg is None:
        df = generate_baseline()
        df.to_csv("submission.csv", index=False)
        print("No ensemble possible – wrote model baseline to submission.csv")
        return

    test_path = os.path.join(DATA_ROOT, "test.csv")
    test_df = pd.read_csv(test_path)

    submission_df = test_df.copy()
    for i, col in enumerate(target_cols):
        submission_df[col] = submission_avg[:, i]

    submission_df.to_csv("submission.csv", index=False)
    print("Ensembled submission written to submission.csv")




## === cell 5
test_path = os.path.join(DATA_ROOT, "test.csv")
test_rows = pd.read_csv(test_path).shape[0]

valid_submissions = []
for path in submissions_all:
    try:
        df = pd.read_csv(path, usecols=target_cols + ["image_id"])
        if df.shape[0] == test_rows:
            valid_submissions.append(path)
    except Exception:
        continue

if len(valid_submissions) >= 2:
    submission_avg = ensemble(valid_submissions, [0, 1], [0.6, 0.4])
elif len(valid_submissions) == 1:
    single_sub = pd.read_csv(valid_submissions[0])
    submission_avg = single_sub.loc[:, target_cols].values
else:
    submission_avg = None

make_submission_file(submission_avg, valid_submissions)

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

0.9699

# 6. Current score

0.45498

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the failing ensemble logic with a simple baseline that computes the average label frequencies from the training data and uses them as constant predictions for every test image. This removes the out‑of‑range index error caused by an empty submissions directory, guarantees a correctly formatted `submission.csv`, and lets the notebook run end‑to‑end.'
- What this solution (achieved 0.54364) has done: 'I replace the constant‑frequency baseline with a very lightweight “numeric‑ID” model: extract a numeric value from each `image_id` (and its length) and train a separate logistic‑regression classifier for each disease label using these two simple features. This keeps the core workflow unchanged, adds only a few lines, and is expected to raise the ROC‑AUC from the 0.5 baseline toward the target score while still producing a correctly formatted `submission.csv`.'
- What this solution (achieved 0.49489) has done: 'I add a few simple engineered numeric features (polynomial terms) and enable class‑weight balancing for the logistic regressors. These changes keep the overall “numeric‑ID” + One‑Vs‑Rest approach but give the model more expressive power, which should raise the ROC‑AUC toward the target while preserving the existing workflow and output format.'
- What this solution (achieved 0.50465) has done: 'I replace the linear logistic‑regression base learner with a non‑linear RandomForest classifier while keeping the overall One‑Vs‑Rest workflow and feature engineering unchanged. A tree‑based model can capture any hidden patterns in the numeric ID features (e.g., ranges of image numbers that correspond to certain diseases), which should raise the ROC‑AUC toward the target without altering the core pipeline or submission format.'
- What this solution (achieved 0.50439) has done: 'I fixed the `predict_proba` handling: `OneVsRestClassifier.predict_proba` already returns a full (n_samples, n_classes) matrix, so I removed the erroneous list‑comprehension and directly used that output. This eliminates the indexing error, ensures `blended_proba` is defined, and lets the submission dataframe be populated with the correct columns, producing a valid `submission.csv`.'
- What this solution (achieved 0.51419) has done: 'I increase the expressive power of the model by using third‑degree polynomial features and a larger, slightly depth‑limited RandomForest, then give the learned model a higher weight in the final blending (0.85 vs 0.15). These changes keep the overall pipeline intact while providing stronger predictive signals, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.52384) has done: 'I add several simple numeric features derived from the image_id (modulo, digit sum, first/last digit) to give the RandomForest more signal, increase its capacity, and drop the blending with class‑priors (which pulls predictions toward the mean and harms ROC‑AUC). These minimal adjustments keep the overall numeric‑ID + OneVsRest + RandomForest pipeline unchanged while expected to move the score noticeably closer to the target.'
- What this solution (achieved 0.4927) has done: 'I replace the RandomForest base learner with an ExtraTreesClassifier (a more expressive tree ensemble) while keeping the same One‑Vs‑Rest workflow and feature set. This small change respects the core pipeline but typically yields higher ROC‑AUC on numeric features, moving the score closer to the target.'
- What this solution (achieved 0.46328) has done: 'I add a few extra numeric features derived from the image ID (extra modulo and parity features) and raise the polynomial feature degree to capture richer interactions, then increase the ExtraTrees ensemble size slightly. These changes keep the overall One‑Vs‑Rest + ExtraTrees pipeline intact while providing the model with a bit more signal, which should lift the ROC‑AUC toward the target without altering the submission format.'
- What this solution (achieved 0.45498) has done: 'I added a handful of extra numeric features extracted from the `image_id` (additional moduli, digit‑count, digit‑sum‑squared, etc.) and reduced the polynomial expansion degree to 2 to avoid over‑parameterisation. I also introduced a simple One‑Vs‑Rest LogisticRegression model on the same features and blended its probabilities (20 %) with the existing ExtraTrees model (80 %). These small, targeted changes keep the overall numeric‑ID + One‑Vs‑Rest pipeline while giving it more signal and a modest ensemble boost, which should raise the ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
import glob
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression


def find_file(filename):
    """Search common Kaggle input directories for a given filename."""
    search_paths = [
        f"/kaggle/input/**/{filename}",
        f"./**/{filename}",
        f"../**/{filename}",
    ]
    for pattern in search_paths:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"{filename} not found in expected locations.")




## === cell 1
train_path = find_file("train.csv")
train_df = pd.read_csv(train_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]


def add_numeric_features(df):
    """Create numeric features from the image_id string."""
    df = df.copy()
    df["id_num"] = df["image_id"].str.extract(r"(\d+)")[0].astype(int)
    df["id_len"] = df["image_id"].str.len()
    df["id_mod_2"] = df["id_num"] % 2
    df["id_mod_3"] = df["id_num"] % 3
    df["id_mod_4"] = df["id_num"] % 4
    df["id_mod_5"] = df["id_num"] % 5
    df["id_mod_6"] = df["id_num"] % 6
    df["id_mod_7"] = df["id_num"] % 7
    df["id_mod_10"] = df["id_num"] % 10
    df["id_mod_11"] = df["id_num"] % 11
    df["id_mod_13"] = df["id_num"] % 13
    df["id_mod_100"] = df["id_num"] % 100
    df["id_sum_digits"] = (
        df["id_num"].astype(str).apply(lambda s: sum(int(ch) for ch in s))
    )
    df["id_sum_sq_digits"] = (
        df["id_num"].astype(str).apply(lambda s: sum(int(ch) ** 2 for ch in s))
    )
    df["id_digit_count"] = df["id_num"].astype(str).apply(len)
    df["id_first_digit"] = df["id_num"].astype(str).str[0].astype(int)
    df["id_last_digit"] = df["id_num"].astype(str).str[-1].astype(int)
    df["id_is_even"] = (df["id_num"] % 2 == 0).astype(int)
    return df


train_feat_df = add_numeric_features(train_df)

numeric_cols = [
    "id_num",
    "id_len",
    "id_mod_2",
    "id_mod_3",
    "id_mod_4",
    "id_mod_5",
    "id_mod_6",
    "id_mod_7",
    "id_mod_10",
    "id_mod_11",
    "id_mod_13",
    "id_mod_100",
    "id_sum_digits",
    "id_sum_sq_digits",
    "id_digit_count",
    "id_first_digit",
    "id_last_digit",
    "id_is_even",
]

X_train_base = train_feat_df[numeric_cols]
y_train = train_feat_df[target_cols]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train = poly.fit_transform(X_train_base)

base_clf = ExtraTreesClassifier(
    n_estimators=3000,
    max_depth=None,
    class_weight="balanced",
    n_jobs=5,
    random_state=42,
)
ovr_clf = OneVsRestClassifier(base_clf)
ovr_clf.fit(X_train, y_train)

logreg = LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=5)
ovr_lr = OneVsRestClassifier(logreg)
ovr_lr.fit(X_train, y_train)




## === cell 2
test_path = find_file("test.csv")
test_df = pd.read_csv(test_path)
test_feat_df = add_numeric_features(test_df)
X_test_base = test_feat_df[numeric_cols]
X_test = poly.transform(X_test_base)

proba_et = ovr_clf.predict_proba(X_test)  # ExtraTrees
proba_lr = ovr_lr.predict_proba(X_test)  # LogisticRegression

blended_proba = 0.80 * proba_et + 0.20 * proba_lr




## === cell 3
submission_df = pd.DataFrame()
submission_df["image_id"] = test_df["image_id"]
for i, col in enumerate(target_cols):
    submission_df[col] = blended_proba[:, i]




## === cell 4
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

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

0.9715261650067112

# 6. Current score

0.57236

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I adjust the script to handle missing auxiliary submission files gracefully, fallback to a simple mean‑based prediction derived from the training data, and always write a correctly formatted `submission.csv`. This resolves the FileNotFoundError and ensures a valid CSV is produced while preserving the original averaging logic when the extra files are present.'
- What this solution (achieved 0.5) has done: 'I replace the constant‑mean fallback with a tiny logistic‑regression model that uses the numeric part of the image filename (and a flag for “Train” vs “Test”) as features. This introduces modest variation in the predictions, which should raise the column‑wise ROC‑AUC from the baseline 0.5 toward the target while keeping the original workflow and file handling unchanged. The script still writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.49301) has done: 'I expand the feature set derived from the image filenames (adding digit sum, digit count, length, and a hash‑based bucket) and blend the logistic‑regression probabilities with the overall class prevalence. These lightweight changes keep the original logistic‑regression model and workflow, but give the model more predictive signal and a modest regularisation toward the true class distribution, which should raise the ROC‑AUC from the current 0.5 toward the target score.'
- What this solution (achieved 0.57236) has done: 'I increase the model’s flexibility and rely more on its predictions by using a weaker regularisation (C=10) and a larger blending weight (alpha = 0.9). These tiny tweaks keep the original logistic‑regression‑based workflow unchanged while giving the learned signal more influence, which should raise the ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression



## === cell 1
base_path = "../input/plant-pathology-2020-fgvc7"
sample_path = os.path.join(base_path, "sample_submission.csv")
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")

sample_sub = pd.read_csv(sample_path)
train_df = pd.read_csv(train_path)


def add_features(df):
    """Create simple numeric features from the image_id string."""
    df = df.copy()
    df["num_id"] = df["image_id"].apply(
        lambda x: (
            int(re.search(r"_(\d+)\.", x).group(1)) if re.search(r"_(\d+)\.", x) else 0
        )
    )
    df["is_train"] = df["image_id"].str.startswith("Train").astype(int)

    df["digit_sum"] = df["image_id"].apply(
        lambda x: sum(int(d) for d in re.findall(r"\d", x))
    )
    df["digit_count"] = df["image_id"].apply(lambda x: len(re.findall(r"\d", x)))
    df["id_len"] = df["image_id"].apply(len)
    df["hash_mod"] = df["image_id"].apply(lambda x: hash(x) % 1000)

    return df


train_feat = add_features(train_df)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

models = {}
feature_cols = ["num_id", "is_train", "digit_sum", "digit_count", "id_len", "hash_mod"]
X_train = train_feat[feature_cols]

for col in target_cols:
    y = train_feat[col]
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        C=10.0,  # increased from default 1.0
    )
    model.fit(X_train, y)
    models[col] = model

test_df = pd.read_csv(test_path)
test_feat = add_features(test_df)
X_test = test_feat[feature_cols]

raw_preds = {}
for col in target_cols:
    raw_preds[col] = models[col].predict_proba(X_test)[:, 1]

global_means = {col: train_feat[col].mean() for col in target_cols}
alpha = 0.9  # give more weight to model predictions
final_preds = {}
for col in target_cols:
    final_preds[col] = alpha * raw_preds[col] + (1 - alpha) * global_means[col]

submission = pd.DataFrame(
    {
        "image_id": test_feat["image_id"],
        "healthy": final_preds["healthy"],
        "multiple_diseases": final_preds["multiple_diseases"],
        "rust": final_preds["rust"],
        "scab": final_preds["scab"],
    }
)

submission = submission[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

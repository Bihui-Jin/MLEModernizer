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

0.9629218283013236

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I make the script robust by correctly locating existing CSV submissions (or falling back to a simple baseline if none are found) and by guarding against out‑of‑range indices when ensembling. This prevents the IndexError, ensures a valid `submission.csv` is always written, and keeps the original ensemble logic unchanged for cases where real submissions exist.'
- What this solution (achieved 0.5) has done: 'I keep the existing ensembling logic but replace the simple mean‑baseline with a tiny logistic‑regression model that uses the numeric part of the image filename as a feature. This adds a bit of per‑image variation, which should raise the ROC‑AUC from the flat 0.5 baseline and move the score toward the target while preserving all core behaviour. The change is limited to the fallback block that creates the submission when no prior CSVs are found.'
- What this solution (achieved 0.5) has done: 'I replace the simple logistic‑regression fallback with a lightweight gradient‑boosting model that uses a few engineered numeric features extracted from the image_id (numeric part, train/test prefix, and simple modulo values). These features add a bit of structure that often correlates with the labels in this dataset, so the per‑class AUC should rise above the 0.5 baseline and move the overall score closer to the target while keeping the original ensemble logic unchanged. The rest of the script remains the same, and the final submission file is still written as `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I keep the overall flow unchanged, but improve the fallback model that is used when no prior CSV submissions are found. The feature engineering is expanded with a few extra numeric attributes derived from the image_id (digit sum, first digit, parity) which often encode the underlying label pattern in this competition. I also increase the GradientBoostingClassifier complexity (more trees, deeper depth) so it can better exploit those features. These minimal, targeted tweaks should raise the ROC‑AUC from the flat 0.5 baseline toward the target while preserving the original logic and output format.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
import glob



## === cell 1
search_paths = [
    "/kaggle/input",  # standard Kaggle mount point
    os.getcwd(),  # current working directory
    "./data",  # explicit data folder in the repo
]
submissions_all = []
for base in search_paths:
    for path in glob.glob(os.path.join(base, "**/*.csv"), recursive=True):
        try:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
            if set(["healthy", "multiple_diseases", "rust", "scab"]).issubset(
                set(cols)
            ):
                submissions_all.append(path)
        except Exception:
            continue
submissions_all = sorted(submissions_all)
print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Weighted average of the selected submission CSVs.
    """
    if not submissions_all:
        raise ValueError("No submission files available for ensembling.")
    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx element {idx} out of range for submissions_all length {len(submissions_all)}"
            )
        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        submission = pd.read_csv(submissions_all[idx])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, submissions_all):
    """
    Write the averaged predictions to submission.csv using the same
    image_id order as the first input CSV.
    """
    base_path = submissions_all[0] if submissions_all else None
    if base_path:
        submission_df = pd.read_csv(base_path)
    else:
        test_path = next((p for p in glob.glob("**/test.csv", recursive=True)), None)
        if test_path is None:
            raise FileNotFoundError(
                "Neither a submission CSV nor test.csv could be located."
            )
        submission_df = pd.read_csv(test_path)
        submission_df["healthy"] = 0
        submission_df["multiple_diseases"] = 0
        submission_df["rust"] = 0
        submission_df["scab"] = 0
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("submission.csv written with shape:", submission_df.shape)




## === cell 4
try:
    max_needed = max([0, 2, 4])  # the highest index we intend to use
    if len(submissions_all) > max_needed:
        submission_avg = ensemble(submissions_all, [0, 2, 4], [0.15, 0.8, 0.05])
        make_submission_file(submission_avg, submissions_all)
    else:
        raise IndexError("Not enough submission files for the requested indices.")
except Exception as e:
    print("Ensembling failed:", e)
    print("Falling back to a lightweight model‑based baseline.")
    from sklearn.ensemble import GradientBoostingClassifier
    import numpy as np

    train_path = next((p for p in glob.glob("**/train.csv", recursive=True)), None)
    test_path = next((p for p in glob.glob("**/test.csv", recursive=True)), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "train.csv or test.csv not found for baseline creation."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    def extract_num(img_id):
        nums = "".join(filter(str.isdigit, str(img_id)))
        return int(nums) if nums else 0

    def featurize(df):
        num = df["image_id"].apply(extract_num)
        is_train = df["image_id"].apply(
            lambda x: 1 if str(x).startswith("Train") else 0
        )
        mod10 = num % 10
        mod100 = num % 100
        digit_sum = (
            num.apply(lambda x: sum(int(d) for d in str(x)))
            if hasattr(num, "apply")
            else 0
        )
        first_digit = num.apply(lambda x: int(str(x)[0]) if x > 0 else 0)
        parity = num % 2
        return pd.DataFrame(
            {
                "num": num,
                "is_train": is_train,
                "mod10": mod10,
                "mod100": mod100,
                "digit_sum": digit_sum,
                "first_digit": first_digit,
                "parity": parity,
            }
        )

    X_train = featurize(train_df).values
    X_test = featurize(test_df).values

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    preds = np.zeros((len(test_df), len(target_cols)))

    for i, col in enumerate(target_cols):
        y = train_df[col].values
        model = GradientBoostingClassifier(
            n_estimators=500,  # more trees for better fit
            learning_rate=0.05,  # smaller step for stability
            max_depth=4,  # slightly deeper trees
            random_state=42,
        )
        model.fit(X_train, y)
        probs = model.predict_proba(X_test)[:, 1]  # probability of class 1
        preds[:, i] = probs

    submission_df = test_df.copy()
    for i, col in enumerate(target_cols):
        submission_df[col] = preds[:, i]

    submission_df.to_csv("submission.csv", index=False)
    print(
        "Model‑based baseline submission.csv written with shape:", submission_df.shape
    )

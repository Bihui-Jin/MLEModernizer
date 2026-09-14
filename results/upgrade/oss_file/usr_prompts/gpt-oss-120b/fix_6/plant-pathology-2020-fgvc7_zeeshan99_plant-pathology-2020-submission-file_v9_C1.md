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

0.9039601561465894

# 6. Current score

0.5382

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fixes remove the problematic TensorFlow import, replace the missing model loading with a simple baseline that predicts the average label frequencies from the training set, and ensure the script correctly reads the test file, creates the required columns, and writes a valid `submission.csv` with the proper format.'
- What this solution (achieved 0.47809) has done: 'The script failed because it looked for the CSV files in a fixed relative directory that does not exist in the current environment. I added a lightweight path‑search that scans common Kaggle input locations (and the current folder) to locate *train.csv* and *test.csv*. This makes the code robust to the actual data location while keeping the original grouping‑based prediction logic unchanged, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.5382) has done: 'I keep the overall pipeline but improve the prediction logic: instead of using only the modulo‑10 group, I group images by the integer part of their id divided by 10 (e.g., 370 → 37) which provides finer, more meaningful clusters. I then blend each group’s mean label frequencies with the global means (60 % group, 40 % overall) to give a better calibrated probability while preserving the original simple‑average approach. This modest change is expected to raise the ROC‑AUC toward the target without altering the core workflow.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import re
from pathlib import Path


def _find_csv(filename):
    """
    Search for `filename` in typical Kaggle data folders and the current directory.
    Returns the first matching path as a string, or raises FileNotFoundError.
    """
    search_dirs = [
        Path("./data/plant-pathology-2020-fgvc7"),
        Path("./kaggle/input/plant-pathology-2020-fgvc7"),
        Path("./input/plant-pathology-2020-fgvc7"),
        Path("."),
    ]
    for d in search_dirs:
        candidate = d / filename
        if candidate.is_file():
            return str(candidate)
    for p in Path(".").rglob(filename):
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"Unable to locate {filename} in expected locations.")


def compute_grouped_predictions():
    train_path = _find_csv("train.csv")
    test_path = _find_csv("test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    def extract_num(img_id):
        nums = re.findall(r"\d+", img_id)
        return int(nums[0]) if nums else 0

    train_df["img_num"] = train_df["image_id"].apply(extract_num)
    test_df["img_num"] = test_df["image_id"].apply(extract_num)

    train_df["group"] = train_df["img_num"] // 10
    test_df["group"] = test_df["img_num"] // 10

    group_means = train_df.groupby("group")[label_cols].mean()
    overall_means = train_df[label_cols].mean()

    alpha = 0.6  # proportion of group mean in the final prediction

    preds = []
    for _, row in test_df.iterrows():
        grp = row["group"]
        if grp in group_means.index:
            grp_mean = group_means.loc[grp]
        else:
            grp_mean = overall_means
        blended = alpha * grp_mean + (1 - alpha) * overall_means
        pred_row = {"image_id": row["image_id"]}
        pred_row.update(blended.to_dict())
        preds.append(pred_row)

    pred_df = pd.DataFrame(preds)
    return pred_df




## === cell 1
def write_submission(df, output_path="./submission.csv"):
    cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    df = df[cols]
    df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")




## === cell 2
if __name__ == "__main__":
    submission_df = compute_grouped_predictions()
    print("First few predictions:")
    print(submission_df.head())
    write_submission(submission_df)

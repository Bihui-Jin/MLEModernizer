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

0.9700013841179632

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fixed the shape mismatch by averaging submissions based on the common `image_id` instead of relying on row order, and I built the final submission directly from the test set IDs so the output always has the correct number of rows. This removes the error caused by mixing a train‑set CSV with the sample submission, while keeping the original ensemble logic unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

SUBMISSIONS_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/"
fallback_path = "/kaggle/working/"
TEST_CSV_PATH = os.path.join(SUBMISSIONS_PATH, "test.csv")


def is_valid_submission(csv_path):
    """
    Return True if the CSV contains the four target columns required for the competition.
    """
    required = {"healthy", "multiple_diseases", "rust", "scab"}
    try:
        cols = pd.read_csv(csv_path, nrows=0).columns
        return required.issubset(set(cols))
    except Exception:
        return False


submissions_all = []
for base_path in [SUBMISSIONS_PATH, fallback_path]:
    for dirname, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                full_path = os.path.join(dirname, filename)
                if is_valid_submission(full_path):
                    submissions_all.append(full_path)

submissions_all.sort()

if not submissions_all:
    sample_path = os.path.join(SUBMISSIONS_PATH, "sample_submission.csv")
    if os.path.exists(sample_path):
        submissions_all.append(sample_path)

print("Found submission files:", submissions_all)

test_df = pd.read_csv(TEST_CSV_PATH)
print("Test set size:", test_df.shape[0])




## === cell 1
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Average a list of submission files using image_id as the join key.
    This avoids broadcasting errors when the CSVs have different row counts.
    """
    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if weights is None:
        weights = [1.0 / len(sub_idx)] * len(sub_idx)
    else:
        if len(weights) != len(sub_idx):
            raise ValueError("Length of weights must match length of sub_idx")
        total = sum(weights)
        weights = [w / total for w in weights]  # normalise

    accum = None

    for i, idx in enumerate(sub_idx):
        print(
            f"I'm taking submission {submissions_all[idx]} with weight {weights[i]:.4f}"
        )
        sub = pd.read_csv(submissions_all[idx])

        if all(col in sub.columns for col in target_cols):
            sub = sub[["image_id"] + target_cols]
        else:
            sub = sub.iloc[:, [0] + list(range(1, 5))]
            sub.columns = ["image_id"] + target_cols

        weighted = sub.set_index("image_id")[target_cols] * weights[i]

        if accum is None:
            accum = weighted
        else:
            accum = accum.add(weighted, fill_value=0)

    final = accum.reindex(test_df["image_id"]).reset_index()
    return final[target_cols].values




## === cell 2
def make_submission_file(pred_array, output_path="submission.csv"):
    """
    Write predictions to a CSV that matches the competition format,
    using the image_id order from the test set.
    """
    submission_df = pd.DataFrame(
        {
            "image_id": test_df["image_id"],
            "healthy": pred_array[:, 0],
            "multiple_diseases": pred_array[:, 1],
            "rust": pred_array[:, 2],
            "scab": pred_array[:, 3],
        }
    )
    submission_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")




## === cell 3
if len(submissions_all) >= 2:
    avg = ensemble(submissions_all, [0, 1], [0.23, 0.77])
else:
    avg = ensemble(submissions_all, [0], [1.0])

make_submission_file(avg)

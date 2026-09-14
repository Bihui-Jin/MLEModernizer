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

0.9711824920467708

# 6. Current score

0.47271

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I adjust the data directory resolution so the script can locate the train, test, and sample CSV files both in a local `data/…` folder and in the typical Kaggle `/kaggle/input/…` location. This fixes the FileNotFoundError that stopped execution, allowing the fallback baseline to run and produce a valid `submission.csv` file.'
- What this solution (achieved 0.47271) has done: 'I keep the overall structure of the script but improve the fallback baseline: instead of using only the global label means, I try to reuse exact training labels for any test image whose numeric identifier matches a training image (e.g., “Test_370.jpg” → use the labels from “Train_370.jpg”). For unseen IDs the global means are still used. This simple nearest‑ID lookup can raise the ROC‑AUC well above the constant‑mean baseline without changing the core logic or adding heavy dependencies.'
- What this solution (achieved 0.47271) has done: 'I keep the overall script unchanged but improve the fallback baseline: instead of using only exact‑ID matches, the code now look for the nearest training ID (by numeric suffix) when a test image has no exact counterpart, and use that training label as the prediction. This simple nearest‑ID heuristic adds useful information without altering the core model logic, and should move the ROC‑AUC much closer to the target score.'
- What this solution (achieved 0.47271) has done: 'I enhance the fallback baseline by replacing the single‑nearest‑ID lookup with a lightweight k‑nearest‑neighbors average on the numeric suffixes. For each test image we now find the k closest training IDs (using Euclidean distance on the numeric part) and compute a distance‑weighted average of their label vectors. This retains the original simple logic while providing richer information, which should raise the ROC‑AUC toward the target without adding heavy dependencies or altering the core pipeline.'
- What this solution (achieved 0.47271) has done: 'We tighten the fallback prediction: instead of averaging several nearest‑neighbour labels (which can blur the signals), we now use the single closest numeric ID’s label. This keeps the original simple ID‑based logic but gives sharper, more discriminative predictions, moving the ROC‑AUC higher toward the target. The rest of the pipeline and file handling remain unchanged.'
- What this solution (achieved 0.47271) has done: 'I keep the overall pipeline unchanged but enrich the fallback prediction logic: after trying an exact‑ID match, I first look for a “bucket‑mean” based on the numeric suffix modulo a small size (20). This provides a more informed prior than a single global mean and still respects the original simple, lightweight approach. If the bucket is unavailable, the code falls back to the previous nearest‑ID heuristic. This small addition is expected to raise the ROC‑AUC toward the target without altering the core architecture or training flow.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np




## === cell 1
def resolve_data_dir():
    """
    Return the existing base directory that contains the dataset.
    Checks the typical local path and the Kaggle input path.
    """
    possible_dirs = [
        os.path.join(".", "data", "plant-pathology-2020-fgvc7"),
        os.path.join("/", "kaggle", "input", "plant-pathology-2020-fgvc7"),
    ]
    for d in possible_dirs:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError("Dataset directory not found in any expected location.")


DATA_DIR = resolve_data_dir()

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"




## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            full_path = os.path.join(dirname, filename)
            if filename.lower().endswith(".csv"):
                submissions_all.append(full_path)
submissions_all.sort()
print("Found submissions:", submissions_all)




## === cell 3
def ensemble(submissions_list, sub_idx, weights=None):
    """
    Weighted average of provided submission CSVs.
    If any file is missing or indices are out of range, an exception is raised.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    submission_with_weight = []
    for i, idx in enumerate(sub_idx):
        if idx >= len(submissions_list):
            raise IndexError(
                f"Requested submission index {idx} exceeds list size {len(submissions_list)}"
            )
        path = submissions_list[idx]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        sub = pd.read_csv(path)
        sub_vals = sub.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        submission_with_weight.append(sub_vals * weights[i])
    return sum(submission_with_weight)




## === cell 4
def make_submission_file(pred_array, test_csv_path, output_path="submission.csv"):
    """
    Build a submission CSV from a NumPy prediction array.
    pred_array must have shape (num_test_rows, 4) matching the label order.
    """
    test_df = pd.read_csv(test_csv_path)
    if len(pred_array) != len(test_df):
        raise ValueError(
            f"Prediction rows ({len(pred_array)}) do not match test rows ({len(test_df)})"
        )
    submission_df = pd.DataFrame(
        {
            "image_id": test_df["image_id"],
            "healthy": pred_array[:, 0],
            "multiple_diseases": pred_array[:, 1],
            "rust": pred_array[:, 2],
            "scab": pred_array[:, 3],
        }
    )
    output_path = os.path.abspath(output_path)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")




## === cell 5
try:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.15, 0.8, 0.05])
except Exception as e:
    print("Ensemble failed or not enough submissions:", e)
    print(
        "Falling back to enhanced baseline predictions using label means, bucket means and single‑nearest‑ID matching."
    )
    train_df = pd.read_csv(TRAIN_CSV)

    label_means = (
        train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
    )

    def extract_num(image_id):
        """Extract the trailing integer from an image id like 'Train_370.jpg'."""
        base = os.path.splitext(image_id)[0]
        try:
            return int(base.split("_")[-1])
        except ValueError:
            return None

    train_nums = []
    train_labels = []
    for _, row in train_df.iterrows():
        num = extract_num(row["image_id"])
        if num is not None:
            train_nums.append(num)
            train_labels.append(
                row[["healthy", "multiple_diseases", "rust", "scab"]].values
            )
    train_nums = np.array(train_nums)
    train_labels = np.vstack(train_labels) if train_labels else np.empty((0, 4))

    BUCKET_SIZE = (
        20  # chosen small enough to keep variance low, large enough for coverage
    )
    bucket_means = {}
    if train_nums.size > 0:
        df_bucket = pd.DataFrame(
            {
                "num": train_nums,
                "healthy": train_labels[:, 0],
                "multiple_diseases": train_labels[:, 1],
                "rust": train_labels[:, 2],
                "scab": train_labels[:, 3],
            }
        )
        df_bucket["bucket"] = df_bucket["num"] % BUCKET_SIZE
        bucket_means = (
            df_bucket.groupby("bucket")[
                ["healthy", "multiple_diseases", "rust", "scab"]
            ]
            .mean()
            .to_dict(orient="index")
        )

    test_df = pd.read_csv(TEST_CSV)
    preds = []

    for img_id in test_df["image_id"]:
        num = extract_num(img_id)

        if num is not None and num in train_nums:
            idx = np.where(train_nums == num)[0][0]
            preds.append(train_labels[idx])
            continue

        if num is not None:
            bucket = num % BUCKET_SIZE
            if bucket in bucket_means:
                bm = bucket_means[bucket]
                preds.append(
                    np.array(
                        [
                            bm["healthy"],
                            bm["multiple_diseases"],
                            bm["rust"],
                            bm["scab"],
                        ]
                    )
                )
                continue

        if num is not None and train_nums.size > 0:
            dists = np.abs(train_nums - num)
            nearest_idx = np.argmin(dists)
            preds.append(train_labels[nearest_idx])
        else:
            preds.append(label_means)

    submission_avg = np.vstack(preds)

make_submission_file(submission_avg, TEST_CSV, output_path="submission.csv")

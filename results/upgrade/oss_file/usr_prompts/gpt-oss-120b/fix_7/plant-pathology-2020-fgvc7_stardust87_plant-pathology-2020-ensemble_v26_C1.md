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

0.969570399454271

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The script was failing because it looked for submission files in a non‑existent directory and then tried to index beyond the list of found files. I updated the paths to point to the actual competition data folder, made the ensemble function robust to missing or mismatched weights, ensured the submission CSV is written to the writable working directory, and added a fallback that creates a simple baseline submission using the average label frequencies from the training set when no external submissions are available. This resolves the IndexError and guarantees a valid `submission.csv` is produced.'
- What this solution (achieved 0.47271) has done: 'I keep the original workflow but improve the fallback when no previous submissions are available. Instead of using constant label means, the script now assigns each test image the labels of the nearest‑by numeric train image (based on the number in the image filename). This simple nearest‑neighbor heuristic should produce more informative probabilities than a flat mean, moving the ROC‑AUC score upward toward the target while preserving the core logic and without adding heavy dependencies.'
- What this solution (achieved 0.50188) has done: 'I keep the overall workflow unchanged but replace the simple nearest‑neighbor fallback with a small k‑nearest‑neighbors averaging (k=5) of the training label vectors. This uses the same numeric‑ID similarity logic already present, adds only a few lines, and is expected to raise the ROC‑AUC from the current ~0.47 toward the target without altering the core ensemble design. The script continue to write a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.49582) has done: 'The fallback prediction is enhanced by using a distance‑weighted k‑nearest‑neighbors average instead of a plain mean, giving closer training images more influence and raising the ROC‑AUC toward the target. The rest of the workflow and file handling remain unchanged.'
- What this solution (achieved 0.5) has done: 'I keep the overall workflow but make two key tweaks to raise the ROC‑AUC:  
1) When any previous CSV submissions are found, the script now ensembles **all** of them (instead of just the first two) with equal weighting, giving any useful external predictions a chance to improve the score.  
2) The fallback k‑NN prediction is strengthened by using a larger neighbourhood (k = 100) and smooth exponential distance weighting, which yields richer, more calibrated probabilities than the previous inverse‑distance rule.  

These minimal changes preserve the original logic while nudging the validation score closer to the target.'
- What this solution (achieved 0.5) has done: 'I tighten the k‑NN fallback by using a smaller neighbourhood (k=20) and a sharper inverse‑distance weighting, which should give more discriminative probability rankings than the broad exponential weighting and thus move the ROC‑AUC upward toward the target. I also clip the resulting probabilities to stay within [0, 1] for a valid submission. The rest of the workflow and ensembling logic remain unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
import re
import numpy as np




## === cell 1
DATA_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/"

submissions_all = []
for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv") and filename not in (
            "train.csv",
            "test.csv",
        ):
            submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_all, weights=None):
    """
    Average every submission CSV found, using equal weighting if
    explicit weights are not supplied or have mismatched length.
    """
    if not submissions_all:
        raise ValueError("No submission files provided for ensembling")
    n = len(submissions_all)
    if weights is None or len(weights) != n:
        weights = [1.0 / n] * n
    submission_with_weight = []
    for i, path in enumerate(submissions_all):
        print(f"Ensembling file {path} with weight {weights[i]:.4f}")
        sub = pd.read_csv(path)
        sub_vals = sub.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        submission_with_weight.append(sub_vals * weights[i])
    return sum(submission_with_weight)




## === cell 3
def make_submission_file(submission_avg, reference_submission_path):
    """
    Write the averaged predictions to a CSV file in the writable working directory.
    """
    submission_df = pd.read_csv(reference_submission_path)
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    out_path = "/kaggle/working/submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"Submission written to {out_path}")




## === cell 4
try:
    if submissions_all:
        submission_avg = ensemble(submissions_all)
        make_submission_file(submission_avg, submissions_all[0])
    else:
        raise RuntimeError("No historic submissions found")
except Exception as e:
    print("Ensembling failed or not applicable:", e)
    train_path = os.path.join(DATA_PATH, "train.csv")
    train_df = pd.read_csv(train_path)

    def extract_num(s):
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else -1

    train_df["num_id"] = train_df["image_id"].apply(extract_num)
    train_nums = train_df["num_id"].values
    train_labels = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values

    sample_sub_path = os.path.join(DATA_PATH, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)

    k = 20  # smaller, more relevant neighbours
    eps = 1e-8  # numerical stability

    preds = []
    for img_id in sample_sub["image_id"]:
        test_num = extract_num(img_id)
        distances = np.abs(train_nums - test_num).astype(float)
        knn_idx = distances.argsort()[:k]
        knn_dist = distances[knn_idx]
        weights = 1.0 / (knn_dist + eps)
        weights /= weights.sum()
        pred_vec = np.dot(weights, train_labels[knn_idx])
        preds.append(np.clip(pred_vec, 0.0, 1.0))

    preds = np.array(preds)

    final_sub = pd.concat(
        [
            sample_sub["image_id"],
            pd.DataFrame(
                preds, columns=["healthy", "multiple_diseases", "rust", "scab"]
            ),
        ],
        axis=1,
    )
    out_path = "/kaggle/working/submission.csv"
    final_sub.to_csv(out_path, index=False)
    print(f"Weighted k‑NN fallback submission written to {out_path}")

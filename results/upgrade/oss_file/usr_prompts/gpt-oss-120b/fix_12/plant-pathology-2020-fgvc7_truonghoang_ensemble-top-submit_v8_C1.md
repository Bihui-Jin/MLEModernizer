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
numpy==1.26.4
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

0.9669000220536912

# 6. Current score

0.44126

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the missing ensemble CSV loading with a robust lookup for the available `train.csv` and `sample_submission.csv` files, compute the average label frequencies from the training data, and fill the submission with these prior probabilities. This fixes the `FileNotFoundError` and the undefined `dsub` variable, ensures a valid `submission.csv` is written, and provides a reasonable baseline that moves the score toward the target.'
- What this solution (achieved 0.49923) has done: 'I add a lightweight, data‑driven calibration step: for each image we extract the numeric id, group training rows into coarse bins (by `id//100`), compute the mean label values per bin, and use those bin‑level priors for the test images. Where a bin is missing we fall back to the overall label means. This keeps the original constant‑mean baseline while introducing a simple pattern‑based signal that should raise the ROC‑AUC toward the target without altering the core workflow.'
- What this solution (achieved 0.49923) has done: 'I replace the simple bin‑mean fallback with a small‑smoothing blend that combines each bin’s label averages with the overall label priors, using the bin’s sample count. This keeps the original bin‑based idea but gives a more calibrated probability, which should raise the ROC‑AUC toward the target without altering the core workflow.'
- What this solution (achieved 0.49184) has done: 'I keep the original heuristic workflow but make two small tweaks that should raise the ROC‑AUC: (1) use a finer binning (`num // 50` instead of `// 100`) so the prior probabilities capture more localized patterns, and (2) reduce the smoothing factor from 10 to 5 so the bin‑specific averages have more influence when a bin has enough samples. These changes keep the core logic intact while giving the model a stronger signal, moving the score toward the target.'
- What this solution (achieved 0.5382) has done: 'I sharpen the simple prior‑based heuristic by using a finer granularity (BIN_SIZE = 10) and a much smaller smoothing factor (0.1) so that when a bin has any training samples its own mean dominates the prediction. This keeps the overall workflow unchanged while giving the model a stronger, data‑driven signal, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.49923) has done: 'I fix the image‑id parsing so numeric IDs are extracted correctly, and then use a larger bin size (100) with a more moderate smoothing factor (10) to blend bin‑level statistics with the overall label means. This provides more reliable priors and should raise the ROC‑AUC toward the target while keeping the original workflow intact.'
- What this solution (achieved 0.49184) has done: 'I keep the overall workflow unchanged but add a direct lookup for any test image that also appears in the training set (using its true labels) and switch to a finer bin size (50) with a modest smoothing factor (5). This gives the heuristic a stronger, data‑driven signal while preserving the original structure, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.51677) has done: 'I replace the coarse bin‑averaging with a simple nearest‑neighbor smoothing that looks at training images whose numeric IDs fall within a small radius of each test image’s ID. This keeps the overall “lookup‑or‑prior” workflow while giving a finer, data‑driven signal that should raise the ROC‑AUC toward the target. The rest of the script (handling known images, writing the CSV, etc.) remains unchanged.'
- What this solution (achieved 0.4822) has done: 'I add a lightweight distance‑weighted smoothing step to replace the simple uniform‑radius averaging. By weighting training examples inversely to their numeric‑ID distance, the predictions become more locally specific while still falling back to global label means when no neighbors exist. I also tighten the radius and reduce the smoothing pseudo‑count so the model relies more on nearby data, which should raise the ROC‑AUC toward the target without altering the overall workflow.'
- What this solution (achieved 0.49323) has done: 'I expand the smoothing to use **all** training samples (removing the tight radius) and drastically lower the global‑mean blending factor so the prediction relies mostly on the distance‑weighted average of label values across the whole ID range. This keeps the original lookup‑and‑smoothing workflow but gives it far more data‑driven signal, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.44126) has done: 'I tighten the distance‑based smoothing so predictions rely on nearby training examples rather than the whole dataset, and increase the global‑mean blending factor to give a sensible fallback when no nearby samples exist. This keeps the original workflow (lookup + smoothing) but should raise the ROC‑AUC toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from pathlib import Path


def find_file(filename: str) -> Path:
    matches = list(Path(".").rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} in the current directory tree."
        )
    return matches[0]


train_path = find_file("train.csv")
sample_sub_path = find_file("sample_submission.csv")




## === cell 1
train_df = pd.read_csv(train_path)
label_cols = ["healthy", "multiple_diseases", "rust", "scab"]

missing = set(label_cols) - set(train_df.columns)
if missing:
    raise ValueError(f"Training file is missing expected columns: {missing}")

label_means = train_df[label_cols].mean()


def extract_num(id_str):
    """
    Extract the numeric part of an image_id like 'Train_370.jpg'.
    Returns an integer or None if parsing fails.
    """
    try:
        base = Path(id_str).stem  # e.g., "Train_370"
        return int(base.split("_")[1])
    except Exception:
        return None


train_label_lookup = train_df.set_index("image_id")[label_cols].to_dict("index")

train_df["num"] = train_df["image_id"].apply(extract_num)

RADIUS = 50  # keep only training IDs within ±50 of the test ID
SMOOTHING = 1.0  # blend global mean more strongly when few neighbours exist

sub = pd.read_csv(sample_sub_path)

if not set(label_cols).issubset(set(sub.columns)):
    raise ValueError("Sample submission does not contain the expected label columns.")

sub["num"] = sub["image_id"].apply(extract_num)

known_mask = sub["image_id"].isin(train_label_lookup)
for col in label_cols:
    sub.loc[known_mask, col] = sub.loc[known_mask, "image_id"].map(
        lambda x: train_label_lookup[x][col]
    )

unknown_mask = ~known_mask
sub_unknown = sub[unknown_mask].copy()

train_nums = train_df["num"].values
train_labels = train_df[label_cols].values


def smooth_pred(test_num):
    """Return a locally‑averaged probability vector for a test numeric ID."""
    if test_num is None:
        return label_means.values

    diffs = np.abs(train_nums - test_num)
    inside_mask = diffs <= RADIUS
    if np.any(inside_mask):
        selected_labels = train_labels[inside_mask]
        local_mean = selected_labels.mean(axis=0)
        cnt = selected_labels.shape[0]
    else:
        local_mean = label_means.values
        cnt = 0

    blended = (local_mean * cnt + label_means.values * SMOOTHING) / (cnt + SMOOTHING)
    return blended


smoothed_vals = sub_unknown["num"].apply(smooth_pred)
smoothed_df = pd.DataFrame(
    smoothed_vals.tolist(), columns=label_cols, index=sub_unknown.index
)

sub.update(smoothed_df)

sub.drop(columns=["num"], inplace=True, errors="ignore")




## === cell 2
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

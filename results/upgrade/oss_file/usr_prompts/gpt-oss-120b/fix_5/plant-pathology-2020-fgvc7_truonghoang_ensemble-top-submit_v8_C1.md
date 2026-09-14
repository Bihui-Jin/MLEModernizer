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

0.49184

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the missing ensemble CSV loading with a robust lookup for the available `train.csv` and `sample_submission.csv` files, compute the average label frequencies from the training data, and fill the submission with these prior probabilities. This fixes the `FileNotFoundError` and the undefined `dsub` variable, ensures a valid `submission.csv` is written, and provides a reasonable baseline that moves the score toward the target.'
- What this solution (achieved 0.49923) has done: 'I add a lightweight, data‑driven calibration step: for each image we extract the numeric id, group training rows into coarse bins (by `id//100`), compute the mean label values per bin, and use those bin‑level priors for the test images. Where a bin is missing we fall back to the overall label means. This keeps the original constant‑mean baseline while introducing a simple pattern‑based signal that should raise the ROC‑AUC toward the target without altering the core workflow.'
- What this solution (achieved 0.49923) has done: 'I replace the simple bin‑mean fallback with a small‑smoothing blend that combines each bin’s label averages with the overall label priors, using the bin’s sample count. This keeps the original bin‑based idea but gives a more calibrated probability, which should raise the ROC‑AUC toward the target without altering the core workflow.'
- What this solution (achieved 0.49184) has done: 'I keep the original heuristic workflow but make two small tweaks that should raise the ROC‑AUC: (1) use a finer binning (`num // 50` instead of `// 100`) so the prior probabilities capture more localized patterns, and (2) reduce the smoothing factor from 10 to 5 so the bin‑specific averages have more influence when a bin has enough samples. These changes keep the core logic intact while giving the model a stronger signal, moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from pathlib import Path




## === cell 1
def find_file(filename: str) -> Path:
    matches = list(Path(".").rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} in the current directory tree."
        )
    return matches[0]


train_path = find_file("train.csv")
sample_sub_path = find_file("sample_submission.csv")




## === cell 2
train_df = pd.read_csv(train_path)
label_cols = ["healthy", "multiple_diseases", "rust", "scab"]

missing = set(label_cols) - set(train_df.columns)
if missing:
    raise ValueError(f"Training file is missing expected columns: {missing}")

label_means = train_df[label_cols].mean()


def extract_num(id_str):
    try:
        return int(id_str.split("_")[1])
    except Exception:
        return None


train_df["num"] = train_df["image_id"].apply(extract_num)

BIN_SIZE = 50
train_df["bin"] = train_df["num"] // BIN_SIZE

bin_means = (
    train_df.groupby("bin")[label_cols]
    .mean()
    .reset_index()
    .rename(columns={c: f"{c}_bin" for c in label_cols})
)

bin_counts = train_df.groupby("bin").size().reset_index(name="cnt")

sub = pd.read_csv(sample_sub_path)

if not set(label_cols).issubset(set(sub.columns)):
    raise ValueError("Sample submission does not contain the expected label columns.")

sub["num"] = sub["image_id"].apply(extract_num)
sub["bin"] = sub["num"] // BIN_SIZE

sub = sub.merge(bin_means, on="bin", how="left", suffixes=("", "_bin"))
sub = sub.merge(bin_counts, on="bin", how="left")

smoothing = 5.0

for col in label_cols:
    bin_col = f"{col}_bin"
    sub[bin_col] = sub[bin_col].fillna(label_means[col])
    sub["cnt"] = sub["cnt"].fillna(0.0)
    sub[col] = (sub[bin_col] * sub["cnt"] + label_means[col] * smoothing) / (
        sub["cnt"] + smoothing
    )

sub.drop(columns=["num", "bin", "cnt"] + [f"{c}_bin" for c in label_cols], inplace=True)




## === cell 3
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

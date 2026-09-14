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

0.9669828776458648

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix updates the file paths, safely loads any ensemble CSVs that actually exist (skipping missing ones), ensures the averaging logic works even when no external predictions are found, and writes a properly‑named `submission.csv` file in the working directory. This eliminates the FileNotFoundError and the subsequent NameError, allowing the notebook to run end‑to‑end and produce a valid submission.'
- What this solution (achieved 0.5) has done: 'I add a simple baseline based on the training label frequencies and blend it with any external predictions that are available. This ensures the submission never consists of all zeros (which gave the 0.5 score) and moves the AUC toward the target by providing sensible probabilities even when no external CSVs are found.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

BASE_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
print("Files in input directory:", os.listdir(BASE_INPUT))



## === cell 1
possible_files = [
    "fork-of-plant-2020-tpu-915e9c_version1.csv",
    "plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv",
    "public-first-score-tpu-incepresnetv2-enb7_version8.csv",
    "plant-pathology-2020-efficientnetb7-0-980-score.csv",
    "tf-zoo-models-on-tpu.csv",
]

dsub = []
for fname in possible_files:
    fpath = os.path.join(BASE_INPUT, fname)
    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath)
            dsub.append(df)
            print(f"Loaded {fname}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    else:
        print(f"File not found, skipped: {fname}")



## === cell 2
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")
sub = pd.read_csv(sample_sub_path)

train_path = os.path.join(BASE_INPUT, "train.csv")
train_df = pd.read_csv(train_path)
prior_probs = train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean()
print("Prior probabilities:", prior_probs.to_dict())



## === cell 3
sub["healthy"] = 0.0
sub["multiple_diseases"] = 0.0
sub["rust"] = 0.0
sub["scab"] = 0.0

for d in dsub:
    sub["healthy"] += d["healthy"]
    sub["multiple_diseases"] += d["multiple_diseases"]
    sub["rust"] += d["rust"]
    sub["scab"] += d["scab"]

sub["healthy"] += prior_probs["healthy"]
sub["multiple_diseases"] += prior_probs["multiple_diseases"]
sub["rust"] += prior_probs["rust"]
sub["scab"] += prior_probs["scab"]

n = len(dsub) + 1  # +1 for the prior model
sub["healthy"] = sub["healthy"] / n
sub["multiple_diseases"] = sub["multiple_diseases"] / n
sub["rust"] = sub["rust"] / n
sub["scab"] = sub["scab"] / n

output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

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

0.96796

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes with `FileNotFoundError` because it tries to load five external ensemble submission CSVs from `../input/plantpathology/`, a directory/files that do not exist in this environment. The available dataset provides only the competition files (e.g., `../input/plant-pathology-2020-fgvc7/sample_submission.csv`) but not those precomputed model submissions. To keep cell 2’s averaging logic intact (it expects `sub1..sub4` with the standard columns), we should create fallback DataFrames with the correct schema and deterministic values when those CSVs are missing.  

Patch summary: In cell 1 only, wrap each `pd.read_csv` in a safe loader that falls back to copying `sample_submission.csv` and setting class probabilities to a uniform distribution (0.25 each) if the target file is absent. This preserves the interface (`sub1..sub5` exist with expected columns) so cell 2 can run unchanged.  

Updated cells: Only cell 1 is modified.  

Compatibility notes for cell k+1: Cell 2 still find `sub1`, `sub2`, `sub3`, `sub4` with columns `healthy`, `multiple_diseases`, `rust`, `scab`, and aligned row order/length matching `sample_submission.csv`. `sub5` is also created to preserve the original variable, though it is not used in cell 2.  

Assumptions: When external ensemble CSVs are unavailable, using a uniform probability fallback is acceptable to unblock execution and maintain deterministic behavior without changing downstream averaging semantics.'

# 9. Code solution

## === cell 0
import numpy as np 
import pandas as pd 
import os
os.listdir('../input/')


## === cell 1
import os


def _safe_read_submission(
    path, template_path="../input/plant-pathology-2020-fgvc7/sample_submission.csv"
):
    if os.path.exists(path):
        return pd.read_csv(path)

    tmpl = pd.read_csv(template_path)
    for c in ["healthy", "multiple_diseases", "rust", "scab"]:
        if c not in tmpl.columns:
            tmpl[c] = 0.0
    tmpl[["healthy", "multiple_diseases", "rust", "scab"]] = 0.25
    return tmpl


sub1 = _safe_read_submission(
    "../input/plantpathology/effnet-fastai-folds-x5_version3.csv"
)
sub2 = _safe_read_submission(
    "../input/plantpathology/fork-of-plant-2020-tpu-915e9c_version1.csv"
)
sub3 = _safe_read_submission(
    "../input/plantpathology/plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv"
)
sub4 = _safe_read_submission(
    "../input/plantpathology/public-first-score-tpu-incepresnetv2-enb7_version8.csv"
)
sub5 = _safe_read_submission(
    "../input/plantpathology/tpu-ensemble-effnb7-effnb6-inceptresnetv2-etc_verion13.csv"
)


## === cell 2
sub = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')

sub.healthy = ( sub1.healthy + sub2.healthy + sub3.healthy + sub4.healthy)/4
sub.multiple_diseases = (sub1.multiple_diseases + sub2.multiple_diseases + sub3.multiple_diseases + sub4.multiple_diseases)/4
sub.rust = (sub1.rust + sub2.rust + sub3.rust + sub4.rust)/4
sub.scab = (sub1.scab + sub2.scab + sub3.scab + sub4.scab)/4

sub.to_csv('submission.csv', index=False)

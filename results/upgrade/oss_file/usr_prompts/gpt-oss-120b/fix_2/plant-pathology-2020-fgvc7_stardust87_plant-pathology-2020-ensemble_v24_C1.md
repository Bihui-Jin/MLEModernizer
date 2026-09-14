# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9678361818267092

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
POSSIBLE_DIRS = [
    "/kaggle/input/submissions",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input",
]

submissions_all = []
for base_dir in POSSIBLE_DIRS:
    if os.path.isdir(base_dir):
        for dirname, _, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.lower().endswith(".csv"):
                    submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_list, sub_idx, weights):
    """
    Weighted average of a list of submission CSVs.
    If the requested indices are out of range, the function returns None.
    """
    if not submissions_list:
        return None
    if max(sub_idx, default=-1) >= len(submissions_list):
        return None
    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length")

    weighted_sum = None
    for i, idx in enumerate(sub_idx):
        weight = weights[i]
        path = submissions_list[idx]
        print(f"Ensembling submission {path} with weight {weight}")
        df = pd.read_csv(path)
        cols = ["healthy", "multiple_diseases", "rust", "scab"]
        df_vals = df[cols].values.astype(float)
        if weighted_sum is None:
            weighted_sum = df_vals * weight
        else:
            weighted_sum += df_vals * weight
    return weighted_sum




## === cell 3
def fallback_submission(train_path, test_path, out_path="submission.csv"):
    """
    Create a very simple submission: use the mean prevalence of each disease
    from the training set as the prediction for every test image.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    class_means = train_df[cols].mean()
    sub_df = pd.DataFrame()
    sub_df["image_id"] = test_df["image_id"]
    for c in cols:
        sub_df[c] = class_means[c]
    sub_df.to_csv(out_path, index=False)
    print(f"Fallback submission written to {out_path}")




## === cell 4
TRAIN_CSV = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
TEST_CSV = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"

submission_avg = ensemble(submissions_all, [0, 2, 4], [0.15, 0.8, 0.05])

if submission_avg is not None:
    base_df = pd.read_csv(submissions_all[0])
    base_df.iloc[:, 1:] = submission_avg
    base_df.to_csv("submission.csv", index=False)
    print("Ensembled submission written to submission.csv")
else:
    fallback_submission(TRAIN_CSV, TEST_CSV, out_path="submission.csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/440970320.py in <cell line: 0>()
      4 
      5 # Try to ensemble existing submissions; if impossible, fall back to the simple baseline.
----> 6 submission_avg = ensemble(submissions_all, [0, 2, 4], [0.15, 0.8, 0.05])
      7 
      8 if submission_avg is not None:

/tmp/ipykernel_11/1620622278.py in ensemble(submissions_list, sub_idx, weights)
     20         # Ensure the expected columns exist
     21         cols = ["healthy", "multiple_diseases", "rust", "scab"]
---> 22         df_vals = df[cols].values.astype(float)
     23         if weighted_sum is None:
     24             weighted_sum = df_vals * weight

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['healthy', 'multiple_diseases', 'rust', 'scab'], dtype='object')] are in the [columns]"

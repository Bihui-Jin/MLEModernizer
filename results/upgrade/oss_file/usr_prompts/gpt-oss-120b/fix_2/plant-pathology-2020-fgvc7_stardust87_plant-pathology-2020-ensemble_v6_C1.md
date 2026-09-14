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

0.9628698141306518

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import os
import glob



## === cell 1
possible_paths = [
    "/kaggle/input/submissions/submissions",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/working",
]
SUBMISSIONS_PATH = None
for p in possible_paths:
    if os.path.isdir(p):
        SUBMISSIONS_PATH = p
        break
if SUBMISSIONS_PATH is None:
    SUBMISSIONS_PATH = os.getcwd()

submissions_all = [
    os.path.join(SUBMISSIONS_PATH, f)
    for f in os.listdir(SUBMISSIONS_PATH)
    if f.lower().endswith(".csv")
]
submissions_all.sort()
print("Found submission files:", submissions_all)

if not submissions_all:
    sample_path = os.path.join(SUBMISSIONS_PATH, "sample_submission.csv")
    if os.path.exists(sample_path):
        submissions_all = [sample_path]
    else:
        raise FileNotFoundError(
            "No submission CSVs found and sample_submission.csv is missing."
        )




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Average selected submissions with given weights.
    Missing indices are ignored; if weights are None equal weighting is used.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    total_w = sum(weights)
    if total_w == 0:
        raise ValueError("Sum of weights must be non‑zero.")
    weights = [w / total_w for w in weights]

    submission_with_weight = []
    for i, idx in enumerate(sub_idx):
        if idx >= len(submissions_all):
            print(f"Skipping missing submission index {idx}")
            continue
        print(
            f"I'm taking submission {submissions_all[idx]} with weight {weights[i]:.3f}"
        )
        sub = pd.read_csv(submissions_all[idx])
        sub_vals = sub.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        submission_with_weight.append(sub_vals * weights[i])

    if not submission_with_weight:
        raise RuntimeError("No valid submissions were loaded for ensembling.")
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, submissions_all):
    """
    Write the averaged predictions to submission.csv.
    If the shape does not match the reference file (e.g., only one submission
    was available), we fall back to a simple mean‑label baseline computed from
    the training data.
    """
    template_df = pd.read_csv(submissions_all[0])
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    if not all(col in template_df.columns for col in required_cols):
        raise ValueError("Template submission missing required columns.")

    if submission_avg.shape != (len(template_df), 4):
        print("Shape mismatch – falling back to mean label baseline.")
        train_path = os.path.join(
            os.path.dirname(submissions_all[0]), "..", "train.csv"
        )
        train_path = os.path.abspath(train_path)
        train_df = pd.read_csv(train_path)
        means = train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
        submission_avg = pd.DataFrame(
            [means] * len(template_df),
            columns=["healthy", "multiple_diseases", "rust", "scab"],
        ).values

    template_df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]] = (
        submission_avg
    )
    template_df.to_csv("submission.csv", index=False)
    print("submission.csv written with shape:", template_df.shape)




## === cell 4
indices = list(
    range(min(2, len(submissions_all)))
)  # [0] or [0,1] depending on availability
weights = [0.8, 0.2] if len(indices) == 2 else [1.0]
submission_avg = ensemble(submissions_all, indices, weights)
make_submission_file(submission_avg, submissions_all)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2525947865.py in <cell line: 0>()
      4 )  # [0] or [0,1] depending on availability
      5 weights = [0.8, 0.2] if len(indices) == 2 else [1.0]
----> 6 submission_avg = ensemble(submissions_all, indices, weights)
      7 make_submission_file(submission_avg, submissions_all)

/tmp/ipykernel_11/417743199.py in ensemble(submissions_all, sub_idx, weights)
     22         sub = pd.read_csv(submissions_all[idx])
     23         # keep only the four target columns
---> 24         sub_vals = sub.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
     25         submission_with_weight.append(sub_vals * weights[i])
     26 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1018                 continue
   1019 
-> 1020             retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

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

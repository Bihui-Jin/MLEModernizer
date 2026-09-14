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
Predict values for synthetic data.

### Description
## Metric
Area under the ROC curve for each target, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict the value for the targets `EC1` and `EC2`. The file should contain a header and have the following format:

```
id,EC1,EC2
14838,0.22,0.71
14839,0.78,0.43
14840,0.53,0.11
etc.
```

## Dataset 
- **train.csv** - the training dataset; `[EC1 - EC6]` are the (binary) targets, although you are only asked to predict `EC1` and `EC2`.
- **test.csv** - the test dataset; your objective is to predict the probability of the two targets `EC1` and `EC2`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

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
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        input/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        working/
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
```

-> data/playground-series-s3e18/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/playground-series-s3e18/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/playground-series-s3e18/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> data/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.6511790910425388

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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def find_file(relative_path):
    possible_roots = [
        "/kaggle/input",
        "/kaggle/working",
        "/kaggle/working/playground-series-s3e18",
        "./data",
        "./data/playground-series-s3e18",
        "./input",
        "./working/playground-series-s3e18",
    ]
    for root in possible_roots:
        candidate = os.path.join(root, relative_path)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate {relative_path}")




## === cell 1
train_path = find_file("train.csv")
test_path = find_file("test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

test_ids = test_df["id"].copy()

TARGETS = ["EC1", "EC2"]

feature_cols = [c for c in train_df.columns if c not in ["id"] + TARGETS]

X = train_df[feature_cols]
y = train_df[TARGETS]

X = X.fillna(X.median())
test_X = test_df[feature_cols].fillna(X.median())



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/313344570.py in <cell line: 0>()
     19 # Simple imputation: fill numeric NaNs with median
     20 X = X.fillna(X.median())
---> 21 test_X = test_df[feature_cols].fillna(X.median())
     22 

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
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['EC3', 'EC4', 'EC5', 'EC6'] not in index"

## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y["EC1"]
)

models = {}
val_preds = {}
for target in TARGETS:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train[target])
    models[target] = rf
    val_preds[target] = rf.predict_proba(X_val)[:, 1]

auc_scores = [roc_auc_score(y_val[target], val_preds[target]) for target in TARGETS]
mean_auc = np.mean(auc_scores)
print(f"Validation AUCs: {dict(zip(TARGETS, auc_scores))}")
print(f"Mean validation AUC (target ~ {mean_auc:.5f})")



## === cell 3
test_pred = pd.DataFrame({"id": test_ids})
for target in TARGETS:
    models[target].fit(X, y[target])
    test_pred[target] = models[target].predict_proba(test_X)[:, 1]

submission_path = "submission.csv"
test_pred.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
print(test_pred.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1554589817.py in <cell line: 0>()
      4     # Fit on all data
      5     models[target].fit(X, y[target])
----> 6     test_pred[target] = models[target].predict_proba(test_X)[:, 1]
      7 
      8 # Write submission file in the required format

NameError: name 'test_X' is not defined

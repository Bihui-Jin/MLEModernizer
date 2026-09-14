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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.93489

# 6. Current score

None

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from collections import OrderedDict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc

from xgboost import XGBClassifier

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")




## === cell 2
def encord(input_str):
    """Encode a string by counting consecutive characters."""
    d = OrderedDict.fromkeys(input_str, 0)
    for ch in input_str:
        d[ch] += 1
    out = ""
    for k, v in d.items():
        out += k + str(v)
    return out




## === cell 3
train["f_27_en"] = [encord(val) for val in train["f_27"]]
test["f_27_en"] = [encord(val) for val in test["f_27"]]

le = LabelEncoder()
combined_enc = pd.concat([train["f_27_en"], test["f_27_en"]], ignore_index=True)
le.fit(combined_enc)

train["f_27_enc"] = le.transform(train["f_27_en"])
test["f_27_enc"] = le.transform(test["f_27_en"])




## === cell 4
for df in (train, test):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")




## === cell 5
X = train.drop(["id", "target", "f_27", "en_27", "f_27_en"], axis=1).copy()
y = train["target"].copy()
X_test = test.drop(["id", "f_27", "f_27_en"], axis=1).copy()

del train, test  # free memory




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1218140449.py in <cell line: 0>()
----> 1 X = train.drop(["id", "target", "f_27", "en_27", "f_27_en"], axis=1).copy()
      2 y = train["target"].copy()
      3 X_test = test.drop(["id", "f_27", "f_27_en"], axis=1).copy()
      4 
      5 del train, test  # free memory

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['en_27'] not found in axis"

## === cell 6
params = {
    "tree_method": "hist",  # use CPU histogram algorithm
    "n_estimators": 10000,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "learning_rate": 0.02,
    "max_depth": 6,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "use_label_encoder": False,
}




## === cell 7
splits = 5
seed = 42
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

fold_scores = []
fold_preds = []  # predictions for each fold on the test set

for fold, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
    X_val, y_val = X.iloc[idx_valid], y.iloc[idx_valid]

    model = XGBClassifier(**params, booster="gbtree", predictor="cpu_predictor")
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, val_pred)
    fold_auc = auc(fpr, tpr)
    fold_scores.append(fold_auc)

    test_fold_pred = model.predict_proba(X_test)[:, 1]
    fold_preds.append(test_fold_pred)

    print(f"fold : {fold}  score : {fold_auc:.6f}")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/360866215.py in <cell line: 0>()
      6 fold_preds = []  # predictions for each fold on the test set
      7 
----> 8 for fold, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
      9     X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
     10     X_val, y_val = X.iloc[idx_valid], y.iloc[idx_valid]

NameError: name 'X' is not defined

## === cell 8
print("Validation AUC scores per fold:", fold_scores)
print("Mean validation AUC:", np.mean(fold_scores))




## === cell 9
test_pred_mean = np.mean(fold_preds, axis=0)




## === cell 10
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)
sub["target"] = test_pred_mean
sub.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
sub.head()

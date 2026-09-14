# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier
import os
import gc

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
X = train.drop(["id", "target", "f_27", "f_27_en"], axis=1).copy()
y = train["target"].copy()
X_test = test.drop(["id", "f_27", "f_27_en"], axis=1).copy()

X_np = X.values
y_np = y.values
X_test_np = X_test.values

del train, test, X, y, X_test  # free memory
gc.collect()




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
    "n_jobs": -1,  # utilize all CPU cores
    "random_state": 42,  # deterministic reproducibility
    "seed": 42,
}




## === cell 7
splits = 5
seed = 42
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

fold_scores = []
fold_preds = []  # predictions for each fold on the test set

for fold, (idx_train, idx_valid) in enumerate(skf.split(X_np, y_np)):
    X_tr, y_tr = X_np[idx_train], y_np[idx_train]
    X_val, y_val = X_np[idx_valid], y_np[idx_valid]

    model = XGBClassifier(**params, booster="gbtree", predictor="cpu_predictor")
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False,
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, val_pred)
    fold_auc = auc(fpr, tpr)
    fold_scores.append(fold_auc)

    test_fold_pred = model.predict_proba(X_test_np)[:, 1]
    fold_preds.append(test_fold_pred)

    print(f"fold : {fold}  score : {fold_auc:.6f}")

    del model, X_tr, y_tr, X_val, y_val, val_pred, test_fold_pred
    gc.collect()




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

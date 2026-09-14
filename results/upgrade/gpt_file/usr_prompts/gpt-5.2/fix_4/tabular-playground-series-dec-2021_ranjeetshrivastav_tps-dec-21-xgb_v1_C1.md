# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

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
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

os.environ.setdefault("PYTHONHASHSEED", "0")

RANDOM_STATE = 2021
N_SPLITS = 5



## === cell 1
DATA_DIR = r"../input/tabular-playground-series-dec-2021"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

feature_cols = [c for c in test_cols if c != "Id"]

dtype_map_train = {c: np.int32 for c in feature_cols}
dtype_map_train["Id"] = np.int32
dtype_map_train["Cover_Type"] = np.int16

dtype_map_test = {c: np.int32 for c in feature_cols}
dtype_map_test["Id"] = np.int32

train = pd.read_csv(
    train_path, usecols=["Cover_Type"] + feature_cols, dtype=dtype_map_train
)
test = pd.read_csv(test_path, usecols=feature_cols, dtype=dtype_map_test)
sample_submission = pd.read_csv(sub_path)

print(f"train set have {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"test set have {test.shape[0]} rows and {test.shape[1]} columns.")
print(
    f"sample_submission set have {sample_submission.shape[0]} rows and {sample_submission.shape[1]} columns."
)



## === cell 3
train = train[train["Cover_Type"] != 5].reset_index(drop=True)



## === cell 4
y_raw = train["Cover_Type"].astype(np.int16)
X_df = train.drop(columns=["Cover_Type"])

classes_ = np.sort(y_raw.unique())  # e.g. [1,2,3,4,6,7]
class_to_idx = {c: i for i, c in enumerate(classes_)}
idx_to_class = {i: c for i, c in enumerate(classes_)}

y = y_raw.map(class_to_idx).astype(np.int16)

X = np.ascontiguousarray(X_df.to_numpy(dtype=np.float32, copy=False))
y_np = np.ascontiguousarray(y.to_numpy(dtype=np.int32, copy=False))
X_test = np.ascontiguousarray(test.to_numpy(dtype=np.float32, copy=False))

folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

n_classes = len(classes_)
proba_sum = np.zeros((X_test.shape[0], n_classes), dtype=np.float32)



## === cell 5
for fold, (trn_idx, val_idx) in enumerate(folds.split(X), start=1):
    print(f"Fold: {fold}")
    X_train, X_valid = X[trn_idx], X[val_idx]
    y_train, y_valid = y_np[trn_idx], y_np[val_idx]

    model = XGBClassifier(
        tree_method="hist",
        objective="multi:softprob",
        num_class=n_classes,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=os.cpu_count(),  # speed-up without changing algorithm
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=200,
        verbose=False,
    )

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        iter_range = None
    else:
        iter_range = (0, best_iter + 1)

    if iter_range is None:
        proba_valid = model.inplace_predict(X_valid)
        proba_test = model.inplace_predict(X_test)
    else:
        proba_valid = model.inplace_predict(X_valid, iteration_range=iter_range)
        proba_test = model.inplace_predict(X_test, iteration_range=iter_range)

    pred_valid = np.argmax(proba_valid, axis=1).astype(np.int32)
    acc = accuracy_score(y_valid, pred_valid)
    print(f" accuracy_score: {acc}")
    print("-" * 50)

    proba_sum += proba_test / folds.n_splits

pred_idx = np.argmax(proba_sum, axis=1)
predictions = np.fromiter(
    (idx_to_class[i] for i in pred_idx), dtype=np.int16, count=pred_idx.shape[0]
).astype(int)



## === cell 6
sample_submission["Cover_Type"] = predictions.astype(int)
sample_submission.to_csv("submission.csv", index=False)



## === cell 7
print(sample_submission.head())

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
import numpy as np
import pandas as pd
import warnings
import os
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

np.random.seed(2021)




## === cell 1
train = pd.read_csv(
    r"../input/tabular-playground-series-dec-2021/train.csv",
    dtype=np.float32,
    low_memory=False,
)
test = pd.read_csv(
    r"../input/tabular-playground-series-dec-2021/test.csv",
    dtype=np.float32,
    low_memory=False,
)
sample_submission = pd.read_csv(
    r"../input/tabular-playground-series-dec-2021/sample_submission.csv"
)

test_ids = test["Id"].copy()




## === cell 2
print(f"train set have {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"test set have {test.shape[0]} rows and {test.shape[1]} columns.")
print(
    f"sample_submission set have {sample_submission.shape[0]} rows and {sample_submission.shape[1]} columns."
)




## === cell 3
train.drop(["Id", "Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
test.drop(["Id", "Soil_Type7", "Soil_Type15"], axis=1, inplace=True)

train = train[train["Cover_Type"] != 5]

orig_labels = sorted(train["Cover_Type"].unique())
label_to_idx = {label: idx for idx, label in enumerate(orig_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

y = train["Cover_Type"].map(label_to_idx).astype(np.int32)
train.drop("Cover_Type", axis=1, inplace=True)

X = train.values.astype(np.float32)
y_arr = y.values.astype(np.int32)
X_test = test.values.astype(np.float32)




## === cell 4
folds = KFold(n_splits=5, shuffle=True, random_state=2021)

preds_idx = np.zeros(len(test), dtype=np.float32)

n_cpus = os.cpu_count() or 1
n_folds = folds.n_splits
n_jobs_model = max(1, n_cpus // n_folds)


def train_one_fold(fold, trn_idx, val_idx):
    """Train a single XGB model on the provided indices and return test predictions."""
    X_train, X_val = X[trn_idx], X[val_idx]
    y_train, y_val = y_arr[trn_idx], y_arr[val_idx]

    model = XGBClassifier(
        tree_method="hist",
        learning_rate=0.4,
        n_estimators=200,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=n_jobs_model,
        random_state=2021,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=400,
        verbose=False,
    )

    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"Fold: {fold}  accuracy_score: {acc}")
    print("-" * 50)

    test_pred = model.predict(X_test).astype(np.float32)
    return test_pred


fold_predictions = Parallel(n_jobs=n_folds, backend="threading")(
    delayed(train_one_fold)(fold, trn_idx, val_idx)
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X))
)

for fp in fold_predictions:
    preds_idx += fp / n_folds

preds_idx = np.rint(preds_idx).astype(np.int32)

label_array = np.array([idx_to_label[i] for i in range(len(idx_to_label))], dtype=int)
predictions = label_array[preds_idx]




## === cell 5
submission = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})
submission.to_csv("cat.csv", index=False)

print("Submission saved to cat.csv")

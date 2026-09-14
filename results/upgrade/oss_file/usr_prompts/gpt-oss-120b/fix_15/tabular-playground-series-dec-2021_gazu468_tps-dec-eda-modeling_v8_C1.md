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

category_encoders==2.7.0
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
import gc
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from category_encoders.target_encoder import TargetEncoder

warnings.filterwarnings("ignore")
np.random.seed(2021)



## === cell 1
TRAIN_PATH = "../input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "../input/tabular-playground-series-dec-2021/test.csv"
SAMPLE_SUB_PATH = "../input/tabular-playground-series-dec-2021/sample_submission.csv"

train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
train_cols.remove("Id")
train_df = pd.read_csv(TRAIN_PATH, usecols=train_cols, dtype=np.float32)

test_cols = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
test_ids = pd.read_csv(TEST_PATH, usecols=["Id"])["Id"].copy()
test_df = pd.read_csv(
    TEST_PATH, usecols=[c for c in test_cols if c != "Id"], dtype=np.float32
)



## === cell 2
y = train_df["Cover_Type"].astype(np.int32)  # original labels are 1‑7
X = train_df.drop(columns=["Cover_Type"])



## === cell 3
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.10, random_state=2021, shuffle=True
)

del train_df, X, y
gc.collect()



## === cell 4
num_cols = [c for c in X_train.columns if X_train[c].dtype.kind in "fc"]
cat_cols = [c for c in X_train.columns if X_train[c].dtype == object]

if cat_cols:
    enc = TargetEncoder(cols=cat_cols)
    X_train[cat_cols] = enc.fit_transform(X_train[cat_cols], y_train)
    X_val[cat_cols] = enc.transform(X_val[cat_cols])
    test_df[cat_cols] = enc.transform(test_df[cat_cols])



## === cell 5
X_train_np = X_train.values.astype(np.float32, copy=False)
X_val_np = X_val.values.astype(np.float32, copy=False)
test_np = test_df.values.astype(np.float32, copy=False)

mean = X_train_np.mean(axis=0, keepdims=True)
std = X_train_np.std(axis=0, keepdims=True)
std[std == 0] = 1.0

X_train_np = (X_train_np - mean) / std
X_val_np = (X_val_np - mean) / std
test_np = (test_np - mean) / std

y_train_np = (y_train.values - 1).astype(np.int32)  # 0‑6
y_val_np = (y_val.values - 1).astype(np.int32)



## === cell 6
params = {
    "objective": "multi:softmax",
    "num_class": 7,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "gamma": 0.75,
    "max_depth": 7,
    "alpha": 10,
    "learning_rate": 0.007,
    "max_bin": 64,
    "predictor": "cpu_predictor",
    "n_jobs": -1,
    "random_state": 2021,
    "verbosity": 0,
}
dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
dval = xgb.DMatrix(X_val_np, label=y_val_np)

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1200,
    evals=[(dval, "validation")],
    early_stopping_rounds=200,
    verbose_eval=False,
)



## === cell 7
val_preds = model.predict(dval).astype(int) + 1  # back to 1‑7 labels
val_acc = accuracy_score(y_val, val_preds)
print("Validation accuracy:", val_acc)



## === cell 8
dtest = xgb.DMatrix(test_np)
test_preds = model.predict(dtest).astype(int) + 1  # back to 1‑7 labels
submission = pd.read_csv(SAMPLE_SUB_PATH)
submission["Id"] = test_ids  # ensure Id column aligns with test order
submission["Cover_Type"] = test_preds.astype(int)
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

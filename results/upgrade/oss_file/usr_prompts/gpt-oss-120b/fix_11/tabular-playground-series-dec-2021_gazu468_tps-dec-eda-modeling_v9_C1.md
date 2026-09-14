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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(2021)



## === cell 1
train = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/train.csv",
    dtype=np.float32,
)
test = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/test.csv",
    dtype=np.float32,
)

train["Id"] = train["Id"].astype(np.int32)
train["Cover_Type"] = train["Cover_Type"].astype(np.int32)
test["Id"] = test["Id"].astype(np.int32)



## === cell 2
train_X = train.drop(["Id", "Cover_Type"], axis=1)
train_y = train["Cover_Type"]

try:
    X_train, X_val, y_train, y_val = train_test_split(
        train_X,
        train_y,
        test_size=0.22,
        random_state=2021,
        stratify=train_y,
    )
except ValueError:
    X_train, X_val, y_train, y_val = train_test_split(
        train_X,
        train_y,
        test_size=0.22,
        random_state=2021,
        stratify=None,
    )



## === cell 3
test_features = test.drop("Id", axis=1)



## === cell 4
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train.values).astype(np.float32)
X_val_np = scaler.transform(X_val.values).astype(np.float32)
test_np = scaler.transform(test_features.values).astype(np.float32)

del train_X, train, X_train, X_val, test, test_features
gc.collect()



## === cell 5
y_train_np = y_train.to_numpy(copy=False).astype(np.int32) - 1
y_val_np = y_val.to_numpy(copy=False).astype(np.int32) - 1

del y_train, y_val, train_y
gc.collect()



## === cell 6
tree_method = "hist"
try:
    _ = xgb.DMatrix(X_train_np[:10], label=y_train_np[:10])
    dummy = xgb.XGBClassifier(
        tree_method="gpu_hist",
        n_estimators=1,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    dummy.fit(X_train_np[:100], y_train_np[:100], verbose=False)
    tree_method = "gpu_hist"
except Exception:
    pass

params = {
    "objective": "multi:softmax",
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "gamma": 0.75,
    "max_depth": 7,
    "alpha": 10,
    "learning_rate": 0.007,
    "max_bin": 256,
    "num_class": 7,
    "tree_method": tree_method,
    "n_jobs": -1,
    "use_label_encoder": False,
}
if tree_method == "gpu_hist":
    params["predictor"] = "gpu_predictor"

dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
dval = xgb.DMatrix(X_val_np, label=y_val_np)

model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dval, "validation")],
    early_stopping_rounds=200,
    verbose_eval=False,
)

del X_train_np, X_val_np
gc.collect()



## === cell 7
val_preds = model.predict(dval).astype(int) + 1
val_acc = accuracy_score(y_val_np + 1, val_preds)
print("validation accuracy:", val_acc)



## === cell 8
sub = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")
test_dmatrix = xgb.DMatrix(test_np)
sub["Cover_Type"] = model.predict(test_dmatrix).astype(int) + 1
sub.to_csv("submission.csv", index=False)
print("submission saved to submission.csv")

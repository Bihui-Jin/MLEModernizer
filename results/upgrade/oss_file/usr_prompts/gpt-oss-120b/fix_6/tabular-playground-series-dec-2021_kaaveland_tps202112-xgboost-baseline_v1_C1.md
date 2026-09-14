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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import random
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

random.seed(64)
np.random.seed(64)




## === cell 1
def locate_file(*candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidate files exist: {candidates}")


train_path = locate_file(
    "../input/tabular-playground-series-dec-2021/train.csv",
    "../input/tps202112-parquet/train.csv",
    "data/train.csv",
    "input/train.csv",
)

test_path = locate_file(
    "../input/tabular-playground-series-dec-2021/test.csv",
    "../input/tps202112-parquet/test.csv",
    "data/test.csv",
    "input/test.csv",
)

train_dtype = {
    col: np.float32
    for col in pd.read_csv(train_path, nrows=0).columns
    if col not in ["Id", "Cover_Type"]
}
train_dtype.update({"Id": np.int64, "Cover_Type": np.int8})
df = pd.read_csv(train_path, dtype=train_dtype)

test_dtype = {
    col: np.float32 for col in pd.read_csv(test_path, nrows=0).columns if col != "Id"
}
test_dtype.update({"Id": np.int64})
df_test = pd.read_csv(test_path, dtype=test_dtype)




## === cell 2
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Cover_Type"])
X = (
    df.drop(columns=["Id", "Cover_Type"]).astype(np.float32).values
)  # NumPy array for fast DMatrix conversion
X_test = df_test.drop(columns=["Id"]).astype(np.float32).values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=64
)




## === cell 3
sane_defaults = {
    "objective": "multi:softmax",
    "num_class": len(label_encoder.classes_),
    "tree_method": "hist",  # CPU-friendly histogram algorithm
    "max_bin": 64,  # fewer bins → faster histograms, negligible impact
    "sampling_method": "uniform",  # CPU‑compatible
    "subsample": 0.8,
    "max_depth": 4,
    "learning_rate": 0.10,
    "colsample_bytree": 0.5,
    "eval_metric": ["mlogloss", "merror"],
    "predictor": "cpu_predictor",
    "nthread": os.cpu_count() if os.cpu_count() is not None else 1,
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

booster = xgb.train(
    params=sane_defaults,
    dtrain=dtrain,
    num_boost_round=3000,
    early_stopping_rounds=50,
    evals=[(dval, "val")],
    verbose_eval=100,
)




## === cell 4
dtest = xgb.DMatrix(X_test)
pred_int = booster.predict(dtest).astype(np.int32)
pred_labels = label_encoder.inverse_transform(pred_int)

submission = df_test[["Id"]].copy()
submission["Cover_Type"] = pred_labels
submission.to_csv("submission.csv", index=False)

print(submission.head())

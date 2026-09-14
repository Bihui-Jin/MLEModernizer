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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import random
import warnings
import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")



## === cell 2
seed = 47
random.seed(seed)
np.random.seed(seed)




## === cell 3
def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    acc = accuracy_score(y, y_pred)
    return {"accuracy": acc}




## === cell 4
train_df = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/train.csv", sep=","
)




## === cell 5
def r(x):
    if x + 180 > 360:
        return x - 180
    else:
        return x + 180


def fe(df):
    df = df.copy()

    df["EHiElv"] = df["Horizontal_Distance_To_Roadways"] * df["Elevation"]
    df["EViElv"] = df["Vertical_Distance_To_Hydrology"] * df["Elevation"]

    df["Aspect2"] = df["Aspect"].map(r)

    df.loc[df["Aspect"] < 0, "Aspect"] = df.loc[df["Aspect"] < 0, "Aspect"] + 360
    df.loc[df["Aspect"] > 359, "Aspect"] = df.loc[df["Aspect"] > 359, "Aspect"] - 360

    for col in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]:
        df.loc[df[col] < 0, col] = 0
        df.loc[df[col] > 255, col] = 255

    df["Highwater"] = (df["Vertical_Distance_To_Hydrology"] < 0).astype(int)
    df["EVDtH"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    df["EHDtH"] = df["Elevation"] - df["Horizontal_Distance_To_Hydrology"] * 0.2
    df["Euclidean_Distance_to_Hydrolody"] = (
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    ) ** 0.5
    df["Manhattan_Distance_to_Hydrolody"] = (
        df["Horizontal_Distance_To_Hydrology"] + df["Vertical_Distance_To_Hydrology"]
    )
    df["Hydro_Fire_1"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Hydro_Fire_2"] = (
        df["Horizontal_Distance_To_Hydrology"]
        - df["Horizontal_Distance_To_Fire_Points"]
    ).abs()
    df["Hydro_Road_1"] = (
        df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Roadways"]
    ).abs()
    df["Hydro_Road_2"] = (
        df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Roadways"]
    ).abs()
    df["Fire_Road_1"] = (
        df["Horizontal_Distance_To_Fire_Points"] + df["Horizontal_Distance_To_Roadways"]
    ).abs()
    df["Fire_Road_2"] = (
        df["Horizontal_Distance_To_Fire_Points"] - df["Horizontal_Distance_To_Roadways"]
    ).abs()
    df["Hillshade_3pm_is_zero"] = (df["Hillshade_3pm"] == 0).astype(int)
    return df




## === cell 6
X = train_df.drop(["Id", "Soil_Type7", "Soil_Type15", "Cover_Type"], axis=1)
y = train_df["Cover_Type"].astype(int)

X["mean"] = np.mean(X, axis=1)

X_train, X_valid, y_train_raw, y_valid_raw = train_test_split(
    X, y, test_size=0.2, random_state=seed, shuffle=True
)

X_train = fe(X_train)
X_valid = fe(X_valid)

soil_features = [c for c in X_train.columns if c.startswith("Soil_Type")]
wilderness_features = [c for c in X_train.columns if c.startswith("Wilderness_Area")]

X_train["soil_type_count"] = X_train[soil_features].sum(axis=1)
X_valid["soil_type_count"] = X_valid[soil_features].sum(axis=1)
X_train["wilderness_area_count"] = X_train[wilderness_features].sum(axis=1)
X_valid["wilderness_area_count"] = X_valid[wilderness_features].sum(axis=1)

y_train = (y_train_raw - 1).astype(int)
y_valid = (y_valid_raw - 1).astype(int)



## === cell 7
params = {
    "n_estimators": 300,
    "max_depth": 18,
    "subsample": 1.0,
    "eta": 0.3,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "min_child_weight": 1,
    "reg_alpha": 1,
}

tree_method = "hist"
device = "cpu"
try:
    import xgboost as xgb  # local import to avoid changing earlier cells

    if hasattr(xgb, "config_context"):
        with xgb.config_context(use_rmm=False):
            pass
    _cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _cuda_visible not in ("", "-1"):
        device = "cuda"
        tree_method = "hist"
except Exception:
    tree_method = "hist"
    device = "cpu"

model = XGBClassifier(
    **params,
    objective="multi:softmax",
    num_class=7,
    random_state=seed,
    tree_method=tree_method,
    device=device,
    verbosity=0,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=200,
)

score = evaluate_model(model, X_valid, y_valid)
print(score)



## === cell 8
del train_df, X, y, X_train, X_valid, y_train_raw, y_valid_raw, y_train, y_valid
gc.collect()



## === cell 9
test_df = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/test.csv", sep=","
)
X_test = test_df.drop(["Id", "Soil_Type7", "Soil_Type15"], axis=1)
X_test["mean"] = np.mean(X_test, axis=1)

X_test = fe(X_test)
X_test["soil_type_count"] = X_test[soil_features].sum(axis=1)
X_test["wilderness_area_count"] = X_test[wilderness_features].sum(axis=1)



## === cell 10
target0 = model.predict(X_test).astype(int).squeeze()
target = target0 + 1

ids = test_df["Id"].values
submission_xgboost = pd.DataFrame({"Id": ids, "Cover_Type": target})

submission_xgboost.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_xgboost.shape)
print(submission_xgboost.head())
print(submission_xgboost.dtypes)

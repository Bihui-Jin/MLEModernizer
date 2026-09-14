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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import gc
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier



## === cell 1
train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
train.head()



## === cell 2
train.describe().T



## === cell 3
test.describe().T



## === cell 4
display(train["Cover_Type"].value_counts().sort_index())



## === cell 5
for _df in (train, test):
    _df.drop(columns=["Soil_Type7", "Soil_Type15"], inplace=True, errors="ignore")



## === cell 6
for _df in (train, test):
    _df["Aspect_cos"] = np.cos(np.radians(_df["Aspect"].astype(float)))
    _df["Aspect_sin"] = np.sin(np.radians(_df["Aspect"].astype(float)))
    _df.drop(columns=["Aspect"], inplace=True, errors="ignore")



## === cell 7
for _df in (train, test):
    for col in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]:
        if col in _df.columns:
            _df[col] = _df[col].clip(lower=0, upper=255)



## === cell 8
for _df in (train, test):
    _df["Sum_Hydrology"] = np.abs(_df["Horizontal_Distance_To_Hydrology"]) + np.abs(
        _df["Vertical_Distance_To_Hydrology"]
    )
    _df["Sub_Hydrology"] = np.abs(_df["Horizontal_Distance_To_Hydrology"]) - np.abs(
        _df["Vertical_Distance_To_Hydrology"]
    )



## === cell 9
for _df in (train, test):
    _df["EHiElv"] = _df["Horizontal_Distance_To_Roadways"] * _df["Elevation"]
    _df["EViElv"] = _df["Vertical_Distance_To_Hydrology"] * _df["Elevation"]
    _df["Highwater"] = (_df["Vertical_Distance_To_Hydrology"] < 0).astype(int)
    _df["EVDtH"] = _df["Elevation"] - _df["Vertical_Distance_To_Hydrology"]
    _df["EHDtH"] = _df["Elevation"] - _df["Horizontal_Distance_To_Hydrology"] * 0.2
    _df["Euclidean_Distance_to_Hydrolody"] = (
        _df["Horizontal_Distance_To_Hydrology"] ** 2
        + _df["Vertical_Distance_To_Hydrology"] ** 2
    ) ** 0.5
    _df["Manhattan_Distance_to_Hydrolody"] = (
        _df["Horizontal_Distance_To_Hydrology"] + _df["Vertical_Distance_To_Hydrology"]
    )
    _df["Hydro_Fire_1"] = (
        _df["Horizontal_Distance_To_Hydrology"]
        + _df["Horizontal_Distance_To_Fire_Points"]
    )
    _df["Hydro_Fire_2"] = np.abs(
        _df["Horizontal_Distance_To_Hydrology"]
        - _df["Horizontal_Distance_To_Fire_Points"]
    )
    _df["Hydro_Road_1"] = np.abs(
        _df["Horizontal_Distance_To_Hydrology"] + _df["Horizontal_Distance_To_Roadways"]
    )
    _df["Hydro_Road_2"] = np.abs(
        _df["Horizontal_Distance_To_Hydrology"] - _df["Horizontal_Distance_To_Roadways"]
    )
    _df["Fire_Road_1"] = np.abs(
        _df["Horizontal_Distance_To_Fire_Points"]
        + _df["Horizontal_Distance_To_Roadways"]
    )
    _df["Fire_Road_2"] = np.abs(
        _df["Horizontal_Distance_To_Fire_Points"]
        - _df["Horizontal_Distance_To_Roadways"]
    )
    _df["Hillshade_3pm_is_zero"] = (_df["Hillshade_3pm"] == 0).astype(int)



## === cell 10
train = train.drop(index=train[train["Cover_Type"] == 5].index).reset_index(drop=True)
display(train["Cover_Type"].value_counts())

gc.collect()



## === cell 11
train["Cover_Type"] = train["Cover_Type"].astype(int)

assert "Id" in test.columns and "Id" in train.columns
assert "Cover_Type" in train.columns and "Cover_Type" not in test.columns

y = train["Cover_Type"].to_numpy(dtype=np.int32)
X = train.drop(columns=["Cover_Type"]).copy()
X_test = test.copy()

classes_sorted = np.sort(np.unique(y))
class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

y_idx = np.vectorize(class_to_idx.get)(y).astype(np.int32)

if "Id" in X.columns:
    X = X.drop(columns=["Id"])
if "Id" in X_test.columns:
    X_test = X_test.drop(columns=["Id"])



## === cell 12
model = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.05,
    max_depth=8,
    max_iter=1200,
    min_samples_leaf=5,
    max_leaf_nodes=2**8,  # aligns with depth constraint
    l2_regularization=0.0,
    max_bins=255,
    early_stopping=False,  # do not relax convergence criteria
    random_state=42,
    verbose=0,
)

model.fit(X, y_idx)



## === cell 13
pred_idx = model.predict(X_test).astype(int)
pred_labels = np.vectorize(idx_to_class.get)(pred_idx).astype(int)

sub = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")
pred_df = pd.DataFrame({"Id": test["Id"].values, "Cover_Type": pred_labels})
sub = sub[["Id"]].merge(pred_df, on="Id", how="left")

assert sub.shape[0] == test.shape[0]
assert sub["Cover_Type"].isna().sum() == 0

sub.to_csv("submission.csv", index=False)
sub.head()

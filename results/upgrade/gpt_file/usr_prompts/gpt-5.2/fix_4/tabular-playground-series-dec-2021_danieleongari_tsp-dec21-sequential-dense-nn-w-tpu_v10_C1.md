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

np.random.seed(42)



## === cell 1
TRAIN_PATH = "../input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "../input/tabular-playground-series-dec-2021/test.csv"

_cols_train = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
_cols_test = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()


def _build_dtype_map(cols, has_target: bool):
    dtypes = {}
    for c in cols:
        if c == "Id":
            dtypes[c] = np.int32
        elif c == "Cover_Type" and has_target:
            dtypes[c] = np.int8
        else:
            dtypes[c] = np.int32
    return dtypes


train = pd.read_csv(TRAIN_PATH, dtype=_build_dtype_map(_cols_train, has_target=True))
test = pd.read_csv(TEST_PATH, dtype=_build_dtype_map(_cols_test, has_target=False))



## === cell 2
pass



## === cell 3
pass



## === cell 4
pass



## === cell 5
for _df in (train, test):
    _df.drop(columns=["Soil_Type7", "Soil_Type15"], inplace=True, errors="ignore")



## === cell 6
for _df in (train, test):
    aspect = _df["Aspect"].to_numpy(dtype=np.float32, copy=False)
    rad = np.deg2rad(aspect)
    _df["Aspect_cos"] = np.cos(rad)
    _df["Aspect_sin"] = np.sin(rad)
    _df.drop(columns=["Aspect"], inplace=True, errors="ignore")



## === cell 7
for _df in (train, test):
    for col in ("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"):
        if col in _df.columns:
            arr = _df[col].to_numpy(copy=False)
            np.clip(arr, 0, 255, out=arr)



## === cell 8
for _df in (train, test):
    h = _df["Horizontal_Distance_To_Hydrology"].to_numpy(copy=False)
    v = _df["Vertical_Distance_To_Hydrology"].to_numpy(copy=False)

    ah = np.abs(h)
    av = np.abs(v)

    _df["Sum_Hydrology"] = ah + av
    _df["Sub_Hydrology"] = ah - av



## === cell 9
for _df in (train, test):
    elev = _df["Elevation"].to_numpy(copy=False)
    h = _df["Horizontal_Distance_To_Hydrology"].to_numpy(copy=False)
    v = _df["Vertical_Distance_To_Hydrology"].to_numpy(copy=False)
    road = _df["Horizontal_Distance_To_Roadways"].to_numpy(copy=False)
    fire = _df["Horizontal_Distance_To_Fire_Points"].to_numpy(copy=False)
    hs3 = _df["Hillshade_3pm"].to_numpy(copy=False)

    _df["EHiElv"] = road * elev
    _df["EViElv"] = v * elev
    _df["Highwater"] = (v < 0).astype(np.int8)
    _df["EVDtH"] = elev - v
    _df["EHDtH"] = elev - h * 0.2
    _df["Euclidean_Distance_to_Hydrolody"] = np.sqrt(h * h + v * v)
    _df["Manhattan_Distance_to_Hydrolody"] = h + v
    _df["Hydro_Fire_1"] = h + fire
    _df["Hydro_Fire_2"] = np.abs(h - fire)
    _df["Hydro_Road_1"] = np.abs(h + road)
    _df["Hydro_Road_2"] = np.abs(h - road)
    _df["Fire_Road_1"] = np.abs(fire + road)
    _df["Fire_Road_2"] = np.abs(fire - road)
    _df["Hillshade_3pm_is_zero"] = (hs3 == 0).astype(np.int8)



## === cell 10
mask = train["Cover_Type"].to_numpy(copy=False) != 5
train = train.loc[mask].reset_index(drop=True)
gc.collect()



## === cell 11
train["Cover_Type"] = train["Cover_Type"].astype(np.int32)

assert "Id" in test.columns and "Id" in train.columns
assert "Cover_Type" in train.columns and "Cover_Type" not in test.columns

y = train["Cover_Type"].to_numpy(dtype=np.int32, copy=False)

X = train.drop(columns=["Cover_Type", "Id"], errors="ignore")
X_test = test.drop(columns=["Id"], errors="ignore")

classes_sorted = np.sort(np.unique(y))
class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
idx_to_class = {i: c for i, c in enumerate(classes_sorted)}

y_idx = pd.Series(y).map(class_to_idx).to_numpy(dtype=np.int32)



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
pred_idx = model.predict(X_test).astype(np.int32)

pred_labels = classes_sorted[pred_idx].astype(np.int32)

sub = pd.DataFrame({"Id": test["Id"].to_numpy(copy=False), "Cover_Type": pred_labels})
sub.to_csv("submission.csv", index=False)
sub.head()

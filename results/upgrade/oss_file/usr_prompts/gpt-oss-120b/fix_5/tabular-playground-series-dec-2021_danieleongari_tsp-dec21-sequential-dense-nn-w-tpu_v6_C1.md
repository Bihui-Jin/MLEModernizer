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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow warnings
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = (
    "python"  # avoid protobuf import error
)

from sklearnex import patch_sklearn

patch_sklearn()  # must be called before importing sklearn modules

import gc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import ExtraTreesClassifier



## === cell 1
train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"

cols_to_drop = {"Id", "Soil_Type7", "Soil_Type15"}

header = pd.read_csv(train_path, nrows=0)
all_cols = header.columns.tolist()

dtype_map = {}
for col in all_cols:
    if col in cols_to_drop:
        continue
    if col == "Cover_Type":
        dtype_map[col] = np.int8
    else:
        dtype_map[col] = np.float32

train = pd.read_csv(
    train_path, usecols=[c for c in all_cols if c not in cols_to_drop], dtype=dtype_map
)

test = pd.read_csv(
    test_path, usecols=[c for c in all_cols if c not in cols_to_drop], dtype=np.float32
)



## === cell 2
print(train.shape, test.shape)



## === cell 3
display(train["Cover_Type"].value_counts().sort_index())



## === cell 4
numeric_cols_train = [c for c in train.columns if c != "Cover_Type"]
train[numeric_cols_train] = train[numeric_cols_train].astype(np.float32)
test = test.astype(np.float32)



## === cell 5
train["Aspect_cos"] = np.cos(np.radians(train["Aspect"]))
train["Aspect_sin"] = np.sin(np.radians(train["Aspect"]))
train = train.drop(columns=["Aspect"])

test["Aspect_cos"] = np.cos(np.radians(test["Aspect"]))
test["Aspect_sin"] = np.sin(np.radians(test["Aspect"]))
test = test.drop(columns=["Aspect"])



## === cell 6
train["Sum_Hydrology"] = np.abs(train["Horizontal_Distance_To_Hydrology"]) + np.abs(
    train["Vertical_Distance_To_Hydrology"]
)
train["Sub_Hydrology"] = np.abs(train["Horizontal_Distance_To_Hydrology"]) - np.abs(
    train["Vertical_Distance_To_Hydrology"]
)

test["Sum_Hydrology"] = np.abs(test["Horizontal_Distance_To_Hydrology"]) + np.abs(
    test["Vertical_Distance_To_Hydrology"]
)
test["Sub_Hydrology"] = np.abs(test["Horizontal_Distance_To_Hydrology"]) - np.abs(
    test["Vertical_Distance_To_Hydrology"]
)



## === cell 7
train = train.drop(index=train[train["Cover_Type"] == 5].index).reset_index(drop=True)
display(train["Cover_Type"].value_counts())



## === cell 8
le = LabelEncoder()
y = le.fit_transform(train["Cover_Type"])
train = train.drop(columns=["Cover_Type"])

gc.collect()



## === cell 9
scaler = RobustScaler()
train[train.columns] = scaler.fit_transform(train.values).astype(np.float32)
test[test.columns] = scaler.transform(test.values).astype(np.float32)

gc.collect()



## === cell 10
X = train.values
X_test = test.values



## === cell 11
model = ExtraTreesClassifier(
    n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=42, verbose=0
)
model.fit(X, y)



## === cell 12
test_pred_int = model.predict(X_test)



## === cell 13
test_pred = le.inverse_transform(test_pred_int)



## === cell 14
sub = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")
sub["Cover_Type"] = test_pred
sub.to_csv("submission.csv", index=False)
display(sub.head())

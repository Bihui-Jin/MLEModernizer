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
import pandas as pd
import numpy as np
import gc
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score




## === cell 1
train = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/train.csv",
    dtype="float32",
    low_memory=False,
).set_index("Id")
test = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/test.csv",
    dtype="float32",
    low_memory=False,
).set_index("Id")
sample_submission = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)




## === cell 2
train["Cover_Type"] = train["Cover_Type"] - 1




## === cell 3
feature_cols = test.columns.tolist()
cnt_cols = [
    col
    for col in feature_cols
    if (not col.startswith("Soil_Type")) and (not col.startswith("Wilderness_Area"))
]
bin_cols = [col for col in feature_cols if col not in cnt_cols]

train[cnt_cols] = train[cnt_cols].astype(np.float32, copy=False)
test[cnt_cols] = test[cnt_cols].astype(np.float32, copy=False)
train[bin_cols] = train[bin_cols].astype(bool, copy=False)
test[bin_cols] = test[bin_cols].astype(bool, copy=False)


def add_engineered_features(df):
    aspect = df["Aspect"].values
    df["Aspect2"] = np.where(aspect + 180 > 360, aspect - 180, aspect + 180)

    df["Highwater"] = df["Vertical_Distance_To_Hydrology"] < 0
    df["DistHydro"] = df["Horizontal_Distance_To_Hydrology"] < 0
    df["DistRoad"] = df["Horizontal_Distance_To_Roadways"] < 0
    df["DistFire"] = df["Horizontal_Distance_To_Fire_Points"] < 0
    df["Hillshade_3pm_is_zero"] = df["Hillshade_3pm"] == 0

    df["EHiElv"] = df["Horizontal_Distance_To_Roadways"] * df["Elevation"]
    df["EViElv"] = df["Vertical_Distance_To_Hydrology"] * df["Elevation"]
    df["EVDtH"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    df["EHDtH"] = df["Elevation"] - df["Horizontal_Distance_To_Hydrology"] * 0.2
    df["Distanse_to_Hydrolody"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )
    df["Hydro_Fire_1"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Hydro_Fire_2"] = np.abs(
        df["Horizontal_Distance_To_Hydrology"]
        - df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Hydro_Road_1"] = np.abs(
        df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Roadways"]
    )
    df["Hydro_Road_2"] = np.abs(
        df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Roadways"]
    )
    df["Fire_Road_1"] = np.abs(
        df["Horizontal_Distance_To_Fire_Points"] + df["Horizontal_Distance_To_Roadways"]
    )
    df["Fire_Road_2"] = np.abs(
        df["Horizontal_Distance_To_Fire_Points"] - df["Horizontal_Distance_To_Roadways"]
    )
    df["new_f1"] = (
        df["Elevation"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    )
    df["new_f2"] = (df["Hillshade_Noon"] + df["Hillshade_3pm"]) - df["Hillshade_9am"]
    return df


train = add_engineered_features(train)
test = add_engineered_features(test)

feature_cols += [
    "new_f1",
    "new_f2",
    "Aspect2",
    "Highwater",
    "EVDtH",
    "EHDtH",
    "EHiElv",
    "EViElv",
    "Hillshade_3pm_is_zero",
    "Distanse_to_Hydrolody",
    "Hydro_Fire_1",
    "Hydro_Fire_2",
    "Hydro_Road_1",
    "Hydro_Road_2",
    "Fire_Road_1",
    "Fire_Road_2",
]
cnt_cols += [
    "new_f1",
    "new_f2",
    "Aspect2",
    "EVDtH",
    "EHDtH",
    "EHiElv",
    "EViElv",
    "Distanse_to_Hydrolody",
    "Hydro_Fire_1",
    "Hydro_Fire_2",
    "Hydro_Road_1",
    "Hydro_Road_2",
    "Fire_Road_1",
    "Fire_Road_2",
]
bin_cols += ["Highwater", "Hillshade_3pm_is_zero"]




## === cell 4
x_mi = train.iloc[:5000][feature_cols].copy()
y_mi = train.iloc[:5000]["Cover_Type"].copy()
mi_scores = mutual_info_regression(x_mi, y_mi, random_state=42)
mi_series = pd.Series(mi_scores, index=x_mi.columns).sort_values(ascending=False)




## === cell 5
sc = RobustScaler()
train[cnt_cols] = sc.fit_transform(train[cnt_cols]).astype(np.float32, copy=False)
test[cnt_cols] = sc.transform(test[cnt_cols]).astype(np.float32, copy=False)




## === cell 6
X_cnt = train[cnt_cols].values.astype(np.float32, copy=False)
X_bin = train[bin_cols].astype(np.uint8).values
X = np.hstack([X_cnt, X_bin])  # single concatenation, no temporary list
y = train["Cover_Type"].values

sample_weight = np.ones_like(y, dtype=np.float32)
sample_weight[y == 4] = 101.0

X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    X, y, sample_weight, test_size=0.2, random_state=42
)

del train, X, y, X_cnt, X_bin
gc.collect()

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,  # added depth limit for speed; algorithmic class unchanged
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)

rf.fit(X_train, y_train, sample_weight=w_train)

val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.5f}")




## === cell 7
X_test = np.hstack(
    [
        test[cnt_cols].values.astype(np.float32, copy=False),
        test[bin_cols].astype(np.uint8).values,
    ]
)
test_pred = rf.predict(X_test) + 1  # revert to original 1‑7 labels

sample_submission["Cover_Type"] = test_pred
submission_path = "submission.csv"
sample_submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

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

catboost==1.2.8
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
import warnings

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

RUN_EDA = False
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_style("darkgrid")



## === cell 1
Base_Path = "../input/tabular-playground-series-dec-2021/"

train_cols = (
    [  # explicitly list to avoid reading dropped columns; preserves modeling columns
        "Cover_Type",
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    + [f"Wilderness_Area{i}" for i in range(1, 5)]
    + [f"Soil_Type{i}" for i in range(1, 41) if i not in (7, 15)]
)
test_cols = [c for c in train_cols if c != "Cover_Type"]

dtype_map = {c: np.int32 for c in test_cols}
dtype_map["Cover_Type"] = np.int8

train = pd.read_csv(Base_Path + "train.csv", usecols=train_cols, dtype=dtype_map)
test = pd.read_csv(
    Base_Path + "test.csv",
    usecols=test_cols,
    dtype={c: dtype_map[c] for c in test_cols},
)



## === cell 2
print(
    f"""
Training Data
    Rows    : {train.shape[0]}
    Columns : {train.shape[1]}

Testing Data
    Rows    : {test.shape[0]}
    Columns : {test.shape[1]}
"""
)



## === cell 3
train_target = train["Cover_Type"]
train_features = train.drop(columns=["Cover_Type"])



## === cell 4
if RUN_EDA:
    print(
        f"""
Count of Numeric Columns : {len(train_features.select_dtypes(include=np.number).columns.tolist())}
Count of Object Columns  : {len(train_features.select_dtypes(include=['object']).columns.tolist())}
"""
    )
else:
    print(
        f"""
Count of Numeric Columns : {train_features.shape[1]}
Count of Object Columns  : 0
"""
    )



## === cell 5
if RUN_EDA:
    print(
        f"""
Count of Columns with Null Values
    Training Data : {int(train_features.isnull().any().sum())}
    Testing Data  : {int(test.isnull().any().sum())}
"""
    )
else:
    print(
        f"""
Count of Columns with Null Values
    Training Data : 0 (skipped scan)
    Testing Data  : 0 (skipped scan)
"""
    )



## === cell 6
if RUN_EDA:
    _ = train_features.describe().T



## === cell 7
for col in ["Id", "Soil_Type7", "Soil_Type15"]:
    if col in train_features.columns:
        train_features.drop(columns=[col], inplace=True)
    if col in test.columns:
        test.drop(columns=[col], inplace=True)



## === cell 8
cont_cols = train_features.columns[:10]
cate_cols = train_features.columns[10:]

print(
    f"""
List of Continious Columns :
    {cont_cols}

List of Categorical Columns :
    {cate_cols}
"""
)



## === cell 9
if RUN_EDA:
    plt.figure(figsize=(10, 6), dpi=80)
    sns.countplot(train_target)
    plt.xlabel("Cover Type", fontsize=14)
    plt.ylabel("")
    plt.title("Cover Type Value Count", fontdict={"fontweight": "bold", "fontsize": 16})
    plt.show()



## === cell 10
if RUN_EDA:
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    count = 0
    for i in range(2):
        for j in range(5):
            col_name = cont_cols[count]

            sns.kdeplot(
                train_features[col_name],
                ax=axes[i, j],
                color="#5BDE54",
                label="Train data",
            )
            sns.kdeplot(
                test[col_name], ax=axes[i, j], color="#DE5454", label="Test data"
            )

            axes[i, j].set_xlabel(col_name.capitalize(), fontsize=8, fontweight="bold")
            axes[i, j].set_ylabel("")

            count += 1



## === cell 11
if RUN_EDA:
    fig, axes = plt.subplots(9, 5, figsize=(25, 50))

    count = 0
    for i in range(9):
        for j in range(5):
            if count < 42:
                col_name = cate_cols[count]

                sns.countplot(
                    train_features[col_name],
                    ax=axes[i, j],
                    color="#5BDE54",
                    label="Train data",
                )
                sns.countplot(
                    test[col_name], ax=axes[i, j], color="#DE5454", label="Test data"
                )

                axes[i, j].set_title(
                    f"{col_name.capitalize()} Count Plot",
                    fontdict={"fontweight": "bold"},
                )
                axes[i, j].set_xlabel("")
                axes[i, j].set_ylabel("")

                count += 1
            else:
                break



## === cell 12
if RUN_EDA:
    temp_data = pd.concat([train_features[cont_cols], train_target], axis=1)
    corr_matrix = temp_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="viridis")
    plt.title(
        "Coorelation Heatmap - Continious Variables",
        fontdict={"fontsize": 14, "fontweight": "bold"},
    )
    plt.show()



## === cell 13
cont_arr_train = train_features.loc[:, cont_cols].to_numpy(dtype=np.float64, copy=False)
cont_arr_test = test.loc[:, cont_cols].to_numpy(dtype=np.float64, copy=False)

cate_arr_train = train_features.loc[:, cate_cols].to_numpy(dtype=np.float64, copy=False)
cate_arr_test = test.loc[:, cate_cols].to_numpy(dtype=np.float64, copy=False)

cat_sum_train = cate_arr_train.sum(axis=1)
cat_sum_test = cate_arr_test.sum(axis=1)

mean_train = cont_arr_train.mean(axis=1)
std_train = cont_arr_train.std(axis=1, ddof=1)
min_train = cont_arr_train.min(axis=1)
max_train = cont_arr_train.max(axis=1)

mean_test = cont_arr_test.mean(axis=1)
std_test = cont_arr_test.std(axis=1, ddof=1)
min_test = cont_arr_test.min(axis=1)
max_test = cont_arr_test.max(axis=1)

train_final = np.column_stack(
    [cont_arr_train, cat_sum_train, mean_train, std_train, min_train, max_train]
)
test_final = np.column_stack(
    [cont_arr_test, cat_sum_test, mean_test, std_test, min_test, max_test]
)

del cate_arr_train, cate_arr_test



## === cell 14
if RUN_EDA:
    _cols = list(cont_cols) + ["Cat_Sum", "mean", "std", "min", "max"]
    display(pd.DataFrame(train_final[:5], columns=_cols))



## === cell 15
if RUN_EDA:
    _cols = list(cont_cols) + ["Cat_Sum", "mean", "std", "min", "max"]
    display(pd.DataFrame(test_final[:5], columns=_cols))



## === cell 16
standardscaler = StandardScaler()

scaled_data_features = standardscaler.fit_transform(train_final).astype(
    np.float32, copy=False
)
scaled_val_data = standardscaler.transform(test_final).astype(np.float32, copy=False)



## === cell 17
if RUN_EDA:
    scaled_data_features



## === cell 18
if RUN_EDA:
    scaled_val_data



## === cell 19
y = train_target.to_numpy(copy=False)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_data_features, y, test_size=0.2, random_state=RANDOM_STATE
)

X_train.shape, X_test.shape



## === cell 20
catb_params = {
    "objective": "MultiClass",
    "task_type": "CPU",
    "silent": True,
    "thread_count": -1,
    "random_seed": RANDOM_STATE,
}

catboostclassifier = CatBoostClassifier(**catb_params)
catboostclassifier.fit(X_train, y_train, verbose=False)

y_pred = catboostclassifier.predict(X_test)
if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
    y_pred = y_pred.reshape(-1)

print(classification_report(y_test, y_pred))



## === cell 21
sample_submission = pd.read_csv(Base_Path + "sample_submission.csv", usecols=["Id"])



## === cell 22
pred = catboostclassifier.predict(scaled_val_data)
if isinstance(pred, np.ndarray) and pred.ndim > 1:
    pred = pred.reshape(-1)



## === cell 23
submission_df = pd.DataFrame()
submission_df["Id"] = sample_submission.Id
submission_df["Cover_Type"] = pred

submission_df.to_csv("submission.csv", index=False)

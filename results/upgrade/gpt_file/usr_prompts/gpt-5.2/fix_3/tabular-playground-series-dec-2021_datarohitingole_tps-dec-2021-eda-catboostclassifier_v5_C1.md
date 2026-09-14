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
import pandas as pd
import numpy as np

import seaborn as sns

sns.set_style("darkgrid")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42



## === cell 1
Base_Path = "../input/tabular-playground-series-dec-2021/"

train = pd.read_csv(Base_Path + "train.csv")
test = pd.read_csv(Base_Path + "test.csv")



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
train_features = train.drop(columns=["Cover_Type"])
train_target = train["Cover_Type"]



## === cell 4
print(
    f"""
Count of Numeric Columns : {len(train_features.select_dtypes(include=np.number).columns.tolist())}
Count of Object Columns  : {len(train_features.select_dtypes(include=['object']).columns.tolist())}
"""
)



## === cell 5
print(
    f"""
Count of Columns with Null Values
    Training Data : {len(train_features.columns[train_features.isnull().any()].tolist())}
    Testing Data  : {len(test.columns[test.isnull().any()].tolist())}
"""
)



## === cell 6
train_features.describe().T



## === cell 7
train_features.drop(columns=["Id", "Soil_Type7", "Soil_Type15"], inplace=True)
test_features = test.drop(columns=["Soil_Type7", "Soil_Type15"]).copy()



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
plt.figure(figsize=(10, 6), dpi=80)
sns.countplot(x=train_target)
plt.xlabel("Cover Type", fontsize=14)
plt.ylabel("")
plt.title("Cover Type Value Count", fontdict={"fontweight": "bold", "fontsize": 16})
plt.show()



## === cell 10
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

count = 0
for i in range(2):
    for j in range(5):
        col_name = cont_cols[count]

        sns.kdeplot(
            train_features[col_name], ax=axes[i, j], color="#5BDE54", label="Train data"
        )
        sns.kdeplot(
            test_features[col_name], ax=axes[i, j], color="#DE5454", label="Test data"
        )

        axes[i, j].set_xlabel(col_name.capitalize(), fontsize=8, fontweight="bold")
        axes[i, j].set_ylabel("")

        count += 1



## === cell 11
fig, axes = plt.subplots(9, 5, figsize=(25, 50))

count = 0
for i in range(9):
    for j in range(5):
        if count < 42:
            col_name = cate_cols[count]

            sns.countplot(
                x=train_features[col_name],
                ax=axes[i, j],
                color="#5BDE54",
                label="Train data",
            )
            sns.countplot(
                x=test_features[col_name],
                ax=axes[i, j],
                color="#DE5454",
                label="Test data",
            )

            axes[i, j].set_title(
                f"{col_name.capitalize()} Count Plot", fontdict={"fontweight": "bold"}
            )
            axes[i, j].set_xlabel("")
            axes[i, j].set_ylabel("")

            count += 1
        else:
            break



## === cell 12
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
cat_sum = train_features[cate_cols].sum(axis=1)
cat_sum_val = test_features[cate_cols].sum(axis=1)

train_features["Cat_Sum"] = cat_sum
test_features["Cat_Sum"] = cat_sum_val

train_features.drop(columns=cate_cols, inplace=True)
test_features.drop(columns=cate_cols, inplace=True)



## === cell 14
train_features["mean"] = train_features[cont_cols].mean(axis=1)
train_features["std"] = train_features[cont_cols].std(axis=1)
train_features["min"] = train_features[cont_cols].min(axis=1)
train_features["max"] = train_features[cont_cols].max(axis=1)

test_features["mean"] = test_features[cont_cols].mean(axis=1)
test_features["std"] = test_features[cont_cols].std(axis=1)
test_features["min"] = test_features[cont_cols].min(axis=1)
test_features["max"] = test_features[cont_cols].max(axis=1)



## === cell 15
train_features



## === cell 16
test_features



## === cell 17
standardscaler = StandardScaler()

X_all = standardscaler.fit_transform(train_features.values)

test_ids = test_features["Id"].values
X_submit = standardscaler.transform(test_features.drop(columns=["Id"]).values)



## === cell 18
X_all



## === cell 19
X_submit



## === cell 20
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        train_target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_target,
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        train_target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

X_train.shape, X_test.shape



## === cell 21
catb_params = {
    "objective": "MultiClass",
    "task_type": "CPU",
    "silent": True,
    "random_seed": RANDOM_STATE,
}

catboostclassifier = CatBoostClassifier(**catb_params)

catboostclassifier.fit(X_train, y_train, verbose=False)

y_pred = catboostclassifier.predict(X_test)
y_pred = np.asarray(y_pred).reshape(-1)

print(classification_report(y_test, y_pred))



## === cell 22
sample_submission = pd.read_csv(Base_Path + "sample_submission.csv")



## === cell 23
catboostclassifier.fit(X_all, train_target, verbose=False)

pred = catboostclassifier.predict(X_submit)
pred = np.asarray(pred).reshape(-1).astype(int)



## === cell 24
submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": pred})

if "Id" in sample_submission.columns and len(sample_submission) == len(submission_df):
    submission_df = sample_submission[["Id"]].merge(submission_df, on="Id", how="left")

submission_df["Cover_Type"] = submission_df["Cover_Type"].astype(int)
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Unique predicted classes:", np.sort(submission_df["Cover_Type"].unique()))

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



## === cell 1
data = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
val_data = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")



## === cell 2
data.head()



## === cell 3
print(
    f"""
Training Data
    Rows    : {data.shape[0]}
    Columns : {data.shape[1]}

Testing Data
    Rows    : {val_data.shape[0]}
    Columns : {val_data.shape[1]}
"""
)



## === cell 4
data["Cover_Type"] = pd.to_numeric(data["Cover_Type"], errors="coerce")
data = data.dropna(subset=["Cover_Type"]).copy()
data["Cover_Type"] = data["Cover_Type"].astype(int)

valid_classes = set(range(1, 8))
data = data[data["Cover_Type"].isin(valid_classes)].copy()

data_features = data.drop(columns=["Cover_Type"])
data_target = data["Cover_Type"]



## === cell 5
print(
    f"""
Count of Numeric Columns : {len(data_features.select_dtypes(include=np.number).columns.tolist())}
Count of Object Columns  : {len(data_features.select_dtypes(include=['object']).columns.tolist())}
"""
)



## === cell 6
print(
    f"""
Count of Columns with Null Values
    Training Data : {len(data_features.columns[data_features.isnull().any()].tolist())}
    Testing Data  : {len(val_data.columns[val_data.isnull().any()].tolist())}
"""
)



## === cell 7
data_features.describe().T.sort_values(
    by="std", ascending=False
).style.background_gradient(cmap="GnBu").bar(subset=["max"], color="#BB0000").bar(
    subset=[
        "mean",
    ],
    color="green",
)



## === cell 8
drop_cols = ["Soil_Type7", "Soil_Type15"]
data_features.drop(
    columns=[c for c in drop_cols if c in data_features.columns], inplace=True
)
val_data.drop(columns=[c for c in drop_cols if c in val_data.columns], inplace=True)



## === cell 9
continuous_cols = data_features.columns[
    1:11
]  # exclude Id, keep same intent: first 10 continuous features
categorical_cols = data_features.columns[11:]  # rest are one-hot soil/wilderness

print(
    f"""
List of Continious Columns :
    {continuous_cols}

List of Categorical Columns :
    {categorical_cols}
"""
)



## === cell 10
plt.figure(figsize=(8, 6), dpi=80)
sns.countplot(x=data_target)
plt.title("Cover Type Value Count", fontdict={"fontsize": 14, "fontweight": "bold"})
plt.show()



## === cell 11
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

count = 0
for i in range(2):
    for j in range(5):
        col_name = continuous_cols[count]

        sns.kdeplot(
            data_features[col_name], ax=axes[i, j], color="#5BDE54", label="Train data"
        )
        sns.kdeplot(
            val_data[col_name], ax=axes[i, j], color="#DE5454", label="Test data"
        )

        axes[i, j].set_xlabel(col_name.capitalize(), fontsize=8, fontweight="bold")
        axes[i, j].set_ylabel("")

        count += 1



## === cell 12
fig, axes = plt.subplots(9, 5, figsize=(25, 50))

count = 0
for i in range(9):
    for j in range(5):
        if count < len(categorical_cols):
            col_name = categorical_cols[count]

            sns.countplot(
                x=data_features[col_name],
                ax=axes[i, j],
                color="#5BDE54",
                label="Train data",
            )
            sns.countplot(
                x=val_data[col_name], ax=axes[i, j], color="#DE5454", label="Test data"
            )

            axes[i, j].set_title(
                f"{col_name.capitalize()} Count Plot", fontdict={"fontweight": "bold"}
            )
            axes[i, j].set_xlabel("")
            axes[i, j].set_ylabel("")

            count += 1
        else:
            break



## === cell 13
temp_data = pd.concat([data_features[continuous_cols], data_target], axis=1)

corr_matrix = temp_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="viridis")
plt.title(
    "Coorelation Heatmap - Continious Variables",
    fontdict={"fontsize": 14, "fontweight": "bold"},
)
plt.show()



## === cell 14
cat_sum = data_features[categorical_cols].sum(axis=1)
cat_sum_val = val_data[categorical_cols].sum(axis=1)

data_features["Cat_Sum"] = cat_sum
val_data["Cat_Sum"] = cat_sum_val

data_features.drop(columns=categorical_cols, inplace=True)
val_data.drop(columns=categorical_cols, inplace=True)



## === cell 15
data_features["Cat_Sum"].unique()



## === cell 16
train_ids = data_features["Id"].values
test_ids = val_data["Id"].values

data_features_model = data_features.drop(columns=["Id"])
val_data_model = val_data.drop(columns=["Id"])

data_features_model, val_data_model = data_features_model.align(
    val_data_model, join="inner", axis=1
)

scaler = StandardScaler()
scaled_data_features = scaler.fit_transform(data_features_model)
scaled_val_data = scaler.transform(val_data_model)



## === cell 17
vc = data_target.value_counts()
too_small = vc[vc < 2].index.tolist()
if len(too_small) > 0:
    mask = ~data_target.isin(too_small)
    scaled_data_features = scaled_data_features[mask.values]
    data_target = data_target.loc[mask].copy()

X_train, X_test, y_train, y_test = train_test_split(
    scaled_data_features,
    data_target,
    test_size=0.2,
    random_state=42,
    stratify=data_target,
)



## === cell 18
catboostclassifier = CatBoostClassifier(silent=True, task_type="CPU", random_seed=42)

catboostclassifier.fit(X_train, y_train, verbose=False)

y_pred = catboostclassifier.predict(X_test)

print(classification_report(y_test, y_pred))



## === cell 19
sample_submission = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)
sample_submission.head()



## === cell 20
pred = catboostclassifier.predict(scaled_val_data)
pred = np.asarray(pred).reshape(-1).astype(int)

pred = np.clip(pred, 1, 7)

submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": pred})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Cover_Type value counts (submission):")
print(submission_df["Cover_Type"].value_counts().sort_index())

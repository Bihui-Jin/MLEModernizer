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
missingno==0.5.2
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn")
sns.set(font_scale=2.5)



## === cell 1
path = "../input/tabular-playground-series-dec-2021/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")



## === cell 2
print(train.shape)
train.head()



## === cell 3
print(train.describe())



## === cell 4
print(f"# of train data : {train.shape[0]}")
print(f"# of train features : {train.shape[1] - 1}")
print("")
print("=" * 15, " >> Null Data << ", "=" * 15)
null_feature = []
for col in train.columns:
    msg = (
        "column: {:>35}\t {:>10d} of {:<10d} ( Percent of Null value: {:.2f}% )".format(
            col,
            train[col].isnull().sum(),
            train[col].shape[0],
            100 * (train[col].isnull().sum() / train[col].shape[0]),
        )
    )
    print(msg)
    if train[col].isnull().sum() != 0:
        null_feature.append(col)

if len(null_feature) != 0:
    print("")
    print("=" * 15, " >> Warning << ", "=" * 15)
    print("NULL Feature : ", null_feature)



## === cell 5
print("=" * 15, " >> Unique Data << ", "=" * 15)
unique_col = []
for col in train.columns:
    msg = "column: {:>35}\t {:>10d}".format(col, len(train[col].unique()))
    print(msg)
    if len(train[col].unique()) == 1:
        unique_col.append(col)

if len(unique_col) != 0:
    print("")
    print("=" * 15, " >> Warning << ", "=" * 15)
    print("Unique Feature : ", unique_col)



## === cell 6
f, ax = plt.subplots(1, 2, figsize=(20, 12))
train["Cover_Type"].value_counts().plot.pie(autopct="%1.4f%%", ax=ax[0], shadow=True)
ax[0].set_title("Pie plot - Cover_Type", fontsize=16)
ax[0].set_ylabel("")
ax[0].tick_params(axis="both", labelsize=14)

ax[1].set_title("Count plot - Cover_Type", fontsize=16)
sns.countplot(x="Cover_Type", data=train, ax=ax[1])  # fixed argument order
ax[1].set_ylabel("count", fontsize=14)
ax[1].set_xlabel("Cover_Type", fontsize=14)
ax[1].tick_params(axis="both", labelsize=14)

plt.show()



## === cell 7
print("=" * 15, " >> Target Data << ", "=" * 15)
num_target = train["Cover_Type"].value_counts()
targets = np.sort(train["Cover_Type"].unique())
for target in targets:
    msg = "target: {:>3}\t {:>7d} of {:<10d} ( Percent of Null value: {:.2f}% )".format(
        target,
        num_target[target],
        train.shape[0],
        100 * (num_target[target] / train.shape[0]),
    )
    print(msg)




## === cell 8
def EngineerFunction(df, is_train=True):
    df = df.drop(["Id"], axis=1)

    df["Pythagorian_Distance_To_Hydrology"] = np.hypot(
        df["Horizontal_Distance_To_Hydrology"], df["Vertical_Distance_To_Hydrology"]
    )

    unique_cols = ["Soil_Type7", "Soil_Type15"]
    df = df.drop(unique_cols, axis=1)

    df.loc[df["Aspect"] < 0, "Aspect"] += 360
    df.loc[df["Aspect"] > 359, "Aspect"] -= 360

    for col in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]:
        df.loc[df[col] < 0, col] = 0
        df.loc[df[col] > 255, col] = 255

    return df


train = EngineerFunction(train, is_train=True)
test = EngineerFunction(test, is_train=False)



## === cell 9
X_train = train.drop("Cover_Type", axis=1)
y_train = train["Cover_Type"]
X_test = test



## === cell 10
from sklearn.preprocessing import RobustScaler

RS = RobustScaler().fit(X_train)
X_train = RS.transform(X_train)
X_test = RS.transform(X_test)



## === cell 11
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=21
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)

rf.fit(X_tr, y_tr)

val_pred = rf.predict(X_val)
print("Validation accuracy :", accuracy_score(y_val, val_pred))



## === cell 12
rf.fit(X_train, y_train)



## === cell 13
submission = pd.read_csv(path + "sample_submission.csv")



## === cell 14
pred_test = rf.predict(X_test)
submission["Cover_Type"] = pred_test



## === cell 15
submission.head()



## === cell 16
submission.to_csv("submission.csv", index=False)

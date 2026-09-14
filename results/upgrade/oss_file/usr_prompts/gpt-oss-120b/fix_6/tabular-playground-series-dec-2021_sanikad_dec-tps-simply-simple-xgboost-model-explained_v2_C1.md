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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df




## === cell 1
train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)




## === cell 2
train.drop("Id", axis=1, inplace=True)




## === cell 3
plt.figure(figsize=(10, 8))
plt.title("TARGET ANALYSIS: Cover_Type")
train["Cover_Type"].value_counts(normalize=True).plot.bar(color="green")
plt.show()




## === cell 4
soil_types = [col for col in train.columns if col.startswith("Soil")]
wilderness = [col for col in train.columns if col.startswith("Wilderness")]




## === cell 5
for col in soil_types:
    if train[col].value_counts()[0] / len(train) * 100 >= 90:
        print(col, train[col].value_counts()[0] / len(train) * 100)




## === cell 6
for col in wilderness:
    if train[col].value_counts()[0] / len(train) * 100 >= 90:
        print(col, train[col].value_counts()[0] / len(train) * 100)




## === cell 7
train["soil_type"] = train[soil_types].sum(axis=1)
test["soil_type"] = test[soil_types].sum(axis=1)




## === cell 8
train.groupby(["soil_type"])["Cover_Type"].agg(pd.Series.mode)




## === cell 9
cols_drop = []
cols_drop.extend(soil_types)




## === cell 10
train["wildness"] = train[wilderness].sum(axis=1)
test["wildness"] = test[wilderness].sum(axis=1)




## === cell 11
cols_drop.extend(wilderness)




## === cell 12
plt.figure()
fig, ax = plt.subplots(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title("Wildness Train")
train["wildness"].value_counts(normalize=True).plot.bar(color="red")

plt.subplot(2, 2, 2)
plt.title("Wildness Test")
test["wildness"].value_counts(normalize=True).plot.bar()

plt.show()




## === cell 13
plt.figure()
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.title("Soil Type Train")
train["soil_type"].value_counts(normalize=True).plot.bar(color="red")

plt.subplot(2, 2, 2)
plt.title("Soil Type Test")
test["soil_type"].value_counts(normalize=True).plot.bar()

plt.show()




## === cell 14
train["Aspect"][train["Aspect"] < 0] += 360
train["Aspect"][train["Aspect"] > 359] -= 360

test["Aspect"][test["Aspect"] < 0] += 360
test["Aspect"][test["Aspect"] > 359] -= 360




## === cell 15
train.loc[train["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
train.loc[train["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
train.loc[train["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
train.loc[train["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
train.loc[train["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
train.loc[train["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255

test.loc[test["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
test.loc[test["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
test.loc[test["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
test.loc[test["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
test.loc[test["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
test.loc[test["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255




## === cell 16
plt.figure()
fig, ax = plt.subplots(figsize=(15, 15))
plt.subplot(3, 3, 1)
sns.boxplot(x=train["Cover_Type"], y=train["Aspect"])
plt.title("Aspect vs Cover Type")

plt.subplot(3, 3, 2)
sns.boxplot(x=train["Cover_Type"], y=train["Elevation"])
plt.title("Elevation vs Cover Type")

plt.subplot(3, 3, 3)
sns.boxplot(x=train["Cover_Type"], y=train["Slope"])
plt.title("Slope vs Cover Type")

plt.subplot(3, 3, 4)
sns.boxplot(x=train["Cover_Type"], y=train["Horizontal_Distance_To_Hydrology"])
plt.title("Horizontal_Distance_To_Hydrology vs Cover Type")

plt.subplot(3, 3, 5)
sns.boxplot(x=train["Cover_Type"], y=train["Vertical_Distance_To_Hydrology"])
plt.title("Vertical_Distance_To_Hydrology vs Cover Type")

plt.subplot(3, 3, 6)
sns.boxplot(x=train["Cover_Type"], y=train["Horizontal_Distance_To_Roadways"])
plt.title("Horizontal_Distance_To_Roadways vs Cover Type")

plt.subplot(3, 3, 7)
sns.boxplot(x=train["Cover_Type"], y=train["Hillshade_9am"])
plt.title("Hillshade_9am vs Cover Type")

plt.subplot(3, 3, 8)
sns.boxplot(x=train["Cover_Type"], y=train["Hillshade_Noon"])
plt.title("Hillshade_Noon vs Cover Type")

plt.subplot(3, 3, 9)
sns.boxplot(x=train["Cover_Type"], y=train["Hillshade_3pm"])
plt.title("Hillshade_3pm vs Cover Type")

plt.show()




## === cell 17
train["Hydrological_Distance"] = (
    train["Horizontal_Distance_To_Hydrology"] ** 2
    + train["Vertical_Distance_To_Hydrology"] ** 2
) ** 0.5
test["Hydrological_Distance"] = (
    test["Horizontal_Distance_To_Hydrology"] ** 2
    + test["Vertical_Distance_To_Hydrology"] ** 2
) ** 0.5




## === cell 18
cols_drop.extend(["Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology"])




## === cell 19
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




## === cell 20
X = train.drop("Cover_Type", axis=1)
Y = train["Cover_Type"]

X_train = X.copy()
X_valid = X.copy()
Y_train_raw = Y.copy()
Y_valid_raw = Y.copy()
Y_train = Y_train_raw - 1  # encode 0‑6
Y_valid = Y_valid_raw - 1




## === cell 21
xgb_without_fe = XGBClassifier(
    n_estimators=400,
    n_jobs=-1,
    booster="gbtree",
    tree_method="hist",
    objective="multi:softprob",
    eval_metric="mlogloss",
    use_label_encoder=False,
    num_class=7,
)

xgb_without_fe.fit(X_train, Y_train)
pred_valid_enc = xgb_without_fe.predict(X_valid)
pred_valid = pred_valid_enc + 1
acc_score_without_fe = accuracy_score(Y_valid_raw, pred_valid)
print("Accuracy without engineered feature drop:", acc_score_without_fe)




## === cell 22
X_train_drop = X_train.drop(cols_drop, axis=1)
X_valid_drop = X_valid.drop(cols_drop, axis=1)

xgb_with_fe = XGBClassifier(
    n_estimators=400,
    n_jobs=-1,
    booster="gbtree",
    tree_method="hist",
    objective="multi:softprob",
    eval_metric="mlogloss",
    use_label_encoder=False,
    num_class=7,
)

xgb_with_fe.fit(X_train_drop, Y_train)
pred_valid_enc_fe = xgb_with_fe.predict(X_valid_drop)
pred_valid_fe = pred_valid_enc_fe + 1
acc_score_with_fe = accuracy_score(Y_valid_raw, pred_valid_fe)
print("Accuracy with engineered feature drop:", acc_score_with_fe)




## === cell 23
compare = pd.DataFrame(
    {"With_Fe": [acc_score_with_fe], "Without_Fe": [acc_score_without_fe]}
)
print(compare)




## === cell 24
from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(xgb_without_fe, max_num_features=10)
plt.show()




## === cell 25
test_id = test["Id"].copy()
test.drop("Id", axis=1, inplace=True)




## === cell 26
test = test[X_train.columns]  # ensure same column order as training




## === cell 27
test_pred_enc = xgb_without_fe.predict(test)
test_pred = test_pred_enc + 1
submission = pd.DataFrame({"Id": test_id, "Cover_Type": test_pred})




## === cell 28
submission.head()




## === cell 29
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")




## === cell 30
print(pd.read_csv("submission.csv").head())

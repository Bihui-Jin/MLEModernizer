# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.95353

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")



## === cell 1
train = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/test.csv")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1384002722.py in <cell line: 0>()
----> 1 train = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/train.csv")
      2 test = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/test.csv")
      3 

NameError: name 'pd' is not defined

## === cell 2
train.drop("Id", axis=1, inplace=True)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1049355978.py in <cell line: 0>()
----> 1 train.drop("Id", axis=1, inplace=True)
      2 

NameError: name 'train' is not defined

## === cell 3
train.memory_usage().sum() / 1024**2




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3306084597.py in <cell line: 0>()
----> 1 train.memory_usage().sum() / 1024**2
      2 
      3 

NameError: name 'train' is not defined

## === cell 4
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




## === cell 5
train = reduce_mem_usage(train)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3256657337.py in <cell line: 0>()
----> 1 train = reduce_mem_usage(train)
      2 

NameError: name 'train' is not defined

## === cell 6
plt.figure(figsize=(10, 8))
plt.title("TARGET ANALYSIS:Cover_Type")
train["Cover_Type"].value_counts(normalize=True).plot.bar(color="green")
plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1090001315.py in <cell line: 0>()
      1 plt.figure(figsize=(10, 8))
      2 plt.title("TARGET ANALYSIS:Cover_Type")
----> 3 train["Cover_Type"].value_counts(normalize=True).plot.bar(color="green")
      4 plt.show()
      5 

NameError: name 'train' is not defined

## === cell 7
soil_types = [col for col in train.columns if col.startswith("Soil")]
wilderness = [col for col in train.columns if col.startswith("Wilderness")]



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3508784169.py in <cell line: 0>()
----> 1 soil_types = [col for col in train.columns if col.startswith("Soil")]
      2 wilderness = [col for col in train.columns if col.startswith("Wilderness")]
      3 

NameError: name 'train' is not defined

## === cell 8
for col in soil_types:
    if train[col].value_counts()[0] / len(train) * 100 >= 90:
        print(col, train[col].value_counts()[0] / len(train) * 100)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2352054410.py in <cell line: 0>()
----> 1 for col in soil_types:
      2     if train[col].value_counts()[0] / len(train) * 100 >= 90:
      3         print(col, train[col].value_counts()[0] / len(train) * 100)
      4 

NameError: name 'soil_types' is not defined

## === cell 9
for col in wilderness:
    if train[col].value_counts()[0] / len(train) * 100 >= 90:
        print(col, train[col].value_counts()[0] / len(train) * 100)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/509253046.py in <cell line: 0>()
----> 1 for col in wilderness:
      2     if train[col].value_counts()[0] / len(train) * 100 >= 90:
      3         print(col, train[col].value_counts()[0] / len(train) * 100)
      4 

NameError: name 'wilderness' is not defined

## === cell 10
train["soil_type"] = train[soil_types].sum(axis=1)
test["soil_type"] = test[soil_types].sum(axis=1)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1379263444.py in <cell line: 0>()
----> 1 train["soil_type"] = train[soil_types].sum(axis=1)
      2 test["soil_type"] = test[soil_types].sum(axis=1)
      3 

NameError: name 'train' is not defined

## === cell 11
train.groupby(["soil_type"])["Cover_Type"].agg(pd.Series.mode)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4261480414.py in <cell line: 0>()
----> 1 train.groupby(["soil_type"])["Cover_Type"].agg(pd.Series.mode)
      2 

NameError: name 'train' is not defined

## === cell 12
cols_drop = []
cols_drop = soil_types



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2004455338.py in <cell line: 0>()
      1 cols_drop = []
----> 2 cols_drop = soil_types
      3 

NameError: name 'soil_types' is not defined

## === cell 13
train["wildness"] = train[wilderness].sum(axis=1)
test["wildness"] = test[wilderness].sum(axis=1)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/834861934.py in <cell line: 0>()
----> 1 train["wildness"] = train[wilderness].sum(axis=1)
      2 test["wildness"] = test[wilderness].sum(axis=1)
      3 

NameError: name 'train' is not defined

## === cell 14
cols_drop.extend(wilderness)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/406912017.py in <cell line: 0>()
----> 1 cols_drop.extend(wilderness)
      2 

NameError: name 'wilderness' is not defined

## === cell 15
plt.figure()
fig, ax = plt.subplots(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title("Wildness_Train")
train["wildness"].value_counts(normalize=True).plot.bar(color="red")

plt.subplot(2, 2, 2)
plt.title("Wildness_Test")
test["wildness"].value_counts(normalize=True).plot.bar()

plt.show()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2002302383.py in <cell line: 0>()
      4 plt.subplot(2, 2, 1)
      5 plt.title("Wildness_Train")
----> 6 train["wildness"].value_counts(normalize=True).plot.bar(color="red")
      7 
      8 plt.subplot(2, 2, 2)

NameError: name 'train' is not defined

## === cell 16
plt.figure()
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.title("Soil Type Train")
train["soil_type"].value_counts(normalize=True).plot.bar(color="red")

plt.subplot(2, 2, 2)
plt.title("Soil Type Test")
test["soil_type"].value_counts(normalize=True).plot.bar()

plt.show()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1452179996.py in <cell line: 0>()
      3 plt.subplot(2, 2, 1)
      4 plt.title("Soil Type Train")
----> 5 train["soil_type"].value_counts(normalize=True).plot.bar(color="red")
      6 
      7 plt.subplot(2, 2, 2)

NameError: name 'train' is not defined

## === cell 17
train.Aspect.describe()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2019336347.py in <cell line: 0>()
----> 1 train.Aspect.describe()
      2 

NameError: name 'train' is not defined

## === cell 18
train["Aspect"][train["Aspect"] < 0] += 360
train["Aspect"][train["Aspect"] > 359] -= 360

test["Aspect"][test["Aspect"] < 0] += 360
test["Aspect"][test["Aspect"] > 359] -= 360



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4214443056.py in <cell line: 0>()
----> 1 train["Aspect"][train["Aspect"] < 0] += 360
      2 train["Aspect"][train["Aspect"] > 359] -= 360
      3 
      4 test["Aspect"][test["Aspect"] < 0] += 360
      5 test["Aspect"][test["Aspect"] > 359] -= 360

NameError: name 'train' is not defined

## === cell 19
train[["Hillshade_Noon", "Hillshade_9am", "Hillshade_3pm"]].describe()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3987977041.py in <cell line: 0>()
----> 1 train[["Hillshade_Noon", "Hillshade_9am", "Hillshade_3pm"]].describe()
      2 

NameError: name 'train' is not defined

## === cell 20
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



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3542406675.py in <cell line: 0>()
----> 1 train.loc[train["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
      2 train.loc[train["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
      3 train.loc[train["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
      4 train.loc[train["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
      5 train.loc[train["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255

NameError: name 'train' is not defined

## === cell 21
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



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1746538351.py in <cell line: 0>()
      2 fig, ax = plt.subplots(figsize=(15, 15))
      3 plt.subplot(3, 3, 1)
----> 4 sns.boxplot(x=train["Cover_Type"], y=train["Aspect"])
      5 plt.title("Aspect vs Cover Type")
      6 

NameError: name 'train' is not defined

## === cell 22
train[
    [
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
].describe()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4075518547.py in <cell line: 0>()
----> 1 train[
      2     [
      3         "Horizontal_Distance_To_Hydrology",
      4         "Vertical_Distance_To_Hydrology",
      5         "Horizontal_Distance_To_Roadways",

NameError: name 'train' is not defined

## === cell 23
train["Hydrological_Distance"] = (
    train["Horizontal_Distance_To_Hydrology"] ** 2
    + train["Vertical_Distance_To_Hydrology"] ** 2
) ** 0.5
test["Hydrological_Distance"] = (
    test["Horizontal_Distance_To_Hydrology"] ** 2
    + test["Vertical_Distance_To_Hydrology"] ** 2
) ** 0.5



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1277348501.py in <cell line: 0>()
      1 train["Hydrological_Distance"] = (
----> 2     train["Horizontal_Distance_To_Hydrology"] ** 2
      3     + train["Vertical_Distance_To_Hydrology"] ** 2
      4 ) ** 0.5
      5 test["Hydrological_Distance"] = (

NameError: name 'train' is not defined

## === cell 24
cols_drop.extend(["Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology"])



## === cell 25
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



## === cell 26
X = train.drop("Cover_Type", axis=1)
Y = train["Cover_Type"]



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/800050020.py in <cell line: 0>()
----> 1 X = train.drop("Cover_Type", axis=1)
      2 Y = train["Cover_Type"]
      3 

NameError: name 'train' is not defined

## === cell 27
X_train, X_valid, Y_train_raw, Y_valid_raw = train_test_split(
    X, Y, test_size=0.30, random_state=42, shuffle=True, stratify=Y
)
Y_train = Y_train_raw - 1  # encode 0‑6
Y_valid = Y_valid_raw - 1



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3415328400.py in <cell line: 0>()
      1 # Use stratified split to keep every class in the training fold
      2 X_train, X_valid, Y_train_raw, Y_valid_raw = train_test_split(
----> 3     X, Y, test_size=0.30, random_state=42, shuffle=True, stratify=Y
      4 )
      5 Y_train = Y_train_raw - 1  # encode 0‑6

NameError: name 'X' is not defined

## === cell 28
xgb_without_fe = XGBClassifier(
    n_estimators=200,
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
pred_valid = pred_valid_enc + 1  # decode back to original classes
acc_score_without_fe = accuracy_score(Y_valid_raw, pred_valid)
acc_score_without_fe



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1781825615.py in <cell line: 0>()
     10 )
     11 
---> 12 xgb_without_fe.fit(X_train, Y_train)
     13 pred_valid_enc = xgb_without_fe.predict(X_valid)
     14 pred_valid = pred_valid_enc + 1  # decode back to original classes

NameError: name 'X_train' is not defined

## === cell 29
X_train_drop = X_train.drop(cols_drop, axis=1)
X_valid_drop = X_valid.drop(cols_drop, axis=1)

xgb_with_fe = XGBClassifier(
    n_estimators=200,
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
acc_score_with_fe



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/891667655.py in <cell line: 0>()
----> 1 X_train_drop = X_train.drop(cols_drop, axis=1)
      2 X_valid_drop = X_valid.drop(cols_drop, axis=1)
      3 
      4 xgb_with_fe = XGBClassifier(
      5     n_estimators=200,

NameError: name 'X_train' is not defined

## === cell 30
compare = pd.DataFrame(
    {"With_Fe": [acc_score_with_fe], "Without_Fe": [acc_score_without_fe]}
)
compare



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1795934036.py in <cell line: 0>()
----> 1 compare = pd.DataFrame(
      2     {"With_Fe": [acc_score_with_fe], "Without_Fe": [acc_score_without_fe]}
      3 )
      4 compare
      5 

NameError: name 'pd' is not defined

## === cell 31
from xgboost import plot_importance
import matplotlib.pyplot as plt



## === cell 32
plot_importance(xgb_without_fe, max_num_features=10)
plt.show()



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2351305401.py in <cell line: 0>()
----> 1 plot_importance(xgb_without_fe, max_num_features=10)
      2 plt.show()
      3 

/usr/local/lib/python3.11/dist-packages/xgboost/plotting.py in plot_importance(booster, ax, height, xlim, ylim, title, xlabel, ylabel, fmap, importance_type, max_num_features, grid, show_values, values_format, **kwargs)
     86 
     87     if isinstance(booster, XGBModel):
---> 88         importance = booster.get_booster().get_score(
     89             importance_type=importance_type, fmap=fmap
     90         )

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 33
test_id = test["Id"].copy()
test.drop("Id", axis=1, inplace=True)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2936506480.py in <cell line: 0>()
----> 1 test_id = test["Id"].copy()
      2 test.drop("Id", axis=1, inplace=True)
      3 

NameError: name 'test' is not defined

## === cell 34
test = test[X_train.columns]



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/817953348.py in <cell line: 0>()
----> 1 test = test[X_train.columns]
      2 

NameError: name 'test' is not defined

## === cell 35
test_pred_enc = xgb_without_fe.predict(test)
test_pred = test_pred_enc + 1
submission = pd.DataFrame({"Id": test_id, "Cover_Type": test_pred})



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3553669973.py in <cell line: 0>()
----> 1 test_pred_enc = xgb_without_fe.predict(test)
      2 test_pred = test_pred_enc + 1
      3 submission = pd.DataFrame({"Id": test_id, "Cover_Type": test_pred})
      4 

NameError: name 'test' is not defined

## === cell 36
submission.head()



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2023294942.py in <cell line: 0>()
----> 1 submission.head()
      2 

NameError: name 'submission' is not defined

## === cell 37
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/799702916.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 

NameError: name 'submission' is not defined

## === cell 38
pd.read_csv("submission.csv").head()

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/533439942.py in <cell line: 0>()
----> 1 pd.read_csv("submission.csv").head()

NameError: name 'pd' is not defined

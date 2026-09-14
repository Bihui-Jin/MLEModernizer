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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier




## === cell 1
train_df = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test_df = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")




## === cell 2
data = []

for f in train_df.columns:
    if f == "Cover_Type":  # fixed typo
        role = "target"
    elif f == "Id":  # correct id column name
        role = "id"
    else:
        role = "input"

    if "Type" in f or "Area" in f or f == "Cover_Type" or f == "Id":
        level = "nominal"
    elif "cat" in f or f == "Id":
        level = "nominal"
    elif train_df[f].dtype == float:
        level = "interval"
    elif train_df[f].dtype == int:
        level = "ordinal"
    else:
        level = "nominal"

    keep = True
    if f == "Id":  # we do not keep Id for modelling
        keep = False

    dtype = train_df[f].dtype

    f_dict = {"varname": f, "role": role, "level": level, "keep": keep, "dtype": dtype}
    data.append(f_dict)

meta = pd.DataFrame(data, columns=["varname", "role", "level", "keep", "dtype"])
meta.set_index("varname", inplace=True)




## === cell 3
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
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df




## === cell 4
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)




## === cell 5
train_df.info()




## === cell 6
v = train_df.columns
for f in v:
    dist_value = train_df[f].value_counts().shape[0]
    print("Variables {:>40} has {} distinct values".format(f, dist_value))




## === cell 7
v = test_df.columns
for f in v:
    dist_value = test_df[f].value_counts().shape[0]
    print("Variables {:>40} has {} distinct values".format(f, dist_value))




## === cell 8
missing = 0
for f in train_df.columns:
    missing += train_df[f].isnull().sum()
    print("Variables : {:>30}\t missings : {}".format(f, train_df[f].isnull().sum()))
print("Sum of missing_value : {}".format(missing))




## === cell 9
v = meta[(meta.level == "nominal") & meta.keep].index
train_df[v].describe()




## === cell 10
for i in v:
    print(i)




## === cell 11
v = meta[(meta.level == "ordinal") & meta.keep].index
for i in v:
    print(i)




## === cell 12
s1 = train_df.sample(frac=0.2, random_state=42)
s2 = test_df.sample(frac=0.2, random_state=42)




## === cell 13
i = 1
plt.figure()
fig, ax = plt.subplots(2, 5, figsize=(20, 12))
for f in v:
    plt.subplot(2, 5, i)
    sns.histplot(s1[f], color="blue", kde=True, bins=100, label="train_" + f)
    sns.histplot(s2[f], color="olive", kde=True, bins=100, label="test_" + f)
    plt.xlabel(f, fontsize=9)
    plt.legend()
    i += 1
plt.show()




## === cell 14
def corr_heatmap(v):
    correlations = train_df[v].corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, ax = plt.subplots(figsize=(30, 10))
    sns.heatmap(
        correlations,
        cmap=cmap,
        vmax=1.0,
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        annot=True,
        cbar_kws={"shrink": 0.75},
    )
    plt.show()


corr_heatmap(v)




## === cell 15
train_df["Cover_Type"].value_counts()




## === cell 16
sns.catplot(x="Cover_Type", kind="count", palette="ch:.25", data=train_df)




## === cell 17
test_df.columns




## === cell 18
target = train_df["Cover_Type"]
train_df.drop(
    columns=["Id", "Cover_Type", "Soil_Type7", "Soil_Type15"], axis=1, inplace=True
)
test_df.drop(columns=["Id", "Soil_Type7", "Soil_Type15"], axis=1, inplace=True)




## === cell 19
def polynomial(df):
    v = [
        col
        for col in meta[(meta.level == "ordinal") & (meta.keep)].index
        if col in df.columns
    ]
    if not v:
        return df.copy()
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    transformed = poly.fit_transform(df[v])
    new_cols = poly.get_feature_names_out(v)
    interactions = pd.DataFrame(data=transformed, columns=new_cols, index=df.index)
    interactions.drop(columns=v, inplace=True)
    print(
        "Before creating interactions we have {} variables in train".format(df.shape[1])
    )
    df = pd.concat([df, interactions], axis=1)
    print(
        "After creating interactions we have {} variables in train".format(df.shape[1])
    )
    return df




## === cell 20
scaler = StandardScaler()




## === cell 21
X_poly = polynomial(train_df)
X_scaled = scaler.fit_transform(X_poly)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, target, test_size=0.2, shuffle=True, random_state=42
)




## === cell 22
model = CatBoostClassifier(thread_count=4, verbose=False)  # CPU mode for compatibility
model.fit(X_train, y_train)




## === cell 23
print("Accuracy Score : ", accuracy_score(y_val, model.predict(X_val)))




## === cell 24
submission = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)
test_poly = polynomial(test_df)
test_scaled = scaler.transform(test_poly)
submission["Cover_Type"] = model.predict(test_scaled)
submission.to_csv("submission.csv", index=False)

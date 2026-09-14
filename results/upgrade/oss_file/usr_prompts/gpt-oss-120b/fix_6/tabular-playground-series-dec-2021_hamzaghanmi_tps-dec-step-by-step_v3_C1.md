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
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import gc
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from lightgbm import LGBMClassifier, plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import Parallel, delayed




## === cell 1
train = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/test.csv")

cols = [e for e in test.columns if e != "Id"]
continous_features = cols[:10]
categorical_features = cols[10:]




## === cell 2
train.head()




## === cell 3
test.head()




## === cell 4
train.info()




## === cell 5
test.info()




## === cell 6
train.isnull().sum()




## === cell 7
test.isnull().sum()




## === cell 8
train[continous_features].describe()




## === cell 9
test[continous_features].describe()




## === cell 10
i = 1
plt.figure()
fig, ax = plt.subplots(2, 5, figsize=(20, 12))
for feature in continous_features:
    plt.subplot(2, 5, i)
    sns.histplot(
        train[feature], color="blue", kde=True, bins=100, label="train_" + feature
    )
    sns.histplot(
        test[feature], color="olive", kde=True, bins=100, label="test_" + feature
    )
    plt.xlabel(feature, fontsize=9)
    plt.legend()
    i += 1
plt.show()




## === cell 11
"""# plot the categorical features 
i = 1
plt.figure()
fig, ax = plt.subplots(11, 4, figsize=(20, 22))
for feature in categorical_features:
    plt.subplot(11, 4, i)
    sns.catplot(x=feature, color="blue", data=train, label='train_' + feature)
    sns.catplot(x=feature, color="olive", data=test, label='test_' + feature)
    plt.xlabel(feature, fontsize=9)
    plt.legend()
    i += 1
plt.show()"""




## === cell 12
sns.catplot(x="Cover_Type", kind="count", palette="ch:.25", data=train)




## === cell 13
train["Cover_Type"].value_counts()




## === cell 14
corr = train[continous_features + ["Cover_Type"]].corr()
corr.style.background_gradient(cmap="coolwarm").format(precision=3)




## === cell 15
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_memory = df.memory_usage().sum() / 1024**2
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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
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
    end_memory = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe after reduction {end_memory} MB")
        print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
    return df




## === cell 16
train[cols] = reduce_mem_usage(train[cols])




## === cell 17
test[cols] = reduce_mem_usage(test[cols])




## === cell 18
train.drop(train[train["Cover_Type"] == 5].index, inplace=True)




## === cell 19
train["mean"] = train[continous_features].mean(axis=1)
train["min"] = train[continous_features].min(axis=1)
train["max"] = train[continous_features].max(axis=1)

test["mean"] = test[continous_features].mean(axis=1)
test["min"] = test[continous_features].min(axis=1)
test["max"] = test[continous_features].max(axis=1)

cols = [e for e in test.columns if e != "Id"]




## === cell 20
params = {
    "objective": "multiclass",
    "random_state": 48,
    "n_estimators": 20000,
    "n_jobs": 5,  # allow LightGBM to use threads
    "reg_alpha": 0.0010309124257626384,
    "reg_lambda": 9.48149567512538,
    "colsample_bytree": 0.5,
    "subsample": 1,
    "learning_rate": 0.2,
    "max_depth": 100,
    "num_leaves": 142,
    "min_child_samples": 204,
    "cat_smooth": 99,
    "verbose": -1,  # suppress LightGBM printing
}




## === cell 21
X = train[cols].values
y = train["Cover_Type"].values
X_test = test[cols].values


def _train_fold(fold_idx, trn_idx, val_idx):
    """Train a single LightGBM fold using pre‑converted NumPy arrays."""
    X_tr, X_val = X[trn_idx], X[val_idx]
    y_tr, y_val = y[trn_idx], y[val_idx]

    model = LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

    pred_test = model.predict(X_test)
    acc = accuracy_score(y_val, model.predict(X_val))
    return (fold_idx, pred_test, acc, model)


kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)

results = Parallel(n_jobs=5, backend="threading")(
    delayed(_train_fold)(i, trn_idx, val_idx)
    for i, (trn_idx, val_idx) in enumerate(kf.split(X, y))
)

results.sort(key=lambda x: x[0])
preds = [r[1] for r in results]
acc = [r[2] for r in results]
model = results[-1][3]

for i, a in enumerate(acc):
    print(f"fold: {i+1} , accuracy: {round(a*100, 3)}%")




## === cell 22
print(f"The mean Accuracy is : {round(np.mean(acc)*100, 3)}%")




## === cell 23
plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()




## === cell 24
preds_array = np.vstack(preds)  # shape: (n_folds, n_test)
mode_result = stats.mode(preds_array, axis=0)
prediction = mode_result.mode[0].astype(int)  # final prediction per test row

sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)
sub["Cover_Type"] = prediction
sub.to_csv("submission.csv", index=False)




## === cell 25
sub.head()

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
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from scipy import stats
import matplotlib.pyplot as plt



## === cell 1
int_cols = [
    "Id",
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4",
    "Soil_Type1",
    "Soil_Type2",
    "Soil_Type3",
    "Soil_Type4",
    "Soil_Type5",
    "Soil_Type6",
    "Soil_Type7",
    "Soil_Type8",
    "Soil_Type9",
    "Soil_Type10",
    "Soil_Type11",
    "Soil_Type12",
    "Soil_Type13",
    "Soil_Type14",
    "Soil_Type15",
    "Soil_Type16",
    "Soil_Type17",
    "Soil_Type18",
    "Soil_Type19",
    "Soil_Type20",
    "Soil_Type21",
    "Soil_Type22",
    "Soil_Type23",
    "Soil_Type24",
    "Soil_Type25",
    "Soil_Type26",
    "Soil_Type27",
    "Soil_Type28",
    "Soil_Type29",
    "Soil_Type30",
    "Soil_Type31",
    "Soil_Type32",
    "Soil_Type33",
    "Soil_Type34",
    "Soil_Type35",
    "Soil_Type36",
    "Soil_Type37",
    "Soil_Type38",
    "Soil_Type39",
    "Soil_Type40",
    "Soil_Type41",
    "Soil_Type42",
    "Soil_Type43",
    "Soil_Type44",
    "Soil_Type45",
    "Soil_Type46",
    "Soil_Type47",
    "Soil_Type48",
    "Soil_Type49",
    "Soil_Type50",
    "Soil_Type51",
    "Soil_Type52",
    "Soil_Type53",
    "Soil_Type54",
    "Soil_Type55",
    "Soil_Type56",
    "Soil_Type57",
    "Soil_Type58",
    "Soil_Type59",
    "Soil_Type60",
]
float_cols = [
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

dtype_dict = {c: np.int8 for c in int_cols}
dtype_dict.update({c: np.float32 for c in float_cols})

train = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/train.csv", dtype=dtype_dict
)
test = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/test.csv", dtype=dtype_dict
)

cols = [e for e in test.columns if e != "Id"]
continous_features = cols[:10]
categorical_features = cols[10:]



## === cell 2
pass



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass




## === cell 14
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
    end_memory = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Memory usage after reduction: {end_memory:.2f} MB "
            f"({100 * (start_memory - end_memory) / start_memory:.2f}% reduced)"
        )
    return df




## === cell 15
train[cols] = reduce_mem_usage(train[cols])



## === cell 16
test[cols] = reduce_mem_usage(test[cols])



## === cell 17
train.drop(train[train["Cover_Type"] == 5].index, inplace=True)



## === cell 18
train["binned_elevation"] = (train["Elevation"] // 50).astype(np.int16)
test["binned_elevation"] = (test["Elevation"] // 50).astype(np.int16)

train["Horizontal_Distance_To_Roadways_Log"] = np.log(
    train["Horizontal_Distance_To_Roadways"] + 300
)
test["Horizontal_Distance_To_Roadways_Log"] = np.log(
    test["Horizontal_Distance_To_Roadways"] + 300
)

train["Soil_Type12_32"] = train["Soil_Type32"] + train["Soil_Type12"]
test["Soil_Type12_32"] = test["Soil_Type32"] + test["Soil_Type12"]
train["Soil_Type23_22_32_33"] = (
    train["Soil_Type23"]
    + train["Soil_Type22"]
    + train["Soil_Type32"]
    + train["Soil_Type33"]
)
test["Soil_Type23_22_32_33"] = (
    test["Soil_Type23"]
    + test["Soil_Type22"]
    + test["Soil_Type32"]
    + test["Soil_Type33"]
)

cols = [e for e in test.columns if e != "Id"]



## === cell 19
scaler = StandardScaler()
train[cols] = scaler.fit_transform(train[cols]).astype(np.float32)
test[cols] = scaler.transform(test[cols]).astype(np.float32)



## === cell 20
params = {
    "objective": "multiclass",
    "random_state": 48,
    "n_estimators": 2000,  # kept as‑is
    "n_jobs": 1,  # each thread will manage its own LightGBM threads (set to 1 to avoid oversubscription)
    "reg_alpha": 0.9481920810028138,
    "reg_lambda": 8.15049828410672,
    "colsample_bytree": 0.5,
    "subsample": 0.8,
    "learning_rate": 0.2,
    "max_depth": 100,
    "num_leaves": 26,
    "min_child_samples": 88,
    "cat_smooth": 78,
    "verbose": -1,
}



## === cell 21
from concurrent.futures import ThreadPoolExecutor, as_completed

kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)

X_all = train[cols].values
y_all = train["Cover_Type"].values.astype(np.int32)
X_test = test[cols].values

fold_indices = list(kf.split(X_all, y_all))

preds = []
acc = []
models = []


def train_and_predict(fold_tuple):
    i, (trn_idx, val_idx) = fold_tuple
    X_tr, X_val = X_all[trn_idx], X_all[val_idx]
    y_tr, y_val = y_all[trn_idx], y_all[val_idx]

    model = LGBMClassifier(**params)
    model.fit(X_tr, y_tr)

    pred_test = model.predict(X_test)
    fold_acc = accuracy_score(y_val, model.predict(X_val))

    return i, pred_test, fold_acc, model


with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_fold = {
        executor.submit(train_and_predict, (i, split)): i
        for i, split in enumerate(fold_indices, start=1)
    }
    for future in as_completed(future_to_fold):
        i, pred_test, fold_acc, model = future.result()
        preds.append(pred_test)
        acc.append(fold_acc)
        models.append(model)
        print(f"fold: {i} , accuracy: {round(fold_acc * 100, 3)}")
        gc.collect()

model = models[-1]
gc.collect()



## === cell 22
print(f"The mean Accuracy is : {round(np.mean(acc) * 100, 3)}")



## === cell 23
try:
    from optuna.integration import lightgbm as opt_lgb

    opt_lgb.plot_importance(model, max_num_features=30, figsize=(10, 10))
    plt.show()
except Exception as e:
    print("Optuna LightGBM integration not available; skipping importance plot.", e)



## === cell 24
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)
preds_array = np.column_stack(preds)  # shape (n_test, n_folds)
mode_result = stats.mode(preds_array, axis=1)
prediction = mode_result.mode.squeeze().astype(int)
sub["Cover_Type"] = prediction
sub.to_csv("submission.csv", index=False)



## === cell 25
sub.head()

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
imbalanced-learn==0.13.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import optuna




## === cell 1
possible_paths = [
    pathlib.Path("/kaggle/input/tabular-playground-series-dec-2021"),
    pathlib.Path("kaggle/data/tabular-playground-series-dec-2021"),
    pathlib.Path("data/tabular-playground-series-dec-2021"),
    pathlib.Path("input/tabular-playground-series-dec-2021"),
]
base_path = next((p for p in possible_paths if p.exists()), None)
if base_path is None:
    raise FileNotFoundError(
        "Could not locate the tabular-playground-series-dec-2021 directory."
    )

train_path = base_path / "train.csv"
test_path = base_path / "test.csv"
sample_path = base_path / "sample_submission.csv"

df_train_og = pd.read_csv(train_path)
df_test_og = pd.read_csv(test_path)
submission = pd.read_csv(sample_path)




## === cell 2
def reduce_mem_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
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
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df




## === cell 3
df_train = reduce_mem_usage(df_train_og)
df_test = reduce_mem_usage(df_test_og)
del df_train_og, df_test_og




## === cell 4
cat_count = collections.Counter(df_train["Cover_Type"])
cat = list(cat_count.keys())
cat_freq = list(cat_count.values())
plt.bar(cat, cat_freq)
plt.xlabel("Cover_Type")
plt.ylabel("Count")
plt.show()




## === cell 5
if "Cover_Type" in df_train.columns:
    df_train = df_train[(df_train["Cover_Type"] != 4) & (df_train["Cover_Type"] != 5)]

drop_cols = ["Id", "Cover_Type", "Soil_Type7", "Soil_Type15"]
existing_drop = [c for c in drop_cols if c in df_train.columns]
X = df_train.drop(columns=existing_drop)

y_raw = df_train["Cover_Type"]
le = LabelEncoder()
y = le.fit_transform(y_raw)  # y now has values 0 … (n_classes-1)




## === cell 6
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




## === cell 7
def objective_xgb(trial):
    xgb_params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "tree_method": "hist",
        "booster": "gbtree",
        "eval_metric": "mlogloss",
        "objective": "multi:softmax",
        "n_estimators": trial.suggest_int("n_estimators", 500, 1000, step=100),
        "reg_lambda": trial.suggest_int("reg_lambda", 1, 100),
        "reg_alpha": trial.suggest_int("reg_alpha", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.2, 0.8, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "gamma": trial.suggest_float("gamma", 0, 20),
        "n_jobs": 4,
        "random_state": 42,
    }

    pipe = Pipeline(
        steps=[("scaler", StandardScaler()), ("clf", XGBClassifier(**xgb_params))]
    )
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)
    return accuracy_score(y_test, preds)




## === cell 8
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=20, timeout=300)




## === cell 9
if study_xgb.trials:
    best_params_xgb = study_xgb.best_params
else:
    best_params_xgb = {
        "learning_rate": 0.05,
        "n_estimators": 800,
        "reg_lambda": 10,
        "reg_alpha": 5,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "max_depth": 6,
        "min_child_weight": 5,
        "gamma": 0.0,
    }




## === cell 10
final_pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            XGBClassifier(
                **best_params_xgb,
                n_jobs=4,
                random_state=42,
                tree_method="hist",
                eval_metric="mlogloss",
                objective="multi:softmax",
            ),
        ),
    ]
)
final_pipe.fit(X, y)




## === cell 11
test_drop = [c for c in ["Id", "Soil_Type7", "Soil_Type15"] if c in df_test.columns]
df_test_processed = df_test.drop(columns=test_drop)

test_pred_enc = final_pipe.predict(df_test_processed)
test_pred = le.inverse_transform(test_pred_enc.astype(int))




## === cell 12
submission["Cover_Type"] = test_pred
submission.to_csv("submission.csv", index=False)

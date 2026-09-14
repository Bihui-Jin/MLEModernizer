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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

try:
    from imblearn.under_sampling import (
        RandomUnderSampler,
    )  # Used for under sampling. explained further in notebook.
except ModuleNotFoundError:
    RandomUnderSampler = None
import collections
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import optuna


## === cell 1
df_train_og = pd.read_csv('../input/tabular-playground-series-dec-2021/train.csv')
df_test_og  = pd.read_csv('../input/tabular-playground-series-dec-2021/test.csv')
submission  = pd.read_csv('../input/tabular-playground-series-dec-2021/sample_submission.csv')


## === cell 2
df_train_og.shape


## === cell 3
df_train_og.head()


## === cell 4
df_train_og.nunique()


## === cell 5
def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df


## === cell 6
df_train = reduce_mem_usage(df_train_og)
df_test = reduce_mem_usage(df_test_og)
del df_train_og
del df_test_og


## === cell 7
cat_count = collections.Counter(df_train['Cover_Type'])
cat_freq = cat_count.values()
cat = cat_count.keys()
plt.bar(cat , cat_freq)

print(cat_count)


## === cell 8
df_train = df_train[(df_train['Cover_Type'] != 4) & (df_train['Cover_Type'] != 5)]


## === cell 9

X = df_train.drop(columns=["Id", "Cover_Type", "Soil_Type7", "Soil_Type15"])
y = df_train["Cover_Type"]

class_counts = y.value_counts()
minority_class = class_counts.idxmin()
minority_n = int(class_counts.min())

rng_seed = 42  # fixed seed for determinism comparable to a fixed random_state
df_xy = df_train.loc[:, X.columns.tolist() + ["Cover_Type"]]

parts = []
for cls, cnt in class_counts.items():
    cls_rows = df_xy[df_xy["Cover_Type"] == cls]
    if cls == minority_class:
        parts.append(cls_rows)
    else:
        parts.append(
            cls_rows.sample(n=minority_n, random_state=rng_seed, replace=False)
        )

df_res = (
    pd.concat(parts, axis=0)
    .sample(frac=1.0, random_state=rng_seed)
    .reset_index(drop=True)
)

X_res = df_res.drop(columns=["Cover_Type"])
y_res = df_res["Cover_Type"]


## === cell 10
cat_count = collections.Counter(y_res)
cat_freq = cat_count.values()
cat = cat_count.keys()
plt.bar(cat , cat_freq)

print(cat_count)


## === cell 11
x_train,x_test,y_train,y_test = train_test_split(X_res,y_res,test_size = 0.2)


## === cell 12
def objective_xgb(trial):
    xgb_params = {
        'learning_rate': 0.03,
        'tree_method': 'gpu_hist',
        'booster': 'gbtree',
        'eval_metric' : 'mlogloss',
        'objective' : 'multi:softmax',
        'n_estimators': trial.suggest_int('n_estimators', 500, 1000, 100),
        'subsample': trial.suggest_float('subsample', 0.2, 0.8, step=0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10), 
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'predictor' : 'gpu_predictor'
    }
    
    pipe = Pipeline(steps = [
    
    ('step1' , StandardScaler()),
    ('step2' , XGBClassifier(**xgb_params))
     ])
    
    pipe.fit(x_train,y_train)
    y_pred = pipe.predict(x_test)
    return accuracy_score(y_test,y_pred)


## === cell 13
from sklearn.preprocessing import LabelEncoder

_le = LabelEncoder()
y_train_enc = _le.fit_transform(y_train)
y_test_enc = _le.transform(y_test)


def _xgb_cuda_available() -> bool:
    try:
        XGBClassifier(
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            objective="multi:softmax",
            eval_metric="mlogloss",
            n_estimators=1,
            max_depth=1,
            learning_rate=0.1,
            subsample=1.0,
            verbosity=0,
        ).fit(
            np.zeros((4, 2), dtype=np.float32), np.array([0, 1, 0, 1], dtype=np.int32)
        )
        return True
    except Exception:
        return False


_USE_GPU = _xgb_cuda_available()


def objective_xgb(trial):
    xgb_params = {
        "learning_rate": 0.03,
        "tree_method": "gpu_hist" if _USE_GPU else "hist",
        "booster": "gbtree",
        "eval_metric": "mlogloss",
        "objective": "multi:softmax",
        "n_estimators": trial.suggest_int("n_estimators", 500, 1000, 100),
        "subsample": trial.suggest_float("subsample", 0.2, 0.8, step=0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "gamma": trial.suggest_float("gamma", 0, 1.0),
        "predictor": "gpu_predictor" if _USE_GPU else "auto",
    }

    pipe = Pipeline(
        steps=[("step1", StandardScaler()), ("step2", XGBClassifier(**xgb_params))]
    )

    pipe.fit(x_train, y_train_enc)
    y_pred = pipe.predict(x_test)
    return accuracy_score(y_test_enc, y_pred)


study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=50)


## === cell 14
best_params_xgb = study_xgb.best_params


## === cell 15
pipe = Pipeline(steps = [
    
    ('step1' , StandardScaler()),
    ('step2' , XGBClassifier(**best_params_xgb))
     ])


## === cell 16
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
print(accuracy_score(y_test,y_pred))


## === cell 17
df_test = df_test.drop(columns = ['Id' , 'Soil_Type7' , 'Soil_Type15'])
Final_pred = pipe.predict(df_test)


## === cell 18
submission['Cover_Type'] = Final_pred
submission.to_csv('Submission.csv' , index=False)

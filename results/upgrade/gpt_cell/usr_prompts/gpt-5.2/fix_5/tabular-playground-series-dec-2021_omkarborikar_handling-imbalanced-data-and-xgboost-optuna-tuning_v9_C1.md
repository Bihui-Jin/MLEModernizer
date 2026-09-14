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
    from imblearn.under_sampling import RandomUnderSampler  # Used for under sampling.
except Exception:

    class RandomUnderSampler:
        def __init__(self, sampling_strategy="auto", random_state=None):
            self.sampling_strategy = sampling_strategy
            self.random_state = random_state

        def fit_resample(self, X, y):
            X_arr = np.asarray(X)
            y_arr = np.asarray(y)

            rng = np.random.RandomState(self.random_state)
            classes, counts = np.unique(y_arr, return_counts=True)

            if isinstance(self.sampling_strategy, dict):
                target_counts = {
                    c: int(self.sampling_strategy.get(c, counts[i]))
                    for i, c in enumerate(classes)
                }
            else:
                min_count = int(counts.min())
                target_counts = {c: min_count for c in classes}

            idx_keep = []
            for c in classes:
                idx_c = np.flatnonzero(y_arr == c)
                n_target = min(len(idx_c), int(target_counts.get(c, len(idx_c))))
                if n_target <= 0:
                    continue
                chosen = rng.choice(idx_c, size=n_target, replace=False)
                idx_keep.append(chosen)

            if len(idx_keep) == 0:
                return X_arr[:0], y_arr[:0]

            idx_keep = np.concatenate(idx_keep)
            idx_keep.sort()

            X_res = X_arr[idx_keep]
            y_res = y_arr[idx_keep]

            if isinstance(X, pd.DataFrame):
                X_res = X.iloc[idx_keep].copy()
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_res = y.iloc[idx_keep].copy()

            return X_res, y_res


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
rus = RandomUnderSampler(sampling_strategy = "not minority")
X  = df_train.drop(columns = ['Id' , 'Cover_Type','Soil_Type7' , 'Soil_Type15'])
y = df_train['Cover_Type']
X_res,y_res = rus.fit_resample(X,y)


## === cell 10
cat_count = collections.Counter(y_res)
cat_freq = cat_count.values()
cat = cat_count.keys()
plt.bar(cat , cat_freq)

print(cat_count)


## === cell 12
from sklearn.feature_selection import SelectKBest,f_classif
selector = SelectKBest(f_classif,k="all")
fitter = selector.fit(X_res,y_res)
scores_df = pd.DataFrame(fitter.scores_) 
columns_df = pd.DataFrame(X_res.columns)
featurescores = pd.concat([scores_df ,columns_df] , axis=1)
featurescores.columns=['score','column name']
plt.figure(figsize=(20,5))
plt.bar(featurescores['column name'] , featurescores['score'],width=0.4)
plt.xticks(rotation = 'vertical')
plt.plot()


## === cell 13
featurescores = featurescores.sort_values(by = 'score' , ascending=False)


## === cell 14

useful_features = featurescores['column name'].head(20)


## === cell 15
useful_features


## === cell 16
X_res = X_res[useful_features]


## === cell 17
x_train,x_test,y_train,y_test = train_test_split(X_res,y_res,test_size = 0.2)


## === cell 18
def objective_xgb(trial):
    xgb_params = {
        'learning_rate': 0.01,
        'tree_method': 'gpu_hist',
        'booster': 'gbtree',
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000, 100),
        'reg_lambda': trial.suggest_int('reg_lambda', 1, 100),
        'reg_alpha': trial.suggest_int('reg_alpha', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0, step=0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10), 
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
        'gamma': trial.suggest_float('gamma', 0, 20)        
    }
    
    pipe = Pipeline(steps = [
    
    ('step1' , StandardScaler()),
    ('step2' , XGBClassifier(**xgb_params))
     ])
    
    pipe.fit(x_train,y_train)
    y_pred = pipe.predict(x_test)
    return accuracy_score(y_test,y_pred)


## === cell 19
def _make_zero_based_labels(y_tr, y_te):
    y_tr = np.asarray(y_tr)
    y_te = np.asarray(y_te)
    classes = np.unique(y_tr)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_tr_mapped = np.vectorize(class_to_idx.get)(y_tr)
    y_te_mapped = np.vectorize(class_to_idx.get)(y_te)
    return y_tr_mapped, y_te_mapped


def _has_xgb_gpu():
    try:
        import xgboost as xgb

        cfg = xgb.config_context()
        if isinstance(cfg, dict):
            device = str(cfg.get("device", "")).lower()
            gpu_id = cfg.get("gpu_id", None)
            if device == "cuda":
                return True
            if gpu_id is not None:
                try:
                    return int(gpu_id) >= 0
                except Exception:
                    return False
        return False
    except Exception:
        return False


_XGB_TREE_METHOD = "gpu_hist" if _has_xgb_gpu() else "hist"


def objective_xgb(trial):
    xgb_params = {
        "learning_rate": 0.01,
        "tree_method": _XGB_TREE_METHOD,
        "booster": "gbtree",
        "n_estimators": trial.suggest_int("n_estimators", 500, 4000, 100),
        "reg_lambda": trial.suggest_int("reg_lambda", 1, 100),
        "reg_alpha": trial.suggest_int("reg_alpha", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "gamma": trial.suggest_float("gamma", 0, 20),
    }

    pipe = Pipeline(
        steps=[("step1", StandardScaler()), ("step2", XGBClassifier(**xgb_params))]
    )

    y_train_mapped, y_test_mapped = _make_zero_based_labels(y_train, y_test)

    pipe.fit(x_train, y_train_mapped)
    y_pred = pipe.predict(x_test)
    return accuracy_score(y_test_mapped, y_pred)


study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=50)


## === cell 20
best_params_xgb = study_xgb.best_params


## === cell 21
pipe = Pipeline(steps = [
    
    ('step1' , StandardScaler()),
    ('step2' , XGBClassifier(**best_params_xgb))
     ])


## === cell 22
y_train_mapped, _ = _make_zero_based_labels(y_train, y_test)
pipe.fit(x_train, y_train_mapped)


## === cell 23
df_test = df_test[useful_features]
Final_pred = pipe.predict(df_test)


## === cell 24
submission['Cover_Type'] = Final_pred
submission.to_csv('Submission.csv' , index=False)

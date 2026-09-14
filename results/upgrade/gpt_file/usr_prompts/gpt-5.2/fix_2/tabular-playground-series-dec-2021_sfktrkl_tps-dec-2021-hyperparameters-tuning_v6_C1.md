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
scipy==1.15.3
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

0.95456

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold



## === cell 1
train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")




## === cell 2
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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
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


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)



## === cell 3
train.head()



## === cell 4
train.describe()



## === cell 5
print("Columns: \n{0}".format(list(train.columns)))



## === cell 6
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)



## === cell 7
missing_cols_train = train.columns[train.isna().any()]
print("Missing values in train data: {0}".format(list(missing_cols_train)))

missing_cols_test = test.columns[test.isna().any()]
print("Missing values in test data: {0}".format(list(missing_cols_test)))



## === cell 8
duplicates_train = train.duplicated().sum()
print("Duplicates in train data: {0}".format(duplicates_train))

duplicates_test = test.duplicated().sum()
print("Duplicates in test data: {0}".format(duplicates_test))



## === cell 9
categorical_features = train.columns[11:-1:]
print("Categorical Columns: \n{0}".format(list(categorical_features)))



## === cell 10
numerical_features = train.columns[1:11]
print("Numerical Columns: \n{0}".format(list(train.columns[1:11])))
train[numerical_features].describe()



## === cell 11
plt.figure(figsize=(10, 6))
plt.title("Target distribution")
sns.countplot(x=train["Cover_Type"], data=train)



## === cell 12
cType5 = train[train["Cover_Type"] == 5].index
print("Number of rows with Cover_Type = 5: {0}".format(len(cType5)))



## === cell 13
print(
    "Unique values in Soil_Type7 column train data: {0}".format(
        train["Soil_Type7"].unique()
    )
)
print(
    "Unique values in Soil_Type15 column train data: {0}".format(
        train["Soil_Type15"].unique()
    )
)

print(
    "Unique values in Soil_Type7 column test data: {0}".format(
        test["Soil_Type7"].unique()
    )
)
print(
    "Unique values in Soil_Type15 column test data: {0}".format(
        test["Soil_Type15"].unique()
    )
)



## === cell 14
train.drop(cType5, axis=0, inplace=True)

train.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
test.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)



## === cell 15
X = train.iloc[:, 1:-1].copy()
y = train.Cover_Type.copy()



## === cell 16
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


def _xgb_gpu_params():
    try:
        import xgboost as xgb  # noqa: F401

        if os.environ.get("FORCE_XGB_CPU", "0") == "1":
            return {"tree_method": "hist", "predictor": "auto"}
        return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
    except Exception:
        return {"tree_method": "hist", "predictor": "auto"}


def run_model(model):
    model.fit(
        train_X,
        train_y,
        eval_set=[(val_X, val_y)],
        early_stopping_rounds=40,
        eval_metric="mlogloss",
        verbose=False,
    )
    predictions = model.predict(val_X)
    score = accuracy_score(val_y, predictions)
    return score, "Accuracy score:  {:.6f}".format(score)


def evaluate_model(model):
    print("Accuracy score:", accuracy_score(train_y, model.predict(train_X)))




## === cell 17
def run_xgboost_model(c, max_score):
    try:
        params = dict(
            seed=1,
            learning_rate=float(c[0]),
            gamma=float(c[1]),
            max_depth=int(c[2]),
            reg_alpha=float(c[3]),
            reg_lambda=float(c[4]),
            n_estimators=int(c[5]),
            eval_metric="mlogloss",
        )
        params.update(_xgb_gpu_params())
        value = run_model(XGBClassifier(**params))
        if value[0] > max_score[0]:
            max_score[0] = value[0]
            max_score[1] = c
        print(
            "Combination: learning_rate: {0}, gamma: {1}, max_depth: {2}, reg_alpha: {3}, reg_lambda: {4}, n_estimators: {5}, {6}".format(
                c[0], c[1], c[2], c[3], c[4], c[5], value[1]
            )
        )
    except Exception as e:
        print(
            "Invalid combination: learning_rate: {0}, gamma: {1}, max_depth: {2}, reg_alpha: {3}, reg_lambda: {4}, n_estimators: {5}. Error: {6}".format(
                c[0], c[1], c[2], c[3], c[4], c[5], str(e)[:200]
            )
        )
        pass




## === cell 18
learning_rate = [0.5]
gamma = [1.0]
max_depth = [8]

reg_alpha = [0]
reg_lambda = [0, 0.1, 0.2]
n_estimators = [100]



## === cell 19
max_score = [0, []]
combinations = np.array(
    np.meshgrid(learning_rate, gamma, max_depth, reg_alpha, reg_lambda, n_estimators)
).T.reshape(-1, 6)
for combination in combinations:
    run_xgboost_model(combination, max_score)

if len(max_score[1]) == 6:
    print(
        "Best score: learning_rate: {0}, gamma: {1}, max_depth: {2}, reg_alpha: {3}, reg_lambda: {4}, n_estimators: {5}, Accuracy score: {6:.6f}.".format(
            max_score[1][0],
            max_score[1][1],
            max_score[1][2],
            max_score[1][3],
            max_score[1][4],
            max_score[1][5],
            max_score[0],
        )
    )
else:
    print(
        "No valid hyperparameter combination completed; proceeding with default CV model settings."
    )



## === cell 20
le = LabelEncoder()
y_enc = le.fit_transform(y)

test_X = test.iloc[:, 1:]

params = dict(
    seed=1, eval_metric="mlogloss", learning_rate=0.3, gamma=1.6, max_depth=10
)
params.update(_xgb_gpu_params())
model = XGBClassifier(**params)

fold = 1
accuracy_scores = []
test_predictions = []

skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_idx, test_idx in skf.split(X, y_enc):
    train_X, val_X = X.iloc[train_idx], X.iloc[test_idx]
    train_y, val_y = y_enc[train_idx], y_enc[test_idx]

    model.fit(
        train_X,
        train_y,
        early_stopping_rounds=40,
        eval_set=[(val_X, val_y)],
        verbose=False,
    )
    predictions = model.predict(val_X)
    score = accuracy_score(val_y, predictions)
    print("Fold: {0}  \t\t Accuracy score:  {1:.6f}".format(fold, score))
    accuracy_scores.append(score)

    test_predictions.append(model.predict(test_X))
    fold += 1

test_pred_matrix = np.column_stack(test_predictions)  # shape: (n_test, n_folds)
maj_vote_enc = np.apply_along_axis(
    lambda r: np.bincount(r.astype(np.int64)).argmax(), 1, test_pred_matrix
)

test_predictions_final = le.inverse_transform(maj_vote_enc.astype(int))

print("Mean accuracy score: {0:.6f}".format(np.mean(accuracy_scores)))



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/3609532895.py in <cell line: 0>()
     21     train_y, val_y = y_enc[train_idx], y_enc[test_idx]
     22 
---> 23     model.fit(
     24         train_X,
     25         train_y,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1517             )
   1518 
-> 1519             self._Booster = train(
   1520                 params,
   1521                 train_dmatrix,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)
    179         if cb_container.before_iteration(bst, i, dtrain, evals):
    180             break
--> 181         bst.update(dtrain, i, obj)
    182         if cb_container.after_iteration(bst, i, dtrain, evals):
    183             break

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in update(self, dtrain, iteration, fobj)
   2048 
   2049         if fobj is None:
-> 2050             _check_call(
   2051                 _LIB.XGBoosterUpdateOneIter(
   2052                     self.handle, ctypes.c_int(iteration), dtrain.handle

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [18:50:51] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [18:50:51] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7f98c4954f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7f98c496b95a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7f98c49753cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7f98c428dc79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x461d09) [0x7f98c428ed09]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7f98c42f24f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7f98c3f8eef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7f9937187e2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7f9937184493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7f98c4954f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7f98c49755c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7f98c428dc79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x461d09) [0x7f98c428ed09]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7f98c42f24f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7f98c3f8eef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7f9937187e2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7f9937184493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7f99371974d8]



## === cell 21
output = pd.DataFrame({"Id": test.Id.values, "Cover_Type": test_predictions_final})
output.to_csv("submission.csv", index=False)
output.head()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/279355247.py in <cell line: 0>()
----> 1 output = pd.DataFrame({"Id": test.Id.values, "Cover_Type": test_predictions_final})
      2 output.to_csv("submission.csv", index=False)
      3 output.head()

NameError: name 'test_predictions_final' is not defined

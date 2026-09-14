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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

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
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.9677

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## === cell 1
train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv", index_col="id")

submission = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv', index_col="id")


## === cell 2
train.head()


## === cell 3
train.dtypes


## === cell 4
train.describe().T


## === cell 5
train.isna().sum()


## === cell 6
fig = plt.figure(figsize=(20, 12))
hist = train.hist(ax = fig.gca(), bins=20)


## === cell 7
CONTINUOUS = train.columns[train.dtypes == 'float64'].to_list()
DISCRETE = train.columns[train.dtypes == 'int64'].to_list()
OBJECT = train.columns[train.dtypes == 'object'].to_list()

DISCRETE.remove("target")

print("CONTINUOUS:", ', '.join(CONTINUOUS))
print("DISCRETE:", ', '.join(DISCRETE))
print("OBJECT:", ', '.join(OBJECT))


## === cell 8
train.f_27


## === cell 9
def get_unique_letters(arr):
    unique = set()
    for val in arr:
        unique |= set(val)
    l = list(unique)
    l.sort()
    return l

unique_letters = get_unique_letters(train.f_27)
print(unique_letters)


## === cell 10
LETTER_COUNTS = [letter + '_num' for letter in unique_letters]

def augment(df, unique_letters):
    for col, letter in zip(LETTER_COUNTS, unique_letters):
        df[col] = df.f_27.str.count(letter)

augment(train, unique_letters)
augment(test, unique_letters)


## === cell 11
train.head()


## === cell 12
train.target.value_counts()


## === cell 13
corr = train.corr()

plt.figure(figsize=(12, 12))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2880001401.py in <cell line: 0>()
----> 1 corr = train.corr()
      2 
      3 plt.figure(figsize=(12, 12))
      4 sns.heatmap(corr, 
      5         xticklabels=corr.columns,

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: 'ACBADABECB'

## === cell 14
ALL_TRAIN_COLUMNS = CONTINUOUS + DISCRETE + LETTER_COUNTS


## === cell 15
from sklearn.model_selection import train_test_split
VAL_SPLIT = 0.2
X_train, X_valid, y_train, y_valid = train_test_split(train[ALL_TRAIN_COLUMNS], train.target, 
                                                      test_size = VAL_SPLIT, random_state = 42)


## === cell 16
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators    = 4096,
    random_state    = 42,
    tree_method     = "gpu_hist",
    early_stopping_rounds = 256, 
    eval_metric     = ['auc']
)

model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], verbose = 100)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/2160697106.py in <cell line: 0>()
      9 )
     10 
---> 11 model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], verbose = 100)

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

XGBoostError: [02:23:57] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [02:23:57] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff4179af2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fff417b195a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fff417bb3cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff410d3c79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff410d476c]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff411384f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff40dd4ef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff4179af2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fff417bb5c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff410d3c79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff410d476c]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff411384f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff40dd4ef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]



## === cell 17
history = model.evals_result()

LABELS = {"validation_0": "train", "validation_1": "test"}

def plot_metric(metric):
    fig, axis = plt.subplots(1, 1)
    for name, label in LABELS.items():
        data = history[name][metric]
        plt.plot(data, label=label)
    plt.title(metric)
    plt.grid()
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

plot_metric("auc")


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/997931190.py in <cell line: 0>()
----> 1 history = model.evals_result()
      2 
      3 LABELS = {"validation_0": "train", "validation_1": "test"}
      4 
      5 def plot_metric(metric):

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in evals_result(self)
   1258             evals_result = self.evals_result_
   1259         else:
-> 1260             raise XGBoostError(
   1261                 "No evaluation result, `eval_set` is not used during training."
   1262             )

XGBoostError: No evaluation result, `eval_set` is not used during training.

## === cell 18
def plot_importance(importance, names, max_features = 10):
    zipped = list(zip(importance, names))
    zipped.sort(reverse=True)

    plt.figure(figsize=(12,10))
    palette = sns.color_palette("Reds_r", max_features)
    
    sns.barplot(x=[imp for imp, _ in zipped][:max_features], y=[name for _, name in zipped][:max_features],
               palette = palette)
    
    plt.title("Feature Importances")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    
plot_importance(model.feature_importances_, ALL_TRAIN_COLUMNS, max_features = 50)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2977922095.py in <cell line: 0>()
     13     plt.ylabel('Features')
     14 
---> 15 plot_importance(model.feature_importances_, ALL_TRAIN_COLUMNS, max_features = 50)

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in feature_importances_(self)
   1312 
   1313         """
-> 1314         b: Booster = self.get_booster()
   1315 
   1316         def dft() -> str:

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 19
predictions = model.predict_proba(test[ALL_TRAIN_COLUMNS])[:, 1]


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3875126359.py in <cell line: 0>()
----> 1 predictions = model.predict_proba(test[ALL_TRAIN_COLUMNS])[:, 1]

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict_proba(self, X, validate_features, base_margin, iteration_range)
   1630             class_prob = softmax(raw_predt, axis=1)
   1631             return class_prob
-> 1632         class_probs = super().predict(
   1633             X=X,
   1634             validate_features=validate_features,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 20
submission['target'] = predictions
submission.to_csv('submission.csv')


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1712374874.py in <cell line: 0>()
----> 1 submission['target'] = predictions
      2 submission.to_csv('submission.csv')

NameError: name 'predictions' is not defined

## === cell 21
submission.head()

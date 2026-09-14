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
lightgbm==4.6.0
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

0.98462

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings
import pandas as pd
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")



## === cell 1
DATA_ROWS = None
NROWS = 50
NCOLS = 15
BASE_PATH = "..."



## === cell 2
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.max_columns", NCOLS)
pd.set_option("display.max_rows", NROWS)



## === cell 3
trn_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
tst_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")

sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)



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
categ_cols = [
    "f_29",
    "f_30",
    "f_13",
    "f_18",
    "f_17",
    "f_14",
    "f_11",
    "f_10",
    "f_09",
    "f_15",
    "f_07",
    "f_12",
    "f_16",
    "f_08",
    "f_27",
]
trn_data[categ_cols].sample(5)



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
pass



## === cell 17
pass




## === cell 18
def count_sequence(df, field):
    """
    For each letter of the provided sequence it returns a new feature with the number of occurrences.
    """
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for letter in alphabet:
        df[letter + "_count"] = df[field].str.count(letter)
    df["unique_characters"] = df["f_27"].apply(lambda s: len(set(s)))
    return df




## === cell 19
def count_chars(df, field):
    """
    Creates numeric character position features and a unique‑character count.
    Vectorized implementation to reduce Python‑level loops.
    """
    padded = df[field].fillna("").astype(str).str.pad(10, side="right", fillchar="A")
    encoded = padded.str.encode("utf-8")
    char_arrays = np.stack(
        encoded.apply(lambda b: np.frombuffer(b, dtype=np.uint8)[:10]), axis=0
    )
    char_arrays = char_arrays - ord("A")
    for i in range(10):
        df[f"ch_{i}"] = char_arrays[:, i].astype(np.int8)
    df["unique_characters"] = df[field].apply(
        lambda s: len(set(s)) if isinstance(s, str) else 0
    )
    return df




## === cell 20
trn_data = count_chars(trn_data, "f_27")
tst_data = count_chars(tst_data, "f_27")



## === cell 21
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
trn_data["f_27_enc"] = encoder.fit_transform(trn_data["f_27"])
tst_data["f_27_enc"] = encoder.transform(tst_data["f_27"])



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _encode(values, uniques, check_unknown)
    223         try:
--> 224             return _map_to_integer(values, uniques)
    225         except KeyError as e:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _map_to_integer(values, uniques)
    163     table = _nandict({val: i for i, val in enumerate(uniques)})
--> 164     return np.array([table[v] for v in values])
    165 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in <listcomp>(.0)
    163     table = _nandict({val: i for i, val in enumerate(uniques)})
--> 164     return np.array([table[v] for v in values])
    165 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in __missing__(self, key)
    157             return self.nan_value
--> 158         raise KeyError(key)
    159 

KeyError: 'AABBBBDHCC'

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1556338743.py in <cell line: 0>()
      3 encoder = LabelEncoder()
      4 trn_data["f_27_enc"] = encoder.fit_transform(trn_data["f_27"])
----> 5 tst_data["f_27_enc"] = encoder.transform(tst_data["f_27"])
      6 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_label.py in transform(self, y)
    137             return np.array([])
    138 
--> 139         return _encode(y, uniques=self.classes_)
    140 
    141     def inverse_transform(self, y):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _encode(values, uniques, check_unknown)
    224             return _map_to_integer(values, uniques)
    225         except KeyError as e:
--> 226             raise ValueError(f"y contains previously unseen labels: {str(e)}")
    227     else:
    228         if check_unknown:

ValueError: y contains previously unseen labels: 'AABBBBDHCC'

## === cell 22
trn_data.head()



## === cell 23
ignore = ["id", "target", "f_27", "f_27_enc"]  # f_27 has been label encoded...
features = [feat for feat in trn_data.columns if feat not in ignore]
target_feature = "target"



## === cell 24
from sklearn.model_selection import train_test_split

test_size_pct = 0.20
X_train, X_valid, y_train, y_valid = train_test_split(
    trn_data[features],
    trn_data[target_feature],
    test_size=test_size_pct,
    random_state=42,
)



## === cell 25
xgb = None
lgb = None



## === cell 26
pass



## === cell 27
pass



## === cell 28
pass



## === cell 29
lgb_params = {
    "n_estimators": 8192,
    "min_child_samples": 96,
    "max_bins": 512,
    "random_state": 46,
    "n_jobs": 1,  # single‑thread per LightGBM model (kept for completeness)
}



## === cell 30
pass



## === cell 31
pass



## === cell 32
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed  # parallel execution for folds



## === cell 33
xgb_params = {
    "max_depth": 6,
    "learning_rate": 0.15,
    "subsample": 0.95,
    "colsample_bytree": 0.95,
    "reg_lambda": 1.50,
    "reg_alpha": 1.50,
    "gamma": 1.50,
    "max_bin": 512,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "seed": 46,
    "verbosity": 0,
    "n_estimators": 8192,
}



## === cell 34
import xgboost as xgb

X_all = trn_data[features].values.astype(np.float32)
y_all = trn_data[target_feature].values.astype(np.int32)
X_test = tst_data[features].values.astype(np.float32)

dtest = xgb.DMatrix(X_test)


def train_fold(fold, trn_idx, val_idx):
    dtrain = xgb.DMatrix(X_all[trn_idx], label=y_all[trn_idx])
    dval = xgb.DMatrix(X_all[val_idx], label=y_all[val_idx])

    evals = [(dval, "validation")]
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=xgb_params["n_estimators"],
        evals=evals,
        early_stopping_rounds=256,
        verbose_eval=False,
    )

    val_pred = model.predict(dval)
    score = roc_auc_score(y_all[val_idx], val_pred)

    tst_pred = model.predict(dtest)
    return (fold, score, tst_pred)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = Parallel(n_jobs=5, backend="loky")(
    delayed(train_fold)(fold, trn_idx, val_idx)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_all))
)

score_list = []
predictions = []
for fold, score, tst_pred in sorted(results, key=lambda x: x[0]):
    print(f"Fold {fold}, AUC = {score:.3f}\n")
    score_list.append(score)
    predictions.append(tst_pred)

print(f"OOF AUC: {np.mean(score_list):.3f}")
print(".........")




## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/backend/queues.py", line 159, in _feed
    obj_ = dumps(obj, reducers=reducers)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/backend/reduction.py", line 214, in dumps
    dump(obj, buf, reducers=reducers, protocol=protocol)
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/backend/reduction.py", line 207, in dump
    _LokyPickler(file, reducers=reducers, protocol=protocol).dump(obj)
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/cloudpickle/cloudpickle.py", line 1303, in dump
    return super().dump(obj)
           ^^^^^^^^^^^^^^^^^
ValueError: ctypes objects containing pointers cannot be pickled
"""

The above exception was the direct cause of the following exception:

PicklingError                             Traceback (most recent call last)
/tmp/ipykernel_11/2403977165.py in <cell line: 0>()
     35 kf = KFold(n_splits=5, shuffle=True, random_state=42)
     36 
---> 37 results = Parallel(n_jobs=5, backend="loky")(
     38     delayed(train_fold)(fold, trn_idx, val_idx)
     39     for fold, (trn_idx, val_idx) in enumerate(kf.split(X_all))

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

PicklingError: Could not pickle the task to send it to the workers.

## === cell 35
def plot_feature_importance(importance, names, model_type, max_features=10):
    import matplotlib.pyplot as plt
    import seaborn as sns

    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)
    fi_df = fi_df.head(max_features)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    plt.title(model_type + " FEATURE IMPORTANCE")
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")
    plt.show()




## === cell 36
if "model" in globals():
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_feature_importance(
        model.feature_importances_, X_train.columns, "XGB ", max_features=25
    )
else:
    print("Model not available for plotting – skipping feature importance.")



## === cell 37
sub.head()



## === cell 38
sub["target"] = np.mean(predictions, axis=0)
sub.to_csv("my_submission_043022.csv", index=False)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1788384864.py in <cell line: 0>()
----> 1 sub["target"] = np.mean(predictions, axis=0)
      2 sub.to_csv("my_submission_043022.csv", index=False)
      3 

NameError: name 'predictions' is not defined

## === cell 39
sub.head()

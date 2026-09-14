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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.9969

# 6. Current score

None

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
import warnings

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
except Exception as e:
    print("TensorFlow import failed:", e)
    tf = None

np.random.seed(42)
if tf is not None:
    tf.random.set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
sample_submission = pd.read_csv(
    "../input/tabular-playground-series-may-2022/sample_submission.csv"
)




## === cell 3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import scipy.stats
from joblib import Parallel, delayed




## === cell 4
if tf is not None:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except Exception:
        strategy = tf.distribute.get_strategy()
    print("REPLICAS:", strategy.num_replicas_in_sync)
else:
    strategy = None
    print("TensorFlow not available – skipping TPU/strategy setup.")




## === cell 5
stat = ["mean", "std", "sum", "var", "euc", "ptp", "logsumexp"]
import scipy.special
from scipy.signal import lfilter  # fast exponential smoothing


def add_new_features(df):
    float_cols = df.select_dtypes(
        include=["float32", "float64", "float"]
    ).columns.tolist()
    if "target" in float_cols:
        float_cols.remove("target")
    df_float = df[float_cols].astype(np.float32, copy=False)
    arr = df_float.to_numpy()

    df["mean"] = arr.mean(axis=1)
    df["std"] = arr.std(axis=1)
    df["sum"] = arr.sum(axis=1)
    df["var"] = arr.var(axis=1)
    df["euc"] = np.linalg.norm(arr, axis=1)
    df["logsumexp"] = scipy.special.logsumexp(arr, axis=1)
    df["ptp"] = arr.ptp(axis=1)

    com = 0.8
    alpha = 1.0 / (com + 1.0)
    smoothed = lfilter([alpha], [1.0, -(1.0 - alpha)], arr, axis=0)
    roll_cols = [f"roll_{c}" for c in float_cols]
    df[roll_cols] = smoothed.astype(np.float32, copy=False)


add_new_features(train)
add_new_features(test)




## === cell 6
scale_cols = [
    col
    for col in train.columns
    if col not in ["id", "target"] and pd.api.types.is_numeric_dtype(train[col])
]

sc = StandardScaler()
scaled_train = sc.fit_transform(train[scale_cols].values.astype(np.float32))
scaled_test = sc.transform(test[scale_cols].values.astype(np.float32))

train[scale_cols] = scaled_train.astype(np.float32)
test[scale_cols] = scaled_test.astype(np.float32)




## === cell 7
labels = train["target"].values
combined_features = [col for col in train.columns if col not in ["id", "target"]]




## === cell 8
X_flat = train[combined_features].values.astype(np.float32)
X_test_flat = test[combined_features].values.astype(np.float32)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4135141960.py in <cell line: 0>()
      1 # Build flat feature matrices for GradientBoosting
----> 2 X_flat = train[combined_features].values.astype(np.float32)
      3 X_test_flat = test[combined_features].values.astype(np.float32)
      4 
      5 

ValueError: could not convert string to float: 'ACBADABECB'

## === cell 9
cv = 2  # number of folds
random_states = np.linspace(789, 9876, cv).astype(int)
print("random_states:", random_states)

pred_list = []


def run_fold(rs):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_flat, labels, test_size=0.06, random_state=int(rs)
    )
    gbdt = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=int(rs),
    )
    gbdt.fit(X_tr, y_tr)
    val_pred = gbdt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"Fold rs={rs} – Validation AUC: {auc:.5f}")
    test_pred = gbdt.predict_proba(X_test_flat)[:, 1]
    return scipy.stats.rankdata(test_pred)


fold_results = Parallel(n_jobs=cv, backend="loky")(
    delayed(run_fold)(rs) for rs in random_states
)

pred_list.extend(fold_results)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 490, in _process_worker
    r = call_item()
        ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 291, in __call__
    return self.fn(*self.args, **self.kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/34364995.py", line 11, in run_fold
NameError: name 'X_flat' is not defined
"""

The above exception was the direct cause of the following exception:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/34364995.py in <cell line: 0>()
     27 
     28 
---> 29 fold_results = Parallel(n_jobs=cv, backend="loky")(
     30     delayed(run_fold)(rs) for rs in random_states
     31 )

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

NameError: name 'X_flat' is not defined

## === cell 10
mean_cnn = np.mean(pred_list, axis=0)




## === cell 11
sample_submission["target"] = mean_cnn
sample_submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")

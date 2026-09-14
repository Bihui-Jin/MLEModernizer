# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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



## === cell 1
train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
sample_submission = pd.read_csv(
    "../input/tabular-playground-series-may-2022/sample_submission.csv"
)



## === cell 2
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



## === cell 3
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



## === cell 4
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



## === cell 5
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

cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0).astype("category")
    train[col] = combined.iloc[: len(train)].cat.codes.astype(np.int32)
    test[col] = combined.iloc[len(train) :].cat.codes.astype(np.int32)



## === cell 6
labels = train["target"].values
combined_features = [col for col in train.columns if col not in ["id", "target"]]



## === cell 7
X_flat = train[combined_features].values.astype(np.float32)
X_test_flat = test[combined_features].values.astype(np.float32)



## === cell 8
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



## === cell 9
mean_cnn = np.mean(pred_list, axis=0)



## === cell 10
sample_submission["target"] = mean_cnn
sample_submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")

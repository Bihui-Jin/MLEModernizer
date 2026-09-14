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




## === cell 2
for df in [train, test]:
    sliced = df["f_27"].str.slice(0, 10).fillna("")
    arr = np.frombuffer(sliced.str.encode("utf-8").values.tobytes(), dtype=np.uint8)
    arr = arr.reshape(-1, 10) - ord("A")
    chars_df = pd.DataFrame(arr, index=df.index, columns=[f"ch{i}" for i in range(10)])
    df[chars_df.columns] = chars_df
    df["unique_characters"] = sliced.apply(lambda s: len(set(s)))


features = [f for f in test.columns if f != "id" and f != "f_27"]
test[features].head(2)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2852738597.py in <cell line: 0>()
      8     # Reshape to (n_rows, 10) and compute offsets
      9     arr = arr.reshape(-1, 10) - ord("A")
---> 10     chars_df = pd.DataFrame(arr, index=df.index, columns=[f"ch{i}" for i in range(10)])
     11     df[chars_df.columns] = chars_df
     12     df["unique_characters"] = sliced.apply(lambda s: len(set(s)))

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    825                 )
    826             else:
--> 827                 mgr = ndarray_to_mgr(
    828                     data,
    829                     index,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in ndarray_to_mgr(values, index, columns, dtype, copy, typ)
    334     )
    335 
--> 336     _check_values_indices_shape_match(values, index, columns)
    337 
    338     if typ == "array":

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _check_values_indices_shape_match(values, index, columns)
    418         passed = values.shape
    419         implied = (len(index), len(columns))
--> 420         raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")
    421 
    422 

ValueError: Shape of passed values is (640000, 10), indices imply (800000, 10)

## === cell 3
from sklearn.preprocessing import StandardScaler




## === cell 4
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)




## === cell 5
stat = ["mean", "std", "sum", "var", "euc", "ptp", "logsumexp"]




## === cell 6
import scipy.special
from scipy.signal import lfilter  # fast exponential smoothing

cols = train.dtypes[train.dtypes == float].index.to_list()
roll_cols = [f"roll_{c}" for c in cols]


def new_feats(df):
    df_float = df[cols].astype(np.float32, copy=False)
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
    df[roll_cols] = smoothed.astype(np.float32, copy=False)


new_feats(train)
new_feats(test)




## === cell 7
rolls = [f for f in test.columns if f.startswith("roll")]




## === cell 8
scale_cols = [f for f in test.columns if train[f].dtype == float]




## === cell 9
sc = StandardScaler()
scaled_train = sc.fit_transform(train[scale_cols].values.astype(np.float32), copy=False)
scaled_test = sc.transform(test[scale_cols].values.astype(np.float32), copy=False)
train[scale_cols] = scaled_train.astype(np.float32, copy=False)
test[scale_cols] = scaled_test.astype(np.float32, copy=False)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3555935486.py in <cell line: 0>()
      1 # Apply scaling in‑place to avoid extra memory copies
      2 sc = StandardScaler()
----> 3 scaled_train = sc.fit_transform(train[scale_cols].values.astype(np.float32), copy=False)
      4 scaled_test = sc.transform(test[scale_cols].values.astype(np.float32), copy=False)
      5 train[scale_cols] = scaled_train.astype(np.float32, copy=False)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    876         if y is None:
    877             # fit method of arity 1 (unsupervised transformation)
--> 878             return self.fit(X, **fit_params).transform(X)
    879         else:
    880             # fit method of arity 2 (supervised transformation)

TypeError: StandardScaler.fit() got an unexpected keyword argument 'copy'

## === cell 10
train[features + stat + rolls].values.reshape(8, 8, train.shape[0], 1).shape




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/89723638.py in <cell line: 0>()
----> 1 train[features + stat + rolls].values.reshape(8, 8, train.shape[0], 1).shape
      2 
      3 

NameError: name 'features' is not defined

## === cell 11
traincon = train[features + stat + rolls].values.reshape(train.shape[0], 8, 8)
testcon = test[features + stat + rolls].values.reshape(test.shape[0], 8, 8)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1941487050.py in <cell line: 0>()
----> 1 traincon = train[features + stat + rolls].values.reshape(train.shape[0], 8, 8)
      2 testcon = test[features + stat + rolls].values.reshape(test.shape[0], 8, 8)
      3 
      4 

NameError: name 'features' is not defined

## === cell 12
labels = train.target




## === cell 13
import random




## === cell 14
train_cols = test.columns.to_list()
train_cols.remove("id")




## === cell 15
len(features + stat + rolls)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3312130527.py in <cell line: 0>()
----> 1 len(features + stat + rolls)
      2 
      3 

NameError: name 'features' is not defined

## === cell 16
import warnings

warnings.filterwarnings("ignore")




## === cell 17
from sklearn.metrics import roc_curve


def plot_loss_auc(history, y_true, prediction):
    """
    history: dummy placeholder for compatibility
    y_true: true validation set or test set labels
    prediction: prediction on val set or test set
    """
    fp, tp, _ = roc_curve(y_true, prediction)
    _, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 3))
    ax[0].plot(fp, tp, label="ROC", linewidth=2)
    ax[0].set_xlabel("False positives")
    ax[0].set_ylabel("True positives")
    ax[0].set_title("ROC")
    plt.show()




## === cell 18
from sklearn.metrics import confusion_matrix, classification_report


def plot_cm(y_true, prediction, p=0.5):
    """
    y_true: true validation set or test set labels
    prediction: probability predictions on val set
    """
    cm = confusion_matrix(y_true, prediction > p)
    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title("Confusion matrix @{:.2f}".format(p))
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()




## === cell 19
class reps(tf.keras.callbacks.Callback if tf is not None else object):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        auc = logs.get("auc")
        val_auc = logs.get("val_auc")
        loss_str = f"{loss:.4f}" if loss is not None else "nan"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "nan"
        auc_str = f"{auc:.4f}" if auc is not None else "nan"
        val_auc_str = f"{val_auc:.4f}" if val_auc is not None else "nan"
        print(
            f"epoch : {epoch}, loss : {loss_str}, val loss : {val_loss_str}, auc : {auc_str}, val auc : {val_auc_str}"
        )


report_callback = reps()




## === cell 20
epochs = 50
exp = -0.05
lr = 0.001  # lr at begin
start = 10  # epoch
mid = 17  # epoch


def lr_decay(epoch, lr):
    if epoch < start:
        return lr
    elif epoch < mid:
        return (
            lr * (1 + np.cos(epoch / (epochs - 1.5) * np.pi)) / 1.8
        )  # from reference 1 notebook
    else:
        return lr * tf.math.exp(exp) if tf is not None else lr


def plot_lr_decay(epochs, lr):
    x = np.arange(0, epochs)
    lrs = []
    lr2 = lr
    for epoch in x:
        lr = lr_decay(epoch, lr)
        lrs.append(lr)
    y = np.array(lrs)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.vlines(
        x=start - 1, linestyles="--", colors="g", ymin=y[-1], ymax=lr2, linewidth=0.95
    )
    plt.vlines(
        x=mid - 1,
        linestyles="--",
        colors="orange",
        ymin=y[-1],
        ymax=lr2,
        linewidth=0.95,
    )
    plt.vlines(x=35, linestyles="--", colors="r", ymin=y[-1], ymax=lr2, linewidth=0.95)
    plt.hlines(
        y=0, linestyles="--", colors="r", xmin=start - 1, xmax=50, linewidth=0.95
    )
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.title("learning rate decay")




## === cell 21
from tensorflow.keras.layers import (
    Conv1D,
    Conv2D,
    Flatten,
    Dropout,
    Conv1DTranspose,
    Conv2DTranspose,
    Dense,
    Reshape,
    GlobalAveragePooling1D,
)
from tensorflow.keras import Input
from tensorflow import keras




## === cell 22
lr = 0.001
input_shape = (8, 8)


def cnn2():

    model = keras.Sequential(
        [
            Input(shape=input_shape),
            Reshape(target_shape=(8, 8, 1)),
            Conv2D(
                filters=144,
                kernel_size=(7, 7),
                padding="same",
                strides=2,
                activation="relu",
            ),
            Dropout(rate=0.1),
            Reshape(target_shape=(4 * 12, 4 * 12)),
            Conv1D(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Conv1DTranspose(
                filters=36, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Conv1DTranspose(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Dropout(rate=0.1),
            Conv1DTranspose(
                filters=128, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Flatten(),
            Dropout(rate=0.4),
            Dense(1, activation="sigmoid"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr) if tf is not None else None
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["AUC"])
    return model




## === cell 23
model = cnn2()
if model is not None:
    model.summary()




## === cell 24
import scipy.stats
from sklearn.model_selection import KFold, train_test_split
from joblib import Parallel, delayed




## === cell 25
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

epochs = 45  # kept for naming consistency
batch_size = 1024  # not used by sklearn
pred_list = []
cv = 2  # number of folds / splits
random_states = np.linspace(789, 9876, cv).astype(int)
print("random_states:", random_states)

X_flat = traincon.reshape(train.shape[0], -1).astype(np.float32)
X_test_flat = testcon.reshape(test.shape[0], -1).astype(np.float32)

combined_features = features + stat + rolls


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




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/327427657.py in <cell line: 0>()
      9 print("random_states:", random_states)
     10 
---> 11 X_flat = traincon.reshape(train.shape[0], -1).astype(np.float32)
     12 X_test_flat = testcon.reshape(test.shape[0], -1).astype(np.float32)
     13 

NameError: name 'traincon' is not defined

## === cell 26
mean_cnn = np.array(pred_list).mean(axis=0)




## === cell 27
sample_submission["target"] = mean_cnn
sample_submission.to_csv("submission.csv", index=False)
sample_submission

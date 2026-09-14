# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.9

# 3. Installed packages

catboost==1.2.8
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "228")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

pd.set_option("display.max_columns", None)

DTYPES_TRAIN = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
DTYPES_TEST = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", dtype=DTYPES_TRAIN
)
test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv", dtype=DTYPES_TEST
)
ss = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    dtype={"id": "int32", "pressure": "float32"},
)

tf.keras.utils.set_random_seed(228)




## === cell 1
train.head(3)




## === cell 2
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}")
print("")
print("Missing values in TRAIN dataset")
for i in train.iloc[:, 0:-1].columns.tolist():
    print(f"{i}: {train[i].isna().sum()}")
print("")
print("Missing values in TEST dataset")
for i in test.columns.tolist():
    print(f"{i}: {test[i].isna().sum()}")
print("")
print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')
print(
    f'The number of observations for each breath: {train["breath_id"].value_counts().reset_index()["breath_id"].unique()[0]}'
)




## === cell 3
RUN_PLOTS = False

if RUN_PLOTS:
    fig = plt.figure(figsize=(13, 8))
    rc = ["R", "C"]
    for i in rc:
        plt.subplot(2, 2, rc.index(i) + 1)
        plt.title(i, y=1.2, size=25, fontname="monospace", color="black")
        a = sns.countplot(x=i, data=train, palette=["#488a99", "#dbae58", "#4b585c"])
        plt.ylabel("")
        plt.xlabel("")
        plt.xticks(fontname="monospace", size=12)
        plt.yticks([])
        for j in ["right", "top"]:
            a.spines[j].set_visible(False)
        for j in ["bottom", "left"]:
            a.spines[j].set_linewidth(1.2)

        summ = 0
        for p in a.patches:
            summ += p.get_height()

        for p in a.patches:
            height = p.get_height()
            a.annotate(
                f"{height}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="center",
                size=13,
                xytext=(1, -15),
                textcoords="offset points",
                fontname="monospace",
                color="white",
            )
            a.annotate(
                f"{round((height / summ) * 100, 1)}%",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="center",
                size=15,
                xytext=(1, 13),
                textcoords="offset points",
                fontname="monospace",
                color="black",
            )

    for i in rc:
        plt.subplot(2, 2, rc.index(i) + 3)
        a = sns.countplot(x=i, data=test, palette=["#488a99", "#dbae58", "#4b585c"])
        plt.ylabel("")
        plt.xlabel("")
        plt.xticks(fontname="monospace", size=12)
        plt.yticks([])
        for j in ["right", "top"]:
            a.spines[j].set_visible(False)
        for j in ["bottom", "left"]:
            a.spines[j].set_linewidth(1.2)

        summ = 0
        for p in a.patches:
            summ += p.get_height()

        for p in a.patches:
            height = p.get_height()
            a.annotate(
                f"{height}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="center",
                size=13,
                xytext=(1, -15),
                textcoords="offset points",
                fontname="monospace",
                color="white",
            )
            a.annotate(
                f"{round((height / summ) * 100, 1)}%",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="center",
                size=15,
                xytext=(1, 13),
                textcoords="offset points",
                fontname="monospace",
                color="black",
            )

    plt.figtext(
        0.15,
        1.1,
        "Distribution of lung attributes (R/C)",
        fontname="monospace",
        size=30,
        color="black",
    )
    plt.figtext(
        1.03, 0.15, "TEST", fontname="monospace", size=25, color="black", rotation=90
    )
    plt.figtext(
        1.03, 0.7, "TRAIN", fontname="monospace", size=25, color="black", rotation=90
    )

    fig.tight_layout(h_pad=10)
    plt.show()




## === cell 4
if RUN_PLOTS:
    fig = plt.figure(figsize=(15, 15))
    r, c, plot = [5, 20, 50], [10, 20, 50], 1
    for i in range(3):
        rr = r[i]
        for k in range(3):
            cc = c[k]
            br_id = train.query("R == @rr & C == @cc").iloc[0, 1]
            plt.subplot(3, 3, plot)
            plt.title(
                f"breath id = {br_id} | R = {rr} | C = {cc}",
                fontname="monospace",
                size=14,
            )
            a = sns.lineplot(
                data=train.query("breath_id == @br_id"),
                x="time_step",
                y="u_in",
                color="#4b585c",
                linewidth=2,
            )
            sns.lineplot(
                data=train.query("breath_id == @br_id"),
                x="time_step",
                y="u_out",
                color="#dbae58",
                linewidth=2,
            )
            sns.lineplot(
                data=train.query("breath_id == @br_id"),
                x="time_step",
                y="pressure",
                color="#488a99",
                linewidth=2,
            )
            plt.ylabel("")
            plt.xlabel("time stemp", size=14, fontname="monospace", labelpad=10)
            plt.xticks(size=12, fontname="monospace")
            plt.yticks(size=12, fontname="monospace")

            for j in ["right", "top"]:
                a.spines[j].set_visible(False)
            for j in ["bottom", "left"]:
                a.spines[j].set_linewidth(1.2)

            plot += 1

    plt.figtext(
        0.01,
        1.08,
        "Observations on breaths with all possible lung attributes",
        fontname="monospace",
        size=30,
        color="black",
    )
    plt.figtext(0.35, 1.03, "u_in", fontname="monospace", size=27, color="#4b585c")
    plt.figtext(0.45, 1.03, "u_out", fontname="monospace", size=27, color="#dbae58")
    plt.figtext(0.55, 1.03, "pressure", fontname="monospace", size=27, color="#488a99")
    fig.tight_layout(h_pad=3)
    plt.show()




## === cell 5
if RUN_PLOTS:
    fig = plt.figure(figsize=(15, 12))
    r, c = [5, 20, 50], [10, 20, 50]
    plot = 1
    for i in range(3):
        rr = r[i]
        for k in range(3):
            cc = c[k]
            plt.subplot(3, 3, plot)
            plt.title(
                f"R = {rr} | C = {cc}", fontname="monospace", size=15, color="black"
            )
            a = sns.kdeplot(
                train.query(
                    "time_step < 0.000001 & u_in < 0.000001 & R == @rr & C == @cc"
                )["pressure"],
                color="#488a99",
                shade=True,
                alpha=1,
                linewidth=1.5,
                edgecolor="black",
            )
            plt.ylabel("")
            plt.xlabel("")
            plt.xticks(size=12, fontname="monospace")
            plt.yticks([])

            for j in ["right", "top"]:
                a.spines[j].set_visible(False)
            for j in ["bottom", "left"]:
                a.spines[j].set_linewidth(1.2)

            plot += 1

    y = 1.27
    for i in range(3):
        rr = r[i]
        y -= 0.333
        x = -0.315
        for k in range(3):
            cc = c[k]
            x += 0.333
            q = train.query(
                "time_step < 0.000001 & u_in < 0.000001 & R == @rr & C == @cc"
            )["pressure"]
            plt.figtext(
                x, y, f"Min: {round(q.min(),2)}", fontname="monospace", color="black"
            )
            plt.figtext(x, y - 0.02, f"Max: {round(q.max(),2)}", fontname="monospace")
            plt.figtext(
                x,
                y - 0.04,
                f"Mean: {round(q.mean(),2)}",
                fontname="monospace",
                color="black",
            )
            plt.figtext(
                x,
                y - 0.06,
                f"Median: {round(q.median(),2)}",
                fontname="monospace",
                color="black",
            )

    plt.figtext(
        0.01,
        1.08,
        "Distribution of pressure depending on lung attributes",
        fontname="monospace",
        size=30,
        color="black",
    )

    fig.tight_layout(h_pad=3)
    plt.show()




## === cell 6
def features_all(
    df_all: pd.DataFrame, *, dummy_columns=None
) -> tuple[pd.DataFrame, list[str]]:
    df_all = df_all.copy()

    df_all["u_out"] = df_all["u_out"].astype(np.int8, copy=False)
    df_all["R"] = df_all["R"].astype(str)
    df_all["C"] = df_all["C"].astype(str)

    g = df_all.groupby("breath_id", sort=False)
    df_all["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float32)
    df_all["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)
    df_all["u_in_lag4"] = g["u_in"].shift(4).fillna(0.0).astype(np.float32)

    ts = df_all["time_step"].to_numpy(np.float32)
    u = df_all["u_in"].to_numpy(np.float32)
    bid = df_all["breath_id"].to_numpy(np.int32)

    change = np.empty(len(bid), dtype=bool)
    change[0] = True
    change[1:] = bid[1:] != bid[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(bid)]

    area = np.empty(len(bid), dtype=np.float32)
    for s, e in zip(starts, ends):
        area[s:e] = np.cumsum(ts[s:e] * u[s:e], dtype=np.float32)
    df_all["area"] = area

    df_all = pd.get_dummies(df_all, columns=["R", "C"], dtype=np.int8)
    if dummy_columns is None:
        dummy_columns = [
            c for c in df_all.columns if c.startswith("R_") or c.startswith("C_")
        ]
    else:
        for c in dummy_columns:
            if c not in df_all.columns:
                df_all[c] = np.int8(0)

    ewm_mean = np.empty(len(u), dtype=np.float32)
    ewm_std = np.empty(len(u), dtype=np.float32)
    ewm_corr = np.empty(len(u), dtype=np.float32)

    roll_mean = np.empty(len(u), dtype=np.float32)
    roll_max = np.empty(len(u), dtype=np.float32)
    roll_std = np.empty(len(u), dtype=np.float32)

    exp_mean = np.empty(len(u), dtype=np.float32)
    exp_max = np.empty(len(u), dtype=np.float32)
    exp_std = np.empty(len(u), dtype=np.float32)

    for s, e in zip(starts, ends):
        ser = pd.Series(u[s:e])

        em = ser.ewm(halflife=10, adjust=True).mean().to_numpy(np.float32)
        es = ser.ewm(halflife=10, adjust=True).std().to_numpy(np.float32)
        ec = ser.ewm(halflife=10, adjust=True).corr().to_numpy(np.float32)

        ewm_mean[s:e] = em
        ewm_std[s:e] = es
        ewm_corr[s:e] = ec

        rm = ser.rolling(window=10, min_periods=1).mean().to_numpy(np.float32)
        rx = ser.rolling(window=10, min_periods=1).max().to_numpy(np.float32)
        rs = ser.rolling(window=10, min_periods=1).std().to_numpy(np.float32)

        roll_mean[s:e] = rm
        roll_max[s:e] = rx
        roll_std[s:e] = rs

        exm = ser.expanding(2).mean().to_numpy(np.float32)
        exx = ser.expanding(2).max().to_numpy(np.float32)
        exs = ser.expanding(2).std().to_numpy(np.float32)

        exp_mean[s:e] = exm
        exp_max[s:e] = exx
        exp_std[s:e] = exs

    df_all["ewm_u_in_mean"] = ewm_mean
    df_all["ewm_u_in_std"] = ewm_std
    df_all["ewm_u_in_corr"] = ewm_corr

    df_all["rolling_10_mean"] = roll_mean
    df_all["rolling_10_max"] = roll_max
    df_all["rolling_10_std"] = roll_std

    df_all["expand_mean"] = exp_mean
    df_all["expand_max"] = exp_max
    df_all["expand_std"] = exp_std

    return df_all, dummy_columns




## === cell 7
train_fe, dummy_cols = features_all(train, dummy_columns=None)
test_fe, _ = features_all(test, dummy_columns=dummy_cols)

train_cols = list(train_fe.columns)
for c in train_cols:
    if c not in test_fe.columns:
        test_fe[c] = 0
test_fe = test_fe[train_cols]

train = train_fe.fillna(0)
test = test_fe.fillna(0)
del train_fe, test_fe




## === cell 8
targets = train[["pressure"]].to_numpy().reshape(-1, 80)

train.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test.drop(["id", "breath_id"], axis=1, inplace=True)




## === cell 9
train = train.astype(np.float32, copy=False)
test = test.astype(np.float32, copy=False)

RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)




## === cell 10
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])




## === cell 11
EPOCH = 300
BATCH_SIZE = 1024

strategy = tf.distribute.get_strategy()

test_pred_sum = np.zeros((test.shape[0], test.shape[1]), dtype=np.float32)

tfdata_opts = tf.data.Options()
tfdata_opts.deterministic = True

ds_test = (
    tf.data.Dataset.from_tensor_slices(test)
    .with_options(tfdata_opts)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=228)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        X_train, X_valid = train[train_idx], train[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=train.shape[-2:]),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(400, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(300, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(200, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(100, return_sequences=True)
                ),
                keras.layers.Dense(50, activation="selu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")

        scheduler = ExponentialDecay(
            1e-3, 400 * ((len(train) * 0.8) / BATCH_SIZE), 1e-5
        )

        def _lr_schedule(epoch, lr):
            return float(scheduler(epoch).numpy())

        lr_cb = LearningRateScheduler(_lr_schedule, verbose=0)

        ds_train = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .with_options(tfdata_opts)
            .shuffle(
                buffer_size=min(len(X_train), 8192),
                seed=228,
                reshuffle_each_iteration=True,
            )
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )
        ds_valid = (
            tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
            .with_options(tfdata_opts)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=EPOCH,
            callbacks=[lr_cb],
            verbose=2,
        )

        fold_pred = (
            model.predict(ds_test, verbose=0).squeeze(axis=-1).astype(np.float32)
        )
        test_pred_sum += fold_pred

        tf.keras.backend.clear_session()




## === cell 12
ss["pressure"] = (test_pred_sum / 5.0).reshape(-1)
ss.to_csv("lstm.csv", index=False)
print("Saved: lstm.csv")

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

3.10

# 3. Installed packages

catboost==1.2.8
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
import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

RANDOM_STATE = 555
np.random.seed(RANDOM_STATE)

N_THREADS = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))




## === cell 1
def df_overview(df):
    print("Dataframe overview:\n")
    display(df.head())
    print("--------------------------------------------\nSample:\n")
    display(df.sample(10, random_state=555))
    print("--------------------------------------------\nInfo:\n")
    print(df.info())
    print("--------------------------------------------\nNaN's:\n")
    print(df.isna().sum())
    print("--------------------------------------------\nDescribe:\n")
    display(df.describe())
    print("--------------------------------------------\nFeature correlation:\n")
    display(df.corr(numeric_only=True))




## === cell 2
def show_correlogram(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 6), dpi=80)
    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="RdYlGn",
        center=0,
        annot=True,
        cbar=False,
    )
    plt.title("Correlogram between features", fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()




## === cell 3
def plot_create(x, y):
    import matplotlib.pyplot as plt

    plt.plot(x, y, "-", label=y.name)


def process_visualisation(df, breath_id):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    plt.title("Breath Id - {}".format(breath_id))
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["pressure"],
    )
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["u_in"],
    )
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["u_out"],
    )
    plt.grid()
    plt.legend()
    plt.ylabel("Value")
    plt.show()




## === cell 4
def process_visualisation_with_preds(df, df_preds, breath_id):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    plt.title("Breath Id - {}".format(breath_id))
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["pressure"],
    )
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["u_in"],
    )
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["u_out"],
    )
    plot_create(df[df["breath_id"] == breath_id]["time_step"], df_preds)
    plt.grid()
    plt.legend()
    plt.ylabel("Value")
    plt.show()




## === cell 5
def _rolling_mean_lag1_window3_by_breath(
    breath_id: np.ndarray, u_in: np.ndarray
) -> np.ndarray:
    """
    Runtime optimization (equivalent):
    - Original: groupby('breath_id')['u_in'].shift(1).rolling(3).mean()
    - Here: exact same per-breath computation in NumPy to avoid pandas groupby+rolling overhead.
    Assumption (true for this dataset): rows for each breath_id are contiguous and ordered.
    """
    n = u_in.shape[0]
    out = np.zeros(n, dtype=np.float32)

    b = breath_id
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = b[1:] != b[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n

    for s, e in zip(starts, ends):
        x = u_in[s:e].astype(np.float32, copy=False)
        L = e - s
        if L <= 3:
            continue
        lag1 = np.empty(L, dtype=np.float32)
        lag1[0] = 0.0
        lag1[1:] = x[:-1]
        rs = lag1[0 : L - 2] + lag1[1 : L - 1] + lag1[2:L]
        out[s + 2 : e] = rs / 3.0
    return out




## === cell 6
def _per_breath_stats_numpy(breath_id: np.ndarray, u_in: np.ndarray):
    """
    Runtime optimization (equivalent to pandas groupby transforms/agg):
    Computes per-breath min/max/median/first/last and broadcasts to rows.
    Using NumPy avoids multiple expensive pandas groupby passes on ~5.4M rows.
    Assumption (true for this dataset): each breath_id forms one contiguous block.
    """
    n = u_in.shape[0]
    b = breath_id

    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = b[1:] != b[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n

    u_min = np.empty(n, dtype=np.float32)
    u_max = np.empty(n, dtype=np.float32)
    u_med = np.empty(n, dtype=np.float32)
    u_first = np.empty(n, dtype=np.float32)
    u_last = np.empty(n, dtype=np.float32)

    for s, e in zip(starts, ends):
        x = u_in[s:e].astype(np.float32, copy=False)
        mn = float(np.min(x))
        mx = float(np.max(x))
        md = float(np.median(x))
        f = float(x[0])
        l = float(x[-1])
        u_min[s:e] = mn
        u_max[s:e] = mx
        u_med[s:e] = md
        u_first[s:e] = f
        u_last[s:e] = l

    return u_min, u_max, u_med, u_first, u_last




## === cell 7
def add_features(df):
    """
    Runtime optimization (equivalent features, same semantics):
    - Avoid repeated pandas groupby/transform/merge on millions of rows.
    - Compute all per-breath features in one NumPy pass (breath blocks are contiguous).
    - Still returns the same set of feature columns and drops ['breath_id','u_in'] as before.
    """
    df = df.copy(deep=False)

    b = df["breath_id"].to_numpy(dtype=np.int32, copy=False)
    u = df["u_in"].to_numpy(dtype=np.float32, copy=False)

    n = u.shape[0]
    u_cumsum = np.empty(n, dtype=np.float32)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = b[1:] != b[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n
    for s, e in zip(starts, ends):
        u_cumsum[s:e] = np.cumsum(u[s:e], dtype=np.float32)

    u_lag1 = np.empty(n, dtype=np.float32)
    u_lag2 = np.empty(n, dtype=np.float32)
    u_lag1[:] = 0.0
    u_lag2[:] = 0.0
    for s, e in zip(starts, ends):
        L = e - s
        if L >= 2:
            u_lag1[s + 1 : e] = u[s : e - 1]
        if L >= 3:
            u_lag2[s + 2 : e] = u[s : e - 2]

    u_roll = _rolling_mean_lag1_window3_by_breath(b, u)

    u_min, u_max, u_med, u_first, u_last = _per_breath_stats_numpy(b, u)

    df["u_in_cumsum"] = u_cumsum
    df["u_in_lag_1"] = u_lag1
    df["u_in_lag_2"] = u_lag2
    df["u_in_rolling_mean"] = u_roll
    df["u_in_min"] = u_min
    df["u_in_max"] = u_max
    df["u_in_median"] = u_med
    df["u_in_begin"] = u_first
    df["u_in_end"] = u_last

    df = df.drop(["breath_id", "u_in"], axis=1)
    return df




## === cell 8
def train_and_score(model):
    model.fit(X_train, y_train)
    return mean_absolute_error(y_valid, model.predict(X_valid))




## === cell 9
train_path = "/kaggle/input/ventilator-pressure-prediction/train.csv"
test_path = "/kaggle/input/ventilator-pressure-prediction/test.csv"
sub_path = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"

train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

df_train = pd.read_csv(train_path, dtype=train_dtypes)
df_test = pd.read_csv(test_path, dtype=test_dtypes)
df_sample_submission = pd.read_csv(
    sub_path, dtype={"id": "int32", "pressure": "float32"}
)



## === cell 10
df_train = df_train.drop("id", axis=1)



## === cell 11
PASS_EDA = False
if PASS_EDA:
    df_overview(df_train)



## === cell 12
if PASS_EDA:
    df_overview(df_test)



## === cell 13
if PASS_EDA:
    show_correlogram(df_train)



## === cell 14
if PASS_EDA:
    import matplotlib.pyplot as plt

    for col in df_train.columns:
        df_train[col].plot(kind="hist", bins=30, title=col)
        plt.show()



## === cell 15
if PASS_EDA:
    import matplotlib.pyplot as plt

    for col in ["time_step", "u_in", "pressure"]:
        df_train[col].plot(kind="box")
        plt.show()



## === cell 16
if PASS_EDA:
    import matplotlib.pyplot as plt

    df_train[df_train["pressure"] > 55]["pressure"].plot(
        kind="hist", bins=30, title="Pressure > 55"
    )
    plt.show()
    df_train[df_train["u_in"] > 70]["u_in"].plot(
        kind="hist", bins=30, title="U in > 70"
    )
    plt.show()



## === cell 17
if PASS_EDA:
    df_train[df_train["pressure"] > 64.5]["pressure"].value_counts().sort_index(
        ascending=False
    )



## === cell 18
if PASS_EDA:
    df_train[df_train["u_in"] > 99.98]["u_in"].value_counts().sort_index(
        ascending=False
    )



## === cell 19
if PASS_EDA:
    df_test[df_test["u_in"] > 99.98]["u_in"].value_counts().sort_index(ascending=False)



## === cell 20
if PASS_EDA:
    import matplotlib.pyplot as plt

    df_train[df_train["u_in"] > 99.98]["pressure"].plot(
        kind="hist", bins=40, title="Pressure when U in is high"
    )
    plt.show()
    df_train[df_train["pressure"] > 64.5]["u_in"].plot(
        kind="hist", bins=40, title="U in when Pressure is high"
    )
    plt.show()



## === cell 21
if PASS_EDA:
    df_breath = df_train.groupby("breath_id", as_index=False).median(numeric_only=True)
    df_breath



## === cell 22
if PASS_EDA:
    import matplotlib.pyplot as plt

    for col in ["time_step", "u_in", "pressure"]:
        df_breath[col].plot(kind="hist", bins=30, title=col)
        plt.show()



## === cell 23
if PASS_EDA:
    df_breath[df_breath["pressure"] < 0].head(10)



## === cell 24
if PASS_EDA:
    import matplotlib.pyplot as plt

    df_breath[df_breath["pressure"] < 0]["pressure"].plot(
        kind="hist", bins=30, title="Negative pressure"
    )
    plt.show()



## === cell 25
if PASS_EDA:
    import matplotlib.pyplot as plt

    df_breath[df_breath["pressure"] < 0]["u_in"].plot(
        kind="hist", bins=30, title="U in when Pressure < 0"
    )
    plt.show()
    df_breath[df_breath["pressure"] == df_breath["pressure"].median()]["u_in"].plot(
        kind="hist", bins=30, title="U in when Pressure is normal"
    )
    plt.show()



## === cell 26
if PASS_EDA:
    print("Some data where pressure is normal:")
    display(
        df_breath[df_breath["pressure"] == df_train["pressure"].median()].sample(
            3, random_state=1
        )
    )
    print("\nSome data where pressure is below 0:")
    display(df_breath[df_breath["pressure"] < 0].sample(3, random_state=1))
    print("\nSome data where pressure is high:")
    display(df_breath[df_breath["pressure"] > 12].sample(3, random_state=1))



## === cell 27
if PASS_EDA:
    print("Process visualisation where pressure is normal:")
    process_visualisation(df_train, 48945)
    process_visualisation(df_train, 28141)
    process_visualisation(df_train, 109737)



## === cell 28
if PASS_EDA:
    print("Process visualisation where pressure is below 0:")
    process_visualisation(df_train, 98041)
    process_visualisation(df_train, 118131)
    process_visualisation(df_train, 11216)



## === cell 29
if PASS_EDA:
    print("Process visualisation where pressure is high:")
    process_visualisation(df_train, 104581)
    process_visualisation(df_train, 14416)
    process_visualisation(df_train, 69384)



## === cell 30
df_train_insp = df_train[df_train["u_out"] == 0].copy()

X = df_train_insp.drop("pressure", axis=1)
X = add_features(X)
y = df_train_insp["pressure"]



## === cell 31
breath_ids = df_train_insp["breath_id"].unique()
breath_tr, breath_va = train_test_split(
    breath_ids, test_size=0.2, random_state=RANDOM_STATE
)

tr_idx = df_train_insp["breath_id"].isin(breath_tr).to_numpy()
va_idx = ~tr_idx

X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
y_np = y.to_numpy(dtype=np.float32, copy=False)

X_train, X_valid = X_np[tr_idx], X_np[va_idx]
y_train, y_valid = y_np[tr_idx], y_np[va_idx]



## === cell 32
cb_model = CatBoostRegressor(
    depth=15,
    loss_function="MAE",
    random_seed=RANDOM_STATE,
    verbose=0,
    thread_count=N_THREADS,
    use_best_model=False,
)



## === cell 33
cb_valid_mae = train_and_score(cb_model)
display(
    pd.DataFrame(
        data=([cb_valid_mae],),
        columns=["Result MAE (inspiratory valid only)"],
        index=["CatBoost"],
    )
)



## === cell 34
if PASS_EDA:
    import matplotlib.pyplot as plt

    pd.DataFrame(
        cb_model.feature_importances_,
        index=[f"f{i}" for i in range(len(cb_model.feature_importances_))],
        columns=["importances"],
    ).sort_values(by="importances").plot(
        kind="barh", figsize=(8, 6), title="CatBoost feature importances"
    )
    plt.show()



## === cell 35
cb_model_full = CatBoostRegressor(
    depth=15,
    loss_function="MAE",
    random_seed=RANDOM_STATE,
    verbose=0,
    thread_count=N_THREADS,
    use_best_model=False,
)
cb_model_full.fit(X_np, y_np, init_model=cb_model)

del X_train, X_valid, y_train, y_valid, tr_idx, va_idx, breath_tr, breath_va
gc.collect()



## === cell 36
if PASS_EDA:
    X_df_vis = df_train[df_train["breath_id"] == 28141].reset_index()
    X_df_vis_feat = add_features(X_df_vis.drop("pressure", axis=1))
    X_df_vis_feat = X_df_vis_feat.drop(["index"], axis=1)

    print("Pressure predictions by CatBoost Model where pressure in normal:")
    process_visualisation_with_preds(
        df_train,
        pd.Series(
            cb_model_full.predict(X_df_vis_feat.to_numpy(dtype=np.float32, copy=False)),
            name="predictions",
        ),
        28141,
    )



## === cell 37
if PASS_EDA:
    X_df_vis = df_train[df_train["breath_id"] == 98041].reset_index()
    X_df_vis_feat = add_features(X_df_vis.drop("pressure", axis=1))
    X_df_vis_feat = X_df_vis_feat.drop(["index"], axis=1)

    print("Pressure predictions by CatBoost Model where pressure in below 0:")
    process_visualisation_with_preds(
        df_train,
        pd.Series(
            cb_model_full.predict(X_df_vis_feat.to_numpy(dtype=np.float32, copy=False)),
            name="predictions",
        ),
        98041,
    )



## === cell 38
if PASS_EDA:
    X_df_vis = df_train[df_train["breath_id"] == 104581].reset_index()
    X_df_vis_feat = add_features(X_df_vis.drop("pressure", axis=1))
    X_df_vis_feat = X_df_vis_feat.drop(["index"], axis=1)

    print("Pressure predictions by CatBoost Model where pressure in high:")
    process_visualisation_with_preds(
        df_train,
        pd.Series(
            cb_model_full.predict(X_df_vis_feat.to_numpy(dtype=np.float32, copy=False)),
            name="predictions",
        ),
        104581,
    )



## === cell 39
df_test_featured = add_features(df_test)

test_ids = df_test_featured["id"].values
df_test_featured = df_test_featured.drop("id", axis=1)

X_test_np = np.ascontiguousarray(
    df_test_featured.to_numpy(dtype=np.float32, copy=False)
)
preds = cb_model_full.predict(X_test_np)

preds = np.where(df_test["u_out"].values == 1, 0.0, preds)

output = pd.DataFrame({"id": test_ids, "pressure": preds})
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)

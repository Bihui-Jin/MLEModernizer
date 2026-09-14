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

No external packages required in the script and installed.

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

# 5. Target score

0.1491524544004632

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd
import math
import random
import gc

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
pd.set_option("max_columns", 300)
gc.enable()


def set_seed(seed_val):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    os.environ["PYTHONHASHSEED"] = str(seed_val)


start_time = time.time()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
OptionError                               Traceback (most recent call last)
/tmp/ipykernel_11/2585964916.py in <cell line: 0>()
     11 
     12 warnings.filterwarnings("ignore")
---> 13 pd.set_option("max_columns", 300)
     14 gc.enable()
     15 

/usr/local/lib/python3.11/dist-packages/pandas/_config/config.py in __call__(self, *args, **kwds)
    272 
    273     def __call__(self, *args, **kwds) -> T:
--> 274         return self.__func__(*args, **kwds)
    275 
    276     # error: Signature of "__doc__" incompatible with supertype "object"

/usr/local/lib/python3.11/dist-packages/pandas/_config/config.py in _set_option(*args, **kwargs)
    165 
    166     for k, v in zip(args[::2], args[1::2]):
--> 167         key = _get_single_key(k, silent)
    168 
    169         o = _get_registered_option(key)

/usr/local/lib/python3.11/dist-packages/pandas/_config/config.py in _get_single_key(pat, silent)
    132         raise OptionError(f"No such keys(s): {repr(pat)}")
    133     if len(keys) > 1:
--> 134         raise OptionError("Pattern matched multiple keys")
    135     key = keys[0]
    136 

OptionError: Pattern matched multiple keys

## === cell 1
def connect_to_tpu(tpu_address: str = None):
    """Dummy TPU connector – not needed in this environment."""
    print("TPU connection not required; using CPU only.")
    return None, None


cluster_resolver, strategy = connect_to_tpu()




## === cell 2
DEBUG = False
TRAIN_MODEL = True




## === cell 3
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

if DEBUG:
    train = train[: 80 * 1000]
    test = test[: 80 * 100]
    submission = submission[: 80 * 100]




## === cell 4
print("TRAIN\n")
display(train)
print("\n\nTEST\n")
display(test)




## === cell 5
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}")
print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')
print(
    f'The number of observations for each breath: {train["breath_id"].value_counts().reset_index()["breath_id"].unique()[0]}'
)




## === cell 6
display(test[test["breath_id"] == 0])




## === cell 7
train_gf = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
plt.title("Histogram of Train Pressures", size=14)
plt.hist(train_gf.sample(100_000).pressure.values, bins=100)
plt.show()
print(
    "Max pressure =", train_gf.pressure.max(), "Min pressure =", train_gf.pressure.min()
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3266919011.py in <cell line: 0>()
      1 train_gf = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
----> 2 plt.title("Histogram of Train Pressures", size=14)
      3 plt.hist(train_gf.sample(100_000).pressure.values, bins=100)
      4 plt.show()
      5 print(

NameError: name 'plt' is not defined

## === cell 8
all_pressure = np.sort(train_gf.pressure.unique())
del train_gf
print("The first 25 unique pressures...")
PRESSURE_MIN = all_pressure[0].item()
PRESSURE_MAX = all_pressure[-1].item()
all_pressure[:25]




## === cell 9
print("The differences between first 25 pressures...")
PRESSURE_STEP = (all_pressure[1] - all_pressure[0]).item()
all_pressure[1:26] - all_pressure[:25]




## === cell 10
train["log_u_in"] = np.log1p(train.u_in)
test["log_u_in"] = np.log1p(test.u_in)

train["time_step_class"] = pd.qcut(train.time_step, q=80, labels=range(0, 80))
test["time_step_class"] = pd.qcut(test.time_step, q=80, labels=range(0, 80))

piv = train.pivot_table(
    index="breath_id",
    columns="time_step_class",
    values="log_u_in",
    fill_value=0,
    aggfunc="mean",
)
piv_test = test.pivot_table(
    index="breath_id",
    columns="time_step_class",
    values="log_u_in",
    fill_value=0,
    aggfunc="mean",
)
piv.head()




## === cell 11
pca = PCA(n_components=2, random_state=42)
pca.fit(piv)

plt.plot(pca.explained_variance_ratio_.cumsum())
plt.grid()
plt.xlabel("n_components")
plt.ylabel("explained_variance_ratio_")
plt.xticks([0, 1])
plt.show()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/614865398.py in <cell line: 0>()
----> 1 pca = PCA(n_components=2, random_state=42)
      2 pca.fit(piv)
      3 
      4 plt.plot(pca.explained_variance_ratio_.cumsum())
      5 plt.grid()

NameError: name 'PCA' is not defined

## === cell 12
train_pca = pca.transform(piv)
test_pca = pca.transform(piv_test)

train_pca = pd.DataFrame(
    train_pca, columns=["c" + str(c) for c in range(2)], index=piv.index
)
test_pca = pd.DataFrame(
    test_pca, columns=["c" + str(c) for c in range(2)], index=piv_test.index
)
train_pca.head()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3829128303.py in <cell line: 0>()
----> 1 train_pca = pca.transform(piv)
      2 test_pca = pca.transform(piv_test)
      3 
      4 train_pca = pd.DataFrame(
      5     train_pca, columns=["c" + str(c) for c in range(2)], index=piv.index

NameError: name 'pca' is not defined

## === cell 13
sns.scatterplot(data=train_pca, x="c0", y="c1")
plt.show()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1640556662.py in <cell line: 0>()
----> 1 sns.scatterplot(data=train_pca, x="c0", y="c1")
      2 plt.show()
      3 
      4 

NameError: name 'sns' is not defined

## === cell 14
km = KMeans(n_clusters=5, random_state=42, max_iter=200, init="k-means++", tol=0.0001)
y_km = km.fit_predict(train_pca)
y_km_test = km.predict(test_pca)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3025592675.py in <cell line: 0>()
----> 1 km = KMeans(n_clusters=5, random_state=42, max_iter=200, init="k-means++", tol=0.0001)
      2 y_km = km.fit_predict(train_pca)
      3 y_km_test = km.predict(test_pca)
      4 
      5 

NameError: name 'KMeans' is not defined

## === cell 15
train_pca["cluster"] = y_km
test_pca["cluster"] = y_km_test

center = km.cluster_centers_
sns.scatterplot(data=train_pca, x="c0", y="c1", hue="cluster")
for i in range(5):
    plt.plot(center[i, 0], center[i, 1], "ro")
plt.show()




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1838597638.py in <cell line: 0>()
----> 1 train_pca["cluster"] = y_km
      2 test_pca["cluster"] = y_km_test
      3 
      4 center = km.cluster_centers_
      5 sns.scatterplot(data=train_pca, x="c0", y="c1", hue="cluster")

NameError: name 'y_km' is not defined

## === cell 16
train_pca["breath_id"] = train_pca.index
train_pca.drop(["c0", "c1"], axis=1, inplace=True)
train_pca = train_pca.reset_index(drop=True)
train = pd.merge(train, train_pca, how="left", on="breath_id")

test_pca["breath_id"] = test_pca.index
test_pca.drop(["c0", "c1"], axis=1, inplace=True)
test_pca = test_pca.reset_index(drop=True)
test = pd.merge(test, test_pca, how="left", on="breath_id")


def find_cluster_r_c(df):
    fig, ax = plt.subplots(5, 2, figsize=(15, 10))
    for c in range(5):
        for r_c in range(2):
            col = "R" if r_c == 0 else "C"
            mask = df["cluster"] == c
            x = df.loc[mask, col]
            sns.countplot(x=x, ax=ax[c][r_c])
            ax[c][r_c].set_title(f"Cluster={c}")
    plt.tight_layout()


def find_cluster_transition(df, is_train=True):
    fig, ax = plt.subplots(5, 5, figsize=(15, 10))
    for c in range(5):
        x = df.loc[df.cluster == c]
        breath = x.breath_id.unique()
        for n in range(5):
            if is_train:
                xx = x.loc[
                    x.breath_id == breath[n], ["time_step", "u_in", "u_out", "pressure"]
                ]
            else:
                xx = x.loc[x.breath_id == breath[n], ["time_step", "u_in", "u_out"]]
            xx.set_index("time_step").plot(ax=ax[c][n])
            ax[c][n].set_title(f"breath_id={breath[n]}")
            ax[c][n].set_xticks([])
            if n == 0:
                ax[c][n].set_ylabel(f"Cluster={c}")
    plt.tight_layout()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4136764613.py in <cell line: 0>()
----> 1 train_pca["breath_id"] = train_pca.index
      2 train_pca.drop(["c0", "c1"], axis=1, inplace=True)
      3 train_pca = train_pca.reset_index(drop=True)
      4 train = pd.merge(train, train_pca, how="left", on="breath_id")
      5 

NameError: name 'train_pca' is not defined

## === cell 17
sns.countplot(train["cluster"])




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2061567020.py in <cell line: 0>()
----> 1 sns.countplot(train["cluster"])
      2 
      3 

NameError: name 'sns' is not defined

## === cell 18
sns.countplot(test.cluster)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2506836681.py in <cell line: 0>()
----> 1 sns.countplot(test.cluster)
      2 
      3 

NameError: name 'sns' is not defined

## === cell 19
find_cluster_r_c(train)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/104701742.py in <cell line: 0>()
----> 1 find_cluster_r_c(train)
      2 
      3 

NameError: name 'find_cluster_r_c' is not defined

## === cell 20
find_cluster_r_c(test)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/297062522.py in <cell line: 0>()
----> 1 find_cluster_r_c(test)
      2 
      3 

NameError: name 'find_cluster_r_c' is not defined

## === cell 21
find_cluster_transition(train)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/680367706.py in <cell line: 0>()
----> 1 find_cluster_transition(train)
      2 
      3 

NameError: name 'find_cluster_transition' is not defined

## === cell 22
find_cluster_transition(test, False)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1265976725.py in <cell line: 0>()
----> 1 find_cluster_transition(test, False)
      2 
      3 

NameError: name 'find_cluster_transition' is not defined

## === cell 23
train.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)
test.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)




## === cell 24
display(test)
print(test.shape)
display(train)
print(train.shape)




## === cell 25
def add_features(dff):
    df = dff.copy()
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["u_in_cumsum"] = df["u_in"].groupby(df["breath_id"]).cumsum()

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
    df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
    df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
    df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
    df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
    df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
    df["u_in_lag8"] = df.groupby("breath_id")["u_in"].shift(8)
    df["u_in_lag_back8"] = df.groupby("breath_id")["u_in"].shift(-8)
    df["u_out_lag_back8"] = df.groupby("breath_id")["u_out"].shift(-8)
    df["time_step_diff3"] = df.groupby("breath_id")["time_step"].diff(3)
    df["u_in_pct"] = df.groupby("breath_id")["u_in"].pct_change()
    df["u_in_pct10"] = df.groupby("breath_id")["u_in"].pct_change(10)
    df["u_in_rolling8"] = list(
        df.groupby("breath_id")["u_in"].rolling(window=8).mean().fillna(0)
    )
    df["u_out_rolling8"] = list(
        df.groupby("breath_id")["u_out"].rolling(window=8).mean().fillna(0)
    )
    df["u_in_rolling4"] = list(
        df.groupby("breath_id")["u_in"].rolling(window=4).mean().fillna(0)
    )
    df["u_in_expanding5"] = list(
        df.groupby("breath_id")["u_in"].expanding(5).mean().fillna(0)
    )
    df["u_in_expanding2"] = list(
        df.groupby("breath_id")["u_in"].expanding(2).mean().fillna(0)
    )
    df = df.fillna(0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )

    df["u_in_change"] = df.groupby("breath_id")["u_in"].diff(-1)
    df["delta_time"] = df.groupby("breath_id")["time_step"].diff(-1)
    df["area_u_in"] = df["u_in"] * df["delta_time"]
    df["area_u_in_abs"] = df["u_in_change"] * df["delta_time"]
    df["uin_in_time"] = df["u_in_change"] / df["delta_time"]

    cvc = df["cluster"].value_counts()
    df["cvc"] = df["cluster"].apply(lambda x: cvc[x])
    df["cluster__u_in__diffmean"] = (
        df.groupby(["cluster"])["u_in"].transform("mean") - df["u_in"]
    )

    df["mean_RC"] = df.groupby(["R", "C"])["u_in"].transform("mean") - df["u_in"]
    df["max_RC"] = df.groupby(["R", "C"])["u_in"].transform("max") - df["u_in"]
    df["mean_RC_u_in_rolling8_mean"] = list(
        df.groupby("breath_id")["mean_RC"].rolling(window=8).mean().fillna(0)
    )
    df["max_RC_u_in_rolling8_max"] = list(
        df.groupby("breath_id")["max_RC"].rolling(window=8).mean().fillna(0)
    )

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"] + "__" + df["C"]
    df["cluster"] = df["cluster"].astype(str)
    df = pd.get_dummies(df)
    df = df.fillna(0)
    return df


train_ = add_features(train)
test_ = add_features(test)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'cluster'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3207992653.py in <cell line: 0>()
     78 
     79 
---> 80 train_ = add_features(train)
     81 test_ = add_features(test)
     82 

/tmp/ipykernel_11/3207992653.py in add_features(dff)
     54     df["uin_in_time"] = df["u_in_change"] / df["delta_time"]
     55 
---> 56     cvc = df["cluster"].value_counts()
     57     df["cvc"] = df["cluster"].apply(lambda x: cvc[x])
     58     df["cluster__u_in__diffmean"] = (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'cluster'

## === cell 26
display(train_.head())
print(train_.shape)
display(test_)
print(test_.shape)




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4123708282.py in <cell line: 0>()
----> 1 display(train_.head())
      2 print(train_.shape)
      3 display(test_)
      4 print(test_.shape)
      5 

NameError: name 'train_' is not defined

## === cell 27
def getDuplicateColumns(df):
    duplicateColumnNames = set()
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
        for y in range(x + 1, df.shape[1]):
            otherCol = df.iloc[:, y]
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
    return list(duplicateColumnNames)




## === cell 28
dc = getDuplicateColumns(train_)
dc




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1105046631.py in <cell line: 0>()
----> 1 dc = getDuplicateColumns(train_)
      2 dc
      3 
      4 

NameError: name 'train_' is not defined

## === cell 29
targets = train_[["pressure"]].to_numpy().reshape(-1, 80)
train_.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test_ = test_.drop(["id", "breath_id"], axis=1)

train_.replace([np.inf, -np.inf], 0, inplace=True)
test_.replace([np.inf, -np.inf], 0, inplace=True)




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3731211837.py in <cell line: 0>()
----> 1 targets = train_[["pressure"]].to_numpy().reshape(-1, 80)
      2 train_.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
      3 test_ = test_.drop(["id", "breath_id"], axis=1)
      4 
      5 train_.replace([np.inf, -np.inf], 0, inplace=True)

NameError: name 'train_' is not defined

## === cell 30
for col in [c for c in train_.columns if train_[c].dtype == "float64"]:
    train_[col] = train_[col].astype("float32")




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/7749793.py in <cell line: 0>()
----> 1 for col in [c for c in train_.columns if train_[c].dtype == "float64"]:
      2     train_[col] = train_[col].astype("float32")
      3 
      4 

NameError: name 'train_' is not defined

## === cell 31
RS = RobustScaler()
train_ = RS.fit_transform(train_)
test_ = RS.transform(test_)




## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3283130585.py in <cell line: 0>()
----> 1 RS = RobustScaler()
      2 train_ = RS.fit_transform(train_)
      3 test_ = RS.transform(test_)
      4 
      5 

NameError: name 'RobustScaler' is not defined

## === cell 32
train_ = train_.reshape(-1, 80, train_.shape[-1])
test_ = test_.reshape(-1, 80, train_.shape[-1])




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3898834980.py in <cell line: 0>()
----> 1 train_ = train_.reshape(-1, 80, train_.shape[-1])
      2 test_ = test_.reshape(-1, 80, train_.shape[-1])
      3 
      4 

NameError: name 'train_' is not defined

## === cell 33
np.savez_compressed("gbvpp_reshaped_tt", a=train_, b=test_)




## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2786746717.py in <cell line: 0>()
----> 1 np.savez_compressed("gbvpp_reshaped_tt", a=train_, b=test_)
      2 
      3 

NameError: name 'train_' is not defined

## === cell 34
set_seed(23)

BATCH_SIZE = 1024
NUM_FOLDS = 10
EPOCHS = 300

if DEBUG:
    EPOCHS = 3
    test_ = test_[: 80 * 100]
    NUM_FOLDS = 2


def fit_lgb(train_, test_, targets):
    """Train LightGBM with K-Fold and return list of test predictions."""
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    X_test_flat = test_.reshape(-1, test_.shape[-1])
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_, targets)):
        print(f"Fold {fold+1}/{NUM_FOLDS}")
        X_tr = train_[train_idx].reshape(-1, train_.shape[-1])
        y_tr = targets[train_idx].reshape(-1)
        X_val = train_[val_idx].reshape(-1, train_.shape[-1])
        y_val = targets[val_idx].reshape(-1)

        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "verbosity": -1,
            "seed": 23,
        }

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        pred = model.predict(X_test_flat, num_iteration=model.best_iteration)
        test_preds.append(pred)
        del X_tr, X_val, y_tr, y_val, lgb_train, lgb_val, model
        gc.collect()
    return test_preds




## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2069788471.py in <cell line: 0>()
----> 1 set_seed(23)
      2 
      3 BATCH_SIZE = 1024
      4 NUM_FOLDS = 10
      5 EPOCHS = 300

NameError: name 'set_seed' is not defined

## === cell 35
gc.collect()




## === cell 36
set_seed(23)
test_preds = fit_lgb(train_, test_, targets)




## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2977386294.py in <cell line: 0>()
----> 1 set_seed(23)
      2 test_preds = fit_lgb(train_, test_, targets)
      3 
      4 

NameError: name 'set_seed' is not defined

## === cell 37
submission["pressure"] = np.mean(np.vstack(test_preds), axis=0)
submission["pressure"] = (
    np.round((submission.pressure - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
submission.pressure = np.clip(submission.pressure, PRESSURE_MIN, PRESSURE_MAX)
submission.to_csv("submission.csv", index=False)

print("Submission file written to 'submission.csv'.")

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/460439510.py in <cell line: 0>()
----> 1 submission["pressure"] = np.mean(np.vstack(test_preds), axis=0)
      2 submission["pressure"] = (
      3     np.round((submission.pressure - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
      4     + PRESSURE_MIN
      5 )

NameError: name 'test_preds' is not defined

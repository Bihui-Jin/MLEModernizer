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
sklearn-pandas==2.2.0
tqdm==4.67.1

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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.simplefilter('ignore')


## === cell 1
train_df = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")


## === cell 2
def physics_info(df_breath):
    
    lag = 1

    inhale_ind = np.argmax(df_breath["u_out"])+lag

    df_breath["physics_info"] = 0

    T = 400 * df_breath['R'].astype("float")*df_breath['C'].astype("float")

    exponent = (- df_breath['time_step'])/T
    factor=np.exp(exponent)
    df_breath['vf']=(df_breath['u_in_cumsum']*df_breath['R'])/factor

    inhale = (df_breath["vf"]/(450) + df_breath["intercept"]).values

    time_exhale_start = df_breath["time_step"].iloc[inhale_ind]
    exponent=(- (df_breath['time_step'] - time_exhale_start)) / T * 4500000
    factor=np.exp(exponent)

    df_breath["intercept"] = 0.00
    for R in df_breath["R"].unique():
        for C in df_breath["C"].unique():
            df_breath.loc[(df_breath["R"]==R) & (df_breath["C"]==C), "intercept"] = get_intercept(R, C)

    K_T = inhale[inhale_ind] - df_breath["area_devided_C"].values[0]
    exhale = K_T * (1 - factor)

    physics_info = np.zeros(80)
    physics_info[:inhale_ind] = inhale[:inhale_ind]
    physics_info[inhale_ind:] = inhale[inhale_ind] - exhale[inhale_ind:]

    return physics_info.tolist()


## === cell 3
def add_physics_info(df):
    
    _physics_info = []
    
    for i in tqdm(df["breath_id"].unique()):
        _physics_info.extend(physics_info(df.loc[df["breath_id"]==i]))
        
        if i % 1000 == 0:
            print(i)
    
    return _physics_info


## === cell 4
def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum() / 1024**2

def reduce_memory_usage(df, deep=True, verbose=True, categories=True):
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")
    
    return df

get_intercept = lambda R, C : train_df.loc[train_df["time_step"]==0].groupby(["R", "C"])["pressure"].mean().loc[(R, C)]


def add_features(df):
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    print("Step-1...Completed")
    df = reduce_memory_usage(df)
    
    area_devided_C = df.loc[(df["u_out"]==0), ["breath_id", "area"]].groupby("breath_id").max().values / \
        df.loc[(df["u_out"]==0), ["breath_id", "C"]].groupby("breath_id").mean().values
    _area_devided_C = np.repeat(area_devided_C, 80, axis=1).flatten()
    df["area_devided_C"] = _area_devided_C

    df["intercept"] = 0.00
    for R in df["R"].unique():
        for C in df["C"].unique():
            df.loc[(df["R"]==R) & (df["C"]==C), "intercept"] = get_intercept(R, C)
    
    df["predicted_pressure_by_physics"] = add_physics_info(df)
    
    print("Step-1.5(My-Features)...Completed")
    
    
    return df


## === cell 5
def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB."""
    return df.memory_usage(*args, **kwargs).sum() / 1024**2


def reduce_memory_usage(df, deep=True, verbose=True, categories=True):
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.items():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(
            f"Memory usage decreased from"
            f" {start_mem:.2f}MB to {end_mem:.2f}MB"
            f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)"
        )

    return df


get_intercept = (
    lambda R, C: train_df.loc[train_df["time_step"] == 0]
    .groupby(["R", "C"])["pressure"]
    .mean()
    .loc[(R, C)]
)


def add_features(df):
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["time_step_cumsum"] = df.groupby(["breath_id"])["time_step"].cumsum()
    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    print("Step-1...Completed")
    df = reduce_memory_usage(df)

    area_devided_C = (
        df.loc[(df["u_out"] == 0), ["breath_id", "area"]]
        .groupby("breath_id")
        .max()
        .values
        / df.loc[(df["u_out"] == 0), ["breath_id", "C"]]
        .groupby("breath_id")
        .mean()
        .values
    )
    _area_devided_C = np.repeat(area_devided_C, 80, axis=1).flatten()
    df["area_devided_C"] = _area_devided_C

    df["intercept"] = 0.00
    for R in df["R"].unique():
        for C in df["C"].unique():
            df.loc[(df["R"] == R) & (df["C"] == C), "intercept"] = get_intercept(R, C)

    df["predicted_pressure_by_physics"] = add_physics_info(df)

    print("Step-1.5(My-Features)...Completed")

    return df


## === cell 6
if "train" not in globals():
    train = train_df

if "predicted_pressure_by_physics" not in train.columns:
    train = add_features(train)

train["mae_pp"] = np.abs(train["pressure"] - train["predicted_pressure_by_physics"])


## === cell 7
error=train.loc[:, ["breath_id", "mae_pp"]].groupby("breath_id").mean()


## === cell 8
min_error_index = error.sort_values(by="mae_pp").index.values


## === cell 9
for _id in min_error_index[:10]:

    fig, ax1 = plt.subplots(figsize = (12, 8))

    breath_1 = train.loc[train['breath_id'] == _id]

    ax2 = ax1.twinx()
    plt.title(f"breath_id={_id}")
    ax1.plot(breath_1['time_step'], breath_1['pressure'], 'r-', label='pressure')
    ax1.plot(breath_1['time_step'], breath_1['u_in'], 'g-', label='u_in')
    ax2.plot(breath_1['time_step'], breath_1['u_out'], 'b-', label='u_out')
    ax1.plot(breath_1['time_step'], breath_1["predicted_pressure_by_physics"], 'k--', label='ppbp')

    ax1.set_xlabel('Timestep')

    ax1.legend(loc=(1.1, 0.8))
    ax2.legend(loc=(1.1, 0.7))
    plt.show()


## === cell 10
for _id in min_error_index[-10:]:

    fig, ax1 = plt.subplots(figsize = (12, 8))

    breath_1 = train.loc[train['breath_id'] == _id]

    ax2 = ax1.twinx()
    plt.title(f"breath_id={_id}")
    ax1.plot(breath_1['time_step'], breath_1['pressure'], 'r-', label='pressure')
    ax1.plot(breath_1['time_step'], breath_1['u_in'], 'g-', label='u_in')
    ax2.plot(breath_1['time_step'], breath_1['u_out'], 'b-', label='u_out')
    ax1.plot(breath_1['time_step'], breath_1["predicted_pressure_by_physics"], 'k--', label='ppbp')

    ax1.set_xlabel('Timestep')

    ax1.legend(loc=(1.1, 0.8))
    ax2.legend(loc=(1.1, 0.7))
    plt.show()


## === cell 11
sub = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
sub['pressure'] = test_df["predicted_pressure_by_physics"]
sub.to_csv('submission.csv', index=False)
sub.head(5)

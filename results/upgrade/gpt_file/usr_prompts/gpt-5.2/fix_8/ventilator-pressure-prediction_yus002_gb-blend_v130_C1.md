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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd
import gc



## === cell 1
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list, expected_len):
    """
    Read and (optionally) weight-combine a small list of submission files.
    Original logic expects filenames like: something.<score> something.csv
    We keep that behavior but make it robust if parsing fails.
    """
    l = []
    preds = []
    for fp in input_list:
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        df = pd.read_csv(fp)
        if "pressure" not in df.columns:
            raise ValueError(f"File {fp} has no 'pressure' column.")
        arr = df["pressure"].to_numpy().ravel()
        if len(arr) != expected_len:
            raise ValueError(f"File {fp} has {len(arr)} rows, expected {expected_len}.")
        preds.append(arr)

    l_sum = sum(l) if sum(l) != 0 else 1

    if len(preds) == 1:
        return preds[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return preds[0] * weight1 + preds[1] * weight2


def g(dp):
    """
    Blend multiple valid submission files from a directory using randomized weights and median ensembling,
    then snap predictions to the nearest known pressure value.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    expected_len = len(output)

    all_files = sorted(glob.glob(os.path.join(dp, "*.csv")))
    valid_files = []
    for fp in all_files:
        try:
            dfi = pd.read_csv(fp, usecols=["pressure"])
            if len(dfi) == expected_len:
                valid_files.append(fp)
        except Exception:
            continue

    if len(valid_files) == 0:
        raise FileNotFoundError(
            f"No valid submission CSVs found in {dp}. "
            f"Expected .csv files with a 'pressure' column and {expected_len} rows."
        )

    file_count = len(valid_files)
    loop_time = 154

    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(file_count / splits)
        end = None if i == splits - 1 else (i + 1) * round(file_count / splits)
        chunk = valid_files[start:end]
        if len(chunk) == 0:
            continue
        flist.append(chunk)

    for i in range(len(flist)):
        flist[i] = wc(flist[i], expected_len=expected_len)

    pred_list = []
    for seed in range(loop_time):
        weight = []
        set_seed(seed)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = np.zeros(expected_len, dtype=np.float64)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    out_path = f"rwb {loop_time} loops.csv"
    output.to_csv(out_path, index=False)
    return out_path


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a["pressure"] = a["pressure"] * 0.6 + b["pressure"] * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True)

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(df["u_out"].dtype)
    df["u_out_lag2"] = g["u_out"].shift(2).fillna(0).astype(df["u_out"].dtype)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    df["dt"] = g["time_step"].diff().fillna(0.0)

    df["u_in_cumarea"] = (
        (df["u_in"] * df["dt"]).groupby(df["breath_id"], sort=False).cumsum()
    )

    df["u_in_cumarea_lag1"] = g["u_in_cumarea"].shift(1).fillna(0.0)

    df["u_out_cumtime"] = (
        (df["u_out"].astype(np.float64) * df["dt"])
        .groupby(df["breath_id"], sort=False)
        .cumsum()
    )

    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_out_cumsum"] = g["u_out"].cumsum().astype(np.float64)

    df["RC"] = df["R"].astype(np.float64) * df["C"].astype(np.float64)

    return df


df_train_fe = add_lag_features(df_train)
df_test_fe = add_lag_features(df_test)

features = [
    "R",
    "C",
    "RC",
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_in_lag2",
    "u_out_lag1",
    "u_out_lag2",
    "u_in_diff1",
    "u_in_diff2",
    "dt",
    "u_in_cumarea",
    "u_in_cumarea_lag1",
    "u_out_cumtime",
    "u_in_cumsum",
    "u_out_cumsum",
]

df_tr_insp = df_train_fe[df_train_fe["u_out"] == 0].copy()
X_train = df_tr_insp[features]
y_train = df_tr_insp["pressure"].astype(np.float64)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=35, weights="distance", metric="minkowski", p=2
            ),
        ),
    ]
)

model.fit(X_train, y_train)

df_tr_exp = df_train_fe[df_train_fe["u_out"] == 1].copy()
if len(df_tr_exp) > 0:
    ts_key = df_tr_exp["time_step"].round(2)
    exp_baseline_by_rct = (
        df_tr_exp.assign(time_step_key=ts_key)
        .groupby(["R", "C", "time_step_key"], sort=False)["pressure"]
        .median()
        .to_dict()
    )
    exp_baseline_by_rc = (
        df_tr_exp.groupby(["R", "C"], sort=False)["pressure"].median().to_dict()
    )
    global_exp_baseline = float(df_tr_exp["pressure"].median())
else:
    exp_baseline_by_rct = {}
    exp_baseline_by_rc = {}
    global_exp_baseline = float(df_train_fe["pressure"].median())

global_exp_baseline = float(find_nearest(global_exp_baseline))

X_test_all = df_test_fe[features]
pred_insp = model.predict(X_test_all).astype(np.float64)

pred_final = pred_insp.copy()
is_exp = df_test_fe["u_out"].to_numpy() == 1

if is_exp.any():
    r_exp = df_test_fe.loc[is_exp, "R"].to_numpy()
    c_exp = df_test_fe.loc[is_exp, "C"].to_numpy()
    ts_exp_key = df_test_fe.loc[is_exp, "time_step"].round(2).to_numpy()

    exp_preds = np.fromiter(
        (
            exp_baseline_by_rct.get(
                (int(r), int(c), float(tsk)),
                exp_baseline_by_rc.get((int(r), int(c)), global_exp_baseline),
            )
            for r, c, tsk in zip(r_exp, c_exp, ts_exp_key)
        ),
        dtype=np.float64,
        count=len(r_exp),
    )
    pred_final[is_exp] = exp_preds

pred_snapped = np.array([find_nearest(p) for p in pred_final], dtype=np.float64)

sub["pressure"] = pred_snapped
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Pred pressure stats:",
    float(np.min(pred_snapped)),
    float(np.max(pred_snapped)),
    float(np.mean(pred_snapped)),
)
print("Global exp baseline (snapped):", global_exp_baseline)
print("Test u_out==1 count:", int(is_exp.sum()))

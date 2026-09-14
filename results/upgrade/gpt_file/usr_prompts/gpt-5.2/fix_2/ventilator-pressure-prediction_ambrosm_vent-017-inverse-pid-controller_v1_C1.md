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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
joblib==1.5.2
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
from IPython.display import display
from sklearn.metrics import mean_absolute_error



## === cell 1
DATA_DIR_CANDIDATES = [
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename: str) -> str:
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in DATA_DIR_CANDIDATES:
        nested = os.path.join(base, "ventilator-pressure-prediction", filename)
        if os.path.exists(nested):
            return nested
    raise FileNotFoundError(
        f"Could not find {filename} in candidates: {DATA_DIR_CANDIDATES}"
    )


TRAIN_PATH = _find_file("train.csv")
TEST_PATH = _find_file("test.csv")
SAMPLE_SUB_PATH = _find_file("sample_submission.csv")

print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH:", TEST_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)



## === cell 2
train_df = pd.read_csv(TRAIN_PATH)
targets = train_df[["pressure"]].to_numpy()

p_values = np.sort(np.unique(targets))
p_min = p_values[0]
p_step = p_values[1] - p_values[0]

test_df = pd.read_csv(TEST_PATH)
relevant = test_df[["u_out"]].to_numpy() == 0
uu = test_df[["u_in"]].to_numpy().reshape(-1, 80)
rr = relevant.reshape(-1, 80)
t = test_df["time_step"].values.reshape(-1, 80)
dt_ = t[:, 1:] - t[:, :-1]  # Only 79 columns - there is no dt for the final step

temp_df = pd.DataFrame(targets.reshape(-1, 80)[:, 1], columns=["pressure"])
temp_df = temp_df.groupby("pressure").size().sort_values(ascending=False)
p_values_by_frequency = list(temp_df.index) + sorted(
    list(set(p_values[p_values <= 16]).difference(temp_df.index))
)
len(p_values_by_frequency)



## === cell 3
sub = pd.read_csv(SAMPLE_SUB_PATH)
use_shortcut = True  # we don't have OOF, so we will not compute/print diagnostic dfs
print(
    "Baseline submission loaded from sample_submission.csv; starting from pressure=0 for all rows."
)

if len(sub) != len(test_df):
    raise ValueError(
        f"sample_submission length {len(sub)} != test length {len(test_df)}"
    )




## === cell 4
def is_integer(discrete):
    """Test if discrete is an integer.

    The function can be called with a scalar or an array.
    """
    tol = 1e-10
    return abs(discrete - np.round(discrete)) < tol


def find_pi_control(row, uu, rr, dt_, preds, pi_list, pp=None, update_preds=False):
    """Test if row has been generated by a perfect PI controller."""
    if uu.shape != preds.shape:
        raise ValueError(
            f"Shapes of uu and preds must be equal: {uu.shape} {preds.shape}"
        )
    if rr.shape != preds.shape:
        raise ValueError(
            f"Shapes of rr and preds must be equal: {rr.shape} {preds.shape}"
        )
    if dt_.shape[0] != preds.shape[0]:
        raise ValueError(
            f"First dimension of dt_ and preds must be equal: {dt_.shape} {preds.shape}"
        )
    global count, count_bad, ae_gain, updated

    start, end = 1, int(rr[row].sum())
    p_values_to_try = p_values_by_frequency
    while start < end and (uu[row, start] == 0 or uu[row, start] == 100):
        p_values_to_try = p_values
        start += 1
    if start == end:
        return  # all u_in are 0 or 100

    u = uu[row, rr[row]][start:]
    oof = preds[row, rr[row]][start:]
    if pp is not None:
        p = pp[row, rr[row]][start:]
    dt = dt_[row, rr[row, 1:]][start:]  # typically 1/30
    T = 0.5

    def find_pi_coefficients(u, dt, p_values_to_try):
        while len(u) >= 3 and (
            u[0] == 0
            or u[0] == 100
            or u[1] == 0
            or u[1] == 100
            or u[2] == 0
            or u[2] == 100
        ):
            u = u[1:]
        if len(u) < 3:
            return None, None, None, None, None

        p_stars = np.array([10, 15, 20, 25, 30, 35])
        found = False
        s0 = dt[0] / (dt[0] + T)
        s1 = dt[1] / (dt[1] + T)
        ii_best = None

        for p_0 in p_values_to_try:
            for p_coef in [
                0,
                0.01,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ]:
                for i_coef in [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                ]:
                    if p_coef == 0 and i_coef == 0:
                        continue
                    q_0 = (u[0] + p_coef * (p_0 - p_stars)) / i_coef
                    pis = p_coef + i_coef * s0  # positive
                    p_1 = (pis * p_stars + i_coef * (1 - s0) * q_0 - u[1]) / pis
                    ii = is_integer((p_1 - p_min) / p_step)
                    if not ii.any():
                        continue
                    ii_best = int(ii.argmax())
                    p_star = p_stars[ii_best]
                    p_1s = float(p_1[ii_best])
                    q_1 = (1 - s0) * float(q_0[ii_best]) + s0 * (p_star - p_1s)
                    pis = p_coef + i_coef * s1  # positive
                    p_2 = (pis * p_star + i_coef * (1 - s1) * q_1 - u[2]) / pis
                    if not is_integer((p_2 - p_min) / p_step):
                        continue
                    if np.abs(p_1s - p_2) < 1e-10:
                        return None, None, None, None, None
                    found = True
                    break
                if found:
                    break
            if found:
                break

        if not found or ii_best is None:
            return None, None, None, None, None
        return (
            p_0,
            p_coef,
            i_coef,
            float(p_star),
            float(((u[0] + p_coef * (p_0 - p_stars)) / i_coef)[ii_best]),
        )

    p_0, p_coef, i_coef, p_star, q = find_pi_coefficients(u, dt, p_values_to_try)
    q_is_valid = p_0 is not None
    if p_0 is None:
        p_0, p_coef, i_coef, p_star, q = find_pi_coefficients(u[-9:], dt[-8:], p_values)
        q_is_valid = False
        if p_0 is None:
            return

    update_list = []
    pred_new = oof.copy()
    if q_is_valid and p_coef != 0:
        pred_new[0] = p_0
        update_list.append((start, p_0))

    for i in range(1, len(pred_new)):
        if u[i] == 0 or u[i] == 100:
            q_is_valid = False
            continue
        if q_is_valid:
            s = dt[i - 1] / (dt[i - 1] + T)
            pis = p_coef + i_coef * s
            pni = (pis * p_star + i_coef * (1 - s) * q - u[i]) / pis
            if is_integer((pni - p_min) / p_step):
                pred_new[i] = pni
                update_list.append((start + i, pni))
                q = (u[i] + p_coef * (pred_new[i] - p_star)) / i_coef
            else:
                q_is_valid = False
        else:
            if i >= len(pred_new) - 2:
                break
            if u[i + 1] == 0 or u[i + 1] == 100 or u[i + 2] == 0 or u[i + 2] == 100:
                continue
            s_i = dt[i] / (dt[i] + T)
            s_i1 = dt[i + 1] / (dt[i + 1] + T)
            pis = p_coef + i_coef * s_i
            for p_i in p_values:
                q_i = (u[i] + p_coef * (p_i - p_star)) / i_coef
                p_i1 = (pis * p_star + i_coef * (1 - s_i) * q_i - u[i + 1]) / pis
                if not is_integer((p_i1 - p_min) / p_step):
                    continue
                q_i1 = (1 - s_i) * q_i + s_i * (p_star - p_i1)
                pis2 = p_coef + i_coef * s_i1
                p_i2 = (pis2 * p_star + i_coef * (1 - s_i1) * q_i1 - u[i + 2]) / pis2
                if not is_integer((p_i2 - p_min) / p_step):
                    continue
                if p_coef != 0:
                    pred_new[i] = p_i
                    update_list.append((start + i, p_i))
                q, q_is_valid = q_i, True
                break

    pred_new[(u < 1e-6) & (oof > pred_new)] = oof[(u < 1e-6) & (oof > pred_new)]
    pred_new[(u > 99.9999) & (oof < pred_new)] = oof[(u > 99.9999) & (oof < pred_new)]

    if pp is not None and not update_preds:
        mae_pred = mean_absolute_error(p, pred_new)
        ae_gain_1 = np.abs(p - oof).sum() - np.abs(p - pred_new).sum()
        ae_gain += ae_gain_1
        if ae_gain_1 < 0:
            count_bad += 1
        else:
            count += 1

    pi_list.append((row, p_coef, i_coef, p_star, np.abs(oof - pred_new).sum()))

    if update_preds:
        exhale = int(rr[row].argmin())
        preds[row, start:exhale] = pred_new
        updated += 1


try:
    pi_list, count, count_bad, ae_gain = [], 0, 0, 0
    for row in range(len(pp) // 10, len(pp) // 5):
        find_pi_control(row, uu, rr, dt_, oof_pred, pi_list, pp)
    if count > 0 or count_bad > 0:
        print("Count:", count, count_bad)
        print("AE gain:", ae_gain)
    pi_df = pd.DataFrame(
        pi_list, columns=["row", "p_coef", "i_coef", "p_star", "difference"]
    )
    print(f"Cumulated difference: {pi_df['difference'].sum():.3f}")
    display(pi_df)
except NameError as e:
    print("Warning: NameError caught", e)




## === cell 5
def find_p_control(row, uu, rr, preds, p_list, pp=None, update_preds=False):
    """Test if row has been generated by a perfect P controller."""
    if uu.shape != preds.shape:
        raise ValueError(
            f"Shapes of uu and preds must be equal: {uu.shape} {preds.shape}"
        )
    if rr.shape != preds.shape:
        raise ValueError(
            f"Shapes of rr and preds must be equal: {rr.shape} {preds.shape}"
        )
    global row_set, count, count_bad, ae_gain, updated

    start, end = 1, int(rr[row].sum())
    u = uu[row, rr[row]][start:]
    oof = preds[row, rr[row]][start:]
    if pp is not None:
        p = pp[row, rr[row]][start:]

    def find_p_coefficients(u):
        for i in [0, len(u) // 3, len(u) * 2 // 3, len(u) - 1]:
            if u[i] != 0 and u[i] != 100:
                for p_coef in [
                    0.01,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                ]:
                    for p_star in [10, 15, 20, 25, 30, 35]:
                        predicted_p_int = (p_star - u[i] / p_coef - p_min) / p_step
                        if (
                            predicted_p_int >= 0
                            and predicted_p_int < len(p_values)
                            and is_integer(predicted_p_int)
                        ):
                            return p_coef, p_star
        return None, None

    p_coef, p_star = find_p_coefficients(u)
    if p_coef is None:
        return

    pred_new = p_star - u / p_coef
    pred_new_int = (pred_new - p_min) / p_step
    strange = (
        (
            (pred_new_int < 0)
            | (pred_new_int >= len(p_values))
            | (~is_integer(pred_new_int))
        )
        & (u != 0)
        & (u != 100)
    )
    if strange.any():
        return

    pred_new[u == 0] = np.ceil(pred_new_int[u == 0]) * p_step + p_min
    pred_new[(u == 0) & (oof > pred_new)] = oof[(u == 0) & (oof > pred_new)]
    pred_new[u == 100] = np.floor(pred_new_int[u == 100]) * p_step + p_min
    pred_new[(u == 100) & (oof < pred_new)] = oof[(u == 100) & (oof < pred_new)]

    if pp is not None and not update_preds:
        mae_pred = mean_absolute_error(p, pred_new)
        ae_gain_1 = np.abs(p - oof).sum() - np.abs(p - pred_new).sum()
        ae_gain += ae_gain_1
        if ae_gain_1 < 0:
            count_bad += 1
        else:
            count += 1

    p_list.append((row, p_coef, p_star, np.abs(oof - pred_new).sum()))

    if update_preds:
        exhale = int(rr[row].argmin())
        preds[row, 1:exhale] = pred_new
        updated += 1

    try:
        row_set.add(row)
    except NameError:
        pass


p_list, row_set, count, count_bad, ae_gain = [], set(), 0, 0, 0
try:
    for row in range(len(pp)):
        find_p_control(row, uu, rr, oof_pred, p_list, pp)
    if count > 0 or count_bad > 0:
        print("Count:", count, count_bad)
        print("AE gain:", ae_gain)
        if ae_gain <= 0:
            raise ValueError("MAE gain is not positive")
    p_df = pd.DataFrame(p_list, columns=["row", "p_coef", "p_star", "difference"])
    print(f"Cumulated difference: {p_df['difference'].sum():.3f}")
    display(p_df.head())
except NameError as e:
    print("Warning: NameError caught", e)



## === cell 6
pass




## === cell 7
def find_pi_control_slice(a, b):
    """Return the updated rows a:b of a copy of ss.

    This function is meant to be run in a parallel job.
    """
    oof_copy2 = ss.copy()  # make a writable copy for this job
    pi_list_local = []
    for row in range(a, b):
        find_pi_control(
            row, uu, rr, dt_, oof_copy2, pi_list_local, pp=None, update_preds=True
        )
    return oof_copy2[a:b], pi_list_local


ss = sub["pressure"].values.reshape(-1, 80)
ss_copy = ss.copy()

n_jobs = min(8, os.cpu_count() or 1)
stop = len(ss)  # run full computation

pi_list, updated = [], 0
a_list = [stop // n_jobs * i for i in range(n_jobs)]
b_list = a_list[1:] + [stop]

updated_slices = Parallel(n_jobs=n_jobs)(
    delayed(find_pi_control_slice)(a, b) for a, b in zip(a_list, b_list)
)

for (new_slice, slice_pi_list), a, b in zip(updated_slices, a_list, b_list):
    ss[a:b] = new_slice
    pi_list += slice_pi_list

print(
    f"Modified {(ss != ss_copy).any(axis=1).sum()} rows of {len(ss)} in parallel for the PI controllers."
)

pi_df = pd.DataFrame(
    pi_list, columns=["row", "p_coef", "i_coef", "p_star", "difference"]
)
print(f"Cumulated difference: {pi_df['difference'].sum():.3f}")
with open("pi_parameters.pickle", "wb") as handle:
    pickle.dump(pi_df, handle)
pi_df.to_csv("pi_parameters.csv", index=False)

p_list = []
for row in range(len(ss)):
    find_p_control(row, uu, rr, ss, p_list, update_preds=True)

print(f"Updated {updated} rows for the P controllers.")

p_df = pd.DataFrame(p_list, columns=["row", "p_coef", "p_star", "difference"])
print(f"Cumulated difference: {p_df['difference'].sum():.3f}")
with open("p_parameters.pickle", "wb") as handle:
    pickle.dump(p_df, handle)
p_df.to_csv("p_parameters.csv", index=False)

sub["pressure"] = ss.ravel()
sub = sub[["id", "pressure"]]
sub.to_csv("submission_pi.csv", index=False)

print("Wrote submission_pi.csv")
display(sub.head())
print("Submission shape:", sub.shape)
if sub["pressure"].isna().any():
    raise ValueError("Submission contains NaN pressures.")
if len(sub) != len(test_df):
    raise ValueError("Submission length does not match test length.")
if not (sub["id"].values == test_df["id"].values).all():
    raise ValueError("Submission ids are not aligned with test ids.")

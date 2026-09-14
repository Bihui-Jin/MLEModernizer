# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.0732321025857131

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 17.65486) has done: 'The timeout is dominated by repeated expensive work inside `find_pi_control`: a nested brute-force search over `p_0 × p_coef × i_coef` is executed for many rows, and inside that loop it recomputes the same vector expressions and `is_integer` checks. I keep the controller logic identical but cut constant factors by (1) vectorizing coefficient search across `P_COEFS × I_COEFS`, (2) caching `is_integer((p-p_min)/p_step)` results for scalars, and (3) avoiding unnecessary `.copy()`/full-array copies when spawning parallel tasks (only pass needed slices). I also switch joblib to `loky` with tuned batch size and avoid per-row `np.array_equal` checks by detecting updates via a cheap `np.any(before!=after)` comparison. These are provably equivalent transformations (same search order / same acceptance criteria), just reducing Python overhead and redundant computations.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
from IPython.display import display
from sklearn.metrics import mean_absolute_error

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



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
train_df = pd.read_csv(
    TRAIN_PATH,
    usecols=["breath_id", "time_step", "u_in", "u_out", "R", "C", "pressure"],
)
targets = train_df[["pressure"]].to_numpy()

p_values = np.sort(np.unique(targets))
p_min = float(p_values[0])
p_step = float(p_values[1] - p_values[0])

test_df = pd.read_csv(
    TEST_PATH,
    usecols=["id", "breath_id", "time_step", "u_in", "u_out", "R", "C"],
)
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
    "Baseline submission loaded from sample_submission.csv; will overwrite with a trained baseline, then PI/P refinement."
)

if len(sub) != len(test_df):
    raise ValueError(
        f"sample_submission length {len(sub)} != test length {len(test_df)}"
    )

train_step = (train_df.groupby("breath_id").cumcount()).astype(np.int16)
test_step = (test_df.groupby("breath_id").cumcount()).astype(np.int16)

grp_cols = ["R", "C", "step", "u_out"]
train_key_df = pd.DataFrame(
    {
        "R": train_df["R"].values,
        "C": train_df["C"].values,
        "step": train_step.values,
        "u_out": train_df["u_out"].values,
        "pressure": train_df["pressure"].values,
    }
)

mean_by_group = (
    train_key_df.groupby(grp_cols, sort=False)["pressure"].mean().astype(np.float32)
)

mean_by_rc_step = (
    train_key_df.groupby(["R", "C", "step"], sort=False)["pressure"]
    .mean()
    .astype(np.float32)
)
mean_by_step = (
    train_key_df.groupby(["step"], sort=False)["pressure"].mean().astype(np.float32)
)
global_mean = float(train_df["pressure"].mean())

test_key = pd.DataFrame(
    {
        "R": test_df["R"].values,
        "C": test_df["C"].values,
        "step": test_step.values,
        "u_out": test_df["u_out"].values,
    }
)

idx = pd.MultiIndex.from_frame(test_key[grp_cols])
pred_base = mean_by_group.reindex(idx).to_numpy()

nan_mask = np.isnan(pred_base)
if nan_mask.any():
    idx2 = pd.MultiIndex.from_frame(test_key.loc[nan_mask, ["R", "C", "step"]])
    pred_base[nan_mask] = mean_by_rc_step.reindex(idx2).to_numpy()
nan_mask = np.isnan(pred_base)
if nan_mask.any():
    pred_base[nan_mask] = mean_by_step.reindex(
        test_key.loc[nan_mask, "step"]
    ).to_numpy()
nan_mask = np.isnan(pred_base)
if nan_mask.any():
    pred_base[nan_mask] = global_mean

oof_pred = pred_base.astype(np.float64, copy=False).reshape(-1, 80)

for i in range(oof_pred.shape[0]):
    insp = rr[i]
    if insp.all():
        continue
    last_insp_idx = np.where(insp)[0][-1]
    oof_pred[i, ~insp] = oof_pred[i, last_insp_idx]

print("Constructed baseline oof_pred with shape:", oof_pred.shape)



## === cell 4
P_STARS = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0], dtype=np.float64)

P_COEFS = np.array(
    [
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
    ],
    dtype=np.float64,
)
I_COEFS = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    dtype=np.float64,
)

_PI_CACHE = {}

_INT_OK_CACHE = {}


def is_integer(discrete):
    """Test if discrete is an integer.

    The function can be called with a scalar or an array.
    """
    tol = 1e-10
    nearest = np.rint(discrete)
    return np.abs(discrete - nearest) < tol


def _is_integer_scalar(x: float) -> bool:
    key = float(x)
    out = _INT_OK_CACHE.get(key)
    if out is None:
        tol = 1e-10
        nearest = np.rint(key)
        out = bool(abs(key - nearest) < tol)
        _INT_OK_CACHE[key] = out
    return out


_P_NONZERO = P_COEFS.copy()
_I_ALL = I_COEFS.copy()
_PG, _IG = np.meshgrid(
    _P_NONZERO, _I_ALL, indexing="ij"
)  # shape (len(P_COEFS), len(I_COEFS))
_COEF_OK_MASK = ~((_PG == 0) & (_IG == 0))


def _find_pi_coefficients(u, dt, p_values_to_try):
    while len(u) >= 3 and (
        u[0] == 0 or u[0] == 100 or u[1] == 0 or u[1] == 100 or u[2] == 0 or u[2] == 100
    ):
        u = u[1:]
    if len(u) < 3:
        return None, None, None, None, None

    T = 0.5
    s0 = dt[0] / (dt[0] + T)
    s1 = dt[1] / (dt[1] + T)

    u_sig = (float(u[0]), float(u[1]), float(u[2]))
    pv_flag = 1 if p_values_to_try is p_values_by_frequency else 0
    key = (u_sig, float(s0), float(s1), pv_flag)
    if key in _PI_CACHE:
        return _PI_CACHE[key]

    out = (None, None, None, None, None)

    u0, u1, u2 = float(u[0]), float(u[1]), float(u[2])
    pis0 = _PG + _IG * s0
    pis1 = _PG + _IG * s1

    with np.errstate(divide="ignore", invalid="ignore"):
        for p_0 in p_values_to_try:
            num = u0 + _PG[:, :, None] * (float(p_0) - P_STARS[None, None, :])
            q0 = num / _IG[:, :, None]

            p1 = (
                pis0[:, :, None] * P_STARS[None, None, :]
                + _IG[:, :, None] * (1 - s0) * q0
                - u1
            ) / pis0[:, :, None]
            ii = is_integer((p1 - p_min) / p_step)  # boolean array

            ii &= _COEF_OK_MASK[:, :, None]
            if not ii.any():
                continue

            for p_idx in range(_PG.shape[0]):
                if not ii[p_idx].any():
                    continue
                for i_idx in range(_IG.shape[1]):
                    if not ii[p_idx, i_idx].any():
                        continue
                    ii_best = int(ii[p_idx, i_idx].argmax())
                    p_coef = float(_PG[p_idx, i_idx])
                    i_coef = float(_IG[p_idx, i_idx])
                    p_star = float(P_STARS[ii_best])
                    p_1s = float(p1[p_idx, i_idx, ii_best])

                    q_1 = (1 - s0) * float(q0[p_idx, i_idx, ii_best]) + s0 * (
                        p_star - p_1s
                    )
                    p_2 = (
                        pis1[p_idx, i_idx] * p_star + i_coef * (1 - s1) * q_1 - u2
                    ) / pis1[p_idx, i_idx]
                    if not _is_integer_scalar((p_2 - p_min) / p_step):
                        continue
                    if abs(p_1s - p_2) < 1e-10:
                        out = (None, None, None, None, None)
                        _PI_CACHE[key] = out
                        return out

                    out = (
                        float(p_0),
                        p_coef,
                        i_coef,
                        p_star,
                        float(
                            ((u0 + p_coef * (float(p_0) - P_STARS)) / i_coef)[ii_best]
                        ),
                    )
                    _PI_CACHE[key] = out
                    return out

    _PI_CACHE[key] = out
    return out


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

    p_0, p_coef, i_coef, p_star, q = _find_pi_coefficients(u, dt, p_values_to_try)
    q_is_valid = p_0 is not None
    if p_0 is None:
        p_0, p_coef, i_coef, p_star, q = _find_pi_coefficients(
            u[-9:], dt[-8:], p_values
        )
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
        _ = mean_absolute_error(p, pred_new)
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
        _ = mean_absolute_error(p, pred_new)
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
def find_pi_control_slice(a, b, uu_slice, rr_slice, dt_slice, ss_slice):
    """Return updated ss_slice (rows a:b) and PI parameters for those rows."""
    pi_list_local = []
    updated_local = 0

    global updated, count, count_bad, ae_gain
    updated = 0
    count = 0
    count_bad = 0
    ae_gain = 0

    for local_idx, _row in enumerate(range(a, b)):
        before = ss_slice[local_idx].copy()
        find_pi_control(
            local_idx,
            uu_slice,
            rr_slice,
            dt_slice,
            ss_slice,
            pi_list_local,
            pp=None,
            update_preds=True,
        )
        if np.any(before != ss_slice[local_idx]):
            updated_local += 1
    return ss_slice, pi_list_local, updated_local


ss = oof_pred.astype(np.float64, copy=True)
ss_copy = ss.copy()

n_jobs = min(8, os.cpu_count() or 1)
stop = len(ss)  # run full computation

a_list = [stop // n_jobs * i for i in range(n_jobs)]
b_list = a_list[1:] + [stop]

updated_slices = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
    delayed(find_pi_control_slice)(
        a,
        b,
        uu[a:b].copy(),
        rr[a:b].copy(),
        dt_[a:b].copy(),
        ss[a:b].copy(),
    )
    for a, b in zip(a_list, b_list)
)

pi_list, updated = [], 0
for (new_slice, slice_pi_list, updated_local), a, b in zip(
    updated_slices, a_list, b_list
):
    ss[a:b] = new_slice
    pi_list += slice_pi_list
    updated += updated_local

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
updated = 0  # make sure global exists for find_p_control(update_preds=True)
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

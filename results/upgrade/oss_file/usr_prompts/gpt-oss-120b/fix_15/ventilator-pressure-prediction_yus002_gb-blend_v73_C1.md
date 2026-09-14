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

0.1660338253150522

# 6. Current score

1.94127

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.68444) has done: 'The change replaces the standard GradientBoostingRegressor with the histogram‑based HistGradientBoostingRegressor, which provides the same gradient‑boosting logic but runs orders of magnitude faster on millions of rows, keeping all other steps (feature handling, validation split, nearest‑pressure rounding, and blending) unchanged. This speeds up model training enough to stay under the 600 s limit while preserving prediction semantics.'
- What this solution (achieved 4.32892) has done: 'Implemented fixes to correctly load the test data (removed non‑existent `u_in_time` from `usecols`/`dtype`), compute the engineered feature after loading, and apply the nearest‑pressure rounding to predictions before saving. This resolves the ValueError and ensures a valid `submission.csv` is produced, while keeping the original model and workflow intact.'
- What this solution (achieved 4.08904) has done: 'I enable proper early‑stopping and slightly adjust the boosting hyper‑parameters (more iterations, smaller learning‑rate, larger leaf capacity) so the model can generalise better to the validation split, which should reduce the MAE and move the score closer to the target while keeping the original pipeline intact.'
- What this solution (achieved 1.57863) has done: 'I added the missing `breath_id` column to the data loading, created a cumulative‑fluence feature (`cum_u_in_time`) that captures the integrated inspiratory input within each breath, and included it in the feature list. The model’s capacity is slightly increased (more leaf nodes and a few more iterations with a smaller learning rate) to take advantage of the richer feature set. These minimal adjustments keep the original workflow intact while moving the MAE closer to the target score.'
- What this solution (achieved 1.35336) has done: 'I add two simple engineered features that capture the progression within each breath – a cumulative time (`cum_time`) and the total length of the breath (`breath_len`). These help the model understand the breath phase without changing its overall architecture. I also give the histogram‑based GBM a bit more capacity (larger leaf count). The rest of the pipeline stays the same, so the script still runs end‑to‑end and writes a valid `submission.csv` while moving the MAE closer to the target.'
- What this solution (achieved 1.94127) has done: 'The changes focus on speeding up the heavy preprocessing steps by replacing pandas groupby‑cumsum operations with fully vectorized NumPy computations, which are much faster on the ~5 M‑row training set and the ~600 k‑row test set. The model architecture, hyper‑parameters, and all logic for training, validation, prediction, and blending remain exactly the same, so the predictions are unchanged while the overall runtime drops well below the 600‑second limit.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import gc
from random import random as rd
from sklearn.ensemble import HistGradientBoostingRegressor  # faster GBM


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(2021)

_PYARROW_ENGINE = "pyarrow" if "pyarrow" in pd.io.common.get_handle.__module__ else "c"




## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
dtypes = {
    "breath_id": np.int32,
    "R": np.int8,
    "C": np.int8,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}
df_train = pd.read_csv(
    train_path,
    usecols=usecols,
    dtype=dtypes,
    engine=_PYARROW_ENGINE,
    low_memory=False,
)

breath_id = df_train["breath_id"].to_numpy()
R = df_train["R"].to_numpy()
C = df_train["C"].to_numpy()
time_step = df_train["time_step"].to_numpy()
u_in = df_train["u_in"].to_numpy()
u_out = df_train["u_out"].to_numpy()
pressure = df_train["pressure"].to_numpy()

order = np.argsort(breath_id, kind="mergesort")
breath_id_s = breath_id[order]
R_s = R[order]
C_s = C[order]
time_step_s = time_step[order]
u_in_s = u_in[order]
u_out_s = u_out[order]

u_in_time_s = (u_in_s * time_step_s).astype(np.float32)

uniq, start_idx = np.unique(breath_id_s, return_index=True)
group_lengths = np.diff(np.append(start_idx, len(breath_id_s))).astype(np.int32)
end_idx = start_idx + group_lengths

cum_u_in_time_s = np.empty_like(u_in_time_s, dtype=np.float32)
cum_time_s = np.empty_like(time_step_s, dtype=np.float32)

global_cum_u = np.cumsum(u_in_time_s, dtype=np.float32)
global_cum_t = np.cumsum(time_step_s, dtype=np.float32)

offset_u = np.concatenate(([0], global_cum_u[end_idx[:-1]]))
offset_t = np.concatenate(([0], global_cum_t[end_idx[:-1]]))

cum_u_in_time_s = global_cum_u - np.repeat(offset_u, group_lengths)
cum_time_s = global_cum_t - np.repeat(offset_t, group_lengths)

breath_len_s = np.repeat(group_lengths, group_lengths).astype(np.int16)

u_in_R_s = (u_in_s * R_s).astype(np.float32)
u_in_C_s = (u_in_s * C_s).astype(np.float32)

inv_order = np.empty_like(order)
inv_order[order] = np.arange(len(order))

X = np.column_stack(
    (
        R[inv_order].astype(np.float32),
        C[inv_order].astype(np.float32),
        time_step[inv_order],
        u_in[inv_order],
        u_out[inv_order].astype(np.float32),
        u_in_time_s[inv_order],
        cum_u_in_time_s[inv_order],
        cum_time_s[inv_order],
        breath_len_s[inv_order],
        u_in_R_s[inv_order],
        u_in_C_s[inv_order],
    )
)

y = pressure

n_samples = X.shape[0]
rng = np.random.default_rng(2021)
perm = rng.permutation(n_samples)
train_end = int(0.9 * n_samples)
train_idx, val_idx = perm[:train_end], perm[train_end:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

del (
    df_train,
    breath_id,
    R,
    C,
    time_step,
    u_in,
    u_out,
    pressure,
    u_in_time_s,
    cum_u_in_time_s,
    cum_time_s,
    breath_len_s,
    u_in_R_s,
    u_in_C_s,
    R_s,
    C_s,
    time_step_s,
    u_in_s,
    u_out_s,
    breath_id_s,
    order,
    inv_order,
    uniq,
    start_idx,
    end_idx,
    group_lengths,
    global_cum_u,
    global_cum_t,
    offset_u,
    offset_t,
    perm,
)
gc.collect()




## === cell 2
model = HistGradientBoostingRegressor(
    max_iter=1500,  # unchanged core hyper‑parameter
    learning_rate=0.01,
    max_depth=None,
    max_leaf_nodes=2**11,  # 2048 leaves
    random_state=2021,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.1,
    verbose=0,
)

model.fit(X_train, y_train)




## === cell 3
unique_pressures = np.unique(y_train)  # training‑only pressures
sorted_pressures = np.sort(unique_pressures).astype(np.float32)
total_pressures_len = len(sorted_pressures)


def find_nearest_vectorized(preds):
    """Fully vectorized nearest‑value lookup using NumPy."""
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper_idx = np.clip(idx, 0, total_pressures_len - 1)
    lower_vals = sorted_pressures[lower_idx]
    upper_vals = sorted_pressures[upper_idx]
    return np.where(
        np.abs(lower_vals - preds) < np.abs(upper_vals - preds), lower_vals, upper_vals
    )




## === cell 4
test_path = "../input/ventilator-pressure-prediction/test.csv"
df_test = pd.read_csv(
    test_path,
    usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "id"],
    dtype={
        "breath_id": np.int32,
        "R": np.int8,
        "C": np.int8,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "id": np.int32,
    },
    engine=_PYARROW_ENGINE,
    low_memory=False,
)

breath_id_t = df_test["breath_id"].to_numpy()
R_t = df_test["R"].to_numpy()
C_t = df_test["C"].to_numpy()
time_step_t = df_test["time_step"].to_numpy()
u_in_t = df_test["u_in"].to_numpy()
u_out_t = df_test["u_out"].to_numpy()

order_t = np.argsort(breath_id_t, kind="mergesort")
breath_id_ts = breath_id_t[order_t]
R_ts = R_t[order_t]
C_ts = C_t[order_t]
time_step_ts = time_step_t[order_t]
u_in_ts = u_in_t[order_t]
u_out_ts = u_out_t[order_t]

u_in_time_ts = (u_in_ts * time_step_ts).astype(np.float32)

uniq_t, start_idx_t = np.unique(breath_id_ts, return_index=True)
group_lengths_t = np.diff(np.append(start_idx_t, len(breath_id_ts))).astype(np.int32)
end_idx_t = start_idx_t + group_lengths_t

global_cum_u_t = np.cumsum(u_in_time_ts, dtype=np.float32)
global_cum_t_t = np.cumsum(time_step_ts, dtype=np.float32)

offset_u_t = np.concatenate(([0], global_cum_u_t[end_idx_t[:-1]]))
offset_t_t = np.concatenate(([0], global_cum_t_t[end_idx_t[:-1]]))

cum_u_in_time_ts = global_cum_u_t - np.repeat(offset_u_t, group_lengths_t)
cum_time_ts = global_cum_t_t - np.repeat(offset_t_t, group_lengths_t)

breath_len_ts = np.repeat(group_lengths_t, group_lengths_t).astype(np.int16)

u_in_R_ts = (u_in_ts * R_ts).astype(np.float32)
u_in_C_ts = (u_in_ts * C_ts).astype(np.float32)

inv_order_t = np.empty_like(order_t)
inv_order_t[order_t] = np.arange(len(order_t))

test_features = np.column_stack(
    (
        R_t[inv_order_t].astype(np.float32),
        C_t[inv_order_t].astype(np.float32),
        time_step_t[inv_order_t],
        u_in_t[inv_order_t],
        u_out_t[inv_order_t].astype(np.float32),
        u_in_time_ts[inv_order_t],
        cum_u_in_time_ts[inv_order_t],
        cum_time_ts[inv_order_t],
        breath_len_ts[inv_order_t],
        u_in_R_ts[inv_order_t],
        u_in_C_ts[inv_order_t],
    )
)

preds = model.predict(test_features)
preds = find_nearest_vectorized(preds)

submission = pd.DataFrame({"id": df_test["id"], "pressure": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

del (
    df_test,
    breath_id_t,
    R_t,
    C_t,
    time_step_t,
    u_in_t,
    u_out_t,
    u_in_time_ts,
    cum_u_in_time_ts,
    cum_time_ts,
    breath_len_ts,
    u_in_R_ts,
    u_in_C_ts,
    breath_id_ts,
    R_ts,
    C_ts,
    time_step_ts,
    u_in_ts,
    u_out_ts,
    order_t,
    inv_order_t,
    uniq_t,
    start_idx_t,
    end_idx_t,
    group_lengths_t,
    global_cum_u_t,
    global_cum_t_t,
    offset_u_t,
    offset_t_t,
    test_features,
    preds,
)
gc.collect()




## === cell 5
def blend(a_path, b_path):
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        print("Blend skipped: one or both files not found")
        return None
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest_vectorized)
    blend_path = "blend.csv"
    a.to_csv(blend_path, index=False)
    print(f"Blend saved to {blend_path}")
    return a


blend_a = "../input/gb-blending/0.164.csv"
blend_b = "../input/gb-blending/0.167 blend.csv"
blend(blend_a, blend_b)

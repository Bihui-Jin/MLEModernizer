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
import glob
import numpy as np
import pandas as pd



## === cell 1
SAMPLE_PATHS = [
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/ventilator-pressure-prediction/sample_submission.csv",
]
sample_path = next((p for p in SAMPLE_PATHS if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle input paths."
    )
sub = pd.read_csv(sample_path)

CANDIDATES = {
    "sub_1": [
        "../input/vpp-lstm-baseline-median-pp/submission.csv",
        "/kaggle/input/vpp-lstm-baseline-median-pp/submission.csv",
    ],
    "sub_2": [
        "../input/blend-of-blend-of-blend-of-blend-of-blend-of-ble/submission.csv",
        "/kaggle/input/blend-of-blend-of-blend-of-blend-of-blend-of-ble/submission.csv",
    ],
    "sub_3": [
        "../input/gb-vpp-whoppity-dub-dub/median_submission.csv",
        "/kaggle/input/gb-vpp-whoppity-dub-dub/median_submission.csv",
    ],
    "sub_4": [
        "../input/random-weights-blending-tool-ventilator-pressure/rwb 125 loops.csv",
        "/kaggle/input/random-weights-blending-tool-ventilator-pressure/rwb 125 loops.csv",
    ],
}


def _find_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    for p in paths:
        if p.startswith("/kaggle/input/"):
            pattern = "/kaggle/input/**/" + os.path.basename(p)
            hits = glob.glob(pattern, recursive=True)
            if hits:
                return hits[0]
    return None


loaded = {}
missing = []
for k, paths in CANDIDATES.items():
    p = _find_existing(paths)
    if p is None:
        missing.append(k)
        continue
    df = pd.read_csv(p)
    if "pressure" not in df.columns:
        raise ValueError(f"{k} at {p} does not contain 'pressure' column.")
    loaded[k] = df

missing



## === cell 2
TEST_PATHS = [
    "../input/ventilator-pressure-prediction/test.csv",
    "/kaggle/input/ventilator-pressure-prediction/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/ventilator-pressure-prediction/test.csv",
]
test_path_for_mask = next((p for p in TEST_PATHS if os.path.exists(p)), None)
if test_path_for_mask is None:
    raise FileNotFoundError("Could not find test.csv for building u_out mask.")
test_mask_df = pd.read_csv(test_path_for_mask, usecols=["id", "u_out"])
test_mask_df = (
    test_mask_df.merge(sub[["id"]], on="id", how="right")
    .sort_values("id", kind="mergesort")
    .reset_index(drop=True)
)
u_out_mask_aligned = test_mask_df["u_out"].values.astype(np.int8)

if len(loaded) < 4:
    TRAIN_PATHS = [
        "../input/ventilator-pressure-prediction/train.csv",
        "/kaggle/input/ventilator-pressure-prediction/train.csv",
        "/kaggle/data/train.csv",
        "/kaggle/data/ventilator-pressure-prediction/train.csv",
    ]
    train_path = next((p for p in TRAIN_PATHS if os.path.exists(p)), None)
    test_path = next((p for p in TEST_PATHS if os.path.exists(p)), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Could not find train.csv/test.csv in expected Kaggle input paths."
        )

    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    )
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    )

    train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    for df in (train, test):
        df["step"] = df.groupby("breath_id", sort=False).cumcount().astype("int16")
        df["time_step_bin"] = df["step"]

    for df in (train, test):
        df["u_in_prev"] = (
            df.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
        )
        df["u_in_lag2"] = (
            df.groupby("breath_id", sort=False)["u_in"].shift(2).fillna(0.0)
        )
        df["u_in_cum"] = df.groupby("breath_id", sort=False)["u_in"].cumsum()

    UIN_CUM_CAP = 1000.0
    for df in (train, test):
        df["u_in_cum_cap"] = df["u_in_cum"].clip(upper=UIN_CUM_CAP)

    for df in (train, test):
        df["u_in_r"] = (df["u_in"] * 10.0).round().astype("int32")  # 0.1 resolution
        df["u_in_prev_r"] = (df["u_in_prev"] * 10.0).round().astype("int32")
        df["u_in_lag2_r"] = (df["u_in_lag2"] * 10.0).round().astype("int32")
        df["u_in_cum_r"] = (df["u_in_cum"] * 10.0).round().astype("int32")

        df["u_in_cum_cap_r"] = (df["u_in_cum_cap"] * 10.0).round().astype("int32")

        df["u_in_r_02"] = (df["u_in"] * 5.0).round().astype("int32")  # 0.2 resolution
        df["u_in_prev_r_02"] = (df["u_in_prev"] * 5.0).round().astype("int32")
        df["u_in_cum_r_02"] = (df["u_in_cum"] * 5.0).round().astype("int32")

        df["u_in_r_05"] = (df["u_in"] * 2.0).round().astype("int32")  # 0.5 resolution
        df["u_in_prev_r_05"] = (df["u_in_prev"] * 2.0).round().astype("int32")
        df["u_in_cum_r_05"] = (df["u_in_cum"] * 2.0).round().astype("int32")

    train_insp = train[train["u_out"] == 0].copy()

    def _mode_agg(s: pd.Series) -> float:
        vc = s.value_counts()
        if vc.empty:
            return np.nan
        return float(vc.index[0])

    def _build_mode_median_table(df_insp: pd.DataFrame, keys, col_mode, col_med):
        g = df_insp.groupby(keys, sort=False)["pressure"]
        mode_df = g.agg(_mode_agg).rename(col_mode).reset_index()
        med_df = g.median().rename(col_med).reset_index()
        return mode_df.merge(med_df, on=keys, how="left")

    keys_lag2 = [
        "R",
        "C",
        "time_step_bin",
        "u_out",
        "u_in_r",
        "u_in_prev_r",
        "u_in_lag2_r",
        "u_in_cum_r",
    ]
    tab_lag2 = _build_mode_median_table(
        train_insp, keys_lag2, "pred_lag2_mode", "pred_lag2_med"
    )
    test = test.merge(tab_lag2, how="left", on=keys_lag2)

    keys_lag2_cap = [
        "R",
        "C",
        "time_step_bin",
        "u_out",
        "u_in_r",
        "u_in_prev_r",
        "u_in_lag2_r",
        "u_in_cum_cap_r",
    ]
    tab_lag2_cap = _build_mode_median_table(
        train_insp, keys_lag2_cap, "pred_lag2cap_mode", "pred_lag2cap_med"
    )
    test = test.merge(tab_lag2_cap, how="left", on=keys_lag2_cap)

    keys = ["R", "C", "time_step_bin", "u_out", "u_in_r", "u_in_prev_r", "u_in_cum_r"]
    tab = _build_mode_median_table(train_insp, keys, "pred_mode", "pred_med")
    test = test.merge(tab, how="left", on=keys)

    keys_cap = [
        "R",
        "C",
        "time_step_bin",
        "u_out",
        "u_in_r",
        "u_in_prev_r",
        "u_in_cum_cap_r",
    ]
    tab_cap = _build_mode_median_table(
        train_insp, keys_cap, "pred_cap_mode", "pred_cap_med"
    )
    test = test.merge(tab_cap, how="left", on=keys_cap)

    keys_02 = [
        "R",
        "C",
        "time_step_bin",
        "u_out",
        "u_in_r_02",
        "u_in_prev_r_02",
        "u_in_cum_r_02",
    ]
    tab_02 = _build_mode_median_table(
        train_insp, keys_02, "pred_02_mode", "pred_02_med"
    )
    test = test.merge(tab_02, how="left", on=keys_02)

    keys_05 = [
        "R",
        "C",
        "time_step_bin",
        "u_out",
        "u_in_r_05",
        "u_in_prev_r_05",
        "u_in_cum_r_05",
    ]
    tab_05 = _build_mode_median_table(
        train_insp, keys_05, "pred_05_mode", "pred_05_med"
    )
    test = test.merge(tab_05, how="left", on=keys_05)

    keys_1b = ["R", "C", "time_step_bin", "u_out", "u_in_r", "u_in_prev_r"]
    tab_1b = _build_mode_median_table(train_insp, keys_1b, "pred1b_mode", "pred1b_med")
    test = test.merge(tab_1b, how="left", on=keys_1b)

    keys_1b_nouout = ["R", "C", "time_step_bin", "u_in_r", "u_in_prev_r"]
    tab_1b_nouout = _build_mode_median_table(
        train_insp, keys_1b_nouout, "pred1b_nouout_mode", "pred1b_nouout_med"
    )
    test = test.merge(tab_1b_nouout, how="left", on=keys_1b_nouout)

    keys_1c = ["R", "C", "time_step_bin", "u_out", "u_in_r"]
    tab_1c = _build_mode_median_table(train_insp, keys_1c, "pred1c_mode", "pred1c_med")
    test = test.merge(tab_1c, how="left", on=keys_1c)

    keys_1c_nouout = ["R", "C", "time_step_bin", "u_in_r"]
    tab_1c_nouout = _build_mode_median_table(
        train_insp, keys_1c_nouout, "pred1c_nouout_mode", "pred1c_nouout_med"
    )
    test = test.merge(tab_1c_nouout, how="left", on=keys_1c_nouout)

    keys_3 = ["R", "C", "time_step_bin", "u_out"]
    tab_3 = _build_mode_median_table(train_insp, keys_3, "pred3_mode", "pred3_med")
    test = test.merge(tab_3, how="left", on=keys_3)

    keys_4 = ["R", "C", "time_step_bin"]
    tab_4 = _build_mode_median_table(train_insp, keys_4, "pred4_mode", "pred4_med")
    test = test.merge(tab_4, how="left", on=keys_4)

    keys_5 = ["R", "C"]
    tab_5 = _build_mode_median_table(train_insp, keys_5, "pred5_mode", "pred5_med")
    test = test.merge(tab_5, how="left", on=keys_5)

    global_mode = float(train_insp["pressure"].value_counts().index[0])
    global_med = float(train_insp["pressure"].median())

    pred = test["pred_lag2_mode"].fillna(test["pred_lag2_med"])
    pred = (
        pred.fillna(test["pred_lag2cap_mode"])
        .fillna(test["pred_lag2cap_med"])
        .fillna(test["pred_mode"])
        .fillna(test["pred_med"])
        .fillna(test["pred_cap_mode"])
        .fillna(test["pred_cap_med"])
        .fillna(test["pred_02_mode"])
        .fillna(test["pred_02_med"])
        .fillna(test["pred_05_mode"])
        .fillna(test["pred_05_med"])
        .fillna(test["pred1b_mode"])
        .fillna(test["pred1b_med"])
        .fillna(test["pred1b_nouout_mode"])
        .fillna(test["pred1b_nouout_med"])
        .fillna(test["pred1c_mode"])
        .fillna(test["pred1c_med"])
        .fillna(test["pred1c_nouout_mode"])
        .fillna(test["pred1c_nouout_med"])
        .fillna(test["pred3_mode"])
        .fillna(test["pred3_med"])
        .fillna(test["pred4_mode"])
        .fillna(test["pred4_med"])
        .fillna(test["pred5_mode"])
        .fillna(test["pred5_med"])
        .fillna(global_mode)
        .fillna(global_med)
    )

    pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)

    def _snap_to_grid(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        idx = np.searchsorted(grid, x, side="left")
        idx = np.clip(idx, 0, len(grid) - 1)
        left = grid[np.clip(idx - 1, 0, len(grid) - 1)]
        right = grid[idx]
        choose_left = (idx > 0) & ((x - left) <= (right - x))
        out = right.copy()
        out[choose_left] = left[choose_left]
        return out

    pred = _snap_to_grid(pred.values, pressure_grid)

    pred = np.where(test["u_out"].values == 0, pred, 0.0).astype(np.float32)

    fallback = pd.DataFrame({"id": test["id"].values, "pressure": pred})

    for k in ["sub_1", "sub_2", "sub_3", "sub_4"]:
        if k not in loaded:
            loaded[k] = fallback.copy()

_have_grid = "pressure_grid" in globals()

for k in ["sub_1", "sub_2", "sub_3", "sub_4"]:
    df = loaded[k]

    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    df = df[["id", "pressure"]]

    df2 = (
        df.merge(sub[["id"]], on="id", how="right")
        .sort_values("id", kind="mergesort")
        .reset_index(drop=True)
    )

    if _have_grid:
        df2["pressure"] = _snap_to_grid(df2["pressure"].values, pressure_grid)

    df2["pressure"] = np.where(u_out_mask_aligned == 0, df2["pressure"].values, 0.0)

    loaded[k] = df2

sub_1, sub_2, sub_3, sub_4 = (
    loaded["sub_1"],
    loaded["sub_2"],
    loaded["sub_3"],
    loaded["sub_4"],
)
(len(sub), len(sub_1), len(sub_2), len(sub_3), len(sub_4))



## === cell 3
sub["pressure"] = (
    (sub_1["pressure"].values * 0)
    + (sub_2["pressure"].values * 0.14)
    + (sub_3["pressure"].values * 0.68)
    + (sub_4["pressure"].values * 0.18)
)

if "pressure_grid" in globals():
    sub["pressure"] = _snap_to_grid(sub["pressure"].values, pressure_grid)

sub["pressure"] = np.where(u_out_mask_aligned == 0, sub["pressure"].values, 0.0)

sub[["id", "pressure"]].to_csv("submission.csv", index=False)
sub.head(5)

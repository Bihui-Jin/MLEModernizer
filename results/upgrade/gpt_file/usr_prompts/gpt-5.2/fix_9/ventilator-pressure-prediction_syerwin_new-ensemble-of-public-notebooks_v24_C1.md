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

0.1534396912767476

# 6. Current score

4.14834

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.24715) has done: 'I remove the dependency on missing external notebook datasets (the FileNotFoundError sources) and instead build a single-model baseline using only the provided competition `train.csv`/`test.csv`. To keep the “core logic” intent (a simple blending/postprocess script) while making it runnable end-to-end, I generate predictions via a leak-free, per-(R,C,time_step,u_in,u_out) median lookup from the training data, with sensible fallbacks when a combination is unseen. This produces a valid `submission.csv` with the required `id,pressure` columns and should score reasonably (often competitive for a simple baseline) without introducing heavy ML dependencies. Paths are kept within `../input/ventilator-pressure-prediction/` as in your original code, and the output file is guaranteed to have a `.csv` suffix.'
- What this solution (achieved 5.03567) has done: 'Your current score (7.24715 MAE) is far from the target (0.15344), and that gap is consistent with a submission whose predictions are mostly the global median (because exact float matches on `time_step`/`u_in` almost never occur). I keep the same “training-data lookup with fallbacks” core logic, but make the keys matchable by quantizing `time_step` and `u_in` to their natural grid and using an `is_insp` filter so expiratory phase predictions don’t pollute the inspiratory scoring. I also add a final, still-lookup-based fallback at the `(R,C,step)` level to further reduce missingness before resorting to coarse medians. These are minimal changes that preserve the overall approach (groupby medians + merges) but should move the score dramatically toward the target band.'
- What this solution (achieved 4.41216) has done: 'Your score is far above (worse than) the target, so we need a modest but meaningful improvement while keeping the same “training-data lookup via groupby medians + fallbacks” core logic. The biggest issue is that exact matching on `u_in` (even after 2-decimal quantization) still leaves many unseen combinations, so I add a very small, still-lookup-based nearest-neighbor fallback within each `(R,C,step,u_out)` by rounding `u_in` to a coarser bin (0.5) and using that median before dropping `u_in` entirely. I also ensure expiratory rows are handled in a metric-aligned way by predicting 0 for `u_out==1` (these rows are not scored, and setting them to 0 is a safe/stable choice). These changes keep the same semantics (grouped medians merged onto test) but reduce missingness and should move MAE substantially toward the target band.'
- What this solution (achieved 4.40612) has done: 'Your current MAE (4.41) is still far above (worse than) the target (0.153), and this lookup approach is likely failing because `u_in` values in test often don’t exactly match train, even with 0.5 binning. To move toward the target while preserving the same “groupby-median lookup + fallbacks” core logic, I add two minimal, still-lookup-based nearest-bin fallbacks for inspiratory rows: (1) bin `u_in` at 1.0 and (2) bin `u_in` at 2.0, merged before dropping `u_in` entirely. I also align `time_step` quantization to the dataset’s native resolution by rounding `time_step` to 2 decimals (then deriving `step` from that), reducing mismatch caused by float noise. Everything else (no ML model, same merge/fillna cascade, `u_out==1` set to 0, same output `submission.csv`) remains the same.'
- What this solution (achieved 4.40612) has done: 'Your current MAE (4.406) is still far worse than the target (0.153), so we need a clear improvement without changing the core “groupby-median lookup + merge + fillna cascade” approach. The biggest remaining mismatch is that `u_in` is continuous and binning still leaves many unseen keys, so I add a minimal nearest-bin fallback that interpolates between the two closest `u_in` bins within each `(R,C,step,u_out)` using precomputed per-bin medians from train (still a lookup, no new model/training loop). This keeps evaluation semantics intact and should reduce error especially where the exact/rounded `u_in` key is missing or noisy. I also fix `sub` construction to merge predictions onto the actual test `id` order directly (to avoid any accidental misalignment), while keeping the same output path/name `submission.csv`.'
- What this solution (achieved 4.14834) has done: 'Your MAE is still far worse than the target, so we need a small but meaningful improvement while keeping the same “train median lookup + merge + fallback cascade (+ simple interpolation)” core logic. The biggest remaining issue is that your `step` key derived from `time_step` can mismatch between train/test due to rounding-to-2-decimals then dividing by 0.03, which is unstable; we instead compute `step` directly as `round(time_step/0.03)` and clip to the valid [0, 90] range to improve key hit-rate without changing the modeling approach. Next, we switch the primary `u_in` quantization from `round(u_in*100)` to `round(u_in*20)` (0.05 resolution) because exact 0.01-level matching is too strict and causes excessive fallback to coarse medians; the rest of the fallback ladder remains unchanged. Finally, for `u_out==1` rows (not scored), we keep predicting a constant but use the global inspiratory median (instead of 0) to avoid extreme outputs that could hurt if scoring rules differ slightly from expectations.'
- What this solution (achieved 4.14834) has done: 'Your current MAE (4.148) is far worse than the target (0.153), so we need a meaningful improvement while preserving your same core “groupby median lookup + merge + fallback cascade (+ simple interpolation)” approach. The biggest bug hurting accuracy is that you compute all lookup tables only on inspiratory rows (u_out==0) but still join using `u_out` as a key; for test expiratory rows (`u_out==1`), every merge miss and you fall back to a constant, which can also indirectly reduce hit-rate and stability. I build lookup tables separately for inspiratory and expiratory phases (still identical aggregation logic) and apply them phase-wise, while keeping the final output semantics (expiratory predictions are constant and not scored). I also switch the constant for expiratory rows back to 0 (safe since they are not scored in this competition), which should reduce the risk of unintended leakage into any local checks while not worsening the Kaggle metric.'
- What this solution (achieved 4.14834) has done: 'Your current MAE is still far above (worse than) the target, and the biggest remaining issue is that the lookup keys still don’t match well because `u_in` varies continuously and your quantization/fallbacks don’t ensure coverage in the inspiratory phase. I keep the exact same “groupby medians + merge + fillna cascade (+ simple interpolation)” approach, but add one minimal, metric-aligned fallback: a per-(R,C,step) linear regression fit of `pressure ~ u_in` (computed only on inspiratory rows), used only when all existing median/interp fallbacks are missing. This is still a lightweight closed-form lookup-style mapping (no new training loop/architecture), and it should substantially reduce the large errors coming from coarse/global fallbacks. I also restrict that regression fallback to inspiratory predictions only (expiratory remains constant 0 as you already do), and clip predictions to the observed train pressure range for stability.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "../input/ventilator-pressure-prediction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

required_train_cols = {"R", "C", "time_step", "u_in", "u_out", "pressure"}
required_test_cols = {"id", "R", "C", "time_step", "u_in", "u_out"}
if not required_train_cols.issubset(train.columns):
    missing = required_train_cols - set(train.columns)
    raise ValueError(f"train.csv missing columns: {missing}")
if not required_test_cols.issubset(test.columns):
    missing = required_test_cols - set(test.columns)
    raise ValueError(f"test.csv missing columns: {missing}")
if not {"id", "pressure"}.issubset(sub.columns):
    raise ValueError("sample_submission.csv must contain columns: id,pressure")

for col in ["R", "C", "u_out"]:
    train[col] = train[col].astype(np.int16)
    test[col] = test[col].astype(np.int16)

for col in ["time_step", "u_in"]:
    train[col] = train[col].astype(np.float32)
    test[col] = test[col].astype(np.float32)

train["pressure"] = train["pressure"].astype(np.float32)

PRESSURE_MIN = float(train["pressure"].min())
PRESSURE_MAX = float(train["pressure"].max())


def add_quantized_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ts = df["time_step"].to_numpy(dtype=np.float32)
    step = np.rint(ts / np.float32(0.03)).astype(np.int16)
    step = np.clip(step, 0, 90).astype(np.int16)
    df["step"] = step

    df["u_in_q"] = np.rint(
        df["u_in"].to_numpy(dtype=np.float32) * np.float32(20)
    ).astype(
        np.int16
    )  # 0.05

    df["u_in_q05"] = np.rint(
        df["u_in"].to_numpy(dtype=np.float32) * np.float32(2)
    ).astype(
        np.int16
    )  # 0.5

    df["u_in_q1"] = np.rint(
        df["u_in"].to_numpy(dtype=np.float32) * np.float32(1)
    ).astype(
        np.int16
    )  # 1.0

    df["u_in_q2"] = np.rint(
        df["u_in"].to_numpy(dtype=np.float32) * np.float32(0.5)
    ).astype(
        np.int16
    )  # 2.0

    df["u_in_int"] = np.floor(df["u_in"].to_numpy(dtype=np.float32)).astype(np.int16)

    df["is_insp"] = (df["u_out"] == 0).astype(np.int8)
    return df


train_q = add_quantized_keys(train)
test_q = add_quantized_keys(test)

train_insp = train_q[train_q["is_insp"] == 1].copy()
train_exp = train_q[train_q["is_insp"] == 0].copy()


def build_tables(train_slice: pd.DataFrame):
    keys_full = ["R", "C", "step", "u_in_q", "u_out"]
    median_full = (
        train_slice.groupby(keys_full, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_full"})
    )

    keys_uin05 = ["R", "C", "step", "u_in_q05", "u_out"]
    median_uin05 = (
        train_slice.groupby(keys_uin05, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_uin05"})
    )

    keys_uin1 = ["R", "C", "step", "u_in_q1", "u_out"]
    median_uin1 = (
        train_slice.groupby(keys_uin1, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_uin1"})
    )

    keys_uin2 = ["R", "C", "step", "u_in_q2", "u_out"]
    median_uin2 = (
        train_slice.groupby(keys_uin2, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_uin2"})
    )

    keys_no_uin = ["R", "C", "step", "u_out"]
    median_no_uin = (
        train_slice.groupby(keys_no_uin, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_no_uin"})
    )

    keys_rc_step = ["R", "C", "step"]
    median_rc_step = (
        train_slice.groupby(keys_rc_step, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_rc_step"})
    )

    keys_rc_uout = ["R", "C", "u_out"]
    median_rc_uout = (
        train_slice.groupby(keys_rc_uout, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_rc_uout"})
    )

    keys_uin_int = ["R", "C", "step", "u_out", "u_in_int"]
    median_uin_int = (
        train_slice.groupby(keys_uin_int, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_uin_int"})
    )

    global_median = (
        float(train_slice["pressure"].median())
        if len(train_slice)
        else float(train_q["pressure"].median())
    )

    return {
        "keys_full": keys_full,
        "median_full": median_full,
        "keys_uin05": keys_uin05,
        "median_uin05": median_uin05,
        "keys_uin1": keys_uin1,
        "median_uin1": median_uin1,
        "keys_uin2": keys_uin2,
        "median_uin2": median_uin2,
        "keys_no_uin": keys_no_uin,
        "median_no_uin": median_no_uin,
        "keys_rc_step": keys_rc_step,
        "median_rc_step": median_rc_step,
        "keys_rc_uout": keys_rc_uout,
        "median_rc_uout": median_rc_uout,
        "keys_uin_int": keys_uin_int,
        "median_uin_int": median_uin_int,
        "global_median": global_median,
    }


tables_insp = build_tables(train_insp)
tables_exp = build_tables(train_exp)


def build_linfit_rc_step(train_insp_slice: pd.DataFrame) -> pd.DataFrame:
    gcols = ["R", "C", "step"]
    grp = train_insp_slice.groupby(gcols, sort=False)

    n = grp["u_in"].count().astype(np.float32)
    sum_x = grp["u_in"].sum().astype(np.float32)
    sum_y = grp["pressure"].sum().astype(np.float32)
    sum_x2 = grp["u_in"].apply(
        lambda s: np.float32((s.to_numpy(np.float32) ** 2).sum())
    )
    sum_xy = grp.apply(
        lambda df: np.float32(
            (
                df["u_in"].to_numpy(np.float32) * df["pressure"].to_numpy(np.float32)
            ).sum()
        )
    )

    out = pd.DataFrame(
        {
            "R": n.index.get_level_values(0).astype(np.int16),
            "C": n.index.get_level_values(1).astype(np.int16),
            "step": n.index.get_level_values(2).astype(np.int16),
            "n": n.to_numpy(np.float32),
            "sum_x": sum_x.to_numpy(np.float32),
            "sum_y": sum_y.to_numpy(np.float32),
            "sum_x2": np.asarray(sum_x2, dtype=np.float32),
            "sum_xy": np.asarray(sum_xy, dtype=np.float32),
        }
    )

    denom = out["n"] * out["sum_x2"] - out["sum_x"] * out["sum_x"]
    slope = np.where(
        np.abs(denom.to_numpy(np.float32)) > np.float32(1e-6),
        (
            out["n"].to_numpy(np.float32) * out["sum_xy"].to_numpy(np.float32)
            - out["sum_x"].to_numpy(np.float32) * out["sum_y"].to_numpy(np.float32)
        )
        / denom.to_numpy(np.float32),
        np.float32(0.0),
    ).astype(np.float32)

    mean_x = (
        out["sum_x"].to_numpy(np.float32)
        / np.maximum(out["n"].to_numpy(np.float32), np.float32(1.0))
    ).astype(np.float32)
    mean_y = (
        out["sum_y"].to_numpy(np.float32)
        / np.maximum(out["n"].to_numpy(np.float32), np.float32(1.0))
    ).astype(np.float32)

    intercept = (mean_y - slope * mean_x).astype(np.float32)

    out = out[["R", "C", "step"]].copy()
    out["lin_slope"] = slope
    out["lin_intercept"] = intercept
    return out


linfit_insp = build_linfit_rc_step(train_insp)


def predict_with_tables(
    test_slice: pd.DataFrame, tables: dict, linfit_table: pd.DataFrame | None = None
) -> pd.Series:
    pred_cols = [
        "id",
        "R",
        "C",
        "step",
        "u_in",
        "u_in_q",
        "u_in_q05",
        "u_in_q1",
        "u_in_q2",
        "u_in_int",
        "u_out",
        "is_insp",
    ]
    pred = test_slice[pred_cols].copy()

    pred = pred.merge(tables["median_full"], on=tables["keys_full"], how="left")
    pred = pred.merge(tables["median_uin05"], on=tables["keys_uin05"], how="left")
    pred = pred.merge(tables["median_uin1"], on=tables["keys_uin1"], how="left")
    pred = pred.merge(tables["median_uin2"], on=tables["keys_uin2"], how="left")
    pred = pred.merge(tables["median_no_uin"], on=tables["keys_no_uin"], how="left")
    pred = pred.merge(tables["median_rc_step"], on=tables["keys_rc_step"], how="left")
    pred = pred.merge(tables["median_rc_uout"], on=tables["keys_rc_uout"], how="left")

    u0 = pred["u_in_int"].to_numpy(dtype=np.int16)
    u1 = (u0 + 1).astype(np.int16)
    w = (pred["u_in"].to_numpy(dtype=np.float32) - u0.astype(np.float32)).clip(0.0, 1.0)

    base_keys = ["R", "C", "step", "u_out"]
    tmp0 = pred[base_keys].copy()
    tmp0["u_in_int"] = u0
    tmp1 = pred[base_keys].copy()
    tmp1["u_in_int"] = u1

    tmp0 = tmp0.merge(
        tables["median_uin_int"], on=tables["keys_uin_int"], how="left"
    ).rename(columns={"pred_uin_int": "pred_u0"})
    tmp1 = tmp1.merge(
        tables["median_uin_int"], on=tables["keys_uin_int"], how="left"
    ).rename(columns={"pred_uin_int": "pred_u1"})

    pred_u0 = tmp0["pred_u0"].to_numpy(dtype=np.float32)
    pred_u1 = tmp1["pred_u1"].to_numpy(dtype=np.float32)

    pred_interp = pred_u0 * (1.0 - w) + pred_u1 * w
    pred_interp = np.where(np.isfinite(pred_interp), pred_interp, np.nan).astype(
        np.float32
    )

    pred_pressure = pred["pred_full"].astype(np.float32)
    pred_pressure = pred_pressure.fillna(pred["pred_uin05"])
    pred_pressure = pred_pressure.fillna(pred["pred_uin1"])
    pred_pressure = pred_pressure.fillna(pred["pred_uin2"])
    pred_pressure = pred_pressure.fillna(pd.Series(pred_interp, index=pred.index))
    pred_pressure = pred_pressure.fillna(pred["pred_no_uin"])
    pred_pressure = pred_pressure.fillna(pred["pred_rc_step"])
    pred_pressure = pred_pressure.fillna(pred["pred_rc_uout"])

    if linfit_table is not None and len(linfit_table) > 0:
        lin = pred[["R", "C", "step", "u_in"]].merge(
            linfit_table, on=["R", "C", "step"], how="left"
        )
        lin_pred = (
            lin["lin_slope"].to_numpy(np.float32) * lin["u_in"].to_numpy(np.float32)
            + lin["lin_intercept"].to_numpy(np.float32)
        ).astype(np.float32)
        lin_pred = np.where(np.isfinite(lin_pred), lin_pred, np.nan).astype(np.float32)
        pred_pressure = pred_pressure.fillna(pd.Series(lin_pred, index=pred.index))

    pred_pressure = pred_pressure.fillna(tables["global_median"]).astype(np.float32)

    pred_pressure = pred_pressure.clip(PRESSURE_MIN, PRESSURE_MAX).astype(np.float32)
    return pred_pressure


test_insp = test_q[test_q["is_insp"] == 1].copy()
test_exp = test_q[test_q["is_insp"] == 0].copy()

pred_insp = predict_with_tables(test_insp, tables_insp, linfit_table=linfit_insp)
pred_exp = predict_with_tables(test_exp, tables_exp, linfit_table=None)

pred_pressure = pd.Series(index=test_q.index, dtype=np.float32)
pred_pressure.loc[test_insp.index] = pred_insp.to_numpy(dtype=np.float32)

pred_pressure.loc[test_exp.index] = np.float32(0.0)

pred_pressure = pred_pressure.to_numpy(dtype=np.float32)

sub_out = test[["id"]].copy()
sub_out["pressure"] = pred_pressure

if len(sub_out) != len(test):
    raise RuntimeError(
        f"Submission row count {len(sub_out)} != test row count {len(test)}"
    )
if sub_out["pressure"].isna().any():
    raise RuntimeError("NaNs found in predicted pressure")

sub_out.to_csv("submission.csv", index=False)
sub_out.head()

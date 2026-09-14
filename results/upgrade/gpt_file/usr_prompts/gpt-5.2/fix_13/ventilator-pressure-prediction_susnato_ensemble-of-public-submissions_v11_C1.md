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

3.9

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

0.2098365471597277

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your notebook fails because it tries to read multiple other notebooks’ `../input/.../submission.csv` files that don’t exist in this Kaggle environment, so `sub_1`…`sub_9` are never defined and the ensemble step crashes. To make this run end-to-end and still follow the intended “submission blending” core logic, I (1) load the official `sample_submission.csv` as the template, (2) attempt to load those external submissions only if present, and (3) fall back to a simple, valid baseline prediction (pressure=0) if none are available. I also fix a small logic bug where `sub_8` was accidentally used twice and `sub_9` was never used, and ensure the output is written as `submission.csv` with the required columns.'
- What this solution (achieved 10.73281) has done: 'Your current script is only averaging other notebooks’ submission files; since none exist in this environment, it falls back to predicting all zeros, which explains the very poor MAE. To move your score much closer to the target while preserving the “simple, no-training, direct submission generation” core logic, I keep the same pipeline but replace the fallback with a legitimate, fast baseline derived from the provided train/test inputs: predict pressure as a lookup table over the exact observed `(R, C, u_in, u_out, time_step)` combinations. This is minimal, deterministic, and uses only allowed data (no leakage from test labels). If the external submissions are present, the code still blend them, but it also blend in this baseline to avoid catastrophic failure.'
- What this solution (achieved 4.00105) has done: 'Your current fallback baseline is a coarse lookup on raw floating `time_step` and `u_in`, which causes many unseen key combinations in test and forces lots of fill-with-global-median, keeping MAE high. To move the score much closer to the target while preserving the same “no-training, submission blending with a baseline fallback” core logic, I keep the lookup-table approach but (1) quantize `u_in` and `time_step` to stabilize key matching, and (2) add a minimal hierarchical fallback: exact `(R,C,u_out,t,u_in)` median → `(R,C,u_out,t)` median → `(R,C,u_out)` median → global median. This only changes the baseline construction (still deterministic and purely train-derived) and keeps the external-submission blending intact. It should substantially reduce the fraction of NA baseline rows and improve MAE toward your target without changing the overall pipeline structure.'
- What this solution (achieved 4.14367) has done: 'We keep your “no-training + lookup-table baseline + optional external blending” core logic, but make the baseline match more test rows by quantizing `time_step` and `u_in` to the *true* grids used in this dataset (time_step is 0.03s increments; u_in is effectively 0.1). Then we add one additional minimal hierarchical fallback keyed by `(R,C,u_out,u_in)` to recover cases where time alignment differs slightly, reducing the number of rows that fall back to the global median. Finally, we blend the baseline with external submissions using a small fixed weight (instead of a plain mean) so the baseline can meaningfully pull predictions away from weak/mismatched externals, aiming to reduce MAE toward your target.'
- What this solution (achieved 4.14367) has done: 'Your current MAE is far above the target, so we need a real (but still “no-training”) improvement while keeping your core “lookup-table baseline + optional blending” approach intact. The biggest issue is that pressure is only scored during inspiration (`u_out==0`), yet the baseline currently mixes expiratory (`u_out==1`) pressure patterns into the lookup, which hurts predictions a lot. I rebuild the lookup tables using only inspiratory rows from train (and fall back to global inspiratory median), while keeping the same hierarchical merge logic and the same external-submission blending. This is a minimal change that should materially reduce MAE and move you toward the target without changing the overall method.'
- What this solution (achieved 4.14367) has done: 'Your current score is far worse than the target (lower is better), so we need a meaningful improvement while keeping your “no-training lookup baseline + optional blending” core logic unchanged. The biggest remaining issue is that the lookup tables are built only from inspiratory rows but you still try to match/merge with `u_out` in the keys, which makes all `u_out==1` test rows miss the LUT and fall back to coarse global medians; while expiratory rows aren’t scored, bad expiratory predictions can still leak into blending/averaging behavior and can also indicate mismatched alignment. I keep the same hierarchical LUT approach, but build two parallel LUT stacks: one from inspiratory (`u_out==0`) and one from expiratory (`u_out==1`), and then use the appropriate stack per-row; this is minimal, deterministic, and should reduce fallback usage and improve MAE toward your target. I also add one extra very-small fallback keyed by `(R,C,time_step_q,u_in_q)` (dropping `u_out`) only when the correct-phase LUT misses, to catch rare `u_out` inconsistencies without changing the overall method.'
- What this solution (achieved 4.24782) has done: 'We keep your no-training lookup-table baseline + optional external blending intact, but fix two high-impact issues that are currently holding MAE around ~4. First, the LUT uses median pressure; switching the aggregation to mean (still a deterministic train-only lookup) is better aligned with MAE and typically reduces error without changing the overall approach. Second, we add a tiny “pressure grid snapping” post-process: in this competition pressure takes values on a fixed discrete grid, so snapping predictions to the nearest observed training pressure value often yields a large MAE drop while preserving the same prediction semantics (still producing a pressure per id). These are minimal changes, fast, and should move your score substantially closer to the 0.21 target.'
- What this solution (achieved 4.24782) has done: 'Your current score (4.24782, lower-is-better) is still far from the target (0.2098), so we need a meaningful but still “no-training LUT baseline + optional blending” improvement. The main bug is that your quantization doesn’t match the dataset grid: `u_in` should be quantized to 0.1 increments (not rounded to 1 decimal without scaling), and `time_step` is safer quantized on the exact 0.03 grid using integer rounding; the current mismatch causes widespread LUT misses and forces global-mean fallbacks. I fix quantization, and add one minimal extra hierarchical fallback keyed by `(R, C, time_step_q, u_in_q)` built from the correct phase (insp/exp) to recover cases where `u_out` mismatches; this preserves your core lookup/merge logic while reducing NA and should move MAE substantially toward the target. The rest of your pipeline (external blending, grid snapping, submission writing) stays intact.'
- What this solution (achieved 4.24782) has done: 'Your MAE is still far above the target (lower is better), so we should improve the existing lookup-table baseline without changing the overall “no-training LUT + optional blending + pressure snapping” approach. The biggest win available with minimal disruption is to add a more appropriate fallback that uses the known discretization of `pressure`: instead of falling back to global means when a key is unseen, fall back to the nearest available `u_in_q` at the same `(R,C,u_out,time_step_q)` (and then the nearest `time_step_q` if needed). This keeps the same LUT logic but drastically reduces “uninformed” fills that cause large errors. Finally, we keep your blending and snapping, but we also snap the baseline-derived fills via the same pressure grid (still deterministic) so the ensemble input is on-manifold.'
- What this solution (achieved 4.24782) has done: 'Your current MAE (4.24782, lower-is-better) is still far above the target (0.2098), so we need to substantially improve the lookup-table baseline while preserving your “no-training LUT + optional blending + pressure snapping” approach. The biggest remaining gap is that the LUT is trying to predict a continuous target, but in this competition `pressure` lies on a fixed discrete grid; we can reduce MAE a lot by doing all LUT aggregation and all fills on that grid (train-only), instead of only snapping at the very end. Concretely, we (1) snap `train["pressure"]` to the nearest training grid value before building LUTs (this preserves semantics because it’s the same grid), and (2) also snap the nearest-neighbor fallback fills to the grid before mixing into the baseline. Everything else (quantization, hierarchical fallbacks, optional external blending, and final snapping) stays the same.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
if not os.path.exists(sub_path):
    sub_path = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
sub = pd.read_csv(sub_path)

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
if not os.path.exists(train_path):
    train_path = "/kaggle/input/ventilator-pressure-prediction/train.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/input/ventilator-pressure-prediction/test.csv"

train = pd.read_csv(
    train_path, usecols=["R", "C", "time_step", "u_in", "u_out", "pressure"]
)
test = pd.read_csv(test_path, usecols=["id", "R", "C", "time_step", "u_in", "u_out"])

candidate_paths = [
    ("sub_1", "../input/improvement-base-on-tensor-bidirect-lstm-0-173/submission.csv"),
    ("sub_2", "../input/ventilator-pressure-eda-lstm-0-189/lstm.csv"),
    ("sub_3", "../input/k/shivansh002/i-am-groot/submission.csv"),
    ("sub_4", "../input/tensorflow-bi-lstm-with-tpu/submission.csv"),
    ("sub_5", "../input/tensorflow-bidirectional-lstm-0-234/submission.csv"),
    ("sub_6", "../input/lightautoml-continuer/submission.csv"),
    ("sub_7", "../input/lightautoml-starter/submission.csv"),
    ("sub_8", "../input/tensorflow/submission.csv"),
    ("sub_9", "../input/tensorflow-lstm-baseline/submission.csv"),
]

loaded = []
for name, path in candidate_paths:
    if path.startswith("../input/"):
        alt_path2 = path.replace("../input/", "/kaggle/input/")
    else:
        alt_path2 = path

    found_path = None
    for p in (path, alt_path2):
        if os.path.exists(p):
            found_path = p
            break
    if found_path is None:
        continue

    df = pd.read_csv(found_path)
    if "pressure" not in df.columns:
        continue
    if len(df) != len(sub):
        continue

    loaded.append(df[["pressure"]].rename(columns={"pressure": name}))

pressure_grid = np.sort(train["pressure"].unique().astype(np.float64))


def _snap_to_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    idx = np.searchsorted(grid, values, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    idx_left = np.clip(idx - 1, 0, len(grid) - 1)
    right = grid[idx]
    left = grid[idx_left]
    return np.where(
        np.abs(values - left) <= np.abs(values - right), left, right
    ).astype(np.float64)


train = train.copy()
train["pressure"] = _snap_to_grid(
    train["pressure"].to_numpy(dtype=np.float64), pressure_grid
)


def _quantize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["time_step"].astype(np.float64).to_numpy()
    ui = df["u_in"].astype(np.float64).to_numpy()

    df["time_step_q"] = (np.rint(ts / 0.03).astype(np.int32) * 0.03).astype(np.float64)
    df["time_step_q"] = np.round(df["time_step_q"], 2)

    df["u_in_q"] = (np.rint(ui * 10.0).astype(np.int32) / 10.0).astype(np.float64)
    df["u_in_q"] = np.round(df["u_in_q"], 1)
    return df


train_q = _quantize_features(train)
test_q = _quantize_features(test)

train_insp = train_q[train_q["u_out"].values == 0].copy()
train_exp = train_q[train_q["u_out"].values == 1].copy()

global_mean_insp = float(train_insp["pressure"].mean())
global_mean_exp = (
    float(train_exp["pressure"].mean())
    if len(train_exp)
    else float(train_q["pressure"].mean())
)
global_mean_all = float(train_q["pressure"].mean())

keys_full = ["R", "C", "u_out", "time_step_q", "u_in_q"]
keys_t = ["R", "C", "u_out", "time_step_q"]
keys_uin = ["R", "C", "u_out", "u_in_q"]
keys_uout = ["R", "C", "u_out"]
keys_full_nouout = ["R", "C", "time_step_q", "u_in_q"]


def _build_luts(df_phase: pd.DataFrame, suffix: str):
    lut_full = (
        df_phase.groupby(keys_full, sort=False)["pressure"]
        .mean()
        .rename(f"p_full{suffix}")
        .reset_index()
    )
    lut_t = (
        df_phase.groupby(keys_t, sort=False)["pressure"]
        .mean()
        .rename(f"p_t{suffix}")
        .reset_index()
    )
    lut_uin = (
        df_phase.groupby(keys_uin, sort=False)["pressure"]
        .mean()
        .rename(f"p_uin{suffix}")
        .reset_index()
    )
    lut_uout = (
        df_phase.groupby(keys_uout, sort=False)["pressure"]
        .mean()
        .rename(f"p_uout{suffix}")
        .reset_index()
    )
    lut_full_nouout = (
        df_phase.groupby(keys_full_nouout, sort=False)["pressure"]
        .mean()
        .rename(f"p_full_nouout{suffix}")
        .reset_index()
    )
    return lut_full, lut_t, lut_uin, lut_uout, lut_full_nouout


lut_full_i, lut_t_i, lut_uin_i, lut_uout_i, lut_full_nouout_i = _build_luts(
    train_insp, "_i"
)
lut_full_e, lut_t_e, lut_uin_e, lut_uout_e, lut_full_nouout_e = (
    _build_luts(train_exp, "_e") if len(train_exp) else (None, None, None, None, None)
)

test_with_base = test_q.copy()

test_with_base = test_with_base.merge(lut_full_i, on=keys_full, how="left", copy=False)
test_with_base = test_with_base.merge(lut_t_i, on=keys_t, how="left", copy=False)
test_with_base = test_with_base.merge(lut_uin_i, on=keys_uin, how="left", copy=False)
test_with_base = test_with_base.merge(lut_uout_i, on=keys_uout, how="left", copy=False)
test_with_base = test_with_base.merge(
    lut_full_nouout_i, on=keys_full_nouout, how="left", copy=False
)

if len(train_exp):
    test_with_base = test_with_base.merge(
        lut_full_e, on=keys_full, how="left", copy=False
    )
    test_with_base = test_with_base.merge(lut_t_e, on=keys_t, how="left", copy=False)
    test_with_base = test_with_base.merge(
        lut_uin_e, on=keys_uin, how="left", copy=False
    )
    test_with_base = test_with_base.merge(
        lut_uout_e, on=keys_uout, how="left", copy=False
    )
    test_with_base = test_with_base.merge(
        lut_full_nouout_e, on=keys_full_nouout, how="left", copy=False
    )
else:
    for col in ["p_full_e", "p_t_e", "p_uin_e", "p_uout_e", "p_full_nouout_e"]:
        test_with_base[col] = np.nan

u_out_arr = test_with_base["u_out"].values.astype(int)

p_i = test_with_base["p_full_i"]
p_i = p_i.fillna(test_with_base["p_t_i"])
p_i = p_i.fillna(test_with_base["p_uin_i"])
p_i = p_i.fillna(test_with_base["p_uout_i"])
p_i = p_i.fillna(test_with_base["p_full_nouout_i"])

p_e = test_with_base["p_full_e"]
p_e = p_e.fillna(test_with_base["p_t_e"])
p_e = p_e.fillna(test_with_base["p_uin_e"])
p_e = p_e.fillna(test_with_base["p_uout_e"])
p_e = p_e.fillna(test_with_base["p_full_nouout_e"])

baseline = np.where(u_out_arr == 0, p_i.to_numpy(), p_e.to_numpy()).astype(float)


def _nearest_uin_fill_asof(
    train_phase: pd.DataFrame, test_phase: pd.DataFrame
) -> np.ndarray:
    if len(test_phase) == 0:
        return np.empty((0,), dtype=np.float64)

    g = (
        train_phase.groupby(["R", "C", "u_out", "time_step_q", "u_in_q"], sort=False)[
            "pressure"
        ]
        .mean()
        .reset_index()
    )

    sort_u_cols = ["R", "C", "u_out", "time_step_q", "u_in_q"]
    g_u = g.sort_values(sort_u_cols, kind="mergesort").reset_index(drop=True)

    t_u = test_phase[["R", "C", "u_out", "time_step_q", "u_in_q"]].copy()
    t_u["_row"] = np.arange(len(t_u), dtype=np.int32)
    t_u = t_u.sort_values(sort_u_cols, kind="mergesort").reset_index(drop=True)

    m_u = pd.merge_asof(
        t_u,
        g_u,
        on="u_in_q",
        by=["R", "C", "u_out", "time_step_q"],
        direction="nearest",
        allow_exact_matches=True,
    )
    out = np.full(len(test_phase), np.nan, dtype=np.float64)
    out[m_u["_row"].to_numpy()] = m_u["pressure"].to_numpy(dtype=np.float64)

    miss_mask = np.isnan(out)
    if miss_mask.any():
        g_ts = (
            train_phase.groupby(["R", "C", "u_out", "time_step_q"], sort=False)[
                "pressure"
            ]
            .mean()
            .reset_index()
        )
        sort_ts_cols = ["R", "C", "u_out", "time_step_q"]
        g_ts = g_ts.sort_values(sort_ts_cols, kind="mergesort").reset_index(drop=True)

        t_ts = test_phase.loc[miss_mask, ["R", "C", "u_out", "time_step_q"]].copy()
        t_ts["_row"] = np.flatnonzero(miss_mask).astype(np.int32)
        t_ts = t_ts.sort_values(sort_ts_cols, kind="mergesort").reset_index(drop=True)

        m_ts = pd.merge_asof(
            t_ts,
            g_ts,
            on="time_step_q",
            by=["R", "C", "u_out"],
            direction="nearest",
            allow_exact_matches=True,
        )
        out[m_ts["_row"].to_numpy()] = m_ts["pressure"].to_numpy(dtype=np.float64)

    return out


test_insp = test_with_base[u_out_arr == 0].copy()
test_exp = test_with_base[u_out_arr == 1].copy()

nn_fill_insp = _nearest_uin_fill_asof(train_insp, test_insp)
nn_fill_exp = (
    _nearest_uin_fill_asof(train_exp, test_exp)
    if len(train_exp)
    else np.full(len(test_exp), np.nan, dtype=np.float64)
)

fill_vals = np.empty(len(test_with_base), dtype=np.float64)
fill_vals[u_out_arr == 0] = nn_fill_insp
fill_vals[u_out_arr == 1] = nn_fill_exp

fill_vals = _snap_to_grid(
    pd.Series(fill_vals, index=test_with_base.index)
    .fillna(
        pd.Series(
            np.where(u_out_arr == 0, global_mean_insp, global_mean_exp),
            index=test_with_base.index,
        ).astype(float)
    )
    .to_numpy(dtype=np.float64),
    pressure_grid,
)

baseline = (
    pd.Series(baseline, index=test_with_base.index)
    .fillna(pd.Series(fill_vals, index=test_with_base.index))
    .astype(float)
)

baseline = _snap_to_grid(baseline.to_numpy(dtype=np.float64), pressure_grid)
test_with_base["baseline"] = baseline

baseline_pred = (
    test_with_base.set_index("id")
    .loc[sub["id"].values, "baseline"]
    .values.astype(float)
)

if loaded:
    preds = pd.concat(loaded, axis=1)
    preds["baseline"] = baseline_pred
else:
    preds = pd.DataFrame({"baseline": baseline_pred}, index=sub.index)

na_rate_full_i = float(test_with_base["p_full_i"].isna().mean())
na_rate_full_e = float(test_with_base["p_full_e"].isna().mean())
print(
    f"Loaded {len(loaded)} external submissions for blending. "
    f"Phase full-key NA rates: insp={na_rate_full_i:.3f}, exp={na_rate_full_e:.3f}"
)
print(
    f"Global means: insp(u_out==0)={global_mean_insp:.5f} | exp(u_out==1)={global_mean_exp:.5f} | all={global_mean_all:.5f}"
)
print(preds.describe().T[["mean", "std", "min", "max"]].head(10))
print(
    f"Pressure grid size: {pressure_grid.size} | min={pressure_grid.min():.2f} max={pressure_grid.max():.2f}"
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2998731707.py in <cell line: 0>()
    263 test_exp = test_with_base[u_out_arr == 1].copy()
    264 
--> 265 nn_fill_insp = _nearest_uin_fill_asof(train_insp, test_insp)
    266 nn_fill_exp = (
    267     _nearest_uin_fill_asof(train_exp, test_exp)

/tmp/ipykernel_11/2998731707.py in _nearest_uin_fill_asof(train_phase, test_phase)
    220     t_u = t_u.sort_values(sort_u_cols, kind="mergesort").reset_index(drop=True)
    221 
--> 222     m_u = pd.merge_asof(
    223         t_u,
    224         g_u,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge_asof(left, right, on, left_on, right_on, left_index, right_index, by, left_by, right_by, suffixes, tolerance, allow_exact_matches, direction)
    706         direction=direction,
    707     )
--> 708     return op.get_result()
    709 
    710 

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in get_result(self, copy)
   1924 
   1925     def get_result(self, copy: bool | None = True) -> DataFrame:
-> 1926         join_index, left_indexer, right_indexer = self._get_join_info()
   1927 
   1928         left_join_indexer: npt.NDArray[np.intp] | None

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_join_info(self)
   1149             )
   1150         else:
-> 1151             (left_indexer, right_indexer) = self._get_join_indexers()
   1152 
   1153             if self.right_index:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_join_indexers(self)
   2236 
   2237         # initial type conversion as needed
-> 2238         left_values = self._convert_values_for_libjoin(left_values, "left")
   2239         right_values = self._convert_values_for_libjoin(right_values, "right")
   2240 

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _convert_values_for_libjoin(self, values, side)
   2180             if isna(values).any():
   2181                 raise ValueError(f"Merge keys contain null values on {side} side")
-> 2182             raise ValueError(f"{side} keys must be sorted")
   2183 
   2184         if isinstance(values, ArrowExtensionArray):

ValueError: left keys must be sorted

## === cell 1
if loaded:
    ext_cols = [c for c in preds.columns if c != "baseline"]
    ext_mean = preds[ext_cols].mean(axis=1).astype(float).values
    base = preds["baseline"].astype(float).values
    w_base = 0.70  # keep your blending scheme; baseline should now be stronger and more consistent
    final_pred = w_base * base + (1.0 - w_base) * ext_mean
else:
    final_pred = preds["baseline"].astype(float).values

idx = np.searchsorted(pressure_grid, final_pred, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)
idx_left = np.clip(idx - 1, 0, len(pressure_grid) - 1)
right = pressure_grid[idx]
left = pressure_grid[idx_left]
final_pred_snapped = np.where(
    np.abs(final_pred - left) <= np.abs(final_pred - right), left, right
).astype(float)

sub["pressure"] = final_pred_snapped
sub[["id", "pressure"]].to_csv("submission.csv", index=False)
sub.head(5)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1961568260.py in <cell line: 0>()
      6     final_pred = w_base * base + (1.0 - w_base) * ext_mean
      7 else:
----> 8     final_pred = preds["baseline"].astype(float).values
      9 
     10 idx = np.searchsorted(pressure_grid, final_pred, side="left")

NameError: name 'preds' is not defined

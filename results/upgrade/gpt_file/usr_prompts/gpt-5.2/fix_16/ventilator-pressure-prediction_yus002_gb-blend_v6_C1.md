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

3.9

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

0.4008231252106993

# 6. Current score

3.1719

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.63464) has done: 'Your script currently can’t yield a Kaggle score because it expects external prediction CSVs in `../input/gb-blending`, which don’t exist in the provided data paths, and it also writes a submission filename without the required `.csv` suffix. To produce a valid, scorable submission with minimal change and without altering the overall “blend predictions into sample_submission” core idea, I (1) fall back to generating a simple baseline prediction directly from `test.csv` when the blend folder is missing/empty, and (2) always write `submission.csv` with the required `id,pressure` columns. Since lower MAE is better and we currently have “Not yielded”, any valid nontrivial baseline move the score toward the target (0.4008) compared to no score. The baseline uses a known safe heuristic for this competition: predict 0 pressure during expiration (`u_out==1`) and a constant inspiratory pressure during inspiration.'
- What this solution (achieved 6.10613) has done: 'Your current score (7.63464, lower-is-better) is far from the target (~0.4008), so we should make a small but meaningful improvement without changing the overall “produce submission via fallback when blend files are missing” core flow. The biggest issue in the fallback is predicting a constant pressure during inspiration, which is much too crude; we can instead predict the mean pressure for each `(R, C, time_step)` during inspiration from the training data and use that lookup on test (a common, lightweight baseline for this competition). This preserves the same basic approach (no ML training loop, no architecture changes) while aligning better with the metric that only scores inspiratory (`u_out==0`) timesteps. We keep output formatting intact and still write `submission.csv` with `id,pressure`.'
- What this solution (achieved 6.19013) has done: 'Your current fallback builds predictions by averaging pressure over `(R, C, time_step)` during inspiration, but rounding `time_step` to 2 decimals still leaves many test timesteps with no exact match, forcing lots of fill-with-global-mean and hurting MAE. To move the score substantially closer to the target while keeping the same “lookup from train, merge onto test, zero during expiration” core logic, I (1) switch the lookup key from rounded `time_step` to an integer timestep index within each breath (1..80), which aligns perfectly between train/test, and (2) add a tiny hierarchical fallback: first `(R,C,step)` mean, then `(R,C)` mean, then global inspiratory mean. This keeps evaluation semantics identical and should reduce the large error coming from mismatched time keys, while still writing a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 5.78556) has done: 'We need to move the MAE down from 6.19 toward 0.4008 (lower is better), so we should improve the fallback predictor while keeping the same “lookup means from train and merge onto test; set expiration to 0; write submission.csv” core logic. The biggest remaining issue is that we’re averaging across all breaths, which blurs different inspiratory trajectories; adding a simple per-breath normalization feature (`u_in` cumulative integral and lag) to the grouping keeps the approach as a pure groupby-lookup but makes the mapping much sharper. We also ensure step indexing is 0..79 consistently and keep hierarchical fallbacks to avoid missing keys. This should reduce large systematic errors without introducing any training loop or model changes and still produces a valid `submission.csv`.'
- What this solution (achieved 4.00309) has done: 'We need to move MAE down from 5.78556 toward 0.4008 (lower is better), so we should make the smallest change that meaningfully improves the existing “groupby-lookup from train then merge onto test; set expiration to 0” fallback without changing the overall approach. The biggest remaining weakness is the coarse binning of the dynamic features (`u_in_cum_bin`, `u_in_lag1_bin`) which creates lots of unseen keys in test and forces frequent fallback to broad averages; tightening this by using integer quantization (robust keys) and adding an additional intermediate fallback on `u_in` itself reduces missing merges while preserving the same lookup/merge semantics. We also ensure the `id` alignment is preserved by sorting back to the sample submission `id` order before writing. These changes keep runtime within limits and still write a valid `submission.csv`.'
- What this solution (achieved 3.62605) has done: 'We keep your existing “fallback lookup from train then merge onto test; set `u_out==1` to 0; write `submission.csv`” core logic, but make the lookup slightly less sparse so fewer test rows fall back to coarse averages. Concretely, we add one extra intermediate fallback table keyed by `(R, C, step, u_in_lag1_q)` and insert it between your current `step_dyn` and `step_uin` fallbacks, which should reduce MAE without changing the approach or adding any ML training loop. We also clip predictions to the observed training pressure range to prevent rare out-of-range means from increasing MAE. Paths, output columns, and the `.csv` submission writing are kept intact.'
- What this solution (achieved 3.65917) has done: 'We’re still far from the target (3.626 → 0.401 MAE, lower is better), so we make one minimal improvement to your existing groupby-lookup fallback without changing the overall “derive mean pressure tables from train, merge onto test, hierarchical fill, set expiration to 0, write submission.csv” core logic. The main adjustment is to add `u_in` itself into the most-detailed lookup key (alongside your existing `u_in_cum_q` and `u_in_lag1_q`) to reduce collisions where different trajectories share similar cumulative/lag values but differ in current valve opening. We keep your same hierarchical fallback structure and keep clipping to the observed train pressure range for stability. Output path/filename and required `id,pressure` format remain unchanged.'
- What this solution (achieved 3.47366) has done: 'We need to move your MAE down from 3.659 toward 0.401 (lower is better), while keeping the same “train-derived lookup tables → merge onto test → hierarchical fill → set `u_out==1` to 0 → write `submission.csv`” core logic. The most impactful minimal change is to prevent over-sparsity in the most-detailed key: we keep your current keys, but add a parallel “softer” dynamic table where `u_in_cum` is quantized less aggressively (fewer misses), then insert it between the current `step_dyn` and `step_lag` fallbacks. This preserves evaluation semantics, doesn’t add any ML training loop, and typically reduces the number of rows falling back to coarse averages. We also keep your clipping and exact submission formatting unchanged.'
- What this solution (achieved 3.22738) has done: 'Your current score (3.47366, lower-is-better) is still far above the target (0.4008), so we should make a small, safe improvement to your existing “train-derived lookup tables → merge onto test → hierarchical fill → set `u_out==1` to 0 → write `submission.csv`” fallback without changing the overall approach. The biggest remaining gap comes from the lookup keys being too sparse, so many rows fall back to coarse averages; we add one intermediate lookup keyed by `(R,C,step,u_in_q)` plus a coarser cumulative bin and keep the same fill hierarchy. We also align the clipping with what’s actually evaluated by clipping only inspiratory predictions (leaving expiration forced to 0), which avoids unintended distortion. Paths and submission formatting remain unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 3.21466) has done: 'We keep your existing “train-derived lookup tables → merge onto test → hierarchical fill → set `u_out==1` to 0 → write submission.csv” core logic, but make the lookup less sparse in a minimal way so fewer test rows fall back to coarse averages (which is currently keeping MAE at ~3.23, far above the 0.40 target). Concretely, we add one intermediate mean table keyed by `(R, C, step, u_in_q)` with a *coarser* `u_in_q` (integer `u_in`), and use it only as an additional fallback between your current `step_uin` and `step` tables. This preserves semantics (still pure groupby means, no training loop/model changes) while improving match rate and should move the score downward toward the target. We also keep the expiratory forcing to 0 and the inspiratory clipping exactly as before to avoid unintended metric regressions.'
- What this solution (achieved 3.21461) has done: 'Your current MAE (3.21466, lower-is-better) is still far above the target (0.40082), so we should make a small improvement that keeps your exact “train-derived mean lookup tables → merge onto test → hierarchical fill → force `u_out==1` to 0 → write submission.csv” core logic. The biggest low-risk gain available without changing the approach is to better respect the competition’s discretized pressure levels by snapping inspiratory predictions to the nearest observed training pressure value (a standard post-processing for this competition that typically reduces MAE). We keep all your existing lookup tables and fallback order intact, and only add a precomputed `pressure_grid` from train plus a vectorized nearest-neighbor quantization step before writing the submission. This should move the score downward toward the target while preserving evaluation semantics and producing the same valid `submission.csv`.'
- What this solution (achieved 3.21459) has done: 'We keep your current “train-derived mean lookup tables → merge onto test → hierarchical fill → expiration=0 → write submission.csv” logic intact, but make one minimal post-processing improvement that typically reduces MAE for this competition. Specifically, after snapping to the nearest observed pressure grid, we add an optional second snap using a small local neighborhood to correct rare ties/edge cases caused by floating-point/searchsorted boundaries. This is a tiny, deterministic adjustment (no new model, no new training) and should nudge the MAE downward toward the target while preserving the same overall approach and submission format. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 3.173) has done: 'Your current MAE (3.21459, lower-is-better) is still far above the target (0.40082), so we should make a small, safe improvement that keeps your exact “train-derived mean lookup tables → merge onto test → hierarchical fill → expiration=0 → write submission.csv” core logic intact. The minimal change most likely to reduce error is to make the most-detailed lookup slightly less sparse by adding one extra intermediate mean table keyed on a *coarser* current-`u_in` quantization (integer `u_in`) together with your existing cumulative/lag dynamics, so more test rows match without falling back to coarse averages. We keep all existing fallbacks, clipping, and the pressure-grid snapping, and we do not change any paths or the submission schema. This should nudge the score downward (better) toward the target while remaining deterministic and within time limits.'
- What this solution (achieved 3.1719) has done: 'I fix the `IndexError` in the nearest-neighbor fallback by ensuring we never index `uniq_k[pos]` when `pos == len(uniq_k)` (NumPy evaluates both sides of `&`, so the previous mask didn’t protect the indexing). This is a pure bugfix that preserves your existing lookup/merge/blend logic and should allow the notebook to run end-to-end and write `submission.csv`. I also make the input paths robust to your provided `/kaggle/input/...` layout while keeping the original relative paths as fallbacks, so the code can actually find the CSVs in this environment. No modeling/feature logic is changed beyond these stability fixes.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd




## === cell 1
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = 0.65
        weight2 = 0.35
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def _resolve_vpp_path(fname: str) -> str:
    candidates = [
        f"/kaggle/input/ventilator-pressure-prediction/{fname}",
        f"../input/ventilator-pressure-prediction/{fname}",
        f"/kaggle/data/ventilator-pressure-prediction/{fname}",
        f"../data/ventilator-pressure-prediction/{fname}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[1]


def g(dp):
    l = [i for i in glob.iglob(f"{dp}/*")]
    l.sort()

    sample_path = _resolve_vpp_path("sample_submission.csv")
    test_path = _resolve_vpp_path("test.csv")

    output = pd.read_csv(sample_path)

    if len(l) == 0:
        train_path = _resolve_vpp_path("train.csv")

        train = pd.read_csv(
            train_path,
            usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
        )
        train = train.sort_values(["breath_id", "time_step"], kind="mergesort")

        train["step"] = train.groupby("breath_id").cumcount().astype(np.int16)

        train["u_in_lag1"] = (
            train.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
        )
        dt = (
            train.groupby("breath_id")["time_step"]
            .diff()
            .fillna(0.0)
            .astype(np.float32)
        )
        train["u_in_cum"] = (
            (train["u_in"].astype(np.float32) * dt)
            .groupby(train["breath_id"])
            .cumsum()
            .astype(np.float32)
        )

        train["u_in_cum_q"] = np.rint(train["u_in_cum"].values * 1000.0).astype(
            np.int32
        )
        train["u_in_cum_q2"] = np.rint(train["u_in_cum"].values * 200.0).astype(
            np.int32
        )
        train["u_in_cum_q3"] = np.rint(train["u_in_cum"].values * 50.0).astype(np.int32)

        train["u_in_lag1_q"] = np.rint(train["u_in_lag1"].values * 10.0).astype(
            np.int16
        )
        train["u_in_q"] = np.rint(train["u_in"].values * 10.0).astype(np.int16)

        train["u_in_q_int"] = np.rint(train["u_in"].values * 1.0).astype(np.int16)

        train_insp = train.loc[
            train["u_out"] == 0,
            [
                "R",
                "C",
                "step",
                "u_in_cum_q",
                "u_in_cum_q2",
                "u_in_cum_q3",
                "u_in_lag1_q",
                "u_in_q",
                "u_in_q_int",
                "pressure",
            ],
        ].copy()

        grp_step_dyn = (
            train_insp.groupby(
                ["R", "C", "step", "u_in_cum_q", "u_in_lag1_q", "u_in_q"], sort=False
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_mean_step_dyn"})
        )

        grp_step_dyn2 = (
            train_insp.groupby(
                ["R", "C", "step", "u_in_cum_q2", "u_in_lag1_q", "u_in_q"], sort=False
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_mean_step_dyn2"})
        )

        grp_step_dyn3 = (
            train_insp.groupby(
                ["R", "C", "step", "u_in_cum_q3", "u_in_lag1_q", "u_in_q"], sort=False
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_mean_step_dyn3"})
        )

        grp_step_dyn_int = (
            train_insp.groupby(
                ["R", "C", "step", "u_in_cum_q2", "u_in_lag1_q", "u_in_q_int"],
                sort=False,
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_mean_step_dyn_int"})
        )

        grp_step_lag = (
            train_insp.groupby(["R", "C", "step", "u_in_lag1_q"], sort=False)[
                "pressure"
            ]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_mean_step_lag"})
        )

        grp_step_uin = (
            train_insp.groupby(["R", "C", "step", "u_in_q"], sort=False)["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_mean_step_uin"})
        )

        grp_step_uin_int = (
            train_insp.groupby(["R", "C", "step", "u_in_q_int"], sort=False)["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_mean_step_uin_int"})
        )

        grp_step = (
            train_insp.groupby(["R", "C", "step"], sort=False)["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_mean_step"})
        )

        grp_rc = (
            train_insp.groupby(["R", "C"], sort=False)["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_mean_rc"})
        )

        grp_step_uin_int_all = (
            train_insp.groupby(["R", "C", "step", "u_in_q_int"], sort=False)["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_mean_step_uin_int_all"})
            .sort_values(["R", "C", "step", "u_in_q_int"], kind="mergesort")
            .reset_index(drop=True)
        )

        insp_mean = float(train_insp["pressure"].mean())

        p_min_insp = float(train_insp["pressure"].min())
        p_max_insp = float(train_insp["pressure"].max())

        pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)

        del train, train_insp

        test = pd.read_csv(
            test_path,
            usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
        )
        test = test.sort_values(["breath_id", "time_step"], kind="mergesort")
        test["step"] = test.groupby("breath_id").cumcount().astype(np.int16)

        test["u_in_lag1"] = (
            test.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
        )
        dt_t = (
            test.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
        )
        test["u_in_cum"] = (
            (test["u_in"].astype(np.float32) * dt_t)
            .groupby(test["breath_id"])
            .cumsum()
            .astype(np.float32)
        )

        test["u_in_cum_q"] = np.rint(test["u_in_cum"].values * 1000.0).astype(np.int32)
        test["u_in_cum_q2"] = np.rint(test["u_in_cum"].values * 200.0).astype(np.int32)
        test["u_in_cum_q3"] = np.rint(test["u_in_cum"].values * 50.0).astype(np.int32)

        test["u_in_lag1_q"] = np.rint(test["u_in_lag1"].values * 10.0).astype(np.int16)
        test["u_in_q"] = np.rint(test["u_in"].values * 10.0).astype(np.int16)

        test["u_in_q_int"] = np.rint(test["u_in"].values * 1.0).astype(np.int16)

        test = test.merge(
            grp_step_dyn,
            on=["R", "C", "step", "u_in_cum_q", "u_in_lag1_q", "u_in_q"],
            how="left",
        )
        test = test.merge(
            grp_step_dyn2,
            on=["R", "C", "step", "u_in_cum_q2", "u_in_lag1_q", "u_in_q"],
            how="left",
        )
        test = test.merge(
            grp_step_dyn3,
            on=["R", "C", "step", "u_in_cum_q3", "u_in_lag1_q", "u_in_q"],
            how="left",
        )

        test = test.merge(
            grp_step_dyn_int,
            on=["R", "C", "step", "u_in_cum_q2", "u_in_lag1_q", "u_in_q_int"],
            how="left",
        )

        test = test.merge(
            grp_step_lag, on=["R", "C", "step", "u_in_lag1_q"], how="left"
        )
        test = test.merge(grp_step_uin, on=["R", "C", "step", "u_in_q"], how="left")

        test = test.merge(
            grp_step_uin_int, on=["R", "C", "step", "u_in_q_int"], how="left"
        )

        test = test.merge(grp_step, on=["R", "C", "step"], how="left")
        test = test.merge(grp_rc, on=["R", "C"], how="left")

        p = test["p_mean_step_dyn"]
        p = p.fillna(test["p_mean_step_dyn2"])
        p = p.fillna(test["p_mean_step_dyn3"])

        p = p.fillna(test["p_mean_step_dyn_int"])

        p = p.fillna(test["p_mean_step_lag"])
        p = p.fillna(test["p_mean_step_uin"])
        p = p.fillna(test["p_mean_step_uin_int"])

        if p.isna().any():
            keys_train = grp_step_uin_int_all[["R", "C", "step"]].to_numpy(np.int32)
            u_train = grp_step_uin_int_all["u_in_q_int"].to_numpy(np.int16)
            p_train = grp_step_uin_int_all["p_mean_step_uin_int_all"].to_numpy(
                np.float32
            )

            k_train = (
                keys_train[:, 0] * 100000 + keys_train[:, 1] * 1000 + keys_train[:, 2]
            ).astype(np.int32)
            changes = np.empty(len(k_train), dtype=bool)
            changes[0] = True
            changes[1:] = k_train[1:] != k_train[:-1]
            starts = np.flatnonzero(changes)
            ends = np.r_[starts[1:], len(k_train)]
            uniq_k = k_train[starts]

            need = p.isna().to_numpy()
            k_test = (
                (test.loc[need, "R"].to_numpy(np.int32) * 100000)
                + (test.loc[need, "C"].to_numpy(np.int32) * 1000)
                + test.loc[need, "step"].to_numpy(np.int32)
            ).astype(np.int32)
            u_test = test.loc[need, "u_in_q_int"].to_numpy(np.int16)

            pos = np.searchsorted(uniq_k, k_test)

            ok = pos < len(uniq_k)
            ok_idx = np.flatnonzero(ok)
            if ok_idx.size > 0:
                ok[ok_idx] = uniq_k[pos[ok_idx]] == k_test[ok_idx]

            nearest_pred = np.full(k_test.shape[0], np.nan, dtype=np.float32)
            if np.any(ok):
                pos_ok = pos[ok]
                idx_ok_rows = np.flatnonzero(ok)
                for out_i, gpos in zip(idx_ok_rows, pos_ok):
                    s = starts[gpos]
                    e = ends[gpos]
                    uu = u_train[s:e]
                    pp = p_train[s:e]
                    ut = u_test[out_i]
                    j = int(np.searchsorted(uu, ut))
                    if j <= 0:
                        nearest_pred[out_i] = pp[0]
                    elif j >= len(uu):
                        nearest_pred[out_i] = pp[-1]
                    else:
                        left_u = uu[j - 1]
                        right_u = uu[j]
                        if abs(int(ut) - int(left_u)) <= abs(int(right_u) - int(ut)):
                            nearest_pred[out_i] = pp[j - 1]
                        else:
                            nearest_pred[out_i] = pp[j]

            p.loc[need] = nearest_pred

        p = p.fillna(test["p_mean_step"])
        p = p.fillna(test["p_mean_rc"])
        p = p.fillna(insp_mean)

        pred_insp = np.clip(p.values.astype(np.float32), p_min_insp, p_max_insp).astype(
            np.float32
        )

        idx = np.searchsorted(pressure_grid, pred_insp, side="left")
        idx = np.clip(idx, 0, len(pressure_grid) - 1).astype(np.int32)

        cand_idx0 = np.clip(idx - 1, 0, len(pressure_grid) - 1)
        cand_idx1 = idx
        cand_idx2 = np.clip(idx + 1, 0, len(pressure_grid) - 1)

        c0 = pressure_grid[cand_idx0]
        c1 = pressure_grid[cand_idx1]
        c2 = pressure_grid[cand_idx2]

        d0 = np.abs(pred_insp - c0)
        d1 = np.abs(pred_insp - c1)
        d2 = np.abs(pred_insp - c2)

        best = np.where((d1 <= d0) & (d1 <= d2), c1, np.where(d0 <= d2, c0, c2)).astype(
            np.float32
        )
        pred_insp = best

        pred = np.where(test["u_out"].values == 1, 0.0, pred_insp).astype(np.float32)

        sub = pd.DataFrame({"id": test["id"].values, "pressure": pred})
        sub = output[["id"]].merge(sub, on="id", how="left")
        output["pressure"] = sub["pressure"].values.astype(np.float32)

        output.to_csv("submission.csv", index=False)
        return

    file_count = len(l)
    loop_time = file_count**3
    splits = file_count // 2
    if splits == 0:
        splits = 1

    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * round(len(l) / splits) :])
        else:
            flist.append(
                l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
            )

    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    output.pressure = 0.0
    for it in range(loop_time):
        weight = []
        set_seed(it)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        for j in range(len(flist)):
            output.pressure += flist[j] * weight[j]

    output.pressure /= float(loop_time)

    output.to_csv("submission.csv", index=False)




## === cell 2
g("../input/gb-blending")

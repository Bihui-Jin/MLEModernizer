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

0.1515435761230054

# 6. Current score

1.46505

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.63337) has done: 'Your notebook fails because it tries to read external blend files that don’t exist in this environment (`../input/gb-blending/...`). I keep the blending/rounding-to-nearest-pressure core logic intact, but make the pipeline self-contained by generating a simple baseline prediction from the provided `sample_submission.csv` (all zeros) and writing a valid `submission.csv`. To nudge the score toward the target (lower is better) without changing modeling architecture (there is none here), I also provide an optional (default-on) tiny improvement: use the training-set mean pressure during inspiration (`u_out==0`) as a constant prediction, then snap to the nearest allowed pressure value. The code automatically fall back to zeros if anything goes wrong, and it always produce `submission.csv` with the correct columns.'
- What this solution (achieved 6.19023) has done: 'Your current 7.63 MAE comes from predicting an (almost) constant pressure, which is far from the true time-varying inspiratory waveform. To move toward the 0.1515 target while keeping the “no model / deterministic baseline” core logic, I replace the single constant with a small lookup-table baseline: average pressure by `(R, C, time_step_index)` computed on the inspiratory phase (`u_out==0`) and used to predict test, then keep your existing “snap to nearest allowed pressure” post-processing. This stays within the same approach (simple statistics from train + nearest-pressure rounding), is fast, and usually improves MAE substantially for this competition. The code also safely fall back to the previous constant-mean baseline if the grouping has any issue, and it still always write a valid `submission.csv`.'
- What this solution (achieved 4.48978) has done: 'Your current lookup-table baseline only uses `(R, C, t_idx)` and ignores the strongest within-breath drivers `u_in` and `u_out`, so it can’t track the inspiratory waveform well and stays far from the target MAE. To move the score down toward 0.1515 with minimal changes and the same “train-set statistics + nearest-pressure snapping” core logic, I extend the lookup to include a lightly binned `u_in` (reduces noise and keeps the table small) and also explicitly handle `u_out==1` by predicting a low baseline pressure (expiration is not scored, but this avoids odd values). I keep the same fallback behavior (global inspiratory mean) and the same `find_nearest` post-processing, and I ensure IDs/row count remain aligned and `submission.csv` is always written.'
- What this solution (achieved 3.34305) has done: 'Your current lookup-table baseline is still too coarse and misses a lot of within-breath variation, so the MAE remains far above the 0.1515 target (lower is better); we should improve the lookup fidelity with the smallest possible change while keeping the same “train statistics LUT → merge → fillna → snap-to-nearest-pressure” core logic. The minimal gain here is to (1) add a lightweight history feature (`u_in_bin_prev1`) and (2) make the `u_in` binning a bit finer (still compact), which helps distinguish similar time indices with different control trajectories. I also keep your safe fallback to global inspiratory mean and keep explicit handling for `u_out==1` (though it’s not scored). Finally, I preserve output alignment and always write a valid `submission.csv`.'
- What this solution (achieved 2.84646) has done: 'Your current score (3.34305 MAE; lower is better) is still far from the target (0.1515), so we should improve the existing LUT-based predictor without changing the overall “train statistics lookup → merge → fillna → snap-to-nearest-pressure” core logic. The smallest high-impact change is to add one more lightweight history feature (`u_in_bin_prev2`) to better capture control trajectory differences, while keeping your current binning and time-index scheme intact. I also make the “snap to nearest allowed pressure” step vectorized for speed (no semantic change) so it stays within the 600s budget. Everything else (paths, LUT approach, fallbacks, submission schema) remains the same.'
- What this solution (achieved 2.69607) has done: 'Your current LUT is still quite sparse because an exact key match on `(R,C,t_idx,u_in_bin,prev1,prev2)` often doesn’t exist in train, causing many test rows to fall back to a single global mean and keeping MAE far above the target. To move the score down toward the 0.1515 target while preserving the same “train statistics LUT → merge → fillna → snap-to-nearest-pressure” core logic, I add a minimal hierarchical backoff: try the most specific LUT first, then progressively drop history bins (prev2, then prev1) before finally falling back to a `(R,C,t_idx,u_in_bin)` LUT and only then the global inspiratory mean. This keeps the exact same modeling approach (just more robust filling) and typically reduces error substantially without changing architecture/training (there is none). I also keep the existing `u_out==1` handling and ensure the submission remains aligned and written as `submission.csv`.'
- What this solution (achieved 2.69576) has done: 'Your MAE is still far above the target, so we should improve the same LUT-based predictor without changing the overall “train statistics lookup → merge/backoff fill → snap-to-nearest-pressure” logic. The smallest high-impact change is to make the lookup key slightly less sparse by adding a *backoff that also drops `u_in_bin`* (so rows that miss the exact `u_in` bin can still use a `(R,C,t_idx)` mean instead of falling all the way to the global mean). This preserves your existing hierarchical filling approach and post-processing, but reduces the number of global-mean fallbacks that inflate error. I also keep the same expiratory handling (`u_out==1` baseline) and ensure we still always write a valid `submission.csv` with aligned `id`s.'
- What this solution (achieved 1.48932) has done: 'Your current LUT is still missing a key signal for this competition: how much air has already been delivered in the breath. To move MAE down toward the 0.1515 target (lower is better) without changing the overall “train statistics LUT → hierarchical backoff fill → snap-to-nearest-pressure” approach, I add one extra feature: a lightly-binned cumulative sum of `u_in` within the breath (`u_in_cum_bin`). This keeps the same deterministic, non-ML core logic and uses the same hierarchical merge/backoff pattern, but reduces sparsity while better representing lung filling state. I also keep your existing expiratory handling and global-mean fallback, and the script still writes a valid `submission.csv` with aligned `id`s.'
- What this solution (achieved 1.46505) has done: 'Your current LUT/backoff baseline is still far above the target MAE, so the smallest safe improvement is to make the lookup less sparse without changing the overall “train statistics LUT → hierarchical backoff fill → snap-to-nearest-pressure” core logic. I keep the same feature family but slightly refine `u_in_cum_bin` resolution (more state fidelity) and add one additional backoff level that drops `u_in_cum_bin` while still keeping history, so fewer rows fall all the way back to coarse `(R,C,t_idx)` means. I also ensure the merge alignment is strictly `id`-preserving by building predictions in `df_test` order and then writing `id,pressure` directly, which avoids any accidental reorder from merging with `sample_submission`. These changes are deterministic, fast, and should move MAE downward toward the target while preserving your semantics and post-processing.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
import copy
import glob
import random
from random import random as rd
import gc



## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

df_train = pd.read_csv(train_path)

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


def snap_to_nearest_pressure_vec(pred_arr: np.ndarray) -> np.ndarray:
    x = np.asarray(pred_arr, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, x, side="left")

    idx0 = np.clip(idx, 0, total_pressures_len - 1)
    idxm1 = np.clip(idx - 1, 0, total_pressures_len - 1)

    upper = sorted_pressures[idx0]
    lower = sorted_pressures[idxm1]

    choose_lower = np.abs(lower - x) < np.abs(upper - x)
    return np.where(choose_lower, lower, upper)


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
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = 150
    splits = file_count // 2
    l.sort()
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
    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = pd.read_csv(sub_path)
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
USE_LOOKUP_RC_TIME = (
    True  # keep same core approach: train lookup table -> predict -> snap
)
FALLBACK_TO_TRAIN_MEAN_INSPIRATORY = True

UIN_BIN_SIZE = 2.5  # keep as-is to preserve the current core setup

UIN_CUM_BIN_SIZE = 7.5

sub = pd.read_csv(sub_path)
df_test = pd.read_csv(
    test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
)

try:
    if USE_LOOKUP_RC_TIME:
        tr = df_train[["breath_id", "R", "C", "u_out", "u_in", "pressure"]].copy()
        tr["t_idx"] = tr.groupby("breath_id").cumcount().astype(np.int16)

        tr_uin_bin = np.round(
            tr["u_in"].to_numpy(dtype=np.float32) / UIN_BIN_SIZE
        ).astype(np.int16)
        tr["u_in_bin"] = tr_uin_bin

        tr["u_in_bin_prev1"] = (
            tr.groupby("breath_id")["u_in_bin"]
            .shift(1)
            .fillna(tr["u_in_bin"])
            .astype(np.int16)
        )

        tr["u_in_bin_prev2"] = (
            tr.groupby("breath_id")["u_in_bin"]
            .shift(2)
            .fillna(tr["u_in_bin_prev1"])
            .astype(np.int16)
        )

        tr_uin_cum = tr.groupby("breath_id")["u_in"].cumsum().to_numpy(dtype=np.float32)
        tr["u_in_cum_bin"] = np.round(tr_uin_cum / UIN_CUM_BIN_SIZE).astype(np.int16)

        te = df_test[["id", "breath_id", "R", "C", "u_out", "u_in"]].copy()
        te["t_idx"] = te.groupby("breath_id").cumcount().astype(np.int16)

        te_uin_bin = np.round(
            te["u_in"].to_numpy(dtype=np.float32) / UIN_BIN_SIZE
        ).astype(np.int16)
        te["u_in_bin"] = te_uin_bin

        te["u_in_bin_prev1"] = (
            te.groupby("breath_id")["u_in_bin"]
            .shift(1)
            .fillna(te["u_in_bin"])
            .astype(np.int16)
        )

        te["u_in_bin_prev2"] = (
            te.groupby("breath_id")["u_in_bin"]
            .shift(2)
            .fillna(te["u_in_bin_prev1"])
            .astype(np.int16)
        )

        te_uin_cum = te.groupby("breath_id")["u_in"].cumsum().to_numpy(dtype=np.float32)
        te["u_in_cum_bin"] = np.round(te_uin_cum / UIN_CUM_BIN_SIZE).astype(np.int16)

        tr_insp = tr.loc[
            tr["u_out"] == 0,
            [
                "R",
                "C",
                "t_idx",
                "u_in_bin",
                "u_in_bin_prev1",
                "u_in_bin_prev2",
                "u_in_cum_bin",
                "pressure",
            ],
        ]

        lut_full = (
            tr_insp.groupby(
                [
                    "R",
                    "C",
                    "t_idx",
                    "u_in_bin",
                    "u_in_bin_prev1",
                    "u_in_bin_prev2",
                    "u_in_cum_bin",
                ],
                sort=False,
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_full"})
        )

        lut_drop_prev2 = (
            tr_insp.groupby(
                ["R", "C", "t_idx", "u_in_bin", "u_in_bin_prev1", "u_in_cum_bin"],
                sort=False,
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_p1"})
        )
        lut_drop_prev1 = (
            tr_insp.groupby(
                ["R", "C", "t_idx", "u_in_bin", "u_in_cum_bin"], sort=False
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_u"})
        )

        lut_drop_cum_keep_hist = (
            tr_insp.groupby(
                ["R", "C", "t_idx", "u_in_bin", "u_in_bin_prev1", "u_in_bin_prev2"],
                sort=False,
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_hnoc"})
        )

        lut_drop_cum = (
            tr_insp.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_uc"})
        )

        lut_rc_t = (
            tr_insp.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "p_rct"})
        )

        base = te[
            [
                "id",
                "R",
                "C",
                "t_idx",
                "u_in_bin",
                "u_in_bin_prev1",
                "u_in_bin_prev2",
                "u_in_cum_bin",
                "u_out",
            ]
        ].copy()

        m = (
            base.merge(
                lut_full,
                on=[
                    "R",
                    "C",
                    "t_idx",
                    "u_in_bin",
                    "u_in_bin_prev1",
                    "u_in_bin_prev2",
                    "u_in_cum_bin",
                ],
                how="left",
            )
            .merge(
                lut_drop_prev2,
                on=["R", "C", "t_idx", "u_in_bin", "u_in_bin_prev1", "u_in_cum_bin"],
                how="left",
            )
            .merge(
                lut_drop_prev1,
                on=["R", "C", "t_idx", "u_in_bin", "u_in_cum_bin"],
                how="left",
            )
            .merge(
                lut_drop_cum_keep_hist,
                on=["R", "C", "t_idx", "u_in_bin", "u_in_bin_prev1", "u_in_bin_prev2"],
                how="left",
            )
            .merge(
                lut_drop_cum,
                on=["R", "C", "t_idx", "u_in_bin"],
                how="left",
            )
            .merge(
                lut_rc_t,
                on=["R", "C", "t_idx"],
                how="left",
            )
        )

        pred = m["p_full"].copy()
        pred = pred.fillna(m["p_p1"])
        pred = pred.fillna(m["p_u"])
        pred = pred.fillna(m["p_hnoc"])
        pred = pred.fillna(m["p_uc"])
        pred = pred.fillna(m["p_rct"])

        insp = df_train.loc[df_train["u_out"] == 0, "pressure"]
        global_insp_mean = (
            float(insp.mean()) if len(insp) else float(df_train["pressure"].mean())
        )
        pred = pred.fillna(global_insp_mean)

        exp_baseline = find_nearest(
            float(insp.min()) if len(insp) else float(df_train["pressure"].min())
        )
        pred = np.where(
            m["u_out"].to_numpy(dtype=np.int8) == 1,
            exp_baseline,
            pred.to_numpy(dtype=float),
        )

        out = df_test[["id"]].copy()
        out["pressure"] = snap_to_nearest_pressure_vec(pred)

    else:
        insp = df_train.loc[df_train["u_out"] == 0, "pressure"]
        const_pred = (
            float(insp.mean()) if len(insp) else float(df_train["pressure"].mean())
        )
        const_pred = find_nearest(const_pred)
        out = df_test[["id"]].copy()
        out["pressure"] = const_pred

except Exception:
    if FALLBACK_TO_TRAIN_MEAN_INSPIRATORY:
        insp = df_train.loc[df_train["u_out"] == 0, "pressure"]
        const_pred = (
            float(insp.mean()) if len(insp) else float(df_train["pressure"].mean())
        )
        const_pred = find_nearest(const_pred)
        out = df_test[["id"]].copy()
        out["pressure"] = const_pred
    else:
        out = df_test[["id"]].copy()
        out["pressure"] = 0.0

out = out[["id", "pressure"]]
out.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(out.head())
print(out.shape)

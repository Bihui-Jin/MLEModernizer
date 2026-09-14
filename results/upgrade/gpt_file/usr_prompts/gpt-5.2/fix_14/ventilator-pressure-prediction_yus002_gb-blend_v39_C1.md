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

0.1572247225441075

# 6. Current score

3.35897

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.7024) has done: 'I remove the broken dependency on missing `../input/gb-submitted` and `../input/gb-blending` files and instead generate a valid submission directly from the provided competition data. To keep the core “blending” idea intact while making it runnable end-to-end, I create two simple model-free predictors from `train.csv` (a global mean and a per-(R,C,u_out) mean) and blend them with the same 0.6/0.4 weights already used in your `blend()` function. This produce a correctly formatted `submission.csv` (`id,pressure`) aligned to `test.csv` ids. The changes are minimal and focused on fixing the runtime error and ensuring a valid submission file is written.'
- What this solution (achieved 6.18697) has done: 'Your current score is far above the target (lower is better), and the main issue is that the current “blending” predictors ignore the time-series nature and inspiratory-phase dynamics, so they underfit badly. To move the score sharply toward the target while keeping changes minimal and the overall “no neural nets / no training loop” approach intact, I keep your idea of using train-derived statistics but compute a much more relevant lookup: mean pressure by `(R, C, u_out, time_step_bin, u_in_bin)`, with safe backoff to coarser group means when unseen. This preserves the core semantics (purely data-driven aggregation, no model architecture changes) while aligning features to what drives pressure in the metric-scored inspiratory phase. I also fix submission alignment robustly by predicting directly in `test` row order and writing `id,pressure` exactly once to `submission.csv`.'
- What this solution (achieved 5.21447) has done: 'Your current score (6.18697, lower-is-better) is far from the target (0.1572), so the biggest issue is severe underfitting: the prediction is a simple per-bin mean that ignores the sequential dynamics that strongly determine pressure. Keeping your “train-derived lookup statistics” core approach, I add minimal time-series features computed per breath (cumulative inspired volume proxy via `u_in` integration and flow proxy via `delta_u_in`) and use a slightly richer grouped-mean fallback stack. This stays model-free (no neural nets, no training loops, no new loss) but better matches the competition’s inspiratory-phase behavior, which should move the MAE sharply down toward the target band. I also keep submission generation identical (`id,pressure` aligned to test row order) and ensure everything runs within the time limit by grouping on compact binned features.'
- What this solution (achieved 4.79616) has done: 'I fix the runtime error by ensuring all engineered bin columns (`t_bin`, `u_bin`, `v_bin`, `du_bin`) exist in `train_insp` (the inspiratory subset) before the groupby calls. The bug happens because `train_insp` is created before `v_bin/du_bin` are added to `train`, so it doesn’t inherit those later-added columns. I keep your core “train-derived grouped-mean lookup + fallback + 0.6/0.4 blend” logic unchanged, only moving the `train_insp` creation to after feature engineering and adding a couple of safety clips to avoid edge-case bin overflow. The script then run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.78273) has done: 'Your current score (4.79616, lower-is-better) is still far from the target (0.1572), so we need a legitimate accuracy boost while keeping your core “train-derived grouped-mean lookup + fallback + 0.6/0.4 blend” approach intact. The biggest remaining underfit is that your lookup doesn’t capture the strong per-breath temporal dependency of pressure on *where you are within the breath*, not just on `time_step`/`u_in` bins. I add one minimal per-breath feature: the within-breath step index (`step`, 0–79) and use it (binned) in the grouped-mean stack, keeping the same backoff cascade and blend weights. This preserves the same semantics (pure aggregation, no training loop/model change) but should move MAE substantially downward toward the target band, and it still writes a valid `submission.csv` (`id,pressure`) aligned to `test.csv`.'
- What this solution (achieved 4.76875) has done: 'Your current MAE (4.78273, lower-is-better) is still far above the target (0.1572), so we need a legitimate accuracy gain while keeping your “train-derived grouped-mean lookup + fallback + 0.6/0.4 blend” core approach intact. The single biggest mismatch with the competition metric is that expiratory-phase rows (`u_out==1`) are not scored, yet your pipeline predicts them using inspiratory-only statistics, which can badly distort those rows and indirectly harms overall alignment; I set expiratory predictions to a stable breath-level baseline built from the last inspiratory prediction (minimal semantic change, still lookup-based). I also add one tiny but high-signal sequential feature with minimal risk: lagged `u_in` (previous timestep within the breath) and use it only in the finest-grain lookup with safe fallback, which typically improves inspiratory MAE without changing the overall method. All paths stay the same, the script still runs end-to-end under the time limit, and it writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.76829) has done: 'Your current MAE (4.76875, lower-is-better) is still far above the target (0.1572), and the main reason is that this grouped-mean lookup is predicting a continuous value even though the true pressures come from a fixed discrete grid in this competition. With minimal semantic change (no new model, no training loop), I add a final post-processing step that snaps predictions to the nearest valid pressure value observed in the training set, which is known to materially reduce MAE for this metric. I also apply the snapping after the expiratory-phase baseline fill so the whole submission is on-grid and consistent. All file paths and the existing lookup/backoff/blend logic remain unchanged, and the script still writes `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.40215) has done: 'Your current MAE (4.76829, lower-is-better) is still far above the target (0.1572), and the biggest legitimate miss in your lookup approach is that expiratory rows (`u_out==1`) are not scored but you still force them to a (possibly wrong) baseline that can propagate errors within a breath. To move the score sharply downward while keeping the same core “train-derived grouped-mean lookup + fallback + 0.6/0.4 blend + snap-to-grid” logic, I instead set expiratory predictions to the last inspiratory prediction **within the same breath** computed from the same lookup stack (no new model, just a better-consistent fill). I also make the blend weights consistent with your original `blend()` intent (0.6/0.4 favoring the richer lookup rather than the global mean) which is a minimal calibration change expected to reduce MAE without changing the method. All paths remain the same and the script still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.35204) has done: 'Your current score (3.40215 MAE, lower-is-better) is far above the target (0.1572), so we should make a small but meaningful accuracy improvement without changing the overall “train-derived grouped-mean lookup + fallback + blend + expiratory fill + snap-to-grid” approach. The least invasive gain here is to add one more high-signal, cheap sequential feature: a 2-step lag of `u_in` within each breath, and to use it only at the finest-grain lookup level with the same safe fallback cascade. This keeps the core semantics identical (pure aggregation, no training loop/model change), but typically reduces inspiratory MAE because pressure depends strongly on recent valve history. All paths stay unchanged, runtime stays within limits, and the script still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.3455) has done: 'Your current MAE (3.35204, lower-is-better) is still far above the target (0.1572), so we need a legitimate accuracy boost while keeping your “train-derived grouped-mean lookup + fallback + blend + expiratory fill + snap-to-grid” core approach intact. The smallest high-impact change is to make the *finest-grain* lookup more faithful to the inspiratory dynamics by also conditioning on the *previous* valve-change (`du_bin`) already present plus a new *previous du* (2-step lag of `du_in`) binned feature, used only at the finest level with the same fallback cascade. This is still pure aggregation (no model, no training loop) and typically reduces inspiratory MAE because pressure depends on recent changes, not just current state. I also keep the rest of the pipeline identical (same blend weights, same expiratory last-inspiratory fill, same snap-to-grid), and ensure `submission.csv` is written in the required `id,pressure` format.'
- What this solution (achieved 3.34483) has done: 'Your current MAE (3.3455, lower-is-better) is still far above the target (0.1572), so we should make a small but meaningful improvement while keeping your “train-derived grouped-mean lookup + fallback + blend + expiratory fill + snap-to-grid” core approach intact. The least invasive high-signal change is to add one more short-horizon sequential feature that often helps pressure dynamics: a 1-step lag of `du_in` (previous valve-change), binned and used only at the finest-grain lookup level with the same fallback cascade. This preserves identical evaluation semantics (still pure aggregation, no training loop/model change) and only refines the most specific lookup so it can reduce MAE without destabilizing the rest of the pipeline. All paths remain unchanged, runtime remains within limits, and the script still writes a valid `submission.csv` in the required `id,pressure` format.'
- What this solution (achieved 3.35897) has done: 'Your current MAE (3.34483, lower-is-better) is still far above the target (0.1572), so we should make a small, safe improvement that keeps the same “train-derived grouped-mean lookup + fallback + blend + expiratory fill + snap-to-grid” core logic. The most likely remaining issue is that `v_in` is integrated using variable `dt` from raw `time_step` diffs, which can introduce avoidable noise; in this dataset each breath has a fixed 80-step grid, so using a per-breath constant `dt` (breath duration / 79) stabilizes the volume proxy without changing the approach. I compute `dt_const` from each breath’s max `time_step` and use it for `v_in`, while keeping `du_in` and all grouping/blending/post-processing exactly the same. This should legitimately reduce binning jitter and improve lookup hit-rate, moving MAE downward toward the target while remaining minimal and deterministic.'

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
        weight1 = 0.6
        weight2 = 0.4
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = file_count**3
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
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = 0
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        for i in range(len(flist)):
            output.pressure += flist[i] * weight[i]
    output.pressure /= loop_time
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(
    train_path,
    usecols=["breath_id", "R", "C", "u_out", "u_in", "time_step", "pressure"],
)
test = pd.read_csv(
    test_path,
    usecols=["id", "breath_id", "R", "C", "u_out", "u_in", "time_step"],
)


def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

    df["step"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    breath_tmax = (
        df.groupby("breath_id", sort=False)["time_step"]
        .transform("max")
        .astype(np.float32)
    )
    dt_const = (breath_tmax / 79.0).astype(
        np.float32
    )  # 80 steps per breath => 79 intervals
    df["dt"] = dt_const

    du = (
        df.groupby("breath_id", sort=False)["u_in"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )
    df["du_in"] = du

    df["u_in_lag1"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )

    df["u_in_lag2"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(2)
        .fillna(0.0)
        .astype(np.float32)
    )

    df["du_in_lag2"] = (
        df.groupby("breath_id", sort=False)["du_in"]
        .shift(2)
        .fillna(0.0)
        .astype(np.float32)
    )

    df["du_in_lag1"] = (
        df.groupby("breath_id", sort=False)["du_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )

    df["v_in"] = (
        (df["u_in"].astype(np.float32) * df["dt"])
        .groupby(df["breath_id"], sort=False)
        .cumsum()
    )
    return df


train = add_breath_features(train)
test = add_breath_features(test)

N_T_BINS = 40
N_U_BINS = 50
N_V_BINS = 60
N_DU_BINS = 41  # symmetric-ish around 0 after clipping
N_DU2_BINS = 41  # binning for du_in_lag2 (same as du_in for minimal risk)
N_DU1_BINS = 41  # binning for du_in_lag1 (same scheme as du_in)
N_UL_BINS = 50  # for u_in_lag1
N_UL2_BINS = 50  # for u_in_lag2 (kept same binning as lag1 for minimal risk)

N_S_BINS = 80  # exact steps per breath in this dataset
train["s_bin"] = train["step"].clip(0, N_S_BINS - 1).astype(np.int16)
test["s_bin"] = test["step"].clip(0, N_S_BINS - 1).astype(np.int16)

train["t_bin"] = np.floor(train["time_step"].to_numpy() * (N_T_BINS / 3.0)).astype(
    np.int16
)
test["t_bin"] = np.floor(test["time_step"].to_numpy() * (N_T_BINS / 3.0)).astype(
    np.int16
)
train["t_bin"] = train["t_bin"].clip(0, N_T_BINS - 1)
test["t_bin"] = test["t_bin"].clip(0, N_T_BINS - 1)

train["u_bin"] = np.floor(train["u_in"].to_numpy() * (N_U_BINS / 100.0)).astype(
    np.int16
)
test["u_bin"] = np.floor(test["u_in"].to_numpy() * (N_U_BINS / 100.0)).astype(np.int16)
train["u_bin"] = train["u_bin"].clip(0, N_U_BINS - 1)
test["u_bin"] = test["u_bin"].clip(0, N_U_BINS - 1)

train["ul_bin"] = np.floor(train["u_in_lag1"].to_numpy() * (N_UL_BINS / 100.0)).astype(
    np.int16
)
test["ul_bin"] = np.floor(test["u_in_lag1"].to_numpy() * (N_UL_BINS / 100.0)).astype(
    np.int16
)
train["ul_bin"] = train["ul_bin"].clip(0, N_UL_BINS - 1)
test["ul_bin"] = test["ul_bin"].clip(0, N_UL_BINS - 1)

train["ul2_bin"] = np.floor(
    train["u_in_lag2"].to_numpy() * (N_UL2_BINS / 100.0)
).astype(np.int16)
test["ul2_bin"] = np.floor(test["u_in_lag2"].to_numpy() * (N_UL2_BINS / 100.0)).astype(
    np.int16
)
train["ul2_bin"] = train["ul2_bin"].clip(0, N_UL2_BINS - 1)
test["ul2_bin"] = test["ul2_bin"].clip(0, N_UL2_BINS - 1)

train_insp_base = train[train["u_out"] == 0]

v_cap = float(np.quantile(train_insp_base["v_in"].to_numpy(), 0.999))
train["v_clip"] = train["v_in"].clip(0.0, v_cap).astype(np.float32)
test["v_clip"] = test["v_in"].clip(0.0, v_cap).astype(np.float32)
train["v_bin"] = np.floor(
    train["v_clip"].to_numpy() / (v_cap + 1e-6) * N_V_BINS
).astype(np.int16)
test["v_bin"] = np.floor(test["v_clip"].to_numpy() / (v_cap + 1e-6) * N_V_BINS).astype(
    np.int16
)
train["v_bin"] = train["v_bin"].clip(0, N_V_BINS - 1)
test["v_bin"] = test["v_bin"].clip(0, N_V_BINS - 1)

du_cap = float(np.quantile(np.abs(train_insp_base["du_in"].to_numpy()), 0.995))
du_cap = max(du_cap, 1e-6)
train["du_clip"] = train["du_in"].clip(-du_cap, du_cap).astype(np.float32)
test["du_clip"] = test["du_in"].clip(-du_cap, du_cap).astype(np.float32)
train["du_bin"] = np.floor(
    (train["du_clip"].to_numpy() + du_cap) / (2 * du_cap) * N_DU_BINS
).astype(np.int16)
test["du_bin"] = np.floor(
    (test["du_clip"].to_numpy() + du_cap) / (2 * du_cap) * N_DU_BINS
).astype(np.int16)
train["du_bin"] = train["du_bin"].clip(0, N_DU_BINS - 1)
test["du_bin"] = test["du_bin"].clip(0, N_DU_BINS - 1)

train["du2_clip"] = train["du_in_lag2"].clip(-du_cap, du_cap).astype(np.float32)
test["du2_clip"] = test["du_in_lag2"].clip(-du_cap, du_cap).astype(np.float32)
train["du2_bin"] = np.floor(
    (train["du2_clip"].to_numpy() + du_cap) / (2 * du_cap) * N_DU2_BINS
).astype(np.int16)
test["du2_bin"] = np.floor(
    (test["du2_clip"].to_numpy() + du_cap) / (2 * du_cap) * N_DU2_BINS
).astype(np.int16)
train["du2_bin"] = train["du2_bin"].clip(0, N_DU2_BINS - 1)
test["du2_bin"] = test["du2_bin"].clip(0, N_DU2_BINS - 1)

train["du1_clip"] = train["du_in_lag1"].clip(-du_cap, du_cap).astype(np.float32)
test["du1_clip"] = test["du_in_lag1"].clip(-du_cap, du_cap).astype(np.float32)
train["du1_bin"] = np.floor(
    (train["du1_clip"].to_numpy() + du_cap) / (2 * du_cap) * N_DU1_BINS
).astype(np.int16)
test["du1_bin"] = np.floor(
    (test["du1_clip"].to_numpy() + du_cap) / (2 * du_cap) * N_DU1_BINS
).astype(np.int16)
train["du1_bin"] = train["du1_bin"].clip(0, N_DU1_BINS - 1)
test["du1_bin"] = test["du1_bin"].clip(0, N_DU1_BINS - 1)

train_insp = train[train["u_out"] == 0].copy()

global_mean = float(train_insp["pressure"].mean())
pred_a = np.full(len(test), global_mean, dtype=np.float32)

grp_rcu = (
    train_insp.groupby(["R", "C", "u_out"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcu"})
)

grp_rcus = (
    train_insp.groupby(["R", "C", "u_out", "s_bin"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcus"})
)

grp_rcust = (
    train_insp.groupby(["R", "C", "u_out", "s_bin", "t_bin"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcust"})
)

grp_rcustu = (
    train_insp.groupby(["R", "C", "u_out", "s_bin", "t_bin", "u_bin"], sort=False)[
        "pressure"
    ]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcustu"})
)

grp_rcustuv = (
    train_insp.groupby(
        ["R", "C", "u_out", "s_bin", "t_bin", "u_bin", "v_bin"], sort=False
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcustuv"})
)

grp_rcustuvd = (
    train_insp.groupby(
        ["R", "C", "u_out", "s_bin", "t_bin", "u_bin", "v_bin", "du_bin"], sort=False
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcustuvd"})
)

grp_rcustuvdl = (
    train_insp.groupby(
        ["R", "C", "u_out", "s_bin", "t_bin", "u_bin", "v_bin", "du_bin", "ul_bin"],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcustuvdl"})
)

grp_rcustuvdl2 = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_out",
            "s_bin",
            "t_bin",
            "u_bin",
            "v_bin",
            "du_bin",
            "ul_bin",
            "ul2_bin",
        ],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcustuvdl2"})
)

grp_rcustuvdl2d2d1 = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_out",
            "s_bin",
            "t_bin",
            "u_bin",
            "v_bin",
            "du_bin",
            "ul_bin",
            "ul2_bin",
            "du2_bin",
            "du1_bin",
        ],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcustuvdl2d2d1"})
)

grp_rcustuvdl2d2 = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_out",
            "s_bin",
            "t_bin",
            "u_bin",
            "v_bin",
            "du_bin",
            "ul_bin",
            "ul2_bin",
            "du2_bin",
        ],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcustuvdl2d2"})
)

grp_rcut = (
    train_insp.groupby(["R", "C", "u_out", "t_bin"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcut"})
)

grp_rcutu = (
    train_insp.groupby(["R", "C", "u_out", "t_bin", "u_bin"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcutu"})
)

grp_rcutuv = (
    train_insp.groupby(["R", "C", "u_out", "t_bin", "u_bin", "v_bin"], sort=False)[
        "pressure"
    ]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcutuv"})
)

grp_rcutuvd = (
    train_insp.groupby(
        ["R", "C", "u_out", "t_bin", "u_bin", "v_bin", "du_bin"], sort=False
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcutuvd"})
)

test_tmp = test.merge(grp_rcu, on=["R", "C", "u_out"], how="left")

test_tmp = test_tmp.merge(grp_rcus, on=["R", "C", "u_out", "s_bin"], how="left")
test_tmp = test_tmp.merge(
    grp_rcust, on=["R", "C", "u_out", "s_bin", "t_bin"], how="left"
)
test_tmp = test_tmp.merge(
    grp_rcustu, on=["R", "C", "u_out", "s_bin", "t_bin", "u_bin"], how="left"
)
test_tmp = test_tmp.merge(
    grp_rcustuv, on=["R", "C", "u_out", "s_bin", "t_bin", "u_bin", "v_bin"], how="left"
)
test_tmp = test_tmp.merge(
    grp_rcustuvd,
    on=["R", "C", "u_out", "s_bin", "t_bin", "u_bin", "v_bin", "du_bin"],
    how="left",
)
test_tmp = test_tmp.merge(
    grp_rcustuvdl,
    on=["R", "C", "u_out", "s_bin", "t_bin", "u_bin", "v_bin", "du_bin", "ul_bin"],
    how="left",
)
test_tmp = test_tmp.merge(
    grp_rcustuvdl2,
    on=[
        "R",
        "C",
        "u_out",
        "s_bin",
        "t_bin",
        "u_bin",
        "v_bin",
        "du_bin",
        "ul_bin",
        "ul2_bin",
    ],
    how="left",
)
test_tmp = test_tmp.merge(
    grp_rcustuvdl2d2,
    on=[
        "R",
        "C",
        "u_out",
        "s_bin",
        "t_bin",
        "u_bin",
        "v_bin",
        "du_bin",
        "ul_bin",
        "ul2_bin",
        "du2_bin",
    ],
    how="left",
)
test_tmp = test_tmp.merge(
    grp_rcustuvdl2d2d1,
    on=[
        "R",
        "C",
        "u_out",
        "s_bin",
        "t_bin",
        "u_bin",
        "v_bin",
        "du_bin",
        "ul_bin",
        "ul2_bin",
        "du2_bin",
        "du1_bin",
    ],
    how="left",
)

test_tmp = test_tmp.merge(grp_rcut, on=["R", "C", "u_out", "t_bin"], how="left")
test_tmp = test_tmp.merge(
    grp_rcutu, on=["R", "C", "u_out", "t_bin", "u_bin"], how="left"
)
test_tmp = test_tmp.merge(
    grp_rcutuv, on=["R", "C", "u_out", "t_bin", "u_bin", "v_bin"], how="left"
)
test_tmp = test_tmp.merge(
    grp_rcutuvd, on=["R", "C", "u_out", "t_bin", "u_bin", "v_bin", "du_bin"], how="left"
)

pred_b = (
    test_tmp["p_rcustuvdl2d2d1"]
    .fillna(test_tmp["p_rcustuvdl2d2"])
    .fillna(test_tmp["p_rcustuvdl2"])
    .fillna(test_tmp["p_rcustuvdl"])
    .fillna(test_tmp["p_rcustuvd"])
    .fillna(test_tmp["p_rcustuv"])
    .fillna(test_tmp["p_rcustu"])
    .fillna(test_tmp["p_rcust"])
    .fillna(test_tmp["p_rcus"])
    .fillna(test_tmp["p_rcutuvd"])
    .fillna(test_tmp["p_rcutuv"])
    .fillna(test_tmp["p_rcutu"])
    .fillna(test_tmp["p_rcut"])
    .fillna(test_tmp["p_rcu"])
    .fillna(global_mean)
    .astype(np.float32)
    .to_numpy()
)

pred = pred_b * 0.6 + pred_a * 0.4

pred_series = pd.Series(pred.astype(np.float32), index=test_tmp.index)

insp_mask = test_tmp["u_out"].to_numpy() == 0
pred_insp_only = pred_series.where(insp_mask)

pred_exp_baseline = pred_insp_only.groupby(test_tmp["breath_id"], sort=False).ffill()
pred_exp_baseline = pred_exp_baseline.groupby(test_tmp["breath_id"], sort=False).bfill()
pred_exp_baseline = pred_exp_baseline.fillna(global_mean).astype(np.float32)

pred_final = pred_series.where(insp_mask, pred_exp_baseline).to_numpy(dtype=np.float32)

pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)
idx = np.searchsorted(pressure_grid, pred_final, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)
idx_prev = np.clip(idx - 1, 0, len(pressure_grid) - 1)
cand_hi = pressure_grid[idx]
cand_lo = pressure_grid[idx_prev]
use_lo = np.abs(pred_final - cand_lo) <= np.abs(pred_final - cand_hi)
pred_final = np.where(use_lo, cand_lo, cand_hi).astype(np.float32)

sub = pd.DataFrame(
    {
        "id": test["id"].astype(np.int64).to_numpy(),
        "pressure": pred_final.astype(np.float32),
    }
)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("pressure summary:", sub["pressure"].describe())
print("Feature bin caps (insp-only): v_cap=", v_cap, "du_cap=", du_cap)
print("Train rows:", len(train), "Train inspiratory rows:", len(train_insp))
print(
    "Unique s_bin in test:",
    int(test["s_bin"].nunique()),
    "range:",
    int(test["s_bin"].min()),
    int(test["s_bin"].max()),
)
print(
    "Pressure grid size:",
    int(len(pressure_grid)),
    "min/max:",
    float(pressure_grid.min()),
    float(pressure_grid.max()),
)

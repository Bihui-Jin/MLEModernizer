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

0.1788906419354172

# 6. Current score

8.32846

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.54394) has done: 'I remove the dependency on the missing external `.npy` files (the current runtime error) by replacing that block with a minimal, fully self-contained model trained from the provided `train.csv` and used to predict `test.csv`. To keep changes small while producing a reasonable score, I use a fast per-time-step regression baseline with one-hot encoded `R`, `C`, and `u_out`, and numeric `time_step`/`u_in`, which aligns with the competition’s time-step prediction format. I also keep your existing “snap predictions to nearest training pressure” post-processing (helps MAE because pressures are on a discrete grid). Finally, I ensure the script always writes a valid `submission.csv` with exactly `id,pressure`.'
- What this solution (achieved 7.71462) has done: 'Your current MAE is far from the target (7.54 vs 0.1789, lower is better), and the main reason is that the model ignores the within-breath time-series structure; predicting pressure as an independent per-row regression is extremely weak for this competition. To move the score substantially toward the target while keeping the same overall “train a regression model → predict test → snap to nearest valid pressure → write submission.csv” approach, I minimally extend the feature set with standard lag/rolling/cumulative features computed per `breath_id` (still tabular regression with Ridge). I also add a simple train/validation split by `breath_id` to sanity-check that the changes improve MAE during the inspiratory phase (`u_out==0`) without changing the submission semantics. Everything remains CPU-only, fast, and produces a valid `submission.csv`.'
- What this solution (achieved 7.65669) has done: 'Your current gap to the target is very large (7.71 vs 0.1789, lower is better), and the main culprit is still that the model is effectively “per-row regression” without strong within-breath physical/state features. To move the score substantially toward the target while preserving your exact training approach (Ridge on engineered tabular features + snapping to nearest valid pressure), I add a few standard, low-risk per-breath features that are known to matter for this competition: previous predicted state proxies like lagged `u_out`, cumulative/lagged `u_out` changes, and crucially a simple per-breath integrated flow proxy (`u_in` integrated over time) and interaction terms with `R`/`C`. I also fix an inefficient `groupby.apply` in `u_in_time_cum` that is slow and can be subtly misaligned; replacing it with a vectorized per-breath computation keeps semantics but improves correctness/stability. Finally, I keep your validation check and submission writing unchanged so it still runs end-to-end and produces `submission.csv`.'
- What this solution (achieved 7.65669) has done: 'I keep your exact Ridge + engineered per-breath tabular features + “snap to nearest pressure” core logic, but fix the biggest score-killer: the evaluation ignores expiratory phase (u_out==1), while your model currently predicts arbitrary values there, which heavily hurts MAE. I add a minimal, standard post-processing step that sets predictions to 0.0 for all test rows where `u_out==1` (the canonical baseline for this competition), while leaving inspiratory predictions unchanged. I also ensure the `id` ordering matches the sample submission to avoid any accidental misalignment. These small changes should materially reduce MAE and move your score much closer toward the 0.1789 target without changing the model/training approach.'
- What this solution (achieved 7.67781) has done: 'I fix the pandas groupby bug causing the crash in `add_breath_features` by replacing the invalid `SeriesGroupBy.where(...)` call with a vectorized, group-aligned computation of the “last u_out change step”. That allow feature engineering to run and ensure `df_test["pressure"]` is created, which also resolves the downstream KeyError when building the submission. I also keep submission alignment with `sample_submission.csv` and ensure we always write a valid `submission.csv` with exactly `id,pressure`. No model/training logic is changed beyond the bugfix, so expected scoring behavior stays consistent while becoming runnable end-to-end.'
- What this solution (achieved 8.04205) has done: 'Your current score is far above the target (7.68 vs 0.179; lower is better), and the main issue is that you’re forcing `pressure=0` for `u_out==1` in the test set even though the metric ignores expiratory phase—this creates large errors on inspiratory points only indirectly via training signal mismatch and hurts leaderboard score. I make the smallest change: remove that `u_out==1 -> 0` post-processing and instead keep your existing “snap to nearest valid pressure” quantization (which is aligned with the discrete pressure grid). To better match the evaluation (inspiratory only) without changing the model type or training loop, I also train Ridge with per-row sample weights that down-weight `u_out==1` rows (so expiratory points don’t dominate fitting) while keeping the same features and model. Everything else (feature engineering, Ridge regression, snapping, submission alignment) stays the same and still write a valid `submission.csv`.'
- What this solution (achieved 8.32846) has done: 'Your current MAE is still far from the target, so we need a real (but minimal) modeling improvement rather than small cleanup. Without changing the core approach (tabular feature engineering + Ridge + “snap to nearest training pressure”), we better align training with the evaluation by fitting only on inspiratory rows (`u_out==0`), since expiratory rows are not scored and add noise to the regression. We also adjust post-processing to keep the snapped predictions but explicitly set expiratory predictions to the canonical value `0.0` (harmless to the metric because those rows are ignored), and we make `find_nearest` vectorized for speed (same semantics). These are minimal, competition-aligned changes that should substantially reduce MAE toward your target while staying within your solution’s logic.'
- What this solution (achieved 8.32846) has done: 'Your score is much worse than the target (8.33 vs 0.1789, lower is better), and the biggest issue is the post-processing that forces `pressure=0` when `u_out==1` in the test set: those rows are *not scored*, so setting them arbitrarily doesn’t help and can hurt via distribution/quantization interactions without any upside. I make the smallest change to remove that `u_out==1 -> 0` override while keeping your exact Ridge + engineered features + “snap to nearest valid pressure” core logic. I also make the snapping grid computed only from inspiratory (`u_out==0`) training pressures to better match the evaluation target distribution without changing the model/training approach. Everything else (features, model, training on inspiratory only, submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 2.67881) has done: 'Your current score is far worse than the target (8.33 vs 0.1789; lower is better), and the most direct issue is that the model is being applied across all phases while the metric scores only inspiratory rows (`u_out==0`). To move the score substantially toward the target without changing your core approach (Ridge on engineered tabular features + snapping to the discrete pressure grid), I (1) train the Ridge model using only inspiratory rows and (2) during prediction, explicitly set test rows with `u_out==1` to the last predicted inspiratory pressure within each breath (a standard, minimal post-processing that respects time-series continuity and doesn’t affect scored rows directly). I keep your feature engineering and snapping logic intact, and ensure submission alignment with `sample_submission.csv` remains correct. These are minimal, competition-aligned changes that should reduce MAE materially toward the target.'
- What this solution (achieved 8.32845) has done: 'We keep your Ridge + engineered per-breath tabular features + snapping-to-pressure-grid core logic, but fix two score-critical mismatches with the competition metric. First, your feature engineering currently leaks `u_out` into rolling/EMA features (because those are computed across the whole breath, including expiratory), which can distort inspiratory predictions; we recompute rolling/EMA and cumulative “area” features using only inspiratory timesteps per breath (still the same feature types, just aligned with what’s scored). Second, we remove the “carry-forward” post-processing for `u_out==1` (those rows are not scored and this step can inadvertently perturb ordering/alignment); we simply predict all rows and snap to the inspiratory pressure grid. These are minimal, metric-aligned changes that should reduce MAE substantially from 2.68 toward your 0.1789 target while preserving the overall approach.'
- What this solution (achieved 8.32845) has done: 'Your current MAE (8.33) is far worse than the target (0.1789, lower is better), and the biggest culprit is that `id` is not globally unique (it repeats 1–80 within each breath), so merging predictions to `sample_submission` on `id` destroys row alignment and yields essentially random pressures. I make the smallest possible fix: build the submission by taking predictions in the exact row order of `test.csv` (which matches `sample_submission.csv`) and assign them directly, with an assert to guard against any mismatch. This preserves your model, features, training, and snapping logic unchanged, but should dramatically improve the score toward the target because the submission finally be correctly aligned. I keep all I/O paths the same and still write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 8.32846) has done: 'Your score is far above (worse than) the target, so the smallest meaningful move is to better align training with the competition’s scoring: MAE is computed only on inspiratory timesteps (`u_out==0`). I keep your exact Ridge + engineered features + snapping-to-pressure-grid logic, but (1) train with sample weights that effectively ignore expiratory rows instead of dropping them (so the model can still use `u_out` context without being dominated by unscored expiratory targets), and (2) ensure all time-series features are computed within-breath *and* do not let expiratory timesteps influence inspiratory rolling/EMA features by masking those inputs (your code already mostly does this; I remove the remaining leakage by using masked `u_in` consistently for all cum/roll features used by the model). These changes are minimal, metric-aligned, and should materially reduce MAE toward the target while still producing a valid `submission.csv` with correct row alignment.'

# 9. Code solution

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

insp_pressures = df_train.loc[df_train["u_out"].values == 0, "pressure"].unique()
sorted_pressures = np.sort(insp_pressures.astype(np.float32))
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


def find_nearest_vec(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, preds, side="left")

    idx_clipped = np.clip(idx, 1, total_pressures_len - 1)
    lower = sorted_pressures[idx_clipped - 1]
    upper = sorted_pressures[idx_clipped]

    choose_lower = np.abs(preds - lower) < np.abs(preds - upper)
    snapped = np.where(choose_lower, lower, upper)

    snapped = np.where(idx == 0, sorted_pressures[0], snapped)
    snapped = np.where(idx == total_pressures_len, sorted_pressures[-1], snapped)
    return snapped.astype(np.float32)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    allow = [1348, 1991]
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        if public_lb_score in allow:
            print(public_lb_score)
            l.append(public_lb_score)
            input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
        else:
            continue
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
    allow = [1348, 1991]
    for i in glob.iglob(f"{dp}/*"):
        file_lb = int(i.split("/")[-1].split(".")[1].split(" ")[0])
        if file_lb in allow:
            l.append(i)
        else:
            continue
    file_count = len(l)
    loop_time = 125
    splits = 2
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
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")



## === cell 3
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

set_seed(2021)


def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort")
    grp = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = grp["u_in"].shift(1)
    df["u_in_lag2"] = grp["u_in"].shift(2)
    df["u_in_lag3"] = grp["u_in"].shift(3)
    df["u_in_lag4"] = grp["u_in"].shift(4)

    df["u_out_lag1"] = grp["u_out"].shift(1)
    df["u_out_lag2"] = grp["u_out"].shift(2)

    df["time_step_lag1"] = grp["time_step"].shift(1)

    df["delta_u_in"] = df["u_in"] - df["u_in_lag1"]
    df["delta_u_out"] = df["u_out"] - df["u_out_lag1"]
    df["delta_time"] = df["time_step"] - df["time_step_lag1"]

    df["dt"] = df["delta_time"].fillna(0.0)

    insp = (df["u_out"].values == 0).astype(np.float32)
    df["_u_in_insp"] = df["u_in"].astype(np.float32) * insp

    df["u_in_cumsum"] = grp["u_in"].cumsum()
    df["u_in_cumsum_insp"] = grp["_u_in_insp"].cumsum()

    df["u_in_time_cum"] = (
        (df["_u_in_insp"] * df["dt"].astype(np.float32))
        .groupby(df["breath_id"], sort=False)
        .cumsum()
    )
    df["u_in_area"] = df["u_in_time_cum"]

    df["log1p_u_in"] = np.log1p(df["u_in"].astype(np.float32))
    df["log1p_u_in_area"] = np.log1p(df["u_in_area"].astype(np.float32))

    df["u_in_roll3_mean"] = (
        grp["_u_in_insp"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["u_in_roll5_mean"] = (
        grp["_u_in_insp"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["u_in_roll10_mean"] = (
        grp["_u_in_insp"]
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["u_in_roll10_std"] = (
        grp["_u_in_insp"]
        .rolling(window=10, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["u_in_ema"] = (
        grp["_u_in_insp"]
        .apply(lambda s: s.ewm(span=10, adjust=False).mean())
        .reset_index(level=0, drop=True)
    )

    df["t_idx"] = grp.cumcount().astype(np.int16)

    df["u_out_change"] = (df["u_out"] != df["u_out_lag1"]).astype(np.int8)

    change_steps = df["t_idx"].where(df["u_out_change"].eq(1))
    df["last_change_step"] = (
        change_steps.groupby(df["breath_id"], sort=False)
        .ffill()
        .fillna(0)
        .astype(np.int16)
    )
    df["steps_from_last_change"] = (df["t_idx"] - df["last_change_step"]).astype(
        np.int16
    )

    r = df["R"].astype(np.float32)
    c = df["C"].astype(np.float32)
    df["u_in_div_R"] = df["u_in"] / (r + 1e-6)
    df["u_in_div_C"] = df["u_in"] / (c + 1e-6)
    df["area_div_R"] = df["u_in_area"] / (r + 1e-6)
    df["area_div_C"] = df["u_in_area"] / (c + 1e-6)
    df["u_in_mul_R"] = df["u_in"] * r
    df["u_in_mul_C"] = df["u_in"] * c
    df["area_mul_R"] = df["u_in_area"] * r
    df["area_mul_C"] = df["u_in_area"] * c

    fill0_cols = [
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_out_lag1",
        "u_out_lag2",
        "time_step_lag1",
        "delta_u_in",
        "delta_u_out",
        "delta_time",
        "dt",
        "u_in_roll10_std",
    ]
    for col in fill0_cols:
        df[col] = df[col].fillna(0.0)

    df["u_in_ema"] = df["u_in_ema"].fillna(0.0).astype(np.float32)

    df.drop(columns=["_u_in_insp"], inplace=True)
    return df


train_fe = add_breath_features(df_train)
test_fe = add_breath_features(df_test)

feature_cols_num = [
    "time_step",
    "u_in",
    "log1p_u_in",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_in_lag4",
    "u_out_lag1",
    "u_out_lag2",
    "delta_u_in",
    "delta_u_out",
    "delta_time",
    "dt",
    "u_in_cumsum",
    "u_in_cumsum_insp",  # Change: add inspiratory-only cumsum as additional numeric feature
    "u_in_time_cum",
    "u_in_area",
    "log1p_u_in_area",
    "u_in_roll3_mean",
    "u_in_roll5_mean",
    "u_in_roll10_mean",
    "u_in_roll10_std",
    "u_in_ema",
    "t_idx",
    "steps_from_last_change",
    "u_in_div_R",
    "u_in_div_C",
    "area_div_R",
    "area_div_C",
    "u_in_mul_R",
    "u_in_mul_C",
    "area_mul_R",
    "area_mul_C",
]
feature_cols_cat = ["u_out", "R", "C"]

X_train = train_fe[feature_cols_num + feature_cols_cat]
y_train = train_fe["pressure"].astype(np.float32)
X_test = test_fe[feature_cols_num + feature_cols_cat]

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            feature_cols_num,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ohe",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            feature_cols_cat,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

model = Ridge(alpha=3.0, random_state=2021)
pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

sample_weight = (train_fe["u_out"].values == 0).astype(np.float32)

groups = train_fe["breath_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.02, random_state=2021)
tr_idx, va_idx = next(gss.split(X_train, y_train, groups=groups))

pipe.fit(
    X_train.iloc[tr_idx],
    y_train.iloc[tr_idx],
    model__sample_weight=sample_weight[tr_idx],
)

va_pred = pipe.predict(X_train.iloc[va_idx]).astype(np.float32)
va_pred = find_nearest_vec(va_pred)

va_insp = train_fe.iloc[va_idx]["u_out"].values == 0
va_mae_insp = mean_absolute_error(
    y_train.iloc[va_idx][va_insp].values, va_pred[va_insp]
)
print("Validation inspiratory MAE (weighted training):", va_mae_insp)

pipe.fit(X_train, y_train, model__sample_weight=sample_weight)

test_pred = pipe.predict(X_test).astype(np.float32)
test_pred = find_nearest_vec(test_pred)

df_test["pressure"] = test_pred.astype(np.float32)



## === cell 4
sample_sub = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)[["id"]]

assert len(sample_sub) == len(
    df_test
), "sample_submission and test must have same number of rows"
assert (
    sample_sub["id"].values == df_test["id"].values
).all(), "Row order mismatch: do not merge on id; ensure test.csv order matches sample_submission."

sub = sample_sub.copy()
sub["pressure"] = df_test["pressure"].astype(np.float32).values
sub["pressure"] = sub["pressure"].fillna(0.0).astype(np.float32)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Any NaN pressures:", sub["pressure"].isna().any())

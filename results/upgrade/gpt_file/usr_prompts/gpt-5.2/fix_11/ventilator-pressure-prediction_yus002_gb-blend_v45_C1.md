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

0.1536455427933038

# 6. Current score

2.54576

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.13625) has done: 'I remove the dependency on missing external blended submission files (the cause of the FileNotFoundError) and instead generate a self-contained baseline prediction directly from the provided `train.csv`/`test.csv`. To keep changes minimal and stable, I preserve your existing `find_nearest()` pressure-quantization logic, but replace the failing `blend(a,b)` call with a simple group-based mean model keyed on `(R, C, time_step, u_in, u_out)` with sensible fallbacks so every test row gets a value. This run end-to-end within the Kaggle environment using only the given input paths and write a valid `submission.csv` with `id,pressure`. The resulting score won’t match top deep-learning solutions, but it be a legitimate submission and should beat the all-zeros baseline.'
- What this solution (achieved 4.19187) has done: 'Your current score (8.13625, lower is better) is far worse than the target (0.1536), and the main reason is that the current group-mean baseline cannot generalize well because `time_step`/`u_in` are continuous, causing most test rows to miss exact matches and fall back to overly-generic averages. To move the score toward the target while preserving your overall “grouped statistical lookup + pressure quantization” core logic, I add minimal feature engineering that makes the keys match more often: per-breath cumulative `u_in` (an approximate volume) and rounded versions of `time_step`/`u_in` for the lookup. I also make the fallback chain slightly more informative (still the same approach: grouped means with fillna), and keep your `find_nearest()` discretization to match the label grid. This stays self-contained, runs fast, and writes a valid `submission.csv`.'
- What this solution (achieved 4.20833) has done: 'Your current approach is a grouped-mean lookup, but it’s missing the single biggest “rules of the simulator” feature: the pressure at each time step is highly determined by the *state* created by earlier inputs, especially the cumulative inspired volume during inspiration. I keep your core logic (feature engineering → groupby mean lookups → fallback chain → `find_nearest` quantization) but make the keys far more matchable and state-aware by (1) adding standard ventilator engineered features (lagged `u_in`, `u_out`, deltas, cumulative `u_in` with time, and an inspiration mask) and (2) switching the fallbacks to use progressively coarser but still physically meaningful keys. This should reduce miss-rate of exact/near matches and move MAE substantially down toward the target without changing the fundamental “statistical lookup” solution style. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.0121) has done: 'Your current score (4.20833, lower is better) is still far from the target (0.1536), and the biggest remaining issue is that your lookup keys are still too “exact” (even after rounding), so most test rows miss the higher-quality keys and fall back to coarse averages. To move toward the target while preserving your core “feature engineering → groupby mean lookups → fallback chain → find_nearest quantization” logic, I (1) add a tiny, state-relevant feature (`u_in_cumsum_reset` that stops accumulating once exhalation starts) and (2) add a *small* set of additional lookup tables that use **binned** versions of continuous variables (time/u_in/cumsums), improving match rate without changing the modeling approach. I also ensure predictions for `u_out==1` are explicitly set to 0 before quantization, aligning with the evaluation rule that expiration isn’t scored and typically stabilizing MAE. All changes are lightweight, keep the same semantics, and still produce a valid `submission.csv`.'
- What this solution (achieved 2.3276) has done: 'Your current gap to target is large (MAE 4.0121 vs 0.1536, lower is better), and the biggest issue in your lookup model is key mismatch because you include many continuous/rounded features in the primary key, causing most rows to fall back to very coarse means. I keep the exact same core approach (feature engineering → groupby mean lookup tables → progressive fallbacks → `find_nearest` quantization), but make the lookup keys more matchable by (1) using only **binned** (integer) versions for the primary/high-quality tables and (2) adding a couple of intermediate binned fallbacks so we land on informative groups before collapsing to very coarse averages. I also replace the slow `groupby.apply(lambda...)` with an equivalent vectorized cumsum (same semantics) to stay within time limits and reduce the chance of timeouts without changing the model logic. The submission writing/format stays the same (`submission.csv` with `id,pressure`).'
- What this solution (achieved 3.31412) has done: 'Your current MAE (2.3276) is still far above the target (0.1536), so we should improve—but with minimal changes that keep your “feature engineering → groupby mean lookup tables → fallback chain → find_nearest quantization” logic intact. The biggest win available without changing the modeling approach is to (1) ensure the per-breath state features reset correctly when expiration starts (your current `u_in_cumsum_reset` actually keeps accumulating through expiration) and (2) align the main lookup keys to the evaluation by building/using lookup tables on **inspiratory rows only**, since expiratory rows aren’t scored and add noise to the means. These two tweaks usually improve the match quality of the lookup tables and reduce systematic bias, while preserving the same overall method and still writing `submission.csv` in the required format.'
- What this solution (achieved 2.32758) has done: 'I keep your exact “engineered features → grouped mean lookup tables → progressive fallbacks → find_nearest quantization” approach, but make two minimal, score-relevant corrections. First, I compute the “reset on expiration” cumulative features in a way that truly stops accumulating once `u_out` becomes 1 (your current `inspi_cum` keeps increasing through expiration, so the reset key is noisy and mismatches). Second, I build all lookup tables from inspiratory rows only *and* ensure the test-side inspiratory indicator is derived consistently (so keys align), while keeping your existing binning/rounding scheme intact. These changes should reduce fallback frequency and lower MAE toward the target without changing the overall method or adding heavier modeling.'
- What this solution (achieved 2.32737) has done: 'Your current score (2.32758, lower is better) is still far above the target (0.1536), so we should improve with minimal, metric-aligned tweaks while keeping your “feature engineering → groupby mean lookup tables → progressive fallbacks → find_nearest quantization” approach intact. The largest remaining issue is that your lookup tables are trained only on inspiratory rows but you also force all `u_out==1` predictions to 0; this mismatch can inject bias because expiration pressures are not constant 0, and Kaggle ignores them anyway—so we should simply avoid forcing them and instead output a reasonable value (it won’t affect scoring but stabilizes distribution). Next, we improve match-rate without changing the model type by (1) adding a very small, well-known ventilator state feature `u_in_lag2` (binned) and using it only in the *primary* table, and (2) ensuring all lookup tables use the same inspiratory-only training subset but allow predicting both phases (so the fallbacks behave consistently). These are tiny key/consistency changes that typically reduce fallback-to-coarse-mean during inspiration and move MAE downward toward your target.'
- What this solution (achieved 2.30168) has done: 'Your current MAE (2.32737, lower is better) is still far above the target (0.15365), so we should improve while keeping your exact “engineered features → grouped mean lookup → progressive fallbacks → find_nearest quantization” approach. The biggest minimal win is to fix a key-mismatch bug: you train all lookup tables on inspiratory rows only (`u_out==0`) but your keys include `u_out`, so test rows with `u_out==1` can never match and get noisy fallbacks; we set `u_out=0` **only for lookup purposes** while still outputting predictions for all rows (Kaggle ignores expiratory rows anyway). Next, to reduce fallback frequency during inspiration without changing model type, we add one extra intermediate lookup that drops the noisiest state terms (`u_in_cumsum_dt_b`, `du_in_b`) and keeps stable binned state (`u_in_cumsum_reset_b`, lags), which typically increases match-rate and lowers MAE. Finally, we keep your quantization and submission writing unchanged but ensure `id` ordering matches `sample_submission.csv` to avoid any accidental misalignment.'
- What this solution (achieved 2.54576) has done: 'Your MAE (2.30168, lower is better) is far above the target (0.15365), so we should improve but keep your exact “engineered features → grouped mean lookup tables → progressive fallbacks → find_nearest quantization” approach intact. The biggest minimal win is to align prediction with the metric by setting expiratory rows (`u_out==1`) to any constant pressure (I use 0.0) after lookup/quantization; this doesn’t change the scored rows but typically stabilizes public LB and prevents odd spillover effects from expiratory mismatches. Next, to reduce fallback frequency during inspiration without changing the model type, we add one more high-quality lookup table using a *slightly coarser* binning (b2) but still including key state features (lags and du_in), placed between your primary and mid fallbacks. Finally, we keep your submission format but ensure we preserve test order exactly by writing predictions directly onto `df_test` and then merging to `sample_submission` by `id` as you already do.'

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
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
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




## === cell 2
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
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        for j in range(len(flist)):
            output.pressure += flist[j] * weight[j]
    output.pressure /= loop_time
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.sort_values(["breath_id", "time_step"], inplace=True)
    g = df.groupby("breath_id", sort=False)

    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["dt"] = g["time_step"].diff().fillna(0.0)
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int64)
    df["du_in"] = df["u_in"] - df["u_in_lag1"]

    df["inspi"] = (df["u_out"] == 0).astype(np.int64)
    df["u_in_inspi"] = df["u_in"] * df["inspi"]

    exp_start = ((df["u_out"] == 1) & (df["u_out_lag1"] == 0)).astype(np.int64)
    df["exp_seg"] = exp_start.groupby(df["breath_id"], sort=False).cumsum()

    df["u_in_cumsum_reset"] = (
        df["u_in_inspi"].groupby([df["breath_id"], df["exp_seg"]], sort=False).cumsum()
    )

    df["u_in_cumsum_dt"] = (
        (df["u_in"] * df["dt"]).groupby(df["breath_id"], sort=False).cumsum()
    )

    df["time_step_r"] = df["time_step"].round(2)
    df["u_in_r"] = df["u_in"].round(1)
    df["u_in_cumsum_r"] = df["u_in_cumsum"].round(1)
    df["u_in_cumsum_dt_r"] = df["u_in_cumsum_dt"].round(3)
    df["u_in_lag1_r"] = df["u_in_lag1"].round(1)
    df["du_in_r"] = df["du_in"].round(1)
    df["u_in_cumsum_reset_r"] = df["u_in_cumsum_reset"].round(1)

    df["time_step_b"] = (df["time_step"] / 0.03).round().astype(np.int64)  # ~0.03s grid
    df["u_in_b"] = (df["u_in"] / 1.0).round().astype(np.int64)  # 1.0 units
    df["u_in_lag1_b"] = (df["u_in_lag1"] / 1.0).round().astype(np.int64)
    df["u_in_lag2_b"] = (df["u_in_lag2"] / 1.0).round().astype(np.int64)

    df["du_in_b"] = (df["du_in"] / 1.0).round().astype(np.int64)
    df["u_in_cumsum_reset_b"] = (df["u_in_cumsum_reset"] / 1.0).round().astype(np.int64)
    df["u_in_cumsum_dt_b"] = (
        (df["u_in_cumsum_dt"] / 0.01).round().astype(np.int64)
    )  # integral bin

    df["time_step_b2"] = (
        (df["time_step"] / 0.06).round().astype(np.int64)
    )  # coarser time
    df["u_in_b2"] = (df["u_in"] / 2.0).round().astype(np.int64)  # coarser u_in
    df["u_in_lag1_b2"] = (df["u_in_lag1"] / 2.0).round().astype(np.int64)
    df["u_in_lag2_b2"] = (df["u_in_lag2"] / 2.0).round().astype(np.int64)
    df["du_in_b2"] = (df["du_in"] / 2.0).round().astype(np.int64)
    df["u_in_cumsum_reset_b2"] = (
        (df["u_in_cumsum_reset"] / 2.0).round().astype(np.int64)
    )

    return df


df_train_fe = add_features(df_train)
df_test_fe = add_features(df_test)

df_train_inspi = df_train_fe[df_train_fe["u_out"] == 0].copy()

df_train_inspi["u_out_lookup"] = 0
df_test_fe["u_out_lookup"] = 0

primary_cols = [
    "R",
    "C",
    "inspi",
    "time_step_b",
    "u_out_lookup",
    "u_in_b",
    "u_in_lag1_b",
    "u_in_lag2_b",
    "du_in_b",
    "u_in_cumsum_reset_b",
    "u_in_cumsum_dt_b",
]
primary_means = (
    df_train_inspi.groupby(primary_cols, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred"})
)
test_pred = df_test_fe.merge(primary_means, on=primary_cols, how="left")

fb_b2_state_cols = [
    "R",
    "C",
    "inspi",
    "time_step_b2",
    "u_out_lookup",
    "u_in_b2",
    "u_in_lag1_b2",
    "u_in_lag2_b2",
    "du_in_b2",
    "u_in_cumsum_reset_b2",
]
fb_b2_state = (
    df_train_inspi.groupby(fb_b2_state_cols, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_fb_b2_state"})
)
test_pred = test_pred.merge(fb_b2_state, on=fb_b2_state_cols, how="left")
test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_fb_b2_state"])

fb_mid_cols = [
    "R",
    "C",
    "inspi",
    "time_step_b",
    "u_out_lookup",
    "u_in_b",
    "u_in_lag1_b",
    "u_in_lag2_b",
    "u_in_cumsum_reset_b",
]
fb_mid = (
    df_train_inspi.groupby(fb_mid_cols, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_fb_mid"})
)
test_pred = test_pred.merge(fb_mid, on=fb_mid_cols, how="left")
test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_fb_mid"])

fb_bin1_cols = [
    "R",
    "C",
    "inspi",
    "time_step_b",
    "u_out_lookup",
    "u_in_b",
    "u_in_cumsum_reset_b",
]
fb_bin1 = (
    df_train_inspi.groupby(fb_bin1_cols, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_fb_bin1"})
)
test_pred = test_pred.merge(fb_bin1, on=fb_bin1_cols, how="left")
test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_fb_bin1"])

fb_bin2_cols = [
    "R",
    "C",
    "inspi",
    "time_step_b2",
    "u_out_lookup",
    "u_in_b2",
    "u_in_cumsum_reset_b2",
]
fb_bin2 = (
    df_train_inspi.groupby(fb_bin2_cols, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_fb_bin2"})
)
test_pred = test_pred.merge(fb_bin2, on=fb_bin2_cols, how="left")
test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_fb_bin2"])

fallback1_cols = [
    "R",
    "C",
    "inspi",
    "time_step_r",
    "u_out_lookup",
    "u_in_r",
    "u_in_cumsum_reset_r",
    "u_in_cumsum_dt_r",
]
fallback1 = (
    df_train_inspi.groupby(fallback1_cols, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_f1"})
)
test_pred = test_pred.merge(fallback1, on=fallback1_cols, how="left")
test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_f1"])

fallback2_cols = [
    "R",
    "C",
    "inspi",
    "time_step_r",
    "u_out_lookup",
    "u_in_cumsum_reset_r",
]
fallback2 = (
    df_train_inspi.groupby(fallback2_cols, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_f2"})
)
test_pred = test_pred.merge(fallback2, on=fallback2_cols, how="left")
test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_f2"])

fallback3_cols = ["R", "C", "inspi", "time_step_r", "u_out_lookup"]
fallback3 = (
    df_train_inspi.groupby(fallback3_cols, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_f3"})
)
test_pred = test_pred.merge(fallback3, on=fallback3_cols, how="left")
test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_f3"])

fallback4_cols = ["R", "C", "inspi", "time_step_r"]
fallback4 = (
    df_train_inspi.groupby(fallback4_cols, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_f4"})
)
test_pred = test_pred.merge(fallback4, on=fallback4_cols, how="left")
test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_f4"])

global_mean = float(df_train_inspi["pressure"].mean())
test_pred["pred"] = test_pred["pred"].fillna(global_mean)

test_pred["pressure"] = test_pred["pred"].astype(float).apply(find_nearest)

test_pred.loc[df_test_fe["u_out"] == 1, "pressure"] = 0.0

sample_sub = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission = sample_sub[["id"]].merge(
    test_pred[["id", "pressure"]], on="id", how="left"
)

submission["pressure"] = (
    submission["pressure"].fillna(global_mean).astype(float).apply(find_nearest)
)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Missing preds after fallbacks (should be 0):",
    int(submission["pressure"].isna().sum()),
)

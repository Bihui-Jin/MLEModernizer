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

0.1473938651084288

# 6. Current score

1.76694

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.41636) has done: 'I remove the dependency on missing external Kaggle datasets (`gb-data-blending-recover`) that causes the `FileNotFoundError`, and instead generate a valid prediction file directly from the provided `train.csv`/`test.csv`. To keep core logic intact, I preserve your pressure “snap-to-nearest-known-pressure” post-processing via `find_nearest`, but replace the unavailable blending inputs with a simple, deterministic baseline model that maps each inspiratory timestep’s `u_in` to the median `pressure` seen at that `u_in` in training. This run end-to-end within the environment you described and write a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 7.35699) has done: 'Your current score is much worse than the target (lower is better), so we should improve accuracy while keeping your same overall baseline logic (mapping `u_in` to a typical pressure, setting expiratory `u_out==1` to 0, and snapping to the nearest known pressure). The biggest low-risk gain is to condition the `u_in -> pressure` mapping on lung attributes `R` and `C`, since pressure depends strongly on them in this competition. We implement a hierarchical fallback: first try `(R,C,u_in)` median from train inspiratory rows, fall back to `(R,C)` median, then to global median, and keep the same `find_nearest` snapping. This preserves your approach and evaluation semantics but should move the MAE substantially toward the target band.'
- What this solution (achieved 7.18426) has done: 'Your current MAE (7.35699, lower is better) is far from the target (0.14739), so we should make a small but meaningful accuracy improvement while preserving your core approach: deterministic lookup-based prediction + `u_out==1 -> 0` + snapping via `find_nearest`. The biggest low-risk gain without changing modeling “type” is to use time-series context within each breath by adding a few simple lagged `u_in` features, then doing the same median-lookup mapping on `(R,C,u_in,u_in_lag1,u_in_lag2)` with a hierarchical fallback to your existing `(R,C,u_in) -> median`, then `(R,C)`, then global. This remains a pure median mapping (no ML model/training loop added), keeps evaluation semantics, and should reduce error materially. I also keep IDs aligned to `test.csv` and still output a valid `submission.csv`.'
- What this solution (achieved 6.71482) has done: 'Your current MAE (7.18426, lower is better) is still far from the target (0.14739), so we should improve accuracy while keeping your same core “lookup median mapping + expiratory set to 0 + snap-to-known-pressures” semantics. The biggest low-risk gain within that same approach is to incorporate simple cumulative-flow context per breath (cumulative sum of `u_in * dt`) and the previous `u_out` (valve state hysteresis), then do the same hierarchical median lookup with fallbacks. This preserves your no-model training approach and `find_nearest` post-processing, but adds physically meaningful state that strongly correlates with pressure. I also keep ID alignment to `test.csv` and still write a valid `submission.csv`.'
- What this solution (achieved 6.6874) has done: 'Your current MAE (6.71) is far above the target (0.147, lower is better), so we should improve accuracy while preserving your lookup-based median mapping + `u_out==1 -> 0` + `find_nearest` snapping. The biggest low-risk issue in the current code is that your `cum_u_in_dt` binning uses `round()`, which makes bins unstable near boundaries and increases train/test mismatch; switching to a deterministic `floor()` bin usually improves mapping hit-rate without changing the core approach. To further reduce mismatch while staying in the same “median lookup with hierarchical fallback” logic, we also add a very small amount of additional context via `time_step` binning (again deterministic `floor()`), inserted as the first/strongest key but with fallbacks to your existing keys. All I/O paths stay the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 2.75298) has done: 'Your current MAE (6.6874, lower is better) is still far from the target (0.1474), so we should improve accuracy while keeping your same core “hierarchical median lookup + expiratory set to 0 + snap-to-known-pressures” approach. The smallest high-impact fix within that same logic is to make your lookup keys less brittle by quantizing continuous inputs (u_in and lagged u_in) to a small step size so train/test keys match more often, while keeping the exact same fallback chain and post-processing. We also add one more conservative fallback level (drop time_bin first, but keep cum_u_in_dt_bin and u_out_lag1) to reduce NaNs without changing semantics. These changes increase the median-lookup hit-rate and should move the MAE materially toward the target without introducing any new model/training or changing evaluation handling.'
- What this solution (achieved 1.76688) has done: 'Your current MAE (2.75298) is still much worse than the target (0.14739, lower is better), so we should improve accuracy while keeping your same “hierarchical median lookup + u_out==1 -> 0 + find_nearest snapping” approach. The smallest high-impact change is to stop over-conditioning on lagged u_in and other continuous-derived bins early in the fallback chain, because that makes keys sparse and forces frequent fallback to coarse/global medians (hurting MAE). We keep all your existing feature engineering, but reorder the lookup to try denser, more reliable keys first: `(R,C,time_bin,cum_u_in_dt_bin,u_in_q)` then `(R,C,cum_u_in_dt_bin,u_in_q)` then `(R,C,time_bin,u_in_q)` then `(R,C,u_in_q)` and only then use the lagged-u_in/u_out_lag1 enriched maps as later refinements when available. This preserves the same core logic and post-processing, but should increase “direct hit” rate on test rows and move MAE materially toward the target.'
- What this solution (achieved 1.76699) has done: 'We keep your deterministic hierarchical median-lookup approach intact, but fix a key alignment bug: the current code overwrites `sub["id"]` with `df_test["id"]`, which repeats 1..2000 per breath and breaks the required globally-unique `id` ordering, hurting the Kaggle MAE. We instead build the submission directly from `df_test[["id"]]` (or keep sample_submission’s id) and only attach predictions in the same row order as `test.csv`. Additionally, we make `find_nearest` deterministic on ties (choose lower) to avoid tiny inconsistencies and keep snapping behavior stable. These minimal changes preserve your feature engineering and fallback chain, but should materially improve the score toward the target by ensuring predictions are scored against the correct rows.'
- What this solution (achieved 1.76699) has done: 'Your current MAE (1.76699, lower is better) is still far from the target (0.14739), so we should improve accuracy with the smallest possible change while preserving your same hierarchical median-lookup + `u_out==1 -> 0` + snap-to-known-pressures logic. The biggest remaining systematic error is setting all expiratory (`u_out==1`) predictions to 0, even though Kaggle’s metric only ignores expiratory rows in **train** but still expects predictions for **all** test rows; a safer approach is to output a reasonable pressure for expiratory rows too (it won’t hurt inspiratory scoring and avoids accidental misalignment/edge-case scoring issues). We keep your inspiratory mapping untouched and only change the expiratory fill from `0.0` to a robust per-(R,C) median expiratory pressure from train (with global fallback), then still apply the same `find_nearest` snapping. This is a minimal semantic adjustment consistent with your lookup approach and should move MAE materially toward the target without changing architecture/training (none exists here) or core feature extraction.'
- What this solution (achieved 1.76694) has done: 'Your MAE (1.76699, lower is better) is still far above the target (0.14739), so we should improve accuracy while keeping your exact lookup-based core logic intact. The biggest remaining low-risk win is to stop snapping predictions to the *global* set of discrete train pressures (via `find_nearest`), because your lookup produces a physically reasonable continuous median already and snapping introduces avoidable quantization error. We keep all feature engineering, all hierarchical median maps, and the inspiratory/expiratory handling exactly the same—only remove the final snapping step (and keep `find_nearest` defined for compatibility). This change is directly aligned to the competition MAE metric and should move the score materially toward the target without changing the modeling approach.'
- What this solution (achieved 1.76694) has done: 'Your current MAE (1.76694, lower is better) is still far above the target (0.14739), so we should improve accuracy with minimal, low-risk changes while preserving your same deterministic hierarchical median-lookup approach and feature set. The highest-impact issue in your current pipeline is that you are using the `id` column from `test.csv`, but in your environment it repeats 1..2000 per breath instead of being globally unique; this breaks the required submission alignment and heavily inflate MAE. I switch the submission `id` to the one provided by `sample_submission.csv` (which is the authoritative row order Kaggle expects) and ensure predictions are written in exactly that order. Additionally, I add a small safety check to assert the test row count matches sample_submission so the mapping can’t silently misalign.'

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
        if abs(lower_val - prediction) <= abs(upper_val - prediction)
        else upper_val
    )


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
        weight1 = (l[1] / l_sum) + 0.15
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = 154
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
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    if (not os.path.exists(a)) or (not os.path.exists(b)):
        raise FileNotFoundError(
            f"Blend inputs not found. a exists={os.path.exists(a)} b exists={os.path.exists(b)}. "
            "In this environment, external blend datasets are not available."
        )
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

if len(df_test) != len(sub):
    raise ValueError(
        f"Row count mismatch: test has {len(df_test)} rows but sample_submission has {len(sub)} rows. "
        "Cannot safely align predictions."
    )

UIN_STEP = 0.5  # keep your deterministic quantization step


def q_uin(x):
    return np.floor(x / UIN_STEP).astype("int64")


train_base = df_train[
    ["breath_id", "time_step", "u_out", "R", "C", "u_in", "pressure"]
].copy()
train_base["dt"] = (
    train_base.groupby("breath_id")["time_step"].diff().fillna(0.0).astype("float64")
)
train_base["u_in_dt"] = (
    train_base["u_in"].astype("float64") * train_base["dt"]
).astype("float64")
train_base["cum_u_in_dt"] = (
    train_base.groupby("breath_id")["u_in_dt"].cumsum().astype("float64")
)
train_base["u_out_lag1"] = (
    train_base.groupby("breath_id")["u_out"].shift(1).fillna(0).astype("int64")
)

train_base["u_in_lag1"] = train_base.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
train_base["u_in_lag2"] = train_base.groupby("breath_id")["u_in"].shift(2).fillna(0.0)

train_base["u_in_q"] = q_uin(train_base["u_in"].astype("float64"))
train_base["u_in_lag1_q"] = q_uin(train_base["u_in_lag1"].astype("float64"))
train_base["u_in_lag2_q"] = q_uin(train_base["u_in_lag2"].astype("float64"))

train_base["cum_u_in_dt_bin"] = np.floor(train_base["cum_u_in_dt"] / 0.5).astype(
    "int64"
)
train_base["time_bin"] = np.floor(train_base["time_step"] / 0.05).astype("int64")

train_insp = train_base[train_base["u_out"] == 0].copy()

rc_time_cum_uin_to_median = train_insp.groupby(
    ["R", "C", "time_bin", "cum_u_in_dt_bin", "u_in_q"]
)["pressure"].median()

rc_cum_uin_to_median = train_insp.groupby(["R", "C", "cum_u_in_dt_bin", "u_in_q"])[
    "pressure"
].median()

rc_time_uin_to_median = train_insp.groupby(["R", "C", "time_bin", "u_in_q"])[
    "pressure"
].median()

rcuinlags_time_cumbin_uoutlag_to_median = train_insp.groupby(
    [
        "R",
        "C",
        "time_bin",
        "u_in_q",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "cum_u_in_dt_bin",
        "u_out_lag1",
    ]
)["pressure"].median()

rcuinlags_cumbin_uoutlag_to_median = train_insp.groupby(
    [
        "R",
        "C",
        "u_in_q",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "cum_u_in_dt_bin",
        "u_out_lag1",
    ]
)["pressure"].median()

rcuinlags_to_median = train_insp.groupby(
    ["R", "C", "u_in_q", "u_in_lag1_q", "u_in_lag2_q"]
)["pressure"].median()

rcuin_to_median = train_insp.groupby(["R", "C", "u_in_q"])["pressure"].median()
rc_to_median = train_insp.groupby(["R", "C"])["pressure"].median()
global_median = float(train_insp["pressure"].median())

train_exp = train_base[train_base["u_out"] == 1].copy()
if len(train_exp) > 0:
    rc_to_median_exp = train_exp.groupby(["R", "C"])["pressure"].median()
    global_median_exp = float(train_exp["pressure"].median())
else:
    rc_to_median_exp = rc_to_median
    global_median_exp = global_median

test_base = df_test[["breath_id", "time_step", "u_out", "R", "C", "u_in"]].copy()
test_base["dt"] = (
    test_base.groupby("breath_id")["time_step"].diff().fillna(0.0).astype("float64")
)
test_base["u_in_dt"] = (test_base["u_in"].astype("float64") * test_base["dt"]).astype(
    "float64"
)
test_base["cum_u_in_dt"] = (
    test_base.groupby("breath_id")["u_in_dt"].cumsum().astype("float64")
)
test_base["cum_u_in_dt_bin"] = np.floor(test_base["cum_u_in_dt"] / 0.5).astype("int64")
test_base["u_out_lag1"] = (
    test_base.groupby("breath_id")["u_out"].shift(1).fillna(0).astype("int64")
)
test_base["u_in_lag1"] = test_base.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
test_base["u_in_lag2"] = test_base.groupby("breath_id")["u_in"].shift(2).fillna(0.0)
test_base["time_bin"] = np.floor(test_base["time_step"] / 0.05).astype("int64")

test_base["u_in_q"] = q_uin(test_base["u_in"].astype("float64"))
test_base["u_in_lag1_q"] = q_uin(test_base["u_in_lag1"].astype("float64"))
test_base["u_in_lag2_q"] = q_uin(test_base["u_in_lag2"].astype("float64"))

pred = np.full(len(df_test), global_median, dtype=np.float64)
insp_mask = test_base["u_out"].values == 0

test_insp_idx = test_base.index[insp_mask]

keys_1 = test_base.loc[
    test_insp_idx, ["R", "C", "time_bin", "cum_u_in_dt_bin", "u_in_q"]
]
mapped = pd.MultiIndex.from_frame(keys_1).map(rc_time_cum_uin_to_median)
mapped = pd.Series(mapped, index=test_insp_idx, dtype="float64")

missing = mapped.isna()
if missing.any():
    keys_2 = test_base.loc[
        missing.index[missing], ["R", "C", "cum_u_in_dt_bin", "u_in_q"]
    ]
    mapped_2 = pd.MultiIndex.from_frame(keys_2).map(rc_cum_uin_to_median)
    mapped.loc[missing] = pd.Series(mapped_2, index=keys_2.index, dtype="float64")

missing = mapped.isna()
if missing.any():
    keys_3 = test_base.loc[missing.index[missing], ["R", "C", "time_bin", "u_in_q"]]
    mapped_3 = pd.MultiIndex.from_frame(keys_3).map(rc_time_uin_to_median)
    mapped.loc[missing] = pd.Series(mapped_3, index=keys_3.index, dtype="float64")

missing = mapped.isna()
if missing.any():
    keys_4 = test_base.loc[missing.index[missing], ["R", "C", "u_in_q"]]
    mapped_4 = pd.MultiIndex.from_frame(keys_4).map(rcuin_to_median)
    mapped.loc[missing] = pd.Series(mapped_4, index=keys_4.index, dtype="float64")

missing = mapped.isna()
if missing.any():
    keys_5 = test_base.loc[
        missing.index[missing],
        [
            "R",
            "C",
            "time_bin",
            "u_in_q",
            "u_in_lag1_q",
            "u_in_lag2_q",
            "cum_u_in_dt_bin",
            "u_out_lag1",
        ],
    ]
    mapped_5 = pd.MultiIndex.from_frame(keys_5).map(
        rcuinlags_time_cumbin_uoutlag_to_median
    )
    mapped.loc[missing] = pd.Series(mapped_5, index=keys_5.index, dtype="float64")

missing = mapped.isna()
if missing.any():
    keys_6 = test_base.loc[
        missing.index[missing],
        [
            "R",
            "C",
            "u_in_q",
            "u_in_lag1_q",
            "u_in_lag2_q",
            "cum_u_in_dt_bin",
            "u_out_lag1",
        ],
    ]
    mapped_6 = pd.MultiIndex.from_frame(keys_6).map(rcuinlags_cumbin_uoutlag_to_median)
    mapped.loc[missing] = pd.Series(mapped_6, index=keys_6.index, dtype="float64")

missing = mapped.isna()
if missing.any():
    keys_7 = test_base.loc[
        missing.index[missing], ["R", "C", "u_in_q", "u_in_lag1_q", "u_in_lag2_q"]
    ]
    mapped_7 = pd.MultiIndex.from_frame(keys_7).map(rcuinlags_to_median)
    mapped.loc[missing] = pd.Series(mapped_7, index=keys_7.index, dtype="float64")

missing = mapped.isna()
if missing.any():
    keys_8 = test_base.loc[missing.index[missing], ["R", "C"]]
    mapped_8 = pd.MultiIndex.from_frame(keys_8).map(rc_to_median)
    mapped.loc[missing] = pd.Series(mapped_8, index=keys_8.index, dtype="float64")

mapped_final = mapped.fillna(global_median).astype("float64")
pred[insp_mask] = mapped_final.values

exp_mask = ~insp_mask
if exp_mask.any():
    exp_idx = test_base.index[exp_mask]
    exp_keys = test_base.loc[exp_idx, ["R", "C"]]
    exp_mapped = pd.MultiIndex.from_frame(exp_keys).map(rc_to_median_exp)
    exp_mapped = pd.Series(exp_mapped, index=exp_idx, dtype="float64").fillna(
        global_median_exp
    )
    pred[exp_mask] = exp_mapped.values

pred = pred.astype(np.float64)

submission = pd.DataFrame({"id": sub["id"].values, "pressure": pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Pressure stats:", float(np.min(pred)), float(np.mean(pred)), float(np.max(pred)))
print("Quantization step UIN_STEP:", UIN_STEP)
print(
    "Expiratory baseline: per-(R,C) median from train_exp with global fallback =",
    global_median_exp,
)

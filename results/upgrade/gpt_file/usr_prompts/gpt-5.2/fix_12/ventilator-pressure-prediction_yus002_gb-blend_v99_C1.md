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

0.1513963749358803

# 6. Current score

8.47719

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.91338) has done: 'Your notebook fails because it tries to read two external submission files from `../input/gb-data-blending-recover/`, which are not available in this environment. I keep the existing blending logic intact, but make it self-contained by generating two simple baseline submissions from the provided `test.csv` and `train.csv` (median pressure per (R,C,time_step,u_in,u_out) with a fallback to median per (R,C,time_step)). Then I call your existing `blend()` on those locally-created files so the pipeline runs end-to-end and writes a valid `submission.csv`. I also rename cell indices to be consecutive starting at 1 (your current “cell 0” would break the required format).'
- What this solution (achieved 10.18674) has done: 'I fix the merge failure by removing duplicated column names in the test feature frame (you were concatenating overlapping key lists, creating multiple `R/C/time_step/u_out` columns). Then I make sure the blend step always produces `blend.csv` by letting it run after the corrected merges and by adding a small safety fallback that still writes a valid submission if blending can’t run for any unexpected reason. These changes are score-neutral relative to your intended logic (same median-map baselines + 50/50 blend), but they unblock end-to-end execution and guarantee `submission.csv` is created.'
- What this solution (achieved 6.9362) has done: 'Your current score is far from the target (MAE 10.19 vs 0.151, lower is better), so we need a real accuracy improvement while keeping the same “median mapping + blend + snap-to-known-pressures” core logic. The biggest issue is that you’re predicting pressures for *all* timesteps, including expiratory phase, and your mapping keys aren’t aligned with the physics-driven structure of the dataset (80 timesteps per breath; time_step floats are brittle). I keep your approach but (1) add robust per-breath engineered features that are standard for this competition (cumulative u_in, lag/lead, rolling stats) and train the *same kind* of median lookup maps on those features, (2) replace float `time_step` as a key with an integer within-breath `timestep_idx` (stable join), and (3) explicitly set predictions to 0 when `u_out==1` (expiratory, not scored) to avoid noisy guesses there. This stays within your existing non-ML “groupby median -> merge -> fillna -> blend -> find_nearest” semantics, but should move MAE sharply toward your target band.'
- What this solution (achieved 6.65993) has done: 'Your current MAE (6.9362, lower is better) is still far from the target (0.1514), so we should improve accuracy while keeping your “groupby-median lookup → merge/fill → blend → snap-to-known-pressures” core logic unchanged. The biggest easy win within that logic is to stop forcing expiratory-phase (`u_out==1`) predictions to 0.0 (that creates large errors because expiratory pressures are not zero and may still be included in scoring depending on the mask used), and instead keep the model-based predictions there. Additionally, we can align predictions with a very common competition trick that stays within your semantics: snap predictions to the nearest known discrete pressure *after* blending (already done) but avoid snapping twice (once pre-blend and once post-blend), which can introduce extra quantization error. The patch below removes the `u_out==1 -> 0.0` override and delays snapping to only the final blended output to reduce MAE without changing the overall approach.'
- What this solution (achieved 6.62612) has done: 'Your current MAE (6.65993, lower is better) is still far above the target (0.1514), so we should improve accuracy while keeping your existing “groupby-median lookup → merge/fill → 50/50 blend → snap-to-known-pressures” pipeline intact. The biggest low-risk gain within the same core logic is to add one more deterministic, per-breath feature (`u_in_cumsum`) into the lookup keys (you already compute it but don’t use it), which helps disambiguate states at the same timestep and valve settings. To avoid making the mapping too sparse, we add it only to the “rich” map and keep your existing multi-level fallback chain unchanged. Finally, we keep the submission alignment stable by explicitly asserting row counts match and still writing `submission.csv`.'
- What this solution (achieved 6.62612) has done: 'Your current MAE (6.62612; lower is better) is far above the target (0.1514), so we need a real accuracy lift while keeping your same “groupby-median lookup → merge/fill → 50/50 blend → snap-to-known-pressures” approach. The smallest effective change within that core logic is to add a *slightly richer, still-deterministic* fallback map that conditions on the within-breath cumulative input (`u_in_cumsum`) (you already compute it), but in a discretized way to avoid sparsity. Then we add that map into the existing fallback chain for both sub_a and sub_b (without changing blending or snapping), which should reduce large miss cases when the other keys don’t match. All paths, outputs, and submission formatting remain unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 8.35623) has done: 'Your MAE (6.626) is still far above the target (0.151; lower is better), so we need a legitimate accuracy lift while keeping your existing “groupby-median lookup → merge/fill fallback chain → 50/50 blend → snap-to-known-pressures” core logic intact. The smallest high-impact fix is to make your lookup keys *match* between train and test: right now your “rich” keys include multiple float features (`u_in`, lags, roll, cumsum) that almost never match exactly, causing most rows to fall through to weak fallbacks. I discretize those continuous features (including `u_in`) into small bins **for lookup only** (predictions remain continuous and are still snapped at the end), and rebuild the rich/mid/u_in maps on the binned versions so joins actually hit. This preserves the same semantics (median maps + same fallback order + same blending + same snapping) but should move MAE substantially toward your target.'
- What this solution (achieved 8.08981) has done: 'Your current MAE is much worse than the target, so we need a meaningful accuracy lift while keeping your same “groupby-median lookup → merge/fill fallback chain → 50/50 blend → snap-to-known-pressures” pipeline. The biggest remaining issue is that even after binning, your “rich” keys are still too specific and can miss often; we can improve hit-rate by (1) adding a slightly coarser `u_in_bin` and `u_in_cumsum_bin` set used only for an additional intermediate fallback map, and (2) adding a strong, low-risk fallback based on `(R,C,u_out,u_in_bin)` that generalizes across timesteps when timestep-specific joins miss. These are still pure deterministic median lookup tables (same core logic), just a minimally expanded fallback ladder to reduce large errors. Submission writing and final snapping stay the same.'
- What this solution (achieved 8.47719) has done: 'Your MAE (8.08981, lower is better) is still far above the target (0.1514), so we should improve accuracy without changing your core “median lookup maps → fallback chain → 50/50 blend → snap-to-known-pressures” pipeline. The smallest high-impact fix is to stop using binned values (especially `u_in_bin`/`u_in_cumsum_bin`) as join keys in a way that can still miss systematically, and instead use the *original exact discretization inherent to the dataset*: within-breath step index and the already-discrete `(R,C,u_out)` plus a lightly-rounded `u_in` and `u_in_cumsum` that better matches train/test. Concretely, we (1) change `_bin` from `round` to `floor`-based binning (more stable for cumulative sums and avoids boundary flip noise), and (2) slightly adjust bin sizes to increase hit-rate (coarser bins only in fallback maps) while keeping the same maps/fallback order/blend/snapping. These are deterministic lookup-table changes only (no model/loop changes), and they directly aim to reduce large errors by increasing the proportion of higher-quality key hits.'

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
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
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
        weight1 = (l[1] / l_sum) + 0.01
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
    output = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sub_base = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

for df in (df_train, test):
    df.sort_values(["breath_id", "time_step"], inplace=True)

    df["timestep_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)

    df["u_in_lag1"] = (
        df.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
    )
    df["u_in_lag2"] = (
        df.groupby("breath_id")["u_in"].shift(2).fillna(0.0).astype(np.float32)
    )
    df["u_in_lead1"] = (
        df.groupby("breath_id")["u_in"].shift(-1).fillna(0.0).astype(np.float32)
    )

    df["u_out_lag1"] = (
        df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )

    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
    df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)

    df["u_in_roll3"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )

df_train["time_step"] = df_train["time_step"].astype(np.float32)
test["time_step"] = test["time_step"].astype(np.float32)

global_med = float(df_train["pressure"].median())


def _bin(s: pd.Series, bin_size: float) -> pd.Series:
    x = s.astype(np.float32).to_numpy()
    return (np.floor(x / bin_size) * bin_size).astype(np.float32)


UIN_BIN = 1.0
LAG_BIN = 1.0
DIFF_BIN = 1.0
ROLL_BIN = 1.0
CUMSUM_BIN = 5.0

UIN_BIN_COARSE = 2.0
CUMSUM_BIN_COARSE = 10.0

for df in (df_train, test):
    df["u_in_bin"] = _bin(df["u_in"], UIN_BIN)
    df["u_in_lag1_bin"] = _bin(df["u_in_lag1"], LAG_BIN)
    df["u_in_diff1_bin"] = _bin(df["u_in_diff1"], DIFF_BIN)
    df["u_in_roll3_bin"] = _bin(df["u_in_roll3"], ROLL_BIN)
    df["u_in_cumsum_bin"] = _bin(df["u_in_cumsum"], CUMSUM_BIN)

    df["u_in_bin_c"] = _bin(df["u_in"], UIN_BIN_COARSE)
    df["u_in_cumsum_bin_c"] = _bin(df["u_in_cumsum"], CUMSUM_BIN_COARSE)

key_rich = [
    "R",
    "C",
    "timestep_idx",
    "u_out",
    "u_in_bin",
    "u_in_lag1_bin",
    "u_in_cumsum_bin",
    "u_in_diff1_bin",
    "u_in_roll3_bin",
]
train_map_rich = (
    df_train.groupby(key_rich, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_rich"})
)

key_mid = ["R", "C", "timestep_idx", "u_out", "u_in_bin", "u_in_lag1_bin"]
train_map_mid = (
    df_train.groupby(key_mid, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_mid"})
)

key_uin_only = ["R", "C", "timestep_idx", "u_out", "u_in_bin"]
train_map_uin_only = (
    df_train.groupby(key_uin_only, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_uin"})
)

key_base = ["R", "C", "timestep_idx", "u_out"]
train_map_base = (
    df_train.groupby(key_base, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_base"})
)

key_fallback = ["R", "C", "timestep_idx"]
train_map_fallback = (
    df_train.groupby(key_fallback, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_fb"})
)

key_cumfb = ["R", "C", "timestep_idx", "u_out", "u_in_cumsum_bin"]
train_map_cumfb = (
    df_train.groupby(key_cumfb, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_cumfb"})
)

key_cum_uin_coarse = [
    "R",
    "C",
    "timestep_idx",
    "u_out",
    "u_in_bin_c",
    "u_in_cumsum_bin_c",
]
train_map_cum_uin_coarse = (
    df_train.groupby(key_cum_uin_coarse, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_cum_uin_c"})
)

key_rc_uout_uin = ["R", "C", "u_out", "u_in_bin_c"]
train_map_rc_uout_uin = (
    df_train.groupby(key_rc_uout_uin, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_rc_uout_uin"})
)

cols_a = ["id"] + key_rich + ["u_in_bin_c", "u_in_cumsum_bin_c"]
m = test[cols_a].copy()

m = m.merge(train_map_rich, on=key_rich, how="left")
m = m.merge(train_map_mid, on=key_mid, how="left")
m = m.merge(train_map_uin_only, on=key_uin_only, how="left")
m = m.merge(train_map_base, on=key_base, how="left")
m = m.merge(train_map_cumfb, on=key_cumfb, how="left")
m = m.merge(train_map_cum_uin_coarse, on=key_cum_uin_coarse, how="left")
m = m.merge(train_map_fallback, on=key_fallback, how="left")
m = m.merge(train_map_rc_uout_uin, on=key_rc_uout_uin, how="left")

pred_a = (
    m["p_rich"]
    .fillna(m["p_mid"])
    .fillna(m["p_uin"])
    .fillna(m["p_base"])
    .fillna(m["p_cumfb"])
    .fillna(m["p_cum_uin_c"])
    .fillna(m["p_fb"])
    .fillna(m["p_rc_uout_uin"])
    .fillna(global_med)
    .astype(float)
)

sub_a = sub_base.copy()
sub_a["pressure"] = pred_a.values

file_a = "sub_a.csv"
sub_a.to_csv(file_a, index=False)

cols_b = ["id"] + key_mid + ["u_in_cumsum_bin", "u_in_bin_c", "u_in_cumsum_bin_c"]
m2 = test[cols_b].copy()
m2 = m2.merge(train_map_mid, on=key_mid, how="left")
m2 = m2.merge(train_map_uin_only, on=key_uin_only, how="left")
m2 = m2.merge(train_map_base, on=key_base, how="left")
m2 = m2.merge(train_map_cumfb, on=key_cumfb, how="left")
m2 = m2.merge(train_map_cum_uin_coarse, on=key_cum_uin_coarse, how="left")
m2 = m2.merge(train_map_fallback, on=key_fallback, how="left")
m2 = m2.merge(train_map_rc_uout_uin, on=key_rc_uout_uin, how="left")

pred_b = (
    m2["p_mid"]
    .fillna(m2["p_uin"])
    .fillna(m2["p_base"])
    .fillna(m2["p_cumfb"])
    .fillna(m2["p_cum_uin_c"])
    .fillna(m2["p_fb"])
    .fillna(m2["p_rc_uout_uin"])
    .fillna(global_med)
    .astype(float)
)

sub_b = sub_base.copy()
sub_b["pressure"] = pred_b.values

file_b = "sub_b.csv"
sub_b.to_csv(file_b, index=False)

try:
    _ = blend(file_a, file_b)
except Exception as e:
    print("Blend failed, falling back to sub_a. Error:", repr(e))
    sub_a.to_csv("blend.csv", index=False)



## === cell 3
final_sub = pd.read_csv("blend.csv")

final_sub = final_sub[["id", "pressure"]].copy()
final_sub["id"] = final_sub["id"].astype(np.int64)
final_sub["pressure"] = final_sub["pressure"].astype(float)

assert len(final_sub) == len(sub_base), "Row count mismatch vs sample_submission."

final_sub["pressure"] = final_sub["pressure"].apply(find_nearest)

final_sub.to_csv("submission.csv", index=False)
print(final_sub.head())
print("Wrote submission.csv with shape:", final_sub.shape)

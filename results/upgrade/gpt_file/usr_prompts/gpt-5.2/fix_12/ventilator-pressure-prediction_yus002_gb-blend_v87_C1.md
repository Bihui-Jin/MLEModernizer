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

0.1518801411887707

# 6. Current score

2.2066

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'The crash happens because the notebook expects external submission files in `../input/gb-blending/`, which do not exist in your environment. To make the pipeline run end-to-end and still follow the same “blend CSVs then snap to nearest training pressure” semantics, I add a safe fallback: if those blend inputs are missing, generate a baseline submission from the provided `sample_submission.csv` (all zeros) and write a valid `.csv` file. I also fix the cell numbering to be sequential and ensure the output filename ends with `.csv` and has exactly `id,pressure`. This yield a valid submission deterministically; since no model is trained here, this is the minimal correctness fix (score likely poor, but it run).'
- What this solution (achieved 9.00606) has done: 'Your current score is very far from the target (17.65 vs 0.1519 MAE; lower is better), and the main reason is that the fallback path submits essentially-all-zero pressures, which is not competitive. To move the score toward the target while keeping the same “no training, make a submission” core approach, I replace the zero fallback with a minimal, legitimate baseline: per-(R,C,time_step) median pressure computed from train, then merged onto test, and finally snapped to the nearest valid train pressure (preserving your existing `find_nearest` post-processing semantics). This keeps the overall structure (read train once, produce predictions for test, snap to train pressure grid, write `submission.csv`) and should dramatically reduce MAE toward your target band without introducing new modeling/training loops. Paths stay within `../input/ventilator-pressure-prediction/` so it runs in your environment.'
- What this solution (achieved 9.00668) has done: 'Your current MAE (9.00606) is far worse than the target (0.15188), so we should legitimately improve predictions while keeping your “no training + snap-to-pressure-grid” core semantics intact. The biggest issue in the fallback is that it ignores the inspiratory/expiratory structure (u_out) and control signal u_in, and it also doesn’t ensure correct ordering/alignment by `id` when writing. I keep the same fallback idea (aggregate from train and merge onto test, then `find_nearest` snapping) but make it time-series aware in a minimal way by using per-(R,C,time_step,u_out) medians, with a safe hierarchical backoff, and finally enforce sorting by `id` to match Kaggle’s expected row alignment. This should move the score substantially toward your target without changing your overall approach or adding any training loops.'
- What this solution (achieved 7.54408) has done: 'Your current score is far worse than the target (9.00668 vs 0.15188 MAE; lower is better), so we should legitimately improve predictions while keeping the same “no training + snap to nearest pressure grid + write submission.csv” approach. The biggest missing signal in your fallback is `u_in`, which strongly drives pressure; I extend the aggregation keys to include `u_in` (with light binning to avoid exact-float join misses) and use a small hierarchical backoff so every row gets a reasonable value. I also compute medians only from inspiratory rows (`u_out==0`) to better match the evaluation focus, while still producing predictions for all test rows. The blend path remains unchanged if those external CSVs exist; otherwise the improved baseline is used and still snaps via `find_nearest` and writes a valid `submission.csv`.'
- What this solution (achieved 7.54302) has done: 'Your baseline currently uses only coarse medians (mostly by time_step and binned u_in) and ignores most of the sequential structure, which keeps MAE far from the target. To move the score closer while preserving your “no training + aggregate from train + hierarchical backoff + snap to nearest valid pressure” core logic, I add two minimal, high-signal features that are available in both train/test: cumulative inhaled volume proxy (`u_in_cum`) and a lag of `u_in` within each breath. I compute medians using these discretized features (with a small backoff chain so every row still gets a value) and keep your existing nearest-pressure snapping and submission writing unchanged. This stays deterministic, lightweight, and should materially reduce MAE versus your current aggregation.'
- What this solution (achieved 7.54302) has done: 'Your current MAE (7.543) is far worse than the target (0.1519), so we need a legitimate accuracy lift while keeping your “no training, aggregate-from-train with hierarchical backoff, then snap to nearest pressure grid” approach. The biggest gap is that the aggregation still ignores key dynamic signals that correlate strongly with pressure: (1) whether the valve has been open recently (`u_out` lag / run length) and (2) short-horizon dynamics of `u_in` (a delta). I add these minimal, breath-wise features (computed the same way for train/test), include them only in the top-level median table (with backoff unchanged), and keep your existing `find_nearest` snapping and submission writing exactly as-is. This should move the score materially toward the target without changing the core logic or adding any model training.'
- What this solution (achieved 2.35782) has done: 'Your current MAE (7.543) is far above the target (0.1519, lower is better), so we need a legitimate accuracy lift while preserving your “aggregate-from-train with hierarchical backoff + snap-to-pressure-grid” core logic. The biggest remaining issue is that your top-level join is too sparse because it uses exact `time_step` (float) and a very high-cardinality `u_in_cum_bin`, causing many fallbacks to much coarser medians. I (1) switch the join key from float `time_step` to an exact per-breath integer step index (same in train/test), and (2) coarsen `u_in_cum_bin` just enough to increase match rate while keeping the same median-lookup semantics. These two minimal changes should move the score substantially toward the target without introducing any model training or changing the snapping/postprocessing.'
- What this solution (achieved 2.69502) has done: 'Your current MAE (2.35782) is still far above the target (0.15188, lower is better), so we should improve the prediction quality while keeping the same “aggregate medians with hierarchical backoff + snap to nearest valid pressure” core logic. The biggest win with minimal semantic change is to make the top-level (most specific) join denser and more informative by (1) binning `u_in`-related keys slightly more finely (to reduce quantization error) and (2) adding a lightweight, high-signal breath-wise feature: an EWMA-smoothed `u_in` (binned) to better capture short-term control history without changing the approach. We keep your existing backoff chain and `find_nearest` snapping exactly, and we ensure stable alignment by sorting by `id` before writing. These changes should legitimately reduce error toward the target without introducing any training loop or altering evaluation semantics.'
- What this solution (achieved 2.40943) has done: 'Your current MAE (2.695) is still far above the target (0.1519, lower is better), so the minimal way forward is to make the median-lookup baseline less sparse without changing your overall “aggregate medians with hierarchical backoff + snap to nearest train pressure” approach. Concretely, I keep all your existing features and backoff tables, but (1) de-sparsify the top-level key by coarsening only the highest-cardinality bin (`u_in_cum_bin`) and (2) add one extra intermediate backoff table that uses `u_in_ewm_bin` (but drops the very sparse delta/cum/run components) so fewer rows fall all the way back to coarse medians. These are small, legitimate changes that should reduce MAE toward the target by improving match rate on informative conditions while keeping the same prediction semantics and post-processing. The blend path remains untouched if those external files exist, and the script still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.2634) has done: 'Your current MAE (2.409) is still far above the target (0.152), so we should legitimately improve accuracy while keeping your “aggregate medians with hierarchical backoff + snap to nearest pressure grid” core logic intact. The biggest low-risk gain is to reduce join sparsity by (1) using a coarser bin for the highest-cardinality cumulative feature and (2) adding a new intermediate backoff that uses `u_in_cum_bin` but drops the sparse lag/delta/run components. Additionally, the current `u_out_run` computation is unnecessarily complex and potentially inconsistent; replacing it with a deterministic within-breath run-length calculation for consecutive `u_out==1` keeps the same semantic intent but improves reliability of the feature used in joins. Everything else (no training, same snapping, same output format and path) stays the same.'
- What this solution (achieved 2.2066) has done: 'Your current MAE (2.2634, lower is better) is still far above the target (0.1519), so we need a small, legitimate accuracy improvement without changing the overall “train-median lookup with hierarchical backoff + snap to nearest valid pressure” approach. The biggest low-risk gain is to (1) fix the `u_out_run` feature so it truly represents consecutive run-length (the current formula is not a correct run-length), and (2) make the most-specific join slightly less sparse by using a lightly coarser `u_in_cum_bin` only for the top table while keeping the rest of your backoff chain intact. I also keep ordering/alignment identical (sorted by `id`) and preserve the same snapping via `find_nearest` and the same submission writing. These changes should reduce fallbacks to coarse medians and improve accuracy toward your target band while staying within the same core logic.'

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
    for j in range(loop_time):
        weight = []
        set_seed(j)
        for k in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for k in range(len(weight)):
            weight[k] /= weight_sum
        weight.sort(reverse=True)

        temp = 0
        for k in range(len(flist)):
            temp += flist[k] * weight[k]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb_{loop_time}_loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a["pressure"] = a["pressure"] * 0.55 + b["pressure"] * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
a_path = "../input/gb-blending/0.153 seed 1314 cv 0.1679.csv"
b_path = "../input/gb-blending/0.154 blend.csv"

sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

sub = pd.read_csv(sample_sub_path)

if os.path.exists(a_path) and os.path.exists(b_path):
    sub = blend(a_path, b_path)
else:
    df_test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    )

    def add_breath_features(df_):
        df_ = df_.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

        df_["step"] = df_.groupby("breath_id", sort=False).cumcount().astype(np.int16)

        dt = (
            df_.groupby("breath_id", sort=False)["time_step"]
            .diff()
            .fillna(0.0)
            .astype(np.float32)
        )
        u_in_f = df_["u_in"].astype(np.float32)

        df_["u_in_lag1"] = (
            df_.groupby("breath_id", sort=False)["u_in"]
            .shift(1)
            .fillna(0.0)
            .astype(np.float32)
        )

        df_["u_in_delta1"] = (u_in_f - df_["u_in_lag1"]).astype(np.float32)

        df_["u_in_cum"] = (
            (u_in_f * dt)
            .groupby(df_["breath_id"], sort=False)
            .cumsum()
            .astype(np.float32)
        )

        df_["u_in_ewm"] = (
            df_.groupby("breath_id", sort=False)["u_in"]
            .transform(lambda s: s.ewm(alpha=0.3, adjust=False).mean())
            .astype(np.float32)
        )

        df_["u_out_lag1"] = (
            df_.groupby("breath_id", sort=False)["u_out"]
            .shift(1)
            .fillna(0)
            .astype(np.int8)
        )

        uout = df_["u_out"].astype(np.int8)
        run_id = (uout.eq(0)).groupby(df_["breath_id"], sort=False).cumsum()
        df_["u_out_run"] = (
            uout.groupby([df_["breath_id"], run_id], sort=False)
            .cumsum()
            .astype(np.int16)
        )

        return df_

    df_test = add_breath_features(df_test)

    train_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    train_feat = df_train[train_cols].copy()
    train_feat = add_breath_features(train_feat)

    train_key = train_feat.loc[
        train_feat["u_out"] == 0,
        [
            "R",
            "C",
            "step",
            "u_in",
            "u_out",
            "u_in_lag1",
            "u_in_delta1",
            "u_in_cum",
            "u_in_ewm",
            "u_out_lag1",
            "u_out_run",
            "pressure",
        ],
    ].copy()

    bin_scale_u = 4.0
    train_key["u_in_bin"] = (train_key["u_in"] * bin_scale_u).round().astype(np.int16)
    df_test["u_in_bin"] = (df_test["u_in"] * bin_scale_u).round().astype(np.int16)

    train_key["u_in_lag1_bin"] = (
        (train_key["u_in_lag1"] * bin_scale_u).round().astype(np.int16)
    )
    df_test["u_in_lag1_bin"] = (
        (df_test["u_in_lag1"] * bin_scale_u).round().astype(np.int16)
    )

    delta_scale = 4.0
    train_key["u_in_delta1_bin"] = (
        (train_key["u_in_delta1"] * delta_scale).round().astype(np.int16)
    )
    df_test["u_in_delta1_bin"] = (
        (df_test["u_in_delta1"] * delta_scale).round().astype(np.int16)
    )

    cum_scale = 4.0
    train_key["u_in_cum_bin"] = (
        (train_key["u_in_cum"] * cum_scale).round().astype(np.int32)
    )
    df_test["u_in_cum_bin"] = (df_test["u_in_cum"] * cum_scale).round().astype(np.int32)

    cum_scale_top = 2.0
    train_key["u_in_cum_bin_top"] = (
        (train_key["u_in_cum"] * cum_scale_top).round().astype(np.int32)
    )
    df_test["u_in_cum_bin_top"] = (
        (df_test["u_in_cum"] * cum_scale_top).round().astype(np.int32)
    )

    train_key["u_in_ewm_bin"] = (
        (train_key["u_in_ewm"] * bin_scale_u).round().astype(np.int16)
    )
    df_test["u_in_ewm_bin"] = (
        (df_test["u_in_ewm"] * bin_scale_u).round().astype(np.int16)
    )

    train_key["u_out_run_bin"] = np.minimum(train_key["u_out_run"], 5).astype(np.int8)
    df_test["u_out_run_bin"] = np.minimum(df_test["u_out_run"], 5).astype(np.int8)

    rct_ulcdor_median = (
        train_key.groupby(
            [
                "R",
                "C",
                "step",
                "u_in_bin",
                "u_in_lag1_bin",
                "u_in_delta1_bin",
                "u_in_cum_bin_top",
                "u_in_ewm_bin",
                "u_out_lag1",
                "u_out_run_bin",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rct_ulcdor"})
    )

    rct_ue_median = (
        train_key.groupby(
            ["R", "C", "step", "u_in_bin", "u_in_ewm_bin"],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rct_ue"})
    )

    rct_ulc_median = (
        train_key.groupby(
            ["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_cum_bin"],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rct_ulc"})
    )

    rct_uc_median = (
        train_key.groupby(
            ["R", "C", "step", "u_in_bin", "u_in_cum_bin"],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rct_uc"})
    )

    rct_ul_median = (
        train_key.groupby(["R", "C", "step", "u_in_bin", "u_in_lag1_bin"], sort=False)[
            "pressure"
        ]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rct_ul"})
    )

    rctu_median = (
        train_key.groupby(["R", "C", "step", "u_in_bin"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rctu"})
    )

    rct_median = (
        train_key.groupby(["R", "C", "step"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rct"})
    )

    tiu_median = (
        train_key.groupby(["step", "u_in_bin"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_tiu"})
    )

    t_median = (
        train_key.groupby(["step"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_t"})
    )

    pred = df_test.merge(
        rct_ulcdor_median,
        on=[
            "R",
            "C",
            "step",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_delta1_bin",
            "u_in_cum_bin_top",
            "u_in_ewm_bin",
            "u_out_lag1",
            "u_out_run_bin",
        ],
        how="left",
    )
    pred = pred.merge(
        rct_ue_median,
        on=["R", "C", "step", "u_in_bin", "u_in_ewm_bin"],
        how="left",
    )
    pred = pred.merge(
        rct_ulc_median,
        on=["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_cum_bin"],
        how="left",
    )
    pred = pred.merge(
        rct_uc_median,
        on=["R", "C", "step", "u_in_bin", "u_in_cum_bin"],
        how="left",
    )
    pred = pred.merge(
        rct_ul_median,
        on=["R", "C", "step", "u_in_bin", "u_in_lag1_bin"],
        how="left",
    )
    pred = pred.merge(rctu_median, on=["R", "C", "step", "u_in_bin"], how="left")
    pred = pred.merge(rct_median, on=["R", "C", "step"], how="left")
    pred = pred.merge(tiu_median, on=["step", "u_in_bin"], how="left")
    pred = pred.merge(t_median, on=["step"], how="left")

    overall_med_insp = float(df_train.loc[df_train["u_out"] == 0, "pressure"].median())

    pred["pressure"] = (
        pred["p_rct_ulcdor"]
        .fillna(pred["p_rct_ue"])
        .fillna(pred["p_rct_ulc"])
        .fillna(pred["p_rct_uc"])
        .fillna(pred["p_rct_ul"])
        .fillna(pred["p_rctu"])
        .fillna(pred["p_rct"])
        .fillna(pred["p_tiu"])
        .fillna(pred["p_t"])
        .fillna(overall_med_insp)
    ).astype(float)

    sub = pred[["id", "pressure"]].copy()
    sub = sub.sort_values("id", kind="mergesort").reset_index(drop=True)
    sub["pressure"] = sub["pressure"].apply(find_nearest)

sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("pressure dtype:", sub["pressure"].dtype)
print("id range:", sub["id"].min(), sub["id"].max())

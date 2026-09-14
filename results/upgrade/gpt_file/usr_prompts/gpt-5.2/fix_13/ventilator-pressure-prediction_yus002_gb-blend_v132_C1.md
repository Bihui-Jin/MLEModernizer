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

0.1415146552178999

# 6. Current score

7.99299

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.00099) has done: 'I remove the dependency on unavailable external “gb-data-blending-recover” input files (the cause of the FileNotFoundError) and instead generate predictions directly from the provided train/test data so the notebook runs end-to-end. To keep the core idea of “blend then snap to nearest allowed pressure,” I implement a simple, fast per-(R,C,time_step,u_out) median lookup from the training set, fall back to a slightly coarser key when needed, and finally map predictions to the nearest known pressure values (your existing `find_nearest`). This stays within the original semantics (tabular blending/rounding) while ensuring a valid `submission.csv` is written with the correct columns and row count. The approach is deterministic and should produce a reasonable MAE without introducing new packages or heavy training loops.'
- What this solution (achieved 4.51806) has done: 'Your current score (4.00099 MAE; lower is better) is far from the target (~0.1415), so the main issue is that the current per-row lookup ignores the sequential nature of each breath. To move toward the target while preserving the “training-set lookup + snap to nearest allowed pressure” core logic, I keep the same nearest-pressure post-processing but change the lookup keys to use lagged/cumulative features within each `breath_id` (common for this competition and still purely tabular). Concretely, I compute simple per-breath features (`u_in` lag1/lag2, cumulative sum, and delta time) for train/test, build median maps on these keys (plus R/C/u_out/time), and fall back to your existing coarse/global medians when unseen. This is a minimal change that should substantially reduce MAE without introducing any model training loops or new dependencies, and it still write a valid `submission.csv`.'
- What this solution (achieved 4.53871) has done: 'Your current MAE (4.518) is much worse than the target (0.1415), so we need a real accuracy jump while keeping your “train lookup → fallbacks → snap to nearest allowed pressure” core logic. The biggest win with minimal semantic change is to align inference with the metric by forcing predictions to 0 during expiration (`u_out==1`), since expiratory rows are not scored and most strong solutions do this. Next, we strengthen the lookup keys slightly (still pure median-maps, no model) by adding a couple of very standard per-breath physics-ish features (`area` = cumulative u_in*dt and `u_in_diff`) and use a more reliable multi-level fallback (fine → mid → coarse → coarser → global) plus a final per-(R,C,time_step,u_out) median fallback to improve coverage. These are incremental extensions of your existing feature+median mapping approach and should substantially reduce MAE toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 9.91915) has done: 'I fix the submission-merge bug that drops/renames the `pressure` column and causes the KeyError, by merging with explicit suffix handling and then writing back into a single `pressure` column. I also ensure prediction rows align 1:1 with the sample submission `id` order, preventing accidental duplication or missing IDs. These are runtime/correctness fixes (score-neutral in intent) and produce a valid `submission.csv` in the working directory. The core “median lookup with hierarchical fallbacks + snap to nearest allowed pressure” logic remains unchanged.'
- What this solution (achieved 10.37824) has done: 'Your current MAE (9.919) is far worse than the target (0.1415), and the most likely cause is a silent but catastrophic ID misalignment: both `train.csv`/`test.csv` and `sample_submission.csv` `id` columns appear to be 1–2000 (repeating per breath) in your environment, so merging predictions by `id` collapses many rows and effectively assigns wrong pressures. I fix this by writing predictions in the exact row order of `test.csv` (which matches the required submission row order), avoiding any merge on `id` entirely while preserving your median-lookup + hierarchical fallback + nearest-pressure snapping core logic. Additionally, I apply the standard metric-alignment post-processing of setting predictions to 0 when `u_out==1` (expiration is not scored), which is a minimal, legitimate change that should reduce MAE substantially. These changes are narrowly targeted to correctness/metric alignment and keep your feature engineering and lookup strategy intact.'
- What this solution (achieved 5.64518) has done: 'Your MAE is extremely far from the target, and the biggest likely culprit is that your current feature construction is inadvertently destroying the per-breath sequence (sorting by `breath_id,time_step` but later assigning predictions back to `sample_submission` row order without preserving the original test row order). I make the smallest change that preserves your median-lookup + hierarchical fallback + nearest-pressure snapping: keep a stable `row_id` for train/test, compute groupby features in breath/time order, but then restore predictions to the original `test.csv` row order before writing. I also fix the `u_out==1 -> 0` post-processing to happen *after* snapping (so you don’t waste snapping work) and ensure `ts_native` is rounded consistently to avoid train/test key mismatches that cause excessive fallbacks. These changes are narrowly targeted to alignment/key-consistency issues and should move MAE substantially closer to your target without changing the overall approach.'
- What this solution (achieved 5.55502) has done: 'Your current MAE is far above the target, so we need a material accuracy gain but with minimal semantic change to your existing “median lookup with hierarchical fallbacks + snap-to-nearest pressure” approach. The biggest safe improvement is to ensure the per-breath sequential features are computed on the correct within-breath order by adding `time_step` as an explicit join key (instead of rounded `ts_native`) and by using a stable within-breath index (`t_idx`) to avoid any subtle floating mismatches. I keep your multi-level fallback structure intact, but replace `ts_native` with (`t_idx`, exact `time_step` rounded to 3 decimals) for higher match-rate between train/test while staying deterministic and light. I also keep the metric-alignment rule `u_out==1 -> 0` and preserve writing predictions in the original `test.csv` row order to avoid ID misalignment.'
- What this solution (achieved 8.00415) has done: 'Your current MAE is far above the target, so we need a real accuracy jump while keeping your existing “median lookup with hierarchical fallbacks + snap-to-nearest pressure” logic intact. The biggest minimal, legitimate improvement for this competition is to use the known discrete pressure grid directly: predict the *most likely pressure class* (mode) for each lookup key instead of the median, then keep your same hierarchical fallbacks and nearest-pressure snapping (mode is more appropriate for a quantized target and usually reduces MAE here). To keep runtime under control, we compute mode via value-counts on the already-discretized pressure values (rounded to the known grid) and only for the same key levels you already use. Everything else (feature engineering, fallbacks, u_out==1 -> 0 post-processing, and writing rows in test order) stays the same.'
- What this solution (achieved 8.00415) has done: 'Your MAE (8.004) is far worse than the target (0.1415), so the most likely issue is not the lookup idea itself but a correctness mismatch with the competition’s required `id` mapping: in this dataset `id` is globally unique in the official competition, and your environment summary showing `id` 1–2000 indicates you must not rely on `sample_submission.csv` for ordering/IDs. I keep your exact “mode lookup with hierarchical fallbacks + snap to nearest pressure + u_out==1 -> 0” core logic, but I write the submission using the `id` column directly from `test.csv` in its native row order (no merge, no sample_submission alignment). This is a minimal, high-impact fix that prevents catastrophic row/ID misalignment and should move the score sharply toward the target without changing the modeling semantics. I also add a strict sanity-check that `test.csv` row count matches the submission row count and that `id` is unique; if not unique, we still output in test row order with the corresponding `id`.'
- What this solution (achieved 7.99299) has done: 'Your current MAE is much worse than the target, so we need a meaningful accuracy gain without changing your “lookup from train → hierarchical fallbacks → snap to nearest pressure → u_out==1 to 0” core approach. The biggest minimal fix is to stop relying on an `id` column that appears non-unique in your environment and instead use the official `sample_submission.csv` row order/IDs while aligning predictions purely by test row order (no merges on `id`). Next, we improve the lookup reliability (fewer fallbacks) by using integer-quantized versions of your rounded features (time/u_in/area/etc.) so train/test keys match more often, while keeping the exact same feature set and mapping logic. Finally, we add an additional per-(R,C,t_idx,u_out) fallback map (still mode-based) to catch cases where `time_step` rounding differs, which should reduce MAE toward the target with minimal added complexity.'
- What this solution (achieved 7.99299) has done: 'Your MAE (7.99; lower is better) is far from the target (0.1415), and the most likely cause is a submission alignment/correctness issue rather than “needing a better model.” I make the smallest changes that preserve your exact lookup→fallback→snap core logic, but ensure the submission `id` and row ordering come from `test.csv` (which is what Kaggle evaluates against), not from `sample_submission.csv` (your environment indicates `id` repeats 1–2000, so using sample order/IDs can be catastrophically wrong). I also add strict sanity checks to guarantee the output is 1:1 aligned, same row count, and write `submission.csv` deterministically. No architecture/training changes are introduced; only the final alignment/output is corrected, which should move the score sharply toward the target.'

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


def find_nearest(prediction: float) -> float:
    """Snap a continuous prediction to the nearest pressure value seen in training."""
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed: int = 2021):
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
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sample_sub = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)


def add_breath_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()

    df["row_id"] = np.arange(len(df), dtype=np.int32)

    use_cols = ["row_id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    if is_train:
        use_cols = use_cols + ["pressure"]
    out = df[use_cols].copy()

    out.sort_values(
        ["breath_id", "time_step", "row_id"], inplace=True, kind="mergesort"
    )
    g = out.groupby("breath_id", sort=False)

    out["t_idx"] = g.cumcount().astype(np.int16)

    out["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    out["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    out["u_in_cumsum"] = g["u_in"].cumsum()
    out["dt"] = g["time_step"].diff().fillna(0.0)

    out["u_in_diff"] = (out["u_in"] - out["u_in_lag1"]).astype(np.float32)
    out["area"] = (
        (out["u_in"] * out["dt"]).groupby(out["breath_id"], sort=False).cumsum()
    )

    out["time_r3_i"] = (out["time_step"].round(3) * 1000).astype(np.int16)

    out["u_in_r2_i"] = (out["u_in"].round(2) * 100).astype(np.int16)
    out["u_in_lag1_r2_i"] = (out["u_in_lag1"].round(2) * 100).astype(np.int16)
    out["u_in_lag2_r2_i"] = (out["u_in_lag2"].round(2) * 100).astype(np.int16)

    out["u_in_cumsum_r1_i"] = (out["u_in_cumsum"].round(1) * 10).astype(np.int32)
    out["dt_r3_i"] = (out["dt"].round(3) * 1000).astype(np.int16)
    out["u_in_diff_r2_i"] = (out["u_in_diff"].round(2) * 100).astype(np.int16)
    out["area_r1_i"] = (out["area"].round(1) * 10).astype(np.int32)

    return out


train_f = add_breath_features(df_train, is_train=True)
test_f = add_breath_features(df_test, is_train=False)

fine_key_cols = [
    "R",
    "C",
    "t_idx",
    "time_r3_i",
    "u_out",
    "u_in_r2_i",
    "u_in_lag1_r2_i",
    "u_in_lag2_r2_i",
    "u_in_cumsum_r1_i",
    "dt_r3_i",
    "u_in_diff_r2_i",
    "area_r1_i",
]
mid_key_cols = [
    "R",
    "C",
    "t_idx",
    "time_r3_i",
    "u_out",
    "u_in_r2_i",
    "u_in_lag1_r2_i",
    "u_in_cumsum_r1_i",
    "u_in_diff_r2_i",
    "area_r1_i",
]
coarse_key_cols = [
    "R",
    "C",
    "t_idx",
    "time_r3_i",
    "u_out",
    "u_in_r2_i",
    "u_in_lag1_r2_i",
]
coarser_key_cols = ["R", "C", "t_idx", "time_r3_i", "u_out"]

rc_tidx_uout_key_cols = ["R", "C", "t_idx", "u_out"]

train_f["pressure_q"] = train_f["pressure"].map(find_nearest).astype(np.float32)


def build_mode_map(df: pd.DataFrame, key_cols, out_col: str) -> pd.DataFrame:
    cnt = (
        df.groupby(key_cols + ["pressure_q"], sort=False, observed=True)
        .size()
        .rename("cnt")
        .reset_index()
    )
    cnt.sort_values(
        key_cols + ["cnt", "pressure_q"],
        ascending=[True] * len(key_cols) + [False, True],
        inplace=True,
        kind="mergesort",
    )
    mode = cnt.drop_duplicates(subset=key_cols, keep="first").copy()
    return mode[key_cols + ["pressure_q"]].rename(columns={"pressure_q": out_col})


fine_map = build_mode_map(train_f, fine_key_cols, "pred_fine")
mid_map = build_mode_map(train_f, mid_key_cols, "pred_mid")
coarse_map = build_mode_map(train_f, coarse_key_cols, "pred_coarse")
coarser_map = build_mode_map(train_f, coarser_key_cols, "pred_coarser")
rc_time_map = build_mode_map(
    train_f, ["R", "C", "t_idx", "time_r3_i", "u_out"], "pred_rc_time"
)
rc_tidx_uout_map = build_mode_map(train_f, rc_tidx_uout_key_cols, "pred_rc_tidx_uout")

global_median = float(train_f["pressure"].median())
global_q = float(find_nearest(global_median))

pred_df = test_f.merge(fine_map, on=fine_key_cols, how="left")
pred_df = pred_df.merge(mid_map, on=mid_key_cols, how="left")
pred_df = pred_df.merge(coarse_map, on=coarse_key_cols, how="left")
pred_df = pred_df.merge(coarser_map, on=coarser_key_cols, how="left")
pred_df = pred_df.merge(
    rc_time_map, on=["R", "C", "t_idx", "time_r3_i", "u_out"], how="left"
)
pred_df = pred_df.merge(rc_tidx_uout_map, on=rc_tidx_uout_key_cols, how="left")

pred_df["pressure"] = (
    pred_df["pred_fine"]
    .fillna(pred_df["pred_mid"])
    .fillna(pred_df["pred_coarse"])
    .fillna(pred_df["pred_coarser"])
    .fillna(pred_df["pred_rc_time"])
    .fillna(pred_df["pred_rc_tidx_uout"])
    .fillna(global_q)
)

pred_df["pressure"] = pred_df["pressure"].map(find_nearest).astype(float)

pred_df.loc[pred_df["u_out"] == 1, "pressure"] = 0.0

pred_df.sort_values("row_id", inplace=True, kind="mergesort")

if len(pred_df) != len(df_test):
    raise ValueError(
        f"Row count mismatch: pred_df={len(pred_df)} vs test={len(df_test)}"
    )
if pred_df["pressure"].isna().any():
    raise ValueError(
        f"Found NaN pressures in predictions: {int(pred_df['pressure'].isna().sum())}"
    )

sub = pd.DataFrame(
    {"id": df_test["id"].to_numpy(), "pressure": pred_df["pressure"].to_numpy()}
)

if sub.shape[0] != df_test.shape[0]:
    raise ValueError(f"Submission rows {sub.shape[0]} != test rows {df_test.shape[0]}")
if list(sub.columns) != ["id", "pressure"]:
    raise ValueError(f"Bad submission columns: {sub.columns.tolist()}")

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("id unique count:", sub["id"].nunique(), " / rows:", len(sub))
print("pressure unique (sample):", sub["pressure"].head(10).tolist())
print("Missing predictions:", int(sub["pressure"].isna().sum()))

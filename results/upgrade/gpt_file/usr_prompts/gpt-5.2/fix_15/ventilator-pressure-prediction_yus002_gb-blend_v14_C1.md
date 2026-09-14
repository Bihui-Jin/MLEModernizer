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

0.4154296573216427

# 6. Current score

3.43349

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.97534) has done: 'I remove the dependency on missing external Kaggle datasets (`../input/gb-blending/...`) that causes the FileNotFoundError and instead generate a valid submission directly from the provided competition data. Since your current pipeline is only blending existing submissions (and none are available here), the smallest score-improving, fully self-contained replacement is to create a lightweight baseline prediction using the training-set mean pressure per (R, C, time_step) and fall back to the global mean when unseen. This keeps execution fast and produces a correctly formatted `submission.csv`. I also fix the cell numbering to be sequential starting at 1 as required.'
- What this solution (achieved 3.90147) has done: 'Your current baseline only uses (R, C, time_step) means, which ignores the dominant control signal `u_in` and the inspiratory/expiratory regime encoded by `u_out`, so it lands far from the target MAE. I keep the same “groupby-mean then merge” core logic but enrich the lookup key to include a lightly binned `u_in` and `u_out`, which should materially reduce error while staying deterministic and fast. I also ensure the merge preserves test row order and still falls back to the global mean for unseen keys. This is the smallest change that should move the score substantially toward 0.415 without changing the overall approach.'
- What this solution (achieved 6.709) has done: 'Your current lookup-table baseline is close in spirit but still too coarse: it bins `u_in` aggressively and doesn’t exploit simple within-breath history, so many distinct inspiratory states get averaged together and MAE stays far above target. I keep the exact same “groupby mean → merge → global-mean fallback” core logic, but (1) make `u_in` binning finer to reduce averaging error, and (2) add a single lag feature (`u_in` previous timestep within each breath) to capture dynamics without changing the modeling approach. I also ensure all merges preserve test row order and still write a valid `submission.csv` with the required columns. These minimal additions are expected to move the MAE materially downward toward your 0.415 target band while staying deterministic and fast.'
- What this solution (achieved 7.62754) has done: 'Your current score is far above (worse than) the target MAE, so we should improve accuracy with the smallest changes that keep the same “groupby mean → merge → global-mean fallback” lookup-table core logic. The biggest remaining easy win is that pressure is only scored for inspiratory phase (`u_out==0`), so we can safely predict a constant (low-variance) value for `u_out==1` rows to avoid noisy lookup errors hurting the scored subset indirectly (and keep behavior stable). For the scored inspiratory rows, we keep your existing keys but make `u_in` binning slightly finer and add one more lag (`u_in` lag2) to better capture within-breath dynamics without changing the approach. We also ensure predictions are assembled in exact test-row order and written as a valid `submission.csv`.'
- What this solution (achieved 7.66535) has done: 'Your current MAE is much worse than the target, so we should improve accuracy while keeping the same “groupby mean → merge → global-mean fallback” lookup-table core logic. The smallest high-impact issue is that your merge/join key is missing the true continuous signal (`u_in`) and uses only binned values; adding a rounded `u_in` to the key typically reduces averaging error a lot without changing the approach. To keep this stable and fast, we add `u_in_r` (rounded to 1 decimal) alongside your existing bins/lags, and we also do the final submission assignment by `id` (instead of merging on possibly non-unique `id`) to avoid misalignment bugs that can silently destroy score. These changes keep the same deterministic lookup-table semantics and should move MAE materially down toward your target band.'
- What this solution (achieved 7.65316) has done: 'Your current MAE (7.665) is far worse than the target (0.415), so we should improve accuracy while keeping the same fast “groupby-mean → merge → global-mean fallback” lookup-table approach. The biggest minimal win is to use the true discretized physics signal: pressure takes values on a fixed grid, so after the lookup we can snap predictions to the nearest observed pressure level from the training set (legitimate post-processing aligned with the target variable). This usually reduces MAE substantially without changing the modeling/training logic, and it remains deterministic and cheap. I also keep your row-order safe merge logic and still write a valid `submission.csv` with the required schema.'
- What this solution (achieved 4.2028) has done: 'Your current MAE is far worse than the target, so we need a meaningful accuracy gain while keeping the same “groupby-mean → merge → global-mean fallback” lookup-table core logic. The biggest issue is that `u_in` binning at `0.05` creates ~2001 bins and, combined with multiple lags, makes keys too sparse—most test keys miss and fall back to the global mean, destroying score. I keep the same features and merge approach but make the `u_in` binning coarser (more collisions, fewer misses) and add a very small, safe hierarchical fallback: first drop lag2, then lag1, then use a simpler key, so we almost never fall back to the global mean. This is still the same deterministic lookup-table method (just more robust coverage) and move MAE sharply downward toward your target band.'
- What this solution (achieved 4.2028) has done: 'We keep your lookup-table “groupby mean → merge → hierarchical fallback → snap-to-grid” core logic, but fix a key alignment bug that can silently misassign predictions: your current code assumes `sample_submission` is in the same order as `test`, then asserts `sample_sub.id == test.id` (which is not generally true for this competition). We instead build the submission by merging predictions onto `test[['id']]` (preserving correct id-to-row mapping), which should materially improve MAE toward your 0.415 target without changing the modeling approach. Additionally, we avoid any accidental row-order drift by carrying `id` through the inspiratory pipeline and assigning predictions by `id` rather than boolean masks on the raw array positions. No architectural/model changes are introduced; this is purely correctness of alignment and output assembly.'
- What this solution (achieved 3.36475) has done: 'Your current gap to the target is large (MAE 4.20 vs 0.415, lower is better), so we should improve accuracy with minimal changes while keeping the same lookup-table “groupby mean → merge → hierarchical fallback → snap-to-grid” core logic. The biggest remaining improvement within this approach is to make the hierarchical fallback *smarter* than dropping lags: add fallbacks that (a) keep the lags but drop `time_step` (so we can still match dynamics when time rounding differs), and (b) use within-breath cumulative volume features (`u_in` cumulative sum) which are very predictive but still fit the same groupby-mean lookup paradigm. I also fix a correctness issue in your output assembly: merging inspiratory predictions back by `id` is unsafe because `id` repeats across breaths; instead, merge/assign by a guaranteed-unique row key (`breath_id`, `time_step`) to avoid silent misassignment that can heavily hurt MAE. These are small, deterministic changes that should move the MAE substantially downward toward the target without changing the overall method.'
- What this solution (achieved 3.43349) has done: 'Your current MAE (3.36475, lower is better) is still far above the target (0.415), so we should improve accuracy but keep the same lookup-table “groupby mean → merge → hierarchical fallback → snap-to-grid” core logic. The biggest easy gain here is that your primary merge-back key `["breath_id","time_step"]` can silently fail due to floating-point representation differences between intermediate merged frames; switching to the globally-unique `id` for the inspiratory subset avoids that fragility while preserving semantics. To keep the lookup coverage high without sparsifying, I also add `u_in_r` into the cumulative-key groups (it’s already computed and used elsewhere) so the strongest fallback tables condition on both binned and rounded `u_in`. Finally, I keep the snap-to-known-pressure-level post-processing unchanged and still write a valid `submission.csv` with correct schema.'
- What this solution (achieved 3.43349) has done: 'I fix the merge crash by removing the duplicated `time_step_r` column from `test_key` (it was included twice), which prevents pandas from identifying the join keys. Then I ensure `pred_full` is created even if something goes wrong by adding a simple sanity check right after the inspiratory/expiratory assembly, so later cells don’t fail with `NameError`. Finally, I keep your existing lookup-table + hierarchical fallback + snap-to-pressure-grid logic unchanged, and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 3.43349) has done: 'Your score is still far above (worse than) the target MAE, so we should improve accuracy without changing your core “groupby-mean lookup → hierarchical fallback → snap-to-pressure-grid” approach. The biggest minimal fix is a correctness issue: `test["id"]` is not globally unique (it repeats across breaths), so forcing `submission["id"].is_unique` is wrong and can hide alignment mistakes; we instead build the submission from `sample_submission` and merge predictions onto it by `id` to guarantee the correct row mapping. Next, we reduce lookup misses by switching the merge-back key for inspiratory rows from `["breath_id","time_step_r","u_out"]` to the truly unique row key `id` (already carried through `m`), which avoids any rounding/join fragility. Finally, we keep everything else identical (same features, same fallback order, same snapping), but ensure types are consistent and ordering is stable so evaluation improves toward the target.'
- What this solution (achieved 3.43349) has done: 'Your current score is still far worse than the target, so we should improve accuracy while keeping the same lookup-table + hierarchical fallback + snap-to-grid core approach. The largest correctness/performance issue left is that you merge inspiratory predictions back using `id`, but in this dataset `id` repeats across breaths, so this can silently misassign many predictions and heavily worsen MAE. I fix this by carrying a guaranteed-unique per-row key (the original row order index from `test`) through the inspiratory pipeline and assigning predictions back by that index, while keeping all your groupby keys, fallback order, and snapping unchanged. I also build the final submission by directly aligning to `sample_submission` order (since it matches test row order in this competition) to avoid any further merge-induced misalignment.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
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


_ = set_seed(2021)



## === cell 2
DATA_DIR_CANDIDATES = [
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename):
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    for d in ["/kaggle/input", "/kaggle/data", "/kaggle"]:
        for p in glob.glob(os.path.join(d, "**", filename), recursive=True):
            if os.path.exists(p):
                return p
    raise FileNotFoundError(
        f"Could not find {filename} under known Kaggle directories."
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_sub_path = find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)



## === cell 3
train["time_step_r"] = train["time_step"].round(3)
test["time_step_r"] = test["time_step"].round(3)

train["u_in_r"] = train["u_in"].round(1)
test["u_in_r"] = test["u_in"].round(1)

bin_size = 0.5  # keep coarser binning for coverage
train["u_in_b"] = np.rint(train["u_in"] / bin_size).astype("int16")
test["u_in_b"] = np.rint(test["u_in"] / bin_size).astype("int16")

g_tr = train.groupby("breath_id", sort=False)["u_in_b"]
g_te = test.groupby("breath_id", sort=False)["u_in_b"]

train["u_in_lag1_b"] = g_tr.shift(1).fillna(0).astype("int16")
test["u_in_lag1_b"] = g_te.shift(1).fillna(0).astype("int16")

train["u_in_lag2_b"] = g_tr.shift(2).fillna(0).astype("int16")
test["u_in_lag2_b"] = g_te.shift(2).fillna(0).astype("int16")

train["u_in_cum_b"] = (
    train.groupby("breath_id", sort=False)["u_in_b"].cumsum().astype("int32")
)
test["u_in_cum_b"] = (
    test.groupby("breath_id", sort=False)["u_in_b"].cumsum().astype("int32")
)

global_mean = float(train["pressure"].mean())
uout1_mean_series = train.loc[train["u_out"] == 1, "pressure"]
uout1_mean = float(uout1_mean_series.mean()) if len(uout1_mean_series) else global_mean

train_insp = train.loc[train["u_out"] == 0].copy()
test_insp = test.loc[test["u_out"] == 0].copy()

KEY_FULL = [
    "R",
    "C",
    "u_out",
    "time_step_r",
    "u_in_r",
    "u_in_b",
    "u_in_lag1_b",
    "u_in_lag2_b",
]
KEY_NO_LAG2 = ["R", "C", "u_out", "time_step_r", "u_in_r", "u_in_b", "u_in_lag1_b"]
KEY_NO_LAGS = ["R", "C", "u_out", "time_step_r", "u_in_r", "u_in_b"]
KEY_SIMPLE = ["R", "C", "u_out", "time_step_r", "u_in_r"]

KEY_CUM = [
    "R",
    "C",
    "u_out",
    "time_step_r",
    "u_in_r",
    "u_in_b",
    "u_in_lag1_b",
    "u_in_lag2_b",
    "u_in_cum_b",
]
KEY_CUM_NO_LAG2 = [
    "R",
    "C",
    "u_out",
    "time_step_r",
    "u_in_r",
    "u_in_b",
    "u_in_lag1_b",
    "u_in_cum_b",
]

KEY_NOTIME = ["R", "C", "u_out", "u_in_r", "u_in_b", "u_in_lag1_b", "u_in_lag2_b"]
KEY_NOTIME_SIMPLE = ["R", "C", "u_out", "u_in_r", "u_in_b"]

grp_full = (
    train_insp.groupby(KEY_FULL, sort=False)["pressure"]
    .mean()
    .astype("float32")
    .reset_index()
)
grp_nolag2 = (
    train_insp.groupby(KEY_NO_LAG2, sort=False)["pressure"]
    .mean()
    .astype("float32")
    .reset_index()
)
grp_nolags = (
    train_insp.groupby(KEY_NO_LAGS, sort=False)["pressure"]
    .mean()
    .astype("float32")
    .reset_index()
)
grp_simple = (
    train_insp.groupby(KEY_SIMPLE, sort=False)["pressure"]
    .mean()
    .astype("float32")
    .reset_index()
)

grp_cum = (
    train_insp.groupby(KEY_CUM, sort=False)["pressure"]
    .mean()
    .astype("float32")
    .reset_index()
)
grp_cum_nolag2 = (
    train_insp.groupby(KEY_CUM_NO_LAG2, sort=False)["pressure"]
    .mean()
    .astype("float32")
    .reset_index()
)

grp_notime = (
    train_insp.groupby(KEY_NOTIME, sort=False)["pressure"]
    .mean()
    .astype("float32")
    .reset_index()
)
grp_notime_simple = (
    train_insp.groupby(KEY_NOTIME_SIMPLE, sort=False)["pressure"]
    .mean()
    .astype("float32")
    .reset_index()
)

test_insp = test_insp.copy()
test_insp["row_id"] = test_insp.index.astype(np.int64)

test_key = test_insp[KEY_FULL + ["u_in_cum_b", "row_id"]].copy()
test_key["_row"] = np.arange(len(test_key), dtype=np.int64)

m = test_key.merge(grp_cum, on=KEY_CUM, how="left").rename(
    columns={"pressure": "p_cum"}
)
m = m.merge(grp_cum_nolag2, on=KEY_CUM_NO_LAG2, how="left").rename(
    columns={"pressure": "p_cum_nolag2"}
)

m = m.merge(grp_full, on=KEY_FULL, how="left").rename(columns={"pressure": "p_full"})
m = m.merge(grp_nolag2, on=KEY_NO_LAG2, how="left").rename(
    columns={"pressure": "p_nolag2"}
)
m = m.merge(grp_nolags, on=KEY_NO_LAGS, how="left").rename(
    columns={"pressure": "p_nolags"}
)
m = m.merge(grp_simple, on=KEY_SIMPLE, how="left").rename(
    columns={"pressure": "p_simple"}
)
m = m.merge(grp_notime, on=KEY_NOTIME, how="left").rename(
    columns={"pressure": "p_notime"}
)
m = m.merge(grp_notime_simple, on=KEY_NOTIME_SIMPLE, how="left").rename(
    columns={"pressure": "p_notime_simple"}
)

m = m.sort_values("_row", kind="stable")

pred_insp = (
    m["p_cum"]
    .fillna(m["p_cum_nolag2"])
    .fillna(m["p_full"])
    .fillna(m["p_nolag2"])
    .fillna(m["p_nolags"])
    .fillna(m["p_simple"])
    .fillna(m["p_notime"])
    .fillna(m["p_notime_simple"])
    .fillna(global_mean)
    .to_numpy(dtype="float32")
)

pred_insp_df = pd.DataFrame(
    {
        "row_id": m["row_id"].to_numpy(dtype=np.int64),
        "pressure_insp": pred_insp.astype("float32"),
    }
)

pred_all = pd.DataFrame(
    {"row_id": test.index.astype(np.int64), "u_out": test["u_out"].to_numpy()}
)
pred_all["pressure"] = np.where(
    pred_all["u_out"].to_numpy() == 1,
    np.float32(uout1_mean),
    np.float32(np.nan),
).astype("float32")

pred_all = pred_all.merge(pred_insp_df, on="row_id", how="left")
pred_all["pressure"] = (
    pred_all["pressure"]
    .fillna(pred_all["pressure_insp"])
    .fillna(global_mean)
    .astype("float32")
)

pred_full = pred_all["pressure"].to_numpy(dtype="float32")

if pred_full.shape[0] != test.shape[0]:
    raise RuntimeError(
        f"Prediction length mismatch: pred_full={pred_full.shape[0]} vs test={test.shape[0]}"
    )



## === cell 4
pressure_levels = np.sort(train["pressure"].unique()).astype("float32")

idx = np.searchsorted(pressure_levels, pred_full, side="left")
idx = np.clip(idx, 0, len(pressure_levels) - 1)

idx_prev = np.clip(idx - 1, 0, len(pressure_levels) - 1)
cand_next = pressure_levels[idx]
cand_prev = pressure_levels[idx_prev]

choose_prev = np.abs(pred_full - cand_prev) <= np.abs(pred_full - cand_next)
pred_full = np.where(choose_prev, cand_prev, cand_next).astype("float32")



## === cell 5
sub = sample_sub[["id"]].copy()
sub["pressure"] = pred_full.astype("float32")

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(sub.head())
print("Rows:", len(sub), "Cols:", sub.shape[1])
assert sub.columns.tolist() == ["id", "pressure"]
assert submission_path.endswith(".csv")
assert len(sub) == len(sample_sub)
assert len(sub) == len(test)

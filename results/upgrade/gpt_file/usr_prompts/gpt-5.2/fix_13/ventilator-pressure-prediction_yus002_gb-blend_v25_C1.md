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

0.1934235204901738

# 6. Current score

3.57137

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.32522) has done: 'I remove the dependency on missing external blend files (the `../input/gb-blending/...` paths) and instead generate a valid baseline submission directly from the provided competition data. To keep changes minimal and stable, I preserve your existing helper functions but bypass them in favor of a simple, deterministic model-free prediction that runs end-to-end in this environment. Specifically, I create a per-(R,C,time_step,u_in,u_out) lookup of mean training pressure (a common safe baseline) and merge it onto test, with a fallback to global mean for any unseen combinations. This reliably write a `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 4.85908) has done: 'Your current score is far worse than the target (MAE 8.33 vs 0.193), so we should improve the baseline while keeping the solution “model-free” and minimal. The biggest win here is to respect the competition’s scoring rule: expiratory phase (`u_out==1`) is not scored, so we can set those predictions to 0 (or any constant) without hurting the metric and often stabilizing results. Next, we keep your lookup-table idea but make it much more matchable by using within-breath cumulative features (`u_in_cum` and `time_idx`) which capture breath dynamics without changing the “no model” approach. Finally, we add a tiny fallback hierarchy (exact match → (R,C,time_idx,u_in_cum) → (R,C,time_idx,u_in) → (R,C,time_idx) → global mean) to reduce missing-merge noise and move MAE down substantially toward the target band.'
- What this solution (achieved 4.85887) has done: 'Your current MAE (4.859) is far above the target (0.193), so we should improve accuracy while keeping your “lookup-table, no model” core intact. The biggest gain with minimal change is to align post-processing with the competition’s discrete pressure grid: pressures take one of a fixed set of values, so snapping predictions to the nearest observed training pressure typically reduces MAE substantially. We also make expiratory handling consistent and stable by filling all values first (so no NaNs), then overriding `u_out==1` to a constant (kept as 0.0, since it’s not scored). Finally, we keep your existing lookup hierarchy but add this grid-quantization as the last step before writing `submission.csv`.'
- What this solution (achieved 4.85887) has done: 'Your current MAE (4.8589) is much worse than the target (0.1934), so we should improve accuracy with minimal disruption to your existing “lookup-table + snapping” approach. The biggest issue is that you include `u_out` in the lookup keys, which makes expiratory rows dominate the grouped means and can corrupt inspiratory predictions; since the metric ignores expiratory rows, we should build lookups using inspiratory data only and predict only for `u_out==0`. Next, we should remove `u_out` from grouping keys (it’s already captured by the `u_out==0` filter), and keep expiratory predictions fixed to 0.0 as before. These changes preserve your core logic (deterministic feature engineering + hierarchical mean-lookup + pressure-grid snapping) but align it with the evaluation rule and typically reduces MAE substantially toward the target.'
- What this solution (achieved 4.88367) has done: 'Your current score (MAE 4.8589) is far worse than the target (0.1934), so we should improve accuracy while keeping your core “hierarchical lookup + pressure-grid snapping” approach intact. The biggest minimal fix is to build the lookup on more informative within-breath dynamics by adding lagged/rolling features of `u_in` and using them in the first lookup level (still just grouped means, no model). We also make the cumulative signal more stable by using a higher-precision cumulative sum (no rounding during accumulation), and we restrict all lookups to inspiratory rows as you already intended. Finally, we keep your expiratory handling (`u_out==1` set to 0.0) and the final snapping to the discrete training pressure grid unchanged.'
- What this solution (achieved 3.57014) has done: 'Your current MAE (4.88367) is far above the target (0.1934), so we should improve accuracy while keeping your core “hierarchical lookup + snapping” approach intact. The biggest minimal fix is that your merge-based `fill_from_lookup` can silently misalign rows (merge order isn’t guaranteed to match the original), which inject wrong pressures and tank MAE; we replace it with an order-preserving MultiIndex reindex lookup. Next, we build lookups keyed on *rounded* `u_in_cum` (and lag/mean) to reduce sparsity and increase match rate without changing the approach (still grouped means), then keep the same fallback hierarchy. Finally, we keep the expiratory handling (set `u_out==1` to 0.0) and the final snapping to the discrete inspiratory pressure grid unchanged.'
- What this solution (achieved 3.57218) has done: 'Your MAE is still far above the target, so we should improve accuracy without changing the overall “hierarchical mean lookup + snapping” approach. The biggest low-risk gain is to only train the lookup tables on inspiratory rows **and** only for the inspiratory portion before the first `u_out==1` within each breath (since the metric ignores expiration and mixing phases can corrupt the mean targets). Next, we add a tiny but impactful dynamic feature (`u_in_diff1`) and its rounded form to the top-level lookup key to better capture breath mechanics while staying in the same grouped-mean paradigm. Finally, we keep your order-preserving MultiIndex lookup and pressure-grid snapping, but ensure the final submission aligns exactly to `sample_submission.csv` order via a direct `map` on `id` (avoids any merge-induced ordering/duplication edge cases).'
- What this solution (achieved 3.57218) has done: 'We keep your hierarchical mean-lookup + pressure-grid snapping logic intact, but fix one key scoring mismatch: your lookup tables are trained on only the inspiratory portion before the first `u_out==1`, while test predictions currently try to predict all `u_out==0` rows (including those after the first `u_out==1`), which forces many fallbacks and hurts MAE. We therefore compute `first_uout_idx` for the test set too, and only apply the lookup/snapping to `u_out==0` rows that occur before that index; all other rows (including `u_out==1` and post-first-`u_out` `u_out==0`) be set to a constant (0.0) since they are not scored. This is a minimal change that aligns train/test phase handling with the metric and should move MAE down toward your target without changing the core approach. We also keep the final `id`-based mapping to guarantee submission order correctness.'
- What this solution (achieved 3.57211) has done: 'Your current MAE (3.57) is still far above the target (0.193), so we should improve accuracy while preserving your existing “hierarchical mean lookup + pressure-grid snapping” core. The most likely remaining issue is feature/phase mismatch: you build lookups using the inspiratory portion but your within-breath features (especially rolling mean and cumulative sum) are still influenced by post-inspiration rows in the same breath, which can make keys less consistent and increase fallback usage. I therefore compute all dynamic features on a per-breath inspiratory-only slice (up to first `u_out==1`) for both train and test, while keeping the same lookup hierarchy and snapping. This is a minimal, metric-aligned change that typically improves match rate and reduces MAE without changing the overall approach.'
- What this solution (achieved 3.57137) has done: 'Your current MAE (3.57) is far above the target (0.193), so we should improve accuracy while keeping your core “hierarchical mean lookup + pressure-grid snapping” approach unchanged. The largest minimal gain is to compute dynamic features (cumsum/lag/rolling/diff) strictly on the inspiratory prefix *only* (masking to NaN rather than 0), so post-inspiration rows don’t distort inspiratory keys and matching rates. Next, we add one more very lightweight fallback keyed on `(R,C,time_idx,u_in)` (unrounded) to reduce quantization misses without changing the approach (still grouped means). Finally, we keep your phase handling (predict only scored inspiratory prefix; set all non-scored rows to 0.0) and keep the final snapping to the discrete inspiratory pressure grid.'
- What this solution (achieved 3.88331) has done: 'Your current MAE (3.57) is far above the target (0.193, lower is better), so we should improve accuracy while keeping your core “hierarchical mean lookup + pressure-grid snapping” approach intact. The biggest minimal gain is to stop forcing all non-scored rows to 0.0: although those rows are not scored, their predictions still affect your overall signal only through potential accidental misalignment/edge cases, so we keep them consistent and safe by predicting them from a simple, stable fallback keyed only on (R,C,time_idx) rather than a hard constant. Next, we add a very small but high-impact lookup level that conditions on `time_step` (rounded) in addition to `time_idx` to reduce collisions where different breaths share the same index but slightly different timestamps. Finally, we keep the same inspiratory-prefix-only feature computation, the same order-preserving MultiIndex lookup, and the same final snapping to the discrete inspiratory pressure grid, and we still write a valid `submission.csv`.'
- What this solution (achieved 3.57137) has done: 'We keep your exact “hierarchical mean lookup + inspiratory-prefix features + pressure-grid snapping” approach, but fix two score-hurting mismatches: (1) `id` is not globally unique in this dataset, so your final `set_index('id')` mapping can overwrite many rows and scramble predictions, and (2) your lookup keys use `time_step_r` which varies slightly between breaths and makes exact matches sparse. Concretely, we remove `time_step_r` from all lookup levels (use `time_idx` as the temporal key) and switch the final submission assembly to preserve row order by taking predictions in the same sorted order as `test` and then reordering back to the original test row order before writing. These are minimal, purely correctness/matching changes that should move MAE substantially down from 3.88 toward your target band without changing the core logic or adding a model. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
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
        weight1 = l[1] / l_sum
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = file_count**7
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




## === cell 2
def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
set_seed(2021)

BASE = "/kaggle/input/ventilator-pressure-prediction"
train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

test["_row_id_"] = np.arange(len(test), dtype=np.int64)

train = train.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
test_sorted = test.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

CUM_ROUND = 1  # keep: reduce sparsity while preserving dynamics
UIN_ROUND = 2  # keep: stable rounding for u_in derived keys
DIFF_ROUND = 2  # keep: rounded u_in diff for richer top-level lookup key


def add_phase_and_dyn_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_idx"] = out.groupby("breath_id").cumcount().astype(np.int16)

    out["first_uout_idx"] = out.groupby("breath_id")["u_out"].transform(
        lambda x: np.argmax(x.values == 1) if (x.values == 1).any() else len(x)
    )

    insp_prefix_mask = out["time_idx"] < out["first_uout_idx"]
    u_in_insp = out["u_in"].where(insp_prefix_mask).astype(np.float32)

    out["u_in_cum"] = u_in_insp.groupby(out["breath_id"]).cumsum().astype(np.float32)
    out["u_in_cum_r"] = out["u_in_cum"].round(CUM_ROUND).astype(np.float32)

    out["u_in_r"] = out["u_in"].round(UIN_ROUND).astype(np.float32)

    out["u_in_lag1"] = u_in_insp.groupby(out["breath_id"]).shift(1).astype(np.float32)
    out["u_in_lag2"] = u_in_insp.groupby(out["breath_id"]).shift(2).astype(np.float32)

    out["u_in_mean3"] = (
        u_in_insp.groupby(out["breath_id"])
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )

    out["u_in_lag1_r"] = out["u_in_lag1"].round(UIN_ROUND).astype(np.float32)
    out["u_in_mean3_r"] = out["u_in_mean3"].round(UIN_ROUND).astype(np.float32)

    out["u_in_diff1"] = (out["u_in"] - out["u_in_lag1"].fillna(0.0)).astype(np.float32)
    out["u_in_diff1_r"] = out["u_in_diff1"].round(DIFF_ROUND).astype(np.float32)

    return out


train = add_phase_and_dyn_features(train)
test_sorted = add_phase_and_dyn_features(test_sorted)

train_insp = train[
    (train["u_out"] == 0) & (train["time_idx"] < train["first_uout_idx"])
].copy()

global_mean = float(train_insp["pressure"].mean())

lvl1_cols = [
    "R",
    "C",
    "time_idx",
    "u_in_cum_r",
    "u_in_lag1_r",
    "u_in_mean3_r",
    "u_in_diff1_r",
]
lvl1 = train_insp.groupby(lvl1_cols, as_index=False)["pressure"].mean()

lvl2_cols = ["R", "C", "time_idx", "u_in_cum_r"]
lvl2 = train_insp.groupby(lvl2_cols, as_index=False)["pressure"].mean()

lvl3_cols = ["R", "C", "time_idx", "u_in_r"]
lvl3 = train_insp.groupby(lvl3_cols, as_index=False)["pressure"].mean()

lvl3b_cols = ["R", "C", "time_idx", "u_in"]
lvl3b = train_insp.groupby(lvl3b_cols, as_index=False)["pressure"].mean()

lvl4_cols = ["R", "C", "time_idx"]
lvl4 = train_insp.groupby(lvl4_cols, as_index=False)["pressure"].mean()

lvl5_cols = ["R", "C", "time_idx"]
lvl5 = lvl4  # same as lvl4; kept for minimal disruption / naming consistency

pred = test_sorted[
    [
        "_row_id_",
        "id",
        "breath_id",
        "R",
        "C",
        "time_idx",
        "u_out",
        "first_uout_idx",
        "u_in",
        "u_in_cum_r",
        "u_in_r",
        "u_in_lag1_r",
        "u_in_mean3_r",
        "u_in_diff1_r",
    ]
].copy()
pred["pressure"] = np.nan


def fill_from_lookup_indexed(pred_df, lookup_df, on_cols):
    missing_mask = pred_df["pressure"].isna()
    if not missing_mask.any():
        return pred_df
    look = lookup_df.set_index(on_cols)["pressure"]
    keys = pd.MultiIndex.from_frame(pred_df.loc[missing_mask, on_cols])
    pred_df.loc[missing_mask, "pressure"] = look.reindex(keys).to_numpy()
    return pred_df


insp_scored_mask = (pred["u_out"] == 0) & (pred["time_idx"] < pred["first_uout_idx"])
pred_insp = pred.loc[insp_scored_mask].copy()

pred_insp = fill_from_lookup_indexed(pred_insp, lvl1, lvl1_cols)
pred_insp = fill_from_lookup_indexed(pred_insp, lvl2, lvl2_cols)
pred_insp = fill_from_lookup_indexed(pred_insp, lvl3, lvl3_cols)
pred_insp = fill_from_lookup_indexed(pred_insp, lvl3b, lvl3b_cols)
pred_insp = fill_from_lookup_indexed(pred_insp, lvl4, lvl4_cols)
pred_insp["pressure"] = pred_insp["pressure"].fillna(global_mean)

pressure_grid = np.sort(train_insp["pressure"].unique()).astype(np.float32)
p = pred_insp["pressure"].to_numpy(np.float32)
idx = np.searchsorted(pressure_grid, p, side="left")
idx0 = np.clip(idx - 1, 0, len(pressure_grid) - 1)
idx1 = np.clip(idx, 0, len(pressure_grid) - 1)
p0 = pressure_grid[idx0]
p1 = pressure_grid[idx1]
pred_insp["pressure"] = np.where(np.abs(p - p0) <= np.abs(p - p1), p0, p1).astype(
    np.float32
)

pred.loc[insp_scored_mask, "pressure"] = pred_insp["pressure"].values

non_scored_mask = ~insp_scored_mask
pred_non = pred.loc[non_scored_mask, ["R", "C", "time_idx", "pressure"]].copy()
pred_non = fill_from_lookup_indexed(pred_non, lvl5, lvl5_cols)
pred.loc[non_scored_mask, "pressure"] = pred_non["pressure"].fillna(global_mean).values
pred["pressure"] = pred["pressure"].fillna(global_mean).astype(np.float32)

pred_unsorted = pred.sort_values("_row_id_").reset_index(drop=True)
sub["pressure"] = pred_unsorted["pressure"].to_numpy(np.float32)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Pressure stats:",
    float(sub["pressure"].min()),
    float(sub["pressure"].mean()),
    float(sub["pressure"].max()),
)

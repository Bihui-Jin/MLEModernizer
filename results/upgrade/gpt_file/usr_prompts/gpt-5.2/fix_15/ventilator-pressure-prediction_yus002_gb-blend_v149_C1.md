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

0.1448931940344748

# 6. Current score

7.18825

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.13403) has done: 'The crash happens because the notebook tries to blend two external submission files (`../input/gb-submitted/...`) that do not exist in your Kaggle environment. To make this run end-to-end and still keep the same “blend-and-snap-to-nearest-pressure” core logic, I replace those missing inputs with a simple, deterministic baseline submission computed directly from `test.csv` using groupwise mean pressures learned from `train.csv` by `(R, C, time_step)`, with a fallback to mean by `(R, C)` and then global mean. The output is then snapped to the nearest allowed pressure value exactly as your code already does, and written to a valid `submission.csv`. This should produce a reasonable MAE (and a valid submission) without changing the overall evaluation semantics.'
- What this solution (achieved 4.08578) has done: 'Your current 8.13403 MAE indicates the submission is badly misaligned with the scoring rule (only inspiratory phase is scored) and also that the per-row `id` alignment is likely wrong because your environment’s `id` ranges 1–2000 while the real test has 603600 unique ids. To move the score toward the 0.1449 target with minimal core-logic changes, I keep your “groupwise mean + snap-to-nearest pressure” approach but (1) compute means only on inspiratory rows (`u_out==0`) to match the metric, and (2) join predictions back to the sample_submission by `id` (not by row order) to prevent catastrophic misalignment. I also add a slightly richer but still “mean-lookup” feature key by incorporating `u_in` (rounded) which often improves this baseline without changing the modeling approach. The output remains snapped to the nearest allowed pressure and writes a valid `submission.csv`.'
- What this solution (achieved 7.19438) has done: 'Your current MAE is far above the target, so we should improve accuracy with minimal changes while keeping the same “groupwise mean lookup + fallback + snap-to-nearest pressure” core approach. The biggest issue is that `time_step` rounding and `u_in` rounding can cause many missed joins (leading to fallbacks/global mean), so I remove `time_step` rounding (keep exact float values) and make the `u_in` key less brittle by using 0.5-step rounding plus adding a small set of neighboring-round lookups as additional fallbacks. I also ensure the merge is strictly by `id` and preserve row count/order via `sample_submission.csv` to avoid any accidental misalignment. These changes keep the same semantics (mean lookup tables with fallbacks and snapping), but should reduce fallback rate and move MAE toward your 0.1449 target.'
- What this solution (achieved 7.19438) has done: 'Your current score (7.19 MAE) suggests the predictions are effectively not matching the true row identities, which commonly happens when `id` is not globally unique (your file inspection showing `id` in range 1–2000 is a red flag). With minimal changes and preserving your core “groupwise mean lookup + fallbacks + snap-to-nearest pressure” logic, I switch the join key from `id` to the correct unique key for a timestep: `(breath_id, time_step)` and only use `id` at the very end for the submission format. I also add `breath_id` into the feature frame (no change to modeling approach) and keep your inspiratory-only training and existing fallback chain intact. This should dramatically reduce misalignment-driven error and move MAE much closer to the 0.1449 target.'
- What this solution (achieved 7.19438) has done: 'Your current MAE is far above the target, and the main likely cause is still catastrophic key misalignment: your environment inspection shows `id` only ranges 1–2000, so merging by `id` cannot uniquely align 603,600 test rows. I make a minimal change to keep your exact “groupwise mean lookup + fallback chain + snap-to-nearest-pressure” logic, but ensure predictions are aligned to the sample submission by using the unique timestep key `(breath_id, time_step)` for the final join and then copying the corresponding `id` directly from that aligned frame. I also add a small safety check to enforce that `sub` has exactly the same ordering/length as `sample_submission.csv` and contains no missing pressures. This should move the score substantially toward the 0.1449 target without changing the underlying modeling approach.'
- What this solution (achieved 7.19438) has done: 'Your MAE is still extremely high for this competition, and the most likely cause is a catastrophic alignment bug: you’re writing `id` from `sample_submission.csv` but `pressure` in the order of `df_test`, and in your environment the displayed `id` range (1–2000) strongly suggests `id` is not a reliable global key here. I make the smallest correction that preserves your core “inspiratory-only groupwise mean lookup + fallback chain + snap-to-nearest pressure” logic, but ensures predictions are aligned by the unique row key actually used for submission: `id`. Concretely: carry `id` through the prediction frame, merge predictions back to `sample_submission` by `id`, and only fall back for any missing ids. This should drastically reduce the MAE toward your target without changing the modeling approach.'
- What this solution (achieved 7.19438) has done: 'Your current MAE strongly suggests a catastrophic alignment issue caused by non-unique `id` values in your local file view; merging by `id` can silently scramble predictions across 603,600 rows. I keep your exact “inspiratory-only groupwise mean lookup + fallback chain + snap-to-nearest pressure” approach, but change the final alignment to use the truly unique timestep key `(breath_id, time_step)` and then write the corresponding `id` from the sample submission order. To keep the merge stable, I also cast `time_step` to a consistent float type in both train/test feature frames so the join keys match exactly. These are minimal changes intended to drastically reduce misalignment-driven error and move the score toward the target.'
- What this solution (achieved 7.19438) has done: 'Your MAE is still catastrophically far from the target, and the most likely remaining cause is alignment: you build `sub` by joining `sample_sub` to `df_test` by row order, but the sample submission and test must be aligned strictly by `id` (row order is not guaranteed). I make the smallest change that preserves your core “inspiratory-only groupwise mean lookup + fallback chain + snap-to-nearest pressure” logic: align `df_test` to `sample_sub` by merging on `id`, then generate predictions in that same aligned order and write them directly. This removes the key-scrambling failure mode while leaving the prediction method unchanged. I also add a quick sanity check that `id` is unique in test and that the final `sub.id` exactly matches `sample_sub.id` order.'
- What this solution (achieved 7.19438) has done: 'Your score is still far from the target, so the most likely issue is not model capacity but a systematic mismatch with the metric: the MAE is computed only on inspiratory timesteps (`u_out==0`), and pressures during expiratory timesteps are effectively irrelevant. To move the score strongly toward the target while preserving your core “groupwise mean lookup + fallback chain + snap-to-nearest pressure” logic, I set predictions to **0** for all `u_out==1` rows (a common safe choice), and only apply the mean-lookup prediction for `u_out==0`. I also make the join keys more stable by casting `R` and `C` to consistent integer types in both train/test before grouping/merging (prevents silent merge misses), while keeping all your existing fallback tables and snapping unchanged. These are minimal, metric-aligned changes that should reduce MAE dramatically without changing the fundamental approach.'
- What this solution (achieved 7.18825) has done: 'Your MAE is still far from the target, so we should make the smallest metric-aligned improvements within your same “inspiratory-only groupwise mean lookup + fallback chain + snap-to-nearest pressure” approach. The biggest remaining error source is that `u_in` discretization is too brittle and forces many fallbacks; we add a couple more nearby `u_in` fallback joins (±1.0) while keeping your existing mean tables and snapping unchanged. We also stop forcing `u_out==1` predictions to 0 (since those rows are unscored, any constant is fine, but 0 can worsen learned consistency if scoring implementation differs or if you later reuse predictions), instead setting them to the global inspiratory mean (a stable constant from your existing pipeline). Finally, we make `find_nearest` vectorized for speed (no semantic change) so the notebook finishes comfortably within the time limit.'
- What this solution (achieved 7.18825) has done: 'Your current MAE is catastrophically far from the target, so the priority is fixing the systematic mismatch with the competition’s “inspiratory-only” scoring. With your existing mean-lookup + fallback + snap core logic unchanged, I (1) compute lookup tables from inspiratory rows only (already done) and (2) **apply predictions only on inspiratory rows in test and force expiratory rows (`u_out==1`) to 0**, which is safe because they are unscored and avoids bleeding noisy estimates into rows that can confuse debugging/alignment. I also remove the fragile `id`-uniqueness assertion (your file inspection shows `id` 1–2000, so that check can incorrectly block correct runs) and instead align deterministically by row order between `test.csv` and `sample_submission.csv`, which is the standard for this competition and prevents merge-induced duplication/misalignment. These are minimal changes intended to move the score sharply toward the 0.1449 target without changing the underlying approach.'
- What this solution (achieved 7.18825) has done: 'Your MAE is far above the target (lower is better), and the most likely cause is still a catastrophic `id` alignment issue: your current code overwrites `test.id` with `sample_submission.id`, but in this competition `test.csv` already contains the correct globally-unique `id` and should not be replaced. I make the smallest fix by keeping your exact mean-lookup + fallback chain + snap-to-nearest-pressure logic, but aligning predictions by the true `id` (merge back to `sample_submission` by `id`) and preserving `test`’s original `id`. I also keep the metric-aligned choice of training only on inspiratory rows (`u_out==0`), and for unscored expiratory rows set pressure to 0 as you already do. These changes should dramatically reduce the MAE by restoring correct row-to-row mapping without changing the core modeling approach.'
- What this solution (achieved 7.18825) has done: 'Your MAE (7.19) is so far from the target (0.145) that it strongly indicates a systematic bug rather than a weak baseline. The smallest, most relevant fix is to stop merging predictions back by `id` (your environment shows `id` repeats 1–2000, so that merge scrambles/duplicates rows) and instead keep the strict row-order alignment between `test.csv` and `sample_submission.csv` (the standard for this competition). To preserve your core “inspiratory-only mean lookup + fallback chain + snap-to-nearest pressure” logic, I only change the final assembly: write `pressure` in the same row order as `sample_submission`, and carry `id` directly from `sample_submission`. This should remove the catastrophic misalignment and move the score sharply toward the target without changing your modeling approach.'
- What this solution (achieved 7.18825) has done: 'Your score is still catastrophically far from the 0.1449 target (lower is better), which is much more consistent with a systematic alignment bug than with a weak mean-lookup baseline. The smallest fix that preserves your exact lookup/fallback/snap logic is to stop pairing `pressure` predictions with `sample_submission` by row order and instead output predictions for the corresponding `test.csv` `id` (which is the actual submission key in this competition). I also add a strict sanity check that `test.id` matches `sample_submission.id` exactly (same values and order); if not, we fall back to explicitly aligning by merging on `id` (still minimal and deterministic). These changes should move the MAE sharply toward the target band without altering your feature extraction, lookup tables, or snapping behavior.'

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


def find_nearest_vec(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64)
    idx_left = np.searchsorted(sorted_pressures, preds, side="left")
    idx = np.clip(idx_left, 0, total_pressures_len - 1)

    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper_idx = idx

    lower = sorted_pressures[lower_idx]
    upper = sorted_pressures[upper_idx]

    choose_lower = np.abs(lower - preds) < np.abs(upper - preds)
    out = np.where(choose_lower, lower, upper)

    out = np.where(idx_left >= total_pressures_len, sorted_pressures[-1], out)
    out = np.where(idx_left <= 0, sorted_pressures[0], out)
    return out.astype(np.float64)


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
    a.pressure = a.pressure * 0.65 + b.pressure * 0.35
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sample_sub = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

if df_test.shape[0] != sample_sub.shape[0]:
    raise RuntimeError(
        f"Row count mismatch: test={df_test.shape[0]} vs sample={sample_sub.shape[0]}"
    )

df_test_aligned = df_test.copy()

train_insp = df_train[df_train["u_out"] == 0].copy()


def add_features(df):
    out = df[["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]].copy()
    out["R"] = out["R"].astype(np.int16)
    out["C"] = out["C"].astype(np.int16)
    out["time_step"] = out["time_step"].astype(np.float64)

    out["u_in_round"] = (out["u_in"] * 2.0).round() / 2.0
    out["u_in_round_m05"] = out["u_in_round"] - 0.5
    out["u_in_round_p05"] = out["u_in_round"] + 0.5
    out["u_in_round_m10"] = out["u_in_round"] - 1.0
    out["u_in_round_p10"] = out["u_in_round"] + 1.0
    return out


train_feats = add_features(train_insp)
test_feats = add_features(df_test_aligned)

train_join_base = pd.concat([train_feats, train_insp["pressure"]], axis=1)

mean_r_c_t_u = (
    train_join_base.groupby(["R", "C", "time_step", "u_in_round"], sort=False)[
        "pressure"
    ]
    .mean()
    .rename("p_mean_rctu")
    .reset_index()
)

mean_r_c_t_u_m05 = mean_r_c_t_u.rename(
    columns={"u_in_round": "u_in_round_m05", "p_mean_rctu": "p_mean_rctu_m05"}
)
mean_r_c_t_u_p05 = mean_r_c_t_u.rename(
    columns={"u_in_round": "u_in_round_p05", "p_mean_rctu": "p_mean_rctu_p05"}
)
mean_r_c_t_u_m10 = mean_r_c_t_u.rename(
    columns={"u_in_round": "u_in_round_m10", "p_mean_rctu": "p_mean_rctu_m10"}
)
mean_r_c_t_u_p10 = mean_r_c_t_u.rename(
    columns={"u_in_round": "u_in_round_p10", "p_mean_rctu": "p_mean_rctu_p10"}
)

mean_r_c_t = (
    train_join_base.groupby(["R", "C", "time_step"], sort=False)["pressure"]
    .mean()
    .rename("p_mean_rct")
    .reset_index()
)

mean_r_c = (
    train_join_base.groupby(["R", "C"], sort=False)["pressure"]
    .mean()
    .rename("p_mean_rc")
    .reset_index()
)

global_mean = float(train_insp["pressure"].mean())

pred_df = test_feats.merge(
    mean_r_c_t_u, on=["R", "C", "time_step", "u_in_round"], how="left"
)
pred_df = pred_df.merge(
    mean_r_c_t_u_m05, on=["R", "C", "time_step", "u_in_round_m05"], how="left"
)
pred_df = pred_df.merge(
    mean_r_c_t_u_p05, on=["R", "C", "time_step", "u_in_round_p05"], how="left"
)
pred_df = pred_df.merge(
    mean_r_c_t_u_m10, on=["R", "C", "time_step", "u_in_round_m10"], how="left"
)
pred_df = pred_df.merge(
    mean_r_c_t_u_p10, on=["R", "C", "time_step", "u_in_round_p10"], how="left"
)
pred_df = pred_df.merge(mean_r_c_t, on=["R", "C", "time_step"], how="left")
pred_df = pred_df.merge(mean_r_c, on=["R", "C"], how="left")

pred = pred_df["p_mean_rctu"].to_numpy()
fallback_m05 = pred_df["p_mean_rctu_m05"].to_numpy()
fallback_p05 = pred_df["p_mean_rctu_p05"].to_numpy()
fallback_m10 = pred_df["p_mean_rctu_m10"].to_numpy()
fallback_p10 = pred_df["p_mean_rctu_p10"].to_numpy()
fallback_rct = pred_df["p_mean_rct"].to_numpy()
fallback_rc = pred_df["p_mean_rc"].to_numpy()

mask = np.isnan(pred)
pred[mask] = fallback_m05[mask]
mask = np.isnan(pred)
pred[mask] = fallback_p05[mask]
mask = np.isnan(pred)
pred[mask] = fallback_m10[mask]
mask = np.isnan(pred)
pred[mask] = fallback_p10[mask]
mask = np.isnan(pred)
pred[mask] = fallback_rct[mask]
mask = np.isnan(pred)
pred[mask] = fallback_rc[mask]
pred = np.where(np.isnan(pred), global_mean, pred)

pred = np.where(test_feats["u_out"].to_numpy() == 1, 0.0, pred)

pred = find_nearest_vec(pred)

test_ids = df_test_aligned["id"].to_numpy()
sample_ids = sample_sub["id"].to_numpy()

if test_ids.shape != sample_ids.shape:
    raise RuntimeError(
        "Unexpected id shape mismatch between test and sample_submission."
    )

if np.array_equal(test_ids, sample_ids):
    sub = pd.DataFrame({"id": sample_ids, "pressure": pred})
else:
    pred_by_id = pd.DataFrame({"id": test_ids, "pressure": pred})
    sub = sample_sub[["id"]].merge(pred_by_id, on="id", how="left")
    if sub["pressure"].isna().any():
        sub["pressure"] = sub["pressure"].fillna(global_mean)
        sub["pressure"] = find_nearest_vec(sub["pressure"].to_numpy(dtype=np.float64))

if sub.shape[0] != sample_sub.shape[0]:
    raise RuntimeError(
        f"Row count mismatch: sub={sub.shape[0]} vs sample={sample_sub.shape[0]}"
    )
if sub["pressure"].isna().any():
    raise RuntimeError("Submission contains NaN pressures after prediction.")
if not np.array_equal(sub["id"].to_numpy(), sample_sub["id"].to_numpy()):
    raise RuntimeError("Final submission id order does not match sample_submission.")

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("pressure nan count:", int(sub["pressure"].isna().sum()))
print("pressure min/max:", float(sub["pressure"].min()), float(sub["pressure"].max()))
print("u_out==1 count:", int((df_test_aligned["u_out"] == 1).sum()))

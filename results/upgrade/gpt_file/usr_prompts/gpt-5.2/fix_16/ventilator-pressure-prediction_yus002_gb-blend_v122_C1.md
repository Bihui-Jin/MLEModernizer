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

0.1449434657072991

# 6. Current score

0.80654

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.45185) has done: 'I remove the dependency on the missing external dataset (`../input/gb-data-blending-recover/mean_submission.csv`) that currently crashes the notebook, and instead generate a valid prediction directly from the provided competition `train.csv`/`test.csv`. To keep changes minimal and score reasonable, I use a simple, deterministic baseline: predict pressure by mapping each test row to the mean training pressure for the same `(R, C, u_in, u_out)` pattern, with sensible fallbacks when an exact key is unseen. I also snap predictions to the nearest valid pressure level from the training set (consistent with your existing logic) and write a proper `submission.csv` with `id,pressure`. This run end-to-end in the given environment and produce a valid `.csv` submission file.'
- What this solution (achieved 1.54179) has done: 'Your current baseline is too coarse because it ignores the sequential nature of each breath and the fact that pressure is only scored on the inspiratory phase; this leads to very large MAE. To move the score much closer to your target while keeping changes minimal and fully deterministic, I keep your “snap-to-valid-pressure” post-processing but replace the row-wise mean lookup with a per-breath nearest-neighbor lookup over full sequences using only the provided features `(R, C, u_in, u_out)` across 80 time steps. This preserves the overall “predict by matching patterns in train” approach (no new model/training loop), but uses much more of the available signal with a small, runtime-safe subset search per `(R,C)` group. The output still be a valid `submission.csv` with `id,pressure` and unchanged I/O paths.'
- What this solution (achieved 0.67489) has done: 'Your current nearest-neighbor approach is directionally right but is held back by (1) using an arbitrary cap `K=250` that can exclude the true nearest neighbor within an (R,C) group and (2) scoring being only on inspiratory phase, so the distance function should emphasize matching the inspiratory part of the sequence rather than treating all 80 steps equally. To move the MAE down toward your target with minimal changes and identical “match a test breath to the closest train breath by sequence distance” core logic, I keep the same RC-grouped 1-NN retrieval but (a) search the full (R,C) candidate set efficiently in chunks (so no K-truncation) and (b) compute distance primarily over inspiratory timesteps (`u_out==0`) with a small contribution from the full sequence for stability. I keep your snap-to-valid-pressure post-processing and the same submission writing/format.'
- What this solution (achieved 0.69972) has done: 'Your current 1-NN within (R,C) groups is close in spirit but still mismatched to the metric because it (a) uses the test inspiratory mask rather than the candidate’s inspiratory phase, and (b) treats inspiratory timesteps with a simple 0/1 mask instead of weighting by how much air is flowing in (u_in), which correlates strongly with pressure changes. I keep the exact same retrieval core logic (RC-grouped 1-NN over full candidate set in chunks) but adjust the distance to use the *candidate’s* inspiratory phase mask and a u_in-based weighting, plus a small additional penalty when u_out patterns differ during inspiratory. I also ensure id alignment is preserved by writing predictions in the original test row order (so there’s no risk of merge/order artifacts) while keeping the same submission schema and snapping-to-valid-pressure postprocess. These are minimal, deterministic changes intended to reduce MAE (lower is better) and move your 0.67489 closer to the 0.14494 target.'
- What this solution (achieved 0.83377) has done: 'Your current 1-NN retrieval is already aligned with the “match test breath to a similar train breath” core logic, but the distance is still dominated by amplitude scale and doesn’t explicitly use the fact that pressure is much more tied to *integrated* flow (cumulative u_in) during inspiration. I keep the exact same RC-grouped full-candidate chunked 1-NN search, but change the distance to include (and emphasize) cumulative u_in (and cumulative u_out) features in addition to the raw per-step signals, which is a minimal metric-alignment tweak rather than a new model. I also make the inspiratory weighting use both train and test inspiratory masks (intersection) to avoid counting mismatched phases heavily, while still retaining a small full-sequence term for stability. These changes are deterministic, keep I/O and snapping-to-valid-pressure intact, and should reduce MAE (lower is better) toward your target.'
- What this solution (achieved 0.696) has done: 'Your current score (0.83377, lower-is-better) is still far above the target (0.14494), so we need a real improvement while keeping your same RC-grouped 1-NN, chunked full-candidate search, and snap-to-valid-pressure logic. The minimal change most likely to reduce MAE is to make the distance reflect the ventilator physics better by comparing **cumulative inspired volume proxy** (`∫ u_in dt`) using the actual `time_step` deltas, rather than plain cumulative sums that ignore irregular timing. I keep all your existing terms and weights but replace the cum-feature construction with **time-weighted integrals** (and correspondingly adjust the cum-distance term), which should select closer neighbors and reduce error without changing the overall approach. I also keep your submission alignment logic intact and still write `submission.csv` with `id,pressure`.'
- What this solution (achieved 0.696) has done: 'Your current gap to target is large (0.696 vs 0.1449, lower-is-better), so we need a meaningful but still “same-core-logic” improvement inside your existing RC-grouped chunked 1-NN retrieval. The biggest metric mismatch left is that the competition only scores inspiratory timesteps, yet your final prediction uses the neighbor’s raw pressure for *all* timesteps; we can safely improve by forcing expiratory predictions (u_out==1 in test) to a reasonable baseline (e.g., last inspiratory predicted pressure per breath), which directly reduces error on scored parts indirectly by preventing snapping artifacts and keeps semantics legitimate. Additionally, your time-integral features currently depend on `prepend=train_t[:, :1]` which makes the first dt=0 and can underweight early inspiration; switching to a stable dt construction (`dt[:,0]=t[:,0]` and `diff` thereafter) is a tiny numeric fix that improves neighbor selection without changing the approach. Finally, we keep snapping-to-valid-pressure and submission alignment unchanged.'
- What this solution (achieved 0.80993) has done: 'Your current 1-NN-by-(R,C) sequence matching is reasonable, but the distance is still not aligned enough with how pressure evolves: it depends much more on **cumulative inspired volume** than on raw u_in at each step. I keep your exact retrieval approach (RC-grouped, chunked full-candidate search, 1-NN) and snapping, but minimally adjust the distance to (1) compare **integrals of u_in and u_in² over time** (better proxy for delivered volume/energy), and (2) include a small penalty on mismatch of **change in u_in (du_in)** to favor similarly-shaped inspiratory ramps. These additions are deterministic, fast (only extra cumsums/diffs), and should move MAE down from ~0.696 toward your target. I also keep your expiratory fill behavior and submission alignment exactly as-is.'
- What this solution (achieved 0.8101) has done: 'Your current approach (RC-grouped, chunked full-candidate 1-NN sequence matching + snap-to-valid-pressure) is still far from the target, so the smallest meaningful improvement is to better align the distance with the metric by comparing **pressure-relevant cumulative volume during inspiration** more strongly and by preventing the distance from being dominated by arbitrary absolute scale. I keep the exact same retrieval loop and post-processing, but (1) add a **time-weighted integral of positive flow only** (`u_in_pos_int`) which is a closer proxy to delivered volume, and (2) add a **per-breath normalization** of the sequence features used inside the distance (center/scale `u_in` and its derivative) to improve neighbor selection without changing the model family. I also fix a subtle id-alignment risk by avoiding `merge` (which can reorder) and instead write predictions by mapping directly to the original `id` order. These are minimal, deterministic changes aimed at reducing MAE (lower is better) toward your target.'
- What this solution (achieved 0.81138) has done: 'Your current MAE (0.8101, lower-is-better) is still far above the target (0.14494), so we need a meaningful improvement while preserving your exact “RC-grouped full-candidate chunked 1-NN sequence matching + snap-to-valid-pressure” core logic. The smallest high-impact fix is to align the distance computation with the evaluation: it is only scored on inspiratory timesteps, so we should compute inspiratory distance using the *test* inspiratory mask (not the intersection), and de-emphasize/ignore expiratory mismatch in distance (since it’s unscored) while still keeping a small full-sequence stabilizer. Additionally, your per-breath z-normalization currently uses all 80 steps; switching the normalization stats to inspiratory-only better matches the scored region and improves nearest-neighbor selection without changing the approach. These changes keep I/O, chunked search, 1-NN retrieval, snapping, and submission writing intact, and are deterministic.'
- What this solution (achieved 0.83222) has done: 'Your current score (0.81138 MAE, lower-is-better) is far above the target (0.14494), so we need a meaningful improvement while keeping your exact RC-grouped, chunked full-candidate 1-NN retrieval and snap-to-valid-pressure core logic intact. The biggest issue is that the distance uses the test inspiratory mask but still lets non-inspiratory (unscored) structure influence matching via full-sequence terms; I gate those “full” terms to inspiratory too, keeping only a tiny stabilizer so we don’t degrade robustness. I also add a very small, physics-aligned term based on the test breath’s time-weighted integral of (u_in − u_out·u_in) to better represent delivered volume during inspiration without changing the approach. Finally, I keep submission writing identical but make the snapping step vectorized (same semantics) to reduce runtime risk under Kaggle limits.'
- What this solution (achieved 0.81119) has done: 'Your current approach is still a deterministic RC-grouped 1-NN sequence match, but the distance is dominated by raw amplitude/scale and heavy cumulative terms, which tends to pick the wrong neighbor and keeps MAE high. To move the score down toward your target while preserving the same core logic, I (1) make the distance scale-robust by using per-breath z-scored signals as the primary inspiratory matching signal (keeping cumulative/shape terms but down-weighted), and (2) add a small but important physics-aligned term comparing the *delivered volume proxy* (time integral of effective flow) at the end of inspiration (a scalar per breath), which improves neighbor selection with minimal extra computation. I also fix a subtle but impactful inconsistency: the distance currently weights by `w_test` but not by `dt`, so breaths with slightly different timing can mismatch; I include `dt` in the inspiratory weighting (still using the same features and loop). Submission writing, snapping-to-valid-pressure, and alignment by `id` remain unchanged.'
- What this solution (achieved 0.80237) has done: 'Your current MAE (0.81119, lower-is-better) is far above the target (0.14494), so we should improve nearest-neighbor selection while keeping your exact RC-grouped chunked 1-NN + snap-to-valid-pressure core logic unchanged. The highest-impact minimal fix is that `dist_full` is currently identical to `dist_insp` (both use the inspiratory mask), so your `full_weight` term provides no extra information; I make `dist_full` truly cover the full sequence (all 80 steps) with time/flow weights, preserving the same distance family and weights. I also fix a bug in computing the “last inspiratory index”: `argmax(mask*arange)` returns 0 for breaths that are all-inspiration, which miscomputes the end-of-inspiration scalar feature; switching to a safe “max index where mask is true” improves the end feature consistently without changing semantics. These two targeted fixes should reduce MAE toward the target without changing I/O, snapping, or the 1-NN retrieval loop structure.'
- What this solution (achieved 0.81136) has done: 'Your current score (0.80237 MAE, lower-is-better) is far above the target (0.14494), so we need a meaningful improvement while keeping your exact RC-grouped, chunked full-candidate 1-NN retrieval and snap-to-valid-pressure logic intact. The smallest high-impact fix is to align the distance computation with the evaluation metric by **computing all sequence-distance terms only on inspiratory timesteps** (test `u_out==0`), while keeping just a tiny full-sequence stabilizer so we don’t overfit the mask. In practice, this means masking `dist_full` and all cumulative/shape distances to inspiratory (using the already-defined `cmask`) rather than letting expiratory mismatches dominate neighbor choice even though they’re unscored. I also reduce the (previously non-negligible) penalty from `du_out` over the full sequence to near-zero, since expiratory pattern mismatch should not steer the 1-NN selection strongly for this metric. These are minimal, deterministic weight/masking adjustments inside your existing loop and should move MAE down toward the target without changing I/O or submission format.'
- What this solution (achieved 0.80654) has done: 'Your current score (0.81136 MAE, lower-is-better) is far above the target (0.14494), so we need a real gain while keeping your exact RC-grouped, chunked full-candidate 1-NN retrieval + snap-to-valid-pressure logic intact. The biggest issue in the current distance is that your “full sequence” term is not actually full (it reuses the inspiratory mask), and several cumulative/shape distances are being pointlessly re-masked (they are already cumulative, so masking only the inspiratory timesteps underweights the decisive early/overall inspiration volume). I (1) make `dist_full` truly full-sequence (all 80 steps) with time/flow weights, and (2) compute the cumulative distances against the full cumulative curves but only *weight* them by inspiratory time-weights (dt-weighted), which better reflects the scored region without throwing away informative cumulative history. These are minimal, local changes inside your existing distance computation; no model/training/feature family changes, and the same submission alignment + snapping remain.'

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
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

required_train_cols = {"breath_id", "R", "C", "u_in", "u_out", "pressure", "time_step"}
required_test_cols = {"breath_id", "R", "C", "u_in", "u_out", "id", "time_step"}
if not required_train_cols.issubset(df_train.columns):
    raise ValueError(
        f"train.csv missing columns: {required_train_cols - set(df_train.columns)}"
    )
if not required_test_cols.issubset(df_test.columns):
    raise ValueError(
        f"test.csv missing columns: {required_test_cols - set(df_test.columns)}"
    )

df_test_orig = df_test.copy()

df_train = df_train.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
df_test_sorted = df_test.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

T = 80
train_breath_ids = df_train["breath_id"].unique()
test_breath_ids = df_test_sorted["breath_id"].unique()

n_train_breaths = len(train_breath_ids)
n_test_breaths = len(test_breath_ids)

if len(df_train) != n_train_breaths * T:
    raise RuntimeError("Unexpected train shape; cannot reshape to (n_breaths, 80).")
if len(df_test_sorted) != n_test_breaths * T:
    raise RuntimeError("Unexpected test shape; cannot reshape to (n_breaths, 80).")

train_RC = (
    df_train.groupby("breath_id")[["R", "C"]].first().loc[train_breath_ids].to_numpy()
)
test_RC = (
    df_test_sorted.groupby("breath_id")[["R", "C"]]
    .first()
    .loc[test_breath_ids]
    .to_numpy()
)

train_u_in = df_train["u_in"].to_numpy(dtype=np.float32).reshape(n_train_breaths, T)
train_u_out = df_train["u_out"].to_numpy(dtype=np.float32).reshape(n_train_breaths, T)
train_pressure = (
    df_train["pressure"].to_numpy(dtype=np.float32).reshape(n_train_breaths, T)
)

test_u_in = df_test_sorted["u_in"].to_numpy(dtype=np.float32).reshape(n_test_breaths, T)
test_u_out = (
    df_test_sorted["u_out"].to_numpy(dtype=np.float32).reshape(n_test_breaths, T)
)

train_t = df_train["time_step"].to_numpy(dtype=np.float32).reshape(n_train_breaths, T)
test_t = (
    df_test_sorted["time_step"].to_numpy(dtype=np.float32).reshape(n_test_breaths, T)
)

train_dt = np.empty_like(train_t, dtype=np.float32)
test_dt = np.empty_like(test_t, dtype=np.float32)
train_dt[:, 0] = np.clip(train_t[:, 0], 0.0, None)
test_dt[:, 0] = np.clip(test_t[:, 0], 0.0, None)
train_dt[:, 1:] = np.clip(np.diff(train_t, axis=1), 0.0, None)
test_dt[:, 1:] = np.clip(np.diff(test_t, axis=1), 0.0, None)

train_u_in_pos = np.clip(train_u_in, 0.0, None)
test_u_in_pos = np.clip(test_u_in, 0.0, None)
train_u_in_pos_int = np.cumsum(train_u_in_pos * train_dt, axis=1, dtype=np.float32)
test_u_in_pos_int = np.cumsum(test_u_in_pos * test_dt, axis=1, dtype=np.float32)

train_u_in_int = np.cumsum(train_u_in * train_dt, axis=1, dtype=np.float32)
test_u_in_int = np.cumsum(test_u_in * test_dt, axis=1, dtype=np.float32)
train_u_out_int = np.cumsum(train_u_out * train_dt, axis=1, dtype=np.float32)
test_u_out_int = np.cumsum(test_u_out * test_dt, axis=1, dtype=np.float32)

train_u_in2_int = np.cumsum(
    (train_u_in * train_u_in) * train_dt, axis=1, dtype=np.float32
)
test_u_in2_int = np.cumsum((test_u_in * test_u_in) * test_dt, axis=1, dtype=np.float32)

train_du_in = np.diff(train_u_in, axis=1, prepend=train_u_in[:, :1]).astype(np.float32)
test_du_in = np.diff(test_u_in, axis=1, prepend=test_u_in[:, :1]).astype(np.float32)

train_flow_eff = train_u_in * (1.0 - train_u_out)
test_flow_eff = test_u_in * (1.0 - test_u_out)
train_flow_eff_int = np.cumsum(train_flow_eff * train_dt, axis=1, dtype=np.float32)
test_flow_eff_int = np.cumsum(test_flow_eff * test_dt, axis=1, dtype=np.float32)

eps = np.float32(1e-3)
train_insp_mask = (train_u_out < 0.5).astype(np.float32)  # (n_train, T)
test_insp_mask_2d = (test_u_out < 0.5).astype(np.float32)  # (n_test, T)

train_insp_cnt = np.clip(train_insp_mask.sum(axis=1, keepdims=True), 1.0, None).astype(
    np.float32
)
test_insp_cnt = np.clip(test_insp_mask_2d.sum(axis=1, keepdims=True), 1.0, None).astype(
    np.float32
)

train_u_in_mean = (
    (train_u_in * train_insp_mask).sum(axis=1, keepdims=True) / train_insp_cnt
).astype(np.float32)
test_u_in_mean = (
    (test_u_in * test_insp_mask_2d).sum(axis=1, keepdims=True) / test_insp_cnt
).astype(np.float32)

train_u_in_var = (
    (train_insp_mask * (train_u_in - train_u_in_mean) ** 2).sum(axis=1, keepdims=True)
    / train_insp_cnt
).astype(np.float32)
test_u_in_var = (
    (test_insp_mask_2d * (test_u_in - test_u_in_mean) ** 2).sum(axis=1, keepdims=True)
    / test_insp_cnt
).astype(np.float32)

train_u_in_std = (np.sqrt(train_u_in_var) + eps).astype(np.float32)
test_u_in_std = (np.sqrt(test_u_in_var) + eps).astype(np.float32)

train_u_in_z = (train_u_in - train_u_in_mean) / train_u_in_std
test_u_in_z = (test_u_in - test_u_in_mean) / test_u_in_std

train_du_in_var = (
    (train_insp_mask * (train_du_in) ** 2).sum(axis=1, keepdims=True) / train_insp_cnt
).astype(np.float32)
test_du_in_var = (
    (test_insp_mask_2d * (test_du_in) ** 2).sum(axis=1, keepdims=True) / test_insp_cnt
).astype(np.float32)

train_du_in_std = (np.sqrt(train_du_in_var) + eps).astype(np.float32)
test_du_in_std = (np.sqrt(test_du_in_var) + eps).astype(np.float32)

train_du_in_z = train_du_in / train_du_in_std
test_du_in_z = test_du_in / test_du_in_std

train_last_insp_idx = np.where(
    train_insp_mask > 0.5, np.arange(T, dtype=np.int32), -1
).max(axis=1)
test_last_insp_idx = np.where(
    test_insp_mask_2d > 0.5, np.arange(T, dtype=np.int32), -1
).max(axis=1)
train_last_insp_idx = np.clip(train_last_insp_idx, 0, T - 1)
test_last_insp_idx = np.clip(test_last_insp_idx, 0, T - 1)

train_flow_eff_end = train_flow_eff_int[
    np.arange(n_train_breaths), train_last_insp_idx
].astype(np.float32)
test_flow_eff_end = test_flow_eff_int[
    np.arange(n_test_breaths), test_last_insp_idx
].astype(np.float32)

train_rc_tuples = [tuple(x) for x in train_RC.astype(int)]
rc_to_train_indices = {}
for i, rc in enumerate(train_rc_tuples):
    rc_to_train_indices.setdefault(rc, []).append(i)

pred_pressure = np.empty((n_test_breaths, T), dtype=np.float32)

chunk_size = 2048

uin_weight_power = 1.0  # keep conservative weighting as before

uout_insp_penalty = 6.0
uout_full_penalty = 0.05

insp_weight = 2.5
full_weight = 0.01

cum_weight = 1.1
uin2_cum_weight = 0.15
duin_weight = 0.18
u_in_pos_cum_weight = 0.55
uin_shape_weight = 3.5  # make z-shape the main driver (still same 1-NN logic)

flow_eff_cum_weight = 0.35

flow_end_weight = 2.0

global_mean_pressure = float(df_train["pressure"].mean())

for i in range(n_test_breaths):
    rc = tuple(test_RC[i].astype(int))
    cand_list = rc_to_train_indices.get(rc, None)

    if cand_list is None or len(cand_list) == 0:
        pred_pressure[i, :] = global_mean_pressure
        continue

    w_test = np.clip(test_u_in[i], 0.0, 100.0) / 100.0
    w_test = (0.1 + w_test) ** uin_weight_power  # (T,)
    w_time = np.clip(test_dt[i], 0.0, None)  # (T,)

    w_step_full = (w_test * (0.2 + w_time)).astype(np.float32)  # (T,)

    test_insp_mask = (test_u_out[i] < 0.5).astype(np.float32)  # (T,)

    best_dist = np.inf
    best_idx = cand_list[0]

    test_end = float(test_flow_eff_end[i])

    for start in range(0, len(cand_list), chunk_size):
        cand = np.asarray(cand_list[start : start + chunk_size], dtype=np.int32)

        du_in = train_u_in[cand] - test_u_in[i]  # (chunk, T)
        du_out = train_u_out[cand] - test_u_out[i]  # (chunk, T)

        du_in_int = train_u_in_int[cand] - test_u_in_int[i]
        du_out_int = train_u_out_int[cand] - test_u_out_int[i]

        du_in2_int = train_u_in2_int[cand] - test_u_in2_int[i]
        ddu_in = train_du_in_z[cand] - test_du_in_z[i]

        du_in_pos_int = train_u_in_pos_int[cand] - test_u_in_pos_int[i]
        du_in_z = train_u_in_z[cand] - test_u_in_z[i]

        d_flow_eff_int = train_flow_eff_int[cand] - test_flow_eff_int[i]

        cmask = test_insp_mask[None, :]  # (1, T)

        w_insp = (cmask * w_step_full[None, :]).astype(np.float32)

        w_full = w_step_full[None, :].astype(np.float32)

        dist_insp = ((du_in * du_in) * w_insp).sum(axis=1) + uout_insp_penalty * (
            (du_out * du_out) * cmask
        ).sum(axis=1)

        dist_full = ((du_in * du_in) * w_full).sum(axis=1) + uout_full_penalty * (
            (du_out * du_out)
        ).sum(axis=1)

        dist_cum = ((du_in_int * du_in_int) * w_insp).sum(axis=1) + 0.25 * (
            (du_out_int * du_out_int) * w_insp
        ).sum(axis=1)

        dist_uin2 = ((du_in2_int * du_in2_int) * w_insp).sum(axis=1)
        dist_duin_shape = ((ddu_in * ddu_in) * w_insp).sum(axis=1)

        dist_uin_pos_cum = ((du_in_pos_int * du_in_pos_int) * w_insp).sum(axis=1)
        dist_uin_shape = ((du_in_z * du_in_z) * w_insp).sum(axis=1)

        dist_flow_eff = ((d_flow_eff_int * d_flow_eff_int) * w_insp).sum(axis=1)

        d_end = train_flow_eff_end[cand].astype(np.float32) - np.float32(test_end)
        dist_end = d_end * d_end

        dist = (
            insp_weight * dist_insp
            + full_weight * dist_full
            + cum_weight * dist_cum
            + uin2_cum_weight * dist_uin2
            + duin_weight * dist_duin_shape
            + u_in_pos_cum_weight * dist_uin_pos_cum
            + uin_shape_weight * dist_uin_shape
            + flow_eff_cum_weight * dist_flow_eff
            + flow_end_weight * dist_end
        )

        local_argmin = int(np.argmin(dist))
        local_best_dist = float(dist[local_argmin])
        if local_best_dist < best_dist:
            best_dist = local_best_dist
            best_idx = int(cand[local_argmin])

    pred_pressure[i, :] = train_pressure[best_idx, :]

test_insp_mask_bool = test_u_out < 0.5
for i in range(n_test_breaths):
    insp_idx = np.where(test_insp_mask_bool[i])[0]
    if insp_idx.size > 0:
        last_insp = pred_pressure[i, int(insp_idx[-1])]
        pred_pressure[i, ~test_insp_mask_bool[i]] = last_insp

pred_flat_sorted = pred_pressure.reshape(-1).astype(np.float64)

sp = sorted_pressures.astype(np.float64)
idx = np.searchsorted(sp, pred_flat_sorted, side="left")
idx = np.clip(idx, 0, len(sp) - 1)
idx_lo = np.clip(idx - 1, 0, len(sp) - 1)
hi = sp[idx]
lo = sp[idx_lo]
choose_lo = np.abs(pred_flat_sorted - lo) < np.abs(hi - pred_flat_sorted)
pred_snapped_sorted = np.where(choose_lo, lo, hi).astype(np.float64)

out_sorted = df_test_sorted[["id"]].copy()
out_sorted["pressure"] = pred_snapped_sorted
id_to_pred = pd.Series(out_sorted["pressure"].values, index=out_sorted["id"].values)

sub["pressure"] = sub["id"].map(id_to_pred).astype(np.float64)

if sub["pressure"].isna().any():
    raise RuntimeError("Submission has NaN pressures after prediction fill.")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

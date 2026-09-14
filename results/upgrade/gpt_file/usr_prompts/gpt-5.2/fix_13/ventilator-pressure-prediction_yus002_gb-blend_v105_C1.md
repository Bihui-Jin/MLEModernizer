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

0.1514367885201152

# 6. Current score

4.44208

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.91814) has done: 'The failure comes from trying to read external submission files (`../input/gb-data-blending-recover/...`) that are not available in this environment. I keep your pressure-snapping logic (`find_nearest`) intact, but replace the unavailable blending step with a self-contained baseline that produces a valid `submission.csv` from the provided train/test files. To keep changes minimal and score-reasonable, the baseline predict per-(R,C,time_step,u_in,u_out) median pressure from train and fall back to per-(R,C,time_step) median, then global median, and finally snap to the nearest allowed pressure. This runs end-to-end and writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 4.00098) has done: 'Your current baseline (groupby medians on raw `time_step` and `u_in`) barely matches between train and test because those floats rarely align exactly, so most rows fall back to the global median and the MAE stays very high. To move the score much closer to your target with minimal logic change, I keep the same “median lookup then fallback then snap-to-allowed-pressures” approach, but make the join keys matchable by quantizing `time_step` (to 0.01s) and `u_in` (to 0.1) in both train and test. This preserves your core semantics while dramatically increasing the hit-rate of the informative median tables. I also ensure the submission is aligned to `sample_submission.csv`’s id order (no accidental misalignment).'
- What this solution (achieved 4.14362) has done: 'I fix the `KeyError: 'R'` by ensuring the intermediate `test_pred` dataframe still contains the merge key columns (`R`, `C`, `time_step_q`, `u_out`) when performing subsequent merges; the error happens because those columns were dropped earlier. I do this with minimal changes by starting `test_pred` from the full keyed test rows and then left-joining the median tables in sequence. I also add a small safety check to guarantee there are no missing predictions after merging, and keep your existing quantization + median fallback + pressure snapping logic unchanged. Finally, the script always write a valid `submission.csv` with `id,pressure` in the sample submission’s id order.'
- What this solution (achieved 4.08981) has done: 'Your current approach is solid but is still effectively “missing the key” too often because the `time_step_q` quantization grid (0.03s) is too coarse for matching the true 0.02s sampling and because `u_in_q` rounding can shift values into adjacent bins unnecessarily. I keep the exact same median-lookup → fallback → snap-to-allowed-pressures logic, but change quantization to (a) map `time_step` to the nearest of the 80 canonical steps per breath and (b) bin `u_in` more conservatively with `floor` (stable, fewer boundary flips). I also add one extra fallback table at the `(R,C,time_step_q,u_in_q)` level (dropping `u_out`) which is still the same semantics (median tables + fallback) but materially increases hit-rate and should move MAE much closer to your target. Submission writing and pressure snapping remain unchanged.'
- What this solution (achieved 3.92095) has done: 'Your current score (4.08981 MAE) is far worse than the target (0.1514), so we should improve accuracy while keeping your same “median lookup → fallback → snap-to-allowed-pressures” core logic. The biggest remaining issue is that quantizing `u_in` to 0.1 and then taking medians still yields lots of mismatches/noisy bins; we can keep the same approach but switch to a more match-friendly and stable `u_in` binning (integer `u_in` bins) and add one more still-on-theme fallback at the `(R,C,time_step_q,u_in_q)` level (already present) but with a better `u_in_q`, plus a breath-position fallback `(R,C,u_in_q,u_out)` to catch repeated control patterns across time. We also restrict training medians to inspiratory phase (`u_out==0`) because the metric only scores inspiratory timesteps; this keeps evaluation semantics aligned without changing the overall approach. All other pieces (pressure snapping, merge-based prediction assembly, and writing `submission.csv`) remain intact.'
- What this solution (achieved 3.92095) has done: 'Your MAE is still very far from the target, so we should improve accuracy while keeping your same “median lookup → fallback → snap-to-allowed-pressures” approach. The biggest easy win is to stop mixing expiratory behavior into predictions: for `u_out==1` timesteps we can directly predict 0 pressure (common strong baseline), while keeping your median tables trained on inspiratory-only data. Next, we increase median-table hit-rate without changing the approach by adding one additional, still-consistent fallback keyed on the full control state `(R,C,u_in_q,u_out)` computed from all rows (insp+exp), which helps when time-step-specific medians miss. Everything else (quantization to canonical 0.02 grid + integer u_in bins, merge-based assembly, snapping, and writing `submission.csv`) remains the same.'
- What this solution (achieved 3.90792) has done: 'Your current score is far above (worse than) the target, and the biggest issue is that the median-lookup tables still miss too often because `u_in` is continuous and integer-binning is too coarse, pushing many rows to weak fallbacks. I keep your exact merge-based “median tables → fallback chain → global median → snap to allowed pressures” core logic, but change only the `u_in` quantization to a match-friendly 0.5 resolution (reduces sparsity while still stable) and add one additional minimal fallback keyed on `(R,C,time_step_q,u_out)` computed from **all** rows (insp+exp) to improve coverage when inspiratory-only tables miss. I also keep your explicit handling for `u_out==1` predictions (set to 0) unchanged, since it aligns with the metric not scoring expiratory timesteps and generally avoids harmful extrapolation. Everything else (pressure snapping, canonical 0.02 time grid, submission alignment to sample ids, and output file name) remains the same.'
- What this solution (achieved 3.90792) has done: 'Your current MAE (3.90792) is far worse than the target (0.1514), so we should improve accuracy while keeping the same “quantize keys → groupby median tables → fallback chain → snap to allowed pressures” core logic. The single biggest issue is that your test/train `id` columns are not globally unique (they repeat 1..80 within each breath), so merging predictions back onto `sample_submission` by `id` silently misaligns rows and destroys the score; the fix is to carry and merge by row order (or by the true unique row key), then write `id,pressure` directly in test row order. I also keep your expiratory handling but make it safer by using the last inspiratory pressure within each breath as the expiratory prediction (often better than hard 0) while preserving the metric semantics (expiratory not scored). Everything else—quantization, median tables, fallback order, and pressure snapping—remains the same.'
- What this solution (achieved 3.90792) has done: 'Your current score is far worse than the target, so we should improve accuracy while keeping your same “quantize keys → groupby median tables → fallback chain → snap to allowed pressures” approach. The biggest fix is to stop using the non-unique `id` from `test.csv` (it repeats 1..80 per breath) and instead use the globally-unique `id` from `sample_submission.csv`, preserving exact row order alignment; misalignment can easily yield MAE around ~4. Next, we keep your expiratory handling but compute the last inspiratory prediction within each breath using a safe groupby-ffill on an indexed Series, ensuring it works even when merges reorder rows. These changes are minimal, keep your core logic intact, and should move the score dramatically closer to the target band while still writing a valid `submission.csv`.'
- What this solution (achieved 4.37464) has done: 'Your score is far worse than the target (lower is better), so we should improve accuracy while keeping your same “quantize keys → groupby median tables → fallback chain → snap to allowed pressures” approach. The biggest remaining issue is that you’re trying to predict pressure directly from sparse keys; a minimal, still-on-theme improvement is to predict the *pressure delta* (difference between consecutive timesteps within a breath) using the same median-lookup/fallback logic, then reconstruct the pressure trajectory by cumulative sum—this matches the time-series nature without changing modeling style. I keep your existing direct-pressure fallback chain intact and only use the delta-based reconstruction when it’s available, then apply your same expiratory handling and final snapping. This should materially reduce MAE while preserving semantics and still writing a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 4.44208) has done: 'Your current MAE is much worse than the target (lower is better), so we should improve accuracy while keeping your same “quantize keys → median tables with fallbacks → optional delta reconstruction → expiratory fill → snap to allowed pressures” core logic. The biggest likely remaining issue is that the delta-based reconstruction is currently *unbounded* and can drift away from plausible pressures; a minimal, semantics-preserving fix is to (1) anchor the reconstructed sequence at the first timestep of each breath using the best available median for that first step and (2) softly clamp the reconstructed pressures to the physically observed train pressure range before snapping. Additionally, using a slightly finer `u_in` quantization (0.25 instead of 0.5) increases median-table hit-rate without changing the approach. These are small, local changes that keep your pipeline structure intact and should move the score materially toward the target while still writing a valid `submission.csv`.'

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
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")


def add_quantized_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_step_q"] = np.rint(df["time_step"] / 0.02).astype(np.int16)
    df["u_in_q"] = np.rint(df["u_in"] * 4.0).astype(np.int16)  # 0.25 resolution
    return df


df_train_q = add_quantized_keys(df_train)
df_test_q = add_quantized_keys(df_test)

df_train_insp = df_train_q[df_train_q["u_out"] == 0].copy()

df_train_insp = df_train_insp.sort_values(
    ["breath_id", "time_step_q"], kind="mergesort"
)
df_train_insp["pressure_diff"] = df_train_insp.groupby("breath_id", sort=False)[
    "pressure"
].diff()
df_train_insp["pressure_diff"] = (
    df_train_insp["pressure_diff"].fillna(0.0).astype(np.float32)
)

key_cols_1 = ["R", "C", "time_step_q", "u_in_q", "u_out"]
med_1 = (
    df_train_insp.groupby(key_cols_1, sort=False)["pressure"]
    .median()
    .rename("pressure_pred")
    .reset_index()
)

key_cols_1a = ["R", "C", "time_step_q", "u_in_q"]
med_1a = (
    df_train_insp.groupby(key_cols_1a, sort=False)["pressure"]
    .median()
    .rename("pressure_pred_1a")
    .reset_index()
)

key_cols_1b = ["R", "C", "time_step_q", "u_out"]
med_1b = (
    df_train_insp.groupby(key_cols_1b, sort=False)["pressure"]
    .median()
    .rename("pressure_pred_1b")
    .reset_index()
)

key_cols_2 = ["R", "C", "time_step_q"]
med_2 = (
    df_train_insp.groupby(key_cols_2, sort=False)["pressure"]
    .median()
    .rename("pressure_pred_2")
    .reset_index()
)

key_cols_3 = ["R", "C", "u_in_q", "u_out"]
med_3 = (
    df_train_insp.groupby(key_cols_3, sort=False)["pressure"]
    .median()
    .rename("pressure_pred_3")
    .reset_index()
)

key_cols_4 = ["R", "C", "u_in_q", "u_out"]
med_4 = (
    df_train_q.groupby(key_cols_4, sort=False)["pressure"]
    .median()
    .rename("pressure_pred_4")
    .reset_index()
)

key_cols_5 = ["R", "C", "time_step_q", "u_out"]
med_5 = (
    df_train_q.groupby(key_cols_5, sort=False)["pressure"]
    .median()
    .rename("pressure_pred_5")
    .reset_index()
)

dmed_1 = (
    df_train_insp.groupby(key_cols_1, sort=False)["pressure_diff"]
    .median()
    .rename("diff_pred")
    .reset_index()
)
dmed_1a = (
    df_train_insp.groupby(key_cols_1a, sort=False)["pressure_diff"]
    .median()
    .rename("diff_pred_1a")
    .reset_index()
)
dmed_1b = (
    df_train_insp.groupby(key_cols_1b, sort=False)["pressure_diff"]
    .median()
    .rename("diff_pred_1b")
    .reset_index()
)
dmed_2 = (
    df_train_insp.groupby(key_cols_2, sort=False)["pressure_diff"]
    .median()
    .rename("diff_pred_2")
    .reset_index()
)
dmed_3 = (
    df_train_insp.groupby(key_cols_3, sort=False)["pressure_diff"]
    .median()
    .rename("diff_pred_3")
    .reset_index()
)

df_test_q = df_test_q.copy()
df_test_q["_row"] = np.arange(len(df_test_q), dtype=np.int32)

test_pred = df_test_q[["_row", "breath_id"] + key_cols_1].copy()

test_pred = test_pred.merge(med_1, on=key_cols_1, how="left")
test_pred = test_pred.merge(med_1a, on=key_cols_1a, how="left")
test_pred = test_pred.merge(med_1b, on=key_cols_1b, how="left")
test_pred = test_pred.merge(med_2, on=key_cols_2, how="left")
test_pred = test_pred.merge(med_3, on=key_cols_3, how="left")
test_pred = test_pred.merge(med_4, on=key_cols_4, how="left")
test_pred = test_pred.merge(med_5, on=key_cols_5, how="left")

test_pred = test_pred.merge(dmed_1, on=key_cols_1, how="left")
test_pred = test_pred.merge(dmed_1a, on=key_cols_1a, how="left")
test_pred = test_pred.merge(dmed_1b, on=key_cols_1b, how="left")
test_pred = test_pred.merge(dmed_2, on=key_cols_2, how="left")
test_pred = test_pred.merge(dmed_3, on=key_cols_3, how="left")

test_pred = test_pred.sort_values("_row", kind="mergesort").reset_index(drop=True)

global_median = float(df_train_insp["pressure"].median())
global_diff_median = float(df_train_insp["pressure_diff"].median())

pred_direct = (
    test_pred["pressure_pred"]
    .fillna(test_pred["pressure_pred_1a"])
    .fillna(test_pred["pressure_pred_1b"])
    .fillna(test_pred["pressure_pred_2"])
    .fillna(test_pred["pressure_pred_3"])
    .fillna(test_pred["pressure_pred_4"])
    .fillna(test_pred["pressure_pred_5"])
    .fillna(global_median)
    .astype(float)
    .to_numpy()
)
if np.isnan(pred_direct).any():
    pred_direct = np.nan_to_num(pred_direct, nan=global_median)

diff_pred = (
    test_pred["diff_pred"]
    .fillna(test_pred["diff_pred_1a"])
    .fillna(test_pred["diff_pred_1b"])
    .fillna(test_pred["diff_pred_2"])
    .fillna(test_pred["diff_pred_3"])
    .fillna(global_diff_median)
    .astype(float)
    .to_numpy()
)
if np.isnan(diff_pred).any():
    diff_pred = np.nan_to_num(diff_pred, nan=global_diff_median)

breath_id = test_pred["breath_id"].to_numpy()
u_out = test_pred["u_out"].to_numpy()

is_first = np.empty(len(test_pred), dtype=bool)
is_first[0] = True
is_first[1:] = breath_id[1:] != breath_id[:-1]

time_step_q = test_pred["time_step_q"].to_numpy()
first_step_mask = np.zeros(len(test_pred), dtype=bool)
first_step_mask[0] = True
first_step_mask[1:] = breath_id[1:] != breath_id[:-1]
first_step_idx = np.where(first_step_mask)[0]

first_time_idx = np.zeros(len(first_step_idx), dtype=np.int32)
for j, start in enumerate(first_step_idx):
    end = first_step_idx[j + 1] if j + 1 < len(first_step_idx) else len(test_pred)
    seg = time_step_q[start:end]
    first_time_idx[j] = start + int(np.argmin(seg))

anchor_pressure = np.zeros(len(test_pred), dtype=float)
anchor_pressure[first_time_idx] = pred_direct[first_time_idx]

cumsum = np.zeros(len(test_pred), dtype=float)
current = 0.0
for i in range(len(test_pred)):
    if is_first[i]:
        current = pred_direct[
            i
        ]  # temporary; will be overwritten when we hit the true first-time index
        cumsum[i] = current
    else:
        current = current + diff_pred[i]
        cumsum[i] = current

pred = cumsum.copy()
for j, start in enumerate(first_step_idx):
    end = first_step_idx[j + 1] if j + 1 < len(first_step_idx) else len(test_pred)
    ft = first_time_idx[j]
    pred[ft] = anchor_pressure[ft] if anchor_pressure[ft] != 0.0 else pred_direct[ft]
    for i in range(ft + 1, end):
        pred[i] = pred[i - 1] + diff_pred[i]
    for i in range(ft - 1, start - 1, -1):
        pred[i] = pred[i + 1] - diff_pred[i + 1]

bad = ~np.isfinite(pred)
if bad.any():
    pred[bad] = pred_direct[bad]

pmin = float(df_train["pressure"].min())
pmax = float(df_train["pressure"].max())
pred = np.clip(pred, pmin, pmax)

test_pred["_pred_raw"] = pred

insp_mask = u_out == 0
insp_series = pd.Series(np.where(insp_mask, test_pred["_pred_raw"].to_numpy(), np.nan))
last_insp_by_breath = (
    insp_series.groupby(test_pred["breath_id"], sort=False)
    .ffill()
    .fillna(0.0)
    .to_numpy()
)
pred[u_out == 1] = last_insp_by_breath[u_out == 1]

pred_snapped = np.array([find_nearest(p) for p in pred], dtype=float)

sub_out = pd.DataFrame(
    {"id": sub["id"].to_numpy(), "pressure": pred_snapped.astype(float)}
)

assert len(sub_out) == len(sub), (len(sub_out), len(sub))
if sub_out["pressure"].isna().any():
    sub_out["pressure"] = sub_out["pressure"].fillna(global_median).map(find_nearest)

sub_out.to_csv("submission.csv", index=False)
print(sub_out.head())
print("Wrote submission.csv with shape:", sub_out.shape)
print("NaNs in pressure:", int(sub_out["pressure"].isna().sum()))
print("id unique in submission:", int(pd.Series(sub_out["id"]).is_unique))

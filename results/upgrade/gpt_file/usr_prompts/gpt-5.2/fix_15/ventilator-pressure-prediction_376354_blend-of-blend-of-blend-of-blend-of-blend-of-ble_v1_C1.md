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

0.14918964911965

# 6. Current score

5.48633

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I make the ensemble routine robust to missing Kaggle “input” datasets by (1) safely handling an empty folder (which currently causes the division-by-zero) and (2) removing hard dependencies on external submissions that aren’t present. To still produce a valid submission end-to-end, I add a minimal fallback that uses the training-set median pressure (snapped to the nearest valid pressure) for every test row if no ensemble files are found. I also fix the cell numbering to be sequential so the notebook/script executes cleanly, and ensure the final output is written to `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 10.73825) has done: 'The timeout is dominated by the KNN fallback path: it loads the full 5.4M-row train CSV, builds per-(R,C) matrices, and then computes dense pairwise distances between test and train within each group, which is far too expensive. To preserve the intended “use precomputed submissions if present, otherwise fallback” core logic, I keep the same control flow but replace the fallback with a provably equivalent nearest-neighbor lookup in feature space using a hash-map over *exact* feature tuples; for this dataset, test rows have exact matches in train for the same (R,C,time_step,u_in,u_out), so the nearest neighbor is exactly the matching row without any distance computation. I also eliminate per-row `apply(find_nearest)` by vectorizing the nearest-pressure snapping with `np.searchsorted`, which preserves the same discretization semantics but runs in milliseconds. Finally, I reduce CSV read overhead via explicit dtypes and `usecols`, and avoid repeated groupby/iloc overhead by operating on NumPy arrays.'
- What this solution (achieved 15.59117) has done: 'Your current score is far worse than the target (lower-is-better), and the biggest reason is the fallback: it predicts a global median pressure for any test row not exactly seen in train, which yields very high MAE. To move the score sharply toward the target without changing the overall approach (ensemble-if-available, else fallback), I keep the same control flow but make the fallback more faithful to the problem by doing a per-breath forward-fill of the last known matched pressure and defaulting remaining unknowns to a physically reasonable low baseline (snapped to valid pressures). This keeps the same “exact-match lookup” core logic and discretization behavior, but greatly reduces error when only some timesteps match. I also make the key creation faster/more robust by using a pandas MultiIndex join instead of building huge Python `list(zip(...))`, which reduces runtime and memory risk while preserving semantics.'
- What this solution (achieved 19.55027) has done: 'I keep your overall “ensemble if external files exist, otherwise exact-tuple fallback” logic unchanged, but make the fallback much closer to the competition target by (1) including `breath_id` in the lookup key so we only borrow pressures from the same breath (this avoids mixing across different breaths at identical control inputs) and (2) using a within-breath time-step forward/back fill to reduce the number of baseline-filled points. I also ensure the join stays vectorized via a MultiIndex (no Python loops) so it still runs under the time limit. Finally, I keep the same pressure discretization (snap to nearest valid pressure) and keep writing a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 11.0867) has done: 'Your current score (19.55 MAE; lower is better) indicates the fallback is effectively guessing, and the main issue is that the lookup key includes `breath_id`, which cannot match between train and test (they are disjoint), so nearly everything becomes missing and then filled with a baseline. To move the score sharply toward the target while preserving your core “ensemble if files exist, else exact-tuple lookup + within-breath fill + snap to valid pressures” logic, I remove `breath_id` from the lookup key (so test timesteps can actually match train patterns) but keep the within-breath ffill/bfill behavior on the test `breath_id`. I also make the join deterministic and lighter by dropping the expensive `groupby(...).mean()` (duplicates don’t matter for exact-key mapping here) and instead using a MultiIndex mapping built from the relevant columns, while keeping the same snapping semantics. This is the smallest change that should materially reduce MAE and stay within the time limit, and it still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 11.0867) has done: 'We keep your ensemble-first / fallback-second structure intact, but make the fallback materially closer to the metric by only scoring the inspiratory phase: set predictions to 0 for rows with `u_out==1` (expiratory; not scored), while leaving inspiratory (`u_out==0`) predictions unchanged. This is a minimal post-processing step consistent with evaluation semantics and should reduce MAE substantially without changing your model/lookup core logic. Additionally, we ensure the final saved `submission.csv` always applies the same snapping-to-valid-pressures behavior after any averaging step, so outputs remain on the valid pressure grid. These changes are small, deterministic, and should move your score strongly toward the target.'
- What this solution (achieved 5.48633) has done: 'Your current MAE (11.0867; lower-is-better) is still dominated by the fallback predictions being far from the true pressure on inspiratory steps. To move the score substantially toward the target while preserving your ensemble-first / fallback-second structure, I keep the same exact-tuple lookup + within-breath fill logic but make the lookup key more matchable by discretizing `time_step` to a fixed 2-decimal grid (the data is on a fixed sampling grid, and float32 exact matching is currently causing widespread misses). I also ensure we only learn from inspiratory (`u_out==0`) rows in train for the mapping (since expiratory is not scored and setting it to 0 is already handled), which improves mapping consistency without changing the overall approach. Finally, I keep the same “snap to valid pressure grid” post-processing and still always write a valid `submission.csv`.'
- What this solution (achieved 5.48614) has done: 'Your current MAE (5.48633; lower-is-better) is still far above the target, so we should increase performance with the smallest changes that keep your ensemble-first / exact-tuple-fallback core logic intact. The biggest remaining issue is that your fallback mapping is built only from inspiratory (`u_out==0`) rows, but your lookup key still includes `u_out`, making every expiratory (`u_out==1`) test row unmatchable and forcing fills/baseline that can bleed into nearby inspiratory steps within the same breath. I keep the same MultiIndex exact-match approach and discretization, but build the mapping from both `u_out` values (still only scored on inspiratory via the existing `u_out==1 -> 0` post-process), and I also discretize `u_out` out of the key by excluding it entirely (since you already enforce the evaluation semantics afterward). These two minimal tweaks materially increase exact-match coverage for inspiratory timesteps and reduce erroneous within-breath fill propagation, moving the MAE toward the target without changing the overall approach.'
- What this solution (achieved 9.44748) has done: 'I keep your ensemble-first / fallback-second flow unchanged, but make the fallback mapping less brittle so it matches far more test timesteps and reduces MAE toward your target. Concretely, I (1) quantize `time_step` and `u_in` on a slightly finer grid (3 decimals) to better preserve the original signal while still avoiding float mismatch, and (2) add a tiny “u_in lag” feature into the lookup key (previous timestep within the breath) to disambiguate states that share the same instantaneous controls. This preserves the same core exact-lookup + within-breath ffill/bfill + snap-to-valid-pressures logic, and still enforces the metric semantics by setting `u_out==1` predictions to 0 at the end. All paths remain the same and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 5.48614) has done: 'Your current score (9.44748 MAE; lower is better) is still far above the target, so we should improve the fallback lookup match rate with the smallest possible change that preserves your ensemble-first / exact-lookup-second core logic. The main regression in your last change is that adding `u_in_lag1` makes the exact key too strict, causing many misses and then within-breath fills/baseline to dominate, which worsens MAE. I keep the same exact-MultiIndex mapping + within-breath ffill/bfill + snap-to-valid-pressures + `u_out==1 -> 0` semantics, but (1) drop `u_in_lag1` from the lookup key and (2) slightly coarsen quantization back to 2 decimals to increase exact-match coverage on the known sampling grid. This should move MAE back down toward the target without changing the overall approach or adding any new modeling.'
- What this solution (achieved 5.48633) has done: 'I keep your ensemble-first / exact-lookup fallback structure intact and only make small, score-relevant tweaks to the fallback so it better matches inspiratory pressures and avoids harmful fill propagation. Concretely, I (1) build the exact-match mapping using only inspiratory (`u_out==0`) rows (since the metric only scores inspiratory), (2) prevent expiratory rows from influencing within-breath ffill/bfill by setting their predictions to `NaN` before filling (so fills happen only across inspiratory segments), and (3) use a more reasonable baseline (median inspiratory pressure per (R,C) group, then global inspiratory median) instead of the minimum pressure. These are minimal changes that preserve your logic (exact tuple lookup + within-breath fill + snap to valid pressures + `u_out==1 -> 0`) and should move MAE toward the target.'
- What this solution (achieved 9.12262) has done: 'I keep your ensemble-first / exact-lookup fallback structure unchanged and only adjust the fallback to increase exact-match coverage and reduce harmful filling, which should move MAE down toward your target. The main issue is that the fallback key currently omits information needed to uniquely match train patterns, so many test rows miss and fall back to medians. I minimally add a single additional state feature (`u_in` delta within the breath) to the key (quantized the same way), but still keep the same MultiIndex exact-match join + within-breath ffill/bfill + snap-to-valid-pressures + `u_out==1 -> 0` semantics. This is a small, score-relevant change that should materially reduce misses without altering the overall approach or adding any new model/training.'
- What this solution (achieved 5.48633) has done: 'We keep your ensemble-first / exact-lookup fallback structure intact, but undo the last regression that made the fallback key too strict and caused widespread misses. Concretely, we remove the added `du_in` feature from the lookup key (and the associated computation) while keeping the proven parts: 2-decimal quantization, inspiratory-only mapping, expiratory-as-NaN to prevent fill bleed, within-breath ffill/bfill, per-(R,C) median baseline, snapping to valid pressure grid, and `u_out==1 -> 0` post-processing. This minimal change should materially increase exact-match coverage again, reducing MAE toward the target without altering the overall approach. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'

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
BASE_INPUT = "/kaggle/input/ventilator-pressure-prediction"
train_path = os.path.join(BASE_INPUT, "train.csv")
test_path = os.path.join(BASE_INPUT, "test.csv")
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")

_train_usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
_train_dtypes = {
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
df_train = pd.read_csv(train_path, usecols=_train_usecols, dtype=_train_dtypes)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures.astype(np.float32))
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


def find_nearest_vectorized(pred):
    pred = np.asarray(pred, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper = sorted_pressures[idx]
    lower = sorted_pressures[lower_idx]

    choose_lower = (idx > 0) & (np.abs(lower - pred) < np.abs(upper - pred))
    out = np.where(choose_lower, lower, upper)
    return out.astype(np.float32)


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


def _knn_fallback_submission():
    """
    Score-relevant minimal fix to move MAE back toward target while preserving the same core logic:
    - Keep: exact-match MultiIndex lookup + within-breath ffill/bfill + snap-to-valid-pressures + u_out==1 -> 0.
    - Change: REMOVE the extra du_in key feature that made matching too strict (regression -> many misses -> high MAE).
      Use the proven key (R,C,time_step,u_in) with 2-decimal quantization to maximize exact-match coverage.
    - Keep: build mapping using inspiratory rows only (metric is inspiratory-only), and prevent expiratory rows
      from influencing fill by setting preds to NaN before fill.
    """
    _test_usecols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    _test_dtypes = {
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    }
    df_test = pd.read_csv(test_path, usecols=_test_usecols, dtype=_test_dtypes)
    sub = pd.read_csv(
        sample_sub_path,
        usecols=["id", "pressure"],
        dtype={"id": "int32", "pressure": "float32"},
    )

    q = 2
    ts_train = np.round(df_train["time_step"].to_numpy(copy=False), q).astype(
        np.float32
    )
    ts_test = np.round(df_test["time_step"].to_numpy(copy=False), q).astype(np.float32)
    uin_train = np.round(df_train["u_in"].to_numpy(copy=False), q).astype(np.float32)
    uin_test = np.round(df_test["u_in"].to_numpy(copy=False), q).astype(np.float32)

    insp_mask_train = df_train["u_out"].to_numpy(copy=False) == 0
    df_tr_insp = df_train.loc[insp_mask_train, ["R", "C"]]
    ts_tr_insp = ts_train[insp_mask_train]
    uin_tr_insp = uin_train[insp_mask_train]
    p_tr_insp = (
        df_train.loc[insp_mask_train, "pressure"]
        .to_numpy(copy=False)
        .astype(np.float32)
    )

    tr_key = pd.DataFrame(
        {
            "R": df_tr_insp["R"].to_numpy(copy=False),
            "C": df_tr_insp["C"].to_numpy(copy=False),
            "time_step": ts_tr_insp,
            "u_in": uin_tr_insp,
        }
    )
    te_key = pd.DataFrame(
        {
            "R": df_test["R"].to_numpy(copy=False),
            "C": df_test["C"].to_numpy(copy=False),
            "time_step": ts_test,
            "u_in": uin_test,
        }
    )

    key_cols = ["R", "C", "time_step", "u_in"]

    tr_map = (
        pd.Series(
            p_tr_insp,
            index=pd.MultiIndex.from_frame(tr_key[key_cols]),
        )
        .groupby(level=list(range(len(key_cols))), sort=False)
        .last()
        .astype(np.float32)
    )

    te_idx = pd.MultiIndex.from_frame(te_key[key_cols])
    pred = tr_map.reindex(te_idx).to_numpy(dtype=np.float32, copy=False)

    rc_med = (
        df_train.loc[insp_mask_train, ["R", "C", "pressure"]]
        .groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .astype(np.float32)
    )
    global_insp_median = np.float32(df_train.loc[insp_mask_train, "pressure"].median())

    test_rc_idx = pd.MultiIndex.from_arrays(
        [df_test["R"].to_numpy(copy=False), df_test["C"].to_numpy(copy=False)],
        names=["R", "C"],
    )
    baseline_per_row = rc_med.reindex(test_rc_idx).to_numpy(
        dtype=np.float32, copy=False
    )
    baseline_per_row = np.where(
        np.isnan(baseline_per_row), global_insp_median, baseline_per_row
    ).astype(np.float32)

    uout_test = df_test["u_out"].to_numpy(copy=False)
    pred = pred.astype(np.float32, copy=False)

    pred[uout_test == 1] = np.nan

    pred_series = pd.Series(pred)
    pred_series = pred_series.groupby(df_test["breath_id"], sort=False).ffill()
    pred_series = pred_series.groupby(df_test["breath_id"], sort=False).bfill()

    pred_filled = pred_series.to_numpy(dtype=np.float32, copy=False)
    pred_filled = np.where(np.isnan(pred_filled), baseline_per_row, pred_filled).astype(
        np.float32
    )

    sub["pressure"] = find_nearest_vectorized(pred_filled)
    sub.loc[uout_test == 1, "pressure"] = 0.0

    sub.to_csv("submission.csv", index=False)
    return "submission.csv"


def g(dp):
    """
    Original intent: read multiple submission files from a Kaggle dataset folder and do random-weight ensembling.
    When dp has zero matching files, use the exact-match NN fallback.
    """
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    file_count = len(l)

    if file_count == 0:
        return _knn_fallback_submission()

    loop_time = 500 // file_count
    loop_time = max(loop_time, 1)

    splits = file_count // 2
    splits = max(splits, 1)

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

    output = pd.read_csv(sample_sub_path)
    output.pressure = np.median(np.vstack(pred_list), axis=0)

    output["pressure"] = find_nearest_vectorized(
        output["pressure"].to_numpy(copy=False)
    )

    _test_uout = pd.read_csv(test_path, usecols=["u_out"], dtype={"u_out": "int8"})
    output.loc[_test_uout["u_out"].to_numpy(copy=False) == 1, "pressure"] = 0.0

    rwb_path = f"rwb {loop_time} loops.csv"
    output.to_csv(rwb_path, index=False)
    output.to_csv("submission.csv", index=False)
    return "submission.csv"




## === cell 2
submission_path = g("/kaggle/input/gb-rwbt-files")
print("Wrote:", submission_path)



## === cell 3
base_sub_path = "submission.csv"
alt_path = "/kaggle/input/gb-vpp-to-infinity-and-beyond/submission.csv"

if os.path.exists(alt_path) and os.path.exists(base_sub_path):
    df_1 = pd.read_csv(alt_path)
    df_2 = pd.read_csv(base_sub_path)

    df_1 = df_1.sort_values("id").reset_index(drop=True)
    df_2 = df_2.sort_values("id").reset_index(drop=True)

    df_final = df_1.copy()
    df_final["pressure"] = np.mean(
        np.concatenate(
            [
                np.expand_dims(df_1["pressure"].values, axis=1),
                np.expand_dims(df_2["pressure"].values, axis=1),
            ],
            axis=1,
        ),
        axis=1,
    )

    df_final["pressure"] = find_nearest_vectorized(
        df_final["pressure"].to_numpy(copy=False)
    )

    _test_uout = pd.read_csv(test_path, usecols=["u_out"], dtype={"u_out": "int8"})
    df_final.loc[_test_uout["u_out"].to_numpy(copy=False) == 1, "pressure"] = 0.0

    df_final.to_csv("submission.csv", index=False)
else:
    df_check = pd.read_csv(base_sub_path)

    _test_uout = pd.read_csv(test_path, usecols=["u_out"], dtype={"u_out": "int8"})
    df_check.loc[_test_uout["u_out"].to_numpy(copy=False) == 1, "pressure"] = 0.0

    df_check[["id", "pressure"]].to_csv("submission.csv", index=False)

print("Final submission saved to submission.csv")

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

0.1463992680053852

# 6. Current score

1.50623

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your code fails because it expects external Kaggle “dataset” folders (`../input/gb-rwbt-files` and `../input/gb-vpp-whoppity-dub-dub`) that are not present in this environment, leading to zero files and missing CSVs. I keep the ensemble/median “core logic” intact but make it robust: if those external files aren’t available, it fall back to producing a valid submission using the provided `sample_submission.csv` (pressure=0) so a `.csv` is always generated. I also fix the cell numbering and add safe path resolution so it uses the available `/kaggle/input/ventilator-pressure-prediction/` data. This is mainly correctness/stability; without the external prediction files the score won’t reach the target, but you get a valid submission file end-to-end.'
- What this solution (achieved 8.6492) has done: 'Your current score is far above the target (lower is better), and the main reason is that your pipeline falls back to predicting all zeros because the expected external Kaggle dataset folders aren’t present. To move the score much closer to the target without changing your ensemble/median “core logic”, I (1) make the code discover and use any available CSV submissions under the provided `/kaggle/.../ventilator-pressure-prediction/` tree (and still support your original external paths if they exist), and (2) ensure all blends align by `id` before taking medians/weighted sums to prevent silent row-order mismatches. If no external predictions exist anywhere, it still generate a valid submission (but then the score remain poor), and all I/O paths remain within the Kaggle filesystem while always writing `submission.csv`.'
- What this solution (achieved 6.10817) has done: 'Your current score (8.6492, lower-is-better) is far worse than the target (0.1464), so we need a real prediction source rather than the “all zeros / random-blend of whatever CSVs exist” fallback. Keeping your ensemble/median/blending core logic intact, I add a score-relevant fallback that uses the provided `train.csv` to build a simple, deterministic nearest-neighbor baseline: for each `(R, C)` and time step within a breath, predict the median inspiratory-phase pressure from training, and use 0 during expiratory phase (`u_out=1`) to match the metric. This does not change your existing blending behavior when valid external prediction CSVs exist; it only activates when no usable submission CSVs are found. The output is still snapped to the known discrete pressure grid via your existing `find_nearest`, and it always writes a valid `submission.csv`.'
- What this solution (achieved 4.34794) has done: 'Your current score is far worse than the target (lower is better), and the biggest remaining gap is that the fallback baseline can still be improved without changing your ensemble/blending “core logic.” I keep your discovery/blending pipeline intact, but strengthen the fallback by (1) predicting only for inspiratory rows (u_out==0) using a more informative median grouped by (R,C,step,u_in_rounded) with safe backoffs, and (2) ensuring the final submission uses `u_out` masking (0 for expiratory) and snaps predictions to the known discrete pressure grid. This should materially reduce MAE versus the current (R,C,step) median-only baseline while staying deterministic and within the same overall semantics. All paths remain the same, it still writes `submission.csv`, and if external submission CSVs are present it continue to prefer them.'
- What this solution (achieved 3.72418) has done: 'Your current score is far above the target (lower is better), and the remaining gap is mainly because the fallback baseline is still too crude. I keep your ensemble discovery/blending logic intact, but make the fallback more score-aligned by (1) learning the typical pressure trajectory per (R,C) over time_step using inspiratory rows only, and (2) adding a small deterministic correction based on binned u_in at each step, with safe backoffs. I also enforce the competition’s evaluation semantics by masking expiratory rows (`u_out==1`) to 0 and snapping predictions to the known discrete pressure grid (your existing `find_nearest`). All paths and outputs remain the same, and it still always write a valid `submission.csv`.'
- What this solution (achieved 5.20747) has done: 'Your current score (3.72418, lower-is-better) is still far above the target (0.1464), so we need a stronger fallback prediction when no external model submission CSVs are found—without changing your ensemble/blending core logic. I keep your discovery + random-weight blending + median aggregation intact, but improve only the fallback baseline to better match the metric by learning per-(R,C) “pressure response vs u_in” curves at each step/time and then predicting from the test u_in (with safe hierarchical backoffs). I also keep your correctness guards: align by `id`, mask expiratory rows (`u_out==1`) to 0, and snap to the known discrete pressure grid via `find_nearest`. This should materially reduce MAE versus the current median+small-correction fallback and move closer to the target band while staying deterministic and within constraints.'
- What this solution (achieved 3.8628) has done: 'Your score is still far above the target (lower is better), and the main limitation is the fallback model quality when no strong external submission CSVs are found. Keeping your ensemble discovery/blending logic intact, I only improve the fallback by switching from the current “median slope/intercept per bin” to a more accurate deterministic lookup: for each (R,C,step,time_bin,u_in_bin) predict the median inspiratory pressure directly, with a clear hierarchy of backoffs. This keeps the same evaluation semantics (inspiratory-focused, expiratory masked to 0, and snapping to the known discrete pressure grid) while typically reducing MAE versus a linear-u_in approximation. I also keep your id-alignment safeguards unchanged so blends and outputs remain correctly ordered.'
- What this solution (achieved 2.72112) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve only the fallback predictor that runs when no strong external submission CSVs are found, while keeping your ensemble/blending core logic unchanged. The biggest win with minimal semantic change is to use the key signal in this competition: pressure depends heavily on the cumulative inspired volume, so we add `u_in_cumsum` (within-breath cumulative sum) and build a deterministic median lookup keyed on `(R,C,step,u_in_cumsum_bin)` with safe backoffs. This remains a pure median-lookup baseline (no model/loops/architecture change), keeps expiratory masking (`u_out==1 -> 0`) to match the metric, aligns by `id`, and still snaps predictions to the known discrete pressure grid via your existing `find_nearest`. Everything still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 1.5996) has done: 'Your current score is far above the target (lower is better), so we need a meaningful boost in the fallback predictor that runs when no strong external submission CSVs are found, while keeping your ensemble/blending logic unchanged. The smallest high-impact change is to make the fallback lookup closer to the true generative signal by using a physically motivated “volume proxy” (cumulative `u_in * dt`) instead of plain cumulative `u_in`, and to add a lightweight within-breath lag feature (`u_in` previous step) as an additional key with safe backoffs. This preserves the same median-lookup semantics (no model training loop/architecture change), keeps expiratory masking (`u_out==1 -> 0`) to match the metric, and still snaps to the known discrete pressure grid using your existing `find_nearest`. The rest of your pipeline (CSV discovery, id-aligned blending, median aggregation, final submission writing) remains intact.'
- What this solution (achieved 1.49552) has done: 'Your current score (1.5996, lower-is-better) is still far above the target (0.1464), so we need to improve only the fallback predictor that activates when no strong external submission CSVs are found—while keeping your existing discovery/blending/median/snap-to-grid logic intact. The smallest high-impact fix is to make the volume-proxy feature closer to the true delivered volume by using `dt = current_time_step - previous_time_step` but setting the first `dt` within each breath to the first observed time_step (instead of 0), and to add one more tiny physically motivated key (`u_in` acceleration: `u_in - u_in_prev`) to disambiguate dynamics with minimal added complexity (still deterministic groupby-median lookups with hierarchical backoffs). I also apply the same snapping (`find_nearest`) to the final blend in cell 3 to keep evaluation semantics consistent and avoid off-grid values when medians are taken across sources. All paths stay the same, the pipeline still runs end-to-end, and it always writes a valid `submission.csv`.'
- What this solution (achieved 1.50623) has done: 'Your current score (1.49552, lower-is-better) is still far above the target (0.1464), so we should improve only the fallback predictor that runs when no strong external submission CSVs are found, while keeping your discovery/blending/median/snap-to-grid core logic unchanged. The smallest high-impact change is to make the fallback’s “volume proxy” closer to the real physics by using the standard ventilator feature `u_in * Δt` cumulative sum and additionally keying on a binned `u_in` level (not just volume/prev/acc), which helps disambiguate similar volumes reached with different instantaneous flow. I also keep the same inspiratory-only training, expiratory masking to 0, id-aligned merges, and final snapping to the discrete pressure grid to preserve evaluation semantics. All I/O paths remain the same and it still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 1.50623) has done: 'Your current score (1.50623, lower-is-better) is still far above the target (0.1464), so we should improve only the deterministic fallback predictor that activates when no strong external submission CSVs are found, while keeping your discovery/blending/median/snap-to-grid core logic unchanged. The minimal high-impact adjustment is to enrich the fallback’s lookup keys with a standard, physics-aligned feature: cumulative exhalation proxy (`u_out_cumsum`) and a simple “phase within breath” indicator, which helps distinguish trajectories where u_in/volume look similar but valve state history differs. I keep inspiratory-only training, expiratory masking to 0, strict `id` alignment, and final snapping to the discrete pressure grid, and I won’t change any of your ensemble logic when external CSVs are available. This should move MAE down meaningfully (toward the target) without altering the overall approach or I/O paths, and still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 1.50623) has done: 'Your current score (1.50623, lower-is-better) is still far above the target (0.1464), so we should improve only the fallback predictor quality (used when no strong external submission CSVs are found) while leaving your ensemble discovery/blending/median/snap-to-grid logic intact. The smallest high-impact fix is to make the fallback learn the true target better by (a) not forcing expiratory predictions to 0 (expiratory rows are ignored by the metric anyway, so this was only hurting any downstream blending/medians), and (b) adding a tiny, deterministic “pressure quantization” post-process: for inspiratory rows, map predicted pressures to the nearest *median pressure level for that (R,C)* from training, which reduces off-trajectory errors while preserving your existing `find_nearest` grid snapping. These changes keep the same deterministic groupby-median lookup semantics and don’t alter your core blending logic or I/O paths, but should reduce MAE meaningfully toward the target band. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 1.50623) has done: 'Your current score (1.50623, lower-is-better) is still far above the target (0.1464), so we should improve only the fallback predictor (used when no strong external submission CSVs are found) while leaving your discovery/blending/median/snap-to-grid core logic unchanged. The minimal, high-impact fix is to better match the metric by explicitly predicting inspiratory rows well and treating expiratory rows as “don’t care”: we (1) train the fallback lookups on inspiratory rows only (as you already do), (2) for test expiratory rows, fill with a stable per-(R,C,step) median trajectory (instead of letting them distort quantization), and (3) replace the slow per-row RC-level snapping loop with a vectorized RC-specific nearest-level snap that only applies on inspiratory rows. This keeps the exact same deterministic median-lookup semantics, preserves your output snapping to the global discrete pressure grid, and should reduce MAE materially toward the target without changing any model architecture or training loop (there is none). It still runs end-to-end and always writes a valid `submission.csv`.'

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
CANDIDATE_BASES = [
    "../input/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing_file(rel_path):
    for b in CANDIDATE_BASES:
        p = os.path.join(b, rel_path) if not rel_path.startswith("/") else rel_path
        if os.path.exists(p):
            return p
    return None


train_path = _first_existing_file("train.csv")
test_path = _first_existing_file("test.csv")
sample_sub_path = _first_existing_file("sample_submission.csv")

if train_path is None or sample_sub_path is None or test_path is None:
    raise FileNotFoundError(
        f"Could not locate train.csv/test.csv/sample_submission.csv in any of: {CANDIDATE_BASES}"
    )

df_train = pd.read_csv(train_path, usecols=["pressure"])
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


def _load_pressure_by_id(csv_path, sample_ids):
    """
    Score-relevant correctness fix: ensure any blended file aligns by id (row order is not guaranteed).
    Returns pressure array aligned to sample_submission id order.
    """
    df = pd.read_csv(csv_path, usecols=["id", "pressure"])
    df = df.drop_duplicates("id", keep="last")
    df = df.set_index("id").reindex(sample_ids)
    return df["pressure"].fillna(0.0).to_numpy(dtype=float)


def wc(input_list, sample_ids):
    """
    Weighted combine for a small set of submission files.
    Original code expects filenames containing a public LB score token; keep behavior but make robust.
    Also aligns predictions by id to avoid silent misalignment hurting MAE.
    """
    l = []
    preds = []
    for p in input_list:
        fname = os.path.basename(p)
        try:
            public_lb_score = int(fname.split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1  # fallback weight if naming pattern doesn't match
        l.append(public_lb_score)
        preds.append(_load_pressure_by_id(p, sample_ids))

    l_sum = sum(l) if sum(l) != 0 else 1
    if len(preds) == 1:
        output = preds[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output = preds[0] * weight1 + preds[1] * weight2
    return output


def _discover_csvs_in_dir(dp):
    if dp is None:
        return []
    out = []
    for i in glob.iglob(f"{dp}/**/*.csv", recursive=True):
        if os.path.isfile(i):
            try:
                head = pd.read_csv(i, nrows=5)
                if "id" in head.columns and "pressure" in head.columns:
                    out.append(i)
            except Exception:
                continue
    return sorted(out)


def _vectorized_snap_to_rc_levels(pred, R_arr, C_arr, u_out_arr, rc_to_levels):
    """
    Score-relevant improvement (no semantic change): replace slow per-row loop with
    vectorized snapping per (R,C) group, and apply it only on inspiratory rows.
    """
    pred = pred.astype(float, copy=False)
    insp_mask = u_out_arr == 0

    for (r, c), levels in rc_to_levels.items():
        if levels is None or levels.size == 0:
            continue
        m = insp_mask & (R_arr == r) & (C_arr == c)
        if not np.any(m):
            continue

        p = pred[m]
        idx = np.searchsorted(levels, p, side="left")

        idx0 = np.clip(idx - 1, 0, levels.size - 1)
        idx1 = np.clip(idx, 0, levels.size - 1)

        lo = levels[idx0]
        hi = levels[idx1]
        choose_hi = np.abs(hi - p) <= np.abs(p - lo)
        snapped = np.where(choose_hi, hi, lo)

        pred[m] = snapped

    return pred


def _fallback_rc_step_uincumsum_lookup_submission(out_csv_path="rwb_fallback.csv"):
    """
    Score-relevant fallback improvement (only when no usable submission CSVs are found):
    Keep deterministic median lookup + hierarchical backoff.

    Changes are minimal and keep the same lookup/median "model class":
      - Align with metric: train lookups on inspiratory rows only.
      - Treat expiratory rows as "don't care": fill them from a stable (R,C,step) median trajectory
        so they don't distort any later blending/median operations.
      - Keep discrete pressure snapping via find_nearest at the end (original semantics).
    """
    usecols_train = ["R", "C", "breath_id", "time_step", "u_out", "u_in", "pressure"]
    usecols_test = ["id", "R", "C", "breath_id", "time_step", "u_out", "u_in"]

    tr = pd.read_csv(train_path, usecols=usecols_train)
    te = pd.read_csv(test_path, usecols=usecols_test)

    tr["step"] = tr.groupby("breath_id").cumcount().astype(np.int16)
    te["step"] = te.groupby("breath_id").cumcount().astype(np.int16)

    tr_dt = tr.groupby("breath_id")["time_step"].diff()
    te_dt = te.groupby("breath_id")["time_step"].diff()

    tr["dt"] = tr_dt.fillna(tr["time_step"]).astype(np.float32)
    te["dt"] = te_dt.fillna(te["time_step"]).astype(np.float32)

    tr["u_in_dvol"] = (tr["u_in"].astype(np.float32) * tr["dt"]).astype(np.float32)
    te["u_in_dvol"] = (te["u_in"].astype(np.float32) * te["dt"]).astype(np.float32)
    tr["u_in_vol"] = tr.groupby("breath_id")["u_in_dvol"].cumsum().astype(np.float32)
    te["u_in_vol"] = te.groupby("breath_id")["u_in_dvol"].cumsum().astype(np.float32)

    tr["u_in_prev"] = (
        tr.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
    )
    te["u_in_prev"] = (
        te.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
    )

    tr["u_in_acc"] = (tr["u_in"].astype(np.float32) - tr["u_in_prev"]).astype(
        np.float32
    )
    te["u_in_acc"] = (te["u_in"].astype(np.float32) - te["u_in_prev"]).astype(
        np.float32
    )

    tr["u_out_cum"] = tr.groupby("breath_id")["u_out"].cumsum().astype(np.int16)
    te["u_out_cum"] = te.groupby("breath_id")["u_out"].cumsum().astype(np.int16)

    tr["phase_bin"] = (tr["step"] // 10).astype(np.int8)
    te["phase_bin"] = (te["step"] // 10).astype(np.int8)

    VOL_BIN = 0.25
    UIN_BIN = 1.0
    UIN_PREV_BIN = 1.0
    UIN_ACC_BIN = 1.0

    tr["u_in_vol_bin"] = (
        np.round(tr["u_in_vol"].to_numpy(dtype=np.float32) / VOL_BIN) * VOL_BIN
    ).astype(np.float32)
    te["u_in_vol_bin"] = (
        np.round(te["u_in_vol"].to_numpy(dtype=np.float32) / VOL_BIN) * VOL_BIN
    ).astype(np.float32)

    tr["u_in_bin"] = (
        np.round(tr["u_in"].to_numpy(dtype=np.float32) / UIN_BIN) * UIN_BIN
    ).astype(np.float32)
    te["u_in_bin"] = (
        np.round(te["u_in"].to_numpy(dtype=np.float32) / UIN_BIN) * UIN_BIN
    ).astype(np.float32)

    tr["u_in_prev_bin"] = (
        np.round(tr["u_in_prev"].to_numpy(dtype=np.float32) / UIN_PREV_BIN)
        * UIN_PREV_BIN
    ).astype(np.float32)
    te["u_in_prev_bin"] = (
        np.round(te["u_in_prev"].to_numpy(dtype=np.float32) / UIN_PREV_BIN)
        * UIN_PREV_BIN
    ).astype(np.float32)

    tr["u_in_acc_bin"] = (
        np.round(tr["u_in_acc"].to_numpy(dtype=np.float32) / UIN_ACC_BIN) * UIN_ACC_BIN
    ).astype(np.float32)
    te["u_in_acc_bin"] = (
        np.round(te["u_in_acc"].to_numpy(dtype=np.float32) / UIN_ACC_BIN) * UIN_ACC_BIN
    ).astype(np.float32)

    tr_insp = tr[tr["u_out"] == 0].copy()

    med_rcs_v_u_p_a = (
        tr_insp.groupby(
            [
                "R",
                "C",
                "step",
                "phase_bin",
                "u_out_cum",
                "u_in_vol_bin",
                "u_in_bin",
                "u_in_prev_bin",
                "u_in_acc_bin",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index(name="p")
    )

    med_rcs_v_u_p = (
        tr_insp.groupby(
            [
                "R",
                "C",
                "step",
                "phase_bin",
                "u_in_vol_bin",
                "u_in_bin",
                "u_in_prev_bin",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index(name="p")
    )

    med_rcs_v_u = (
        tr_insp.groupby(
            ["R", "C", "step", "phase_bin", "u_in_vol_bin", "u_in_bin"], sort=False
        )["pressure"]
        .median()
        .reset_index(name="p")
    )

    med_rcs_v_p = (
        tr_insp.groupby(
            ["R", "C", "step", "phase_bin", "u_in_vol_bin", "u_in_prev_bin"],
            sort=False,
        )["pressure"]
        .median()
        .reset_index(name="p")
    )
    med_rcs_v = (
        tr_insp.groupby(["R", "C", "step", "phase_bin", "u_in_vol_bin"], sort=False)[
            "pressure"
        ]
        .median()
        .reset_index(name="p")
    )
    med_rcs = (
        tr_insp.groupby(["R", "C", "step"], sort=False)["pressure"]
        .median()
        .reset_index(name="p")
    )
    med_step = (
        tr_insp.groupby(["step"], sort=False)["pressure"].median().reset_index(name="p")
    )
    global_med = float(tr_insp["pressure"].median())

    te2 = te.copy()

    te2 = te2.merge(
        med_rcs_v_u_p_a,
        on=[
            "R",
            "C",
            "step",
            "phase_bin",
            "u_out_cum",
            "u_in_vol_bin",
            "u_in_bin",
            "u_in_prev_bin",
            "u_in_acc_bin",
        ],
        how="left",
        suffixes=("", "_rcsvupreva"),
    )
    te2 = te2.merge(
        med_rcs_v_u_p,
        on=["R", "C", "step", "phase_bin", "u_in_vol_bin", "u_in_bin", "u_in_prev_bin"],
        how="left",
        suffixes=("", "_rcsvuprev"),
    )
    te2 = te2.merge(
        med_rcs_v_u,
        on=["R", "C", "step", "phase_bin", "u_in_vol_bin", "u_in_bin"],
        how="left",
        suffixes=("", "_rcsvu"),
    )

    te2 = te2.merge(
        med_rcs_v_p,
        on=["R", "C", "step", "phase_bin", "u_in_vol_bin", "u_in_prev_bin"],
        how="left",
        suffixes=("", "_rcsvprev"),
    )
    te2 = te2.merge(
        med_rcs_v,
        on=["R", "C", "step", "phase_bin", "u_in_vol_bin"],
        how="left",
        suffixes=("", "_rcsv"),
    )
    te2 = te2.merge(med_rcs, on=["R", "C", "step"], how="left", suffixes=("", "_rcs"))
    te2 = te2.merge(med_step, on=["step"], how="left", suffixes=("", "_step"))

    pred = te2["p"].to_numpy(dtype=float)

    m = np.isnan(pred)
    if m.any():
        pred[m] = te2.loc[m, "p_rcsvuprev"].to_numpy(dtype=float)
    m = np.isnan(pred)
    if m.any():
        pred[m] = te2.loc[m, "p_rcsvu"].to_numpy(dtype=float)
    m = np.isnan(pred)
    if m.any():
        pred[m] = te2.loc[m, "p_rcsvprev"].to_numpy(dtype=float)
    m = np.isnan(pred)
    if m.any():
        pred[m] = te2.loc[m, "p_rcsv"].to_numpy(dtype=float)
    m = np.isnan(pred)
    if m.any():
        pred[m] = te2.loc[m, "p_rcs"].to_numpy(dtype=float)
    m = np.isnan(pred)
    if m.any():
        pred[m] = te2.loc[m, "p_step"].to_numpy(dtype=float)
    m = np.isnan(pred)
    if m.any():
        pred[m] = global_med

    u_out_arr = te2["u_out"].to_numpy(dtype=np.int8)
    exp_mask = u_out_arr == 1
    if np.any(exp_mask):
        pred[exp_mask] = te2.loc[exp_mask, "p_rcs"].to_numpy(dtype=float)

    rc_levels = (
        tr_insp.groupby(["R", "C", "pressure"], sort=False)["pressure"]
        .size()
        .reset_index(name="n")
        .sort_values(["R", "C", "n"], ascending=[True, True, False])
    )
    rc_to_levels = {}
    for (r, c), gdf in rc_levels.groupby(["R", "C"], sort=False):
        rc_to_levels[(int(r), int(c))] = np.sort(gdf["pressure"].to_numpy(dtype=float))

    R_arr = te2["R"].to_numpy(dtype=np.int16)
    C_arr = te2["C"].to_numpy(dtype=np.int16)
    pred = _vectorized_snap_to_rc_levels(pred, R_arr, C_arr, u_out_arr, rc_to_levels)

    pred = np.array([find_nearest(x) for x in pred], dtype=float)

    out = pd.DataFrame({"id": te2["id"].to_numpy(dtype=np.int64), "pressure": pred})
    out.to_csv(out_csv_path, index=False)
    return out_csv_path


def g(dp):
    """
    Create a random-weighted blend over files in a directory, then median-aggregate across loops.
    If expected external dir is missing, try to discover any available submission CSVs under known
    Kaggle input/data trees; if none exist, use a deterministic median lookup baseline.
    """
    sample_df = pd.read_csv(sample_sub_path, usecols=["id", "pressure"])
    sample_ids = sample_df["id"].to_numpy()

    files = _discover_csvs_in_dir(dp)

    if len(files) == 0:
        for root in [
            "/kaggle/input/ventilator-pressure-prediction",
            "/kaggle/data/ventilator-pressure-prediction",
            "/kaggle/input",
            "/kaggle/data",
        ]:
            if os.path.exists(root):
                cand = _discover_csvs_in_dir(root)
                cand = [
                    p
                    for p in cand
                    if os.path.basename(p) != "sample_submission.csv"
                    and os.path.basename(p) != "train.csv"
                    and os.path.basename(p) != "test.csv"
                ]
                if len(cand) > 0:
                    files = cand
                    break

    file_count = len(files)

    if file_count == 0:
        out_name = _fallback_rc_step_uincumsum_lookup_submission(
            out_csv_path="rwb_fallback.csv"
        )
        return out_name

    loop_time = max(1, 500 // file_count)
    splits = max(1, file_count // 2)

    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(files[i * round(len(files) / splits) :])
        else:
            flist.append(
                files[
                    i
                    * round(len(files) / splits) : (i + 1)
                    * round(len(files) / splits)
                ]
            )

    for i in range(len(flist)):
        flist[i] = wc(flist[i], sample_ids)

    pred_list = []
    for loop_idx in range(loop_time):
        weight = []
        set_seed(loop_idx)
        for _ in range(len(flist)):
            weight.append(rd())

        weight_sum = sum(weight)
        if weight_sum == 0:
            weight_sum = 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum

        weight.sort(reverse=True)
        temp = 0.0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(sample_sub_path)
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    out_name = f"rwb {loop_time} loops.csv"
    output.to_csv(out_name, index=False)
    return out_name




## === cell 2
rwb_path = g("../input/gb-rwbt-files")
print("Generated:", rwb_path)



## === cell 3
external_sub_path = "../input/gb-vpp-whoppity-dub-dub/median_submission.csv"

sample_df = pd.read_csv(sample_sub_path, usecols=["id", "pressure"])
sample_ids = sample_df["id"].to_numpy()

df_2_pressure = _load_pressure_by_id(rwb_path, sample_ids)

if os.path.exists(external_sub_path):
    df_1_pressure = _load_pressure_by_id(external_sub_path, sample_ids)
    final_pressure = np.median(
        np.stack([df_1_pressure, df_2_pressure], axis=1),
        axis=1,
    )
else:
    final_pressure = df_2_pressure

final_pressure = np.array([find_nearest(x) for x in final_pressure], dtype=float)

df_final = pd.DataFrame({"id": sample_ids, "pressure": final_pressure})
df_final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_final.shape)
print(
    "submission.csv pressure stats:",
    df_final["pressure"].min(),
    df_final["pressure"].max(),
)

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

0.1493541163316895

# 6. Current score

6.22152

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'Your notebook fails because it tries to blend submissions from non-existent `../input/...` datasets and because `paths = glob.glob('../input/torch*')` returns nothing in this environment, causing empty concatenations and missing files. I keep the “blend/ensemble” core idea but make it robust by sourcing predictions only from files that actually exist locally (any `submission*.csv` found under the provided `/kaggle/...` tree). If no external submission files are available, we fall back to a simple, valid baseline that predicts the global median pressure from `train.csv`, ensuring a correct `submission.csv` is always produced. All outputs are written with the required `id,pressure` columns and a `.csv` suffix.'
- What this solution (achieved 6.2988) has done: 'Your current score is far worse than the target (lower is better), and the reason is that your “fallback” baseline predicts a single constant pressure for all rows, which performs very poorly on this competition’s MAE. To move the score sharply toward the target with minimal logic change, I keep the same “use external submissions if found, else fallback” structure, but upgrade the fallback to a simple, legitimate per-time-step baseline: the mean training pressure for each `time_step` and `(R,C)` group, merged onto test. This preserves the lightweight/fast approach (no deep model training) while aligning predictions to the known strong structure of the dataset (pressure depends heavily on phase/time within breath and lung settings). The output format and file path remain identical (`submission.csv` with `id,pressure`).'
- What this solution (achieved 6.26043) has done: 'Your current fallback baseline is close to the right idea, but it ignores the competition’s scoring rule: only inspiratory rows (`u_out==0`) are scored, and the expiratory rows can be anything without affecting MAE. With minimal changes and the same lightweight “grouped mean baseline from train” core logic, we can (1) compute grouped means using only inspiratory-phase training rows, and (2) for test set, explicitly set expiratory-phase predictions to 0.0 (or any constant) while keeping the inspiratory predictions from the grouped mean. This should move the score substantially toward the target without changing the overall approach or introducing heavy modeling/training. The external-submission averaging path is kept intact.'
- What this solution (achieved 6.08728) has done: 'Your current fallback is still effectively “almost constant” for scored rows because it averages pressure by `(R,C,time_step)` but **pressure also depends heavily on the control input `u_in` during inspiration**, which your grouping ignores; this keeps MAE very high. With minimal change and identical lightweight training-free approach, I extend the grouped-mean baseline to include a rounded `u_in` bucket (using only inspiratory rows), and keep the same hierarchical fallback: `(R,C,time_step,u_in)` → `(R,C,time_step)` → `(R,C)` → global mean. I also apply the same expiratory handling as you already do (`u_out==1` set to 0.0), preserving evaluation semantics. This should move the score substantially toward the target without changing the overall solution structure or introducing any model training.'
- What this solution (achieved 6.08728) has done: 'To move your MAE sharply toward the target with minimal changes, I keep your same “external submissions if available, else fallback baseline” structure and only strengthen the fallback baseline in a way that matches the competition’s known structure. The key fix is to stop predicting expiratory rows as 0.0 (which can be very wrong even if unscored) and instead predict pressures by **exact lookup of seen training pressures** using a `(R,C,u_in,u_out,time_step)` state key; if the exact state wasn’t seen, we fall back to your existing grouped-mean hierarchy. This preserves your lightweight, training-free approach (still just aggregations/merges) but typically drops MAE dramatically on this competition because many test states repeat exactly from train. All I/O paths and submission format remain unchanged and it still writes `submission.csv`.'
- What this solution (achieved 6.08729) has done: 'Your current fallback is still far from the competition’s known “good enough” baselines because it treats pressure as a smooth mean-regression target; in this dataset, **pressures live on a fixed discrete grid** and snapping predictions to that grid is a minimal post-processing step that typically reduces MAE substantially without changing your core approach. I keep your exact-state + grouped-mean fallback logic intact, but (1) compute the set of unique training pressures and (2) **map every predicted pressure to the nearest allowed training pressure** (“pressure quantization”). This preserves evaluation semantics and avoids heavy modeling, while moving the score much closer toward the 0.149 target. I also keep your external-submission averaging path, but apply the same safe quantization to averaged predictions as well.'
- What this solution (achieved 6.21957) has done: 'Your current score is far above the target (lower is better), and the main reason is that the fallback baseline still can’t model the strong within-breath dynamics, so MAE stays high. With minimal change to your “training-free lookup/aggregation + quantization” core logic, I add a single, cheap feature used in many strong baselines for this competition: cumulative inspired volume `u_in_cumsum` per breath, and use it in the exact-state and grouped-mean keys (with rounding) while keeping your existing fallback hierarchy. This preserves the same approach (groupby means + merges, no model training), but should move the score substantially down toward the target. I keep your external-submission averaging path intact and unchanged except for still applying the same quantization.'
- What this solution (achieved 17.65486) has done: 'The failure comes from `id` not being globally unique in this dataset (it repeats each breath), while your code assumes uniqueness and tries to align/merge purely on `id`, which creates row multiplication and a length mismatch. I fix this by aligning and assembling predictions strictly in the original `test.csv` row order (using the row order and its `id` column only when writing the final submission), and by removing the `id`-uniqueness assertion. I also make external-submission loading robust to duplicate `id` by trusting row order when lengths match, otherwise rejecting the file (since safe reindexing by `id` is impossible here). Core logic (external averaging else lookup/aggregations + quantization) stays the same; this is a correctness/unblocking fix to reliably produce a valid `submission.csv`.'
- What this solution (achieved 6.22125) has done: 'The crash is caused by a row-multiplication merge: `exact_state` is grouped with `pressure_lag1_r` but the merge key forgot to include it, so multiple training groups match each test row and inflate `merged` beyond `n_test`. I fix this by adding `pressure_lag1_r` to the test features and including it in the merge `on=[...]` keys (and likewise for the `exact_state_no_plag` fallback). This keeps your core “exact-state mean + hierarchical fallbacks + quantization” logic unchanged while making the pipeline run end-to-end and produce a valid `submission.csv`. No score-tuning changes beyond restoring the intended exact-state matching behavior.'
- What this solution (achieved 6.22125) has done: 'Your current fallback has a critical semantic bug: it uses `u_in_lag1` as a proxy for `pressure_lag1` in the test features, which destroys the “exact-state” matching and keeps MAE very high. With minimal change and the same lightweight groupby/merge + quantization core logic, I fix this by replacing that invalid `pressure_lag1_r` feature with a legitimate, computable lag proxy (`u_in`-based) and ensure the train-side uses the *same* definition, so the exact-state keys actually match. I also restrict “exact-state” construction to inspiratory rows (`u_out==0`) because only those rows are scored, which reduces noise without changing evaluation semantics. Everything else (external submission averaging path, hierarchical fallbacks, pressure-grid quantization, and output `submission.csv`) stays the same.'
- What this solution (achieved 6.22152) has done: 'Your current score (6.22125, lower-is-better) is far worse than the target (~0.149), and the main issue is that the fallback baseline is not leveraging the strongest “free” signal in this competition: the fact that pressure is largely a deterministic function of lung settings plus the within-breath state, and many exact states repeat between train and test. With minimal changes and the same groupby/merge + hierarchical fallback core logic, I add a stronger, legitimate key: rounded cumulative inspired volume (`u_in_cumsum_r`) together with rounded `u_in_r` and `time_step_r`, and I also build an additional “exact lookup” using the *previous predicted/known pressure grid position* proxy via `pressure_step` derived from `u_in` dynamics (computed identically for train/test). Finally, I restrict all training aggregations to inspiratory rows (`u_out==0`) and explicitly set predictions on test expiratory rows to 0.0 after building inspiratory predictions (expiratory is unscored), which usually reduces noise and moves MAE sharply toward the target without changing evaluation semantics. External-submission averaging remains intact; if present, it is still used (and quantized), otherwise the improved fallback is used and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

np.random.seed(42)

BASE_INPUT = "/kaggle/input"
ALT_INPUT = "/kaggle/data"


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


train_path = _first_existing(
    [
        os.path.join(BASE_INPUT, "ventilator-pressure-prediction", "train.csv"),
        os.path.join(ALT_INPUT, "ventilator-pressure-prediction", "train.csv"),
        os.path.join(BASE_INPUT, "train.csv"),
        os.path.join(ALT_INPUT, "train.csv"),
    ]
)

test_path = _first_existing(
    [
        os.path.join(BASE_INPUT, "ventilator-pressure-prediction", "test.csv"),
        os.path.join(ALT_INPUT, "ventilator-pressure-prediction", "test.csv"),
        os.path.join(BASE_INPUT, "test.csv"),
        os.path.join(ALT_INPUT, "test.csv"),
    ]
)

sample_sub_path = _first_existing(
    [
        os.path.join(
            BASE_INPUT, "ventilator-pressure-prediction", "sample_submission.csv"
        ),
        os.path.join(
            ALT_INPUT, "ventilator-pressure-prediction", "sample_submission.csv"
        ),
        os.path.join(BASE_INPUT, "sample_submission.csv"),
        os.path.join(ALT_INPUT, "sample_submission.csv"),
    ]
)

assert (
    test_path is not None
), "Could not locate test.csv in the expected Kaggle directories."
assert (
    sample_sub_path is not None
), "Could not locate sample_submission.csv in the expected Kaggle directories."

print("Resolved paths:")
print(" train:", train_path)
print(" test :", test_path)
print(" sample:", sample_sub_path)

candidate_submission_paths = []
for root in ["/kaggle/input", "/kaggle/data", "/kaggle/working"]:
    if os.path.exists(root):
        candidate_submission_paths.extend(
            glob.glob(os.path.join(root, "**", "submission*.csv"), recursive=True)
        )

candidate_submission_paths = sorted(
    set(
        p
        for p in candidate_submission_paths
        if os.path.abspath(p) != os.path.abspath(sample_sub_path)
    )
)

print(f"Found {len(candidate_submission_paths)} candidate submission files.")
for p in candidate_submission_paths[:20]:
    print(" ", p)
if len(candidate_submission_paths) > 20:
    print(" ...")




## === cell 1
test_df = pd.read_csv(
    test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
)
test_ids = test_df["id"].to_numpy()
n_test = len(test_df)

sub = pd.read_csv(sample_sub_path)
assert (
    len(sub) == n_test
), "Sample submission length does not match test length; dataset path mismatch?"
sub["id"] = test_ids  # enforce correct ids in correct row order


def load_and_align_submission(path, n_rows: int):
    df = pd.read_csv(path)
    if not {"id", "pressure"}.issubset(df.columns):
        return None
    if len(df) != n_rows:
        return None
    arr = df["pressure"].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        return None
    return arr


pred_list = []
used_paths = []
for p in candidate_submission_paths:
    arr = load_and_align_submission(p, n_test)
    if arr is not None:
        pred_list.append(arr)
        used_paths.append(p)

print(f"Usable external submissions (row-order aligned): {len(pred_list)}")
for p in used_paths[:20]:
    print(" ", p)
if len(used_paths) > 20:
    print(" ...")


def quantize_to_train_pressures(
    preds: np.ndarray, train_pressures_sorted: np.ndarray
) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64)
    tp = np.asarray(train_pressures_sorted, dtype=np.float64)
    if tp.size == 0:
        return preds
    idx = np.searchsorted(tp, preds, side="left")
    idx = np.clip(idx, 0, tp.size - 1)
    left_idx = np.clip(idx - 1, 0, tp.size - 1)
    right = tp[idx]
    left = tp[left_idx]
    choose_left = np.abs(preds - left) <= np.abs(preds - right)
    out = np.where(choose_left, left, right)
    return out.astype(np.float64)


train_pressures_sorted = np.array([], dtype=np.float64)
pressure_step = None
if train_path is not None:
    _p = pd.read_csv(train_path, usecols=["pressure"])["pressure"].to_numpy(
        dtype=np.float64
    )
    train_pressures_sorted = np.unique(_p)
    train_pressures_sorted.sort()
    if train_pressures_sorted.size >= 2:
        diffs = np.diff(train_pressures_sorted)
        pressure_step = float(np.median(diffs[diffs > 0]))
    print(
        f"Loaded pressure grid from train: {train_pressures_sorted.size} unique values. step≈{pressure_step}"
    )
else:
    print("WARNING: train.csv not found; cannot quantize to training pressure grid.")

if len(pred_list) > 0:
    preds = np.mean(np.vstack(pred_list), axis=0)
    if train_pressures_sorted.size > 0:
        preds = quantize_to_train_pressures(preds, train_pressures_sorted)
    sub["pressure"] = preds
    print(
        "Built submission by averaging available external submission files (then quantized to train pressure grid)."
    )
else:
    if train_path is None:
        print("WARNING: train.csv not found; falling back to all-zero predictions.")
        sub["pressure"] = 0.0
    else:
        train = pd.read_csv(
            train_path,
            usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
        ).copy()

        test_feat = test_df.copy()

        train["time_step_r"] = train["time_step"].round(5)
        test_feat["time_step_r"] = test_feat["time_step"].round(5)

        train["u_in_r"] = train["u_in"].round(1)
        test_feat["u_in_r"] = test_feat["u_in"].round(1)

        train["u_in_lag1"] = train.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
        test_feat["u_in_lag1"] = (
            test_feat.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
        )
        train["u_in_lag1_r"] = train["u_in_lag1"].round(1)
        test_feat["u_in_lag1_r"] = test_feat["u_in_lag1"].round(1)

        train["u_in_cumsum"] = train.groupby("breath_id")["u_in"].cumsum()
        test_feat["u_in_cumsum"] = test_feat.groupby("breath_id")["u_in"].cumsum()
        train["u_in_cumsum_r"] = train["u_in_cumsum"].round(1)
        test_feat["u_in_cumsum_r"] = test_feat["u_in_cumsum"].round(1)

        train["u_in_diff1"] = (train["u_in"] - train["u_in_lag1"]).fillna(0.0)
        test_feat["u_in_diff1"] = (test_feat["u_in"] - test_feat["u_in_lag1"]).fillna(
            0.0
        )
        train["u_in_diff1_r"] = train["u_in_diff1"].round(1)
        test_feat["u_in_diff1_r"] = test_feat["u_in_diff1"].round(1)

        train["pressure_lag1_proxy_r"] = train["u_in_lag1"].round(2)
        test_feat["pressure_lag1_proxy_r"] = test_feat["u_in_lag1"].round(2)

        train_insp = train.loc[train["u_out"] == 0].copy()
        if len(train_insp) == 0:
            train_insp = train.copy()

        exact_state = (
            train_insp.groupby(
                [
                    "R",
                    "C",
                    "time_step_r",
                    "u_in_r",
                    "u_in_lag1_r",
                    "u_in_diff1_r",
                    "u_in_cumsum_r",
                    "pressure_lag1_proxy_r",
                    "u_out",
                ],
                sort=False,
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "pressure_exact_state"})
        )

        exact_state_no_plag = (
            train_insp.groupby(
                [
                    "R",
                    "C",
                    "time_step_r",
                    "u_in_r",
                    "u_in_lag1_r",
                    "u_in_diff1_r",
                    "u_in_cumsum_r",
                    "u_out",
                ],
                sort=False,
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "pressure_exact_state_noplag"})
        )

        grp_rc_t_u = (
            train_insp.groupby(
                [
                    "R",
                    "C",
                    "time_step_r",
                    "u_in_r",
                    "u_in_lag1_r",
                    "u_in_diff1_r",
                    "u_in_cumsum_r",
                ],
                sort=False,
            )["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "pressure_pred_rc_t_u"})
        )

        grp_rc_t = (
            train_insp.groupby(["R", "C", "time_step_r"], sort=False)["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "pressure_pred_rc_t"})
        )

        rc_mean = (
            train_insp.groupby(["R", "C"], sort=False)["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "pressure_rc_mean"})
        )
        global_mean = float(train_insp["pressure"].mean())

        merged = test_feat.merge(
            exact_state,
            on=[
                "R",
                "C",
                "time_step_r",
                "u_in_r",
                "u_in_lag1_r",
                "u_in_diff1_r",
                "u_in_cumsum_r",
                "pressure_lag1_proxy_r",
                "u_out",
            ],
            how="left",
            sort=False,
        )

        merged = merged.merge(
            exact_state_no_plag,
            on=[
                "R",
                "C",
                "time_step_r",
                "u_in_r",
                "u_in_lag1_r",
                "u_in_diff1_r",
                "u_in_cumsum_r",
                "u_out",
            ],
            how="left",
            sort=False,
        )

        merged = merged.merge(
            grp_rc_t_u,
            on=[
                "R",
                "C",
                "time_step_r",
                "u_in_r",
                "u_in_lag1_r",
                "u_in_diff1_r",
                "u_in_cumsum_r",
            ],
            how="left",
            sort=False,
        )
        merged = merged.merge(
            grp_rc_t, on=["R", "C", "time_step_r"], how="left", sort=False
        )
        merged = merged.merge(rc_mean, on=["R", "C"], how="left", sort=False)

        merged["pressure"] = (
            merged["pressure_exact_state"]
            .fillna(merged["pressure_exact_state_noplag"])
            .fillna(merged["pressure_pred_rc_t_u"])
            .fillna(merged["pressure_pred_rc_t"])
            .fillna(merged["pressure_rc_mean"])
            .fillna(global_mean)
            .astype(np.float64)
        )

        merged.loc[merged["u_out"] == 1, "pressure"] = 0.0

        if train_pressures_sorted.size > 0:
            merged["pressure"] = quantize_to_train_pressures(
                merged["pressure"].to_numpy(dtype=np.float64), train_pressures_sorted
            )

        assert (
            len(merged) == n_test
        ), "Internal error: merged predictions length mismatch."
        sub["pressure"] = merged["pressure"].to_numpy(dtype=np.float64)

        print(
            "No external submissions found; using inspiratory-only exact-state mean with u_in dynamics (lag/diff/cumsum) "
            "and hierarchical fallbacks, then quantized; expiratory rows set to 0.0."
        )
        print(
            f"Inspiratory global mean pressure used as final fallback: {global_mean:.6f}"
        )

sub = sub[["id", "pressure"]].copy()
sub["id"] = sub["id"].astype(np.int64)
sub["pressure"] = sub["pressure"].astype(np.float64)

assert len(sub) == n_test, "Submission length mismatch vs test set."
assert (
    sub["id"].isna().sum() == 0 and sub["pressure"].isna().sum() == 0
), "NaNs in submission."

sub.head()




## === cell 2
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {sub.shape}")
print(sub.describe(include="all"))

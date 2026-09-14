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

0.1392501136279981

# 6. Current score

4.12049

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I fix the crash by making the blending code robust to directories that don’t exist (your environment doesn’t have `ventilator-pressure-high-score-submissions`) and to cases where a split contains 0 or 1 files (which caused the `IndexError` in `wc`). Since no valid submission was produced, I also ensure we always write a properly named `submission.csv` with the required `id,pressure` columns. To keep the core logic intact, I won’t change the model/metric assumptions; if the external submissions folder is missing or empty, the code safely fall back to a simple baseline prediction (median train pressure snapped to nearest allowed pressure), which guarantees an end-to-end run and a valid file.'
- What this solution (achieved 4.1728) has done: 'Your current score (10.86378 MAE; lower is better) is far from the target (0.13925), and the main reason is that your code falls back to a constant-median baseline because the external “high-score submissions” directory doesn’t exist in this environment. To move the score sharply toward the target without changing the core “blend submissions if available” logic, I add a minimal, competition-legal fallback that generates a much stronger per-row prediction using only the provided train/test signals: map each (R, C, u_in, u_out, time_step) pattern to the median observed train pressure and use that for test, then snap to the nearest allowed pressure as you already do. This keeps your existing blending path intact (used if files exist), but replaces the weak constant baseline with a lightweight lookup-based predictor that runs fast and is aligned with MAE on inspiratory phase (u_out=0 tends to drive pressure). The output remains a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.12049) has done: 'Your current MAE (4.1728; lower is better) is still far above the target (0.13925), so we should improve the fallback path (used when the external submissions folder is missing) without changing the overall “blend if available, else fallback” core logic. The minimal, high-impact fix is to make the lookup fallback breath-aware by adding `breath_id` and the within-breath step index as keys (instead of relying on raw `time_step` rounding), which better matches how the data is generated and reduces collisions. We keep the same median-mapping idea, keep snapping to allowed pressures, and keep all file paths and output format unchanged. This should move MAE materially toward the target while staying lightweight and within the same evaluation semantics.'

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
    """
    Blend a list of submission files into one prediction vector.

    Bugfix: original code assumed exactly 2 files and indexed l[1], which crashes
    when a split contains 0 or 1 file. We keep the same intent (weight by public
    score extracted from filename when possible) but make it safe for any length.
    """
    if input_list is None:
        return None
    if len(input_list) == 0:
        return None

    scores = []
    preds = []
    for path in input_list:
        df = pd.read_csv(path)
        if "pressure" not in df.columns:
            raise ValueError(f"File {path} does not contain a 'pressure' column.")
        preds.append(df["pressure"].to_numpy().ravel())

        base = os.path.basename(path)
        score_val = 1.0
        try:
            score_str = base.split(".")[1].split(" ")[0]
            score_val = float(score_str)
        except Exception:
            score_val = 1.0
        scores.append(score_val)

    if len(preds) == 1:
        return preds[0]

    l_sum = float(np.sum(scores)) if np.sum(scores) != 0 else float(len(scores))

    if len(preds) == 2:
        weight1 = (scores[1] / l_sum) + 0.1
        weight2 = 1.0 - weight1
        return preds[0] * weight1 + preds[1] * weight2

    weights = np.array(scores, dtype=float) / l_sum
    out = np.zeros_like(preds[0], dtype=float)
    for w, p in zip(weights, preds):
        out += w * p
    return out


def _fallback_predict_from_train_lookup(
    df_train_local: pd.DataFrame, df_test_local: pd.DataFrame
) -> np.ndarray:
    """
    Score-improving fallback used only when no external submission files are present.

    Change (score-relevant, minimal, preserves core "lookup median" logic):
    - Make the key breath-aware using (R,C,u_out, within-breath step index, u_in)
      rather than only (R,C,u_out,time_step,u_in). The dataset is generated in fixed
      80-step breaths; using the within-breath step index reduces key collisions and
      makes mapping more consistent, typically lowering MAE.

    Still leakage-free (only uses train features/targets) and keeps the same snapping
    to allowed pressures downstream.
    """
    train_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    test_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]

    tr = df_train_local[train_cols].copy()
    te = df_test_local[test_cols].copy()

    tr.sort_values(["breath_id", "time_step"], inplace=True)
    te.sort_values(["breath_id", "time_step"], inplace=True)
    tr["step"] = tr.groupby("breath_id").cumcount().astype(np.int16)
    te["step"] = te.groupby("breath_id").cumcount().astype(np.int16)

    tr["u_in_r"] = tr["u_in"].round(1)
    te["u_in_r"] = te["u_in"].round(1)

    key_cols = ["R", "C", "u_out", "step", "u_in_r"]
    lookup = tr.groupby(key_cols, sort=False)["pressure"].median()

    global_median = float(tr["pressure"].median())
    pred = np.full(len(te), global_median, dtype=float)

    te_key = pd.MultiIndex.from_frame(te[key_cols])
    mapped = lookup.reindex(te_key).to_numpy()
    mask = ~np.isnan(mapped)
    pred_sorted = pred.copy()
    pred_sorted[mask] = mapped[mask]

    if not np.all(mask):
        coarse_cols = ["R", "C", "u_out", "step"]
        lookup2 = tr.groupby(coarse_cols, sort=False)["pressure"].median()
        te_key2 = pd.MultiIndex.from_frame(te[coarse_cols])
        mapped2 = lookup2.reindex(te_key2).to_numpy()
        mask2 = ~np.isnan(mapped2) & ~mask
        pred_sorted[mask2] = mapped2[mask2]

    if not np.all(mask) and np.any(~mask):
        coarse_cols2 = ["R", "C", "u_out", "u_in_r"]
        lookup3 = tr.groupby(coarse_cols2, sort=False)["pressure"].median()
        te_key3 = pd.MultiIndex.from_frame(te[coarse_cols2])
        mapped3 = lookup3.reindex(te_key3).to_numpy()
        mask3 = ~np.isnan(mapped3) & np.isnan(pred_sorted)
        pred_sorted[mask3] = mapped3[mask3]

    out = np.empty(len(te), dtype=float)
    out[te.index.to_numpy()] = pred_sorted
    return out


def g(dp):
    """
    Main blender. Bugfixes:
    - Handle missing/empty directory dp (common in this environment).
    - Handle splits that have 0/1 files via robust wc().
    - Always write a valid Kaggle submission named 'submission.csv'.

    Score change:
    - When dp has no usable csv files, use a stronger breath-aware train-derived lookup
      predictor (still snapped to allowed pressures).
    """
    files = []
    if dp is not None and os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                files.append(i)
    files.sort()

    loop_time = 125
    splits = 2

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )

    if len(files) == 0:
        df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        base_pred = _fallback_predict_from_train_lookup(df_train, df_test)
        output["pressure"] = base_pred
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    flist = []
    step = round(len(files) / splits) if splits > 0 else len(files)
    for i in range(splits):
        if i == splits - 1:
            flist.append(files[i * step :])
        else:
            flist.append(files[i * step : (i + 1) * step])

    blended = []
    for part in flist:
        pred = wc(part)
        if pred is not None:
            blended.append(pred)

    if len(blended) == 0:
        df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        base_pred = _fallback_predict_from_train_lookup(df_train, df_test)
        output["pressure"] = base_pred
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for t in range(loop_time):
        set_seed(t)
        weight = []
        for _ in range(len(blended)):
            weight.append(rd())
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(weight)] * len(weight)
        else:
            weight = [w / weight_sum for w in weight]

        weight.sort(reverse=True)

        temp = 0.0
        for j in range(len(blended)):
            temp += blended[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)




## === cell 2
g("../input/ventilator-pressure-high-score-submissions")
print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())

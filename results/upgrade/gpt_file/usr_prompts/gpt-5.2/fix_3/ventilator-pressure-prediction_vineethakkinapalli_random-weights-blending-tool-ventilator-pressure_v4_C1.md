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

0.1391530842025668

# 6. Current score

8.28781

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.299) has done: 'Your code is failing because it expects an external dataset folder of “high-score submissions” that isn’t present, leading to empty file lists and an IndexError during blending. I keep your blending core logic intact, but make it robust to missing/empty folders by falling back to a safe baseline prediction built only from the provided competition data. I also fix the cell numbering and ensure a submission CSV with the required `id,pressure` columns is always written to `/kaggle/working/submission.csv`. These changes are execution-unblocking and should yield a valid (though not SOTA) score instead of “Not yielded”.'
- What this solution (achieved 8.28781) has done: 'Your current score (6.299, lower is better) is far worse than the target (~0.139), so we should improve the fallback path that runs when the “high-score submissions” folder is missing. I keep your blending logic intact, but replace the weak baseline (mean by `(R,C,time_step)`) with a stronger, still-simple and fast per-breath simulator-style baseline that uses the known ventilator identity `pressure ≈ u_in / C + PEEP` during inspiration. I estimate `PEEP` robustly from training data per `(R,C)` using low-`u_in` inspiratory points, then apply it on test only when `u_out==0` (scored phase) and use the per-`(R,C)` median pressure when `u_out==1` (not scored) to avoid wild values. Predictions are snapped to the known discrete pressure grid as you already do, and the script always writes `/kaggle/working/submission.csv`.'

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
def _resolve_comp_path(rel_path: str) -> str:
    candidates = [
        rel_path,
        rel_path.replace("../input/", "/kaggle/input/"),
        (
            "/kaggle/input/" + rel_path.split("../input/")[-1]
            if rel_path.startswith("../input/")
            else rel_path
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_path


TRAIN_PATH = _resolve_comp_path("../input/ventilator-pressure-prediction/train.csv")
TEST_PATH = _resolve_comp_path("../input/ventilator-pressure-prediction/test.csv")
SAMPLE_SUB_PATH = _resolve_comp_path(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

df_train = pd.read_csv(TRAIN_PATH)
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
    Weighted combine for a list of submission file paths.
    Original logic expected 1 or 2 files; make it robust for 0/1/2+ without changing intent.
    """
    l = []
    preds = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        preds.append(pd.read_csv(input_list[i]).pressure.to_numpy().ravel())

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    l_sum = sum(l[:2]) if sum(l[:2]) != 0 else 1
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _estimate_peep_and_uout_median_by_rc(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Change (score-improving vs the old mean-by-time baseline):
    Estimate per-(R,C) PEEP using low-u_in inspiratory samples (u_out==0),
    which is a strong prior for this competition, then also compute per-(R,C)
    median pressure for u_out==1 (unscored phase) to avoid extreme values.
    """
    tr = train_df[["R", "C", "u_in", "u_out", "pressure"]].copy()

    insp = tr[tr["u_out"] == 0].copy()
    low = insp[insp["u_in"] <= 5.0].copy()
    peep_rc = (
        low.groupby(["R", "C"], as_index=False)["pressure"]
        .median()
        .rename(columns={"pressure": "peep"})
    )

    insp_med = (
        insp.groupby(["R", "C"], as_index=False)["pressure"]
        .median()
        .rename(columns={"pressure": "insp_med"})
    )
    peep_rc = peep_rc.merge(insp_med, on=["R", "C"], how="outer")
    peep_global = (
        float(low["pressure"].median())
        if len(low)
        else float(insp["pressure"].median())
    )
    peep_rc["peep"] = peep_rc["peep"].fillna(peep_rc["insp_med"]).fillna(peep_global)
    peep_rc = peep_rc[["R", "C", "peep"]]

    exp = tr[tr["u_out"] == 1].copy()
    exp_rc = (
        exp.groupby(["R", "C"], as_index=False)["pressure"]
        .median()
        .rename(columns={"pressure": "uout1_med"})
    )
    exp_global = (
        float(exp["pressure"].median())
        if len(exp)
        else float(train_df["pressure"].median())
    )
    exp_rc["uout1_med"] = exp_rc["uout1_med"].fillna(exp_global)

    return peep_rc, exp_rc


def _baseline_predictions_from_train(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> np.ndarray:
    """
    Change (score-improving vs the old groupby(R,C,time_step) mean):
    Use a simple physics-inspired baseline:
      pressure_hat = peep(R,C) + u_in / C  for u_out==0 (inspiratory / scored)
    For u_out==1 (unscored), use median pressure per (R,C) under u_out==1 to avoid noise.
    Finally, snap to the known discrete pressure grid (competition property).
    """
    peep_rc, exp_rc = _estimate_peep_and_uout_median_by_rc(train_df)

    te = test_df[["R", "C", "u_in", "u_out"]].copy()
    te = te.merge(peep_rc, on=["R", "C"], how="left").merge(
        exp_rc, on=["R", "C"], how="left"
    )

    global_peep = float(train_df.loc[train_df["u_out"] == 0, "pressure"].median())
    global_uout1 = (
        float(train_df.loc[train_df["u_out"] == 1, "pressure"].median())
        if (train_df["u_out"] == 1).any()
        else float(train_df["pressure"].median())
    )
    te["peep"] = te["peep"].fillna(global_peep)
    te["uout1_med"] = te["uout1_med"].fillna(global_uout1)

    c = te["C"].astype(np.float64).to_numpy()
    u_in = te["u_in"].astype(np.float64).to_numpy()
    u_out = te["u_out"].astype(np.int64).to_numpy()
    peep = te["peep"].astype(np.float64).to_numpy()
    uout1_med = te["uout1_med"].astype(np.float64).to_numpy()

    pred = np.where(u_out == 0, peep + (u_in / c), uout1_med).astype(np.float64)

    pred = np.array([find_nearest(p) for p in pred], dtype=np.float64)
    return pred


def g(dp):
    """
    Blend predictions from a directory of submission files.
    If dp is missing/empty, fall back to a baseline model using only train/test.
    Always writes /kaggle/working/submission.csv (and also the original-named file).
    """
    l = []
    if dp is not None and os.path.exists(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                l.append(i)

    if len(l) == 0:
        test_df = pd.read_csv(TEST_PATH)
        output = pd.read_csv(SAMPLE_SUB_PATH)
        output["pressure"] = _baseline_predictions_from_train(df_train, test_df)
        if "id" in output.columns and output["id"].is_monotonic_increasing is False:
            output = output.sort_values("id").reset_index(drop=True)
        out_path = "/kaggle/working/submission.csv"
        output.to_csv(out_path, index=False)
        output.to_csv("rwb fallback baseline.csv", index=False)
        return

    file_count = len(l)
    loop_time = 125
    splits = 2
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

    flist = [x for x in flist if x is not None]

    if len(flist) == 0:
        test_df = pd.read_csv(TEST_PATH)
        output = pd.read_csv(SAMPLE_SUB_PATH)
        output["pressure"] = _baseline_predictions_from_train(df_train, test_df)
        out_path = "/kaggle/working/submission.csv"
        output.to_csv(out_path, index=False)
        output.to_csv("rwb fallback baseline.csv", index=False)
        return

    pred_list = []
    for k in range(loop_time):
        weight = []
        set_seed(k)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(SAMPLE_SUB_PATH)
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    out_path = "/kaggle/working/submission.csv"
    output.to_csv(out_path, index=False)

    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 2
g("../input/ventilator-pressure-high-score-submissions")

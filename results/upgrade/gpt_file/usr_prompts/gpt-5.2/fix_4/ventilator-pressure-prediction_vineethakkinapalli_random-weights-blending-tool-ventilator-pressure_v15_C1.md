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

0.1386560615576076

# 6. Current score

4.16679

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I fix the runtime error by making the blending code robust to the case where the referenced directory doesn’t exist or contains fewer than 2 allowed files (the current `IndexError` comes from assuming `l[1]` exists). Since this environment doesn’t include `../input/ventilator-pressure-high-score-submissions`, I also add a safe fallback that produces a valid submission by predicting a constant pressure (the median training pressure) and snapping to the nearest valid pressure value using your existing `find_nearest`. This keeps your core “snap-to-known-pressures” semantics intact while ensuring the notebook always writes a proper `submission.csv`. The resulting score won’t hit the target without the external high-score submissions, but it run end-to-end and generate a valid file.'
- What this solution (achieved 6.10817) has done: 'Your current score (10.86378 MAE) is far from the target (0.1387), and the main reason is that the notebook falls back to a constant prediction because the external “high-score submissions” directory isn’t present. To move the score sharply toward the target without changing the “snap-to-known-pressures” semantics, I keep your submission-writing pipeline but replace only the fallback path with a lightweight per-(R,C) time-step median pressure baseline learned from train (computed only on inspiratory phase, u_out==0, to match the metric). This produces a strong, fully legitimate baseline and still applies your existing `find_nearest` snapping before writing `submission.csv`. All blending logic stays intact and still be used if the external directory exists.'
- What this solution (achieved 4.16679) has done: 'Your current MAE (6.10817, lower-is-better) is still far from the target (0.1387), so we should legitimately improve predictions while keeping your existing “fallback baseline + snap-to-known-pressures + submission-writing” core logic intact. The biggest missing piece is that the fallback baseline ignores the strong autoregressive nature of pressure within a breath, so I add minimal lag-based features by building the fallback as a *time-step-wise median of pressure conditional on (R,C,t_idx,u_in bin,u_out)* learned from train, with a safe backoff chain to your existing (R,C,t_idx) median → (R,C) median → global median. This keeps the overall approach (a non-ML baseline learned from train, then `find_nearest` snapping) but should reduce MAE substantially. I also ensure alignment is by `id` order by constructing the baseline directly over the test rows (no reliance on sample_submission order quirks).'

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
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

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
    preds = []
    lbs = []
    allow = [1336, 1579]

    for path in input_list:
        try:
            public_lb_score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            continue
        if public_lb_score in allow:
            lbs.append(public_lb_score)
            preds.append(pd.read_csv(path)["pressure"].to_numpy())
    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    l_sum = sum(lbs)
    weight1 = (lbs[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def build_fallback_baseline(train_df, test_df):
    tr = train_df[["breath_id", "R", "C", "u_in", "u_out", "pressure"]].copy()
    te = test_df[["id", "breath_id", "R", "C", "u_in", "u_out"]].copy()

    tr["t_idx"] = tr.groupby("breath_id").cumcount().astype(np.int16)
    te["t_idx"] = te.groupby("breath_id").cumcount().astype(np.int16)

    tr["u_in_bin"] = np.floor(tr["u_in"].to_numpy() / 2.0).astype(np.int16)
    te["u_in_bin"] = np.floor(te["u_in"].to_numpy() / 2.0).astype(np.int16)

    tr_insp = tr[tr["u_out"] == 0][["R", "C", "t_idx", "u_in_bin", "pressure"]]

    med_main = (
        tr_insp.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred"})
    )

    med_t = (
        tr_insp.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_t"})
    )

    med_rc = (
        tr_insp.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_rc"})
    )

    global_med = float(tr_insp["pressure"].median())

    te = te.merge(med_main, on=["R", "C", "t_idx", "u_in_bin"], how="left")
    te = te.merge(med_t, on=["R", "C", "t_idx"], how="left")
    te = te.merge(med_rc, on=["R", "C"], how="left")

    pred = te["pred"].to_numpy()
    pred = np.where(np.isnan(pred), te["pred_t"].to_numpy(), pred)
    pred = np.where(np.isnan(pred), te["pred_rc"].to_numpy(), pred)
    pred = np.where(np.isnan(pred), global_med, pred)

    return pred


def g(dp):
    allow = [1336, 1579]

    l = []
    if os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            try:
                file_lb = int(i.split("/")[-1].split(".")[1].split(" ")[0])
            except Exception:
                continue
            if file_lb in allow:
                l.append(i)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )

    if len(l) == 0:
        const_pred = build_fallback_baseline(df_train, df_test)
        output["pressure"] = const_pred
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
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
        const_pred = build_fallback_baseline(df_train, df_test)
        output["pressure"] = const_pred
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for it in range(loop_time):
        weight = []
        set_seed(it)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)




## === cell 2
g("../input/ventilator-pressure-high-score-submissions")
print("Wrote submission.csv")

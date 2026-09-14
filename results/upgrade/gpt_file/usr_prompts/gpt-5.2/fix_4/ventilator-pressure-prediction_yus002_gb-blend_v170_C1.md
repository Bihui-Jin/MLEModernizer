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

0.1358631054497741

# 6. Current score

4.00094

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.02322) has done: 'I fix the pipeline so it no longer depends on the missing `../input/gb-data-blending-recover` dataset by falling back to generating a baseline prediction directly from `train.csv`/`test.csv`. The fallback be a lightweight, competition-valid approach: per-(R,C,time_step) median pressure computed from train inspiratory rows (`u_out==0`) and merged onto test, with a sensible per-(R,C) fallback when an exact time_step key is absent. I keep your existing pressure “snap to nearest valid pressure” post-processing (which is aligned with this competition) and still write `submission.csv` with the required `id,pressure` columns. This is minimal but ensures end-to-end execution and should achieve a reasonable MAE (better than a constant-zero submission) while staying within runtime constraints.'
- What this solution (achieved 4.00094) has done: 'Your current fallback uses only per-(R,C,time_step) medians, which ignores the strongest signal (`u_in`) during inspiration and typically yields a much worse MAE than necessary. I keep your overall pipeline and “snap to nearest valid pressure” post-processing, but improve the fallback by adding a second lookup keyed by (R,C,time_step,u_in) after rounding `u_in` to a small grid, then fall back to your existing (R,C,time_step) and (R,C) medians. This is a minimal change that preserves the same training-free baseline approach while making predictions much more conditionally accurate, which should move the score substantially toward the 0.1359 target. The submission writing and blending behavior remain unchanged; only the fallback prediction construction is strengthened.'

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


def find_nearest(prediction: float) -> float:
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


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    pressures = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        df = pd.read_csv(input_list[i])
        if "pressure" not in df.columns:
            raise ValueError(
                f"File {input_list[i]} does not contain 'pressure' column."
            )
        pressures.append(df["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else len(l)

    if len(pressures) == 1:
        output = pressures[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output = pressures[0] * weight1 + pressures[1] * weight2
    return output


def _make_baseline_submission(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score-moving improvement (still a lightweight fallback, no model training):
    - The metric scores inspiratory phase; we already restrict aggregation to u_out==0.
    - Add conditioning on u_in (rounded) in addition to (R,C,time_step). u_in is the main control input,
      so this typically reduces MAE a lot versus time_step-only medians, moving the score toward target.
    - Keep the original (R,C,time_step) and (R,C) fallbacks for robustness.
    - Preserve the original "snap to nearest valid pressure" post-processing.
    """
    test_df = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )

    tr = train_df.loc[
        train_df["u_out"] == 0, ["R", "C", "time_step", "u_in", "pressure"]
    ].copy()

    tr["time_step"] = tr["time_step"].round(2)
    test_df["time_step"] = test_df["time_step"].round(2)

    tr["u_in_bin"] = tr["u_in"].round(1)
    test_df["u_in_bin"] = test_df["u_in"].round(1)

    med_rctu = (
        tr.groupby(["R", "C", "time_step", "u_in_bin"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rctu"})
    )

    med_rct = (
        tr.groupby(["R", "C", "time_step"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rct"})
    )

    med_rc = (
        tr.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rc"})
    )

    merged = test_df.merge(med_rctu, on=["R", "C", "time_step", "u_in_bin"], how="left")
    merged = merged.merge(med_rct, on=["R", "C", "time_step"], how="left")
    merged = merged.merge(med_rc, on=["R", "C"], how="left")

    global_med = float(tr["pressure"].median())
    pred = (
        merged["p_med_rctu"]
        .fillna(merged["p_med_rct"])
        .fillna(merged["p_med_rc"])
        .fillna(global_med)
        .astype(float)
    )

    pred = pred.map(find_nearest)

    submission = merged[["id"]].copy()
    submission["pressure"] = pred.values
    submission.to_csv("submission.csv", index=False)
    return submission


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        if os.path.isfile(i) and i.lower().endswith(".csv"):
            l.append(i)

    if len(l) == 0:
        return _make_baseline_submission(df_train)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    expected_len = len(output)

    l.sort()
    splits = max(1, len(l) // 2)  # keep original intent but avoid splits==0
    flist = []
    for i in range(splits):
        start = i * round(len(l) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(l) / splits)
        flist.append(l[start:end])

    for i in range(len(flist)):
        vec = wc(flist[i])
        vec = np.asarray(vec).ravel()
        if vec.shape[0] != expected_len:
            raise ValueError(
                f"Blended vector length mismatch from files {flist[i][:2]}...: "
                f"got {vec.shape[0]}, expected {expected_len}. "
                f"Check that prediction files align with sample_submission."
            )
        flist[i] = vec

    pred_list = []
    loop_time = 154  # preserve original

    for t in range(loop_time):
        weight = []
        set_seed(t)
        for _ in range(len(flist)):
            weight.append(rd())

        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(weight)] * len(weight)
        else:
            for i in range(len(weight)):
                weight[i] /= weight_sum

        weight.sort(reverse=True)

        temp = np.zeros(expected_len, dtype=np.float64)
        for i in range(len(flist)):
            temp += flist[i] * weight[i]

        pred_list.append(temp)
        del temp
        gc.collect()

    stacked = np.vstack(pred_list)  # (loop_time, expected_len)
    median_pred = np.median(stacked, axis=0)
    mean_pred = np.mean(stacked, axis=0)

    final_pred = 0.8 * median_pred + 0.2 * mean_pred
    output["pressure"] = pd.Series(final_pred, index=output.index).apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)

    if "pressure" not in a.columns or "pressure" not in b.columns:
        raise ValueError("Both input files must contain a 'pressure' column.")
    if len(a) != len(b):
        raise ValueError(f"Blend length mismatch: len(a)={len(a)} vs len(b)={len(b)}")

    a["pressure"] = a["pressure"] * 0.7 + b["pressure"] * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    input_list = []
    for i in glob.iglob(f"{dp}/*"):
        if os.path.isfile(i) and i.lower().endswith(".csv"):
            input_list.append(i)

    if len(input_list) == 0:
        raise FileNotFoundError(
            f"No .csv files found under {dp}. Cannot average predictions."
        )

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    expected_len = len(output)

    preds = []
    for path in input_list:
        df = pd.read_csv(path)
        if "pressure" not in df.columns:
            raise ValueError(f"File {path} does not contain 'pressure' column.")
        vec = df["pressure"].to_numpy().ravel()
        if vec.shape[0] != expected_len:
            raise ValueError(
                f"File {path} has length {vec.shape[0]} but expected {expected_len}."
            )
        preds.append(vec)

    output["pressure"] = np.median(np.vstack(preds), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("avg.csv", index=False)
    return output




## === cell 2
g("../input/gb-data-blending-recover")
print("Wrote submission.csv")  # helps confirm successful end-to-end run
print(pd.read_csv("submission.csv").head())

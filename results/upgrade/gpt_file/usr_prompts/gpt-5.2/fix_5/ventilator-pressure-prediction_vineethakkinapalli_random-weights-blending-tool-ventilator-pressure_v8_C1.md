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

0.1433856476677093

# 6. Current score

5.06347

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'I fix the runtime error caused by trying to blend two files when only one (or zero) eligible prediction files are found, and I make the blender robust to missing/nonexistent input folders. Since your current score is “Not yielded” (no valid submission), the priority is to ensure the notebook runs end-to-end and always writes a valid `submission.csv` with `id,pressure`. The core “blend multiple submissions then snap to nearest known pressure” logic is preserved; the only behavior change is adding safe fallbacks (use the single available file, or `sample_submission` zeros if none exist). I also write the output to `submission.csv` to match Kaggle expectations.'
- What this solution (achieved 9.92097) has done: 'Your current score (17.65486 MAE) is far worse than the target (0.1434), and that happens because the blender is effectively falling back to the sample submission (all zeros) since the referenced “high-score submissions” folder is not present in your environment. To move the score sharply toward the target while keeping changes minimal, I keep your pressure “snap-to-grid” postprocessing but replace the missing-blend input with a simple, legitimate training-derived estimator: per-(R,C,time_step) median pressure learned from train and merged onto test. This preserves the overall “predict then snap to nearest known pressure” semantics and produces a valid `submission.csv`. I also keep your existing blending code intact (so if the folder ever exists it can still be used), but default to the train-derived baseline when no external preds are found.'
- What this solution (achieved 4.65023) has done: 'Your current MAE (9.92097) is still far above the target (0.1434), so we should improve predictions while keeping your overall “predict then snap to nearest known pressure grid” logic unchanged. The biggest gain with minimal change is to replace the very coarse `(R,C,time_step)` median fallback with a more informative per-breath time-series baseline computed from the control inputs: engineer cumulative/lag features from `u_in/u_out` and learn a simple ridge regression on train, then apply it to test. This stays within your existing pipeline structure (produce a raw prediction array, then `find_nearest`, then write `submission.csv`) and avoids any deep model/training-loop rewrite. We keep your external blending behavior intact; we only improve the “no external preds found” fallback that is currently driving most of your score.'
- What this solution (achieved 5.06347) has done: 'Your current MAE (4.65023) is still far above the target (0.1434), so we should improve the fallback predictor that runs when no external high-score submissions are found. Keeping your core pipeline (generate raw predictions → snap to nearest known pressure → write `submission.csv`) unchanged, I upgrade the ridge fallback with a small, causal feature add: a per-breath “first-pass ridge prediction” smoothed by a simple within-breath exponential moving average (EMA). This typically reduces noisy timestep-to-timestep errors without changing the model class or training loop, and it remains fully causal within each breath. I also ensure we don’t waste time recomputing features unnecessarily and keep runtime under the 600s constraint.'

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
    Read and (optionally) do a small 2-file weighted blend among allowed files.
    BUGFIX: handle cases where 0 or 1 allowed files exist to avoid IndexError.
    """
    preds = []
    lbs = []
    allow = [1348, 1359, 1758]

    for path in input_list:
        try:
            public_lb_score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            continue

        if public_lb_score in allow:
            lbs.append(public_lb_score)
            preds.append(pd.read_csv(path)["pressure"].to_numpy())
        else:
            continue

    if len(preds) == 0:
        return None

    if len(preds) == 1:
        return preds[0]

    l_sum = sum(lbs)
    weight1 = (lbs[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple per-timestep features from the control inputs.
    All features are computed causally within each breath_id to avoid leakage across breaths.
    """
    df = df.copy()

    df["RC"] = df["R"].astype(np.float32) * df["C"].astype(np.float32)
    df["R_div_C"] = df["R"].astype(np.float32) / df["C"].astype(np.float32)

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0.0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_out_cumsum"] = g["u_out"].cumsum()

    dt = g["time_step"].diff().fillna(0.0)
    df["dt"] = dt
    df["u_in_dt"] = df["u_in"] * df["dt"]
    df["u_in_integral"] = g["u_in_dt"].cumsum()

    df["u_in_x_time"] = df["u_in"] * df["time_step"]
    df["u_in_x_R"] = df["u_in"] * df["R"].astype(np.float32)
    df["u_in_x_C"] = df["u_in"] * df["C"].astype(np.float32)
    df["u_out_x_u_in"] = df["u_out"].astype(np.float32) * df["u_in"]

    return df


def _ema_within_breath(
    pred: np.ndarray, breath_id: np.ndarray, alpha: float = 0.15
) -> np.ndarray:
    """
    CHANGE (score improvement toward target):
    Smooth ridge predictions with a simple causal EMA within each breath_id.
    This keeps evaluation semantics intact (still predicting pressure per timestep),
    reduces high-frequency noise, and is fully causal (uses only past within-breath info).
    """
    out = pred.astype(np.float32).copy()
    n = len(out)
    i = 0
    while i < n:
        bid = breath_id[i]
        j = i + 1
        while j < n and breath_id[j] == bid:
            j += 1
        prev = out[i]
        for k in range(i + 1, j):
            prev = alpha * out[k] + (1.0 - alpha) * prev
            out[k] = prev
        i = j
    return out


def _fit_ridge_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Use a lightweight ridge regression to map engineered features to pressure,
    then apply a causal within-breath EMA smoothing to reduce MAE.
    """
    from sklearn.linear_model import Ridge

    tr = _make_features(train_df)
    te = _make_features(test_df)

    feature_cols = [
        "R",
        "C",
        "time_step",
        "RC",
        "R_div_C",
        "u_in",
        "u_out",
        "u_in_lag1",
        "u_in_lag2",
        "u_out_lag1",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_cumsum",
        "u_out_cumsum",
        "dt",
        "u_in_integral",
        "u_in_x_time",
        "u_in_x_R",
        "u_in_x_C",
        "u_out_x_u_in",
    ]

    X_tr = tr[feature_cols].astype(np.float32).to_numpy()
    y_tr = tr["pressure"].astype(np.float32).to_numpy()
    X_te = te[feature_cols].astype(np.float32).to_numpy()

    model = Ridge(alpha=1.0, random_state=2021)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te).astype(np.float32)

    pred = _ema_within_breath(pred, test_df["breath_id"].to_numpy(), alpha=0.15)
    return pred


def g(dp):
    """
    Main blending function.

    If external high-score submissions are missing/empty, use a train-fitted ridge regression
    on causal per-breath control features, then snap to the known pressure grid.
    """
    allow = [1348, 1359, 1758]

    l = []
    if os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            try:
                file_lb = int(i.split("/")[-1].split(".")[1].split(" ")[0])
            except Exception:
                continue
            if file_lb in allow:
                l.append(i)
    l.sort()

    loop_time = 125
    splits = 2

    flist = []
    if len(l) == 0:
        flist = []
    else:
        for si in range(splits):
            start = si * round(len(l) / splits)
            if si == splits - 1:
                flist.append(l[start:])
            else:
                end = (si + 1) * round(len(l) / splits)
                flist.append(l[start:end])

    for i in range(len(flist)):
        flist[i] = wc(flist[i])
    flist = [x for x in flist if x is not None]

    sample = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )

    if len(flist) == 0:
        df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

        pred = _fit_ridge_and_predict(df_train, df_test)

        out = sample.copy()
        out["pressure"] = pred
        out["pressure"] = out["pressure"].apply(find_nearest)
        out.to_csv("submission.csv", index=False)
        return

    if len(flist) == 1:
        out = sample.copy()
        out["pressure"] = flist[0]
        out["pressure"] = out["pressure"].apply(find_nearest)
        out.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for t in range(loop_time):
        weight = []
        set_seed(t)
        for j in range(len(flist)):
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

    out = sample.copy()
    out["pressure"] = np.median(np.vstack(pred_list), axis=0)
    out["pressure"] = out["pressure"].apply(find_nearest)
    out.to_csv("submission.csv", index=False)




## === cell 2
g("../input/ventilator-pressure-high-score-submissions")
print("Wrote submission.csv:", os.path.exists("submission.csv"))
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns)

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

0.1432423129282703

# 6. Current score

9.88149

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92111) has done: 'Your current code fails because it expects a directory of external “high-score submissions” that is not present in this Kaggle environment, causing empty lists and an `IndexError` during blending. To keep the core “blend multiple submissions then snap to nearest valid pressure” logic intact while making it runnable end-to-end, I add a safe fallback: if no eligible submission files are found, train a simple per-(R,C,time_step,u_out) median lookup model from `train.csv` and predict for `test.csv`. This is score-reasonable (not random), uses only provided data, and still outputs `id,pressure` with valid snapping via your existing `find_nearest`. Finally, I ensure the notebook writes a `.csv` submission file to the working directory.'
- What this solution (achieved 9.88149) has done: 'Your current fallback median-lookup is valid but far too weak for this competition, which is why the MAE is ~9.92 vs your target ~0.143 (lower is better). To move substantially toward the target without changing the “no ML model / pure pandas aggregation + snapping to valid pressures” core approach, I upgrade the fallback to a breath-aware physical proxy: compute cumulative inspired volume (`u_in` integrated over time) per breath and learn a per-(R,C,time_step,u_out) median mapping from that state to pressure, with safe backoffs. This keeps the solution as a deterministic lookup/regression-by-aggregation and retains your existing `find_nearest` snapping and submission writing. The blending path is kept intact; it only run if those external submissions actually exist.'

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
    Blend a list of submission filepaths.
    Original code assumed exactly 2 items and specific filename patterns; we keep the intent
    but make it robust to 0/1/N files so it doesn't crash.
    """
    preds = []
    scores = []
    allow = [1348, 1358, 1758]

    for fp in input_list:
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            continue

        if public_lb_score in allow:
            scores.append(public_lb_score)
            preds.append(pd.read_csv(fp).pressure.to_numpy().ravel())

    if len(preds) == 0:
        return None

    if len(preds) == 1:
        return preds[0]

    order = np.argsort(scores)[::-1]
    preds = [preds[i] for i in order[:2]]
    scores = [scores[i] for i in order[:2]]

    l_sum = sum(scores)
    weight1 = (scores[1] / l_sum) + 0.1
    weight1 = float(np.clip(weight1, 0.0, 1.0))
    weight2 = 1.0 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def _add_engineered_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, competition-relevant feature engineering for the fallback:
    - cumulative inspired volume proxy: u_in integrated over time within each breath
    This is directly tied to pressure dynamics and uses only provided columns.
    """
    out = df.copy()

    out.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    dt = out.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0).to_numpy()
    out["dt"] = dt
    out["u_in_dt"] = out["u_in"].to_numpy() * out["dt"].to_numpy()
    out["u_in_cum"] = out.groupby("breath_id", sort=False)["u_in_dt"].cumsum()

    out.sort_values("id", inplace=True, kind="mergesort")
    return out


def _fallback_model_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> np.ndarray:
    """
    Stronger fallback when external submission files aren't available.

    Change (score-improving, still same core semantics: pure aggregation lookup + snapping):
    Learn median pressure as a function of (R,C,u_out,time_step,u_in_cum_bin). This captures
    the main physical state (volume) instead of only instantaneous controls.

    Safe backoffs:
      1) (R,C,u_out,time_step,u_in_cum_bin)
      2) (R,C,u_out,time_step)
      3) (R,C,time_step)
      4) global median
    """
    tr = _add_engineered_state(train_df)
    te = _add_engineered_state(test_df)

    q = np.linspace(0, 1, 51)  # 50 bins
    edges = np.unique(np.quantile(tr["u_in_cum"].to_numpy(), q))
    if edges.size < 3:
        edges = np.array([tr["u_in_cum"].min(), tr["u_in_cum"].max() + 1e-6])

    def to_bin(x: np.ndarray) -> np.ndarray:
        b = np.digitize(x, edges[1:-1], right=False).astype(np.int16)
        return b

    tr["u_in_cum_bin"] = to_bin(tr["u_in_cum"].to_numpy())
    te["u_in_cum_bin"] = to_bin(te["u_in_cum"].to_numpy())

    keys1 = ["R", "C", "u_out", "time_step", "u_in_cum_bin"]
    keys2 = ["R", "C", "u_out", "time_step"]
    keys3 = ["R", "C", "time_step"]

    med1 = tr.groupby(keys1, sort=False)["pressure"].median()
    med2 = tr.groupby(keys2, sort=False)["pressure"].median()
    med3 = tr.groupby(keys3, sort=False)["pressure"].median()
    global_med = float(tr["pressure"].median())

    k1 = pd.MultiIndex.from_frame(te[keys1])
    k2 = pd.MultiIndex.from_frame(te[keys2])
    k3 = pd.MultiIndex.from_frame(te[keys3])

    p = med1.reindex(k1).to_numpy()
    mask = np.isnan(p)
    if mask.any():
        p2 = med2.reindex(k2[mask]).to_numpy()
        p[mask] = p2

        mask = np.isnan(p)
        if mask.any():
            p3 = med3.reindex(k3[mask]).to_numpy()
            p[mask] = p3

            mask = np.isnan(p)
            if mask.any():
                p[mask] = global_med

    return p


def g(dp):
    """
    Original intent: blend a set of existing high-score submissions from dp.
    Fix: if dp doesn't exist / contains no eligible files, fall back to a simple trained lookup
    so we still generate a valid .csv submission end-to-end.
    """
    allow = [1348, 1358, 1758]
    l = []

    if dp is not None and os.path.isdir(dp):
        for fp in glob.iglob(f"{dp}/*"):
            try:
                file_lb = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
            except Exception:
                continue
            if file_lb in allow:
                l.append(fp)

    l.sort()

    if len(l) > 0:
        loop_time = 125
        splits = 2

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
            df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
            preds = _fallback_model_predict(df_train, df_test)
            output = pd.read_csv(
                "../input/ventilator-pressure-prediction/sample_submission.csv"
            )
            output["pressure"] = preds
            output["pressure"] = output["pressure"].apply(find_nearest)
            output.to_csv("submission.csv", index=False)
            return

        pred_list = []
        for seed in range(loop_time):
            weight = []
            set_seed(seed)
            for _ in range(len(flist)):
                weight.append(rd())
            weight_sum = sum(weight)
            weight = [w / weight_sum for w in weight]
            weight.sort(reverse=True)

            temp = 0
            for j in range(len(flist)):
                temp += flist[j] * weight[j]
            pred_list.append(temp)
            del temp
            gc.collect()

        output = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        output["pressure"] = np.median(np.vstack(pred_list), axis=0)
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    preds = _fallback_model_predict(df_train, df_test)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output["pressure"] = preds
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)




## === cell 2
g("../input/ventilator-pressure-high-score-submissions")
print("Wrote submission.csv")
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())
print(sub.isna().sum().to_dict())

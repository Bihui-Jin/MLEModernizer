# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
    Deterministic feature engineering for lookup-median regression:
    - Integer time index within each breath (robust vs float-key mismatches).
    - Cumulative inspired volume proxy: integral of u_in over time (u_in_dt cumulative sum).
    - Small lag history of controls/state (u_in, u_out, u_in_cum) to capture dynamics.
    """
    out = df.copy()
    out.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    out["t_idx"] = out.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    dt = out.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0).to_numpy()
    out["dt"] = dt.astype(np.float32)
    out["u_in_dt"] = (
        out["u_in"].to_numpy(dtype=np.float32) * out["dt"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    out["u_in_cum"] = (
        out.groupby("breath_id", sort=False)["u_in_dt"].cumsum().astype(np.float32)
    )

    g = out.groupby("breath_id", sort=False)
    out["u_in_l1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    out["u_in_l2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)
    out["u_out_l1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)
    out["u_in_cum_l1"] = g["u_in_cum"].shift(1).fillna(0.0).astype(np.float32)

    out.sort_values("id", inplace=True, kind="mergesort")
    return out


def _fallback_model_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> np.ndarray:
    """
    Fallback when external submission files aren't available.

    Changes (score-improving, still same core semantics: deterministic median lookup + snapping):
    1) Train medians ONLY on inspiratory phase (u_out==0) to match the evaluation metric.
    2) Add a small sequential "previous pressure bin" key and predict within each breath in order,
       which stabilizes the pressure trajectory without any ML training loops.

    For u_out==1 (expiratory, not scored), we carry forward the last predicted inspiratory pressure
    within the breath (common simple strategy; avoids injecting noisy lookup there).
    """
    tr_full = _add_engineered_state(train_df)
    te = _add_engineered_state(test_df)

    tr = tr_full[tr_full["u_out"].to_numpy() == 0].copy()

    q_cum = np.linspace(0, 1, 81)  # 80 bins for cumulative volume
    edges_cum = np.unique(np.quantile(tr["u_in_cum"].to_numpy(), q_cum))
    if edges_cum.size < 3:
        edges_cum = np.array([tr["u_in_cum"].min(), tr["u_in_cum"].max() + 1e-6])

    q_uin = np.linspace(0, 1, 51)  # 50 bins for instantaneous u_in / lagged u_in
    edges_uin = np.unique(np.quantile(tr["u_in"].to_numpy(), q_uin))
    if edges_uin.size < 3:
        edges_uin = np.array([tr["u_in"].min(), tr["u_in"].max() + 1e-6])

    def pressure_to_bin(p_arr: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(sorted_pressures, p_arr, side="left")
        idx = np.clip(idx, 0, total_pressures_len - 1).astype(np.int16)
        return idx

    def to_bin(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.digitize(x, edges[1:-1], right=False).astype(np.int16)

    for df in (tr, te):
        df["u_in_cum_bin"] = to_bin(df["u_in_cum"].to_numpy(), edges_cum)
        df["u_in_bin"] = to_bin(df["u_in"].to_numpy(), edges_uin)
        df["u_in_l1_bin"] = to_bin(df["u_in_l1"].to_numpy(), edges_uin)

    tr["p_bin"] = pressure_to_bin(tr["pressure"].to_numpy(dtype=np.float32))
    tr.sort_values(["breath_id", "t_idx"], inplace=True, kind="mergesort")
    tr["p_prev_bin"] = (
        tr.groupby("breath_id", sort=False)["p_bin"]
        .shift(1)
        .fillna(-1)
        .astype(np.int16)
    )

    keys_seq = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_cum_bin",
        "u_in_bin",
        "u_in_l1_bin",
        "u_out_l1",
        "p_prev_bin",
    ]
    keys_base = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_cum_bin",
        "u_in_bin",
        "u_in_l1_bin",
        "u_out_l1",
    ]
    keys_rc_t = ["R", "C", "t_idx"]

    med_seq = tr.groupby(keys_seq, sort=False)["pressure"].median()
    med_base = tr.groupby(keys_base, sort=False)["pressure"].median()
    med_rc_t = tr.groupby(keys_rc_t, sort=False)["pressure"].median()
    global_med = float(tr["pressure"].median())

    te_sorted = te.sort_values(["breath_id", "t_idx"], kind="mergesort").copy()
    pred = np.empty(len(te_sorted), dtype=np.float32)

    def _reindex_one(series, row_df, key_cols):
        mi = pd.MultiIndex.from_frame(row_df[key_cols])
        return series.reindex(mi).to_numpy()

    te_sorted["p_prev_bin"] = -1  # will be updated per breath in loop

    breath_groups = te_sorted.groupby("breath_id", sort=False).indices
    for bid, idxs in breath_groups.items():
        last_pbin = -1
        for pos in idxs:
            te_sorted.iat[pos, te_sorted.columns.get_loc("p_prev_bin")] = last_pbin
            row = te_sorted.iloc[[pos]]

            p_val = _reindex_one(med_seq, row, keys_seq)[0]
            if np.isnan(p_val):
                p_val = _reindex_one(med_base, row, keys_base)[0]
            if np.isnan(p_val):
                p_val = _reindex_one(med_rc_t, row, keys_rc_t)[0]
            if np.isnan(p_val):
                p_val = global_med

            u_out_val = int(row["u_out"].iloc[0])
            if u_out_val == 1 and last_pbin != -1:
                p_val = float(sorted_pressures[last_pbin])

            pred[pos] = p_val
            snapped = find_nearest(p_val)
            last_pbin = int(np.searchsorted(sorted_pressures, snapped, side="left"))
            last_pbin = max(0, min(total_pressures_len - 1, last_pbin))

    te_sorted["pred"] = pred
    te_out = te_sorted.sort_values("id", kind="mergesort")
    return te_out["pred"].to_numpy()


def g(dp):
    """
    Original intent: blend a set of existing high-score submissions from dp.
    If dp doesn't exist / contains no eligible files, fall back to deterministic lookup
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

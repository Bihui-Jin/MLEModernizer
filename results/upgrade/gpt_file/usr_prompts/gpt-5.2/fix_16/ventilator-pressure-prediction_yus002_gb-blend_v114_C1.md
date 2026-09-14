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

SAMPLE_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"

sample_submission_df = pd.read_csv(SAMPLE_PATH)
sample_ids_global = sample_submission_df["id"].to_numpy()
n_global = len(sample_submission_df)

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


def snap_to_known_pressures(pred_arr: np.ndarray) -> np.ndarray:
    x = np.asarray(pred_arr, dtype=np.float64).reshape(-1)
    idx = np.searchsorted(sorted_pressures, x, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)
    prev_idx = np.clip(idx - 1, 0, total_pressures_len - 1)

    upper = sorted_pressures[idx]
    lower = sorted_pressures[prev_idx]
    choose_lower = np.abs(x - lower) < np.abs(upper - x)
    out = np.where(choose_lower, lower, upper)
    return out.astype(np.float64, copy=False)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def _load_pred_file(fp, sample_ids):
    """
    Robustly load a prediction file and align to sample_submission ids.
    Returns a numpy array of shape (n_rows,) or None if invalid.
    """
    try:
        df = pd.read_csv(fp)
    except Exception:
        return None

    if "pressure" not in df.columns:
        return None

    if "id" in df.columns:
        try:
            df2 = df[["id", "pressure"]].copy()
            df2 = df2.dropna(subset=["id", "pressure"])
            df2["id"] = df2["id"].astype(np.int64, errors="ignore")
            df2 = df2.set_index("id")
            aligned = df2.reindex(sample_ids)["pressure"].to_numpy()
        except Exception:
            aligned = df["pressure"].to_numpy()
    else:
        aligned = df["pressure"].to_numpy()

    if aligned.ndim != 1:
        aligned = aligned.reshape(-1)

    if len(aligned) != len(sample_ids):
        return None

    return aligned.astype(np.float64, copy=False)


def wc(input_list):
    l = []
    preds = []
    sample_ids = sample_ids_global

    for i in range(len(input_list)):
        fp = input_list[i]
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        pred = _load_pred_file(fp, sample_ids)
        if pred is None:
            continue
        l.append(public_lb_score)
        preds.append(pred)

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    l_sum = sum(l) if sum(l) != 0 else 1
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _make_breath_features(df, is_train=False):
    """
    Existing baseline feature engineering (kept) for fallback group-mean lookup.
    """
    out = df.copy()
    out["ts2"] = np.round(out["time_step"].to_numpy(), 2)
    out["u_in_lag1"] = out.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
    out["u_in_cum"] = out.groupby("breath_id", sort=False)["u_in"].cumsum()
    out["u_in_lag1_b"] = np.round(out["u_in_lag1"].to_numpy(), 1)
    out["u_in_cum_b"] = np.round(out["u_in_cum"].to_numpy(), 1)
    if is_train:
        out["pressure"] = out["pressure"].astype(np.float64)
    return out


def _build_timebucket_baseline_predictions(sample_ids):
    """
    Existing (kept) training-free baseline for fallback.
    """
    test = pd.read_csv(TEST_PATH)

    tr = _make_breath_features(
        df_train[["breath_id", "R", "C", "u_out", "time_step", "u_in", "pressure"]],
        is_train=True,
    )
    te = _make_breath_features(
        test[["id", "breath_id", "R", "C", "u_out", "time_step", "u_in"]],
        is_train=False,
    )

    mean_map_hist = (
        tr.groupby(["R", "C", "u_out", "ts2", "u_in_lag1_b", "u_in_cum_b"], sort=False)[
            "pressure"
        ]
        .mean()
        .astype(np.float64)
    )

    mean_map_ts = (
        tr.groupby(["R", "C", "u_out", "ts2"], sort=False)["pressure"]
        .mean()
        .astype(np.float64)
    )
    mean_rcu = (
        tr.groupby(["R", "C", "u_out"], sort=False)["pressure"]
        .mean()
        .astype(np.float64)
    )
    global_mean = float(tr["pressure"].mean())

    te = te.join(
        mean_map_hist.rename("p_hist"),
        on=["R", "C", "u_out", "ts2", "u_in_lag1_b", "u_in_cum_b"],
    )
    te = te.join(mean_map_ts.rename("p_ts"), on=["R", "C", "u_out", "ts2"])
    te = te.join(mean_rcu.rename("p_rcu"), on=["R", "C", "u_out"])

    pred = te["p_hist"].to_numpy(dtype=np.float64)
    m = np.isnan(pred)
    if m.any():
        pred[m] = te.loc[m, "p_ts"].to_numpy(dtype=np.float64)
    m2 = np.isnan(pred)
    if m2.any():
        pred[m2] = te.loc[m2, "p_rcu"].to_numpy(dtype=np.float64)
    m3 = np.isnan(pred)
    if m3.any():
        pred[m3] = global_mean

    te["pred_filled"] = pred
    pred_aligned = (
        te.set_index("id").reindex(sample_ids)["pred_filled"].to_numpy(dtype=np.float64)
    )
    pred_aligned = np.asarray(pred_aligned, dtype=np.float64).reshape(-1)
    return pred_aligned


def _build_nn_breath_predictions(sample_ids):
    """
    Keep the same lookup-only core, but speed up per-breath fallback by avoiding
    per-breath DataFrame filtering (isin/loc) and instead using already-grouped
    rows aligned to ids_arr. This preserves identical semantics.
    """
    test = pd.read_csv(
        TEST_PATH,
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )
    tr = df_train[
        ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    ].copy()

    fallback_pred = _build_timebucket_baseline_predictions(sample_ids)
    pred_by_id = pd.Series(index=sample_ids, dtype=np.float64)

    def _traj_key(u_in_arr, u_out_arr, ui_dec=2):
        ui = np.ascontiguousarray(
            np.round(np.asarray(u_in_arr, dtype=np.float32), ui_dec)
        )
        uo = np.ascontiguousarray(np.asarray(u_out_arr, dtype=np.int8))
        return ui.tobytes() + b"|" + uo.tobytes()

    tr["_ts_i"] = (
        np.round(tr["time_step"].to_numpy(dtype=np.float64), 2) * 100.0
    ).astype(np.int16)
    test["_ts_i"] = (
        np.round(test["time_step"].to_numpy(dtype=np.float64), 2) * 100.0
    ).astype(np.int16)

    tr["_uin_i"] = (np.round(tr["u_in"].to_numpy(dtype=np.float64), 2) * 100.0).astype(
        np.int16
    )
    test["_uin_i"] = (
        np.round(test["u_in"].to_numpy(dtype=np.float64), 2) * 100.0
    ).astype(np.int16)

    tr["_ucum_i"] = (
        np.round(tr.groupby("breath_id", sort=False)["u_in"].cumsum().to_numpy(), 2)
        * 20.0
    ).astype(np.int32)
    test["_ucum_i"] = (
        np.round(test.groupby("breath_id", sort=False)["u_in"].cumsum().to_numpy(), 2)
        * 20.0
    ).astype(np.int32)

    def _fill_ucum_nearest(p_arr, uc_arr, mean_step_ucum_slice):
        m = np.isnan(p_arr)
        if not m.any():
            return p_arr
        idx = mean_step_ucum_slice.index
        if len(idx) == 0:
            return p_arr
        avail_uc = idx.to_numpy(dtype=np.int64)
        avail_p = mean_step_ucum_slice.to_numpy(dtype=np.float64)
        miss_uc = uc_arr[m].astype(np.int64, copy=False)
        order = np.argsort(avail_uc)
        auc = avail_uc[order]
        ap = avail_p[order]
        pos = np.searchsorted(auc, miss_uc, side="left")
        pos = np.clip(pos, 0, len(auc) - 1)
        pos0 = np.clip(pos - 1, 0, len(auc) - 1)
        upper_uc = auc[pos]
        lower_uc = auc[pos0]
        choose_lower = np.abs(miss_uc - lower_uc) <= np.abs(upper_uc - miss_uc)
        nearest_pos = np.where(choose_lower, pos0, pos)
        p_arr[m] = ap[nearest_pos]
        return p_arr

    rc_keys = sorted(set(zip(test["R"].values.tolist(), test["C"].values.tolist())))
    for R, C in rc_keys:
        tr_block = tr[(tr["R"].values == R) & (tr["C"].values == C)]
        te_block = test[(test["R"].values == R) & (test["C"].values == C)]
        if te_block.empty or tr_block.empty:
            continue

        mean_step_uin = (
            tr_block.groupby(["u_out", "_ts_i", "_uin_i"], sort=False)["pressure"]
            .mean()
            .astype(np.float64)
        )
        mean_step = (
            tr_block.groupby(["u_out", "_ts_i"], sort=False)["pressure"]
            .mean()
            .astype(np.float64)
        )

        mean_step_ucum = (
            tr_block.groupby(["u_out", "_ts_i", "_ucum_i"], sort=False)["pressure"]
            .mean()
            .astype(np.float64)
        )

        tr_in = tr_block.groupby("breath_id", sort=False)["u_in"].apply(np.asarray)
        tr_out = tr_block.groupby("breath_id", sort=False)["u_out"].apply(np.asarray)
        tr_p = tr_block.groupby("breath_id", sort=False)["pressure"].apply(np.asarray)

        te_in = te_block.groupby("breath_id", sort=False)["u_in"].apply(np.asarray)
        te_out = te_block.groupby("breath_id", sort=False)["u_out"].apply(np.asarray)
        te_ids = te_block.groupby("breath_id", sort=False)["id"].apply(np.asarray)
        te_uout = te_block.groupby("breath_id", sort=False)["u_out"].apply(np.asarray)
        te_ts = te_block.groupby("breath_id", sort=False)["_ts_i"].apply(np.asarray)
        te_uin_i = te_block.groupby("breath_id", sort=False)["_uin_i"].apply(np.asarray)
        te_ucum_i = te_block.groupby("breath_id", sort=False)["_ucum_i"].apply(
            np.asarray
        )

        lengths_tr = tr_in.apply(len).to_numpy()
        if len(lengths_tr) == 0:
            continue
        modal_len = int(pd.Series(lengths_tr).mode().iloc[0])

        ok_tr = lengths_tr == modal_len
        tr_in = tr_in.loc[ok_tr]
        tr_out = tr_out.loc[ok_tr]
        tr_p = tr_p.loc[ok_tr]
        if tr_in.empty:
            continue

        lengths_te = te_in.apply(len).to_numpy()
        if len(lengths_te) == 0:
            continue
        ok_te = lengths_te == modal_len
        te_in_ok = te_in.loc[ok_te]
        te_out_ok = te_out.loc[ok_te]
        te_ids_ok = te_ids.loc[ok_te]
        te_uout_ok = te_uout.loc[ok_te]
        te_ts_ok = te_ts.loc[ok_te]
        te_uin_i_ok = te_uin_i.loc[ok_te]
        te_ucum_i_ok = te_ucum_i.loc[ok_te]
        if te_in_ok.empty:
            continue

        tr_keys = {}
        for bid in tr_in.index.to_numpy():
            k = _traj_key(tr_in.loc[bid], tr_out.loc[bid], ui_dec=2)
            p = np.asarray(tr_p.loc[bid], dtype=np.float32)
            if k in tr_keys:
                tr_keys[k][0] += p
                tr_keys[k][1] += 1
            else:
                tr_keys[k] = [p.copy(), 1]

        tr_keys_relaxed = {}
        for bid in tr_in.index.to_numpy():
            k2 = _traj_key(tr_in.loc[bid], tr_out.loc[bid], ui_dec=1)
            p = np.asarray(tr_p.loc[bid], dtype=np.float32)
            if k2 in tr_keys_relaxed:
                tr_keys_relaxed[k2][0] += p
                tr_keys_relaxed[k2][1] += 1
            else:
                tr_keys_relaxed[k2] = [p.copy(), 1]

        for bid in te_in_ok.index.to_numpy():
            k = _traj_key(te_in_ok.loc[bid], te_out_ok.loc[bid], ui_dec=2)
            if k in tr_keys:
                p_sum, cnt = tr_keys[k]
                p_avg = (p_sum / float(cnt)).astype(np.float64)
                pred_by_id.loc[te_ids_ok.loc[bid]] = p_avg
                continue

            k2 = _traj_key(te_in_ok.loc[bid], te_out_ok.loc[bid], ui_dec=1)
            if k2 in tr_keys_relaxed:
                p_sum, cnt = tr_keys_relaxed[k2]
                p_avg = (p_sum / float(cnt)).astype(np.float64)
                pred_by_id.loc[te_ids_ok.loc[bid]] = p_avg
                continue

            ids_arr = te_ids_ok.loc[bid]
            uo = te_uout_ok.loc[bid].astype(np.int64, copy=False)
            ts = te_ts_ok.loc[bid].astype(np.int64, copy=False)
            uin_i = te_uin_i_ok.loc[bid].astype(np.int64, copy=False)
            uc = te_ucum_i_ok.loc[bid].astype(np.int64, copy=False)

            mi_uc = pd.MultiIndex.from_arrays(
                [uo, ts, uc],
                names=["u_out", "_ts_i", "_ucum_i"],
            )
            p = mean_step_ucum.reindex(mi_uc).to_numpy(dtype=np.float64)

            m_uc = np.isnan(p)
            if m_uc.any():
                pairs = np.unique(np.stack([uo[m_uc], ts[m_uc]], axis=1), axis=0)
                for key in pairs:
                    uok, tsk = int(key[0]), int(key[1])
                    mask = m_uc & (uo == uok) & (ts == tsk)
                    if not mask.any():
                        continue
                    try:
                        slice_series = mean_step_ucum.xs(
                            (uok, tsk), level=("u_out", "_ts_i"), drop_level=False
                        )
                    except KeyError:
                        continue
                    p_masked = p[mask].copy()
                    p_filled = _fill_ucum_nearest(
                        p_masked,
                        uc[mask],
                        slice_series.droplevel(["u_out", "_ts_i"]),
                    )
                    p[mask] = p_filled

            m = np.isnan(p)
            if m.any():
                mi_uin = pd.MultiIndex.from_arrays(
                    [uo[m], ts[m], uin_i[m]],
                    names=["u_out", "_ts_i", "_uin_i"],
                )
                p[m] = mean_step_uin.reindex(mi_uin).to_numpy(dtype=np.float64)

            m2 = np.isnan(p)
            if m2.any():
                mi2 = pd.MultiIndex.from_arrays(
                    [uo[m2], ts[m2]],
                    names=["u_out", "_ts_i"],
                )
                p[m2] = mean_step.reindex(mi2).to_numpy(dtype=np.float64)

            pred_by_id.loc[ids_arr] = p

        del tr_block, te_block, tr_in, tr_out, tr_p, te_in, te_out, te_ids
        gc.collect()

    fb_map = pd.Series(fallback_pred, index=sample_ids, dtype=np.float64)
    missing = pred_by_id.isna()
    if missing.any():
        pred_by_id.loc[missing] = fb_map.loc[pred_by_id.index[missing]].to_numpy(
            dtype=np.float64
        )

    pred_aligned = pred_by_id.reindex(sample_ids).to_numpy(dtype=np.float64)
    return pred_aligned


def g(dp):
    sample = sample_submission_df
    sample_ids = sample_ids_global
    n = n_global

    l = []
    if os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                l.append(i)

    file_count = len(l)
    loop_time = 154

    if file_count == 0:
        baseline_pred = _build_nn_breath_predictions(sample_ids)
        merged = [baseline_pred]
    else:
        splits = max(1, file_count // 2)
        l.sort()
        flist = []
        for i in range(splits):
            if i == splits - 1:
                flist.append(l[i * round(len(l) / splits) :])
            else:
                flist.append(
                    l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
                )

        merged = []
        for i in range(len(flist)):
            pred = wc(flist[i])
            if pred is not None and len(pred) == n:
                merged.append(pred)

        if len(merged) == 0:
            baseline_pred = _build_nn_breath_predictions(sample_ids)
            merged = [baseline_pred]

    mlen = len(merged)

    pred_mat = np.empty((loop_time, n), dtype=np.float32)

    for k in range(loop_time):
        weight = []
        set_seed(k)
        for j in range(mlen):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(mlen):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for j in range(mlen):
            temp += merged[j] * weight[j]
        pred_mat[k, :] = temp.astype(np.float32, copy=False)

    output = sample.copy()
    output_pred = np.median(pred_mat, axis=0).astype(np.float64, copy=False)
    output["pressure"] = snap_to_known_pressures(output_pred)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    sample = sample_submission_df
    sample_ids = sample_ids_global

    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)

    if "id" in a_df.columns:
        a_p = a_df.set_index("id").reindex(sample_ids)["pressure"].to_numpy()
    else:
        a_p = a_df["pressure"].to_numpy()
    if "id" in b_df.columns:
        b_p = b_df.set_index("id").reindex(sample_ids)["pressure"].to_numpy()
    else:
        b_p = b_df["pressure"].to_numpy()

    out = sample.copy()
    out["pressure"] = a_p * 0.6 + b_p * 0.4
    out["pressure"] = snap_to_known_pressures(
        out["pressure"].to_numpy(dtype=np.float64)
    )
    out.to_csv("blend.csv", index=False)
    return out




## === cell 1
g("../input/gb-data-blending-recover")

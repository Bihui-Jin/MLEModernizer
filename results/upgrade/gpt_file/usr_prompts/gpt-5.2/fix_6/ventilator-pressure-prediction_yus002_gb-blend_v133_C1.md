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

0.1363000140389698

# 6. Current score

0.68119

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.31337) has done: 'The crash happens because the blend directory (`../input/gb-data-blending-recover`) is not available in your environment, so the glob finds no prediction CSVs and the code ends up producing a single scalar median instead of a 603600-length vector. I add a safe fallback that (1) searches for any out-of-fold/prediction CSVs under `../input`, and if none exist, (2) trains a minimal baseline (group-wise mean pressure by `(R,C,time_step,u_in,u_out)` with a global mean fallback) to guarantee a valid `submission.csv`. I also make the prediction-loading more robust (auto-detect the right column, enforce correct length, and align to the sample submission `id` order) and ensure the output filename ends with `.csv`. This preserves the original ensemble/blending intent when prediction files exist, and otherwise produces a legitimate baseline submission that runs end-to-end.'
- What this solution (achieved 1.95174) has done: 'Your current score is far worse than the target (lower is better), and the biggest issue is that the code falls back to a very weak baseline when the external blend directory is missing. To move the score sharply toward the target while keeping the overall “no deep model, simple pipeline” core logic, I keep the same submission-writing flow but upgrade the fallback baseline to a known strong classical approach for this competition: nearest-neighbor matching on engineered time-series features per breath, with a safe fallback when no neighbor is found. I also ensure we only score inspiratory-like rows by using the known physical constraint that pressure during `u_out==1` is near the minimum (this helps MAE because expiratory phase isn’t scored, but models often mispredict there), and then still snap predictions to the nearest allowed pressure values via your existing `find_nearest`. These changes stay within the existing structure (a fallback generator) and should reduce MAE substantially toward your target without altering your blending logic when external prediction CSVs exist.'
- What this solution (achieved 0.65554) has done: 'Your current score (1.95174, lower-is-better) is far worse than the target, so we should strengthen the existing fallback while keeping your blending/median ensemble flow intact. The biggest score drag in your fallback is forcing `u_out==1` rows to the minimum pressure; those rows are *not scored* but are included in the MAE denominator, so setting them to a constant often hurts badly—removing that override should improve the score substantially without changing your model/blending logic. Additionally, your nearest-neighbor matching currently uses only a coarse per-breath summary; we can keep the same NN approach but make the match more faithful by comparing full inspiratory `u_in` sequences (still classical, still the same fallback pipeline). Finally, we keep snapping to the discrete allowed pressure grid via your existing `find_nearest` and ensure the submission aligns to `sample_submission.csv` by `id`.'
- What this solution (achieved 0.65554) has done: 'Your score is much worse than the target (lower is better), so we should improve the fallback path (which is what you’re actually using because `../input/gb-data-blending-recover` doesn’t exist). The smallest high-impact change without altering the overall “nearest-neighbor breath matching + copy pressure trajectory + snap-to-grid” core logic is to make the NN matching cheaper and more accurate by precomputing `breath_id -> row index` maps instead of repeatedly searching with `np.where` (which is both slow and can cause timeouts/instability). I also add a deterministic tie-break (use L2 on the same inspiratory `u_in` sequence when L1 ties) while keeping the same features and copying behavior; this typically nudges MAE down without changing the approach. Finally, I keep the submission alignment-by-`id` but add a hard length check to guarantee a valid 603600-row `submission.csv`.'
- What this solution (achieved 0.68119) has done: 'Your current MAE (0.65554, lower-is-better) is far above the target, so we should improve the score by making the existing nearest-neighbor fallback more faithful to the competition metric without changing the overall approach. The smallest high-impact fix is to ensure we only copy/compare inspiratory portions (where `u_out==0`) and to avoid using expiratory time steps in the breath-distance calculation, since expiratory rows are not scored and can distort NN matching. We keep the same “match a test breath to a train breath, copy the pressure trajectory, fallback to grouped-mean, then snap to the discrete pressure grid” logic, but compute NN distances on inspiratory indices only and copy pressures for inspiratory steps while using the grouped-mean fallback for expiratory steps. This typically reduces MAE substantially versus mixing inspiratory/expiratory dynamics in the distance and copy.'

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
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

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


def _read_pred_file(fp, expected_len):
    """Robustly read a prediction file and return a 1D numpy array of length expected_len (or raise)."""
    df = pd.read_csv(fp)
    if "pressure" in df.columns:
        arr = df["pressure"].to_numpy()
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 0:
            raise ValueError(f"No numeric columns found in {fp}")
        arr = df[num_cols[-1]].to_numpy()
    arr = np.asarray(arr).ravel()
    if len(arr) != expected_len:
        raise ValueError(
            f"Prediction length mismatch in {fp}: got {len(arr)} expected {expected_len}"
        )
    return arr


def wc(input_list):
    pressures = []
    lbs = []
    for i in range(len(input_list)):
        fp = input_list[i]
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
            lbs.append(public_lb_score)
        except Exception:
            lbs.append(1)
        pressures.append(_read_pred_file(fp, expected_len=603600))
    lbs_sum = sum(lbs) if sum(lbs) != 0 else len(lbs)

    if len(pressures) == 1:
        return pressures[0]

    weight1 = (lbs[1] / lbs_sum) + 0.1
    weight1 = float(np.clip(weight1, 0.0, 1.0))
    weight2 = 1.0 - weight1
    return pressures[0] * weight1 + pressures[1] * weight2


def _make_breath_features(df):
    """
    Minimal feature engineering used only inside the fallback baseline.
    This is directly score-relevant for this competition and keeps the rest of the pipeline intact.
    """
    df = df.copy()

    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    df["t_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)

    df["u_in_lag1"] = (
        df.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
    )
    df["u_out_lag1"] = (
        df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )

    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum().astype(np.float32)

    df["u_in_x_time"] = (df["u_in"] * df["time_step"]).astype(np.float32)
    df["is_exhale"] = (df["u_out"] == 1).astype(np.int8)

    return df


def _nearest_neighbor_fallback(out_path="submission.csv"):
    """
    Score-improving fallback: nearest-neighbor matching between test breaths and train breaths
    using time-series features. Keeps the same 'fallback generator' logic.

    Change (score relevant, core logic preserved):
    - Compute NN distance using ONLY inspiratory indices (u_out==0) so expiratory phase
      does not distort matching (metric scores inspiratory only).
    - When copying pressure trajectory from matched train breath, copy ONLY inspiratory
      time steps; for expiratory steps use the grouped-mean fallback (same fallback already present).
    """
    df_test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_SUB_PATH)

    tr = _make_breath_features(df_train)
    te = _make_breath_features(df_test)

    def _build_uin_uout_matrix(dff):
        dff = dff.sort_values(["breath_id", "t_idx"], kind="mergesort")
        bids = dff["breath_id"].unique()
        bid_to_i = {int(b): i for i, b in enumerate(bids)}
        mat_uin = np.zeros((len(bids), 80), dtype=np.float32)
        mat_uout = np.ones((len(bids), 80), dtype=np.int8)  # default exhale
        idx = dff["breath_id"].map(bid_to_i).to_numpy()
        tidx = dff["t_idx"].to_numpy()
        mat_uin[idx, tidx] = dff["u_in"].to_numpy(dtype=np.float32)
        mat_uout[idx, tidx] = dff["u_out"].to_numpy(dtype=np.int8)
        rc = (
            dff.groupby("breath_id", sort=False)[["R", "C"]]
            .first()
            .reset_index()
            .set_index("breath_id")
        )
        return bids, bid_to_i, mat_uin, mat_uout, rc

    tr_bids, tr_bid_to_row, tr_uin_mat, tr_uout_mat, tr_rc = _build_uin_uout_matrix(tr)
    te_bids, te_bid_to_row, te_uin_mat, te_uout_mat, te_rc = _build_uin_uout_matrix(te)

    tr_press = tr[["breath_id", "t_idx", "pressure"]].copy()
    tr_press.sort_values(["breath_id", "t_idx"], inplace=True, kind="mergesort")
    train_breath_ids = tr_press["breath_id"].unique()
    breath_to_offset = {int(bid): i for i, bid in enumerate(train_breath_ids)}
    press_matrix = np.full((len(train_breath_ids), 80), np.nan, dtype=np.float32)
    bidx = tr_press["breath_id"].map(breath_to_offset).to_numpy()
    tidx = tr_press["t_idx"].to_numpy()
    press_matrix[bidx, tidx] = tr_press["pressure"].to_numpy(dtype=np.float32)

    match = {}
    te_rc_df = te_rc.reset_index()
    tr_rc_df = tr_rc.reset_index()

    tr_index_by_rc = {}
    for r, c in tr_rc_df[["R", "C"]].drop_duplicates().itertuples(index=False):
        sel = (tr_rc_df["R"].values == r) & (tr_rc_df["C"].values == c)
        tr_index_by_rc[(int(r), int(c))] = tr_rc_df.loc[sel, "breath_id"].to_numpy()

    chunk = 128
    for (r, c), te_grp in te_rc_df.groupby(["R", "C"], sort=False):
        tr_blist = tr_index_by_rc.get((int(r), int(c)), None)
        if tr_blist is None or len(tr_blist) == 0:
            continue

        tr_rows = np.fromiter((tr_bid_to_row[int(b)] for b in tr_blist), dtype=np.int32)
        te_blist = te_grp["breath_id"].to_numpy()
        te_rows = np.fromiter((te_bid_to_row[int(b)] for b in te_blist), dtype=np.int32)

        tr_uin = tr_uin_mat[tr_rows]  # (n_tr, 80)
        tr_uout = tr_uout_mat[tr_rows]  # (n_tr, 80)
        te_uin = te_uin_mat[te_rows]  # (n_te, 80)
        te_uout = te_uout_mat[te_rows]  # (n_te, 80)

        for i in range(0, len(te_blist), chunk):
            te_chunk_uin = te_uin[i : i + chunk]  # (k, 80)
            te_chunk_uout = te_uout[i : i + chunk]  # (k, 80)

            insp_mask = (te_chunk_uout == 0).astype(np.float32)  # (k, 80), 1 on insp
            diff = (te_chunk_uin[:, None, :] - tr_uin[None, :, :]) * insp_mask[
                :, None, :
            ]
            d1 = np.abs(diff).sum(axis=2)  # (k, n_tr)

            nn_idx = d1.argmin(axis=1)
            min_d1 = d1[np.arange(d1.shape[0]), nn_idx]

            for j in range(d1.shape[0]):
                ties = np.flatnonzero(d1[j] == min_d1[j])
                if ties.size > 1:
                    diff2 = diff[j, ties, :]
                    d2 = (diff2 * diff2).sum(axis=1)
                    nn_idx[j] = ties[int(d2.argmin())]

            for j, tb in enumerate(te_blist[i : i + chunk]):
                match[int(tb)] = int(tr_blist[int(nn_idx[j])])

    key = ["R", "C", "time_step", "u_in", "u_out"]
    agg_mean = df_train.groupby(key, sort=False)["pressure"].mean()
    global_mean = float(df_train["pressure"].mean())

    te_sorted = te.sort_values(["breath_id", "t_idx"], inplace=False, kind="mergesort")
    te_bids_sorted = te_sorted["breath_id"].to_numpy()
    te_tidx_sorted = te_sorted["t_idx"].to_numpy()
    te_uout_sorted = te_sorted["u_out"].to_numpy(dtype=np.int8)

    pred = np.empty(len(te_sorted), dtype=np.float32)
    pred.fill(np.nan)

    for i in range(len(te_sorted)):
        if te_uout_sorted[i] == 1:
            continue
        tb = int(te_bids_sorted[i])
        tt = int(te_tidx_sorted[i])
        mb = match.get(tb, None)
        if mb is not None and mb in breath_to_offset:
            pred[i] = press_matrix[breath_to_offset[mb], tt]

    mapped = pd.MultiIndex.from_frame(te_sorted[key])
    base = agg_mean.reindex(mapped).to_numpy(dtype=np.float32)
    base = np.where(np.isnan(base), global_mean, base).astype(np.float32)
    pred = np.where(np.isnan(pred), base, pred)

    pred = np.array([find_nearest(float(x)) for x in pred], dtype=float)

    sub = sample.copy()
    pred_by_id = pd.Series(pred, index=te_sorted["id"].to_numpy())
    sub["pressure"] = pred_by_id.reindex(sub["id"].values).to_numpy()

    if len(sub) != 603600 or sub["pressure"].isna().any():
        raise RuntimeError(
            f"Invalid submission generated: len={len(sub)}, nan_pressure={sub['pressure'].isna().sum()}"
        )

    sub.to_csv(out_path, index=False)
    return sub


def _baseline_submission(out_path="submission.csv"):
    df_test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_SUB_PATH)

    key = ["R", "C", "time_step", "u_in", "u_out"]
    agg = df_train.groupby(key, sort=False)["pressure"].mean()
    global_mean = float(df_train["pressure"].mean())

    mapped = pd.MultiIndex.from_frame(df_test[key])
    pred = agg.reindex(mapped).to_numpy()
    pred = np.where(np.isnan(pred), global_mean, pred).astype(float)

    pred = np.array([find_nearest(x) for x in pred], dtype=float)

    sub = sample.copy()
    if "id" in df_test.columns and len(df_test) == len(sub):
        test_idx = pd.Index(df_test["id"].values)
        pred_series = pd.Series(pred, index=test_idx)
        sub["pressure"] = pred_series.reindex(sub["id"].values).to_numpy()
    else:
        sub["pressure"] = pred

    if len(sub) != 603600 or sub["pressure"].isna().any():
        raise RuntimeError(
            f"Invalid submission generated: len={len(sub)}, nan_pressure={sub['pressure'].isna().sum()}"
        )

    sub.to_csv(out_path, index=False)
    return sub


def g(dp):
    loop_time = 154

    files = sorted([p for p in glob.glob(f"{dp}/*.csv")])
    if len(files) == 0:
        files = sorted(
            [
                p
                for p in glob.iglob(f"{dp}/*")
                if os.path.isfile(p) and p.lower().endswith(".csv")
            ]
        )

    if len(files) == 0:
        return _nearest_neighbor_fallback(out_path="submission.csv")

    file_count = len(files)
    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        if len(chunk) > 0:
            flist.append(chunk)

    blended_chunks = []
    for ch in flist:
        try:
            blended_chunks.append(wc(ch))
        except Exception:
            continue

    if len(blended_chunks) == 0:
        return _nearest_neighbor_fallback(out_path="submission.csv")

    pred_list = []
    for seed in range(loop_time):
        weight = []
        set_seed(seed)
        for _ in range(len(blended_chunks)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else len(weight)
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(603600, dtype=float)
        for i in range(len(blended_chunks)):
            temp += blended_chunks[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(SAMPLE_SUB_PATH)
    median_pred = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = pd.Series(median_pred).apply(find_nearest).to_numpy()
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
_ = g("../input/gb-data-blending-recover")
print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())

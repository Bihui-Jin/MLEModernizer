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

0.1395938254400038

# 6. Current score

3.70239

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.99689) has done: 'I fix the baseline fallback so it only assigns predictions for inspiratory rows (u_out==0) and leaves expiratory rows as 0, resolving the shape-mismatch crash that prevented any submission from being written. I also make the ID alignment robust by matching the sample submission order (no unintended re-sorting) and ensuring the blended predictions have the correct length with NaNs handled safely. These changes preserve the core logic (weighted blending of four component submissions; baseline only used when those files are missing) while guaranteeing an end-to-end run that writes `submission.csv`. No score-tuning is attempted beyond making the fallback baseline correct, since the current score wasn’t yielded due to runtime errors.'
- What this solution (achieved 3.70239) has done: 'The timeout is dominated by the fallback KNN baseline: it loads the full 5.4M-row train set, then repeatedly builds large distance matrices via `Xt @ Xtr.T` per (R,C) group, which is far too slow in a CPU-only 600s environment. To preserve core logic and output semantics, the optimized version keeps the exact same KNN (same normalization, weighting, k, chunking, clipping) but replaces the quadratic-distance computation with an exact nearest-neighbor search using `sklearn.neighbors.NearestNeighbors` (KDTree/BallTree), which is mathematically equivalent for Euclidean distance and drastically reduces runtime. It also avoids repeated pandas indexing/reindexing by aligning submissions via a single merge-sort index mapping and uses faster grouping (precomputed indices) while keeping deterministic behavior. All file paths and the ensemble weights remain unchanged; only the slow parts are swapped for provably equivalent implementations.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

paths = {
    "sub_0": "../input/gb-vpp-pulp-fiction/median_submission.csv",
    "sub_1": "../input/new-ensemble-of-public-notebooks/submission.csv",
    "sub_2": "../input/gaps-features-tf-lstm-resnet-like-ff/sub.csv",
    "sub_3": "../input/vent-011-median-wins/submission.csv",
}

loaded = {}
missing = []
for k, p in paths.items():
    if os.path.exists(p):
        loaded[k] = pd.read_csv(p, usecols=["id", "pressure"])
    else:
        missing.append(k)

if missing:
    from sklearn.neighbors import NearestNeighbors

    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    train = pd.read_csv(
        train_path,
        usecols=["R", "C", "time_step", "u_in", "u_out", "pressure"],
        dtype={
            "R": "int16",
            "C": "int16",
            "u_out": "int8",
            "time_step": "float32",
            "u_in": "float32",
            "pressure": "float32",
        },
    )
    test = pd.read_csv(
        test_path,
        usecols=["id", "R", "C", "time_step", "u_in", "u_out"],
        dtype={
            "id": "int32",
            "R": "int16",
            "C": "int16",
            "u_out": "int8",
            "time_step": "float32",
            "u_in": "float32",
        },
    )

    tr_R = train["R"].to_numpy()
    tr_C = train["C"].to_numpy()
    tr_uout = train["u_out"].to_numpy()
    tr_time = train["time_step"].to_numpy()
    tr_uin = train["u_in"].to_numpy()
    tr_press = train["pressure"].to_numpy()

    te_id = test["id"].to_numpy()
    te_R = test["R"].to_numpy()
    te_C = test["C"].to_numpy()
    te_uout = test["u_out"].to_numpy()
    te_time = test["time_step"].to_numpy()
    te_uin = test["u_in"].to_numpy()

    train_ins_mask = tr_uout == 0
    test_ins_mask = te_uout == 0

    feat_cols = ["time_step", "u_in", "u_out"]  # kept as-is for semantic consistency.

    test_ins_idx_by_RC = {}
    if test_ins_mask.any():
        ins_pos = np.nonzero(test_ins_mask)[0]
        rc_ins = (te_R[ins_pos].astype(np.int32) << 16) | te_C[ins_pos].astype(np.int32)
        order = np.argsort(rc_ins, kind="mergesort")  # stable/deterministic
        rc_sorted = rc_ins[order]
        pos_sorted = ins_pos[order]
        uniq, start = np.unique(rc_sorted, return_index=True)
        end = np.empty_like(start)
        end[:-1] = start[1:]
        end[-1] = rc_sorted.size
        for u, s, e in zip(uniq.tolist(), start.tolist(), end.tolist()):
            Rv = int((u >> 16) & 0xFFFF)
            Cv = int(u & 0xFFFF)
            test_ins_idx_by_RC[(Rv, Cv)] = pos_sorted[s:e].astype(np.int64, copy=False)

    def knn_predict_group(
        Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int = 25
    ) -> np.ndarray:
        if Xte.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        if Xtr.shape[0] == 0:
            return np.zeros((Xte.shape[0],), dtype=np.float32)
        if Xtr.shape[0] < k:
            return np.full((Xte.shape[0],), float(np.mean(ytr)), dtype=np.float32)

        mu = Xtr.mean(axis=0, keepdims=True)
        sig = Xtr.std(axis=0, keepdims=True)
        sig = np.where(sig < 1e-6, 1.0, sig)
        Xtrn = (Xtr - mu) / sig
        Xten = (Xte - mu) / sig

        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        nn.fit(Xtrn)

        out = np.empty((Xten.shape[0],), dtype=np.float32)
        chunk = 4096  # NOTE (speed): larger chunk reduces sklearn call overhead; output is identical.
        for i in range(0, Xten.shape[0], chunk):
            Xt = Xten[i : i + chunk]
            dist, idx = nn.kneighbors(Xt, return_distance=True)
            d2_k = dist * dist
            y_k = ytr[idx]
            w = 1.0 / (d2_k + 1e-6)
            pred = (w * y_k).sum(axis=1) / w.sum(axis=1)
            out[i : i + chunk] = pred.astype(np.float32, copy=False)
        return out

    pred_full = np.zeros(len(test), dtype=np.float32)

    unique_R = np.unique(tr_R[train_ins_mask])
    unique_C = np.unique(tr_C[train_ins_mask])

    tr_uout_f = tr_uout.astype(np.float32, copy=False)
    te_uout_f = te_uout.astype(np.float32, copy=False)

    for Rv in unique_R.tolist():
        for Cv in unique_C.tolist():
            te_pos = test_ins_idx_by_RC.get((int(Rv), int(Cv)))
            if te_pos is None or te_pos.size == 0:
                continue

            tr_mask = train_ins_mask & (tr_R == Rv) & (tr_C == Cv)
            if not tr_mask.any():
                continue

            Xtr = np.stack(
                (tr_time[tr_mask], tr_uin[tr_mask], tr_uout_f[tr_mask]), axis=1
            ).astype(np.float32, copy=False)
            ytr = tr_press[tr_mask].astype(np.float32, copy=False)
            Xte = np.stack(
                (te_time[te_pos], te_uin[te_pos], te_uout_f[te_pos]), axis=1
            ).astype(np.float32, copy=False)

            pred_full[te_pos] = knn_predict_group(Xtr, ytr, Xte, k=25)

    pmin = float(tr_press.min())
    pmax = float(tr_press.max())
    pred_full = np.clip(pred_full, pmin, pmax)

    baseline_df = pd.DataFrame({"id": te_id, "pressure": pred_full})
    for k in missing:
        loaded[k] = baseline_df.copy()

sub_0 = loaded["sub_0"]
sub_1 = loaded["sub_1"]
sub_2 = loaded["sub_2"]
sub_3 = loaded["sub_3"]

target_ids = sub["id"].to_numpy()


def align_to_sample(df: pd.DataFrame) -> np.ndarray:
    if "id" in df.columns and "pressure" in df.columns:
        ids = df["id"].to_numpy()
        pres = df["pressure"].to_numpy()
        order = np.argsort(ids, kind="mergesort")
        ids_sorted = ids[order]
        pres_sorted = pres[order]
        pos = np.searchsorted(ids_sorted, target_ids)
        ok = (pos < ids_sorted.size) & (ids_sorted[pos] == target_ids)
        out = np.full(target_ids.shape[0], np.nan, dtype=pres_sorted.dtype)
        out[ok] = pres_sorted[pos[ok]]
        return out
    return df["pressure"].to_numpy()


p0 = align_to_sample(sub_0)
p1 = align_to_sample(sub_1)
p2 = align_to_sample(sub_2)
p3 = align_to_sample(sub_3)

p0 = np.nan_to_num(p0, nan=0.0)
p1 = np.nan_to_num(p1, nan=0.0)
p2 = np.nan_to_num(p2, nan=0.0)
p3 = np.nan_to_num(p3, nan=0.0)



## === cell 1
sub["pressure"] = (p0 * 0.16) + (p1 * 0.22) + (p2 * 0.10) + (p3 * 0.52)

test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    usecols=["id", "u_out"],
    dtype={"id": "int32", "u_out": "int8"},
)
exp_mask = test["u_out"].to_numpy() == 1
sub.loc[exp_mask, "pressure"] = 0.0

sub[["id", "pressure"]].to_csv("submission.csv", index=False)
sub.head(5)

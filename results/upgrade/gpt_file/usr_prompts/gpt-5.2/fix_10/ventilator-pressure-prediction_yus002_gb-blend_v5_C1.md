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

3.9

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

0.4074916150958257

# 6. Current score

0.68438

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 13.43268) has done: 'Your script currently can’t yield a score because it expects a folder of prior submission files (`../input/gb-blending`) that isn’t present in your listed data paths, so it never produces a valid submission for this environment. I keep your “blend predictions from CSVs” core logic, but add a safe fallback: if no blend files are found, create a simple baseline prediction from `test.csv` that is valid for submission (and non-trivial vs all-zeros), so you can actually submit and get a score. I also fix the `id` handling by building the submission directly from `test.csv` (instead of `sample_submission.csv`, which can risk misalignment) and ensure the output filename is a standard `submission.csv`. These are minimal changes focused on unblocking a valid submission and improving from “not yielded” toward your target.'
- What this solution (achieved 1.42386) has done: 'Your current fallback baseline is extremely weak for this competition (it ignores breath dynamics, R/C effects, and time-step history), which explains the very large MAE (13.43) versus your target (0.407). To move the score sharply toward the target while keeping changes minimal and preserving the “simple baseline fallback” core approach (no new model/training loop), I replace the linear `pressure ≈ a*u_in` with a fast nearest-neighbor lookup on engineered breath-level time-series features built from train inspiratory steps only. This stays within pandas/numpy/sklearn only, runs within the time limit by aggregating per-breath (not per-row) for matching, and then maps matched train-breath pressure trajectories onto each test breath. Submission writing, paths, and your blend logic are kept intact; only the fallback prediction is strengthened.'
- What this solution (achieved 0.77568) has done: 'Your current fallback “nearest-neighbor breath retrieval” still loses a lot of accuracy because it only matches on coarse aggregated statistics, so it often picks a wrong training breath even when the full inspiratory control sequence is very different. Keeping the same core logic (retrieve a whole inspiratory pressure trajectory from the most similar training breath, per (R,C)), I change the distance computation to use the full inspiratory `u_in` time-series (downsampled to a small fixed number of points) plus the existing summary features, which is a minimal but much stronger similarity signal for this competition. I also make the mapping from retrieved inspiratory pressure sequence to each test breath robust by interpolating to the test inspiratory length instead of truncation/padding, reducing systematic alignment error. These changes should reduce MAE substantially (lower-is-better), moving your score closer to the 0.407 target while staying within the same non-training, lookup-based approach and writing a valid `submission.csv`.'
- What this solution (achieved 0.67466) has done: 'Your current score (0.77568, lower-is-better) is still far above the target (0.40749), so we should improve the fallback retrieval baseline while preserving the same “match a train breath within (R,C) and copy/interpolate its inspiratory pressure trajectory” core logic. The smallest high-impact change is to make the nearest-neighbor distance better reflect the evaluation: (1) include both `u_in` and `u_out` time-series signatures (not just `u_in`), and (2) weight time-series signature dimensions more strongly than coarse summary stats so the retrieved breath is more often dynamically similar. To reduce systematic errors from pressure quantization in this competition, we also snap predictions to the discrete pressure grid observed in train (a standard post-processing that keeps the same semantics and usually reduces MAE). All paths and submission writing remain the same, and if blend files exist the original blending behavior is untouched.'
- What this solution (achieved 0.6741) has done: 'Your current approach is a non-training “retrieve nearest train breath (within R,C) and copy its inspiratory pressure trajectory” baseline, and the score gap to target is still large (0.67466 vs 0.40749, lower is better), so we should strengthen the retrieval similarity signal while keeping the same core logic. The most direct minimal upgrade is to compute the nearest neighbor using a higher-fidelity time-series signature that focuses on the inspiratory phase (the only scored region) and uses cumulative control history (cumsum of `u_in`) plus valve timing (`u_out`) to better match dynamics. We keep your interpolation, per-(R,C) grouping, and pressure-grid snapping, but adjust signature construction and weighting so the selected neighbor is more often truly similar where MAE is computed. All paths and the blend-or-fallback behavior remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.6782) has done: 'To move your MAE down toward the 0.4075 target (lower-is-better) while keeping the same “nearest-breath retrieval within (R,C) + copy/interpolate inspiratory pressure trajectory” core logic, I make the distance metric better match what’s scored: compare inspiratory control sequences in a time-aligned way using both `u_in` and its time-integral (`∫u_in dt`) rather than `cumsum(u_in)` alone. I also add a tiny extra summary feature (`u_in` end-of-inspiration slope) and slightly rebalance the existing weights so the retrieval prioritizes the inspiratory waveform (which drives pressure) over coarse stats. The blending path/behavior is unchanged; if `../input/gb-blending` is absent, the improved fallback baseline is used and still writes a valid `submission.csv`. These are minimal, local changes intended to reduce the retrieval mismatch errors that are dominating your current 0.6741 score.'
- What this solution (achieved 0.68468) has done: 'Your current MAE (0.6782, lower-is-better) is still far above the target (0.4075), so we should improve the nearest-breath retrieval while keeping the exact same core approach: within each (R,C), pick the closest train breath by a hand-crafted signature and copy/interpolate its inspiratory pressure trajectory. The smallest likely high-impact change is to (1) build the time-series signatures on a *common time grid* (0..t_end) instead of index-position, and (2) include an inspiratory “valve-close timing” feature (fraction of u_out==1) to better match phase behavior; both reduce mismatches that dominate MAE without changing the overall algorithm. I also make the distance computation scale-stable by robustly normalizing the non-time-series summary features within each (R,C) group (still L1 distance, still same retrieval). Blending logic, file paths, and submission writing remain unchanged, and the script still produces `submission.csv` end-to-end.'
- What this solution (achieved 0.68468) has done: 'I keep your existing “blend if files exist, otherwise fallback to nearest-breath retrieval within (R,C)” core logic, but fix two issues that are currently hurting MAE. First, the fallback currently computes robust normalization (median/IQR) *inside the per-test-breath loop*, which is both slow and (more importantly) can introduce subtle inconsistency; I compute and store those stats once per (R,C) group and reuse them. Second, because the metric ignores expiratory phase, I explicitly set predictions for `u_out==1` to 0 (or any constant) so we don’t carry over unintended copied values and risk extra error; inspiratory predictions remain unchanged. These minimal, local changes should move your score down (better, lower-is-better) toward the 0.407 target while preserving your approach and producing `submission.csv`.'
- What this solution (achieved 0.68438) has done: 'We keep your blending-or-fallback structure exactly as-is, but tighten the fallback retrieval to better match the competition scoring: the MAE is computed only on inspiratory timesteps (`u_out==0`), so the nearest-neighbor distance should be driven primarily by inspiratory control history. Concretely, we (1) build the `u_out` signature from the full breath (so it can actually capture valve-open timing), (2) switch the time-integral signature to a correct cumulative trapezoid integral (better physical alignment), and (3) slightly rebalance the feature weights toward the time-series signatures and away from coarse summary stats to reduce mismatched neighbor selection. These are local changes inside the same “retrieve closest train breath within (R,C) and copy/interpolate inspiratory pressure trajectory” logic, and the script still writes a valid `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd




## === cell 1
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Keep original blending logic: read each candidate submission and compute a weighted blend.
    """
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = l[1] / l_sum
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def _baseline_from_test(test_path, train_path):
    """
    Keep core logic identical: for each test breath, retrieve the closest train breath (within same R,C)
    and copy its inspiratory pressure trajectory (with interpolation to match inspiratory length).

    Changes aimed to reduce MAE toward target (lower-is-better), without changing the algorithm:
    - Build the u_out signature on the full breath (train/test), not only inspiratory rows, so it actually encodes
      "valve-open timing" differences that impact dynamics. This strengthens retrieval without changing the approach.
    - Use a cumulative trapezoid integral for the u_in time-integral signature (better physical alignment than u*dt cumsum).
    - Slightly rebalance weights to emphasize the time-series signatures that dominate the scored region.
    - Keep expiratory predictions forced to 0.0 (not scored) to avoid accidental errors outside inspiratory phase.
    """
    use_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    use_test = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "id"]
    train = pd.read_csv(train_path, usecols=use_train)
    test = pd.read_csv(test_path, usecols=use_test)

    pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)

    tr_insp = train[train["u_out"] == 0].copy()
    te_insp = test[test["u_out"] == 0].copy()

    tr_full = train[["breath_id", "time_step", "u_in", "u_out"]].copy()
    te_full = test[["breath_id", "time_step", "u_in", "u_out"]].copy()

    def make_breath_features(df_insp, prefix):
        g = df_insp.groupby("breath_id", sort=False)

        u_mean = g["u_in"].mean()
        u_max = g["u_in"].max()
        u_std = g["u_in"].std(ddof=0).fillna(0.0)
        u_last = g["u_in"].last()
        u_sum = g["u_in"].sum()

        df_tmp = df_insp[["breath_id", "time_step", "u_in"]].copy()
        df_tmp["dt"] = (
            df_tmp.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
        )
        auc = (
            (df_tmp["u_in"] * df_tmp["dt"])
            .groupby(df_tmp["breath_id"], sort=False)
            .sum()
        )

        def end_slope(s):
            v = s.to_numpy(dtype=np.float32)
            if v.size < 2:
                return np.float32(0.0)
            return np.float32(v[-1] - v[-2])

        u_slope_end = g["u_in"].apply(end_slope)

        uout_mean = g["u_out"].mean()

        feats = pd.DataFrame(
            {
                "breath_id": u_mean.index.values,
                f"{prefix}u_mean": u_mean.values.astype(np.float32),
                f"{prefix}u_max": u_max.values.astype(np.float32),
                f"{prefix}u_std": u_std.values.astype(np.float32),
                f"{prefix}u_last": u_last.values.astype(np.float32),
                f"{prefix}u_sum": u_sum.values.astype(np.float32),
                f"{prefix}u_auc": auc.reindex(u_mean.index).values.astype(np.float32),
                f"{prefix}u_slope_end": u_slope_end.reindex(u_mean.index).values.astype(
                    np.float32
                ),
                f"{prefix}uout_mean": uout_mean.reindex(u_mean.index).values.astype(
                    np.float32
                ),
            }
        )
        return feats

    def make_time_signature(df_phase, col, n_sig=32):
        df_sorted = df_phase.sort_values(["breath_id", "time_step"])
        sig = {}
        for bid, grp in df_sorted.groupby("breath_id", sort=False):
            t = grp["time_step"].to_numpy(dtype=np.float32)
            v = grp[col].to_numpy(dtype=np.float32)
            m = len(v)
            if m == 0:
                sig[int(bid)] = np.zeros(n_sig, dtype=np.float32)
                continue
            if m == 1:
                sig[int(bid)] = np.full(n_sig, float(v[0]), dtype=np.float32)
                continue
            t_end = float(t[-1])
            if t_end <= 0.0:
                sig[int(bid)] = np.full(n_sig, float(v[-1]), dtype=np.float32)
                continue
            t_new = np.linspace(0.0, t_end, n_sig, dtype=np.float32)
            sig[int(bid)] = np.interp(t_new, t, v).astype(np.float32)
        return sig

    def make_time_integral_signature(df_phase, n_sig=32):
        df_sorted = df_phase.sort_values(["breath_id", "time_step"])[
            ["breath_id", "time_step", "u_in"]
        ]
        sig = {}
        for bid, grp in df_sorted.groupby("breath_id", sort=False):
            t = grp["time_step"].to_numpy(dtype=np.float32)
            u = grp["u_in"].to_numpy(dtype=np.float32)
            m = len(u)
            if m < 2:
                sig[int(bid)] = np.zeros(n_sig, dtype=np.float32)
                continue

            dt = np.diff(t).astype(np.float32)
            inc = 0.5 * (u[1:] + u[:-1]) * dt
            integ = np.concatenate([[0.0], np.cumsum(inc, dtype=np.float32)]).astype(
                np.float32
            )

            denom = float(integ[-1]) if float(integ[-1]) != 0.0 else 1.0
            integ = (integ / denom).astype(np.float32)

            t_end = float(t[-1])
            if t_end <= 0.0:
                sig[int(bid)] = np.full(n_sig, float(integ[-1]), dtype=np.float32)
                continue
            t_new = np.linspace(0.0, t_end, n_sig, dtype=np.float32)
            sig[int(bid)] = np.interp(t_new, t, integ).astype(np.float32)
        return sig

    tr_feats = make_breath_features(tr_insp, prefix="")
    te_feats = make_breath_features(te_insp, prefix="")

    tr_rc = train.groupby("breath_id", sort=False)[["R", "C"]].first().reset_index()
    te_rc = test.groupby("breath_id", sort=False)[["R", "C"]].first().reset_index()
    tr_feats = tr_feats.merge(tr_rc, on="breath_id", how="left")
    te_feats = te_feats.merge(te_rc, on="breath_id", how="left")

    tr_uin_sig = make_time_signature(tr_insp, col="u_in", n_sig=32)
    te_uin_sig = make_time_signature(te_insp, col="u_in", n_sig=32)

    tr_uin_isig = make_time_integral_signature(tr_insp, n_sig=32)
    te_uin_isig = make_time_integral_signature(te_insp, n_sig=32)

    tr_uout_sig = make_time_signature(tr_full, col="u_out", n_sig=16)
    te_uout_sig = make_time_signature(te_full, col="u_out", n_sig=16)

    tr_pressure_seq = (
        tr_insp.sort_values(["breath_id", "time_step"])
        .groupby("breath_id", sort=False)["pressure"]
        .apply(lambda s: s.values.astype(np.float32))
    )

    feat_cols = [
        "u_mean",
        "u_max",
        "u_std",
        "u_last",
        "u_sum",
        "u_auc",
        "u_slope_end",
        "uout_mean",
    ]

    pred = np.zeros(len(test), dtype=np.float32)

    tr_rc_norm = {}
    tr_feats_by_rc = {}
    for (r, c), subdf in tr_feats.groupby(["R", "C"], sort=False):
        bids = subdf["breath_id"].to_numpy(dtype=np.int64)
        Xs_raw = subdf[feat_cols].to_numpy(dtype=np.float32)

        med = np.median(Xs_raw, axis=0).astype(np.float32)
        q75 = np.percentile(Xs_raw, 75, axis=0).astype(np.float32)
        q25 = np.percentile(Xs_raw, 25, axis=0).astype(np.float32)
        iqr = (q75 - q25).astype(np.float32)
        iqr[iqr == 0.0] = 1.0
        tr_rc_norm[(int(r), int(c))] = (med, iqr)

        Xs = ((Xs_raw - med) / iqr).astype(np.float32)

        Uin = np.vstack(
            [tr_uin_sig.get(int(b), np.zeros(32, dtype=np.float32)) for b in bids]
        ).astype(np.float32)
        UinI = np.vstack(
            [tr_uin_isig.get(int(b), np.zeros(32, dtype=np.float32)) for b in bids]
        ).astype(np.float32)
        Uout = np.vstack(
            [tr_uout_sig.get(int(b), np.zeros(16, dtype=np.float32)) for b in bids]
        ).astype(np.float32)

        Xfull = np.hstack([0.08 * Xs, 1.55 * Uin, 1.55 * UinI, 0.25 * Uout]).astype(
            np.float32
        )
        tr_feats_by_rc[(int(r), int(c))] = (bids, Xfull)

    te_breath_ids = te_feats["breath_id"].to_numpy(dtype=np.int64)
    te_rc_vals = te_feats[["R", "C"]].to_numpy(dtype=np.int64)
    te_Xs_raw = te_feats[feat_cols].to_numpy(dtype=np.float32)

    te_Uin = np.vstack(
        [te_uin_sig.get(int(b), np.zeros(32, dtype=np.float32)) for b in te_breath_ids]
    ).astype(np.float32)
    te_UinI = np.vstack(
        [te_uin_isig.get(int(b), np.zeros(32, dtype=np.float32)) for b in te_breath_ids]
    ).astype(np.float32)
    te_Uout = np.vstack(
        [te_uout_sig.get(int(b), np.zeros(16, dtype=np.float32)) for b in te_breath_ids]
    ).astype(np.float32)

    test_sorted = test.sort_values(["breath_id", "time_step"]).reset_index()
    row_breath = test_sorted["breath_id"].to_numpy(dtype=np.int64)
    row_u_out = test_sorted["u_out"].to_numpy(dtype=np.int64)

    change = np.r_[True, row_breath[1:] != row_breath[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(test_sorted)]
    te_seg = {
        int(b): (int(s), int(e)) for b, s, e in zip(row_breath[starts], starts, ends)
    }

    def interp_seq(p, n_target):
        p = np.asarray(p, dtype=np.float32)
        m = len(p)
        if n_target <= 0:
            return np.zeros(0, dtype=np.float32)
        if m == 0:
            return np.zeros(n_target, dtype=np.float32)
        if m == 1:
            return np.full(n_target, float(p[0]), dtype=np.float32)
        if m == n_target:
            return p.astype(np.float32)
        x_old = np.linspace(0.0, 1.0, m, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, n_target, dtype=np.float32)
        return np.interp(x_new, x_old, p).astype(np.float32)

    for i in range(len(te_breath_ids)):
        b = int(te_breath_ids[i])
        r, c = int(te_rc_vals[i, 0]), int(te_rc_vals[i, 1])
        key = (r, c)
        if key not in tr_feats_by_rc:
            continue

        tr_bids, tr_X = tr_feats_by_rc[key]

        med, iqr = tr_rc_norm[key]
        Xs_t = ((te_Xs_raw[i] - med) / iqr).astype(np.float32)

        x = np.hstack(
            [0.08 * Xs_t, 1.55 * te_Uin[i], 1.55 * te_UinI[i], 0.25 * te_Uout[i]]
        ).astype(np.float32)

        d = np.abs(tr_X - x).sum(axis=1)
        nn = int(tr_bids[int(d.argmin())])

        p_seq = tr_pressure_seq.get(nn, None)
        if p_seq is None:
            continue

        s, e = te_seg.get(b, (None, None))
        if s is None:
            continue

        idx = test_sorted.loc[s : e - 1, "index"].to_numpy(dtype=np.int64)
        uo = row_u_out[s:e]
        insp_mask = uo == 0
        n_insp = int(insp_mask.sum())
        if n_insp == 0:
            continue

        p_use = interp_seq(p_seq, n_insp)
        pred[idx[insp_mask]] = p_use

    pred = np.clip(pred.astype(np.float32), 0.0, 80.0)

    if pressure_grid.size > 0:
        pos = np.searchsorted(pressure_grid, pred, side="left")
        pos0 = np.clip(pos - 1, 0, pressure_grid.size - 1)
        pos1 = np.clip(pos, 0, pressure_grid.size - 1)
        g0 = pressure_grid[pos0]
        g1 = pressure_grid[pos1]
        choose1 = np.abs(g1 - pred) < np.abs(g0 - pred)
        pred = np.where(choose1, g1, g0).astype(np.float32)

    pred = pred.astype(np.float32)
    pred[test["u_out"].to_numpy(dtype=np.int64) == 1] = np.float32(0.0)

    sub = pd.DataFrame(
        {"id": test["id"].astype(np.int64), "pressure": pred.astype(np.float32)}
    )
    sub = sub.sort_values("id").reset_index(drop=True)
    return sub


def g(dp):
    """
    Original intent: blend many existing submissions from a dataset folder.
    Minimal fix: if no files are found in dp, create a baseline submission from the competition data.
    """
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    if len(l) == 0:
        test_path = (
            "/kaggle/data/test.csv"
            if os.path.exists("/kaggle/data/test.csv")
            else "/kaggle/input/test.csv"
        )
        train_path = (
            "/kaggle/data/train.csv"
            if os.path.exists("/kaggle/data/train.csv")
            else "/kaggle/input/train.csv"
        )
        sub = _baseline_from_test(test_path=test_path, train_path=train_path)
        sub.to_csv("submission.csv", index=False)
        return

    file_count = len(l)
    loop_time = file_count
    splits = file_count // 2 if file_count >= 2 else 1
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

    test_path = (
        "/kaggle/data/test.csv"
        if os.path.exists("/kaggle/data/test.csv")
        else "/kaggle/input/test.csv"
    )
    test = pd.read_csv(test_path, usecols=["id"])
    output = pd.DataFrame(
        {
            "id": test["id"].astype(np.int64),
            "pressure": np.zeros(len(test), dtype=np.float64),
        }
    )

    for i in range(loop_time):
        weight = []
        set_seed(i)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        for j in range(len(flist)):
            output["pressure"] += flist[j] * weight[j]

    output["pressure"] /= loop_time
    output.to_csv("submission.csv", index=False)




## === cell 2
g("../input/gb-blending")

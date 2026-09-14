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

0.159580356454496

# 6. Current score

8.21476

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.5439) has done: 'The current notebook fails because it tries to read four external submission files from `../input/...` datasets that are not present in your environment, so `sub_1`..`sub_4` never get defined and the blend crashes. To keep the “blend submissions” core logic while making it run end-to-end, I add a small fallback: if those external files are missing, train a lightweight in-notebook baseline model (scikit-learn Ridge) on the provided `train.csv` and generate predictions for `test.csv`. Then the script always write a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 7.89004) has done: 'Your current fallback model predicts pressure from per-row features only, but the competition metric is sequence-based and only scores inspiratory phase, so the Ridge baseline lands far from the target. To move the MAE down substantially with minimal changes and without altering the overall “train fallback → make 4 subs → weighted blend” structure, I keep the same approach but enrich the fallback features with simple lag/cumulative features computed within each `breath_id` (no architecture/training loop changes). I also clip predictions to the known training pressure range to reduce extreme errors. These are small, safe changes that should reduce the error toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 7.96719) has done: 'Your current fallback Ridge model is far from the target because it ignores the competition’s key sequence behavior and the fact that only inspiratory steps (u_out=0) are scored. I keep the same “fallback model → create 4 subs → weighted blend” structure, but make minimal feature additions that capture more breath dynamics (extra lags/rolling stats and simple interaction terms) while staying in the same Ridge + preprocessing pipeline. I also bias predictions toward a safe value during the unscored expiratory phase (u_out=1) to reduce model noise without affecting the scored portion, and ensure prediction alignment with `id`. These changes should reduce MAE substantially versus the current 7.89, moving closer toward your 0.159 target without changing the overall approach.'
- What this solution (achieved 7.57753) has done: 'Your current fallback Ridge model is still far from the target because it treats the problem as purely per-row regression; with minimal change we can keep the same Ridge pipeline but add two competition-specific, lightweight post-processing steps that often drastically reduce MAE: (1) snap predictions to the discrete set of pressure values observed in training (pressure is quantized), and (2) apply an inspiratory-only residual correction using the mean error per (R,C,time_step) bucket learned from train (no new model/loop, just a correction table). These changes preserve your overall flow (fallback model → create 4 subs → weighted blend), keep runtime reasonable, and should move the score substantially downward toward the target band. I also keep your existing expiratory-phase handling and clipping, but apply snapping last so outputs remain valid training-like pressures.'
- What this solution (achieved 7.79069) has done: 'Your current score is far worse than the target (lower is better), so we should make small changes that legitimately reduce MAE without changing the overall “fallback model → make 4 subs → weighted blend” core flow. The biggest win with minimal risk is to align training and correction to the metric by fitting the Ridge model only on inspiratory rows (`u_out==0`) and applying the residual correction using a more reliable, slightly coarser time bin (reduces noise/leak from rounding). I also build the residual correction with fast vectorized joins instead of Python loops (same semantics, fewer mistakes), and keep your clipping + snapping-to-quantized-pressure as the final post-process. These changes preserve your model family and pipeline, but should move the MAE substantially downward toward your target band.'
- What this solution (achieved 7.79054) has done: 'Your current MAE is much worse than the target (lower is better), so we should make the smallest changes that materially reduce error while keeping your core “fallback Ridge → make 4 subs → weighted blend” structure unchanged. The biggest issue is that the fallback is effectively using the same prediction four times, so the blend adds no value; we can instead create 4 slightly different (but still Ridge, same features, same training approach) fallback models and blend them to reduce variance and improve MAE. Also, your residual correction currently mismatches rows because it pairs `train_insp2` (already filtered) with predictions taken from the full inspiratory mask over the full training frame; fixing that alignment is a correctness bug that can significantly improve the score without changing the modeling approach. Finally, we keep your clipping + snapping to the quantized pressure grid, but apply snapping after blending so the final output matches the discrete target distribution.'
- What this solution (achieved 7.79054) has done: 'Your score (7.79 MAE, lower is better) is far from the target (0.159), so we need a legitimate but minimal correction that actually matches the competition’s time-series nature without changing your overall “fallback Ridge → make 4 subs → weighted blend” structure. The largest correctness/performance bug is that you compute a strong `preds_blend` (with residual correction + snapping) but then ignore it in the final submission and instead re-blend the raw per-model predictions in cell 3; I fix cell 3 to directly use the already-built `preds_blend` when the fallback path is active. I also ensure each `sub_k` created under fallback uses the same expiratory-phase handling and clipping/snap logic to avoid injecting noisy u_out=1 values back into the blend. These are small, targeted changes that should substantially reduce MAE (move toward the target) while keeping architecture/training approach unchanged and still producing a valid `submission.csv`.'
- What this solution (achieved 8.17313) has done: 'Your current MAE (7.79, lower is better) is extremely far from the target (0.159), so the main issue is not blending but that the fallback predictor is fundamentally misaligned with how pressure evolves over a breath. To keep your core logic (fallback Ridge → 4 subs → weighted blend → clipping → snap-to-grid) intact while moving the score down substantially, I add a single, competition-standard “within-breath cumulative integral” feature (`u_in_cumarea`) and a physically-motivated interaction with `1/C` plus a couple tiny stabilizing terms; these are still just additional engineered columns feeding the same Ridge pipeline. I also change the residual-correction time binning to use exact `time_step` values (they are already on a fixed grid in this dataset), which avoids noisy rounding artifacts and typically improves the correction table quality without changing the approach. Finally, I ensure the expiratory-phase fill uses the (R,C) inspiratory mean on the snapped pressure grid, keeping u_out=1 harmless while not affecting scored rows.'
- What this solution (achieved 8.17313) has done: 'Your MAE is far above the target (lower is better), so the smallest legitimate step toward the target is to make the fallback predictor respect the competition’s *sequence* structure without changing the model family or training loop. I keep your Ridge + engineered-features + residual-correction + snapping/blending pipeline, but fix two high-impact correctness issues: (1) your `id` alignment is currently wrong because you read `test.csv` without `id` and then try to align to it; (2) your residual correction key currently uses mismatched arrays because `ts_bin_train` was built from `train_insp` but assigned to a shorter `train_insp2`. I also ensure the residual correction uses `time_step` itself consistently (as a stable key) and then align predictions to `sub` purely by `id`, which should materially reduce error while preserving your core logic.'
- What this solution (achieved 8.21476) has done: 'Your MAE (8.17, lower is better) is far from the target (0.159), so we need a legitimate improvement while keeping your Ridge+features+residual-correction+snap/blend core intact. The biggest missing piece is that the model is asked to learn absolute pressure without knowing the breath’s initial baseline, which varies strongly by (R,C); we can add a simple per-breath “start pressure prior” feature computed from train and mapped to both train/test, then let Ridge learn deltas around it. Next, we make the residual correction table more reliable by keying it on `time_step` plus a small within-breath step index (0..79) to avoid float-key edge cases and improve alignment. These are minimal feature/key additions (no architecture/training loop change) and should materially reduce error toward the target while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

paths = {
    "sub_1": "../input/improvement-base-on-tensor-bidirect-lstm-0-173/submission.csv",
    "sub_2": "../input/finetune-of-tensorflow-bidirectional-lstm/submission.csv",
    "sub_3": "../input/lightautoml-bidirectional-lstm/submission.csv",
    "sub_4": "../input/tensorflow-bidirectional-lstm-custom-mae-loss/submission.csv",
}

loaded = {}
missing = []
for k, p in paths.items():
    if os.path.exists(p):
        loaded[k] = pd.read_csv(p)
    else:
        missing.append(p)

use_fallback_model = len(missing) > 0



## === cell 2
if use_fallback_model:

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge

    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    )
    test = pd.read_csv(
        test_path,
        usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "id"],
    )

    train_sorted_for_p0 = train.sort_values(["breath_id", "time_step"]).copy()
    p0_by_breath = train_sorted_for_p0.groupby("breath_id", sort=False)[
        "pressure"
    ].first()
    rc_p0_mean = (
        train_sorted_for_p0.assign(
            p0=train_sorted_for_p0["breath_id"].map(p0_by_breath)
        )
        .groupby(["R", "C"], sort=False)["p0"]
        .mean()
        .astype(np.float32)
    )
    global_p0 = float(p0_by_breath.mean())

    def add_breath_features(df: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        df = df.sort_values(["breath_id", "time_step"]).copy()
        g = df.groupby("breath_id", sort=False)

        df["u_in_lag1"] = g["u_in"].shift(1)
        df["u_in_lag2"] = g["u_in"].shift(2)
        df["u_in_lag3"] = g["u_in"].shift(3)
        df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
        df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

        df["u_out_lag1"] = g["u_out"].shift(1)

        df["u_in_cumsum"] = g["u_in"].cumsum()
        df["u_in_cummean"] = df["u_in_cumsum"] / (g.cumcount() + 1)

        df["time_step_lag1"] = g["time_step"].shift(1)
        df["dt"] = df["time_step"] - df["time_step_lag1"]

        dt0 = df["dt"].fillna(0.0).astype(np.float32)
        df["u_in_x_dt"] = (df["u_in"].astype(np.float32) * dt0).astype(np.float32)
        df["u_in_cumarea"] = g["u_in_x_dt"].cumsum().astype(np.float32)

        df["u_in_roll3_mean"] = (
            g["u_in"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df["u_in_roll5_mean"] = (
            g["u_in"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df["u_in_roll5_std"] = (
            g["u_in"].rolling(5, min_periods=1).std().reset_index(level=0, drop=True)
        )

        df["u_in_x_R"] = df["u_in"] * df["R"]
        df["u_in_x_C"] = df["u_in"] * df["C"]

        df["inv_C"] = (1.0 / df["C"].astype(np.float32)).astype(np.float32)
        df["u_in_x_invC"] = (df["u_in"].astype(np.float32) * df["inv_C"]).astype(
            np.float32
        )
        df["cumarea_x_invC"] = (
            df["u_in_cumarea"].astype(np.float32) * df["inv_C"]
        ).astype(np.float32)

        df["u_out_cumsum"] = g["u_out"].cumsum().astype(np.float32)

        df["step"] = g.cumcount().astype(np.int16)

        if is_train:
            df["p0"] = df["breath_id"].map(p0_by_breath).astype(np.float32)
        else:
            rc_index = pd.MultiIndex.from_frame(df[["R", "C"]])
            df["p0"] = rc_index.map(rc_p0_mean).astype("float32")
            df["p0"] = df["p0"].fillna(global_p0).astype(np.float32)

        df["u_in_minus_p0"] = (df["u_in"].astype(np.float32) - df["p0"]).astype(
            np.float32
        )
        df["cumarea_plus_p0"] = (
            df["u_in_cumarea"].astype(np.float32) + df["p0"]
        ).astype(np.float32)

        return df

    train_fe = add_breath_features(train, is_train=True)
    test_fe = add_breath_features(test, is_train=False)

    num_cols = [
        "time_step",
        "u_in",
        "u_out",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_diff1",
        "u_in_diff2",
        "u_out_lag1",
        "u_in_cumsum",
        "u_in_cummean",
        "dt",
        "u_in_x_dt",
        "u_in_cumarea",
        "u_in_roll3_mean",
        "u_in_roll5_mean",
        "u_in_roll5_std",
        "u_in_x_R",
        "u_in_x_C",
        "inv_C",
        "u_in_x_invC",
        "cumarea_x_invC",
        "u_out_cumsum",
        "step",
        "p0",
        "u_in_minus_p0",
        "cumarea_plus_p0",
    ]
    cat_cols = ["R", "C"]

    X_all = train_fe[num_cols + cat_cols]
    y_all = train_fe["pressure"].astype(np.float32)
    X_test = test_fe[num_cols + cat_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    insp_mask_train = train_fe["u_out"].values == 0
    X_train = X_all.loc[insp_mask_train].reset_index(drop=True)
    y_train = y_all.loc[insp_mask_train].reset_index(drop=True)

    ridge_alphas = [0.3, 1.0, 3.0, 10.0]
    preds_list = []

    train_insp = train_fe.loc[insp_mask_train].copy()
    uniq_p = np.sort(train["pressure"].unique()).astype(np.float32)

    def snap_to_nearest(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(grid, values, side="left")
        idx = np.clip(idx, 0, len(grid) - 1)
        left = grid[np.clip(idx - 1, 0, len(grid) - 1)]
        right = grid[idx]
        choose_right = np.abs(values - right) <= np.abs(values - left)
        return np.where(choose_right, right, left).astype(np.float32)

    rc_mean = train_insp.groupby(["R", "C"])["pressure"].mean()
    global_insp_mean = float(train_insp["pressure"].mean())

    test_rc_df = test_fe[["R", "C"]].copy()
    test_rc_df["rc_mean"] = (
        test_rc_df.set_index(["R", "C"]).index.map(rc_mean).astype("float32")
    )
    test_rc_df["rc_mean"] = (
        test_rc_df["rc_mean"].fillna(global_insp_mean).astype(np.float32)
    )
    fallback_means = snap_to_nearest(
        test_rc_df["rc_mean"].values.astype(np.float32), uniq_p
    )

    ts_bin_test = test_fe["time_step"].values.astype(np.float32)
    step_test = test_fe["step"].values.astype(np.int16)
    test_key = pd.MultiIndex.from_arrays(
        [test_fe["R"].values, test_fe["C"].values, step_test, ts_bin_test],
        names=["R", "C", "step", "ts_bin"],
    )

    mask_exp = test_fe["u_out"].values == 1
    mask_insp = ~mask_exp

    for a in ridge_alphas:
        model = Ridge(alpha=a, random_state=42)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test).astype(np.float32)

        X_train_insp_aligned = X_all.loc[insp_mask_train].reset_index(drop=True)
        train_pred_insp = pipe.predict(X_train_insp_aligned).astype(np.float32)

        train_insp2 = train_insp[["R", "C", "time_step", "step", "pressure"]].copy()
        train_insp2["ts_bin"] = train_insp2["time_step"].values.astype(np.float32)
        train_insp2["resid"] = (
            train_insp2["pressure"].values.astype(np.float32) - train_pred_insp
        )

        resid_tbl = (
            train_insp2.groupby(["R", "C", "step", "ts_bin"], sort=False)["resid"]
            .mean()
            .astype(np.float32)
        )

        resid_corr = test_key.map(resid_tbl).astype("float32")
        resid_corr = np.nan_to_num(resid_corr, nan=0.0).astype(np.float32)

        preds_adj = preds.copy()
        preds_adj[mask_insp] = preds_adj[mask_insp] + resid_corr[mask_insp]
        preds_adj[mask_exp] = fallback_means[mask_exp]

        preds_list.append(preds_adj)

    preds_blend = (
        preds_list[0] * 0.1
        + preds_list[1] * 0.5
        + preds_list[2] * 0.26
        + preds_list[3] * 0.14
    ).astype(np.float32)

    pmin = float(train_fe["pressure"].min())
    pmax = float(train_fe["pressure"].max())
    preds_blend = np.clip(preds_blend, pmin, pmax).astype(np.float32)
    preds_blend = snap_to_nearest(preds_blend.astype(np.float32), uniq_p)

    snapped_preds_list = []
    for p in preds_list:
        p2 = np.clip(p.astype(np.float32), pmin, pmax).astype(np.float32)
        p2 = snap_to_nearest(p2, uniq_p)
        snapped_preds_list.append(p2)

    sub_sorted = sub.sort_values("id").reset_index(drop=True)
    test_pred_df = pd.DataFrame({"id": test["id"].values, "pressure": preds_blend})
    test_pred_df = test_pred_df.sort_values("id").reset_index(drop=True)
    preds_aligned_blend = (
        sub_sorted[["id"]]
        .merge(test_pred_df, on="id", how="left")["pressure"]
        .values.astype(np.float32)
    )

    sub_1 = sub_sorted.copy()
    sub_2 = sub_sorted.copy()
    sub_3 = sub_sorted.copy()
    sub_4 = sub_sorted.copy()

    for i, sub_k in enumerate([sub_1, sub_2, sub_3, sub_4]):
        test_pred_k = pd.DataFrame(
            {"id": test["id"].values, "pressure": snapped_preds_list[i]}
        )
        test_pred_k = test_pred_k.sort_values("id").reset_index(drop=True)
        sub_k["pressure"] = (
            sub_sorted[["id"]]
            .merge(test_pred_k, on="id", how="left")["pressure"]
            .values.astype(np.float32)
        )

else:
    sub_1 = loaded["sub_1"]
    sub_2 = loaded["sub_2"]
    sub_3 = loaded["sub_3"]
    sub_4 = loaded["sub_4"]

for df_name in ["sub_1", "sub_2", "sub_3", "sub_4"]:
    df = locals()[df_name]
    if "id" in df.columns:
        locals()[df_name] = df.sort_values("id").reset_index(drop=True)

sub = sub.sort_values("id").reset_index(drop=True)



## === cell 3
if use_fallback_model:
    sub["pressure"] = preds_aligned_blend.astype(np.float32)
else:
    sub["pressure"] = (
        (sub_1["pressure"].values * 0.1)
        + (sub_2["pressure"].values * 0.5)
        + (sub_3["pressure"].values * 0.26)
        + (sub_4["pressure"].values * 0.14)
    )

sub[["id", "pressure"]].to_csv("submission.csv", index=False)
sub.head(5)

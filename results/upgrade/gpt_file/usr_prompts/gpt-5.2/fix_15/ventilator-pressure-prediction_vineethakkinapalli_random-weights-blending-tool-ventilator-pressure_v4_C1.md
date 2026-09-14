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

0.1391530842025668

# 6. Current score

8.61259

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.299) has done: 'Your code is failing because it expects an external dataset folder of “high-score submissions” that isn’t present, leading to empty file lists and an IndexError during blending. I keep your blending core logic intact, but make it robust to missing/empty folders by falling back to a safe baseline prediction built only from the provided competition data. I also fix the cell numbering and ensure a submission CSV with the required `id,pressure` columns is always written to `/kaggle/working/submission.csv`. These changes are execution-unblocking and should yield a valid (though not SOTA) score instead of “Not yielded”.'
- What this solution (achieved 8.28781) has done: 'Your current score (6.299, lower is better) is far worse than the target (~0.139), so we should improve the fallback path that runs when the “high-score submissions” folder is missing. I keep your blending logic intact, but replace the weak baseline (mean by `(R,C,time_step)`) with a stronger, still-simple and fast per-breath simulator-style baseline that uses the known ventilator identity `pressure ≈ u_in / C + PEEP` during inspiration. I estimate `PEEP` robustly from training data per `(R,C)` using low-`u_in` inspiratory points, then apply it on test only when `u_out==0` (scored phase) and use the per-`(R,C)` median pressure when `u_out==1` (not scored) to avoid wild values. Predictions are snapped to the known discrete pressure grid as you already do, and the script always writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 7.24716) has done: 'Your current MAE (8.28781, lower is better) is far worse than the target (~0.139), so we need to improve the fallback baseline path (used because the “high-score submissions” folder is missing) without changing your blending core logic. The main issue is that the baseline ignores the strong per-breath temporal structure; we can significantly reduce error by using only training data to build a fast lookup of the median `pressure` for each `(R, C, time_step, u_in, u_out)` combination, with safe fallbacks to coarser groupings when unseen. This keeps evaluation semantics identical (still predicting `pressure` per row, still snapping to the known discrete pressure grid), stays within Kaggle constraints, and runs quickly. The script still always write a valid `/kaggle/working/submission.csv` with `id,pressure`.'
- What this solution (achieved 3.85526) has done: 'Your current MAE (7.247, lower is better) is still far from the target (~0.139), so we should improve only the fallback predictor (used when the “high-score submissions” folder is missing/empty) while leaving your blending logic intact. The biggest issue is that your lookup keys (`time_step`, `u_in`) are float-based, so exact matching between train/test is extremely sparse; rounding them to fixed grids before grouping/merging massively increases hit-rate and reduces MAE without changing the overall approach. I also add a per-row linear interpolation fallback along the `u_in` axis within each `(R,C,time_step,u_out)` group, which stays “train-only lookup driven” but improves generalization when exact `u_in` isn’t present. The script still always writes a valid `/kaggle/working/submission.csv` with `id,pressure` and keeps snapping to the discrete pressure grid.'
- What this solution (achieved 8.32997) has done: 'Your current score (3.85526 MAE, lower-is-better) is far above the target (~0.139), so the smallest safe way to move toward the target is to strengthen only the fallback predictor (used when the high-score-submissions folder is missing/empty) while leaving your blending logic intact. The main weakness is that the current lookup ignores strong sequential/breath structure and treats each timestep independently; we can improve a lot by adding lag/lead features and using a train-only KNN regressor (distance-weighted) over those features, then still snapping outputs to the known discrete pressure grid (preserving evaluation semantics). This keeps the same overall “train-only baseline when blending inputs are absent” approach, changes no neural architecture/training loops (you have none), and remains fast enough by training on a capped random subset and predicting in chunks. The script still always writes a valid `/kaggle/working/submission.csv` with exactly `id,pressure`.'
- What this solution (achieved 8.51391) has done: 'Your current score (8.32997 MAE, lower-is-better) is far above the target (~0.139), so we should improve only the fallback path (used when the high-score-submissions folder is missing/empty) while keeping your blending logic intact. The weakest link is the KNN baseline: it uses unscaled features and an arbitrary random subsample, which makes distance-based regression perform poorly and unstable. I keep the same KNN regressor approach and the same feature set/lag logic, but (1) standardize features using train statistics (crucial for KNN), (2) sample whole breaths (keeps temporal consistency), and (3) slightly tune `n_neighbors` toward a safer bias/variance point while staying in the same model family. The script still always writes a valid `/kaggle/working/submission.csv` with `id,pressure` and still snaps predictions to the known discrete pressure grid.'
- What this solution (achieved 8.61792) has done: 'I fix the runtime error by ensuring the calibration dataframe uses exactly the same feature columns as the KNN model (it currently duplicates features and produces a 17-vs-14 column mismatch). This is a minimal logic-preserving change: the model, features, training, and calibration approach remain the same, but the matrices are aligned so standardization and prediction work. I also keep the fallback path robust and ensure the submission is always written as `/kaggle/working/submission.csv` with the required `id,pressure` columns. No score-tuning changes beyond this bugfix are introduced; the goal is to get a valid submission produced end-to-end.'
- What this solution (achieved 8.61856) has done: 'I fix the shape-mismatch bug causing the broadcast error by ensuring the calibration dataframe doesn’t duplicate feature columns (it currently appends `R,C,time_step` twice, creating 17 columns instead of the 14 the KNN was trained on). This is execution-unblocking and score-neutral in intent: the model, features, training, residual-calibration logic, and pressure-grid snapping remain the same, but the matrices align correctly for standardization/prediction. I also add a small safety assert to catch any future feature-column drift early and keep the submission-writing logic unchanged so `/kaggle/working/submission.csv` is always produced.'
- What this solution (achieved 8.61565) has done: 'Your current score (8.61856 MAE, lower is better) is far worse than the target (~0.139), so we should improve only the fallback predictor (used because the “high-score submissions” folder is missing) while keeping your overall blending logic and KNN approach intact. The biggest issue is that we are predicting even during `u_out==1` (unscored) and only patching it with a crude `(R,C)` median; instead we can train/predict **only on inspiratory rows (`u_out==0`)** and then fill expiratory rows safely, which directly matches the competition metric without changing the model family. We also fix a subtle bug in `u_in_cum_area` (it was using global `cumsum()` not per-breath), and we make time weighting slightly stronger (still the same KNN, same features) to better respect sequence alignment. These are minimal, metric-aligned changes that should reduce MAE substantially while still writing a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 8.61469) has done: 'Your current MAE (8.61565, lower is better) is far above the target (~0.139), so we should improve only the fallback predictor path (used because the high-score submission folder is missing) while keeping your overall blending logic and KNN approach intact. The biggest issue is a feature bug: `u_in_cum_area` is computed as `cumsum(u_in) * time_step`, but the correct per-step area proxy is `cumsum(u_in * dt)` per breath; fixing this keeps the same feature intent but makes it physically/temporally meaningful and improves neighborhood quality. I also add two minimal, metric-aligned features (`dt` and `u_in_area`) and include them in the same KNN pipeline (no new model family, no new training loop) to better capture flow integration without changing evaluation semantics. Everything still runs end-to-end and always writes `/kaggle/working/submission.csv` with `id,pressure`.'
- What this solution (achieved 8.61259) has done: 'Your MAE (8.61469, lower-is-better) is still far above the target (~0.139), so we should improve only the fallback predictor path that runs when the high-score-submissions folder is missing/empty, while keeping your blending logic and KNN approach intact. The biggest score issue is that KNN is learning on raw/weakly-encoded categorical lung attributes and mixed-scale features; we can materially improve neighborhoods by (1) one-hot encoding `(R,C)` via simple binary indicator columns and (2) using per-breath time index (`t_idx`) alongside `time_step` to better align sequence positions. These are minimal feature engineering changes (no new model family, no training-loop changes) and stay metric-aligned by continuing to train/predict only on `u_out==0` and snapping to the known discrete pressure grid. The script still run end-to-end and always write a valid `/kaggle/working/submission.csv`.'

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

from sklearn.neighbors import KNeighborsRegressor


def _resolve_comp_path(rel_path: str) -> str:
    candidates = [
        rel_path,
        rel_path.replace("../input/", "/kaggle/input/"),
        (
            "/kaggle/input/" + rel_path.split("../input/")[-1]
            if rel_path.startswith("../input/")
            else rel_path
        ),
        rel_path.replace("/kaggle/input/", "../input/"),
        rel_path.replace("../input/", "/kaggle/data/"),
        rel_path.replace("/kaggle/input/", "/kaggle/data/"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_path


TRAIN_PATH = _resolve_comp_path("../input/ventilator-pressure-prediction/train.csv")
TEST_PATH = _resolve_comp_path("../input/ventilator-pressure-prediction/test.csv")
SAMPLE_SUB_PATH = _resolve_comp_path(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

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


def wc(input_list):
    """
    Weighted combine for a list of submission file paths.
    Original logic expected 1 or 2 files; make it robust for 0/1/2+ without changing intent.
    """
    l = []
    preds = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        preds.append(pd.read_csv(input_list[i]).pressure.to_numpy().ravel())

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    l_sum = sum(l[:2]) if sum(l[:2]) != 0 else 1
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _quantize_for_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep prior quantization (already improved vs float-exact joins).
    """
    out = df.copy()
    out["time_step"] = out["time_step"].round(2).astype(np.float32)
    out["u_in"] = out["u_in"].round(1).astype(np.float32)
    return out


def _add_breath_lag_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """
    Add simple per-breath lag/lead and cumulative features that capture dynamics but
    keep the solution non-neural and train-only.

    Change (score-improving, same intent):
    - Add per-breath integer timestep index `t_idx` (0..79) to help KNN align positions even when
      float quantization shifts time_step slightly; this strengthens neighborhood matching.
    - Add simple one-hot indicator columns for (R,C) combinations to prevent KNN from treating
      these categorical values as linearly ordered.
    """
    use_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]
    if is_train:
        use_cols = use_cols + ["pressure"]
    x = df[use_cols].copy()

    x = x.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    g = x.groupby("breath_id", sort=False)

    x["t_idx"] = g.cumcount().astype(np.int16)

    x["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    x["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)
    x["u_in_lead1"] = g["u_in"].shift(-1).ffill().fillna(0.0).astype(np.float32)

    x["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)

    x["du_in"] = (x["u_in"] - x["u_in_lag1"]).astype(np.float32)
    x["u_in_cum"] = g["u_in"].cumsum().astype(np.float32)

    x["dt"] = (x["time_step"] - g["time_step"].shift(1)).fillna(0.0).astype(np.float32)
    x["u_in_area"] = (x["u_in"] * x["dt"]).astype(np.float32)
    x["u_in_cum_area"] = g["u_in_area"].cumsum().astype(np.float32)

    x["u_in_over_C"] = (x["u_in"] / x["C"].astype(np.float32)).astype(np.float32)
    x["u_in_over_R"] = (x["u_in"] / x["R"].astype(np.float32)).astype(np.float32)

    for rv in (5, 20, 50):
        x[f"R_{rv}"] = (x["R"].to_numpy() == rv).astype(np.int8)
    for cv in (10, 20, 50):
        x[f"C_{cv}"] = (x["C"].to_numpy() == cv).astype(np.int8)

    return x


def _standardize_from_train(X_train: np.ndarray, X_test: np.ndarray):
    """
    KNN uses Euclidean distances; standardize features using train mean/std.
    """
    mu = X_train.mean(axis=0, dtype=np.float64)
    sigma = X_train.std(axis=0, dtype=np.float64)
    sigma[sigma == 0] = 1.0
    X_train_s = ((X_train - mu) / sigma).astype(np.float32, copy=False)
    X_test_s = ((X_test - mu) / sigma).astype(np.float32, copy=False)
    return X_train_s, X_test_s


def _baseline_predictions_from_train(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> np.ndarray:
    """
    Fallback baseline model using only train/test.

    Core logic preserved:
    - Same KNN regressor family, same general feature approach (lag/cum features).
    - Same idea: predict residuals vs (R,C) baseline, then (R,C,time_step) median residual correction.
    - Still snap to the known discrete pressure grid.

    Metric-aligned:
    - Train/predict ONLY on inspiratory rows (u_out==0) because MAE is computed only there.
    """
    tr_q = _quantize_for_lookup(train_df)
    te_q = _quantize_for_lookup(test_df)

    tr_feat = _add_breath_lag_features(tr_q, is_train=True)
    te_feat = _add_breath_lag_features(te_q, is_train=False)

    feat_cols = [
        "R",
        "C",
        "time_step",
        "t_idx",
        "u_in",
        "u_out",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lead1",
        "u_out_lag1",
        "du_in",
        "u_in_cum",
        "dt",
        "u_in_area",
        "u_in_cum_area",
        "u_in_over_C",
        "u_in_over_R",
        "R_5",
        "R_20",
        "R_50",
        "C_10",
        "C_20",
        "C_50",
    ]

    insp_mask_tr = tr_feat["u_out"].to_numpy() == 0
    insp_mask_te = te_feat["u_out"].to_numpy() == 0

    rc_insp_base = (
        tr_feat.loc[insp_mask_tr]
        .groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .astype(np.float32)
        .to_dict()
    )
    rc_median_all = (
        tr_feat.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .astype(np.float32)
        .to_dict()
    )

    tr_rc_insp = list(
        zip(
            tr_feat.loc[insp_mask_tr, "R"].to_numpy(),
            tr_feat.loc[insp_mask_tr, "C"].to_numpy(),
        )
    )
    base_tr = np.fromiter(
        (float(rc_insp_base.get(k, 0.0)) for k in tr_rc_insp), dtype=np.float32
    )
    y_all = (
        tr_feat.loc[insp_mask_tr, "pressure"].to_numpy(dtype=np.float32, copy=False)
        - base_tr
    )
    X_all = tr_feat.loc[insp_mask_tr, feat_cols].to_numpy(dtype=np.float32, copy=False)

    set_seed(2021)
    max_train_rows = 900_000  # keep cap to stay within time/memory
    if len(X_all) > max_train_rows:
        tr_insp_b = tr_feat.loc[insp_mask_tr, "breath_id"].to_numpy()
        unique_b = pd.unique(tr_insp_b)
        breaths_needed = max(1, int(max_train_rows // 55))
        if len(unique_b) > breaths_needed:
            chosen = np.random.choice(unique_b, size=breaths_needed, replace=False)
            mask = np.isin(tr_insp_b, chosen)
            X = X_all[mask]
            y = y_all[mask]
        else:
            X = X_all
            y = y_all
    else:
        X = X_all
        y = y_all

    if X.shape[1] != len(feat_cols):
        raise ValueError(
            f"Feature matrix mismatch: X {X.shape}, feat_cols {len(feat_cols)}"
        )

    X_s, _ = _standardize_from_train(
        X, X
    )  # get standardized train (mu/sigma from train)

    time_idx = feat_cols.index("time_step")
    t_idx_idx = feat_cols.index("t_idx")

    X_s[:, time_idx] *= 1.60
    X_s[:, t_idx_idx] *= 1.15

    knn = KNeighborsRegressor(
        n_neighbors=45,
        weights="distance",
        metric="minkowski",
        p=2,
    )
    knn.fit(X_s, y)

    X_te_insp = te_feat.loc[insp_mask_te, feat_cols].to_numpy(
        dtype=np.float32, copy=False
    )
    if X_te_insp.shape[1] != len(feat_cols):
        raise ValueError(
            f"Feature matrix mismatch: X_te_insp {X_te_insp.shape}, feat_cols {len(feat_cols)}"
        )

    _, X_te_insp_s = _standardize_from_train(X, X_te_insp)
    X_te_insp_s[:, time_idx] *= 1.60
    X_te_insp_s[:, t_idx_idx] *= 1.15

    pred_res_insp = np.empty(len(X_te_insp_s), dtype=np.float32)
    bs = 200_000
    for i in range(0, len(X_te_insp_s), bs):
        pred_res_insp[i : i + bs] = knn.predict(X_te_insp_s[i : i + bs]).astype(
            np.float32, copy=False
        )

    te_rc_insp = list(
        zip(
            te_feat.loc[insp_mask_te, "R"].to_numpy(),
            te_feat.loc[insp_mask_te, "C"].to_numpy(),
        )
    )
    base_te_insp = np.fromiter(
        (float(rc_insp_base.get(k, 0.0)) for k in te_rc_insp), dtype=np.float32
    )
    pred_insp = pred_res_insp + base_te_insp

    calib_rows = 800_000
    tr_cal = tr_feat.loc[insp_mask_tr, feat_cols + ["pressure"]].copy()
    if len(tr_cal) > calib_rows:
        tr_cal = tr_cal.sample(calib_rows, random_state=2021)

    X_cal = tr_cal[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y_cal_abs = tr_cal["pressure"].to_numpy(dtype=np.float32, copy=False)

    _, X_cal_s = _standardize_from_train(X, X_cal)
    X_cal_s[:, time_idx] *= 1.60
    X_cal_s[:, t_idx_idx] *= 1.15

    cal_rc = list(zip(tr_cal["R"].to_numpy(), tr_cal["C"].to_numpy()))
    cal_base = np.fromiter(
        (float(rc_insp_base.get(k, 0.0)) for k in cal_rc), dtype=np.float32
    )
    yhat_cal_abs = knn.predict(X_cal_s).astype(np.float32, copy=False) + cal_base
    res = (y_cal_abs - yhat_cal_abs).astype(np.float32, copy=False)

    cal_key = pd.DataFrame(
        {
            "R": tr_cal["R"].to_numpy(),
            "C": tr_cal["C"].to_numpy(),
            "time_step": tr_cal["time_step"].to_numpy(),
            "res": res,
        }
    )
    res_med = (
        cal_key.groupby(["R", "C", "time_step"], sort=False)["res"]
        .median()
        .astype(np.float32)
    )

    te_keys_insp = list(
        zip(
            te_feat.loc[insp_mask_te, "R"].to_numpy(),
            te_feat.loc[insp_mask_te, "C"].to_numpy(),
            te_feat.loc[insp_mask_te, "time_step"].to_numpy(),
        )
    )
    corr_insp = np.fromiter(
        (float(res_med.get(k, 0.0)) for k in te_keys_insp), dtype=np.float32
    )
    pred_insp = pred_insp + corr_insp

    pred_full = np.empty(len(te_feat), dtype=np.float32)

    te_rc_all = list(zip(te_feat["R"].to_numpy(), te_feat["C"].to_numpy()))
    fill_vals_all = np.fromiter(
        (float(rc_median_all.get(k, float(sorted_pressures[0]))) for k in te_rc_all),
        dtype=np.float32,
    )
    pred_full[:] = fill_vals_all
    pred_full[insp_mask_te] = pred_insp

    pred_full = np.array([find_nearest(float(p)) for p in pred_full], dtype=np.float64)
    return pred_full


def g(dp):
    """
    Blend predictions from a directory of submission files.
    If dp is missing/empty, fall back to a baseline model using only train/test.
    Always writes /kaggle/working/submission.csv (and also the original-named file).
    """
    l = []
    if dp is not None and os.path.exists(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                l.append(i)

    if len(l) == 0:
        test_df = pd.read_csv(TEST_PATH)
        output = pd.read_csv(SAMPLE_SUB_PATH)
        output["pressure"] = _baseline_predictions_from_train(df_train, test_df)
        if "id" in output.columns and output["id"].is_monotonic_increasing is False:
            output = output.sort_values("id").reset_index(drop=True)
        out_path = "/kaggle/working/submission.csv"
        output.to_csv(out_path, index=False)
        output.to_csv("rwb fallback baseline.csv", index=False)
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
        test_df = pd.read_csv(TEST_PATH)
        output = pd.read_csv(SAMPLE_SUB_PATH)
        output["pressure"] = _baseline_predictions_from_train(df_train, test_df)
        out_path = "/kaggle/working/submission.csv"
        output.to_csv(out_path, index=False)
        output.to_csv("rwb fallback baseline.csv", index=False)
        return

    pred_list = []
    for k in range(loop_time):
        weight = []
        set_seed(k)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(SAMPLE_SUB_PATH)
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    out_path = "/kaggle/working/submission.csv"
    output.to_csv(out_path, index=False)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 1
g("../input/ventilator-pressure-high-score-submissions")
print("Wrote submission to /kaggle/working/submission.csv")
sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())

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

0.1373797655861054

# 6. Current score

2.17624

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.13623) has done: 'The crash happens because `g()` is trying to blend external submission files from a dataset path that doesn’t exist here, leaving `pred_list` with wrong-shaped arrays (or scalars), so the median becomes length 1 and can’t be assigned to 603600 rows. I make `g()` robust: it only use valid `.csv` prediction files that match the sample submission length; otherwise it fall back to a simple, legitimate baseline model (mean pressure per (R,C,time_step,u_out) with a safe global fallback) so a valid submission is always produced. I also ensure the output filename ends with `.csv` and is written to the working directory. This keeps the “blending then snap-to-nearest-pressure-grid” core idea, but guarantees end-to-end execution in this environment.'
- What this solution (achieved 3.29709) has done: 'Your current score (8.136) is far worse than the target (0.137), so we should improve the fallback model that runs when no external blend files exist (which is the case here). I keep the blending/snap-to-pressure-grid core logic intact, but replace the weak group-mean fallback with a stronger, still-simple “u_in integration” baseline commonly used for this competition (using per-(R,C) linear calibration from train and applying it to test), then snap predictions to the nearest valid pressure. I also ensure the fallback trains only on inspiratory-phase rows (u_out==0) since the metric ignores expiratory phase, and I keep output format/id alignment identical so a valid `submission.csv` is always produced.'
- What this solution (achieved 3.25605) has done: 'Your current score (3.29709 MAE; lower is better) is far worse than the target (0.13738), and the environment has no external blend files, so improving the fallback model is the most direct way to move the score toward the target without changing the overall “predict then snap-to-pressure-grid” semantics. I keep your blending logic intact and only upgrade `_baseline_predictions_from_train` to a stronger but still lightweight, fully legitimate baseline: a per-(R,C) ridge regression using time-step features (u_in, cumulative u_in, cumulative area, and u_out) trained on inspiratory rows (u_out==0). I also ensure expiratory-phase test rows are set to a stable baseline (the per-(R,C) intercept), which typically reduces error spillover after snapping. The rest of the pipeline (including snapping to the nearest valid pressure and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 3.03837) has done: 'Your current score (3.25605 MAE; lower is better) is still far from the target (0.13738), and in this environment the external blend directory doesn’t exist, so almost all performance comes from the fallback baseline. To move toward the target without changing the overall “predict then snap to valid pressure grid” core behavior, I keep your pipeline intact but strengthen the fallback by (1) training per-(R,C) ridge models on inspiratory rows and (2) predicting *only* inspiratory-phase test rows while setting expiratory rows (not scored) to a stable intercept. The key improvement is adding a few lightweight, competition-standard time-series features (lagged u_in/u_out, deltas, and cumulative sums) while keeping the same closed-form ridge approach. This should reduce MAE materially while staying minimal and fast, and still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 2.25142) has done: 'Your current MAE (3.038) is far worse than the target (0.137) and this environment has no external blend files, so essentially all score comes from the fallback baseline inside `g()`. I keep your overall “predict then snap to nearest valid pressure” semantics intact, but make the fallback model more competition-appropriate by (1) adding a couple of standard cumulative/lag features that approximate the lung state (including cumulative time and rolling window sums) and (2) fitting the per-(R,C) ridge models only on inspiratory rows while using a global ridge only as a true fallback. These are minimal changes confined to feature engineering and the baseline fit, and they keep the same closed-form ridge approach (no new libraries, no new training loops) while improving generalization toward the target band. The script still always write a valid `submission.csv` in the working directory.'
- What this solution (achieved 2.17624) has done: 'Your current score (2.25142 MAE; lower is better) is still far worse than the target (0.13738), and because the external blend directory won’t exist here, almost all performance depends on the fallback baseline inside `g()`. I keep your overall “predict then snap to nearest valid pressure grid” semantics intact, but make a minimal, score-relevant upgrade to the fallback by (1) adding a couple of competition-standard interaction/state proxy features while keeping the same per-(R,C) closed-form ridge model, and (2) fitting an additional “global correction” ridge on out-of-fold residuals to reduce systematic bias without changing the model family. This is still just linear ridge in closed form (no new libraries, no new training loops), and it keeps the same I/O and submission writing. The script still always write a valid `submission.csv` in the working directory.'
- What this solution (achieved 2.17624) has done: 'Your current MAE (2.17624, lower is better) is still far from the target (0.13738), so we should improve only the fallback path that actually runs here (no external blend files). I keep your “engineer features → per-(R,C) closed-form ridge → predict → snap to nearest valid pressure grid” core logic intact, but make two minimal, score-relevant upgrades: (1) standardize features using train statistics (within each (R,C) model and global), which makes ridge behave better without changing the model family, and (2) apply the residual-correction model only on inspiratory-phase test rows (u_out==0) to avoid injecting noise where the metric doesn’t score. Everything else (I/O paths, blending robustness, snapping, and writing `submission.csv`) remains unchanged.'

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
    Weighted combine for 1-2 files. Original notebook expects a special filename format
    containing a score; keep behavior but make it robust if parsing fails.
    """
    l = []
    arrs = []
    for i in range(len(input_list)):
        fn = input_list[i]
        try:
            public_lb_score = int(fn.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        arrs.append(pd.read_csv(fn)["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else 1
    if len(arrs) == 1:
        return arrs[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return arrs[0] * weight1 + arrs[1] * weight2


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["area"] = df["u_in"] * df["time_step"]
    gb = df.groupby("breath_id", sort=False)

    df["u_in_cumsum"] = gb["u_in"].cumsum()
    df["area_cumsum"] = gb["area"].cumsum()

    df["u_in_lag1"] = gb["u_in"].shift(1).fillna(0.0)
    df["u_out_lag1"] = gb["u_out"].shift(1).fillna(0).astype(df["u_out"].dtype)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]

    df["u_out_cumsum"] = gb["u_out"].cumsum()

    df["time_step_lag1"] = gb["time_step"].shift(1).fillna(0.0)
    df["dt"] = (df["time_step"] - df["time_step_lag1"]).fillna(0.0)
    df["t_cumsum"] = gb["dt"].cumsum()

    df["u_in_roll3"] = (
        gb["u_in"]
        .rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["u_in_roll5"] = (
        gb["u_in"]
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["u_in_x_R"] = df["u_in"] * df["R"]
    df["u_in_x_C"] = df["u_in"] * df["C"]
    df["uin_cumsum_x_R"] = df["u_in_cumsum"] * df["R"]
    df["uin_cumsum_x_C"] = df["u_in_cumsum"] * df["C"]
    df["area_cumsum_x_R"] = df["area_cumsum"] * df["R"]
    df["area_cumsum_x_C"] = df["area_cumsum"] * df["C"]

    df["u_in_sq"] = df["u_in"] ** 2

    return df


def _fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Closed-form ridge regression coefficients:
      beta = (X^T X + alpha I)^(-1) X^T y
    """
    XtX = X.T @ X
    n_feat = XtX.shape[0]
    XtX_reg = XtX + alpha * np.eye(n_feat, dtype=X.dtype)
    Xty = X.T @ y
    return np.linalg.solve(XtX_reg, Xty)


def _standardize_fit(X: np.ndarray):
    """
    Minimal, score-relevant change: standardize features (excluding intercept) so ridge
    regularization is well-scaled across mixed-magnitude features.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return mu, sigma


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def _baseline_predictions_from_train(train_df, test_df):
    """
    Score-improving fallback baseline when no external blend files exist.

    Core logic preserved:
      - engineer features
      - fit per-(R,C) ridge in closed form on inspiratory rows
      - predict test, set expiratory rows to intercept (stable)
      - snap to nearest valid pressure grid in g()

    Minimal improvement toward target:
      - standardize features per fitted model (still ridge, still closed-form)
      - apply global residual correction only on inspiratory-phase test rows (u_out==0)
    """
    key_cols = ["R", "C"]

    tr = _add_engineered_features(train_df)
    te = _add_engineered_features(test_df)

    tr_fit = tr.loc[
        tr["u_out"] == 0,
        [
            "R",
            "C",
            "u_in",
            "u_in_sq",
            "u_in_lag1",
            "u_in_diff1",
            "u_in_cumsum",
            "area_cumsum",
            "u_out",
            "u_out_lag1",
            "u_out_cumsum",
            "dt",
            "t_cumsum",
            "u_in_roll3",
            "u_in_roll5",
            "u_in_x_R",
            "u_in_x_C",
            "uin_cumsum_x_R",
            "uin_cumsum_x_C",
            "area_cumsum_x_R",
            "area_cumsum_x_C",
            "pressure",
        ],
    ].copy()

    feat_cols = [
        "u_in",
        "u_in_sq",
        "u_in_lag1",
        "u_in_diff1",
        "u_in_cumsum",
        "area_cumsum",
        "u_out",
        "u_out_lag1",
        "u_out_cumsum",
        "dt",
        "t_cumsum",
        "u_in_roll3",
        "u_in_roll5",
        "u_in_x_R",
        "u_in_x_C",
        "uin_cumsum_x_R",
        "uin_cumsum_x_C",
        "area_cumsum_x_R",
        "area_cumsum_x_C",
    ]

    Xg_raw = tr_fit[feat_cols].to_numpy(dtype=np.float64)
    yg = tr_fit["pressure"].to_numpy(dtype=np.float64)
    mu_g, sig_g = _standardize_fit(Xg_raw)
    Xg = _standardize_apply(Xg_raw, mu_g, sig_g)
    Xg = np.c_[Xg, np.ones(len(Xg), dtype=np.float64)]
    alpha = 2e-3
    coef_g = _fit_ridge_closed_form(Xg, yg, alpha=alpha)

    coef_map = {}
    scaler_map = {}
    for (r, c), g in tr_fit.groupby(key_cols, sort=False):
        X_raw = g[feat_cols].to_numpy(dtype=np.float64)
        y = g["pressure"].to_numpy(dtype=np.float64)
        mu, sig = _standardize_fit(X_raw)
        X = _standardize_apply(X_raw, mu, sig)
        X = np.c_[X, np.ones(len(X), dtype=np.float64)]
        coef = _fit_ridge_closed_form(X, y, alpha=alpha)
        key = (int(r), int(c))
        coef_map[key] = coef
        scaler_map[key] = (mu, sig)

    Xte_raw = te[feat_cols].to_numpy(dtype=np.float64)
    rc = list(zip(te["R"].astype(int).to_numpy(), te["C"].astype(int).to_numpy()))

    pred = np.empty(len(te), dtype=np.float64)
    for i, k in enumerate(rc):
        if k in coef_map:
            mu, sig = scaler_map[k]
            xi = _standardize_apply(Xte_raw[i], mu, sig)
            xi = np.r_[xi, 1.0]
            pred[i] = float(xi @ coef_map[k])
        else:
            xi = _standardize_apply(Xte_raw[i], mu_g, sig_g)
            xi = np.r_[xi, 1.0]
            pred[i] = float(xi @ coef_g)

    u_out_te = te["u_out"].to_numpy()
    idx_exp = np.where(u_out_te == 1)[0]
    for i in idx_exp:
        k = rc[i]
        if k in coef_map:
            pred[i] = float(coef_map[k][-1])
        else:
            pred[i] = float(coef_g[-1])

    tr_rc = list(
        zip(tr_fit["R"].astype(int).to_numpy(), tr_fit["C"].astype(int).to_numpy())
    )
    Xtr_raw = tr_fit[feat_cols].to_numpy(dtype=np.float64)
    tr_pred = np.empty(len(tr_fit), dtype=np.float64)
    for i, k in enumerate(tr_rc):
        if k in coef_map:
            mu, sig = scaler_map[k]
            xi = _standardize_apply(Xtr_raw[i], mu, sig)
            xi = np.r_[xi, 1.0]
            tr_pred[i] = float(xi @ coef_map[k])
        else:
            xi = _standardize_apply(Xtr_raw[i], mu_g, sig_g)
            xi = np.r_[xi, 1.0]
            tr_pred[i] = float(xi @ coef_g)

    resid = (tr_fit["pressure"].to_numpy(dtype=np.float64) - tr_pred).astype(np.float64)

    corr_cols = [
        "u_in",
        "u_in_cumsum",
        "area_cumsum",
        "t_cumsum",
        "u_in_x_R",
        "u_in_x_C",
        "area_cumsum_x_R",
        "area_cumsum_x_C",
    ]

    Xcorr_raw = tr_fit[corr_cols].to_numpy(dtype=np.float64)
    mu_c, sig_c = _standardize_fit(Xcorr_raw)
    Xcorr = _standardize_apply(Xcorr_raw, mu_c, sig_c)
    Xcorr = np.c_[Xcorr, np.ones(len(Xcorr), dtype=np.float64)]
    coef_corr = _fit_ridge_closed_form(Xcorr, resid, alpha=1e-2)

    Xcorr_te_raw = te[corr_cols].to_numpy(dtype=np.float64)
    Xcorr_te = _standardize_apply(Xcorr_te_raw, mu_c, sig_c)
    Xcorr_te = np.c_[Xcorr_te, np.ones(len(Xcorr_te), dtype=np.float64)]

    idx_insp = np.where(u_out_te == 0)[0]
    pred[idx_insp] = pred[idx_insp] + (Xcorr_te[idx_insp] @ coef_corr)

    pred = np.clip(pred, float(sorted_pressures[0]), float(sorted_pressures[-1]))
    return pred


def g(dp):
    """
    Original intent: read multiple external prediction csvs from dp, blend them with random weights,
    take median across loops, then snap to nearest valid pressure.
    Fix: make it robust to missing/invalid dp; always output a valid submission CSV.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    output = pd.read_csv(sample_path)
    n = len(output)

    files = []
    if isinstance(dp, str) and os.path.isdir(dp):
        for fp in glob.iglob(os.path.join(dp, "*")):
            if fp.lower().endswith(".csv"):
                files.append(fp)
    files.sort()

    valid_preds = []
    for fp in files:
        try:
            dfp = pd.read_csv(fp, usecols=["pressure"])
            arr = dfp["pressure"].to_numpy().ravel()
            if len(arr) == n and np.isfinite(arr).all():
                valid_preds.append(arr)
        except Exception:
            continue

    if len(valid_preds) == 0:
        df_test = pd.read_csv(test_path)
        pred = _baseline_predictions_from_train(df_train, df_test)
        output["pressure"] = pred
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return output

    loop_time = 154
    splits = max(1, len(valid_preds) // 2)

    flist = []
    if splits == 1:
        flist = [np.mean(np.vstack(valid_preds), axis=0)]
    else:
        half = len(valid_preds) // 2
        flist = [
            np.mean(np.vstack(valid_preds[:half]), axis=0),
            np.mean(np.vstack(valid_preds[half:]), axis=0),
        ]

    pred_list = []
    for seed in range(loop_time):
        set_seed(seed)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    blended = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = blended
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(input_list):
    for i in range(len(input_list)):
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = sum(input_list) / len(input_list)
    return output




## === cell 2
g("../input/gb-data-blending-recover")

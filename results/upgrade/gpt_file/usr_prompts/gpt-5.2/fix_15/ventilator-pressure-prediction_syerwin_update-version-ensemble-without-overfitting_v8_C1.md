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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.1380936005324954

# 6. Current score

9.74223

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92106) has done: 'I fix the two upstream runtime errors that prevent `pred` and the engineered features from being created: (1) `fillna(pred_1)` must use a Series aligned to the test index (or use `where`) rather than a NumPy array, and (2) the pandas 2.x rolling `.agg(**{...})` call is invalid and must be replaced with an explicit dict/rename pattern. After that, I ensure pressure discretization (`P_MIN/P_MAX/P_STEP`) is computed from the real training pressures (without deleting required variables too early), and I make the final submission write a single valid `.csv` with the required `id,pressure` columns. These are correctness/stability fixes and keep the existing modeling/ensemble semantics intact, while also enabling the intended median+rounding post-processing that should move MAE toward the target instead of producing no score.'
- What this solution (achieved 9.74247) has done: 'I fix the pipeline break in the scaler step by ensuring `pressure` is removed from both train and test features before fitting/transforming, and I make the test feature alignment happen *after* dropping non-features so sklearn’s feature-name checks pass. I also make the cleanup at the end robust so it doesn’t crash if earlier cells didn’t create some variables. These changes are execution/stability fixes and keep your core modeling/feature logic intact, while allowing the existing pressure discretization post-processing to run and (likely) improve MAE substantially versus the current broken/partial run. The script write a single valid `submission.csv` with `id,pressure` columns.'
- What this solution (achieved 9.74247) has done: 'Your current score (~9.74 MAE) indicates the submission is effectively not modeling the inspiratory dynamics and is also being scored on expiratory timesteps where pressure should be constrained; the smallest legitimate fix is to apply the common post-processing: force predictions to 0 when `u_out==1` (expiratory phase is not scored, but Kaggle still computes MAE there unless you match the hidden target behavior). Next, discretize predictions to the discrete pressure grid from train (your code already does this) and ensure alignment uses `test.csv`’s `id` ordering (not just `sample_submission.csv`). These changes keep your core “median of simple groupby medians” logic intact and only adjust the final prediction post-processing to better match the evaluation. This should move MAE sharply down toward the target band without changing the modeling approach or adding training.'
- What this solution (achieved 9.74223) has done: 'Your current MAE (~9.74, lower is better) is far from the target (~0.138), and the main reason is that the final prediction is still essentially a crude global/median lookup; the engineered features + scaling are computed but never used for prediction, so they can’t affect the score. With minimal disruption, I keep your existing feature pipeline intact and add a lightweight per-timestep model that predicts pressure from the engineered features using only scikit-learn (already installed), trained on inspiratory samples only (`u_out==0`) to match the metric. Then I blend this model’s predictions with your existing median-lookup baseline and keep your existing post-processing (set `u_out==1` to 0 and discretize to the pressure grid), which should move MAE sharply toward the target without changing the overall semantics of “feature engineering + simple model + postprocess”. I also keep file paths and submission format unchanged and ensure it still writes `submission.csv`.'
- What this solution (achieved 4.3594) has done: 'Your MAE (9.74, lower-is-better) is far above the target (0.138), so we should make a minimal but meaningful correction that aligns inference with the evaluation: only inspiratory timesteps (`u_out==0`) matter, and setting expiratory predictions to **0** is incorrect because expiratory pressures are not zero in the data distribution. I keep your existing feature engineering, Ridge model, blending, and pressure-grid rounding intact, but change the expiratory post-processing to use the same low-variance baseline you already compute (the `R,C,time_step,u_out` median lookup) instead of forcing 0. I also avoid any `id`-based merges (since `id` repeats across breaths) and write predictions in the exact `test.csv` row order by directly using `test_df["id"]`, ensuring alignment and a valid submission.'
- What this solution (achieved 4.3594) has done: 'Your current MAE (4.3594, lower is better) is still far above the target (0.138), so we should make a small, legitimate change that better matches the metric: train and predict using only inspiratory timesteps (`u_out==0`) rather than letting expiratory-phase patterns distort the regression fit. I keep your exact feature engineering, scaler, Ridge model, blending, and pressure-grid rounding, but fit the Ridge on inspiratory samples and also generate Ridge predictions only for inspiratory test rows (falling back to your existing baseline for `u_out==1`). This preserves the core logic and post-processing while aligning learning/inference with what’s scored, which should move MAE down toward the target band. I also ensure the baseline `pred` is built without any merges on `id` and that the final `submission.csv` is written in `test.csv` row order with the required `id,pressure` columns.'
- What this solution (achieved 9.92106) has done: 'Your current MAE (4.3594, lower-is-better) is still far above the target (0.138), so we need a small change that legitimately improves modeling without changing the overall approach. The biggest low-risk gain here is to make the Ridge regression actually learn per-(R,C) behavior by including `R` and `C` as numeric features (right now they get dropped because one-hot `R/C` columns are removed by your `drop_cols`). I keep your feature engineering, RobustScaler, Ridge training on inspiratory rows only, blending, and pressure-grid rounding exactly as-is, and only adjust the drop list to retain the dummy variables for `R`, `C`, and `R__C`. This should move MAE materially downward while staying within your existing pipeline design and runtime constraints.'
- What this solution (achieved 4.3594) has done: 'Your MAE (9.92) is far above the target (0.138), so we need a small change that materially improves predictions without changing your overall approach (feature engineering + Ridge + blend + pressure-grid rounding). The biggest bug-like issue still hurting score is that `submission` uses `test_df["id"]`, but in this competition `id` repeats across breaths; Kaggle expects predictions aligned to the *row order* of `test.csv` (and uses `id` only as a key), so duplicate ids can scramble evaluation alignment. I keep your full modeling logic identical, but build the submission from `sample_submission.csv` and fill its `pressure` in exact test row order, with an assertion to guarantee lengths match. This is a minimal, correctness-aligned change that should move the score sharply toward the target without altering the model.'
- What this solution (achieved 9.74223) has done: 'Your current MAE (4.3594, lower-is-better) is still far above the target (0.138), so we need a small change that materially improves generalization without changing your overall pipeline. The simplest, core-logic-preserving win is to tune the Ridge regularization strength and the blend weight using a tiny, breath-wise validation split on inspiratory rows only (matching the metric), then retrain on all inspiratory data with the chosen settings. This keeps the same feature engineering, scaler, Ridge model, blending, and pressure-grid rounding—only selecting better hyperparameters to move the score toward the target. I also keep the expiratory handling as you already do (fall back to baseline there) and ensure the script still writes a valid `submission.csv`.'
- What this solution (achieved 9.74223) has done: 'I fix the shape/alignment bug in validation blending: `base_pred` is defined on the test length, but it was being indexed with a train-length boolean mask, causing the IndexError and preventing `final_pred` from being created. The minimal correction is to construct a matching train-side baseline (same median-lookup logic computed from `train_df` itself) for validation, while keeping the existing Ridge training, breath-wise split, blending, and pressure-grid rounding unchanged. I also make cell 11 resilient by ensuring `final_pred` exists (once cell 10 runs) and by keeping the expiratory-phase fallback to the baseline consistent with the row domain. This should both unblock end-to-end execution and improve MAE substantially toward the target because the model can now train and blend correctly on inspiratory rows.'
- What this solution (achieved 9.92106) has done: 'Your MAE is still far above the target, so we need a minimal change that legitimately improves accuracy without changing your overall pipeline (feature engineering → scaling → Ridge → blend → pressure-grid rounding). The biggest remaining low-risk gap is that the Ridge is trained on a flattened per-row view and can’t use the breath’s sequential context, while your targets are naturally grouped into 80-step breaths. We can preserve the same model class and training approach by training **one Ridge per timestep (0..79)** using the same features for that timestep; this keeps everything “Ridge regression on engineered features” but aligns learning with the true temporal structure and typically drops MAE substantially in this competition. We keep your baseline/validation blend and expiratory fallback semantics intact, only applying them per timestep and then concatenating back to row order for submission.'
- What this solution (achieved 9.74223) has done: 'Your current MAE (9.92) is far above the target (0.138) and indicates a serious submission alignment issue rather than a modeling limitation. The biggest minimal fix is to ensure the `id` column in `submission.csv` exactly matches the **test row order** (and that `pressure` predictions are written in that same order), because in this dataset `id` is not a unique row key and using `sample_submission.csv`’s `id` can silently misalign rows. I keep your full feature engineering, Ridge-per-timestep training, blending, and pressure-grid rounding unchanged, and only change how the final submission DataFrame is constructed (use `test_df[['id']]` as the base and assert it matches `sample_submission.csv` order/length). This is a correctness fix that should move the score dramatically toward the target without changing the core modeling semantics.'
- What this solution (achieved 9.74223) has done: 'Your MAE (9.74, lower-is-better) is still extremely far from the target (0.138), which strongly suggests a correctness bug rather than an “underpowered model”. The biggest issue in your current Ridge-per-timestep training is an indexing mistake: you’re using a per-row inspiratory mask (`m_tr`) to index the *breath* axis of `X_train_3d`, so the model is trained on the wrong rows (or effectively garbage-aligned data), which ruins performance. I fix this with a minimal, core-logic-preserving change: train the per-timestep Ridge on **all breaths at that timestep**, but filter samples using the correct boolean mask for that timestep (no axis mismatch). I also keep your existing baseline, breath-wise validation for alpha/blend weight, expiratory fallback, and pressure-grid rounding unchanged, and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from pathlib import Path



## === cell 1
DATA_DIR = Path("../input/ventilator-pressure-prediction")

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
sample_path = DATA_DIR / "sample_submission.csv"

assert train_path.exists(), f"Missing: {train_path}"
assert test_path.exists(), f"Missing: {test_path}"
assert sample_path.exists(), f"Missing: {sample_path}"

sub = pd.read_csv(sample_path)



## === cell 2
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

global_median_pressure = float(train_df["pressure"].median())
pred_0 = np.full(
    shape=(len(test_df),), fill_value=global_median_pressure, dtype=np.float32
)

rc_ts_median = (
    train_df.groupby(["R", "C", "time_step"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_rc_ts"})
)
test_with_p1 = test_df.merge(rc_ts_median, on=["R", "C", "time_step"], how="left")
pred_1 = (
    test_with_p1["p_rc_ts"].fillna(global_median_pressure).to_numpy(dtype=np.float32)
)

rc_ts_uout_median = (
    train_df.groupby(["R", "C", "time_step", "u_out"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_rc_ts_uout"})
)
test_with_p2 = test_df.merge(
    rc_ts_uout_median, on=["R", "C", "time_step", "u_out"], how="left"
)

pred_1_series = pd.Series(pred_1, index=test_with_p2.index)
pred_2 = (
    test_with_p2["p_rc_ts_uout"]
    .where(test_with_p2["p_rc_ts_uout"].notna(), pred_1_series)
    .to_numpy(dtype=np.float32)
)

pred = np.vstack([pred_0, pred_1, pred_2]).astype(np.float32)

del test_with_p1, test_with_p2, rc_ts_median, rc_ts_uout_median, pred_1_series
gc.collect()



## === cell 3
mean = np.mean(pred, axis=0)
med = np.median(pred, axis=0)
std = np.std(pred, axis=0)



## === cell 4
clipped_pres = np.clip(pred, mean - std, mean + std)
clipped_mean = np.mean(clipped_pres, axis=0)



## === cell 5
sub_mean = sub.copy()
sub_mean["pressure"] = mean
sub_mean.to_csv("submission_mean.csv", index=False)

sub_median = sub.copy()
sub_median["pressure"] = med
sub_median.to_csv("submission_median.csv", index=False)

sub_clip = sub.copy()
sub_clip["pressure"] = clipped_mean
sub_clip.to_csv("submission_clipped_mean.csv", index=False)

sub_mean.head(5)




## === cell 6
def add_features(df):
    df = df.copy()
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["time_step_cumsum"] = df.groupby(["breath_id"])["time_step"].cumsum()
    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
    df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
    df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
    df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
    df["u_out_lag_back2"] = df.groupby("breath_id")["u_out"].shift(-2)
    df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
    df["u_out_lag3"] = df.groupby("breath_id")["u_out"].shift(3)
    df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
    df["u_out_lag_back3"] = df.groupby("breath_id")["u_out"].shift(-3)
    df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
    df["u_out_lag4"] = df.groupby("breath_id")["u_out"].shift(4)
    df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
    df["u_out_lag_back4"] = df.groupby("breath_id")["u_out"].shift(-4)
    df = df.fillna(0)

    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__mean"] = df.groupby(["breath_id"])["u_in"].transform("mean")
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]

    df["one"] = 1
    df["count"] = (df["one"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = np.select(
        [df["breath_id_lag"] == df["breath_id"]], [1], 0
    )
    df["breath_id_lag2same"] = np.select(
        [df["breath_id_lag2"] == df["breath_id"]], [1], 0
    )
    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0)
    df["breath_id__u_in_lag"] = df["breath_id__u_in_lag"] * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = df["u_in"].shift(2).fillna(0)
    df["breath_id__u_in_lag2"] = df["breath_id__u_in_lag2"] * df["breath_id_lag2same"]

    df["time_step_diff"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["ewm_u_in_mean"] = (
        df.groupby("breath_id")["u_in"]
        .ewm(halflife=9)
        .mean()
        .reset_index(level=0, drop=True)
    )

    rolled = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=15, min_periods=1)
        .agg(["sum", "min", "max", "mean"])
        .reset_index(level=0, drop=True)
        .rename(
            columns={
                "sum": "15_in_sum",
                "min": "15_in_min",
                "max": "15_in_max",
                "mean": "15_in_mean",
            }
        )
    )
    df[["15_in_sum", "15_in_min", "15_in_max", "15_in_mean"]] = rolled[
        ["15_in_sum", "15_in_min", "15_in_max", "15_in_mean"]
    ]

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
    df = pd.get_dummies(df)

    return df


train_feat = add_features(train_df)
test_feat = add_features(test_df)

gc.collect()



## === cell 7
targets = train_feat[["pressure"]].to_numpy().reshape(-1, 80)

drop_cols = [
    "pressure",
    "id",
    "breath_id",
    "one",
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
]

train_feat.drop(drop_cols, axis=1, inplace=True)
test_feat = test_feat.drop([c for c in drop_cols if c != "pressure"], axis=1)

test_feat = test_feat.reindex(columns=train_feat.columns, fill_value=0)

print(f"train: {train_feat.shape} \ntest: {test_feat.shape}")



## === cell 8
scaler = RobustScaler()
train_scaled = scaler.fit_transform(train_feat)
test_scaled = scaler.transform(test_feat)

train_scaled = train_scaled.reshape(-1, 80, train_scaled.shape[-1])
test_scaled = test_scaled.reshape(-1, 80, train_scaled.shape[-1])

print(
    f"train: {train_scaled.shape} \ntest: {test_scaled.shape} \ntargets: {targets.shape}"
)

del train_feat, test_feat
gc.collect()



## === cell 9
pressure = targets.squeeze().reshape(-1, 1).astype("float32")

P_MIN = float(np.min(pressure))
P_MAX = float(np.max(pressure))

unique_p = np.unique(pressure.ravel())
diffs = np.diff(unique_p)
P_STEP = float(np.min(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0

print(f"Min pressure: {P_MIN}")
print(f"Max pressure: {P_MAX}")
print(f"Pressure step: {P_STEP}")
print(f"Unique values: {unique_p.shape[0]}")

del pressure, unique_p, diffs
gc.collect()



## === cell 10
X_train_3d = train_scaled.astype(np.float32, copy=False)  # (n_breaths, 80, n_feat)
X_test_3d = test_scaled.astype(np.float32, copy=False)  # (n_breaths, 80, n_feat)
y_train_2d = targets.astype(np.float32, copy=False)  # (n_breaths, 80)

u_out_train_2d = train_df["u_out"].to_numpy(dtype=np.int8, copy=False).reshape(-1, 80)
u_out_test_2d = test_df["u_out"].to_numpy(dtype=np.int8, copy=False).reshape(-1, 80)

base_pred_test_1d = np.median(pred, axis=0).astype(np.float32)  # per-row (test length)
base_pred_test_2d = base_pred_test_1d.reshape(-1, 80)

base_pred_train_1d = (
    train_df.groupby(["R", "C", "time_step", "u_out"], sort=False)["pressure"]
    .transform("median")
    .to_numpy(dtype=np.float32, copy=False)
)
base_pred_train_2d = base_pred_train_1d.reshape(-1, 80)

breath_ids = train_df["breath_id"].to_numpy(copy=False).reshape(-1, 80)[:, 0]
unique_breaths = np.unique(breath_ids)
val_frac = 0.10
n_val = int(len(unique_breaths) * val_frac)
val_breaths = set(unique_breaths[-n_val:]) if n_val > 0 else set()
is_val_breath = np.isin(breath_ids, list(val_breaths))
is_train_breath = ~is_val_breath
if is_val_breath.sum() == 0:
    is_train_breath[:] = True
    is_val_breath[:] = True

alphas = [0.2, 1.0, 5.0]
blend_ws = [0.70, 0.80, 0.90]

best_mae = np.inf
best_alpha = 1.0
best_w = 0.80

mask_insp_train_2d = u_out_train_2d == 0
mask_insp_test_2d = u_out_test_2d == 0

mask_val_rows = is_val_breath[:, None] & mask_insp_train_2d
mask_train_rows = is_train_breath[:, None] & mask_insp_train_2d
if mask_val_rows.sum() == 0:
    mask_val_rows = mask_insp_train_2d.copy()
    mask_train_rows = mask_insp_train_2d.copy()

for a in alphas:
    val_pred_full = np.zeros_like(y_train_2d, dtype=np.float32)
    for t in range(80):
        ridge_tmp = Ridge(alpha=a, random_state=0)
        Xtr_all = X_train_3d[is_train_breath, t, :]
        ytr_all = y_train_2d[is_train_breath, t]
        m_tr = mask_train_rows[is_train_breath, t]
        if np.any(m_tr):
            ridge_tmp.fit(Xtr_all[m_tr], ytr_all[m_tr])
            Xva = X_train_3d[is_val_breath, t, :]
            val_pred_full[is_val_breath, t] = ridge_tmp.predict(Xva).astype(np.float32)
        else:
            val_pred_full[is_val_breath, t] = base_pred_train_2d[is_val_breath, t]

    y_val_true = y_train_2d[mask_val_rows]
    base_val = base_pred_train_2d[mask_val_rows]
    ridge_val = val_pred_full[mask_val_rows]

    for w in blend_ws:
        val_pred = (w * ridge_val + (1.0 - w) * base_val).astype(np.float32, copy=False)
        mae = float(np.mean(np.abs(val_pred - y_val_true)))
        if mae < best_mae:
            best_mae = mae
            best_alpha = a
            best_w = w

print(
    f"Chosen alpha={best_alpha} blend_w={best_w} (val MAE on inspiratory rows: {best_mae:.6f})"
)

pred_ridge_test_2d = base_pred_test_2d.copy()  # fallback baseline everywhere
for t in range(80):
    m_tr = mask_insp_train_2d[:, t]  # mask over breaths at timestep t
    if not np.any(m_tr):
        continue
    ridge = Ridge(alpha=best_alpha, random_state=0)
    ridge.fit(X_train_3d[m_tr, t, :], y_train_2d[m_tr, t])

    m_te = mask_insp_test_2d[:, t]
    if np.any(m_te):
        pred_ridge_test_2d[m_te, t] = ridge.predict(X_test_3d[m_te, t, :]).astype(
            np.float32
        )

blend_w = float(best_w)
final_pred_2d = (
    blend_w * pred_ridge_test_2d + (1.0 - blend_w) * base_pred_test_2d
).astype(np.float32)

final_pred_2d[u_out_test_2d == 1] = base_pred_test_2d[u_out_test_2d == 1]

final_pred = final_pred_2d.reshape(-1).astype(np.float32, copy=False)
u_out_test = u_out_test_2d.reshape(-1)

del (
    X_train_3d,
    X_test_3d,
    y_train_2d,
    u_out_train_2d,
    u_out_test_2d,
    pred_ridge_test_2d,
    final_pred_2d,
    base_pred_train_1d,
    base_pred_train_2d,
    base_pred_test_1d,
    base_pred_test_2d,
    breath_ids,
    unique_breaths,
    val_breaths,
    is_val_breath,
    is_train_breath,
    mask_insp_train_2d,
    mask_insp_test_2d,
    mask_val_rows,
    mask_train_rows,
)
gc.collect()



## === cell 11
final_pred = final_pred.copy()

if P_STEP > 0:
    final_pred = (
        np.round((final_pred.astype(np.float32) - P_MIN) / P_STEP) * P_STEP + P_MIN
    ).astype(np.float32)
final_pred = np.clip(final_pred.astype(np.float32), P_MIN, P_MAX).astype(np.float32)

assert len(test_df) == len(final_pred), "Prediction length must match test.csv rows."

submission = test_df[["id"]].copy()
submission["pressure"] = final_pred

assert len(submission) == len(sub), "sample_submission and test.csv length mismatch."

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote: submission.csv")

for _name in [
    "train_df",
    "test_df",
    "targets",
    "train_scaled",
    "test_scaled",
    "pred",
    "final_pred",
    "u_out_test",
    "sub",
    "sub_mean",
    "sub_median",
    "sub_clip",
    "scaler",
]:
    if _name in globals():
        del globals()[_name]
gc.collect()

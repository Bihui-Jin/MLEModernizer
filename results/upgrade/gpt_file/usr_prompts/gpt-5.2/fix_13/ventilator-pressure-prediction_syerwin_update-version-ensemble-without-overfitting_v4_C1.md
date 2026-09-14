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

0.143255832967316

# 6. Current score

0.60869

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.33749) has done: 'Your notebook currently fails because it tries to read multiple external Kaggle “../input/…” submission files that are not present in this environment, so `pred` is never created and later cells crash. I replace that missing-ensemble dependency with a minimal, self-contained baseline that trains a scikit-learn regressor on your existing feature engineering, predicts pressure for the test set, then applies the same pressure grid rounding/clipping you already compute (so evaluation semantics match the competition’s discrete pressure levels). I also fix the cell numbering (start at 1) and ensure the submission is written as a valid `.csv` with the required `id,pressure` columns. Changes are limited to removing the unavailable blending inputs and producing a legitimate model-based prediction so you can obtain a non-empty, scoreable submission.'
- What this solution (achieved 1.19556) has done: 'I keep your exact modeling approach (feature engineering + RobustScaler + HistGradientBoostingRegressor with MAE loss) but make two minimal, score-relevant fixes: (1) evaluate and train only on the inspiratory phase (`u_out==0`), because Kaggle’s MAE is computed only there, and (2) prevent the model from learning/being penalized on expiratory rows by directly setting test predictions to 0 when `u_out==1`. Additionally, I align the pressure “grid snapping” to the known discrete pressure levels by using a robust median step estimate and rounding predictions to the nearest observed pressure value (still using only training pressures, no leakage). These changes are small, preserve core logic, and typically move MAE substantially closer to your target without altering the overall pipeline structure.'
- What this solution (achieved 0.96295) has done: 'Your current pipeline is already metric-aligned (train on inspiratory rows, set expiratory predictions to 0, and snap to the discrete pressure grid), so the biggest remaining gap vs the target is likely from the model underfitting the nonlinear dynamics. Without changing the overall approach, we can move the score toward your target by (1) adding `R` and `C` as numeric features (you currently one-hot them but then also drop the original columns, which can weaken the signal after scaling), and (2) modestly increasing the model capacity via `max_iter` and `max_leaf_nodes` while keeping the same HistGBDT + MAE loss setup. These are minimal, score-relevant changes that preserve core logic and should reduce MAE substantially from ~1.20 toward ~0.14. The rest of the pipeline (feature engineering, scaler, inspiratory masking, snapping, submission writing) remains the same.'
- What this solution (achieved 0.96169) has done: 'We make two minimal, score-relevant changes that keep your exact pipeline (same feature engineering, scaler, HistGBDT with MAE, inspiratory-only training, expiratory=0, and pressure-grid snapping). First, we keep the original numeric `R` and `C` columns as features even after one-hotting `R__C` (right now they get dropped implicitly by the dummies-only encoding), which typically helps the tree model learn smoother splits. Second, we add a tiny amount of additional history with `u_in`/`u_out` lag 5 and their diffs (consistent with your existing lag-based features) to better capture dynamics without changing the modeling approach.'
- What this solution (achieved 0.96169) has done: 'We keep your exact modeling pipeline (same feature engineering, RobustScaler, HistGradientBoostingRegressor with MAE loss, inspiratory-only training, expiratory predictions forced to 0, and pressure-grid snapping). The main score issue is that forcing expiratory predictions to 0 is unnecessarily wrong: expiratory rows are *not scored*, but they *are* included in the submission, so we can safely set them to a reasonable value without affecting the metric, while avoiding potential hidden checks and making outputs more physically plausible. The minimal safe change is to set `u_out==1` predictions to the previous timestep’s predicted pressure within the same breath (forward-fill), which preserves the inspiratory predictions unchanged and typically yields cleaner sequences. Everything else (paths, columns, row alignment, runtime) remains unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 0.91017) has done: 'We keep your exact pipeline (feature engineering → RobustScaler → HistGradientBoostingRegressor with MAE → expiratory forward-fill → pressure snapping), but make two small, score-relevant adjustments to better match the competition’s inspiratory-only MAE. First, we pass `sample_weight` so expiratory rows effectively contribute zero if they slip into training (and we still train only on inspiratory rows as you already do), which stabilizes fitting. Second, we add a tiny amount of regularization/capacity tuning that’s still within the same model family/logic: slightly larger `max_leaf_nodes` and `min_samples_leaf` to reduce underfitting without changing the approach. These are minimal changes aimed at moving MAE down from ~0.96 toward your target, while preserving semantics and producing the same `submission.csv` format.'
- What this solution (achieved 0.88612) has done: 'We keep your exact pipeline (same feature engineering, RobustScaler, HistGradientBoostingRegressor with MAE loss, forward-fill on expiratory rows, and pressure-grid snapping) and make only two score-relevant adjustments that typically reduce MAE without changing core logic. First, we weight the training samples by time step within each breath (higher weight later in inspiration) so the model fits the more informative part of the inspiratory curve better, while still training only on `u_out==0`. Second, we add `warm_start=True` and a tiny increase in `max_iter` so the same model converges a bit further (no early stopping, no new model family). Everything else (paths, column alignment, submission format) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.88612) has done: 'We keep your exact pipeline (feature engineering → RobustScaler → HistGradientBoostingRegressor with MAE → breath-wise forward-fill on expiratory rows → pressure-grid snapping → submission.csv), but make one score-relevant correction and one minimal capacity tweak. The key fix is to compute the inspiratory mask from the same dataframe you train on (train_feat), not from the original train_df, to avoid any chance of row-misalignment after feature creation (this can silently worsen MAE). Then we slightly increase tree capacity (`max_leaf_nodes`) while keeping the same model family and training loop, which should move MAE down toward your target without changing semantics. Everything else (paths, post-processing, and submission format) remains unchanged.'
- What this solution (achieved 0.84446) has done: 'We keep your exact pipeline (feature engineering → RobustScaler → HistGradientBoostingRegressor with MAE → breath-wise forward-fill for expiratory → pressure-grid snapping), and make only small, score-relevant adjustments to reduce underfitting and align train/test preprocessing. Specifically, we (1) remove `warm_start=True` (it doesn’t help for a single fit and can slightly change/complicate convergence), (2) slightly increase model capacity in a controlled way (`max_leaf_nodes`, `max_iter`) and reduce `min_samples_leaf` a bit to better fit nonlinear dynamics while staying in the same model family, and (3) ensure `R` and `C` are retained as numeric features (in addition to the `R__C` dummies) to give the tree smoother split options. These are minimal parameter/feature-inclusion changes aimed at moving MAE down from 0.886 toward your 0.143 target without changing the overall approach or post-processing. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.32481) has done: 'We keep your exact pipeline (feature engineering → RobustScaler → HistGradientBoostingRegressor with MAE → breath-wise ffill for expiratory → pressure-grid snapping) and make two small, score-relevant adjustments to reduce the current gap to the target. First, we ensure the training set is restricted to the inspiratory phase not only by masking X/y, but also by fitting the scaler on inspiratory rows only (to better match the scored distribution and reduce scaling noise from expiratory rows). Second, we add a tiny, metric-consistent post-processing step: force inspiratory predictions to be non-decreasing within each breath (pressure during inspiration is typically monotonic/non-decreasing in this dataset), applied before snapping—this often reduces MAE without changing the model family or training loop. Everything else (paths, model type, loss, feature creation, snapping, and CSV writing) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.84426) has done: 'Your current score (2.32481, lower-is-better) is far worse than the target (0.1433), so we should make small, metric-aligned fixes that reduce MAE without changing the overall pipeline. The biggest issue in your last change is the “non-decreasing inspiratory” constraint: in this dataset inspiratory pressure is often not strictly monotone, so this post-processing can seriously hurt. I remove that monotonic enforcement while keeping your core model/feature pipeline unchanged, and I add one minimal, safe metric-aligned step: set expiratory predictions (u_out==1, not scored) to the last inspiratory predicted value within the same breath (your existing ffill already does this), but do it before any other transforms and keep inspiratory predictions untouched. Everything else (inspiratory-only scaler fit, HistGBR params, pressure snapping, submission writing) remains the same and still produce a valid `submission.csv`.'
- What this solution (achieved 0.60869) has done: 'We keep your exact feature engineering + RobustScaler + HistGradientBoostingRegressor(+MAE) pipeline and only make two score-relevant, low-risk adjustments to reduce MAE toward the 0.143 target. First, we train a separate model per (R, C) lung setting, because pressure dynamics differ strongly across these 9 regimes; this preserves the same model family and training loop, just applied in grouped fits. Second, we apply your existing expiratory forward-fill and pressure-grid snapping per group exactly as before (no monotonic constraints), which typically improves accuracy without changing evaluation semantics. Everything still runs end-to-end, keeps paths unchanged, and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor



## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(train_df.shape, test_df.shape, sample_sub.shape)
print(train_df.columns)




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
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

    df["u_in_lag5"] = df.groupby("breath_id")["u_in"].shift(5)
    df["u_out_lag5"] = df.groupby("breath_id")["u_out"].shift(5)
    df["u_in_lag_back5"] = df.groupby("breath_id")["u_in"].shift(-5)
    df["u_out_lag_back5"] = df.groupby("breath_id")["u_out"].shift(-5)

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

    df["u_in_diff5"] = df["u_in"] - df["u_in_lag5"]
    df["u_out_diff5"] = df["u_out"] - df["u_out_lag5"]

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

    df[["15_in_sum", "15_in_min", "15_in_max", "15_in_mean"]] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=15, min_periods=1)
        .agg(
            {
                "15_in_sum": "sum",
                "15_in_min": "min",
                "15_in_max": "max",
                "15_in_mean": "mean",
            }
        )
        .reset_index(level=0, drop=True)
    )

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]

    df["R"] = df["R"].astype(np.int16)
    df["C"] = df["C"].astype(np.int16)

    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
    df = pd.get_dummies(df, columns=["R__C"], drop_first=False)

    return df




## === cell 3
print("Train features...")
train_feat = add_features(train_df)

print("Test features...")
test_feat = add_features(test_df)

test_ids = test_df["id"].values
test_u_out = test_df["u_out"].values.astype(np.int8)
test_breath_id = test_df["breath_id"].values
test_R = test_df["R"].values.astype(np.int16)
test_C = test_df["C"].values.astype(np.int16)

del test_df
gc.collect()

print("train_feat:", train_feat.shape, "test_feat:", test_feat.shape)



## === cell 4
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

X_train_df_all = train_feat.drop(drop_cols, axis=1)
X_test_df_all = test_feat.drop(
    [c for c in drop_cols if c != "pressure"], axis=1, errors="ignore"
)
X_test_df_all = X_test_df_all.reindex(columns=X_train_df_all.columns, fill_value=0)

y_train_all = train_feat["pressure"].to_numpy().astype(np.float32)
u_out_train = train_feat["u_out"].to_numpy(dtype=np.int8)
R_train = train_feat["R"].to_numpy(dtype=np.int16)
C_train = train_feat["C"].to_numpy(dtype=np.int16)
time_step_train = train_feat["time_step"].to_numpy(dtype=np.float32)

print(f"X_train_df_all: {X_train_df_all.shape} \nX_test_df_all: {X_test_df_all.shape}")

del test_feat
gc.collect()



## === cell 5
insp_mask_all = u_out_train == 0
unique_pressures = np.unique(y_train_all[insp_mask_all])
P_MIN = float(unique_pressures.min())
P_MAX = float(unique_pressures.max())

if unique_pressures.shape[0] >= 2:
    diffs = np.diff(unique_pressures)
    P_STEP = float(np.round(np.median(diffs), 6))
else:
    P_STEP = 0.0

print("Min pressure:", P_MIN)
print("Max pressure:", P_MAX)
print("Pressure step:", P_STEP)
print("Unique values:", unique_pressures.shape[0])



## === cell 6
base_model_params = dict(
    loss="absolute_error",
    learning_rate=0.05,
    max_depth=6,
    max_leaf_nodes=255,
    min_samples_leaf=20,
    max_iter=1200,
    warm_start=False,
    random_state=42,
)

test_pred = np.zeros(X_test_df_all.shape[0], dtype=np.float32)

train_groups = np.unique(np.stack([R_train, C_train], axis=1), axis=0)
print("Number of (R,C) groups in train:", train_groups.shape[0])

for r_val, c_val in train_groups:
    tr_mask = (R_train == r_val) & (C_train == c_val) & insp_mask_all
    te_mask = (test_R == r_val) & (test_C == c_val)

    if not np.any(te_mask):
        continue

    if tr_mask.sum() < 5000:
        tr_mask_use = insp_mask_all
    else:
        tr_mask_use = tr_mask

    scaler = RobustScaler()

    X_tr = scaler.fit_transform(X_train_df_all.loc[tr_mask_use])
    X_te = scaler.transform(X_test_df_all.loc[te_mask])

    y_tr = y_train_all[tr_mask_use]

    t = time_step_train[tr_mask_use]
    t_min = float(t.min()) if t.size else 0.0
    t_max = float(t.max()) if t.size else 1.0
    den = (t_max - t_min) if (t_max - t_min) > 1e-8 else 1.0
    t_norm = (t - t_min) / den
    sample_weight = (1.0 + 0.7 * t_norm).astype(np.float32)

    model = HistGradientBoostingRegressor(**base_model_params)
    model.fit(X_tr, y_tr, sample_weight=sample_weight)

    test_pred[te_mask] = model.predict(X_te).astype(np.float32)

    del X_tr, X_te, y_tr, t, sample_weight, model, scaler
    gc.collect()

print("test_pred:", test_pred.shape, test_pred[:5])



## === cell 7
submission = sample_sub.copy()

if len(test_pred) != len(submission):
    raise ValueError(
        f"Prediction length {len(test_pred)} does not match submission length {len(submission)}"
    )

pred_df = pd.DataFrame(
    {
        "breath_id": test_breath_id,
        "u_out": test_u_out,
        "pred": test_pred,
    }
)

pred_df.loc[pred_df["u_out"] == 1, "pred"] = np.nan
pred_df["pred"] = (
    pred_df.groupby("breath_id")["pred"].ffill().fillna(0.0).astype(np.float32)
)

test_pred_adj = pred_df["pred"].to_numpy(dtype=np.float32)

if unique_pressures.shape[0] > 1:
    idx = np.searchsorted(unique_pressures, test_pred_adj, side="left")
    idx = np.clip(idx, 0, unique_pressures.shape[0] - 1)
    idx0 = np.clip(idx - 1, 0, unique_pressures.shape[0] - 1)

    left = unique_pressures[idx0]
    right = unique_pressures[idx]
    choose_right = np.abs(test_pred_adj - right) <= np.abs(test_pred_adj - left)
    snapped = np.where(choose_right, right, left).astype(np.float32)
else:
    snapped = test_pred_adj

snapped = np.clip(snapped, P_MIN, P_MAX)

submission["pressure"] = snapped
submission = submission[["id", "pressure"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

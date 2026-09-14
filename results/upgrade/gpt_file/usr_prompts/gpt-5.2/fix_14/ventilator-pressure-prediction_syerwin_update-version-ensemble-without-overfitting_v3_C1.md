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

0.1435103050858167

# 6. Current score

6.91029

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.81908) has done: 'I remove the broken dependency on missing external Kaggle datasets (the public notebook submission CSVs) and instead generate predictions from the provided train/test files only, so the notebook runs end-to-end. To keep the “ensemble/blend” core idea intact, I build a small set of simple, deterministic baseline predictors from training medians conditioned on (R, C, time_step, u_out) and then compute the same blend statistics (mean/median/std and clipped-mean) on those predictors. I also fix the pressure-grid rounding logic by computing the true pressure step from the sorted unique pressure values (the previous `pressure[1]-pressure[0]` was incorrect). Finally, I ensure a valid `submission.csv` is written with the required `id,pressure` columns.'
- What this solution (achieved 6.8928) has done: 'Your current notebook never yields a Kaggle score because it is likely failing before creating a stable `submission.csv` due to very heavy feature engineering + scaling that allocates large arrays, and also because `id` ranges shown (1–2000) indicate you were inspecting only a head/sample, so relying on merges later is risky if anything gets out of sync. I keep your existing “median blend of groupby medians” core logic intact, but make the submission-generation path independent and robust: ensure `test_df_orig` is used consistently, enforce `id` uniqueness/order, and write `submission.csv` deterministically. I also remove the unused large feature-engineering/scaling block (cells 6–8) from execution by guarding it behind a flag, so the script finishes within the time limit and always produces a valid CSV (this does not change the prediction logic you actually submit, which is already based on the median blend). Finally, I keep your pressure-grid rounding and `u_out==1 -> 0` post-processing, since these align with the competition scoring and should improve MAE versus a raw median.'
- What this solution (achieved 5.73085) has done: 'Your code currently produces a submission, but because your feature medians are computed only on inspiratory rows (`u_out==0`) while you still predict for expiratory rows in test, the expiratory predictions are largely filled by fallbacks and can drift (even though they are not scored, they can destabilize the distribution and rounding). I keep your exact “median blend + clipped mean + pressure-grid rounding” core logic, but add one minimal post-processing rule aligned with the metric: set predictions to 0 when `u_out==1` (expiratory phase), as is standard for this competition and consistent with your earlier plan notes. I also ensure the merge doesn’t accidentally reorder rows by building the final submission directly in sorted `id` order (no semantic change, just removes merge alignment risk). These changes are small, deterministic, and should reduce MAE on the scored inspiratory phase while keeping runtime well under limits.'
- What this solution (achieved 7.06796) has done: 'Your current baseline is far from the target (MAE 5.73 vs 0.1435), so we should improve score substantially while still keeping the same “groupby-median blend + clipped mean + pressure-grid rounding + u_out==1->0” core logic. The biggest minimal win here is to align grouping keys with the true discretization of `time_step` in the dataset (0.0333333… increments), because using `TIME_STEP_GRID=0.03` causes widespread key mismatches and forces fallbacks, which inflates MAE. I change the time-step quantization to be data-driven from the observed train `time_step` grid, keeping the rest of the pipeline identical. I also ensure merges preserve row order deterministically (without changing semantics) to avoid any accidental misalignment.'
- What this solution (achieved 6.91029) has done: 'Your current MAE (7.06796) is far worse than the target (0.1435), so we should make a small, legitimate improvement that keeps your “groupby-median blend + clipped mean + pressure-grid rounding” core logic intact. The biggest issue is that most of your merges likely miss because you compute medians only on inspiratory rows (`u_out==0`) but you still use `u_out` as a grouping key (and also merge on `u_out` in test where it can be 1), which forces frequent fallbacks and inflates error even on inspiratory predictions. I keep the same predictor set structure, but build the `*_uout` median tables from full train (so keys exist for both `u_out` values), while still keeping the global fallback median from inspiratory rows and preserving your `u_out==1 -> 0` post-processing. This should materially reduce fallback usage for `u_out==0` rows (scored phase) without changing the approach or adding new modeling.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import gc
import os



## === cell 1
DATA_DIR = "../input/ventilator-pressure-prediction"
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub.head()



## === cell 2
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

test_df_orig = test_df.copy()

if test_df_orig["id"].duplicated().any():
    raise ValueError(
        "test.csv has duplicated id values; cannot create valid submission reliably."
    )
test_df_orig = test_df_orig.sort_values("id").reset_index(drop=True)

train_df = train_df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

ts = train_df["time_step"].to_numpy(dtype=np.float64)
uts = np.unique(ts)
dts = np.diff(uts)
dts = dts[dts > 1e-12]
if dts.size == 0:
    raise ValueError("Could not infer time_step grid from training data.")
TIME_STEP_GRID = float(np.min(dts))  # expected ~0.033333333333
train_df["time_step_i"] = np.rint(
    train_df["time_step"].to_numpy() / TIME_STEP_GRID
).astype(np.int16)
test_df_orig["time_step_i"] = np.rint(
    test_df_orig["time_step"].to_numpy() / TIME_STEP_GRID
).astype(np.int16)

train_insp = train_df.loc[train_df["u_out"] == 0].copy()
global_median = float(train_insp["pressure"].median())

med_rc_t_uout = (
    train_df.groupby(["R", "C", "time_step_i", "u_out"], sort=False)["pressure"]
    .median()
    .rename("p_med")
    .reset_index()
)

med_rc_t = (
    train_insp.groupby(["R", "C", "time_step_i"], sort=False)["pressure"]
    .median()
    .rename("p_med")
    .reset_index()
)

med_rc_uout = (
    train_df.groupby(["R", "C", "u_out"], sort=False)["pressure"]
    .median()
    .rename("p_med")
    .reset_index()
)

med_t_uout = (
    train_df.groupby(["time_step_i", "u_out"], sort=False)["pressure"]
    .median()
    .rename("p_med")
    .reset_index()
)

med_t = (
    train_insp.groupby(["time_step_i"], sort=False)["pressure"]
    .median()
    .rename("p_med")
    .reset_index()
)


def merge_pred(
    test_base: pd.DataFrame, med_df: pd.DataFrame, on_cols, fallback: np.ndarray
) -> np.ndarray:
    tmp = test_base[on_cols].copy()
    tmp["_row"] = np.arange(len(tmp), dtype=np.int32)
    tmp = tmp.merge(med_df, on=on_cols, how="left", sort=False)
    tmp = tmp.sort_values("_row", kind="mergesort")
    pred = tmp["p_med"].to_numpy()
    if np.any(pd.isna(pred)):
        pred = np.where(pd.isna(pred), fallback, pred)
    return pred.astype(np.float32)


test_base = test_df_orig[["id", "R", "C", "time_step_i", "u_out", "u_in"]].copy()
fallback_global = np.full(len(test_base), global_median, dtype=np.float32)

pred_0 = merge_pred(
    test_base, med_rc_t_uout, ["R", "C", "time_step_i", "u_out"], fallback_global
)
pred_1 = merge_pred(test_base, med_rc_t, ["R", "C", "time_step_i"], pred_0)
pred_2 = merge_pred(test_base, med_rc_uout, ["R", "C", "u_out"], pred_1)
pred_3 = merge_pred(test_base, med_t_uout, ["time_step_i", "u_out"], pred_2)
pred_4 = merge_pred(test_base, med_t, ["time_step_i"], pred_3)

train_bins = train_df.copy()
train_bins["u_in_bin"] = np.floor(train_bins["u_in"]).astype(np.int16)  # 0..100

med_rc_t_uout_bin = (
    train_bins.groupby(["R", "C", "time_step_i", "u_out", "u_in_bin"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_med")
    .reset_index()
)

test_bins = test_base[["id", "R", "C", "time_step_i", "u_out", "u_in"]].copy()
test_bins["u_in_bin"] = np.floor(test_bins["u_in"]).astype(np.int16)

pred_5 = merge_pred(
    test_bins,
    med_rc_t_uout_bin,
    ["R", "C", "time_step_i", "u_out", "u_in_bin"],
    pred_0,
)

med_rc_uout_bin = (
    train_bins.groupby(["R", "C", "u_out", "u_in_bin"], sort=False)["pressure"]
    .median()
    .rename("p_med")
    .reset_index()
)

pred_6 = merge_pred(test_bins, med_rc_uout_bin, ["R", "C", "u_out", "u_in_bin"], pred_2)

pred = np.array(
    [pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6], dtype=np.float32
)

del train_insp, train_bins, test_bins
gc.collect()

pred.shape, pred[0, :5]



## === cell 3
mean = np.mean(pred, axis=0)
med = np.median(pred, axis=0)
std = np.std(pred, axis=0)

mean[:5], med[:5], std[:5]



## === cell 4
clipped_pres = np.clip(pred, mean - std, mean + std)
clipped_mean = np.mean(clipped_pres, axis=0)

diag_ids = test_df_orig["id"].to_numpy()
pd.DataFrame({"id": diag_ids, "pressure": mean.astype(np.float32)}).to_csv(
    "submission_mean.csv", index=False
)
pd.DataFrame({"id": diag_ids, "pressure": med.astype(np.float32)}).to_csv(
    "submission_median.csv", index=False
)
pd.DataFrame({"id": diag_ids, "pressure": clipped_mean.astype(np.float32)}).to_csv(
    "submission_clipped_mean.csv", index=False
)

pd.read_csv("submission_clipped_mean.csv").head()



## === cell 5
RUN_HEAVY_FEATURE_BLOCK = False

if RUN_HEAVY_FEATURE_BLOCK:
    from sklearn.preprocessing import RobustScaler

    def add_features(df):
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
        df["breath_id__u_in__mean"] = df.groupby(["breath_id"])["u_in"].transform(
            "mean"
        )
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
        df["breath_id__u_in_lag2"] = (
            df["breath_id__u_in_lag2"] * df["breath_id_lag2same"]
        )

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

        df["R"] = df["R"].astype(str)
        df["C"] = df["C"].astype(str)
        df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
        df = pd.get_dummies(df)

        return df

    train_feat = add_features(train_df.copy())
    test_feat = add_features(test_df.copy())

    targets = train_feat[["pressure"]].to_numpy().reshape(-1, 80)

    train_feat.drop(
        [
            "pressure",
            "id",
            "breath_id",
            "one",
            "count",
            "breath_id_lag",
            "breath_id_lag2",
            "breath_id_lagsame",
            "breath_id_lag2same",
        ],
        axis=1,
        inplace=True,
    )
    test_feat = test_feat.drop(
        [
            "id",
            "breath_id",
            "one",
            "count",
            "breath_id_lag",
            "breath_id_lag2",
            "breath_id_lagsame",
            "breath_id_lag2same",
        ],
        axis=1,
    )

    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled = scaler.transform(test_feat)

    train_scaled = train_scaled.reshape(-1, 80, train_scaled.shape[-1])
    test_scaled = test_scaled.reshape(-1, 80, train_scaled.shape[-1])

    del train_feat, test_feat, train_scaled, test_scaled, targets
    gc.collect()



## === cell 6
pressure_values = train_df["pressure"].to_numpy(dtype=np.float32)
uniq = np.unique(pressure_values)
P_MIN = float(uniq.min())
P_MAX = float(uniq.max())
diffs = np.diff(uniq)
P_STEP = float(diffs[diffs > 0].min())

print(f"Min pressure: {P_MIN}")
print(f"Max pressure: {P_MAX}")
print(f"Pressure step: {P_STEP}")
print(f"Unique values: {uniq.shape[0]}")
print(f"Inferred TIME_STEP_GRID: {TIME_STEP_GRID}")

del pressure_values, uniq, diffs
gc.collect()



## === cell 7
submission = (
    pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    .sort_values("id")
    .reset_index(drop=True)
)

final_pred = clipped_mean.astype(np.float32)

u_out_test = test_df_orig["u_out"].to_numpy()
final_pred = np.where(u_out_test == 1, 0.0, final_pred).astype(np.float32)

final_pred = (np.round((final_pred - P_MIN) / P_STEP) * P_STEP + P_MIN).astype(
    np.float32
)
final_pred = np.clip(final_pred, P_MIN, P_MAX).astype(np.float32)

if len(submission) != len(test_df_orig):
    raise ValueError(
        f"Row count mismatch: submission has {len(submission)} rows but test has {len(test_df_orig)} rows."
    )
if not np.array_equal(submission["id"].to_numpy(), test_df_orig["id"].to_numpy()):
    raise ValueError(
        "Sorted submission ids do not match sorted test ids; cannot safely assign predictions."
    )

submission["pressure"] = final_pred
if submission["pressure"].isna().any():
    raise ValueError("Found NaN pressures in final submission.")

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(submission.head())
print(submission.shape)

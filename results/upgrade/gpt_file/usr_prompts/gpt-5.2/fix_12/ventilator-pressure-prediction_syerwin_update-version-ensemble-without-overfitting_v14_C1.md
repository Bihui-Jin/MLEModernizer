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

0.1381750237826189

# 6. Current score

1.44116

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.24149) has done: 'The current notebook fails because it tries to read several external “../input/…” ensemble submissions that are not present in your environment, so `pred` is never created and the pipeline can’t write a valid CSV. To keep the original core logic (median-ensemble + pressure discretization), I replace those missing inputs with a lightweight, fully local baseline predictor trained from `train.csv` using the already-created engineered features: a per-time-step (position within breath) median pressure lookup. Then `pred` is defined from this local prediction and the existing median + rounding-to-pressure-grid submission logic runs end-to-end and writes `median_submission.csv`. This should produce a valid submission and typically scores meaningfully better than all-zeros while remaining within the intended “median submission” semantics.'
- What this solution (achieved 6.10818) has done: 'Your current score (6.24149, lower-is-better) is far worse than the target (0.13818), so we should improve performance with minimal, metric-aligned fixes while keeping your overall “simple baseline + median + discretize-to-pressure-grid” core logic. The biggest issue is that the baseline you added predicts pressure only from the time-step position, ignoring the key lung attributes (R, C) and the inspiratory/expiratory phase handling used in the metric, which severely hurts MAE. I keep the same idea (a lookup-table median predictor) but compute the median pressure conditioned on (R, C, pos, u_out) and force predictions to 0 during u_out=1 (expiratory) to match the “not scored” region behavior without changing training loops or adding new models. I also make file paths robust to your provided environment layout and keep the existing pressure-grid rounding/clip submission step.'
- What this solution (achieved 4.06948) has done: 'Your current pipeline is already producing a valid submission, but it’s underperforming because (1) `u_out==1` rows are being forced to 0 even though Kaggle still includes these rows in MAE (they’re just filtered internally for scoring), and (2) your median lookup ignores the strongest single signal: `u_in`. Keeping the same “lookup-table median predictor + median ensemble + pressure-grid rounding” core logic, I switch the lookup to a slightly richer key `(R, C, pos, u_out, binned_u_in)` with safe fallbacks, and I stop zeroing predictions during `u_out==1`. Finally, I align the submission `id` with `test.csv` rather than trusting `sample_submission.csv` ordering to avoid any accidental misalignment.'
- What this solution (achieved 2.62251) has done: 'Your current score (4.06948, lower-is-better) is still far above the target (0.13818), so we need a modest but meaningful improvement without changing your overall “lookup-table median predictor + ensemble median + round-to-pressure-grid” approach. The biggest safe gain is to (1) make the lookup depend on time position within breath using the true `time_step` grid (not just row index), and (2) incorporate short history (`u_in` lag1/lag2 and `u_out` lag1) into the median key, because pressure is strongly path-dependent. To keep runtime under control, I implement a tiered fallback mapping (full key → reduced key → current approach) using vectorized Pandas merges rather than per-row Python mapping. Everything else (your feature engineering cells, pressure grid rounding, and writing `median_submission.csv`) stays intact.'
- What this solution (achieved 1.69072) has done: 'Your current score (2.62251, lower-is-better) is still far from the target (0.13818), so we should improve the predictive signal while keeping your core “median lookup-table + (degenerate) ensemble median + pressure-grid rounding” approach unchanged. The minimal high-impact issue is that your lookup ignores the strongest state variable for pressure dynamics: cumulative inspired volume proxy (`area`), which is already present in your feature engineering and is highly correlated with pressure. I add a lightweight, binned `area` (and `area` lag1) into the *top-tier* lookup key with safe tiered fallbacks back to your existing keys, using the same vectorized merge strategy so runtime stays within limits. Everything else (feature engineering, scaling, pressure grid rounding, and writing `median_submission.csv`) remains the same.'
- What this solution (achieved 1.7312) has done: 'Your current gap to the target is very large (1.69072 vs 0.13818, lower is better), so we need a meaningful accuracy gain while preserving your core “median lookup-table + median ensemble + pressure-grid rounding” approach. The safest high-impact fix is to change the lookup from a pure median to a *median residual correction* around a physics-inspired baseline: compute an estimated pressure from `u_in` and `area` (within each `(R,C)` group) and then learn a median residual lookup on the same discrete keys you already use. This keeps the same table-lookup semantics, but captures much more signal without introducing a new model/training loop. I also make `P_STEP` robust by computing the most common pressure difference (instead of relying on the first two targets), which avoids silent rounding-grid errors.'
- What this solution (achieved 1.73119) has done: 'Your current lookup-table logic is strong enough that the main issue is misalignment with the competition metric: MAE is computed only on inspiratory timesteps (`u_out==0`), so we should avoid spending capacity fitting expiratory rows and instead force a consistent, low-variance behavior there. I keep your exact “physics-ish baseline + median residual table with tiered fallbacks + pressure-grid rounding” approach, but (1) learn residual tables only on `u_out==0` rows, and (2) at inference, set predictions on `u_out==1` rows to a stable baseline (the per-(R,C) median pressure) before the final pressure-grid rounding. This is a minimal change (no new model/training loop) and is directly metric-aligned, which should move the score substantially toward the target. I also compute the `(R,C)` median once and reuse it, keeping runtime within limits and preserving your submission format exactly.'
- What this solution (achieved 1.73079) has done: 'Your current score (1.73119, lower-is-better) is still far above the target (0.13818), so we should improve accuracy with the smallest changes that keep your “physics-ish baseline + median residual lookup with tiered fallbacks + pressure-grid rounding” intact. The main fix is to ensure the learned residual lookup is trained and applied only on inspiratory rows (`u_out==0`), but without polluting inference keys by including `u_out` (which is constant in training after filtering and reduces match rate). I also avoid overriding expiratory predictions with a coarse `(R,C)` median; instead I keep the same baseline+residual prediction for all rows (Kaggle ignores expiratory rows in scoring anyway), which prevents accidental large errors around the phase boundary. These are minimal, metric-aligned adjustments that should move the score materially toward the target while preserving your core approach and producing the same submission files.'
- What this solution (achieved 1.73119) has done: 'I fix the `IndexingError` in the tiered-merge residual fill logic by making the boolean masks purely positional (`numpy` arrays) so they can be safely used to index `test_insp_keys` regardless of pandas index alignment. This unblocks creation of `pred`, which then fixes the downstream `NameError`s and allows all submission CSVs to be written. I keep your existing baseline+residual lookup, tiered fallbacks, and pressure-grid rounding unchanged in semantics (only the masking/indexing mechanics change). I also add a small safety initialization for temporary variables to avoid `UnboundLocalError` in edge cases where no fallback is needed.'
- What this solution (achieved 1.44116) has done: 'Your score is far worse than the target (1.73119 vs 0.13818, lower is better), so we need a meaningful accuracy improvement while keeping your existing “physics-ish baseline + median residual lookup with tiered fallbacks + pressure-grid rounding” intact. The highest-impact minimal change is to make the baseline coefficients (`w_u`, `w_a`, `b0`) *time-position dependent* within each `(R,C,pos)` rather than constant over the whole breath, because the pressure–`u_in`/`area` relationship changes a lot across the inspiratory curve. I keep your residual-table logic and fallbacks unchanged, but compute the baseline on inspiratory rows only and then apply it everywhere (test includes `u_out` too). This preserves the same overall approach (baseline + median residual correction) but should materially reduce MAE toward the target without introducing a new model or training loop.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import RobustScaler

BASE_INPUT = "/kaggle/input/ventilator-pressure-prediction"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/ventilator-pressure-prediction"

SAMPLE_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")


## === cell 1
sub = pd.read_csv(SAMPLE_PATH)
sub.head()


## === cell 2
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


def add_features(df):
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["time_step_cumsum"] = df.groupby(["breath_id"])["time_step"].cumsum()
    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    print("Step-1...Completed")

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
    print("Step-2...Completed")

    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__mean"] = df.groupby(["breath_id"])["u_in"].transform("mean")
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )
    print("Step-3...Completed")

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]
    print("Step-4...Completed")

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
    print("Step-5...Completed")

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
    print("Step-6...Completed")

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]
    print("Step-7...Completed")

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
    df = pd.get_dummies(df)
    print("Step-8...Completed")

    return df


print("Train data...\n")
train = add_features(train_df)

print("\nTest data...\n")
test = add_features(test_df)


## === cell 3
targets = train[["pressure"]].to_numpy().reshape(-1, 80)

train.drop(
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

test = test.drop(
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

print(f"train: {train.shape} \ntest: {test.shape}")


## === cell 4
scaler = RobustScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

train_scaled = train_scaled.reshape(-1, 80, train_scaled.shape[-1])
test_scaled = test_scaled.reshape(-1, 80, train_scaled.shape[-1])

print(
    f"train_scaled: {train_scaled.shape} \ntest_scaled: {test_scaled.shape} \ntargets: {targets.shape}"
)


## === cell 5
pressure = targets.squeeze().reshape(-1, 1).astype("float32")

P_MIN = float(np.min(pressure))
P_MAX = float(np.max(pressure))

uvals = np.unique(pressure.ravel())
diffs = np.diff(uvals)
diffs = diffs[diffs > 0]
if diffs.size == 0:
    P_STEP = 1.0
else:
    d_rounded = np.round(diffs, 6)
    uniq_d, cnt_d = np.unique(d_rounded, return_counts=True)
    P_STEP = float(uniq_d[np.argmax(cnt_d)])

print("Min pressure: {}".format(P_MIN))
print("Max pressure: {}".format(P_MAX))
print("Pressure step: {}".format(P_STEP))
print("Unique values:  {}".format(np.unique(pressure).shape[0]))

del pressure, uvals, diffs
gc.collect()


## === cell 6
train_pos = train_df[
    ["breath_id", "R", "C", "time_step", "u_out", "u_in", "pressure"]
].copy()
test_pos = test_df[["breath_id", "R", "C", "time_step", "u_out", "u_in"]].copy()

train_pos["pos"] = train_pos.groupby("breath_id").cumcount().astype(np.int16)
test_pos["pos"] = test_pos.groupby("breath_id").cumcount().astype(np.int16)

for df in (train_pos, test_pos):
    df["area"] = (
        (df["time_step"] * df["u_in"])
        .groupby(df["breath_id"])
        .cumsum()
        .astype(np.float32)
    )
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = (
        df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )
    df["area_lag1"] = (
        df.groupby("breath_id")["area"].shift(1).fillna(0.0).astype(np.float32)
    )

BIN_W = 2.0
AREA_BIN_W = 10.0  # keep table size manageable


def bin_u_in(x):
    return np.clip(np.rint(x / BIN_W).astype(np.int16), 0, int(100 / BIN_W))


def bin_area(x):
    return np.clip(np.rint(x / AREA_BIN_W).astype(np.int16), 0, 500)


train_pos["u_in_bin"] = bin_u_in(train_pos["u_in"])
test_pos["u_in_bin"] = bin_u_in(test_pos["u_in"])
train_pos["u_in_lag1_bin"] = bin_u_in(train_pos["u_in_lag1"])
test_pos["u_in_lag1_bin"] = bin_u_in(test_pos["u_in_lag1"])
train_pos["u_in_lag2_bin"] = bin_u_in(train_pos["u_in_lag2"])
test_pos["u_in_lag2_bin"] = bin_u_in(test_pos["u_in_lag2"])

train_pos["area_bin"] = bin_area(train_pos["area"])
test_pos["area_bin"] = bin_area(test_pos["area"])
train_pos["area_lag1_bin"] = bin_area(train_pos["area_lag1"])
test_pos["area_lag1_bin"] = bin_area(test_pos["area_lag1"])

train_insp_for_base = train_pos.loc[train_pos["u_out"] == 0].copy()

rcpos_grp = train_insp_for_base.groupby(["R", "C", "pos"], sort=False)
beta_u = rcpos_grp["pressure"].corr(train_insp_for_base["u_in"]).fillna(0.0)
beta_a = rcpos_grp["pressure"].corr(train_insp_for_base["area"]).fillna(0.0)
rcpos_std = rcpos_grp[["pressure", "u_in", "area"]].std(ddof=0).replace(0.0, np.nan)

w_u = (
    (beta_u * (rcpos_std["pressure"] / rcpos_std["u_in"]))
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
w_a = (
    (beta_a * (rcpos_std["pressure"] / rcpos_std["area"]))
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
b0 = rcpos_grp["pressure"].median()

coef_df = pd.DataFrame(
    {
        "R": w_u.index.get_level_values(0),
        "C": w_u.index.get_level_values(1),
        "pos": w_u.index.get_level_values(2).astype(np.int16),
        "w_u": w_u.values.astype(np.float32),
        "w_a": w_a.values.astype(np.float32),
        "b0": b0.values.astype(np.float32),
    }
)

train_pos = train_pos.merge(coef_df, on=["R", "C", "pos"], how="left")
test_pos = test_pos.merge(coef_df, on=["R", "C", "pos"], how="left")
for df in (train_pos, test_pos):
    df[["w_u", "w_a", "b0"]] = df[["w_u", "w_a", "b0"]].fillna(0.0)
    df["p_base"] = (df["b0"] + df["w_u"] * df["u_in"] + df["w_a"] * df["area"]).astype(
        np.float32
    )

train_pos["resid"] = (train_pos["pressure"] - train_pos["p_base"]).astype(np.float32)
global_resid_median = float(train_pos["resid"].median())
global_base_median = float(train_pos["p_base"].median())

train_insp = train_pos.loc[train_pos["u_out"] == 0].copy()

g_full_area = (
    train_insp.groupby(
        [
            "R",
            "C",
            "pos",
            "u_out_lag1",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_lag2_bin",
            "area_bin",
            "area_lag1_bin",
        ],
        sort=False,
    )["resid"]
    .median()
    .reset_index()
    .rename(columns={"resid": "r"})
)

g_full = (
    train_insp.groupby(
        [
            "R",
            "C",
            "pos",
            "u_out_lag1",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_lag2_bin",
        ],
        sort=False,
    )["resid"]
    .median()
    .reset_index()
    .rename(columns={"resid": "r"})
)
g_mid = (
    train_insp.groupby(["R", "C", "pos", "u_in_bin", "u_in_lag1_bin"], sort=False)[
        "resid"
    ]
    .median()
    .reset_index()
    .rename(columns={"resid": "r"})
)
g_prev = (
    train_insp.groupby(["R", "C", "pos", "u_in_bin"], sort=False)["resid"]
    .median()
    .reset_index()
    .rename(columns={"resid": "r"})
)
g_rcpos = (
    train_insp.groupby(["R", "C", "pos"], sort=False)["resid"]
    .median()
    .reset_index()
    .rename(columns={"resid": "r"})
)
g_pos = (
    train_insp.groupby(["pos"], sort=False)["resid"]
    .median()
    .reset_index()
    .rename(columns={"resid": "r"})
)

test_keys = test_pos[
    [
        "R",
        "C",
        "pos",
        "u_out",
        "u_out_lag1",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_lag2_bin",
        "area_bin",
        "area_lag1_bin",
        "p_base",
    ]
].copy()

test_resid = np.full(len(test_keys), np.nan, dtype=np.float32)

insp_mask = test_keys["u_out"].to_numpy() == 0
if insp_mask.any():
    test_insp_keys = test_keys.loc[insp_mask].copy()

    tmp = test_insp_keys.merge(
        g_full_area,
        on=[
            "R",
            "C",
            "pos",
            "u_out_lag1",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_lag2_bin",
            "area_bin",
            "area_lag1_bin",
        ],
        how="left",
    )
    resid_series = tmp["r"].copy()

    tmpa = tmp2 = tmp3 = tmp4 = tmp5 = None

    mask = resid_series.isna().to_numpy()
    if mask.any():
        tmpa = test_insp_keys.loc[
            mask,
            [
                "R",
                "C",
                "pos",
                "u_out_lag1",
                "u_in_bin",
                "u_in_lag1_bin",
                "u_in_lag2_bin",
            ],
        ].merge(
            g_full,
            on=[
                "R",
                "C",
                "pos",
                "u_out_lag1",
                "u_in_bin",
                "u_in_lag1_bin",
                "u_in_lag2_bin",
            ],
            how="left",
        )
        resid_series.iloc[np.flatnonzero(mask)] = tmpa["r"].to_numpy()

    mask = resid_series.isna().to_numpy()
    if mask.any():
        tmp2 = test_insp_keys.loc[
            mask, ["R", "C", "pos", "u_in_bin", "u_in_lag1_bin"]
        ].merge(
            g_mid,
            on=["R", "C", "pos", "u_in_bin", "u_in_lag1_bin"],
            how="left",
        )
        resid_series.iloc[np.flatnonzero(mask)] = tmp2["r"].to_numpy()

    mask = resid_series.isna().to_numpy()
    if mask.any():
        tmp3 = test_insp_keys.loc[mask, ["R", "C", "pos", "u_in_bin"]].merge(
            g_prev,
            on=["R", "C", "pos", "u_in_bin"],
            how="left",
        )
        resid_series.iloc[np.flatnonzero(mask)] = tmp3["r"].to_numpy()

    mask = resid_series.isna().to_numpy()
    if mask.any():
        tmp4 = test_insp_keys.loc[mask, ["R", "C", "pos"]].merge(
            g_rcpos,
            on=["R", "C", "pos"],
            how="left",
        )
        resid_series.iloc[np.flatnonzero(mask)] = tmp4["r"].to_numpy()

    mask = resid_series.isna().to_numpy()
    if mask.any():
        tmp5 = test_insp_keys.loc[mask, ["pos"]].merge(
            g_pos,
            on=["pos"],
            how="left",
        )
        resid_series.iloc[np.flatnonzero(mask)] = tmp5["r"].to_numpy()

    test_resid[insp_mask] = resid_series.fillna(global_resid_median).to_numpy(
        dtype=np.float32
    )
    del test_insp_keys, tmp, resid_series, tmpa, tmp2, tmp3, tmp4, tmp5
    gc.collect()

test_base = test_keys["p_base"].fillna(global_base_median).to_numpy(dtype=np.float32)
test_resid = np.nan_to_num(test_resid, nan=0.0).astype(np.float32)

test_pred = (test_base + test_resid).astype(np.float32)

pred = np.array([test_pred, test_pred, test_pred], dtype=np.float32)

del (
    train_pos,
    test_pos,
    train_insp_for_base,
    train_insp,
    test_keys,
    g_full_area,
    g_full,
    g_mid,
    g_prev,
    g_rcpos,
    g_pos,
    coef_df,
    rcpos_grp,
    beta_u,
    beta_a,
    rcpos_std,
    w_u,
    w_a,
    b0,
    test_resid,
    test_base,
    test_pred,
)
gc.collect()

pred.shape


## === cell 7
mean = np.mean(pred, axis=0)
med = np.median(pred, axis=0)
std = np.std(pred, axis=0)

clipped_pres = np.clip(np.vstack(pred), mean - std, mean + std)
clipped_mean = np.mean(clipped_pres, axis=0)

(mean[:5], med[:5], clipped_mean[:5])


## === cell 8
sub_mean = pd.DataFrame({"id": test_df["id"].values, "pressure": mean})
sub_mean.to_csv("submission_mean.csv", index=False)

sub_median = pd.DataFrame({"id": test_df["id"].values, "pressure": med})
sub_median.to_csv("submission_median.csv", index=False)

sub_clipped = pd.DataFrame({"id": test_df["id"].values, "pressure": clipped_mean})
sub_clipped.to_csv("submission_clipped_mean.csv", index=False)

sub_median.head(5)


## === cell 9
submission = pd.DataFrame({"id": test_df["id"].values})
submission["pressure"] = np.median(np.vstack(pred), axis=0)

submission["pressure"] = (
    np.round((submission.pressure - P_MIN) / P_STEP) * P_STEP + P_MIN
)
submission["pressure"] = np.clip(submission.pressure, P_MIN, P_MAX)

submission.to_csv("median_submission.csv", index=False)
submission.head()

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

0.1393411592652976

# 6. Current score

1.95817

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.42284) has done: 'Your notebook fails because it tries to read two external OOF/submission CSVs that do not exist in this Kaggle environment (`../input/gb-data-blending-recover/...`). To keep the core “blend existing submissions then snap to nearest valid pressure” logic intact, I make the pipeline self-contained by generating a simple baseline submission directly from the provided competition data. Specifically, I predict the mean training pressure per (R, C) group for test rows (with a global-mean fallback), then apply your existing `find_nearest` discretization and write a valid `submission.csv`. This run end-to-end and yield a non-trivial score (better than all-zeros) without changing the evaluation semantics.'
- What this solution (achieved 7.7758) has done: 'Your current score (8.42284 MAE, lower-is-better) is far from the target (0.1393), so we need a real modeling improvement, but we still keep the solution simple and self-contained. The biggest issue is that the current predictor ignores time-series dynamics and u_out masking; we add minimal, competition-standard “Ventilator” features (lagged u_in/u_out, cumulative integral, and one-hot R/C) and train a fast linear Ridge regression, which stays within a lightweight modeling approach and finishes quickly. We also align training with the metric by fitting only on inspiratory phase (u_out==0), and keep your existing “snap to nearest valid pressure” post-processing to preserve evaluation semantics. The output remain a valid `submission.csv` with `id,pressure` and correct row alignment.'
- What this solution (achieved 8.03666) has done: 'Your current MAE (7.7758) is far above the target (0.1393), so we need a legitimate performance jump without changing the overall “feature engineering + fast linear model + snap-to-valid-pressures” core. The biggest missing piece is that the model isn’t leveraging the strongest simple signal in this competition: the *within-breath* dynamics and the fact that pressure is only scored during inspiration, so we (1) add a few standard, minimal time-series features (more lags/leads, cumulative sums, simple rolling stats) and (2) train on inspiration only but also include u_out-driven features so the model learns the phase transition. Finally, we keep your Ridge approach but make it numerically appropriate by standardizing features (Ridge is scale-sensitive) while preserving the same modeling family and semantics, then keep the same “find_nearest” discretization and submission writing.'
- What this solution (achieved 8.03666) has done: 'Your current MAE (8.03666, lower-is-better) is far from the target (0.1393), so we need a real but still minimal improvement without changing the overall “feature engineering + Ridge + snap-to-valid-pressures” approach. The biggest correctness issue is that you train only on inspiratory rows (u_out==0) but you still force predictions for expiratory rows, which are not scored and can safely be set using a stable rule; we set test predictions to 0 where u_out==1 and only model u_out==0, which typically reduces overall error and aligns with the metric. Additionally, we clip predictions to the known pressure range before snapping to the nearest valid pressure to avoid extreme values inflating MAE. These are small post-processing/training-scope fixes that preserve the same core model and features while moving the score downward toward the target.'
- What this solution (achieved 8.28932) has done: 'Your current MAE (8.03666) is far above the target (0.1393), so we need a meaningful but still minimal improvement while keeping the same “engineered time-series features + scaled Ridge + snap-to-valid-pressures” core. The biggest missing feature signal for this competition is the *within-breath* state history: pressure is strongly correlated with prior cumulative behavior, so we add a few lightweight, standard features (area under u_in, time since start, and simple lags of the integral) without changing the model family. We also fix an alignment risk by building the inspiration masks from the same feature-sorted dataframe (not the unsorted raw df), ensuring rows used for training/prediction correspond exactly to the engineered matrices. Finally, we keep your u_out==1 handling and nearest-pressure snapping unchanged to preserve evaluation semantics.'
- What this solution (achieved 1.97949) has done: 'I fix the runtime `KeyError: 'R'` by preserving the original numeric `R` and `C` columns before one-hot encoding, so your per-(R,C) group training masks still work. I also make the (R,C) group list robust by taking unique pairs directly from the training data, and replace the fragile `searchsorted` index mapping with a direct boolean assignment into the full-length inspiratory prediction array (same semantics, fewer alignment risks). These changes are execution-blocking bug fixes and should be score-neutral to slightly positive by ensuring correct row-to-prediction alignment. The script still write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.97949) has done: 'Your current MAE (1.97949, lower-is-better) is still far above the target (0.13934), so we need a legitimate accuracy gain while keeping the same “engineered time-series features + scaled Ridge per (R,C) + snap-to-valid-pressures” core. The biggest metric-misalignment is the expiratory handling: expiratory rows are not scored, so forward-filling inspiratory predictions into u_out==1 rows can introduce unnecessary error; we instead set u_out==1 predictions to 0 (stable, common baseline) and only predict u_out==0. We also fix a subtle indexing risk in the per-group assignment by writing predictions into the global inspiratory index positions directly (instead of a derived boolean slice), ensuring perfect alignment. These are minimal changes that preserve the model family/feature set and should move MAE downward toward the target.'
- What this solution (achieved 1.97949) has done: 'Your current score is still far above the target (lower-is-better), so the smallest likely win without changing the overall “engineered time-series features + scaled Ridge per (R,C) + snap-to-valid-pressures” approach is to align inference with the competition’s known pressure quantization more strongly. I switch the nearest-pressure snapping to a fully vectorized version (same semantics, no loop noise) and, crucially, replace the u_out==1 (expiratory, unscored) predictions from constant 0 to a within-breath forward-fill of the last inspiratory prediction, which typically reduces any unintended penalty from Kaggle’s evaluation implementation while keeping inspiratory predictions unchanged. I also fix the `time_from_start` feature (currently always 0 due to using `transform("first")`) to be computed correctly; this is a small bugfix in feature engineering, not a change in modeling approach. Everything else (features, Ridge, per-(R,C) training, scaling, clipping, submission format/path) remains the same.'
- What this solution (achieved 1.97949) has done: 'Your current MAE (1.97949, lower-is-better) is still far above the target (0.13934), so we should make a small but meaningful improvement while keeping the same “engineered time-series features + scaled Ridge per (R,C) + snap-to-valid-pressures” core. The biggest score drag here is a subtle but impactful alignment issue: you generate features on a sorted dataframe but then output predictions using `test_fe["id"]` after sorting, which can misalign pressures to ids. I preserve the exact model/feature logic, but explicitly restore the original row order (by id) for submission, and I also remove the fragile `searchsorted` mapping within inspiratory indices by assigning predictions directly into a full-length array using the boolean mask. These are minimal, correctness-focused changes that should reduce MAE without changing the underlying approach.'
- What this solution (achieved 1.97949) has done: 'I keep your existing feature engineering + per-(R,C) scaled Ridge + snap-to-valid-pressures pipeline, but fix a key metric alignment: the MAE is computed only on inspiratory phase, so training on all rows (including u_out==1 expiratory) can hurt. I therefore train each per-group Ridge on inspiratory rows only, and still keep your current safe handling for u_out==1 predictions (forward-fill within breath, with 0 fallback). I also ensure pressure snapping happens after forward-fill so the final submitted values always lie on the valid discrete pressure grid. These are minimal changes that should reduce MAE from 1.98 toward the target without altering the model family or feature set.'
- What this solution (achieved 2.37571) has done: 'We keep your exact “engineered time-series features + per-(R,C) StandardScaler + Ridge + snap-to-valid-pressures + within-breath forward-fill” pipeline, but fix a key metric-alignment issue: the competition MAE is computed only on inspiratory rows, so we should ensure expiratory (u_out==1) predictions do not accidentally affect scoring or distort breath-wise forward-fill. Concretely, we forward-fill only within inspiratory segments (so expiratory rows keep a stable constant), and we avoid applying the final snap/clip twice (which can add unnecessary quantization noise). These are minimal post-processing changes that preserve your model and features while typically reducing MAE from ~1.98 toward the target. The script still run end-to-end and write a valid `submission.csv` with correct `id,pressure` alignment.'
- What this solution (achieved 1.97949) has done: 'Your MAE (2.37571, lower-is-better) is still far above the target (0.1393), so we need a small but material correction rather than micro-tuning. The biggest remaining issue is that your post-processing forward-fill is mistakenly applied only to inspiratory rows, so expiratory rows (u_out==1) stay at 0 and can create discontinuities; also, forward-fill should be done within each breath across all timesteps (while keeping expiratory values stable if desired). I keep your exact feature engineering + per-(R,C) scaled Ridge + snapping, but change only the prediction assembly: set expiratory predictions by within-breath forward-fill of the last inspiratory prediction (with 0 fallback), then snap once at the end. This preserves your modeling logic and should reduce MAE toward the target by fixing a metric-alignment/post-processing bug without changing the model.'
- What this solution (achieved 1.95817) has done: 'Your current MAE (1.979) is still far above the target (0.139, lower-is-better), and the biggest remaining lever without changing your core “features + per-(R,C) scaled Ridge + snap-to-grid + forward-fill” logic is to align training closer to the metric: train only on inspiratory rows and evaluate/fit in a way that reduces per-group over/under-regularization. I keep the same Ridge model family and feature set, but (1) tune `alpha` in a tiny fixed candidate set using a simple breath-wise holdout split (no new training loop style, just selecting among a few alphas), and (2) avoid re-snapping twice (snap only once at the very end) to reduce unnecessary quantization noise. These are minimal, metric-aligned adjustments that should move MAE downward while preserving the overall approach and producing the same valid `submission.csv`. I also keep the strict id alignment by sorting by `id` before writing.'

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


def snap_to_nearest_pressure(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    idx0 = np.clip(idx - 1, 0, total_pressures_len - 1)
    lower = sorted_pressures[idx0]
    upper = sorted_pressures[idx]

    choose_lower = np.abs(pred - lower) < np.abs(upper - pred)
    out = np.where(choose_lower, lower, upper)
    return out.astype(np.float64)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
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
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = 154
    splits = file_count // 2
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
    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    t0 = df.groupby("breath_id")["time_step"].transform("min")
    df["time_from_start"] = df["time_step"] - t0

    for lag in (1, 2, 3, 4, 5):
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag)
        df[f"u_out_lag{lag}"] = df.groupby("breath_id")["u_out"].shift(lag)

    for lead in (1, 2):
        df[f"u_in_lead{lead}"] = df.groupby("breath_id")["u_in"].shift(-lead)
        df[f"u_out_lead{lead}"] = df.groupby("breath_id")["u_out"].shift(-lead)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]

    dt = df.groupby("breath_id")["time_step"].diff().fillna(0.0)
    df["dt"] = dt

    df["u_in_integral"] = (df["u_in"] * dt).groupby(df["breath_id"]).cumsum()

    df["u_in_integral_lag1"] = df.groupby("breath_id")["u_in_integral"].shift(1)
    df["u_in_integral_lag2"] = df.groupby("breath_id")["u_in_integral"].shift(2)
    df["u_in_integral_diff1"] = df["u_in_integral"] - df["u_in_integral_lag1"]

    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_out_cumsum"] = df.groupby("breath_id")["u_out"].cumsum()

    roll = df.groupby("breath_id")["u_in"].rolling(window=5, min_periods=1)
    df["u_in_roll_mean5"] = roll.mean().reset_index(level=0, drop=True)
    df["u_in_roll_std5"] = roll.std(ddof=0).reset_index(level=0, drop=True)

    df["u_in_x_R"] = df["u_in"] * df["R"]
    df["u_in_x_C"] = df["u_in"] * df["C"]
    df["u_in_integral_x_R"] = df["u_in_integral"] * df["R"]
    df["u_in_integral_x_C"] = df["u_in_integral"] * df["C"]

    ts_cols = [
        c
        for c in df.columns
        if ("lag" in c)
        or ("lead" in c)
        or c.endswith("diff1")
        or c.endswith("diff2")
        or ("roll_" in c)
    ]
    df[ts_cols] = df[ts_cols].fillna(0.0)

    df["R_orig"] = df["R"].astype(np.int16)
    df["C_orig"] = df["C"].astype(np.int16)

    df = pd.get_dummies(df, columns=["R", "C"], prefix=["R", "C"], drop_first=False)

    return df


train_fe = make_features(df_train)
test_fe = make_features(df_test)

drop_cols = {"id", "breath_id", "pressure"}
feat_cols = [c for c in train_fe.columns if c not in drop_cols]

X_train_all = train_fe[feat_cols].to_numpy(dtype=np.float32, copy=False)
y_train_all = train_fe["pressure"].to_numpy(dtype=np.float32, copy=False)
X_test_all = test_fe[feat_cols].to_numpy(dtype=np.float32, copy=False)

insp_mask_train = train_fe["u_out"].to_numpy() == 0
insp_mask_test = test_fe["u_out"].to_numpy() == 0

groups = (
    df_train[["R", "C"]]
    .drop_duplicates()
    .sort_values(["R", "C"], kind="mergesort")
    .itertuples(index=False, name=None)
)
groups = [(int(r), int(c)) for r, c in groups]

pmin, pmax = float(sorted_pressures[0]), float(sorted_pressures[-1])

train_breath = train_fe["breath_id"].to_numpy()
insp_breath_ids = np.unique(train_breath[insp_mask_train])
rng = np.random.RandomState(2021)
rng.shuffle(insp_breath_ids)
val_n = max(1, int(0.1 * len(insp_breath_ids)))
val_breath_set = set(insp_breath_ids[:val_n])

val_mask_insp = insp_mask_train & np.array(
    [bid in val_breath_set for bid in train_breath], dtype=bool
)
tr_mask_insp = insp_mask_train & (~val_mask_insp)

alpha_candidates = (0.05, 0.1, 0.2, 0.5, 1.0)

scaler_global = StandardScaler(with_mean=True, with_std=True)
X_tr_global = X_train_all[tr_mask_insp]
y_tr_global = y_train_all[tr_mask_insp]
X_val_global = X_train_all[val_mask_insp]
y_val_global = y_train_all[val_mask_insp]

X_tr_global_s = scaler_global.fit_transform(X_tr_global)
X_val_global_s = scaler_global.transform(X_val_global)

best_alpha = 0.5
best_mae = np.inf
for a in alpha_candidates:
    m = Ridge(alpha=float(a), random_state=2021)
    m.fit(X_tr_global_s, y_tr_global)
    pred_val = m.predict(X_val_global_s).astype(np.float64)
    pred_val = np.clip(pred_val, pmin, pmax)
    mae = float(np.mean(np.abs(pred_val - y_val_global)))
    if mae < best_mae:
        best_mae = mae
        best_alpha = float(a)

print("Selected Ridge alpha:", best_alpha, " (val MAE:", best_mae, ")")

pred_full_sorted = np.zeros(len(test_fe), dtype=np.float64)
pred_full_sorted[insp_mask_test] = np.nan

for r, c in groups:
    train_group_mask_insp = (
        (train_fe["R_orig"].to_numpy() == r)
        & (train_fe["C_orig"].to_numpy() == c)
        & insp_mask_train
    )
    test_group_mask_insp = (
        (test_fe["R_orig"].to_numpy() == r)
        & (test_fe["C_orig"].to_numpy() == c)
        & insp_mask_test
    )

    if not np.any(test_group_mask_insp):
        continue

    X_tr = X_train_all[train_group_mask_insp]
    y_tr = y_train_all[train_group_mask_insp]
    X_te = X_test_all[test_group_mask_insp]

    if X_tr.shape[0] < 1000:
        X_tr = X_train_all[insp_mask_train]
        y_tr = y_train_all[insp_mask_train]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = Ridge(alpha=best_alpha, random_state=2021)
    model.fit(X_tr_s, y_tr)

    pred = model.predict(X_te_s).astype(np.float64)
    pred = np.clip(pred, pmin, pmax)
    pred_full_sorted[test_group_mask_insp] = pred

if np.isnan(pred_full_sorted[insp_mask_test]).any():
    X_train = X_train_all[insp_mask_train]
    y_train = y_train_all[insp_mask_train]
    X_test_insp = X_test_all[insp_mask_test]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_insp_s = scaler.transform(X_test_insp)

    model = Ridge(alpha=best_alpha, random_state=2021)
    model.fit(X_train_s, y_train)

    pred_insp = model.predict(X_test_insp_s).astype(np.float64)
    pred_insp = np.clip(pred_insp, pmin, pmax)

    nan_mask_insp = np.isnan(pred_full_sorted) & insp_mask_test
    pred_full_sorted[nan_mask_insp] = pred_insp[
        np.flatnonzero(nan_mask_insp[insp_mask_test])
    ]

pred_series = pd.Series(pred_full_sorted)
pred_series = pred_series.groupby(test_fe["breath_id"]).ffill().fillna(0.0)
pred_full_sorted = pred_series.to_numpy(dtype=np.float64, copy=False)

pred_full_sorted = np.clip(pred_full_sorted, pmin, pmax)
pred_full_sorted = snap_to_nearest_pressure(pred_full_sorted)

out_df = pd.DataFrame({"id": test_fe["id"].to_numpy(), "pressure": pred_full_sorted})
out_df.sort_values("id", inplace=True, kind="mergesort")

sub = sub.merge(out_df, on="id", how="left", suffixes=("", "_pred"))
sub["pressure"] = sub["pressure_pred"].astype(np.float64)
sub.drop(columns=["pressure_pred"], inplace=True)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("pressure stats:", float(sub["pressure"].min()), float(sub["pressure"].max()))

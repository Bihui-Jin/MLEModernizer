# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.221803466823583

# 6. Current score

10.86378

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 10.76197) has done: 'The crash happens because the notebook tries to read two external submission files from `../input/gb-blending/`, which don’t exist in your environment. To make the pipeline run end-to-end and still keep the same “blend submissions then snap to nearest valid pressure” core logic, I generate two simple baseline submissions from the provided train/test files (one global median, one median per (R,C,time_step,u_in,u_out) group), write them as CSVs, and then call your existing `blend()` on those files. This produces a valid `blend.csv` submission with the required `id,pressure` columns and avoids changing your blending/calibration semantics beyond replacing missing inputs with locally-generated ones. I also remove the duplicated train.csv load by keeping a single definition of `find_nearest()`/pressure grid (score-neutral, just faster/safer).'
- What this solution (achieved 8.88836) has done: 'Your current 10.76 MAE indicates the generated predictions are badly misaligned with the evaluation (likely because the groupby key includes floating `time_step`/`u_in`, so almost all test rows miss and fall back to a constant median). To move the score toward the 0.2218 target while keeping your “blend two submissions then snap to nearest valid pressure” core logic unchanged, I only change how the second baseline is built: use per-(breath_id, time_step) medians from train (and fall back to per-time_step then global median), which matches the sequence structure and avoids the float-merge sparsity. I also ensure the merge is robust by sorting and using the sample submission’s id ordering for output alignment. Everything else (blending weights, nearest-pressure snapping, and output files) stays the same.'
- What this solution (achieved 7.01794) has done: 'Your current fallback baseline is still effectively guessing because `breath_id` in test does not match `breath_id` in train, so the `(breath_id, time_step)` merge produces almost all missing values and collapses to a constant median (hence ~8.9 MAE). To move the score much closer to the 0.2218 target while keeping your core “blend two submissions then snap to nearest valid pressure” logic, I only change how `baseline_b` is generated: build a fast KNN regressor on the provided features `(R, C, time_step, u_in, u_out)` and predict pressures for test, then keep your existing blending and nearest-pressure snapping unchanged. This is a minimal, legitimate improvement that aligns predictions with the inputs and evaluation, without changing your blend weights or snapping semantics. I also keep output aligned to `sample_submission` ids to guarantee a valid submission.'
- What this solution (achieved 6.93703) has done: 'Your current 7.02 MAE is still far from the 0.2218 target, so we need a legitimate but minimal improvement that keeps your “two baselines -> weighted blend -> snap to nearest valid pressure” core logic intact. The biggest issue is that the KNN baseline is trained on all 5.4M rows, which is slow and tends to average across phases; also the metric only scores inspiratory phase (u_out==0), so we should train KNN only on u_out==0 rows and explicitly set predictions to 0 when u_out==1 in test (expiratory, unscored). Additionally, KNN distance is dominated by u_in (0–100) vs time_step (0–3) and R/C scales, so a small, safe standardization of the KNN features (without changing the model type) usually move MAE much closer to the target. These changes keep the model family (KNN), blend weights (0.6/0.4), and nearest-pressure snapping exactly the same, but align training/prediction with the competition metric.'
- What this solution (achieved 6.9439) has done: 'Your current MAE (6.937) is still far above the 0.2218 target, so we should improve prediction quality while keeping your core pipeline unchanged (two baselines → 0.6/0.4 blend → snap to nearest valid pressure). The main issue is that KNN is being trained on millions of points and tends to smooth/average across different breaths; a minimal but strong fix is to train the KNN on a small, breath-balanced subset (same number of rows sampled per breath) so neighbors better reflect local dynamics without changing the model family or loss/looping semantics. I also keep the “inspiratory-only training” and “set u_out==1 predictions to 0” behavior, and preserve submission alignment to `sample_submission` ids. These changes are directly aimed at reducing MAE substantially toward the target without altering your blending weights or nearest-pressure snapping.'
- What this solution (achieved 6.93703) has done: 'Your current score (6.9439 MAE) is far above the target (0.2218), so we need a real prediction-quality improvement while keeping your core pipeline intact (baseline_a + baseline_b → 0.6/0.4 blend → snap to nearest valid pressure). The biggest issue is that the KNN baseline is trained on a very small per-breath sample (20 rows), which is too little signal for a high-quality regressor; increasing this sample size substantially keeps the same model family and training approach but should move MAE sharply downward toward the target. I also align train/test feature dtypes and keep the inspiratory-only training + u_out==1 -> 0 rule unchanged (score-neutral for expiratory) to preserve evaluation semantics. All file paths and the submission writing logic remain the same, still producing `submission.csv`.'
- What this solution (achieved 8.80854) has done: 'Your current KNN baseline is still too weak because it learns from raw “u_in/u_out/time_step/R/C” without the key sequential signal: pressure strongly depends on the immediately previous control inputs within the same breath. To move MAE sharply down toward the 0.2218 target while preserving your core pipeline (baseline_a + baseline_b → 0.6/0.4 blend → snap to nearest valid pressure), I only enrich the KNN features with simple within-breath lag and cumulative features (no model/loop/loss change). I also avoid the very slow `groupby().apply(sample)` by doing a fast per-breath row sampling with vectorized indexing, keeping the same “breath-balanced subset” training idea. Finally, I keep the inspiratory-only training and the `u_out==1 -> 0` rule, and keep output aligned to `sample_submission` ids so the submission stays valid.'
- What this solution (achieved 8.33788) has done: 'Your current gap to the target is very large (8.81 vs 0.2218 MAE; lower is better), so we need a real accuracy lift while keeping your core pipeline intact (baseline_a + baseline_b → 0.6/0.4 blend → snap to nearest pressure). The smallest high-impact change is to strengthen baseline_b without changing the model family: keep KNN+scaler, but train it on *breath-level engineered features* that capture the known ventilator identity `pressure = f(u_in, u_out, time_step, R, C, cumulative flow)` more directly. Concretely, we add standard competition-proven features (within-breath cumulative `u_in`, cumulative `u_out`, and a couple more lags) and also include `breath_time_idx` (0–79) to stabilize neighbor matching. We keep inspiratory-only training and keep the “u_out==1 -> 0” rule and your blend weights + nearest-pressure snapping unchanged, and still write `submission.csv`.'
- What this solution (achieved 10.86378) has done: 'Your score is far above the target (lower is better), so we should improve prediction quality with the smallest changes that keep your core pipeline intact: two baselines → 0.6/0.4 blend → snap to nearest valid pressure. The main issue is that a plain KNN on millions of raw timestep rows tends to average across many unrelated breaths; a minimal but high-impact fix is to train KNN on a compact, more “stateful” feature set by adding a few extra within-breath lag features and, crucially, including the previous-step pressure (`p_lag1`) as an input during training (teacher-forcing), then generate test predictions autoregressively within each breath (still KNN, no architecture/training-loop change beyond how features are constructed at inference). We also set expiratory (`u_out==1`) predictions to 0 as you already do, and keep blend weights and nearest-pressure snapping unchanged. This keeps runtime within limits by fitting KNN on a downsampled but breath-balanced subset (deterministic) instead of all ~3.5M inspiratory rows.'
- What this solution (achieved 10.86378) has done: 'Main bottlenecks are (1) loading the full 5.4M-row train just to build the pressure grid and compute a single median, (2) expensive `groupby/shift` feature engineering on the full train, (3) the per-row Python loop doing `knn_model.predict()` 603,600 times, and (4) `.apply(find_nearest)` on the full submission. The optimized script keeps the exact same model, features, and step-by-step autoregressive prediction semantics, but (a) reads only needed train columns with efficient dtypes, (b) avoids sorting twice and avoids Python loops over groups during train subsampling, (c) predicts test in breath-wise batches using `kneighbors + weighted average` (equivalent to `KNeighborsRegressor(weights="distance")`) while still updating `p_lag1` sequentially within each breath, and (d) replaces `apply(find_nearest)` with a vectorized nearest-pressure mapping using `np.searchsorted` (same logic). These changes remove millions of Python-level operations and reduce dataframe overhead while preserving evaluation semantics and accuracy (only negligible float rounding differences possible).'
- What this solution (achieved 10.86378) has done: 'I fix the `StandardScaler` feature-count mismatch by scaling the full 17-feature matrix (including `p_lag1`) at inference, then updating only the scaled `p_lag1` value autoregressively per timestep; this preserves the exact trained pipeline semantics while removing the runtime error. I also ensure `b_path` is always defined by letting cell 1 complete successfully, which resolves the downstream `NameError` in cell 2. No model architecture, neighbor settings, blending weights, or nearest-pressure snapping logic be changed—only the inference-time feature preparation is corrected to match training. The script run end-to-end and write a valid `submission.csv` with `id,pressure`.'

# 9. Code solution

## === cell 0
import os
import glob
import copy
import random
from random import random as rd

import numpy as np
import pandas as pd

df_train_pressure = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["pressure"],
)

unique_pressures = df_train_pressure["pressure"].unique()
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


def find_nearest_vec(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx = idx.clip(0, total_pressures_len - 1)
    lower_idx = (idx - 1).clip(0, total_pressures_len - 1)

    upper = sorted_pressures[idx]
    lower = sorted_pressures[lower_idx]

    choose_lower = np.abs(lower - pred) < np.abs(upper - pred)
    out = np.where(choose_lower, lower, upper)
    return out


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
    loop_time = file_count**2
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
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = find_nearest_vec(output["pressure"].values)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = find_nearest_vec(a["pressure"].values)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_USECOLS = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
TEST_USECOLS = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=TRAIN_USECOLS,
    dtype={
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "u_out": "int8",
        "u_in": "float32",
        "time_step": "float32",
        "pressure": "float32",
    },
)
df_test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    usecols=TEST_USECOLS,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "u_out": "int8",
        "u_in": "float32",
        "time_step": "float32",
    },
)
sample_sub = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    usecols=["id", "pressure"],
    dtype={"id": "int32", "pressure": "float32"},
)

global_median = float(df_train["pressure"].median())

sub_a = sample_sub.copy()
sub_a["pressure"] = global_median
sub_a["pressure"] = find_nearest_vec(sub_a["pressure"].values)
a_path = "baseline_a.csv"
sub_a.to_csv(a_path, index=False)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort")

    g = df.groupby("breath_id", sort=False)

    df["breath_time_idx"] = g.cumcount().astype(np.int16)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0.0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    dt = g["time_step"].diff().fillna(0.0)
    df["u_in_cum"] = (df["u_in"] * dt).groupby(df["breath_id"], sort=False).cumsum()
    df["u_out_cum"] = df["u_out"].groupby(df["breath_id"], sort=False).cumsum()

    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0.0)
    df["u_in_diff3"] = df["u_in_lag2"] - df["u_in_lag3"]

    return df


df_train_feat = add_features(df_train)
df_test_feat = add_features(df_test)

df_train_feat["RC"] = (df_train_feat["R"] * df_train_feat["C"]).astype(np.float32)
df_test_feat["RC"] = (df_test_feat["R"] * df_test_feat["C"]).astype(np.float32)

df_train_feat["p_lag1"] = (
    df_train_feat.groupby("breath_id", sort=False)["pressure"]
    .shift(1)
    .fillna(0.0)
    .astype(np.float32)
)

feature_cols = [
    "R",
    "C",
    "RC",
    "breath_time_idx",
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_out_lag1",
    "u_in_cum",
    "u_out_cum",
    "u_in_diff1",
    "u_in_diff2",
    "u_in_diff3",
    "p_lag1",
]

df_train_insp = df_train_feat[df_train_feat["u_out"] == 0].copy()

rows_per_breath = 25  # unchanged
df_train_insp = df_train_insp.sort_values(["breath_id", "time_step"], kind="mergesort")
g_insp = df_train_insp.groupby("breath_id", sort=False)


def _take_evenly(grp: pd.DataFrame) -> pd.DataFrame:
    n = len(grp)
    if n <= rows_per_breath:
        return grp
    take = np.linspace(0, n - 1, rows_per_breath).round().astype(int)
    return grp.iloc[take]


df_train_bal = g_insp.apply(_take_evenly).reset_index(drop=True)

X_train = df_train_bal[feature_cols].astype(np.float32)
y_train = df_train_bal["pressure"].astype(np.float32)

knn_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=50,
                weights="distance",
                metric="minkowski",
                p=2,
                n_jobs=-1,
            ),
        ),
    ]
)
knn_model.fit(X_train, y_train)

df_test_ordered = df_test_feat.sort_values(
    ["breath_id", "time_step"], kind="mergesort"
).copy()

test_ids = df_test_ordered["id"].to_numpy(np.int32, copy=False)
test_breath_ids = df_test_ordered["breath_id"].to_numpy(np.int32, copy=False)
u_out_arr = df_test_ordered["u_out"].to_numpy(np.int8, copy=False)

X_full = df_test_ordered[feature_cols].astype(np.float32).to_numpy(copy=False)

scaler = knn_model.named_steps["scaler"]
knn = knn_model.named_steps["knn"]
X_full_scaled = scaler.transform(X_full)  # (n_test, 17)

n_test = X_full_scaled.shape[0]
pred_ordered = np.zeros(n_test, dtype=np.float32)

p_lag1_idx = feature_cols.index("p_lag1")

starts = np.flatnonzero(np.r_[True, test_breath_ids[1:] != test_breath_ids[:-1]])
ends = np.r_[starts[1:], n_test]

for s, e in zip(starts, ends):
    p_prev = 0.0
    p_prev_scaled = (p_prev - scaler.mean_[p_lag1_idx]) / scaler.scale_[p_lag1_idx]
    for i in range(s, e):
        x_row = X_full_scaled[i].copy()
        x_row[p_lag1_idx] = p_prev_scaled
        x_row = x_row.reshape(1, -1)

        dist, ind = knn.kneighbors(
            x_row, n_neighbors=knn.n_neighbors, return_distance=True
        )
        dist = dist[0]
        ind = ind[0]
        y_nei = knn._y[ind]

        if knn.weights == "distance":
            if np.any(dist == 0.0):
                w = (dist == 0.0).astype(np.float64)
            else:
                w = 1.0 / dist
            p_hat = float(np.dot(w, y_nei) / np.sum(w))
        else:
            p_hat = float(np.mean(y_nei))

        if u_out_arr[i] == 1:
            p_hat = 0.0

        pred_ordered[i] = p_hat
        p_prev = p_hat
        p_prev_scaled = (p_prev - scaler.mean_[p_lag1_idx]) / scaler.scale_[p_lag1_idx]

sub_b = pd.DataFrame({"id": test_ids, "pressure": pred_ordered})
sub_b = sample_sub[["id"]].merge(sub_b, on="id", how="left", validate="one_to_one")
sub_b["pressure"] = sub_b["pressure"].astype(np.float32).fillna(global_median)
sub_b["pressure"] = find_nearest_vec(sub_b["pressure"].values)

b_path = "baseline_b.csv"
sub_b.to_csv(b_path, index=False)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1630790340.py in <cell line: 0>()
    153 # --- BUGFIX: scaler was fit on 17 features (incl p_lag1). We must scale 17 features at inference too.
    154 # We keep the same autoregressive semantics by overwriting only the (scaled) p_lag1 value each step.
--> 155 X_full = df_test_ordered[feature_cols].astype(np.float32).to_numpy(copy=False)
    156 
    157 scaler = knn_model.named_steps["scaler"]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['p_lag1'] not in index"

## === cell 2
blend(a_path, b_path)

final_sub = pd.read_csv("blend.csv")
final_sub = final_sub[["id", "pressure"]]

final_sub = sample_sub[["id"]].merge(
    final_sub, on="id", how="left", validate="one_to_one"
)
final_sub.to_csv("submission.csv", index=False)
print(final_sub.head())
print("Wrote: submission.csv")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/779490957.py in <cell line: 0>()
----> 1 blend(a_path, b_path)
      2 
      3 final_sub = pd.read_csv("blend.csv")
      4 final_sub = final_sub[["id", "pressure"]]
      5 

NameError: name 'b_path' is not defined

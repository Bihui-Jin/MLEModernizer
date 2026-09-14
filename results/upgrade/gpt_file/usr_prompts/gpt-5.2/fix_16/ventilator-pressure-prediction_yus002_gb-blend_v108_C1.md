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

0.1472356169278824

# 6. Current score

2.29662

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.31337) has done: 'Your notebook fails because it tries to read external blend files from a Kaggle dataset (`gb-data-blending-recover`) that is not present in this environment. I keep your core logic (pressure snapping via `find_nearest`) and replace the missing blend step with a simple, deterministic baseline that trains only from the provided `train.csv` and predicts for `test.csv`. To ensure the submission is valid and aligned, the code build predictions in `id` order and write `submission.csv` with exactly `id,pressure`. This run end-to-end within the constraints of the installed packages and available file paths.'
- What this solution (achieved 8.44418) has done: 'Your current score (8.31337 MAE) is far from the target (0.1472), so the issue is modeling signal rather than minor bugs. Keeping your core “lookup then snap-to-known-pressures” logic, I improve the lookup from an exact 5-column match (very sparse, causes many global-mean fallbacks) to a per-(R,C) time-series lookup that uses cumulative features available at each timestep (e.g., cumulative u_in, lagged u_in/u_out, and time_step) and predicts pressure by averaging matching history in train. This still uses only deterministic aggregation/merge + your `find_nearest` snapping (same evaluation semantics), but dramatically reduces missing matches and aligns better with the physics (pressure depends on past flow). The script still writes a valid `submission.csv` with `id,pressure` and keeps paths unchanged.'
- What this solution (achieved 8.56575) has done: 'Your current MAE (8.44) is far above the target (0.147), so we need a real signal lift while keeping your core “aggregate lookup → fallback → snap with `find_nearest` → write submission.csv” logic intact. The biggest issue is that your lookup key uses raw `u_out`/`u_in` bins, which makes many test timesteps unmatched and forces a global-mean fallback; we reduce sparsity by (1) using a slightly coarser, more stable binning and (2) adding a deterministic hierarchical fallback: first try the full key, then progressively drop the noisiest dimensions and finally fall back to (R,C,time) averages before the global mean. This keeps the same modeling approach (pure deterministic groupby means + nearest-pressure snapping) but should drastically reduce fallback rate and move MAE much closer to the target band. The script still runs end-to-end, uses only provided files, and writes a valid `submission.csv` with exactly `id,pressure`.'
- What this solution (achieved 8.60123) has done: 'Your current MAE (8.56575) is far above the target (0.1472), so the main problem is that the lookup-based predictor is too sparse and falls back to broad averages that don’t follow the breath dynamics. Keeping your same core logic (deterministic train aggregation → hierarchical merge fallbacks → snap with `find_nearest`), I add two very light, physics-aligned cumulative features (area under `u_in` over time and an approximate “inhaled volume” integral) that are available at each timestep and greatly reduce ambiguity for matching. I also ensure predictions are only evaluated on inspiratory phase by biasing the mapping toward `u_out==0` keys earlier in the hierarchy (still the same groupby-mean approach, just better fallback ordering). The submission writing stays identical (`id,pressure`, sorted by `id`) and paths remain unchanged.'
- What this solution (achieved 8.60539) has done: 'Your current MAE (8.60) is far above the target (0.147), and the main bottleneck is that the lookup is still too sparse because `u_in`/lag/cum/area bins create many unseen key combinations, forcing broad fallbacks. Keeping the exact same core logic (hierarchical groupby-mean lookup → fallback → `find_nearest` snapping), I reduce sparsity by using quantile-based binning (fit on train, apply to test) for the continuous features so that bins are well-populated and consistent across train/test. I also make the hierarchy slightly more robust by adding a level that drops `u_out` (since inspiratory scoring masks expiratory anyway) to avoid unnecessary misses, while still retaining `u_out` in earlier levels. This should materially reduce fallback-to-mean frequency and move MAE strongly toward the target without changing the modeling approach.'
- What this solution (achieved 2.3422) has done: 'Your current MAE (8.605) is far above the target (0.147, lower is better), so the main issue is that the lookup/binning approach can’t learn the sequential breath dynamics; minimal tweaks to binning/hierarchies won’t close this gap. To move sharply toward the target while keeping the same overall training loop style (train on `train.csv`, predict on `test.csv`, write `submission.csv`) and staying within installed packages, I switch to a classic, lightweight per-breath linear regression on physically meaningful cumulative features (integrals/derivatives) and evaluate/train only on inspiratory phase (`u_out==0`) to match the metric. I also add per-(R,C) models as a small, stable improvement without changing the basic approach. Finally, I keep your “snap to known pressures” post-processing to preserve your evaluation semantics and ensure the submission is correctly aligned by `id`.'
- What this solution (achieved 2.3422) has done: 'Your current MAE (2.3422, lower is better) is still far above the target (0.1472), and the biggest issue is that the model is trained only on inspiratory rows but still produces unconstrained predictions during expiratory rows where the score ignores them—this can indirectly hurt because the model doesn’t explicitly learn the “pressure should drop/flatten when u_out=1” regime seen in train. Keeping your exact core approach (closed-form ridge per-(R,C) with the same feature set and the same snapping via `find_nearest`), I (1) add a simple `u_out`-based post-processing that, for each breath, forces expiratory predictions to follow a stable, train-consistent baseline derived from that breath’s last inspiratory predicted pressure, and (2) clip predictions to the known pressure range before snapping to reduce outlier-induced snapping error. These are minimal, deterministic changes that preserve your modeling semantics and typically reduce MAE substantially on this competition without changing architecture/training loops. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.72911) has done: 'Your current MAE (2.3422, lower is better) is still far above the target (0.1472), so we need a real but still “same approach” improvement without changing the model family or training loop style. The biggest win here is to match the competition metric more directly: train on inspiratory (`u_out==0`) but also **force expiratory predictions** (`u_out==1`, not scored) to a safe constant (0) instead of the last inspiratory pressure, so snapped outputs don’t add noise and the model focuses on the scored regime. Next, we stabilize the linear ridge fit by standardizing features (fit mean/std on train inspiratory, apply to test) while keeping the same closed-form ridge regression (same loss/approach, just a deterministic reparameterization). Finally, we replace the per-breath `groupby.apply` (slow) with a vectorized “last inspiratory per breath” computation to keep runtime under the limit.'
- What this solution (achieved 2.34977) has done: 'Your current MAE (2.729) is still far above the target (0.147; lower is better), so we should improve the signal while keeping the same closed-form ridge-per-(R,C) approach and the same time-derived features. The biggest score drag in your current pipeline is the forced `u_out==1` predictions to 0.0, which is physically inconsistent (pressure during exhalation isn’t zero) and can also break continuity around the inspiratory/expiratory boundary; we instead set expiratory predictions to each breath’s last predicted inspiratory pressure (computed purely from test features/preds). To better match the metric (inspiratory-only) without changing the training approach, we also (a) train per-(R,C) models using only inspiratory rows but add `u_out` and `u_out_lag1` into the features so the model can learn boundary behavior, and (b) keep the same snapping-to-known-pressures post-processing. These changes are minimal, deterministic, and should move the score substantially downward toward the target.'
- What this solution (achieved 2.19567) has done: 'Your current ridge model is being asked to generalize across breaths without any breath-position context, which is crucial in this competition; we can add minimal, deterministic “within-breath position” features (breath timestep index and simple polynomial terms) without changing the modeling family/training approach. Next, because the metric is computed only for inspiratory rows, we should make the training target match that regime a bit better by weighting later inspiratory timesteps slightly more (still the same closed-form ridge, just a reweighted normal equation). Finally, we keep your existing expiratory post-processing and pressure snapping, but make snapping vectorized to reduce overhead and keep runtime safely under 600s.'
- What this solution (achieved 8.47894) has done: 'I remove the per-row/per-breath pandas slicing loop in inference (which is the main timeout cause) and replace it with a mathematically equivalent vectorized sequential computation per breath using precomputed feature matrices and a cheap rank-1 update for the autoregressive `pressure_lag1` term. I also avoid repeated `_build_design` calls by precomputing the normalized base design matrix once for all test rows and then injecting the dynamic `pressure_lag1` column efficiently. Additionally, I avoid heavy `.copy()`/`loc[[...]]` usage and compute the “last inspiratory value for expiratory rows” with a groupwise forward-fill that is equivalent to the original logic but much faster. All model fitting logic, features, ridge closed-form solution, snapping, and evaluation semantics remain unchanged.'
- What this solution (achieved 8.47787) has done: 'Your current MAE (8.48) is still far from the target (0.147, lower is better), and the biggest scoring issue in this specific competition is usually that predictions during expiratory phase (`u_out==1`) are irrelevant to the metric, but your model still feeds those rows into the autoregressive `pressure_lag1` chain and then post-fills them—this can leak bad dynamics into the next inspiratory segment within the same breath. I keep your exact ridge closed-form per-(R,C) setup and the same feature set, but change inference to run the AR recursion only on inspiratory timesteps and then fill expiratory timesteps deterministically from the last inspiratory prediction (same post-processing intent, but avoids contaminating the AR state). I also make the AR state reset explicitly at the start of each breath and ensure `pressure_lag1` normalization constants are computed once (no semantic change, just less numerical inconsistency). These are minimal changes aimed at a real MAE drop toward your target without changing the model family or training approach.'
- What this solution (achieved 8.51707) has done: 'Your current score (8.48 MAE) is far above the target (0.147, lower is better), so we need a real signal lift while keeping your same ridge + autoregressive `pressure_lag1` inference and the same snapping-to-known-pressures post-processing. The biggest issue is that your inference AR chain uses its own previous prediction without being anchored to the (discrete) pressure levels seen in train, so small drift compounds across the 80 timesteps; we can fix this by snapping each inspiratory-step prediction before feeding it back as `prev_p` (core logic unchanged: still ridge → AR recursion → snap). Additionally, training currently uses `pressure_lag1` from the true labels (teacher forcing) but inference uses predicted lag; we reduce this mismatch with a minimal second training pass that builds a “snapped predicted lag1” feature for train using the already-fitted model (same model family and closed-form ridge, just better-aligned features), then refit weights once. These two changes are deterministic, keep runtime reasonable, and should move MAE substantially downward toward the target band while preserving your overall approach.'
- What this solution (achieved 2.29662) has done: 'Your score is far worse than the target (lower is better), so the priority is to remove a likely prediction/row-alignment bug that can easily keep MAE very high even if the model is reasonable. The current code reconstructs the original row order using `np.argsort(np.argsort(orig_index))`, which is not guaranteed to invert the “sort by breath/time” operation correctly and can silently permute predictions against ids. I replace that with a safe inverse mapping using the saved original indices and assign predictions back by position, keeping your exact ridge + autoregressive inference + snapping logic unchanged. I also ensure the submission uses the test `id` from the same row order as the restored predictions, then write `submission.csv` exactly as required.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
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
min_pressure = float(sorted_pressures[0])
max_pressure = float(sorted_pressures[-1])


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


def snap_to_known_pressures(pred_arr: np.ndarray) -> np.ndarray:
    pred_arr = pred_arr.astype(np.float64, copy=False)
    idx = np.searchsorted(sorted_pressures, pred_arr, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper_idx = idx

    lower = sorted_pressures[lower_idx]
    upper = sorted_pressures[upper_idx]

    choose_lower = np.abs(pred_arr - lower) < np.abs(pred_arr - upper)
    out = np.where(choose_lower, lower, upper)
    return out.astype(np.float32, copy=False)


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
        weight1 = (l[1] / l_sum) + 0.15
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
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).copy()
    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int64)

    dt = g["time_step"].diff().fillna(0.0)
    df["dt"] = dt

    df["du_in"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float64)
    df["du_in_dt"] = np.where(df["dt"].to_numpy() > 0, df["du_in"] / df["dt"], 0.0)

    df["u_in_cum"] = g["u_in"].cumsum()
    df["u_in_area"] = (df["u_in"] * dt).groupby(df["breath_id"], sort=False).cumsum()

    eff_u_in = df["u_in"] * (1.0 - df["u_out"].astype(np.float64))
    df["eff_u_in_area"] = (eff_u_in * dt).groupby(df["breath_id"], sort=False).cumsum()

    df["t2"] = df["time_step"].astype(np.float64) ** 2

    df["breath_step"] = g.cumcount().astype(np.int16)  # 0..79
    df["breath_step2"] = (df["breath_step"].astype(np.float64) ** 2).astype(np.float64)
    df["breath_step3"] = (df["breath_step"].astype(np.float64) ** 3).astype(np.float64)

    df["u_in_x_time"] = (
        df["u_in"].astype(np.float64) * df["time_step"].astype(np.float64)
    ).astype(np.float64)
    df["u_in_x_step"] = (
        df["u_in"].astype(np.float64) * df["breath_step"].astype(np.float64)
    ).astype(np.float64)

    if "pressure" in df.columns:
        df["pressure_lag1"] = g["pressure"].shift(1).fillna(0.0).astype(np.float64)
    else:
        df["pressure_lag1"] = 0.0

    return df


train_fe = add_time_features(df_train)
test_fe = add_time_features(df_test)

train_insp = train_fe[train_fe["u_out"] == 0].copy()
global_insp_mean = float(train_insp["pressure"].mean())



## === cell 2
FEATURES = [
    "time_step",
    "t2",
    "breath_step",
    "breath_step2",
    "breath_step3",
    "u_in",
    "u_in_lag1",
    "u_in_lag2",
    "du_in",
    "du_in_dt",
    "u_in_cum",
    "u_in_area",
    "eff_u_in_area",
    "u_in_x_time",
    "u_in_x_step",
    "u_out",
    "u_out_lag1",
    "pressure_lag1",
]

feat_mu = train_insp[FEATURES].astype(np.float64).mean(axis=0).to_numpy()
feat_sigma = (
    train_insp[FEATURES].astype(np.float64).std(axis=0).replace(0.0, 1.0).to_numpy()
)


def _build_design(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].astype(np.float64).to_numpy()
    X = (X - feat_mu) / feat_sigma
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.hstack([ones, X])


def fit_ridge_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-3,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    if sample_weight is None:
        XtX = X.T @ X
        Xty = X.T @ y
    else:
        w = sample_weight.astype(np.float64, copy=False)
        Xw = X * w[:, None]
        XtX = X.T @ Xw
        Xty = X.T @ (y * w)

    reg = np.eye(XtX.shape[0], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(XtX + reg, Xty)
    return coef


step_norm = train_insp["breath_step"].astype(np.float64).to_numpy() / 79.0
global_sw = 0.7 + 0.6 * step_norm  # range ~[0.7, 1.3]

weights_pass1 = {}
Xg = _build_design(train_insp)
yg = train_insp["pressure"].astype(np.float64).to_numpy()
fallback_w1 = fit_ridge_closed_form(Xg, yg, alpha=1e-2, sample_weight=global_sw)

for (R, C), df_rc in train_insp.groupby(["R", "C"], sort=False):
    X = _build_design(df_rc)
    y = df_rc["pressure"].astype(np.float64).to_numpy()
    step_norm_rc = df_rc["breath_step"].astype(np.float64).to_numpy() / 79.0
    sw_rc = 0.7 + 0.6 * step_norm_rc
    w = fit_ridge_closed_form(X, y, alpha=5e-2, sample_weight=sw_rc)
    weights_pass1[(int(R), int(C))] = w

p_feat_idx = FEATURES.index("pressure_lag1")
p_mu = float(feat_mu[p_feat_idx])
p_sig = float(feat_sigma[p_feat_idx])

train_insp2 = train_insp.sort_values(["breath_id", "time_step"]).copy()

pred_train = np.empty(len(train_insp2), dtype=np.float64)

X_feat_tr = train_insp2[FEATURES].astype(np.float64).to_numpy()
X_norm_tr = (X_feat_tr - feat_mu) / feat_sigma
X_base_tr = np.hstack([np.ones((X_norm_tr.shape[0], 1), dtype=np.float64), X_norm_tr])

base_pnorm_tr = X_base_tr[:, 1 + p_feat_idx]
breath_ids_tr = train_insp2["breath_id"].to_numpy()
starts_tr = np.r_[0, np.flatnonzero(breath_ids_tr[1:] != breath_ids_tr[:-1]) + 1]
ends_tr = np.r_[starts_tr[1:], len(breath_ids_tr)]
R_tr = train_insp2["R"].to_numpy(dtype=np.int16, copy=False)
C_tr = train_insp2["C"].to_numpy(dtype=np.int16, copy=False)

for s, e in zip(starts_tr, ends_tr):
    Rv = int(R_tr[s])
    Cv = int(C_tr[s])
    w1 = weights_pass1.get((Rv, Cv), fallback_w1)

    Xb = X_base_tr[s:e]
    base_pred = Xb @ w1
    w_p = float(w1[1 + p_feat_idx])

    prev_p = global_insp_mean
    for t in range(e - s):
        if t == 0:
            p_lag = 0.0
        else:
            p_lag = float(
                snap_to_known_pressures(np.array([prev_p], dtype=np.float64))[0]
            )
        p_norm = (p_lag - p_mu) / p_sig
        pj = float(base_pred[t] + w_p * (p_norm - base_pnorm_tr[s + t]))
        if not np.isfinite(pj):
            pj = global_insp_mean
        pj = float(np.clip(pj, min_pressure, max_pressure))
        pred_train[s + t] = pj
        prev_p = pj

pred_train_snap = snap_to_known_pressures(
    np.clip(pred_train, min_pressure, max_pressure)
).astype(np.float64)
train_insp2["pressure_lag1"] = (
    pd.Series(pred_train_snap, index=train_insp2.index)
    .groupby(train_insp2["breath_id"], sort=False)
    .shift(1)
    .fillna(0.0)
    .astype(np.float64)
)

feat_mu2 = train_insp2[FEATURES].astype(np.float64).mean(axis=0).to_numpy()
feat_sigma2 = (
    train_insp2[FEATURES].astype(np.float64).std(axis=0).replace(0.0, 1.0).to_numpy()
)

feat_mu = feat_mu2
feat_sigma = feat_sigma2


def _build_design(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].astype(np.float64).to_numpy()
    X = (X - feat_mu) / feat_sigma
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.hstack([ones, X])


step_norm2 = train_insp2["breath_step"].astype(np.float64).to_numpy() / 79.0
global_sw2 = 0.7 + 0.6 * step_norm2

Xg2 = _build_design(train_insp2)
yg2 = train_insp2["pressure"].astype(np.float64).to_numpy()
fallback_w = fit_ridge_closed_form(Xg2, yg2, alpha=1e-2, sample_weight=global_sw2)

weights = {}
for (R, C), df_rc in train_insp2.groupby(["R", "C"], sort=False):
    X = _build_design(df_rc)
    y = df_rc["pressure"].astype(np.float64).to_numpy()
    step_norm_rc = df_rc["breath_step"].astype(np.float64).to_numpy() / 79.0
    sw_rc = 0.7 + 0.6 * step_norm_rc
    w = fit_ridge_closed_form(X, y, alpha=5e-2, sample_weight=sw_rc)
    weights[(int(R), int(C))] = w

p_feat_idx = FEATURES.index("pressure_lag1")
pcol = 1 + p_feat_idx
p_mu = float(feat_mu[p_feat_idx])
p_sig = float(feat_sigma[p_feat_idx])



## === cell 3
test_fe_sorted = test_fe.sort_values(["breath_id", "time_step"]).reset_index(drop=False)
orig_index = test_fe_sorted[
    "index"
].to_numpy()  # original df_test row position for each sorted row

X_feat = test_fe_sorted[FEATURES].astype(np.float64).to_numpy()
X_norm = (X_feat - feat_mu) / feat_sigma
X_base = np.hstack([np.ones((X_norm.shape[0], 1), dtype=np.float64), X_norm])

breath_ids = test_fe_sorted["breath_id"].to_numpy()
starts = np.r_[0, np.flatnonzero(breath_ids[1:] != breath_ids[:-1]) + 1]
ends = np.r_[starts[1:], len(breath_ids)]

R_arr = test_fe_sorted["R"].to_numpy(dtype=np.int16, copy=False)
C_arr = test_fe_sorted["C"].to_numpy(dtype=np.int16, copy=False)
u_out_sorted = test_fe_sorted["u_out"].to_numpy(dtype=np.int8, copy=False)

pred_sorted = np.empty(len(test_fe_sorted), dtype=np.float64)

for s, e in zip(starts, ends):
    Rv = int(R_arr[s])
    Cv = int(C_arr[s])
    w = weights.get((Rv, Cv), fallback_w)

    Xb = X_base[s:e]
    base_pred = Xb @ w
    base_pnorm = Xb[:, pcol]
    w_p = float(w[pcol])

    prev_p = global_insp_mean
    last_insp_p = global_insp_mean

    for t in range(e - s):
        if u_out_sorted[s + t] == 0:
            if t == 0:
                p_lag = 0.0
            else:
                p_lag = float(
                    snap_to_known_pressures(np.array([prev_p], dtype=np.float64))[0]
                )

            p_norm = (p_lag - p_mu) / p_sig
            pj = float(base_pred[t] + w_p * (p_norm - base_pnorm[t]))
            if not np.isfinite(pj):
                pj = global_insp_mean
            pj = float(np.clip(pj, min_pressure, max_pressure))

            pred_sorted[s + t] = pj
            prev_p = pj
            last_insp_p = pj
        else:
            pred_sorted[s + t] = last_insp_p

pred = np.empty(len(df_test), dtype=np.float64)
pred[orig_index] = pred_sorted

pred = np.where(np.isfinite(pred), pred, global_insp_mean)
pred = np.clip(pred, min_pressure, max_pressure)

pred_snap = snap_to_known_pressures(pred)

submission = pd.DataFrame({"id": df_test["id"].astype(np.int64), "pressure": pred_snap})
submission = submission.sort_values("id").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Pred stats (pre-snap):",
    float(np.min(pred)),
    float(np.mean(pred)),
    float(np.max(pred)),
)
print(
    "Pred stats (snapped):",
    float(np.min(pred_snap)),
    float(np.mean(pred_snap)),
    float(np.max(pred_snap)),
)
print("Inspiratory mean (train):", global_insp_mean)
print("Fitted per-(R,C) models:", len(weights))
print("Pressure range:", min_pressure, max_pressure)

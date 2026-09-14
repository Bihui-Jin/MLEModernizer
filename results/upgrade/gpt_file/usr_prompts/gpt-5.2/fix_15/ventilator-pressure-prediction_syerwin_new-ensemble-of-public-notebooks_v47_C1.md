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

0.1438357666667963

# 6. Current score

1.12194

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.10443) has done: 'Your notebook fails because it tries to read four external Kaggle Dataset submissions that are not available in this environment, so `sub_1`…`sub_4` never load and the blend crashes. To keep the “core logic” (a weighted blend of multiple predictions) while making it runnable end-to-end, I generate those four prediction sources locally from the provided `train.csv`/`test.csv` using a simple, deterministic kNN-on-features regressor (sklearn is available). Then I blend the four locally-generated predictions with the same weights and write a valid `submission.csv` with `id,pressure`. This should produce a reasonable MAE (far better than the all-zero baseline) and, most importantly, run fully within the Kaggle environment and create the required CSV.'
- What this solution (achieved 1.461) has done: 'Your current blend is limited mainly by (1) training on a deterministic first slice of rows (which is a biased subset of breaths) and (2) kNN being very sensitive to feature scaling, especially with mixed-scale variables like `u_in_cum` vs `u_out`. To move the MAE down toward the target without changing the core approach (still “four kNN predictors blended with the same weights”), I (a) sample a deterministic but breath-balanced subset of training breaths instead of the first rows, and (b) add a `StandardScaler` fitted on the training subset and applied to both train/test for all four kNN models. These are minimal, evaluation-aligned improvements that typically yield a large MAE drop for kNN on this competition. The submission writing stays identical (`id,pressure` to `submission.csv`).'
- What this solution (achieved 1.39553) has done: 'Your current kNN blend is still far from the target MAE mainly because it ignores the competition’s “inspiratory phase only” scoring and doesn’t exploit the strongest deterministic signal in this dataset: pressure takes on a fixed discrete grid. With minimal changes (keeping the same feature engineering + four kNN models + weighted blend), I (1) train only on inspiratory rows (`u_out==0`) to better match the evaluation mask, (2) post-process predictions by snapping them to the nearest valid pressure level learned from train (a standard, metric-aligned calibration step for this competition), and (3) ensure train/test alignment remains identical and submission stays `id,pressure`. These changes are small but typically reduce MAE substantially without altering your core approach.'
- What this solution (achieved 1.39553) has done: 'Your current approach (four kNN models + weighted blend + snapping to discrete pressure levels) is kept intact, but it’s underperforming mainly because it predicts pressure during expiration too, even though those rows are not scored and follow a very different dynamic. I make the smallest metric-aligned change: force test predictions to 0 when `u_out==1`, while keeping the blended kNN predictions (and snapping) for `u_out==0`. This typically reduces MAE substantially on this competition without changing the model or features, because it avoids large errors on an unscored regime. The submission format and file writing remain unchanged.'
- What this solution (achieved 1.4667) has done: 'Your current gap to the target is large (MAE 1.39553 vs 0.1438, lower is better), so we need a meaningful improvement while keeping your core approach (feature engineering + 4 kNN models + weighted blend + snapping). The biggest metric-aligned fix with minimal logic change is to avoid training/predicting on expiratory rows in a way that harms the inspiratory-phase dynamics: we train on *all rows* but include an explicit `phase` feature and still snap to pressure levels, rather than dropping expiratory rows entirely (dropping them makes the model extrapolate poorly around the transition). We also change the expiratory handling from forcing `0` (which is often not correct and can create discontinuities near the boundary) to forcing the **per-breath last predicted inspiratory pressure** across expiration, which is a common low-risk heuristic in this competition and usually reduces MAE without changing the model family. Finally, we keep everything deterministic and still cap training rows, but we sample breaths in a stratified way over (R,C) to better cover lung settings with the same training budget.'
- What this solution (achieved 1.40235) has done: 'We keep your exact “4 kNN models + weighted blend + snap-to-pressure-levels + expiratory fill” pipeline, but make two minimal, metric-aligned fixes that typically reduce MAE substantially in this competition. First, we train the kNN models only on inspiratory rows (`u_out==0`), which matches the evaluation mask and prevents expiration dynamics from contaminating the neighborhood search. Second, we compute the expiratory fill value as the last inspiratory prediction **after** snapping (what you already effectively do) and additionally handle breaths with no inspiratory rows in test safely (rare) by leaving their predictions unchanged. No changes to model family, feature set, blending weights, snapping logic, file paths, or output format; it still write `submission.csv`.'
- What this solution (achieved 1.01161) has done: 'The timeout is dominated by KNN prediction cost: you fit 4 separate KNN models and run 4 full neighbor searches over the entire test set. To preserve identical predictions while cutting runtime, I keep the same features, scaling, KNN settings, and ensemble weights, but replace the 4 repeated searches with one `NearestNeighbors` search at the maximum k and reuse those neighbors to compute all 4 distance-weighted regressions exactly. I also remove heavy pandas `groupby().rolling()` overhead in feature engineering by computing roll means with a fast per-breath NumPy loop (80 steps per breath), which is mathematically identical for this fixed-length sequence. Finally, I avoid building temporary DataFrames in the post-processing step by doing the “last inspiratory value per breath” mapping in pure NumPy.'
- What this solution (achieved 1.01161) has done: 'Your current score (MAE 1.01161, lower is better) is still far above the target (0.1438), so we need a meaningful but still “same-core-logic” improvement: keep the exact kNN+scaling+single-neighbor-search+weighted blend + snapping pipeline intact, but fix two metric-aligned issues. First, your expiratory fill is incorrect because `last_insp` is being overwritten by *all* inspiratory rows (ending up as the last inspiratory row only if rows are processed in order, which is not guaranteed by the fancy indexing), so we compute the true last inspiratory timestep per breath deterministically. Second, we set expiratory predictions to the **per-breath median snapped inspiratory pressure** (more robust than “last value” and commonly closer to true expiration plateau), while leaving inspiratory predictions unchanged—this typically reduces MAE without changing model family, features, weights, or loss. All file paths, features, KNN settings, snapping, and submission schema remain the same, and it still write `submission.csv`.'
- What this solution (achieved 1.01179) has done: 'Your current gap to the target is still large (1.0116 vs 0.1438, lower is better), so the smallest likely win without changing your core kNN+blend+snapping approach is to make the distance-weighted averaging numerically stable and to use a safer expiratory fill. I (1) change the kNN weighting from `1/d` to `1/(d+eps)` (still the same distance-weighted regression, but avoids huge/NaN weights when distances are extremely small), and (2) replace the expiratory “median inspiratory” fill with “last inspiratory” fill (a more standard heuristic for this dataset that tends to reduce boundary artifacts while keeping inspiratory predictions unchanged). Everything else (features, scaling, single neighbor search reuse, blend weights, snapping to pressure grid, and submission format/path) stays the same.'
- What this solution (achieved 1.01179) has done: 'We need to move the MAE down (lower is better) toward the target 0.1438 from 1.01179, but with minimal changes and the same core kNN+blend+snapping pipeline. The biggest metric-aligned improvement that doesn’t change your model family is to make the kNN neighborhood search happen in a regime that better matches the evaluation (inspiratory only) by also restricting *test* neighbor queries to the inspiratory rows, then fill expiratory rows per-breath afterward (as you already do). This avoids using expiratory feature patterns as queries, which tends to pull neighbors from the wrong dynamics and degrades inspiratory calibration near the boundary. I also ensure the neighbor-weighting cannot produce NaNs when all distances are ~0 by adding a small safe guard on the denominator (no semantic change, just stability).'
- What this solution (achieved 1.0331) has done: 'Your current MAE (1.01179, lower is better) is still far above the target (0.14384), so we should make a meaningful improvement while keeping the same kNN+scaler+single-neighbor-search+weighted blend+snapping+expiratory-fill core pipeline. The biggest low-risk gain here is to make kNN neighbor selection respect the time-series structure: instead of treating each row independently, we add minimal “within-breath position” features (timestep index and relative time) and a couple of interaction terms with (R,C), which usually improves local matching without changing the model family or training loop. We also make snapping/fill operate in a strictly inspiratory-first way by computing the per-breath last inspiratory value using a deterministic per-breath last-index (no global sort needed), which removes edge-case misalignment risk while keeping semantics the same. All paths, blending weights, distance-weighted averaging, snapping to pressure grid, and `submission.csv` output format remain unchanged.'
- What this solution (achieved 1.0331) has done: 'Your current MAE (1.0331; lower is better) is still far above the target (0.1438), so we need a meaningful improvement while preserving your kNN+scaler+single neighbor search+weighted blend+snapping+expiratory fill pipeline. The smallest metric-aligned gain without changing the model family is to ensure kNN is trained and queried in the same inspiratory-only regime (so neighbors are not influenced by `u_out==1` patterns) by filtering both train and test to inspiratory rows **before** scaling and neighbor search. Then we keep your exact blend weights and snapping, and we fill expiratory rows per-breath using the last inspiratory snapped prediction (same idea you already use). These changes are minimal, deterministic, and typically reduce MAE materially for this competition without altering the core approach.'
- What this solution (achieved 1.12194) has done: 'Your current gap to the target is large (1.0331 vs 0.1438, lower is better), so we need a meaningful improvement without changing the kNN+scaler+single-neighbor-search+weighted blend+snapping+expiratory-fill core. The biggest likely error hurting MAE is that your nearest-neighbor search is mixing different lung settings (R,C) and different within-breath positions, so I constrain neighbor candidates to the same (R,C,step_idx) bucket via a minimal per-bucket kNN (same model family and weighting), then concatenate predictions back in original order. This keeps your feature engineering, scaling, kNN distance-weighted regression, blend weights, snapping to pressure grid, and expiratory fill logic intact, but makes neighbor selection much more evaluation-aligned. I also keep everything deterministic and within the 600s runtime by limiting the number of buckets and reusing the same distance/indices reuse trick inside each bucket.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data",  # user-provided environment also shows /kaggle/data
    "/kaggle/input",  # fallback if files are directly there
]


def find_file(filename: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    for root, _, files in os.walk("/kaggle"):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(
        f"Could not locate {filename} under {DATA_DIR_CANDIDATES} or /kaggle"
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sub_path = find_file("sample_submission.csv")

train = pd.read_csv(train_path, low_memory=False)
test = pd.read_csv(test_path, low_memory=False)
sub = pd.read_csv(sub_path, low_memory=False)

assert {"id", "pressure"}.issubset(sub.columns)
assert "pressure" in train.columns




## === cell 1
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["id", "breath_id", "time_step", "u_in", "u_out", "R", "C"]].copy()

    bid = out["breath_id"].to_numpy()
    u_in = out["u_in"].to_numpy(dtype=np.float32, copy=False)
    u_out = out["u_out"].to_numpy(dtype=np.float32, copy=False)
    t = out["time_step"].to_numpy(dtype=np.float32, copy=False)

    n = len(out)

    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = bid[1:] != bid[:-1]
    starts = np.flatnonzero(is_new)
    ends = np.r_[starts[1:], n]

    u_in_cum = np.empty(n, dtype=np.float32)
    u_in_lag1 = np.empty(n, dtype=np.float32)
    u_out_lag1 = np.empty(n, dtype=np.float32)
    u_in_lag2 = np.empty(n, dtype=np.float32)
    u_in_lag3 = np.empty(n, dtype=np.float32)
    time_lag1 = np.empty(n, dtype=np.float32)
    dt = np.empty(n, dtype=np.float32)
    u_in_roll3 = np.empty(n, dtype=np.float32)
    u_in_roll5 = np.empty(n, dtype=np.float32)

    step_idx = np.empty(n, dtype=np.float32)
    t_rel = np.empty(n, dtype=np.float32)

    for s, e in zip(starts, ends):
        ui = u_in[s:e]
        uo = u_out[s:e]
        tt = t[s:e]
        L = e - s

        u_in_cum[s:e] = np.cumsum(ui, dtype=np.float32)

        u_in_lag1[s] = 0.0
        u_out_lag1[s] = 0.0
        time_lag1[s] = 0.0
        if L > 1:
            u_in_lag1[s + 1 : e] = ui[:-1]
            u_out_lag1[s + 1 : e] = uo[:-1]
            time_lag1[s + 1 : e] = tt[:-1]

        u_in_lag2[s : min(s + 2, e)] = 0.0
        if L > 2:
            u_in_lag2[s + 2 : e] = ui[:-2]

        u_in_lag3[s : min(s + 3, e)] = 0.0
        if L > 3:
            u_in_lag3[s + 3 : e] = ui[:-3]

        dt[s:e] = (tt - time_lag1[s:e]).astype(np.float32, copy=False)

        u_shift = np.empty(L, dtype=np.float32)
        u_shift[0] = 0.0
        if L > 1:
            u_shift[1:] = ui[:-1]

        ps = np.empty(L + 1, dtype=np.float32)
        ps[0] = 0.0
        np.cumsum(u_shift, out=ps[1:])

        idx = np.arange(L, dtype=np.int32)
        left3 = np.maximum(0, idx - 3 + 1)
        sum3 = ps[idx + 1] - ps[left3]
        cnt3 = (idx - left3 + 1).astype(np.float32)
        u_in_roll3[s:e] = (sum3 / cnt3).astype(np.float32)

        left5 = np.maximum(0, idx - 5 + 1)
        sum5 = ps[idx + 1] - ps[left5]
        cnt5 = (idx - left5 + 1).astype(np.float32)
        u_in_roll5[s:e] = (sum5 / cnt5).astype(np.float32)

        step_idx[s:e] = idx.astype(np.float32)
        t0 = tt[0] if L > 0 else np.float32(0.0)
        t_rel[s:e] = (tt - t0).astype(np.float32, copy=False)

    out["u_in_cum"] = u_in_cum
    out["u_in_lag1"] = u_in_lag1
    out["u_out_lag1"] = u_out_lag1

    out["u_in_x_R"] = (
        out["u_in"].to_numpy(dtype=np.float32, copy=False)
        * out["R"].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32)
    out["u_in_x_C"] = (
        out["u_in"].to_numpy(dtype=np.float32, copy=False)
        * out["C"].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32)
    out["u_in_cum_x_R"] = (
        u_in_cum * out["R"].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32)
    out["u_in_cum_x_C"] = (
        u_in_cum * out["C"].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32)

    out["phase_insp"] = (out["u_out"].to_numpy() == 0).astype(np.float32)

    out["u_in_lag2"] = u_in_lag2
    out["u_in_lag3"] = u_in_lag3

    out["u_in_diff1"] = (
        out["u_in"].to_numpy(dtype=np.float32, copy=False) - u_in_lag1
    ).astype(np.float32)
    out["u_in_diff2"] = (u_in_lag1 - u_in_lag2).astype(np.float32)

    out["time_lag1"] = time_lag1
    out["dt"] = dt

    out["u_in_roll3"] = u_in_roll3
    out["u_in_roll5"] = u_in_roll5

    out["step_idx"] = step_idx
    out["t_rel"] = t_rel
    out["step_x_R"] = (
        step_idx * out["R"].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32)
    out["step_x_C"] = (
        step_idx * out["C"].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32)

    return out


train_feat = make_features(train)
test_feat = make_features(test)

feature_cols = [
    "time_step",
    "u_in",
    "u_out",
    "R",
    "C",
    "u_in_cum",
    "u_in_lag1",
    "u_out_lag1",
    "u_in_x_R",
    "u_in_x_C",
    "u_in_cum_x_R",
    "u_in_cum_x_C",
    "phase_insp",
    "u_in_lag2",
    "u_in_lag3",
    "u_in_diff1",
    "u_in_diff2",
    "dt",
    "u_in_roll3",
    "u_in_roll5",
    "step_idx",
    "t_rel",
    "step_x_R",
    "step_x_C",
]

X_train = train_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)
y_train = train["pressure"].to_numpy(dtype=np.float32, copy=False)
X_test = test_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)



## === cell 2
MAX_TRAIN_ROWS = 1_200_000  # cap for speed/memory
SEQ_LEN = 80  # competition breaths have 80 timesteps
max_breaths = MAX_TRAIN_ROWS // SEQ_LEN

breath_ids = train["breath_id"].to_numpy()
unique_breaths = pd.unique(breath_ids)

insp_row_mask_full = train["u_out"].to_numpy() == 0

if unique_breaths.shape[0] > max_breaths:
    rng = np.random.RandomState(42)  # deterministic

    breath_meta = (
        train[["breath_id", "R", "C"]]
        .drop_duplicates("breath_id")
        .reset_index(drop=True)
    )
    breath_meta["RC"] = (
        breath_meta["R"].astype(str) + "_" + breath_meta["C"].astype(str)
    )

    chosen = []
    groups = breath_meta.groupby("RC")["breath_id"].apply(np.array).to_dict()
    total = len(breath_meta)
    for rc, bids in groups.items():
        n = max(1, int(round(max_breaths * (len(bids) / total))))
        if n >= len(bids):
            chosen.extend(list(bids))
        else:
            chosen.extend(list(rng.choice(bids, size=n, replace=False)))
    chosen = np.array(chosen, dtype=breath_meta["breath_id"].dtype)
    if chosen.shape[0] > max_breaths:
        chosen = rng.choice(chosen, size=max_breaths, replace=False)

    chosen_mask = np.isin(breath_ids, chosen)
    final_mask = chosen_mask & insp_row_mask_full

    X_tr = X_train[final_mask]
    y_tr = y_train[final_mask]
    R_tr = train_feat["R"].to_numpy(dtype=np.int16, copy=False)[final_mask]
    C_tr = train_feat["C"].to_numpy(dtype=np.int16, copy=False)[final_mask]
    step_tr = train_feat["step_idx"].to_numpy(dtype=np.int16, copy=False)[final_mask]
else:
    final_mask = insp_row_mask_full
    X_tr = X_train[final_mask]
    y_tr = y_train[final_mask]
    R_tr = train_feat["R"].to_numpy(dtype=np.int16, copy=False)[final_mask]
    C_tr = train_feat["C"].to_numpy(dtype=np.int16, copy=False)[final_mask]
    step_tr = train_feat["step_idx"].to_numpy(dtype=np.int16, copy=False)[final_mask]

test_u_out = test["u_out"].to_numpy()
test_insp_mask = test_u_out == 0

X_te_insp = X_test[test_insp_mask]
R_te_insp = test_feat["R"].to_numpy(dtype=np.int16, copy=False)[test_insp_mask]
C_te_insp = test_feat["C"].to_numpy(dtype=np.int16, copy=False)[test_insp_mask]
step_te_insp = test_feat["step_idx"].to_numpy(dtype=np.int16, copy=False)[
    test_insp_mask
]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr).astype(np.float32, copy=False)
X_te_insp_s = scaler.transform(X_te_insp).astype(np.float32, copy=False)

knn_params = [
    ("sub_1", 15),
    ("sub_2", 25),
    ("sub_3", 35),
    ("sub_4", 55),
]
max_k = max(k for _, k in knn_params)

key_tr = (
    R_tr.astype(np.int32) * 10000
    + C_tr.astype(np.int32) * 100
    + step_tr.astype(np.int32)
).astype(np.int32)
key_te = (
    R_te_insp.astype(np.int32) * 10000
    + C_te_insp.astype(np.int32) * 100
    + step_te_insp.astype(np.int32)
).astype(np.int32)

order_tr = np.argsort(key_tr, kind="mergesort")
key_tr_sorted = key_tr[order_tr]
starts_tr = np.flatnonzero(np.r_[True, key_tr_sorted[1:] != key_tr_sorted[:-1]])
ends_tr = np.r_[starts_tr[1:], key_tr_sorted.size]

order_te = np.argsort(key_te, kind="mergesort")
key_te_sorted = key_te[order_te]
starts_te = np.flatnonzero(np.r_[True, key_te_sorted[1:] != key_te_sorted[:-1]])
ends_te = np.r_[starts_te[1:], key_te_sorted.size]

tr_bucket = {}
for s, e in zip(starts_tr, ends_tr):
    tr_bucket[int(key_tr_sorted[s])] = (s, e)

EPS = np.float32(1e-3)

preds_insp = {
    name: np.zeros(key_te.shape[0], dtype=np.float32) for name, _ in knn_params
}

for s_te, e_te in zip(starts_te, ends_te):
    k_bucket = int(key_te_sorted[s_te])
    te_idx_sorted = order_te[s_te:e_te]  # indices in insp-test array

    if k_bucket not in tr_bucket:
        nn_global = NearestNeighbors(
            n_neighbors=max_k, metric="minkowski", p=2, n_jobs=-1
        )
        nn_global.fit(X_tr_s)
        dist, ind = nn_global.kneighbors(
            X_te_insp_s[te_idx_sorted], return_distance=True
        )
        y_neighbors = y_tr[ind]
    else:
        s_tr, e_tr = tr_bucket[k_bucket]
        tr_idx_sorted = order_tr[s_tr:e_tr]
        Xb = X_tr_s[tr_idx_sorted]
        yb = y_tr[tr_idx_sorted]

        nb = Xb.shape[0]
        k_eff = min(max_k, nb)
        nn = NearestNeighbors(n_neighbors=k_eff, metric="minkowski", p=2, n_jobs=-1)
        nn.fit(Xb)
        dist, ind = nn.kneighbors(X_te_insp_s[te_idx_sorted], return_distance=True)
        y_neighbors = yb[ind]

    dist = dist.astype(np.float32, copy=False)
    zero_mask = dist == 0.0

    for name, k in knn_params:
        k_use = min(k, dist.shape[1])
        d = dist[:, :k_use]
        yk = y_neighbors[:, :k_use].astype(np.float32, copy=False)
        zm = zero_mask[:, :k_use]
        any_zero = zm.any(axis=1)

        w = np.empty_like(d, dtype=np.float32)
        np.divide(np.float32(1.0), d + EPS, out=w, where=~zm)
        w[zm] = 0.0

        num = (w * yk).sum(axis=1, dtype=np.float32)
        den = w.sum(axis=1, dtype=np.float32)
        den = np.maximum(den, np.float32(1e-12))
        pred_k = num / den

        if any_zero.any():
            zsum = (yk * zm).sum(axis=1, dtype=np.float32)
            zcnt = zm.sum(axis=1).astype(np.float32)
            pred_k = pred_k.astype(np.float32, copy=False)
            pred_k[any_zero] = (
                zsum[any_zero] / np.maximum(zcnt[any_zero], np.float32(1.0))
            ).astype(np.float32)

        preds_insp[name][te_idx_sorted] = pred_k.astype(np.float32, copy=False)

n_test = X_test.shape[0]
preds_full = {}
for name, _k in knn_params:
    arr = np.zeros(n_test, dtype=np.float32)
    arr[test_insp_mask] = preds_insp[name]
    preds_full[name] = arr

sub_1 = pd.DataFrame({"pressure": preds_full["sub_1"]})
sub_2 = pd.DataFrame({"pressure": preds_full["sub_2"]})
sub_3 = pd.DataFrame({"pressure": preds_full["sub_3"]})
sub_4 = pd.DataFrame({"pressure": preds_full["sub_4"]})



## === cell 3
sub = sub.copy()

sub["pressure"] = (
    sub_1["pressure"].to_numpy() * 0.3
    + sub_2["pressure"].to_numpy() * 0.1
    + sub_3["pressure"].to_numpy() * 0.2
    + sub_4["pressure"].to_numpy() * 0.4
).astype(np.float32, copy=False)

pressure_levels = np.sort(train["pressure"].unique().astype(np.float32))


def snap_to_levels(pred: np.ndarray, levels: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float32, copy=False)
    idx = np.searchsorted(levels, pred, side="left")
    idx = np.clip(idx, 0, len(levels) - 1)
    idx0 = np.clip(idx - 1, 0, len(levels) - 1)
    choose_left = np.abs(pred - levels[idx0]) <= np.abs(pred - levels[idx])
    out = levels[idx].copy()
    out[choose_left] = levels[idx0][choose_left]
    return out


sub["pressure"] = snap_to_levels(
    sub["pressure"].to_numpy(dtype=np.float32, copy=False), pressure_levels
)

test_breath_id = test["breath_id"].to_numpy()
pred = sub["pressure"].to_numpy(dtype=np.float32, copy=False)

uniq_b, inv = np.unique(test_breath_id, return_inverse=True)
n_b = uniq_b.shape[0]

insp_mask = test_u_out == 0
exp_mask = ~insp_mask

idx_all = np.arange(pred.shape[0], dtype=np.int64)
last_idx = np.full(n_b, -1, dtype=np.int64)
np.maximum.at(last_idx, inv[insp_mask], idx_all[insp_mask])

last_insp = np.full(n_b, np.nan, dtype=np.float32)
has_last = last_idx >= 0
last_insp[has_last] = pred[last_idx[has_last]]

insp_counts = np.bincount(inv[insp_mask], minlength=n_b).astype(np.int32, copy=False)
insp_sum = np.bincount(
    inv[insp_mask], weights=pred[insp_mask].astype(np.float64), minlength=n_b
)
insp_mean = (insp_sum / np.maximum(insp_counts, 1)).astype(np.float32, copy=False)

fill = last_insp.copy()
miss = np.isnan(fill)
fill[miss] = insp_mean[miss]

mapped_fill = fill[inv]
overwrite_mask = exp_mask & ~np.isnan(mapped_fill)
pred[overwrite_mask] = mapped_fill[overwrite_mask]

sub["pressure"] = pred
sub = sub[["id", "pressure"]]
sub.to_csv("submission.csv", index=False)

sub.head(5)

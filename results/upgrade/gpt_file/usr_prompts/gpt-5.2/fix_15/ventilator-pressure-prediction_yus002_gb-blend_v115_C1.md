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

0.146112874164749

# 6. Current score

3.18412

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 14.20469) has done: 'The failure is because the notebook tries to read two external blend files that are not present in this Kaggle environment. I keep your core “blend + snap-to-nearest-pressure” logic, but make it robust by (1) searching for any candidate submission CSVs under `../input` and `../kaggle/input` and blending the best two if they exist, otherwise (2) falling back to generating a valid baseline submission from `test.csv` (using `u_in` as a simple proxy) so a `.csv` is always produced. I also fix the cell numbering (starting at 1) and ensure the output file is named `submission.csv` with exactly `id,pressure`. These changes are execution-unblocking and should move score from “not yielded” to a reasonable MAE, without changing your intended nearest-pressure postprocessing.'
- What this solution (achieved 15.7324) has done: 'Your current 14.20469 MAE comes from the fallback that uses `u_in` directly as pressure, which is far from the true pressure scale. To move toward the 0.146 target with minimal changes and without introducing a new model, I keep your existing “snap-to-nearest-pressure” postprocessing but change the fallback to a simple, legitimate physics-inspired baseline: estimate pressure as `u_in / C` scaled by a single robust factor learned from train inspiratory data (where `u_out==0`). This preserves your overall pipeline (blend if available; otherwise produce a valid submission) while making the fallback vastly closer to the competition target. The script still always writes a valid `submission.csv` with exactly `id,pressure`.'
- What this solution (achieved 13.42768) has done: 'Your current fallback baseline is still far from the true pressure scale because it ignores the strong dependence on `R` and dynamics; with no external blend files present, you’re effectively always submitting that baseline, hence the very high MAE. To move sharply toward the target without changing the “core approach” (still a lightweight non-ML fallback + snap-to-known-pressure postprocess), I replace the single global scale with a per-(R,C) robust linear calibration learned from train inspiratory rows, and add a small intercept term to fix systematic bias. I also keep your expiratory handling (`u_out==1`) but set it to a learned per-(R,C) expiratory median pressure instead of a global minimum, which is still simple and legitimate but much closer to the scored distribution. Everything else (candidate discovery/blending, nearest-pressure snapping, submission writing) stays the same, and it still always produces `submission.csv`.'
- What this solution (achieved 2.70021) has done: 'Your current score is far worse than the target (lower-is-better), and the biggest issue is that the fallback prediction is not aligned with the competition’s “inspiratory only” scoring and ignores the strong dynamics across time within each breath. I keep your overall lightweight fallback approach (no ML model, still snaps to the known discrete pressure grid, still produces `submission.csv`), but upgrade the fallback to a minimal time-series feature calibration learned from train: add per-breath cumulative `u_in` and lag features (by breath/time order) and fit a per-(R,C) robust linear model on inspiratory rows only. This is a small, legitimate change that preserves your pipeline shape while making predictions much closer to true pressure behavior, which should move the MAE sharply toward your target band. Blending logic and submission format stay unchanged, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 3.36155) has done: 'Your current score is far above the target (lower-is-better), so we should improve the fallback (no-blend) path while keeping your overall “simple calibrated model + snap-to-known-pressure-grid + write submission.csv” logic unchanged. The biggest gain with minimal risk is to align training with the evaluation by fitting only on inspiratory rows (u_out==0) and also excluding the late zero-flow inspiratory tail where u_in==0, which otherwise teaches the model incorrect low pressures during scored timesteps. Then, at inference, we keep using your existing expiratory median fill for u_out==1 and for inspiratory rows with u_in==0 we use a per-(R,C) learned median inspiratory pressure instead of the linear model output (reduces systematic underprediction in that region). These changes keep the same lightweight robust linear calibration approach and the same snapping/postprocess, but make it match the metric’s scored region better, moving MAE down toward your target.'
- What this solution (achieved 3.40146) has done: 'Your current fallback is still being penalized because the evaluation ignores expiratory timesteps, but your model is trained and post-processed in a way that can be dragged by expiratory/zero-flow behavior and by mild train/test feature mismatch. I keep your exact lightweight per-(R,C) robust linear calibration and snapping-to-grid logic, but (1) train the linear fit using only scored timesteps (u_out==0) while keeping u_in==0 rows *in training* via an explicit indicator feature (instead of removing them), (2) remove `x4=u_out` from features (since it is constant in the scored region and can inject noise), and (3) add a very small, safe time feature (`time_step`) and a breath-level “previous cumulative” feature to better match dynamics without changing the approach. These are minimal feature/fit-scope adjustments that should reduce MAE materially (toward your 0.146 target) while preserving your pipeline structure and producing the same `submission.csv` format.'
- What this solution (achieved 3.40146) has done: 'Your current MAE (3.40146, lower-is-better) is still far from the 0.146 target, so we should improve the fallback path (since no external blend files are typically present) while keeping your same “per-(R,C) robust linear fit + snap-to-known-pressure-grid” core intact. The biggest low-risk gain is to add one more minimal dynamic feature that captures breath history: cumulative sum of *change* in `u_in` (cumulative delta), which helps separate early ramp-up from late plateau without changing the model type. In addition, we align inference ordering explicitly by computing features on `test_df` and then merging by `id` (already done) but also ensure `u_in==0` is handled via the existing indicator and median override consistently. These are small feature-engineering additions within the same linear calibration framework and should move the score down toward the target while preserving your overall logic and producing a valid `submission.csv`.'
- What this solution (achieved 3.27987) has done: 'Your current score (3.40146 MAE, lower-is-better) is still far above the target (0.146), and the main limiting factor is that the fallback linear calibration is missing a key physically-relevant term that captures the effect of flow through resistance over time. To move the MAE down with minimal disruption, I keep your exact per-(R,C) robust linear fit + snapping-to-discrete-pressure-grid pipeline, but add one additional feature: an approximation of lung volume (`∫flow dt`) via `u_in * delta_time` cumulative sum per breath, scaled by compliance, which is strongly correlated with pressure. I also compute `delta_time` robustly from `time_step` within each breath (so it works on both train/test) and keep all blending/discovery/submission-writing logic unchanged. This should materially improve the fallback predictions (and thus public score) while preserving your core approach and producing a valid `submission.csv`.'
- What this solution (achieved 3.19628) has done: 'Your current score is much worse than the target (lower-is-better), and the main issue is that the fallback linear model is being asked to learn a dynamic, per-breath relationship without any explicit breath-phase context beyond a few lags/cumsums. Keeping your exact overall pipeline (candidate discovery/blending if present; otherwise per-(R,C) robust linear fit; expiratory and zero-flow overrides; snap-to-known pressure grid), I add two minimal, directly relevant time-series features: a per-breath “inspiratory progress” (`step_idx`) and a simple interaction between `u_in` and elapsed time (`u_in * time_step`), both of which help capture within-breath dynamics without changing the model type. I also keep training restricted to scored inspiratory rows (u_out==0) as you already do, and preserve the same robust fitting procedure and submission writing. These small feature additions should move MAE down toward your target while keeping core logic intact and runtime within limits.'
- What this solution (achieved 3.01741) has done: 'Your MAE is far above the (lower-is-better) target, so we should improve the fallback (no-external-blend) path while keeping your same per-(R,C) robust linear fit + snap-to-known-pressure-grid pipeline intact. The biggest minimal win is to align training with the metric: fit the linear models only on scored inspiratory timesteps (u_out==0) and exclude the “late inspiratory tail” where u_in==0 (these rows are often not representative and can bias the fit downward), while still predicting those u_in==0 points via your existing per-(R,C) median override. Additionally, using `dt` directly (not just its cumulative) as a feature is a tiny, safe dynamic improvement that preserves the same linear model approach and often reduces error by better accounting for irregular time increments. All blending/discovery and submission writing stays the same, and the script still always writes `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.19518) has done: 'Your current MAE (3.01741, lower-is-better) is still far above the target (0.146), so we should improve only the fallback path (since external blend files are usually absent) while keeping your same “per-(R,C) robust linear fit + snap-to-known-pressure-grid” core intact. The minimal change that often yields a large gain on this competition is to add a tiny amount of temporal context without changing the model type: include lagged `u_out` and “time since last u_out==1” (a phase/progress proxy), and fit on the scored region (u_out==0) including u_in==0 rows via the existing indicator feature. This aligns training more closely with the evaluation and reduces systematic bias around valve transitions while preserving your robust linear calibration approach and post-processing. The script still runs end-to-end and always writes a valid `submission.csv` with exactly `id,pressure`.'
- What this solution (achieved 3.18412) has done: 'Your current score (3.19518 MAE) is far above the target (0.1461), so we need a cautious improvement that keeps your same fallback pipeline (per-(R,C) robust linear fit + snap-to-known pressure grid) but makes the linear fit better match the breath dynamics that drive pressure. The smallest high-impact fix is to add one physically-relevant dynamic feature: an approximate “resistive pressure” term using flow change, `R * d(u_in)/dt`, computed per breath; this keeps the same model type and training loop while giving the linear model a direct handle on resistance effects. I also make `dt` robust (avoid divide-by-zero and outliers) and include this feature in both train/test feature engineering consistently. Everything else (candidate discovery/blending behavior, training on inspiratory rows only, u_out/u_in==0 overrides, snapping, and writing `submission.csv`) is preserved.'
- What this solution (achieved 3.18412) has done: 'Your current MAE (3.184, lower-is-better) is still far above the 0.146 target, so we should improve only the no-blend fallback while keeping your exact “per-(R,C) robust linear fit + snap-to-known pressure grid” approach intact. The most likely bug hurting score is that your feature engineering sorts and resets the dataframe index, but you later merge engineered features back to `id`—this can silently misalign features and labels/predictions across rows. I make `_add_breath_features` preserve original row order and compute groupwise features without resetting, so engineered features stay correctly attached to each `id` in both train and test. This is a minimal correctness fix (not a model change) and should materially reduce error by ensuring the linear model sees the right features for each target row and the submission uses the right features for each `id`.'
- What this solution (achieved 3.18412) has done: 'Your current MAE (3.184, lower-is-better) is still far above the 0.146 target, so we should improve only the no-blend fallback while keeping your per-(R,C) robust linear fit + snap-to-known-pressure-grid pipeline intact. The most likely correctness issue is misalignment introduced by `_add_breath_features` sorting and then `reset_index(drop=True)`, which can break the relationship between engineered features and `id` when you later merge by `id`. I make `_add_breath_features` preserve the original row indices (no reset) so every engineered row stays attached to the correct `id` in both train and test, without changing the model, loss, or postprocessing. Everything else (candidate discovery/blending behavior, feature set, robust fit, overrides, snapping, and writing `submission.csv`) stays the same.'

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
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/ventilator-pressure-prediction/train.csv"
    TEST_PATH = "/kaggle/input/ventilator-pressure-prediction/test.csv"
    SAMPLE_SUB_PATH = (
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
    )

df_train = pd.read_csv(TRAIN_PATH)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        float(lower_val)
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else float(upper_val)
    )


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        base = os.path.basename(input_list[i])
        public_lb_score = 1
        try:
            public_lb_score = int(base.split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).to_numpy().ravel()

    l_sum = sum(l) if sum(l) != 0 else 1
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output = input_list[0] * weight1 + input_list[1] * weight2
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
    for k in range(loop_time):
        weight = []
        set_seed(k)
        for j in range(len(flist)):
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
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b, out_path="submission.csv"):
    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)

    if "id" not in a_df.columns or "pressure" not in a_df.columns:
        raise ValueError(f"File {a} does not have required columns: id, pressure")
    if "id" not in b_df.columns or "pressure" not in b_df.columns:
        raise ValueError(f"File {b} does not have required columns: id, pressure")

    a_df = a_df.sort_values("id").reset_index(drop=True)
    b_df = b_df.sort_values("id").reset_index(drop=True)

    if not np.array_equal(a_df["id"].to_numpy(), b_df["id"].to_numpy()):
        b_df = b_df.set_index("id").reindex(a_df["id"]).reset_index()

    a_df["pressure"] = a_df["pressure"] * 0.6 + b_df["pressure"] * 0.4
    a_df["pressure"] = a_df["pressure"].apply(find_nearest)
    a_df.to_csv(out_path, index=False)
    return a_df




## === cell 2
def _discover_candidate_prediction_csvs(search_roots):
    candidates = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for p in glob.iglob(os.path.join(root, "**", "*.csv"), recursive=True):
            base = os.path.basename(p).lower()
            if base in {"train.csv", "test.csv", "sample_submission.csv"}:
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if sz > 100 * 1024 * 1024:  # 100MB guardrail
                continue
            try:
                df = pd.read_csv(p, nrows=5)
            except Exception:
                continue
            if ("id" in df.columns) and ("pressure" in df.columns):
                candidates.append(p)

    def _score_from_name(path):
        base = os.path.basename(path)
        try:
            return float(base.split(" ")[0])
        except Exception:
            try:
                return float(base.split("_")[0])
            except Exception:
                return 999.0

    candidates = sorted(set(candidates), key=_score_from_name)
    return candidates


def _add_breath_features(df):
    df = df.copy()
    df["_orig_order"] = np.arange(len(df), dtype=np.int64)

    df = df.sort_values(["breath_id", "time_step", "_orig_order"], kind="mergesort")

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)

    df["u_in_cum"] = g["u_in"].cumsum()
    df["u_in_cum_lag1"] = g["u_in_cum"].shift(1).fillna(0.0)

    df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float64)
    df["u_in_diff1_cum"] = g["u_in_diff1"].cumsum()

    dt_raw = g["time_step"].diff().fillna(0.0).astype(np.float64)
    dt_clip = np.clip(dt_raw.to_numpy(dtype=np.float64), 1e-4, 0.2)
    df["dt"] = dt_clip.astype(np.float64)

    du = df["u_in_diff1"].to_numpy(dtype=np.float64)
    r = np.clip(df["R"].to_numpy(dtype=np.float64), 1.0, None)
    df["du_dt"] = (du / dt_clip).astype(np.float64)
    df["r_du_dt"] = (r * df["du_dt"].to_numpy(dtype=np.float64)).astype(np.float64)

    df["u_in_dt"] = (
        df["u_in"].to_numpy(dtype=np.float64) * df["dt"].to_numpy(dtype=np.float64)
    ).astype(np.float64)
    df["u_in_dt_cum"] = g["u_in_dt"].cumsum().astype(np.float64)

    df["step_idx"] = g.cumcount().astype(np.float64)
    df["u_in_t"] = (
        df["u_in"].to_numpy(dtype=np.float64)
        * df["time_step"].to_numpy(dtype=np.float64)
    ).astype(np.float64)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.float64)
    last_uout1_time = df["time_step"].where(df["u_out"] == 1.0)
    last_uout1_time = (
        last_uout1_time.groupby(df["breath_id"], sort=False).ffill().fillna(0.0)
    )
    df["time_since_uout1"] = (df["time_step"] - last_uout1_time).astype(np.float64)

    c = np.clip(df["C"].to_numpy(dtype=np.float64), 1.0, None)
    r = np.clip(df["R"].to_numpy(dtype=np.float64), 1.0, None)

    df["x0"] = df["u_in"].to_numpy(dtype=np.float64) / c
    df["x1"] = df["u_in_cum"].to_numpy(dtype=np.float64) / c
    df["x2"] = df["u_in_lag1"].to_numpy(dtype=np.float64) / c
    df["x3"] = df["u_in_lag2"].to_numpy(dtype=np.float64) / c
    df["x4"] = df["u_in_cum_lag1"].to_numpy(dtype=np.float64) / c
    df["x5"] = df["u_in"].to_numpy(dtype=np.float64) / (c * np.sqrt(r))
    df["x6"] = df["time_step"].to_numpy(dtype=np.float64)
    df["x7"] = (df["u_in"].to_numpy(dtype=np.float64) == 0.0).astype(np.float64)

    df["x8"] = df["u_in_diff1_cum"].to_numpy(dtype=np.float64) / c
    df["x9"] = df["u_in_dt_cum"].to_numpy(dtype=np.float64) / c

    df["x10"] = df["step_idx"].to_numpy(dtype=np.float64) / 80.0  # normalize ~[0,1]
    df["x11"] = df["u_in_t"].to_numpy(dtype=np.float64) / c

    df["x12"] = df["dt"].to_numpy(dtype=np.float64)

    df["x13"] = df["u_out_lag1"].to_numpy(dtype=np.float64)
    df["x14"] = df["time_since_uout1"].to_numpy(dtype=np.float64)

    df["x15"] = (df["r_du_dt"].to_numpy(dtype=np.float64) / 1000.0).astype(np.float64)

    df = df.sort_values("_orig_order", kind="mergesort").drop(columns=["_orig_order"])
    return df


def _robust_fit_linear(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    Xd = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])

    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta

    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    s = 1.4826 * mad
    k = 1.345 * s
    w = 1.0 / np.maximum(1.0, np.abs(resid) / k)  # in (0,1]
    W = w[:, None]

    beta2, *_ = np.linalg.lstsq(Xd * W, y * w, rcond=None)
    return beta2  # includes intercept as beta2[0]


search_roots = [
    "../input",
    "/kaggle/input",
    "../kaggle/input",
]
cands = _discover_candidate_prediction_csvs(search_roots)

sample = pd.read_csv(SAMPLE_SUB_PATH)
sample = sample.sort_values("id").reset_index(drop=True)

test_df = pd.read_csv(
    TEST_PATH, usecols=["id", "breath_id", "time_step", "u_in", "u_out", "C", "R"]
)

if len(cands) >= 2:
    sub = blend(cands[0], cands[1], out_path="submission.csv")
elif len(cands) == 1:
    one = pd.read_csv(cands[0]).sort_values("id").reset_index(drop=True)
    if not np.array_equal(one["id"].to_numpy(), sample["id"].to_numpy()):
        one = one.set_index("id").reindex(sample["id"]).reset_index()
    one["pressure"] = one["pressure"].apply(find_nearest)
    one[["id", "pressure"]].to_csv("submission.csv", index=False)
    sub = one[["id", "pressure"]]
else:
    train_cols = ["breath_id", "time_step", "u_in", "u_out", "C", "R", "pressure"]
    tr = df_train[train_cols].copy()
    tr = _add_breath_features(tr)

    te = test_df.copy()
    te = _add_breath_features(te)

    feat_cols = [
        "x0",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
    ]

    insp_fit = tr.loc[(tr["u_out"] == 0)].copy()

    params = []
    for (r, c), gdf in insp_fit.groupby(["R", "C"], sort=False):
        X = gdf[feat_cols].to_numpy(dtype=np.float64)
        y = gdf["pressure"].to_numpy(dtype=np.float64)
        beta = _robust_fit_linear(X, y)
        params.append((int(r), int(c)) + tuple(beta.tolist()))
    params_df = pd.DataFrame(
        params,
        columns=["R", "C"] + [f"b{i}" for i in range(0, len(feat_cols) + 1)],
    )

    Xg = insp_fit[feat_cols].to_numpy(dtype=np.float64)
    yg = insp_fit["pressure"].to_numpy(dtype=np.float64)
    beta_g = _robust_fit_linear(Xg, yg)

    exp = tr.loc[tr["u_out"] == 1, ["R", "C", "pressure"]]
    exp_med = (
        exp.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "exp_pressure"})
    )
    exp_global = float(df_train.loc[df_train["u_out"] == 1, "pressure"].median())

    insp0 = tr.loc[(tr["u_out"] == 0) & (tr["u_in"] == 0), ["R", "C", "pressure"]]
    insp0_med = (
        insp0.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "insp0_pressure"})
    )
    insp0_global = float(
        df_train.loc[
            (df_train["u_out"] == 0) & (df_train["u_in"] == 0), "pressure"
        ].median()
    )

    merged = sample.merge(test_df, on="id", how="left")
    merged = merged.merge(te[["id"] + feat_cols], on="id", how="left")
    merged = merged.merge(params_df, on=["R", "C"], how="left")
    merged = merged.merge(exp_med, on=["R", "C"], how="left")
    merged = merged.merge(insp0_med, on=["R", "C"], how="left")

    beta_cols = [f"b{i}" for i in range(0, len(feat_cols) + 1)]
    for k, v in zip(beta_cols, beta_g.tolist()):
        merged[k] = merged[k].fillna(float(v))
    merged["exp_pressure"] = merged["exp_pressure"].fillna(exp_global)
    merged["insp0_pressure"] = merged["insp0_pressure"].fillna(insp0_global)

    X_test = merged[feat_cols].to_numpy(dtype=np.float64)
    b0 = merged["b0"].to_numpy(dtype=np.float64)
    B = merged[[f"b{i}" for i in range(1, len(feat_cols) + 1)]].to_numpy(
        dtype=np.float64
    )
    pred_insp = b0 + np.sum(B * X_test, axis=1)

    u_out_arr = merged["u_out"].to_numpy(dtype=np.int64)
    u_in_arr = merged["u_in"].to_numpy(dtype=np.float64)

    pred = pred_insp.copy()
    pred = np.where(
        u_out_arr == 1, merged["exp_pressure"].to_numpy(dtype=np.float64), pred
    )
    pred = np.where(
        (u_out_arr == 0) & (u_in_arr == 0),
        merged["insp0_pressure"].to_numpy(dtype=np.float64),
        pred,
    )

    merged["pressure"] = pred
    merged["pressure"] = merged["pressure"].apply(find_nearest)
    merged[["id", "pressure"]].to_csv("submission.csv", index=False)
    sub = merged[["id", "pressure"]]

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

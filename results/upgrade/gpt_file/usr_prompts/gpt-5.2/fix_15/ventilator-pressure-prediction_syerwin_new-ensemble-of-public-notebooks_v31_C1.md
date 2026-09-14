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

0.1496426051358578

# 6. Current score

8.50217

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.29564) has done: 'The notebook fails because it tries to read four external submission files that are not present in this Kaggle environment, so downstream blending variables are undefined. To keep the core “blend submissions” logic but make it runnable end-to-end, I (1) robustly load those files only if they exist, (2) fall back to a strong baseline: generate a prediction by mapping `u_in` to the nearest pressure seen in train for each `(R,C)` group (and a global fallback). This produces a valid `submission.csv` with the required `id,pressure` columns and should score far better than all-zeros, moving toward your target. The script also validates alignment by merging on `id` so row order mismatches won’t corrupt predictions.'
- What this solution (achieved 8.6643) has done: 'Your current fallback baseline is too weak because it only uses instantaneous `u_in` (binned) and ignores the strong temporal/physical structure of pressure (especially dependence on cumulative inhaled volume/flow history). To move the MAE down toward your target while keeping the same “no-model” mapping core idea, I keep the submission-blending branch intact, but substantially strengthen the fallback by adding a few standard Ventilator Pressure Prediction engineered features (cumulative `u_in`, lagged `u_in`, and `u_in` change) computed per `breath_id`, then doing the same group-median mapping (still a pure lookup/median approach, no training loop). I also avoid hard-setting `u_out==1` predictions to 0 (that can hurt because those rows are not scored and Kaggle may still expect realistic continuity), instead using the same mapping for all rows and letting the metric ignore expiratory phase. Finally, I make the `id` alignment explicit by building the prediction dataframe from `test` and merging into `sample_submission` by `id`.'
- What this solution (achieved 8.25906) has done: 'Your current fallback is a pure lookup/median mapper, but it’s being hurt by (1) very coarse quantization and (2) using time-dependent features (like cumsum) without also using the time index, which creates many-to-one collisions across different phases of the breath. I keep the same “group-median mapping” core logic, but make the mapping more phase-aware by adding a quantized `time_step` to the key and tightening the feature set to stable lags/diffs that generalize from train to test. I also make the quantization finer (especially for `u_in` and `time_step`) so the lookup is less lossy, while preserving the same merge/fillna fallback chain and submission writing behavior. These changes should legitimately reduce MAE (lower-is-better), moving the score toward your target without introducing any new model training.'
- What this solution (achieved 8.27526) has done: 'Your current score (8.25906 MAE; lower is better) is far from the target (~0.1496), so we need a meaningful but still “same-core-logic” improvement. I keep your non-ML lookup/median-mapping approach intact, but (1) make the mapping “breath-phase aware” by adding a quantized within-breath step index as a key (time_step quantization can collide slightly), and (2) snap predictions to the known discrete pressure grid from train (a standard Ventilator trick) to reduce MAE without changing the method. I also include `u_out_q` only in the key-building stage to reduce mismatched joins from slight distribution differences, while still training the maps on inspiratory rows (as you already do). The output remains a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 8.25872) has done: 'Your current MAE (8.275) is far above the target (0.1496), so we need a meaningful improvement while keeping the same “lookup/median mapping (no ML training)” core. The biggest gain with minimal change is to make the mapping explicitly *inspiratory-phase aware* by adding a quantized time key based on `time_step` (not just `step`), because the same `step` can correspond to slightly different times and dynamics across breaths. We also tighten/standardize quantization using integer bins via `np.rint` and add a very small extra fallback map that uses `time_step_q` + `u_in_q` (without lags) to reduce missing-join cases. Finally, we keep your pressure-grid snapping (good for this competition) and preserve the blending branch unchanged.'
- What this solution (achieved 8.35286) has done: 'Your current MAE (8.2587; lower is better) is far above the target (0.1496), and the main reason is that the fallback is still a coarse per-row lookup that cannot capture the strong within-breath dynamics. Keeping your core “no-ML lookup/median mapping + fallback chain + pressure-grid snapping” approach intact, I make the mapping explicitly within-breath-position aware by adding the `step` index into the keys (instead of relying on `time_step` alone), and I compute the maps on a stable `(R,C,step, u_in, lag, diff, cumsum)` quantized space. I also add one extra intermediate fallback map that uses `(R,C,step,u_in,cumsum)` to reduce missing-join frequency without changing the method. This should legitimately reduce MAE while keeping the same overall logic and still producing a valid `submission.csv`.'
- What this solution (achieved 8.38321) has done: 'Your current MAE (8.35; lower is better) is far from the target (0.1496), so we need a meaningful improvement while keeping your core “lookup/median mapping + fallback chain + pressure-grid snapping” approach. The biggest minimal win is to make the lookup keys more *breath-dynamics aware* without adding any ML: incorporate a physically-relevant feature (`u_in * dt` and its cumulative sum, i.e., approximate inhaled volume) and use it (quantized) in the primary and mid-level maps. This reduces collisions where the same `(step, u_in, lags)` occurs at different effective delivered volumes, which is a common cause of large errors in this competition. I also fix a small inconsistency: `time_step_q` was computed but unused; we use `dt`/volume instead (more stable than raw `time_step`), keeping runtime within limits and still writing a valid `submission.csv`.'
- What this solution (achieved 8.3558) has done: 'Your current MAE (8.383) is far worse than the target (0.1496), so we need a real accuracy lift while keeping the same “no-ML lookup/median mapping + fallback chain + pressure-grid snapping” core. The main issue is that your lookup keys are too detailed in some places (causing lots of missing joins) and not explicitly aligned to the *scored inspiratory region*, so the effective predictions often fall back to coarse/global medians. I keep the exact same median-lookup approach, but (1) add a very lightweight inspiratory-phase time index (`insp_step`) computed only over `u_out==0` and use it as an additional mapping axis, and (2) add one extra “high-coverage but still phase-aware” fallback map using `(R,C,insp_step,u_in)` so fewer rows drop to global. This should reduce the MAE meaningfully (lower-is-better) without introducing any model training or changing your blending branch behavior.'
- What this solution (achieved 8.38819) has done: 'Your current MAE (8.3558; lower is better) is far above the target (0.1496), so we need to improve accuracy meaningfully while keeping your existing “median lookup + fallback chain + pressure-grid snapping” approach. The largest minimal win is to stop throwing away expiratory rows in the training map: instead, train maps on all rows but include `u_out` in the key so inspiratory and expiratory dynamics don’t collide, which greatly improves coverage and reduces harmful fallbacks. In addition, use a strict within-breath `step` (0–79) as a core phase key for all rows (instead of `insp_step=-1` for expiratory), and add one high-coverage fallback keyed by `(R,C,u_out,step,u_in)` to reduce NaNs. These changes preserve your core logic (no ML training, same feature engineering style, same median-merge pipeline, same pressure snapping) but should move the MAE down substantially toward the target band.'
- What this solution (achieved 8.34971) has done: 'Your current MAE is far above the target, so we need a real but still “same core logic” lift inside your median-lookup pipeline (no ML). The biggest low-risk gain is to (1) build lookup maps primarily on the *scored inspiratory rows* (`u_out==0`) while still keeping a separate expiratory map for completeness, and (2) add a stronger, high-coverage inspiratory fallback keyed by `(R,C,step,u_in)` (without `u_in_dt_cumsum`) to avoid frequent NaN fallbacks caused by cumulative-feature mismatch. I also fix a key issue in your pressure snapping: use the train pressure *grid* directly but snap after blending inspiratory/expiratory predictions so inspiratory rows benefit most from grid alignment. These are minimal changes that preserve your approach (feature engineering + quantized group-median maps + fallback chain + pressure-grid snapping) and should reduce MAE (lower is better) toward your target.'
- What this solution (achieved 8.05712) has done: 'Your current MAE is far above the target, so we should improve accuracy while keeping the same core “median lookup maps + fallback chain + pressure-grid snapping” approach. The biggest issue is that your strongest key includes `u_in_dt_cumsum_q`, which is sensitive to tiny `dt` differences and causes many missing joins, forcing fallbacks and inflating MAE. I keep your feature set and merge-based pipeline, but (1) quantize `time_step` and compute a stable `t_insp` (time since start of inspiration) to use as a better phase key than `u_in_dt_cumsum`, and (2) slightly relax the primary key by dropping the cumulative term from the *top* map while keeping a mid-level cumulative/time-aware map for disambiguation. This increases match coverage on inspiratory rows (the only scored region) without changing the overall method, and still writes a valid `submission.csv`.'
- What this solution (achieved 8.10454) has done: 'Your current MAE is far above the target (lower is better), so we need a meaningful lift while keeping your exact “median-lookup maps + fallback chain + pressure-grid snapping” core intact. The biggest likely issue is key mismatch: your `t_insp` is currently computed as “time since first inspiratory sample” which can drift and fragment joins; replacing it with a stable per-breath inspiratory time (`t_insp = time_step` during inspiration, else 0) increases map coverage without changing the approach. Additionally, your quantization is relatively fine for `u_in` and `t_insp`, which creates sparse groups and forces fallbacks; slightly coarsening them (still deterministic) typically reduces MAE for lookup methods by increasing match rate. Finally, we keep expiratory handling and pressure-grid snapping exactly as you do, but we train the time-aware maps strictly on inspiratory rows and only use them for inspiratory predictions (as intended by the metric).'
- What this solution (achieved 8.49011) has done: 'Your current score is far above the target (lower-is-better), so we need a real accuracy lift while keeping your exact “median lookup maps + fallback chain + pressure-grid snapping” core intact. The biggest minimal win here is to explicitly use the competition’s key lung attributes by adding `R*C` (and `R/C`) and a more stable “delivered volume proxy” (`u_in` cumulative and `u_in*dt` cumulative) into an additional *high-priority* inspiratory lookup, while keeping your existing maps unchanged as fallbacks. This improves match specificity without making keys so sparse that you fall back to global medians, and it stays within the same no-ML group-median mapping approach. I also correct a likely data issue in your environment: the `id` range you printed (1–2000) indicates the CSV preview was truncated; so the code keeps merging by `id` (safe) and writes a valid `submission.csv` unchanged.'
- What this solution (achieved 8.50217) has done: 'Your current MAE is far above the target (lower is better), and the biggest likely issue is that the lookup keys are still too sparse/mismatched so many test rows fall back to coarse/global medians. Keeping the exact same “median lookup maps + fallback chain + pressure-grid snapping” core, I (1) slightly coarsen the most fragile quantizations (`t_insp`, `u_in_cumsum`, `R/C`) to increase join coverage, and (2) add one high-coverage, inspiratory-only map keyed by `(R,C,step,u_in)` but with *coarsened u_in* as an intermediate fallback (reduces NaNs without changing the approach). I also ensure the lung-attribute physics maps join on consistently-quantized columns (using the already-created `_q` columns only), preserving semantics while reducing silent mismatches. The output remains `submission.csv` with `id,pressure` and the blend-files branch remains unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

blend_paths = [
    "../input/random-weights-blending-tool-ventilator-pressure/rwb 125 loops.csv",
    "../input/blend-of-blend-of-blend-of-blend-of-blend-of-ble/submission.csv",
    "../input/ensemble-folds-with-median-0-153/submission_median_round_LB153.csv",
    "../input/vpp-lstm-baseline-median-pp/submission.csv",
]

loaded_subs = []
for p in blend_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        if "id" in df.columns and "pressure" in df.columns:
            loaded_subs.append((p, df[["id", "pressure"]].copy()))

if len(loaded_subs) > 0:
    base_weights = np.array([0.2, 0.7, 0.1, 0.0], dtype=float)
    weights = base_weights[: len(loaded_subs)]
    if weights.sum() == 0:
        weights = np.ones(len(loaded_subs), dtype=float)
    weights = weights / weights.sum()

    merged = sub[["id"]].copy()
    for i, (_, df) in enumerate(loaded_subs):
        merged = merged.merge(
            df.rename(columns={"pressure": f"pressure_{i}"}), on="id", how="left"
        )

    pred = np.zeros(len(merged), dtype=float)
    for i in range(len(loaded_subs)):
        pred += merged[f"pressure_{i}"].fillna(0).to_numpy(dtype=float) * weights[i]

    sub["pressure"] = pred

else:
    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "time_step", "R", "C", "u_in", "u_out", "pressure"],
    )
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "time_step", "R", "C", "u_in", "u_out"]
    )

    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
        g = df.groupby("breath_id", sort=False)

        df["step"] = g.cumcount().astype(np.int16)

        df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
        df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
        df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
        df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

        df["u_in_cumsum"] = g["u_in"].cumsum()

        df["dt"] = g["time_step"].diff().fillna(0.0)
        df["u_in_dt"] = df["u_in"] * df["dt"]
        df["u_in_dt_cumsum"] = g["u_in_dt"].cumsum()

        df["insp_step"] = (
            df["u_out"]
            .eq(0)
            .groupby(df["breath_id"], sort=False)
            .cumsum()
            .astype(np.int16)
            - 1
        )
        df.loc[df["u_out"].ne(0), "insp_step"] = -1

        df["t_insp"] = np.where(
            df["u_out"].to_numpy(dtype=int) == 0, df["time_step"], 0.0
        )

        df["RC"] = (df["R"].astype(np.int32) * df["C"].astype(np.int32)).astype(
            np.int32
        )
        df["R_div_C"] = (
            df["R"].astype(np.float32) / df["C"].astype(np.float32)
        ).astype(np.float32)

        return df

    train_f = add_features(train)
    test_f = add_features(test)

    q_step = 1
    q_uout = 1

    q_u = 1.0
    q_diff = 1.0
    q_dt_cum = 0.10

    q_t = 0.10  # was 0.05

    q_u_cum = 10.0  # was 5.0

    q_rc = 50.0

    q_rdivc = 0.2  # was 0.1

    q_u_coarse = 2.0

    def qbin(arr, q):
        return np.rint(arr / q).astype(np.int32)

    for col, q in [
        ("step", q_step),
        ("u_out", q_uout),
        ("u_in", q_u),
        ("u_in_lag1", q_u),
        ("u_in_diff1", q_diff),
        ("u_in_diff2", q_diff),
        ("u_in_dt_cumsum", q_dt_cum),
        ("u_in_cumsum", q_u_cum),
        ("time_step", q_t),
        ("t_insp", q_t),
        ("RC", q_rc),
        ("R_div_C", q_rdivc),
    ]:
        train_f[col + "_q"] = qbin(train_f[col].to_numpy(dtype=float), q)
        test_f[col + "_q"] = qbin(test_f[col].to_numpy(dtype=float), q)

    train_f["u_in_q2"] = qbin(train_f["u_in"].to_numpy(dtype=float), q_u_coarse)
    test_f["u_in_q2"] = qbin(test_f["u_in"].to_numpy(dtype=float), q_u_coarse)

    train_insp = train_f.loc[train_f["u_out"].eq(0)].copy()
    train_exp = train_f.loc[train_f["u_out"].ne(0)].copy()

    global_const_all = float(train_f["pressure"].median())
    global_const_insp = float(train_insp["pressure"].median())
    global_const_exp = (
        float(train_exp["pressure"].median()) if len(train_exp) else global_const_all
    )

    key_cols_main_insp = [
        "R",
        "C",
        "step_q",
        "u_in_q",
        "u_in_lag1_q",
        "u_in_diff1_q",
    ]

    key_cols_time_insp = [
        "R",
        "C",
        "t_insp_q",
        "u_in_q",
        "u_in_lag1_q",
        "u_in_diff1_q",
    ]

    grp_map_insp_main = (
        train_insp.groupby(key_cols_main_insp, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_insp_main"})
    )

    grp_map_insp_time = (
        train_insp.groupby(key_cols_time_insp, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_insp_time"})
    )

    grp_map_insp_step_u = (
        train_insp.groupby(["R", "C", "step_q", "u_in_q"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_insp_step_u"})
    )

    grp_map_insp_mid = (
        train_insp.groupby(["R", "C", "step_q", "u_in_q", "t_insp_q"], sort=False)[
            "pressure"
        ]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_insp_mid"})
    )

    grp_map_insp2 = (
        train_insp.groupby(
            ["R", "C", "step_q", "u_in_q", "u_in_lag1_q", "u_in_diff1_q"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_insp2"})
    )

    global_map_insp = (
        train_insp.groupby(["step_q", "u_in_q"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_insp_global"})
    )

    grp_map_insp_phys = (
        train_insp.groupby(
            ["RC_q", "R_div_C_q", "step_q", "u_in_q", "u_in_cumsum_q"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_insp_phys"})
    )

    grp_map_insp_phys2 = (
        train_insp.groupby(["RC_q", "step_q", "u_in_q", "u_in_cumsum_q"], sort=False)[
            "pressure"
        ]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_insp_phys2"})
    )

    grp_map_insp_step_u2 = (
        train_insp.groupby(["R", "C", "step_q", "u_in_q2"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_insp_step_u2"})
    )

    grp_map_exp_step_u = (
        train_exp.groupby(["R", "C", "step_q", "u_in_q"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_exp_step_u"})
    )
    global_map_exp = (
        train_exp.groupby(["step_q", "u_in_q"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_exp_global"})
    )

    test_pred = test_f[
        [
            "id",
            "R",
            "C",
            "u_out",
            "u_out_q",
            "step_q",
            "u_in_q",
            "u_in_q2",
            "u_in_lag1_q",
            "u_in_diff1_q",
            "u_in_dt_cumsum_q",
            "u_in_cumsum_q",
            "time_step_q",
            "t_insp_q",
            "RC_q",
            "R_div_C_q",
        ]
    ].copy()

    test_pred = test_pred.merge(grp_map_insp_main, on=key_cols_main_insp, how="left")
    test_pred = test_pred.merge(grp_map_insp_time, on=key_cols_time_insp, how="left")
    test_pred = test_pred.merge(
        grp_map_insp_mid, on=["R", "C", "step_q", "u_in_q", "t_insp_q"], how="left"
    )
    test_pred = test_pred.merge(
        grp_map_insp2,
        on=["R", "C", "step_q", "u_in_q", "u_in_lag1_q", "u_in_diff1_q"],
        how="left",
    )
    test_pred = test_pred.merge(
        grp_map_insp_step_u, on=["R", "C", "step_q", "u_in_q"], how="left"
    )

    test_pred = test_pred.merge(
        grp_map_insp_step_u2, on=["R", "C", "step_q", "u_in_q2"], how="left"
    )

    test_pred = test_pred.merge(global_map_insp, on=["step_q", "u_in_q"], how="left")

    test_pred = test_pred.merge(
        grp_map_insp_phys,
        on=["RC_q", "R_div_C_q", "step_q", "u_in_q", "u_in_cumsum_q"],
        how="left",
    )
    test_pred = test_pred.merge(
        grp_map_insp_phys2,
        on=["RC_q", "step_q", "u_in_q", "u_in_cumsum_q"],
        how="left",
    )

    test_pred = test_pred.merge(
        grp_map_exp_step_u, on=["R", "C", "step_q", "u_in_q"], how="left"
    )
    test_pred = test_pred.merge(global_map_exp, on=["step_q", "u_in_q"], how="left")

    pred_insp = (
        test_pred["pred_insp_phys"]
        .fillna(test_pred["pred_insp_phys2"])
        .fillna(test_pred["pred_insp_main"])
        .fillna(test_pred["pred_insp_time"])
        .fillna(test_pred["pred_insp_mid"])
        .fillna(test_pred["pred_insp2"])
        .fillna(test_pred["pred_insp_step_u"])
        .fillna(test_pred["pred_insp_step_u2"])
        .fillna(test_pred["pred_insp_global"])
        .fillna(global_const_insp)
        .to_numpy(dtype=float)
    )
    pred_exp = (
        test_pred["pred_exp_step_u"]
        .fillna(test_pred["pred_exp_global"])
        .fillna(global_const_exp)
        .to_numpy(dtype=float)
    )

    pred = np.where(test_pred["u_out"].to_numpy(dtype=int) == 0, pred_insp, pred_exp)

    pressure_grid = np.sort(train["pressure"].unique())
    idx = np.searchsorted(pressure_grid, pred, side="left")
    idx = np.clip(idx, 0, len(pressure_grid) - 1)
    left = pressure_grid[np.clip(idx - 1, 0, len(pressure_grid) - 1)]
    right = pressure_grid[idx]
    choose_right = (idx == 0) | (
        (idx > 0) & (np.abs(right - pred) <= np.abs(pred - left))
    )
    pred_snapped = np.where(choose_right, right, left)

    out = pd.DataFrame({"id": test["id"].to_numpy(), "pressure": pred_snapped})
    sub = sub[["id"]].merge(out, on="id", how="left")
    sub["pressure"] = sub["pressure"].fillna(global_const_all)

sub.to_csv("submission.csv", index=False)
sub.head(5)

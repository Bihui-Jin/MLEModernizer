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

0.1567

# 6. Current score

7.1924

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The crash happens because cell 1 tries to read four Kaggle Dataset input files that do not exist in this environment under `../input/...`. To keep the notebook logic intact (an ensemble of four submissions), the minimal fix is to point those reads to files that *do* exist here. The only available submission-like CSV in the provided filesystem is the competition `sample_submission.csv`, so we load that same file for `sub_1..sub_4` as a deterministic fallback, preserving the expected columns and shapes for cell 2. This unblocks execution without changing any later code or interfaces.'
- What this solution (achieved 5.8959) has done: 'Your current score is very far from the target because the ensemble inputs are missing, so you’re effectively submitting all zeros from `sample_submission.csv`. To move the score sharply toward the target with minimal changes and without changing the “blend submissions then write CSV” core logic, I replace the missing external submission fallbacks with a simple, legitimate baseline model trained on the provided `train.csv` and used to generate four diverse-but-related predictions. Specifically, I fit a lightweight regularized linear regression on inspiratory-phase data using basic features (`u_in`, `time_step`, `R`, `C`, `u_out`) and a few safe interactions, then create the four “submissions” via slightly different feature sets/regularization to preserve the ensemble structure. This keeps the approach as “ensemble of four submissions” while producing non-trivial predictions from available data, yielding a much lower MAE and moving toward the 0.1567 target (though likely not all the way).'
- What this solution (achieved 3.45044) has done: 'Your current MAE (5.8959) is still far above the target (0.1567), so we should improve the baseline predictions while keeping the “train simple models → make four submissions → weighted blend → write submission.csv” core logic intact. The biggest issue is setting expiratory-phase (`u_out==1`) predictions to 0; since expiratory phase is not scored, we should instead keep reasonable predictions there so the model isn’t forced into an unnatural discontinuity. Next, we add a minimal amount of time-series signal without changing the approach: include per-breath cumulative `u_in` (“volume proxy”) and lagged `u_in`/`u_out` features; these are standard for this competition and substantially reduce MAE with linear models. Finally, we clip predictions to the observed training pressure range to avoid outliers that hurt MAE.'
- What this solution (achieved 3.4504) has done: 'Your current MAE (3.45044) is still far above the target (0.1567), so we need a meaningful but still minimal improvement while keeping the same “train four simple models → weighted blend → write submission.csv” structure. The biggest gain available without changing the modeling approach is to respect the competition’s discrete pressure levels: we can quantize each model’s predictions to the nearest pressure value seen in training, which typically reduces MAE a lot for this dataset. We also ensure the `id` alignment is stable by building predictions in the sorted test order and then mapping back to the sample submission order without relying on potentially non-unique indices. These changes keep the same Ridge models and features, but improve post-processing to better match the evaluation target.'
- What this solution (achieved 3.78963) has done: 'We keep your existing “fit four Ridge models → blend → snap to pressure grid → write submission.csv” core logic intact, but fix two high-impact issues that are currently inflating MAE. First, you are fitting only on inspiratory rows (`u_out==0`) yet predicting all rows; this mismatched distribution hurts the learned mapping, so we train on all rows but weight expiratory rows to zero in the loss (via `sample_weight`), which matches the metric (expiratory not scored) without changing the model family. Second, we add a minimal, standard one-step “breath anchoring” correction: within each breath, shift predictions so that the first timestep matches a simple per-(R,C) baseline pressure (median at time_step==0), then re-snap to the pressure grid; this reduces per-breath bias and typically moves MAE sharply toward the target with minimal risk. These are small, metric-aligned adjustments and should move your 3.4504 much closer to 0.1567 without changing file paths or submission format.'
- What this solution (achieved 3.65521) has done: 'The crash happens inside `add_group_features`: it tries to access `g["u_in_cum"]` via the groupby object `g` before `u_in_cum` exists on the original `df` used to create `g`, so pandas raises `KeyError: 'Column not found: u_in_cum'`. The minimal fix is to compute lagged versions of `u_in_cum` from the already-created `out["u_in_cum"]` by grouping `out` (or by grouping the series directly), instead of indexing `g["u_in_cum"]`. This preserves the feature logic and values while avoiding the missing-column access. No other cells need changes, and the outputs (`sub_1..sub_4`, `pressure_values`) remain the same interface for cell 2. Assumption: only this KeyError blocks execution; the data files load correctly from the provided paths.'
- What this solution (achieved 3.59847) has done: 'The crash happens inside `add_group_features`: it tries to access `g["u_in_cum"]` from the original groupby object `g`, but `u_in_cum` is a newly created column on `out` and doesn’t exist in the original `df`. The minimal fix is to compute lags of `u_in_cum` from `out.groupby("breath_id")` (or by grouping the `out["u_in_cum"]` series), instead of using the earlier `g`. This preserves the exact intended feature logic and keeps all downstream variables (`train_f`, `test_f`, `pressure_values`, `sub_1..sub_4`) unchanged in interface. No other cells need changes.'
- What this solution (achieved 7.1924) has done: 'Your current MAE (3.59847) is far above the target (0.1567), so we should make a minimal, metric-aligned improvement that keeps your existing “fit four Ridge models → blend → snap to pressure grid → write submission.csv” core logic. The biggest issue is that the model is trying to learn pressure directly, but the evaluation ignores expiratory (`u_out==1`) rows; with plain sample weights the Ridge still optimizes an objective that doesn’t explicitly match “MAE on inspiratory only.” With minimal change, we can keep Ridge but shift the target to predict *delta from a strong per-(R,C,time_step,u_in)* baseline computed from training inspiratory rows, and then add the baseline back at inference; this typically reduces systematic error a lot for linear models on this competition. We keep all existing features, ensemble structure, anchoring, and grid-snapping; we only add the baseline and switch the learned target from absolute pressure to residuals (still Ridge regression).'

# 9. Code solution

## === cell 0
import pandas as pd



## === cell 1
import os
import numpy as np

TRAIN_PATH = "/kaggle/data/train.csv"
TEST_PATH = "/kaggle/data/test.csv"
SAMPLE_SUB_PATH = "/kaggle/data/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

sub = pd.read_csv(SAMPLE_SUB_PATH)

paths = [
    "../input/tensorflow-bidirectional-lstm-custom-mae-loss/submission.csv",
    "../input/finetune-of-tensorflow-bidirectional-lstm/submission.csv",
    "../input/a-dummy-approach-to-improve-your-score-postprocess/submission.csv",
    "../input/ensemble-folds-with-median-0-153/submission_median_round_LB153.csv",
]

have_any = any(os.path.exists(p) for p in paths)

if have_any:
    sub_1 = pd.read_csv(paths[0]) if os.path.exists(paths[0]) else sub.copy()
    sub_2 = pd.read_csv(paths[1]) if os.path.exists(paths[1]) else sub.copy()
    sub_3 = pd.read_csv(paths[2]) if os.path.exists(paths[2]) else sub.copy()
    sub_4 = pd.read_csv(paths[3]) if os.path.exists(paths[3]) else sub.copy()
else:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    p_min = float(train["pressure"].min())
    p_max = float(train["pressure"].max())

    pressure_values = np.sort(train["pressure"].unique().astype(np.float32))

    def snap_to_pressure_grid(preds: np.ndarray) -> np.ndarray:
        preds = preds.astype(np.float32)
        preds = np.clip(preds, p_min, p_max).astype(np.float32)
        idx = np.searchsorted(pressure_values, preds, side="left")
        idx = np.clip(idx, 0, len(pressure_values) - 1)
        prev_idx = np.clip(idx - 1, 0, len(pressure_values) - 1)
        next_val = pressure_values[idx]
        prev_val = pressure_values[prev_idx]
        choose_prev = np.abs(preds - prev_val) <= np.abs(next_val - preds)
        snapped = np.where(choose_prev, prev_val, next_val).astype(np.float32)
        return snapped

    def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("breath_id", sort=False)
        out = df.copy()

        out["dt"] = g["time_step"].diff().fillna(0.0).astype(np.float32)

        out["u_in_cum"] = (
            (out["u_in"] * out["dt"])
            .groupby(out["breath_id"], sort=False)
            .cumsum()
            .astype(np.float32)
        )

        out["u_in_sum"] = g["u_in"].cumsum().astype(np.float32)

        out["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
        out["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)
        out["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int16)

        out["du_in"] = (out["u_in"] - out["u_in_lag1"]).astype(np.float32)

        out["u_in_cum_lag1"] = (
            out["u_in_cum"]
            .groupby(out["breath_id"], sort=False)
            .shift(1)
            .fillna(0.0)
            .astype(np.float32)
        )
        out["u_in_cum_lag2"] = (
            out["u_in_cum"]
            .groupby(out["breath_id"], sort=False)
            .shift(2)
            .fillna(0.0)
            .astype(np.float32)
        )
        out["du_in_cum"] = (out["u_in_cum"] - out["u_in_cum_lag1"]).astype(np.float32)

        out["u_in_cummean"] = (
            g["u_in"]
            .transform(
                lambda s: (s.cumsum() / (np.arange(len(s), dtype=np.float32) + 1.0))
            )
            .astype(np.float32)
        )

        return out

    train_f = add_group_features(train)
    test_f = add_group_features(test)

    train_insp = train_f[train_f["u_out"].values == 0].copy()
    rc_t_uin_baseline = (
        train_insp.groupby(["R", "C", "time_step", "u_in"], sort=False)["pressure"]
        .median()
        .astype(np.float32)
    )
    rc_t_baseline = (
        train_insp.groupby(["R", "C", "time_step"], sort=False)["pressure"]
        .median()
        .astype(np.float32)
    )
    rc_baseline_static = (
        train_insp.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .astype(np.float32)
    )
    global_baseline = float(train_insp["pressure"].median())

    def baseline_pressure_for_rows(df: pd.DataFrame) -> np.ndarray:
        keys1 = list(
            zip(
                df["R"].values,
                df["C"].values,
                df["time_step"].values,
                df["u_in"].values,
            )
        )
        b1 = np.array(
            [float(rc_t_uin_baseline.get(k, np.nan)) for k in keys1], dtype=np.float32
        )

        miss1 = np.isnan(b1)
        if miss1.any():
            keys2 = list(
                zip(
                    df.loc[miss1, "R"].values,
                    df.loc[miss1, "C"].values,
                    df.loc[miss1, "time_step"].values,
                )
            )
            b2 = np.array(
                [float(rc_t_baseline.get(k, np.nan)) for k in keys2], dtype=np.float32
            )
            b1[miss1] = b2

        miss2 = np.isnan(b1)
        if miss2.any():
            keys3 = list(zip(df.loc[miss2, "R"].values, df.loc[miss2, "C"].values))
            b3 = np.array(
                [float(rc_baseline_static.get(k, global_baseline)) for k in keys3],
                dtype=np.float32,
            )
            b1[miss2] = b3

        b1 = np.where(np.isnan(b1), np.float32(global_baseline), b1).astype(np.float32)
        return b1

    base_train = baseline_pressure_for_rows(train_f)
    base_test = baseline_pressure_for_rows(test_f)

    y = (train_f["pressure"].astype(np.float32).values - base_train).astype(np.float32)
    sample_weight = (train_f["u_out"].values == 0).astype(np.float32)

    first_train = (
        train_f.groupby("breath_id", sort=False).head(1)[["R", "C", "pressure"]].copy()
    )
    rc_anchor = (
        first_train.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .astype(np.float32)
    )
    global_anchor = float(first_train["pressure"].median())

    def make_features(df: pd.DataFrame, variant: int) -> pd.DataFrame:
        X = df[
            [
                "u_in",
                "u_out",
                "time_step",
                "R",
                "C",
                "dt",
                "u_in_cum",
                "u_in_sum",
                "u_in_lag1",
                "u_in_lag2",
                "u_out_lag1",
                "du_in",
                "u_in_cum_lag1",
                "u_in_cum_lag2",
                "du_in_cum",
                "u_in_cummean",
            ]
        ].copy()

        if variant in (2, 3, 4):
            X["u_in_x_time"] = (X["u_in"] * X["time_step"]).astype(np.float32)
        if variant in (3, 4):
            X["u_in_div_C"] = (X["u_in"] / X["C"].astype(float)).astype(np.float32)
            X["u_in_cum_div_C"] = (X["u_in_cum"] / X["C"].astype(float)).astype(
                np.float32
            )
            X["u_in_sum_div_C"] = (X["u_in_sum"] / X["C"].astype(float)).astype(
                np.float32
            )
        if variant in (4,):
            X["u_in_x_R"] = (X["u_in"] * X["R"]).astype(np.float32)
            X["u_in_cum_x_R"] = (X["u_in_cum"] * X["R"]).astype(np.float32)
            X["u_in_sum_x_R"] = (X["u_in_sum"] * X["R"]).astype(np.float32)
        return X

    def fit_predict(variant: int, alpha: float) -> np.ndarray:
        X_tr = make_features(train_f, variant)
        X_te = make_features(test_f, variant)

        num_cols = list(X_tr.columns)

        model = Pipeline(
            steps=[
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=alpha, random_state=42)),
            ]
        )

        model.fit(X_tr[num_cols].values, y, ridge__sample_weight=sample_weight)

        preds = (
            model.predict(X_te[num_cols].values).astype(np.float32) + base_test
        ).astype(np.float32)

        pred_df_local = pd.DataFrame(
            {
                "breath_id": test_f["breath_id"].values,
                "R": test_f["R"].values,
                "C": test_f["C"].values,
                "pred": preds,
            }
        )
        first_test = (
            pred_df_local.groupby("breath_id", sort=False)
            .head(1)[["breath_id", "R", "C", "pred"]]
            .copy()
        )

        if len(first_test) > 0:
            rc_key = list(zip(first_test["R"].values, first_test["C"].values))
            baseline_vals = np.array(
                [float(rc_anchor.get(k, global_anchor)) for k in rc_key],
                dtype=np.float32,
            )
            delta = (
                baseline_vals - first_test["pred"].values.astype(np.float32)
            ).astype(np.float32)
            delta_by_breath = dict(zip(first_test["breath_id"].values, delta))
            preds = preds + np.array(
                [
                    delta_by_breath.get(bid, 0.0)
                    for bid in pred_df_local["breath_id"].values
                ],
                dtype=np.float32,
            )

        preds = snap_to_pressure_grid(preds)
        return preds

    pred1 = fit_predict(variant=1, alpha=2.0)
    pred2 = fit_predict(variant=2, alpha=5.0)
    pred3 = fit_predict(variant=3, alpha=10.0)
    pred4 = fit_predict(variant=4, alpha=20.0)

    pred_df = pd.DataFrame(
        {
            "id": test_f["id"].values,
            "pred1": pred1,
            "pred2": pred2,
            "pred3": pred3,
            "pred4": pred4,
        }
    )

    sub_1 = sub.merge(pred_df[["id", "pred1"]], on="id", how="left")
    sub_2 = sub.merge(pred_df[["id", "pred2"]], on="id", how="left")
    sub_3 = sub.merge(pred_df[["id", "pred3"]], on="id", how="left")
    sub_4 = sub.merge(pred_df[["id", "pred4"]], on="id", how="left")

    for s, col in [
        (sub_1, "pred1"),
        (sub_2, "pred2"),
        (sub_3, "pred3"),
        (sub_4, "pred4"),
    ]:
        s[col] = s[col].fillna(global_baseline).astype(np.float32)

    sub_1["pressure"] = sub_1["pred1"].astype(np.float32)
    sub_2["pressure"] = sub_2["pred2"].astype(np.float32)
    sub_3["pressure"] = sub_3["pred3"].astype(np.float32)
    sub_4["pressure"] = sub_4["pred4"].astype(np.float32)

    sub_1 = sub_1[["id", "pressure"]]
    sub_2 = sub_2[["id", "pressure"]]
    sub_3 = sub_3[["id", "pressure"]]
    sub_4 = sub_4[["id", "pressure"]]



## === cell 2
sub["pressure"] = (
    (sub_1["pressure"].values * 0.1)
    + (sub_2["pressure"].values * 0.2)
    + (sub_3["pressure"].values * 0.2)
    + (sub_4["pressure"].values * 0.5)
).astype(np.float32)

if "pressure_values" in globals():
    pv = pressure_values  # from training
    idx = np.searchsorted(pv, sub["pressure"].values.astype(np.float32), side="left")
    idx = np.clip(idx, 0, len(pv) - 1)
    prev_idx = np.clip(idx - 1, 0, len(pv) - 1)
    next_val = pv[idx]
    prev_val = pv[prev_idx]
    x = sub["pressure"].values.astype(np.float32)
    choose_prev = np.abs(x - prev_val) <= np.abs(next_val - x)
    sub["pressure"] = np.where(choose_prev, prev_val, next_val).astype(np.float32)

sub.to_csv("submission.csv", index=False)
sub.head(5)

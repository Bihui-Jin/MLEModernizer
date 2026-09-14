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

0.1565308632644423

# 6. Current score

1.11513

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.35471) has done: 'Your notebook fails because it tries to read out-of-environment Kaggle Dataset paths (e.g., `../input/ventilator-pressure-prediction-lstm-gpu-infer/...`) that don’t exist here, so the ensemble submissions can’t be loaded. To make this run end-to-end and generate a valid `submission.csv`, I replace the missing external submission inputs with a minimal in-notebook baseline model trained from `train.csv` and used to predict `test.csv`. Since no prior score was produced, this change is necessary (not optional) to yield a submission; it keeps the approach simple and stable in the provided package constraints (pandas + scikit-learn). The output is written with the correct `id,pressure` columns and row alignment to the test file.'
- What this solution (achieved 6.65982) has done: 'Your current score is far worse than the target (lower is better), so we should improve materially while keeping the same overall approach (feature engineering + scikit-learn regressor). The biggest issue for this competition is that MAE is computed only on inspiratory timesteps (`u_out==0`), so we can move predictions for expiratory timesteps toward a safe value without harming the metric much and sometimes improving generalization; additionally, this task benefits from grouping by lung attributes, so we keep the same model type but train separate models per `(R, C)` pair. Finally, we add a small, deterministic validation split by `breath_id` and report MAE on `u_out==0` so you can sanity-check that the changes move in the right direction before submitting.'
- What this solution (achieved 6.39417) has done: 'We keep your current feature set and HistGradientBoostingRegressor approach, but fix two score-critical mismatches with the competition: (1) the evaluation ignores expiratory steps, so setting `u_out==1` predictions to a constant 0 can create unrealistic discontinuities—use the model’s predicted value (or carry-forward) instead; and (2) pressure takes on a discrete grid in this dataset, so snapping predictions to the nearest allowed pressure value typically reduces MAE without changing the core model. We also train with `sample_weight` so inspiratory timesteps (`u_out==0`) are emphasized in the absolute-error objective, aligning training more tightly with the metric while preserving the same model family and training loop. Finally, we keep the same group-by `(R,C)` modeling and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 6.41687) has done: 'Your score is far above (worse than) the target, so we should improve materially without changing the core approach (same features + per-(R,C) HistGradientBoostingRegressor with absolute-error loss). The biggest issue is that `id` is not globally unique in this dataset (it repeats 1–80 for every breath), so your submission is misaligned and gets punished heavily; we fix this by using the `id` column from `sample_submission.csv` (which is globally unique) and enforcing exact row-count/alignment. Next, we align the training objective more closely to the metric by giving expiratory timesteps (`u_out==1`, not scored) essentially zero weight (but not exactly zero, to keep stability). Finally, we keep your pressure-grid snapping, but make it fast and deterministic, and add a couple of guard-asserts to ensure the output schema matches Kaggle exactly.'
- What this solution (achieved 6.41819) has done: 'Your current score is much worse than the target (lower is better), and the main reason is still output misalignment: you’re pairing predictions in test row-order with `sample_submission` ids, but the competition expects predictions matched to the **test.csv `id`**. I make the smallest fix to guarantee correct alignment by building the submission directly from `test["id"]` (and sorting by `id` to match the sample). Since the metric ignores `u_out==1`, I also set expiratory predictions to a stable, reasonable value (the last predicted inspiratory pressure per breath) to avoid noisy, unconstrained outputs without changing your model/training core. Everything else (features, per-(R,C) HGBR with absolute-error, pressure-grid snapping) stays the same.'
- What this solution (achieved 6.41819) has done: 'Your MAE is still extremely high because the inference-time “fill u_out==1 with last inspiratory per breath” step is currently misaligned: after sorting `tp` you later do `tp.sort_index()` which restores the *original row indices* (not the original test row order), so many predictions get attached to the wrong `id`. I make the smallest fix by carrying the original test row position through the `tp` dataframe and then restoring predictions by that position (guaranteeing correct alignment to `test["id"]`). Everything else (features, per-(R,C) HistGradientBoostingRegressor, absolute-error loss, snapping to pressure grid, weights) stays the same to preserve core logic while materially improving score toward your target. This should move you from a “broken alignment” regime (very large MAE) back to a reasonable baseline score.'
- What this solution (achieved 9.54856) has done: 'Your current score is far worse than the target, and with the present model family the most likely cause is still that predictions are being generated in a different row order than `test.csv` and then attached to the wrong `id` (sorting inside feature engineering changes row order). I make the smallest alignment-safe change: carry the original test row order through feature engineering and always write predictions back in that original order before creating `submission.csv`. I keep your per-(R,C) HistGradientBoostingRegressor setup, features, absolute-error loss, weighting, and pressure-grid snapping identical. This should materially reduce MAE by fixing row/id mismatches without altering the modeling logic.'
- What this solution (achieved 1.17478) has done: 'Your current MAE is far worse than the target, so the safest way to move toward the target without changing your model family is to fix two score-critical issues: (1) you are asserting `test["id"].is_unique`, but in this competition `id` repeats within each breath in many exports/environments—so we must align predictions to the `sample_submission` row order/ids instead of relying on `test["id"]`; and (2) your test-time group prediction uses `groupby().indices` then indexes `X_test.loc[idx]`, which is label-based and can silently misalign when the index is not a clean RangeIndex after sorting—so we force a stable positional index (`RangeIndex`) after feature engineering and use that consistently. These are minimal alignment/stability changes that preserve your features, per-(R,C) HGBR setup, absolute-error loss, sample weights, and pressure-grid snapping. The result should materially reduce MAE by ensuring each prediction lands on the correct submission row/id.'
- What this solution (achieved 1.17711) has done: 'Your current score is still far from the target (lower is better), so the most likely remaining issue is not the model but silent row misalignment introduced by feature engineering: you assign `row_pos` before sorting, then `reset_index()` destroys that original mapping and the stored `row_pos` becomes incorrect. I make the smallest alignment-safe fix by creating `row_pos` *before* any sorting and then carrying it through sorting without resetting it away, so predictions can be restored to the exact original `test.csv` row order deterministically. I also make the per-(R,C) training loop use the group indices directly (instead of recomputing boolean masks) to avoid subtle shape/order mistakes, while keeping the exact same model family, loss, features, weights, and snapping. This should materially reduce MAE by ensuring each predicted pressure lands on the correct submission row/id.'
- What this solution (achieved 1.55363) has done: 'I fix the KeyError in the per-(R,C) training loop by ensuring that the group indices are aligned to the *same index space* as the train/val subsets (right now you’re using full-data indices to index into the subset position maps). This is a correctness/stability fix that should also materially improve score because the current training/validation routing is broken (many samples never get trained/predicted by the intended group model). I keep the same feature engineering, per-(R,C) HistGradientBoostingRegressor setup, absolute-error loss, weighting, and pressure-grid snapping intact. Finally, I keep the submission writing logic unchanged except for small guards to ensure the pipeline always finishes and writes a valid `submission.csv`.'
- What this solution (achieved 1.51908) has done: 'Your current score (1.55363, lower-is-better) is still far above the target (~0.1565), and the remaining biggest “minimal-change” win is to align training with the metric more closely: Kaggle scores **only inspiratory** timesteps (`u_out==0`), but your model still trains on expiratory rows with non-trivial weight (0.01), which can noticeably hurt. I reduce expiratory sample weight to a much smaller value to effectively focus the absolute-error objective on the scored region while keeping the same model family, features, and training loop. I also snap **after** the `u_out==1` fill (so filled values are on-grid too), which is consistent with your existing discretization step and can shave MAE without changing core logic. Everything else (per-(R,C) HGBR, feature engineering, validation scheme, and submission alignment via `row_pos`) stays intact.'
- What this solution (achieved 1.54363) has done: 'Your current score is still much worse than the target (lower is better), so we should make a small, metric-aligned change that improves MAE without changing your model family or feature set. The key remaining mismatch is that you train on all timesteps (including expiratory) with a nonzero weight and use raw `pressure` as the target, while the metric only scores inspiratory (`u_out==0`) and pressures are discrete; we can keep the exact same model but (a) train only on inspiratory rows and (b) predict *grid class indices* (via the existing pressure grid) and then map back to pressure—this often reduces MAE on this competition while staying within your current approach (same regressor, same loss, same loop). We also keep your per-(R,C) training and your safe u_out==1 fill, but we snap using the predicted class index mapping (deterministic). This should move the score materially toward your target without altering the overall pipeline structure.'
- What this solution (achieved 1.11513) has done: 'Your current MAE (1.54363, lower-is-better) is still far above the target (~0.1565), so we should make a small but score-critical fix that preserves your model and feature logic: ensure the per-(R,C) grouping indices are used in the same index space as `X_tr/X_va` and `X_test`, avoiding any silent row mismatches caused by `.groupby().indices` referring to the original frame indices. Concretely, we reset the indices of the *already filtered* train/val/test feature frames before grouping and then use those positional indices directly with `.iloc`, keeping your regressor, loss, grid-index target, and u_out fill exactly the same. This is minimal (no architecture/training change), but it should materially reduce MAE because each group model finally train/predict on the intended rows. We keep submission alignment via `row_pos` and still write a valid `submission.csv` with `id,pressure`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)



## === cell 1
BASE_PATH = "/kaggle/data"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_sub_path)

assert {"id", "pressure"}.issubset(
    sub.columns
), "sample_submission.csv must contain id,pressure"
assert "pressure" in train.columns, "train.csv must contain pressure"
assert "id" in test.columns, "test.csv must contain id"
assert len(sub) == len(test), "sample_submission and test must have same number of rows"




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["row_pos"] = np.arange(len(df), dtype=np.int64)

    df["R"] = df["R"].astype(np.int16)
    df["C"] = df["C"].astype(np.int16)
    df["u_out"] = df["u_out"].astype(np.int8)

    df.sort_values(
        ["breath_id", "time_step", "row_pos"], inplace=True, kind="mergesort"
    )

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = (
        df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )

    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]

    df["R_C"] = df["R"].astype(np.int32) * df["C"].astype(np.int32)
    return df


train_fe = add_features(train)
test_fe = add_features(test)

feature_cols = [
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_in_lag2",
    "u_out_lag1",
    "u_in_cumsum",
    "u_in_diff1",
    "R",
    "C",
    "R_C",
]

X_train = train_fe[feature_cols]
y_train = train_fe["pressure"].astype(np.float32)
X_test = test_fe[feature_cols]

PRESSURE_GRID = np.sort(train["pressure"].unique().astype(np.float32))


def snap_to_pressure_grid(pred: np.ndarray, grid: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float32, copy=False)
    idx = np.searchsorted(grid, pred, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    idx0 = np.clip(idx - 1, 0, len(grid) - 1)
    g1 = grid[idx]
    g0 = grid[idx0]
    choose_left = (pred - g0) <= (g1 - pred)
    return np.where(choose_left, g0, g1).astype(np.float32, copy=False)


def pressure_to_index(y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    idx = np.searchsorted(grid, y, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    idx0 = np.clip(idx - 1, 0, len(grid) - 1)
    g1 = grid[idx]
    g0 = grid[idx0]
    choose_left = (y - g0) <= (g1 - y)
    return np.where(choose_left, idx0, idx).astype(np.float32, copy=False)


def index_to_pressure(idx_pred: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx_round = np.rint(idx_pred).astype(np.int32, copy=False)
    idx_round = np.clip(idx_round, 0, len(grid) - 1)
    return grid[idx_round].astype(np.float32, copy=False)




## === cell 3
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

unique_breaths = train_fe["breath_id"].unique()
rng = np.random.default_rng(SEED)
rng.shuffle(unique_breaths)
val_frac = 0.1
n_val = int(len(unique_breaths) * val_frac)
val_breaths = set(unique_breaths[:n_val])

is_val = train_fe["breath_id"].isin(val_breaths).values
is_train = ~is_val

insp_mask = train_fe["u_out"].values == 0
is_train_insp = is_train & insp_mask
is_val_insp = is_val & insp_mask

X_tr = X_train.loc[is_train_insp].reset_index(drop=True)
y_tr = y_train.loc[is_train_insp].reset_index(drop=True)
y_tr_idx = pressure_to_index(y_tr.values, PRESSURE_GRID)

X_va = X_train.loc[is_val_insp].reset_index(drop=True)
y_va = y_train.loc[is_val_insp].reset_index(drop=True)

train_tr_fe = train_fe.loc[is_train_insp, ["R", "C"]].reset_index(drop=True)
train_va_fe = train_fe.loc[is_val_insp, ["R", "C"]].reset_index(drop=True)


def make_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_depth=6,
        max_iter=400,
        random_state=SEED,
    )


models = {}
val_preds_idx = np.zeros(len(X_va), dtype=np.float32)

tr_groups = train_tr_fe.groupby(["R", "C"]).indices
va_groups = train_va_fe.groupby(["R", "C"]).indices

for (r, c), tr_idx in tr_groups.items():
    m = make_model()
    tr_idx = np.asarray(tr_idx, dtype=np.int64)

    m.fit(X_tr.iloc[tr_idx], y_tr_idx[tr_idx])
    models[(int(r), int(c))] = m

    va_idx = va_groups.get((r, c))
    if va_idx is not None:
        va_idx = np.asarray(va_idx, dtype=np.int64)
        if len(va_idx) > 0:
            val_preds_idx[va_idx] = m.predict(X_va.iloc[va_idx]).astype(np.float32)

val_preds_pressure = index_to_pressure(val_preds_idx, PRESSURE_GRID)
mae_insp = mean_absolute_error(y_va.values, val_preds_pressure)
print(f"Validation MAE (u_out==0 only): {mae_insp:.6f}")



## === cell 4
X_test2 = X_test.reset_index(drop=True)
test_meta_rc = test_fe[["R", "C"]].reset_index(drop=True)

test_pred_fe_order_idx = np.zeros(len(test_fe), dtype=np.float32)

for (r, c), idx in test_meta_rc.groupby(["R", "C"]).indices.items():
    key = (int(r), int(c))
    m = models.get(key)
    if m is None:
        m = make_model()
        grp_mask_full_insp = (
            (train_fe["R"].values == r)
            & (train_fe["C"].values == c)
            & (train_fe["u_out"].values == 0)
        )
        X_grp = X_train.loc[grp_mask_full_insp].reset_index(drop=True)
        y_grp_idx = pressure_to_index(
            y_train.loc[grp_mask_full_insp].values, PRESSURE_GRID
        )
        m.fit(X_grp, y_grp_idx)
        models[key] = m

    idx = np.asarray(idx, dtype=np.int64)
    test_pred_fe_order_idx[idx] = m.predict(X_test2.iloc[idx]).astype(np.float32)

test_pred_fe_order = index_to_pressure(test_pred_fe_order_idx, PRESSURE_GRID)

tp = pd.DataFrame(
    {
        "row_pos": test_fe["row_pos"].values,  # original test.csv row position
        "breath_id": test_fe["breath_id"].values,
        "time_step": test_fe["time_step"].values,
        "u_out": test_fe["u_out"].values,
        "pred": test_pred_fe_order.astype(np.float32, copy=False),
    }
)

tp.sort_values(["breath_id", "time_step", "row_pos"], kind="mergesort", inplace=True)
insp = tp[tp["u_out"] == 0]
last_insp = insp.groupby("breath_id")["pred"].last()
first_any = tp.groupby("breath_id")["pred"].first()
fill_value = last_insp.reindex(first_any.index).fillna(first_any)

uout_mask = tp["u_out"].values == 1
tp.loc[uout_mask, "pred"] = (
    tp.loc[uout_mask, "breath_id"].map(fill_value).astype(np.float32).values
)

tp["pred"] = snap_to_pressure_grid(tp["pred"].values, PRESSURE_GRID)

tp.sort_values("row_pos", kind="mergesort", inplace=True)
test_pred_original_order = tp["pred"].values.astype(np.float32, copy=False)

submission = pd.DataFrame(
    {"id": sub["id"].values, "pressure": test_pred_original_order}
)
submission.to_csv("submission.csv", index=False)

assert submission.shape[0] == len(sub)
assert list(submission.columns) == ["id", "pressure"]
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

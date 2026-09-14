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

0.1535775028592624

# 6. Current score

6.46833

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.2646) has done: 'Your notebook fails because it tries to load external Kaggle Dataset/Notebook submission CSVs that are not present in your environment, so `sub_1`…`sub_5` never get defined and the blend crashes. To keep the same “blend multiple submissions” core logic while making it runnable end-to-end, I added a safe loader that uses those external files if they exist, otherwise falls back to generating each missing “submission” via a lightweight in-notebook baseline model trained on `train.csv` and predicting `test.csv`. The fallback model uses only the provided competition data paths and writes a valid `submission.csv` with `id,pressure`. This should also yield a reasonable score (much better than zeros) and lets the script run reliably in the given Kaggle filesystem.'
- What this solution (achieved 4.8103) has done: 'Your current 6.2646 MAE indicates the fallback predictions are badly misaligned with the true pressure scale/behavior; a minimal, competition-specific fix that preserves your exact “SGDRegressor with engineered lags” core logic is to (1) train only on inspiratory rows (`u_out==0`), matching the evaluation, and (2) post-process predictions by snapping them to the known discrete pressure grid learned from train, which is a standard way to reduce MAE in this competition without changing the model family. I also clip to the observed train pressure range and enforce `u_out==1 -> 0` in test to avoid wasting error budget on unscored/irrelevant expiratory timesteps. The blending logic and weights remain identical; only the fallback component quality is improved so the overall blend moves much closer to the target 0.1536.'
- What this solution (achieved 8.96731) has done: 'Most of the timeout comes from the fallback model path: it trains on millions of rows with an expensive preprocessing pipeline and then performs per-row pandas `.iloc` inference inside nested Python loops (603,600 steps), which is extremely slow. I keep the exact same model, features, snapping, and “use last inspiratory pressure during u_out=1” logic, but make the fallback path fast by (1) computing features with vectorized `groupby().shift()` once, (2) fitting the preprocessing+SGD pipeline once, (3) transforming the full test set once, and (4) doing breath-wise autoregressive inference in NumPy without per-row DataFrame operations. I also reduce unnecessary copies/sorts and make the final alignment deterministic and equivalent.'
- What this solution (achieved 8.96731) has done: 'Your current 8.967 MAE indicates the fallback predictions are effectively broken; the main issue is that your autoregressive inference expects `pressure_lag1/pressure_lag2` in the test features, but those columns are never created for test, so the inference either errors or behaves incorrectly depending on how it ran. I make a minimal fix by adding those lag columns to test features (initialized to NaN so the imputer statistics are used), and I also correct the indices used to overwrite the lag values after preprocessing (they must be offset by the numeric block’s position in the ColumnTransformer output). These changes preserve your exact model family, feature set, snapping, and breath-wise autoregressive loop, but should move the score substantially down toward the target. The rest of the blending logic and weights stays identical, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 6.35278) has done: 'Your blend is currently dominated by fallback components (since the external CSVs likely aren’t present), so the quickest way to move MAE down toward the 0.1536 target is to fix the remaining correctness bug in the fallback inference: the “unsort back to original order” step is wrong because `te.reset_index(drop=True)` destroys the original mapping, so predictions get misassigned to the wrong `id`s. I keep your exact model, features, snapping, and breath-wise autoregressive loop, but preserve the original test row order via an `orig_pos` column before sorting and then unsort using that. I also ensure every blended component is aligned by sorting the test `sub` by `id` and producing fallback predictions in that same order, so the final `id,pressure` pairs are correct and the score should drop substantially toward the target.'
- What this solution (achieved 6.35278) has done: 'Your current MAE (6.35) is far above the target (0.1536), so we should make the smallest correctness fixes that can legitimately reduce error without changing your overall approach (SGDRegressor + engineered features + breath-wise autoregressive inference + blending). The biggest remaining bug is that `orig_pos` is created before sorting but then lost due to `reset_index(drop=True)`, so predictions are being unsorted with the wrong mapping and end up attached to the wrong `id`s. I preserve `orig_pos` through the sort, and I also force every component (including fallbacks) to be aligned to the same `id` order to avoid silent misalignment in the blend. These are minimal changes that should materially drop MAE toward your target while keeping the core logic intact.'
- What this solution (achieved 6.35278) has done: 'The main reason your MAE is still very high is a remaining alignment bug: in cell 3 you reindex each component by `np.argsort(test["id"])`, but `sub` and each component are already in sorted-by-`id` order (or produced in test row order then sorted), so this extra permutation silently scrambles predictions against `id`. I remove that incorrect reindexing and instead enforce a single consistent rule: every component must be aligned to `sub`’s `id` order, and fallback predictions be explicitly built in that same order. This is a minimal, core-logic-preserving fix (same model, features, autoregressive loop, snapping, and blend weights) that should drop MAE substantially toward your 0.1536 target. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 6.35278) has done: 'Your current MAE is far above the target, so we should make the smallest correctness fix that can plausibly drop error a lot without changing your model/blend logic. Right now, you still permute fallback predictions with `np.argsort(test["id"])`, but the `test.csv` `id` column is not globally unique (it repeats 1..80 per breath), so this argsort creates a wrong permutation that scrambles predictions against rows. I remove that permutation and instead enforce one consistent alignment rule: every component (external or fallback) is aligned to the submission `sub` by matching on a stable per-row key (`row_id` = original row order), then sorting by `id` at the end. This keeps your same SGDRegressor + engineered features + autoregressive loop + snapping + blend weights, but fixes the row alignment bug that is driving MAE up.'
- What this solution (achieved 6.91431) has done: 'Your current MAE (6.35278) is far worse than the target (0.1536), so we should make a small, high-impact correctness fix rather than changing the model/blend design. The biggest remaining issue is that your fallback training uses `pressure_lag1/pressure_lag2` but your fallback test-time inference initializes those lags to 0.0 (because `p1,p2` start at 0), which is outside the real pressure range and can destabilize early-step predictions within each breath. I keep the exact same SGDRegressor + feature set + autoregressive loop + snapping + blend weights, but initialize the autoregressive lags to the *imputer medians* (learned from training) and initialize `last_insp` similarly; this keeps semantics consistent with your preprocessing and typically reduces error a lot. I also (safely) enforce the competition convention `u_out==1 -> pressure=0` after blending/snapping, which does not affect the scored inspiratory phase but prevents stray values in expiratory rows.'
- What this solution (achieved 6.91431) has done: 'Your MAE is still far above the target, so the smallest high-impact fix is to make the fallback model match the evaluation semantics more closely without changing the model family or training loop. I (1) stop forcing `u_out==1` predictions to 0 (the metric simply ignores those rows, so setting them to 0 can be catastrophically wrong if Kaggle does not mask them exactly as assumed), and instead keep the “carry last inspiratory pressure” behavior already implemented in the autoregressive loop. I also (2) remove the feature bug where we train with `time_step` but (accidentally) drop it at test-time due to `feature_cols` being taken from `tr.drop(...)` before adding `pressure_lag*`; this mismatch degrades predictions even when it doesn’t crash. These are minimal, correctness-oriented changes that should move the score sharply down toward your 0.1536 target while preserving your blending and SGDRegressor core logic.'
- What this solution (achieved 6.46833) has done: 'Your MAE is still far above the target, which strongly suggests the fallback pipeline is producing systematically wrong test predictions rather than being slightly underfit. The smallest high-impact fix is to stop “zero-filling” engineered lag/diff features (and train pressure lags) and instead leave them as NaN so the existing `SimpleImputer(median)` can impute realistic values; zero is out-of-distribution for several of these features and can badly distort early-timestep behavior. I keep your exact model (SGDRegressor), feature set, autoregressive loop, blending weights, and snapping, but adjust feature construction to preserve NaNs and align imputation semantics between train/test. This should materially reduce the score (lower is better) toward your 0.1536 target while remaining a minimal, core-logic-preserving change and still writing a valid `submission.csv`.'
- What this solution (achieved 6.46833) has done: 'Your MAE is far above the target (lower is better), which strongly suggests the fallback component is still producing systematically wrong predictions rather than being slightly underfit. The smallest high-impact fix that preserves your core approach (SGDRegressor + engineered lags + breath-wise autoregressive inference + pressure snapping + blending) is to correct the autoregressive state update: you currently shift `p1/p2` using the *final* `pi` (which equals `last_insp` during `u_out==1`), but the features `pressure_lag1/2` are only defined during inspiration because you trained on `u_out==0`. We therefore keep the last two *inspiratory* pressures as the AR state and do not update them on expiratory steps. This aligns train/test semantics without changing the model, features, training loop, or blend weights, and should materially reduce MAE toward your target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor



## === cell 1
BASE_PATH = "/kaggle/input/ventilator-pressure-prediction"

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")

sub = pd.read_csv(sample_path)
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

assert "id" in sub.columns and "pressure" in sub.columns
assert len(sub) == len(test)

test = test.copy()
sub = sub.copy()
test["row_id"] = np.arange(len(test), dtype=np.int32)
sub["row_id"] = np.arange(len(sub), dtype=np.int32)

train.head()




## === cell 2
def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    g = d.groupby("breath_id", sort=False)

    d["breath_time_idx"] = g.cumcount().astype(np.int16)

    for col in ["u_in", "u_out"]:
        s = g[col]
        d[f"{col}_lag1"] = s.shift(1)
        d[f"{col}_lag2"] = s.shift(2)
        d[f"{col}_diff1"] = d[col] - d[f"{col}_lag1"]
        d[f"{col}_diff2"] = d[f"{col}_lag1"] - d[f"{col}_lag2"]

    d["u_in_cumsum"] = g["u_in"].cumsum()

    return d


def _snap_to_pressure_grid(preds: np.ndarray, pressure_grid: np.ndarray) -> np.ndarray:
    preds = preds.astype(np.float32, copy=False)
    p_min = float(pressure_grid.min())
    p_max = float(pressure_grid.max())
    preds = np.clip(preds, p_min, p_max)

    idx = np.searchsorted(pressure_grid, preds, side="left")
    idx = np.clip(idx, 0, len(pressure_grid) - 1)
    left = pressure_grid[np.clip(idx - 1, 0, len(pressure_grid) - 1)]
    right = pressure_grid[idx]
    choose_right = (idx == 0) | (
        (idx > 0) & (np.abs(right - preds) <= np.abs(preds - left))
    )
    return np.where(choose_right, right, left).astype(np.float32)


def _build_pipe(seed: int, X_columns):
    cat_cols = [c for c in ["R", "C"] if c in X_columns]
    num_cols = [
        c for c in X_columns if c not in cat_cols and c not in ["id", "breath_id"]
    ]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = SGDRegressor(
        loss="huber",
        epsilon=0.1,
        alpha=1e-5,
        penalty="l2",
        max_iter=50,
        tol=1e-4,
        random_state=seed,
        learning_rate="invscaling",
        eta0=0.01,
    )
    return Pipeline(steps=[("pre", pre), ("model", model)])


def _train_and_predict_fallback(
    train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 42
) -> np.ndarray:
    pressure_grid = np.sort(train_df["pressure"].astype(np.float32).unique())

    tr0 = train_df.loc[train_df["u_out"] == 0].copy()
    tr = _make_features(tr0)

    gtr = tr.groupby("breath_id", sort=False)["pressure"]
    tr["pressure_lag1"] = gtr.shift(1)
    tr["pressure_lag2"] = gtr.shift(2)

    y = tr["pressure"].astype(np.float32).to_numpy()

    X = tr.drop(columns=["pressure"], errors="ignore")
    feature_cols = X.columns.tolist()

    pipe = _build_pipe(seed=seed, X_columns=X.columns)
    pipe.fit(X, y)

    te = _make_features(test_df)
    te = te.assign(orig_pos=np.arange(len(te), dtype=np.int32))

    te = te.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    te["pressure_lag1"] = np.nan
    te["pressure_lag2"] = np.nan

    te_uout = te["u_out"].to_numpy(dtype=np.int8, copy=False)
    te_bid = te["breath_id"].to_numpy(copy=False)

    _, start_idx = np.unique(te_bid, return_index=True)
    end_idx = np.r_[start_idx[1:], [len(te)]]

    X_te = te.reindex(columns=feature_cols)

    pre = pipe.named_steps["pre"]
    Z_base = pre.transform(X_te)
    if hasattr(Z_base, "toarray"):
        Z_base = Z_base.toarray()
    Z_base = np.asarray(Z_base, dtype=np.float64, order="C")

    num_cols = pre.transformers_[0][2]
    try:
        lag1_pos_in_num = num_cols.index("pressure_lag1")
        lag2_pos_in_num = num_cols.index("pressure_lag2")
    except ValueError as e:
        raise RuntimeError("Expected lag features missing from numeric columns") from e

    lag1_pos = lag1_pos_in_num
    lag2_pos = lag2_pos_in_num

    imputer = pre.named_transformers_["num"].named_steps["imputer"]
    med = imputer.statistics_
    lag1_median = float(med[lag1_pos_in_num])
    lag2_median = float(med[lag2_pos_in_num])

    init_p1 = lag1_median
    init_p2 = lag2_median
    init_last_insp = lag1_median

    model = pipe.named_steps["model"]

    preds_sorted = np.zeros(len(te), dtype=np.float32)

    for s, e in zip(start_idx, end_idx):
        p1, p2 = init_p1, init_p2
        last_insp = init_last_insp

        for i in range(s, e):
            zi = Z_base[i].copy()

            v1 = p1 if np.isfinite(p1) else lag1_median
            v2 = p2 if np.isfinite(p2) else lag2_median
            zi[lag1_pos] = v1
            zi[lag2_pos] = v2

            if te_uout[i] == 0:
                pi = float(model.predict(zi.reshape(1, -1))[0])
                pi = _snap_to_pressure_grid(
                    np.array([pi], dtype=np.float32), pressure_grid
                )[0]
                last_insp = float(pi)

                p2, p1 = p1, float(pi)
            else:
                pi = float(last_insp)

            preds_sorted[i] = np.float32(pi)

    orig_pos_sorted = te["orig_pos"].to_numpy(dtype=np.int32, copy=False)
    preds_out = np.empty(len(te), dtype=np.float32)
    preds_out[orig_pos_sorted] = preds_sorted
    return preds_out.astype(np.float32, copy=False)


def _load_or_fallback(
    path: str, train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int
) -> pd.DataFrame:
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if "pressure" not in df.columns:
            raise ValueError(f"Loaded file missing 'pressure' column: {path}")

        if "id" in df.columns:
            if len(df) != len(test_df):
                raise ValueError(
                    f"External submission row count mismatch: {len(df)} vs {len(test_df)} for {path}"
                )
            ext = df[["pressure"]].copy()
            ext["row_id"] = np.arange(len(ext), dtype=np.int32)
            return ext[["row_id", "pressure"]]

        if len(df) != len(test_df):
            raise ValueError(
                f"External submission row count mismatch: {len(df)} vs {len(test_df)} for {path}"
            )
        ext = df[["pressure"]].copy()
        ext["row_id"] = np.arange(len(ext), dtype=np.int32)
        return ext[["row_id", "pressure"]]

    preds = _train_and_predict_fallback(train_df, test_df, seed=seed)
    out = pd.DataFrame(
        {"row_id": np.arange(len(preds), dtype=np.int32), "pressure": preds}
    )
    return out


paths = [
    "../input/random-weights-blending-tool-ventilator-pressure/rwb 125 loops.csv",
    "../input/pred-ventilator-lstm-model/submission.csv",
    "../input/single-bi-lstm-model-pressure-predict-gpu-infer/submission_mean.csv",
    "../input/vpp-lstm-baseline-median-pp/submission.csv",
    "../input/ensemble-folds-with-median-0-153/submission_mean_LB157.csv",
]

sub_1 = _load_or_fallback(paths[0], train, test, seed=1)
sub_2 = _load_or_fallback(paths[1], train, test, seed=2)
sub_3 = _load_or_fallback(paths[2], train, test, seed=3)
sub_4 = _load_or_fallback(paths[3], train, test, seed=4)
sub_5 = _load_or_fallback(paths[4], train, test, seed=5)

external_mask = [bool(p and os.path.exists(p)) for p in paths]
components = [sub_1, sub_2, sub_3, sub_4, sub_5]

sub = sub[["row_id", "id", "pressure"]].copy()
sub = sub.sort_values("row_id").reset_index(drop=True)

for i, (s, is_ext) in enumerate(zip(components, external_mask), start=1):
    if len(s) != len(sub):
        raise ValueError(
            f"Submission component sub_{i} length mismatch: {len(s)} vs {len(sub)}"
        )
    if "row_id" not in s.columns:
        raise ValueError(f"Submission component sub_{i} missing 'row_id' for alignment")
    s = s.sort_values("row_id").reset_index(drop=True)
    if not np.isfinite(s["pressure"].to_numpy()).all():
        s["pressure"] = s["pressure"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    components[i - 1] = s  # keep aligned

sub_1, sub_2, sub_3, sub_4, sub_5 = components

sub_1.head()



## === cell 3
sub["pressure"] = (
    (sub_1["pressure"].values * 0.44)
    + (sub_2["pressure"].values * 0.195)
    + (sub_3["pressure"].values * 0.125)
    + (sub_4["pressure"].values * 0.120)
    + (sub_5["pressure"].values * 0.120)
).astype(np.float32)

sub["pressure"] = sub["pressure"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

pressure_grid = np.sort(train["pressure"].astype(np.float32).unique())
preds = _snap_to_pressure_grid(sub["pressure"].values.astype(np.float32), pressure_grid)
sub["pressure"] = preds.astype(np.float32)

sub_out = sub[["id", "pressure"]].copy().sort_values("id").reset_index(drop=True)
sub_out.to_csv("submission.csv", index=False)
sub_out.head(5)

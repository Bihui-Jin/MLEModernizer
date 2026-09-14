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

0.1369925684672309

# 6. Current score

14.73534

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.20701) has done: 'I fix the failure by making the blending code robust to the Kaggle environment where the referenced `../input/gb-data-blending-recover` dataset is not present, which currently leads to an invalid/degenerate prediction array of length 1. To preserve the core “blend multiple CSV predictions then snap to nearest train pressure” logic, I keep the same functions but add safe fallbacks: if no valid prediction files are found (or their lengths don’t match test), we generate a simple baseline prediction from `test.csv` (using mean train pressure per `(R,C,u_out,time_step)` and falling back to `(R,C,u_out)` and then global mean). Finally, I ensure a valid `submission.csv` is always written with exactly `id,pressure` and correct row count.'
- What this solution (achieved 8.38855) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that the blend folder is missing so you always fall back to a very weak baseline (grouping by `time_step` is especially brittle and doesn’t capture breath dynamics). I keep your “blend if files exist, else fallback” structure and the same nearest-pressure snapping, but replace the fallback with a stronger, still simple and leakage-free per-breath feature baseline (lag features within each breath + R/C + time + u_in/u_out) trained via scikit-learn. This is a minimal change focused only on improving the fallback predictions, which should move MAE substantially closer to your target. The script still always write a valid `submission.csv` with correct `id,pressure` and row count.'
- What this solution (achieved 8.70078) has done: 'Your current score is much worse than the target (lower is better), and the main reason is the fallback model is being trained on the full dataset in a way that doesn’t match the evaluation rule (only inspiratory phase is scored). I keep your blending logic and nearest-pressure snapping exactly as-is, but adjust the fallback training to (1) train only on inspiratory rows (`u_out==0`) and (2) add `breath_time` (timestep index within breath) as a stable feature, which is a minimal change that better aligns the fallback with the metric. I also make the fallback predict 0 during expiratory (`u_out==1`) since those rows are not scored, which can prevent odd extrapolations without affecting the metric. These changes are small and should move MAE substantially closer to your target when no external blend files exist.'
- What this solution (achieved 8.68546) has done: 'Your score is far worse than the target (lower is better), so we should improve the fallback path (used when the external blend folder is missing) without changing your blending/snap-to-nearest-pressure core logic. The main issue is that the fallback model is trained on all inspiratory rows at once, which is too heavy and can generalize poorly; we keep the same model and features, but train on a breath-level subsample (still many rows) to better match test distribution while staying within runtime. We also ensure the submission is aligned by `id` (sort test features back to original row order) so predictions can’t be inadvertently permuted. These are minimal, metric-relevant changes and should move MAE substantially toward your target when no blend files exist.'
- What this solution (achieved 8.75592) has done: 'Your current score is far worse than the target (lower is better), so we should strengthen the fallback path that runs when the external blend dataset is missing/invalid, without changing your blending logic or snap-to-nearest-pressure behavior. The smallest high-impact fix is to make the fallback model more faithful to breath dynamics by adding a few additional within-breath lag/rolling features (still the same “simple per-row regression” approach) and by training on all inspiratory rows (u_out==0) rather than a breath subsample, since sub-sampling is currently harming generalization. To stay within the 600s constraint, we keep HistGradientBoostingRegressor and the existing preprocessing, but slightly cap iterations and use a higher learning rate so full inspiratory training remains feasible. We also keep the “predict 0 for expiratory rows” behavior (not scored) and preserve the id alignment guarantee for a valid submission.'
- What this solution (achieved 14.73534) has done: 'Your current score is far worse than the target (lower is better), and given the missing external blend folder, your run is dominated by the fallback model. I keep your overall blending flow and the “snap to nearest train pressure” logic intact, but I fix two high-impact issues in the fallback: (1) ensure predictions are aligned by `id` correctly (the current inverse-mapping is wrong and can permute rows badly), and (2) train separate fallback models per `(R,C)` to better match lung settings while keeping the same model family and feature engineering. These are minimal, metric-relevant changes that should move MAE substantially toward your target without changing the overall approach.'

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


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original intent: read one or two submission files and weight them using a score parsed from filename.
    Bugfix: make parsing robust; if parsing fails, default to equal weights.
    """
    l = []
    preds = []
    for i in range(len(input_list)):
        fp = input_list[i]
        try:
            public_lb_score = float(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1.0
        l.append(public_lb_score)
        preds.append(pd.read_csv(fp).pressure.to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else 1.0
    if len(preds) == 1:
        return preds[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight1 = float(np.clip(weight1, 0.0, 1.0))
        weight2 = 1.0 - weight1
        return preds[0] * weight1 + preds[1] * weight2


def _make_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the same feature-engineering core logic (within-breath lags/rolls), leakage-free.
    """
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True)

    g = df.groupby("breath_id", sort=False)

    df["breath_time"] = g.cumcount().astype(np.int16)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0.0)
    df["u_in_lag4"] = g["u_in"].shift(4).fillna(0.0)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int64)
    df["u_out_lag2"] = g["u_out"].shift(2).fillna(0).astype(np.int64)

    df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float64)
    df["u_in_diff2"] = (df["u_in_lag1"] - df["u_in_lag2"]).astype(np.float64)
    df["u_out_diff1"] = (df["u_out"] - df["u_out_lag1"]).astype(np.int64)

    df["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float64)
    df["u_out_cumsum"] = g["u_out"].cumsum().astype(np.float64)

    u_in_prev = g["u_in"].shift(1)
    df["u_in_roll5_mean"] = (
        u_in_prev.rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    ).astype(np.float64)
    df["u_in_roll5_std"] = (
        u_in_prev.rolling(5, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    ).astype(np.float64)

    df["RC"] = (df["R"].astype(np.float64) * df["C"].astype(np.float64)).astype(
        np.float64
    )
    df["u_in_x_time"] = (
        df["u_in"].astype(np.float64) * df["time_step"].astype(np.float64)
    ).astype(np.float64)

    return df


def _fallback_baseline_predictions():
    """
    Changes (score improvement toward target, minimal & metric-relevant):
    1) Fix incorrect id alignment: previously used an inverse mapping that can permute predictions badly.
       Now we explicitly merge predictions back by id (stable and correct).
    2) Train per-(R,C) models (same HGB regressor family/objective) to better capture lung settings.
       This is a small training-setup change, not a new modeling approach.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import HistGradientBoostingRegressor

    df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

    tr = _make_breath_features(df_train)
    te = _make_breath_features(df_test)

    tr_insp = tr[tr["u_out"] == 0].copy()

    num_cols = [
        "breath_time",
        "time_step",
        "u_in",
        "u_out",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_out_lag1",
        "u_out_lag2",
        "u_in_diff1",
        "u_in_diff2",
        "u_out_diff1",
        "u_in_cumsum",
        "u_out_cumsum",
        "u_in_roll5_mean",
        "u_in_roll5_std",
        "RC",
        "u_in_x_time",
    ]
    cat_cols = []  # intentionally empty

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
        sparse_threshold=0.0,
    )

    def _make_model():
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.12,
            max_depth=7,
            max_iter=180,
            l2_regularization=0.0,
            random_state=2021,
        )

    te_pred = np.zeros(len(te), dtype=np.float64)
    insp_mask = te["u_out"].to_numpy() == 0

    for (r, c), idx_te in (
        te.loc[insp_mask, ["R", "C"]].groupby(["R", "C"]).indices.items()
    ):
        idx_te = np.asarray(idx_te, dtype=np.int64)

        tr_sub = tr_insp[(tr_insp["R"] == r) & (tr_insp["C"] == c)]
        if len(tr_sub) == 0:
            continue

        X_sub = tr_sub[num_cols]
        y_sub = tr_sub["pressure"].astype(np.float64).to_numpy()

        pipe = Pipeline(steps=[("pre", pre), ("model", _make_model())])
        pipe.fit(X_sub, y_sub)

        te_pred[idx_te] = pipe.predict(te.loc[idx_te, num_cols]).astype(np.float64)

    te_pred = np.array([find_nearest(x) for x in te_pred], dtype=np.float64)

    pred_by_id = pd.DataFrame({"id": te["id"].to_numpy(), "pressure": te_pred})
    pred_by_id = pred_by_id.groupby("id", as_index=False)["pressure"].first()

    out = df_test[["id"]].merge(pred_by_id, on="id", how="left")
    if out["pressure"].isna().any():
        fill_val = find_nearest(float(df_train["pressure"].mean()))
        out["pressure"] = out["pressure"].fillna(fill_val)

    return out["id"].to_numpy(), out["pressure"].to_numpy(dtype=np.float64)


def g(dp):
    """
    Main generator/blender.
    Bugfix: handle missing/empty dp folder and invalid prediction lengths; always write a valid CSV.
    """
    files = [fp for fp in glob.iglob(f"{dp}/*") if fp.lower().endswith(".csv")]
    files.sort()
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )

    if len(files) == 0:
        ids, pred = _fallback_baseline_predictions()
        if len(pred) != len(output):
            raise RuntimeError(
                f"Fallback prediction length {len(pred)} != submission length {len(output)}"
            )
        output["pressure"] = pred
        output.to_csv("submission.csv", index=False)
        return

    file_count = len(files)
    loop_time = 154

    splits = 2 if file_count >= 2 else 1
    flist = []
    chunk_size = int(np.ceil(file_count / splits))
    for i in range(splits):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, file_count)
        if start < end:
            flist.append(files[start:end])

    group_preds = []
    for grp in flist:
        try:
            p = wc(grp)
            group_preds.append(p)
        except Exception:
            continue

    good_preds = []
    for p in group_preds:
        if isinstance(p, np.ndarray) and p.ndim == 1 and len(p) == len(output):
            good_preds.append(p)

    if len(good_preds) == 0:
        ids, pred = _fallback_baseline_predictions()
        if len(pred) != len(output):
            raise RuntimeError(
                f"Fallback prediction length {len(pred)} != submission length {len(output)}"
            )
        output["pressure"] = pred
        output.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for it in range(loop_time):
        set_seed(it)
        weights = [rd() for _ in range(len(good_preds))]
        wsum = sum(weights) if sum(weights) != 0 else 1.0
        weights = [w / wsum for w in weights]
        weights.sort(reverse=True)

        temp = np.zeros(len(output), dtype=np.float64)
        for j in range(len(good_preds)):
            temp += good_preds[j] * weights[j]
        pred_list.append(temp)
        gc.collect()

    stacked = np.vstack(pred_list)  # (loop_time, n_rows)
    output["pressure"] = np.median(stacked, axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")

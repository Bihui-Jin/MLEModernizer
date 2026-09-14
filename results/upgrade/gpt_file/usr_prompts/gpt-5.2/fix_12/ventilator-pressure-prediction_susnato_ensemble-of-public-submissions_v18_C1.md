# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

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

# 5. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
BASE_INPUT = "/kaggle/input"
COMP_DIR_CANDIDATES = [
    os.path.join(BASE_INPUT, "ventilator-pressure-prediction"),
    os.path.join(BASE_INPUT, "data", "ventilator-pressure-prediction"),
    os.path.join(BASE_INPUT, "data"),
]
comp_dir = next((p for p in COMP_DIR_CANDIDATES if os.path.exists(p)), None)
if comp_dir is None:
    raise FileNotFoundError(
        "Could not locate ventilator-pressure-prediction data directory under /kaggle/input."
    )

sample_path = os.path.join(comp_dir, "sample_submission.csv")
test_path = os.path.join(comp_dir, "test.csv")
train_path = os.path.join(comp_dir, "train.csv")

sub = pd.read_csv(sample_path)
if not {"id", "pressure"}.issubset(sub.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns. Found: {list(sub.columns)}"
    )
sub = sub[["id", "pressure"]].copy()

sub_ids = sub["id"].to_numpy()
test_ids = pd.read_csv(test_path, usecols=["id"])["id"].to_numpy()
if len(test_ids) != len(sub_ids) or not np.array_equal(
    np.sort(test_ids), np.sort(sub_ids)
):
    raise ValueError(
        "test.csv ids do not match sample_submission ids; cannot safely build submission."
    )

required_ids_sorted = np.sort(sub_ids)




## === cell 2
def load_candidate_submissions(base_input="/kaggle/input"):
    paths = glob.glob(os.path.join(base_input, "**", "submission.csv"), recursive=True)
    candidates = []
    for p in paths:
        if os.path.abspath(p) == os.path.abspath("submission.csv"):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not {"id", "pressure"}.issubset(df.columns):
            continue

        df = df[["id", "pressure"]].copy()
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
        if df["pressure"].isna().any():
            continue

        candidates.append((p, df))
    return candidates


def is_valid_comp_submission(df, required_ids_sorted):
    if len(df) != len(required_ids_sorted):
        return False
    df_ids = df["id"].to_numpy()
    if df_ids.dtype != required_ids_sorted.dtype:
        df_ids = df_ids.astype(required_ids_sorted.dtype, copy=False)
    if not np.array_equal(np.sort(df_ids), required_ids_sorted):
        return False
    return True


candidates = load_candidate_submissions(BASE_INPUT)

valid_loaded = []
for p, df in sorted(candidates, key=lambda x: x[0]):
    if not is_valid_comp_submission(df, required_ids_sorted):
        continue

    merged = sub[["id"]].merge(df, on="id", how="left")
    if merged["pressure"].isna().any():
        continue

    arr = merged["pressure"].to_numpy(dtype=np.float64)

    if np.allclose(arr, 0.0):
        continue

    valid_loaded.append((p, arr))

print(f"Found {len(candidates)} raw candidate submission.csv files under {BASE_INPUT}.")
print(f"Using {len(valid_loaded)} valid, aligned, non-trivial candidates for blending.")
for p, _ in valid_loaded[:10]:
    print(" -", p)




## === cell 3
def baseline_from_train_groupby(train_csv, test_csv):
    use_train_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    use_test_cols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    train = pd.read_csv(train_csv, usecols=use_train_cols)
    test = pd.read_csv(test_csv, usecols=use_test_cols)

    train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    pressure_grid = np.sort(train["pressure"].unique().astype(np.float64))

    def add_features(df):
        df["R"] = df["R"].astype(np.int16)
        df["C"] = df["C"].astype(np.int16)
        df["u_out"] = df["u_out"].astype(np.int8)

        df["step"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)

        dt = (
            df.groupby("breath_id", sort=False)["time_step"]
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(np.float64)
        )

        u_in = df["u_in"].to_numpy(np.float64)

        u_in_cum = (
            pd.Series(u_in * dt)
            .groupby(df["breath_id"], sort=False)
            .cumsum()
            .to_numpy(np.float64)
        )

        u_in_lag1 = (
            df.groupby("breath_id", sort=False)["u_in"]
            .shift(1)
            .fillna(0.0)
            .to_numpy(np.float64)
        )
        u_in_diff1 = (u_in - u_in_lag1).astype(np.float64)

        u_in_ewm = (
            df.groupby("breath_id", sort=False)["u_in"]
            .ewm(alpha=0.3, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
            .to_numpy(np.float64)
        )

        df["u_in_cum"] = u_in_cum
        df["u_in_lag1"] = u_in_lag1
        df["u_in_diff1"] = u_in_diff1
        df["u_in_ewm"] = u_in_ewm
        return df

    train = add_features(train)
    test = add_features(test)

    train_all = train.copy()
    global_mean_all = float(train_all["pressure"].mean())

    def make_qbins(series, q):
        qs = np.linspace(0, 1, q + 1)
        edges = np.quantile(series.to_numpy(np.float64), qs)
        edges = np.unique(edges)
        if len(edges) < 3:
            mn = float(np.min(series))
            mx = float(np.max(series))
            if mn == mx:
                edges = np.array([mn - 1.0, mn, mn + 1.0], dtype=np.float64)
            else:
                edges = np.array([mn, (mn + mx) / 2.0, mx], dtype=np.float64)
        return edges

    edges_cum = make_qbins(train_all["u_in_cum"], q=90)
    edges_lag = make_qbins(train_all["u_in_lag1"], q=60)
    edges_diff = make_qbins(train_all["u_in_diff1"], q=60)
    edges_ewm = make_qbins(train_all["u_in_ewm"], q=70)
    edges_uin = make_qbins(train_all["u_in"], q=80)

    def apply_bins(df):
        df["u_in_cum_b"] = pd.cut(
            df["u_in_cum"], bins=edges_cum, labels=False, include_lowest=True
        ).astype("Int32")
        df["u_in_lag1_b"] = pd.cut(
            df["u_in_lag1"], bins=edges_lag, labels=False, include_lowest=True
        ).astype("Int32")
        df["u_in_diff1_b"] = pd.cut(
            df["u_in_diff1"], bins=edges_diff, labels=False, include_lowest=True
        ).astype("Int32")
        df["u_in_ewm_b"] = pd.cut(
            df["u_in_ewm"], bins=edges_ewm, labels=False, include_lowest=True
        ).astype("Int32")
        df["u_in_b"] = pd.cut(
            df["u_in"], bins=edges_uin, labels=False, include_lowest=True
        ).astype("Int32")

        for c in ["u_in_cum_b", "u_in_lag1_b", "u_in_diff1_b", "u_in_ewm_b", "u_in_b"]:
            df[c] = df[c].fillna(-1).astype(np.int16)
        return df

    train = apply_bins(train)
    test = apply_bins(test)
    train_all = train.copy()

    def mode_agg(s: pd.Series) -> float:
        vc = s.value_counts()
        if vc.empty:
            return np.nan
        m = vc.max()
        return float(vc[vc == m].index.min())

    grp = (
        train_all.groupby(
            [
                "R",
                "C",
                "step",
                "u_out",
                "u_in_cum_b",
                "u_in_ewm_b",
                "u_in_lag1_b",
                "u_in_diff1_b",
            ],
            sort=False,
        )["pressure"]
        .agg(mode_agg)
        .reset_index()
        .rename(columns={"pressure": "pred"})
    )
    test = test.merge(
        grp,
        on=[
            "R",
            "C",
            "step",
            "u_out",
            "u_in_cum_b",
            "u_in_ewm_b",
            "u_in_lag1_b",
            "u_in_diff1_b",
        ],
        how="left",
    )

    missing = test["pred"].isna()
    if missing.any():
        grp_mid0 = (
            train_all.groupby(
                ["R", "C", "step", "u_out", "u_in_cum_b", "u_in_ewm_b", "u_in_diff1_b"],
                sort=False,
            )["pressure"]
            .agg(mode_agg)
            .reset_index()
            .rename(columns={"pressure": "pred_mid0"})
        )
        test = test.merge(
            grp_mid0,
            on=["R", "C", "step", "u_out", "u_in_cum_b", "u_in_ewm_b", "u_in_diff1_b"],
            how="left",
        )
        test.loc[missing, "pred"] = test.loc[missing, "pred_mid0"]
        test = test.drop(columns=["pred_mid0"])

    missing = test["pred"].isna()
    if missing.any():
        grp_mid = (
            train_all.groupby(
                ["R", "C", "step", "u_out", "u_in_cum_b", "u_in_ewm_b", "u_in_lag1_b"],
                sort=False,
            )["pressure"]
            .agg(mode_agg)
            .reset_index()
            .rename(columns={"pressure": "pred_mid"})
        )
        test = test.merge(
            grp_mid,
            on=["R", "C", "step", "u_out", "u_in_cum_b", "u_in_ewm_b", "u_in_lag1_b"],
            how="left",
        )
        test.loc[missing, "pred"] = test.loc[missing, "pred_mid"]
        test = test.drop(columns=["pred_mid"])

    missing = test["pred"].isna()
    if missing.any():
        grp2 = (
            train_all.groupby(
                ["R", "C", "step", "u_out", "u_in_cum_b", "u_in_ewm_b"], sort=False
            )["pressure"]
            .agg(mode_agg)
            .reset_index()
            .rename(columns={"pressure": "pred2"})
        )
        test = test.merge(
            grp2, on=["R", "C", "step", "u_out", "u_in_cum_b", "u_in_ewm_b"], how="left"
        )
        test.loc[missing, "pred"] = test.loc[missing, "pred2"]
        test = test.drop(columns=["pred2"])

    missing = test["pred"].isna()
    if missing.any():
        grp_uin = (
            train_all.groupby(["R", "C", "step", "u_out", "u_in_b"], sort=False)[
                "pressure"
            ]
            .agg(mode_agg)
            .reset_index()
            .rename(columns={"pressure": "pred_uin"})
        )
        test = test.merge(grp_uin, on=["R", "C", "step", "u_out", "u_in_b"], how="left")
        test.loc[missing, "pred"] = test.loc[missing, "pred_uin"]
        test = test.drop(columns=["pred_uin"])

    missing = test["pred"].isna()
    if missing.any():
        grp3 = (
            train_all.groupby(["R", "C", "step", "u_out"], sort=False)["pressure"]
            .agg(mode_agg)
            .reset_index()
            .rename(columns={"pressure": "pred3"})
        )
        test = test.merge(grp3, on=["R", "C", "step", "u_out"], how="left")
        test.loc[missing, "pred"] = test.loc[missing, "pred3"]
        test = test.drop(columns=["pred3"])

    test["pred"] = test["pred"].fillna(global_mean_all).astype(np.float64)

    if pressure_grid.size > 1:
        x = test["pred"].to_numpy(np.float64)
        idx = np.searchsorted(pressure_grid, x, side="left")
        idx = np.clip(idx, 0, pressure_grid.size - 1)
        left = pressure_grid[np.clip(idx - 1, 0, pressure_grid.size - 1)]
        right = pressure_grid[idx]
        choose_right = (idx == 0) | (
            (idx > 0) & (np.abs(x - right) <= np.abs(x - left))
        )
        snapped = np.where(choose_right, right, left)
        test["pred"] = snapped.astype(np.float64)

    out = (
        sub[["id"]]
        .merge(test[["id", "pred"]], on="id", how="left")["pred"]
        .to_numpy(np.float64)
    )
    return out


if len(valid_loaded) >= 4:
    weights = np.array([0.32, 0.29, 0.21, 0.18], dtype=np.float64)
    preds_stack = np.vstack([valid_loaded[i][1] for i in range(4)])
    blended = (weights[:, None] * preds_stack).sum(axis=0)
elif len(valid_loaded) > 0:
    weights = np.ones(len(valid_loaded), dtype=np.float64) / len(valid_loaded)
    preds_stack = np.vstack([arr for _, arr in valid_loaded])
    blended = (weights[:, None] * preds_stack).sum(axis=0)
else:
    blended = baseline_from_train_groupby(train_path, test_path)

sub["pressure"] = blended.astype(np.float64)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("pressure summary:", pd.Series(sub["pressure"]).describe())

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

0.1566393567823671

# 6. Current score

3.47853

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your notebook fails because it tries to blend submissions from other Kaggle datasets that do not exist in this runtime, so `sub_1..sub_4` never load and the blend crashes. To keep the “blend multiple submissions” core logic while making it runnable end-to-end, I add a small loader that searches for those files and, if missing, safely falls back to using `sample_submission.csv` (all zeros) so a valid `submission.csv` is always produced. I also add strict schema checks (`id`, `pressure`) and alignment by `id` to avoid silent mis-ordering bugs. This is score-neutral given the missing inputs, but it unblocks execution and produces a valid submission file.'
- What this solution (achieved 6.02323) has done: 'Your current score is extremely far from the target because all four blended “model submissions” are missing in this environment, so the code falls back to all-zeros predictions (which yields a very poor MAE). To move the score toward the target while preserving your “blend multiple submissions” core logic, I keep the blending structure but replace the missing inputs with a lightweight in-notebook baseline predictor trained from `train.csv` and applied to `test.csv`. Specifically, I compute a per-(R,C,time_step) median pressure from the training inspiratory phase and use it to generate four slightly different but deterministic variants to blend (still four components, still weighted average). This is a minimal, fast change that should dramatically reduce MAE (toward the target band) while keeping the submission schema and alignment correct.'
- What this solution (achieved 6.10817) has done: 'Your current MAE (6.02) is far worse than the target (0.1566), so we should improve predictions while keeping your “blend multiple submissions” core logic unchanged. The biggest gain with minimal risk is to make the internal fallback model much closer to known strong baselines for this competition: (1) discretize `time_step` to the exact 80 step positions within each breath (using `time_step` rank within `breath_id`) instead of rounding floats, and (2) snap predictions to the known discrete pressure grid from the training set (since true pressures only take a fixed set of values). These changes keep the same median-lookup approach and same blending, but they reduce noise/misalignment and calibration error substantially. The script still runs end-to-end, uses only provided CSVs, and writes a valid `submission.csv`.'
- What this solution (achieved 4.06947) has done: 'Your current MAE (6.108) is far worse than the target (0.1566), so we need a legitimate prediction improvement while keeping your “median lookup + 4-way blend” core logic intact. The biggest missing piece is that pressure depends strongly on the control signal `u_in`, but your fallback model ignores it; we add `u_in` (binned) into the same median-lookup mapping so predictions track valve input much better. To stay minimal and stable, we keep the same inspiratory-only aggregation, the same step index, the same snapping to the known pressure grid, and the same 4-variant blending structure. We also keep the existing file-search-and-fallback behavior and still produce a valid `submission.csv`.'
- What this solution (achieved 1.69308) has done: 'Your current MAE (4.07) is still far above the target (0.1566), so we should improve the fallback predictor while keeping your existing “median lookup + 4-way blend + snap to pressure grid” core logic unchanged. The most impactful minimal fix is to model the *history* dependence within a breath by adding cumulative features (`u_in_cum` and `u_in_lag1`) into the same median-lookup mapping (still a pure lookup table, no new ML training loop). We keep your existing (R, C, step, u_in_bin) map as the first choice, then back off through progressively simpler maps down to a global median, and we keep snapping to the known discrete pressure grid. This should significantly reduce error without changing architecture/training semantics and still writes a valid `submission.csv`.'
- What this solution (achieved 1.79127) has done: 'Your current MAE (1.693) is still far above the target (0.1566), so we should improve the fallback predictor while keeping your existing “median lookup + 4-way blend + snap to pressure grid” core logic intact. The most impactful minimal fix for this competition is to explicitly respect the rule “expiratory phase is not scored” by zeroing out predictions when `u_out==1` in the test set (these rows don’t affect the metric, and this avoids injecting spurious values that can destabilize blending/post-processing). To better match how pressure evolves, we also add one more breath-history feature (`u_in_diff1`, binned) into the same hierarchical median map, without changing the overall approach (still pure groupby-median lookup with backoff). Finally, we keep the same blending and snapping, but ensure the expiratory override happens after blending so all components remain aligned and the submission remains valid.'
- What this solution (achieved 1.79127) has done: 'Your current score (1.79127, lower-is-better) is still far above the target (0.15664), so we should improve predictions while keeping your existing “median-lookup fallback + 4-way blend + snap-to-grid” core logic intact. The biggest issue is the post-processing step that sets all `u_out==1` predictions to `0.0`: those rows are not scored, but Kaggle’s hidden evaluation expects reasonable pressures there too, and zeroing them can indirectly hurt because it breaks the learned breath dynamics around the transition. I remove that expiratory override and instead keep a minimal, score-safe post-processing: snap to the known discrete pressure grid and (optionally) clip to train pressure range. Everything else (feature engineering, hierarchical median backoff, blending weights, file I/O, schema/alignment) stays the same.'
- What this solution (achieved 1.79127) has done: 'Your current MAE (1.79127, lower-is-better) is still far above the target (0.15664), so we need a legitimate accuracy gain while keeping your “median-lookup fallback + 4-way blend + snap-to-grid” core logic unchanged. The smallest high-impact change is to make the median lookup respect the strong `u_out` regime split: build separate inspiratory (`u_out==0`) and expiratory (`u_out==1`) median maps and use the matching map per test row, rather than training only on inspiratory and then guessing expiratory dynamics. This preserves the same groupby-median/backoff structure, the same blending, and the same snapping/clipping, but should substantially reduce error on expiratory rows without harming inspiratory rows. I also keep the existing backoff hierarchy and alignment logic intact, only duplicating it for the `u_out` split.'
- What this solution (achieved 3.49852) has done: 'Your current score (1.79127, lower-is-better) is still far above the target (0.15664), so we should improve predictions while keeping your existing “hierarchical median lookup + 4-way blend + snap-to-pressure-grid” core logic intact. The most impactful minimal fix is to stop using binned `u_in`-derived history features inside the lookup keys, because the binning adds discretization error and dramatically increases sparsity, causing frequent backoff to coarse medians. Instead, we keep the same exact mapping/backoff structure but switch the lookup keys to use the raw continuous values (`u_in`, `u_in_cum`, `u_in_lag1`, `u_in_diff1`) rounded to a few decimals (stable, much less sparse than multi-bin cartesian products, yet still fast). We keep the u_out split, the 4-variant blending, the final snapping to the known pressure grid, and the submission writing unchanged, so evaluation semantics and output format remain correct.'
- What this solution (achieved 3.47853) has done: 'Your MAE (3.4985, lower-is-better) is still far above the target (0.1566), so we should legitimately improve predictions while keeping your core “hierarchical median lookup + 4-way blend + snap-to-pressure-grid” structure unchanged. The biggest issue is key sparsity from using several rounded continuous history features inside the *primary* lookup key, which causes heavy backoff to coarse medians and hurts accuracy. I keep all the same features and the same backoff idea, but (1) change the first-stage key to a compact, higher-coverage key and (2) add an intermediate backoff map that uses the history features *without* `u_in_r`, reducing sparsity while still capturing dynamics. Everything else (u_out split, blending weights, snapping/clipping, file search/fallback, submission writing) remains the same.'

# 9. Code solution

## === cell 0
import os
import pandas as pd


INPUT_BASES = [
    "/kaggle/input",  # Kaggle standard
    "/kaggle/data",  # provided in this environment description
    "../input",  # original relative path used by the notebook
    "../data",
]


def first_existing_path(rel_path: str) -> str | None:
    for base in INPUT_BASES:
        candidate = os.path.join(base, rel_path.lstrip("/"))
        if os.path.exists(candidate):
            return candidate
    fname = os.path.basename(rel_path)
    for base in INPUT_BASES:
        if os.path.isdir(base):
            for root, dirs, files in os.walk(base):
                if fname in files:
                    return os.path.join(root, fname)
                if root.count(os.sep) - base.count(os.sep) >= 4:
                    dirs[:] = []
    return None


def read_submission_or_fallback(
    rel_path: str, fallback_df: pd.DataFrame, name: str
) -> pd.DataFrame:
    p = first_existing_path(rel_path)
    if p is None:
        df = fallback_df.copy()
        df["pressure"] = 0.0
        df.attrs["source"] = f"FALLBACK({name})"
        return df

    df = pd.read_csv(p)
    if "id" not in df.columns or "pressure" not in df.columns:
        raise ValueError(
            f"{name} at {p} must contain columns ['id','pressure'], got {list(df.columns)}"
        )
    df = df[["id", "pressure"]].copy()
    df["pressure"] = (
        pd.to_numeric(df["pressure"], errors="coerce").fillna(0.0).astype("float64")
    )
    df.attrs["source"] = p
    return df


def align_on_id(base: pd.DataFrame, other: pd.DataFrame, name: str) -> pd.Series:
    merged = base[["id"]].merge(
        other[["id", "pressure"]], on="id", how="left", validate="one_to_one"
    )
    if merged["pressure"].isna().any():
        merged["pressure"] = merged["pressure"].fillna(0.0)
    return merged["pressure"].astype("float64")


sample_path = first_existing_path(
    "ventilator-pressure-prediction/sample_submission.csv"
) or first_existing_path("sample_submission.csv")
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected input directories."
    )

sub = pd.read_csv(sample_path)[["id", "pressure"]].copy()
sub["pressure"] = (
    pd.to_numeric(sub["pressure"], errors="coerce").fillna(0.0).astype("float64")
)

train_path = first_existing_path(
    "ventilator-pressure-prediction/train.csv"
) or first_existing_path("train.csv")
test_path = first_existing_path(
    "ventilator-pressure-prediction/test.csv"
) or first_existing_path("test.csv")
if train_path is None or test_path is None:
    raise FileNotFoundError(
        "Could not locate train.csv and/or test.csv in expected input directories."
    )

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

for df in (train, test):
    df["R"] = df["R"].astype("int64")
    df["C"] = df["C"].astype("int64")
    df["u_out"] = df["u_out"].astype("int64")
    df["time_step"] = df["time_step"].astype("float64")
    df["u_in"] = df["u_in"].astype("float64")
    df["id"] = df["id"].astype("int64")
    df["breath_id"] = df["breath_id"].astype("int64")

train["pressure"] = train["pressure"].astype("float64")

train["step"] = train.groupby("breath_id", sort=False).cumcount().astype("int16")
test["step"] = test.groupby("breath_id", sort=False).cumcount().astype("int16")

UIN_DECIMALS = 1
UIN_CUM_DECIMALS = 0
UIN_DIFF_DECIMALS = 1

train["u_in_r"] = train["u_in"].round(UIN_DECIMALS).astype("float32")
test["u_in_r"] = test["u_in"].round(UIN_DECIMALS).astype("float32")

train["u_in_cum"] = (
    train.groupby("breath_id", sort=False)["u_in"].cumsum().astype("float64")
)
test["u_in_cum"] = (
    test.groupby("breath_id", sort=False)["u_in"].cumsum().astype("float64")
)
train["u_in_cum_r"] = train["u_in_cum"].round(UIN_CUM_DECIMALS).astype("float32")
test["u_in_cum_r"] = test["u_in_cum"].round(UIN_CUM_DECIMALS).astype("float32")

train["u_in_lag1"] = (
    train.groupby("breath_id", sort=False)["u_in"]
    .shift(1)
    .fillna(0.0)
    .astype("float64")
)
test["u_in_lag1"] = (
    test.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0).astype("float64")
)
train["u_in_lag1_r"] = train["u_in_lag1"].round(UIN_DECIMALS).astype("float32")
test["u_in_lag1_r"] = test["u_in_lag1"].round(UIN_DECIMALS).astype("float32")

train["u_in_diff1"] = (
    train.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0).astype("float64")
)
test["u_in_diff1"] = (
    test.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0).astype("float64")
)
train["u_in_diff1_r"] = train["u_in_diff1"].round(UIN_DIFF_DECIMALS).astype("float32")
test["u_in_diff1_r"] = test["u_in_diff1"].round(UIN_DIFF_DECIMALS).astype("float32")


def build_maps(train_part: pd.DataFrame) -> dict:
    cols = [
        "R",
        "C",
        "step",
        "u_in_r",
        "u_in_cum_r",
        "u_in_lag1_r",
        "u_in_diff1_r",
        "pressure",
    ]
    tp = train_part.loc[:, cols].copy()

    med_map_rcs_hist = (
        tp.groupby(
            ["R", "C", "step", "u_in_cum_r", "u_in_lag1_r", "u_in_diff1_r"],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_rcs_hist"})
    )

    med_map_rcs_u_hist = (
        tp.groupby(
            ["R", "C", "step", "u_in_r", "u_in_cum_r", "u_in_lag1_r", "u_in_diff1_r"],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
    )

    med_map_rcs_u = (
        tp.groupby(["R", "C", "step", "u_in_r"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_rcsu"})
    )

    med_map_rcs = (
        tp.groupby(["R", "C", "step"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_rcs"})
    )

    med_rc = (
        tp.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_rc"})
    )

    global_med = float(tp["pressure"].median())
    return {
        "med_map_rcs_hist": med_map_rcs_hist,
        "med_map_rcs_u_hist": med_map_rcs_u_hist,
        "med_map_rcs_u": med_map_rcs_u,
        "med_map_rcs": med_map_rcs,
        "med_rc": med_rc,
        "global_med": global_med,
    }


maps_insp = build_maps(train.loc[train["u_out"] == 0])
maps_exp = build_maps(train.loc[train["u_out"] == 1])


def predict_with_maps(test_part: pd.DataFrame, maps: dict) -> pd.DataFrame:
    pred = test_part[
        [
            "id",
            "R",
            "C",
            "step",
            "u_in_r",
            "u_in_cum_r",
            "u_in_lag1_r",
            "u_in_diff1_r",
        ]
    ].copy()

    pred = (
        pred.merge(
            maps["med_map_rcs_hist"],
            on=["R", "C", "step", "u_in_cum_r", "u_in_lag1_r", "u_in_diff1_r"],
            how="left",
        )
        .merge(
            maps["med_map_rcs_u_hist"],
            on=[
                "R",
                "C",
                "step",
                "u_in_r",
                "u_in_cum_r",
                "u_in_lag1_r",
                "u_in_diff1_r",
            ],
            how="left",
        )
        .merge(
            maps["med_map_rcs_u"],
            on=["R", "C", "step", "u_in_r"],
            how="left",
        )
        .merge(maps["med_map_rcs"], on=["R", "C", "step"], how="left")
        .merge(maps["med_rc"], on=["R", "C"], how="left")
    )

    pred["pressure"] = (
        pred["pressure_rcs_hist"]
        .fillna(pred["pressure"])
        .fillna(pred["pressure_rcsu"])
        .fillna(pred["pressure_rcs"])
        .fillna(pred["pressure_rc"])
        .fillna(maps["global_med"])
        .astype("float64")
    )
    return pred[["id", "pressure"]]


pred_insp = predict_with_maps(test.loc[test["u_out"] == 0], maps_insp)
pred_exp = predict_with_maps(test.loc[test["u_out"] == 1], maps_exp)

pred_base = pd.concat([pred_insp, pred_exp], axis=0, ignore_index=True)
pred_base = test[["id"]].merge(pred_base, on="id", how="left", validate="one_to_one")
pred_base["pressure"] = pd.to_numeric(pred_base["pressure"], errors="coerce").fillna(
    float(maps_insp["global_med"])
)

pressure_grid = (
    pd.Series(train["pressure"].unique()).astype("float64").sort_values().values
)
idx = pd.Index(pressure_grid).get_indexer(
    pred_base["pressure"].values, method="nearest"
)
pred_base["pressure"] = pressure_grid[idx]

pred_base = pred_base[["id", "pressure"]].copy()
pred_base.attrs["source"] = (
    "FALLBACK_MODEL(u_out_split_hier_median_less_sparse + snap_to_pressure_grid)"
)

sub_1 = read_submission_or_fallback(
    "random-weights-blending-tool-ventilator-pressure/rwb 125 loops.csv",
    pred_base,
    "sub_1",
)
sub_2 = read_submission_or_fallback(
    "bi-lstm-model-pressure-predict-gpu-infer/submission_median_round.csv",
    pred_base,
    "sub_2",
)
sub_3 = read_submission_or_fallback(
    "ensemble-folds-with-median-0-153/submission_median_round_LB153.csv",
    pred_base,
    "sub_3",
)
sub_4 = read_submission_or_fallback(
    "vpp-lstm-baseline-median-pp/submission.csv", pred_base, "sub_4"
)

if str(sub_1.attrs.get("source", "")).startswith("FALLBACK("):
    sub_1 = pred_base.copy()
    sub_1["pressure"] = sub_1["pressure"] * 1.00
    sub_1.attrs["source"] = "INTERNAL_FALLBACK_VARIANT(v1)"

if str(sub_2.attrs.get("source", "")).startswith("FALLBACK("):
    sub_2 = pred_base.copy()
    sub_2["pressure"] = sub_2["pressure"] * 0.995 + 0.05
    idx2 = pd.Index(pressure_grid).get_indexer(
        sub_2["pressure"].values, method="nearest"
    )
    sub_2["pressure"] = pressure_grid[idx2]
    sub_2.attrs["source"] = "INTERNAL_FALLBACK_VARIANT(v2)"

if str(sub_3.attrs.get("source", "")).startswith("FALLBACK("):
    sub_3 = pred_base.copy()
    sub_3["pressure"] = sub_3["pressure"] * 1.005 - 0.05
    idx3 = pd.Index(pressure_grid).get_indexer(
        sub_3["pressure"].values, method="nearest"
    )
    sub_3["pressure"] = pressure_grid[idx3]
    sub_3.attrs["source"] = "INTERNAL_FALLBACK_VARIANT(v3)"

if str(sub_4.attrs.get("source", "")).startswith("FALLBACK("):
    sub_4 = pred_base.copy()
    sub_4["pressure"] = sub_4["pressure"] * 1.00
    sub_4.attrs["source"] = "INTERNAL_FALLBACK_VARIANT(v4)"

p1 = align_on_id(sub, sub_1, "sub_1")
p2 = align_on_id(sub, sub_2, "sub_2")
p3 = align_on_id(sub, sub_3, "sub_3")
p4 = align_on_id(sub, sub_4, "sub_4")

print("Blend sources:")
for i, s in enumerate([sub_1, sub_2, sub_3, sub_4], start=1):
    print(f"  sub_{i}: {s.attrs.get('source', 'unknown')}")




## === cell 1
sub["pressure"] = (
    (p1.values * 0.6) + (p2.values * 0.1) + (p3.values * 0.1) + (p4.values * 0.2)
)

idxf = pd.Index(pressure_grid).get_indexer(sub["pressure"].values, method="nearest")
sub["pressure"] = pressure_grid[idxf]

sub["pressure"] = sub["pressure"].clip(train["pressure"].min(), train["pressure"].max())

sub["id"] = pd.to_numeric(sub["id"], errors="raise").astype("int64")
sub["pressure"] = (
    pd.to_numeric(sub["pressure"], errors="coerce").fillna(0.0).astype("float64")
)

if sub.shape[0] == 0 or sub["id"].isna().any():
    raise RuntimeError("Submission is empty or has invalid ids.")
if sub["id"].duplicated().any():
    raise RuntimeError("Submission has duplicate ids after blending/alignment.")

sub.to_csv("submission.csv", index=False)
sub.head(5)

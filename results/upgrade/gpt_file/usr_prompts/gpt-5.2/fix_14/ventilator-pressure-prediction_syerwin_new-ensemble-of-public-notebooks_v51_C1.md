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

0.1439792876612579

# 6. Current score

2.87368

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your notebook fails because it tries to read four external “submission” files from `../input/...` that are not present in your environment, so `sub_1`..`sub_4` never get defined and the blend crashes. To make it run end-to-end and still follow the same “blend submissions” core idea, I change it to (1) load those files only if they exist and (2) otherwise fall back to a valid baseline submission (all zeros) using the provided `sample_submission.csv`. This guarantees a `submission.csv` is always written with the correct columns and row count; if any of the blend files exist, it blend them with the same weights as your original code.'
- What this solution (achieved 4.95961) has done: 'Your current score is extremely far from the target (MAE 17.65 vs 0.144; lower is better), and that’s because the notebook is effectively submitting an all-zeros pressure vector when the external blend files aren’t available. To move toward the target while keeping changes minimal, I keep the same “produce a submission dataframe and write submission.csv” flow but replace the fallback-from-missing-files behavior with a simple in-notebook baseline model trained on `train.csv` and applied to `test.csv`. This uses only the provided competition data and keeps runtime under the limit by training a lightweight linear model on a small, relevant feature set. The output still be `id,pressure` with exactly the same row count as `sample_submission.csv`.'
- What this solution (achieved 4.95961) has done: 'Your current score (MAE 4.9596; lower is better) is still far from the target (~0.144), so we should improve the model while keeping the same overall “train a simple regressor on engineered lag/cumsum features, predict on test, write submission.csv” core flow. The biggest low-risk gain here is to correctly create features in-place (your `for df in (train, test)` loop currently edits a temporary `df` and then reassigns via a brittle `"pressure" in df.columns` check), and to ensure the submission aligns 1:1 with `test` by predicting in the exact test row order and then writing `id,pressure` directly (no merge that can silently misalign if `id` duplicates). These are minimal semantic fixes plus a small regularization tuning for Ridge that tends to reduce MAE without changing the approach. The result should move the score substantially closer to the target while preserving your model family, feature set, and evaluation semantics.'
- What this solution (achieved 4.95831) has done: 'Your current MAE (4.9596) is far worse than the target (0.144), so we should improve accuracy while keeping the same “Ridge on engineered lag/cumsum features” core approach. The biggest win without changing the model family is to respect the competition’s scoring rule: only inspiratory rows (u_out==0) are scored, so we can safely force expiratory predictions (u_out==1) to a reasonable constant rather than letting the linear model produce arbitrary values there. Additionally, ventilator pressure takes discrete values; snapping predictions to the nearest known training pressure level typically reduces MAE with minimal semantic change. These two post-processing steps keep the training loop and model intact and should move the score substantially toward the target.'
- What this solution (achieved 7.74021) has done: 'Your score is still far above the target (MAE 4.96 vs 0.144; lower is better), so we make small, metric-aligned changes without changing the core “Ridge on engineered lag/cumsum features” approach. First, we add a few additional *lightweight per-breath features* (lags of `u_in`, cumulative `u_out`, and simple interaction terms `u_in*R` and `u_in*C`) that stay within the same linear-model family but typically reduce MAE substantially. Second, instead of setting expiratory (`u_out==1`) predictions to a global mean, we set them to the **last predicted inspiratory pressure of the same breath**, which is usually closer to the true expiratory pressures and avoids breath-to-breath mismatch. We keep the same snapping-to-discrete-pressure post-processing and produce the same `submission.csv` format.'
- What this solution (achieved 8.19259) has done: 'Your MAE (7.74; lower is better) is far worse than the target (0.144), and the biggest issue is that the current model ignores the time-series nature of the problem by training a single global Ridge across all breaths; a minimal, high-impact fix while preserving the same “Ridge on engineered lag/cumsum features + snap-to-levels” core logic is to train separate Ridge models per (R, C) lung setting and predict with the matching model for each test row. This keeps the same feature engineering, same model family, same loss semantics, and same post-processing, but aligns with the competition structure where pressure dynamics differ strongly by R and C. I also remove an accidental redundant sort block in prediction (no semantic benefit) and ensure the per-breath expiratory fill uses the last inspiratory prediction after the per-(R,C) prediction is assembled. The output remains a valid `submission.csv` with `id,pressure` in the original test row order.'
- What this solution (achieved 8.22522) has done: 'Your current score (8.19 MAE; lower is better) is far from the target (0.144), so we should improve accuracy while keeping the same core approach: Ridge regression on the same engineered lag/cumsum features with discrete-pressure snapping and u_out-aware postprocessing. The biggest low-risk issue is that the model currently predicts each time step independently and only uses “past” lags; adding a minimal set of *future* (lead) features per breath (which are available at inference because the entire control sequence is known) typically reduces error a lot without changing the model family or training loop. I also make the expiratory (u_out==1) fill more robust by using the last *inspiratory* prediction computed within each breath after lead features are added (same idea as you have, just kept consistent after new features). All paths and output format remain unchanged and it still writes `submission.csv`.'
- What this solution (achieved 8.38322) has done: 'Your current MAE (8.225) is far above the target (0.144), so we should improve accuracy while keeping the same core approach: per-(R,C) Ridge on the same lag/cumsum/lead features plus expiratory handling and discrete-level snapping. The biggest low-risk fix is to stop treating each row independently by adding a minimal “previous predicted pressure” feature via a two-pass (teacher-forcing on train, recursive on test) refinement, but still using Ridge and the same feature set. We also ensure the discrete snapping is applied after this refinement and keep expiratory rows filled from the last inspiratory prediction per breath (now using the refined prediction). These changes keep runtime low, preserve your model family/training style, and typically move MAE substantially toward the target.'
- What this solution (achieved 2.57148) has done: 'Your current MAE is far worse than the target (lower is better), and the biggest likely cause in this notebook is a silent but fatal misalignment: you generate predictions in `test_sorted` order but then write them out paired with `test["id"]` in the *original* (unsorted) order. I keep your exact modeling approach (per-(R,C) Ridge + recursive lag feature + expiratory fill + snap-to-discrete levels) and make the minimal fix to guarantee 1:1 row alignment by carrying `id` through the sorted prediction frame and writing submission directly from that same frame. I also remove one unused variable and add a small safety assert on row counts to prevent producing an invalid/misaligned submission. These changes should substantially move the score toward your target without changing core logic.'
- What this solution (achieved 2.87368) has done: 'The timeout is dominated by per-row `.iterrows()` loops that call `Pipeline.predict()` thousands of times (both when building student lag features on train and during test inference). I keep the exact modeling logic (Ridge + StandardScaler, same features, same per-breath autoregressive state update) but replace row-wise DataFrame construction/prediction with fast NumPy-based step-wise prediction using the fitted scaler and ridge coefficients directly (mathematically identical). I also cut repeated sorting/copying and avoid expensive `.loc` assignments inside loops by writing into preallocated NumPy arrays. These changes preserve the same predictions up to negligible floating-point differences while reducing runtime by orders of magnitude.'
- What this solution (achieved 2.87368) has done: 'I fix the failing assertion by ensuring the submission uses the competition’s globally-unique `id` and is sorted by `id` before writing, rather than assuming the raw `test.csv` row order is already monotonic. To keep the core modeling logic unchanged, predictions still be generated on the `test_sorted` frame and then aligned back to `id` with a 1:1 merge-free mapping. I also add a strict uniqueness check for `id` to prevent silent misalignment, and remove the brittle monotonicity assertion in favor of explicitly sorting the output. This is score-neutral (format/alignment correctness only) and guarantees a valid `submission.csv` is produced end-to-end.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

sub = pd.read_csv(sub_path)

train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train = pd.read_csv(train_path, dtype=train_dtypes)
test = pd.read_csv(test_path, dtype=test_dtypes)




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

    gb = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = gb["u_in"].shift(1).fillna(0.0).astype("float32")
    df["u_out_lag1"] = gb["u_out"].shift(1).fillna(0.0).astype("float32")
    df["u_in_cum"] = gb["u_in"].cumsum().astype("float32")
    df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype("float32")

    df["u_in_lag2"] = gb["u_in"].shift(2).fillna(0.0).astype("float32")
    df["u_in_diff2"] = (df["u_in_lag1"] - df["u_in_lag2"]).astype("float32")
    df["u_out_cum"] = gb["u_out"].cumsum().astype("float32")

    df["u_in_x_R"] = (df["u_in"] * df["R"].astype("float32")).astype("float32")
    df["u_in_x_C"] = (df["u_in"] * df["C"].astype("float32")).astype("float32")

    df["u_in_lead1"] = gb["u_in"].shift(-1).fillna(0.0).astype("float32")
    df["u_out_lead1"] = gb["u_out"].shift(-1).fillna(0.0).astype("float32")
    df["u_in_lead2"] = gb["u_in"].shift(-2).fillna(0.0).astype("float32")

    return df


train = add_features(train)
test = add_features(test)



## === cell 3
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

base_features = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_out_lag1",
    "u_in_cum",
    "u_in_diff1",
    "u_in_lag2",
    "u_in_diff2",
    "u_out_cum",
    "u_in_x_R",
    "u_in_x_C",
    "u_in_lead1",
    "u_out_lead1",
    "u_in_lead2",
]

train_insp = train[train["u_out"] == 0].copy()
train_insp = train_insp.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

train_insp["pressure_lag1_true"] = (
    train_insp.groupby("breath_id", sort=False)["pressure"]
    .shift(1)
    .fillna(0.0)
    .astype("float32")
)

features = base_features + ["pressure_lag1_true"]

X_train_tf = train_insp[features]
y_train = train_insp["pressure"]




## === cell 4
def _extract_linear_parts(pipeline: Pipeline):
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    ridge: Ridge = pipeline.named_steps["ridge"]
    mean_ = scaler.mean_.astype(np.float64, copy=False)
    scale_ = scaler.scale_.astype(np.float64, copy=False)
    coef_ = ridge.coef_.astype(np.float64, copy=False)
    intercept_ = float(ridge.intercept_)
    return mean_, scale_, coef_, intercept_


def _predict_one_step(
    x_row_f64: np.ndarray,
    mean_: np.ndarray,
    scale_: np.ndarray,
    coef_: np.ndarray,
    intercept_: float,
) -> float:
    z = (x_row_f64 - mean_) / scale_
    return float(z.dot(coef_) + intercept_)


models = {}
global_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=3.0, random_state=42)),
    ]
)
global_model.fit(X_train_tf, y_train)

for (r, c), grp in train_insp.groupby(["R", "C"], sort=False):
    Xg = grp[features]
    yg = grp["pressure"]
    m = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=3.0, random_state=42)),
        ]
    )
    m.fit(Xg, yg)
    models[(int(r), int(c))] = m

student_models = {}
global_student_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=3.0, random_state=42)),
    ]
)

train_insp_student = train_insp.copy()
train_insp_student["pressure_lag1_pred"] = 0.0

g_mean, g_scale, g_coef, g_intercept = _extract_linear_parts(global_model)

base_mat = train_insp_student[base_features].to_numpy(dtype=np.float64, copy=False)
lag_pred_all = np.empty(len(train_insp_student), dtype=np.float32)

gb_idx = train_insp_student.groupby("breath_id", sort=False).indices
for breath_id, idx in gb_idx.items():
    prev_p = 0.0
    for pos in idx:
        lag_pred_all[pos] = np.float32(prev_p)
        x = np.empty(len(base_features) + 1, dtype=np.float64)
        x[:-1] = base_mat[pos]
        x[-1] = prev_p  # "pressure_lag1_true" slot in teacher model inference
        prev_p = _predict_one_step(x, g_mean, g_scale, g_coef, g_intercept)

train_insp_student["pressure_lag1_pred"] = lag_pred_all

student_features = base_features + ["pressure_lag1_pred"]
X_train_student_global = train_insp_student[student_features]
global_student_model.fit(X_train_student_global, y_train)

for (r, c), grp in train_insp_student.groupby(["R", "C"], sort=False):
    Xg = grp[student_features]
    yg = grp["pressure"]
    m = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=3.0, random_state=42)),
        ]
    )
    m.fit(Xg, yg)
    student_models[(int(r), int(c))] = m



## === cell 5
test_sorted = test.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

student_linear = {k: _extract_linear_parts(m) for k, m in student_models.items()}
global_student_linear = _extract_linear_parts(global_student_model)

base_test_mat = test_sorted[base_features].to_numpy(dtype=np.float64, copy=False)
u_out_arr = test_sorted["u_out"].to_numpy(dtype=np.int8, copy=False)

preds_all = np.empty(len(test_sorted), dtype=np.float64)

gb_test_idx = test_sorted.groupby("breath_id", sort=False).indices
for breath_id, idx in gb_test_idx.items():
    first_pos = idx[0]
    r = int(test_sorted["R"].iat[first_pos])
    c = int(test_sorted["C"].iat[first_pos])

    mean_, scale_, coef_, intercept_ = student_linear.get((r, c), global_student_linear)

    prev_pred = 0.0
    for pos in idx:
        x = np.empty(len(base_features) + 1, dtype=np.float64)
        x[:-1] = base_test_mat[pos]
        x[-1] = prev_pred  # "pressure_lag1_pred"
        p = _predict_one_step(x, mean_, scale_, coef_, intercept_)
        preds_all[pos] = p
        if u_out_arr[pos] == 0:
            prev_pred = p  # only advance state during inspiratory phase

test_sorted["pred"] = preds_all

exp_mask = test_sorted["u_out"].to_numpy(dtype=np.int8, copy=False) == 1
pred_vec = test_sorted["pred"].to_numpy(dtype=np.float64, copy=False)
pred_vec[exp_mask] = 0.0
test_sorted["pred"] = pred_vec

pressure_levels = np.sort(train["pressure"].unique().astype(np.float64))
test_pred = test_sorted["pred"].to_numpy(dtype=np.float64, copy=False)

insp_mask = ~exp_mask
insp_pred = test_pred[insp_mask]

idxs = np.searchsorted(pressure_levels, insp_pred, side="left")
idx0 = np.clip(idxs - 1, 0, len(pressure_levels) - 1)
idx1 = np.clip(idxs, 0, len(pressure_levels) - 1)
p0 = pressure_levels[idx0]
p1 = pressure_levels[idx1]
choose_right = np.abs(p1 - insp_pred) < np.abs(insp_pred - p0)
snapped_insp = np.where(choose_right, p1, p0)

snapped_all = test_pred.copy()
snapped_all[insp_mask] = snapped_insp
snapped_all[exp_mask] = 0.0

test_ids_sorted = test_sorted["id"].to_numpy(copy=False)
if pd.Index(test_ids_sorted).has_duplicates:
    raise ValueError(
        "test_sorted contains duplicate ids; cannot map predictions safely."
    )

pred_by_id = pd.Series(snapped_all, index=test_ids_sorted)

submission = pd.DataFrame({"id": test["id"].to_numpy(copy=False)})
if pd.Index(submission["id"]).has_duplicates:
    raise ValueError("test contains duplicate ids; unexpected for this competition.")

submission["pressure"] = (
    submission["id"].map(pred_by_id).to_numpy(dtype=np.float64, copy=False)
)

assert len(submission) == len(
    test
), "Row count mismatch: submission and test must match 1:1"
assert submission["pressure"].notna().all(), "Found NaN predictions after id mapping"

submission = submission.sort_values("id", kind="mergesort").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

submission.head(5)

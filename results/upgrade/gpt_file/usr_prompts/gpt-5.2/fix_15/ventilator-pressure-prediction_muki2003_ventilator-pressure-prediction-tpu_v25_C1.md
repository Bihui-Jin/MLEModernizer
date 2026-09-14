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

No external packages required in the script and installed.

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

0.1388768686943292

# 6. Current score

3.42135

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.26536) has done: 'We need to fix the pipeline failing early because TensorFlow cannot import in this environment due to a protobuf binary mismatch; that cascades into undefined variables and no submission file. The minimal safe fix is to keep the same feature engineering and pressure discretization logic, but replace the TensorFlow model with a lightweight, deterministic fallback that runs with the existing installed stack (NumPy/Pandas/Scikit-learn) and still produces a valid `submission.csv`. Because no score was yielded, the priority is end-to-end correctness; the fallback model (Ridge regression per time step on the same engineered/scaled features) should be a reasonable baseline and move toward the target MAE without changing evaluation semantics. I also fix the cell numbering to start at 1 and ensure paths work under `/kaggle/input/...` with a safe fallback to the provided `../input/...` layout.'
- What this solution (achieved 6.26392) has done: 'Your current score is far worse than the target (lower is better), so we should make a small, legitimate improvement without changing the overall approach (still RobustScaler + linear Ridge + fold-averaging + pressure-grid snapping). The biggest issue is that the validation metric you print is “all phases, unmasked”, but Kaggle scores only the inspiratory phase (`u_out == 0`), so the model selection/calibration is currently misaligned. I (1) fit the scaler only on the training portion inside each fold to avoid subtle fold leakage, and (2) report/track MAE on the correct inspiratory mask, while keeping the final training/prediction logic the same. I also apply the same inspiratory-aware idea to a tiny, safe post-process: keep the existing pressure snapping/clipping but additionally set predictions to 0 when `u_out==1` in test (expiratory phase is unscored, and this often stabilizes without hurting the scored portion).'
- What this solution (achieved 5.0935) has done: 'Your current MAE (6.26392) is far worse than the target (0.1389), so we should make one small, legitimate improvement that keeps the same core pipeline (same features, same scaler+Ridge, same fold averaging, same pressure-grid snapping). The biggest remaining mismatch is that Kaggle scores only inspiratory rows (`u_out==0`), so we should train Ridge using sample weights that ignore expiratory rows (weight 0) and optimize the Ridge `alpha` mildly using inspiratory-only CV MAE (this is still Ridge, not a new model). This aligns the learned coefficients to the scored portion without changing architecture/training loop style. We keep the existing post-process (pressure snapping and setting test `u_out==1` to 0) and still output a valid `submission.csv`.'
- What this solution (achieved 5.09351) has done: 'Your current MAE (5.0935, lower is better) is still far from the target (0.1389), so we need a small but meaningful improvement while keeping the same overall Ridge+RobustScaler+fold-averaging+pressure-grid snapping pipeline. The main issue is the current KFold splits breaths randomly, which mixes identical (R,C,time_step) dynamics across folds in a way that hurts generalization; switching to `GroupKFold` by `breath_id` is a minimal, competition-appropriate fix that preserves the same model and training approach but validates more realistically and typically improves score. I also make the per-fold row-index creation vectorized (faster, same semantics) to stay within the 600s timeout. Everything else (features, Ridge, sample-weighting for inspiratory rows, snapping, and submission writing) remains the same.'
- What this solution (achieved 5.01115) has done: 'Your current MAE (5.09, lower is better) is far above the target (0.139), so we need a legitimate improvement while keeping the same Ridge + RobustScaler + GroupKFold + fold-averaging + pressure-grid snapping core. The biggest remaining low-risk gain is to make the linear model aware of the strong per-breath temporal structure by adding a few simple within-breath lag/cumulative features (same idea as your existing `u_in_lag1/u_in_diff1/u_in_cumsum`, just extended minimally). This preserves the same training loop and model type, but typically reduces error substantially because pressure depends on recent valve history and accumulated volume. I also keep the inspiratory-only weighting and the existing post-process, and still write a valid `submission.csv`.'
- What this solution (achieved 3.56903) has done: 'Your score is much worse than the target (lower is better), so we should make a small but meaningful improvement while keeping the exact same core pipeline (RobustScaler + Ridge + GroupKFold + fold-averaging + pressure-grid snapping). The biggest remaining low-risk win is to add a few more within-breath temporal features that linear models can exploit: short-window rolling means/max, plus a simple interaction between accumulated volume and lung attributes (R, C). These do not change the model type, training loop, or loss—only enrich the existing feature matrix—so evaluation semantics stay identical but MAE should move substantially toward the target. I also keep the existing inspiratory-only sample-weighting and the test `u_out==1` postprocess as-is.'
- What this solution (achieved 3.42135) has done: 'Your current MAE (3.569) is still far above the target (0.139, lower is better), so we make one minimal, score-aligned improvement without changing the core pipeline (same Ridge + RobustScaler + GroupKFold + fold averaging + pressure snapping). The main fix is to make the model aware of the strong “per-time-step” nature of the problem by adding a single deterministic feature: the within-breath time index (0–79), which helps a linear model learn the baseline pressure trajectory. To further align with the metric (inspiratory only), we keep training weights as-is but also apply the u_out==1 post-process to set test pressures to the nearest plausible baseline (0 after snapping stays consistent). Everything else (files/paths, training loop, model, snapping, and submission writing) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/ventilator-pressure-prediction",
    "../input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "../input",
]


def _pick_base():
    for b in BASE_CANDIDATES:
        if os.path.exists(os.path.join(b, "train.csv")) and os.path.exists(
            os.path.join(b, "test.csv")
        ):
            return b
    return "../input/ventilator-pressure-prediction"


BASE = _pick_base()

TRAIN_PATH = os.path.join(BASE, "train.csv")
TEST_PATH = os.path.join(BASE, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "pressure": "float32",
    },
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    },
)



## === cell 2
assert (
    len(train) % 80 == 0
), "Train rows must be multiple of 80 (one breath = 80 time steps)."
assert (
    len(test) % 80 == 0
), "Test rows must be multiple of 80 (one breath = 80 time steps)."




## === cell 3
def add_features_fast(df: pd.DataFrame) -> None:
    """
    Minimal, score-relevant improvement:
    - Keep existing lag/diff/cumsum/rolling/interactions (core logic unchanged).
    - Add within-breath time index (0..79) as a numeric feature. This is a tiny,
      deterministic feature that helps a linear model learn the baseline temporal
      pressure profile and usually improves MAE without changing training semantics.
    """
    n = len(df)
    n_breaths = n // 80

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, 80)
    u_out = df["u_out"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, 80)

    u_in_lag1 = np.empty_like(u_in)
    u_in_lag1[:, 0] = 0.0
    u_in_lag1[:, 1:] = u_in[:, :-1]

    u_in_diff1 = u_in - u_in_lag1
    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32)

    u_in_lag2 = np.empty_like(u_in)
    u_in_lag2[:, :2] = 0.0
    u_in_lag2[:, 2:] = u_in[:, :-2]
    u_in_diff2 = u_in - u_in_lag2

    u_out_lag1 = np.empty_like(u_out)
    u_out_lag1[:, 0] = 0.0
    u_out_lag1[:, 1:] = u_out[:, :-1]
    u_out_cumsum = np.cumsum(u_out, axis=1, dtype=np.float32)

    u_in_pad2 = np.pad(u_in, ((0, 0), (2, 0)), mode="constant", constant_values=0.0)
    u_in_roll3 = (u_in_pad2[:, 2:] + u_in_pad2[:, 1:-1] + u_in_pad2[:, :-2]) / 3.0

    u_in_pad4 = np.pad(u_in, ((0, 0), (4, 0)), mode="constant", constant_values=0.0)
    u_in_roll5 = (
        u_in_pad4[:, 4:]
        + u_in_pad4[:, 3:-1]
        + u_in_pad4[:, 2:-2]
        + u_in_pad4[:, 1:-3]
        + u_in_pad4[:, :-4]
    ) / 5.0

    u_in_roll3_max = np.maximum.reduce(
        [u_in_pad2[:, 2:], u_in_pad2[:, 1:-1], u_in_pad2[:, :-2]]
    )

    R = df["R"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, 80)
    C = df["C"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, 80)

    inv_R = 1.0 / (R + 1e-6)
    inv_C = 1.0 / (C + 1e-6)
    u_in_cumsum_div_R = u_in_cumsum * inv_R
    u_in_cumsum_div_C = u_in_cumsum * inv_C

    t_idx = np.arange(80, dtype=np.float32)[None, :].repeat(n_breaths, axis=0)

    df["u_in_lag1"] = u_in_lag1.reshape(-1)
    df["u_in_diff1"] = u_in_diff1.reshape(-1)
    df["u_in_cumsum"] = u_in_cumsum.reshape(-1)

    df["u_in_lag2"] = u_in_lag2.reshape(-1)
    df["u_in_diff2"] = u_in_diff2.reshape(-1)

    df["u_out_lag1"] = u_out_lag1.reshape(-1)
    df["u_out_cumsum"] = u_out_cumsum.reshape(-1)

    df["u_in_roll3"] = u_in_roll3.reshape(-1)
    df["u_in_roll5"] = u_in_roll5.reshape(-1)
    df["u_in_roll3_max"] = u_in_roll3_max.reshape(-1)

    df["u_in_cumsum_div_R"] = u_in_cumsum_div_R.reshape(-1)
    df["u_in_cumsum_div_C"] = u_in_cumsum_div_C.reshape(-1)

    df["t_idx"] = t_idx.reshape(-1)


add_features_fast(train)
add_features_fast(test)



## === cell 4
targets = train["pressure"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, 1)
test_id = test["id"].copy()

u_out_train = train["u_out"].to_numpy(dtype=np.int8, copy=False)  # row-level
u_out_test = test["u_out"].to_numpy(dtype=np.int8, copy=False)  # row-level

breath_id_train = train["breath_id"].to_numpy(dtype=np.int32, copy=False)

train.drop(columns=["id", "breath_id", "pressure", "time_step"], inplace=True)
test.drop(columns=["id", "breath_id", "time_step"], inplace=True)



## === cell 5
train_mat = train.to_numpy(dtype=np.float32, copy=False)
test_mat = test.to_numpy(dtype=np.float32, copy=False)

train_re = train_mat.reshape(-1, 80, train_mat.shape[-1])
test_re = test_mat.reshape(-1, 80, test_mat.shape[-1])

y_all = targets.reshape(-1).astype(np.float32, copy=False)



## === cell 6
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX



## === cell 7
EPOCH = 260  # kept for parity with original script (not used by Ridge)
BATCH_SIZE = 512  # kept for parity with original script (not used by Ridge)

n_splits = 7

gkf = GroupKFold(n_splits=n_splits)

alpha_grid = [0.3, 1.0, 3.0]

best_alpha = None
best_cv_mae_insp = float("inf")

all_breaths = np.arange(train_re.shape[0], dtype=np.int32)
groups = breath_id_train.reshape(-1, 80)[:, 0]

for alpha in alpha_grid:
    fold_maes = []
    print("=" * 20, f"Alpha={alpha}", "=" * 20)

    for fold, (train_idx_b, valid_idx_b) in enumerate(
        gkf.split(all_breaths, groups=groups), start=1
    ):
        train_rows = (
            train_idx_b[:, None] * 80 + np.arange(80, dtype=np.int32)[None, :]
        ).reshape(-1)
        valid_rows = (
            valid_idx_b[:, None] * 80 + np.arange(80, dtype=np.int32)[None, :]
        ).reshape(-1)

        rb = RobustScaler()
        rb.fit(train_mat[train_rows])

        X_all_scaled = rb.transform(train_mat).astype(np.float32, copy=False)
        X_tr, y_tr = X_all_scaled[train_rows], y_all[train_rows]
        X_va, y_va = X_all_scaled[valid_rows], y_all[valid_rows]

        w_tr = (u_out_train[train_rows] == 0).astype(np.float32, copy=False)

        model = Ridge(alpha=alpha, random_state=SEED)
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        va_pred = model.predict(X_va)

        u_out_va = u_out_train[valid_rows]
        insp_mask = u_out_va == 0
        if np.any(insp_mask):
            va_mae_insp = float(np.mean(np.abs(va_pred[insp_mask] - y_va[insp_mask])))
        else:
            va_mae_insp = float(np.mean(np.abs(va_pred - y_va)))

        fold_maes.append(va_mae_insp)

    cv_mae = float(np.mean(fold_maes))
    print(f"Alpha={alpha} CV MAE (inspiratory only): {cv_mae:.6f}")
    if cv_mae < best_cv_mae_insp:
        best_cv_mae_insp = cv_mae
        best_alpha = alpha

print(
    f"Selected alpha={best_alpha} with inspiratory-only CV MAE={best_cv_mae_insp:.6f}"
)

test_fold_preds = []
ridge_params = dict(alpha=best_alpha, random_state=SEED)

for fold, (train_idx_b, valid_idx_b) in enumerate(
    gkf.split(all_breaths, groups=groups), start=1
):
    print("-" * 30, ">", f"Fold {fold}", "<", "-" * 30)

    train_rows = (
        train_idx_b[:, None] * 80 + np.arange(80, dtype=np.int32)[None, :]
    ).reshape(-1)
    valid_rows = (
        valid_idx_b[:, None] * 80 + np.arange(80, dtype=np.int32)[None, :]
    ).reshape(-1)

    rb = RobustScaler()
    rb.fit(train_mat[train_rows])

    X_all_scaled = rb.transform(train_mat).astype(np.float32, copy=False)
    X_test_all_scaled = rb.transform(test_mat).astype(np.float32, copy=False)

    X_tr, y_tr = X_all_scaled[train_rows], y_all[train_rows]
    X_va, y_va = X_all_scaled[valid_rows], y_all[valid_rows]

    w_tr = (u_out_train[train_rows] == 0).astype(np.float32, copy=False)

    model = Ridge(**ridge_params)
    model.fit(X_tr, y_tr, sample_weight=w_tr)

    va_pred = model.predict(X_va)

    u_out_va = u_out_train[valid_rows]
    insp_mask = u_out_va == 0
    if np.any(insp_mask):
        va_mae_insp = float(np.mean(np.abs(va_pred[insp_mask] - y_va[insp_mask])))
    else:
        va_mae_insp = float(np.mean(np.abs(va_pred - y_va)))

    va_mae_all = float(np.mean(np.abs(va_pred - y_va)))
    print(f"Fold {fold} MAE (inspiratory only, u_out==0): {va_mae_insp:.6f}")
    print(f"Fold {fold} MAE (all phases, unmasked):       {va_mae_all:.6f}")

    fold_pred = (
        model.predict(X_test_all_scaled).reshape(-1, 1).astype(np.float32, copy=False)
    )
    test_fold_preds.append(fold_pred)



## === cell 8
predictions_stack = np.concatenate(test_fold_preds, axis=1)  # (n_rows_test, n_folds)
mean_pre = np.mean(predictions_stack, axis=1)  # (n_rows_test,)

rounding_pre = (
    np.round((mean_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX).astype(
    np.float32, copy=False
)

clipped_pre = clipped_pre.copy()
clipped_pre[u_out_test == 1] = 0.0

mean_pre.shape, clipped_pre.shape



## === cell 9
submission_file = pd.read_csv(SAMPLE_SUB_PATH)

assert len(submission_file) == len(
    clipped_pre
), f"Submission length mismatch: {len(submission_file)} vs {len(clipped_pre)}"

submission_file["pressure"] = clipped_pre
submission_file.to_csv("submission.csv", index=False)

submission_file.head()



## === cell 10
assert os.path.exists("submission.csv")
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["id", "pressure"]
assert len(sub_check) == 603600, f"Unexpected submission rows: {len(sub_check)}"
print("Wrote submission.csv with shape:", sub_check.shape)
print(sub_check.head())

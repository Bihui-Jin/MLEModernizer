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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.140469241408593

# 6. Current score

6.39764

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.39753) has done: 'The crash comes from missing external `.npy` prediction artifacts referenced under `../input/new-model-same-old-mistakes/`, so `train/test` never get any features and later reshapes/divisions fail. I keep the overall “stacking/linear-regression ensemble over per-timestep features + optional discretization” structure, but add a safe fallback that builds simple, local per-breath features from the provided CSVs when those artifacts are not present. I also fix the KFold split to operate on breaths (not mismatched `targets` shape) and avoid mutating `test` inside the fold loop so predictions aggregate correctly. Finally, the script always write a valid `submission_*.csv` with `id,pressure` columns.'
- What this solution (achieved 6.39788) has done: 'Your current score (6.39753, lower-is-better) is far from the target (0.14047), and the main reason is that the code is falling back to very weak “basic local features” instead of using the intended strong external OOF/test prediction artifacts. The smallest meaningful change toward the target is to actually load those external artifacts from the correct Kaggle input location (the current relative path likely resolves to nowhere), while keeping the same stacking + LinearRegression + discretization logic. I add a robust path resolver that tries the common Kaggle dataset mount points and only uses the fallback if nothing exists. This should move the score much closer to the target with minimal code changes and still produce valid `submission.csv`.'
- What this solution (achieved 6.39788) has done: 'Your current score is far worse than the target (lower-is-better), and the most likely reason is that you are still falling back to weak “basic local features” because the external prediction artifacts aren’t being found/loaded correctly. I make the smallest change that increases the chance we actually use the intended external OOF/test `.npy` artifacts: robustly auto-discover the correct Kaggle dataset root under `/kaggle/input/` and validate that each submodel folder exists before attempting to load it. If some listed submodels are missing, we skip only those (instead of triggering a full fallback), keeping the same stacking + LinearRegression + discretization logic. This should move the MAE substantially toward the target while keeping the overall approach identical and still writing a valid `submission.csv`.'
- What this solution (achieved 6.39764) has done: 'Your current score (6.39788, lower-is-better) is far from the target (0.14047), so we should focus on the most likely root cause: you are still not actually using the strong external OOF/test prediction artifacts, and the fallback features are too weak. I make the smallest changes that (1) reliably locate the external dataset under `/kaggle/input` by auto-discovering the correct folder name (even if it’s been “slugified” by Kaggle), and (2) fix a subtle but critical bug in `add_preds`: the loaded `oof_idx` indices are almost certainly “row indices” (0..N*80-1), but you are using them as “breath indices” (0..N-1), which silently scrambles/zeros most training features and ruins MAE. With those two minimal fixes, the stacking + LinearRegression + discretization logic stays the same, but it should move the score sharply toward your target (assuming the dataset is attached and the artifacts exist). The script still fall back safely and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import math
import os
from pathlib import Path
import random
import gc
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression



## === cell 1
pass



## === cell 2
FOLDS = 5
SEED = 23
DEBUG = False



## === cell 3
pass




## === cell 4
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


set_seed(SEED)



## === cell 5
pass



## === cell 6
train_df = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/test.csv")

pressure_values = np.array(sorted(train_df["pressure"].unique().tolist()))

if DEBUG:
    train_df = train_df.iloc[: 80 * 2000].copy()
    test_df = test_df.iloc[: 80 * 500].copy()

sub = test_df[["id"]].copy()

pressure_values.shape



## === cell 7
if (len(train_df) % 80) != 0 or (len(test_df) % 80) != 0:
    raise ValueError(
        "Expected train/test rows to be divisible by 80 (fixed breath length)."
    )

u_out_0 = train_df["u_out"].to_numpy().reshape(-1, 80)
targets = train_df[["pressure"]].to_numpy().reshape(-1, 80)
test_indices = test_df.index

train_breath_ids = train_df["breath_id"].to_numpy().reshape(-1, 80)[:, 0]
test_breath_ids = test_df["breath_id"].to_numpy().reshape(-1, 80)[:, 0]




## === cell 8
def add_preds(
    model_path, train, test, n_folds, with_discrete=True, size=80, preds_folder="preds"
):
    """
    Loads OOF/test predictions from a Kaggle dataset folder and appends them as new features.

    Minimal but critical fix to move score toward target:
    - Many public artifact packs store `val_{fold}.npy` as per-row indices (0..N*80-1),
      but this pipeline needs per-breath indices (0..N-1) because `train` is shaped (N,80,F).
      We therefore detect row-level indices and convert them to unique breath indices via //80.
      Without this, most features remain zeros/misaligned, leading to very poor MAE.
    """
    num_features_to_add = 2 if with_discrete else 1
    train_1 = np.zeros(
        (train.shape[0], train.shape[1], train.shape[2] + num_features_to_add),
        dtype=np.float32,
    )
    test_1 = np.zeros(
        (test.shape[0], test.shape[1], test.shape[2] + num_features_to_add),
        dtype=np.float32,
    )

    test_preds = []
    test_preds_discrete = []

    for fold in range(n_folds):
        idx_path = f"{model_path}/indices/val_{fold}.npy"
        if not os.path.exists(idx_path):
            raise FileNotFoundError(idx_path)

        oof_idx = np.load(idx_path)

        oof_idx = np.asarray(oof_idx)
        if oof_idx.ndim != 1:
            oof_idx = oof_idx.ravel()
        if oof_idx.size > 0 and (oof_idx.max() >= train.shape[0]):
            oof_idx = np.unique(oof_idx // train.shape[1]).astype(np.int64)

        oof_preds = np.load(f"{model_path}/{preds_folder}/val_pred_fold_{fold}.npy")
        oof_preds = oof_preds[np.mod(np.arange(len(oof_preds)), size) < train.shape[1]]
        oof_preds = oof_preds.reshape(-1, train.shape[1], 1)

        test_pred = np.load(f"{model_path}/{preds_folder}/test_pred_fold_{fold}.npy")
        test_pred = test_pred[np.mod(np.arange(len(test_pred)), size) < train.shape[1]]
        test_preds.append(test_pred)

        if with_discrete:
            oof_preds_discrete = np.load(
                f"{model_path}/{preds_folder}/val_pred_fold_{fold}_discrete.npy"
            )
            oof_preds_discrete = oof_preds_discrete[
                np.mod(np.arange(len(oof_preds_discrete)), size) < train.shape[1]
            ]
            oof_preds_discrete = oof_preds_discrete.reshape(-1, train.shape[1], 1)

            test_pred_discrete = np.load(
                f"{model_path}/{preds_folder}/test_pred_fold_{fold}_discrete.npy"
            )
            test_pred_discrete = test_pred_discrete[
                np.mod(np.arange(len(test_pred_discrete)), size) < train.shape[1]
            ]
            test_preds_discrete.append(test_pred_discrete)

            train_1[oof_idx] = np.c_[train[oof_idx], oof_preds, oof_preds_discrete]
        else:
            train_1[oof_idx] = np.c_[train[oof_idx], oof_preds]

    test_preds = np.median(np.vstack(test_preds), axis=0).reshape(-1, train.shape[1], 1)
    test_1 = np.c_[test, test_preds]

    if with_discrete:
        test_preds_discrete = np.median(np.vstack(test_preds_discrete), axis=0).reshape(
            -1, train.shape[1], 1
        )
        test_1 = np.c_[test_1, test_preds_discrete]

    return train_1, test_1


def build_basic_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fallback: if external .npy predictions aren't available, build a minimal local feature tensor
    from provided columns. Keeps the same downstream stacking + LinearRegression logic.
    """

    def _make(df: pd.DataFrame):
        u_in = df["u_in"].to_numpy(dtype=np.float32).reshape(-1, 80, 1)
        u_out = df["u_out"].to_numpy(dtype=np.float32).reshape(-1, 80, 1)
        t = df["time_step"].to_numpy(dtype=np.float32).reshape(-1, 80, 1)

        R = df["R"].to_numpy(dtype=np.float32).reshape(-1, 80, 1)
        C = df["C"].to_numpy(dtype=np.float32).reshape(-1, 80, 1)

        u_in_cum = np.cumsum(u_in, axis=1)
        du_in = np.diff(u_in, axis=1, prepend=u_in[:, :1, :])

        X = np.concatenate([u_in, u_out, t, R, C, u_in_cum, du_in], axis=2)
        return X

    return _make(train_df), _make(test_df)




## === cell 9
def resolve_external_model_root(preferred: str):
    """
    Minimal improvement toward target:
    - Kaggle input datasets may be mounted under a different folder name than expected
      (e.g., creator slug, versioned name, etc.). We search /kaggle/input for any directory
      containing at least one of the expected submodel folders, so we actually use the
      strong external prediction artifacts instead of the weak fallback.
    """
    preferred = (preferred or "").rstrip("/")

    candidates = []
    if preferred:
        candidates.append(preferred)

    leaf = Path(preferred).name if preferred else ""
    if leaf:
        candidates.append(f"/kaggle/input/{leaf}")

    candidates.extend(
        [
            "/kaggle/input/new-model-same-old-mistakes",
            "/kaggle/input/new-model-same-old-mistakes/",
        ]
    )

    for c in candidates:
        if c and os.path.exists(c):
            return str(Path(c).resolve()).rstrip("/")

    base = Path("/kaggle/input")
    if base.exists():
        expected_any = {
            "i_think_i_might_have_built_a_model_v10",
            "i_think_i_might_have_built_a_model_v14",
            "i_think_i_might_have_built_a_model_v16",
            "i_think_i_might_have_built_a_model_v18_19",
            "i_think_i_might_have_built_a_model_v17_21_23_24",
        }
        for p in base.iterdir():
            if not p.is_dir():
                continue
            try:
                names = {x.name for x in p.iterdir() if x.is_dir()}
            except PermissionError:
                continue
            if len(names.intersection(expected_any)) > 0:
                return str(p.resolve()).rstrip("/")

    return preferred


model_path = resolve_external_model_root("../input/new-model-same-old-mistakes/")

m_list = [
    ("i_think_i_might_have_built_a_model_v10", 80, "preds"),
    ("i_think_i_might_have_built_a_model_v14", 80, "preds"),
    ("i_think_i_might_have_built_a_model_v18_19", 80, "preds"),
    ("i_think_i_might_have_built_a_model_v16", 80, "preds"),
    ("i_think_i_might_have_built_a_model_v16", 80, "preds_kaggle"),
    ("i_think_i_might_have_built_a_model_v16", 80, "preds_colab"),
    ("i_think_i_might_have_built_a_model_v17_21_23_24", 80, "preds"),
    ("i_think_i_might_have_built_a_model_colab_seed1", 80, "preds"),
    ("i_think_i_might_have_built_a_model_v20_27_seed2", 80, "preds"),
    ("i_think_i_might_have_built_a_model_v29_31_seed5", 80, "preds"),
]

train = np.empty(shape=(train_df.shape[0] // 80, 80, 0), dtype=np.float32)
test = np.empty(shape=(test_df.shape[0] // 80, 80, 0), dtype=np.float32)
with_discrete = True

used_external = True
used_any_submodel = False

missing_models = []
for m in m_list:
    submodel_root = f"{model_path}/{m[0]}"
    if not os.path.exists(submodel_root):
        missing_models.append(submodel_root)
        continue

    try:
        train, test = add_preds(
            submodel_root,
            train,
            test,
            FOLDS,
            with_discrete=with_discrete,
            size=m[1],
            preds_folder=m[2],
        )
        used_any_submodel = True
    except FileNotFoundError as e:
        missing_models.append(str(e))
        continue

if not used_any_submodel:
    print(
        "External prediction artifacts not found for any listed submodel. "
        "Falling back to basic local features."
    )
    used_external = False
    with_discrete = False
    train, test = build_basic_features(train_df, test_df)
else:
    if missing_models:
        print(f"Warning: skipped {len(missing_models)} missing submodels/files.")

print(
    "model_path:",
    model_path,
    "| train shape:",
    train.shape,
    "| test shape:",
    test.shape,
    "| used_external:",
    used_external,
)



## === cell 10
if used_external:
    tr = train.reshape(-1, train.shape[2])
    u_out_0_flat = u_out_0.ravel().astype(bool)
    u_out_0_flat = ~u_out_0_flat

    i = 0
    for m in m_list:
        if with_discrete:
            if (i * 2 + 1) >= tr.T.shape[0]:
                break
            pred = tr.T[i * 2]
            mae = mean_absolute_error(targets.ravel()[u_out_0_flat], pred[u_out_0_flat])
            pred_discrete = tr.T[i * 2 + 1]
            mae_discrete = mean_absolute_error(
                targets.ravel()[u_out_0_flat], pred_discrete[u_out_0_flat]
            )
            print(
                f"OOF {m[0]}, {m[1]}, {m[2]} || "
                f"MAE score (discrete, u_out==0): {mae_discrete:.6f} || "
                f"MAE score (non-discrete, u_out==0): {mae:.6f}"
            )
        else:
            if i >= tr.T.shape[0]:
                break
            pred = tr.T[i]
            mae = mean_absolute_error(targets.ravel()[u_out_0_flat], pred[u_out_0_flat])
            print(
                f"OOF {m[0]}, {m[1]}, {m[2]} || MAE score (non-discrete, u_out==0): {mae:.6f}"
            )
        i += 1
else:
    print("Skipped external-preds OOF diagnostics (fallback features in use).")



## === cell 11
pass



## === cell 12
diff = np.diff(pressure_values)
step = np.median(diff)
step



## === cell 13
EXTRAPOLATE_KNOTS = 100

left_pressure_extrapolate = np.arange(
    pressure_values[0] - EXTRAPOLATE_KNOTS * step, pressure_values[0] - step, step
)
right_pressure_extrapolate = np.arange(
    pressure_values[-1] + step, pressure_values[-1] + EXTRAPOLATE_KNOTS * step, step
)

pressure_values_extra = np.concatenate(
    [left_pressure_extrapolate, pressure_values, right_pressure_extrapolate]
)
pressure_values_extra_mid_points = (
    pressure_values_extra[1:] + pressure_values_extra[:-1]
) / 2

del diff, left_pressure_extrapolate, right_pressure_extrapolate, pressure_values
gc.collect()




## === cell 14
def discretize_np(y_discr, y_midpoints, y_cont):
    indices = np.searchsorted(y_midpoints, y_cont, side="left")
    return y_discr[indices]




## === cell 15
pass



## === cell 16
k_fold = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_preds = []
oof_preds_discrete = []
oof_targets = []

test_preds = []
test_preds_discrete = []

test_2d = test.reshape(-1, train.shape[2])

for fold, (train_idx, val_idx) in enumerate(k_fold.split(train)):
    print(f"FOLD={fold} started")

    X_train, X_val = train[train_idx], train[val_idx]
    y_train, y_val = targets[train_idx], targets[val_idx]

    u_out_0_val = u_out_0[val_idx]
    u_out_0_val_flat = ~u_out_0_val.ravel().astype(bool)

    y_val_flat = y_val.ravel()

    X_train_2d, y_train_1d = X_train.reshape(-1, train.shape[2]), y_train.ravel()
    X_val_2d, y_val_1d = X_val.reshape(-1, train.shape[2]), y_val.ravel()

    print(X_train_2d.shape, y_train_1d.shape)
    print(X_val_2d.shape, y_val_1d.shape)

    model = LinearRegression(n_jobs=None)
    model.fit(X_train_2d, y_train_1d)
    print(f"Sum of weights: {np.sum(model.coef_):.6f}")

    val_pred = model.predict(X_val_2d).ravel()
    val_pred_discrete = discretize_np(
        pressure_values_extra, pressure_values_extra_mid_points, val_pred
    )

    oof_preds.append(val_pred[u_out_0_val_flat])
    oof_preds_discrete.append(val_pred_discrete[u_out_0_val_flat])
    oof_targets.append(y_val_flat[u_out_0_val_flat])

    val_mae = mean_absolute_error(
        y_val_flat[u_out_0_val_flat], val_pred[u_out_0_val_flat]
    )
    val_mae_discrete = mean_absolute_error(
        y_val_flat[u_out_0_val_flat], val_pred_discrete[u_out_0_val_flat]
    )
    print(
        f"FOLD={fold} | MAE (discrete, u_out==0): {val_mae_discrete:.6f}, MAE (non-discrete, u_out==0): {val_mae:.6f}"
    )

    test_pred = model.predict(test_2d).ravel()
    test_preds.append(test_pred)

    test_pred_discrete = discretize_np(
        pressure_values_extra, pressure_values_extra_mid_points, test_pred
    )
    test_preds_discrete.append(test_pred_discrete)

    print(f"FOLD={fold} finished")



## === cell 17
oof_preds = np.concatenate(oof_preds)
oof_preds_discrete = np.concatenate(oof_preds_discrete)
oof_targets = np.concatenate(oof_targets)

oof_mae = mean_absolute_error(oof_targets, oof_preds)
oof_mae_discrete = mean_absolute_error(oof_targets, oof_preds_discrete)
print(
    f"OOF | MAE (discrete, u_out==0): {oof_mae_discrete:.6f}, MAE (non-discrete, u_out==0): {oof_mae:.6f}"
)



## === cell 18
sub["pressure"] = 0.0

sub.loc[test_indices, "pressure"] = np.mean(np.vstack(test_preds), axis=0)
sub[["id", "pressure"]].to_csv("submission_mean.csv", index=False)

sub.loc[test_indices, "pressure"] = np.median(np.vstack(test_preds), axis=0)
sub[["id", "pressure"]].to_csv("submission_median.csv", index=False)

sub[["id", "pressure"]].head()



## === cell 19
sub.loc[test_indices, "pressure"] = np.mean(np.vstack(test_preds_discrete), axis=0)
sub[["id", "pressure"]].to_csv("submission_mean_discrete.csv", index=False)

sub.loc[test_indices, "pressure"] = np.median(np.vstack(test_preds_discrete), axis=0)
sub[["id", "pressure"]].to_csv("submission_median_discrete.csv", index=False)

sub.loc[test_indices, "pressure"] = np.median(np.vstack(test_preds_discrete), axis=0)
sub[["id", "pressure"]].to_csv("submission.csv", index=False)

print("Wrote: submission.csv (and *_mean/_median variants)")
sub[["id", "pressure"]].head()

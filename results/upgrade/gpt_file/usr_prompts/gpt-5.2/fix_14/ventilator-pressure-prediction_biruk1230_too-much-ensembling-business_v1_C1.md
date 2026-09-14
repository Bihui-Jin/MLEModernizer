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

6.39753

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.39753) has done: 'The crash comes from missing external `.npy` prediction artifacts referenced under `../input/new-model-same-old-mistakes/`, so `train/test` never get any features and later reshapes/divisions fail. I keep the overall “stacking/linear-regression ensemble over per-timestep features + optional discretization” structure, but add a safe fallback that builds simple, local per-breath features from the provided CSVs when those artifacts are not present. I also fix the KFold split to operate on breaths (not mismatched `targets` shape) and avoid mutating `test` inside the fold loop so predictions aggregate correctly. Finally, the script always write a valid `submission_*.csv` with `id,pressure` columns.'
- What this solution (achieved 6.39788) has done: 'Your current score (6.39753, lower-is-better) is far from the target (0.14047), and the main reason is that the code is falling back to very weak “basic local features” instead of using the intended strong external OOF/test prediction artifacts. The smallest meaningful change toward the target is to actually load those external artifacts from the correct Kaggle input location (the current relative path likely resolves to nowhere), while keeping the same stacking + LinearRegression + discretization logic. I add a robust path resolver that tries the common Kaggle dataset mount points and only uses the fallback if nothing exists. This should move the score much closer to the target with minimal code changes and still produce valid `submission.csv`.'
- What this solution (achieved 6.39788) has done: 'Your current score is far worse than the target (lower-is-better), and the most likely reason is that you are still falling back to weak “basic local features” because the external prediction artifacts aren’t being found/loaded correctly. I make the smallest change that increases the chance we actually use the intended external OOF/test `.npy` artifacts: robustly auto-discover the correct Kaggle dataset root under `/kaggle/input/` and validate that each submodel folder exists before attempting to load it. If some listed submodels are missing, we skip only those (instead of triggering a full fallback), keeping the same stacking + LinearRegression + discretization logic. This should move the MAE substantially toward the target while keeping the overall approach identical and still writing a valid `submission.csv`.'
- What this solution (achieved 6.39764) has done: 'Your current score (6.39788, lower-is-better) is far from the target (0.14047), so we should focus on the most likely root cause: you are still not actually using the strong external OOF/test prediction artifacts, and the fallback features are too weak. I make the smallest changes that (1) reliably locate the external dataset under `/kaggle/input` by auto-discovering the correct folder name (even if it’s been “slugified” by Kaggle), and (2) fix a subtle but critical bug in `add_preds`: the loaded `oof_idx` indices are almost certainly “row indices” (0..N*80-1), but you are using them as “breath indices” (0..N-1), which silently scrambles/zeros most training features and ruins MAE. With those two minimal fixes, the stacking + LinearRegression + discretization logic stays the same, but it should move the score sharply toward your target (assuming the dataset is attached and the artifacts exist). The script still fall back safely and always write a valid `submission.csv`.'
- What this solution (achieved 6.39788) has done: 'Your score (6.39764 MAE, lower-is-better) is far from the target (0.14047), so the main goal is to ensure the stacking features are correctly aligned and actually populated from the external OOF/test artifacts; otherwise LinearRegression is trained on mostly-zeros/misaligned columns and performs terribly. I make the smallest fixes inside `add_preds` to (1) handle fold indices robustly (row-level vs breath-level), (2) correctly map OOF predictions into the right breaths even when the artifact is stored as a full-length vector (N_breaths*80) rather than already-shaped, and (3) always aggregate test predictions fold-wise without accidentally reshaping the wrong length. These changes keep the same model (LinearRegression), same training loop, same discretization, and same submission logic, but fix the likely silent misalignment that is dominating your error. The script still fall back safely if no external artifacts exist, and still write `submission.csv`.'
- What this solution (achieved 6.39788) has done: 'Your current MAE (6.39788, lower-is-better) is far from the target (0.14047), so the minimal change likely to move you toward the target is to fix feature corruption caused by leaving large parts of `train_1` as zeros when external OOF indices don’t cover all breaths. I keep the exact same stacking + LinearRegression + discretization flow, but in `add_preds` I always carry forward the existing features into `train_1/test_1` first, then overwrite only the fold OOF breaths with the augmented features; this prevents training on mostly-zero/missing columns. I also make the test-side aggregation robust when some folds/submodels are missing (avoid stacking empty lists), and I ensure indices are mapped consistently as breath-level without silently dropping most samples. These are small, directly score-relevant fixes that should substantially reduce MAE while preserving your intended approach and still writing a valid `submission.csv`.'
- What this solution (achieved 6.39753) has done: 'Your MAE (6.39788, lower-is-better) is still far from the target (0.14047), so the smallest score-relevant change is to stop feeding the stacker misaligned/mostly-empty external features. I keep the same external-preds stacking + LinearRegression + discretization flow, but fix `add_preds` so it correctly maps fold OOF predictions into the *selected* validation breaths (using the provided `val_{fold}.npy` indices), rather than assuming prediction arrays are aligned to global breath indices. I also make test-pred aggregation robust by always filling the correct feature slots for each submodel (so later submodels don’t overwrite earlier ones) and by using mean across folds in a deterministic way. These are minimal, directly score-related alignment fixes that should move MAE sharply toward your target while still producing the same submission format.'
- What this solution (achieved 6.39753) has done: 'Your MAE is far worse than the target (lower-is-better), and the most likely cause is a silent feature-corruption bug: `add_preds()` currently reinitializes `train_1/test_1` to zeros each submodel call and only fills OOF rows for that fold, leaving most breaths as zeros for that submodel’s feature columns. I make the smallest score-relevant fix by initializing the newly-added feature columns to NaN, then after loading all folds fill any remaining NaNs using a per-timestep mean of the available OOF predictions (so every breath gets a sensible value and we avoid training on mostly-zeros). I also make the fold index handling robust by detecting whether `val_{fold}.npy` is breath-level or row-level indices and mapping accordingly, without changing the model (LinearRegression), training loop, discretization, or submission format. This should move MAE substantially toward your target while keeping the core logic identical and still writing `submission.csv`.'
- What this solution (achieved 6.39753) has done: 'Your current MAE (6.39753, lower-is-better) is far from the target (0.14047), so we should focus on the most score-relevant, minimal fix: your CV split is currently done on the 3D `train` tensor (breath-level), but the competition metric is computed only on inspiratory timesteps (u_out==0) and your external OOF features are likely aligned by `breath_id`; a breath-wise split is correct, but we must also ensure we never train/validate on breaths whose external features are still “filled” by the global mean fallback (which can heavily degrade the LinearRegression fit). I add a tiny “feature validity mask” derived from whether a breath received any real OOF values for the first stacked column, and restrict KFold to only those breaths when `used_external=True` (keeping identical model and training loop). I also ensure test prediction aggregation uses `np.mean(test_preds_discrete, axis=0)` without unnecessary vstack/median duplication and that submission rows align 1:1 with `test_df` order. These changes preserve the exact stacking + LinearRegression + discretization semantics, but remove the biggest silent quality issue (training on “imputed/empty” external features), which should move MAE substantially toward your target when artifacts exist.'
- What this solution (achieved 6.39753) has done: 'Your current MAE (6.39753, lower-is-better) is far worse than the target (0.14047), so we should make a small change that is very likely to reduce error without changing the core approach (external-preds stacking → LinearRegression → optional discretization). The biggest silent score-killer here is that `add_preds()` can still mis-map OOF predictions to breaths when `val_{fold}.npy` contains row indices: it currently “guesses” breath indices via `// 80`, but then truncates by `k=min(len(val_idx), oof_preds.shape[0])`, which can easily pair the wrong breaths with the wrong predictions. I minimally fix alignment by converting row indices to a stable, ordered list of unique breath indices and then mapping those breath indices to the corresponding prediction rows via a deterministic order (so the i-th prediction row is placed into the i-th breath in that fold). I also add one tiny, score-relevant post-processing change: for the final `submission.csv` we output the *discrete-mean* ensemble (still using your existing discretization function/pressure grid), which typically matches the competition’s quantized pressure levels better than the raw continuous mean.'
- What this solution (achieved 6.39764) has done: 'Your current MAE (6.39753, lower-is-better) is far from the target (0.14047), so the minimal score-relevant goal is to stop corrupting the stacker features due to fold OOF misalignment. I make a small, contained fix inside `add_preds()` to correctly map fold validation indices to the exact breath IDs and row positions, by using `train_breath_ids` to build a breath-index mapping and by supporting both “fold stores breath indices” and “fold stores row indices” cases without truncation-based pairing. This keeps the same overall logic (external preds stacked as features → LinearRegression → discretize to pressure grid → mean/median ensembling) but should materially reduce MAE by ensuring the LinearRegression sees correctly-aligned OOF features. The rest of the pipeline (CV, discretization, submission writing) is left intact, and it still falls back safely if no external artifacts exist.'
- What this solution (achieved 6.39764) has done: 'Your current MAE (6.39764, lower-is-better) is far from the target (0.14047), and the biggest likely cause is still misalignment between the fold validation indices and the OOF prediction rows inside `add_preds()`, which silently scrambles the stacked features. I make the smallest fix that preserves your exact stacking→LinearRegression→discretization pipeline: for each fold, we map validation indices to *exact breath indices* and then place OOF predictions into those breaths, but with a deterministic “shape-aware” rule that handles the common artifact formats (OOF stored for only the fold’s validation breaths, or stored for all breaths with non-val rows missing/zero). I also make the “valid external feature” mask more robust (use NaN-based validity before `nan_to_num`) so CV isn’t accidentally trained on imputed zeros when external artifacts are partially missing. These changes are narrowly targeted to correct feature population/alignment (the main score driver here) and keep everything else (model, loss, folds, discretization, submission writing) the same.'
- What this solution (achieved 6.39753) has done: 'Your MAE is still extremely far from target, which strongly suggests the external OOF/test artifacts are either not being used or are being mapped to the wrong breaths, so the stacker is effectively learning from garbage. I make the smallest score-relevant change inside `add_preds()` to correctly align fold validation indices to the corresponding OOF prediction rows by using `train_breath_ids` + `breath_id` extracted from `train_df` (row-level), instead of guessing via `//80` and relying on truncation order. This preserves your exact pipeline (stack external preds as features → LinearRegression → discretize to pressure grid → mean/median ensemble), but fixes the most likely silent corruption: “OOF predictions for fold k placed into the wrong breaths”. I also make the “valid external feature mask” stricter by using `np.isnan`-based coverage before `nan_to_num`, so CV doesn’t train on mean-imputed folds when artifacts are partially missing.'

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

train_breath_id_row = train_df["breath_id"].to_numpy().astype(np.int64)




## === cell 8
def add_preds(
    model_path,
    train,
    test,
    n_folds,
    with_discrete=True,
    size=80,
    preds_folder="preds",
    *,
    train_breath_ids=None,
    train_breath_id_row=None,
):
    """
    Loads OOF/test predictions from an external folder and appends them as new features.

    Score-relevant minimal fix (alignment):
    - Previous versions can silently scramble mapping when val_{fold}.npy stores row indices.
      Using `val_idx//80` assumes row order equals breath order, which may not match after any
      preprocessing in the artifact pack; truncation-based pairing can then mis-assign OOF rows.
    - Fix: if we have per-row breath_id (`train_breath_id_row`), convert row indices -> breath_id
      -> breath index in the current `train` tensor. This produces a stable, correct val_breath_idx.
    - Then we support both artifact formats:
        (A) oof_preds has length == n_breaths_train (already global-aligned)
        (B) oof_preds has length == #val_breaths (fold-only); assign in the computed val order.
    """
    num_features_to_add = 2 if with_discrete else 1
    base_dim = train.shape[2]

    train_1 = np.empty(
        (train.shape[0], train.shape[1], base_dim + num_features_to_add),
        dtype=np.float32,
    )
    test_1 = np.empty(
        (test.shape[0], test.shape[1], base_dim + num_features_to_add),
        dtype=np.float32,
    )

    if base_dim > 0:
        train_1[:, :, :base_dim] = train
        test_1[:, :, :base_dim] = test

    train_1[:, :, base_dim:] = np.nan
    test_1[:, :, base_dim:] = np.nan

    def _load_and_reshape_pred(path: str, expected_steps: int):
        arr = np.load(path)
        arr = np.asarray(arr)

        if arr.ndim == 1:
            if arr.size % expected_steps != 0:
                raise ValueError(f"Unexpected 1D pred length in {path}: {arr.size}")
            arr = arr.reshape(-1, expected_steps, 1)
        elif arr.ndim == 2:
            if arr.shape[1] != expected_steps:
                if arr.shape[1] == 1 and arr.shape[0] % expected_steps == 0:
                    arr = arr.reshape(-1).reshape(-1, expected_steps, 1)
                else:
                    raise ValueError(f"Unexpected 2D pred shape in {path}: {arr.shape}")
            else:
                arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
        elif arr.ndim == 3:
            if arr.shape[1] != expected_steps:
                raise ValueError(f"Unexpected 3D pred shape in {path}: {arr.shape}")
            if arr.shape[2] != 1:
                arr = arr[:, :, :1]
        else:
            raise ValueError(f"Unexpected pred ndim in {path}: {arr.ndim}")

        return arr.astype(np.float32)

    def _ordered_unique(x: np.ndarray) -> np.ndarray:
        seen = set()
        out = []
        for v in x.tolist():
            if v not in seen:
                seen.add(v)
                out.append(v)
        return np.asarray(out, dtype=np.int64)

    n_breaths_train = train.shape[0]
    n_steps = train.shape[1]
    n_breaths_test = test.shape[0]

    breathid_to_idx = None
    if train_breath_ids is not None:
        train_breath_ids = np.asarray(train_breath_ids).ravel()
        if train_breath_ids.size == n_breaths_train:
            breathid_to_idx = {
                int(b): i for i, b in enumerate(train_breath_ids.tolist())
            }

    oof_acc = np.full((n_breaths_train, n_steps, 1), np.nan, dtype=np.float32)
    oof_acc_d = (
        np.full((n_breaths_train, n_steps, 1), np.nan, dtype=np.float32)
        if with_discrete
        else None
    )

    def _to_val_breath_idx(val_idx_raw: np.ndarray) -> np.ndarray:
        val_idx_raw = np.asarray(val_idx_raw).ravel().astype(np.int64)
        if val_idx_raw.size == 0:
            return val_idx_raw

        if val_idx_raw.max() >= n_breaths_train:
            if train_breath_id_row is not None and breathid_to_idx is not None:
                row_idx = val_idx_raw
                row_idx = row_idx[(row_idx >= 0) & (row_idx < train_breath_id_row.size)]
                breath_ids = train_breath_id_row[row_idx]
                mapped = [breathid_to_idx[int(b)] for b in _ordered_unique(breath_ids)]
                val_breath_idx = np.asarray(mapped, dtype=np.int64)
            else:
                val_breath_idx = _ordered_unique(
                    (val_idx_raw // n_steps).astype(np.int64)
                )
        else:
            if breathid_to_idx is not None and all(
                int(v) in breathid_to_idx for v in val_idx_raw.tolist()
            ):
                mapped = [breathid_to_idx[int(v)] for v in val_idx_raw.tolist()]
                val_breath_idx = _ordered_unique(np.asarray(mapped, dtype=np.int64))
            else:
                val_breath_idx = _ordered_unique(val_idx_raw.astype(np.int64))

        val_breath_idx = val_breath_idx[
            (val_breath_idx >= 0) & (val_breath_idx < n_breaths_train)
        ]
        return val_breath_idx

    test_preds = []
    test_preds_discrete = []

    for fold in range(n_folds):
        idx_path = f"{model_path}/indices/val_{fold}.npy"
        if not os.path.exists(idx_path):
            raise FileNotFoundError(idx_path)

        val_idx_raw = np.load(idx_path)
        val_breath_idx = _to_val_breath_idx(val_idx_raw)
        if val_breath_idx.size == 0:
            continue

        oof_preds = _load_and_reshape_pred(
            f"{model_path}/{preds_folder}/val_pred_fold_{fold}.npy",
            expected_steps=n_steps,
        )
        test_pred = _load_and_reshape_pred(
            f"{model_path}/{preds_folder}/test_pred_fold_{fold}.npy",
            expected_steps=n_steps,
        )

        if oof_preds.shape[0] == n_breaths_train:
            train_1[:, :, base_dim : base_dim + 1] = oof_preds
            oof_acc[:, :, :] = oof_preds
        else:
            k = min(val_breath_idx.size, oof_preds.shape[0])
            if k > 0:
                bidx = val_breath_idx[:k]
                train_1[bidx, :, base_dim : base_dim + 1] = oof_preds[:k]
                oof_acc[bidx] = oof_preds[:k]

        test_preds.append(test_pred[: min(n_breaths_test, test_pred.shape[0])])

        if with_discrete:
            oof_preds_discrete = _load_and_reshape_pred(
                f"{model_path}/{preds_folder}/val_pred_fold_{fold}_discrete.npy",
                expected_steps=n_steps,
            )
            test_pred_discrete = _load_and_reshape_pred(
                f"{model_path}/{preds_folder}/test_pred_fold_{fold}_discrete.npy",
                expected_steps=n_steps,
            )

            if oof_preds_discrete.shape[0] == n_breaths_train:
                train_1[:, :, base_dim + 1 : base_dim + 2] = oof_preds_discrete
                oof_acc_d[:, :, :] = oof_preds_discrete
            else:
                kd = min(val_breath_idx.size, oof_preds_discrete.shape[0])
                if kd > 0:
                    bidxd = val_breath_idx[:kd]
                    train_1[bidxd, :, base_dim + 1 : base_dim + 2] = oof_preds_discrete[
                        :kd
                    ]
                    oof_acc_d[bidxd] = oof_preds_discrete[:kd]

            test_preds_discrete.append(
                test_pred_discrete[: min(n_breaths_test, test_pred_discrete.shape[0])]
            )

    if len(test_preds) == 0:
        raise FileNotFoundError(
            f"No usable test predictions found under: {model_path}/{preds_folder}"
        )

    oof_mean = np.nanmean(oof_acc, axis=0, keepdims=True)  # (1, steps, 1)
    if np.isnan(oof_mean).any():
        oof_mean = np.nan_to_num(oof_mean, nan=0.0).astype(np.float32)
    miss = np.isnan(train_1[:, :, base_dim : base_dim + 1])
    if miss.any():
        train_1[:, :, base_dim : base_dim + 1][miss] = np.broadcast_to(
            oof_mean, (n_breaths_train, n_steps, 1)
        )[miss]

    if with_discrete:
        oof_mean_d = np.nanmean(oof_acc_d, axis=0, keepdims=True)
        if np.isnan(oof_mean_d).any():
            oof_mean_d = np.nan_to_num(oof_mean_d, nan=0.0).astype(np.float32)
        miss_d = np.isnan(train_1[:, :, base_dim + 1 : base_dim + 2])
        if miss_d.any():
            train_1[:, :, base_dim + 1 : base_dim + 2][miss_d] = np.broadcast_to(
                oof_mean_d, (n_breaths_train, n_steps, 1)
            )[miss_d]

    common_test_breaths = min(x.shape[0] for x in test_preds)
    test_preds_mean = np.mean(
        np.stack([x[:common_test_breaths] for x in test_preds], axis=0), axis=0
    )
    test_1[:common_test_breaths, :, base_dim : base_dim + 1] = test_preds_mean
    if common_test_breaths < n_breaths_test:
        fill_val = test_preds_mean[-1:, :, :]
        test_1[common_test_breaths:, :, base_dim : base_dim + 1] = np.broadcast_to(
            fill_val, (n_breaths_test - common_test_breaths, n_steps, 1)
        )

    if with_discrete:
        if len(test_preds_discrete) == 0:
            raise FileNotFoundError(
                f"No usable discrete test predictions found under: {model_path}/{preds_folder}"
            )
        common_test_breaths_d = min(x.shape[0] for x in test_preds_discrete)
        common = min(common_test_breaths, common_test_breaths_d)
        test_preds_discrete_mean = np.mean(
            np.stack([x[:common] for x in test_preds_discrete], axis=0), axis=0
        )
        test_1[:common, :, base_dim + 1 : base_dim + 2] = test_preds_discrete_mean
        if common < n_breaths_test:
            fill_val_d = test_preds_discrete_mean[-1:, :, :]
            test_1[common:, :, base_dim + 1 : base_dim + 2] = np.broadcast_to(
                fill_val_d, (n_breaths_test - common, n_steps, 1)
            )

    train_1 = np.nan_to_num(train_1, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    test_1 = np.nan_to_num(test_1, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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
            train_breath_ids=train_breath_ids,
            train_breath_id_row=train_breath_id_row,
        )
        used_any_submodel = True
    except FileNotFoundError as e:
        missing_models.append(str(e))
        continue
    except Exception as e:
        missing_models.append(f"{submodel_root} :: {repr(e)}")
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
if used_external:
    first_col = train[:, :, 0]
    valid_breath_mask = np.mean(np.abs(first_col), axis=1) > 1e-6
    valid_breath_idx = np.where(valid_breath_mask)[0]
    if valid_breath_idx.size < train.shape[0]:
        print(
            f"Restricting CV to breaths with non-trivial external features: "
            f"{valid_breath_idx.size}/{train.shape[0]}"
        )
else:
    valid_breath_idx = np.arange(train.shape[0], dtype=np.int64)

k_fold = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_preds = []
oof_preds_discrete = []
oof_targets = []

test_preds = []
test_preds_discrete = []

test_2d = test.reshape(-1, train.shape[2])

for fold, (tr_i, va_i) in enumerate(k_fold.split(valid_breath_idx)):
    print(f"FOLD={fold} started")

    train_idx = valid_breath_idx[tr_i]
    val_idx = valid_breath_idx[va_i]

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
sub["pressure"] = np.mean(np.asarray(test_preds), axis=0)
sub[["id", "pressure"]].to_csv("submission_mean.csv", index=False)

sub["pressure"] = np.median(np.asarray(test_preds), axis=0)
sub[["id", "pressure"]].to_csv("submission_median.csv", index=False)

sub[["id", "pressure"]].head()



## === cell 19
sub["pressure"] = np.mean(np.asarray(test_preds_discrete), axis=0)
sub[["id", "pressure"]].to_csv("submission_mean_discrete.csv", index=False)

sub["pressure"] = np.median(np.asarray(test_preds_discrete), axis=0)
sub[["id", "pressure"]].to_csv("submission_median_discrete.csv", index=False)

sub["pressure"] = np.mean(np.asarray(test_preds_discrete), axis=0)
sub[["id", "pressure"]].to_csv("submission.csv", index=False)

print("Wrote: submission.csv (and *_mean/_median variants)")
sub[["id", "pressure"]].head()

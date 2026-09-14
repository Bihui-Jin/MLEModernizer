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

0.1441035344899513

# 6. Current score

7.82194

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.31337) has done: 'I make the script robust to missing external Kaggle “input datasets” (the `gb-rwbt-files` and `vent-011-median-wins` folders), which currently cause the pipeline to crash before writing any submission. Concretely, I add safe fallbacks that (1) if no blend files exist, generate a baseline prediction directly from `test.csv` (using mean `pressure` per `(R,C,time_step,u_in,u_out)` with a global fallback), and (2) if the optional `vent-011-median-wins` submission file isn’t available, skip that blend step and just use the generated predictions. The existing nearest-pressure snapping logic is preserved to keep evaluation semantics aligned with the competition’s discrete pressure levels. Finally, the code always write a valid `submission.csv` with columns `id,pressure`.'
- What this solution (achieved 4.31827) has done: 'Your current score (8.31337, lower-is-better) is far from the target (0.1441), so we need a real accuracy lift while keeping the same overall “generate predictions then snap to nearest discrete pressure” logic. The minimal high-impact change here is to replace the overly-specific `(R,C,time_step,u_in,u_out)` mean lookup (which rarely matches exactly due to continuous `u_in`) with a per-breath, time-series-aware baseline: compute cumulative inspired volume and estimate pressure via a simple physical relation using `(R,C)` plus a tiny fitted offset from train. This stays within your existing pipeline structure (still produces a single prediction file, still optional blending, still snapping), but should dramatically reduce MAE. I also make path handling robust to your provided `/kaggle/input/...` structure without changing the intended I/O outputs (`submission.csv` in the working directory).'
- What this solution (achieved 4.33588) has done: 'Your current MAE (4.318) is far from the target (0.144), so we need a real accuracy lift without changing the overall pipeline (generate predictions → optional blend → snap to discrete pressures → write submission.csv). The biggest issue is that the “physics” baseline is mis-specified (it uses `u_in * R`, which is not a flow/pressure-drop relation) and it doesn’t enforce the competition’s “don’t-care” expiratory phase behavior (u_out==1) which can hurt public LB even though it’s not scored. I keep the same lightweight per-row linear least-squares fit, but change it to a more appropriate linear surrogate: `pressure ≈ a + b*(V/C) + c*(flow*R) + d*(u_out==0)`, where `flow` is approximated by `u_in` during inspiration, and I explicitly set predictions during expiration to 0 before snapping. I also fit on more breaths (still bounded for speed) to reduce coefficient noise while staying well within the 600s budget and preserving the same end-to-end semantics and file outputs.'
- What this solution (achieved 4.22921) has done: 'Your current MAE (4.33588, lower-is-better) is still far above the target (0.1441), so we need a meaningful accuracy lift while keeping your same overall pipeline (physics baseline → optional blend → snap to discrete pressures → write `submission.csv`). The smallest high-impact fix is to make the “physics” surrogate consistent with a standard linear RC model: approximate flow as the change in inspired volume per unit time, not `u_in` directly, and add a small `u_in` term to absorb valve nonlinearity while still using the same single least-squares fit. This keeps the same training approach (one `np.linalg.lstsq`), uses the same features you already compute (dt, cumulative volume, R, C), and preserves the same expiratory handling (`u_out==1` → 0) and snapping. These changes should substantially reduce the MAE without changing the overall structure or I/O.'
- What this solution (achieved 3.41594) has done: 'Your current MAE (4.229, lower-is-better) is still far above the target (0.144), so we need a real accuracy lift while keeping the same “single least-squares physics baseline → optional blend → snap to discrete pressures → write submission.csv” structure. The smallest high-impact fix is to make the surrogate model align better with the competition’s RC dynamics by fitting/predicting using per-breath lagged pressure and lagged flow/volume terms (autoregressive features), which is still just one `np.linalg.lstsq` and keeps the same overall approach. This also removes the big mismatch between training (true pressure depends strongly on previous pressure state) and your current stateless linear model. We keep the expiration handling (`u_out==1` → 0.0) and the same nearest-pressure snapping and output format.'
- What this solution (achieved 3.17467) has done: 'Your current MAE (3.41594, lower-is-better) is still far above the target (0.1441), so we need a meaningful accuracy lift while keeping your same overall “single least-squares physics baseline → optional blend → snap to discrete pressures → write CSV” structure. The biggest remaining issue is that the autoregressive term is trained on true previous pressure but, at test-time, you feed back the model’s own previous prediction, which can drift; we can reduce this by training in the same “closed-loop” way the model is used at inference (teacher-forcing mismatch fix) without changing the model form. Concretely, we keep the exact same features and linear least-squares approach, but build the design matrix using per-breath simulated previous predictions (with the same recursion you use in `_predict_physics`) and then solve once with `np.linalg.lstsq`. This is a minimal change that directly targets the dominant error source (state drift) and should move the MAE downward toward your target while preserving your snapping and submission semantics.'
- What this solution (achieved 3.17467) has done: 'We keep your same pipeline (fit one linear least-squares “physics+AR” model → predict sequentially per breath → snap to nearest discrete pressure → write `submission.csv`), but fix a key mismatch in your closed-loop training. Right now the recursion during training updates `y_prev` using a non-AR warm-start (`pred_i0 = base0`), which does not match inference where `y_prev` is the full recursive prediction; we update training to use the same recursive formula when building the design matrix so coefficients are fit under the same dynamics. This is a minimal code change confined to `_fit_physics_linear_coeffs` and should reduce drift error, moving MAE down toward your target. All I/O paths and the final submission format remain unchanged.'
- What this solution (achieved 3.96859) has done: 'We keep your exact pipeline (single linear least-squares “physics+AR” model → sequential per-breath prediction → snap to discrete pressures → optional blend → write `submission.csv`) but fix the main remaining accuracy bug: your “closed-loop training” still uses a base-model warm-start (`pred_i0`) to update `y_prev`, which does not match inference recursion and causes drift. The minimal change is to update training so `y_prev` is advanced with the same full recursive formula used at inference, using the current coefficients and previous simulated state. This keeps the model form and loss identical (still one `np.linalg.lstsq`), but better aligns train/test dynamics, which should reduce MAE toward your target. All paths and the submission-writing behavior remain unchanged.'
- What this solution (achieved 3.92674) has done: 'We keep your exact “fit one linear least-squares physics+AR model → sequential per-breath prediction → snap to discrete pressures → optional blend → write CSV” pipeline, but fix the main accuracy regression: closed-loop training still advances `y_prev` using a different recursion than inference. Specifically, while building the training design matrix we iteratively re-predict the current step with the latest coefficients and use that as the next `y_prev`, matching inference dynamics. This is a minimal, localized change inside `_fit_physics_linear_coeffs` (no new models, no new features, no new training loops beyond the existing iterative refinement) and should move MAE down toward your target. All paths and the final `submission.csv` schema remain unchanged.'
- What this solution (achieved 3.92674) has done: 'We keep your exact pipeline (single least-squares “physics+AR” model → sequential per-breath prediction → snap to discrete pressures → optional blend → write `submission.csv`) but fix two small issues that strongly affect MAE. First, we stop forcing expiration (`u_out==1`) predictions to 0.0, because Kaggle’s metric *ignores* expiration rather than expecting zeros; hard-zeroing can still hurt because the scoring mask is based on the true `u_out` and any mismatch can leak into scored steps around transitions. Second, we prevent autoregressive state drift by resetting the AR state when inspiration starts (i.e., when `u_out` switches 1→0), which keeps the recursion stable without changing model form or features. These are localized changes only inside `_predict_physics` and the training design-matrix construction, and they should move your MAE downward toward the target while preserving snapping and output format.'
- What this solution (achieved 3.92674) has done: 'We keep your exact pipeline (fit one linear least-squares physics+AR model → sequential per-breath prediction → snap to discrete pressures → optional blend → write `submission.csv`) but fix two issues that typically inflate MAE a lot in this competition. First, we train and predict only on the inspiratory phase (`u_out==0`) since the metric ignores expiration; this removes a large amount of unscored (and model-mismatched) data from the fit and prevents expiration dynamics from contaminating coefficients. Second, we set test predictions during expiration to the last inspiratory value within the breath (instead of continuing the recursion), which is safe because expiration is not scored and stabilizes transitions so the first inspiratory points don’t get dragged by an arbitrary expiration recursion. These are localized changes inside `_fit_physics_linear_coeffs` and `_predict_physics` only, preserving model form, features, snapping, and output format.'
- What this solution (achieved 7.79334) has done: 'We keep your exact “single least-squares physics+AR model → sequential per-breath prediction → snap to discrete pressures → optional blend → write submission.csv” pipeline, but fix the biggest scoring issue: your predictions are being aligned to `sample_submission.csv` order instead of `test.csv` order, which can silently permute `id`↔`pressure` and explode MAE. Concretely, we always build the submission from `test.csv[['id']]` in the same sorted order used for prediction, then merge predictions back to original row order by index and finally write `id,pressure`. This is a minimal change localized to the fallback path in `g()` (and keeps blending logic unchanged) and should move MAE strongly downward toward your target without changing the model form. We also add a small safety assertion on row counts to prevent future misalignment regressions.'
- What this solution (achieved 7.82194) has done: 'Your current MAE is far worse than the target, so we need a clear accuracy lift while keeping your same “single least-squares physics+AR model → sequential per-breath prediction → snap to discrete pressures → write submission.csv” structure. The highest-impact minimal change is to use the true time step size (fixed ~0.03s) instead of `dt = diff(time_step)` (which makes `Q≈u_in` and breaks the intended RC-feature meaning), and to compute `dV` using that fixed `dt` so `V` and `Q` are physically consistent. We also clip predictions to the known pressure range before snapping (still the same snapping semantics, just avoids extreme outliers that increase MAE). All file paths, blending behavior, and output schema remain unchanged.'
- What this solution (achieved 7.82194) has done: 'Your score is far above the target (lower-is-better), and the biggest likely culprit left is that predictions are being generated in `test.csv` row order while `id` is being taken from an unsorted `df_test`, which can silently misalign `id`↔`pressure` and massively inflate MAE. I make the fallback path build the submission strictly from the same sorted frame used for prediction, then restore original `test.csv` row order before writing, guaranteeing correct alignment. I apply the same alignment safeguard to the blend path (which currently assumes `sample_submission.csv` order matches the prediction vectors). These are minimal, localized changes that preserve your model, snapping, and overall pipeline, but should move the MAE sharply down toward the target by fixing ordering.'
- What this solution (achieved 7.82194) has done: 'The current MAE (7.82) is far from the target (0.144, lower-is-better), so the most likely remaining issue is still catastrophic `id`↔`pressure` misalignment in the blend path: you assume the blended vector is already in the exact `test.csv` row order, but external files are often in `sample_submission`/sorted-by-`id` order. I keep your core pipeline (same physics+AR model, same blending strategy, same snapping), but add a minimal, deterministic alignment step: always build the final prediction series keyed by `id` and reindex it to `test.csv['id']` before writing. I apply the same safeguard to the external `vent-011-median-wins` blend so both sources are aligned on `id` before taking the median. This should drastically reduce MAE if ordering was the cause, while keeping semantics unchanged when ordering was already correct.'

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
def _resolve_path(rel_path: str) -> str:
    candidates = [
        rel_path,
        rel_path.replace("../input/", "/kaggle/input/"),
        rel_path.replace("/kaggle/input/", "../input/"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_path  # let downstream error if truly missing


train_path = _resolve_path("../input/ventilator-pressure-prediction/train.csv")
df_train = pd.read_csv(train_path)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)

P_MIN = float(sorted_pressures[0])
P_MAX = float(sorted_pressures[-1])

DT_FIXED = 0.03


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
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def _fit_physics_linear_coeffs(
    df_tr: pd.DataFrame,
    max_breaths: int = 12000,
    seed: int = 2021,
    refine_iters: int = 3,
):
    rng = np.random.RandomState(seed)
    breath_ids = df_tr["breath_id"].unique()
    if len(breath_ids) > max_breaths:
        breath_ids = rng.choice(breath_ids, size=max_breaths, replace=False)
    d = df_tr[df_tr["breath_id"].isin(breath_ids)].copy()
    d.sort_values(["breath_id", "time_step"], inplace=True)

    dt = np.full(len(d), DT_FIXED, dtype=float)

    u_in = d["u_in"].to_numpy(dtype=float)
    u_out = d["u_out"].to_numpy(dtype=int)
    insp = u_out == 0

    dV = (u_in * insp.astype(float)) * dt
    V = pd.Series(dV, index=d.index).groupby(d["breath_id"]).cumsum().to_numpy()

    Q = dV / dt

    C = d["C"].to_numpy(dtype=float)
    R = d["R"].to_numpy(dtype=float)

    x1 = V / C
    x2 = R * Q
    x3 = u_in * insp
    x4 = insp.astype(float)

    y = d["pressure"].to_numpy(dtype=float)

    breath_ids_arr = d["breath_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(breath_ids_arr[1:] != breath_ids_arr[:-1]) + 1]
    ends = np.r_[starts[1:], len(d)]

    m0 = insp
    X0 = np.vstack(
        [
            np.ones_like(x1[m0]),
            x1[m0],
            x2[m0],
            x3[m0],
            x4[m0],
        ]
    ).T
    y0 = y[m0]
    coef0, _, _, _ = np.linalg.lstsq(X0, y0, rcond=None)
    a0, b0, c0, d0, e0 = coef0

    coef = np.array([a0, b0, c0, d0, e0, 1.0, 0.0, 0.0], dtype=float)

    for _ in range(int(refine_iters)):
        X_rows = []
        y_rows = []
        a, b, c, dd, ee, pp, bb1, cc2 = coef

        for s, e in zip(starts, ends):
            y_prev = 0.0
            x1_prev = 0.0
            x2_prev = 0.0
            prev_insp = False
            for i in range(s, e):
                if insp[i] and (not prev_insp):
                    y_prev = 0.0
                    x1_prev = 0.0
                    x2_prev = 0.0

                if not insp[i]:
                    prev_insp = False
                    continue

                X_rows.append(
                    [
                        1.0,
                        x1[i],
                        x2[i],
                        x3[i],
                        x4[i],
                        y_prev,
                        x1_prev,
                        x2_prev,
                    ]
                )
                y_rows.append(y[i])

                base = a + b * x1[i] + c * x2[i] + dd * x3[i] + ee * x4[i]
                pred_i = base + pp * y_prev + bb1 * x1_prev + cc2 * x2_prev

                y_prev = pred_i
                x1_prev = x1[i]
                x2_prev = x2[i]
                prev_insp = True

        X = np.asarray(X_rows, dtype=float)
        y_m = np.asarray(y_rows, dtype=float)
        coef_new, _, _, _ = np.linalg.lstsq(X, y_m, rcond=None)
        coef = coef_new

    return coef  # [a,b,c,d,e, p_prev, x1_prev, x2_prev]


PHYS_COEF = _fit_physics_linear_coeffs(
    df_train, max_breaths=12000, seed=2021, refine_iters=3
)


def _predict_physics(df_te: pd.DataFrame, coef):
    d = df_te.copy()
    d.sort_values(["breath_id", "time_step"], inplace=True)

    dt = np.full(len(d), DT_FIXED, dtype=float)

    u_in = d["u_in"].to_numpy(dtype=float)
    u_out = d["u_out"].to_numpy(dtype=int)
    insp = u_out == 0

    dV = (u_in * insp.astype(float)) * dt
    V = pd.Series(dV, index=d.index).groupby(d["breath_id"]).cumsum().to_numpy()
    Q = dV / dt

    C = d["C"].to_numpy(dtype=float)
    R = d["R"].to_numpy(dtype=float)

    x1 = V / C
    x2 = R * Q
    x3 = u_in * insp
    x4 = insp.astype(float)

    a, b, c, dd, ee, pp, bb1, cc2 = coef

    pred = np.zeros(len(d), dtype=float)

    breath_ids = d["breath_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(breath_ids[1:] != breath_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(d)]

    for s, e in zip(starts, ends):
        y_prev = 0.0
        x1_prev = 0.0
        x2_prev = 0.0
        prev_insp = False
        last_insp_pred = 0.0
        for i in range(s, e):
            if insp[i] and (not prev_insp):
                y_prev = 0.0
                x1_prev = 0.0
                x2_prev = 0.0

            if not insp[i]:
                pred[i] = last_insp_pred
                prev_insp = False
                continue

            base = a + b * x1[i] + c * x2[i] + dd * x3[i] + ee * x4[i]
            pred_i = base + pp * y_prev + bb1 * x1_prev + cc2 * x2_prev
            pred[i] = pred_i

            y_prev = pred_i
            x1_prev = x1[i]
            x2_prev = x2[i]
            prev_insp = True
            last_insp_pred = pred_i

    pred_series = pd.Series(pred, index=d.index).sort_index()
    return pred_series.to_numpy()


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    if len(l) == 0:
        test_path = _resolve_path("../input/ventilator-pressure-prediction/test.csv")
        df_test = pd.read_csv(test_path)

        d_sorted = df_test.sort_values(["breath_id", "time_step"]).copy()
        pred_sorted = _predict_physics(d_sorted, PHYS_COEF)

        pred_by_index = pd.Series(pred_sorted, index=d_sorted.index).sort_index()
        assert len(pred_by_index) == len(df_test)

        output = df_test[["id"]].copy()
        output["pressure"] = pred_by_index.loc[df_test.index].to_numpy()

        output["pressure"] = output["pressure"].clip(P_MIN, P_MAX)
        output["pressure"] = output["pressure"].apply(find_nearest)

        output.to_csv("rwb 0 loops.csv", index=False)
        return

    file_count = len(l)
    loop_time = max(1, 500 // file_count)

    splits = file_count // 2
    l.sort()
    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * round(len(l) / splits) :])
        else:
            flist.append(
                l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
            )
    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    test_path = _resolve_path("../input/ventilator-pressure-prediction/test.csv")
    df_test = pd.read_csv(test_path)

    blended_pred = np.median(np.vstack(pred_list), axis=0)

    if len(blended_pred) != len(df_test):
        raise ValueError(
            f"Blended prediction length {len(blended_pred)} != test rows {len(df_test)}; cannot align safely."
        )

    ids_test = df_test["id"].to_numpy()
    order_by_id = np.argsort(ids_test)
    inv_order = np.empty_like(order_by_id)
    inv_order[order_by_id] = np.arange(len(order_by_id))
    blended_by_id_sorted = blended_pred[order_by_id]
    blended_aligned = blended_by_id_sorted[inv_order]

    output = df_test[["id"]].copy()
    output["pressure"] = blended_aligned
    output["pressure"] = output["pressure"].clip(P_MIN, P_MAX)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 2
g(_resolve_path("../input/gb-rwbt-files"))



## === cell 3
rwb_candidates = sorted(glob.glob("rwb * loops.csv"), key=lambda x: os.path.getmtime(x))
if len(rwb_candidates) == 0:
    raise FileNotFoundError(
        "No rwb prediction file was generated; expected g() to create one."
    )

df_2 = pd.read_csv(rwb_candidates[-1])

test_path = _resolve_path("../input/ventilator-pressure-prediction/test.csv")
df_test = pd.read_csv(test_path)

df_2 = df_2[["id", "pressure"]].copy()
df_2 = df_2.drop_duplicates(subset=["id"], keep="last")
df_2 = df_2.set_index("id").reindex(df_test["id"]).reset_index()
if df_2["pressure"].isna().any():
    raise ValueError(
        "rwb predictions missing some test ids after alignment; cannot submit safely."
    )

path_df1 = _resolve_path("../input/vent-011-median-wins/submission.csv")
if os.path.exists(path_df1):
    df_1 = pd.read_csv(path_df1)[["id", "pressure"]].copy()
    df_1 = df_1.drop_duplicates(subset=["id"], keep="last")
    df_1 = df_1.set_index("id").reindex(df_test["id"]).reset_index()
    if df_1["pressure"].isna().any():
        df_final = df_2.copy()
    else:
        df_final = df_test[["id"]].copy()
        df_final["pressure"] = np.median(
            np.vstack([df_1["pressure"].to_numpy(), df_2["pressure"].to_numpy()]),
            axis=0,
        )
else:
    df_final = df_2.copy()

df_final = df_final[["id", "pressure"]]
df_final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_final.shape)
print(df_final.head())

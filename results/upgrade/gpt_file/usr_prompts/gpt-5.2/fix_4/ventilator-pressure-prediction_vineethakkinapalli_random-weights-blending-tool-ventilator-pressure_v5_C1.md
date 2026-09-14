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

0.1475814798797685

# 6. Current score

0.9277

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The crash happens because the script expects a second “allowed” file to exist in `../input/ventilator-pressure-high-score-submissions`, but that dataset path is not available here, so `l` has length 0 or 1 and `l[1]` raises `IndexError`. I keep your blending/median logic intact, but make it robust to missing/insufficient external submissions and automatically fall back to a simple, legitimate baseline prediction derived from the training pressure distribution (no leakage). I also ensure the output file is always created with the required `id,pressure` columns and a `.csv` suffix under the working directory. These changes are primarily to fix execution and produce a valid submission; the fallback baseline should also be better than an all-zero submission.'
- What this solution (achieved 3.83592) has done: 'Your current score (10.86378 MAE) is far from the target (0.1476), and the main reason is that you’re not actually blending any strong submissions in this environment—so you fall back to a constant median baseline, which performs very poorly. To move the score sharply toward the target without changing your blending logic, I replace the fallback baseline with a simple, legitimate, competition-standard “nearest neighbor on (R,C,u_in,u_out,time_step)” regressor using scikit-learn, and I keep the same snapping-to-valid-pressure post-processing. This preserves the overall structure (blend if available, else fallback) while making the fallback far stronger and still fully offline/no leakage. I also fix the data path to the dataset that actually exists here (your files are under `/kaggle/input/...`), ensuring it runs end-to-end and always writes `submission.csv`.'
- What this solution (achieved 0.9277) has done: 'Your current MAE (3.83592) is still far above the target (0.1476), and the main limiting factor is the weak fallback model: row-wise KNN across all breaths ignores the strong within-breath temporal structure of the problem. To move substantially toward the target while keeping the “fallback if no external submissions” core flow intact, I replace the fallback with a breath-wise KNN that uses the full 80-step sequence of controls plus (R,C) to find similar breaths, then predicts the entire pressure sequence by averaging neighbor pressure sequences. I keep your pressure “snapping” post-processing (nearest valid pressure) and ensure predictions are only required/used for inspiratory phase (u_out==0), setting expiratory predictions to 0 which is unscored. This change is still a simple scikit-learn KNN baseline (same general training approach) but is far more aligned with the competition metric and should reduce MAE toward your target band.'

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

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler



## === cell 1
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction: float) -> float:
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx >= total_pressures_len:
        return float(sorted_pressures[-1])
    if insert_idx <= 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Blend (up to) two submissions in input_list that match the expected filename score format.
    Bugfix: handle cases where 0 or 1 allowed files exist to avoid IndexError.
    """
    preds = []
    l = []
    allow = [1348, 1991]

    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        if public_lb_score in allow:
            l.append(public_lb_score)
            preds.append(pd.read_csv(input_list[i]).pressure.to_numpy().ravel())

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    l_sum = l[0] + l[1]
    if l_sum == 0:
        weight1 = 0.5
    else:
        weight1 = (l[1] / l_sum) + 0.1
    weight1 = float(np.clip(weight1, 0.0, 1.0))
    weight2 = 1.0 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def _make_breath_matrix(df: pd.DataFrame, feat_cols, target_col=None):
    """
    Create breath-level matrices:
      X: (n_breaths, len(feat_cols)*80 + 2)  with appended [R, C] at the end
      y: (n_breaths, 80) if target_col provided, else None
      breath_ids: (n_breaths,)
      row_ids: (n_breaths, 80) test row ids for reconstruction
      u_out: (n_breaths, 80) used to zero out expiratory predictions (unscored)
    Assumes exactly 80 rows per breath (true for this competition).
    """
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    breath_ids = df["breath_id"].to_numpy()
    _, idx_start = np.unique(breath_ids, return_index=True)
    idx_start.sort()
    n_breaths = len(idx_start)

    feat_arr = df[feat_cols].to_numpy(dtype=np.float32)
    try:
        feat_arr = feat_arr.reshape(n_breaths, 80, len(feat_cols))
    except Exception as e:
        raise RuntimeError(
            "Unexpected breath length; expected exactly 80 rows per breath. "
            f"Reshape failed: {e}"
        )

    R = df["R"].to_numpy(dtype=np.float32).reshape(n_breaths, 80)[:, 0:1]
    C = df["C"].to_numpy(dtype=np.float32).reshape(n_breaths, 80)[:, 0:1]

    X_seq = feat_arr.reshape(n_breaths, 80 * len(feat_cols))
    X = np.concatenate([X_seq, R, C], axis=1)

    row_ids = df["id"].to_numpy(dtype=np.int64).reshape(n_breaths, 80)
    u_out = df["u_out"].to_numpy(dtype=np.int8).reshape(n_breaths, 80)

    y = None
    if target_col is not None:
        y = df[target_col].to_numpy(dtype=np.float32).reshape(n_breaths, 80)

    breath_ids_unique = (
        df["breath_id"].to_numpy(dtype=np.int64).reshape(n_breaths, 80)[:, 0]
    )
    return X, y, breath_ids_unique, row_ids, u_out


def _baseline_predictions():
    """
    Score-improving fallback (still a simple KNN, but aligned with the evaluation):
    - Build breath-level features (full 80-step u_in/u_out/time_step sequence + R,C)
    - Find nearest neighbor breaths in training
    - Predict the full 80-step pressure sequence by averaging neighbor sequences
    - Set expiratory (u_out==1) predictions to 0.0 because they are unscored
    - Snap inspiratory predictions to nearest valid pressure levels

    Why this moves score toward target:
      - The metric is MAE per timestep during inspiratory phase; breath-wise matching captures
        dynamics much better than row-wise matching across unrelated breaths.
      - Averaging neighbor pressure sequences is a legitimate non-leaky baseline and typically
        yields a large MAE drop versus row-wise KNN, moving toward the 0.14 range.
    """
    feat_cols = ["time_step", "u_in", "u_out"]

    train_usecols = [
        "breath_id",
        "id",
        "R",
        "C",
        "time_step",
        "u_in",
        "u_out",
        "pressure",
    ]
    test_usecols = ["breath_id", "id", "R", "C", "time_step", "u_in", "u_out"]

    train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=train_usecols)
    test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=test_usecols)

    X_tr, y_tr, tr_bids, _, _ = _make_breath_matrix(
        train, feat_cols=feat_cols, target_col="pressure"
    )
    X_te, _, te_bids, te_row_ids, te_u_out = _make_breath_matrix(
        test, feat_cols=feat_cols, target_col=None
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    knn = KNeighborsRegressor(
        n_neighbors=25,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )
    knn.fit(X_tr_s, y_tr)  # multi-output regression: predicts (80,) sequence per breath
    pred_seq = knn.predict(X_te_s).astype(np.float32)  # (n_test_breaths, 80)

    pred_seq = np.where(te_u_out == 1, 0.0, pred_seq)

    n_breaths = pred_seq.shape[0]
    pred_flat = pred_seq.reshape(-1)
    u_out_flat = te_u_out.reshape(-1)
    id_flat = te_row_ids.reshape(-1)

    inspiratory_mask = u_out_flat == 0
    snapped = pred_flat.copy()
    snapped[inspiratory_mask] = np.fromiter(
        (find_nearest(float(x)) for x in pred_flat[inspiratory_mask]),
        dtype=np.float32,
        count=int(inspiratory_mask.sum()),
    )

    sub = pd.DataFrame({"id": id_flat, "pressure": snapped.astype(np.float32)})
    sub = sub.sort_values("id").reset_index(drop=True)
    return sub


def g(dp):
    """
    Try to blend external submissions from dp if present; otherwise fall back to baseline.
    Bugfixes:
      - handle missing directory / no matching files / insufficient files robustly
      - always write a valid .csv submission with correct columns and numeric pressure
    """
    allow = [1348, 1991]
    l = []

    if os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            try:
                file_lb = int(i.split("/")[-1].split(".")[1].split(" ")[0])
            except Exception:
                continue
            if file_lb in allow:
                l.append(i)

    if len(l) == 0:
        sub = _baseline_predictions()
        sub.to_csv("submission.csv", index=False)
        return

    loop_time = 125
    splits = 2
    l.sort()

    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * round(len(l) / splits) :])
        else:
            flist.append(
                l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
            )

    blended = []
    for i in range(len(flist)):
        temp = wc(flist[i])
        if temp is not None:
            blended.append(temp)

    if len(blended) == 0:
        sub = _baseline_predictions()
        sub.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for j in range(loop_time):
        weight = []
        set_seed(j)
        for k in range(len(blended)):
            weight.append(rd())
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(blended)] * len(blended)
        else:
            for k in range(len(weight)):
                weight[k] /= weight_sum
        weight.sort(reverse=True)

        temp_pred = 0.0
        for k in range(len(blended)):
            temp_pred += blended[k] * weight[k]
        pred_list.append(temp_pred)
        del temp_pred
        gc.collect()

    output = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)




## === cell 2
g("/kaggle/input/ventilator-pressure-high-score-submissions")
print("Wrote: submission.csv")
print(pd.read_csv("submission.csv").head())
print(pd.read_csv("submission.csv").shape)
print(pd.read_csv("submission.csv").columns.tolist())

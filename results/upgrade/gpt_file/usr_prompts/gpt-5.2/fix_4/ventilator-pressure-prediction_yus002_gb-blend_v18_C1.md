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

3.9

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

0.2873555354677188

# 6. Current score

1.37289

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.83097) has done: 'I remove the hardcoded dependency on external “gb-blending” input files (which don’t exist in your environment) and instead make the notebook always produce a valid `submission.csv` directly from the provided competition data. Since your current script contains no train/inference model logic (only blending), the safest score-improving path toward the target is to generate a simple but valid baseline prediction using only available columns (a fast k-NN regression on engineered time-step features), which should beat the all-zeros sample submission while keeping runtime under the limit. I also fix pathing to use the existing `/kaggle/input/ventilator-pressure-prediction/` files and ensure the output format is exactly `id,pressure` with the correct row order. Finally, I keep your existing blending utilities intact but guard them so they don’t crash when files are missing.'
- What this solution (achieved 1.37299) has done: 'Your current score (1.83097, lower-is-better) is far from the target (0.28736), so we should improve performance while keeping the same overall approach (a simple k-NN on per-timestep engineered features). The biggest issue is that k-NN on all 5.4M rows is both slow and poorly aligned with the metric (only inspiratory phase is scored), so we train only on inspiratory rows (`u_out==0`) and keep predictions unchanged for expiratory rows (set to 0), which matches the evaluation semantics and typically yields a large MAE drop. To keep runtime under 600s without changing the model class, we also reduce the k-NN training set size via deterministic uniform downsampling of inspiratory rows (still the same k-NN logic), and we standardize features (k-NN is distance-based) using train-set mean/std to improve distance quality. Finally, we ensure submission alignment by directly assigning predictions in the test row order rather than merging on `id`.'
- What this solution (achieved 1.37289) has done: 'Your current MAE (1.37299, lower-is-better) is still far above the target (0.28736), so we should improve predictive accuracy with minimal changes while keeping the same k-NN baseline and feature set. The biggest gain, without changing the model class, is to (1) standardize k-NN features using global (numpy) stats for speed and consistency, and (2) post-process predictions by snapping them to the discrete set of pressures seen in training (the true target takes only fixed increments), which typically yields a large MAE improvement on this competition. We keep the inspiratory-only training/prediction semantics (u_out==0) exactly as you already do, and keep expiratory predictions at 0 (not scored). All paths stay the same, runtime remains under the limit, and a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os
import glob
import copy
import random
from random import random as rd

import numpy as np
import pandas as pd




## === cell 1
def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.RandomState(seed)


set_seed(2021)




## === cell 2
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
        weight1 = 0.8
        weight2 = 0.2
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = file_count**3
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
    output = pd.read_csv(
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = 0
    for _ in range(loop_time):
        weight = []
        set_seed(_)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        for j in range(len(flist)):
            output.pressure += flist[j] * weight[j]
    output.pressure /= loop_time
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 3
def blend(a, b, out_path="blend.csv"):
    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)
    a_df["pressure"] = a_df["pressure"] * 0.7 + b_df["pressure"] * 0.3
    a_df.to_csv(out_path, index=False)
    return a_df




## === cell 4
from sklearn.neighbors import KNeighborsRegressor

DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
    df["u_out_lag1"] = (
        df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )
    return df


train_fe = add_features(train)
test_fe = add_features(test)

feature_cols = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "u_in_cumsum",
    "u_in_lag1",
    "u_out_lag1",
]

train_insp_mask = train_fe["u_out"].values == 0
insp_idx = np.flatnonzero(train_insp_mask)

max_train_rows = 350000  # keep your existing runtime/quality tradeoff
if insp_idx.size > max_train_rows:
    take = np.linspace(0, insp_idx.size - 1, max_train_rows, dtype=np.int64)
    sel_idx = insp_idx[take]
else:
    sel_idx = insp_idx

X_train = train_fe.loc[sel_idx, feature_cols].astype(np.float32)
y_train = train_fe.loc[sel_idx, "pressure"].astype(np.float32)
X_test = test_fe[feature_cols].astype(np.float32)

Xtr = X_train.to_numpy(dtype=np.float32, copy=False)
Xte = X_test.to_numpy(dtype=np.float32, copy=False)
mu = Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
sigma = Xtr.std(axis=0, dtype=np.float64).astype(np.float32)
sigma[sigma == 0.0] = 1.0

Xtr_std = (Xtr - mu) / sigma
Xte_std = (Xte - mu) / sigma

knn = KNeighborsRegressor(
    n_neighbors=25, weights="distance", metric="minkowski", p=2, n_jobs=-1
)
knn.fit(Xtr_std, y_train.to_numpy(dtype=np.float32, copy=False))

test_insp_mask = test_fe["u_out"].values == 0
pred = np.zeros(len(test_fe), dtype=np.float32)

pred_insp = knn.predict(Xte_std[test_insp_mask]).astype(np.float32)

pressure_grid = np.sort(train_fe["pressure"].unique()).astype(np.float32)

idx = np.searchsorted(pressure_grid, pred_insp, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)
idx_left = np.clip(idx - 1, 0, len(pressure_grid) - 1)
choose_left = (idx > 0) & (
    np.abs(pred_insp - pressure_grid[idx_left])
    <= np.abs(pred_insp - pressure_grid[idx])
)
nearest = pressure_grid[idx]
nearest[choose_left] = pressure_grid[idx_left[choose_left]]

pred[test_insp_mask] = nearest

submission = pd.DataFrame(
    {"id": test_fe["id"].values, "pressure": pred.astype(np.float32)}
)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(submission.head())
print(submission.shape)

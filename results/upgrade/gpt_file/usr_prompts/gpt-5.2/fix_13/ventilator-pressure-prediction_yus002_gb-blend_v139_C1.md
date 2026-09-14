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

0.1387971868983444

# 6. Current score

5.14201

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.24999) has done: 'Your notebook fails because it tries to read external blend CSVs that aren’t present in this environment, so no submission ever gets written. I keep your existing pressure-rounding logic (nearest known pressure) but replace the missing-file blend with a minimal, self-contained baseline that trains on train.csv and predicts test.csv, then writes `submission.csv` with the required `id,pressure` columns. To stay within the installed packages (no deep learning libs available), the smallest stable approach is a simple per-(R,C,time_step) median target with safe fallbacks, and then apply your `find_nearest` discretization so predictions lie on the known pressure grid. This run end-to-end and produce a valid `.csv` submission.'
- What this solution (achieved 7.79119) has done: 'Your current score (7.24999 MAE) is far worse than the target (0.1388), so we should improve but with minimal changes and without changing the overall “train-on-train, predict-on-test, then snap to known pressure grid” logic. The biggest issue in your baseline is that it ignores key time-series dynamics and `u_in/u_out`, so I add a tiny set of per-row engineered features (cumulative `u_in`, lagged `u_in/u_out`, and a couple interactions) and switch from a pure median lookup to a lightweight linear regression fit on inspiratory rows only. This stays within installed packages (sklearn is available via sklearn-pandas dependency), keeps runtime reasonable by sampling breaths for training, and still applies your `find_nearest` discretization so predictions land on the known pressure grid. The output remain a valid `submission.csv` with exactly `id,pressure` aligned to the test set.'
- What this solution (achieved 7.98017) has done: 'Your current score is far above the target (lower is better), and the biggest preventable issue is that the model is trained only on inspiratory rows (`u_out==0`) but you predict all test rows, including expiratory phase which is not scored but can still hurt if badly miscalibrated. I keep your exact core approach (Ridge on your engineered features + snapping to the known pressure grid), but (1) train on all rows with a simple `sample_weight` to emphasize inspiratory phase rather than discarding expiratory data, and (2) add a tiny, metric-aligned postprocess: for `u_out==1` set pressure to a stable low baseline (nearest grid value of 0), which tends to be safer given the lung is venting. These are minimal changes that usually reduce MAE substantially without changing the overall pipeline, and the script still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.33408) has done: 'We need to move your MAE down toward the target (lower is better), and the current 7.98 indicates the model/predictions are badly misaligned with the competition structure. The biggest fix that stays within your core approach (Ridge + engineered features + snapping to the known pressure grid) is to build *breath-wise sequential features* (cumulative and lag features are good, but we also need “phase within breath” and “time since u_out opened”), and to ensure predictions are aligned to the correct `id` ordering after sorting during feature engineering. I also remove the `u_out==1 -> 0 pressure` hard override (it can be harmful because inspiratory-only scoring means you mainly need to be accurate when `u_out==0`; forcing zeros can distort transitions near the boundary). Finally, I keep snapping to the known pressure grid, but do it vectorized for speed and determinism.'
- What this solution (achieved 7.83802) has done: 'Your current MAE (4.334) is still far from the target (0.1388, lower is better), so we should improve accuracy while keeping your core approach unchanged: Ridge regression on engineered sequential features plus snapping predictions to the known pressure grid. The smallest high-impact improvement within that constraint is to (1) one-hot encode the discrete lung attributes (R, C, and their combination) instead of treating them as numeric, and (2) add a couple of lightweight “within-breath state” features (rolling mean of u_in, and cumulative count of inspiratory steps) that better capture phase dynamics without changing the training loop. I also fix a subtle ordering issue by ensuring we use the same sorted test rows for prediction and then map back to original order correctly, while still writing `submission.csv` with `id,pressure`. These changes are minimal, deterministic, and should reduce MAE materially toward your target.'
- What this solution (achieved 4.14793) has done: 'Your current MAE is far worse than the target (lower is better), so we should improve accuracy while keeping your existing core pipeline: Ridge regression on engineered sequential features plus snapping to the known pressure grid. The biggest fix with minimal conceptual change is to (1) remove the breath downsampling (it throws away too much signal) and instead train on all breaths with float32 features to keep memory/time under control, and (2) standardize features (Ridge is sensitive to feature scale, and your one-hot + continuous mix benefits materially from scaling). I also keep your test-row reordering logic but simplify it to a direct restore-by-index to avoid subtle misalignment risk. These changes preserve the same model family, loss, and overall workflow, and should move the score substantially toward the target band while still writing a valid `submission.csv`.'
- What this solution (achieved 4.13389) has done: 'Your current MAE (4.14793, lower is better) is still far above the target (0.1388), so we should improve accuracy while keeping the same core pipeline: engineered sequential features → scaled Ridge regression → snap predictions to the known pressure grid → write `submission.csv`. The biggest likely issue left is that the model is trained on *all rows* but Kaggle only scores inspiratory rows (`u_out==0`), so we keep training on all rows but (a) increase emphasis on inspiratory rows via `sample_weight`, and (b) add a minimal, metric-aligned postprocess that uses a separate inspiratory-only Ridge to generate predictions specifically for `u_out==0` rows in test (while keeping expiratory predictions from the all-rows model). This does not change the model family or loss—just a small weighting/segmentation tweak consistent with the evaluation—and should move the score materially toward the target. We also clip predictions to the min/max known pressure before snapping to avoid edge artifacts.'
- What this solution (achieved 4.13389) has done: 'Your current MAE is far above the target (lower is better), so we need a real accuracy gain without changing the core pipeline (engineered sequential features → scaled Ridge → snap to known pressure grid). The biggest minimal-impact fix is to align training with the metric: train the model(s) only on inspiratory rows (`u_out==0`), because expiratory rows are not scored and add a lot of contradictory signal. Then, for test expiratory rows, output a simple stable baseline (the minimum known pressure snapped to the grid), while keeping Ridge predictions for inspiratory rows. This keeps the same model family, feature set, scaling, and snapping—just changes the training subset and a metric-aligned postprocess to reduce MAE toward your target.'
- What this solution (achieved 2.31695) has done: 'Your MAE (4.13389) is still far above the target (0.1388, lower is better), and the biggest remaining issue that fits your “same core Ridge + engineered sequential features + snap-to-grid” pipeline is that you’re training one global model across all lung settings, which forces the Ridge to average very different dynamics across (R,C). With minimal conceptual change, we instead fit separate Ridge models per (R,C) group using the same features and scaler logic (fit scaler+model within each group), then stitch predictions back together; this usually yields a large MAE drop on this competition while staying within your exact modeling family and feature set. We keep training only on inspiratory rows to align with the metric, keep expiratory rows at the minimum pressure baseline, and keep the exact pressure-grid snapping postprocess. The output remains a valid `submission.csv` with `id,pressure` aligned to the original test order.'
- What this solution (achieved 5.14201) has done: 'Your current MAE (2.31695) is still much worse than the target (0.1388, lower is better), so we should improve accuracy while keeping the exact same core pipeline (engineered sequential features → per-(R,C) scaled Ridge → snap-to-known-pressure-grid → write submission). The smallest high-impact change is to make the training target match the metric by excluding “inspiratory-only” rows that are *not scored*: in this competition, only rows with `u_out==0` **and** `u_in>0` are scored, so we should train on that same subset instead of all `u_out==0` rows (which includes many near-zero u_in steps that behave differently). Then, at prediction time, we continue to only predict those scored rows with the Ridge model and keep all other rows at a safe baseline (minimum pressure on the grid), preserving your overall semantics. This is a minimal, metric-aligned masking change and should move the MAE down materially toward your target without changing model family, loss, or features.'
- What this solution (achieved 17.93847) has done: 'Your current MAE is far above the target (lower is better), so we should improve accuracy while keeping your same core pipeline (feature engineering → per-(R,C) scaled Ridge → snap-to-known-pressure-grid). The biggest minimal fix is to stop training on *all* rows while only predicting a subset: we train the per-(R,C) Ridge models on the same “scored-like” subset you predict for (`u_out==0 & u_in>0`) and keep the rest at a safe baseline, which avoids learning contradictory low-u_in behavior that isn’t evaluated. To reduce a common failure mode in this competition without changing the model family, we also add two tiny within-breath features (lagged pressure and lagged pressure difference) computed only from train and used as inputs, with safe zero fill for test; this preserves the same training loop but gives the linear model a strong state signal. Finally, we keep your ordering/alignment and snapping logic identical so the submission stays valid.'
- What this solution (achieved 5.14201) has done: 'Your current MAE is far worse than the target (lower is better), and the biggest issue is that you’re using `pressure_lag1/2` and `pressure_diff1` as features but they are available only in train (set to 0 in test), creating a large train/test mismatch that inflates error. I keep your exact core pipeline (same per-(R,C) scaled Ridge, same engineered non-target sequential features, same inspiratory-like mask, same snap-to-grid postprocess) but remove those target-derived features entirely to make train and test feature distributions consistent. I also make the sort/unsort alignment deterministic by resetting index after sorting inside `add_features` (so `_orig_order` is unambiguous), without changing semantics. The result should move the MAE substantially down toward your target while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
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


def snap_to_pressure_grid(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    left_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    right_idx = idx

    left_val = sorted_pressures[left_idx]
    right_val = sorted_pressures[right_idx]

    choose_left = np.abs(pred - left_val) < np.abs(pred - right_val)
    out = np.where(choose_left, left_val, right_val)

    return out.astype(float)


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


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = 154
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
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

set_seed(2021)

df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")


def add_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()

    df["_orig_order"] = np.arange(len(df), dtype=np.int64)
    df.sort_values(["breath_id", "time_step"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = (
        df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int16)
    )

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_time"] = df["u_in"] * df["time_step"]
    df["u_in_cumsum_time"] = df["u_in_cumsum"] * df["time_step"]

    df["u_in_over_R"] = df["u_in"] / df["R"].astype(float)
    df["u_in_over_C"] = df["u_in"] / df["C"].astype(float)

    df["timestep_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)
    df["is_insp"] = (df["u_out"] == 0).astype(np.int8)

    df["u_in_rollmean_5"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )
    df["insp_count"] = df.groupby("breath_id")["is_insp"].cumsum().astype(np.int16)

    u_out_change = df.groupby("breath_id")["u_out"].diff().fillna(0).astype(np.int8)
    opened = (u_out_change == 1).astype(np.int8)
    df["u_out_opened"] = opened
    df["u_out_open_event_id"] = (
        df.groupby("breath_id")["u_out_opened"].cumsum().astype(np.int16)
    )
    df["time_since_u_out_open"] = (
        df.groupby(["breath_id", "u_out_open_event_id"]).cumcount().astype(np.int16)
    )
    df.drop(columns=["u_out_open_event_id"], inplace=True)

    df["RC"] = (df["R"].astype(str) + "_" + df["C"].astype(str)).astype("category")
    df["R_cat"] = df["R"].astype("category")
    df["C_cat"] = df["C"].astype("category")

    return df


train_feat = add_features(df_train, is_train=True)
test_feat = add_features(df_test, is_train=False)

combined = pd.concat(
    [train_feat[["RC", "R_cat", "C_cat"]], test_feat[["RC", "R_cat", "C_cat"]]], axis=0
)
combined_dummies = pd.get_dummies(
    combined, columns=["RC", "R_cat", "C_cat"], drop_first=False, dtype=np.int8
)

train_dummies = combined_dummies.iloc[: len(train_feat)].reset_index(drop=True)
test_dummies = combined_dummies.iloc[len(train_feat) :].reset_index(drop=True)

train_feat = train_feat.reset_index(drop=True)
test_feat = test_feat.reset_index(drop=True)

train_feat = pd.concat([train_feat, train_dummies], axis=1)
test_feat = pd.concat([test_feat, test_dummies], axis=1)

feature_cols = [
    "time_step",
    "timestep_idx",
    "u_in",
    "u_out",
    "is_insp",
    "u_in_lag1",
    "u_in_lag2",
    "u_out_lag1",
    "u_in_diff1",
    "u_in_diff2",
    "u_in_cumsum",
    "u_in_time",
    "u_in_cumsum_time",
    "u_in_over_R",
    "u_in_over_C",
    "u_out_opened",
    "time_since_u_out_open",
    "u_in_rollmean_5",
    "insp_count",
]

dummy_cols = [
    c
    for c in train_feat.columns
    if c.startswith("RC_") or c.startswith("R_cat_") or c.startswith("C_cat_")
]
feature_cols = feature_cols + dummy_cols

X_train = train_feat[feature_cols].astype(np.float32).values
y_train = train_feat["pressure"].astype(np.float32).values
X_test_sorted = test_feat[feature_cols].astype(np.float32).values

insp_mask_train = (train_feat["u_out"].values == 0) & (train_feat["u_in"].values > 0.0)
insp_mask_test_sorted = (test_feat["u_out"].values == 0) & (
    test_feat["u_in"].values > 0.0
)

test_pred_sorted = np.full(
    shape=(len(test_feat),), fill_value=float(sorted_pressures[0]), dtype=np.float32
)

train_groups = train_feat[["R", "C"]].astype(np.int16).values
test_groups = test_feat[["R", "C"]].astype(np.int16).values
unique_groups = np.unique(train_groups, axis=0)

for Rv, Cv in unique_groups:
    tr_g = (train_groups[:, 0] == Rv) & (train_groups[:, 1] == Cv) & insp_mask_train
    te_g = (test_groups[:, 0] == Rv) & (test_groups[:, 1] == Cv) & insp_mask_test_sorted

    if not np.any(te_g):
        continue
    if not np.any(tr_g):
        continue

    scaler_g = StandardScaler(with_mean=True, with_std=True)
    Xtr_g = scaler_g.fit_transform(X_train[tr_g])
    Xte_g = scaler_g.transform(X_test_sorted[te_g])

    model_g = Ridge(alpha=0.5, random_state=2021)
    model_g.fit(Xtr_g, y_train[tr_g])

    test_pred_sorted[te_g] = model_g.predict(Xte_g).astype(np.float32)

test_pred = np.empty_like(test_pred_sorted)
test_pred[test_feat["_orig_order"].values] = test_pred_sorted

pmin, pmax = float(sorted_pressures[0]), float(sorted_pressures[-1])
test_pred = np.clip(test_pred, pmin, pmax)
test_pred = snap_to_pressure_grid(test_pred)

submission = pd.DataFrame({"id": df_test["id"].values, "pressure": test_pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Pressure unique (sample):", np.unique(submission["pressure"].values)[:10])

# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.1578757127431918

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'The notebook fails because it tries to read a non-existent external dataset (`ensemble-of-public-submissions`), so `df_sub` is never created and all downstream cells crash. To make it run end-to-end in your environment, I replace that input with the competition’s provided `sample_submission.csv` (same required columns) so we always have a valid submission template. Since this baseline has no model predictions available, I keep the original core “round-to-known-pressures” logic but apply it to a constant initial prediction (0), which produces a valid submission file (score be poor but the pipeline work). I also remove IPython magics (`%matplotlib inline`, `%%time`) that can error in script execution and ensure the output file is named `submission.csv`.'
- What this solution (achieved 9.01216) has done: 'Your current 17.65 MAE comes from effectively predicting a constant (0) then rounding, which is far from the target 0.158, so we need a real (but minimal) predictive signal. To keep core logic intact, I only change how the initial `pressure` values are filled in `df_sub`: use a lightweight per‑(R,C,time_step,u_out) median lookup built from train (no model/loops added), then apply your existing probability-weighted rounding exactly as before. This stays aligned with the metric by learning typical inspiratory pressures at each time step and valve state, while remaining fast (<600s) and producing a valid `submission.csv`. I also keep the original rounding baseline cells, but make them operate on these filled predictions rather than zeros.'
- What this solution (achieved 7.53955) has done: 'Your current score (9.01216 MAE) is far above the target (0.1579), so we need a meaningful but still lightweight predictive improvement while keeping your “median lookup + rounding-to-known-pressures” core logic intact. The smallest high-impact fix is to make the median lookup respect the competition’s scoring rule (only inspiratory phase is scored) by building the medians using only `u_out == 0` rows from train, then using the same medians for all test rows. Additionally, using a more breath-specific key by adding `u_in` (rounded to 1 decimal to keep the map size manageable) typically improves pressure estimation without changing the approach. Everything else (rounding functions, submission format, paths) stays the same.'
- What this solution (achieved 7.84143) has done: 'Your current approach is a fast “train median lookup → fill test predictions → round to known pressure classes”, but it’s leaving accuracy on the table because the mapping is built only on inspiratory rows yet is still keyed by `u_out` (which becomes constant 0) and doesn’t use any within-breath dynamics. I keep the same core logic, but add a minimal, cheap breath-dynamics feature: cumulative inspired volume (`u_in` integrated over time) and its interaction with `(R,C)`, then rebuild the same median lookup using these keys (still only from `u_out==0` to match the metric). This typically reduces MAE substantially while preserving your overall pipeline, rounding logic, and submission format. I also keep the original fallbacks, just updated to include the new feature where appropriate, and ensure everything still runs under the 600s constraint.'
- What this solution (achieved 8.23944) has done: 'Your current MAE (7.84) is far worse than the target (0.158), so we should improve predictions while keeping your same “train median lookup → fill test → rounding to known pressures” pipeline intact. The biggest issue is that your lookup keys include raw `time_step` and fine rounded `u_in/cum_u_in`, which often won’t match between train/test due to floating precision and over-fragmentation, causing heavy fallback usage and poor accuracy. I make a minimal, metric-aligned adjustment: compute an integer time index per breath (0..79) and use that instead of float `time_step` in the grouping keys (same logic, just a stable join key). I also slightly coarsen `cum_u_in` rounding (keeping the same feature) to reduce sparsity and increase exact key hits, which should move the score substantially toward the target while preserving your overall approach and producing `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing: {TRAIN_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_PATH), f"Missing: {TEST_PATH}"



## === cell 1
import matplotlib.pyplot as plt



## === cell 2
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
df_sub = pd.read_csv(SAMPLE_SUB_PATH)

if list(df_sub.columns) != ["id", "pressure"]:
    df_sub = df_sub[["id", "pressure"]].copy()

if not df_sub["id"].equals(df_test["id"]):
    df_sub = df_test[["id"]].copy()
    df_sub["pressure"] = 0.0


def add_breath_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort")

    dt = df.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
    df["u_in_dt"] = df["u_in"] * dt
    df["cum_u_in"] = df.groupby("breath_id", sort=False)["u_in_dt"].cumsum()

    df["t_idx"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    return df


df_train = add_breath_dynamics_features(df_train)
df_test_feat = add_breath_dynamics_features(df_test)

df_train_insp = df_train[df_train["u_out"] == 0].copy()

UIN_ROUND_DECIMALS = 1
CUMUIN_ROUND_DECIMALS = 1

df_train_insp["u_in_r"] = df_train_insp["u_in"].round(UIN_ROUND_DECIMALS)
df_test_feat["u_in_r"] = df_test_feat["u_in"].round(UIN_ROUND_DECIMALS)

df_train_insp["cum_u_in_r"] = df_train_insp["cum_u_in"].round(CUMUIN_ROUND_DECIMALS)
df_test_feat["cum_u_in_r"] = df_test_feat["cum_u_in"].round(CUMUIN_ROUND_DECIMALS)

rc_stats = (
    df_train_insp.groupby(["R", "C", "t_idx"], sort=False)["cum_u_in"]
    .agg(["median"])
    .rename(columns={"median": "cum_u_in_med_rc_t"})
    .reset_index()
)
df_train_insp = df_train_insp.merge(rc_stats, on=["R", "C", "t_idx"], how="left")
df_test_feat = df_test_feat.merge(rc_stats, on=["R", "C", "t_idx"], how="left")

df_train_insp["cum_u_in_norm"] = df_train_insp["cum_u_in"] / df_train_insp[
    "cum_u_in_med_rc_t"
].replace(0.0, 1.0).fillna(1.0)
df_test_feat["cum_u_in_norm"] = df_test_feat["cum_u_in"] / df_test_feat[
    "cum_u_in_med_rc_t"
].replace(0.0, 1.0).fillna(1.0)

CUMNORM_ROUND_DECIMALS = 2
df_train_insp["cum_u_in_norm_r"] = df_train_insp["cum_u_in_norm"].round(
    CUMNORM_ROUND_DECIMALS
)
df_test_feat["cum_u_in_norm_r"] = df_test_feat["cum_u_in_norm"].round(
    CUMNORM_ROUND_DECIMALS
)

train_key_cols = ["R", "C", "t_idx", "u_in_r", "cum_u_in_r", "cum_u_in_norm_r"]
test_key_cols = ["R", "C", "t_idx", "u_in_r", "cum_u_in_r", "cum_u_in_norm_r"]

median_map = (
    df_train_insp.groupby(train_key_cols, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure"})
)

median_fallback = (
    df_train_insp.groupby(["R", "C", "t_idx", "u_in_r", "cum_u_in_norm_r"], sort=False)[
        "pressure"
    ]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback"})
)

median_fallback2 = (
    df_train_insp.groupby(["R", "C", "t_idx", "u_in_r"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback2"})
)

median_fallback3 = (
    df_train_insp.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback3"})
)

median_fallback4 = (
    df_train_insp.groupby(["t_idx"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback4"})
)

test_pred = df_test_feat[
    ["id"] + test_key_cols + ["R", "C", "t_idx", "u_in_r", "cum_u_in_norm_r"]
].merge(median_map, on=test_key_cols, how="left")
test_pred = test_pred.merge(
    median_fallback, on=["R", "C", "t_idx", "u_in_r", "cum_u_in_norm_r"], how="left"
)
test_pred = test_pred.merge(
    median_fallback2, on=["R", "C", "t_idx", "u_in_r"], how="left"
)
test_pred = test_pred.merge(median_fallback3, on=["R", "C", "t_idx"], how="left")
test_pred = test_pred.merge(median_fallback4, on=["t_idx"], how="left")

global_median = float(df_train_insp["pressure"].median())
test_pred["pred_pressure"] = (
    test_pred["pred_pressure"]
    .fillna(test_pred["pred_pressure_fallback"])
    .fillna(test_pred["pred_pressure_fallback2"])
    .fillna(test_pred["pred_pressure_fallback3"])
    .fillna(test_pred["pred_pressure_fallback4"])
    .fillna(global_median)
)

df_sub = df_test[["id"]].copy()
df_sub["pressure"] = test_pred["pred_pressure"].astype(float).values



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/879757954.py in <cell line: 0>()
    111 test_pred = df_test_feat[
    112     ["id"] + test_key_cols + ["R", "C", "t_idx", "u_in_r", "cum_u_in_norm_r"]
--> 113 ].merge(median_map, on=test_key_cols, how="left")
    114 test_pred = test_pred.merge(
    115     median_fallback, on=["R", "C", "t_idx", "u_in_r", "cum_u_in_norm_r"], how="left"

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1308                         #  the latter of which will raise
   1309                         lk = cast(Hashable, lk)
-> 1310                         left_keys.append(left._get_label_or_level_values(lk))
   1311                         join_names.append(lk)
   1312                     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1923 
   1924             label_axis_name = "column" if axis == 0 else "index"
-> 1925             raise ValueError(
   1926                 f"The {label_axis_name} label '{key}' is not unique.{multi_message}"
   1927             )

ValueError: The column label 'R' is not unique.

## === cell 3
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)



## === cell 4
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction: float) -> float:
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )




## === cell 5
df_sub["pressure"] = df_sub["pressure"].astype(float).apply(find_nearest)



## === cell 6
df_sub.head()



## === cell 7
df_sub.to_csv("submission.csv", index=False)
submission_round_LB154 = df_sub.copy()



## === cell 8
assert submission_round_LB154.shape[0] == df_test.shape[0]
assert list(submission_round_LB154.columns) == ["id", "pressure"]
assert (
    submission_round_LB154["id"].is_monotonic_increasing
    or submission_round_LB154["id"].nunique() == submission_round_LB154.shape[0]
)



## === cell 9
pressure_freq = df_train["pressure"].value_counts().to_frame()
pressure_freq["freq"] = df_train["pressure"].value_counts(normalize=True).values
pressure_freq = pressure_freq.sort_index().reset_index()
pressure_freq.columns = ["pressure", "count", "freq"]



## === cell 10
pressure_freq.head()



## === cell 11
pressure_freq["pressure_pre"] = pressure_freq["pressure"].shift(1)
pressure_freq["pressure_step"] = (
    pressure_freq["pressure"] - pressure_freq["pressure_pre"]
)

pressure_freq["freq_pre"] = pressure_freq["freq"].shift(1)
pressure_freq["freq_relative_pct"] = pressure_freq["freq"] / (
    pressure_freq["freq"] + pressure_freq["freq_pre"]
)
pressure_freq["pressure_prob"] = (
    pressure_freq["pressure_pre"]
    + pressure_freq["pressure_step"] * pressure_freq["freq_relative_pct"]
)



## === cell 12
pressure_freq["pressure_half"] = (
    pressure_freq["pressure_pre"] + pressure_freq["pressure"]
) / 2
plt.figure(figsize=(10, 6))
(pressure_freq["pressure_prob"] - pressure_freq["pressure_half"]).plot()
plt.title("Probability-weighted rounding threshold offset")
plt.tight_layout()
plt.show()



## === cell 13
total_pressures_len = len(sorted_pressures)


def find_nearest_prob(prediction: float) -> float:
    """
    Probability weighted rounding.
    Keeps the original notebook logic; selects rounding cutpoint based on empirical class frequencies.
    """
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    cut_val = pressure_freq["pressure_prob"].iloc[insert_idx]
    if pd.isna(cut_val):
        cut_val = (lower_val + upper_val) / 2.0
    return float(lower_val if prediction < cut_val else upper_val)




## === cell 14
df_sub = df_test[["id"]].copy()
df_sub["pressure"] = test_pred["pred_pressure"].astype(float).values
df_sub["pressure"] = df_sub["pressure"].astype(float).apply(find_nearest_prob)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1097449815.py in <cell line: 0>()
      1 df_sub = df_test[["id"]].copy()
----> 2 df_sub["pressure"] = test_pred["pred_pressure"].astype(float).values
      3 df_sub["pressure"] = df_sub["pressure"].astype(float).apply(find_nearest_prob)
      4 

NameError: name 'test_pred' is not defined

## === cell 15
df_sub.to_csv("submission.csv", index=False)
submission_prob = df_sub.copy()



## === cell 16
submission_prob.head()



## === cell 17
assert os.path.exists("submission.csv"), "submission.csv was not created."



## === cell 18
y_old = submission_round_LB154["pressure"]
y_new = submission_prob["pressure"]



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'pressure'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3730445487.py in <cell line: 0>()
      1 y_old = submission_round_LB154["pressure"]
----> 2 y_new = submission_prob["pressure"]
      3 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'pressure'

## === cell 19
(y_new - y_old).describe().round(6)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1575199154.py in <cell line: 0>()
----> 1 (y_new - y_old).describe().round(6)
      2 

NameError: name 'y_new' is not defined

## === cell 20
int((y_new > y_old).sum()), float((y_new > y_old).mean())



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1514105443.py in <cell line: 0>()
----> 1 int((y_new > y_old).sum()), float((y_new > y_old).mean())
      2 

NameError: name 'y_new' is not defined

## === cell 21
int((y_new < y_old).sum()), float((y_new < y_old).mean())

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/688599251.py in <cell line: 0>()
----> 1 int((y_new < y_old).sum()), float((y_new < y_old).mean())

NameError: name 'y_new' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission must contain the 'pressure' column

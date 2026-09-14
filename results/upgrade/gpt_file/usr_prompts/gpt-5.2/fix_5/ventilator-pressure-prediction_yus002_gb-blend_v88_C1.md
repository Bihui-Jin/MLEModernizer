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

0.1552281166569305

# 6. Current score

7.8339

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.44669) has done: 'I fix the runtime error by ensuring the intermediate `.map(...)` calls produce a proper `Series` aligned to `test_b.index` (pandas 2.x returns an `Index` for `Index.map`, which `fillna` rejects). I also keep the existing blending/nearest-pressure snapping logic intact and only adjust the mapping fallback code to be type-safe and deterministic. Finally, I make sure the script always writes a valid `submission.csv` with the required `id,pressure` columns in the working directory.'
- What this solution (achieved 7.70246) has done: 'Your current score (7.44669 MAE; lower is better) is far from the target (0.1552), so we need a meaningful but still “logic-preserving” improvement. The biggest issue is that your predictions ignore the time-series dynamics and the scoring mask (only inspiratory phase where `u_out==0` matters), so I keep your mapping/blending structure but add a minimal, deterministic per-breath cumulative feature (`u_in` integral) and use the same groupby-median mapping idea on that feature, prioritizing `u_out==0` rows during mapping to better match the metric. I also ensure the final submission keeps `id` alignment and continues snapping to the nearest allowed pressure values. These changes keep the same overall approach (rule-based median mappings + blend + snap) while substantially reducing MAE toward the target.'
- What this solution (achieved 7.8339) has done: 'Your current MAE (7.70; lower is better) is far above the target (0.155), so we need a meaningful improvement while keeping your current “median-mapping + blend + snap-to-grid” core logic unchanged. The biggest win with minimal conceptual change is to (1) build mappings separately for inspiratory vs expiratory (`u_out`) because pressure behavior differs and the metric only scores inspiratory, and (2) replace the current global fill fallback with a more specific per-(R,C,u_out) median fallback before going to global medians. This keeps your exact approach (groupby median lookups + hierarchical fillna + 50/50 blend + nearest-pressure snapping) but makes the lookup hierarchy better aligned to the physics and the scoring mask. The submission writing, file paths, and snapping logic remain unchanged.'

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
    loop_time = 150
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
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
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
    if isinstance(a, str):
        a = pd.read_csv(a)
    else:
        a = a.copy()
    if isinstance(b, str):
        b = pd.read_csv(b)
    else:
        b = b.copy()

    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub_template = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

train_insp = df_train[df_train["u_out"] == 0].copy()
train_exp = df_train[df_train["u_out"] == 1].copy()

global_median_insp = float(train_insp["pressure"].median())
global_median_exp = float(train_exp["pressure"].median())
global_median_all = float(df_train["pressure"].median())


def add_cumuin_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True)
    dt = df.groupby("breath_id")["time_step"].diff().fillna(0.0).astype("float64")
    df["cum_uin"] = (
        (df["u_in"].astype("float64") * dt).groupby(df["breath_id"]).cumsum()
    )
    df["cum_uin_r2"] = df["cum_uin"].round(2)
    df["u_in_r0"] = df["u_in"].round(0).astype("int64")
    df["u_in_r1"] = df["u_in"].round(1)
    return df


train_feat = add_cumuin_feature(df_train)
test_feat = add_cumuin_feature(df_test)

map_uin_insp = (
    train_feat.loc[train_feat["u_out"] == 0].groupby("u_in_r1")["pressure"].median()
)
map_uin_exp = (
    train_feat.loc[train_feat["u_out"] == 1].groupby("u_in_r1")["pressure"].median()
)

test_a = test_feat.copy()
pred_a_series = pd.Series(index=test_a.index, dtype="float64")

mask_insp = test_a["u_out"].to_numpy() == 0
mask_exp = ~mask_insp

pred_a_series.loc[mask_insp] = (
    test_a.loc[mask_insp, "u_in_r1"].map(map_uin_insp).astype("float64")
)
pred_a_series.loc[mask_exp] = (
    test_a.loc[mask_exp, "u_in_r1"].map(map_uin_exp).astype("float64")
)

pred_a_series.loc[mask_insp] = pred_a_series.loc[mask_insp].fillna(global_median_insp)
pred_a_series.loc[mask_exp] = pred_a_series.loc[mask_exp].fillna(global_median_exp)
pred_a = pred_a_series.fillna(global_median_all).to_numpy()

sub_a = sub_template.copy()
sub_a["pressure"] = pred_a

train_b = train_feat.copy()
test_b = test_feat.copy()

map_rcuout_uin_cum = train_b.groupby(["R", "C", "u_out", "u_in_r0", "cum_uin_r2"])[
    "pressure"
].median()
map_rcuout_uin = train_b.groupby(["R", "C", "u_out", "u_in_r0"])["pressure"].median()
map_rcuout = train_b.groupby(["R", "C", "u_out"])["pressure"].median()

map_rco_uout = map_rcuout  # alias for clarity

map_ruout = train_b.groupby(["R", "u_out"])["pressure"].median()
map_cuout = train_b.groupby(["C", "u_out"])["pressure"].median()

rcuout_uin_cum_idx = pd.MultiIndex.from_frame(
    test_b[["R", "C", "u_out", "u_in_r0", "cum_uin_r2"]]
)
pred_b = pd.Series(
    rcuout_uin_cum_idx.map(map_rcuout_uin_cum), index=test_b.index, dtype="float64"
)

rcuout_uin_idx = pd.MultiIndex.from_frame(test_b[["R", "C", "u_out", "u_in_r0"]])
pred_b = pred_b.fillna(
    pd.Series(rcuout_uin_idx.map(map_rcuout_uin), index=test_b.index, dtype="float64")
)

rcuout_idx = pd.MultiIndex.from_frame(test_b[["R", "C", "u_out"]])
pred_b = pred_b.fillna(
    pd.Series(rcuout_idx.map(map_rcuout), index=test_b.index, dtype="float64")
)

ruout_idx = pd.MultiIndex.from_frame(test_b[["R", "u_out"]])
pred_b = pred_b.fillna(
    pd.Series(ruout_idx.map(map_ruout), index=test_b.index, dtype="float64")
)

cuout_idx = pd.MultiIndex.from_frame(test_b[["C", "u_out"]])
pred_b = pred_b.fillna(
    pd.Series(cuout_idx.map(map_cuout), index=test_b.index, dtype="float64")
)

mask_insp_b = test_b["u_out"].to_numpy() == 0
mask_exp_b = ~mask_insp_b
pred_b.loc[mask_insp_b] = pred_b.loc[mask_insp_b].fillna(global_median_insp)
pred_b.loc[mask_exp_b] = pred_b.loc[mask_exp_b].fillna(global_median_exp)

pred_b = pred_b.fillna(global_median_all).to_numpy()

sub_b = sub_template.copy()
sub_b["pressure"] = pred_b

sub = blend(sub_a, sub_b)

submission = sub_template[["id"]].copy()
submission["pressure"] = sub["pressure"].to_numpy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv columns:", submission.columns.tolist())
print("pressure dtype:", submission["pressure"].dtype)

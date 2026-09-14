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

0.1558650678850587

# 6. Current score

1.36514

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 19.55027) has done: 'The crash happens because the notebook tries to read external submission files from `../input/gb-blending/`, which are not present in your environment. To keep the core “blend two submissions then snap to nearest allowed pressure” logic intact while making it run end-to-end, I add a safe fallback that creates two simple baseline submissions from the provided `sample_submission.csv` (constant predictions) when those files are missing. This guarantees a valid `submission.csv` is written with the required `id,pressure` columns and no runtime errors. The score won’t be competitive (since we’re not training a model), but it yield a valid submission and preserve the blending pipeline structure.'
- What this solution (achieved 8.25062) has done: 'Your current score is far worse than the target (lower is better), mainly because the fallback creates constant predictions that ignore the time-series controls and lung attributes. To move the score substantially toward the target while keeping changes minimal, I keep your “snap to nearest allowed pressure” post-processing and submission writing intact, but replace the fallback constants with a simple, legitimate, data-driven baseline: a k-nearest-neighbors regressor trained on per-timestep features available in train/test. This preserves the overall structure (read train → predict test → snap → write CSV) and avoids any missing external files, producing a valid `submission.csv` end-to-end. The blend step is kept, but both inputs become model-based so the blend remains meaningful.'
- What this solution (achieved 1.58066) has done: 'Your current MAE (8.25062; lower is better) is far from the target (0.1559), so the main issue is that the KNN fallback is not modeling the per-breath time-series structure well enough. Keeping your existing “train → predict test → snap to nearest allowed pressure → blend → write submission.csv” core flow intact, I replace the KNN fallback with a light, competition-standard per-timestep regression baseline (HistGradientBoostingRegressor) and add a few simple lag/interaction features that don’t change evaluation semantics. I also fix the id/order alignment bug in the fallback generator (it currently risks mismatching predictions to ids), which can severely hurt score even if the model is decent. The blend and nearest-pressure snapping remain exactly as you designed.'
- What this solution (achieved 1.47652) has done: 'Your current score is much worse than the target (lower is better), so we should make small, legitimate changes that better match the evaluation: only inspiratory timesteps (u_out==0) matter. I keep your exact overall flow (build two model submissions → blend → snap to nearest allowed pressure → write submission.csv) and your model family (HistGradientBoostingRegressor), but I (1) train using sample weights to focus learning on u_out==0 without discarding data, and (2) add a couple of very cheap per-breath features (rolling mean and a u_out run-length) that help the model capture the time-series structure while staying within the same feature-engineering approach. These are minimal changes that typically move MAE substantially toward strong baselines on this competition, while preserving your blending and snapping logic. The submission alignment by id is kept explicit to avoid silent ordering issues.'
- What this solution (achieved 1.36514) has done: 'Your current score is far worse than the target (lower is better), and a big part of the gap is that the model is being trained to predict pressure on *all* timesteps even though Kaggle only scores inspiratory timesteps (`u_out==0`). To move the MAE toward the target without changing your overall pipeline, I keep your exact model family (HistGradientBoostingRegressor), blending, and “snap to nearest allowed pressure”, but I (1) train only on inspiratory rows (so the loss matches the evaluation), and (2) force test predictions on expiratory rows (`u_out==1`) to a safe constant (0) because they are not scored but snapping them to pressure grid can still add noise to the blend. I also add two very cheap per-breath cumulative features that usually help tree models capture dynamics without changing the approach. The submission writing and id alignment remain explicit and unchanged.'

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
        input_list[i] = (pd.read_csv(input_list[i]).pressure).to_numpy().ravel()
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
    for loop_idx in range(loop_time):
        weight = []
        set_seed(loop_idx)
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
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def _add_features(df):
    df = df.sort_values(["breath_id", "time_step"]).copy()

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int64)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / (g.cumcount() + 1)

    df["RC"] = df["R"].astype(np.float32) * df["C"].astype(np.float32)
    df["u_in_x_R"] = df["u_in"] * df["R"]
    df["u_in_x_C"] = df["u_in"] * df["C"]

    df["u_in_roll3_mean"] = (
        g["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )

    uout = df["u_out"].to_numpy()
    bid = df["breath_id"].to_numpy()
    run = np.zeros(len(df), dtype=np.int16)
    r = 0
    prev_bid = bid[0] if len(bid) else -1
    for i in range(len(df)):
        if bid[i] != prev_bid:
            r = 0
            prev_bid = bid[i]
        if uout[i] == 1:
            r += 1
        else:
            r = 0
        run[i] = r
    df["u_out_runlen"] = run.astype(np.int16)

    df["u_out_cumsum"] = g["u_out"].cumsum().astype(np.int16)
    df["u_in_ewm_span5"] = (
        g["u_in"]
        .ewm(span=5, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )

    return df


def _make_hgb_submission(path, max_iter=250, learning_rate=0.08, max_depth=7):
    from sklearn.ensemble import HistGradientBoostingRegressor

    train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

    train = _add_features(train)
    test = _add_features(test)

    feats = [
        "R",
        "C",
        "time_step",
        "u_in",
        "u_out",
        "u_in_lag1",
        "u_in_lag2",
        "u_out_lag1",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_cumsum",
        "u_in_cummean",
        "RC",
        "u_in_x_R",
        "u_in_x_C",
        "u_in_roll3_mean",
        "u_out_runlen",
        "u_out_cumsum",
        "u_in_ewm_span5",
    ]

    insp_mask = train["u_out"].to_numpy() == 0
    X_train = train.loc[insp_mask, feats].to_numpy(dtype=np.float32, copy=False)
    y_train = train.loc[insp_mask, "pressure"].to_numpy(dtype=np.float32, copy=False)

    X_test = test[feats].to_numpy(dtype=np.float32, copy=False)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        max_iter=int(max_iter),
        random_state=2021,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(np.float32)

    test_u_out = test["u_out"].to_numpy()
    pred = np.where(test_u_out == 1, 0.0, pred).astype(np.float32)

    sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
    pred_df = pd.DataFrame({"id": test["id"].to_numpy(), "pressure": pred})
    pred_df = pred_df.sort_values("id").reset_index(drop=True)
    sub = sub.sort_values("id").reset_index(drop=True)
    sub["pressure"] = pred_df["pressure"].to_numpy()

    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv(path, index=False)
    return path




## === cell 2
a = "../input/gb-blending/0.156 seed 33.csv"
b = "../input/gb-blending/0.157 blend.csv"

if not os.path.exists(a) or not os.path.exists(b):
    a = _make_hgb_submission(
        "fallback_a.csv", max_iter=300, learning_rate=0.08, max_depth=7
    )
    b = _make_hgb_submission(
        "fallback_b.csv", max_iter=260, learning_rate=0.06, max_depth=8
    )

blend_df = blend(a, b)

blend_df[["id", "pressure"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", blend_df[["id", "pressure"]].shape)
print(blend_df.head())

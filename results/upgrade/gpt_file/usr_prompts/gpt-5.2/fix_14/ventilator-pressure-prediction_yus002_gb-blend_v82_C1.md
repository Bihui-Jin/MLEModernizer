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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 19.55027) has done: 'The crash happens because the notebook tries to read external submission files from `../input/gb-blending/`, which are not present in your environment. To keep the core “blend two submissions then snap to nearest allowed pressure” logic intact while making it run end-to-end, I add a safe fallback that creates two simple baseline submissions from the provided `sample_submission.csv` (constant predictions) when those files are missing. This guarantees a valid `submission.csv` is written with the required `id,pressure` columns and no runtime errors. The score won’t be competitive (since we’re not training a model), but it yield a valid submission and preserve the blending pipeline structure.'
- What this solution (achieved 8.25062) has done: 'Your current score is far worse than the target (lower is better), mainly because the fallback creates constant predictions that ignore the time-series controls and lung attributes. To move the score substantially toward the target while keeping changes minimal, I keep your “snap to nearest allowed pressure” post-processing and submission writing intact, but replace the fallback constants with a simple, legitimate, data-driven baseline: a k-nearest-neighbors regressor trained on per-timestep features available in train/test. This preserves the overall structure (read train → predict test → snap → write CSV) and avoids any missing external files, producing a valid `submission.csv` end-to-end. The blend step is kept, but both inputs become model-based so the blend remains meaningful.'
- What this solution (achieved 1.58066) has done: 'Your current MAE (8.25062; lower is better) is far from the target (0.1559), so the main issue is that the KNN fallback is not modeling the per-breath time-series structure well enough. Keeping your existing “train → predict test → snap to nearest allowed pressure → blend → write submission.csv” core flow intact, I replace the KNN fallback with a light, competition-standard per-timestep regression baseline (HistGradientBoostingRegressor) and add a few simple lag/interaction features that don’t change evaluation semantics. I also fix the id/order alignment bug in the fallback generator (it currently risks mismatching predictions to ids), which can severely hurt score even if the model is decent. The blend and nearest-pressure snapping remain exactly as you designed.'
- What this solution (achieved 1.47652) has done: 'Your current score is much worse than the target (lower is better), so we should make small, legitimate changes that better match the evaluation: only inspiratory timesteps (u_out==0) matter. I keep your exact overall flow (build two model submissions → blend → snap to nearest allowed pressure → write submission.csv) and your model family (HistGradientBoostingRegressor), but I (1) train using sample weights to focus learning on u_out==0 without discarding data, and (2) add a couple of very cheap per-breath features (rolling mean and a u_out run-length) that help the model capture the time-series structure while staying within the same feature-engineering approach. These are minimal changes that typically move MAE substantially toward strong baselines on this competition, while preserving your blending and snapping logic. The submission alignment by id is kept explicit to avoid silent ordering issues.'
- What this solution (achieved 1.36514) has done: 'Your current score is far worse than the target (lower is better), and a big part of the gap is that the model is being trained to predict pressure on *all* timesteps even though Kaggle only scores inspiratory timesteps (`u_out==0`). To move the MAE toward the target without changing your overall pipeline, I keep your exact model family (HistGradientBoostingRegressor), blending, and “snap to nearest allowed pressure”, but I (1) train only on inspiratory rows (so the loss matches the evaluation), and (2) force test predictions on expiratory rows (`u_out==1`) to a safe constant (0) because they are not scored but snapping them to pressure grid can still add noise to the blend. I also add two very cheap per-breath cumulative features that usually help tree models capture dynamics without changing the approach. The submission writing and id alignment remain explicit and unchanged.'
- What this solution (achieved 1.34105) has done: 'Your score is still far from the target (lower is better), so we should make a minimal change that directly better matches the evaluation: Kaggle scores only inspiratory timesteps (`u_out==0`). I keep your exact model family (HistGradientBoostingRegressor), feature set, blending, and “snap to nearest allowed pressure” logic, but change the post-processing so we only snap inspiratory predictions and we set expiratory (`u_out==1`) predictions to 0 after blending (so expiratory values don’t contaminate the inspiratory blend via snapping/median effects). I also fix the `id` alignment robustly by merging on `id` (instead of relying on sort/reset) to eliminate any silent mismatch risk. These changes are small, preserve the core approach, and should move MAE toward the target without trying to over-optimize.'
- What this solution (achieved 1.41258) has done: 'Your current MAE (1.341) is still far worse than the target (0.1559; lower is better), so we should make a small change that better matches the competition’s known structure without changing your overall approach (HGBRegressor → blend → snap-to-grid → write submission). The biggest easy win here is to encode the (R, C) pair categorically (9 combos) instead of leaving them as raw integers, which tree models often benefit from, while keeping the same model family and training loop. I also align the snapping to only happen on inspiratory rows both inside `_make_hgb_submission` and after blending (you already do this, but we make it fully consistent and avoid any accidental snapping of expiratory values). Finally, I keep your robust id-based merge so row order can’t silently break the submission.'
- What this solution (achieved 1.33453) has done: 'Your current score is far worse than the target (lower is better), so we should make a small change that legitimately improves MAE without changing your overall pipeline (HGBRegressor → blend → snap-to-grid → write submission). The biggest low-risk win is to stop treating `RC_cat` as a numeric magnitude and instead one-hot encode the 9 possible (R,C) combinations using `OneHotEncoder` inside a `ColumnTransformer`, while keeping the same HistGradientBoostingRegressor and the same feature set. This preserves the training approach and evaluation semantics, but gives the model a cleaner way to use (R,C) as categorical context, which typically improves this competition baseline materially. I also keep your inspiratory-only training and inspiratory-only snapping exactly as-is, and still set expiratory predictions to 0.'

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

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["pressure"],
    dtype={"pressure": np.float32},
)
sorted_pressures = np.unique(df_train["pressure"].to_numpy(np.float32, copy=False))
sorted_pressures.sort()
sorted_pressures = sorted_pressures.astype(np.float32, copy=False)
total_pressures_len = int(sorted_pressures.shape[0])
del df_train
gc.collect()


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


def find_nearest_vec(pred):
    pred = np.asarray(pred, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx0 = np.clip(idx, 0, total_pressures_len - 1)
    idx1 = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper = sorted_pressures[idx0]
    lower = sorted_pressures[idx1]
    choose_lower = np.abs(lower - pred) < np.abs(upper - pred)
    out = np.where(
        idx == 0,
        sorted_pressures[0],
        np.where(
            idx == total_pressures_len,
            sorted_pressures[-1],
            np.where(choose_lower, lower, upper),
        ),
    )
    return out.astype(np.float32, copy=False)


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
    output["pressure"] = find_nearest_vec(
        output["pressure"].to_numpy(np.float32, copy=False)
    )
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a_df = pd.read_csv(
        a, usecols=["id", "pressure"], dtype={"id": np.int64, "pressure": np.float32}
    )
    b_p = pd.read_csv(b, usecols=["pressure"], dtype={"pressure": np.float32})[
        "pressure"
    ].to_numpy(np.float32, copy=False)
    a_p = a_df["pressure"].to_numpy(np.float32, copy=False)
    a_df["pressure"] = a_p * np.float32(0.55) + b_p * np.float32(0.45)
    a_df.to_csv("blend.csv", index=False)
    return a_df


def _add_features(df):
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

    n = len(df)
    if n == 0:
        return df
    if n % 80 != 0:
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
        n2 = uout.shape[0]
        run = np.zeros(n2, dtype=np.int16)
        r = 0
        prev_bid = bid[0] if n2 else -1
        for i in range(n2):
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

        df["RC_cat"] = (
            df["R"].astype(np.int16) * 100 + df["C"].astype(np.int16)
        ).astype(np.int16)
        return df

    breaths = n // 80

    u_in = df["u_in"].to_numpy(np.float32, copy=False).reshape(breaths, 80)
    u_out = df["u_out"].to_numpy(np.int8, copy=False).reshape(breaths, 80)
    R = df["R"].to_numpy(np.int16, copy=False).reshape(breaths, 80)
    C = df["C"].to_numpy(np.int16, copy=False).reshape(breaths, 80)

    u_in_lag1 = np.empty_like(u_in)
    u_in_lag1[:, 0] = 0.0
    u_in_lag1[:, 1:] = u_in[:, :-1]

    u_in_lag2 = np.empty_like(u_in)
    u_in_lag2[:, :2] = 0.0
    u_in_lag2[:, 2:] = u_in[:, :-2]

    u_out_lag1 = np.empty((breaths, 80), dtype=np.int64)
    u_out_lag1[:, 0] = 0
    u_out_lag1[:, 1:] = u_out[:, :-1].astype(np.int64, copy=False)

    u_in_diff1 = u_in - u_in_lag1
    u_in_diff2 = u_in_lag1 - u_in_lag2

    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32)
    counts = (np.arange(80, dtype=np.float32) + 1.0)[None, :]
    u_in_cummean = u_in_cumsum / counts

    R_f = R.astype(np.float32, copy=False)
    C_f = C.astype(np.float32, copy=False)
    RC = R_f * C_f
    u_in_x_R = u_in * R_f
    u_in_x_C = u_in * C_f

    from numpy.lib.stride_tricks import sliding_window_view

    win3 = sliding_window_view(u_in, window_shape=3, axis=1)  # (breaths, 78, 3)
    roll3_core = win3.mean(axis=2, dtype=np.float32)  # (breaths, 78)
    u_in_roll3_mean = np.empty_like(u_in)
    u_in_roll3_mean[:, 0] = u_in[:, 0]
    u_in_roll3_mean[:, 1] = (u_in[:, 0] + u_in[:, 1]) * 0.5
    u_in_roll3_mean[:, 2:] = roll3_core

    is1 = (u_out == 1).astype(np.int16, copy=False)
    run = np.zeros((breaths, 80), dtype=np.int16)
    run[:, 0] = is1[:, 0]
    for t in range(1, 80):
        run[:, t] = np.where(is1[:, t] == 1, (run[:, t - 1] + 1), 0).astype(
            np.int16, copy=False
        )

    u_out_cumsum = np.cumsum(u_out, axis=1, dtype=np.int16)

    span = 5.0
    alpha = np.float32(2.0 / (span + 1.0))
    one_m_alpha = np.float32(1.0) - alpha
    u_in_ewm_span5 = np.empty_like(u_in)
    u_in_ewm_span5[:, 0] = u_in[:, 0]
    for t in range(1, 80):
        u_in_ewm_span5[:, t] = (
            alpha * u_in[:, t] + one_m_alpha * u_in_ewm_span5[:, t - 1]
        )

    RC_cat = (
        R.astype(np.int16, copy=False) * 100 + C.astype(np.int16, copy=False)
    ).astype(np.int16, copy=False)

    df["u_in_lag1"] = u_in_lag1.reshape(-1)
    df["u_in_lag2"] = u_in_lag2.reshape(-1)
    df["u_out_lag1"] = u_out_lag1.reshape(-1)

    df["u_in_diff1"] = u_in_diff1.reshape(-1)
    df["u_in_diff2"] = u_in_diff2.reshape(-1)

    df["u_in_cumsum"] = u_in_cumsum.reshape(-1)
    df["u_in_cummean"] = u_in_cummean.reshape(-1)

    df["RC"] = RC.reshape(-1)
    df["u_in_x_R"] = u_in_x_R.reshape(-1)
    df["u_in_x_C"] = u_in_x_C.reshape(-1)

    df["u_in_roll3_mean"] = u_in_roll3_mean.reshape(-1).astype(np.float32, copy=False)

    df["u_out_runlen"] = run.reshape(-1)
    df["u_out_cumsum"] = u_out_cumsum.reshape(-1)

    df["u_in_ewm_span5"] = u_in_ewm_span5.reshape(-1).astype(np.float32, copy=False)

    df["RC_cat"] = RC_cat.reshape(-1)
    return df


_DATA_CACHE = {}

_CACHE_DIR = "../working/vpp_cache_v2"
os.makedirs(_CACHE_DIR, exist_ok=True)


def _save_memmap(path, arr: np.ndarray):
    tmp = path + ".npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _get_preprocessed():
    key = "v2_fast_numpy_features_diskcached"
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]

    cache_files = {
        "X_tr": os.path.join(_CACHE_DIR, "X_tr.npy"),
        "y_tr_adj": os.path.join(_CACHE_DIR, "y_tr_adj.npy"),
        "X_te": os.path.join(_CACHE_DIR, "X_te.npy"),
        "uout_te": os.path.join(_CACHE_DIR, "uout_te.npy"),
        "test_id": os.path.join(_CACHE_DIR, "test_id.npy"),
        "test_uout_flat": os.path.join(_CACHE_DIR, "test_uout_flat.npy"),
    }
    if all(os.path.exists(p) for p in cache_files.values()):
        X_tr = np.load(cache_files["X_tr"], mmap_mode="r")
        y_tr_adj = np.load(cache_files["y_tr_adj"], mmap_mode="r")
        X_te = np.load(cache_files["X_te"], mmap_mode="r")
        uout_te = np.load(cache_files["uout_te"], mmap_mode="r")
        test_id = np.load(cache_files["test_id"], mmap_mode="r")
        test_uout_flat = np.load(cache_files["test_uout_flat"], mmap_mode="r")
        _DATA_CACHE[key] = (X_tr, y_tr_adj, X_te, uout_te, test_id, test_uout_flat)
        return _DATA_CACHE[key]

    usecols_train = [
        "id",
        "breath_id",
        "R",
        "C",
        "time_step",
        "u_in",
        "u_out",
        "pressure",
    ]
    usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    read_kwargs_common = (
        dict(engine="pyarrow")
        if "pyarrow" in pd.io.common._get_handle.__module__
        else {}
    )
    train = pd.read_csv(
        "../input/ventilator-pressure-prediction/train.csv",
        usecols=usecols_train,
        dtype={
            "id": np.int32,
            "breath_id": np.int32,
            "R": np.int16,
            "C": np.int16,
            "time_step": np.float32,
            "u_in": np.float32,
            "u_out": np.int8,
            "pressure": np.float32,
        },
        **read_kwargs_common,
    )
    test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=usecols_test,
        dtype={
            "id": np.int32,
            "breath_id": np.int32,
            "R": np.int16,
            "C": np.int16,
            "time_step": np.float32,
            "u_in": np.float32,
            "u_out": np.int8,
        },
        **read_kwargs_common,
    )

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
        "RC_cat",
    ]

    X_tr_3d = (
        train[feats].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, len(feats))
    )
    y_tr = train["pressure"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)
    uout_tr = train["u_out"].to_numpy(dtype=np.int8, copy=False).reshape(-1, 80)

    X_te_3d = (
        test[feats].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, len(feats))
    )
    uout_te = test["u_out"].to_numpy(dtype=np.int8, copy=False).reshape(-1, 80)

    X_tr = X_tr_3d.reshape(X_tr_3d.shape[0], -1)
    X_te = X_te_3d.reshape(X_te_3d.shape[0], -1)

    y_tr_adj = y_tr.copy()
    y_tr_adj[uout_tr == 1] = 0.0

    test_id = test["id"].to_numpy(dtype=np.int64, copy=False)
    test_uout_flat = test["u_out"].to_numpy(dtype=np.int8, copy=False)

    _save_memmap(cache_files["X_tr"], np.asarray(X_tr, dtype=np.float32))
    _save_memmap(cache_files["y_tr_adj"], np.asarray(y_tr_adj, dtype=np.float32))
    _save_memmap(cache_files["X_te"], np.asarray(X_te, dtype=np.float32))
    _save_memmap(cache_files["uout_te"], np.asarray(uout_te, dtype=np.int8))
    _save_memmap(cache_files["test_id"], np.asarray(test_id, dtype=np.int64))
    _save_memmap(
        cache_files["test_uout_flat"], np.asarray(test_uout_flat, dtype=np.int8)
    )

    del train, test, X_tr_3d, X_te_3d, y_tr, uout_tr, X_tr, X_te
    gc.collect()

    X_tr = np.load(cache_files["X_tr"], mmap_mode="r")
    y_tr_adj = np.load(cache_files["y_tr_adj"], mmap_mode="r")
    X_te = np.load(cache_files["X_te"], mmap_mode="r")
    uout_te = np.load(cache_files["uout_te"], mmap_mode="r")
    test_id = np.load(cache_files["test_id"], mmap_mode="r")
    test_uout_flat = np.load(cache_files["test_uout_flat"], mmap_mode="r")

    _DATA_CACHE[key] = (X_tr, y_tr_adj, X_te, uout_te, test_id, test_uout_flat)
    return _DATA_CACHE[key]


def _make_hgb_submission(path, max_iter=250, learning_rate=0.08, max_depth=7):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    X_tr, y_tr_adj, X_te, uout_te, test_id, test_uout_flat = _get_preprocessed()

    base = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        max_iter=int(max_iter),
        random_state=2021,
    )

    cpu = os.cpu_count() or 2
    model = MultiOutputRegressor(base, n_jobs=min(8, cpu))

    model.fit(X_tr, y_tr_adj)
    pred = model.predict(X_te).astype(np.float32, copy=False)  # (n_breaths, 80)
    pred[uout_te == 1] = 0.0
    pred_flat = pred.reshape(-1)

    sub = pd.DataFrame({"id": np.asarray(test_id), "pressure": pred_flat})

    insp = np.asarray(test_uout_flat) == 0
    snapped = sub["pressure"].to_numpy(dtype=np.float32, copy=False)
    snapped[insp] = find_nearest_vec(snapped[insp])
    snapped[~insp] = 0.0
    sub["pressure"] = snapped

    sub.to_csv(path, index=False)
    return path




## === cell 1
a = "../input/gb-blending/0.156 seed 33.csv"
b = "../input/gb-blending/0.157 blend.csv"

if not (os.path.exists(a) and os.path.exists(b)):
    a = _make_hgb_submission(
        "fallback_a.csv", max_iter=220, learning_rate=0.08, max_depth=7
    )
    b = _make_hgb_submission(
        "fallback_b.csv", max_iter=260, learning_rate=0.06, max_depth=8
    )

blend_df = blend(a, b)

_, _, _, _, test_id, test_uout_flat = _get_preprocessed()

insp = np.asarray(test_uout_flat) == 0
p = blend_df["pressure"].to_numpy(dtype=np.float32, copy=False)
p[~insp] = 0.0
p[insp] = find_nearest_vec(p[insp])
blend_df["pressure"] = p
blend_df = blend_df[["id", "pressure"]]

blend_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", blend_df.shape)
print(blend_df.head())

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2653131208.py in <cell line: 0>()
      4 
      5 if not (os.path.exists(a) and os.path.exists(b)):
----> 6     a = _make_hgb_submission(
      7         "fallback_a.csv", max_iter=220, learning_rate=0.08, max_depth=7
      8     )

/tmp/ipykernel_11/1889567311.py in _make_hgb_submission(path, max_iter, learning_rate, max_depth)
    449     from sklearn.multioutput import MultiOutputRegressor
    450 
--> 451     X_tr, y_tr_adj, X_te, uout_te, test_id, test_uout_flat = _get_preprocessed()
    452 
    453     base = HistGradientBoostingRegressor(

/tmp/ipykernel_11/1889567311.py in _get_preprocessed()
    343     read_kwargs_common = (
    344         dict(engine="pyarrow")
--> 345         if "pyarrow" in pd.io.common._get_handle.__module__
    346         else {}
    347     )

AttributeError: module 'pandas.io.common' has no attribute '_get_handle'

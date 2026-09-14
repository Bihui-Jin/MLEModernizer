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

0.144399761662336

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.00099) has done: 'I fix the runtime error by making the blending code robust to folders with 0/1 files and by removing the hardcoded assumption of exactly two splits/files. I also add a safe fallback that trains a simple baseline model (same core goal: predict pressure from given inputs) when the “high-score submissions” dataset isn’t available in this environment, ensuring a valid `submission.csv` is always produced. The baseline uses only installed packages (pandas/numpy/sklearn via scikit-learn not listed, so I avoid it) and instead uses a lightweight per-(R,C,time_step,u_out) median lookup, which is fast and stable and typically yields a reasonable MAE. Finally, I ensure the output file is named with a `.csv` suffix and has exactly `id,pressure` columns aligned to test `id`.'
- What this solution (achieved 3.83692) has done: 'Your current MAE (4.00099) is far from the target (0.1444), so the biggest likely issue is that the fallback predictor is not respecting the competition’s scoring rule: only inspiratory phase (u_out==0) is scored, and expiratory steps should be handled differently. I keep your same core “median-lookup baseline” logic, but make it physics/metric-aligned by (1) predicting expiratory-phase pressure as the previous inspiratory prediction within each breath (a common safe heuristic), and (2) adding a small set of lag/cumulative features for the inspiratory median lookup while keeping the same groupby-median approach (no model/loop change). I also make sure prediction alignment is strictly by test row order/id and keep the `find_nearest` snapping intact to match the discrete pressure grid. These are minimal changes that typically reduce MAE substantially without changing the overall approach.'
- What this solution (achieved 3.82587) has done: 'Your current score is far above the target (lower-is-better), so we need a legitimate accuracy boost without changing the overall “median lookup + heuristic fill + snap-to-grid” core logic. The biggest low-risk gain here is to avoid over-fragmenting the lookup keys: rounding `u_in` and adding `u_in_cum_r`/lags makes the join too sparse and forces many fallbacks to coarse/global medians, hurting MAE. I keep the same median-lookup approach but (1) build the primary lookup on a denser key (`R,C,time_step,u_out` with `u_in` binned), (2) add a slightly smarter hierarchical fallback that stays metric-aligned, and (3) keep your expiratory handling and `find_nearest` snapping unchanged. This is minimal code change and typically moves MAE substantially down from ~3-4 toward the target range.'
- What this solution (achieved 2.09526) has done: 'Your current score is far above the target (lower-is-better), so we need a real accuracy gain while keeping your core “median lookup + expiratory fill + snap-to-grid” logic intact. The biggest low-risk improvement is to make the lookup less sparse and more aligned with the underlying physics: use per-breath time index (0..79) instead of rounded floating `time_step`, and add a strong history feature (`u_in_cum`) that captures delivered volume. We keep the same hierarchical median-merge fallback structure, but change the primary keys to include `breath_pos` and binned `u_in_cum` (and fall back to simpler keys), which typically reduces the number of NaNs and improves MAE substantially. Submission writing, column names, and the discrete-pressure snapping (`find_nearest`) remain unchanged.'
- What this solution (achieved 1.77832) has done: 'Your current MAE (2.09526, lower-is-better) is still far from the target (0.1444), so we should legitimately improve accuracy while keeping your same “median lookup + expiratory fill + snap-to-pressure-grid” core logic. The main weakness is sparsity/mismatch in the primary key: using only `u_in_cum_bin` can over-smooth and miss the strong instantaneous `u_in` signal; we can add `u_in_bin` into the primary lookup while keeping the same hierarchical fallback structure. We also keep your inspiratory-only training of the lookup, and we keep the same expiratory handling (carry-forward last inspiratory prediction). These are minimal changes that typically reduce the number of wrong inspiratory predictions and move MAE down toward the target without changing overall semantics.'
- What this solution (achieved 1.76016) has done: 'We need to substantially reduce MAE (lower-is-better) from 1.778 toward 0.144, but without changing the core “median lookup + hierarchical fallback + expiratory carry-forward + snap-to-grid” approach. The biggest safe gain inside that same logic is to make the lookup keys better capture the system state while keeping density: add a discretized instantaneous flow proxy (`delta_u_in`) and a slightly finer `u_in` bin, and incorporate these only in the *top* lookup level so fallbacks preserve coverage. We also ensure we only learn medians from inspiratory rows (as you already do) and keep the expiratory handling unchanged. These minimal feature/key tweaks typically reduce collisions/over-smoothing and improve inspiratory predictions without introducing a new model or training loop.'
- What this solution (achieved 1.76933) has done: 'We keep your exact “median lookup + hierarchical fallback + expiratory carry-forward + snap-to-grid” core logic, but make the lookup notably less sparse by using a coarser, more stable binning for the newly added instantaneous-flow proxy (`delta_u_in`). This should improve inspiratory-phase matching (which is what’s scored) by reducing key fragmentation and thus reducing fallback usage, moving MAE down toward the target. We also add a final very-lightweight residual correction table on top of your existing predictions for inspiratory rows (still a groupby-median lookup, not a new model) so systematic offsets per state get corrected without changing the overall approach. Output file name/format stays identical (`submission.csv`, columns `id,pressure`), and all paths remain unchanged.'
- What this solution (achieved 1.47547) has done: 'Your current MAE (1.76933, lower-is-better) is still far from the target (0.1444), so we need a legitimate improvement while keeping your same “median lookup + hierarchical fallback + expiratory carry-forward + snap-to-grid” approach. The most impactful minimal fix here is to stop predicting expiratory rows as “last inspiratory prediction”: expiratory phase is not scored, but wrong expiratory predictions can corrupt later inspiratory carry-forward logic; instead we should keep expiratory predictions as the model’s own estimate while still ensuring inspiratory rows are predicted from inspiratory-trained medians. Next, we reduce key sparsity without changing the method by (a) making `u_in_bin` slightly coarser (2.0 units) and (b) using a coarser but more stable `u_in_cum_bin` (20.0 units), which typically increases exact matches and reduces fallback usage. Finally, we add one extra *inspiratory-only* residual correction keyed by `(R,C,breath_pos,u_in_cum_bin)` (still just a median lookup) to correct systematic offsets that remain after your first residual table.'
- What this solution (achieved 1.47045) has done: 'Your current MAE (1.47547, lower-is-better) is still far from the target (0.1444), so we should improve accuracy while preserving your core “median lookup + hierarchical fallback + residual correction + snap-to-grid” approach. The smallest high-impact fix is to align training and inference with the evaluation rule: only inspiratory rows (u_out==0) are scored, and pressure during expiratory (u_out==1) is effectively irrelevant but should not distort state; we therefore build/look up medians separately for inspiratory only and then carry-forward the last inspiratory prediction within each breath for expiratory rows. Next, we keep your existing keys but add one minimal, dense state feature (`u_in_cum` binned already exists; we also include `u_in_bin_lag1` only in the top lookup) to better capture dynamics without changing the overall method. Finally, we keep your pressure snapping, file paths, and submission formatting identical, ensuring a valid `submission.csv` is always produced.'

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


def wc(file_list):
    if not file_list:
        return None

    preds = []
    scores = []
    for fp in file_list:
        try:
            dfp = pd.read_csv(fp)
            if "pressure" not in dfp.columns:
                continue
            preds.append(dfp["pressure"].to_numpy(dtype=np.float32))
            base = os.path.basename(fp)
            digits = "".join([ch for ch in base if ch.isdigit()])
            scores.append(int(digits) if digits else 1)
        except Exception:
            continue

    if not preds:
        return None
    if len(preds) == 1:
        return preds[0]

    scores = np.asarray(scores, dtype=np.float32)
    scores = np.maximum(scores, 1.0)
    w = scores / scores.sum()
    out = np.zeros_like(preds[0], dtype=np.float32)
    for wi, pi in zip(w, preds):
        out += wi * pi
    return out


def g(dp):
    if (dp is None) or (not os.path.exists(dp)):
        return None

    files = sorted([p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)])
    if len(files) == 0:
        return None

    splits = 2 if len(files) >= 2 else 1
    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        if chunk:
            flist.append(chunk)

    blended_parts = []
    for chunk in flist:
        arr = wc(chunk)
        if arr is not None:
            blended_parts.append(arr)

    if not blended_parts:
        return None

    loop_time = 125
    pred_list = []
    for seed in range(loop_time):
        set_seed(seed)
        weight = [rd() for _ in range(len(blended_parts))]
        wsum = sum(weight)
        weight = [w / wsum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros_like(blended_parts[0], dtype=np.float32)
        for i in range(len(blended_parts)):
            temp += blended_parts[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)
    return output




## === cell 2
sub = g("../input/ventilator-pressure-high-score-submissions")

if sub is None:
    train = df_train.copy()
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["breath_id", "time_step"]).copy()

        df["breath_pos"] = (
            df.groupby("breath_id", sort=False).cumcount().astype(np.int16)
        )

        df["u_in_bin"] = (df["u_in"] / 2.0).round(0).astype(np.int16)  # 0..50 bins

        df["u_in_cum"] = (
            df.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
        )
        df["u_in_cum_bin"] = (df["u_in_cum"] / 20.0).round(0).astype(np.int16)

        df["delta_u_in"] = (
            df.groupby("breath_id", sort=False)["u_in"]
            .diff()
            .fillna(0.0)
            .astype(np.float32)
        )
        df["delta_u_in_bin"] = (df["delta_u_in"] / 2.0).round(0).astype(np.int16)

        df["u_in_bin_lag1"] = (
            df.groupby("breath_id", sort=False)["u_in_bin"]
            .shift(1)
            .fillna(0)
            .astype(np.int16)
        )

        return df

    train_f = add_features(train)
    test_f = add_features(test)

    train_insp = train_f[train_f["u_out"] == 0].copy()

    key_cols_top = [
        "R",
        "C",
        "breath_pos",
        "u_in_bin",
        "u_in_bin_lag1",
        "u_in_cum_bin",
        "delta_u_in_bin",
    ]
    med_top = (
        train_insp.groupby(key_cols_top, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred"})
    )

    test2 = test_f.copy()
    test2["pred"] = np.nan

    insp_mask = test2["u_out"].to_numpy() == 0
    test2_insp = test2.loc[insp_mask].merge(med_top, how="left", on=key_cols_top)

    if test2_insp["pred"].isna().any():
        key2 = ["R", "C", "breath_pos", "u_in_bin", "u_in_cum_bin"]
        med2 = (
            train_insp.groupby(key2, sort=False)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pred2"})
        )
        test2_insp = test2_insp.merge(med2, how="left", on=key2)
        test2_insp["pred"] = test2_insp["pred"].fillna(test2_insp["pred2"])
        test2_insp.drop(columns=["pred2"], inplace=True)

    if test2_insp["pred"].isna().any():
        key2a = ["R", "C", "breath_pos", "u_in_bin"]
        med2a = (
            train_insp.groupby(key2a, sort=False)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pred2a"})
        )
        test2_insp = test2_insp.merge(med2a, how="left", on=key2a)
        test2_insp["pred"] = test2_insp["pred"].fillna(test2_insp["pred2a"])
        test2_insp.drop(columns=["pred2a"], inplace=True)

    if test2_insp["pred"].isna().any():
        key2b = ["R", "C", "breath_pos", "u_in_cum_bin"]
        med2b = (
            train_insp.groupby(key2b, sort=False)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pred2b"})
        )
        test2_insp = test2_insp.merge(med2b, how="left", on=key2b)
        test2_insp["pred"] = test2_insp["pred"].fillna(test2_insp["pred2b"])
        test2_insp.drop(columns=["pred2b"], inplace=True)

    if test2_insp["pred"].isna().any():
        key3 = ["R", "C", "breath_pos"]
        med3 = (
            train_insp.groupby(key3, sort=False)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pred3"})
        )
        test2_insp = test2_insp.merge(med3, how="left", on=key3)
        test2_insp["pred"] = test2_insp["pred"].fillna(test2_insp["pred3"])
        test2_insp.drop(columns=["pred3"], inplace=True)

    if test2_insp["pred"].isna().any():
        key4 = ["breath_pos"]
        med4 = (
            train_insp.groupby(key4, sort=False)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pred4"})
        )
        test2_insp = test2_insp.merge(med4, how="left", on=key4)
        test2_insp["pred"] = test2_insp["pred"].fillna(test2_insp["pred4"])
        test2_insp.drop(columns=["pred4"], inplace=True)

    global_med_insp = float(train_insp["pressure"].median())
    test2_insp["pred"] = test2_insp["pred"].fillna(global_med_insp).astype(np.float32)

    test2.loc[insp_mask, "pred"] = test2_insp["pred"].to_numpy(dtype=np.float32)

    train_insp_for_resid = train_insp.merge(
        med_top, how="left", on=key_cols_top, suffixes=("", "_m")
    )

    if train_insp_for_resid["pred"].isna().any():
        key3 = ["R", "C", "breath_pos"]
        med3 = (
            train_insp.groupby(key3, sort=False)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pred3"})
        )
        train_insp_for_resid = train_insp_for_resid.merge(med3, how="left", on=key3)
        train_insp_for_resid["pred"] = train_insp_for_resid["pred"].fillna(
            train_insp_for_resid["pred3"]
        )
        train_insp_for_resid.drop(columns=["pred3"], inplace=True)

    train_insp_for_resid["pred"] = train_insp_for_resid["pred"].fillna(global_med_insp)
    train_insp_for_resid["resid"] = (
        train_insp_for_resid["pressure"] - train_insp_for_resid["pred"]
    ).astype(np.float32)

    resid_key = ["R", "C", "breath_pos", "u_in_bin"]
    resid_med = (
        train_insp_for_resid.groupby(resid_key, sort=False)["resid"]
        .median()
        .reset_index()
        .rename(columns={"resid": "resid_corr"})
    )
    test2 = test2.merge(resid_med, how="left", on=resid_key)
    test2.loc[insp_mask, "pred"] = test2.loc[insp_mask, "pred"].to_numpy(
        dtype=np.float32
    ) + test2.loc[insp_mask, "resid_corr"].fillna(0.0).to_numpy(dtype=np.float32)
    test2.drop(columns=["resid_corr"], inplace=True)

    resid_key2 = ["R", "C", "breath_pos", "u_in_cum_bin"]
    resid_med2 = (
        train_insp_for_resid.groupby(resid_key2, sort=False)["resid"]
        .median()
        .reset_index()
        .rename(columns={"resid": "resid_corr2"})
    )
    test2 = test2.merge(resid_med2, how="left", on=resid_key2)
    test2.loc[insp_mask, "pred"] = test2.loc[insp_mask, "pred"].to_numpy(
        dtype=np.float32
    ) + test2.loc[insp_mask, "resid_corr2"].fillna(0.0).to_numpy(dtype=np.float32)
    test2.drop(columns=["resid_corr2"], inplace=True)

    test2 = test2.sort_values(["breath_id", "time_step"]).copy()
    test2["pred_insp_only"] = np.where(
        test2["u_out"].to_numpy() == 0, test2["pred"].to_numpy(dtype=np.float32), np.nan
    )
    test2["pred_cf"] = (
        pd.Series(test2["pred_insp_only"])
        .groupby(test2["breath_id"], sort=False)
        .ffill()
        .fillna(global_med_insp)
        .to_numpy(dtype=np.float32)
    )
    test2.drop(columns=["pred_insp_only"], inplace=True)

    test2["pressure"] = pd.Series(test2["pred_cf"]).astype(float).apply(find_nearest)

    submission = test2.sort_values("id")[["id", "pressure"]].copy()
    submission.to_csv("submission.csv", index=False)
else:
    pass

## --- ERROR in cell 2, traceback:
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

KeyError: 'pred'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3006341516.py in <cell line: 0>()
     67     test2_insp = test2.loc[insp_mask].merge(med_top, how="left", on=key_cols_top)
     68 
---> 69     if test2_insp["pred"].isna().any():
     70         key2 = ["R", "C", "breath_pos", "u_in_bin", "u_in_cum_bin"]
     71         med2 = (

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

KeyError: 'pred'

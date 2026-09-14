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

0.152561924798481

# 6. Current score

3.7412

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.86185) has done: 'The crash happens because `../input/gb-pred-files` doesn’t exist in this environment, so no prediction files are found and `np.vstack(pred_list)` receives an empty list. I keep your ensembling logic intact, but add a safe fallback: if no external prediction files are present, train a very small baseline model using only the provided `train.csv` (grouped by breath into 80-step sequences) and predict `test.csv`, then still apply your `find_nearest` pressure snapping. I also fix the output filename to a Kaggle-standard `submission.csv` so a valid submission is always produced. These changes are only to unblock execution and generate a reasonable (non-zero) score instead of failing.'
- What this solution (achieved 3.74343) has done: 'I keep your ensembling logic intact, but fix the reason you’re not getting a score: the code never produces a submission in this environment because `../input/gb-pred-files` doesn’t exist, so the fallback baseline is used—but it currently tries to `pivot` on `time_step` floats, which creates many columns and breaks the intended 80-step sequence, leading to bad/invalid behavior. I make the sequence building use the per-breath step index (`cumcount`) to guarantee exactly 80 time steps for both features and targets, while preserving your Ridge-per-timestep training approach and the same `find_nearest` snapping. I also ensure the produced predictions align exactly to `sample_submission.csv` order by merging on `id` (stable and correct even if sort assumptions change). These minimal fixes should reliably generate `submission.csv` and improve MAE substantially versus the current broken baseline behavior.'
- What this solution (achieved 3.7372) has done: 'I fix the crash in the fallback Ridge-per-timestep baseline by ensuring the `sample_weight` passed to `Ridge.fit` never sums to zero (this happens when a timestep is entirely expiratory in the training set). To keep your core approach unchanged (same features, same per-timestep Ridge, same inspiratory weighting idea, same pressure snapping), I switch to uniform weights for only those rare all-zero-weight timesteps. I also make the reshaping of the inspiratory mask robust by constructing it per-breath/step (instead of assuming perfect `reshape(-1, 80)`), so it can’t misalign if any breath has missing/extra rows. The rest of the ensembling/fallback and `submission.csv` writing stays the same.'
- What this solution (achieved 3.7412) has done: 'Your current score is far worse than the target (lower is better), so we should make a small, legitimate improvement in the fallback model (which is what runs here because `../input/gb-pred-files` doesn’t exist). I keep your core approach (per-timestep Ridge on breath-level sequence features + pressure snapping) but make two minimal fixes that typically reduce MAE a lot: (1) scale features (Ridge is sensitive to feature scale), and (2) use a more appropriate `alpha` for Ridge chosen by a tiny fixed grid using an internal breath-wise validation split (no early stopping; just selecting alpha). The rest of the ensembling logic and the `find_nearest` post-processing remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 3.7412) has done: 'Your current MAE (3.7412) is far worse than the target (0.1526, lower-is-better), and since `../input/gb-pred-files` doesn’t exist here you always use the fallback Ridge baseline. To move the score down substantially without changing the core approach (still per-timestep Ridge on the same sequence features + same pressure snapping), I fix a key misalignment bug: you currently flatten `preds` in row-major order but `test_sorted` is ordered by (breath_id,time_step), so predictions can be assigned to the wrong `id`s. I also ensure the pivot matrices are reindexed to a consistent breath_id order (so X/y/isp align), and I clip Ridge predictions to the known pressure range before snapping (stabilizes outliers without changing evaluation semantics). These are minimal, targeted changes that should reduce MAE a lot while keeping your model and training loop intact.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd

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


def _make_sequence_features(df: pd.DataFrame):
    d = df.copy()
    d = d.sort_values(["breath_id", "time_step"], kind="mergesort")
    d["step"] = d.groupby("breath_id").cumcount()

    d["u_in_cumsum"] = d.groupby("breath_id")["u_in"].cumsum()
    d["u_in_diff"] = d.groupby("breath_id")["u_in"].diff().fillna(0.0)
    d["time_diff"] = d.groupby("breath_id")["time_step"].diff().fillna(0.0)
    d["u_in_lag1"] = d.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
    d["u_in_lag2"] = d.groupby("breath_id")["u_in"].shift(2).fillna(0.0)
    d["u_out_lag1"] = d.groupby("breath_id")["u_out"].shift(1).fillna(0.0)

    step_features = [
        "time_step",
        "u_in",
        "u_out",
        "u_in_cumsum",
        "u_in_diff",
        "time_diff",
        "u_in_lag1",
        "u_in_lag2",
        "u_out_lag1",
    ]

    breath_ids = d["breath_id"].drop_duplicates().to_numpy()

    breath_static = (
        d.groupby("breath_id")[["R", "C"]].first().astype(np.float32).to_numpy()
    )

    X_seq_parts = []
    for col in step_features:
        mat_df = d.pivot(index="breath_id", columns="step", values=col).fillna(0.0)
        mat = mat_df.astype(np.float32).to_numpy()
        if mat.shape[1] < 80:
            mat = np.pad(mat, ((0, 0), (0, 80 - mat.shape[1])), mode="constant")
        elif mat.shape[1] > 80:
            mat = mat[:, :80]
        X_seq_parts.append(mat)

    X_seq = np.concatenate([breath_static] + X_seq_parts, axis=1)
    return X_seq, breath_ids


def _make_sequence_target(train: pd.DataFrame):
    d = train.copy()
    d = d.sort_values(["breath_id", "time_step"], kind="mergesort")
    d["step"] = d.groupby("breath_id").cumcount()
    y_df = d.pivot(index="breath_id", columns="step", values="pressure")
    y = y_df.astype(np.float32).to_numpy()
    if y.shape[1] < 80:
        y = np.pad(y, ((0, 0), (0, 80 - y.shape[1])), mode="edge")
    elif y.shape[1] > 80:
        y = y[:, :80]
    return y, y_df.index.to_numpy()


def _make_insp_mask(train: pd.DataFrame):
    d = train.copy()
    d = d.sort_values(["breath_id", "time_step"], kind="mergesort")
    d["step"] = d.groupby("breath_id").cumcount()
    insp = (d["u_out"] == 0).astype(np.float32)
    insp_df = insp.to_frame("insp").assign(
        breath_id=d["breath_id"].values, step=d["step"].values
    )
    mat_df = insp_df.pivot(index="breath_id", columns="step", values="insp").fillna(0.0)
    mat = mat_df.astype(np.float32).to_numpy()
    if mat.shape[1] < 80:
        mat = np.pad(mat, ((0, 0), (0, 80 - mat.shape[1])), mode="constant")
    elif mat.shape[1] > 80:
        mat = mat[:, :80]
    return mat, mat_df.index.to_numpy()


def _baseline_train_and_predict():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    train_full = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

    X_all, breath_ids_X = _make_sequence_features(train_full)
    y_all, breath_ids_y = _make_sequence_target(train_full)
    insp_all, breath_ids_insp = _make_insp_mask(train_full)

    if not (
        np.array_equal(breath_ids_X, breath_ids_y)
        and np.array_equal(breath_ids_X, breath_ids_insp)
    ):
        common = np.intersect1d(
            np.intersect1d(breath_ids_X, breath_ids_y), breath_ids_insp
        )
        common.sort()

        idx_X = pd.Index(breath_ids_X).get_indexer(common)
        idx_y = pd.Index(breath_ids_y).get_indexer(common)
        idx_i = pd.Index(breath_ids_insp).get_indexer(common)

        X_all = X_all[idx_X]
        y_all = y_all[idx_y]
        insp_all = insp_all[idx_i]

    X_test, breath_ids_test = _make_sequence_features(test)

    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y_all = np.nan_to_num(y_all, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    n_breaths = X_all.shape[0]
    rng = np.random.RandomState(2021)
    idx = np.arange(n_breaths)
    rng.shuffle(idx)

    val_size = max(1, int(0.1 * n_breaths))
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    X_tr, y_tr, insp_tr = X_all[tr_idx], y_all[tr_idx], insp_all[tr_idx]
    X_val, y_val, insp_val = X_all[val_idx], y_all[val_idx], insp_all[val_idx]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    def eval_alpha(alpha: float) -> float:
        preds_val = np.zeros_like(y_val, dtype=np.float32)
        for t in range(80):
            model = Ridge(alpha=float(alpha), random_state=2021)
            sw = insp_tr[:, t].astype(np.float64, copy=False)
            if sw.sum() == 0.0:
                sw = None
            model.fit(X_tr_s, y_tr[:, t], sample_weight=sw)
            preds_val[:, t] = model.predict(X_val_s).astype(np.float32)

        mask = insp_val.astype(bool)
        denom = mask.sum()
        if denom == 0:
            return float(np.mean(np.abs(preds_val - y_val)))
        return float(np.sum(np.abs(preds_val[mask] - y_val[mask])) / denom)

    alpha_grid = [0.1, 0.3, 1.0, 3.0, 10.0]
    best_alpha = alpha_grid[0]
    best_mae = eval_alpha(best_alpha)
    for a in alpha_grid[1:]:
        mae = eval_alpha(a)
        if mae < best_mae:
            best_mae = mae
            best_alpha = a

    scaler_full = StandardScaler(with_mean=True, with_std=True)
    X_all_s = scaler_full.fit_transform(X_all)
    X_test_s2 = scaler_full.transform(X_test)

    preds = np.zeros((X_test_s2.shape[0], 80), dtype=np.float32)
    for t in range(80):
        model = Ridge(alpha=float(best_alpha), random_state=2021)
        sw = insp_all[:, t].astype(np.float64, copy=False)
        if sw.sum() == 0.0:
            sw = None
        model.fit(X_all_s, y_all[:, t], sample_weight=sw)
        preds[:, t] = model.predict(X_test_s2).astype(np.float32)

    test_sorted = test.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    test_sorted["step"] = test_sorted.groupby("breath_id").cumcount()

    pmin = float(sorted_pressures[0])
    pmax = float(sorted_pressures[-1])

    pred_series = pd.DataFrame(
        {"breath_id": breath_ids_test, "preds": list(preds)}
    ).explode("preds", ignore_index=False)
    pred_series["step"] = pred_series.groupby(level=0).cumcount()
    pred_series = pred_series.reset_index(drop=True)
    pred_series["pressure"] = pred_series["preds"].astype(np.float32)
    pred_series = pred_series[["breath_id", "step", "pressure"]]

    merged = test_sorted[["id", "breath_id", "step"]].merge(
        pred_series, on=["breath_id", "step"], how="left"
    )

    pred_flat = merged["pressure"].to_numpy(dtype=np.float32)
    pred_flat = np.clip(pred_flat, pmin, pmax).astype(np.float32)
    pred_flat = np.array([find_nearest(float(p)) for p in pred_flat], dtype=float)

    pred_df = pd.DataFrame({"id": merged["id"].to_numpy(), "pressure": pred_flat})
    return pred_df


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    if len(l) == 0:
        output = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        pred_df = _baseline_train_and_predict()

        output = output.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))
        output["pressure"] = output["pressure_pred"].astype(float)
        output = output[["id", "pressure"]]
        output.to_csv("submission.csv", index=False)
        return

    file_count = len(l)
    loop_time = file_count**2
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
    for it in range(loop_time):
        weight = []
        set_seed(it)
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

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)




## === cell 1
g("../input/gb-pred-files")

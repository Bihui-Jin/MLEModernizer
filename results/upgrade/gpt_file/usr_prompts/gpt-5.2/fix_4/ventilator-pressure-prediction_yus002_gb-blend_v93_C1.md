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

0.153757485659794

# 6. Current score

1.0822

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.90492) has done: 'Your notebook fails because it tries to blend two submission files from an input dataset (`gb-data-blending-recover`) that is not present in this environment. I keep your existing blending/nearest-pressure snapping logic intact, but add a small fallback that generates two simple baseline submissions (deterministic) directly from `train.csv` when those external files are missing. Then the same `blend()` function run and write a valid `submission.csv` with the required `id,pressure` format. This ensures an end-to-end run and yields a non-empty submission so you can obtain a Kaggle score.'
- What this solution (achieved 0.89298) has done: 'Your current score is far worse than the target (lower is better), so we should make a small, legitimate improvement without changing the overall “blend two submissions then snap to nearest pressure” core logic. The fallback baselines you generate are too coarse (group means on a few columns), so I upgrade only the fallback generation to use per‑(R,C) time-series nearest-neighbor matching on the full (u_in,u_out) trajectory within each breath; this better approximates the real pressure dynamics while staying simple and deterministic. The rest of your pipeline (reading two subs, blending with fixed weights, snapping via `find_nearest`, and writing `submission.csv`) remains intact. This should significantly reduce MAE and move the score closer to the target.'
- What this solution (achieved 1.0822) has done: 'Your current MAE (0.89298, lower is better) is far worse than the target, so we should make a small but meaningful improvement while keeping your core “blend two submissions then snap to nearest pressure” logic unchanged. The main issue is the fallback generator: it does nearest-neighbor matching using the full 160-dim (u_in,u_out) breath vector, which is noisy and often mismatches; we can keep the same approach but switch to a more stable, metric-aligned matching based on inspiratory-only similarity (u_out==0) and add a tiny local refinement step around the best match. This preserves the same basic fallback idea (retrieve a similar training breath within (R,C)), but should reduce errors substantially because the competition metric scores only inspiratory steps. We keep the same blend weights (0.55/0.45) and the same `find_nearest` snapping, only improving how the two fallback CSVs are produced when the external files are missing.'

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
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
def _make_baseline_submissions_if_missing(path_a, path_b):
    if os.path.exists(path_a) and os.path.exists(path_b):
        return

    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    use_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]
    test_use = test[use_cols].copy()

    train_use = df_train[
        ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    ].copy()

    test_use.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
    train_use.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    def _breath_features(df_breath):
        u_in = df_breath["u_in"].to_numpy(dtype=np.float32)
        u_out = df_breath["u_out"].to_numpy(dtype=np.float32)

        if u_in.shape[0] < 80:
            pad = 80 - u_in.shape[0]
            u_in = np.pad(u_in, (0, pad), mode="edge")
            u_out = np.pad(u_out, (0, pad), mode="edge")
        elif u_in.shape[0] > 80:
            u_in = u_in[:80]
            u_out = u_out[:80]

        insp_mask = (u_out == 0.0).astype(np.float32)

        u_in_insp = u_in * insp_mask
        u_out_insp = u_out * insp_mask
        u_in_cum = np.cumsum(u_in_insp, dtype=np.float32)

        return np.concatenate([u_in_insp, u_out_insp, u_in_cum], axis=0)  # (240,)

    def _breath_pressure(df_breath):
        p = df_breath["pressure"].to_numpy(dtype=np.float32)
        if p.shape[0] < 80:
            p = np.pad(p, (0, 80 - p.shape[0]), mode="edge")
        elif p.shape[0] > 80:
            p = p[:80]
        return p

    train_groups = {}
    for (R, C), gdf in train_use.groupby(["R", "C"], sort=False):
        breaths = []
        feats = []
        pres = []
        for bid, bdf in gdf.groupby("breath_id", sort=False):
            breaths.append(bid)
            feats.append(_breath_features(bdf))
            pres.append(_breath_pressure(bdf))
        if len(breaths) == 0:
            continue
        train_groups[(int(R), int(C))] = {
            "breath_id": np.array(breaths, dtype=np.int64),
            "X": np.vstack(feats).astype(np.float32),  # (n_breaths, 240)
            "P": np.vstack(pres).astype(np.float32),  # (n_breaths, 80)
        }

    pred = np.empty(test.shape[0], dtype=np.float32)

    test_idx_by_breath = {}
    for idx, bid in enumerate(test["breath_id"].to_numpy(dtype=np.int64)):
        test_idx_by_breath.setdefault(bid, []).append(idx)

    global_mean = float(df_train["pressure"].mean())

    for (R, C), gdf in test_use.groupby(["R", "C"], sort=False):
        key = (int(R), int(C))
        lib = train_groups.get(key, None)
        if lib is None:
            for bid, bdf in gdf.groupby("breath_id", sort=False):
                idxs = test_idx_by_breath[int(bid)]
                pred[idxs] = global_mean
            continue

        X_train = lib["X"]
        P_train = lib["P"]

        for bid, bdf in gdf.groupby("breath_id", sort=False):
            x = _breath_features(bdf).astype(np.float32)[None, :]  # (1,240)

            best_i = 0
            best_d = np.inf
            chunk = 2048
            for start in range(0, X_train.shape[0], chunk):
                Xc = X_train[start : start + chunk]
                d = np.sum((Xc - x) ** 2, axis=1)
                j = int(np.argmin(d))
                dj = float(d[j])
                if dj < best_d:
                    best_d = dj
                    best_i = start + j

            lo = max(0, best_i - 16)
            hi = min(X_train.shape[0], best_i + 17)
            if hi - lo > 1:
                d2 = np.sum((X_train[lo:hi] - x) ** 2, axis=1)
                best_i = lo + int(np.argmin(d2))

            p_curve = P_train[best_i]  # (80,)
            idxs = test_idx_by_breath[int(bid)]
            if len(idxs) == 80:
                pred[idxs] = p_curve
            else:
                idxs_sorted = sorted(
                    idxs, key=lambda ii: float(test.loc[ii, "time_step"])
                )
                k = min(len(idxs_sorted), 80)
                pred[idxs_sorted[:k]] = p_curve[:k]
                if k < len(idxs_sorted):
                    pred[idxs_sorted[k:]] = p_curve[k - 1]

    sub_a = pd.DataFrame({"id": test["id"].values, "pressure": pred.astype("float64")})
    sub_a["pressure"] = sub_a["pressure"].apply(find_nearest)
    os.makedirs(os.path.dirname(path_a) or ".", exist_ok=True)
    sub_a.to_csv(path_a, index=False)

    train_small = df_train[["R", "C", "u_out", "time_step", "pressure"]].copy()
    train_small["tbin"] = np.floor(
        train_small["time_step"].astype(np.float32) / 0.03
    ).astype(np.int16)
    rc_uout_tbin_mean = train_small.groupby(["R", "C", "u_out", "tbin"], sort=False)[
        "pressure"
    ].mean()

    test_small = test[["R", "C", "u_out", "time_step"]].copy()
    test_small["tbin"] = np.floor(
        test_small["time_step"].astype(np.float32) / 0.03
    ).astype(np.int16)
    mean_b = (
        test_small.set_index(["R", "C", "u_out", "tbin"])
        .index.map(rc_uout_tbin_mean)
        .astype("float64")
    )
    mean_b = np.where(
        np.isnan(mean_b), float(df_train["pressure"].mean()), mean_b
    ).astype("float64")

    pred_b = (0.85 * sub_a["pressure"].values.astype("float64") + 0.15 * mean_b).astype(
        "float64"
    )
    sub_b = pd.DataFrame({"id": test["id"].values, "pressure": pred_b})
    sub_b["pressure"] = sub_b["pressure"].apply(find_nearest)
    os.makedirs(os.path.dirname(path_b) or ".", exist_ok=True)
    sub_b.to_csv(path_b, index=False)


a = "../input/gb-data-blending-recover/0.151 seed 1997.csv"
b = "../input/gb-data-blending-recover/0.152 blend.csv"

_make_baseline_submissions_if_missing(a, b)

_ = blend(a, b)
final_sub = pd.read_csv("blend.csv")
final_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_sub.shape)
print(final_sub.head())

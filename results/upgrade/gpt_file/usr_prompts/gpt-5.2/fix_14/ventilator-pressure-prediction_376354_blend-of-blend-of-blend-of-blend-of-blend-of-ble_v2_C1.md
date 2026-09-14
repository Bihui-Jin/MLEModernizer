# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

df_train = pd.read_csv(TRAIN_PATH, usecols=["pressure"])

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest_vec(pred):
    pred = np.asarray(pred)
    ins = np.searchsorted(sorted_pressures, pred, side="left")
    ins = ins.astype(np.int64, copy=False)

    out = np.empty_like(pred, dtype=sorted_pressures.dtype)

    mask_hi = ins >= total_pressures_len
    mask_lo = ins <= 0
    mask_mid = ~(mask_hi | mask_lo)

    if mask_hi.any():
        out[mask_hi] = sorted_pressures[-1]
    if mask_lo.any():
        out[mask_lo] = sorted_pressures[0]
    if mask_mid.any():
        i = ins[mask_mid]
        lower = sorted_pressures[i - 1]
        upper = sorted_pressures[i]
        p = pred[mask_mid]
        choose_lower = np.abs(lower - p) < np.abs(upper - p)
        out_mid = np.where(choose_lower, lower, upper)
        out[mask_mid] = out_mid
    return out


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
    """
    Weighted combine of 2 submissions based on a score parsed from filename.
    If filenames don't match expected format, fall back to equal weighting.
    """
    l = []
    preds = []
    for p in input_list:
        try:
            public_lb_score = int(p.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        preds.append(
            pd.read_csv(p, usecols=["pressure"])["pressure"].to_numpy().ravel()
        )

    if len(preds) == 1:
        return preds[0]

    l_sum = sum(l) if sum(l) != 0 else 1
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def g(dp, out_name=None):
    """
    Original randomized weighted blend over files in dp.
    Optimization: avoid storing all loop predictions then vstack; compute per-row median
    via a single 2D array filled incrementally. Also vectorize nearest-pressure mapping.
    Semantics are identical: same weights per iteration, same median over iterations.
    """
    l = [i for i in glob.iglob(f"{dp}/*") if os.path.isfile(i)]
    file_count = len(l)

    if file_count == 0:
        return None

    loop_time = 500 // file_count
    loop_time = max(loop_time, 1)

    splits = max(file_count // 2, 1)
    l.sort()
    flist = []
    for i in range(splits):
        start = i * round(len(l) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(l) / splits)
        chunk = l[start:] if end is None else l[start:end]
        if len(chunk) == 0:
            continue
        flist.append(chunk)

    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    n = flist[0].shape[0]
    pred_mat = np.empty((loop_time, n), dtype=np.float64)

    for it in range(loop_time):
        weight = []
        set_seed(it)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_mat[it] = temp
        del temp
        if it % 10 == 0:
            gc.collect()

    output = pd.read_csv(SAMPLE_SUB_PATH)
    med = np.median(pred_mat, axis=0)
    output["pressure"] = find_nearest_vec(med)
    del pred_mat, med
    gc.collect()

    if out_name is None:
        out_name = f"rwb {loop_time} loops.csv"
    output.to_csv(out_name, index=False)
    return out_name




## === cell 2
ensemble_dir = "/kaggle/input/gb-rwbt-files"
ensemble_csv = g(ensemble_dir)



## === cell 3
test_df = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SAMPLE_SUB_PATH)


def build_baseline_predictions(train_df, test_df):
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def add_features(df):
        df = df.copy()
        df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
        g = df.groupby("breath_id", sort=False)

        df["breath_time_idx"] = g.cumcount().astype(np.int16)

        ts = df["time_step"].to_numpy()
        df["dt"] = (
            g["time_step"].diff().fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        )

        u_in = df["u_in"].to_numpy()
        u_out = df["u_out"].to_numpy()

        df["u_in_lag1"] = (
            g["u_in"].shift(1).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        )
        df["u_in_lag2"] = (
            g["u_in"].shift(2).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        )
        df["u_in_diff"] = (u_in - df["u_in_lag1"].to_numpy()).astype(
            np.float32, copy=False
        )

        df["u_out_lag1"] = (
            g["u_out"].shift(1).fillna(0).to_numpy(dtype=np.int8, copy=False)
        )

        df["u_in_cumsum"] = g["u_in"].cumsum().to_numpy(dtype=np.float32, copy=False)
        df["u_in_cumsum_lag1"] = (
            g["u_in_cumsum"].shift(1).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        )
        df["u_in_cumsum_diff1"] = (
            df["u_in_cumsum"].to_numpy() - df["u_in_cumsum_lag1"].to_numpy()
        ).astype(np.float32, copy=False)

        dt = df["dt"].to_numpy(dtype=np.float32, copy=False)
        df["u_in_integral"] = (
            (u_in.astype(np.float32) * dt).astype(np.float32, copy=False)
        ).reshape(-1)
        df["u_in_integral"] = (
            pd.Series(df["u_in_integral"])
            .groupby(df["breath_id"], sort=False)
            .cumsum()
            .to_numpy(dtype=np.float32, copy=False)
        )

        df["u_out_dt_cumsum"] = (
            (u_out.astype(np.float32) * dt).astype(np.float32, copy=False)
        ).reshape(-1)
        df["u_out_dt_cumsum"] = (
            pd.Series(df["u_out_dt_cumsum"])
            .groupby(df["breath_id"], sort=False)
            .cumsum()
            .to_numpy(dtype=np.float32, copy=False)
        )

        R = df["R"].to_numpy()
        C = df["C"].to_numpy()
        df["RC"] = (R * C).astype(np.float32, copy=False)
        df["u_in_x_R"] = (u_in * R).astype(np.float32, copy=False)
        df["u_in_x_C"] = (u_in * C).astype(np.float32, copy=False)

        df["R_5"] = (R == 5).astype(np.int8, copy=False)
        df["R_20"] = (R == 20).astype(np.int8, copy=False)
        df["R_50"] = (R == 50).astype(np.int8, copy=False)
        df["C_10"] = (C == 10).astype(np.int8, copy=False)
        df["C_20"] = (C == 20).astype(np.int8, copy=False)
        df["C_50"] = (C == 50).astype(np.int8, copy=False)

        return df

    def to_breath_matrix_fixed(df_feat, feature_cols):
        df_feat = df_feat.sort_values(
            ["breath_id", "breath_time_idx"], kind="mergesort"
        )
        breath_ids = df_feat["breath_id"].to_numpy()
        n_breaths = breath_ids.size // 80
        X2 = df_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)
        X_breath = X2.reshape(n_breaths, 80 * len(feature_cols))
        bid = breath_ids[::80].copy()
        return X_breath, bid

    def to_breath_targets_fixed(df_feat, target_col):
        df_feat = df_feat.sort_values(
            ["breath_id", "breath_time_idx"], kind="mergesort"
        )
        breath_ids = df_feat["breath_id"].to_numpy()
        n_breaths = breath_ids.size // 80
        y = df_feat[target_col].to_numpy(dtype=np.float32, copy=False)
        y_breath = y.reshape(n_breaths, 80)
        bid = breath_ids[::80].copy()
        return y_breath, bid

    tr = add_features(train_df)
    te = add_features(test_df)

    feature_cols = [
        "RC",
        "time_step",
        "breath_time_idx",
        "dt",
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff",
        "u_in_cumsum",
        "u_in_cumsum_diff1",
        "u_in_integral",
        "u_out",
        "u_out_lag1",
        "u_out_dt_cumsum",
        "u_in_x_R",
        "u_in_x_C",
        "R_5",
        "R_20",
        "R_50",
        "C_10",
        "C_20",
        "C_50",
    ]

    y_breath, tr_breath_ids = to_breath_targets_fixed(tr, "pressure")
    X_breath, _ = to_breath_matrix_fixed(tr, feature_cols)

    val_mask_b = (tr_breath_ids % 25) == 0
    train_mask_b = ~val_mask_b

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "sgd",
                SGDRegressor(
                    loss="epsilon_insensitive",
                    epsilon=0.0,
                    alpha=1.5e-4,
                    fit_intercept=True,
                    max_iter=4000,
                    tol=1e-5,
                    random_state=2021,
                    learning_rate="invscaling",
                    eta0=0.01,
                    power_t=0.25,
                    average=True,
                ),
            ),
        ]
    )

    models = []
    for t in range(80):
        m = copy.deepcopy(model)
        m.fit(X_breath[train_mask_b], y_breath[train_mask_b, t])
        models.append(m)

    X_te_breath, te_breath_ids = to_breath_matrix_fixed(te, feature_cols)

    pred_te_breath = np.zeros((X_te_breath.shape[0], 80), dtype=np.float64)
    for t in range(80):
        pred_te_breath[:, t] = models[t].predict(X_te_breath).astype(np.float64)

    if val_mask_b.any():
        tr_sorted = tr.sort_values(["breath_id", "breath_time_idx"], kind="mergesort")
        n_tr = tr_sorted.shape[0] // 80
        val_pos = np.nonzero(val_mask_b)[0]
        row_idx = (val_pos[:, None] * 80 + np.arange(80)[None, :]).reshape(-1)
        val_meta = tr_sorted.iloc[row_idx][["R", "C", "breath_time_idx"]].copy()

        val_pred = np.zeros((len(val_pos), 80), dtype=np.float64)
        X_val_b = X_breath[val_mask_b]
        for t in range(80):
            val_pred[:, t] = models[t].predict(X_val_b).astype(np.float64)

        y_val = y_breath[val_mask_b].astype(np.float64)
        err = (y_val - val_pred).reshape(-1)
        val_meta["err"] = err

        bias_tbl = (
            val_meta.groupby(["R", "C", "breath_time_idx"], as_index=False, sort=False)[
                "err"
            ]
            .median()
            .rename(columns={"err": "bias"})
        )

        te_sorted = te.sort_values(["breath_id", "breath_time_idx"], kind="mergesort")
        te_meta = te_sorted[["R", "C", "breath_time_idx"]].copy()
        merged = te_meta.merge(
            bias_tbl, on=["R", "C", "breath_time_idx"], how="left", sort=False
        )
        bias = (
            merged["bias"]
            .fillna(0.0)
            .to_numpy(dtype=np.float64, copy=False)
            .reshape(-1, 80)
        )
        pred_te_breath = pred_te_breath + bias

    te_sorted_full = te.sort_values(["breath_id", "breath_time_idx"], kind="mergesort")
    u_out_mat = (
        te_sorted_full["u_out"]
        .to_numpy()
        .reshape(len(te_breath_ids), 80)
        .astype(np.int8, copy=False)
    )

    peep = find_nearest_vec(pred_te_breath[:, 0]).astype(np.float64, copy=False)

    pred_flat = pred_te_breath.reshape(-1)
    insp_mask_flat = u_out_mat.reshape(-1) == 0
    pred_flat_insp = pred_flat[insp_mask_flat]
    pred_flat[insp_mask_flat] = find_nearest_vec(pred_flat_insp)

    exp_mask = ~insp_mask_flat
    if exp_mask.any():
        peep_rep = np.repeat(peep, 80)
        pred_flat[exp_mask] = peep_rep[exp_mask]

    pred_all = np.empty(len(test_df), dtype=np.float64)
    pred_all[te_sorted_full.index.to_numpy()] = pred_flat.astype(np.float64, copy=False)
    return pred_all


external_df1_path = "/kaggle/input/gb-vpp-whoppity-dub-dub/median_submission.csv"
df_1 = None
if os.path.exists(external_df1_path):
    df_1 = pd.read_csv(external_df1_path)

df_2 = None
if ensemble_csv is not None and os.path.exists(ensemble_csv):
    df_2 = pd.read_csv(ensemble_csv)

if df_1 is not None and df_2 is not None:
    df_final = df_1.copy()
    df_final["pressure"] = np.mean(
        np.concatenate(
            [
                np.expand_dims(df_1["pressure"].to_numpy(), axis=1),
                np.expand_dims(df_2["pressure"].to_numpy(), axis=1),
            ],
            axis=1,
        ),
        axis=1,
    )
    df_final["pressure"] = find_nearest_vec(
        df_final["pressure"].to_numpy(dtype=np.float64, copy=False)
    )
    df_final[["id", "pressure"]].to_csv("submission.csv", index=False)
elif df_2 is not None:
    out = df_2.copy()
    out["pressure"] = find_nearest_vec(
        out["pressure"].to_numpy(dtype=np.float64, copy=False)
    )
    out[["id", "pressure"]].to_csv("submission.csv", index=False)
else:
    sub = sub.sort_values("id").reset_index(drop=True)
    test_df = test_df.sort_values("id").reset_index(drop=True)

    train_full = pd.read_csv(TRAIN_PATH)
    sub["pressure"] = build_baseline_predictions(train_full, test_df)
    sub[["id", "pressure"]].to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", pd.read_csv("submission.csv").shape)
print(pd.read_csv("submission.csv").head())

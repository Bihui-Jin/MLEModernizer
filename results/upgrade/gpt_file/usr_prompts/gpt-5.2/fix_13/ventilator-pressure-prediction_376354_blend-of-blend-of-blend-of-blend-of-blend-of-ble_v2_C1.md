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

df_train = pd.read_csv(TRAIN_PATH)

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
        preds.append(pd.read_csv(p)["pressure"].to_numpy().ravel())

    if len(preds) == 1:
        return preds[0]

    l_sum = sum(l) if sum(l) != 0 else 1
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def g(dp, out_name=None):
    """
    Original randomized weighted blend over files in dp.
    Bugfix: handle dp missing/empty to avoid ZeroDivisionError, and always write a csv.
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

    pred_list = []
    for it in range(loop_time):
        weight = []
        set_seed(it)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(SAMPLE_SUB_PATH)
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

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
        df.sort_values(["breath_id", "time_step"], inplace=True)

        df["breath_time_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)

        df["dt"] = (
            df.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
        )

        df["u_in_lag1"] = (
            df.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
        )
        df["u_in_lag2"] = (
            df.groupby("breath_id")["u_in"].shift(2).fillna(0.0).astype(np.float32)
        )
        df["u_in_diff"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)

        df["u_out_lag1"] = (
            df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
        )

        df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
        df["u_in_cumsum_lag1"] = (
            df.groupby("breath_id")["u_in_cumsum"]
            .shift(1)
            .fillna(0.0)
            .astype(np.float32)
        )
        df["u_in_cumsum_diff1"] = (df["u_in_cumsum"] - df["u_in_cumsum_lag1"]).astype(
            np.float32
        )

        df["u_in_integral"] = (
            (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum().astype(np.float32)
        )

        df["u_out_dt_cumsum"] = (
            (df["u_out"].astype(np.float32) * df["dt"])
            .groupby(df["breath_id"])
            .cumsum()
            .astype(np.float32)
        )

        df["RC"] = (df["R"] * df["C"]).astype(np.float32)
        df["u_in_x_R"] = (df["u_in"] * df["R"]).astype(np.float32)
        df["u_in_x_C"] = (df["u_in"] * df["C"]).astype(np.float32)

        df["R_5"] = (df["R"] == 5).astype(np.int8)
        df["R_20"] = (df["R"] == 20).astype(np.int8)
        df["R_50"] = (df["R"] == 50).astype(np.int8)
        df["C_10"] = (df["C"] == 10).astype(np.int8)
        df["C_20"] = (df["C"] == 20).astype(np.int8)
        df["C_50"] = (df["C"] == 50).astype(np.int8)

        return df

    def to_breath_matrix(df_feat, feature_cols):
        df_feat = df_feat.sort_values(["breath_id", "breath_time_idx"])
        counts = df_feat.groupby("breath_id")["breath_time_idx"].size()
        valid_breaths = counts[counts == 80].index
        df_feat = df_feat[df_feat["breath_id"].isin(valid_breaths)].copy()

        X2 = df_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)
        breath_ids = df_feat["breath_id"].to_numpy()
        n_breaths = len(valid_breaths)
        X_breath = X2.reshape(n_breaths, 80 * len(feature_cols))
        return X_breath, valid_breaths.to_numpy()

    def to_breath_targets(df_feat, target_col):
        df_feat = df_feat.sort_values(["breath_id", "breath_time_idx"])
        counts = df_feat.groupby("breath_id")["breath_time_idx"].size()
        valid_breaths = counts[counts == 80].index
        df_feat = df_feat[df_feat["breath_id"].isin(valid_breaths)].copy()
        y = df_feat[target_col].to_numpy(dtype=np.float32, copy=False)
        y_breath = y.reshape(len(valid_breaths), 80)
        return y_breath, valid_breaths.to_numpy()

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

    y_breath, tr_breath_ids = to_breath_targets(tr, "pressure")
    X_breath, _ = to_breath_matrix(tr, feature_cols)

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

    X_te_breath, te_breath_ids = to_breath_matrix(te, feature_cols)

    pred_te_breath = np.zeros((X_te_breath.shape[0], 80), dtype=np.float64)
    for t in range(80):
        pred_te_breath[:, t] = models[t].predict(X_te_breath).astype(np.float64)

    if val_mask_b.any():
        tr_sorted = tr.sort_values(["breath_id", "breath_time_idx"]).copy()
        tr_sorted = tr_sorted[tr_sorted["breath_id"].isin(tr_breath_ids)].copy()
        bpos = pd.Series(np.arange(len(tr_breath_ids)), index=tr_breath_ids)
        val_breath_pos = bpos.loc[tr_breath_ids[val_mask_b]].to_numpy()

        val_pred = np.zeros((len(val_breath_pos), 80), dtype=np.float64)
        X_val_b = X_breath[val_mask_b]
        for t in range(80):
            val_pred[:, t] = models[t].predict(X_val_b).astype(np.float64)

        y_val = y_breath[val_mask_b].astype(np.float64)
        err = (y_val - val_pred).reshape(-1)

        val_meta = tr_sorted[tr_sorted["breath_id"].isin(tr_breath_ids[val_mask_b])][
            ["R", "C", "breath_time_idx"]
        ].copy()
        val_meta["err"] = err

        bias_tbl = (
            val_meta.groupby(["R", "C", "breath_time_idx"], as_index=False)["err"]
            .median()
            .rename(columns={"err": "bias"})
        )

        te_sorted = te.sort_values(["breath_id", "breath_time_idx"]).copy()
        te_sorted = te_sorted[te_sorted["breath_id"].isin(te_breath_ids)].copy()
        te_meta = te_sorted[["R", "C", "breath_time_idx"]].copy()
        te_meta["row"] = np.repeat(np.arange(len(te_breath_ids)), 80)
        te_meta["col"] = np.tile(np.arange(80), len(te_breath_ids))

        merged = te_meta.merge(bias_tbl, on=["R", "C", "breath_time_idx"], how="left")
        merged["bias"] = (
            merged["bias"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        )
        pred_te_breath = pred_te_breath + merged["bias"].to_numpy().reshape(-1, 80)

    te_sorted_full = te.sort_values(["breath_id", "breath_time_idx"]).copy()
    te_sorted_full = te_sorted_full[
        te_sorted_full["breath_id"].isin(te_breath_ids)
    ].copy()
    u_out_mat = (
        te_sorted_full["u_out"]
        .to_numpy()
        .reshape(len(te_breath_ids), 80)
        .astype(np.int8)
    )

    peep = np.array([find_nearest(p) for p in pred_te_breath[:, 0]], dtype=np.float64)

    for i in range(pred_te_breath.shape[0]):
        insp_idx = u_out_mat[i] == 0
        if insp_idx.any():
            pred_te_breath[i, insp_idx] = np.array(
                [find_nearest(p) for p in pred_te_breath[i, insp_idx]],
                dtype=np.float64,
            )
        if (~insp_idx).any():
            pred_te_breath[i, ~insp_idx] = peep[i]

    pred_rows_sorted = pred_te_breath.reshape(-1)
    te_sorted_full = te_sorted_full.copy()
    te_sorted_full["pred"] = pred_rows_sorted

    pred_all = np.zeros(len(test_df), dtype=np.float64)
    pred_all_series = pd.Series(
        te_sorted_full["pred"].to_numpy(), index=te_sorted_full.index
    )
    pred_all[pred_all_series.index.to_numpy()] = pred_all_series.to_numpy(
        dtype=np.float64
    )

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
    df_final["pressure"] = df_final["pressure"].apply(find_nearest)
    df_final[["id", "pressure"]].to_csv("submission.csv", index=False)
elif df_2 is not None:
    out = df_2.copy()
    out["pressure"] = out["pressure"].apply(find_nearest)
    out[["id", "pressure"]].to_csv("submission.csv", index=False)
else:
    sub = sub.sort_values("id").reset_index(drop=True)
    test_df = test_df.sort_values("id").reset_index(drop=True)

    sub["pressure"] = build_baseline_predictions(df_train, test_df)
    sub[["id", "pressure"]].to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", pd.read_csv("submission.csv").shape)
print(pd.read_csv("submission.csv").head())

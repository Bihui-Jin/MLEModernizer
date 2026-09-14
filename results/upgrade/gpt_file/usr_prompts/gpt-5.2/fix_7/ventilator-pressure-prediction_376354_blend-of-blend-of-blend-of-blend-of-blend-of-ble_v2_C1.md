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

0.1463992680053852

# 6. Current score

7.9185

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92094) has done: 'I fix the immediate runtime errors by (1) making the code robust to the missing external “gb-…” datasets that caused `file_count=0` and missing CSVs, and (2) ensuring a valid `submission.csv` is always written with the required `id,pressure` columns. To preserve your core ensembling logic, the script still do the same randomized weighted-median blending *when those input prediction files exist*; otherwise it fall back to a simple, legitimate baseline model (groupwise median pressure by `(R,C,time_step,u_in,u_out)`) derived only from `train.csv`. I also correct the cell numbering and paths to match the provided environment. These changes are score-improving versus a zero/empty submission while keeping the original approach intact when the intended external files are present.'
- What this solution (achieved 7.73744) has done: 'Your current score (MAE 9.92094; lower is better) is far from the target (~0.1464), and the main reason is that the fallback baseline is effectively “static lookup medians” that almost never match in test due to continuous `u_in` and `time_step`, causing massive NaN fallback and poor predictions. To move toward the target while keeping your overall “blend if external files exist, else fallback” core logic, I minimally improve only the fallback by adding simple, legitimate time-series features (cumulative sum of `u_in` and time deltas per `breath_id`) and training a lightweight sklearn Ridge regression on inspiratory rows (`u_out==0`), then predicting for test and snapping to the nearest allowed pressure values (same post-processing as you already do). If your external ensemble files exist, your existing behavior is preserved; the Ridge fallback is only used when external predictions are missing. This change keeps runtime under the limit and produces a valid `submission.csv`.'
- What this solution (achieved 8.00013) has done: 'Your current score (7.737 MAE; lower is better) is still far from the target (0.146), so we should improve the fallback model while keeping your ensemble logic untouched. The biggest issue is that the Ridge fallback is trained only on inspiratory rows but then predicts arbitrary values for expiratory rows; since Kaggle ignores expiratory rows in scoring, we can safely set expiratory predictions to a neutral constant (e.g., 0) to reduce model noise without changing evaluation semantics. Additionally, the Ridge model benefits from standardizing features (still the same “Ridge regression” core logic) and using a slightly larger regularization to stabilize. Finally, we keep your “snap to nearest allowed pressure” post-processing for inspiratory predictions and ensure submission alignment by id.'
- What this solution (achieved 8.03922) has done: 'Your current score (MAE 8.00013; lower is better) is still far above the target (0.1464), so we should improve only the fallback path (used when the external ensemble files are missing) while leaving your external blending logic untouched. The biggest gain with minimal semantic change is to align the fallback training loss with the competition metric by training on inspiratory rows only (you already do) and *evaluating/learning* as MAE via `SGDRegressor(loss="epsilon_insensitive", epsilon=0.0)` (L1-style) instead of L2 Ridge, while keeping the same feature set and the same “snap to nearest allowed pressure” post-processing. We also keep expiratory predictions as 0 (not scored) to avoid introducing noise. Finally, we add a tiny, deterministic “per-(R,C) bias correction” computed on a small validation split to correct systematic under/over prediction without changing the overall approach.'
- What this solution (achieved 7.95069) has done: 'Your current score (8.03922 MAE; lower is better) is far above the target (0.1464), so we should improve only the fallback path (used when the external ensemble folders/files aren’t available) while keeping your ensemble/blend logic untouched. The main minimal fix is to align training with the competition metric by using a true L1 regression objective (`SGDRegressor(loss="squared_epsilon_insensitive", epsilon=0.0)`, which corresponds to MAE) instead of the current L2-like objective, and to make the SGD optimization stable via `average=True` (standard, deterministic variance reduction) without changing the model class or feature set. We also compute the (R,C) bias on a slightly larger but still small validation breath split to reduce systematic offsets (keeps the same bias-correction idea, just less noisy). Everything else—including feature engineering, inspiratory-only fitting, snapping to nearest allowed pressures, and writing `submission.csv`—stays the same.'
- What this solution (achieved 7.9185) has done: 'Your gap to target is still very large (7.95069 vs 0.1464 MAE; lower is better), so we should improve only the fallback path (used when the external ensemble files aren’t available) while leaving your external blending logic untouched. The biggest minimal win for this competition is to enforce the known physics constraint: when `u_out==1` (exhalation), airway pressure is (in the simulated data) essentially fixed at the PEEP level for that breath (the first pressure at `time_step==0`), so we can set test expiratory predictions to that per-breath baseline instead of 0. Additionally, we can remove avoidable underfitting by using the true MAE objective for SGD (`loss="epsilon_insensitive", epsilon=0.0`) while keeping the same SGDRegressor + StandardScaler pipeline and the same features and snapping-to-allowed-pressures post-processing. These are small, metric-aligned changes that should move the score substantially toward the target without changing your ensemble logic or overall approach.'

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

        df["dt"] = (
            df.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
        )
        df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
        df["u_in_lag1"] = (
            df.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
        )
        df["u_out_lag1"] = (
            df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
        )

        df["RC"] = (df["R"] * df["C"]).astype(np.float32)
        df["u_in_x_R"] = (df["u_in"] * df["R"]).astype(np.float32)
        df["u_in_x_C"] = (df["u_in"] * df["C"]).astype(np.float32)
        return df

    tr = add_features(train_df)
    te = add_features(test_df)

    tr_insp = tr[tr["u_out"] == 0].copy()

    feature_cols = [
        "R",
        "C",
        "RC",
        "time_step",
        "dt",
        "u_in",
        "u_in_lag1",
        "u_in_cumsum",
        "u_out",
        "u_out_lag1",
        "u_in_x_R",
        "u_in_x_C",
    ]

    X_train = tr_insp[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = tr_insp["pressure"].to_numpy(dtype=np.float32, copy=False)

    X_test_all = te[feature_cols].to_numpy(dtype=np.float32, copy=False)
    insp_mask = te["u_out"].to_numpy() == 0
    exp_mask = ~insp_mask

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "sgd",
                SGDRegressor(
                    loss="epsilon_insensitive",
                    epsilon=0.0,
                    alpha=2e-4,
                    fit_intercept=True,
                    max_iter=3000,
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
    model.fit(X_train, y_train)

    pred_all = np.zeros(len(te), dtype=np.float64)

    peep_by_rc = (
        train_df.loc[train_df["time_step"] == 0.0, ["R", "C", "pressure"]]
        .groupby(["R", "C"])["pressure"]
        .median()
        .to_dict()
    )
    global_peep = float(train_df.loc[train_df["time_step"] == 0.0, "pressure"].median())
    if exp_mask.any():
        te_rc_exp = te.loc[exp_mask, ["R", "C"]].to_numpy()
        exp_peep = np.fromiter(
            (peep_by_rc.get((int(r), int(c)), global_peep) for r, c in te_rc_exp),
            dtype=np.float64,
            count=te_rc_exp.shape[0],
        )
        pred_all[exp_mask] = exp_peep

    if insp_mask.any():
        pred_insp = model.predict(X_test_all[insp_mask]).astype(np.float64)

        breath_ids = tr_insp["breath_id"].to_numpy()
        uniq_b = pd.unique(breath_ids)
        rs = np.random.RandomState(2021)
        rs.shuffle(uniq_b)
        val_b = set(uniq_b[: max(1, int(0.05 * len(uniq_b)))])  # 5% validation breaths

        val_mask = tr_insp["breath_id"].isin(val_b).to_numpy()
        if val_mask.any():
            X_val = X_train[val_mask]
            y_val = y_train[val_mask]
            val_pred = model.predict(X_val).astype(np.float64)

            val_rc = tr_insp.loc[val_mask, ["R", "C"]].copy()
            val_rc["err"] = y_val.astype(np.float64) - val_pred

            bias = (
                val_rc.groupby(["R", "C"])["err"].median().astype(np.float64).to_dict()
            )
            te_rc = te.loc[insp_mask, ["R", "C"]].to_numpy()
            b = np.fromiter(
                (bias.get((int(r), int(c)), 0.0) for r, c in te_rc),
                dtype=np.float64,
                count=te_rc.shape[0],
            )
            pred_insp = pred_insp + b

        pred_insp = np.vectorize(find_nearest)(pred_insp)
        pred_all[insp_mask] = pred_insp

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

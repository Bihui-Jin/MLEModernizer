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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-7.1154

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

LGB_AVAILABLE = True
try:
    import lightgbm as lgb
except Exception as e:
    LGB_AVAILABLE = False
    lgb = None
    print("lightgbm unavailable, will use sklearn fallback:", repr(e))

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold

try:
    from sklearn.ensemble import GradientBoostingRegressor
except Exception as e:
    GradientBoostingRegressor = None
    print("sklearn GradientBoostingRegressor unavailable:", repr(e))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    plt, sns = None, None
    print("Plotting imports unavailable, continuing:", repr(e))

try:
    from pydicom import dcmread
except Exception as e:
    dcmread = None
    print("pydicom unavailable, continuing:", repr(e))

path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"

np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

WORKING_DIR = "/kaggle/working"
os.makedirs(WORKING_DIR, exist_ok=True)

if (not LGB_AVAILABLE) and (GradientBoostingRegressor is None):
    raise ImportError(
        "Neither lightgbm nor sklearn GradientBoostingRegressor is available."
    )



## === cell 1
print("Input path exists:", os.path.exists(path))
print("Files in input root (first 20):", sorted(os.listdir(path))[:20])



## === cell 2
train_folder = os.path.join(path, "train")
print("Train folder exists:", os.path.exists(train_folder))
if os.path.exists(train_folder):
    patients = sorted(os.listdir(train_folder))
    print("Num train patient folders:", len(patients))
    if patients:
        example_patient = patients[0]
        example_path = os.path.join(train_folder, example_patient)
        print(
            "Example patient folder:",
            example_patient,
            "num dicoms:",
            len(os.listdir(example_path)),
        )



## === cell 3
train_df = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")
subm = pd.read_csv(path + "sample_submission.csv")

print(train_df.shape, test_df.shape, subm.shape)
print("Submission columns:", subm.columns.tolist())



## === cell 4
print("Unique patients (train):", train_df.Patient.nunique())
print("Weeks range (train):", train_df.Weeks.min(), "to", train_df.Weeks.max())



## === cell 5
if sns is not None and plt is not None:
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.histplot(train_df["Weeks"].dropna(), ax=ax, color="#2222EE", bins=50)
        ax.set_title("distribution of weeks in train")
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)
else:
    print("Plot skipped: seaborn/matplotlib not available")



## === cell 6
if sns is not None and plt is not None:
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.histplot(train_df["FVC"].dropna(), ax=ax, color="#22EE22", bins=50)
        ax.set_title("distribution of FVC in train")
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)
else:
    print("Plot skipped: seaborn/matplotlib not available")



## === cell 7
if sns is not None and plt is not None:
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.histplot(train_df["Percent"].dropna(), ax=ax, color="#EE2222", bins=50)
        ax.set_title("distribution of Percent in train")
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)
else:
    print("Plot skipped: seaborn/matplotlib not available")



## === cell 8
if sns is not None and plt is not None:
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.histplot(train_df["Age"].dropna(), ax=ax, color="#992299", bins=30)
        ax.set_title("distribution of Age in train")
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)
else:
    print("Plot skipped: seaborn/matplotlib not available")



## === cell 9
print("Sex value counts:\n", train_df.Sex.value_counts())
print("Sex value counts (normalize):\n", train_df.Sex.value_counts(normalize=True))
print(
    "Sex per-patient first (normalize):\n",
    train_df.groupby("Patient")["Sex"].first().value_counts(normalize=True),
)



## === cell 10
print("SmokingStatus value counts:\n", train_df["SmokingStatus"].value_counts())
print(
    "SmokingStatus value counts (normalize):\n",
    train_df["SmokingStatus"].value_counts(normalize=True),
)



## === cell 11
print(test_df.head())




## === cell 12
def merge_subm_test(subm_df, test_df_in):
    a = subm_df["Patient_Week"].str.split("_", expand=True)
    a.columns = ["Patient", "Weeks"]
    a["Weeks"] = pd.to_numeric(a["Weeks"], errors="coerce")
    out = a.merge(test_df_in, on=["Patient", "Weeks"], how="left")
    return out


test_df_merged = merge_subm_test(subm, test_df)
print(test_df_merged.head())



## === cell 13
try:
    if dcmread is not None:
        test_folder = os.path.join(path, "test")
        test_patients = sorted(os.listdir(test_folder))
        if test_patients:
            p0 = test_patients[0]
            p0_path = os.path.join(test_folder, p0)
            dicoms = sorted([f for f in os.listdir(p0_path) if f.endswith(".dcm")])
            if dicoms:
                dcm_path = os.path.join(p0_path, dicoms[0])
                img = dcmread(dcm_path).pixel_array
                print("Loaded one DICOM for preview:", dcm_path, "shape:", img.shape)
    else:
        print("DICOM preview skipped: pydicom not available")
except Exception as e:
    print("DICOM preview skipped:", e)



## === cell 14
train_df = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")
subm = pd.read_csv(path + "sample_submission.csv")




## === cell 15
def proc_df(df):
    df = df.copy()

    df = pd.concat(
        [df, pd.get_dummies(df["SmokingStatus"], prefix="Smoke", dtype=int)], axis=1
    )
    df.drop(["SmokingStatus"], axis=1, inplace=True)

    df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1}).astype("float")

    return df


train_df = proc_df(train_df)




## === cell 16
def proc_train(df):
    rows = []
    smoke_cols = [c for c in df.columns if c.startswith("Smoke_")]

    for patient, g in df.groupby("Patient"):
        g = g.sort_values("Weeks").reset_index(drop=True)

        if (g["Weeks"] == 0).any():
            base = g.loc[g["Weeks"] == 0].iloc[0]
        else:
            base = g.iloc[0]

        base_week = float(base["Weeks"])
        base_fvc = float(base["FVC"])
        base_percent = float(base["Percent"])

        for _, r in g.iterrows():
            if float(r["Weeks"]) == base_week:
                continue

            row = {
                "Patient": patient,  # kept for grouping only; excluded from features later
                "Weeks": float(r["Weeks"]),
                "FVC": float(r["FVC"]),
                "base_Week": base_week,
                "base_FVC": base_fvc,
                "base_Percent": base_percent,
                "Percent": float(r["Percent"]),
                "Age": float(r["Age"]),
                "Sex": float(r["Sex"]),
                "week_diff": base_week - float(r["Weeks"]),
            }
            for c in smoke_cols:
                row[c] = float(base[c])
            rows.append(row)

    df_final = pd.DataFrame(rows)
    return df_final.reset_index(drop=True)


train_df = proc_train(train_df)
print("Processed train_df shape:", train_df.shape)
print(train_df.head())



## === cell 17
subm_key = subm[["Patient_Week"]].copy()
a = subm_key["Patient_Week"].str.split("_", expand=True)
a.columns = ["Patient", "Weeks"]

a["Weeks"] = pd.to_numeric(a["Weeks"], errors="coerce").astype("Int64")
if a["Weeks"].isna().any():
    raise ValueError(
        "Found non-numeric Weeks parsed from Patient_Week in submission key."
    )

a["Weeks"] = a["Weeks"].astype(int)

test_raw = pd.read_csv(path + "test.csv")
test_raw = proc_df(test_raw)

test_raw = test_raw.rename(
    columns={
        "Weeks": "base_Week",
        "FVC": "base_FVC",
        "Percent": "base_Percent",
    }
)

test_df = pd.merge(a, test_raw, how="left", on=["Patient"])

for c in ["base_Week", "base_FVC", "base_Percent", "Age", "Sex"]:
    if c in test_df.columns:
        test_df[c] = test_df.groupby("Patient")[c].transform(
            lambda s: s.ffill().bfill()
        )

smoke_cols = [c for c in test_df.columns if c.startswith("Smoke_")]
for c in smoke_cols:
    test_df[c] = test_df.groupby("Patient")[c].transform(lambda s: s.ffill().bfill())

test_df["week_diff"] = test_df["base_Week"].astype(float) - test_df["Weeks"].astype(
    float
)
test_df["Percent"] = test_df["base_Percent"].astype(float)

print("Processed test_df shape:", test_df.shape)
print(test_df.head())



## === cell 18
drop_cols_train = ["FVC"]
drop_cols_test = []

X = train_df.drop(["Patient"] + drop_cols_train, axis=1)
y = train_df["FVC"].astype(float)

test = test_df.drop(["Patient"] + drop_cols_test, axis=1)

X, test = X.align(test, join="left", axis=1, fill_value=0)

X = X.fillna(0)
test = test.fillna(0)

print("X shape:", X.shape, "test shape:", test.shape)
print("Any NaNs in X/test:", X.isna().any().any(), test.isna().any().any())
print("Using LightGBM:", LGB_AVAILABLE)



## === cell 19
num_fold = 5


def get_lgbm_model(X_train, y_train, X_val, y_val, fold, param_choice, columns):
    base_params = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 0.0,
        "random_state": 42,
    }

    if param_choice == "normal":
        params = {"objective": "regression", "metric": "rmse"}
    elif param_choice == "quantile1":
        params = {"objective": "quantile", "alpha": 0.1, "metric": "quantile"}
    elif param_choice == "quantile2":
        params = {"objective": "quantile", "alpha": 0.9, "metric": "quantile"}
    else:
        raise ValueError(f"Unknown param_choice: {param_choice}")

    params = {**base_params, **params}
    model = lgb.LGBMRegressor(**params, n_estimators=1200, n_jobs=-1)

    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=params.get("metric", None),
            callbacks=[lgb.log_evaluation(period=0)],
        )
    except TypeError:
        safe_eval_metric = (
            "rmse"
            if params.get("objective") == "quantile"
            else params.get("metric", None)
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=safe_eval_metric,
            verbose=-1,
        )

    fold_importance = pd.DataFrame()
    fold_importance["feature"] = columns
    fold_importance["importance"] = model.feature_importances_
    fold_importance = fold_importance.sort_values(by=["importance"])
    fold_importance.to_csv(
        os.path.join(
            WORKING_DIR, f"feature_importances_{y_train.name}_{param_choice}{fold}.csv"
        ),
        index=False,
    )
    return model


def get_sklearn_gbr_model(X_train, y_train, X_val, y_val, fold, param_choice, columns):
    if GradientBoostingRegressor is None:
        raise ImportError(
            "No available model backend: lightgbm missing and sklearn GBR missing."
        )

    if param_choice == "normal":
        loss = "squared_error"
        alpha = None
    elif param_choice == "quantile1":
        loss = "quantile"
        alpha = 0.1
    elif param_choice == "quantile2":
        loss = "quantile"
        alpha = 0.9
    else:
        raise ValueError(f"Unknown param_choice: {param_choice}")

    model = GradientBoostingRegressor(
        loss=loss,
        alpha=alpha if alpha is not None else 0.9,
        learning_rate=0.05,
        n_estimators=600,
        max_depth=3,
        random_state=42,
        subsample=0.9,
    )
    model.fit(X_train, y_train)

    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            fold_importance = pd.DataFrame(
                {"feature": columns, "importance": imp}
            ).sort_values("importance")
            fold_importance.to_csv(
                os.path.join(
                    WORKING_DIR,
                    f"feature_importances_{y_train.name}_{param_choice}{fold}.csv",
                ),
                index=False,
            )
    except Exception:
        pass

    return model


def get_model(X_train, y_train, X_val, y_val, fold, param_choice, columns):
    if LGB_AVAILABLE:
        return get_lgbm_model(
            X_train, y_train, X_val, y_val, fold, param_choice, columns
        )
    return get_sklearn_gbr_model(
        X_train, y_train, X_val, y_val, fold, param_choice, columns
    )


def get_lgbm_pred(X, y, test, param_choice, groups=None):
    print("get_lgbm_pred", param_choice)

    pred_te_sum = np.zeros(len(test), dtype=float)
    pred_val = np.zeros(len(X), dtype=float)

    if groups is None:
        splitter = KFold(n_splits=num_fold, random_state=42, shuffle=True)
        split_iter = splitter.split(X, y)
    else:
        gkf = GroupKFold(n_splits=num_fold)
        split_iter = gkf.split(X, y, groups=groups)

    fold = 0
    score = 0.0

    for train_index, test_index in split_iter:
        fold += 1
        print("fold", fold)

        X_train = X.iloc[train_index, :]
        X_val = X.iloc[test_index, :]
        y_train = y.iloc[train_index]
        y_val = y.iloc[test_index]

        model = get_model(
            X_train, y_train, X_val, y_val, fold, param_choice, X_train.columns
        )

        pred_te_sum += model.predict(test)
        pred_val[test_index] = model.predict(X_val)

        score += np.sqrt(mean_squared_error(y_val, pred_val[test_index]))
        print("rmse", str(score / fold))

    with open(os.path.join(WORKING_DIR, "score.txt"), "a+", encoding="utf-8") as f:
        f.write(str(score / num_fold) + ", ")

    pred_te_avg = pred_te_sum / num_fold
    return pred_te_avg, pred_val




## === cell 20
train_groups = train_df["Patient"].values

pred_FVC_te, pred_FVC_tr = get_lgbm_pred(X, y, test, "normal", groups=train_groups)

pred_FVC_te_q1, pred_FVC_tr_q1 = get_lgbm_pred(
    X, y, test, "quantile1", groups=train_groups
)
pred_FVC_te_q2, pred_FVC_tr_q2 = get_lgbm_pred(
    X, y, test, "quantile2", groups=train_groups
)

pred_conf_tr = pred_FVC_tr_q2 - pred_FVC_tr_q1
pred_conf_te = pred_FVC_te_q2 - pred_FVC_te_q1

pred_conf_tr = np.maximum(pred_conf_tr, 70.0)
pred_conf_te = np.maximum(pred_conf_te, 70.0)




## === cell 21
def metric(confidence, fvc, pred_fvc):
    confidence = max(confidence, 70)
    delta = min(abs(fvc - pred_fvc), 1000)
    score = -(math.sqrt(2) * (delta / confidence)) - np.log(math.sqrt(2) * confidence)
    return score


def calc_score(confidence, fvc, pred_fvc):
    score = 0.0
    for n in range(len(confidence)):
        score += metric(float(confidence[n]), float(fvc[n]), float(pred_fvc[n]))
    return score / len(confidence)


cv_score = calc_score(pred_conf_tr, train_df.FVC.values, pred_FVC_tr)
print("OOF metric (approx, uncalibrated):", cv_score)




## === cell 22
def calibrate_conf_multiplier(conf_tr, fvc_true, pred_fvc, grid=None):
    conf_tr = np.asarray(conf_tr, dtype=float)
    fvc_true = np.asarray(fvc_true, dtype=float)
    pred_fvc = np.asarray(pred_fvc, dtype=float)

    if grid is None:
        grid = np.linspace(0.5, 2.0, 61)

    best_m = 1.0
    best_score = -1e18
    for m in grid:
        s = calc_score(np.maximum(conf_tr * m, 70.0), fvc_true, pred_fvc)
        if s > best_score:
            best_score = s
            best_m = float(m)
    return best_m, best_score


conf_mult, cv_score_cal = calibrate_conf_multiplier(
    pred_conf_tr, train_df.FVC.values, pred_FVC_tr
)
print("Chosen confidence multiplier:", conf_mult)
print("OOF metric (approx, calibrated):", cv_score_cal)

pred_conf_te = np.maximum(pred_conf_te * conf_mult, 70.0)
pred_conf_tr = np.maximum(pred_conf_tr * conf_mult, 70.0)

fvc_lo, fvc_hi = np.percentile(train_df["FVC"].values, [0.5, 99.5])
pred_FVC_te = np.clip(pred_FVC_te, fvc_lo, fvc_hi)



## === cell 23
subm_out = subm[["Patient_Week"]].copy()
subm_out["FVC"] = pred_FVC_te
subm_out["Confidence"] = pred_conf_te

subm_out["FVC"] = subm_out["FVC"].astype(float)
subm_out["Confidence"] = (
    subm_out["Confidence"].astype(float).fillna(70.0).clip(lower=70.0)
)

subm_out = subm_out[["Patient_Week", "FVC", "Confidence"]]

if list(subm_out.columns) != ["Patient_Week", "FVC", "Confidence"]:
    raise ValueError(f"Bad submission columns: {subm_out.columns.tolist()}")
if len(subm_out) != len(subm):
    raise ValueError(f"Bad submission rowcount: {len(subm_out)} vs {len(subm)}")
if subm_out["Patient_Week"].isna().any():
    raise ValueError("Submission has NaN Patient_Week")
if subm_out[["FVC", "Confidence"]].isna().any().any():
    raise ValueError("Submission has NaN predictions")

out_path = os.path.join(WORKING_DIR, "submission.csv")
subm_out.to_csv(out_path, index=False)

print("Wrote submission.csv with shape:", subm_out.shape)
print(subm_out.head())
print("submission.csv exists:", os.path.exists(out_path))
print(
    "submission.csv size (bytes):",
    os.path.getsize(out_path) if os.path.exists(out_path) else None,
)

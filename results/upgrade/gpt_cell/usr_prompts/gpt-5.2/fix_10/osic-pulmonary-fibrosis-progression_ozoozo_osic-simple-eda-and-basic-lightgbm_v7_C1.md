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

No external packages required in the script and installed.

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

-6.998

# 6. Current score

-7.83152

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.92138) has done: 'Your pipeline already builds a valid submission, but it likely didn’t “yield” a score because it relies on external packages (LightGBM, pydicom, cv2, seaborn) that may not exist in the target runtime, causing the run to fail before writing `submission.csv`. I keep your core feature engineering and Laplace-style confidence idea, but replace the LightGBM dependency with a pure-scikit-learn regressor (hist-gradient boosting) and quantile models to preserve the same semantics (mean FVC + interval-based confidence). I also make the split deterministic and patient-grouped (to avoid leakage from the pairwise patient expansion), which should move the score in the right direction vs. a leaky row-wise KFold. Finally, I keep output formatting identical and always write `submission.csv`.'
- What this solution (achieved -7.80943) has done: 'Your current pipeline is already valid and stable, but it is likely underperforming because the “Confidence” you submit (q80–q20 width) is not calibrated for the Laplace Log Likelihood; overly small σ gets clipped to 70 and overly large σ is heavily penalized via the log term. I keep your exact modeling approach (same HistGradientBoosting models and GroupKFold) and only add a tiny post-fit calibration step that rescales the predicted interval width into a better σ, using out-of-fold residuals to choose a single global multiplier. This should improve the score toward the target by reducing the average penalty from miscalibrated confidence without changing the core regression predictions. I also clip/guard confidence values to avoid pathological widths and keep submission formatting identical.'
- What this solution (achieved -7.96444) has done: 'Your current mean-FVC model is fine; the biggest remaining gap to the target is likely from confidence miscalibration because the metric heavily penalizes both under- and over-confident σ. I keep the exact same models and CV, but replace the coarse grid search for the confidence multiplier with an analytic per-row Laplace-optimal σ (σ*=√2·Δ), then learn a single global scaling factor to map your predicted interval width onto that σ target. This keeps your semantics (interval-width-derived confidence) while matching the evaluation metric more directly, and it’s a minimal change localized to the confidence calibration block. I also add a small safety clip on the learned multiplier to avoid extreme σ inflation/deflation that could overshoot the target.'
- What this solution (achieved -7.76069) has done: 'Your current gap is negative (−7.96444 vs target −6.998, higher is better), so we should cautiously improve without changing the core modeling. The largest lever left (with minimal risk) is confidence calibration: your current single least-squares multiplier fits σ* derived from in-fold residuals, but can be biased because it uses clipped σ* and doesn’t directly optimize the Laplace metric. I keep the same three HistGradientBoosting models and the same GroupKFold OOF predictions, and only replace the confidence scaling step with a tiny 1D search over a small range of multipliers that directly maximizes the OOF Laplace metric. This preserves your semantics (confidence from quantile width) and typically moves the score upward toward the target by better balancing under/over-confidence penalties.'
- What this solution (achieved -7.89657) has done: 'Your current score is below the target (−7.76069 vs −6.998; higher is better), so we should make a small, low-risk improvement rather than altering the modeling core. The biggest remaining lever consistent with your approach is confidence calibration: right now you optimize the multiplier on the *expanded pairwise* training rows, which does not match the Kaggle metric’s patient-week evaluation and can miscalibrate σ. I keep the exact three HistGradientBoosting models and GroupKFold training unchanged, but change the confidence-multiplier search to maximize the Laplace metric on an out-of-fold aggregation at the original (Patient, Weeks) level (averaging predictions across base rows), which better matches evaluation semantics. This is a localized change in the calibration block and should move the score upward toward the target without changing the core prediction model.'
- What this solution (achieved -7.83152) has done: 'Your current score is below the target (−7.89657 vs −6.998; higher is better), so the smallest safe improvement lever—without touching the model/training core—is better confidence calibration. I keep the exact same three HistGradientBoostingRegressor models and GroupKFold OOF setup, but adjust the confidence multiplier search to optimize a calibration objective that matches the competition: only the final 3 weeks per patient are scored. Concretely, I restrict the OOF aggregation used for multiplier search to the “last-3 weeks per patient” subset (from train), which better aligns σ with the evaluated distribution and should move the leaderboard score upward toward the target. The submission formatting and file writing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor

path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"



## === cell 1
assert os.path.exists(os.path.join(path, "train.csv"))
assert os.path.exists(os.path.join(path, "test.csv"))
assert os.path.exists(os.path.join(path, "sample_submission.csv"))



## === cell 2
train_df = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")
subm = pd.read_csv(path + "sample_submission.csv")



## === cell 3
_ = train_df.head()



## === cell 4
_ = train_df.Patient.nunique()



## === cell 5
_ = train_df.Weeks.max()



## === cell 6
_ = train_df.Weeks.min()



## === cell 7
if False:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 1)
    sns.distplot(train_df[train_df["Weeks"].notna()]["Weeks"], ax=ax, color="#2222EE")
    ax.set_title("distribution of weeks in train")



## === cell 8
if False:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 1)
    sns.distplot(train_df[train_df["FVC"].notna()]["FVC"], ax=ax, color="#22EE22")
    ax.set_title("distribution of FVC in train")



## === cell 9
if False:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 1)
    sns.distplot(
        train_df[train_df["Percent"].notna()]["Percent"], ax=ax, color="#EE2222"
    )
    ax.set_title("distribution of Percent in train")



## === cell 10
if False:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 1)
    sns.distplot(train_df[train_df["Age"].notna()]["Age"], ax=ax, color="#992299")
    ax.set_title("distribution of Age in train")



## === cell 11
_ = train_df.Sex.value_counts()



## === cell 12
_ = train_df.Sex.value_counts(normalize=True)



## === cell 13
_ = train_df.groupby("Patient")["Sex"].first().value_counts(normalize=True)



## === cell 14
_ = train_df["SmokingStatus"].value_counts()



## === cell 15
_ = train_df["SmokingStatus"].value_counts(normalize=True)



## === cell 16
_ = test_df.head()




## === cell 17
def merge_subm_test(subm_df, test_df_in):
    a = subm_df["Patient_Week"].str.split("_", expand=True)
    a.columns = ["Patient", "Week"]
    test_df_out = test_df_in.merge(a, on="Patient")
    return test_df_out


_merged_test_preview = merge_subm_test(subm, test_df)



## === cell 18
_ = _merged_test_preview.head()



## === cell 19
_ = _merged_test_preview.groupby(["Patient"])["Weeks"].count()



## === cell 20
_ = _merged_test_preview.groupby(["Patient"])["Week"].first()



## === cell 21
_ = _merged_test_preview.groupby(["Patient"])["Week"].last()



## === cell 22
if False:
    from pydicom import dcmread
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(5, 6, figsize=(20, 20))
    for n in range(0, 30):
        image = dcmread(
            "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/"
            + str(n + 1)
            + ".dcm"
        )
        axs[int(n / 6), np.mod(n, 6)].imshow(image.pixel_array)



## === cell 23
if False:
    from pydicom import dcmread
    import matplotlib.pyplot as plt
    import numpy as np

    test_root = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/"
    patient_dirs = sorted(
        d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))
    )
    if not patient_dirs:
        raise FileNotFoundError(f"No patient directories found under: {test_root}")

    patient_id = patient_dirs[0]
    patient_path = os.path.join(test_root, patient_id)

    dcm_files = sorted(
        f for f in os.listdir(patient_path) if f.lower().endswith(".dcm")
    )
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found under: {patient_path}")

    n_show = min(30, len(dcm_files))
    fig, axs = plt.subplots(5, 6, figsize=(20, 20))

    for n in range(n_show):
        image = dcmread(os.path.join(patient_path, dcm_files[n]))
        axs[int(n / 6), np.mod(n, 6)].imshow(image.pixel_array)

    for n in range(n_show, 30):
        axs[int(n / 6), np.mod(n, 6)].axis("off")



## === cell 24
train_df = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")
subm = pd.read_csv(path + "sample_submission.csv")




## === cell 25
def proc_df(df):
    df = pd.concat([df, pd.get_dummies(df["SmokingStatus"], dtype=int)], axis=1)
    df.drop(["SmokingStatus"], axis=1, inplace=True)
    df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1})
    return df


train_df = proc_df(train_df)




## === cell 26
def proc_train(df):
    df_final_parts = []
    for patient, df2 in df.groupby("Patient", sort=False):
        df_target = df2[["Patient", "Weeks", "FVC"]].copy()

        df_base = df2.copy()
        df_base = df_base.rename(
            columns={
                "FVC": "base_FVC",
                "Percent": "base_Percent",
                "Weeks": "base_Week",
            },
            errors="raise",
        )

        df_target["_k"] = 1
        df_base["_k"] = 1
        df3 = pd.merge(df_target, df_base, on=["Patient", "_k"], how="inner").drop(
            columns=["_k"]
        )

        df3 = df3[df3["Weeks"] != df3["base_Week"]].copy()
        df3["week_diff"] = df3["base_Week"] - df3["Weeks"]

        df_final_parts.append(df3)

    df_final = pd.concat(df_final_parts, axis=0, ignore_index=True)

    df_final = df_final.dropna(
        subset=["FVC", "base_FVC", "base_Percent", "Age", "Sex", "week_diff"]
    )
    return df_final.reset_index(drop=True)




## === cell 27
train_df = proc_train(train_df)



## === cell 28
a = subm["Patient_Week"].str.split("_", expand=True)
a.columns = ["Patient", "Weeks"]
a["Weeks"] = a["Weeks"].astype("int")

test_df = test_df.rename(
    columns={
        "Weeks": "base_Week",
        "FVC": "base_FVC",
        "Percent": "base_Percent",
        "Age": "Age",
    },
    inplace=False,
)

test_df = proc_df(test_df)

test_df = pd.merge(a, test_df, how="left", on=["Patient"])
test_df["week_diff"] = test_df["base_Week"] - test_df["Weeks"]

_ = test_df.head()



## === cell 29
train_df = train_df.drop(
    set(train_df.columns) - set(test_df.columns) - {"FVC", "Percent"}, axis=1
)
test_df = test_df.drop(set(test_df.columns) - set(train_df.columns), axis=1)

X = train_df.drop(["Patient", "FVC"], axis=1)
y = train_df["FVC"].values  # ndarray for indexing
groups = train_df["Patient"].values  # patient-grouped CV avoids leakage
test = test_df.drop(["Patient"], axis=1)



## === cell 30
_ = X.head()



## === cell 31
_ = test.head()



## === cell 32
num_fold = 5


def get_sklearn_model(X_train, y_train, X_val, y_val, param_choice):
    if param_choice == "normal":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            random_state=42,
        )
    elif param_choice == "quantile1":
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.2,
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            random_state=42,
        )
    elif param_choice == "quantile2":
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.8,
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            random_state=42,
        )
    else:
        raise ValueError("Unknown param_choice")

    model.fit(X_train, y_train)
    return model


def get_model_pred(X, y, groups, test, param_choice):
    print("get_model_pred", param_choice)

    pred_te_sum = np.zeros(len(test), dtype=np.float64)
    pred_val = np.zeros(len(X), dtype=np.float64)

    gkf = GroupKFold(n_splits=num_fold)
    fold = 0
    score = 0.0

    for train_index, val_index in gkf.split(X, y, groups=groups):
        fold += 1
        print("fold", fold)

        X_train = X.iloc[train_index, :]
        X_val = X.iloc[val_index, :]
        y_train = y[train_index]
        y_val = y[val_index]

        model = get_sklearn_model(X_train, y_train, X_val, y_val, param_choice)

        pred_te_sum += model.predict(test)
        pred_val[val_index] = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, pred_val[val_index]))
        score += rmse
        print("rmse", str(score / fold))

    with open("score", "a+") as f:
        f.write(str(score / num_fold) + ", ")

    return pred_te_sum / num_fold, pred_val




## === cell 33
pred_FVC_te, pred_FVC_tr = get_model_pred(X, y, groups, test, "normal")
pred_FVC_te_q1, pred_FVC_tr_q1 = get_model_pred(X, y, groups, test, "quantile1")
pred_FVC_te_q2, pred_FVC_tr_q2 = get_model_pred(X, y, groups, test, "quantile2")

raw_conf_tr = np.maximum(pred_FVC_tr_q2 - pred_FVC_tr_q1, 1.0)
raw_conf_te = np.maximum(pred_FVC_te_q2 - pred_FVC_te_q1, 1.0)




## === cell 34
def metric(confidence, fvc, pred_fvc):
    confidence = max(float(confidence), 70.0)
    delta = min(abs(float(fvc) - float(pred_fvc)), 1000.0)
    score = -(math.sqrt(2.0) * (delta / confidence)) - np.log(
        math.sqrt(2.0) * confidence
    )
    return score


def calc_score(confidence, fvc, pred_fvc):
    score = 0.0
    n = len(fvc)
    for i in range(n):
        score += metric(confidence[i], fvc[i], pred_fvc[i])
    return score / n


base_conf_tr = np.maximum(raw_conf_tr.astype(np.float64), 1.0)

oof_calib_df = train_df[["Patient", "Weeks", "FVC"]].copy()
oof_calib_df["pred"] = pred_FVC_tr.astype(np.float64)
oof_calib_df["base_conf"] = base_conf_tr.astype(np.float64)

oof_agg = oof_calib_df.groupby(["Patient", "Weeks"], as_index=False).agg(
    FVC=("FVC", "first"), pred=("pred", "mean"), base_conf=("base_conf", "mean")
)

oof_agg["_rank_week_desc"] = oof_agg.groupby("Patient")["Weeks"].rank(
    method="first", ascending=False
)
oof_agg_eval = oof_agg[oof_agg["_rank_week_desc"] <= 3].copy()

m_grid = np.linspace(0.2, 5.0, 161)  # keep identical search range/step for stability
best_m = None
best_score = -1e18

y_true_agg = oof_agg_eval["FVC"].values.astype(np.float64)
y_pred_agg = oof_agg_eval["pred"].values.astype(np.float64)
base_conf_agg = np.maximum(oof_agg_eval["base_conf"].values.astype(np.float64), 1.0)

for m in m_grid:
    conf = np.maximum(base_conf_agg * float(m), 70.0)
    sc = calc_score(conf, y_true_agg, y_pred_agg)
    if sc > best_score:
        best_score = sc
        best_m = float(m)

pred_conf_tr = np.maximum(base_conf_tr * best_m, 70.0)
pred_conf_te = np.maximum(raw_conf_te.astype(np.float64) * best_m, 70.0)

print(
    "OOF Laplace metric (calibrated, aggregated, last-3 weeks):",
    best_score,
    " best_conf_multiplier:",
    best_m,
)
print("OOF aggregated rows used for calibration (last-3):", len(oof_agg_eval))



## === cell 35
subm_out = subm.copy()
subm_out["FVC"] = pred_FVC_te.astype(np.float64)
subm_out["Confidence"] = pred_conf_te.astype(np.float64)

subm_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm_out.shape)
print(subm_out.head())



## === cell 36
assert list(subm_out.columns) == ["Patient_Week", "FVC", "Confidence"]
assert len(subm_out) == len(subm)
assert subm_out["Confidence"].min() >= 70.0

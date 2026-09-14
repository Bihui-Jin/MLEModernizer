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

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1
xgboost==2.0.3

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

-6.8465

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn import (
    metrics,
    model_selection,
    linear_model,
    preprocessing,
    utils,
)




## === cell 1
def laplace_likelihood(y, p):
    m, s = p
    diff = np.minimum(1000, np.abs(y - m))
    s = np.maximum(70, s)
    lik = -np.sqrt(2) * diff / s - np.log(np.sqrt(2) * s)
    return np.mean(lik)


def laplace_likelihood_bound(y, p):
    m = p
    diff = np.minimum(1000, np.abs(y - m))
    s = np.maximum(70, np.sqrt(2) * diff)
    lik = -np.sqrt(2) * diff / s - np.log(np.sqrt(2) * s)
    return np.mean(lik)


def laplace_likelihood_avg(y, p):
    m = p
    diff = np.minimum(1000, np.abs(y - m))
    s = np.sqrt(2) * metrics.mean_absolute_error(y, m)
    lik = -np.sqrt(2) * diff / s - np.log(np.sqrt(2) * s)
    return np.mean(lik)




## === cell 2
def build_folds(X, y, group=None, k=5, shuffle=False, train_mask=None, valid_mask=None):
    """
    Minimal fix: when shuffle=True with GroupKFold, we must return folds in original index space.
    Otherwise, masks/indexing can silently mismatch and hurt generalization/score.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame):
        y = y.values
    if isinstance(group, pd.DataFrame):
        group = group.values
    if isinstance(train_mask, pd.DataFrame):
        train_mask = train_mask.values
    if isinstance(valid_mask, pd.DataFrame):
        valid_mask = valid_mask.values

    if group is None:
        folds = list(model_selection.KFold(k, shuffle=True).split(X, y))
    else:
        if shuffle:
            idx = utils.shuffle(np.arange(X.shape[0]))
            Xs, ys, groups = X.copy()[idx], y.copy()[idx], group.copy()[idx]
            folds_shuffled = list(
                model_selection.GroupKFold(k).split(
                    np.array(Xs), np.array(ys), np.array(groups)
                )
            )
            folds = [(idx[tr], idx[va]) for (tr, va) in folds_shuffled]
        else:
            folds = list(model_selection.GroupKFold(k).split(X, y, group))

    if train_mask is not None:
        for i in range(k):
            folds[i] = (
                np.array([j for j in folds[i][0] if train_mask[j]]),
                folds[i][1],
            )
    if valid_mask is not None:
        for i in range(k):
            folds[i] = (
                folds[i][0],
                np.array([j for j in folds[i][1] if valid_mask[j]]),
            )
    return folds




## === cell 3
def feature_eng(df):
    df = df.copy()
    df["n_weeks"] = df["Weeks_target"] - df["Weeks_base"]
    df["symlog_n_weeks"] = np.sign(df["n_weeks"]) * np.log(1 + np.abs(df["n_weeks"]))
    df["symlog_n_weeks2"] = np.sign(df["n_weeks"]) * np.log(
        1 + np.abs(df["n_weeks"]) ** 2
    )
    df["expdecay_n_weeks"] = np.exp(-np.abs(df["n_weeks"]))
    df["Sex_female"] = (df["Sex"] == "Female").astype("float")
    df["Smoking_ex"] = (df["SmokingStatus"] == "Ex-smoker").astype(int)
    df["Smoking_currently"] = (df["SmokingStatus"] == "Currently smokes").astype(int)
    return df




## === cell 4
data_folder = "../input/osic-pulmonary-fibrosis-progression"



## === cell 5
df_train = pd.read_csv(os.path.join(data_folder, "train.csv")).drop_duplicates(
    keep=False, subset=["Patient", "Weeks"]
)
df_train["n_obs"] = df_train.groupby("Patient").Weeks.cumcount()
df_train["till_last"] = (
    df_train.groupby("Patient").Weeks.transform("count") - 1 - df_train["n_obs"]
)

df_train = df_train.merge(
    df_train.drop(["Age", "Sex", "Percent", "SmokingStatus"], axis=1),
    on="Patient",
    suffixes=["_base", "_target"],
)

cols_before_fe = df_train.columns
df_train = feature_eng(df_train)
FEATURES = ["FVC_base", "Percent", "Age"] + [
    c for c in df_train.columns if c not in cols_before_fe
]
df_train[FEATURES].head(2)



## === cell 6
train_mask = (
    (df_train.n_weeks > 0)
    & df_train.n_obs_base.between(0, 3)
    & df_train.till_last_target.between(0, 10)
)
valid_mask = (
    (df_train.n_weeks > 0)
    & df_train.n_obs_base.between(0, 0)
    & df_train.till_last_target.between(0, 3)
)



## === cell 7
df_test = pd.read_csv(os.path.join(data_folder, "test.csv")).rename(
    columns={"Weeks": "Weeks_base", "FVC": "FVC_base"}
)

df_test = (
    df_test.assign(k=0)
    .merge(pd.DataFrame({"Weeks_target": np.arange(-12, 133 + 1), "k": 0}))
    .drop("k", axis=1)
)

df_test = feature_eng(df_test)
df_test.head(2)



## === cell 8
print(50 * "#")
print(
    "##  Train:  %4d rows with %3d unique patients  ##"
    % (df_train[train_mask].shape[0], df_train[train_mask].Patient.nunique())
)
print(
    "##  Valid:  %4d rows with %3d unique patients  ##"
    % (df_train[valid_mask].shape[0], df_train[valid_mask].Patient.nunique())
)
print(
    "##  Test:   %4d rows with %3d unique patients  ##"
    % (df_test.shape[0], df_test.Patient.nunique())
)
print(50 * "#")




## === cell 9
def _transform(x, mode: str = None, df=None):
    if mode is None:
        return x
    if mode.upper() == "DIFF":
        return x - df.FVC_base
    if mode.upper() == "DIFF_FVC":
        return np.where(df.n_weeks == 0, 0.0, x - df.FVC_base)
    if mode.upper() == "PERC":
        return x / df.FVC_base
    if mode.upper() == "PERC_FVC":
        return np.where(df.n_weeks == 0, 1.0, x / df.FVC_base)
    if re.match("^LOG", mode.upper()):
        gamma = re.sub("^LOG[:]", "", mode.upper())
        gamma = float(gamma) if len(gamma) > 0 else 0.0
        return np.log(gamma + x)
    if re.match("^BOXCOX", mode.upper()):
        gamma = re.sub("^BOXCOX[:]", "", mode.upper())
        gamma = float(gamma) if len(gamma) > 0 else 1.0
        return (x**gamma - 1.0) / gamma
    return None


def _inverse_transform(x, mode=None, df=None):
    if mode is None:
        return x
    if mode.upper() == "DIFF":
        return x + df.FVC_base
    if mode.upper() == "DIFF_FVC":
        return np.where(df.n_weeks == 0, df.FVC_base, x + df.FVC_base)
    if mode.upper() == "PERC":
        return x * df.FVC_base
    if mode.upper() == "PERC_FVC":
        return np.where(df.n_weeks == 0, df.FVC_base, x * df.FVC_base)
    if re.match("^LOG", mode.upper()):
        gamma = re.sub("^LOG[:]", "", mode.upper())
        gamma = float(gamma) if len(gamma) > 0 else 0.0
        return np.exp(x) - gamma
    if re.match("^BOXCOX", mode.upper()):
        gamma = re.sub("^BOXCOX[:]", "", mode.upper())
        gamma = float(gamma) if len(gamma) > 0 else 1.0
        return (1.0 + x * gamma) ** (1 / gamma)
    return None


def transform(x, mode=None, df=None):
    out = x.copy()
    if isinstance(mode, list):
        for m in mode:
            out = _transform(out, mode=m, df=df)
    if isinstance(mode, str):
        out = _transform(out, mode=mode, df=df)
    return out


def inverse_transform(x, mode=None, df=None):
    out = x.copy()
    if isinstance(mode, list):
        for m in mode[::-1]:
            out = _inverse_transform(out, mode=m, df=df)
    if isinstance(mode, str):
        out = _inverse_transform(out, mode=mode, df=df)
    return out




## === cell 10
MODE = ["PERC_FVC", "BOXCOX:0.35"]
SIGMA_MODE = ["PERC"]

X = df_train[FEATURES].copy().values
y = df_train["FVC_target"].copy().values
group = df_train["Patient"].values
X_test = df_test[FEATURES].copy().values



## === cell 11
N_FOLDS = 10
folds = build_folds(
    X, y, group, k=N_FOLDS, shuffle=True, train_mask=train_mask, valid_mask=valid_mask
)
valid_folds = pd.DataFrame(
    {
        "idx": np.concatenate([np.array(f[1]) for fi, f in enumerate(folds)]),
        "fold": np.concatenate(
            [
                np.repeat(np.array(fi).reshape(1), len(f[1]))
                for fi, f in enumerate(folds)
            ]
        ),
    }
).set_index("idx")

prep = preprocessing.Normalizer()
Z = prep.fit_transform(X)
Z_test = prep.transform(X_test)

mean_target = transform(y, MODE, df_train)
plt.hist(mean_target, 40)



## === cell 12
pred_oof = np.nan * np.zeros((N_FOLDS, X.shape[0]))
pred_test = np.nan * np.zeros((N_FOLDS, X_test.shape[0]))
for i, (idxT, idxV) in enumerate(tqdm(folds)):
    model = linear_model.LinearRegression()
    model.fit(Z[idxT], mean_target[idxT])
    pred_oof[i] = model.predict(Z)
    pred_oof[i, idxT] = np.nan
    pred_test[i] = model.predict(Z_test)
pred_mean = inverse_transform(np.nanmean(pred_oof, 0), MODE, df_train)
test_mean = inverse_transform(np.nanmean(pred_test, 0), MODE, df_test)



## === cell 13
opt_sigma = np.sqrt(2) * np.clip(np.abs(y - pred_mean), 0, 1000)
sigma_target = transform(opt_sigma, SIGMA_MODE, df_train)
plt.hist(sigma_target[train_mask | valid_mask], 50)



## === cell 14
pred_oof = np.nan * np.zeros((N_FOLDS, X.shape[0]))
pred_test = np.nan * np.zeros((N_FOLDS, X_test.shape[0]))
for i, (idxT, idxV) in enumerate(tqdm(folds)):
    model = linear_model.LinearRegression()
    model.fit(Z[idxT], sigma_target[idxT])
    pred_oof[i] = model.predict(Z)
    pred_oof[i, idxT] = np.nan
    pred_test[i] = model.predict(Z_test)
pred_sigma = inverse_transform(np.nanmean(pred_oof, 0), SIGMA_MODE, df_train)
test_sigma = inverse_transform(np.nanmean(pred_test, 0), SIGMA_MODE, df_test)



## === cell 15
lll_overall = laplace_likelihood(
    y[valid_mask], [pred_mean[valid_mask], pred_sigma[valid_mask]]
)
lllb_overall = laplace_likelihood_bound(y[valid_mask], pred_mean[valid_mask])
y_rmse_overall = metrics.mean_squared_error(y[valid_mask], pred_mean[valid_mask]) ** 0.5
y_mae_overall = metrics.mean_absolute_error(y[valid_mask], pred_mean[valid_mask])
s_rmse_overall = (
    metrics.mean_squared_error(opt_sigma[valid_mask], pred_sigma[valid_mask]) ** 0.5
)
s_mae_overall = metrics.mean_absolute_error(
    opt_sigma[valid_mask], pred_sigma[valid_mask]
)

_df_base = pd.DataFrame(
    {"y": y, "so": opt_sigma, "p": pred_mean, "s": pred_sigma}
).merge(valid_folds, left_index=True, right_index=True)
lllb = _df_base.groupby("fold").apply(
    lambda x: laplace_likelihood_bound(x["y"], x["p"])
)
lll = _df_base.groupby("fold").apply(
    lambda x: laplace_likelihood(x["y"], [x["p"], x["s"]])
)
y_rmse = _df_base.groupby("fold").apply(
    lambda x: metrics.mean_squared_error(x["y"], x["p"]) ** 0.5
)
y_mae = _df_base.groupby("fold").apply(
    lambda x: metrics.mean_absolute_error(x["y"], x["p"])
)
s_rmse = _df_base.groupby("fold").apply(
    lambda x: metrics.mean_squared_error(x["so"], x["s"]) ** 0.5
)
s_mae = _df_base.groupby("fold").apply(
    lambda x: metrics.mean_absolute_error(x["so"], x["s"])
)

print("        OVERALL  ||  FOLD-WISE")
print(
    "Bound:  %7.4f  ||  %7.4f  (+- %7.4f)" % (lllb_overall, lllb.mean(), 2 * lllb.std())
)
print("Score:  %7.4f  ||  %7.4f  (+- %7.4f)" % (lll_overall, lll.mean(), 2 * lll.std()))
print(
    "yRMSE:  %7.2f  ||  %7.2f  (+- %7.2f)"
    % (y_rmse_overall, y_rmse.mean(), 2 * y_rmse.std())
)
print(
    "y-MAE:  %7.2f  ||  %7.2f  (+- %7.2f)"
    % (y_mae_overall, y_mae.mean(), 2 * y_mae.std())
)
print(
    "sRMSE:  %7.2f  ||  %7.2f  (+- %7.2f)"
    % (s_rmse_overall, s_rmse.mean(), 2 * s_rmse.std())
)
print(
    "s-MAE:  %7.2f  ||  %7.2f  (+- %7.2f)"
    % (s_mae_overall, s_mae.mean(), 2 * s_mae.std())
)



## === cell 16
pd.DataFrame({"LLL": lll, "Bound": lllb}).boxplot(figsize=(12, 4))



## === cell 17
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.scatter(y[valid_mask], pred_mean[valid_mask], alpha=0.2)
plt.xlim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
plt.ylim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
plt.plot(plt.xlim(), plt.ylim(), color="tab:red", alpha=0.5, linestyle="--")
plt.subplot(1, 2, 2)
plt.scatter(opt_sigma[valid_mask], pred_sigma[valid_mask], alpha=0.2)
plt.xlim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
plt.ylim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
plt.plot(plt.xlim(), plt.ylim(), color="tab:red", alpha=0.5, linestyle="--")



## === cell 18
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.scatter(df_train.Weeks_target[valid_mask], pred_mean[valid_mask], alpha=0.5)
plt.subplot(1, 2, 2)
plt.scatter(df_train.Weeks_target[valid_mask], pred_sigma[valid_mask], alpha=0.5)



## === cell 19
plt.figure(figsize=(16, 8))
for i, pid in enumerate(df_train[valid_mask].Patient.unique()[:12]):
    plt.subplot(3, 4, i + 1)
    idx = (df_train.Patient == pid) & (df_train.n_obs_base == 0)
    plt.fill_between(
        df_train[idx].Weeks_target,
        pred_mean[idx] - 1 * pred_sigma[idx],
        pred_mean[idx] + 1 * pred_sigma[idx],
        color="tab:blue",
        alpha=0.1,
    )
    plt.fill_between(
        df_train[idx].Weeks_target,
        pred_mean[idx] - 2 * pred_sigma[idx],
        pred_mean[idx] + 2 * pred_sigma[idx],
        color="tab:blue",
        alpha=0.1,
    )
    plt.plot(df_train[idx].Weeks_target, pred_mean[idx], marker="x", color="tab:blue")
    plt.plot(
        df_train[idx].Weeks_target,
        df_train[idx].FVC_target,
        marker="o",
        color="tab:red",
    )



## === cell 20
for i, pid in enumerate(df_train[valid_mask].Patient.unique()):
    idx = (df_train.Patient == pid) & (df_train.n_obs_base == 0)
    plt.plot(
        df_train[idx].n_weeks,
        pred_mean[idx] / df_train[idx].FVC_base,
        color="tab:blue",
        alpha=0.2,
    )



## === cell 21
submission = df_test.copy()[["Patient", "Weeks_target"]]
submission["Patient_Week"] = (
    submission["Patient"] + "_" + submission["Weeks_target"].astype("str")
)

submission["FVC"] = np.asarray(test_mean, dtype=float)
submission["Confidence"] = np.asarray(test_sigma, dtype=float)

submission["FVC"] = np.where(
    np.isfinite(submission["FVC"]), submission["FVC"], submission["FVC"].median()
)
submission["Confidence"] = np.where(
    np.isfinite(submission["Confidence"]), submission["Confidence"], 70.0
)
submission["Confidence"] = np.clip(submission["Confidence"], 70.0, None)

submission = submission.sort_values(["Weeks_target", "Patient"])[
    ["Patient_Week", "FVC", "Confidence"]
]
submission.head()



## === cell 22
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 23
plt.figure(figsize=(16, 8))

test_patients = df_test.Patient.unique()[:6]

for i, pid in enumerate(test_patients):
    plt.subplot(2, 3, i + 1)
    idx = df_test.Patient == pid
    plt.fill_between(
        df_test[idx].Weeks_target,
        test_mean[idx] - 1 * test_sigma[idx],
        test_mean[idx] + 1 * test_sigma[idx],
        color="tab:blue",
        alpha=0.1,
    )
    plt.fill_between(
        df_test[idx].Weeks_target,
        test_mean[idx] - 2 * test_sigma[idx],
        test_mean[idx] + 2 * test_sigma[idx],
        color="tab:blue",
        alpha=0.1,
    )
    plt.plot(df_test[idx].Weeks_target, test_mean[idx], color="tab:blue")
    plt.plot(
        df_test[idx].Weeks_base, df_test[idx].FVC_base, marker="o", color="tab:red"
    )
plt.tight_layout()

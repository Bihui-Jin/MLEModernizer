# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os, re

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".")[0])
    if _pb_major >= 5:
        import sys, subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import xgboost
import lightgbm
import catboost

from sklearn import (
    ensemble,
    metrics,
    model_selection,
    linear_model,
    preprocessing,
    utils,
)

import tensorflow as tf
from tensorflow import keras




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
        idx = utils.shuffle(np.arange(X.shape[0]))
        if shuffle:
            Xs, ys, groups = X.copy()[idx], y.copy()[idx], group.copy()[idx]
            folds = list(
                model_selection.GroupKFold(k).split(
                    np.array(Xs), np.array(ys), np.array(groups)
                )
            )
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
    & df_train.n_obs_base.between(0, 2)
    & df_train.till_last_target.between(0, 10)
)
valid_mask = (
    (df_train.n_weeks > 0)
    & df_train.n_obs_base.between(0, 0)
    & df_train.till_last_target.between(0, 2)
)



## === cell 7
df_test = pd.read_csv(os.path.join(data_folder, "test.csv")).rename(
    columns={"Weeks": "Weeks_base", "FVC": "FVC_base"}
)

df_test = (
    df_test.assign(k=0)
    .merge(pd.DataFrame({"Weeks_target": np.arange(-12, 133 + 1), "k": 0}))
    .drop(columns=["k"])
)

df_test = feature_eng(df_test)

df_test.head(2)



## === cell 8
print(100 * "#")
print(
    "Train: %4d rows with %3d unique patients"
    % (df_train[train_mask].shape[0], df_train[train_mask].Patient.nunique())
)
print(
    "Valid: %4d rows with %3d unique patients"
    % (df_train[valid_mask].shape[0], df_train[valid_mask].Patient.nunique())
)
print(
    "Test:  %4d rows with %3d unique patients"
    % (df_test.shape[0], df_test.Patient.nunique())
)
print(100 * "#")




## === cell 9
def transform(x, mode=None, df=None):
    if mode is None:
        return x
    if mode.upper() == "VAR":
        return x - df.FVC_base
    if mode.upper() == "PERC":
        return x / df.FVC_base
    if mode.upper() == "LOG":
        return np.log(x)


def inverse_transform(x, mode=None, df=None):
    if mode is None:
        return x
    if mode.upper() == "VAR":
        return x + df.FVC_base
    if mode.upper() == "PERC":
        return x * df.FVC_base
    if mode.upper() == "LOG":
        return np.exp(x)




## === cell 10
MODE = "PERC"
SIGMA_MODE = "PERC"

X = df_train[FEATURES].copy()
y = df_train["FVC_target"].copy()
group = df_train["Patient"]
X_test = df_test[FEATURES].copy()



## === cell 11
N_FOLDS = 10
folds = build_folds(
    X, y, group, k=N_FOLDS, train_mask=train_mask, valid_mask=valid_mask
)



## === cell 12
prep = preprocessing.Normalizer()
Z = prep.fit_transform(X)
Z_test = prep.transform(X_test)

mean_target = transform(y, MODE, df_train)



## === cell 13
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



## === cell 14
opt_sigma = np.sqrt(2) * np.clip(np.abs(y - pred_mean), 0, 1000)
sigma_target = transform(opt_sigma, SIGMA_MODE, df_train)
plt.hist(opt_sigma, 50)



## === cell 15
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



## === cell 16
print(
    "Score: %.4f (bounded by mean prediction at %.4f)"
    % (
        laplace_likelihood(
            y[valid_mask], [pred_mean[valid_mask], pred_sigma[valid_mask]]
        ),
        laplace_likelihood_bound(y[valid_mask], pred_mean[valid_mask]),
    )
)



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
sample = pd.read_csv(os.path.join(data_folder, "sample_submission.csv"))
sample = sample.copy()
sample["Patient"] = sample["Patient_Week"].str.split("_").str[0]
sample["Weeks_target"] = sample["Patient_Week"].str.split("_").str[1].astype(int)

pred_df = df_test[["Patient", "Weeks_target"]].copy()
pred_df["FVC"] = test_mean
pred_df["Confidence"] = test_sigma

submission = sample.merge(pred_df, on=["Patient", "Weeks_target"], how="left")

submission["Confidence"] = submission["Confidence"].astype(float)
submission["Confidence"] = submission["Confidence"].fillna(70.0)
submission["Confidence"] = np.maximum(1.0, submission["Confidence"])

submission = submission[["Patient_Week", "FVC", "Confidence"]]

assert list(submission.columns) == ["Patient_Week", "FVC", "Confidence"]
assert submission.shape[0] == sample.shape[0]
assert submission["FVC"].notnull().all()

submission.head()



## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mget_loc[0;34m(self, key)[0m
[1;32m   3804[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3805[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_engine[0m[0;34m.[0m[0mget_loc[0m[0;34m([0m[0mcasted_key[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3806[0m         [0;32mexcept[0m [0mKeyError[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32mindex.pyx[0m in [0;36mpandas._libs.index.IndexEngine.get_loc[0;34m()[0m

[0;32mindex.pyx[0m in [0;36mpandas._libs.index.IndexEngine.get_loc[0;34m()[0m

[0;32mpandas/_libs/hashtable_class_helper.pxi[0m in [0;36mpandas._libs.hashtable.PyObjectHashTable.get_item[0;34m()[0m

[0;32mpandas/_libs/hashtable_class_helper.pxi[0m in [0;36mpandas._libs.hashtable.PyObjectHashTable.get_item[0;34m()[0m

[0;31mKeyError[0m: 'Confidence'

The above exception was the direct cause of the following exception:

[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3976403487.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     13[0m [0;34m[0m[0m
[1;32m     14[0m [0;31m# Change (metric-safety): ensure Confidence is positive; metric clips at 70 anyway, but negative/NaN can break scoring.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 15[0;31m [0msubmission[0m[0;34m[[0m[0;34m"Confidence"[0m[0;34m][0m [0;34m=[0m [0msubmission[0m[0;34m[[0m[0;34m"Confidence"[0m[0;34m][0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mfloat[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m [0msubmission[0m[0;34m[[0m[0;34m"Confidence"[0m[0;34m][0m [0;34m=[0m [0msubmission[0m[0;34m[[0m[0;34m"Confidence"[0m[0;34m][0m[0;34m.[0m[0mfillna[0m[0;34m([0m[0;36m70.0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m [0msubmission[0m[0;34m[[0m[0;34m"Confidence"[0m[0;34m][0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mmaximum[0m[0;34m([0m[0;36m1.0[0m[0;34m,[0m [0msubmission[0m[0;34m[[0m[0;34m"Confidence"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   4100[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0mnlevels[0m [0;34m>[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4101[0m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_multilevel[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4102[0;31m             [0mindexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0mget_loc[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4103[0m             [0;32mif[0m [0mis_integer[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4104[0m                 [0mindexer[0m [0;34m=[0m [0;34m[[0m[0mindexer[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mget_loc[0;34m(self, key)[0m
[1;32m   3810[0m             ):
[1;32m   3811[0m                 [0;32mraise[0m [0mInvalidIndexError[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3812[0;31m             [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0mkey[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3813[0m         [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3814[0m             [0;31m# If we have a listlike key, _check_indexing_error will raise[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: 'Confidence'

## === cell 21
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)

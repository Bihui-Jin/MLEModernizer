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

-6.8495

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import model_selection, preprocessing, linear_model, metrics




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
        idx = np.random.permutation(np.arange(X.shape[0]))
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
    sex_col = "Sex_base" if "Sex_base" in df.columns else "Sex"
    smoke_col = (
        "SmokingStatus_base" if "SmokingStatus_base" in df.columns else "SmokingStatus"
    )
    df["Sex_female"] = (df[sex_col] == "Female").astype(float)
    df["Smoking_ex"] = (df[smoke_col] == "Ex-smoker").astype(int)
    df["Smoking_currently"] = (df[smoke_col] == "Currently smokes").astype(int)
    return df




## === cell 4
data_folder = os.path.join(os.getcwd(), "data", "osic-pulmonary-fibrosis-progression")

df_train = pd.read_csv(os.path.join(data_folder, "train.csv")).drop_duplicates(
    keep=False, subset=["Patient", "Weeks"]
)

df_train["n_obs"] = df_train.groupby("Patient").Weeks.cumcount()
df_train["till_last"] = (
    df_train.groupby("Patient").Weeks.transform("count") - 1 - df_train["n_obs"]
)

df_train["n_obs_base"] = df_train["n_obs"]

df_train = df_train.merge(
    df_train.drop(["Age", "Sex", "Percent", "SmokingStatus"], axis=1),
    on="Patient",
    suffixes=["_base", "_target"],
)

for col in ["Percent_base", "Age_base", "Sex_base", "SmokingStatus_base"]:
    if col not in df_train.columns:
        orig = col.replace("_base", "")
        if orig in df_train.columns:
            df_train.rename(columns={orig: col}, inplace=True)

cols_before_fe = df_train.columns
df_train = feature_eng(df_train)

base_feats = [
    c for c in ["FVC_base", "Percent_base", "Age_base"] if c in df_train.columns
]
FEATURES = base_feats + [c for c in df_train.columns if c not in cols_before_fe]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1127159640.py in <cell line: 0>()
      2 data_folder = os.path.join(os.getcwd(), "data", "osic-pulmonary-fibrosis-progression")
      3 
----> 4 df_train = pd.read_csv(os.path.join(data_folder, "train.csv")).drop_duplicates(
      5     keep=False, subset=["Patient", "Weeks"]
      6 )

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/data/osic-pulmonary-fibrosis-progression/train.csv'

## === cell 5
df_test = pd.read_csv(os.path.join(data_folder, "test.csv")).rename(
    columns={"Weeks": "Weeks_base", "FVC": "FVC_base"}
)

df_test.rename(
    columns={
        "Percent": "Percent_base",
        "Age": "Age_base",
        "Sex": "Sex_base",
        "SmokingStatus": "SmokingStatus_base",
    },
    inplace=True,
)

df_test = (
    df_test.assign(k=0)
    .merge(pd.DataFrame({"Weeks_target": np.arange(-12, 133 + 1), "k": 0}), on="k")
    .drop("k", axis=1)
)
df_test = feature_eng(df_test)

indices = np.arange(len(df_train))
train_idx, valid_idx = model_selection.train_test_split(
    indices, test_size=0.2, random_state=42
)
train_mask = np.zeros(len(df_train), dtype=bool)
valid_mask = np.zeros(len(df_train), dtype=bool)
train_mask[train_idx] = True
valid_mask[valid_idx] = True



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2067018262.py in <cell line: 0>()
----> 1 df_test = pd.read_csv(os.path.join(data_folder, "test.csv")).rename(
      2     columns={"Weeks": "Weeks_base", "FVC": "FVC_base"}
      3 )
      4 
      5 df_test.rename(

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/data/osic-pulmonary-fibrosis-progression/test.csv'

## === cell 6
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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2033268207.py in <cell line: 0>()
      2 print(
      3     "Train: %4d rows with %3d unique patients"
----> 4     % (df_train[train_mask].shape[0], df_train[train_mask].Patient.nunique())
      5 )
      6 print(

NameError: name 'df_train' is not defined

## === cell 7
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




## === cell 8
MODE = "PERC"
SIGMA_MODE = "PERC"

X = df_train[FEATURES].copy()
y = df_train["FVC_target"].copy()
group = df_train["Patient"]
X_test = df_test[FEATURES].copy()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3114904037.py in <cell line: 0>()
      2 SIGMA_MODE = "PERC"
      3 
----> 4 X = df_train[FEATURES].copy()
      5 y = df_train["FVC_target"].copy()
      6 group = df_train["Patient"]

NameError: name 'df_train' is not defined

## === cell 9
N_FOLDS = 10
folds = build_folds(X, y, group, k=N_FOLDS)

prep = preprocessing.StandardScaler()
Z = prep.fit_transform(X)
Z_test = prep.transform(X_test)

mean_target = transform(y, MODE, df_train)

pred_oof = np.full((N_FOLDS, X.shape[0]), np.nan)
pred_test = np.full((N_FOLDS, X_test.shape[0]), np.nan)

for i, (idxT, idxV) in enumerate(tqdm(folds)):
    model = linear_model.Ridge(alpha=0.5, fit_intercept=True)
    model.fit(Z[idxT], mean_target[idxT])
    pred_oof[i] = model.predict(Z)
    pred_oof[i, idxT] = np.nan
    pred_test[i] = model.predict(Z_test)

pred_mean = inverse_transform(np.nanmean(pred_oof, axis=0), MODE, df_train)
test_mean = inverse_transform(np.nanmean(pred_test, axis=0), MODE, df_test)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3486192161.py in <cell line: 0>()
      1 N_FOLDS = 10
----> 2 folds = build_folds(X, y, group, k=N_FOLDS)
      3 
      4 prep = preprocessing.StandardScaler()
      5 Z = prep.fit_transform(X)

NameError: name 'X' is not defined

## === cell 10
SIGMA_SCALE = 1.20  # raised from 1.10

opt_sigma = np.sqrt(2) * np.clip(np.abs(y - pred_mean), 0, 1000)
sigma_target = transform(opt_sigma, SIGMA_MODE, df_train)

pred_oof = np.full((N_FOLDS, X.shape[0]), np.nan)
pred_test = np.full((N_FOLDS, X_test.shape[0]), np.nan)

for i, (idxT, idxV) in enumerate(tqdm(folds)):
    model = linear_model.Ridge(alpha=0.5, fit_intercept=True)
    model.fit(Z[idxT], sigma_target[idxT])
    pred_oof[i] = model.predict(Z)
    pred_oof[i, idxT] = np.nan
    pred_test[i] = model.predict(Z_test)

pred_sigma_raw = inverse_transform(np.nanmean(pred_oof, axis=0), SIGMA_MODE, df_train)
pred_sigma = np.maximum(pred_sigma_raw * SIGMA_SCALE, 70)

test_sigma_raw = inverse_transform(np.nanmean(pred_test, axis=0), SIGMA_MODE, df_test)
test_sigma = np.maximum(test_sigma_raw * SIGMA_SCALE, 70)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/472215169.py in <cell line: 0>()
      2 SIGMA_SCALE = 1.20  # raised from 1.10
      3 
----> 4 opt_sigma = np.sqrt(2) * np.clip(np.abs(y - pred_mean), 0, 1000)
      5 sigma_target = transform(opt_sigma, SIGMA_MODE, df_train)
      6 

NameError: name 'y' is not defined

## === cell 11
print(
    "Score: %.4f (bounded by mean prediction at %.4f)"
    % (
        laplace_likelihood(
            y[valid_mask], [pred_mean[valid_mask], pred_sigma[valid_mask]]
        ),
        laplace_likelihood_bound(y[valid_mask], pred_mean[valid_mask]),
    )
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3149779420.py in <cell line: 0>()
      3     % (
      4         laplace_likelihood(
----> 5             y[valid_mask], [pred_mean[valid_mask], pred_sigma[valid_mask]]
      6         ),
      7         laplace_likelihood_bound(y[valid_mask], pred_mean[valid_mask]),

NameError: name 'y' is not defined

## === cell 12
import matplotlib.pyplot as plt

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
plt.show()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1337798665.py in <cell line: 0>()
      3 plt.figure(figsize=(16, 6))
      4 plt.subplot(1, 2, 1)
----> 5 plt.scatter(y[valid_mask], pred_mean[valid_mask], alpha=0.2)
      6 plt.xlim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
      7 plt.ylim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))

NameError: name 'y' is not defined

## === cell 13
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.scatter(df_train.Weeks_target[valid_mask], pred_mean[valid_mask], alpha=0.5)
plt.subplot(1, 2, 2)
plt.scatter(df_train.Weeks_target[valid_mask], pred_sigma[valid_mask], alpha=0.5)
plt.show()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4153199804.py in <cell line: 0>()
      1 plt.figure(figsize=(16, 6))
      2 plt.subplot(1, 2, 1)
----> 3 plt.scatter(df_train.Weeks_target[valid_mask], pred_mean[valid_mask], alpha=0.5)
      4 plt.subplot(1, 2, 2)
      5 plt.scatter(df_train.Weeks_target[valid_mask], pred_sigma[valid_mask], alpha=0.5)

NameError: name 'df_train' is not defined

## === cell 14
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
plt.tight_layout()
plt.show()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1930122357.py in <cell line: 0>()
      1 plt.figure(figsize=(16, 8))
----> 2 for i, pid in enumerate(df_train[valid_mask].Patient.unique()[:12]):
      3     plt.subplot(3, 4, i + 1)
      4     idx = (df_train.Patient == pid) & (df_train.n_obs_base == 0)
      5     plt.fill_between(

NameError: name 'df_train' is not defined

## === cell 15
submission = df_test[["Patient", "Weeks_target"]].copy()
submission["Patient_Week"] = (
    submission["Patient"] + "_" + submission["Weeks_target"].astype(str)
)
submission["FVC"] = test_mean
submission["Confidence"] = test_sigma
submission = submission.sort_values(["Weeks_target", "Patient"])[
    ["Patient_Week", "FVC", "Confidence"]
]
submission.head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4216212384.py in <cell line: 0>()
----> 1 submission = df_test[["Patient", "Weeks_target"]].copy()
      2 submission["Patient_Week"] = (
      3     submission["Patient"] + "_" + submission["Weeks_target"].astype(str)
      4 )
      5 submission["FVC"] = test_mean

NameError: name 'df_test' is not defined

## === cell 16
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3990991418.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined

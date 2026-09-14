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

-7.1046058579729685

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import pickle  # kept to preserve original imports; no longer required
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error



## === cell 1
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"




## === cell 2
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_all(20)



## === cell 3
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.75
EPOCH = 250
BATCH_SIZE = 128

NFOLD = 5



## === cell 4
train_raw = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

required_cols = ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
for c in required_cols:
    assert c in train_raw.columns, f"train.csv missing column {c}"
    assert c in raw_test.columns, f"test.csv missing column {c}"

assert list(sample_sub.columns) == ["Patient_Week", "FVC", "Confidence"]



## === cell 5
base = (
    train_raw.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    .rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
)
train = train_raw.merge(base, on="Patient", how="left")
train["Base_week"] = train["Weeks"] - train["Min_week"]

train = train.rename(columns={"SmokingStatus": "SmokingStatus"})



## === cell 6
X_prediction = sample_sub.copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)

raw_test_ren = raw_test.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
X_prediction = X_prediction.merge(
    raw_test_ren[
        ["Patient", "Min_week", "Base_FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ],
    on="Patient",
    how="left",
)
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]



## === cell 7
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder


def standardisation(x, u, s):
    s = 1.0 if (s is None or s == 0) else s
    return (x - u) / s


def normalization(x, ma, mi):
    denom = (ma - mi) if (ma - mi) != 0 else 1.0
    return (x - mi) / denom


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehot = SklearnOneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.standardisation = bool_standard
        self.normalization = bool_normalization
        self.fitted = False

    def fit(self, df):
        d = df.copy()
        self.enc_sex.fit(d["Sex"].astype(str).fillna("Unknown").values)
        self.enc_smok.fit(d["SmokingStatus"].astype(str).fillna("Unknown").values)
        smok_int = self.enc_smok.transform(
            d["SmokingStatus"].astype(str).fillna("Unknown").values
        ).reshape(-1, 1)
        self.onehot.fit(smok_int)

        self.base_week_min = float(d["Base_week"].min())
        self.base_week_max = float(d["Base_week"].max())

        self.base_fvc_min = float(d["Base_FVC"].min())
        self.base_fvc_max = float(d["Base_FVC"].max())

        self.base_percent_min = float(d["Percent"].min())
        self.base_percent_max = float(d["Percent"].max())

        self.age_min = float(d["Age"].min())
        self.age_max = float(d["Age"].max())

        self.weeks_min = float(d["Weeks"].min())
        self.weeks_max = float(d["Weeks"].max())

        self.min_week_min = float(d["Min_week"].min())
        self.min_week_max = float(d["Min_week"].max())

        self.fvc_min = float(d["FVC"].min()) if "FVC" in d.columns else None
        self.fvc_max = float(d["FVC"].max()) if "FVC" in d.columns else None

        self.fitted = True
        return self

    def transform(self, df):
        assert self.fitted, "data_preparation must be fitted first"
        d = df.copy()

        d["Sex"] = self.enc_sex.transform(d["Sex"].astype(str).fillna("Unknown").values)

        smok_int = self.enc_smok.transform(
            d["SmokingStatus"].astype(str).fillna("Unknown").values
        ).reshape(-1, 1)
        smok_oh = self.onehot.transform(smok_int)
        smok_cols = [f"_{c}" for c in self.enc_smok.classes_]
        smok_df = pd.DataFrame(smok_oh, columns=smok_cols, index=d.index).astype(int)

        d = pd.concat([d.drop(columns=["SmokingStatus"]), smok_df], axis=1)

        if self.normalization:
            d["Base_week"] = normalization(
                d["Base_week"].astype(float), self.base_week_max, self.base_week_min
            )
            d["Base_FVC"] = normalization(
                d["Base_FVC"].astype(float), self.base_fvc_max, self.base_fvc_min
            )
            d["Percent"] = normalization(
                d["Percent"].astype(float), self.base_percent_max, self.base_percent_min
            )
            d["Age"] = normalization(d["Age"].astype(float), self.age_max, self.age_min)
            d["Weeks"] = normalization(
                d["Weeks"].astype(float), self.weeks_max, self.weeks_min
            )
            d["Min_week"] = normalization(
                d["Min_week"].astype(float), self.min_week_max, self.min_week_min
            )

        return d

    def __call__(self, df):
        if not self.fitted:
            self.fit(df)
        return self.transform(df)


data_prep = data_preparation(bool_normalization=True, bool_standard=False)
train_prep = data_prep(train)
X_pred_prep = data_prep.transform(X_prediction)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Sex'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1333569650.py in <cell line: 0>()
     99 
    100 data_prep = data_preparation(bool_normalization=True, bool_standard=False)
--> 101 train_prep = data_prep(train)
    102 X_pred_prep = data_prep.transform(X_prediction)
    103 

/tmp/ipykernel_11/1333569650.py in __call__(self, df)
     94     def __call__(self, df):
     95         if not self.fitted:
---> 96             self.fit(df)
     97         return self.transform(df)
     98 

/tmp/ipykernel_11/1333569650.py in fit(self, df)
     25     def fit(self, df):
     26         d = df.copy()
---> 27         self.enc_sex.fit(d["Sex"].astype(str).fillna("Unknown").values)
     28         self.enc_smok.fit(d["SmokingStatus"].astype(str).fillna("Unknown").values)
     29         smok_int = self.enc_smok.transform(

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'Sex'

## === cell 8
SELECTED_COLUMNS = [
    "Weeks",
    "Percent",
    "Age",
    "Sex",
    "Min_week",
    "Base_FVC",
    "Base_week",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]

for c in SELECTED_COLUMNS:
    if c not in train_prep.columns:
        train_prep[c] = 0
    if c not in X_pred_prep.columns:
        X_pred_prep[c] = 0

X_train = train_prep[SELECTED_COLUMNS].astype(np.float64).values
y_train = train_raw["FVC"].astype(np.float64).values  # keep target in ml

X_test = X_pred_prep[SELECTED_COLUMNS].astype(np.float64).values




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2564191537.py in <cell line: 0>()
     15 # Ensure these columns exist; if a smoking category is missing in this environment, create it as zeros
     16 for c in SELECTED_COLUMNS:
---> 17     if c not in train_prep.columns:
     18         train_prep[c] = 0
     19     if c not in X_pred_prep.columns:

NameError: name 'train_prep' is not defined

## === cell 9
def fit_ridge(X, y, alpha=1.0):
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0  # don't regularize intercept
    A = Xb.T @ Xb + alpha * I
    b = Xb.T @ y
    w = np.linalg.solve(A, b)
    return w


def predict_ridge(X, w):
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return Xb @ w


def make_quantile_triplet(mean_pred, sigma):
    mid = mean_pred
    low = mid - 0.5 * sigma
    high = mid + 0.5 * sigma
    return np.vstack([low, mid, high]).T




## === cell 10
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)

pred_oof = np.zeros((X_train.shape[0], 3), dtype=np.float64)
pred_test = np.zeros((X_test.shape[0], 3), dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]

    w = fit_ridge(X_tr, y_tr, alpha=10.0)
    va_mean = predict_ridge(X_va, w)
    tr_mean = predict_ridge(X_tr, w)

    resid = y_tr - tr_mean
    sigma = float(
        np.mean(np.abs(resid)) + 70.0
    )  # include metric floor to avoid overconfidence

    pred_oof[val_idx] = make_quantile_triplet(va_mean, sigma)
    pred_test += make_quantile_triplet(predict_ridge(X_test, w), sigma) / NFOLD



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4280846940.py in <cell line: 0>()
      1 kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)
      2 
----> 3 pred_oof = np.zeros((X_train.shape[0], 3), dtype=np.float64)
      4 pred_test = np.zeros((X_test.shape[0], 3), dtype=np.float64)
      5 

NameError: name 'X_train' is not defined

## === cell 11
sigma_opt = float(mean_absolute_error(y_train.reshape(-1, 1), pred_oof[:, 1]))
unc = pred_oof[:, 2] - pred_oof[:, 0]
sigma_mean = float(np.mean(unc))

X_prediction_out = X_prediction.copy()
X_prediction_out["FVC1"] = pred_test[:, 1]
X_prediction_out["Confidence1"] = pred_test[:, 2] - pred_test[:, 0]

subm = X_prediction_out.copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

subm.loc[~subm["FVC1"].isnull(), "FVC"] = subm.loc[~subm["FVC1"].isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = max(70.0, sigma_opt)
else:
    subm.loc[~subm["FVC1"].isnull(), "Confidence"] = subm.loc[
        ~subm["FVC1"].isnull(), "Confidence1"
    ]
    subm["Confidence"] = subm["Confidence"].clip(lower=70.0)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/926985467.py in <cell line: 0>()
      1 # Calibrate sigma similarly to original intent (use MAE of mid prediction), then create submission fields.
----> 2 sigma_opt = float(mean_absolute_error(y_train.reshape(-1, 1), pred_oof[:, 1]))
      3 unc = pred_oof[:, 2] - pred_oof[:, 0]
      4 sigma_mean = float(np.mean(unc))
      5 

NameError: name 'y_train' is not defined

## === cell 12
otest = raw_test.copy()
for i in range(len(otest)):
    pw = otest.loc[i, "Patient"] + "_" + str(int(otest.loc[i, "Weeks"]))
    subm.loc[subm["Patient_Week"] == pw, "FVC"] = float(otest.loc[i, "FVC"])
    subm.loc[subm["Patient_Week"] == pw, "Confidence"] = (
        70.0  # metric floor; not overconfident
    )



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4049611165.py in <cell line: 0>()
      3 for i in range(len(otest)):
      4     pw = otest.loc[i, "Patient"] + "_" + str(int(otest.loc[i, "Weeks"]))
----> 5     subm.loc[subm["Patient_Week"] == pw, "FVC"] = float(otest.loc[i, "FVC"])
      6     subm.loc[subm["Patient_Week"] == pw, "Confidence"] = (
      7         70.0  # metric floor; not overconfident

NameError: name 'subm' is not defined

## === cell 13
out_path = "submission.csv"
subm[["Patient_Week", "FVC", "Confidence"]].to_csv(out_path, index=False)

check = pd.read_csv(out_path)
assert check.shape[0] == sample_sub.shape[0]
assert list(check.columns) == ["Patient_Week", "FVC", "Confidence"]
print("Wrote", out_path, "with shape", check.shape)
print("Confidence summary:", check["Confidence"].describe())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4018258586.py in <cell line: 0>()
      1 # Write valid submission.
      2 out_path = "submission.csv"
----> 3 subm[["Patient_Week", "FVC", "Confidence"]].to_csv(out_path, index=False)
      4 
      5 # Quick validation

NameError: name 'subm' is not defined

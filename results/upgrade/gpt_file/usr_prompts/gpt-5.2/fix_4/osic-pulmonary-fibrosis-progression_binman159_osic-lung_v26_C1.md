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
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
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

-8.1278

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -18.58739) has done: 'I fix the merge/indexing bug that drops `Patient_Week` during `set_index` by avoiding index name collisions and explicitly preserving the `Patient_Week` column. Then I fix the feature mismatch at inference by ensuring the test feature matrix uses the exact same numeric feature columns as training (and excludes `FVC`). Finally, I correct the scorer direction so GridSearchCV selects params that *maximize* the competition metric, and ensure a valid `submission.csv` is always written with the required columns and ordering.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import typing as tp
import pydicom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor

RANDOM_STATE = 27



## === cell 1
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

train_df["Patient_Week"] = (
    train_df["Patient"].astype(str) + "_" + train_df["Weeks"].astype(str)
)
test_df["Patient_Week"] = (
    test_df["Patient"].astype(str) + "_" + test_df["Weeks"].astype(str)
)



## === cell 2
print("Shape of Training data: ", train_df.shape)
print("Shape of Test data: ", test_df.shape)




## === cell 3
def add_height(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    sex = data["Sex"]
    is_male = (sex == "Male") | (sex == 1)
    denom_male = 27.63 - (0.112 * data["Age"])
    denom_female = 21.78 - (0.101 * data["Age"])
    data["Height"] = np.where(
        is_male, data["FVC"] / denom_male, data["FVC"] / denom_female
    )
    return data


def add_norm(data: pd.DataFrame) -> pd.DataFrame:
    mu = data.mean()
    sd = data.std().replace(0, 1.0)
    return (data - mu) / sd




## === cell 4
train_aug = train_df.copy()
test_aug = test_df.copy()


def _baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["absW"] = tmp["Weeks"].abs()
    base = (
        tmp.sort_values(["Patient", "absW", "Weeks"])
        .groupby("Patient", as_index=False)
        .first()
        .drop(columns=["absW"])
    )
    base = base.rename(
        columns={
            "Weeks": "baseline_Weeks",
            "FVC": "baseline_FVC",
            "Percent": "baseline_Percent",
        }
    )
    return base[["Patient", "baseline_Weeks", "baseline_FVC", "baseline_Percent"]]


train_base = _baseline_rows(train_aug)
test_base = _baseline_rows(test_aug)

train_aug = train_aug.merge(
    train_base, on="Patient", how="left", validate="many_to_one"
)
test_aug = test_aug.merge(test_base, on="Patient", how="left", validate="many_to_one")

train_aug["Weeks_delta"] = train_aug["Weeks"] - train_aug["baseline_Weeks"]
test_aug["Weeks_delta"] = test_aug["Weeks"] - test_aug["baseline_Weeks"]



## === cell 5
df = train_aug.copy()

df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1}).astype("float64")
df["SmokingStatus"] = (
    df["SmokingStatus"]
    .map({"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2})
    .astype("float64")
)

df = add_height(df)

df = df.set_index("Patient")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cols_to_norm = [c for c in num_cols if c not in ["FVC", "Sex"]]
df[cols_to_norm] = add_norm(df[cols_to_norm])

df.head()



## === cell 6
sub_df = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
sub_df.drop(["FVC"], axis=1, inplace=True)
sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
sub_df["pred_Weeks"] = (
    sub_df["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
sub_df.head()



## === cell 7
test_FVC = pd.merge(
    sub_df[["Patient_Week", "Patient", "pred_Weeks"]],
    df.reset_index(),
    how="left",
    on="Patient",
    suffixes=("", "_train"),
)

if "Patient_Week" not in test_FVC.columns:
    raise KeyError("Patient_Week column missing after merge; cannot align predictions.")

test_FVC["pred_Weeks_raw"] = test_FVC["pred_Weeks"].astype(float)

_patient_to_baseweek = test_base.set_index("Patient")["baseline_Weeks"].to_dict()
test_FVC["Weeks_delta_raw"] = test_FVC.apply(
    lambda r: float(r["pred_Weeks"])
    - float(_patient_to_baseweek.get(r["Patient"], 0.0)),
    axis=1,
)

test_FVC = test_FVC.set_index("Patient_Week", drop=True)

for col in ["Patient", "pred_Weeks"]:
    if col in test_FVC.columns:
        test_FVC = test_FVC.drop(columns=[col])

test_FVC.head()



## === cell 8
test_conf = pd.merge(
    sub_df[["Patient_Week", "Patient", "pred_Weeks"]],
    df.reset_index(),
    how="left",
    on="Patient",
    suffixes=("", "_train"),
)

if "Patient_Week" not in test_conf.columns:
    raise KeyError(
        "Patient_Week column missing after merge; cannot align confidence inputs."
    )

test_conf["pred_Weeks_raw"] = test_conf["pred_Weeks"].astype(float)
_patient_to_baseweek = test_base.set_index("Patient")["baseline_Weeks"].to_dict()
test_conf["Weeks_delta_raw"] = test_conf.apply(
    lambda r: float(r["pred_Weeks"])
    - float(_patient_to_baseweek.get(r["Patient"], 0.0)),
    axis=1,
)

test_conf = test_conf.set_index("Patient_Week", drop=True)

for col in ["Patient", "pred_Weeks"]:
    if col in test_conf.columns:
        test_conf = test_conf.drop(columns=[col])

test_conf.head()



## === cell 9
print(test_FVC.shape)
print(test_conf.shape)



## === cell 10
X = df.drop(columns=["FVC"])
y = df["FVC"]

X = X.select_dtypes(include=[np.number])

test_FVC_X = test_FVC.select_dtypes(include=[np.number])

if "pred_Weeks_raw" not in X.columns:
    X = X.copy()
    X["pred_Weeks_raw"] = (
        train_aug.set_index("Patient").loc[X.index, "Weeks"].astype(float).values
    )
if "Weeks_delta_raw" not in X.columns:
    X = X.copy()
    X["Weeks_delta_raw"] = (
        train_aug.set_index("Patient").loc[X.index, "Weeks_delta"].astype(float).values
    )

test_FVC_X = test_FVC_X.reindex(columns=X.columns, fill_value=0.0)

n_val = min(100, max(5, int(0.1 * len(X))))
X_train = X.iloc[:-n_val, :]
y_train = y.iloc[:-n_val]
X_val = X.iloc[-n_val:, :]
y_val = y.iloc[-n_val:]

print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print("Aligned test_FVC_X shape:", test_FVC_X.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/881934008.py in <cell line: 0>()
     11     # For training rows, pred_Weeks_raw is just the row's actual Weeks (raw) for learning progression.
     12     X = X.copy()
---> 13     X["pred_Weeks_raw"] = (
     14         train_aug.set_index("Patient").loc[X.index, "Weeks"].astype(float).values
     15     )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (12360) does not match length of index (1394)

## === cell 11
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
train_df.boxplot("FVC", by="SmokingStatus", ax=axs[0, 0])
train_df.boxplot("Percent", by="SmokingStatus", ax=axs[0, 1])
train_df.boxplot("FVC", by="Sex", ax=axs[1, 0])
train_df.boxplot("Percent", by="Sex", ax=axs[1, 1])
plt.show()



## === cell 12
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177411956430/100.dcm"
if os.path.exists(img):
    ds = pydicom.dcmread(img)
    plt.figure(figsize=(7, 7))
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.show()
else:
    print("Example DICOM not found:", img)



## === cell 13
img_1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177411956430/10.dcm"
img_2 = "../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/10.dcm"

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
if os.path.exists(img_1):
    ds = pydicom.dcmread(img_1)
    ax[0].set_title("Patient 1")
    ax[0].imshow(ds.pixel_array, cmap=plt.cm.bone)
else:
    ax[0].set_title("Patient 1 (missing file)")
    ax[0].axis("off")

if os.path.exists(img_2):
    ds = pydicom.dcmread(img_2)
    ax[1].set_title("Patient 2")
    ax[1].imshow(ds.pixel_array, cmap=plt.cm.bone)
else:
    ax[1].set_title("Patient 2 (missing file)")
    ax[1].axis("off")

plt.show()




## === cell 14
def competition_metric(trueFVC, predFVC, predSTD=100):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    error = np.mean(
        -1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD)
    )
    return error




## === cell 15
class model_selection:
    def __init__(self):
        self.y_pred_FVC = pd.DataFrame()
        self.my_scorer = make_scorer(competition_metric, greater_is_better=True)
        self.best_param = None
        self.scoring = 0

    def xgboost(self, X, y, X_val, y_val, test):
        parameters = {
            "learning_rate": [0.002],
            "n_estimators": [4000],
            "max_depth": [4],
            "reg_alpha": [0.005],
        }
        clf = GridSearchCV(
            XGBRegressor(
                min_child_weight=0,
                gamma=0,
                colsample_bytree=0.7,
                objective="reg:squarederror",
                nthread=-1,
                scale_pos_weight=1,
                subsample=0.7,
                seed=RANDOM_STATE,
                random_state=RANDOM_STATE,
            ),
            param_grid=parameters,
            scoring=self.my_scorer,
        )
        clf.fit(X, y)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_xgb_FVC = clf.predict(test)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test.index, name="Patient_Week"),
                pd.Series(y_pred_xgb_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC, self.best_param, self.scoring

    def lightgbm(self, X, y, X_val, y_val, test):
        parameters = {
            "learning_rate": [0.00005, 0.0001, 0.0005],
            "n_estimators": [2500, 3000, 4000],
            "num_leaves": [1, 2, 3],
        }
        clf = GridSearchCV(
            LGBMRegressor(
                objective="regression",
                max_bin=200,
                bagging_fraction=0.75,
                bagging_freq=5,
                bagging_seed=7,
                feature_fraction=0.2,
                feature_fraction_seed=7,
                verbose=-1,
                random_state=RANDOM_STATE,
            ),
            param_grid=parameters,
            scoring=self.my_scorer,
        )
        clf.fit(X, y)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_lgb_FVC = clf.predict(test)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test.index, name="Patient_Week"),
                pd.Series(y_pred_lgb_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC, self.best_param, self.scoring

    def HuberRegressor(self, X, y, test):
        hbr = HuberRegressor(max_iter=200)
        hbr.fit(X, y)
        y_pred_hbr_FVC = hbr.predict(test)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test.index, name="Patient_Week"),
                pd.Series(y_pred_hbr_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC




## === cell 16
model = model_selection()
output = model.xgboost(X_train, y_train, X_val, y_val, test_FVC_X)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1802059631.py in <cell line: 0>()
      1 model = model_selection()
----> 2 output = model.xgboost(X_train, y_train, X_val, y_val, test_FVC_X)
      3 

NameError: name 'X_train' is not defined

## === cell 17
print(output[1])  # best params
print(output[2])  # validation score (competition metric, higher is better)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3165798338.py in <cell line: 0>()
----> 1 print(output[1])  # best params
      2 print(output[2])  # validation score (competition metric, higher is better)
      3 

NameError: name 'output' is not defined

## === cell 18
y_pred_FVC = output[0].copy()
if "Patient_Week" not in y_pred_FVC.columns:
    y_pred_FVC.rename(columns={y_pred_FVC.columns[0]: "Patient_Week"}, inplace=True)
if "FVC" not in y_pred_FVC.columns:
    y_pred_FVC.rename(columns={y_pred_FVC.columns[-1]: "FVC"}, inplace=True)

y_pred_FVC.head()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/526527965.py in <cell line: 0>()
----> 1 y_pred_FVC = output[0].copy()
      2 if "Patient_Week" not in y_pred_FVC.columns:
      3     y_pred_FVC.rename(columns={y_pred_FVC.columns[0]: "Patient_Week"}, inplace=True)
      4 if "FVC" not in y_pred_FVC.columns:
      5     y_pred_FVC.rename(columns={y_pred_FVC.columns[-1]: "FVC"}, inplace=True)

NameError: name 'output' is not defined

## === cell 19
val_pred = output_model_pred = None
try:
    best_est = GridSearchCV(
        XGBRegressor(
            min_child_weight=0,
            gamma=0,
            colsample_bytree=0.7,
            objective="reg:squarederror",
            nthread=-1,
            scale_pos_weight=1,
            subsample=0.7,
            seed=RANDOM_STATE,
            random_state=RANDOM_STATE,
        ),
        param_grid={
            "learning_rate": [output[1]["learning_rate"]],
            "n_estimators": [output[1]["n_estimators"]],
            "max_depth": [output[1]["max_depth"]],
            "reg_alpha": [output[1]["reg_alpha"]],
        },
        scoring=make_scorer(competition_metric, greater_is_better=True),
    )
    best_est.fit(X_train, y_train)
    val_pred = best_est.predict(X_val)
except Exception:
    val_pred = None

if val_pred is None:
    sigma_est = 150.0
else:
    mae = float(np.mean(np.abs(y_val.values - val_pred)))
    sigma_est = max(70.0, mae * np.sqrt(2))

submission = sub_df[["Patient_Week", "Confidence"]].merge(
    y_pred_FVC[["Patient_Week", "FVC"]], how="left", on="Patient_Week"
)

submission["FVC"] = submission["FVC"].fillna(train_df["FVC"].median())

submission["Confidence"] = float(sigma_est)
submission["Confidence"] = submission["Confidence"].clip(lower=70)

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.head()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3094772260.py in <cell line: 0>()
     38 
     39 submission = sub_df[["Patient_Week", "Confidence"]].merge(
---> 40     y_pred_FVC[["Patient_Week", "FVC"]], how="left", on="Patient_Week"
     41 )
     42 

NameError: name 'y_pred_FVC' is not defined

## === cell 20
submission.to_csv("./submission.csv", index=False)
print("Wrote ./submission.csv with shape:", submission.shape)
print("Estimated constant Confidence used:", float(sigma_est))
print(submission.head())

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3908688580.py in <cell line: 0>()
----> 1 submission.to_csv("./submission.csv", index=False)
      2 print("Wrote ./submission.csv with shape:", submission.shape)
      3 print("Estimated constant Confidence used:", float(sigma_est))
      4 print(submission.head())

NameError: name 'submission' is not defined

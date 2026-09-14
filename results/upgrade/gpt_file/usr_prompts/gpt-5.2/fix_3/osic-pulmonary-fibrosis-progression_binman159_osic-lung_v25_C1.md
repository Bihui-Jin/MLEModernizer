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

-8.5999

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import typing as tp
import pydicom
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor

RANDOM_STATE = 42



## === cell 1
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

df_all = pd.concat([train_df, test_df], ignore_index=True)
df_all["Patient_Week"] = (
    df_all["Patient"].astype(str) + "_" + df_all["Weeks"].astype(str)
)



## === cell 2
print("Shape of Training data: ", train_df.shape)
print("Shape of Test data: ", test_df.shape)




## === cell 3
def add_height(data: pd.DataFrame) -> None:
    if "Sex" not in data.columns:
        raise KeyError("Expected column 'Sex' to exist before add_height().")
    denom_f = 21.78 - (0.101 * data["Age"])
    denom_m = 27.63 - (0.112 * data["Age"])
    is_female = (data["Sex"] == 0) | (data["Sex"] == "Female")
    data["Height"] = np.where(is_female, data["FVC"] / denom_f, data["FVC"] / denom_m)


def add_norm(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return out
    std = out[num_cols].std(ddof=0).replace(0, 1.0)
    out[num_cols] = (out[num_cols] - out[num_cols].mean()) / std
    return out




## === cell 4
df = train_df.copy()

df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1})
df["SmokingStatus"] = df["SmokingStatus"].map(
    {"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2}
)

df["Patient_Week"] = df["Patient"].astype(str) + "_" + df["Weeks"].astype(str)
df = df.drop("Patient_Week", axis=1)
df = df.set_index("Patient")

add_height(df)

feature_cols_to_norm = df.columns[~df.columns.isin(["FVC", "Percent"])]
df[feature_cols_to_norm] = add_norm(df[feature_cols_to_norm])

df.head()



## === cell 5
sub_df = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
sub_df.drop(["FVC", "Confidence"], axis=1, inplace=True)

sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.rsplit("_", 1)[0])
sub_df["pred_Weeks"] = (
    sub_df["Patient_Week"].apply(lambda x: x.rsplit("_", 1)[1]).astype(int)
)

sub_df.head()



## === cell 6
test_FVC = pd.merge(
    sub_df[["Patient_Week", "Patient", "pred_Weeks"]],
    df.reset_index(),
    how="left",
    on="Patient",
)

test_FVC = test_FVC.rename(columns={"pred_Weeks": "Weeks"})
if "FVC" in test_FVC.columns:
    test_FVC = test_FVC.drop(["FVC"], axis=1)

test_FVC = test_FVC.set_index("Patient_Week")
test_FVC = test_FVC.drop(columns=["Patient"], errors="ignore")
test_FVC = test_FVC.loc[:, ~test_FVC.columns.duplicated()]

cols_norm = test_FVC.columns[~test_FVC.columns.isin(["Sex", "Percent"])]
test_FVC[cols_norm] = add_norm(test_FVC[cols_norm])

test_FVC.head()



## === cell 7
test_conf = pd.merge(
    sub_df[["Patient_Week", "Patient", "pred_Weeks"]],
    df.reset_index(),
    how="left",
    on="Patient",
)
test_conf = test_conf.rename(columns={"pred_Weeks": "Weeks"})
if "Percent" in test_conf.columns:
    test_conf = test_conf.drop(["Percent"], axis=1)

test_conf = test_conf.set_index("Patient_Week")
test_conf = test_conf.drop(columns=["Patient"], errors="ignore")
test_conf = test_conf.loc[:, ~test_conf.columns.duplicated()]

cols_norm = test_conf.columns[~test_conf.columns.isin(["Sex", "FVC"])]
test_conf[cols_norm] = add_norm(test_conf[cols_norm])

test_conf.head()



## === cell 8
print(test_FVC.shape)
print(test_conf.shape)



## === cell 9
X = df.iloc[:, df.columns != "FVC"]
y = df["FVC"]

X_train = X[:-5]
y_train = y[:-5]
X_val = X[-5:]
y_val = y[-5:]

print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)



## === cell 10
X_conf = df.iloc[:, df.columns != "Percent"]
y_conf = df["Percent"]

X_train_conf = X_conf[:-5]
y_train_conf = y_conf[:-5]
X_val_conf = X_conf[-5:]
y_val_conf = y_conf[-5:]

print(X_conf.shape)
print(y_conf.shape)
print(X_train_conf.shape)
print(y_train_conf.shape)
print(X_val_conf.shape)
print(y_val_conf.shape)



## === cell 11
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
train_df.boxplot("FVC", by="SmokingStatus", ax=axs[0, 0])
train_df.boxplot("Percent", by="SmokingStatus", ax=axs[0, 1])
train_df.boxplot("FVC", by="Sex", ax=axs[1, 0])
train_df.boxplot("Percent", by="Sex", ax=axs[1, 1])

plt.show()



## === cell 12
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"
if os.path.exists(img):
    ds = pydicom.dcmread(img)
    plt.figure(figsize=(7, 7))
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.show()



## === cell 13
img_1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"
img_2 = "../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/10.dcm"

if os.path.exists(img_1) and os.path.exists(img_2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ds = pydicom.dcmread(img_1)
    ax[0].set_title("Patient 1: Ex-Smoker")
    ax[0].imshow(ds.pixel_array, cmap=plt.cm.bone)

    ds = pydicom.dcmread(img_2)
    ax[1].set_title("Patient 2: Never smoked")
    ax[1].imshow(ds.pixel_array, cmap=plt.cm.bone)

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
        base = XGBRegressor(
            min_child_weight=0,
            gamma=0,
            colsample_bytree=0.7,
            objective="reg:squarederror",
            nthread=-1,
            scale_pos_weight=1,
            subsample=0.7,
            seed=27,
            random_state=RANDOM_STATE,
        )
        clf = GridSearchCV(
            base, param_grid=parameters, scoring=self.my_scorer, cv=3, refit=True
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
            cv=3,
            refit=True,
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
        y_pred = hbr.predict(test)
        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test.index, name="Patient_Week"),
                pd.Series(y_pred, name="Confidence"),
            ],
            axis=1,
        )
        return self.y_pred_FVC




## === cell 16
test_FVC_aligned = test_FVC.loc[:, ~test_FVC.columns.duplicated()].reindex(
    columns=X.columns
)
test_FVC_aligned = test_FVC_aligned.fillna(test_FVC_aligned.median(numeric_only=True))

model = model_selection()
output = model.xgboost(X_train, y_train, X_val, y_val, test_FVC_aligned)



## === cell 17
print(output[1])  # best params
print(output[2])  # score on validation (per competition_metric scorer)



## === cell 18
y_pred_FVC = output[0].copy()
y_pred_FVC.head()



## === cell 19
test_conf_aligned = test_conf.loc[:, ~test_conf.columns.duplicated()].reindex(
    columns=X_conf.columns
)
test_conf_aligned = test_conf_aligned.fillna(
    test_conf_aligned.median(numeric_only=True)
)
test_conf_aligned.head()



## === cell 20
model = model_selection()
y_pred_conf = model.HuberRegressor(X_train_conf, y_train_conf, test_conf_aligned)
y_pred_conf.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1824565247.py in <cell line: 0>()
      1 model = model_selection()
----> 2 y_pred_conf = model.HuberRegressor(X_train_conf, y_train_conf, test_conf_aligned)
      3 y_pred_conf.head()
      4 

/tmp/ipykernel_11/3399218054.py in HuberRegressor(self, X, y, test)
     81         hbr = HuberRegressor(max_iter=200)
     82         hbr.fit(X, y)
---> 83         y_pred = hbr.predict(test)
     84         self.y_pred_FVC = pd.concat(
     85             [

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    335         check_is_fitted(self)
    336 
--> 337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
    338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input X contains NaN.
HuberRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 21
submission = sub_df[["Patient_Week"]].merge(y_pred_FVC, how="left", on="Patient_Week")
submission = submission.merge(y_pred_conf, how="left", on="Patient_Week")

submission = submission[["Patient_Week", "FVC", "Confidence"]]

submission["FVC"] = submission["FVC"].fillna(train_df["FVC"].median())
submission["Confidence"] = submission["Confidence"].fillna(train_df["Percent"].median())

submission["Confidence"] = submission["Confidence"].abs()
submission.loc[submission["Confidence"] < 1.0, "Confidence"] = 1.0

submission.head()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3004843048.py in <cell line: 0>()
      1 submission = sub_df[["Patient_Week"]].merge(y_pred_FVC, how="left", on="Patient_Week")
----> 2 submission = submission.merge(y_pred_conf, how="left", on="Patient_Week")
      3 
      4 submission = submission[["Patient_Week", "FVC", "Confidence"]]
      5 

NameError: name 'y_pred_conf' is not defined

## === cell 22
submission.to_csv("./submission.csv", index=False)
print("Wrote submission to ./submission.csv")
print(submission.shape)
print(submission.columns.tolist())

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame must have a 'Confidence' column.

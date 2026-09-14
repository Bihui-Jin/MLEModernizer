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

-7.0154

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

np.random.seed(42)

max_print = 50
printed = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if printed >= max_print:
            break
        print(os.path.join(dirname, filename))
        printed += 1



## === cell 1
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train_df




## === cell 2
def feature_engineer(data):
    """
    method to feature engineer any df, train or test
    """
    df = data.copy()

    df["FirstWeek"] = df.groupby("Patient")["Weeks"].transform("min")

    first_fvc = (
        df.loc[df["Weeks"] == df["FirstWeek"]][["Patient", "FVC"]]
        .groupby("Patient")
        .first()  # some patients have multiple measurements in same week - get the first
        .reset_index()
        .rename(columns={"FVC": "FirstFVC"})
    )

    df = df.merge(first_fvc, on="Patient")  # add FirstFVC column
    df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]

    def calculate_height(row):
        if row["Sex"] == "Male":
            return row["FirstFVC"] / (27.63 - 0.112 * row["Age"])
        else:
            return row["FirstFVC"] / (21.78 - 0.101 * row["Age"])

    df["Height"] = df.apply(calculate_height, axis=1)

    return df


feature_engineer(train_df)



## === cell 3
from sklearn.base import BaseEstimator, TransformerMixin


class MyFeatureEngineerer(BaseEstimator, TransformerMixin):
    """
    Class wrapper around feature_engineer so we can:
    - fit(): compute patient-level baseline features (FirstFVC, FirstWeek, Height, etc.)
    - transform(): apply to a possibly expanded (patient, week) grid

    IMPORTANT FIX for score validity:
    Fit this on TRAIN (full history) and reuse for TEST so FirstWeek/FirstFVC
    are consistent with the model's expectations.
    """

    def __init__(self):
        self.df_ = None

    def fit(self, X, y=None):
        self.df_ = feature_engineer(X)
        return self

    def transform(self, X):
        if self.df_ is None:
            raise ValueError("MyFeatureEngineerer must be fit() before transform().")

        if len(X) != len(self.df_):
            drop = X.columns.values
            base = self.df_.drop(drop, axis=1, errors="ignore").join(
                self.df_["Patient"]
            )
            base = base.groupby("Patient", as_index=False).first()
            df = X.merge(base, on="Patient", how="left")
            df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]
        else:
            df = self.df_
        return df




## === cell 4
def transformed_col_names(col_trans):
    """
    Ensure transformed array column names match ColumnTransformer output
    under sklearn>=1.0 by using get_feature_names_out().
    """
    try:
        return list(col_trans.get_feature_names_out())
    except Exception:
        new_colnames = []
        for name, trans, cols in col_trans.transformers_:
            if trans == "drop":
                continue
            if trans == "passthrough":
                if isinstance(cols, (list, tuple, np.ndarray)):
                    new_colnames.extend(list(cols))
                else:
                    new_colnames.append(str(cols))
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    new_colnames.extend(list(trans.get_feature_names_out(cols)))
                    continue
                except Exception:
                    new_colnames.extend([f"{name}__{c}" for c in cols])
            else:
                new_colnames.extend([f"{name}__{c}" for c in cols])
        return new_colnames




## === cell 5
from sklearn.base import BaseEstimator, TransformerMixin


class ParamMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    custom minmax scaler where min and max are not based on data,
    but are passed in as parameters
    """

    def __init__(self, min_val=0, max_val=100):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = (X - self.min_val) / (self.max_val - self.min_val)
        return data




## === cell 6
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

passthru_features = ["Patient", "FVC"]
onehot_features = ["Sex", "SmokingStatus"]
hundred_features = ["Percent", "Age"]
minmax_features = ["FirstFVC", "FirstWeek", "WeeksPassed", "Height"]

oh_enc = OneHotEncoder(sparse_output=False, drop="if_binary", handle_unknown="ignore")
hundred_minmax = ParamMinMaxScaler()
week_minmax = ParamMinMaxScaler(min_val=-12, max_val=133)
minmax = MinMaxScaler()

col_trans = ColumnTransformer(
    [
        ("original", "passthrough", passthru_features),
        ("week_minmax", week_minmax, ["Weeks"]),
        ("hundred_minmax", hundred_minmax, hundred_features),
        ("minmax", minmax, minmax_features),
        ("onehot", oh_enc, onehot_features),
    ],
    remainder="passthrough",
    sparse_threshold=0,
)



## === cell 7
eng = MyFeatureEngineerer()
train_df_fe = eng.fit_transform(train_df)

new_arr = col_trans.fit_transform(train_df_fe)
train_df = pd.DataFrame(new_arr, columns=transformed_col_names(col_trans))

train_df["FVC"] = pd.to_numeric(train_df["FVC"], errors="coerce")
train_df



## === cell 8
TARGET_COL = "FVC"
ID_COL = "Patient"

drop_features = [ID_COL, TARGET_COL]
X_train = train_df.drop(columns=drop_features, errors="ignore")
y_train = train_df[TARGET_COL].astype(float)

groups = train_df[ID_COL].values

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)



## === cell 9
"""
TensorFlow custom losses are not used with sklearn LinearRegression.
Keep the notebook runnable by leaving these as stubs.
"""


def laplace_log_score(**kwargs):
    raise NotImplementedError("Not used in this sklearn-based solution.")


def pinball_qloss(quantiles):
    raise NotImplementedError("Not used in this sklearn-based solution.")


def weighted_loss(weights, loss_functions):
    raise NotImplementedError("Not used in this sklearn-based solution.")


def mloss():
    raise NotImplementedError("Not used in this sklearn-based solution.")




## === cell 10
from sklearn.linear_model import LinearRegression


def make_model():
    return LinearRegression()




## === cell 11
train_df



## === cell 12
model = make_model()
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_train[:10]



## === cell 13
from sklearn.model_selection import RandomizedSearchCV



## === cell 14
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import make_scorer, mean_absolute_error

NFOLDS = 6
gkf = GroupKFold(n_splits=NFOLDS)

scorer = make_scorer(mean_absolute_error, greater_is_better=False)

cv_scores = cross_val_score(
    model, X_train, y_train, cv=gkf, groups=groups, scoring=scorer
)
mae_scores = -cv_scores
print("CV MAE:", mae_scores)

confidence = float(max(70.0, np.mean(mae_scores)))
print("Confidence used:", confidence)



## === cell 15
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.bar(np.arange(len(X_train.columns)), model.coef_)
plt.xticks(np.arange(len(X_train.columns)), X_train.columns.values, rotation=70)
plt.tight_layout()



## === cell 16
import random

p = random.choice(train_df["Patient"].unique())
mask = train_df["Patient"] == p

week_col = "week_minmax__Weeks" if "week_minmax__Weeks" in train_df.columns else None

cols = [TARGET_COL]
if week_col is not None:
    cols = [week_col] + cols

temp_df = train_df.loc[mask, cols].copy()
temp_df = temp_df.join(
    pd.Series(pred_train, name="FVC_pred", index=train_df.index)
    .loc[mask]
    .reset_index(drop=True)
)

if week_col is not None:
    temp_df.plot(x=week_col, y=[TARGET_COL, "FVC_pred"], title=p)
else:
    temp_df[[TARGET_COL, "FVC_pred"]].plot(title=p)



## === cell 17
input_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
input_df



## === cell 18

input_df2 = input_df.drop(["FVC", "Weeks"], axis=1)

all_weeks = pd.DataFrame({"Weeks": np.arange(-12, 134, dtype=int)})

patients = input_df["Patient"].unique()
patient_weeks = pd.DataFrame(
    {
        "Patient": np.repeat(patients, len(all_weeks)),
        "Weeks": np.tile(all_weeks["Weeks"].values, len(patients)),
    }
)

temp_df = patient_weeks.merge(input_df2, on="Patient", how="left")
new_df = eng.transform(temp_df)
new_df.head()



## === cell 19
new_df = new_df.copy()
new_df["FVC"] = 0.0

new_arr_test = col_trans.transform(new_df)
test_df = pd.DataFrame(new_arr_test, columns=transformed_col_names(col_trans))

test_df["FVC"] = pd.to_numeric(test_df["FVC"], errors="coerce")
test_df.head()



## === cell 20
X_test = test_df.drop(columns=drop_features, errors="ignore")
print("X_test shape:", X_test.shape)
X_test.head()



## === cell 21
pred = model.predict(X_test)
pred[:10]



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2390343684.py in <cell line: 0>()
----> 1 pred = model.predict(X_test)
      2 pred[:10]
      3 

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
LinearRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 22
sub_df = patient_weeks.copy()
sub_df["FVC"] = pred
sub_df["Patient_Week"] = (
    sub_df["Patient"].astype(str) + "_" + sub_df["Weeks"].astype(str)
)
sub_df["Confidence"] = confidence
sub_df.head()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3912118492.py in <cell line: 0>()
      1 sub_df = patient_weeks.copy()
----> 2 sub_df["FVC"] = pred
      3 sub_df["Patient_Week"] = (
      4     sub_df["Patient"].astype(str) + "_" + sub_df["Weeks"].astype(str)
      5 )

NameError: name 'pred' is not defined

## === cell 23
sample_sub = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

submission = sub_df[["Patient_Week", "FVC", "Confidence"]].copy()
submission["FVC"] = submission["FVC"].round().astype(int)
submission["Confidence"] = submission["Confidence"].astype(float)

submission = sample_sub[["Patient_Week"]].merge(
    submission, on="Patient_Week", how="left"
)

submission["FVC"] = (
    submission["FVC"].fillna(submission["FVC"].median()).round().astype(int)
)
submission["Confidence"] = submission["Confidence"].fillna(confidence).astype(float)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/289984473.py in <cell line: 0>()
      4 )
      5 
----> 6 submission = sub_df[["Patient_Week", "FVC", "Confidence"]].copy()
      7 submission["FVC"] = submission["FVC"].round().astype(int)
      8 submission["Confidence"] = submission["Confidence"].astype(float)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['Patient_Week', 'FVC', 'Confidence'], dtype='object')] are in the [columns]"

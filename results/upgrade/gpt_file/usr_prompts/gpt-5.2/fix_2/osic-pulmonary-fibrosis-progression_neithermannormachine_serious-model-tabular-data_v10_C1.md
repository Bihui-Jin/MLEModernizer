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

-6.8685

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



## === cell 1
pass



## === cell 2
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train_df




## === cell 3
def feature_engineer(data):
    """
    Feature engineering for train/test DataFrames.
    """
    df = data.copy()

    df["FirstWeek"] = df.groupby("Patient")["Weeks"].transform("min")

    first_fvc = (
        df.loc[df["Weeks"] == df["FirstWeek"]][["Patient", "FVC"]]
        .groupby("Patient")
        .first()
        .reset_index()
        .rename(columns={"FVC": "FirstFVC"})
    )
    df = df.merge(first_fvc, on="Patient", how="left")

    df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]

    def calculate_height(row):
        if row["Sex"] == "Male":
            return row["FirstFVC"] / (27.63 - 0.112 * row["Age"])
        else:
            return row["FirstFVC"] / (21.78 - 0.101 * row["Age"])

    df["Height"] = df.apply(calculate_height, axis=1)
    return df


feature_engineer(train_df)



## === cell 4
from sklearn.base import BaseEstimator, TransformerMixin


class MyFeatureEngineerer(BaseEstimator, TransformerMixin):
    """
    Stores patient-level baseline-derived features at fit time,
    and can be used to transform a modified frame with extra weeks.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Can only use this estimator on Pandas DataFrame")
        self.df_ = feature_engineer(X)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Can only use this estimator on Pandas DataFrame")

        if len(X) != len(self.df_):
            drop = X.columns.values
            df = self.df_.drop(drop, axis=1, errors="ignore").join(
                self.df_[["Patient"]]
            )
            df = X.merge(df, on="Patient", how="left")
            df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]
        else:
            df = self.df_.copy()
        return df




## === cell 5
from sklearn.base import BaseEstimator, TransformerMixin


class ParamMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Custom min-max scaler with fixed min/max passed as parameters.
    """

    def __init__(self, min_val=0, max_val=100):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.min_val) / (self.max_val - self.min_val)




## === cell 6
def transformed_col_names(col_trans, input_features):
    try:
        return list(col_trans.get_feature_names_out(input_features=input_features))
    except Exception:
        names = []
        for name, trans, cols in col_trans.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if trans == "passthrough":
                if isinstance(cols, (list, tuple, np.ndarray)):
                    names.extend(list(cols))
                else:
                    names.append(str(cols))
            else:
                if isinstance(cols, (list, tuple, np.ndarray)):
                    names.extend(list(cols))
                else:
                    names.append(str(cols))
        return names




## === cell 7
from sklearn_pandas import DataFrameMapper



## === cell 8
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
    transformers=[
        ("original", "passthrough", passthru_features),
        ("week_minmax", week_minmax, ["Weeks"]),
        ("hundred_minmax", hundred_minmax, hundred_features),
        ("minmax", minmax, minmax_features),
        ("onehot", oh_enc, onehot_features),
    ],
    remainder="drop",
    sparse_threshold=0,
)



## === cell 9
pass



## === cell 10
eng_train = MyFeatureEngineerer().fit(train_df)
train_fe = eng_train.transform(train_df)

new_arr = col_trans.fit_transform(train_fe)

feature_names = transformed_col_names(col_trans, input_features=train_fe.columns)
train_df = pd.DataFrame(new_arr, columns=feature_names)

if "original__Patient" in train_df.columns:
    train_df["original__Patient"] = train_df["original__Patient"].astype(str)
if "original__FVC" in train_df.columns:
    train_df["original__FVC"] = pd.to_numeric(
        train_df["original__FVC"], errors="coerce"
    )

train_df




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2890777916.py in <cell line: 0>()
      6 
      7 feature_names = transformed_col_names(col_trans, input_features=train_fe.columns)
----> 8 train_df = pd.DataFrame(new_arr, columns=feature_names)
      9 
     10 # Ensure types are numeric where expected (and keep Patient as string)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    825                 )
    826             else:
--> 827                 mgr = ndarray_to_mgr(
    828                     data,
    829                     index,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in ndarray_to_mgr(values, index, columns, dtype, copy, typ)
    334     )
    335 
--> 336     _check_values_indices_shape_match(values, index, columns)
    337 
    338     if typ == "array":

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _check_values_indices_shape_match(values, index, columns)
    418         passed = values.shape
    419         implied = (len(index), len(columns))
--> 420         raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")
    421 
    422 

ValueError: Shape of passed values is (1394, 13), indices imply (1394, 11)

## === cell 11
def laplace_metric_np(y_true, y_pred_fvc, sigma):
    """
    Returns mean metric (higher is better) for the OSIC competition.
    """
    sigma_clipped = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred_fvc), 1000.0)
    sq2 = np.sqrt(2.0)
    metric = -(sq2 * delta) / sigma_clipped - np.log(sq2 * sigma_clipped)
    return float(np.mean(metric))


def temp_laplace_scorer(estimator, X, y_true, confidence):
    """
    sklearn scorer signature: (estimator, X, y) -> score
    """
    y_pred = estimator.predict(X)
    return laplace_metric_np(
        np.asarray(y_true, dtype=np.float32),
        np.asarray(y_pred, dtype=np.float32),
        sigma=float(confidence),
    )




## === cell 12
from sklearn.linear_model import LinearRegression


def make_model():
    return LinearRegression()




## === cell 13
train_df



## === cell 14
model = make_model()

drop_features = [
    "original__Patient",
    "original__FVC",
    "week_minmax__Weeks",
    "hundred_minmax__Percent",
]

X_train = train_df.drop(columns=drop_features, errors="ignore")
y_train = train_df["original__FVC"].astype(float)

model.fit(X_train, y_train)
X_train



## --- ERROR in cell 14, traceback:
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

KeyError: 'original__FVC'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1111732433.py in <cell line: 0>()
     11 
     12 X_train = train_df.drop(columns=drop_features, errors="ignore")
---> 13 y_train = train_df["original__FVC"].astype(float)
     14 
     15 model.fit(X_train, y_train)

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

KeyError: 'original__FVC'

## === cell 15
from sklearn.model_selection import RandomizedSearchCV

pass



## === cell 16
pass



## === cell 17
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import make_scorer

NFOLDS = 6
gkf = GroupKFold(n_splits=NFOLDS)

groups = train_df["original__Patient"].values

conf = np.arange(100, 401, 5)
conf_df = pd.DataFrame(index=conf, columns=["mean score", "std score"], dtype=float)
conf_df.index.name = "Confidence"

for c in conf:
    scorer = make_scorer(
        lambda yt, yp: laplace_metric_np(
            np.asarray(yt, dtype=np.float32),
            np.asarray(yp, dtype=np.float32),
            sigma=float(c),
        ),
        greater_is_better=True,
    )
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=gkf, groups=groups, scoring=scorer
    )
    conf_df.loc[c, "mean score"] = float(np.mean(cv_scores))
    conf_df.loc[c, "std score"] = float(np.std(cv_scores))

num_std = 2.3
conf_df["worst case"] = (
    conf_df["mean score"] - num_std * conf_df["std score"]
)  # conservative lower bound
conf_df



## --- ERROR in cell 17, traceback:
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

KeyError: 'original__Patient'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3586338956.py in <cell line: 0>()
      5 gkf = GroupKFold(n_splits=NFOLDS)
      6 
----> 7 groups = train_df["original__Patient"].values
      8 
      9 conf = np.arange(100, 401, 5)

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

KeyError: 'original__Patient'

## === cell 18
best = conf_df.nlargest(10, columns=["worst case"], keep="all")
best_fmt = best.copy()
for col in best_fmt.columns:
    best_fmt[col] = best_fmt[col].map(lambda x: f"{x:,.6f}")
best_fmt



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2769041794.py in <cell line: 0>()
      1 # Higher is better, so take largest worst-case
----> 2 best = conf_df.nlargest(10, columns=["worst case"], keep="all")
      3 best_fmt = best.copy()
      4 for col in best_fmt.columns:
      5     best_fmt[col] = best_fmt[col].map(lambda x: f"{x:,.6f}")

NameError: name 'conf_df' is not defined

## === cell 19
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.bar(X_train.columns.values, model.coef_)
plt.xticks(rotation=70)
plt.tight_layout()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1154932871.py in <cell line: 0>()
      2 
      3 plt.figure(figsize=(10, 4))
----> 4 plt.bar(X_train.columns.values, model.coef_)
      5 plt.xticks(rotation=70)
      6 plt.tight_layout()

AttributeError: 'LinearRegression' object has no attribute 'coef_'

## === cell 20
pred_train = model.predict(X_train)
pred_train[:10]



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1725652231.py in <cell line: 0>()
----> 1 pred_train = model.predict(X_train)
      2 pred_train[:10]
      3 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    333 
    334     def _decision_function(self, X):
--> 335         check_is_fitted(self)
    336 
    337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 21
import random

p = random.choice(train_df["original__Patient"].unique())
mask = train_df["original__Patient"] == p

temp_df = pd.DataFrame(
    {
        "Weeks": (train_fe.loc[train_fe["Patient"] == p, "Weeks"].values),
        "FVC": (train_fe.loc[train_fe["Patient"] == p, "FVC"].values),
        "FVC_pred": pd.Series(pred_train, index=train_df.index).loc[mask].values,
    }
)

temp_df = temp_df.sort_values("Weeks")
ax = temp_df.plot(
    x="Weeks", y=["FVC", "FVC_pred"], title=f"Patient {p}", figsize=(7, 4)
)
plt.tight_layout()



## --- ERROR in cell 21, traceback:
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

KeyError: 'original__Patient'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1161289377.py in <cell line: 0>()
      1 import random
      2 
----> 3 p = random.choice(train_df["original__Patient"].unique())
      4 mask = train_df["original__Patient"] == p
      5 

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

KeyError: 'original__Patient'

## === cell 22
from sklearn.pipeline import Pipeline

pass



## === cell 23
input_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
input_df



## === cell 24
eng = MyFeatureEngineerer().fit(input_df)

all_weeks = np.arange(-12, 134, dtype=int)
patients = input_df["Patient"].unique()

patient_weeks = pd.DataFrame(
    {
        "Patient": np.repeat(patients, len(all_weeks)),
        "Weeks": np.tile(all_weeks, len(patients)),
    }
)

input_df2 = input_df.drop(["FVC", "Weeks"], axis=1)
temp_df = patient_weeks.merge(input_df2, on="Patient", how="left")

new_df = eng.transform(temp_df)
new_df



## === cell 25
new_df = new_df.copy()
new_df["FVC"] = 0.0

test_arr = col_trans.transform(new_df)
test_feature_names = transformed_col_names(col_trans, input_features=new_df.columns)
test_df = pd.DataFrame(test_arr, columns=test_feature_names)

if "original__Patient" in test_df.columns:
    test_df["original__Patient"] = test_df["original__Patient"].astype(str)

test_df.head()



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1987568052.py in <cell line: 0>()
      5 test_arr = col_trans.transform(new_df)
      6 test_feature_names = transformed_col_names(col_trans, input_features=new_df.columns)
----> 7 test_df = pd.DataFrame(test_arr, columns=test_feature_names)
      8 
      9 # ensure Patient is str

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    825                 )
    826             else:
--> 827                 mgr = ndarray_to_mgr(
    828                     data,
    829                     index,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in ndarray_to_mgr(values, index, columns, dtype, copy, typ)
    334     )
    335 
--> 336     _check_values_indices_shape_match(values, index, columns)
    337 
    338     if typ == "array":

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _check_values_indices_shape_match(values, index, columns)
    418         passed = values.shape
    419         implied = (len(index), len(columns))
--> 420         raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")
    421 
    422 

ValueError: Shape of passed values is (2628, 13), indices imply (2628, 11)

## === cell 26
X_test = test_df.drop(columns=drop_features, errors="ignore")
X_test.head()



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2482551021.py in <cell line: 0>()
----> 1 X_test = test_df.drop(columns=drop_features, errors="ignore")
      2 X_test.head()
      3 

NameError: name 'test_df' is not defined

## === cell 27
pred = model.predict(X_test)
pred[:10]



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2390343684.py in <cell line: 0>()
----> 1 pred = model.predict(X_test)
      2 pred[:10]
      3 

NameError: name 'X_test' is not defined

## === cell 28
sub_df = patient_weeks.copy()
sub_df["FVC"] = pred.astype(float)
sub_df.head()



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3286062149.py in <cell line: 0>()
      1 sub_df = patient_weeks.copy()
----> 2 sub_df["FVC"] = pred.astype(float)
      3 sub_df.head()
      4 

NameError: name 'pred' is not defined

## === cell 29
plt.figure(figsize=(17, 10))
for i, (patient, frame) in enumerate(sub_df.groupby("Patient")):
    ax = plt.subplot(3, 6, i + 1)
    frame.sort_values("Weeks")[["Weeks", "FVC"]].plot(
        x="Weeks", y="FVC", title=patient, ax=ax, legend=False
    )
plt.tight_layout()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2290760102.py in <cell line: 0>()
      2 for i, (patient, frame) in enumerate(sub_df.groupby("Patient")):
      3     ax = plt.subplot(3, 6, i + 1)
----> 4     frame.sort_values("Weeks")[["Weeks", "FVC"]].plot(
      5         x="Weeks", y="FVC", title=patient, ax=ax, legend=False
      6     )

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
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['FVC'] not in index"

## === cell 30
sub_df["Patient_Week"] = sub_df["Patient"] + "_" + sub_df["Weeks"].astype(str)

best_conf = int(conf_df["worst case"].idxmax())
sub_df["Confidence"] = float(best_conf)

sub_df[["Patient_Week", "FVC", "Confidence"]].head()



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2277741352.py in <cell line: 0>()
      2 
      3 # Pick confidence based on the tuned "worst case" lower bound from CV
----> 4 best_conf = int(conf_df["worst case"].idxmax())
      5 sub_df["Confidence"] = float(best_conf)
      6 

NameError: name 'conf_df' is not defined

## === cell 31
sample_sub = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
out = sample_sub[["Patient_Week"]].merge(
    sub_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)

out["FVC"] = out["FVC"].fillna(out["FVC"].median())
out["Confidence"] = out["Confidence"].fillna(float(best_conf))

out.to_csv("submission.csv", index=False)
out.head()

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2200316454.py in <cell line: 0>()
      4 )
      5 out = sample_sub[["Patient_Week"]].merge(
----> 6     sub_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
      7 )
      8 

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
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['FVC', 'Confidence'] not in index"

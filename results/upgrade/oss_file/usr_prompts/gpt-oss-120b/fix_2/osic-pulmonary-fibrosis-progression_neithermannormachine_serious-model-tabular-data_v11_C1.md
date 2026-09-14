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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""




## === cell 2
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train_df




## === cell 3
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

    df["WeeksPassed_sqrt"] = np.power(df["WeeksPassed"].abs(), 1 / 2) * np.sign(
        df["WeeksPassed"]
    )
    df["WeeksPassed_square"] = df["WeeksPassed"] ** (2)

    def calculate_height(
        row,
    ):  # height can be predictor of FVC -- this estimates the height of patients
        if row["Sex"] == "Male":
            return row["FirstFVC"] / (27.63 - 0.112 * row["Age"])
        else:
            return row["FirstFVC"] / (21.78 - 0.101 * row["Age"])

    df["Height"] = df.apply(calculate_height, axis=1)

    df["HeightWeeks"] = df["WeeksPassed"] * df["Height"]

    return df




## === cell 4
from sklearn.base import BaseEstimator, TransformerMixin


class MyFeatureEngineerer(BaseEstimator, TransformerMixin):
    """
    this is class so that feature engineering can be done on separate sets


    To use, call fit on a DataFrame to compute and record values that need to be saved before modification (ie before adding new weeks)
    Examples of values needed to be saved are: FirstFVC, FirstWeek, ...
    Then transform after modifications are done

    can just fit_transform if not modifying DataFrame further

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        try:
            self.df_ = feature_engineer(X)
        except AttributeError:  # fit should only be called on pandas DataFrame
            raise ValueError("Can only use this estimator on Pandas DataFrame")
        return self  # return fitted self for further method calls

    def transform(self, X):  # honestly fix this up, it's not scalable at all
        """
        X has been modified with additional weeks
        """

        if len(X) != len(self.df_):
            drop = X.columns.values
            df_tmp = self.df_.drop(drop, axis=1).join(
                self.df_["Patient"]
            )  # drop columns already in X, except for patient
            df = X.merge(df_tmp, on="Patient")
            df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]
            df["WeeksPassed_sqrt"] = np.power(df["WeeksPassed"].abs(), 1 / 2) * np.sign(
                df["WeeksPassed"]
            )
            df["WeeksPassed_square"] = df["WeeksPassed"] ** (2)
            df["HeightWeeks"] = df["WeeksPassed"] * df["Height"]
        else:
            df = self.df_  # if not, just return self.df_
        return df


"""
from sklearn.utils.estimator_checks import check_estimator
check_estimator(MyFeatureEngineerer())
"""




## === cell 5
from sklearn.base import BaseEstimator, TransformerMixin


class ParamMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    custom minmax scaler where min and max are not based on data,
    but are passed in as parameters

    pretty good for percentages
    """

    def __init__(self, min_val=0, max_val=100):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):  # don't need to fit at all
        return self

    def transform(self, X):  # do minmax scaling
        data = (X - self.min_val) / (self.max_val - self.min_val)
        return data


"""
from sklearn.utils.estimator_checks import check_estimator
check_estimator(ParamMinMaxScaler())"""




## === cell 6
def transformed_col_names(col_trans):
    """
    Return a list of column names produced by a fitted ColumnTransformer.
    Uses get_feature_names_out when available; otherwise falls back to the
    original column names.
    """
    names = []
    for name, trans, cols in col_trans.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names.extend(trans.get_feature_names_out(cols))
        elif hasattr(trans, "get_feature_names"):
            names.extend(trans.get_feature_names())
        else:
            names.extend(cols)
    return names




## === cell 7
from sklearn_pandas import DataFrameMapper  # yes it works!

help(
    DataFrameMapper
)  # todo: work this into the pipeline, refactor code so it's less crud




## === cell 8

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


passthru_features = ["Patient", "FVC"]
onehot_features = ["Sex", "SmokingStatus"]
hundred_features = ["Percent", "Age"]
minmax_features = [
    "FirstFVC",
    "FirstWeek",
    "WeeksPassed",
    "WeeksPassed_sqrt",
    "WeeksPassed_square",
    "Height",
    "HeightWeeks",
]


oh_enc = OneHotEncoder(sparse=False, drop="if_binary")
hundred_minmax = ParamMinMaxScaler()
week_minmax = ParamMinMaxScaler(min_val=-12, max_val=133)
minmax = MinMaxScaler()

col_trans = ColumnTransformer(
    [
        ("original", "passthrough", passthru_features),
        ("week_minmax", week_minmax, ["Weeks"]),
        ("hundred_minmax", minmax, hundred_features),
        ("minmax", minmax, minmax_features),
        ("onehot", oh_enc, onehot_features),
    ],
    remainder="passthrough",
    sparse_threshold=0,
)




## === cell 9
train_df = MyFeatureEngineerer().fit_transform(train_df)

new_arr = col_trans.fit_transform(train_df)

train_df = pd.DataFrame(new_arr, columns=transformed_col_names(col_trans))
train_df




## === cell 11
from sklearn.linear_model import LinearRegression


def make_model():
    """
    Returns a simple LinearRegression model.
    """
    model = LinearRegression()
    return model




## === cell 12
train_df  # just look over train_df again




## === cell 13
model = make_model()

drop_features = [
    "Patient",
    "FVC",
    "Weeks",
    "Percent",
]  # features to drop from X training data

X_train = train_df.drop(drop_features, axis=1)
y_train = train_df["FVC"]

model.fit(X_train, y_train)
X_train




## === cell 14
from sklearn.model_selection import RandomizedSearchCV

"""
params = {
    
            }
hyper_search = RandomizedSearchCV(model, param_distributions=params, n_iter = 20)"""




## === cell 16
CONFIDENCE_VALUE = 260




## === cell 17
conf_df = pd.DataFrame({"Confidence": [CONFIDENCE_VALUE]})




## === cell 18
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.bar(X_train.columns, model.coef_)
plt.xticks(rotation=70)
plt.title("LinearRegression Coefficients")
plt.show()




## === cell 19
pred_train = model.predict(X_train)
pred_train




## === cell 20
pass




## === cell 21
from sklearn.pipeline import Pipeline


"""
pipeline = Pipeline([
                ('fe', MyFeatureEngineerer()),
                ('ct', col_trans),
                ('model', make_model())
            ])


pipeline.fit(X_train, y_train)"""




## === cell 22
input_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
input_df  # preprocess this to turn into test_df




## === cell 23
all_weeks = pd.DataFrame(np.arange(-12, 134), columns=["Weeks"])
patient_weeks = pd.DataFrame()

for p in input_df["Patient"].unique():
    tdf = all_weeks.copy()
    tdf["Patient"] = p
    patient_weeks = pd.concat([patient_weeks, tdf], ignore_index=True)

test_meta = input_df.drop(columns=["FVC", "Weeks"])
test_expanded = patient_weeks.merge(test_meta, on="Patient", how="left")

test_fe = feature_engineer(test_expanded)

test_fe["FVC"] = 0

test_arr = col_trans.transform(test_fe)
test_df = pd.DataFrame(test_arr, columns=transformed_col_names(col_trans))
test_df




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3965896822.py in <cell line: 0>()
     13 
     14 # Apply the same feature engineering as for training
---> 15 test_fe = feature_engineer(test_expanded)
     16 
     17 # The model expects a placeholder 'FVC' column (passthrough); fill with zeros

/tmp/ipykernel_55/621879147.py in feature_engineer(data)
      9 
     10     first_fvc = (
---> 11         df.loc[df["Weeks"] == df["FirstWeek"]][["Patient", "FVC"]]
     12         .groupby("Patient")
     13         .first()  # some patients have multiple measurements in same week - get the first

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

## === cell 24
X_test = test_df.drop(drop_features, axis=1)
X_test




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1415484391.py in <cell line: 0>()
      1 # Drop the same columns used during training to obtain the final feature matrix
----> 2 X_test = test_df.drop(drop_features, axis=1)
      3 X_test
      4 
      5 

NameError: name 'test_df' is not defined

## === cell 25
pred = model.predict(X_test)
pred




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/738294633.py in <cell line: 0>()
----> 1 pred = model.predict(X_test)
      2 pred
      3 
      4 

NameError: name 'X_test' is not defined

## === cell 26
sub_df = patient_weeks.copy()
sub_df["FVC"] = pred
sub_df




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2579864444.py in <cell line: 0>()
      1 # Assemble submission DataFrame
      2 sub_df = patient_weeks.copy()
----> 3 sub_df["FVC"] = pred
      4 sub_df
      5 

NameError: name 'pred' is not defined

## === cell 27
import matplotlib.pyplot as plt

plt.figure(figsize=(17, 10))
for i, (patient, frame) in enumerate(sub_df.groupby("Patient")):
    ax = plt.subplot(2, 3, i + 1)
    frame[["Weeks", "FVC"]].plot(x="Weeks", y="FVC", title=patient, ax=ax)
plt.tight_layout()
plt.show()




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3785783029.py in <cell line: 0>()
      4 for i, (patient, frame) in enumerate(sub_df.groupby("Patient")):
      5     ax = plt.subplot(2, 3, i + 1)
----> 6     frame[["Weeks", "FVC"]].plot(x="Weeks", y="FVC", title=patient, ax=ax)
      7 plt.tight_layout()
      8 plt.show()

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

## === cell 28
sub_df["Patient_Week"] = sub_df["Patient"] + "_" + sub_df["Weeks"].astype(str)
sub_df["Confidence"] = CONFIDENCE_VALUE
sub_df




## === cell 29
sub_df[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3991796285.py in <cell line: 0>()
----> 1 sub_df[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
      2 
      3 

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
print("Submission file created at:", "submission.csv")

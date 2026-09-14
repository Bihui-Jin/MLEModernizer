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

3.9

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
pillow==11.3.0
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

-6.976

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
import matplotlib.pyplot as plt

from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tqdm.notebook import tqdm

import os
from PIL import Image




## === cell 1
base_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
df = pd.read_csv(base_path + "train.csv")
df.head()




## === cell 2
def get_weeks_passed(df):
    min_week_dict = df.groupby("Patient").min("Weeks")["Weeks"].to_dict()
    df["MinWeek"] = df["Patient"].map(min_week_dict)
    df["WeeksPassed"] = df["Weeks"] - df["MinWeek"]
    return df


def get_baseline_FVC(df):
    _df = (
        df.loc[df.Weeks == df.MinWeek][["Patient", "FVC"]]
        .rename({"FVC": "FirstFVC"}, axis=1)
        .groupby("Patient")
        .first()
    )

    first_FVC_dict = _df.to_dict()["FirstFVC"]
    df["FirstFVC"] = df["Patient"].map(first_FVC_dict)

    return df


def calculate_height(row):
    if row["Sex"] == "Male":
        return row["FirstFVC"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FirstFVC"] / (21.78 - 0.101 * row["Age"])




## === cell 3
df = get_weeks_passed(df)
df = get_baseline_FVC(df)
df["Height"] = df.apply(calculate_height, axis=1)
df["FullFVC"] = df["FVC"] / df["Percent"] * 100
df.head()




## === cell 4
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

no_transform_attribs = ["Patient", "FVC"]
num_attribs = [
    "Percent",
    "Age",
    "WeeksPassed",
    "FirstFVC",
    "Height",
    "Weeks",
    "MinWeek",
    "FullFVC",
]
cat_attribs = ["Sex", "SmokingStatus"]


class NoTransformer:
    """Pass‑through transformer used inside ColumnTransformer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


datawrangler = ColumnTransformer(
    [
        ("original", NoTransformer(), no_transform_attribs),
        ("minmax", MinMaxScaler(), num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
    ]
)




## === cell 5
transformed_data_series = datawrangler.fit_transform(df)

new_col_names = no_transform_attribs + num_attribs

cat_encoder = datawrangler.named_transformers_["cat"]
categorical_values = cat_encoder.get_feature_names_out()
new_col_names += list(categorical_values)

train_sklearn_df = pd.DataFrame(transformed_data_series, columns=new_col_names)
train_sklearn_df.head()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2660244446.py in <cell line: 0>()
      1 # Fit transformer and build a dataframe with proper column names
----> 2 transformed_data_series = datawrangler.fit_transform(df)
      3 
      4 # base column list
      5 new_col_names = no_transform_attribs + num_attribs

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in fit_transform(self, X, y)
    725         self._validate_remainder(X)
    726 
--> 727         result = self._fit_transform(X, y, _fit_transform_one)
    728 
    729         if not result:

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in _fit_transform(self, X, y, func, fitted, column_as_strings)
    656         )
    657         try:
--> 658             return Parallel(n_jobs=self.n_jobs)(
    659                 delayed(func)(
    660                     transformer=clone(trans) if not fitted else trans,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
     61             for delayed_func, args, kwargs in iterable
     62         )
---> 63         return super().__call__(iterable_with_config)
     64 
     65 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   1984             output = self._get_sequential_output(iterable)
   1985             next(output)
-> 1986             return output if self.return_generator else list(output)
   1987 
   1988         # Let's create an ID that uniquely identifies the current call. If the

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_sequential_output(self, iterable)
   1909 
   1910             # Sequentially call the tasks and yield the results.
-> 1911             for func, args, kwargs in iterable:
   1912                 self.n_dispatched_batches += 1
   1913                 self.n_dispatched_tasks += 1

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in <genexpr>(.0)
     57         # pre_dispatch and n_jobs.
     58         config = get_config()
---> 59         iterable_with_config = (
     60             (_with_config(delayed_func, config), args, kwargs)
     61             for delayed_func, args, kwargs in iterable

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in <genexpr>(.0)
    658             return Parallel(n_jobs=self.n_jobs)(
    659                 delayed(func)(
--> 660                     transformer=clone(trans) if not fitted else trans,
    661                     X=_safe_indexing(X, column, axis=1),
    662                     y=y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in clone(estimator, safe)
     77                 )
     78             else:
---> 79                 raise TypeError(
     80                     "Cannot clone object '%s' (type %s): "
     81                     "it does not seem to be a scikit-learn "

TypeError: Cannot clone object '<__main__.NoTransformer object at 0x7fc50979ddd0>' (type <class '__main__.NoTransformer'>): it does not seem to be a scikit-learn estimator as it does not implement a 'get_params' method.

## === cell 6
from sklearn.model_selection import train_test_split

X = train_sklearn_df.drop(columns=["FVC", "Patient"])
y = train_sklearn_df["FVC"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2146494255.py in <cell line: 0>()
      2 
      3 # Features: drop the target and the raw Patient ID (kept only for reference)
----> 4 X = train_sklearn_df.drop(columns=["FVC", "Patient"])
      5 y = train_sklearn_df["FVC"]
      6 

NameError: name 'train_sklearn_df' is not defined

## === cell 7
from sklearn.ensemble import GradientBoostingRegressor

LOWER_ALPHA = 0.1
UPPER_ALPHA = 0.9

lower_huber = GradientBoostingRegressor(
    loss="quantile", alpha=LOWER_ALPHA, random_state=42
)
upper_huber = GradientBoostingRegressor(
    loss="quantile", alpha=UPPER_ALPHA, random_state=42
)
mid_huber = GradientBoostingRegressor(loss="huber", random_state=42)

lower_huber.fit(X_train, y_train)
mid_huber.fit(X_train, y_train)
upper_huber.fit(X_train, y_train)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/801612086.py in <cell line: 0>()
     13 
     14 # Train the three models
---> 15 lower_huber.fit(X_train, y_train)
     16 mid_huber.fit(X_train, y_train)
     17 upper_huber.fit(X_train, y_train)

NameError: name 'X_train' is not defined

## === cell 8
def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    return np.mean(-np.sqrt(2) * deltaFVC / clipSTD - np.log(np.sqrt(2) * clipSTD))




## === cell 9
def _engineer_feature(Week, FVC, Percent, Age, Sex):
    MinWeek = min(Week, 0)
    FirstFVC = FVC
    FullFVC = (FVC / Percent) * 100
    if Sex == "Male":
        Height = FirstFVC / (27.63 - 0.112 * Age)
    else:
        Height = FirstFVC / (21.78 - 0.101 * Age)
    return MinWeek, FirstFVC, FullFVC, Height


def _create_df_with_running_weeks(
    Patient,
    Weeks,
    FVC,
    Percent,
    Age,
    Sex,
    SmokingStatus,
    MinWeek,
    FirstFVC,
    FullFVC,
    Height,
    week_start=-12,
    week_end=134,
):
    week_range = list(range(week_start, week_end))
    df = pd.DataFrame({"Weeks": week_range})
    df["Patient"] = Patient
    df["Sex"] = Sex
    df["Age"] = Age
    df["SmokingStatus"] = SmokingStatus
    df["MinWeek"] = MinWeek
    df["FirstFVC"] = FirstFVC
    df["FullFVC"] = FullFVC
    df["Height"] = Height
    df["Percent"] = Percent
    df["WeeksPassed"] = df["Weeks"] - MinWeek
    df["FVC"] = 0.0  # placeholder
    return df


def _wrangle_data(df):
    transformed_data_series = datawrangler.transform(df)

    base_cols = no_transform_attribs + num_attribs
    cat_names = datawrangler.named_transformers_["cat"].get_feature_names_out()
    all_cols = base_cols + list(cat_names)

    df_transformed = pd.DataFrame(transformed_data_series, columns=all_cols)
    return df, df_transformed


def _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber):
    feature_cols = X.columns  # X from training cell 7
    df_transformed = df_transformed[feature_cols]

    preds_lower = lower_huber.predict(df_transformed)
    preds_mid = mid_huber.predict(df_transformed)
    preds_upper = upper_huber.predict(df_transformed)

    df["Lower"] = preds_lower
    df["Upper"] = preds_upper
    df["FVC"] = preds_mid
    df["Confidence"] = np.abs(preds_upper - preds_lower)
    return df


def huber_predict(
    lower_huber,
    mid_huber,
    upper_huber,
    Patient,
    Week,
    FVC,
    Percent,
    Age,
    Sex,
    SmokingStatus,
    week_start=-12,
    week_end=134,
):
    MinWeek, FirstFVC, FullFVC, Height = _engineer_feature(Week, FVC, Percent, Age, Sex)
    df = _create_df_with_running_weeks(
        Patient,
        Week,
        FVC,
        Percent,
        Age,
        Sex,
        SmokingStatus,
        MinWeek,
        FirstFVC,
        FullFVC,
        Height,
        week_start,
        week_end,
    )
    df, df_transformed = _wrangle_data(df)
    df = _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber)
    return df




## === cell 10
test_df = pd.read_csv(base_path + "test.csv")
test_df.tail()




## === cell 11
df_predict = pd.DataFrame()  # will collect all rows

for idx, row in test_df.iterrows():
    patient_pred = huber_predict(
        lower_huber,
        mid_huber,
        upper_huber,
        Patient=row["Patient"],
        Week=row["Weeks"],
        FVC=row["FVC"],
        Percent=row["Percent"],
        Age=row["Age"],
        Sex=row["Sex"],
        SmokingStatus=row["SmokingStatus"],
        week_start=-12,
        week_end=134,
    )
    df_predict = pd.concat([df_predict, patient_pred], ignore_index=True)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/666933330.py in <cell line: 0>()
      3 
      4 for idx, row in test_df.iterrows():
----> 5     patient_pred = huber_predict(
      6         lower_huber,
      7         mid_huber,

/tmp/ipykernel_55/2643048893.py in huber_predict(lower_huber, mid_huber, upper_huber, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus, week_start, week_end)
     99         week_end,
    100     )
--> 101     df, df_transformed = _wrangle_data(df)
    102     df = _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber)
    103     return df

/tmp/ipykernel_55/2643048893.py in _wrangle_data(df)
     42 
     43 def _wrangle_data(df):
---> 44     transformed_data_series = datawrangler.transform(df)
     45 
     46     # rebuild column list exactly as done when training

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in transform(self, X)
    776 
    777         if fit_dataframe_and_transform_dataframe:
--> 778             named_transformers = self.named_transformers_
    779             # check that all names seen in fit are in transform, unless
    780             # they were dropped

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in named_transformers_(self)
    459         """
    460         # Use Bunch object to improve autocomplete
--> 461         return Bunch(**{name: trans for name, trans, _ in self.transformers_})
    462 
    463     def _get_feature_name_out_for_transformer(

AttributeError: 'ColumnTransformer' object has no attribute 'transformers_'

## === cell 12
df_predict["Patient_Week"] = (
    df_predict["Patient"] + "_" + df_predict["Weeks"].astype(str)
)

submission = df_predict[["Patient_Week", "FVC"]].copy()
submission["Confidence"] = 285

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1720963426.py in <cell line: 0>()
      1 # Build the submission file
      2 df_predict["Patient_Week"] = (
----> 3     df_predict["Patient"] + "_" + df_predict["Weeks"].astype(str)
      4 )
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'Patient'

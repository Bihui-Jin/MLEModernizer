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

3.9

# 2. Installed packages

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".", 1)[0])
except Exception:
    _pb_major = None

if _pb_major is not None and _pb_major >= 6:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf>=5.26.1,<6"]
    )
    import importlib

    importlib.invalidate_caches()

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

from tqdm.notebook import tqdm

from PIL import Image


## === cell 1
base_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/'
df = pd.read_csv(base_path + 'train.csv')
df.head()


## === cell 2
def get_weeks_passed(df):
    min_week_dict = df.groupby('Patient').min('Weeks')['Weeks'].to_dict()
    df['MinWeek'] =  df['Patient'].map(min_week_dict)
    df['WeeksPassed'] = df['Weeks'] - df['MinWeek']
    return df


## === cell 3
def get_baseline_FVC(df):
    _df = (
        df
        .loc[df.Weeks == df.MinWeek][['Patient','FVC']]
        .rename({'FVC': 'FirstFVC'}, axis=1)
        .groupby('Patient')
        .first()
    )
    
    first_FVC_dict = _df.to_dict()['FirstFVC']
    df['FirstFVC'] =  df['Patient'].map(first_FVC_dict)
    
    return df


## === cell 4
def calculate_height(row):
    if row['Sex'] == 'Male':
        return row['FirstFVC'] / (27.63 - 0.112 * row['Age'])
    else:
        return row['FirstFVC'] / (21.78 - 0.101 * row['Age'])
    


## === cell 5
df = get_weeks_passed(df)
df = get_baseline_FVC(df)
df['Height'] = df.apply(calculate_height, axis=1)
df['FullFVC'] = df['FVC']/df['Percent']*100


## === cell 6
df


## === cell 7
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

no_transform_attribs = ['Patient','FVC']
num_attribs = ['Percent', 'Age', 'WeeksPassed', 'FirstFVC','Height', 'Weeks', 'MinWeek', 'FullFVC']
cat_attribs = ['Sex', 'SmokingStatus']


## === cell 8
from sklearn.base import BaseEstimator, TransformerMixin

class NoTransformer(BaseEstimator, TransformerMixin):
    """Passes through data without any change and is compatible with ColumnTransformer class"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X


## === cell 9

datawrangler = ColumnTransformer(([
     ('original', NoTransformer(), no_transform_attribs),
     ('MinMax', MinMaxScaler(), num_attribs),
     ('cat_encoder', OneHotEncoder(), cat_attribs),
    ]))

transformed_data_series = []
transformed_data_series = datawrangler.fit_transform(df)


## === cell 10

new_col_names = no_transform_attribs + num_attribs

categorical_values = list(
    datawrangler.named_transformers_["cat_encoder"].get_feature_names_out(cat_attribs)
)
new_col_names += categorical_values

train_sklearn_df = pd.DataFrame(transformed_data_series, columns=new_col_names)
train_sklearn_df.head()


## === cell 11
from sklearn.model_selection import train_test_split

csv_features_list = [
    "FullFVC",
    "Age",
    "Weeks",
    "MinWeek",
    "WeeksPassed",
    "FirstFVC",
    "Height",
    "x0_Female",
    "x1_Currently smokes",
    "x1_Ex-smoker",
]

_feature_name_map = {
    "x0_Female": "Sex_Female",
    "x1_Currently smokes": "SmokingStatus_Currently smokes",
    "x1_Ex-smoker": "SmokingStatus_Ex-smoker",
}
resolved_features_list = [
    f if f in train_sklearn_df.columns else _feature_name_map.get(f, f)
    for f in csv_features_list
]

X = train_sklearn_df[resolved_features_list].astype(float)

y = train_sklearn_df[["FVC"]].astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


## === cell 12
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor


## === cell 14

LOWER_ALPHA = 0.2
UPPER_ALPHA = 0.8
lower_huber = GradientBoostingRegressor(loss="quantile",                   
                                        alpha=LOWER_ALPHA)
upper_huber = GradientBoostingRegressor(loss="quantile",
                                        alpha=UPPER_ALPHA)

mid_huber = GradientBoostingRegressor(loss="huber")


## === cell 15
lower_huber.fit(X_train, y_train)
mid_huber.fit(X_train, y_train)
upper_huber.fit(X_train, y_train)



preds_lower = lower_huber.predict(X_test)
preds_mid = mid_huber.predict(X_test)
preds_upper = upper_huber.predict(X_test)

preds = pd.DataFrame({'lower':preds_lower, 'mid':preds_mid, 'upper':preds_upper})


## === cell 16
mse = mean_squared_error(
    y_test,
    preds['mid'],
    squared=False
)

mae = mean_absolute_error(
    y_test,
    preds['mid']
)

print('MSE Loss: {0:.2f}'.format(mse))
print('MAE Loss: {0:.2f}'.format(mae))


## === cell 17
def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70 , 9e9)  
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 1000)  
    return np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))
    

print(
    'Competition metric with variable confidence: ', 
    competition_metric(np.ravel(y_test.values), preds['mid'], preds['upper']-preds['lower']) 
)

print(
    'Competition metric with static confidence: ', 
    competition_metric(np.ravel(y_test.values), preds['mid'], 285) 
)


## === cell 20
def load_huber_models(lower_huber_path, mid_huber_path, upper_huber_path, datawrangler_path):
    
    '''
    function to load saved huber modles and saved datawrangler
    
    Param
    -----
    lower_huber_path: string
    mid_huber_path: string
    upper_huber_path: string
    datawrangler_path: string
    
    Return
    ------
    lower_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.1
    mid_huber: saved GradientBoostRegressor from sklearn, with loss = 'huber'
    upper_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.9
    datawrangler: saved columntransformer from sklearn
    
    '''
    
    with open(lower_huber_path, 'rb') as f:
        lower_huber = cPickle.load(f)

    with open(mid_huber_path, 'rb') as f:
        mid_huber = cPickle.load(f)

    with open(upper_huber_path, 'rb') as f:
        upper_huber = cPickle.load(f)

    with open(datawrangler_path, 'rb') as f:
        datawrangler = cPickle.load(f)

    return lower_huber, mid_huber, upper_huber, datawrangler
    

def huber_predict(lower_huber, mid_hubber, upper_huber, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus, week_start=-12, week_end=134):

    '''
    function to predict FVC value and confidence
    
    Param
    -----
    lower_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.1
    mid_huber: saved GradientBoostRegressor from sklearn, with loss = 'huber'
    upper_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.9
    datawrangler: saved columntransformer from sklearn
    
    Week: integer 
        Number of weeks after CT Scan, can be negative
    FVC: integer
        Initial FVC measurement at above `week`
    Percent: integer
        Initial percentage measurement at above `week`
    Age: integer
        Age at above `week`
    Sex: string
        'Male' or 'Female'
    SmokingStatus: string
        'Currently smokes', 'Ex-smoker', or 'Never smoked'
    week_start: integer
        start week to predict
    week_end: interger
        end week to predict
    
    Return
    ------
    df: DataFrame
        DataFrame with user inputs, engineered features, and predictions in running weeks
    
    '''
    
    
    MinWeek, FirstFVC, FullFVC, Height = _engineer_feature(Week, FVC, Percent, Age, Sex)

    df = _create_df_with_running_weeks(Patient,
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
                                        week_end)

    df, df_transformed = _wrangle_data(df)
    
    df = _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber)
    
    return df
        
def _engineer_feature(Week, FVC, Percent, Age, Sex):
    '''
    function to calculate MinWeek, FullFVC, and Height from patient details
    
    '''

    MinWeek = min(Week, 0)
    FirstFVC = FVC
    FullFVC = (FVC/Percent)*100

    if Sex == 'Male':
        Height = FirstFVC / (27.63 - 0.112 * Age)
    else:
        Height = FirstFVC / (21.78 - 0.101 * Age)

    return MinWeek, FirstFVC, FullFVC, Height
    
def _create_df_with_running_weeks(Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus, MinWeek, FirstFVC, FullFVC, Height, week_start, week_end):

    '''
    function to put patient details, engineered features, and running list of weeks into DataFrame
    
    '''

    Weeks = list(range(week_start, week_end))
    df = pd.DataFrame({'Weeks':Weeks})
    df['Patient'] = Patient
    df['Sex'] = Sex
    df['Age'] = Age
    df['SmokingStatus'] = SmokingStatus
    df['MinWeek'] = MinWeek
    df['FirstFVC'] = FVC
    df['FullFVC'] = FullFVC
    df['Height'] = Height
    df['Percent'] = Percent
    df['WeeksPassed'] = df['Weeks'] - df['MinWeek']
    df['FVC'] = 0 # dummy FVC, to be predicted

    return df

def _wrangle_data(df):
    '''
    function to transform patient details into suitable format for models' ingestion
    
    '''

    transformed_data_series = datawrangler.transform(df)


    new_col_names = no_transform_attribs + num_attribs

    categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]
    new_col_names += categorical_values

    df_transformed = pd.DataFrame(transformed_data_series, columns=new_col_names)

    return df, df_transformed


def _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber):
    
    '''
    function to predict lower, upper and mid FVC and confidence interval for patients
    
    '''
    csv_features_list = ['FullFVC','Age','Weeks','MinWeek','WeeksPassed','FirstFVC','Height','x0_Female','x1_Currently smokes','x1_Ex-smoker']
    df_transformed = df_transformed[csv_features_list]

    preds_lower = lower_huber.predict(df_transformed)
    preds_mid = mid_huber.predict(df_transformed)
    preds_upper = upper_huber.predict(df_transformed)
    
    df['Lower'] = preds_lower
    df['Upper'] = preds_upper
    df['FVC'] = preds_mid
    df['Confidence'] = abs(preds_upper - preds_lower)

    return df


## === cell 21
def load_huber_models(
    lower_huber_path, mid_huber_path, upper_huber_path, datawrangler_path
):
    """
    function to load saved huber modles and saved datawrangler

    Param
    -----
    lower_huber_path: string
    mid_huber_path: string
    upper_huber_path: string
    datawrangler_path: string

    Return
    ------
    lower_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.1
    mid_huber: saved GradientBoostRegressor from sklearn, with loss = 'huber'
    upper_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.9
    datawrangler: saved columntransformer from sklearn

    """

    with open(lower_huber_path, "rb") as f:
        lower_huber = cPickle.load(f)

    with open(mid_huber_path, "rb") as f:
        mid_huber = cPickle.load(f)

    with open(upper_huber_path, "rb") as f:
        upper_huber = cPickle.load(f)

    with open(datawrangler_path, "rb") as f:
        datawrangler = cPickle.load(f)

    return lower_huber, mid_huber, upper_huber, datawrangler


def huber_predict(
    lower_huber,
    mid_hubber,
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
    """
    function to predict FVC value and confidence

    Param
    -----
    lower_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.1
    mid_huber: saved GradientBoostRegressor from sklearn, with loss = 'huber'
    upper_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.9
    datawrangler: saved columntransformer from sklearn

    Week: integer
        Number of weeks after CT Scan, can be negative
    FVC: integer
        Initial FVC measurement at above `week`
    Percent: integer
        Initial percentage measurement at above `week`
    Age: integer
        Age at above `week`
    Sex: string
        'Male' or 'Female'
    SmokingStatus: string
        'Currently smokes', 'Ex-smoker', or 'Never smoked'
    week_start: integer
        start week to predict
    week_end: interger
        end week to predict

    Return
    ------
    df: DataFrame
        DataFrame with user inputs, engineered features, and predictions in running weeks

    """

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


def _engineer_feature(Week, FVC, Percent, Age, Sex):
    """
    function to calculate MinWeek, FullFVC, and Height from patient details

    """

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
    week_start,
    week_end,
):
    """
    function to put patient details, engineered features, and running list of weeks into DataFrame

    """

    Weeks = list(range(week_start, week_end))
    df = pd.DataFrame({"Weeks": Weeks})
    df["Patient"] = Patient
    df["Sex"] = Sex
    df["Age"] = Age
    df["SmokingStatus"] = SmokingStatus
    df["MinWeek"] = MinWeek
    df["FirstFVC"] = FVC
    df["FullFVC"] = FullFVC
    df["Height"] = Height
    df["Percent"] = Percent
    df["WeeksPassed"] = df["Weeks"] - df["MinWeek"]
    df["FVC"] = 0  # dummy FVC, to be predicted

    return df


def _wrangle_data(df):
    """
    function to transform patient details into suitable format for models' ingestion

    """

    transformed_data_series = datawrangler.transform(df)

    new_col_names = no_transform_attribs + num_attribs

    categorical_values = list(
        datawrangler.named_transformers_["cat_encoder"].get_feature_names_out(
            cat_attribs
        )
    )
    new_col_names += categorical_values

    df_transformed = pd.DataFrame(transformed_data_series, columns=new_col_names)

    return df, df_transformed


def _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber):
    """
    function to predict lower, upper and mid FVC and confidence interval for patients

    """
    csv_features_list = [
        "FullFVC",
        "Age",
        "Weeks",
        "MinWeek",
        "WeeksPassed",
        "FirstFVC",
        "Height",
        "x0_Female",
        "x1_Currently smokes",
        "x1_Ex-smoker",
    ]
    df_transformed = df_transformed[csv_features_list]

    preds_lower = lower_huber.predict(df_transformed)
    preds_mid = mid_huber.predict(df_transformed)
    preds_upper = upper_huber.predict(df_transformed)

    df["Lower"] = preds_lower
    df["Upper"] = preds_upper
    df["FVC"] = preds_mid
    df["Confidence"] = abs(preds_upper - preds_lower)

    return df


## === cell 22
base_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/'
df = pd.read_csv(base_path + 'test.csv')
df.tail(20)


## === cell 23



for i, row in df.iterrows():
    if i == 0:
        df_predict = huber_predict(lower_huber,
                            mid_huber,
                            upper_huber,
                            row['Patient'],
                            row['Weeks'],
                            row['FVC'],
                            row['Percent'],
                            row['Age'],
                            row['Sex'],
                            row['SmokingStatus']
                           )
    else:
        df_interim = huber_predict(lower_huber,
                    mid_huber,
                    upper_huber,
                    row['Patient'],
                    row['Weeks'],
                    row['FVC'],
                    row['Percent'],
                    row['Age'],
                    row['Sex'],
                    row['SmokingStatus']
                   )
    

        df_predict = df_predict.append(df_interim)


## --- ERROR in cell 23, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_10/3211849242.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      8[0m [0;32mfor[0m [0mi[0m[0;34m,[0m [0mrow[0m [0;32min[0m [0mdf[0m[0;34m.[0m[0miterrows[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m     [0;32mif[0m [0mi[0m [0;34m==[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m         df_predict = huber_predict(lower_huber,
[0m[1;32m     11[0m                             [0mmid_huber[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m                             [0mupper_huber[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_10/2518206529.py[0m in [0;36mhuber_predict[0;34m(lower_huber, mid_hubber, upper_huber, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus, week_start, week_end)[0m
[1;32m    104[0m     [0mdf[0m[0;34m,[0m [0mdf_transformed[0m [0;34m=[0m [0m_wrangle_data[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    105[0m [0;34m[0m[0m
[0;32m--> 106[0;31m     [0mdf[0m [0;34m=[0m [0m_get_predictions[0m[0;34m([0m[0mdf[0m[0;34m,[0m [0mdf_transformed[0m[0;34m,[0m [0mlower_huber[0m[0;34m,[0m [0mmid_huber[0m[0;34m,[0m [0mupper_huber[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    107[0m [0;34m[0m[0m
[1;32m    108[0m     [0;32mreturn[0m [0mdf[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_10/2518206529.py[0m in [0;36m_get_predictions[0;34m(df, df_transformed, lower_huber, mid_huber, upper_huber)[0m
[1;32m    204[0m         [0;34m"x1_Ex-smoker"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    205[0m     ]
[0;32m--> 206[0;31m     [0mdf_transformed[0m [0;34m=[0m [0mdf_transformed[0m[0;34m[[0m[0mcsv_features_list[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    207[0m [0;34m[0m[0m
[1;32m    208[0m     [0mpreds_lower[0m [0;34m=[0m [0mlower_huber[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mdf_transformed[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   4106[0m             [0;32mif[0m [0mis_iterator[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4107[0m                 [0mkey[0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4108[0;31m             [0mindexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0m_get_indexer_strict[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0;34m"columns"[0m[0;34m)[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4109[0m [0;34m[0m[0m
[1;32m   4110[0m         [0;31m# take() does not accept boolean indexers[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36m_get_indexer_strict[0;34m(self, key, axis_name)[0m
[1;32m   6198[0m             [0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m,[0m [0mnew_indexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_reindex_non_unique[0m[0;34m([0m[0mkeyarr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6199[0m [0;34m[0m[0m
[0;32m-> 6200[0;31m         [0mself[0m[0;34m.[0m[0m_raise_if_missing[0m[0;34m([0m[0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m,[0m [0maxis_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6201[0m [0;34m[0m[0m
[1;32m   6202[0m         [0mkeyarr[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtake[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36m_raise_if_missing[0;34m(self, key, indexer, axis_name)[0m
[1;32m   6250[0m [0;34m[0m[0m
[1;32m   6251[0m             [0mnot_found[0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mensure_index[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[[0m[0mmissing_mask[0m[0;34m.[0m[0mnonzero[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0munique[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6252[0;31m             [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0;34mf"{not_found} not in index"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6253[0m [0;34m[0m[0m
[1;32m   6254[0m     [0;34m@[0m[0moverload[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: "['x0_Female', 'x1_Currently smokes', 'x1_Ex-smoker'] not in index"

## === cell 24
df_predict['Patient_Week'] = df_predict['Patient'] + '_' + df_predict['Weeks'].astype(str)

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

-6.9692

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

import tensorflow as tf

from tqdm.notebook import tqdm

import os
from PIL import Image


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]
new_col_names += categorical_values

train_sklearn_df = pd.DataFrame(transformed_data_series, columns=new_col_names)
train_sklearn_df.head()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2753060113.py in <cell line: 0>()
      5 
      6 # extract possible values from the fitted transformer
----> 7 categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]
      8 new_col_names += categorical_values
      9 

AttributeError: 'OneHotEncoder' object has no attribute 'get_feature_names'

## === cell 11
from sklearn.model_selection import train_test_split

csv_features_list = ['FullFVC','Age','Weeks','MinWeek','WeeksPassed','FirstFVC','Height','x0_Female','x1_Currently smokes','x1_Ex-smoker']

X = train_sklearn_df[csv_features_list].astype(float)

y = train_sklearn_df[['FVC']].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1203019177.py in <cell line: 0>()
      3 csv_features_list = ['FullFVC','Age','Weeks','MinWeek','WeeksPassed','FirstFVC','Height','x0_Female','x1_Currently smokes','x1_Ex-smoker']
      4 
----> 5 X = train_sklearn_df[csv_features_list].astype(float)
      6 
      7 y = train_sklearn_df[['FVC']].astype(float)

NameError: name 'train_sklearn_df' is not defined

## === cell 12
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor


## === cell 14

LOWER_ALPHA = 0.25
UPPER_ALPHA = 0.75
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


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3718906340.py in <cell line: 0>()
      1 # Fit models
----> 2 lower_huber.fit(X_train, y_train)
      3 mid_huber.fit(X_train, y_train)
      4 upper_huber.fit(X_train, y_train)
      5 

NameError: name 'X_train' is not defined

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


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/92010253.py in <cell line: 0>()
      1 mse = mean_squared_error(
----> 2     y_test,
      3     preds['mid'],
      4     squared=False
      5 )

NameError: name 'y_test' is not defined

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


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3560619549.py in <cell line: 0>()
      7 print(
      8     'Competition metric with variable confidence: ',
----> 9     competition_metric(np.ravel(y_test.values), preds['mid'], preds['upper']-preds['lower'])
     10 )
     11 

NameError: name 'y_test' is not defined

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
Patient = 'Albert'
Week = -4 # Number of weeks after CT Scan, can be negative
FVC = 3000
Percent = 78
Age = 69
Sex = 'Female'
SmokingStatus = 'Never smoked'



df = huber_predict(lower_huber, mid_huber, upper_huber, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/146603639.py in <cell line: 0>()
     14 #                                                                     '/kaggle/working/datawrangler.pkl')
     15 
---> 16 df = huber_predict(lower_huber, mid_huber, upper_huber, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus)

/tmp/ipykernel_11/4099507354.py in huber_predict(lower_huber, mid_hubber, upper_huber, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus, week_start, week_end)
     88                                         week_end)
     89 
---> 90     df, df_transformed = _wrangle_data(df)
     91 
     92     df = _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber)

/tmp/ipykernel_11/4099507354.py in _wrangle_data(df)
    148 
    149     # extract possible values from the fitted transformer
--> 150     categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]
    151     new_col_names += categorical_values
    152 

AttributeError: 'OneHotEncoder' object has no attribute 'get_feature_names'

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
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3211849242.py in <cell line: 0>()
      8 for i, row in df.iterrows():
      9     if i == 0:
---> 10         df_predict = huber_predict(lower_huber,
     11                             mid_huber,
     12                             upper_huber,

/tmp/ipykernel_11/4099507354.py in huber_predict(lower_huber, mid_hubber, upper_huber, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus, week_start, week_end)
     88                                         week_end)
     89 
---> 90     df, df_transformed = _wrangle_data(df)
     91 
     92     df = _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber)

/tmp/ipykernel_11/4099507354.py in _wrangle_data(df)
    148 
    149     # extract possible values from the fitted transformer
--> 150     categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]
    151     new_col_names += categorical_values
    152 

AttributeError: 'OneHotEncoder' object has no attribute 'get_feature_names'

## === cell 24
df_predict['Patient_Week'] = df_predict['Patient'] + '_' + df_predict['Weeks'].astype(str)
df_predict = df_predict.sort_values('Patient').sort_values('Weeks', ignore_index=True)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/602010596.py in <cell line: 0>()
----> 1 df_predict['Patient_Week'] = df_predict['Patient'] + '_' + df_predict['Weeks'].astype(str)
      2 df_predict = df_predict.sort_values('Patient').sort_values('Weeks', ignore_index=True)

NameError: name 'df_predict' is not defined

## === cell 25
df_submission = df_predict[['Patient_Week', 'FVC']] #, 'Confidence'
df_submission['Confidence'] = 285
df_submission.to_csv('/kaggle/working/submission.csv', index=False)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2529652126.py in <cell line: 0>()
----> 1 df_submission = df_predict[['Patient_Week', 'FVC']] #, 'Confidence'
      2 df_submission['Confidence'] = 285
      3 df_submission.to_csv('/kaggle/working/submission.csv', index=False)

NameError: name 'df_predict' is not defined

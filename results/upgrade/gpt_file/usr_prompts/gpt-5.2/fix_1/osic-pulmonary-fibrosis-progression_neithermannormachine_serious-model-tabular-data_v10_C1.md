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

## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''



## === cell 3
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train_df


## === cell 4
def feature_engineer(data):
    '''
    method to feature engineer any df, train or test
    '''
    
    df = data.copy()
    
    df['FirstWeek'] = df.groupby('Patient')['Weeks'].transform('min')
    
    first_fvc = (df.loc[df['Weeks'] == df['FirstWeek']][['Patient','FVC']]
                        .groupby('Patient')
                        .first() #some patients have multiple measurements in same week - get the first
                        .reset_index()
                         .rename(columns = {'FVC': 'FirstFVC'}) )
    
    df = df.merge(first_fvc, on = 'Patient') #add FirstFVC column
    
    df['WeeksPassed'] = df['Weeks'] - df['FirstWeek']
    
    '''
    
    
    '''
    
    
    
    def calculate_height(row): #height can be predictor of FVC -- this estimates the height of patients
        if row['Sex'] == 'Male':
            return row['FirstFVC'] / (27.63 - 0.112 * row['Age'])
        else:
            return row['FirstFVC'] / (21.78 - 0.101 * row['Age'])

    df['Height'] = df.apply(calculate_height, axis=1)
    
    return df


feature_engineer(train_df) #just looking


## === cell 5
from sklearn.base import BaseEstimator, TransformerMixin

class MyFeatureEngineerer(BaseEstimator, TransformerMixin):
    '''
    this is class so that feature engineering can be done on separate sets
    
    
    To use, call fit on a DataFrame to compute and record values that need to be saved before modification (ie before adding new weeks)
    Examples of values needed to be saved are: FirstFVC, FirstWeek, ...
    Then transform after modifications are done
    
    can just fit_transform if not modifying DataFrame further
    
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        try:
            self.df_ = feature_engineer(X)
        except AttributeError: #fit should only be called on pandas DataFrame
            raise ValueError('Can only use this estimator on Pandas DataFrame')
        return self #return fitted self for further method calls
    
    def transform(self, X):
        '''
        X has been modified with additional weeks
        '''
        
        if len(X) != len(self.df_):
            drop = X.columns.values 
            df = self.df_.drop(drop, axis = 1).join(self.df_['Patient']) #drop columns already in X, except for patient
            df = X.merge(df, on = 'Patient')
            df['WeeksPassed'] = df['Weeks'] - df['FirstWeek']
        else:
            df = self.df_ #if not, just return self.df_
        return df
    
'''
from sklearn.utils.estimator_checks import check_estimator
check_estimator(MyFeatureEngineerer())
'''


## === cell 6
from sklearn.base import BaseEstimator, TransformerMixin

class ParamMinMaxScaler(BaseEstimator, TransformerMixin):
    '''
    custom minmax scaler where min and max are not based on data,
    but are passed in as parameters
    
    pretty good for percentages
    '''
    def __init__(self, min_val = 0, max_val = 100):
        self.min_val = min_val
        self.max_val = max_val
    
    
    def fit(self, X, y=None): #don't need to fit at all
        return self

    def transform(self, X): #do minmax scaling
        data = (X - self.min_val) / (self.max_val - self.min_val)
        return data
'''
from sklearn.utils.estimator_checks import check_estimator
check_estimator(ParamMinMaxScaler())'''


## === cell 7
def transformed_col_names(col_trans):
    '''
    helper function to get column names of dataframe back after column transforming
    because col_trans.get_feature_names() doesn't work very well
    Use this after fitting col_trans
    '''
    import re
    
    new_colnames = []
    for _, t, col in col_trans.transformers_: #loop thru all transformers
        try: #try to get new column names
            temp = t.get_feature_names()
            temp2 = []
            for name in temp: #loop thru feature names returned by t
                match = re.search('x(\d+)+_', name) #look for this ugly bit
                i = int(match.group(1)) #get the feature number
                new_name = col[i] + '_' + name[match.end():] #replace x0 or whatever number with meaningful feature name
                temp2.append(new_name)
            col = temp2
        except AttributeError: #if transformer t does not provide get_feature_names()
            pass #no big deal, just ignore it; we'll extend with original column names
        new_colnames.extend(col) #then append column names to list
        
    return new_colnames


## === cell 8
from sklearn_pandas import DataFrameMapper #yes it works!
help(DataFrameMapper) #todo: work this into the pipeline, refactor code so it's less crud


## === cell 9

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

passthru_features = ['Patient', 'FVC']
onehot_features = ['Sex', 'SmokingStatus']
hundred_features = ['Percent', 'Age']
minmax_features = ['FirstFVC', 'FirstWeek', 'WeeksPassed', 'Height']


oh_enc = OneHotEncoder(sparse = False, drop = 'if_binary')
hundred_minmax = ParamMinMaxScaler()
week_minmax = ParamMinMaxScaler(min_val = -12, max_val = 133)
minmax = MinMaxScaler()

col_trans = ColumnTransformer([
                ('original', 'passthrough', passthru_features),
                ('week_minmax', week_minmax, ['Weeks']),
                ('hundred_minmax', minmax, hundred_features),
                ('minmax', minmax, minmax_features),
                ('onehot', oh_enc, onehot_features)
            ], remainder = 'passthrough', sparse_threshold=0)


## === cell 10
help(ColumnTransformer)


## === cell 11
train_df = MyFeatureEngineerer().fit_transform(train_df)

new_df = col_trans.fit_transform(train_df)

train_df = pd.DataFrame(new_df, columns = transformed_col_names(col_trans))
train_df


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4258748423.py in <cell line: 0>()
      4 
      5 #get the names of the columns back and convert to dataframe
----> 6 train_df = pd.DataFrame(new_df, columns = transformed_col_names(col_trans))
      7 train_df

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

## === cell 12
'''
For sklearn compatibility, functions should have signature f(y_true, y_pred, **kwargs)
For tensorflow compatibility, functions should have signature f(y_true, y_pred)
'''

import tensorflow as tf

def laplace_log_score(**kwargs):
    '''
    The competition metric. Average laplace log score over all examples.
    Technically I don't need to return a separate function, I just do this for consistency
    with the other functions I defined
    for now kwargs is ignored
    
    Returns loss function that returns average laplace log score over all predictions.
    
    
    y_true is shape (N x 1)
    y_pred is shape (N x 3)
    '''
    sigma_min = 70 #confidence can't be lower than this 
    delta_max = 1000 #delta can't be higher than this
    
    def loss(y_true, y_pred):
        tf.dtypes.cast(y_true, tf.float32)
        tf.dtypes.cast(y_pred, tf.float32)

        sigma = y_pred[:, 2] - y_pred[:, 0] 
        fvc_pred = y_pred[:, 1] #looks like we are using the median y_pred (0.5 quantile) as our final prediction


        sigma_clip = tf.maximum(sigma, sigma_min) #can't go lower than confidence of 70
        delta = tf.abs(y_true[:, 0] - fvc_pred) #delta is error
        delta = tf.minimum(delta, delta_max)
        sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
        metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
        return tf.keras.backend.mean(metric).numpy() #return average laplace log score over all examples
        
    return loss

def pinball_qloss(quantiles):
    '''
    Pinball Loss, a metric to measure quantile regression
    Avg pinball loss over all examples
    quantiles is some iterable which holds the quantiles we want to compute loss at
    
    Returns a loss function that returns avg pinball loss over all examples
    
    y_true is shape (N x 1)
    y_pred is shape (N x 3)
    '''
    q = tf.constant(np.array([quantiles]), dtype=tf.float32) #for tensorflow compatibility                   
    def loss(y_true, y_pred): #loss function to return
        e = y_true - y_pred #the error between prediction and true value for each example (shape N x 3)
        v = tf.maximum(q*e, (q-1)*e) #the definition of pinball loss
        return tf.keras.backend.mean(v).numpy() #return avg over all examples
    
    return loss

def weighted_loss(weights, loss_functions):
    '''
    Generic function that takes the weighted average of multiple loss functions
    weights is an iterable of the weights for the corresponding loss function
    loss_functions is an iterable of loss functions of signature f(y_true, y_pred)
    weights should be same length as loss_functions
    
    Returns a loss function that takes weighted average of given loss functions
    '''
    
    weights = np.array(weights) #ensure numpy compatibility
    
    def loss(y_true, y_pred): #loss function to return
        losses = np.array([lf(y_true, y_pred) for lf in loss_functions]) 
        return np.sum(weights * losses) #then compute weighted average (scalar)
    
    return loss


def mloss(w):
    '''
    Convenient function wrapper for weighted_loss with weights and losses already here
    '''
    weights = [w, 1 - w]
    losses = [pinball_qloss([0.2, 0.5, 0.8]), laplace_log_score()]
    lf = weighted_loss(weights, losses)
    
    def loss(y_true, y_pred):
        return lf(y_true, y_pred)
        
    return loss


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 13
import tensorflow as tf
from sklearn.linear_model import LinearRegression, GammaRegressor, TweedieRegressor


def make_model(): #let's start with a simple tabular model; integrate images later
    '''
    creates and returns a model, but does not fit it
    '''
    
    
    loss = mloss(0.8) #loss has signature f(y_true, y_pred)
    
    model = LinearRegression()
    
    return model


## === cell 14
train_df #just look over train_df again


## === cell 15


model = make_model()

drop_features = ['Patient', 'FVC', 'Weeks', 'Percent'] #features to drop from X training data

X_train = train_df.drop(drop_features, axis = 1)
y_train = train_df['FVC']

model.fit(X_train, y_train)
X_train


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3332679863.py in <cell line: 0>()
     15 y_train = train_df['FVC']
     16 
---> 17 model.fit(X_train, y_train)
     18 X_train

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in fit(self, X, y, sample_weight)
    646         accept_sparse = False if self.positive else ["csr", "csc", "coo"]
    647 
--> 648         X, y = self._validate_data(
    649             X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
    650         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __array__(self, dtype, copy)
   2151     ) -> np.ndarray:
   2152         values = self._values
-> 2153         arr = np.asarray(values, dtype=dtype)
   2154         if (
   2155             astype_is_view(values.dtype, arr.dtype)

ValueError: could not convert string to float: 'Male'

## === cell 16
from sklearn.model_selection import RandomizedSearchCV

'''
params = {
    
            }
hyper_search = RandomizedSearchCV(model, param_distributions=params, n_iter = 20)'''


## === cell 18
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import make_scorer, mean_absolute_error

NFOLDS = 6
gkf = GroupKFold(n_splits = NFOLDS) #use groupkfold to prevent same patient in training and test set
groups = train_df['Patient'].values

scorer = make_scorer(mean_absolute_error)




def temp_loss(y_true, y_pred): #just a temp loss function to wrap around laplace log score
    CONFIDENCE = c #c is our loop variable
    y_true = np.expand_dims(y_true, -1)
    y_mod = np.zeros((y_pred.shape[0],3))
    y_mod[:, 1] = y_pred
    y_mod[:, 0] = y_pred - CONFIDENCE / 2
    y_mod[:, 2] = y_pred + CONFIDENCE / 2
    return laplace_log_score()(y_true.astype('float32'),y_mod.astype('float32'))



conf = np.arange(100, 401, 5) #various confidence values
conf_df = pd.DataFrame(index = conf, columns = ['mean score', 'std score'])
conf_df.index.name = 'Confidence'
for c in conf: #optimize over various confidence values
    scorer = make_scorer(temp_loss)
    cv_scores = cross_val_score(model, X_train, y_train, cv = gkf, groups = groups, scoring = scorer)
    
    avg_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    conf_df.loc[c, :] = [avg_score, std_score]
    
    '''
    confidence = np.mean(cv_scores) #temp confidence value for submission
    print(confidence)
    '''


num_std = 2.3 #this number seems to produce worst cases that line up pretty well with leaderboard, at least for simple LinearRegression
conf_df['worst case'] = (conf_df['mean score'] + num_std * conf_df['std score'] )
conf_df = conf_df.convert_dtypes() #ensure numeric

conf_df


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2857446282.py in <cell line: 0>()
     34     scorer = make_scorer(temp_loss)
     35     #print('With confidence value', c)
---> 36     cv_scores = cross_val_score(model, X_train, y_train, cv = gkf, groups = groups, scoring = scorer)
     37     #print(cv_scores)
     38 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py in cross_val_score(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch, error_score)
    513     scorer = check_scoring(estimator, scoring=scoring)
    514 
--> 515     cv_results = cross_validate(
    516         estimator=estimator,
    517         X=X,

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py in cross_validate(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch, return_train_score, return_estimator, error_score)
    283     )
    284 
--> 285     _warn_or_raise_about_fit_failures(results, error_score)
    286 
    287     # For callabe scoring, the return type is only know after calling. If the

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py in _warn_or_raise_about_fit_failures(results, error_score)
    365                 f"Below are more details about the failures:\n{fit_errors_summary}"
    366             )
--> 367             raise ValueError(all_fits_failed_message)
    368 
    369         else:

ValueError: 
All the 6 fits failed.
It is very likely that your model is misconfigured.
You can try to debug the error by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
1 fits failed with the following error:
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py", line 686, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py", line 648, in fit
    X, y = self._validate_data(
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/base.py", line 584, in _validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py", line 1106, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py", line 879, in check_array
    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py", line 185, in _asarray_with_order
    array = numpy.asarray(array, order=order, dtype=dtype)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py", line 2153, in __array__
    arr = np.asarray(values, dtype=dtype)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: could not convert string to float: 'Female'

--------------------------------------------------------------------------------
5 fits failed with the following error:
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py", line 686, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py", line 648, in fit
    X, y = self._validate_data(
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/base.py", line 584, in _validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py", line 1106, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py", line 879, in check_array
    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py", line 185, in _asarray_with_order
    array = numpy.asarray(array, order=order, dtype=dtype)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py", line 2153, in __array__
    arr = np.asarray(values, dtype=dtype)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: could not convert string to float: 'Male'


## === cell 19
best = conf_df.nsmallest(10, columns = ['worst case'], keep = 'all')
best = best.applymap('{:,.4f}'.format) #format for output to 4 decimal places

best#.loc[[200,270,300,350]]


## --- ERROR in cell 19, traceback:
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

KeyError: 'worst case'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1713821580.py in <cell line: 0>()
      1 #keep the best confidence values
----> 2 best = conf_df.nsmallest(10, columns = ['worst case'], keep = 'all')
      3 best = best.applymap('{:,.4f}'.format) #format for output to 4 decimal places
      4 
      5 best#.loc[[200,270,300,350]]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in nsmallest(self, n, columns, keep)
   7754         Nauru         337000  182      NR
   7755         """
-> 7756         return selectn.SelectNFrame(self, n=n, keep=keep, columns=columns).nsmallest()
   7757 
   7758     @doc(

/usr/local/lib/python3.11/dist-packages/pandas/core/methods/selectn.py in nsmallest(self)
     59     @final
     60     def nsmallest(self):
---> 61         return self.compute("nsmallest")
     62 
     63     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/methods/selectn.py in compute(self, method)
    197 
    198         for column in columns:
--> 199             dtype = frame[column].dtype
    200             if not self.is_valid_dtype_n_method(dtype):
    201                 raise TypeError(

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

KeyError: 'worst case'

## === cell 20
import matplotlib.pyplot as plt

plt.bar(X_train.columns.values, model.coef_)
plt.xticks(rotation = 70)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/758671930.py in <cell line: 0>()
      1 import matplotlib.pyplot as plt
      2 
----> 3 plt.bar(X_train.columns.values, model.coef_)
      4 plt.xticks(rotation = 70)

AttributeError: 'LinearRegression' object has no attribute 'coef_'

## === cell 21
pred_train = model.predict(X_train)
pred_train


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1901534468.py in <cell line: 0>()
----> 1 pred_train = model.predict(X_train)
      2 pred_train

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
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __array__(self, dtype, copy)
   2151     ) -> np.ndarray:
   2152         values = self._values
-> 2153         arr = np.asarray(values, dtype=dtype)
   2154         if (
   2155             astype_is_view(values.dtype, arr.dtype)

ValueError: could not convert string to float: 'Male'

## === cell 22
import random

p = random.choice(train_df['Patient'].unique())

mask = train_df['Patient'] == p
ser = pd.Series(pred_train)[mask]

temp_df = train_df.loc[mask, ['Weeks', 'FVC']].join(pd.Series(pred_train, name = 'FVC_pred')[mask])



temp_df.plot(x = 'Weeks', y = ['FVC', 'FVC_pred'])


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2898482324.py in <cell line: 0>()
      6 mask = train_df['Patient'] == p
      7 #print(train_df.loc[mask, ['Weeks', 'FVC']])
----> 8 ser = pd.Series(pred_train)[mask]
      9 
     10 temp_df = train_df.loc[mask, ['Weeks', 'FVC']].join(pd.Series(pred_train, name = 'FVC_pred')[mask])

NameError: name 'pred_train' is not defined

## === cell 23
from sklearn.pipeline import Pipeline


'''
pipeline = Pipeline([
                ('fe', MyFeatureEngineerer()),
                ('ct', col_trans),
                ('model', make_model())
            ])


pipeline.fit(X_train, y_train)'''


## === cell 24
input_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
input_df #preprocess this to turn into test_df


## === cell 25




eng = MyFeatureEngineerer()
eng.fit(input_df)
input_df2 = input_df.drop(['FVC', 'Weeks'], axis = 1) #this info is stored in FirstFVC and FirstWeek of eng
print(input_df)
all_weeks = pd.DataFrame(np.array(range(-12, 134)), columns = ['Weeks'])
patient_weeks = pd.DataFrame()


for p in input_df['Patient'].unique(): #this loop creates rows for every week/patient combo
    tdf = all_weeks.copy()
    tdf['Patient'] = p
    patient_weeks = patient_weeks.append(tdf, ignore_index = True)

temp_df = patient_weeks.merge(input_df2, on = 'Patient')

print(temp_df)
print(eng.df_)
new_df = eng.transform(temp_df)
new_df


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3383702395.py in <cell line: 0>()
     20     tdf = all_weeks.copy()
     21     tdf['Patient'] = p
---> 22     patient_weeks = patient_weeks.append(tdf, ignore_index = True)
     23 
     24 temp_df = patient_weeks.merge(input_df2, on = 'Patient')

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 26
new_df['FVC'] = 0 #need this for column transforming, can drop afterwards
print(new_df.columns)
new_df = col_trans.transform(new_df) #col_trans already fit on train, don't worry
test_df = pd.DataFrame(new_df, columns = transformed_col_names(col_trans))
test_df


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/987405043.py in <cell line: 0>()
----> 1 new_df['FVC'] = 0 #need this for column transforming, can drop afterwards
      2 print(new_df.columns)
      3 new_df = col_trans.transform(new_df) #col_trans already fit on train, don't worry
      4 test_df = pd.DataFrame(new_df, columns = transformed_col_names(col_trans))
      5 test_df

IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

## === cell 27
X_test = test_df.drop(drop_features, axis = 1) 
X_test


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3970795743.py in <cell line: 0>()
----> 1 X_test = test_df.drop(drop_features, axis = 1)
      2 X_test

NameError: name 'test_df' is not defined

## === cell 28
pred = model.predict(X_test)


pred


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2116212246.py in <cell line: 0>()
----> 1 pred = model.predict(X_test)
      2 
      3 
      4 pred

NameError: name 'X_test' is not defined

## === cell 29
sub_df = patient_weeks.join(pd.Series(pred, name = 'FVC'))
sub_df


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3252819594.py in <cell line: 0>()
----> 1 sub_df = patient_weeks.join(pd.Series(pred, name = 'FVC'))
      2 sub_df

NameError: name 'pred' is not defined

## === cell 30
plt.figure(figsize = (17,10))
for i, (patient, frame) in enumerate(sub_df.groupby('Patient')):
    ax = plt.subplot(2,3, i+1)
    frame[['Weeks', 'FVC']].plot(x = 'Weeks', y = 'FVC', title = patient, ax = ax)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1240434273.py in <cell line: 0>()
      1 #visualize results
      2 plt.figure(figsize = (17,10))
----> 3 for i, (patient, frame) in enumerate(sub_df.groupby('Patient')):
      4     ax = plt.subplot(2,3, i+1)
      5     frame[['Weeks', 'FVC']].plot(x = 'Weeks', y = 'FVC', title = patient, ax = ax)

NameError: name 'sub_df' is not defined

## === cell 31

sub_df['Patient_Week'] = sub_df['Patient'] + '_' + sub_df['Weeks'].astype(str)
sub_df['Confidence'] = 240 #choose best confidence (best worst case), as determined by cross-val


sub_df


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/888949536.py in <cell line: 0>()
      1 #format the output
      2 
----> 3 sub_df['Patient_Week'] = sub_df['Patient'] + '_' + sub_df['Weeks'].astype(str)
      4 sub_df['Confidence'] = 240 #choose best confidence (best worst case), as determined by cross-val
      5 

NameError: name 'sub_df' is not defined

## === cell 32
sub_df[['Patient_Week', 'FVC', 'Confidence']].to_csv('submission.csv', index = False)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1076270644.py in <cell line: 0>()
----> 1 sub_df[['Patient_Week', 'FVC', 'Confidence']].to_csv('submission.csv', index = False)

NameError: name 'sub_df' is not defined

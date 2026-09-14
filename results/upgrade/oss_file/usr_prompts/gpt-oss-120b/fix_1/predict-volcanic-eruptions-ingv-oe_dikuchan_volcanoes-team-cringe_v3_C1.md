# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Given readings from several seismic sensors around a volcano, estimate how long it will be until the next eruption.

## Metric
Mean absolute error (MAE) between the predicted loss and the actual loss.

## Submission Format
For every id in the test set, you should predict the time until the next eruption. The file should contain a header and have the following format:

```
segment_id,time_to_eruption
1,1
2,2
3,3
etc.
```

## Data
### Dataset Description

#### Files
**train.csv** Metadata for the train files.

- `segment_id`: ID code for the data segment. Matches the name of the associated data file.
- `time_to_eruption`: The target value, the time until the next eruption.

**[train|test]/*.csv**: the data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano. The readings have been normalized within each segment, in part to ensure that the readings fall within the range of int16 values. If you are using the Pandas library you may find that you still need to load the data as float32 due to the presence of some nulls.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tsfresh==0.21.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
        input/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
        working/
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
```

-> data/predict-volcanic-eruptions-ingv-oe/sample_submission.csv has 444 rows and 2 columns.
The columns are: segment_id, time_to_eruption

-> data/predict-volcanic-eruptions-ingv-oe/test/1003520023.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1004346803.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1007996426.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1009749143.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1016956864.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1024522044.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1028325789.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> (stopped after 10 files for performance)

# 5. Target score

5176291.478340709

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

import optuna

from xgboost import XGBRegressor

## === cell 2
data_folder = Path('../input/predict-volcanic-eruptions-ingv-oe/')

df = pd.read_csv(data_folder / 'train.csv')

df.head().T

## === cell 4
df = df.sort_values('time_to_eruption')

time_min = df.head(1).iloc[0]['segment_id']
time_max = df.tail(1).iloc[0]['segment_id']

df_min = pd.read_csv(data_folder / 'train' / f'{time_min}.csv')
df_max = pd.read_csv(data_folder / 'train' / f'{time_max}.csv')

df_min['time_to_eruption'] = df.head(1).iloc[0]['time_to_eruption']
df_max['time_to_eruption'] = df.tail(1).iloc[0]['time_to_eruption']

## === cell 5
df_min.describe()

## === cell 6
df_max.describe()

## === cell 8
for i in range(10):
    sensor = f'sensor_{i + 1}'
    if df_min[sensor].isnull().all() or df_max[sensor].isnull().all():
        del df_min[sensor]
        del df_max[sensor]

## === cell 9
def plot_sensors_data(ld, rd):
    figure, axs = plt.subplots(5, 2, figsize=(16, 20))

    for (i, c) in zip(range(5), df_min.columns):
        axs[i, 0].plot(ld[c])
        axs[i, 0].set_title(f'Minimal time to eruption, {c}')
    
        axs[i, 1].plot(rd[c])
        axs[i, 1].set_title(f'Maximal time to eruption, {c}')

## === cell 10
plot_sensors_data(df_min, df_max)

## === cell 12
plot_sensors_data(df_min[:100], df_max[:100])

## === cell 14

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

## === cell 15
tsfresh_parameters = MinimalFCParameters()

## === cell 17
del tsfresh_parameters['length']

tsfresh_parameters['skewness'] = None
tsfresh_parameters['kurtosis'] = None
tsfresh_parameters['last_location_of_maximum'] = None
tsfresh_parameters['first_location_of_maximum'] = None
tsfresh_parameters['last_location_of_minimum'] = None
tsfresh_parameters['first_location_of_minimum'] = None
tsfresh_parameters['first_location_of_minimum'] = None
tsfresh_parameters['benford_correlation'] = None
tsfresh_parameters['percentage_of_reoccurring_values_to_all_values'] = None
tsfresh_parameters['percentage_of_reoccurring_datapoints_to_all_datapoints'] = None

tsfresh_parameters['number_peaks'] =  [
    {'n': 1}, 
    {'n': 3}, 
    {'n': 5}, 
    {'n': 10}, 
    {'n': 50}
]
tsfresh_parameters['binned_entropy']  = [
    {'max_bins': 10}
]
tsfresh_parameters['fft_aggregated']  = [
    {'aggtype': 'centroid'},
    {'aggtype': 'variance'},
    {'aggtype': 'skew'},
    {'aggtype': 'kurtosis'}
]
tsfresh_parameters['autocorrelation'] = [
    {'lag': 0},
    {'lag': 1},
    {'lag': 2},
    {'lag': 3},
    {'lag': 4},
    {'lag': 5},
    {'lag': 6},
    {'lag': 7},
    {'lag': 8},
    {'lag': 9}
]
tsfresh_parameters['agg_autocorrelation'] = [
    {'f_agg': 'mean', 'maxlag': 40},
    {'f_agg': 'median', 'maxlag': 40},
    {'f_agg': 'var', 'maxlag': 40}
]
tsfresh_parameters['friedrich_coefficients'] = [
    {'coeff': 0, 'm': 3, 'r': 30},
    {'coeff': 1, 'm': 3, 'r': 30},
    {'coeff': 2, 'm': 3, 'r': 30},
    {'coeff': 3, 'm': 3, 'r': 30}
]
tsfresh_parameters['count_above'] = [{'t': 0}]
tsfresh_parameters['count_below'] = [{'t': 0}]

## === cell 19
def preprocess_timeseries(data, parameters, is_train=True):
    df_result = None
    
    if is_train:
        segments = df.iterrows()
    else:
        segments = enumerate(os.listdir(data_folder / 'test'))
        
    for idx, row in segments:
        if is_train:
            segment, time_to_eruption = row
            segment_timeseries_path = data_folder / 'train/{segment}.csv'
        else:
            segment = row
            segment_timeseries_path = data_folder / 'test/{segment}'

        segment_timeseries = pd.read_csv(segment_timeseries_path)
        segment_timeseries = segment_timeseries.fillna(0).reset_index()
        segment_timeseries['id'] = idx
        
        extracted_features = extract_features(segment_timeseries, \
                                              column_id='id', \
                                              column_sort='index', \
                                              disable_progressbar=True,
                                              default_fc_parameters=parameters)
        extracted_features['segment'] = segment
        
        if is_train:
            extracted_features['time_to_eruption'] = time_to_eruption

        if df_result is None:
            df_result = extracted_features
        else:
            df_result = pd.concat([df_result, extracted_features], \
                                  axis=0, \
                                  ignore_index=True, \
                                  sort=True)
            
        print(f'Processed segment #{idx}')
        
    return df_result

## === cell 21
def save(parameters):
    ts_train = pd.read_csv(data_folder / 'train.csv')
    df_train = preprocess_timeseries(ts_train, parameters)
    df_train.to_csv('train.csv', index=False)
    
    df_test = preprocess_timeseries(None, parameters, is_train=False)
    df_test.to_csv('test.csv', index=False)

## === cell 23
df_train = pd.read_csv('../input/volcanoes-ts-processed-with-tsfresh/train.csv')
df_train = df_train.dropna(axis='columns')

features = [c for c in df_train.columns if c not in ['time_to_eruption', 'segment']]

X = df_train[features]
y = df_train['time_to_eruption']

df_train.head()

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3107467918.py in <cell line: 0>()
----> 1 df_train = pd.read_csv('../input/volcanoes-ts-processed-with-tsfresh/train.csv')
      2 df_train = df_train.dropna(axis='columns')
      3 
      4 features = [c for c in df_train.columns if c not in ['time_to_eruption', 'segment']]
      5 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/volcanoes-ts-processed-with-tsfresh/train.csv'

## === cell 25
class LowImportanceSelector(TransformerMixin):
    def __init__(self, threshold, n_estimators=100):
        self.features = None
        self.threshold = threshold
        self.n_estimators = n_estimators

    def fit(self, X, y):
        estimator = RandomForestRegressor(n_estimators=self.n_estimators)
        estimator.fit(X, y)
    
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': estimator.feature_importances_
        })
        importances = importances[importances['importance'] > self.threshold]
        
        self.features = importances['feature']
        
        return self

    def transform(self, X):
        return X[self.features]

## === cell 27
class CorrelationSelector(TransformerMixin):
    def __init__(self, threshold):
        self.columns = None
        self.threshold = threshold
   
    def fit(self, X, y=None):
        X = X.copy()
        self.columns = set()
        C = X.corr()
        for i in range(len(C.columns)):
            for j in range(i):
                if (C.iloc[i, j] >= self.threshold) and (C.columns[j] not in self.columns):
                    c = C.columns[i]
                    self.columns.add(c)
                    if c in X.columns:
                        del X[c]
        
        return self

    def transform(self, X, y=None):
        X.drop(columns=list(self.columns)).shape
        
        return X.drop(columns=list(self.columns))

## === cell 28
def objective(trial, data=X, target=y):
    parameters = {
        'tree_method': 'gpu_hist',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        'n_estimators': 1000,
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
        'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1337)
    
    model = XGBRegressor(**parameters)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
    return mean_absolute_error(y_test, model.predict(X_test))

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/723205429.py in <cell line: 0>()
----> 1 def objective(trial, data=X, target=y):
      2     parameters = {
      3         'tree_method': 'gpu_hist',
      4         'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
      5         'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),

NameError: name 'X' is not defined

## === cell 29
"""
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f'Number of finished trials: {len(study.trials)}')
print(f'Best trial: {study.best_trial.params}')
"""

## === cell 30
parameters = {
    'lambda': 0.0020555245431348778, 
    'alpha': 0.11298627316540845, 
    'colsample_bytree': 0.6, 
    'subsample': 1.0, 
    'learning_rate': 0.01, 
    'max_depth': 20, 
    'random_state': 48, 
    'min_child_weight': 18
}

## === cell 31
pipe = Pipeline([
    ('correlation', CorrelationSelector(threshold=0.85)),
    ('scaler', MinMaxScaler()),
    ('xgboost', XGBRegressor(objective='reg:squarederror', n_estimators=1000, **parameters))
])

## === cell 32
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2937709293.py in <cell line: 0>()
----> 1 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

NameError: name 'X' is not defined

## === cell 33
pipe.fit(X_train, y_train)

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1500543500.py in <cell line: 0>()
----> 1 pipe.fit(X_train, y_train)

NameError: name 'X_train' is not defined

## === cell 34
pipe.score(X_test, y_test)

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3120704999.py in <cell line: 0>()
----> 1 pipe.score(X_test, y_test)

NameError: name 'X_test' is not defined

## === cell 35
pipe.fit(X, y)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/146570639.py in <cell line: 0>()
----> 1 pipe.fit(X, y)

NameError: name 'X' is not defined

## === cell 36
df_predict = pd.read_csv('../input/volcanoes-ts-processed-with-tsfresh/predict.csv')
df_predict = df_predict.dropna(axis='columns')
df_predict['segment'] = df_predict['segment'].apply(lambda s: s.split('.')[0])

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2559621306.py in <cell line: 0>()
----> 1 df_predict = pd.read_csv('../input/volcanoes-ts-processed-with-tsfresh/predict.csv')
      2 df_predict = df_predict.dropna(axis='columns')
      3 df_predict['segment'] = df_predict['segment'].apply(lambda s: s.split('.')[0])

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/volcanoes-ts-processed-with-tsfresh/predict.csv'

## === cell 37
target = pipe.predict(df_predict.drop(columns='segment'))
target = pd.DataFrame({
    'segment_id': df_predict['segment'], 'time_to_eruption': target
})
target.to_csv('submission.csv', index=False)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2678737246.py in <cell line: 0>()
----> 1 target = pipe.predict(df_predict.drop(columns='segment'))
      2 target = pd.DataFrame({
      3     'segment_id': df_predict['segment'], 'time_to_eruption': target
      4 })
      5 target.to_csv('submission.csv', index=False)

NameError: name 'df_predict' is not defined

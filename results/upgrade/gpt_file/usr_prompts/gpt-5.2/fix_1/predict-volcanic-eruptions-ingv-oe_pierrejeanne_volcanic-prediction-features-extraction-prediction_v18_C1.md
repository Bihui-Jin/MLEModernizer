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
librosa==0.11.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
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

11742913.23097345

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

from scipy.signal import butter,filtfilt

from scipy.stats import skew,kurtosis

from numpy.fft import fft,fftfreq

import librosa as lr
from librosa.core import stft, amplitude_to_db


from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score


from sklearn.model_selection import GridSearchCV


## === cell 1
filename_list = [] 
file_path = r'/kaggle/input/predict-volcanic-eruptions-ingv-oe/train'
all_files = glob.glob(file_path + "/*.csv")

filename_list.append(all_files)

filename_list.append(all_files)

list_sequence = [] 
for file in all_files:
    file = file.split("/")[-1]
    file = file.split(".")[-2]
    list_sequence.append(int(file))

df_list_sequence = pd.DataFrame(list_sequence)
df_list_sequence.columns=['segment_id']

## === cell 2
df_list_sequence

## === cell 3
filename_list_test = [] 
file_path_test = r'/kaggle/input/predict-volcanic-eruptions-ingv-oe/test'
all_files_test = glob.glob(file_path_test + "/*.csv")

filename_list_test.append(all_files_test)

filename_list_test.append(all_files_test)

list_sequence_test = [] 
for file in all_files_test:
    file = file.split("/")[-1]
    file = file.split(".")[-2]
    list_sequence_test.append(int(file))

df_list_sequence_test = pd.DataFrame(list_sequence_test)
df_list_sequence_test.columns=['segment_id']

## === cell 4
train = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train.csv")
train.head()

## === cell 6
#     """
#     remove frequency lower than cutoff
#     Parameters
#     ----------
#     data : array_like
#     cutoff: int
#     fs: float
#     order: int
#     Returns
#     -------
#     y:numpy.ndarray

    


## === cell 14
train_features = pd.read_csv("../input/features-from-version-16/train_features.csv")
train_features = train_features.drop(columns=['Unnamed: 0'])
train_features.head(2)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1580342383.py in <cell line: 0>()
----> 1 train_features = pd.read_csv("../input/features-from-version-16/train_features.csv")
      2 train_features = train_features.drop(columns=['Unnamed: 0'])
      3 train_features.head(2)

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/features-from-version-16/train_features.csv'

## === cell 15
test_features = pd.read_csv("../input/features-from-version-16/test_features.csv")
test_features = test_features.drop(columns=['Unnamed: 0'])
test_features.head(2)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/427466359.py in <cell line: 0>()
----> 1 test_features = pd.read_csv("../input/features-from-version-16/test_features.csv")
      2 test_features = test_features.drop(columns=['Unnamed: 0'])
      3 test_features.head(2)

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/features-from-version-16/test_features.csv'

## === cell 17
X_train = train_features.drop(['time_to_eruption','segment_id'], axis = 1)
y_train = train_features[['time_to_eruption']]

X_test = test_features.drop(['segment_id'], axis = 1)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1150844961.py in <cell line: 0>()
      1 ## scale X --- .value is used to convert the datafrram into a numpy array
      2 ## train set
----> 3 X_train = train_features.drop(['time_to_eruption','segment_id'], axis = 1)
      4 y_train = train_features[['time_to_eruption']]
      5 

NameError: name 'train_features' is not defined

## === cell 18
scalerx = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X_train_scaled = scalerx.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = scalerx.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


scalery = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
y_train_scaled = scalery.fit_transform(y_train)
y_train_scaled = pd.DataFrame(y_train_scaled, columns=y_train.columns, index=y_train.index)

y_train_scaled.head()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/859165285.py in <cell line: 0>()
      1 scalerx = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
----> 2 X_train_scaled = scalerx.fit_transform(X_train)
      3 X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
      4 #
      5 X_test_scaled = scalerx.fit_transform(X_test)

NameError: name 'X_train' is not defined

## === cell 19
Xy_train_scaled = pd.concat([X_train_scaled, y_train_scaled], axis=1, sort=False)

correlation_coef_scale = Xy_train_scaled[Xy_train_scaled.columns[1:]].corr()['time_to_eruption']
df_cor = pd.DataFrame(correlation_coef_scale.sort_values(ascending=False))
df_cor = df_cor[df_cor['time_to_eruption']<-0.24]

X_names = df_cor.T.columns
X_names = X_names.tolist()
len(X_names)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3945049507.py in <cell line: 0>()
----> 1 Xy_train_scaled = pd.concat([X_train_scaled, y_train_scaled], axis=1, sort=False)
      2 
      3 correlation_coef_scale = Xy_train_scaled[Xy_train_scaled.columns[1:]].corr()['time_to_eruption']
      4 df_cor = pd.DataFrame(correlation_coef_scale.sort_values(ascending=False))
      5 # df_cor.head(15)

NameError: name 'X_train_scaled' is not defined

## === cell 20
X_train_scaled = X_train_scaled[X_names].values
X_test_scaled = X_test_scaled[X_names].values
y_train_scaled = y_train_scaled.values

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3478132599.py in <cell line: 0>()
----> 1 X_train_scaled = X_train_scaled[X_names].values
      2 X_test_scaled = X_test_scaled[X_names].values
      3 y_train_scaled = y_train_scaled.values

NameError: name 'X_train_scaled' is not defined

## === cell 21
df_X_train = pd.DataFrame(X_train_scaled, columns=X_names)
df_y_train = pd.DataFrame(y_train_scaled, columns=['time_to_eruption'])
Xy_train = pd.concat([df_X_train, df_y_train], axis=1, sort=False)

Xy_train.plot(x="time_to_eruption", y=X_names[0:], kind = 'line', legend=False, 
                 subplots = True, sharex = True, figsize = (20,30), ls="none", marker="o", layout=(10,9))

plt.show()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3691986429.py in <cell line: 0>()
----> 1 df_X_train = pd.DataFrame(X_train_scaled, columns=X_names)
      2 df_y_train = pd.DataFrame(y_train_scaled, columns=['time_to_eruption'])
      3 Xy_train = pd.concat([df_X_train, df_y_train], axis=1, sort=False)
      4 
      5 Xy_train.plot(x="time_to_eruption", y=X_names[0:], kind = 'line', legend=False, 

NameError: name 'X_train_scaled' is not defined

## === cell 24
from sklearn.svm import SVR
svr_rbf = SVR(kernel = 'rbf',C=10, gamma=0.1, degree=2, epsilon=.1,coef0=0)
svr_rbf.fit(X_train, y_train)

predicted = svr_rbf.predict(X_test)
predicted2 = scalery.inverse_transform(predicted.reshape(-1,1))



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4118206955.py in <cell line: 0>()
      1 from sklearn.svm import SVR
      2 svr_rbf = SVR(kernel = 'rbf',C=10, gamma=0.1, degree=2, epsilon=.1,coef0=0)
----> 3 svr_rbf.fit(X_train, y_train)
      4 
      5 predicted = svr_rbf.predict(X_test)

NameError: name 'X_train' is not defined

## === cell 25
predicted2

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3975199138.py in <cell line: 0>()
----> 1 predicted2

NameError: name 'predicted2' is not defined

## === cell 27
submission = pd.DataFrame()
submission['segment_id'] = df_list_sequence_test["segment_id"]
submission['time_to_eruption'] = predicted
submission.head()
submission.to_csv('submission2.csv', header=True, index=False)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2187393060.py in <cell line: 0>()
      1 submission = pd.DataFrame()
      2 submission['segment_id'] = df_list_sequence_test["segment_id"]
----> 3 submission['time_to_eruption'] = predicted
      4 submission.head()
      5 submission.to_csv('submission2.csv', header=True, index=False)

NameError: name 'predicted' is not defined

## === cell 29
import xgboost as xgb

segment_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

gbm_param_grid = {
    'colsample_bytree': [0.3,0.7],
    'n_estimators': [25],
    'max_depth': range(2, 12)   #range between 2 and 11
}

gbm = xgb.XGBRegressor(n_estimators=10)

randomized_mse = RandomizedSearchCV(estimator=gbm,param_distributions=gbm_param_grid,
scoring="neg_mean_squared_error", n_iter=10,cv=4, verbose=1)


randomized_mse.fit(X,y)

print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1305737170.py in <cell line: 0>()
      3 
      4 # Create the DMatrix: housing_dmatrix
----> 5 segment_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
      6 
      7 gbm_param_grid = {

NameError: name 'X_train' is not defined

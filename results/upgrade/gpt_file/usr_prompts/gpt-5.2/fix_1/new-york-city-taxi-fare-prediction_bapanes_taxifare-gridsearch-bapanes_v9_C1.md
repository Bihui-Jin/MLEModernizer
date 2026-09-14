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
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

3.8575708313744257

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
print("hello moto")

## === cell 1
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import math 

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

print(tf.__version__)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def data_to_np(input_file):
    
    
    df = pd.read_csv(input_file, sep=',', nrows = 900000)
    
    header_names = ['pickup_longitude','pickup_latitude','dropoff_longitude',
                    'dropoff_latitude','passenger_count','distance']
    
    
    df_train = df[header_names]
    np_df_train = df_train.values
    
    df_label = df['fare_amount']
    np_df_label = df_label.values
    
    return np_df_train, np_df_label    

## === cell 4
def global_mean_per_column(mynp_train_list):
    
    sum_mean = 0
    
    for con in range(len(mynp_train_list)):
        sum_mean = sum_mean + np.mean(mynp_train_list[con], axis=0) 
    
    
    mean = sum_mean/len(mynp_train_list)
        
    return mean

## === cell 5
def global_std_per_column(mynp_train_list, global_mean):
    
    sum_mean_x2 = 0
    
    for con in range(len(mynp_train_list)):
        sum_mean_x2 += np.mean((mynp_train_list[con] - global_mean)**2, axis=0)
    
    
    std = np.sqrt(sum_mean_x2/len(mynp_train_list))
    
    return std


## === cell 6
def norm_mynp_train(mynp_train, mean, std):
    
    mynp_train_norm = (mynp_train - mean)/std
       
    return mynp_train_norm    

## === cell 8
mynp_train_0, mynp_label_0 = data_to_np('../input/my-taxi-fare-data/train_r0.csv')
mynp_train_1, mynp_label_1 = data_to_np('../input/my-taxi-fare-data/train_r1.csv')
mynp_train_2, mynp_label_2 = data_to_np('../input/my-taxi-fare-data/train_r2.csv')

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2682256079.py in <cell line: 0>()
----> 1 mynp_train_0, mynp_label_0 = data_to_np('../input/my-taxi-fare-data/train_r0.csv')
      2 mynp_train_1, mynp_label_1 = data_to_np('../input/my-taxi-fare-data/train_r1.csv')
      3 mynp_train_2, mynp_label_2 = data_to_np('../input/my-taxi-fare-data/train_r2.csv')

/tmp/ipykernel_11/3642759654.py in data_to_np(input_file)
      4     #of the output files (maybe we can fix this later)
      5 
----> 6     df = pd.read_csv(input_file, sep=',', nrows = 900000)
      7 
      8     header_names = ['pickup_longitude','pickup_latitude','dropoff_longitude',

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/my-taxi-fare-data/train_r0.csv'

## === cell 10
order = np.argsort(np.random.random(mynp_label_0.shape))

mynp_train_0 = mynp_train_0[order]
mynp_label_0 = mynp_label_0[order]

order = np.argsort(np.random.random(mynp_label_1.shape))

mynp_train_1 = mynp_train_1[order]
mynp_label_1 = mynp_label_1[order]

order = np.argsort(np.random.random(mynp_label_2.shape))

mynp_train_2 = mynp_train_2[order]
mynp_label_2 = mynp_label_2[order]

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1159123523.py in <cell line: 0>()
----> 1 order = np.argsort(np.random.random(mynp_label_0.shape))
      2 
      3 mynp_train_0 = mynp_train_0[order]
      4 mynp_label_0 = mynp_label_0[order]
      5 

NameError: name 'mynp_label_0' is not defined

## === cell 12
mynp_train_list = []
mynp_train_list.append(mynp_train_0)
mynp_train_list.append(mynp_train_1)
mynp_train_list.append(mynp_train_2)

mynp_label_list = []
mynp_label_list.append(mynp_label_0)
mynp_label_list.append(mynp_label_1)
mynp_label_list.append(mynp_label_2)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/812800557.py in <cell line: 0>()
      1 mynp_train_list = []
----> 2 mynp_train_list.append(mynp_train_0)
      3 mynp_train_list.append(mynp_train_1)
      4 mynp_train_list.append(mynp_train_2)
      5 

NameError: name 'mynp_train_0' is not defined

## === cell 14
global_mean = global_mean_per_column(mynp_train_list)
global_mean

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_11/664570670.py in <cell line: 0>()
----> 1 global_mean = global_mean_per_column(mynp_train_list)
      2 global_mean

/tmp/ipykernel_11/3002500203.py in global_mean_per_column(mynp_train_list)
      9     #mean_1 = np.mean(mynp_train_1, axis=0)
     10 
---> 11     mean = sum_mean/len(mynp_train_list)
     12 
     13     return mean

ZeroDivisionError: division by zero

## === cell 15
global_std = global_std_per_column(mynp_train_list,global_mean)
global_std



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1246060199.py in <cell line: 0>()
----> 1 global_std = global_std_per_column(mynp_train_list,global_mean)
      2 global_std
      3 
      4 #notice the small value of the std deviation of lat and lon variables

NameError: name 'global_mean' is not defined

## === cell 17
def build_model(shape_of_np_array):
    model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu, 
                               input_shape=(shape_of_np_array,)),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=["accuracy"])
    return model

## === cell 18
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('epoch')
    print('.'),



## === cell 20
df_test = pd.read_csv('../input/my-taxi-fare-data/test_r0.csv', sep=',')
df_test = df_test[['key','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','distance']]
df_test_fn = df_test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','distance']]

mynp_test = df_test_fn.values
mean_test = np.mean(mynp_test,axis=0)
std_test = mynp_test.std(axis=0)
mynp_test = (mynp_test - mean_test)/std_test

mynp_test

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/525296064.py in <cell line: 0>()
----> 1 df_test = pd.read_csv('../input/my-taxi-fare-data/test_r0.csv', sep=',')
      2 df_test = df_test[['key','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','distance']]
      3 df_test_fn = df_test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','distance']]
      4 
      5 mynp_test = df_test_fn.values

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/my-taxi-fare-data/test_r0.csv'

## === cell 24
mynp_train_norm_0 = norm_mynp_train(mynp_train_list[0], global_mean, global_std)
mynp_train_norm_1 = norm_mynp_train(mynp_train_list[1], global_mean, global_std)
mynp_train_norm_2 = norm_mynp_train(mynp_train_list[2], global_mean, global_std)

mynp_train_concat = np.concatenate((mynp_train_norm_0,mynp_train_norm_1,mynp_train_norm_2),axis=0)
mynp_label_concat = np.concatenate((mynp_label_0,mynp_label_1,mynp_label_2),axis=0)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2932085123.py in <cell line: 0>()
      1 #normaalization
----> 2 mynp_train_norm_0 = norm_mynp_train(mynp_train_list[0], global_mean, global_std)
      3 mynp_train_norm_1 = norm_mynp_train(mynp_train_list[1], global_mean, global_std)
      4 mynp_train_norm_2 = norm_mynp_train(mynp_train_list[2], global_mean, global_std)
      5 

IndexError: list index out of range

## === cell 26
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

## === cell 29
mynp_train_norm_0.shape

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1155277485.py in <cell line: 0>()
----> 1 mynp_train_norm_0.shape

NameError: name 'mynp_train_norm_0' is not defined

## === cell 30

grid_train = mynp_train_concat[:200]
grid_label = mynp_label_concat[:200]

print(grid_train.shape, grid_label.shape)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2776559476.py in <cell line: 0>()
      1 #reading a sub-array from the big one for the grid study
      2 
----> 3 grid_train = mynp_train_concat[:200]
      4 grid_label = mynp_label_concat[:200]
      5 

NameError: name 'mynp_train_concat' is not defined

## === cell 32
model_for_grid = KerasClassifier(build_fn=build_model, shape_of_np_array = mynp_train_0.shape[1], validation_split=0.2, verbose = 0) 
epochs = [1,3,5,7,10] 
param_grid = dict(epochs=epochs) 
grid = GridSearchCV(estimator=model_for_grid, param_grid=param_grid, n_jobs=1) 
grid_result = grid.fit(grid_train, grid_label,callbacks=[early_stop, PrintDot()])

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3133987238.py in <cell line: 0>()
----> 1 model_for_grid = KerasClassifier(build_fn=build_model, shape_of_np_array = mynp_train_0.shape[1], validation_split=0.2, verbose = 0)
      2 epochs = [1,3,5,7,10]
      3 param_grid = dict(epochs=epochs)
      4 grid = GridSearchCV(estimator=model_for_grid, param_grid=param_grid, n_jobs=1)
      5 grid_result = grid.fit(grid_train, grid_label,callbacks=[early_stop, PrintDot()])

NameError: name 'KerasClassifier' is not defined

## === cell 33
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1355970931.py in <cell line: 0>()
----> 1 print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
      2 means = grid_result.cv_results_['mean_test_score']
      3 stds = grid_result.cv_results_['std_test_score']
      4 params = grid_result.cv_results_['params']
      5 for mean, stdev, param in zip(means, stds, params):

NameError: name 'grid_result' is not defined

## === cell 34
model_after_gridSearch = build_model(mynp_train_0.shape[1]) 




EPOCHS = 3

model_after_gridSearch.fit(mynp_train_concat, mynp_label_concat, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2076905061.py in <cell line: 0>()
----> 1 model_after_gridSearch = build_model(mynp_train_0.shape[1])
      2 
      3 
      4 #for con in range(len(mynp_train_super_array)):
      5 

NameError: name 'mynp_train_0' is not defined

## === cell 36
test_predictions = model_after_gridSearch.predict(mynp_test).flatten()
test_predictions

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2944254608.py in <cell line: 0>()
      1 #test_predictions = test_predictions_total/len(mynp_train_super_array)
      2 #test_predictions = test_predictions_total
----> 3 test_predictions = model_after_gridSearch.predict(mynp_test).flatten()
      4 test_predictions

NameError: name 'model_after_gridSearch' is not defined

## === cell 39
test_key_array = df_test['key'].values

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2259810122.py in <cell line: 0>()
----> 1 test_key_array = df_test['key'].values

NameError: name 'df_test' is not defined

## === cell 40
test_key_array

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/376977661.py in <cell line: 0>()
----> 1 test_key_array

NameError: name 'test_key_array' is not defined

## === cell 41
df_output = pd.DataFrame({'key': test_key_array,'fare_amount': test_predictions})

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1034540635.py in <cell line: 0>()
----> 1 df_output = pd.DataFrame({'key': test_key_array,'fare_amount': test_predictions})

NameError: name 'test_key_array' is not defined

## === cell 42
df_output

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2808490814.py in <cell line: 0>()
----> 1 df_output

NameError: name 'df_output' is not defined

## === cell 43
df_output.to_csv('submission_file.csv', index = False)

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1181764526.py in <cell line: 0>()
----> 1 df_output.to_csv('submission_file.csv', index = False)

NameError: name 'df_output' is not defined

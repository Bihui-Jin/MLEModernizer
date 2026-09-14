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
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

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
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

0.1141505871263613

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import numpy as np
import pandas as pd
import gc
import os
import random
import matplotlib.pyplot as plt

## === cell 2
train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')

train['dcount'] = train.groupby('breath_id')['id'].transform('cumcount')
test['dcount'] = test.groupby('breath_id')['id'].transform('cumcount')

train['uo'] = 80 - train.groupby('breath_id')['u_out'].transform('sum')
test['uo'] = 80 - test.groupby('breath_id')['u_out'].transform('sum')

train['time_delta'] = (train['time_step'] - train.groupby('breath_id')['time_step'].shift(1)).fillna(0)
test['time_delta'] = (test['time_step'] - test.groupby('breath_id')['time_step'].shift(1)).fillna(0)

print(train.shape)
train.head()

## === cell 3
oof = pd.read_csv('../input/ventmodels/train-blend-1.csv')
oof.columns = ['id', 'pred']
train = train.merge(oof, on='id', how='left')
train

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1203514196.py in <cell line: 0>()
----> 1 oof = pd.read_csv('../input/ventmodels/train-blend-1.csv')
      2 oof.columns = ['id', 'pred']
      3 train = train.merge(oof, on='id', how='left')
      4 train

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/ventmodels/train-blend-1.csv'

## === cell 4
oof = pd.read_csv('../input/ventmodels/1336_submission.csv')
oof.columns = ['id', 'pred']
test = test.merge(oof, on='id', how='left')
test

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/177394259.py in <cell line: 0>()
----> 1 oof = pd.read_csv('../input/ventmodels/1336_submission.csv')
      2 oof.columns = ['id', 'pred']
      3 test = test.merge(oof, on='id', how='left')
      4 test

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/ventmodels/1336_submission.csv'

## === cell 5
train['error'] = (train['pressure'] - train['pred']).abs()
train.loc[train.u_out>0,'error'] = 0
train.loc[train.u_out==0,'error'].hist(bins=20)

## --- ERROR in cell 5, traceback:
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

KeyError: 'pred'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3560484946.py in <cell line: 0>()
----> 1 train['error'] = (train['pressure'] - train['pred']).abs()
      2 train.loc[train.u_out>0,'error'] = 0
      3 train.loc[train.u_out==0,'error'].hist(bins=20)

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

KeyError: 'pred'

## === cell 6
maxdrift = train.loc[train.u_out==0,'error'].mean() + 3*train.loc[train.u_out==0,'error'].std()
maxdrift

## --- ERROR in cell 6, traceback:
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

KeyError: 'error'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3037468302.py in <cell line: 0>()
----> 1 maxdrift = train.loc[train.u_out==0,'error'].mean() + 3*train.loc[train.u_out==0,'error'].std()
      2 maxdrift

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1366         with suppress(IndexingError):
   1367             tup = self._expand_ellipsis(tup)
-> 1368             return self._getitem_lowerdim(tup)
   1369 
   1370         # no multi-index, so validate all of the indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_lowerdim(self, tup)
   1063                 # We don't need to check for tuples here because those are
   1064                 #  caught by the _is_nested_tuple_indexer check above.
-> 1065                 section = self._getitem_axis(key, axis=i)
   1066 
   1067                 # We should never have a scalar section here, because

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1429         # fall thru to straight lookup
   1430         self._validate_key(key, axis)
-> 1431         return self._get_label(key, axis=axis)
   1432 
   1433     def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_label(self, label, axis)
   1379     def _get_label(self, label, axis: AxisInt):
   1380         # GH#5567 this will fail if the label is not present in the axis.
-> 1381         return self.obj.xs(label, axis=axis)
   1382 
   1383     def _handle_lowerdim_multi_index_axis0(self, tup: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in xs(self, key, axis, level, drop_level)
   4285         if axis == 1:
   4286             if drop_level:
-> 4287                 return self[key]
   4288             index = self.columns
   4289         else:

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

KeyError: 'error'

## === cell 8
p_coef = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
i_coef = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
setpoints = [10, 15, 20, 25, 30, 35]

## === cell 9
unique_pressures = train['pressure'].round(decimals=7).unique()
unique_pressures = list(np.sort(unique_pressures))
len(unique_pressures), unique_pressures[:10]

## === cell 10
maxdrift / (unique_pressures[1]-unique_pressures[0])

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2398209608.py in <cell line: 0>()
----> 1 maxdrift / (unique_pressures[1]-unique_pressures[0])

NameError: name 'maxdrift' is not defined

## === cell 11
max_pressure = 64.82099173863328
min_pressure = -1.895744294564641
diff_pressure = 0.0703021454512

## === cell 12
def generate_u_in(pressure, time_step, kp, ki, kt, integral=0):
    dt = np.diff(time_step, prepend=[0])
    preds = []
    for j in range(32):
        error = kt - pressure[j]
        integral += (error - integral) * (dt[j] / (dt[j] + 0.5))
        preds.append(kp * error + ki * integral)
    return preds

pressure = train[train['breath_id'] == 1]['pressure'].values 
timestep = train[train['breath_id'] == 1]['time_step'].values
u_in = generate_u_in(pressure, timestep, 0.8, 8.0, 20)
noise = train[train['breath_id'] == 1]['u_in'].values[:32] - u_in

plt.figure()
plt.plot(timestep[:32], train[train['breath_id'] == 1]['u_in'].values[:32], label='u_in')
plt.plot(timestep[:32], u_in, label='u_in_hat')
plt.legend()
plt.show()

## === cell 19
MAX_PRESSURE  = PRESSURE_MAX  = 64.82099173863328
MIN_PRESSURE  = PRESSURE_MIN  = -1.895744294564641
DIFF_PRESSURE = PRESSURE_STEP = 0.0703021454512
MIN_PRESSURE2 = PRESSURE_MIN2 = MIN_PRESSURE + DIFF_PRESSURE

def match_breath(u_in, u_out, timestep, kp, ki, kt):
    dt = np.diff(timestep)
    dt2 = dt / (dt + 0.5)
    in_len = np.sum(1 - u_out)
    preds = np.zeros(len(u_in)) - 999
    
    match=0
    for t in range(1, in_len):
        if preds[t - 1] != -999:
            P0 = preds[t - 1]
        else:
            P0 = np.arange(MIN_PRESSURE, MAX_PRESSURE + DIFF_PRESSURE, DIFF_PRESSURE)
            
        I0 = (u_in[t - 1] - kp * (kt - P0))/ki

        I11 = I0 + (kt - MIN_PRESSURE - I0) * dt2[t-1]
        u_in_hat1 = kp * (kt - MIN_PRESSURE) + ki * I11

        I12 = I0 + (kt - MIN_PRESSURE2 - I0) * dt2[t-1]
        u_in_hat2 = kp * (kt - MIN_PRESSURE2) + ki * I12

        slope = u_in_hat2 - u_in_hat1
        x_intersect = (u_in[t] - u_in_hat2) / slope
        
        diff = np.abs(np.round(x_intersect) - x_intersect)
        
        if diff.min() < 1e-10:
            match+=1
            pos = np.argmin(diff)
            
            if preds[t - 1] == -999:
                preds[t - 1] = P0[np.argmin(diff)]
                preds[t] = MIN_PRESSURE + int(x_intersect[pos] + 1) * DIFF_PRESSURE
            else:
                preds[t] = MIN_PRESSURE + (np.round(x_intersect) + 1) * DIFF_PRESSURE
            
    return preds, match

## === cell 20
i = 1
pressure = train[train['breath_id'] == i]['pressure'].values.copy()
timestep = train[train['breath_id'] == i]['time_step'].values.copy()
u_in = train[train['breath_id'] == i]['u_in'].values.copy()
u_out = train[train['breath_id'] == i]['u_out'].values.copy()
ypred = train[train['breath_id'] == i]['pred'].values.copy()

## --- ERROR in cell 20, traceback:
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

KeyError: 'pred'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1542255565.py in <cell line: 0>()
      4 u_in = train[train['breath_id'] == i]['u_in'].values.copy()
      5 u_out = train[train['breath_id'] == i]['u_out'].values.copy()
----> 6 ypred = train[train['breath_id'] == i]['pred'].values.copy()

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

KeyError: 'pred'

## === cell 21
%%time
res, match = match_breath(u_in, u_out, timestep, 1, 8, 20)

## === cell 22
import time


## === cell 25
starttime = time.time()

PIDTEST = []
count = 0
for i in test['breath_id'].unique()[:50]:
    count += 1
    ids = test[test['breath_id'] == i]['id'].values.copy()
    timestep = test[test['breath_id'] == i]['time_step'].values.copy()
    u_in = test[test['breath_id'] == i]['u_in'].values.copy()
    u_out = test[test['breath_id'] == i]['u_out'].values.copy()
    match = 0
    for P in p_coef:
        for I in i_coef:
            for SP in setpoints:
                res, match = match_breath(u_in, u_out, timestep, P, I, SP)
                if match>24:
                    dt = pd.DataFrame({
                        'id': ids,
                        'breath_id': i,
                        'P': P,
                        'I': I,
                        'SP': SP,
                        'pressure': res,
                    })                    
                    PIDTEST.append(dt)
                    print(i, count, match, P, I, SP, (time.time()-starttime), (time.time()-starttime)/count )
                    break
            if match>24:
                break
        if match>24:
            break
            
PIDTEST = pd.concat(PIDTEST).reset_index(drop=True)
PIDTEST.shape

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3134154914.py in <cell line: 0>()
     32             break
     33 
---> 34 PIDTEST = pd.concat(PIDTEST).reset_index(drop=True)
     35 PIDTEST.shape

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in concat(objs, axis, join, ignore_index, keys, levels, names, verify_integrity, sort, copy)
    380         copy = False
    381 
--> 382     op = _Concatenator(
    383         objs,
    384         axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in __init__(self, objs, axis, join, keys, levels, names, ignore_index, verify_integrity, copy, sort)
    443         self.copy = copy
    444 
--> 445         objs, keys = self._clean_keys_and_objs(objs, keys)
    446 
    447         # figure out what our result ndim is going to be

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in _clean_keys_and_objs(self, objs, keys)
    505 
    506         if len(objs_list) == 0:
--> 507             raise ValueError("No objects to concatenate")
    508 
    509         if keys is None:

ValueError: No objects to concatenate

## === cell 27
PIDTEST.to_csv('pid-test-1.csv', index=False)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3958362336.py in <cell line: 0>()
----> 1 PIDTEST.to_csv('pid-test-1.csv', index=False)

AttributeError: 'list' object has no attribute 'to_csv'

## === cell 28

BIDtrain = []
for SP in setpoints:
    for P in p_coef:
        train['u_ctrl'] = ((SP - train['u_in']/P)).round(decimals=7)
        
        train['isclass'] = 0
        train.loc[train['u_ctrl'].isin(unique_pressures), 'isclass'] = 1
        
        dt = train.loc[(train.u_out==0)&(train.dcount>=1)].groupby('breath_id')[['isclass','uo']].agg({'isclass':'sum', 'uo':'first'}).reset_index().sort_values('isclass', ascending=False).reset_index(drop=True)
        dt = dt.loc[dt['isclass']>=(dt['uo']-3)]
        if dt.shape[0]>0:
            print('matches:',dt.shape[0], 'P=',P, 'SP=', SP)
            dt['P'] = P
            dt['SP'] = SP
            BIDtrain.append(dt)
            

BIDtrain = pd.concat(BIDtrain)
print(BIDtrain.shape)
BIDtrain.head(10)

## === cell 29

BIDtest = []
for SP in setpoints:
    for P in p_coef:
        test['u_ctrl'] = ((SP - test['u_in']/P)).round(decimals=7)
        
        test['isclass'] = 0
        test.loc[test['u_ctrl'].isin(unique_pressures), 'isclass'] = 1
        
        dt = test.loc[(test.u_out==0)&(test.dcount>=1)].groupby('breath_id')[['isclass','uo']].agg({'isclass':'sum', 'uo':'first'}).reset_index().sort_values('isclass', ascending=False).reset_index(drop=True)
        dt = dt.loc[dt['isclass']>=(dt['uo']-3)]
        if dt.shape[0]>0:
            print('matches:',dt.shape[0], 'P=',P, 'SP=', SP)
            dt['P'] = P
            dt['SP'] = SP
            BIDtest.append(dt)
            
BIDtest = pd.concat(BIDtest)
print(BIDtest.shape)
BIDtest.head(10)

## === cell 30
for i in range(10):
    bid = BIDtrain.iloc[i]
    P = bid.P
    SP = bid.SP
    tmp = train.loc[train.breath_id == bid.breath_id].copy()
    tmp['u_ctrl'] =  (SP - tmp['u_in']/P)
    tmp.loc[(tmp.u_out==0)&(tmp.dcount>=0)].plot( x='time_step', y=['pressure', 'u_ctrl'], title = 'P='+str(P) + ' SP:'+str(SP) )

## === cell 31
for i in range(10):
    bid = BIDtest.iloc[i]
    P = bid.P
    SP = bid.SP
    tmp = test.loc[test.breath_id == bid.breath_id].copy()
    tmp['u_ctrl'] =  (SP - tmp['u_in']/P)
    tmp.loc[(tmp.u_out==0)&(tmp.dcount>=0)].plot( x='time_step', y=['u_ctrl'], title = 'P='+str(P) + ' SP:'+str(SP) )

## === cell 32
test.head()

## === cell 34
test = test.merge(BIDtest[['breath_id','P','SP']],on='breath_id',how='left')
test['pred'] = (test.SP - test['u_in']/test.P).round(decimals=7)
test_ids = BIDtest.breath_id.values

tmp = test.loc[(test.dcount>0)&(test.breath_id.isin(test_ids))]
tmp

## === cell 35
sub = pd.read_csv('../input/ventmodels/Exp-102-ResBiLSTM-v2-CustomLoss-v1-AdamW-LogUin-30Fold-Seed2025-FineTune.csv')

sub = sub.merge(tmp[['id','pred']],on='id',how='left')

sub.loc[sub.pred.notna(),'pressure'] = sub.loc[sub.pred.notna(),'pred']

del sub['pred']
sub

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3061692352.py in <cell line: 0>()
----> 1 sub = pd.read_csv('../input/ventmodels/Exp-102-ResBiLSTM-v2-CustomLoss-v1-AdamW-LogUin-30Fold-Seed2025-FineTune.csv')
      2 
      3 sub = sub.merge(tmp[['id','pred']],on='id',how='left')
      4 
      5 sub.loc[sub.pred.notna(),'pressure'] = sub.loc[sub.pred.notna(),'pred']

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/ventmodels/Exp-102-ResBiLSTM-v2-CustomLoss-v1-AdamW-LogUin-30Fold-Seed2025-FineTune.csv'

## === cell 37
tmp = pd.read_csv('pid-test-1.csv')
tmp = tmp.loc[ tmp.pressure > - 999, ['id','pressure'] ].reset_index(drop=True)
tmp.columns = ['id', 'pred']
tmp.head()

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2393584749.py in <cell line: 0>()
----> 1 tmp = pd.read_csv('pid-test-1.csv')
      2 tmp = tmp.loc[ tmp.pressure > - 999, ['id','pressure'] ].reset_index(drop=True)
      3 tmp.columns = ['id', 'pred']
      4 tmp.head()

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

FileNotFoundError: [Errno 2] No such file or directory: 'pid-test-1.csv'

## === cell 38
sub = sub.merge(tmp[['id','pred']],on='id',how='left')
sub.loc[sub.pred.notna(),'pressure'] = sub.loc[sub.pred.notna(),'pred']

del sub['pred']
sub

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1678302579.py in <cell line: 0>()
----> 1 sub = sub.merge(tmp[['id','pred']],on='id',how='left')
      2 sub.loc[sub.pred.notna(),'pressure'] = sub.loc[sub.pred.notna(),'pred']
      3 
      4 del sub['pred']
      5 sub

NameError: name 'sub' is not defined

## === cell 39
sub.to_csv('submission-postprocessing.csv', index=False)

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2991825254.py in <cell line: 0>()
----> 1 sub.to_csv('submission-postprocessing.csv', index=False)

NameError: name 'sub' is not defined

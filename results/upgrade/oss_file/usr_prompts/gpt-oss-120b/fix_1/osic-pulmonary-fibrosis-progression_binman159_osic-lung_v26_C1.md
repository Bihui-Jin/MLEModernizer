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
lightgbm==4.6.0
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
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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

-8.1278

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import typing as tp
import pydicom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor


## === cell 1
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
df = pd.concat([train_df, test_df], ignore_index=True)
df['Patient_Week'] = df['Patient'].astype(str) + '_ '+ df['Weeks'].astype(str)


## === cell 3
print('Shape of Training data: ', train_df.shape)
print('Shape of Test data: ', test_df.shape)


## === cell 4
def add_height(data)->'dataframe':
    data['Height'] = 0
    data['Height'] = data.apply(lambda x: x.FVC / (21.78 - (0.101 * x.Age)) if x.Height == 1 else x.FVC / (27.63 - (0.112 * x.Age)), axis=1)
    
def add_norm(data)->'dataframe':
    return (data - data.mean()) / data.std()


## === cell 5
df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1})
df['SmokingStatus'] = df['SmokingStatus'].map({'Currently smokes': 0, 'Never smoked': 1, 'Ex-smoker': 2})
df = df.drop('Patient_Week', axis=1)
df = df.set_index('Patient')
add_height(df)
df[df.columns[~df.columns.isin(['FVC', 'Sex'])]] = add_norm(df[df.columns[~df.columns.isin(['FVC', 'Sex'])]])
df.head()


## === cell 6
sub_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
sub_df.drop(['FVC'], axis=1, inplace=True)
sub_df['Patient'] = sub_df['Patient_Week'].apply(lambda x: x.split('_')[0])
sub_df['pred_Weeks'] = sub_df['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)

sub_df.head()


## === cell 7
test_FVC = pd.merge(sub_df, df, how='left', on=['Patient'])
test_FVC = test_FVC.rename(columns={'Patient_Week_x':'Patient_Week'})
test_FVC = test_FVC.drop(['pred_Weeks', 'FVC', 'Confidence'], axis=1)
test_FVC = test_FVC.groupby('Patient_Week').mean()
test_FVC[test_FVC.columns[~test_FVC.columns.isin(['Sex'])]] = add_norm(test_FVC[test_FVC.columns[~test_FVC.columns.isin(['Sex'])]])

test_FVC.head()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _agg_py_fallback(self, how, values, ndim, alt)
   1941         try:
-> 1942             res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
   1943         except Exception as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in agg_series(self, obj, func, preserve_dtype)
    863 
--> 864         result = self._aggregate_series_pure_python(obj, func)
    865 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _aggregate_series_pure_python(self, obj, func)
    884         for i, group in enumerate(splitter):
--> 885             res = func(group)
    886             res = extract_result(res)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in <lambda>(x)
   2453                 "mean",
-> 2454                 alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
   2455                 numeric_only=numeric_only,

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in mean(self, axis, skipna, numeric_only, **kwargs)
   6548     ):
-> 6549         return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
   6550 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in mean(self, axis, skipna, numeric_only, **kwargs)
  12419     ) -> Series | float:
> 12420         return self._stat_function(
  12421             "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
  12376 
> 12377         return self._reduce(
  12378             func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
   6456                 )
-> 6457             return op(delegate, skipna=skipna, **kwds)
   6458 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in f(values, axis, skipna, **kwds)
    146             else:
--> 147                 result = alt(values, axis=axis, skipna=skipna, **kwds)
    148 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in new_func(values, axis, skipna, mask, **kwargs)
    403 
--> 404         result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
    405 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in nanmean(values, axis, skipna, mask)
    719     the_sum = values.sum(axis, dtype=dtype_sum)
--> 720     the_sum = _ensure_numeric(the_sum)
    721 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in _ensure_numeric(x)
   1700             # GH#44008, GH#36703 avoid casting e.g. strings to numeric
-> 1701             raise TypeError(f"Could not convert string '{x}' to numeric")
   1702         try:

TypeError: Could not convert string 'ID00014637202177757139317' to numeric

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_10/824071582.py in <cell line: 0>()
      2 test_FVC = test_FVC.rename(columns={'Patient_Week_x':'Patient_Week'})
      3 test_FVC = test_FVC.drop(['pred_Weeks', 'FVC', 'Confidence'], axis=1)
----> 4 test_FVC = test_FVC.groupby('Patient_Week').mean()
      5 test_FVC[test_FVC.columns[~test_FVC.columns.isin(['Sex'])]] = add_norm(test_FVC[test_FVC.columns[~test_FVC.columns.isin(['Sex'])]])
      6 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in mean(self, numeric_only, engine, engine_kwargs)
   2450             )
   2451         else:
-> 2452             result = self._cython_agg_general(
   2453                 "mean",
   2454                 alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
   1996             return result
   1997 
-> 1998         new_mgr = data.grouped_reduce(array_func)
   1999         res = self._wrap_agged_manager(new_mgr)
   2000         if how in ["idxmin", "idxmax"]:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in grouped_reduce(self, func)
   1467                 #  while others do not.
   1468                 for sb in blk._split():
-> 1469                     applied = sb.apply(func)
   1470                     result_blocks = extend_blocks(applied, result_blocks)
   1471             else:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in apply(self, func, **kwargs)
    391         one
    392         """
--> 393         result = func(self.values, **kwargs)
    394 
    395         result = maybe_coerce_values(result)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in array_func(values)
   1993 
   1994             assert alt is not None
-> 1995             result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
   1996             return result
   1997 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _agg_py_fallback(self, how, values, ndim, alt)
   1944             msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
   1945             # preserve the kind of exception that raised
-> 1946             raise type(err)(msg) from err
   1947 
   1948         if ser.dtype == object:

TypeError: agg function failed [how->mean,dtype->object]

## === cell 8
test_conf = pd.merge(sub_df, df, how='left', on=['Patient'])
test_conf = test_conf.rename(columns={'Patient_Week_x':'Patient_Week'})
test_conf = test_conf.drop(['pred_Weeks', 'Percent', 'Confidence'], axis=1)
test_conf = test_conf.groupby('Patient_Week').mean()
test_conf[test_conf.columns[~test_conf.columns.isin(['Sex','FVC'])]] = add_norm(test_conf[test_conf.columns[~test_conf.columns.isin(['Sex','FVC'])]])

test_conf.head()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _agg_py_fallback(self, how, values, ndim, alt)
   1941         try:
-> 1942             res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
   1943         except Exception as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in agg_series(self, obj, func, preserve_dtype)
    863 
--> 864         result = self._aggregate_series_pure_python(obj, func)
    865 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _aggregate_series_pure_python(self, obj, func)
    884         for i, group in enumerate(splitter):
--> 885             res = func(group)
    886             res = extract_result(res)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in <lambda>(x)
   2453                 "mean",
-> 2454                 alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
   2455                 numeric_only=numeric_only,

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in mean(self, axis, skipna, numeric_only, **kwargs)
   6548     ):
-> 6549         return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
   6550 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in mean(self, axis, skipna, numeric_only, **kwargs)
  12419     ) -> Series | float:
> 12420         return self._stat_function(
  12421             "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
  12376 
> 12377         return self._reduce(
  12378             func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
   6456                 )
-> 6457             return op(delegate, skipna=skipna, **kwds)
   6458 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in f(values, axis, skipna, **kwds)
    146             else:
--> 147                 result = alt(values, axis=axis, skipna=skipna, **kwds)
    148 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in new_func(values, axis, skipna, mask, **kwargs)
    403 
--> 404         result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
    405 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in nanmean(values, axis, skipna, mask)
    719     the_sum = values.sum(axis, dtype=dtype_sum)
--> 720     the_sum = _ensure_numeric(the_sum)
    721 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in _ensure_numeric(x)
   1700             # GH#44008, GH#36703 avoid casting e.g. strings to numeric
-> 1701             raise TypeError(f"Could not convert string '{x}' to numeric")
   1702         try:

TypeError: Could not convert string 'ID00014637202177757139317' to numeric

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3955835148.py in <cell line: 0>()
      2 test_conf = test_conf.rename(columns={'Patient_Week_x':'Patient_Week'})
      3 test_conf = test_conf.drop(['pred_Weeks', 'Percent', 'Confidence'], axis=1)
----> 4 test_conf = test_conf.groupby('Patient_Week').mean()
      5 test_conf[test_conf.columns[~test_conf.columns.isin(['Sex','FVC'])]] = add_norm(test_conf[test_conf.columns[~test_conf.columns.isin(['Sex','FVC'])]])
      6 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in mean(self, numeric_only, engine, engine_kwargs)
   2450             )
   2451         else:
-> 2452             result = self._cython_agg_general(
   2453                 "mean",
   2454                 alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
   1996             return result
   1997 
-> 1998         new_mgr = data.grouped_reduce(array_func)
   1999         res = self._wrap_agged_manager(new_mgr)
   2000         if how in ["idxmin", "idxmax"]:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in grouped_reduce(self, func)
   1467                 #  while others do not.
   1468                 for sb in blk._split():
-> 1469                     applied = sb.apply(func)
   1470                     result_blocks = extend_blocks(applied, result_blocks)
   1471             else:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in apply(self, func, **kwargs)
    391         one
    392         """
--> 393         result = func(self.values, **kwargs)
    394 
    395         result = maybe_coerce_values(result)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in array_func(values)
   1993 
   1994             assert alt is not None
-> 1995             result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
   1996             return result
   1997 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _agg_py_fallback(self, how, values, ndim, alt)
   1944             msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
   1945             # preserve the kind of exception that raised
-> 1946             raise type(err)(msg) from err
   1947 
   1948         if ser.dtype == object:

TypeError: agg function failed [how->mean,dtype->object]

## === cell 9
print(test_FVC.shape)
print(test_conf.shape)


## === cell 10
X = df.iloc[:, df.columns != "FVC"]
y = df["FVC"]

X_train = X[:-5]
y_train = y[:-5]
X_val = X[-5:]
y_val = y[-5:]

print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


## === cell 11
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
train_df.boxplot('FVC', by='SmokingStatus', ax = axs[0, 0])
train_df.boxplot('Percent', by='SmokingStatus', ax = axs[0, 1])
train_df.boxplot('FVC', by='Sex', ax = axs[1, 0])
train_df.boxplot('Percent', by='Sex', ax = axs[1, 1])

plt.show()


## === cell 12
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"
ds = pydicom.dcmread(img)
plt.figure(figsize = (7,7))
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)


## === cell 13
img_1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"
img_2 = "../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/10.dcm"

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ds = pydicom.dcmread(img_1)
ax[0].set_title('Patient 1: Ex-Smoker')
ax[0].imshow(ds.pixel_array, cmap=plt.cm.bone)

ds = pydicom.dcmread(img_2)
ax[1].set_title('Patient 2: Never smoked')
ax[1].imshow(ds.pixel_array, cmap=plt.cm.bone)

plt.show


## === cell 14
def competition_metric(trueFVC, predFVC, predSTD=100):
    clipSTD = np.clip(predSTD, 70 , 9e9)  
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 1000)  
    error = np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))
    return error


## === cell 15
class model_selection(): 
    
    def __init__(self): 
        self.y_pred_FVC = pd.DataFrame()
        self.my_scorer = make_scorer(competition_metric, greater_is_better=False)
        self.best_param = None
        self.scoring = 0
    
    
    def xgboost(self, X, y, X_val, y_val,test): 
        parameters = {'learning_rate': [0.002], 'n_estimators':[4000], 'max_depth':[4], 'reg_alpha':[0.005]}
        clf = GridSearchCV(XGBRegressor(min_child_weight=0, gamma=0, 
                                                colsample_bytree=0.7, objective='reg:linear', nthread=-1,
                                                scale_pos_weight=1, subsample=.7, seed=27), 
                           param_grid=parameters, 
                           scoring=self.my_scorer)
        clf.fit(X, y)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_xgb_FVC = clf.predict(test)
        
        self.y_pred_FVC = pd.concat([pd.Series(test.index), pd.Series(y_pred_xgb_FVC)], axis=1)
        return self.y_pred_FVC, self.best_param, self.scoring
        
    
    def lightgbm(self, X, y, X_val, y_val,test): 
        parameters = {'learning_rate': [0.00005, 0.0001, 0.0005], 'n_estimators':[2500, 3000, 4000], 'num_leaves':[1, 2, 3]}
        clf = GridSearchCV(LGBMRegressor(objective='regression', 
                                               max_bin=200, 
                                               bagging_fraction=0.75,
                                               bagging_freq=5, 
                                               bagging_seed=7,
                                               feature_fraction=0.2,
                                               feature_fraction_seed=7,
                                               verbose=-1,), 
                           param_grid=parameters, 
                           scoring=self.my_scorer)
        clf.fit(X, y)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_lgb_FVC = clf.predict(test)
        
        self.y_pred_FVC = pd.concat([pd.Series(test.index), pd.Series(y_pred_lgb_FVC)], axis=1)
        return self.y_pred_FVC, self.best_param, self.scoring
    
    
    def HuberRegressor(self, X, y, test): 
        my_scorer = make_scorer(competition_metric, greater_is_better=False)
        hbr = HuberRegressor(max_iter=200)
        hbr.fit(X, y)
        y_pred_hbr_FVC = hbr.predict(test)
        
        self.y_pred_FVC = pd.concat([pd.Series(test.index), pd.Series(y_pred_hbr_FVC)], axis=1)
        return self.y_pred_FVC


## === cell 16
model = model_selection()
output = model.xgboost(X_train, y_train, X_val, y_val,test_FVC)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/1704358735.py in <cell line: 0>()
      1 model = model_selection()
----> 2 output = model.xgboost(X_train, y_train, X_val, y_val,test_FVC)

/tmp/ipykernel_10/472528427.py in xgboost(self, X, y, X_val, y_val, test)
     18         self.best_param = clf.best_params_
     19         self.scoring = clf.score(X_val, y_val)
---> 20         y_pred_xgb_FVC = clf.predict(test)
     21 
     22         self.y_pred_FVC = pd.concat([pd.Series(test.index), pd.Series(y_pred_xgb_FVC)], axis=1)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in predict(self, X)
    497         """
    498         check_is_fitted(self)
--> 499         return self.best_estimator_.predict(X)
    500 
    501     @available_if(_estimator_has("predict_proba"))

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inplace_predict(self, data, iteration_range, predict_type, missing, validate_features, base_margin, strict_shape)
   2414             data = pd.DataFrame(data)
   2415         if _is_pandas_df(data):
-> 2416             data, fns, _ = _transform_pandas_df(data, enable_categorical)
   2417             if validate_features:
   2418                 self._validate_features(fns)

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _transform_pandas_df(data, enable_categorical, feature_names, feature_types, meta, meta_type)
    488             or is_pa_ext_dtype(dtype)
    489         ):
--> 490             _invalid_dataframe_dtype(data)
    491         if is_pa_ext_dtype(dtype):
    492             pyarrow_extension = True

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _invalid_dataframe_dtype(data)
    306     type_err = "DataFrame.dtypes for data must be int, float, bool or category."
    307     msg = f"""{type_err} {_ENABLE_CAT_ERR} {err}"""
--> 308     raise ValueError(msg)
    309 
    310 

ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:Patient_Week: object, Patient: object

## === cell 17
print(output[1]) # best params
print(output[2]) # error on validation


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/734402156.py in <cell line: 0>()
----> 1 print(output[1]) # best params
      2 print(output[2]) # error on validation

NameError: name 'output' is not defined

## === cell 18
y_pred_FVC = output[0]
y_pred_FVC.rename(columns={0:'FVC'}, inplace=True)
y_pred_FVC


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/724768803.py in <cell line: 0>()
----> 1 y_pred_FVC = output[0]
      2 y_pred_FVC.rename(columns={0:'FVC'}, inplace=True)
      3 y_pred_FVC

NameError: name 'output' is not defined

## === cell 23
submission = pd.merge(sub_df,y_pred_FVC, how='left', on=['Patient_Week'])

submission = pd.merge(submission, test_FVC, how='left', on=['Patient_Week']) # Raw Confidence (Percent)
submission = submission[['Patient_Week', 'FVC', 'Confidence']].rename(columns={'Percent':'Confidence'}) # for raw

submission.set_index('Patient_Week', inplace=True)
submission.head()


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4214714445.py in <cell line: 0>()
----> 1 submission = pd.merge(sub_df,y_pred_FVC, how='left', on=['Patient_Week'])
      2 # submission = pd.merge(submission,y_pred_conf, how='left', on=['Patient_Week']) # Predicted Confidence
      3 # submission = submission[['Patient_Week', 'FVC', 'Confidence']].rename(columns={'Percent':'Confidence'})
      4 
      5 # raw Confidence

NameError: name 'y_pred_FVC' is not defined

## === cell 24
submission.to_csv('./submission.csv')


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/365037821.py in <cell line: 0>()
----> 1 submission.to_csv('./submission.csv')

NameError: name 'submission' is not defined

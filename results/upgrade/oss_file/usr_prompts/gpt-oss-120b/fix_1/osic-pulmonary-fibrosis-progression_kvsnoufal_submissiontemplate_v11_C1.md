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
pydicom==3.0.1
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

-7.0605

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
import pydicom
import matplotlib.pyplot as plt
%matplotlib inline


## === cell 1
import random


## === cell 2
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
train_df.head()


## === cell 3
print('Shape of Training data: ', train_df.shape)
print('Shape of Test data: ', test_df.shape)

print(f"The total patient ids are {train_df['Patient'].count()}")
print(f"Number of unique ids are {train_df['Patient'].value_counts().shape[0]} ")


## === cell 4
sub["Patient"]=sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"]=sub["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
sub=sub.drop(["FVC","Patient_Week","Confidence"],axis=1)
submission=sub[["Patient","Weeks"]].merge(test_df,on=["Patient","Weeks"],how="left")
for col in ["Age","Sex","SmokingStatus"]:
    submission[col]=submission.groupby("Patient")[col].apply(lambda x : x.ffill())
    submission[col]=submission.groupby("Patient")[col].apply(lambda x : x.bfill())
submission.head()#.isnull().sum(),submission.shape


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12686     try:
> 12687         reindexed_value = value.reindex(index)._values
  12688     except ValueError as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in reindex(self, index, axis, method, copy, level, fill_value, limit, tolerance)
   5152     ) -> Series:
-> 5153         return super().reindex(
   5154             index=index,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5609         # perform the reindex on the axes
-> 5610         return self._reindex_axes(
   5611             axes, level, limit, tolerance, method, fill_value, copy

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
   5632             ax = self._get_axis(a)
-> 5633             new_index, indexer = ax.reindex(
   5634                 labels, level=level, limit=limit, tolerance=tolerance, method=method

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in reindex(self, target, method, level, limit, tolerance)
   4432 
-> 4433         target = self._wrap_reindex_result(target, indexer, preserve_names)
   4434         return target, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in _wrap_reindex_result(self, target, indexer, preserve_names)
   2716                 try:
-> 2717                     target = MultiIndex.from_tuples(target)
   2718                 except TypeError:

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in new_meth(self_or_cls, *args, **kwargs)
    221 
--> 222         return meth(self_or_cls, *args, **kwargs)
    223 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in from_tuples(cls, tuples, sortorder, names)
    616 
--> 617             arrays = list(lib.tuples_to_object_array(tuples).T)
    618         elif isinstance(tuples, list):

lib.pyx in pandas._libs.lib.tuples_to_object_array()

ValueError: Buffer dtype mismatch, expected 'Python object' but got 'long'

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/609821951.py in <cell line: 0>()
      4 submission=sub[["Patient","Weeks"]].merge(test_df,on=["Patient","Weeks"],how="left")
      5 for col in ["Age","Sex","SmokingStatus"]:
----> 6     submission[col]=submission.groupby("Patient")[col].apply(lambda x : x.ffill())
      7     submission[col]=submission.groupby("Patient")[col].apply(lambda x : x.bfill())
      8 submission.head()#.isnull().sum(),submission.shape

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5261             if not isinstance(value, Series):
   5262                 value = Series(value)
-> 5263             return _reindex_for_setitem(value, self.index)
   5264 
   5265         if is_list_like(value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12692             raise err
  12693 
> 12694         raise TypeError(
  12695             "incompatible index of inserted column with frame index"
  12696         ) from err

TypeError: incompatible index of inserted column with frame index

## === cell 5

print(train_df.shape)
train_df=train_df.drop_duplicates(keep=False, subset=['Patient','Weeks'])
print(train_df.shape)
train_df=train_df[train_df["Patient"].isin(list(submission["Patient"].unique()))==False]
print(train_df.shape)


## === cell 6
train_df.reset_index(drop=True,inplace=True)
train_df=train_df.sort_values("Weeks")
train_df["c_first_week"]=train_df.groupby("Patient")["Weeks"].transform("min")
train_df.loc[train_df["c_first_week"]==train_df["Weeks"],"c_first_FVC"]=train_df["FVC"]
train_df["c_first_FVC"]=train_df.groupby("Patient")["c_first_FVC"].apply(lambda x: x.ffill())
train_df.loc[train_df["c_first_week"]==train_df["Weeks"],"c_first_PCT"]=train_df["Percent"]
train_df["c_first_PCT"]=train_df.groupby("Patient")["c_first_PCT"].apply(lambda x: x.ffill())
train_df["c_week_since_week"]=train_df["Weeks"]-train_df["c_first_week"]
train_df.head()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12686     try:
> 12687         reindexed_value = value.reindex(index)._values
  12688     except ValueError as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in reindex(self, index, axis, method, copy, level, fill_value, limit, tolerance)
   5152     ) -> Series:
-> 5153         return super().reindex(
   5154             index=index,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5609         # perform the reindex on the axes
-> 5610         return self._reindex_axes(
   5611             axes, level, limit, tolerance, method, fill_value, copy

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
   5632             ax = self._get_axis(a)
-> 5633             new_index, indexer = ax.reindex(
   5634                 labels, level=level, limit=limit, tolerance=tolerance, method=method

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in reindex(self, target, method, level, limit, tolerance)
   4432 
-> 4433         target = self._wrap_reindex_result(target, indexer, preserve_names)
   4434         return target, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in _wrap_reindex_result(self, target, indexer, preserve_names)
   2716                 try:
-> 2717                     target = MultiIndex.from_tuples(target)
   2718                 except TypeError:

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in new_meth(self_or_cls, *args, **kwargs)
    221 
--> 222         return meth(self_or_cls, *args, **kwargs)
    223 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in from_tuples(cls, tuples, sortorder, names)
    616 
--> 617             arrays = list(lib.tuples_to_object_array(tuples).T)
    618         elif isinstance(tuples, list):

lib.pyx in pandas._libs.lib.tuples_to_object_array()

ValueError: Buffer dtype mismatch, expected 'Python object' but got 'long'

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3790472695.py in <cell line: 0>()
      4 train_df["c_first_week"]=train_df.groupby("Patient")["Weeks"].transform("min")
      5 train_df.loc[train_df["c_first_week"]==train_df["Weeks"],"c_first_FVC"]=train_df["FVC"]
----> 6 train_df["c_first_FVC"]=train_df.groupby("Patient")["c_first_FVC"].apply(lambda x: x.ffill())
      7 train_df.loc[train_df["c_first_week"]==train_df["Weeks"],"c_first_PCT"]=train_df["Percent"]
      8 train_df["c_first_PCT"]=train_df.groupby("Patient")["c_first_PCT"].apply(lambda x: x.ffill())

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5261             if not isinstance(value, Series):
   5262                 value = Series(value)
-> 5263             return _reindex_for_setitem(value, self.index)
   5264 
   5265         if is_list_like(value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12692             raise err
  12693 
> 12694         raise TypeError(
  12695             "incompatible index of inserted column with frame index"
  12696         ) from err

TypeError: incompatible index of inserted column with frame index

## === cell 8
submission.reset_index(drop=True,inplace=True)
submission.loc[submission["FVC"].notnull(),"c_first_week"]=submission["Weeks"]
submission.loc[submission["FVC"].notnull(),"c_first_FVC"]=submission["FVC"]
submission.loc[submission["Percent"].notnull(),"c_first_PCT"]=submission["Percent"]

submission["c_first_FVC"]=submission.groupby("Patient")["c_first_FVC"].apply(lambda x: x.ffill())
submission["c_first_FVC"]=submission.groupby("Patient")["c_first_FVC"].apply(lambda x: x.bfill())

submission["c_first_PCT"]=submission.groupby("Patient")["c_first_PCT"].apply(lambda x: x.ffill())
submission["c_first_PCT"]=submission.groupby("Patient")["c_first_PCT"].apply(lambda x: x.bfill())

submission["c_first_week"]=submission.groupby("Patient")["c_first_week"].apply(lambda x: x.ffill())
submission["c_first_week"]=submission.groupby("Patient")["c_first_week"].apply(lambda x: x.bfill())

submission["c_week_since_week"]=submission["Weeks"]-submission["c_first_week"]
submission


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12686     try:
> 12687         reindexed_value = value.reindex(index)._values
  12688     except ValueError as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in reindex(self, index, axis, method, copy, level, fill_value, limit, tolerance)
   5152     ) -> Series:
-> 5153         return super().reindex(
   5154             index=index,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5609         # perform the reindex on the axes
-> 5610         return self._reindex_axes(
   5611             axes, level, limit, tolerance, method, fill_value, copy

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
   5632             ax = self._get_axis(a)
-> 5633             new_index, indexer = ax.reindex(
   5634                 labels, level=level, limit=limit, tolerance=tolerance, method=method

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in reindex(self, target, method, level, limit, tolerance)
   4432 
-> 4433         target = self._wrap_reindex_result(target, indexer, preserve_names)
   4434         return target, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in _wrap_reindex_result(self, target, indexer, preserve_names)
   2716                 try:
-> 2717                     target = MultiIndex.from_tuples(target)
   2718                 except TypeError:

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in new_meth(self_or_cls, *args, **kwargs)
    221 
--> 222         return meth(self_or_cls, *args, **kwargs)
    223 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in from_tuples(cls, tuples, sortorder, names)
    616 
--> 617             arrays = list(lib.tuples_to_object_array(tuples).T)
    618         elif isinstance(tuples, list):

lib.pyx in pandas._libs.lib.tuples_to_object_array()

ValueError: Buffer dtype mismatch, expected 'Python object' but got 'long'

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3254375470.py in <cell line: 0>()
      5 submission.loc[submission["Percent"].notnull(),"c_first_PCT"]=submission["Percent"]
      6 
----> 7 submission["c_first_FVC"]=submission.groupby("Patient")["c_first_FVC"].apply(lambda x: x.ffill())
      8 submission["c_first_FVC"]=submission.groupby("Patient")["c_first_FVC"].apply(lambda x: x.bfill())
      9 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5261             if not isinstance(value, Series):
   5262                 value = Series(value)
-> 5263             return _reindex_for_setitem(value, self.index)
   5264 
   5265         if is_list_like(value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12692             raise err
  12693 
> 12694         raise TypeError(
  12695             "incompatible index of inserted column with frame index"
  12696         ) from err

TypeError: incompatible index of inserted column with frame index

## === cell 9
catcols=["SmokingStatus","Sex"]
uval_dicts={}
for col in catcols:
    uvals=train_df[col].unique()
    for val in uvals:
        train_df.loc[train_df[col]==val,"ohe_"+val]=1
        train_df.loc[train_df[col]!=val,"ohe_"+val]=0
        
        submission.loc[submission[col]==val,"ohe_"+val]=1
        submission.loc[submission[col]!=val,"ohe_"+val]=0
    uval_dicts[col]=uvals

    
    
from sklearn import preprocessing

numcols=['Weeks','Age','c_first_week', 'c_first_FVC', 'c_week_since_week', 'c_first_PCT']
for col in numcols:
    le=preprocessing.StandardScaler()
    le.fit(np.array(train_df[col].tolist()+submission[col].tolist()).reshape(-1,1))
    train_df["n_"+col]=le.transform(train_df[col].values.reshape(-1,1)).flatten()
    submission["n_"+col]=le.transform(submission[col].values.reshape(-1,1)).flatten()
    
train_df.head()


submission.head()


## --- ERROR in cell 9, traceback:
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

KeyError: 'c_week_since_week'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2817418373.py in <cell line: 0>()
     22 for col in numcols:
     23     le=preprocessing.StandardScaler()
---> 24     le.fit(np.array(train_df[col].tolist()+submission[col].tolist()).reshape(-1,1))
     25     train_df["n_"+col]=le.transform(train_df[col].values.reshape(-1,1)).flatten()
     26     submission["n_"+col]=le.transform(submission[col].values.reshape(-1,1)).flatten()

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

KeyError: 'c_week_since_week'

## === cell 10
submission.columns


## === cell 11
train_df.columns


## === cell 12
numerical_features=[ 'n_Weeks', 'n_Age', 'n_c_first_week', 'n_c_first_FVC',
       'n_c_week_since_week', 'n_c_first_PCT',]
binary_features=['ohe_Never smoked', 'ohe_Ex-smoker', 'ohe_Currently smokes',
       'ohe_Female', 'ohe_Male']
target="FVC"


## === cell 13
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)
def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss

    
    


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
def make_model(num_inputs,num_blocks,units,dropout):
    input_ = L.Input((num_inputs,), name="Patient")
    for _ in range(num_blocks):
        if _==0:
            x=L.Dense(units,activation="relu")(input_)
        else:
            x=L.Dense(units,activation="relu")(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
    
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = M.Model(input_, preds, name="CNN")
    model.compile(loss=mloss(1), optimizer="adam", metrics=[score])
    return model


## === cell 16
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(2020)


## === cell 17
X=train_df[numerical_features+binary_features].values
y=train_df[target].astype(np.float32).values

X_test=submission[numerical_features+binary_features].values
X.shape,y.shape,X_test.shape


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/30409561.py in <cell line: 0>()
----> 1 X=train_df[numerical_features+binary_features].values
      2 y=train_df[target].astype(np.float32).values
      3 
      4 X_test=submission[numerical_features+binary_features].values
      5 X.shape,y.shape,X_test.shape

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

KeyError: "['n_c_week_since_week', 'n_c_first_PCT'] not in index"

## === cell 18
model_checkpoint=tf.keras.callbacks.ModelCheckpoint("./bestmodel.chkpt",save_best_only=True,save_weights_only=True,\
                                                   mode="min",monitor="val_score")
lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=30, verbose=0, mode='min',
    min_delta=0.0001, cooldown=0, min_lr=0,)
es=tf.keras.callbacks.EarlyStopping(
    monitor='val_score', min_delta=0, patience=20, verbose=0, mode='min', restore_best_weights=True
)
CALLBACKS=[lr]


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1264203696.py in <cell line: 0>()
----> 1 model_checkpoint=tf.keras.callbacks.ModelCheckpoint("./bestmodel.chkpt",save_best_only=True,save_weights_only=True,\
      2                                                    mode="min",monitor="val_score")
      3 lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=30, verbose=0, mode='min',
      4     min_delta=0.0001, cooldown=0, min_lr=0,)
      5 es=tf.keras.callbacks.EarlyStopping(

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=./bestmodel.chkpt

## === cell 19
from sklearn import model_selection
from tqdm import tqdm
import numpy as np

NFOLDS=5
BATCH_SIZE=100
EPOCHS=300
kf=model_selection.KFold(n_splits=NFOLDS, random_state=2020)
dfs=[]
validation=np.zeros((X.shape[0],3))
test_predictions=[]
val_scores=[]


for i,(train_index,val_index) in enumerate(kf.split(X)):
    print("Fold")
    print(i,len(train_index),len(val_index))
    X_train=X[train_index]
    y_train=y[train_index]
    X_val=X[val_index]
    y_val=y[val_index]
    
    model=make_model(X.shape[1],num_blocks=2,units=300,dropout=0.2)
    history=model.fit(X_train,y_train,\
              validation_data=(X_val,y_val),\
              batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=0,\
                     callbacks=CALLBACKS)
    
    y_pred=model.predict(X_val)
    val_score=score(y_val.reshape(-1,1),y_pred).numpy()
    print(f"Fold {i} Val score {val_score}")
    val_scores.append(val_score)
    
    
    test_predictions.append(model.predict(X_test))
    
    histdf=pd.DataFrame(history.history)
    histdf["epoch"]=history.epoch
    dfs.append(histdf)
    
    validation[val_index,:]=y_pred
    
    
    

print(f"mean OOF validation score: {np.mean(val_scores)} ")
print(f"min OOF validation score: {np.min(val_scores)} ")
print(f"max OOF validation score: {np.max(val_scores)} ")


findf=pd.DataFrame()

for col in dfs[0].columns:
    vals=np.zeros((EPOCHS,))
    for df in dfs:
        vals+=df[col].values
    findf[col]=vals/len(dfs)
findf["val_score"].plot()
plt.show()
findf["val_loss"].plot()
plt.show()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1465511320.py in <cell line: 0>()
      6 BATCH_SIZE=100
      7 EPOCHS=300
----> 8 kf=model_selection.KFold(n_splits=NFOLDS, random_state=2020)
      9 dfs=[]
     10 validation=np.zeros((X.shape[0],3))

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in __init__(self, n_splits, shuffle, random_state)
    449 
    450     def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
--> 451         super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    452 
    453     def _iter_test_indices(self, X, y=None, groups=None):

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in __init__(self, n_splits, shuffle, random_state)
    306 
    307         if not shuffle and random_state is not None:  # None is the default
--> 308             raise ValueError(
    309                 "Setting a random_state has no effect since shuffle is "
    310                 "False. You should leave "

ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.

## === cell 20
print(findf["val_score"].tail())
findf["val_score"].plot()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1230930661.py in <cell line: 0>()
----> 1 print(findf["val_score"].tail())
      2 findf["val_score"].plot()

NameError: name 'findf' is not defined

## === cell 21
preds=np.zeros((X_test.shape[0],3))
for p in test_predictions:
    preds += p / NFOLDS
print(preds.shape)
preds[:3]


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4168119152.py in <cell line: 0>()
----> 1 preds=np.zeros((X_test.shape[0],3))
      2 for p in test_predictions:
      3     preds += p / NFOLDS
      4 print(preds.shape)
      5 preds[:3]

NameError: name 'X_test' is not defined

## === cell 22
sub=submission[["Patient","Weeks"]]
sub["FVC_median"]=preds[:,1]
sub["Confidence"]=preds[:,2]-preds[:,0]


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2099480670.py in <cell line: 0>()
      1 sub=submission[["Patient","Weeks"]]
----> 2 sub["FVC_median"]=preds[:,1]
      3 sub["Confidence"]=preds[:,2]-preds[:,0]

NameError: name 'preds' is not defined

## === cell 23
sub.isnull().sum()


## === cell 24
sub.describe().T


## === cell 25
sub["Patient_Week"]=sub.apply(lambda x: x["Patient"]+"_"+str(x["Weeks"]),axis=1)
sub


## === cell 26
sub.rename(columns={"FVC_median":"FVC"})[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv",index=False)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3115987955.py in <cell line: 0>()
----> 1 sub.rename(columns={"FVC_median":"FVC"})[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv",index=False)

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

KeyError: "['FVC', 'Confidence'] not in index"

## === cell 31
def calculate_score(y_true, y_pred):
    
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    print(sigma)
    sigma_clip = sigma#np.clip(sigma,a_min=None,a_max= None)
    print(pd.Series(sigma_clip).value_counts())
    delta = np.abs(y_true - fvc_pred)
    print(delta)
    print(pd.Series(delta).value_counts())
    sq2 = np.sqrt( 2)
    print(sq2)
    metric = (delta / sigma_clip)*sq2+np.log(sigma_clip* sq2) 
    print(metric)
    print(sigma_clip* sq2)
    print(np.log(sigma_clip* sq2))
    return - np.nanmean(metric)

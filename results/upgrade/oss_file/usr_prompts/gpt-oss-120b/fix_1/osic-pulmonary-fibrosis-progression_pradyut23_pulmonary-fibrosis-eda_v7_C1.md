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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
seaborn==0.12.2
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
tf_keras==2.18.0
ydata-profiling==4.17.0

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

-6.9198

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport


## === cell 1
train=pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
print('Train Data:')
print(train.head())

test=pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')
print('\n\nTest Data:')
print(test.head())

sub=pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
print('\n\nSubmission File:')
print(sub.head())


## === cell 2
ProfileReport(train,progress_bar=False)


## === cell 3
ProfileReport(test,progress_bar=False)


## === cell 4
ProfileReport(sub,progress_bar=False)


## === cell 5
print('No of unique patients:',len(train.Patient.unique()))

readings=train.groupby('Patient').Weeks.count()
print('Min no. of readings for a patient:', min(readings))
print('Max no. of readings for a patient:', max(readings))

fig=plt.figure(figsize=(15,5))
sns.barplot(readings.index,readings,color='#7AC8BE')
plt.title('Number of Readings per Patient',size=15)
plt.xlabel('Patient',size=12)
plt.ylabel('# Readings',size=12)
plt.xticks([])


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3495413447.py in <cell line: 0>()
      6 
      7 fig=plt.figure(figsize=(15,5))
----> 8 sns.barplot(readings.index,readings,color='#7AC8BE')
      9 plt.title('Number of Readings per Patient',size=15)
     10 plt.xlabel('Patient',size=12)

TypeError: barplot() takes from 0 to 1 positional arguments but 2 positional arguments (and 1 keyword-only argument) were given

## === cell 6
print('Minimum aged patient:',min(train['Age']))
print('Maximum aged patient:',max(train['Age']))

fig=plt.figure(figsize=(10,5))
sns.distplot(train['Age'])
plt.title('Age Distribution',size=15)
plt.xlabel('Age',size=12)


## === cell 7
sex=train.groupby('Patient').Sex.first()
print('Male Patients:',sex.value_counts()[0])
print('Female Patients:',sex.value_counts()[1])

fig=plt.figure(figsize=(5,5))                                              
sns.countplot(sex)
plt.title('Sex Distribution',size=15)
plt.ylabel('# Patients',size=12)
plt.xlabel('Sex',size=12)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3658183500.py in <cell line: 0>()
      5 
      6 fig=plt.figure(figsize=(5,5))
----> 7 sns.countplot(sex)
      8 plt.title('Sex Distribution',size=15)
      9 plt.ylabel('# Patients',size=12)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in countplot(data, x, y, hue, order, hue_order, orient, color, palette, saturation, width, dodge, ax, **kwargs)
   2941         raise ValueError("Cannot pass values for both `x` and `y`")
   2942 
-> 2943     plotter = _CountPlotter(
   2944         x, y, hue, data, order, hue_order,
   2945         estimator, errorbar, n_boot, units, seed,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, estimator, errorbar, n_boot, units, seed, orient, color, palette, saturation, width, errcolor, errwidth, capsize, dodge)
   1528                  errcolor, errwidth, capsize, dodge):
   1529         """Initialize the plotter."""
-> 1530         self.establish_variables(x, y, hue, data, orient,
   1531                                  order, hue_order, units)
   1532         self.establish_colors(color, palette, saturation)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_variables(self, x, y, hue, data, orient, order, hue_order, units)
    514 
    515                 # Convert to a list of arrays, the common representation
--> 516                 plot_data = [np.asarray(d, float) for d in plot_data]
    517 
    518                 # The group names will just be numeric indices

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in <listcomp>(.0)
    514 
    515                 # Convert to a list of arrays, the common representation
--> 516                 plot_data = [np.asarray(d, float) for d in plot_data]
    517 
    518                 # The group names will just be numeric indices

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __array__(self, dtype, copy)
   1029         """
   1030         values = self._values
-> 1031         arr = np.asarray(values, dtype=dtype)
   1032         if using_copy_on_write() and astype_is_view(values.dtype, arr.dtype):
   1033             arr = arr.view()

ValueError: could not convert string to float: 'Male'

## === cell 8
smoke=train.groupby('Patient').SmokingStatus.first()
print('Ex-smokers:',smoke.value_counts()[0])
print('Patients who never smoked:',smoke.value_counts()[1])
print('Patients who currently smoke:',smoke.value_counts()[2])

fig=plt.figure(figsize=(5,5))                                              
sns.countplot(smoke)
plt.title('Smoking Status',size=15)
plt.ylabel('# Patients',size=12)
plt.xlabel('Status',size=12)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2066830583.py in <cell line: 0>()
      6 
      7 fig=plt.figure(figsize=(5,5))
----> 8 sns.countplot(smoke)
      9 plt.title('Smoking Status',size=15)
     10 plt.ylabel('# Patients',size=12)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in countplot(data, x, y, hue, order, hue_order, orient, color, palette, saturation, width, dodge, ax, **kwargs)
   2941         raise ValueError("Cannot pass values for both `x` and `y`")
   2942 
-> 2943     plotter = _CountPlotter(
   2944         x, y, hue, data, order, hue_order,
   2945         estimator, errorbar, n_boot, units, seed,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, estimator, errorbar, n_boot, units, seed, orient, color, palette, saturation, width, errcolor, errwidth, capsize, dodge)
   1528                  errcolor, errwidth, capsize, dodge):
   1529         """Initialize the plotter."""
-> 1530         self.establish_variables(x, y, hue, data, orient,
   1531                                  order, hue_order, units)
   1532         self.establish_colors(color, palette, saturation)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_variables(self, x, y, hue, data, orient, order, hue_order, units)
    514 
    515                 # Convert to a list of arrays, the common representation
--> 516                 plot_data = [np.asarray(d, float) for d in plot_data]
    517 
    518                 # The group names will just be numeric indices

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in <listcomp>(.0)
    514 
    515                 # Convert to a list of arrays, the common representation
--> 516                 plot_data = [np.asarray(d, float) for d in plot_data]
    517 
    518                 # The group names will just be numeric indices

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __array__(self, dtype, copy)
   1029         """
   1030         values = self._values
-> 1031         arr = np.asarray(values, dtype=dtype)
   1032         if using_copy_on_write() and astype_is_view(values.dtype, arr.dtype):
   1033             arr = arr.view()

ValueError: could not convert string to float: 'Ex-smoker'

## === cell 9
print('Maximum FVC value:',max(train['FVC']))
print('Minimum FVC value:',min(train['FVC']))

fig=plt.figure(figsize=(10,5))
sns.distplot(train['FVC'])
plt.title('FVC Value Distribution',size=15)
plt.xlabel('FVC Value',size=12)


## === cell 10
print('Maximum Percentage:',max(train['Percent']))
print('Minimum Percentage:',min(train['Percent']))

fig=plt.figure(figsize=(10,5))
sns.distplot(train['Percent'])
plt.title('Percentage Distribution',size=15)
plt.xlabel('Percent',size=12)


## === cell 11
a=train[['Age','SmokingStatus','Percent']]
fig=plt.figure(figsize=(15,5))
for i in range(len(a.columns)):
    fig.add_subplot(1,3,i+1)
    sns.scatterplot(x=a.iloc[:,i],y=train['FVC'],hue=train['Sex'],palette=['blue','red'])
plt.tight_layout()
plt.show()


## === cell 12
import pydicom as dicom
import cv2

data_dir='../input/osic-pulmonary-fibrosis-progression/train'
patients=os.listdir(data_dir)
labels_df=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv',index_col=0)
labels_df.head()


## === cell 13

for patient in patients[:1]:
    label=labels_df.loc[patient,'FVC']
    path=data_dir+'/'+patient
    slices=[dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    print('No. of scans:',len(slices))
    print('Height and width of the scan:',slices[0].pixel_array.shape)
    print('\nMetadata of the Dicom File:')
    print(slices[1])


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3602044782.py in <cell line: 0>()
      4     label=labels_df.loc[patient,'FVC']
      5     path=data_dir+'/'+patient
----> 6     slices=[dicom.read_file(path + '/' + s) for s in os.listdir(path)]
      7     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
      8     print('No. of scans:',len(slices))

/tmp/ipykernel_11/3602044782.py in <listcomp>(.0)
      4     label=labels_df.loc[patient,'FVC']
      5     path=data_dir+'/'+patient
----> 6     slices=[dicom.read_file(path + '/' + s) for s in os.listdir(path)]
      7     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
      8     print('No. of scans:',len(slices))

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 14
c=0
for patient in patients:
    try:
        label=labels_df.loc[patient,'FVC']
        path=data_dir+'/'+patient
        slices=[dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        print(len(slices),slices[0].pixel_array.shape)
        c+=1
        if c==5:
            break
    except:
        continue


## === cell 15
min_s=9999
max_s=0
for patient in patients[:]:
    label=labels_df.loc[patient,'FVC']
    path=data_dir+'/'+patient
    slices=[len(s) for s in os.listdir(path)]
    if len(slices)<min_s:
        min_s=len(slices)
    if len(slices)>max_s:
        max_s=len(slices)
print('Minimum number of scans for any patient:',min_s)
print('Maximum number of scans for any patient:',max_s)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine._get_loc_duplicates()

index.pyx in pandas._libs.index.IndexEngine._maybe_get_bool_indexer()

index.pyx in pandas._libs.index._unpack_bool_indexer()

KeyError: 'train'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4290756804.py in <cell line: 0>()
      2 max_s=0
      3 for patient in patients[:]:
----> 4     label=labels_df.loc[patient,'FVC']
      5     path=data_dir+'/'+patient
      6     slices=[len(s) for s in os.listdir(path)]

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
   4299                     new_index = index[loc]
   4300         else:
-> 4301             loc = index.get_loc(key)
   4302 
   4303             if isinstance(loc, np.ndarray):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'train'

## === cell 16
import cv2

for patient in patients[1:2]:
    label=labels_df.loc[patient,'FVC']
    path=data_dir+'/'+patient
    slices=[dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    
    fig=plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.title('CT Scan',size=15)
    plt.imshow(slices[0].pixel_array,cmap='gray')
    plt.show()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3415744157.py in <cell line: 0>()
      5     label=labels_df.loc[patient,'FVC']
      6     path=data_dir+'/'+patient
----> 7     slices=[dicom.read_file(path + '/' + s) for s in os.listdir(path)]
      8     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
      9 

/tmp/ipykernel_11/3415744157.py in <listcomp>(.0)
      5     label=labels_df.loc[patient,'FVC']
      6     path=data_dir+'/'+patient
----> 7     slices=[dicom.read_file(path + '/' + s) for s in os.listdir(path)]
      8     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
      9 

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 17
import cv2

for patient in patients:
    label=labels_df.loc[patient,'FVC']
    path=data_dir+'/'+patient
    slices=[dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    
    img_px_size=150
    
    try:
        if len(slices)<=56:
            fig=plt.figure(figsize=(20,40))
            for num,each_slice in enumerate(slices):
                fig.add_subplot(14,4,num+1)
                new_image=cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size))
                plt.axis('off')
                plt.title(num+1,size=10)
                plt.imshow(new_image,cmap='gray')
            plt.show()
            break
    except:
        continue


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1158272303.py in <cell line: 0>()
      5     label=labels_df.loc[patient,'FVC']
      6     path=data_dir+'/'+patient
----> 7     slices=[dicom.read_file(path + '/' + s) for s in os.listdir(path)]
      8     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
      9 

/tmp/ipykernel_11/1158272303.py in <listcomp>(.0)
      5     label=labels_df.loc[patient,'FVC']
      6     path=data_dir+'/'+patient
----> 7     slices=[dicom.read_file(path + '/' + s) for s in os.listdir(path)]
      8     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
      9 

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 18
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


## === cell 19
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    image[image == -2000] = 0
    
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


## === cell 20
first_patient = load_scan(data_dir + '/' + patients[0])
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
plt.show()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2353757803.py in <cell line: 0>()
----> 1 first_patient = load_scan(data_dir + '/' + patients[0])
      2 first_patient_pixels = get_pixels_hu(first_patient)
      3 plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
      4 plt.xlabel("Hounsfield Units (HU)")
      5 plt.ylabel("Frequency")

/tmp/ipykernel_11/3043122161.py in load_scan(path)
      1 # Load the scans in given folder path
      2 def load_scan(path):
----> 3     slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
      4     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
      5     try:

/tmp/ipykernel_11/3043122161.py in <listcomp>(.0)
      1 # Load the scans in given folder path
      2 def load_scan(path):
----> 3     slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
      4     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
      5     try:

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 21
drop=train[train.duplicated(subset=['Patient','Weeks'],keep='last')]
print('No. of rows to be dropped:',drop.shape[0])
train.drop_duplicates(subset=['Patient','Weeks'],keep='last',inplace=True)


## === cell 22
sub[['Patient','Weeks']]=sub.Patient_Week.str.split("_",expand = True)
sub=sub[['Patient','Weeks','Confidence','Patient_Week']]
sub.head()


## === cell 23
sub=sub.merge(test.drop('Weeks',axis = 1),on="Patient")
sub.head()


## === cell 24
train['Dataset']='train'
sub['Dataset']='test'

data=train.append([sub])
data.reset_index(inplace = True,drop=True)
data.head()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2122276641.py in <cell line: 0>()
      4 sub['Dataset']='test'
      5 
----> 6 data=train.append([sub])
      7 data.reset_index(inplace = True,drop=True)
      8 data.head()

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 25
data = pd.concat([
    data,
    pd.get_dummies(data.Sex),
    pd.get_dummies(data.SmokingStatus)
],axis=1)

data.drop(['Sex','SmokingStatus'],axis=1,inplace=True)
data['Weeks']=data['Weeks'].astype('int64')
data.head()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2325218332.py in <cell line: 0>()
      2 #Conversion
      3 data = pd.concat([
----> 4     data,
      5     pd.get_dummies(data.Sex),
      6     pd.get_dummies(data.SmokingStatus)

NameError: name 'data' is not defined

## === cell 26
def get_baseline(df):  
    _df=df.copy()
    _df['min_week']=_df['Weeks']
    _df.loc[_df.Dataset=='test','min_week']=0
    _df["min_week"]=_df.groupby('Patient')['Weeks'].transform('min')
    _df['baselined_week']=_df['Weeks']-_df['min_week']
    
    return _df   


data['Weeks']=data['Weeks'].astype('int64')
data=get_baseline(data)
data.head()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3237100722.py in <cell line: 0>()
     11 
     12 
---> 13 data['Weeks']=data['Weeks'].astype('int64')
     14 data=get_baseline(data)
     15 data.head()

NameError: name 'data' is not defined

## === cell 27
def get_baseline_FVC(df):
    _df = df.copy()
    base = _df.loc[_df.Weeks == _df.min_week]
    base = base[['Patient','FVC']].copy()
    base.columns = ['Patient','base_FVC']
    
    base['nb'] = 1
    base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
    
    base = base[base.nb == 1]
    base.drop('nb', axis = 1, inplace = True)
    
    _df = _df.merge(base, on = 'Patient', how = 'left')    
    _df.drop(['min_week'], axis = 1)
    
    return _df

data=get_baseline_FVC(data)
data.head()


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3178336373.py in <cell line: 0>()
     20     return _df
     21 
---> 22 data=get_baseline_FVC(data)
     23 data.head()

NameError: name 'data' is not defined

## === cell 28
def scaling(series):
    return (series-series.min())/(series.max()-series.min())

data['Age']=scaling(data['Age'])
data['Percent']=scaling(data['Percent'])
data['baselined_week']=scaling(data['baselined_week'])
data['base_FVC']=scaling(data['base_FVC'])
data.head()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1799658916.py in <cell line: 0>()
      3     return (series-series.min())/(series.max()-series.min())
      4 
----> 5 data['Age']=scaling(data['Age'])
      6 data['Percent']=scaling(data['Percent'])
      7 data['baselined_week']=scaling(data['baselined_week'])

NameError: name 'data' is not defined

## === cell 29
import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Lambda, Input
from tensorflow.keras.models import Sequential, Model

C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

def score(y_true, y_pred):
    """Calculate the competition metric"""
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype = tf.float32) )
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)

def qloss(y_true, y_pred):
    """Calculate Pinball loss"""
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype = tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q-1) * e)
    return K.mean(v)

def mloss(_lambda):
    """Combine Score and qloss"""
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
    return loss


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 30
def make_model(nh):
    z = Input((nh,), name="Patient")
    x = Dense(100, activation="elu", name="d1")(z)
    x = Dense(100, activation="elu", name="d3")(x)
    p1 = Dense(3, activation="linear", name="p1")(x)
    p2 = Dense(3, activation="elu", name="p2")(x)
    preds = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = Model(z,preds,name="CNN")
    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
    return model


## === cell 31

features_list=['baselined_week', 'Percent', 'Age', 'base_FVC', 'Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes']
train=data.loc[data.Dataset == 'train']
sub=data.loc[data.Dataset == 'test']

y=train['FVC'].values.astype(float)

X_train=train[features_list].values
X_test=sub[features_list].values
n_rows=X_train.shape[1]

train_preds=np.zeros((X_train.shape[0], 3))
test_preds=np.zeros((X_test.shape[0], 3))


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1989904784.py in <cell line: 0>()
      3 # get back original data split
      4 features_list=['baselined_week', 'Percent', 'Age', 'base_FVC', 'Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes']
----> 5 train=data.loc[data.Dataset == 'train']
      6 sub=data.loc[data.Dataset == 'test']
      7 

NameError: name 'data' is not defined

## === cell 32
model=make_model(n_rows)
print(model.summary())


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3749915003.py in <cell line: 0>()
----> 1 model=make_model(n_rows)
      2 print(model.summary())

NameError: name 'n_rows' is not defined

## === cell 33
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from keras import backend as K

reduce_lr_loss=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.4,patience=150,verbose=0,epsilon=1e-4,mode='min')

NFOLD = 6
kf = KFold(n_splits=NFOLD)
OOF_val_score=[]

cnt = 0
BATCH_SIZE=128
EPOCHS = 800
for tr_idx, val_idx in kf.split(X_train):
    cnt += 1
    print(f"FOLD {cnt}")
    model=make_model(n_rows)
    history=model.fit(X_train[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, 
            validation_data=(X_train[val_idx], y[val_idx]), verbose=0, callbacks=[reduce_lr_loss])
    print("train", model.evaluate(X_train[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", model.evaluate(X_train[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))
    print("predict val...")
    train_preds[val_idx]=model.predict(X_train[val_idx],batch_size=BATCH_SIZE, verbose=0)
    
    OOF_val_score.append(model.evaluate(X_train[val_idx], y[val_idx], verbose = 0, batch_size = BATCH_SIZE, return_dict = True)['score'])
    
    print("predict test...")
    test_preds+=model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)/NFOLD


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/239518399.py in <cell line: 0>()
     12 BATCH_SIZE=128
     13 EPOCHS = 800
---> 14 for tr_idx, val_idx in kf.split(X_train):
     15     cnt += 1
     16     print(f"FOLD {cnt}")

NameError: name 'X_train' is not defined

## === cell 34
score = history.history['score']
val_score = history.history['val_score']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize = (20,5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, score, label = 'Training Accuracy')
plt.plot(epochs_range, val_score, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.ylim(0.3 * np.mean(val_loss), 1.8 * np.mean(val_loss))

plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.show()


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3505667508.py in <cell line: 0>()
      1 # fetch results from history
----> 2 score = history.history['score']
      3 val_score = history.history['val_score']
      4 
      5 loss = history.history['loss']

NameError: name 'history' is not defined

## === cell 35
np.mean(OOF_val_score)


## === cell 36
sigma_opt = mean_absolute_error(y, train_preds[:,1])
sigma_uncertain = train_preds[:,2] - train_preds[:,0]
sigma_mean = np.mean(sigma_uncertain)
print(sigma_opt, sigma_mean)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2809666322.py in <cell line: 0>()
      1 ## FIND OPTIMIZED STANDARD-DEVIATION
----> 2 sigma_opt = mean_absolute_error(y, train_preds[:,1])
      3 sigma_uncertain = train_preds[:,2] - train_preds[:,0]
      4 sigma_mean = np.mean(sigma_uncertain)
      5 print(sigma_opt, sigma_mean)

NameError: name 'y' is not defined

## === cell 37
sub['FVC1'] = test_preds[:, 1]
sub['Confidence1'] = test_preds[:,2] - test_preds[:,0]

submission = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
submission.loc[~submission.FVC1.isnull()].head(10)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/66350507.py in <cell line: 0>()
      1 ## PREPARE SUBMISSION FILE WITH OUR PREDICTIONS
----> 2 sub['FVC1'] = test_preds[:, 1]
      3 sub['Confidence1'] = test_preds[:,2] - test_preds[:,0]
      4 
      5 # get rid of unused data and show some non-empty data

NameError: name 'test_preds' is not defined

## === cell 38
submission.loc[~submission.FVC1.isnull(),'FVC'] = submission.loc[~submission.FVC1.isnull(),'FVC1']

if sigma_mean < 70:
    submission['Confidence'] = sigma_opt
else:
    submission.loc[~submission.FVC1.isnull(),'Confidence'] = submission.loc[~submission.FVC1.isnull(),'Confidence1']


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1024234948.py in <cell line: 0>()
----> 1 submission.loc[~submission.FVC1.isnull(),'FVC'] = submission.loc[~submission.FVC1.isnull(),'FVC1']
      2 
      3 if sigma_mean < 70:
      4     submission['Confidence'] = sigma_opt
      5 else:

NameError: name 'submission' is not defined

## === cell 39
submission.describe().T


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1258828452.py in <cell line: 0>()
----> 1 submission.describe().T

NameError: name 'submission' is not defined

## === cell 40
org_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(org_test)):
    submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'FVC'] = org_test.FVC[i]
    submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'Confidence'] = 70


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2771494234.py in <cell line: 0>()
      2 
      3 for i in range(len(org_test)):
----> 4     submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'FVC'] = org_test.FVC[i]
      5     submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'Confidence'] = 70

NameError: name 'submission' is not defined

## === cell 41
submission[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index = False)


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4006308278.py in <cell line: 0>()
----> 1 submission[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index = False)

NameError: name 'submission' is not defined

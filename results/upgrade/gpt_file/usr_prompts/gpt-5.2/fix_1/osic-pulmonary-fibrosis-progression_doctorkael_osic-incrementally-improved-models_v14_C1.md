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
scipy==1.15.3
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

-6.8653

# 6. Current score

-11.52489

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pydicom

plt.style.use("dark_background")
%matplotlib inline


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
main_dir = "../input/osic-pulmonary-fibrosis-progression"

!ls {main_dir}


## === cell 2
train_files = tf.io.gfile.glob(main_dir+"/train/*/*")
test_files = tf.io.gfile.glob(main_dir+"/test/*/*")
sample_sub = pd.read_csv(main_dir + "/sample_submission.csv")
train = pd.read_csv(main_dir + "/train.csv")
test = pd.read_csv(main_dir + "/test.csv")

print ("Number of train patients: {}\nNumber of test patients: {:4}"
       .format(train.Patient.nunique(), test.Patient.nunique()))

print ("\nTotal number of Train patient records: {}\nTotal number of Test patient records: {:6}"
       .format(len(train_files), len(test_files)))

train.shape, test.shape, sample_sub.shape


## === cell 3
temp = pydicom.dcmread(train_files[0])
type(temp)


## === cell 4
print ('\n'.join(str(temp).split("\n")[:15]))


## === cell 5
list(temp.keys())[:5]


## === cell 6
print (temp.dir()[:5])


## === cell 7
(
    temp.BitsAllocated, 
    temp.get('BitsAllocated'),
    temp.data_element("BitsAllocated").value, 
    temp[(0x28, 0x100)].value, 
    temp.get([0x28, 0x100]).value
)


## === cell 8
key = (0x08, 0x08)
print ("Accessing by a tuple key returns a", type(temp[key]).__name__)


## === cell 9
print (list(filter(lambda x: "__" not in x, dir(pydicom.DataElement))))


## === cell 10
temp[key].VR


## === cell 11
np.unique(list(map(lambda x: x.VR, temp.iterall())))


## === cell 12
temp[key].description()


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/514720707.py in <cell line: 0>()
      1 # usually the neatly formated
      2 # version of the DICOM Keyword
----> 3 temp[key].description()

AttributeError: 'DataElement' object has no attribute 'description'

## === cell 13
print ("Value: {}\nContains {} elements".format(temp['Modality'].value, temp['Modality'].VM))
print ("\nValue: {}\nContains {} elements".format(temp['ImageType'].value, temp['ImageType'].VM))


## === cell 14
temp['PatientSex'].is_empty, temp['ImageType'].is_empty


## === cell 15
temp[(0x18, 0x1151)].keyword


## === cell 16
print ("{} is saved as {}".format(temp['XRayTubeCurrent'].repval, type(temp['XRayTubeCurrent'].repval)))
print ("{} is saved as {}".format(temp['XRayTubeCurrent'].value, type(temp['XRayTubeCurrent'].value)))


## === cell 17
temp.file_meta


## === cell 18
temp.group_dataset(0x28)


## === cell 19
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(temp.pixel_array, cmap='bone');


## === cell 20
train.head()


## === cell 21
test


## === cell 22
sample_sub.tail()


## === cell 23
train.isna().sum().any(), test.isna().sum().any()


## === cell 24
train.info(null_counts=False)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1348182365.py in <cell line: 0>()
      1 # checking data type of each col
----> 2 train.info(null_counts=False)

TypeError: DataFrame.info() got an unexpected keyword argument 'null_counts'

## === cell 25
(train.groupby('Patient').nunique() != 1).sum() == 0


## === cell 26
train.nunique()


## === cell 27
ages = train.groupby('Patient').Age.head(1)
print ("Max Patient Age: {}\nMin Patient Age: {}".format(ages.min(), ages.max()))
ax = ages.plot(kind='hist', bins=50, edgecolor='red', color='y', figsize=(15, 5), xticks=range(49, 89))
ages.plot(kind='kde', ax=ax, xlim=(47, 90), color='w', secondary_y=True);


## === cell 28
f, ax = plt.subplots(figsize=(15, 5), ncols=2)

train.groupby('Patient').SmokingStatus.head(1).value_counts().plot(
    kind='pie', ax=ax[0], autopct=lambda x: str(int(x))+"%", 
    title='Smoking Status Pie chart', 
    colors=['orange', 'blue', 'green'])

train.groupby('Patient').Sex.head(1).value_counts().plot(
    kind='pie', ax=ax[1], autopct=lambda x: str(int(x))+"%", 
    title='Sex pie chart', colors=['red', 'blue']);


## === cell 29
train.groupby(['SmokingStatus', 'Sex'])['Patient'].nunique().unstack().plot(
    kind='bar', stacked=True, figsize=(10, 6), yticks=range(0, 130, 10),
    rot=0, title='Gender Across Smoking Status');


## === cell 30
train.groupby("Sex").agg(['min', 'max', 'mean', 'std']).drop("Age", axis=1)


## --- ERROR in cell 30, traceback:
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

TypeError: Could not convert string 'ID00123637202217151272140ID00123637202217151272140ID00123637202217151272140ID00123637202217151272140ID00123637202217151272140ID00123637202217151272140ID00123637202217151272140ID00123637202217151272140ID00123637202217151272140ID00298637202280361773446ID00298637202280361773446ID00298637202280361773446ID00298637202280361773446ID00298637202280361773446ID00298637202280361773446ID00298637202280361773446ID00298637202280361773446ID00298637202280361773446ID00255637202267923028520ID00255637202267923028520ID00255637202267923028520ID00255637202267923028520ID00255637202267923028520ID00255637202267923028520ID00255637202267923028520ID00255637202267923028520ID00264637202270643353440ID00264637202270643353440ID00264637202270643353440ID00264637202270643353440ID00264637202270643353440ID00264637202270643353440ID00264637202270643353440ID00264637202270643353440ID00128637202219474716089ID00128637202219474716089ID00128637202219474716089ID00128637202219474716089ID00128637202219474716089ID00128637202219474716089ID00128637202219474716089ID00020637202178344345685ID00020637202178344345685ID00020637202178344345685ID00020637202178344345685ID00020637202178344345685ID00020637202178344345685ID00020637202178344345685ID00020637202178344345685ID00020637202178344345685ID00228637202259965313869ID00228637202259965313869ID00228637202259965313869ID00228637202259965313869ID00228637202259965313869ID00228637202259965313869ID00228637202259965313869ID00228637202259965313869ID00228637202259965313869ID00267637202270790561585ID00267637202270790561585ID00267637202270790561585ID00267637202270790561585ID00267637202270790561585ID00267637202270790561585ID00048637202185016727717ID00048637202185016727717ID00048637202185016727717ID00048637202185016727717ID00048637202185016727717ID00048637202185016727717ID00048637202185016727717ID00048637202185016727717ID00048637202185016727717ID00170637202238079193844ID00170637202238079193844ID00170637202238079193844ID00170637202238079193844ID00170637202238079193844ID00170637202238079193844ID00170637202238079193844ID00170637202238079193844ID00170637202238079193844ID00285637202278913507108ID00285637202278913507108ID00285637202278913507108ID00285637202278913507108ID00285637202278913507108ID00285637202278913507108ID00285637202278913507108ID00285637202278913507108ID00285637202278913507108ID00248637202266698862378ID00248637202266698862378ID00248637202266698862378ID00248637202266698862378ID00248637202266698862378ID00248637202266698862378ID00248637202266698862378ID00248637202266698862378ID00248637202266698862378ID00221637202258717315571ID00221637202258717315571ID00221637202258717315571ID00221637202258717315571ID00221637202258717315571ID00221637202258717315571ID00221637202258717315571ID00221637202258717315571ID00221637202258717315571ID00169637202238024117706ID00169637202238024117706ID00169637202238024117706ID00169637202238024117706ID00169637202238024117706ID00169637202238024117706ID00169637202238024117706ID00169637202238024117706ID00169637202238024117706ID00343637202287577133798ID00343637202287577133798ID00343637202287577133798ID00343637202287577133798ID00343637202287577133798ID00343637202287577133798ID00343637202287577133798ID00343637202287577133798ID00343637202287577133798ID00343637202287577133798ID00401637202305320178010ID00401637202305320178010ID00401637202305320178010ID00401637202305320178010ID00401637202305320178010ID00401637202305320178010ID00401637202305320178010ID00401637202305320178010ID00401637202305320178010ID00051637202185848464638ID00051637202185848464638ID00051637202185848464638ID00051637202185848464638ID00051637202185848464638ID00051637202185848464638ID00051637202185848464638ID00051637202185848464638ID00051637202185848464638ID00232637202260377586117ID00232637202260377586117ID00232637202260377586117ID00232637202260377586117ID00232637202260377586117ID00232637202260377586117ID00232637202260377586117ID00232637202260377586117ID00232637202260377586117ID00149637202232704462834ID00149637202232704462834ID00149637202232704462834ID00149637202232704462834ID00149637202232704462834ID00149637202232704462834ID00149637202232704462834ID00149637202232704462834ID00149637202232704462834ID00312637202282607344793ID00312637202282607344793ID00312637202282607344793ID00312637202282607344793ID00312637202282607344793ID00312637202282607344793ID00312637202282607344793ID00312637202282607344793ID00312637202282607344793ID00367637202296290303449ID00367637202296290303449ID00367637202296290303449ID00367637202296290303449ID00367637202296290303449ID00367637202296290303449ID00367637202296290303449ID00367637202296290303449ID00367637202296290303449ID00288637202279148973731ID00288637202279148973731ID00288637202279148973731ID00288637202279148973731ID00288637202279148973731ID00288637202279148973731ID00288637202279148973731ID00288637202279148973731ID00341637202287410878488ID00341637202287410878488ID00341637202287410878488ID00341637202287410878488ID00341637202287410878488ID00341637202287410878488ID00341637202287410878488ID00341637202287410878488ID00341637202287410878488ID00241637202264294508775ID00241637202264294508775ID00241637202264294508775ID00241637202264294508775ID00241637202264294508775ID00241637202264294508775ID00241637202264294508775ID00241637202264294508775ID00241637202264294508775ID00086637202203494931510ID00086637202203494931510ID00086637202203494931510ID00086637202203494931510ID00086637202203494931510ID00086637202203494931510ID00086637202203494931510ID00086637202203494931510ID00086637202203494931510ID00275637202271440119890ID00275637202271440119890ID00275637202271440119890ID00275637202271440119890ID00275637202271440119890ID00275637202271440119890ID00275637202271440119890ID00275637202271440119890ID00275637202271440119890ID00078637202199415319443ID00078637202199415319443ID00078637202199415319443ID00078637202199415319443ID00078637202199415319443ID00078637202199415319443ID00078637202199415319443ID00078637202199415319443ID00078637202199415319443ID00075637202198610425520ID00075637202198610425520ID00075637202198610425520ID00075637202198610425520ID00075637202198610425520ID00075637202198610425520ID00075637202198610425520ID00075637202198610425520ID00075637202198610425520ID00192637202245493238298ID00192637202245493238298ID00192637202245493238298ID00192637202245493238298ID00192637202245493238298ID00192637202245493238298ID00192637202245493238298ID00192637202245493238298ID00192637202245493238298ID00405637202308359492977ID00405637202308359492977ID00405637202308359492977ID00405637202308359492977ID00405637202308359492977ID00405637202308359492977ID00405637202308359492977ID00405637202308359492977ID00405637202308359492977ID00130637202220059448013ID00130637202220059448013ID00130637202220059448013ID00130637202220059448013ID00130637202220059448013ID00130637202220059448013ID00130637202220059448013ID00130637202220059448013ID00130637202220059448013ID00023637202179104603099ID00023637202179104603099ID00023637202179104603099ID00023637202179104603099ID00023637202179104603099ID00023637202179104603099ID00023637202179104603099ID00023637202179104603099ID00023637202179104603099' to numeric

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3870877791.py in <cell line: 0>()
----> 1 train.groupby("Sex").agg(['min', 'max', 'mean', 'std']).drop("Age", axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
   1430 
   1431         op = GroupByApply(self, func, args=args, kwargs=kwargs)
-> 1432         result = op.agg()
   1433         if not is_dict_like(func) and result is not None:
   1434             # GH #52849

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg(self)
    191         elif is_list_like(func):
    192             # we require a list, but not a 'str'
--> 193             return self.agg_list_like()
    194 
    195         if callable(func):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg_list_like(self)
    324         Result of aggregation.
    325         """
--> 326         return self.agg_or_apply_list_like(op_name="agg")
    327 
    328     def compute_list_like(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg_or_apply_list_like(self, op_name)
   1569             obj, "as_index", True, condition=hasattr(obj, "as_index")
   1570         ):
-> 1571             keys, results = self.compute_list_like(op_name, selected_obj, kwargs)
   1572         result = self.wrap_results_list_like(keys, results)
   1573         return result

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in compute_list_like(self, op_name, selected_obj, kwargs)
    383                     else self.args
    384                 )
--> 385                 new_res = getattr(colg, op_name)(func, *args, **kwargs)
    386                 results.append(new_res)
    387                 indices.append(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
    255             kwargs["engine"] = engine
    256             kwargs["engine_kwargs"] = engine_kwargs
--> 257             ret = self._aggregate_multiple_funcs(func, *args, **kwargs)
    258             if relabeling:
    259                 # columns is not narrowed by mypy from relabeling flag

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in _aggregate_multiple_funcs(self, arg, *args, **kwargs)
    360             for idx, (name, func) in enumerate(arg):
    361                 key = base.OutputKey(label=name, position=idx)
--> 362                 results[key] = self.aggregate(func, *args, **kwargs)
    363 
    364         if any(isinstance(x, DataFrame) for x in results.values()):

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
    247             if engine_kwargs is not None:
    248                 kwargs["engine_kwargs"] = engine_kwargs
--> 249             return getattr(self, func)(*args, **kwargs)
    250 
    251         elif isinstance(func, abc.Iterable):

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

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/base.py in grouped_reduce(self, func)
    365     def grouped_reduce(self, func):
    366         arr = self.array
--> 367         res = func(arr)
    368         index = default_index(len(res))
    369 

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

## === cell 31
f, ax = plt.subplots(nrows=2, figsize=(15, 10))

sc = ax[0].scatter(
    'Age', 'FVC', c=train.Sex.map({'Male': 0, 'Female': 1}), 
    s=(train.Weeks+5), data=train, cmap='brg_r', alpha=0.5)

ax[0].set(
    xlabel='Age', ylabel='FVC', xticks=range(48, 90), 
    title="Age Vs FVC")
ax[0].legend(sc.legend_elements()[0], ['Male', 'Female'])

sc = ax[1].scatter(
    'Age', 'Percent', c=train.Sex.map({'Male': 0, 'Female': 1}), 
    s=(train.Weeks+5), data=train, cmap='brg_r', alpha=0.5)

ax[1].set(
    xlabel='Age', ylabel='FVC', xticks=range(48, 90), 
    title="Age Vs Percent");
ax[1].legend(sc.legend_elements()[0], ['Male', 'Female'])

f.suptitle("Across Different Genders")
f.tight_layout(rect=[0, 0.03, 1, 0.95]);


## === cell 32
(train
 .groupby(["SmokingStatus"])[['Weeks', 'FVC', 'Percent']].agg({
     "Weeks": "count",
     "FVC": ['min', 'mean', 'max'], 
     "Percent": ['min', 'mean', 'max']})
 .rename({"Weeks": "Cumulative Records"}, axis=1))


## === cell 33
(train
 .groupby(["SmokingStatus", 'Sex'])[['Weeks', 'FVC', 'Percent']].agg({
     "Weeks": "count",
     "FVC": ['min', 'mean', 'max'], 
     "Percent": ['min', 'mean', 'max']})
 .rename({"Weeks": "Cumulative Records"}, axis=1))


## === cell 34
from scipy.signal import savgol_filter

def display_FVC_progress(data, title, smooth=True, drop=1, median=True):
    
    agg = ['count', 'min', 'median', 'max']
    if not median:
        agg.remove("median")

    temp = data.groupby('Weeks')[['FVC']].agg(agg)
    temp = temp[temp['FVC']['count'] > drop].drop(("FVC", 'count'), axis=1)

    if smooth:
        temp['FVC', 'max'] = savgol_filter(temp['FVC', 'max'], 9, 3)
        temp['FVC', 'min'] = savgol_filter(temp['FVC', 'min'], 9, 3)

    ax = temp.plot(
        figsize=(15, 5), 
        title=f'Variation & progress of FVC over the Weeks ({title})', 
        legend=True, xticks=range(-10, 150, 5)
    );

    ax.fill_between(temp.index, temp['FVC', 'max'], temp['FVC', 'min'], color='green');


## === cell 35
display_FVC_progress(train, 'All Categories')


## === cell 36
display_FVC_progress(train.loc[train.Sex == 'Male'], 'Only Males', drop=1, smooth=True, median=False)
display_FVC_progress(train.loc[train.Sex == 'Female'], 'Only Females', drop=1, smooth=True, median=False)


## === cell 37
display_FVC_progress(train.loc[train.SmokingStatus == 'Ex-smoker'], 'Category: Ex Smokers', median=False)

display_FVC_progress(
    train.loc[train.SmokingStatus == 'Currently smokes'], 
    'Category: Current Smokers', median=False)

display_FVC_progress(train.loc[train.SmokingStatus == 'Never smoked'], 'Category: Never Smoked', median=False)


## === cell 38
train.groupby('Patient')['Weeks'].count().agg(['min', 'max', 'mean'])


## === cell 39
choice = 2
temp = (train.groupby(['Sex', 'SmokingStatus'])['Patient']
        .apply(lambda x: np.random.choice(np.unique(x), size=choice, replace=False)).reset_index())

f, ax = plt.subplots(ncols=choice, nrows=6, figsize=(20, 30))
for i, sex, status, patients in temp.itertuples():
    
    for j in range(choice):
    
        (train
         .loc[train.Patient == patients[j], ['FVC', 'Weeks']]
         .set_index('Weeks')
         .plot(ax=ax[i][j], title=f"{sex} Patient\n{status}", legend=False)
        )
        
f.tight_layout();


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1213348699.py in <cell line: 0>()
      3 choice = 2
      4 temp = (train.groupby(['Sex', 'SmokingStatus'])['Patient']
----> 5         .apply(lambda x: np.random.choice(np.unique(x), size=choice, replace=False)).reset_index())
      6 
      7 f, ax = plt.subplots(ncols=choice, nrows=6, figsize=(20, 30))

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in apply(self, func, *args, **kwargs)
    228     )
    229     def apply(self, func, *args, **kwargs) -> Series:
--> 230         return super().apply(func, *args, **kwargs)
    231 
    232     @doc(_agg_template_series, examples=_agg_examples_doc, klass="Series")

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in apply(self, func, include_groups, *args, **kwargs)
   1822         with option_context("mode.chained_assignment", None):
   1823             try:
-> 1824                 result = self._python_apply_general(f, self._selected_obj)
   1825                 if (
   1826                     not isinstance(self.obj, Series)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _python_apply_general(self, f, data, not_indexed_same, is_transform, is_agg)
   1883             data after applying f
   1884         """
-> 1885         values, mutated = self._grouper.apply_groupwise(f, data, self.axis)
   1886         if not_indexed_same is None:
   1887             not_indexed_same = mutated

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in apply_groupwise(self, f, data, axis)
    917             # group might be modified
    918             group_axes = group.axes
--> 919             res = f(group)
    920             if not mutated and not _is_indexed_like(res, group_axes, axis):
    921                 mutated = True

/tmp/ipykernel_11/1213348699.py in <lambda>(x)
      3 choice = 2
      4 temp = (train.groupby(['Sex', 'SmokingStatus'])['Patient']
----> 5         .apply(lambda x: np.random.choice(np.unique(x), size=choice, replace=False)).reset_index())
      6 
      7 f, ax = plt.subplots(ncols=choice, nrows=6, figsize=(20, 30))

mtrand.pyx in numpy.random.mtrand.RandomState.choice()

ValueError: Cannot take a larger sample than population when 'replace=False'

## === cell 40
def laplace_log_likelihood(y_true, y_pred, sigma=70):
    sigma_clipped = tf.maximum(sigma, 70)

    delta_clipped = tf.minimum(tf.abs(y_true - y_pred), 1000)
    
    delta_clipped = tf.cast(delta_clipped, dtype=tf.float32)
    sigma_clipped = tf.cast(sigma_clipped, dtype=tf.float32)
    
    score = - tf.sqrt(2.0) * delta_clipped / sigma_clipped - tf.math.log(tf.sqrt(2.0) * sigma_clipped)
    
    return tf.reduce_mean(score)


## === cell 41
laplace_log_likelihood(train['FVC'], train['FVC'], 70)


## === cell 42
high_delta = []
zero_delta = []
for i in range(70, 2000):
    high_delta.append(laplace_log_likelihood(train['FVC'], 9e5, sigma=i).numpy())
    zero_delta.append(laplace_log_likelihood(train['FVC'], train['FVC'], sigma=i).numpy())
    
   
f, ax = plt.subplots(figsize=(20, 5), ncols=2)
ax[0].plot(high_delta)
ax[0].set(xlabel='Confidence', ylabel='Scores', title='$\Delta = 1000$ (Incorrect Predictions)')

ax[1].plot(zero_delta)
ax[1].set(xlabel='Confidence', ylabel='Scores', title='$\Delta = 0$ (Correct Predictions)');


## === cell 43
sub = sample_sub.copy()
sub.FVC = 9e3
sub.Confidence = 250
sub.head()


## === cell 44
sub.to_csv("conf_submission.csv", index=False)


## === cell 45
ages = pd.cut(train.Age, 10).cat.codes

dumb_preds = [
    ("Sample Submission Idea", 2000),
    
    ("Min Scores", train['FVC'].min()), 
    ("25th Quantile Scores", train['FVC'].quantile(0.25)), 
    ("Median Scores", train['FVC'].median()), 
    ("75th Quantile Scores", train['FVC'].quantile(0.75)), 
    ("Max Scores", train['FVC'].max()), 
    
    ("Mean Scores", train['FVC'].mean()),
    
    ("Weeks median", train.groupby('Weeks')['FVC'].transform('median')), 
    ("Binned Age median", train.groupby([ages])['FVC'].transform('median')),
    ("SmokingStatus median", train.groupby(['SmokingStatus'])['FVC'].transform('median')),
    ("Sex median", train.groupby(['Sex'])['FVC'].transform('median')),
    
    ("Age-Sex median", train.groupby([ages, 'Sex'])['FVC'].transform('median')),
    ("Age-SmokingStatus median", train.groupby([ages, 'SmokingStatus'])['FVC'].transform('median')),
    ("Weekly-Sex median", train.groupby(['Weeks', 'Sex'])['FVC'].transform('median')),
    ("Weekly-Smoking median", train.groupby(['Weeks', 'SmokingStatus'])['FVC'].transform('median')),
    ("Weekly-Age median", train.groupby(['Weeks', ages])['FVC'].transform('median')),
    
    ("Weekly-Sex-Smoking median", train.groupby(['Weeks', 'Sex', 'SmokingStatus'])['FVC'].transform('median')),
    ("Weekly-Sex-Age median", train.groupby(["Weeks", "Sex", ages])['FVC'].transform('median')),
    ("Weekly-Smoking-Age median", train.groupby(["Weeks", "SmokingStatus", ages])['FVC'].transform('median')),
]

sigma = 250

sigma_l = train.groupby('Patient')['Weeks'].transform(lambda x: np.linspace(225, 275, len(x))).values


print ("Some Dumb Ideas & their Scores:\n")
for text, preds in dumb_preds:
    
    score = laplace_log_likelihood(train['FVC'], preds, sigma).numpy()
    print (f"\t{text} with fixed conf {' ' * (29 - len(text))}: {score:-6.2f}")
    
    score = laplace_log_likelihood(train['FVC'], preds, sigma_l).numpy()
    print (f"\t{text} with conf swelling {' ' * (26 - len(text))}: {score:-6.2f}\n")


## === cell 46
(train.groupby(["Weeks", "SmokingStatus", ages])['FVC'].count() != 1).sum()


## === cell 47
for name, median in (
    ("Weekly median", train.groupby('Weeks')['FVC'].transform("count").mean()), 
    ("Binned Age median", train.groupby([ages])['FVC'].transform("count").mean()),
    ("SmokingStatus median", train.groupby(['SmokingStatus'])['FVC'].transform("count").mean()),
    ("Sex median", train.groupby(['Sex'])['FVC'].transform("count").mean()),
    ("Age-Sex median", train.groupby([ages, 'Sex'])['FVC'].transform("count").mean()),
    ("Age-SmokingStatus median", train.groupby([ages, 'SmokingStatus'])['FVC'].transform("count").mean()),
    ("Weekly-Sex median", train.groupby(['Weeks', 'Sex'])['FVC'].transform("count").mean()),
    ("Weekly-Smoking median", train.groupby(['Weeks', 'SmokingStatus'])['FVC'].transform("count").mean()),
    ("Weekly-Age median", train.groupby(['Weeks', ages])['FVC'].transform("count").mean()),
    ("Weekly-Sex-Smoking median", train.groupby(['Weeks', 'Sex', 'SmokingStatus'])['FVC'].transform("count").mean()),
    ("Weekly-Sex-Age median", train.groupby(["Weeks", "Sex", ages])['FVC'].transform("count").mean()),
    ("Weekly-Smoking-Age median", train.groupby(["Weeks", "SmokingStatus", ages])['FVC'].transform("count").mean()),
):
    print (f"{name} {' ' * (30 - len(name))}: {median:-6.1f}")


## === cell 48
sub = sample_sub.Patient_Week.str.extract("(ID\w+)_(\-?\d+)").rename({0: "Patient", 1: "Weeks"}, axis=1)
sub['Weeks'] = sub['Weeks'].astype(int)
sub = pd.merge(sub, test[['Patient', 'Sex', 'SmokingStatus']], on='Patient')
sub.head()


## === cell 49
week_temp = train.groupby(["Weeks", 'Sex'])['FVC'].median()
sex_temp = train.groupby(['Sex'])['FVC'].median()

for index, week, sex in sub.iloc[:, 1:3].itertuples():
    if (week, sex) in week_temp:
        sub.loc[index, 'FVC'] = week_temp[week, sex]
        sub.loc[index, 'Confidence'] = sigma
    else:
        sub.loc[index, 'FVC'] = sex_temp[sex]
        sub.loc[index, 'Confidence'] = sigma + 100
        
sub.sample(5)


## === cell 50
sub["Patient_Week"] = sub.Patient + "_" + sub.Weeks.astype(str)
sub.head()


## === cell 51
sub[['Patient_Week', 'FVC', 'Confidence']].to_csv("pd_submission.csv", index=False)


## === cell 52
x = train[['Weeks', 'Age', 'Sex', 'SmokingStatus']].copy()
y = train['FVC'].copy()

stats = x.describe().T

x = pd.get_dummies(x, columns=['Sex', 'SmokingStatus'], drop_first=True)
for col in ['Weeks', 'Age']:
    x[col] = (x[col] - stats.loc[col, 'min']) / (stats.loc[col, 'max'] - stats.loc[col, 'min'])

x.head()


## === cell 53
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer

l1 = (make_scorer(
    lambda x, y: laplace_log_likelihood(x, y, sigma=sigma).numpy(), 
    greater_is_better=False))

cross_val_score(LinearRegression(), x, y, cv=3, scoring=ll)


## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1271813290.py in <cell line: 0>()
      8     greater_is_better=False))
      9 
---> 10 cross_val_score(LinearRegression(), x, y, cv=3, scoring=ll)

NameError: name 'll' is not defined

## === cell 54
x = train.copy()
y = train['FVC'].copy()

x['base_Week'] = x.groupby('Patient')['Weeks'].transform('min')
x['Base_FVC'] = x.groupby('Patient')['FVC'].transform('first')

stats = x.describe().T

x = pd.get_dummies(x, columns=['Sex', 'SmokingStatus'], drop_first=True)

num_cols = [
    'Weeks', 'Age', 'base_Week', 'Base_FVC', 
]

for col in num_cols:
    x[col] = (x[col] - stats.loc[col, 'min']) / (stats.loc[col, 'max'] - stats.loc[col, 'min'])
    
print (x.corr()['FVC'].abs().sort_values(ascending=False)[1:])

x.drop(['Patient', 'Percent', 'FVC'], axis=1, inplace=True)
x.head()


## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2290728768.py in <cell line: 0>()
     21 
     22 # print out how well our features would do
---> 23 print (x.corr()['FVC'].abs().sort_values(ascending=False)[1:])
     24 
     25 x.drop(['Patient', 'Percent', 'FVC'], axis=1, inplace=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: 'ID00133637202223847701934'

## === cell 55
cross_val_score(LinearRegression(), x, y, cv=3, scoring=ll)


## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2788190953.py in <cell line: 0>()
      1 # how does it perform now?
----> 2 cross_val_score(LinearRegression(), x, y, cv=3, scoring=ll)

NameError: name 'll' is not defined

## === cell 56
lr = LinearRegression().fit(x, y)


## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2753525018.py in <cell line: 0>()
      1 # fit on the train dataset
----> 2 lr = LinearRegression().fit(x, y)

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
    808         # Use the original dtype for conversion if dtype is None
    809         new_dtype = dtype_orig if dtype is None else dtype
--> 810         array = array.astype(new_dtype)
    811         # Since we converted here, we do not need to convert again later
    812         dtype = None

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    131     if copy or arr.dtype == object or dtype == object:
    132         # Explicit copy, or required since NumPy can't view from / to object.
--> 133         return arr.astype(dtype, copy=True)
    134 
    135     return arr.astype(dtype, copy=copy)

ValueError: could not convert string to float: 'ID00133637202223847701934'

## === cell 57
x = (sub.drop(['Confidence', 'Patient_Week'], 1)
     .merge(test[['Patient', 'Weeks', 'FVC', 'Age']], on='Patient')
     .rename({"Weeks_y": "base_Week", "FVC_y": "Base_FVC", "Weeks_x": "Weeks"}, axis=1)
     .drop(['Patient', 'FVC_x'], axis=1))

x = pd.get_dummies(x, columns=['Sex', 'SmokingStatus'])

for col in ['Weeks', 'Age', 'base_Week', 'Base_FVC']:
    x[col] = (x[col] - stats.loc[col, 'min']) / (stats.loc[col, 'max'] - stats.loc[col, 'min'])

x = x[['Weeks', 'Age', 'base_Week', 'Base_FVC', 'Sex_Male',
   'SmokingStatus_Ex-smoker', 'SmokingStatus_Never smoked']]

x.head()


## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/916880575.py in <cell line: 0>()
----> 1 x = (sub.drop(['Confidence', 'Patient_Week'], 1)
      2      .merge(test[['Patient', 'Weeks', 'FVC', 'Age']], on='Patient')
      3      .rename({"Weeks_y": "base_Week", "FVC_y": "Base_FVC", "Weeks_x": "Weeks"}, axis=1)
      4      .drop(['Patient', 'FVC_x'], axis=1))
      5 

TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 were given

## === cell 58
sub['FVC'] = lr.predict(x)
sub.head()


## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3145053606.py in <cell line: 0>()
----> 1 sub['FVC'] = lr.predict(x)
      2 sub.head()

NameError: name 'lr' is not defined

## === cell 59
sub[['Patient_Week', 'FVC', 'Confidence']].to_csv("submission.csv", index=False)


## === cell 60
x = train.copy()

temp = (x.groupby("Patient")
        .apply(lambda x: x.loc[int(
            np.percentile(x['Weeks'].index, q=25)
        ), ["Weeks", "FVC", "Percent"]]))

temp.rename(
    {"Weeks": "Base_Week", 
     "FVC": "Base_FVC", 
     "Percent": "Base_Percent"}, 
    axis=1, inplace=True)

x = x.merge(temp, on='Patient')
x['Where'] = 'train'

temp = sub[['Patient', 'Weeks']].merge(
    test.rename({"Weeks": "Base_Week", 
                 "FVC": "Base_FVC", 
                 "Percent": "Base_Percent"}, axis=1), 
    on='Patient')

temp['Where'] = 'test'
x = pd.concat([x, temp], axis=0)

x['Week_Offset'] = x['Weeks'] - x['Base_Week']

x['Sex'] = x['Sex'].map({"Male": 1, "Female": 0})
x['SmokingStatus'] = x['SmokingStatus'].map({"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2})

x = pd.get_dummies(x, columns=['Sex', 'SmokingStatus'], drop_first=True)

x['Bin_base_FVC'] = pd.cut(x['Base_FVC'], bins=range(0, 7501, 500)).cat.codes / 15

num_cols = ['Weeks', 'Week_Offset', 'Base_Week', 'Age', 'Base_FVC', 'Percent', 'Base_Percent']
for col in num_cols:
    x[col] = (x[col] - x[col].min()) / (x[col].max() - x[col].min())

to_drop = (
    ["FVC", 'Percent']
    
    + [
    ] + 
    
    ['Patient']
)

print (x[x.Where == 'train'].corr()['FVC'].abs().sort_values(ascending=False).drop(to_drop[:-1]))

y = x['FVC'].dropna()
x = x.drop(to_drop, axis=1)

x.head()


## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1500621952.py in <cell line: 0>()
     61 
     62 # print out how well our features would do
---> 63 print (x[x.Where == 'train'].corr()['FVC'].abs().sort_values(ascending=False).drop(to_drop[:-1]))
     64 
     65 y = x['FVC'].dropna()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: 'ID00133637202223847701934'

## === cell 61
cross_val_score(LinearRegression(), x[x.Where == 'train'].drop('Where', 1), y, cv=3, scoring=ll)


## --- ERROR in cell 61, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/96121776.py in <cell line: 0>()
      1 # how does it perform now?
----> 2 cross_val_score(LinearRegression(), x[x.Where == 'train'].drop('Where', 1), y, cv=3, scoring=ll)

TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 were given

## === cell 62
lr = LinearRegression().fit(x[x.Where == 'train'].drop('Where', 1), y)
sub['FVC'] = lr.predict(x[x.Where == 'test'].drop('Where', 1))
sub.head()


## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3888609074.py in <cell line: 0>()
----> 1 lr = LinearRegression().fit(x[x.Where == 'train'].drop('Where', 1), y)
      2 sub['FVC'] = lr.predict(x[x.Where == 'test'].drop('Where', 1))
      3 sub.head()

TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 were given

## === cell 63
sub['Confidence'] = sub['Confidence'] - 50
sub[['Patient_Week', 'FVC', 'Confidence']].to_csv("submission.csv", index=False)

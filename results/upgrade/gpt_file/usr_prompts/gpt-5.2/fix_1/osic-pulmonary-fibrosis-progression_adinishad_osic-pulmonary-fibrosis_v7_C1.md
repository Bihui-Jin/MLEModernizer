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
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

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

-8.1528

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


## === cell 1
os.listdir("../input/osic-pulmonary-fibrosis-progression")


## === cell 2
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")


## === cell 3
train.head()


## === cell 4
test.head()


## === cell 5
submission = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")


## === cell 6
submission.head()


## === cell 7
print(f"Train info {train.shape}")
print(f"test info {test.shape}")
print(f"submission info {submission.shape}")


## === cell 8
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns


## === cell 9
print(f"Total Patient Id {train['Patient'].count()}")
print(f"NUmber of Unique Id {train['Patient'].value_counts().shape[0]}")


## === cell 10
train["SmokingStatus"].value_counts().plot(kind="bar")


## === cell 11
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"
ds = pydicom.dcmread(img)
plt.figure(figsize = (5,5))
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)


## === cell 12
import random


def get_random(smokes):
    smoke_pat = train[train["SmokingStatus"]==smokes] 
    patientz = [i for i in smoke_pat["Patient"]] # patient id list
    r_st = random.choice(patientz) # random choice
    print(r_st)
    image_dir = f"../input/osic-pulmonary-fibrosis-progression/train/{r_st}" # image directory
    image_list = os.listdir(image_dir) # list of images
    c = []
    for t in image_list:
        first, exts = os.path.splitext(t) # split text
        first = int(first) # int
        c.append(first) # append
    d = [num for num in range(1, 31)] # num from 1 to 30
    gh = []
    for x in c:
        if x in d:
            gh.append(x) # if number is in list then append
    fig = plt.figure(figsize=(10, 10)) # figure
    columns = 5
    row = 6
    for ab in gh:
        files = image_dir + "/" + str(ab) + ".dcm" # file directory
        ds = pydicom.dcmread(files) # read dcm file
        fig.add_subplot(row, columns, ab) # add plot
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone) # show images
    plt.suptitle(smokes) # title


## === cell 13
get_random("Ex-smoker")


## === cell 15
submission["Patient"] = submission["Patient_Week"].apply(lambda x:x.split("_")[0])
submission["Weeks"] = submission["Patient_Week"].apply(lambda x:x.split("_")[1])

submission =  submission[['Patient','Weeks', 'Confidence','Patient_Week']]
submission = submission.merge(test.drop('Weeks', axis=1), on="Patient")


## === cell 16
submission.tail()


## === cell 17
submission.shape


## === cell 18
train.head()


## === cell 19
test


## === cell 20
sns.heatmap(train.corr(), annot=True)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4132553259.py in <cell line: 0>()
----> 1 sns.heatmap(train.corr(), annot=True)

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

## === cell 21
test


## === cell 22
submission["Patient"].unique()


## === cell 23
train.shape


## === cell 24
train["Dataset"] = "train"
test["Dataset"] = "test"
submission["Dataset"] = "submission"


## === cell 25
dataset = train.append([test, submission])
dataset = dataset.reset_index()
dataset = dataset.drop(columns=['index'])


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/445223662.py in <cell line: 0>()
----> 1 dataset = train.append([test, submission])
      2 dataset = dataset.reset_index()
      3 dataset = dataset.drop(columns=['index'])

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 26
dataset.head()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2754860658.py in <cell line: 0>()
----> 1 dataset.head()

NameError: name 'dataset' is not defined

## === cell 27
dataset["Weeks"] = dataset["Weeks"].astype("int64")


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1610033277.py in <cell line: 0>()
----> 1 dataset["Weeks"] = dataset["Weeks"].astype("int64")

NameError: name 'dataset' is not defined

## === cell 28
dataset.info()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3981094352.py in <cell line: 0>()
----> 1 dataset.info()

NameError: name 'dataset' is not defined

## === cell 29
dataset["First_week"] = dataset["Weeks"]
dataset.loc[dataset.Dataset=='submission','First_week'] = np.nan
dataset["First_week"] = dataset.groupby('Patient')['First_week'].transform('min')


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3693660423.py in <cell line: 0>()
----> 1 dataset["First_week"] = dataset["Weeks"]
      2 dataset.loc[dataset.Dataset=='submission','First_week'] = np.nan
      3 dataset["First_week"] = dataset.groupby('Patient')['First_week'].transform('min')

NameError: name 'dataset' is not defined

## === cell 30
dataset.head()


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2754860658.py in <cell line: 0>()
----> 1 dataset.head()

NameError: name 'dataset' is not defined

## === cell 31
dataset = dataset.merge(dataset[dataset["Weeks"] == dataset["First_week"]][["Patient", "FVC"]].rename({"FVC": "First_FVC"}, axis=1).groupby("Patient").first().reset_index(), on="Patient", how="left")


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/941358041.py in <cell line: 0>()
----> 1 dataset = dataset.merge(dataset[dataset["Weeks"] == dataset["First_week"]][["Patient", "FVC"]].rename({"FVC": "First_FVC"}, axis=1).groupby("Patient").first().reset_index(), on="Patient", how="left")

NameError: name 'dataset' is not defined

## === cell 32
dataset["Week_diff"] = dataset["Weeks"] - dataset["First_week"]

dataset = pd.concat([dataset,pd.get_dummies(dataset.Sex),pd.get_dummies(dataset.SmokingStatus)], axis=1)

dataset = dataset.drop(columns=['Sex', 'SmokingStatus'])


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3337503283.py in <cell line: 0>()
----> 1 dataset["Week_diff"] = dataset["Weeks"] - dataset["First_week"]
      2 # dataset["FVC_diff"] = dataset["First_FVC"] - dataset["FVC"]
      3 
      4 dataset = pd.concat([dataset,pd.get_dummies(dataset.Sex),pd.get_dummies(dataset.SmokingStatus)], axis=1)
      5 

NameError: name 'dataset' is not defined

## === cell 33
dataset.tail()


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2519326997.py in <cell line: 0>()
----> 1 dataset.tail()

NameError: name 'dataset' is not defined

## === cell 34
dataset.info()


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3981094352.py in <cell line: 0>()
----> 1 dataset.info()

NameError: name 'dataset' is not defined

## === cell 35
train = dataset[dataset["Dataset"]=="train"]
test = dataset[dataset["Dataset"]=="test"]
submission = dataset[dataset["Dataset"]=="submission"]


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/190838229.py in <cell line: 0>()
----> 1 train = dataset[dataset["Dataset"]=="train"]
      2 test = dataset[dataset["Dataset"]=="test"]
      3 submission = dataset[dataset["Dataset"]=="submission"]

NameError: name 'dataset' is not defined

## === cell 36
from sklearn.preprocessing import StandardScaler

col = ['Weeks', 'Percent', 'Age', 'First_week', 'First_FVC', 'Week_diff',
       'Female', 'Male', 'Currently smokes', 'Ex-smoker', 'Never smoked']

train_data = train[col]


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2549487289.py in <cell line: 0>()
      4        'Female', 'Male', 'Currently smokes', 'Ex-smoker', 'Never smoked']
      5 
----> 6 train_data = train[col]

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

KeyError: "['First_week', 'First_FVC', 'Week_diff', 'Female', 'Male', 'Currently smokes', 'Ex-smoker', 'Never smoked'] not in index"

## === cell 37
train_data.isnull().any()


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1055957012.py in <cell line: 0>()
----> 1 train_data.isnull().any()

NameError: name 'train_data' is not defined

## === cell 38
plt.subplots(figsize=(14,10))
g = train.corr()
sns.heatmap(g, annot=True, fmt='.2', cmap="Dark2_r")


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4200601982.py in <cell line: 0>()
      1 plt.subplots(figsize=(14,10))
----> 2 g = train.corr()
      3 sns.heatmap(g, annot=True, fmt='.2', cmap="Dark2_r")

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

## === cell 39
g["FVC"].sort_values(ascending=False)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2599361226.py in <cell line: 0>()
----> 1 g["FVC"].sort_values(ascending=False)

NameError: name 'g' is not defined

## === cell 40
stdscale = StandardScaler()
train_data[col] = stdscale.fit_transform(train_data[col])


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3364217010.py in <cell line: 0>()
      1 stdscale = StandardScaler()
----> 2 train_data[col] = stdscale.fit_transform(train_data[col])

NameError: name 'train_data' is not defined

## === cell 41
train_data[col]


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1062915213.py in <cell line: 0>()
----> 1 train_data[col]

NameError: name 'train_data' is not defined

## === cell 42
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


## === cell 43
model_params = {
    "svr": {
        "model": SVR(),
        "params": {
            "C": [1, 10, 20],
            "kernel":['linear', 'poly', 'rbf', 'sigmoid']
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators":[100, 200]
        }
    },
    "LR": {
        "model": LinearRegression(),
        "params": {}
    }
}


## === cell 44
from sklearn.model_selection import GridSearchCV
scores = []
for model_name, param in model_params.items():
    clf = GridSearchCV(param["model"], param["params"], cv=10, return_train_score=False)
    clf.fit(train_data[col], train["FVC"])
    scores.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_,
    })

df = pd.DataFrame(scores, columns=["model", "best_score", "best_params"])
df


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/360157516.py in <cell line: 0>()
      3 for model_name, param in model_params.items():
      4     clf = GridSearchCV(param["model"], param["params"], cv=10, return_train_score=False)
----> 5     clf.fit(train_data[col], train["FVC"])
      6     scores.append({
      7         "model": model_name,

NameError: name 'train_data' is not defined

## === cell 45
model = LinearRegression()

model.fit(train_data[col], train["FVC"])


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/887447493.py in <cell line: 0>()
      1 model = LinearRegression()
      2 
----> 3 model.fit(train_data[col], train["FVC"])

NameError: name 'train_data' is not defined

## === cell 46
pred = model.predict(train_data)
pred


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1090603787.py in <cell line: 0>()
----> 1 pred = model.predict(train_data)
      2 pred

NameError: name 'train_data' is not defined

## === cell 47
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(train["FVC"], pred, squared=False)

print(mse)

mae = mean_absolute_error(train["FVC"], pred)
print(mae)


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/657737966.py in <cell line: 0>()
      1 from sklearn.metrics import mean_squared_error, mean_absolute_error
      2 
----> 3 mse = mean_squared_error(train["FVC"], pred, squared=False)
      4 
      5 print(mse)

NameError: name 'pred' is not defined

## === cell 48
a = list(train["FVC"])
b = list(pred)

sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(15, 5))
plt.plot(b[:10], c='green', label= 'predictions')
plt.plot(a[:10], c='red', label= 'actual')
plt.legend()


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3637477047.py in <cell line: 0>()
      1 a = list(train["FVC"])
----> 2 b = list(pred)
      3 
      4 sns.set_style('whitegrid')
      5 f, ax = plt.subplots(figsize=(15, 5))

NameError: name 'pred' is not defined

## === cell 49
submission[col].isnull().any()


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1492198279.py in <cell line: 0>()
----> 1 submission[col].isnull().any()

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

KeyError: "['First_week', 'First_FVC', 'Week_diff', 'Female', 'Male', 'Currently smokes', 'Ex-smoker', 'Never smoked'] not in index"

## === cell 50
sub_data = submission[col]
sub_data = stdscale.fit_transform(sub_data[col])


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3592765819.py in <cell line: 0>()
----> 1 sub_data = submission[col]
      2 sub_data = stdscale.fit_transform(sub_data[col])

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

KeyError: "['First_week', 'First_FVC', 'Week_diff', 'Female', 'Male', 'Currently smokes', 'Ex-smoker', 'Never smoked'] not in index"

## === cell 51
pred_2 = model.predict(sub_data)


## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1227360477.py in <cell line: 0>()
----> 1 pred_2 = model.predict(sub_data)

NameError: name 'sub_data' is not defined

## === cell 52
pred_2


## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1343341344.py in <cell line: 0>()
----> 1 pred_2

NameError: name 'pred_2' is not defined

## === cell 53
a = list(submission["FVC"])
b = list(pred_2)

sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(15, 5))
plt.plot(b, c='green', label= 'predictions')
plt.plot(a, c='red', label= 'actual')
plt.legend()


## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/86974771.py in <cell line: 0>()
      1 a = list(submission["FVC"])
----> 2 b = list(pred_2)
      3 
      4 sns.set_style('whitegrid')
      5 f, ax = plt.subplots(figsize=(15, 5))

NameError: name 'pred_2' is not defined

## === cell 54
submission["FVC"]


## === cell 55
test


## === cell 56
submission


## === cell 57
train


## === cell 58
submission["FVC_1"] = pred_2

confidence_dict={}
for id in submission['Patient'].unique():
    real=float(test[test['Patient']==id]['FVC'])
    predicted=float(submission[(submission['Patient']==id) & (submission['Weeks'].astype(int)==int(test[test['Patient']==id]['Weeks']))]['FVC_1'])
    confidence_dict[id]=abs(real-predicted)
    
    
confidence=[]
for i in range(len(submission)):
    confidence.append(confidence_dict[submission.iloc[i,0]])
submission['Confidence']=confidence


## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4008217455.py in <cell line: 0>()
----> 1 submission["FVC_1"] = pred_2
      2 
      3 confidence_dict={}
      4 for id in submission['Patient'].unique():
      5     real=float(test[test['Patient']==id]['FVC'])

NameError: name 'pred_2' is not defined

## === cell 59
new = submission[["Patient_Week", "FVC_1", "Confidence"]]
new.rename(columns={"FVC_1":"FVC"}, inplace=True)


## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1076142092.py in <cell line: 0>()
----> 1 new = submission[["Patient_Week", "FVC_1", "Confidence"]]
      2 new.rename(columns={"FVC_1":"FVC"}, inplace=True)

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

KeyError: "['FVC_1'] not in index"

## === cell 60
new.to_csv("submission.csv", index=False)


## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2276357689.py in <cell line: 0>()
----> 1 new.to_csv("submission.csv", index=False)

NameError: name 'new' is not defined

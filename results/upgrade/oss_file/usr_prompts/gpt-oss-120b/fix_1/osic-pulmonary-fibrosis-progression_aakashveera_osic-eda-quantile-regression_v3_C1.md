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
pillow==11.3.0
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

-6.916

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import re
from PIL import Image
import random
import pydicom
import gc

import warnings
warnings.filterwarnings("ignore")


## === cell 1
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
df = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
sample = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")


## === cell 3
df.head()


## === cell 4
df.info()


## === cell 5
print(f"Out of 1549 entried there were only {df['Patient'].nunique()} unique patients")


## === cell 6
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,7))
sns.countplot(df['Sex'],ax=ax1).set_title("GENDER COUNT OF GIVEN DATA");
sns.countplot(df['SmokingStatus'],ax=ax2).set_title("SMOKING STATUS COUNT OF GIVEN DATA");


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/625895044.py in <cell line: 0>()
      1 fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,7))
----> 2 sns.countplot(df['Sex'],ax=ax1).set_title("GENDER COUNT OF GIVEN DATA");
      3 sns.countplot(df['SmokingStatus'],ax=ax2).set_title("SMOKING STATUS COUNT OF GIVEN DATA");

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

## === cell 7
print("Since the data has dupilcated values let's recheck by dropping duplicates")
print()
print("######### GENDER ##########")
print()
print(df[['Patient','Sex','SmokingStatus']].drop_duplicates()['Sex'].value_counts())
print()
print("####### SMOKING STATUS ########")
print()
print(df[['Patient','Sex','SmokingStatus']].drop_duplicates()['SmokingStatus'].value_counts())


## === cell 8
sns.set_style('whitegrid')
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,7))

print(f"FVC minimum value - {df['FVC'].min()}, FVC maximum value - {df['FVC'].max()}                 \
          WEEKS minimum value - {df['Weeks'].min()}, WEEKS maximum value - {df['Weeks'].max()}")

sns.distplot(df['FVC'],ax=ax1,kde=False,bins=40,color='salmon').set_title("FVC DISTRIBUTION");
sns.distplot(df['Weeks'],ax=ax2,kde=False,bins=40,color='salmon').set_title("WEEKS DISTRIBUTION");


## === cell 9
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,7))

print(f"Age minimum value - {df['Age'].min()}, Age maximum value - {df['Age'].max()}                 \
          Percent minimum value - {df['Percent'].min()}, Percent maximum value - {df['Percent'].max()}")

sns.distplot(df['Age'],ax=ax1,kde=False,bins=40,color='plum').set_title("AGE DISTRIBUTION");
sns.distplot(df['Percent'],ax=ax2,kde=False,bins=40,color='plum').set_title("PERCENT DISTRIBUTION");


## === cell 10
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,4))
sns.boxplot(df['Percent'],ax=ax1,palette='winter',orient='h').set_title("PERCENTAGE DETAILS");
sns.boxplot(df['Age'],ax=ax2,palette='winter',orient='h').set_title("AGE DETAILS");


## === cell 11
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,4))
sns.boxplot(df['FVC'],ax=ax1,palette='winter',orient='h').set_title("FVC DETAILS");
sns.boxplot(df['Weeks'],ax=ax2,palette='winter',orient='h').set_title("WEEKS DETAILS");


## === cell 12
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,7));
sns.boxplot(df['Sex'],df['FVC'],palette='winter',orient='v',ax=ax1).set_title("GENDER VS FVC");

sns.boxplot(df['SmokingStatus'],df['FVC'],palette='winter',orient='v',ax=ax2).set_title("SMOKING STAT VS FVC");


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1910663739.py in <cell line: 0>()
      1 fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,7));
----> 2 sns.boxplot(df['Sex'],df['FVC'],palette='winter',orient='v',ax=ax1).set_title("GENDER VS FVC");
      3 
      4 sns.boxplot(df['SmokingStatus'],df['FVC'],palette='winter',orient='v',ax=ax2).set_title("SMOKING STAT VS FVC");

TypeError: boxplot() takes from 0 to 1 positional arguments but 2 positional arguments (and 3 keyword-only arguments) were given

## === cell 13
print("######## FVC #########\n")
print(f"Mean FVC of Male {df[df['Sex']=='Male']['FVC'].mean()}")
print(f"Mean FVC of Female {df[df['Sex']=='Female']['FVC'].mean()}\n")

print("######## Smoking Stat #########\n")
print(f"Mean FVC of Smoker {df[df['SmokingStatus']=='Currently smokes']['FVC'].mean()}")
print(f"Mean FVC of Ex-Smoker {df[df['SmokingStatus']=='Ex-smoker']['FVC'].mean()}")
print(f"Mean FVC of Never Smoked {df[df['SmokingStatus']=='Never smoked']['FVC'].mean()}")


## === cell 14
sns.pairplot(hue='Sex',data=df,x_vars=['Weeks','FVC','Percent','Age'],y_vars=['Weeks','FVC','Percent','Age'],height=3);


## === cell 15
print("FVC decreases over time for most of the cases")
plt.figure(figsize = (15,10))
a = sns.lineplot(x = df["Weeks"], y = df["FVC"], hue = df["Patient"], size=1,legend=False)


## === cell 16
df['Photo count'] = 0
names = df['Patient'].unique()
for name in names:
    file = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'+name
    df.loc[df['Patient']==name,'Photo count'] = len(os.listdir(file))
    
data = df.groupby(by="Patient")["Photo count"].first().reset_index(drop=False)
data = data.sort_values(['Photo count']).reset_index(drop=True)
data['Photo count'].describe()


## === cell 17
plt.figure(figsize=(20,5))
sns.distplot(data['Photo count'],bins=200,kde=False)


## === cell 18
patient_dir = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430"

files = []
for dcm in list(os.listdir(patient_dir)):
    files.append(dcm)
    
files.sort(key=lambda f: int(re.findall('\d+',f)[0]))

datasets = []

for dcm in files:
    path = patient_dir + "/" + dcm
    datasets.append(pydicom.dcmread(path))
    
fig=plt.figure(figsize=(16, 6))

columns = 10
rows = 3

for i in range(columns*rows):
    img = datasets[i].pixel_array
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img, cmap="plasma")
    plt.axis('off');


## === cell 19
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(42)


## === cell 20
df.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])


## === cell 21
sample['Patient'] = sample['Patient_Week'].apply(lambda x:x.split('_')[0])
sample['Weeks'] = sample['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sample =  sample[['Patient','Weeks','Confidence','Patient_Week']]
sample = sample.merge(test.drop('Weeks', axis=1), on="Patient")


## === cell 22
df['WHERE'] = 'train'
test['WHERE'] = 'val'
sample['WHERE'] = 'test'
data = df.append([test, sample])


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1485706808.py in <cell line: 0>()
      2 test['WHERE'] = 'val'
      3 sample['WHERE'] = 'test'
----> 4 data = df.append([test, sample])

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 23
data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')


## --- ERROR in cell 23, traceback:
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

KeyError: 'Weeks'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2242701266.py in <cell line: 0>()
----> 1 data['min_week'] = data['Weeks']
      2 data.loc[data.WHERE=='test','min_week'] = np.nan
      3 data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

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

KeyError: 'Weeks'

## === cell 24
base = data.loc[data.Weeks == data.min_week]
base = base[['Patient','FVC']].copy()
base.columns = ['Patient','min_FVC']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1761360659.py in <cell line: 0>()
----> 1 base = data.loc[data.Weeks == data.min_week]
      2 base = base[['Patient','FVC']].copy()
      3 base.columns = ['Patient','min_FVC']
      4 base['nb'] = 1
      5 base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'Weeks'

## === cell 25
data = data.merge(base, on='Patient', how='left')
data['base_week'] = data['Weeks'] - data['min_week']
del base
gc.collect()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2277863507.py in <cell line: 0>()
----> 1 data = data.merge(base, on='Patient', how='left')
      2 data['base_week'] = data['Weeks'] - data['min_week']
      3 del base
      4 gc.collect()

NameError: name 'base' is not defined

## === cell 26
data[['Sex_Male','SmokingStatus_Ex-smoker','SmokingStatus_Never smoked']] = pd.get_dummies(data[['Sex','SmokingStatus']],drop_first=True)

features = ['Percent','Age','min_FVC','base_week']


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3176415160.py in <cell line: 0>()
----> 1 data[['Sex_Male','SmokingStatus_Ex-smoker','SmokingStatus_Never smoked']] = pd.get_dummies(data[['Sex','SmokingStatus']],drop_first=True)
      2 
      3 features = ['Percent','Age','min_FVC','base_week']

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
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['Sex', 'SmokingStatus'], dtype='object')] are in the [columns]"

## === cell 27
from sklearn.preprocessing import MinMaxScaler

for i in features:
    scaler = MinMaxScaler()
    data[i] = scaler.fit_transform(data[[i]])


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1228716857.py in <cell line: 0>()
      1 from sklearn.preprocessing import MinMaxScaler
      2 
----> 3 for i in features:
      4     scaler = MinMaxScaler()
      5     data[i] = scaler.fit_transform(data[[i]])

NameError: name 'features' is not defined

## === cell 28
df = data.loc[data.WHERE=='train'].drop('WHERE',axis=1)
test = data.loc[data.WHERE=='val'].drop('WHERE',axis=1)
sample = data.loc[data.WHERE=='test'].drop('WHERE',axis=1)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1421847814.py in <cell line: 0>()
----> 1 df = data.loc[data.WHERE=='train'].drop('WHERE',axis=1)
      2 test = data.loc[data.WHERE=='val'].drop('WHERE',axis=1)
      3 sample = data.loc[data.WHERE=='test'].drop('WHERE',axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'WHERE'

## === cell 29
features += ['Sex_Male','SmokingStatus_Ex-smoker','SmokingStatus_Never smoked']


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/220639232.py in <cell line: 0>()
----> 1 features += ['Sex_Male','SmokingStatus_Ex-smoker','SmokingStatus_Never smoked']

NameError: name 'features' is not defined

## === cell 30
df[features].head()


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3080168127.py in <cell line: 0>()
----> 1 df[features].head()

NameError: name 'features' is not defined

## === cell 31
X = df[features].values

y = df['FVC'].values

test_ = sample[features].values

y = y.astype('float64') 


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/647198942.py in <cell line: 0>()
----> 1 X = df[features].values
      2 
      3 y = df['FVC'].values
      4 
      5 test_ = sample[features].values

NameError: name 'features' is not defined

## === cell 32
nh = X.shape[1]
pe = np.zeros((test_.shape[0], 3))
pred = np.zeros((X.shape[0], 3))


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3229659775.py in <cell line: 0>()
----> 1 nh = X.shape[1]
      2 pe = np.zeros((test_.shape[0], 3))
      3 pred = np.zeros((X.shape[0], 3))

NameError: name 'X' is not defined

## === cell 33
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(tf.dtypes.cast(y_true[:, 0], tf.float32) - tf.dtypes.cast(fvc_pred, tf.float32))
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)

def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    e = tf.dtypes.cast(y_true,tf.float32) - tf.dtypes.cast(y_pred,tf.float32)
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss

def make_model(nh):
    z = L.Input((nh,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = M.Model(z, preds, name="CNN")
    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
    return model


## === cell 34
kf = KFold(n_splits=5)
cnt = 0
EPOCHS = 800

for tr_idx, val_idx in kf.split(X):
    cnt += 1
    print(f"FOLD {cnt}")
    model = make_model(nh)
    model.fit(X[tr_idx], y[tr_idx], batch_size=128, epochs=EPOCHS, 
            validation_data=(X[val_idx], y[val_idx]), verbose=0)
    print("train", model.evaluate(X[tr_idx], y[tr_idx], verbose=0, batch_size=128))
    print("val", model.evaluate(X[val_idx], y[val_idx], verbose=0, batch_size=128))
    print("predict val...")
    pred[val_idx] = model.predict(X[val_idx], batch_size=128, verbose=0)
    print("predict test...")
    pe += model.predict(test_, batch_size=128, verbose=0) / 5


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1128283123.py in <cell line: 0>()
      3 EPOCHS = 800
      4 
----> 5 for tr_idx, val_idx in kf.split(X):
      6     cnt += 1
      7     print(f"FOLD {cnt}")

NameError: name 'X' is not defined

## === cell 35
sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:,2] - pred[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/267521803.py in <cell line: 0>()
----> 1 sigma_opt = mean_absolute_error(y, pred[:, 1])
      2 unc = pred[:,2] - pred[:, 0]
      3 sigma_mean = np.mean(unc)
      4 print(sigma_opt, sigma_mean)

NameError: name 'y' is not defined

## === cell 36
idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3813949270.py in <cell line: 0>()
----> 1 idxs = np.random.randint(0, y.shape[0], 100)
      2 plt.plot(y[idxs], label="ground truth")
      3 plt.plot(pred[idxs, 0], label="q25")
      4 plt.plot(pred[idxs, 1], label="q50")
      5 plt.plot(pred[idxs, 2], label="q75")

NameError: name 'y' is not defined

## === cell 37
plt.hist(unc)
plt.title("uncertainty in prediction")
plt.show()


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/260305405.py in <cell line: 0>()
----> 1 plt.hist(unc)
      2 plt.title("uncertainty in prediction")
      3 plt.show()

NameError: name 'unc' is not defined

## === cell 38
sample['FVC1'] = 0.996*pe[:, 1]
sample['Confidence1'] = pe[:, 2] - pe[:, 0]

subm = sample[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3332272464.py in <cell line: 0>()
----> 1 sample['FVC1'] = 0.996*pe[:, 1]
      2 sample['Confidence1'] = pe[:, 2] - pe[:, 0]
      3 
      4 subm = sample[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()

NameError: name 'pe' is not defined

## === cell 39
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if sigma_mean<70:
    subm['Confidence'] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/541779.py in <cell line: 0>()
----> 1 subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
      2 if sigma_mean<70:
      3     subm['Confidence'] = sigma_opt
      4 else:
      5     subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']

NameError: name 'subm' is not defined

## === cell 40
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
for i in range(len(otest)):
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3125650378.py in <cell line: 0>()
      1 otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
      2 for i in range(len(otest)):
----> 3     subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
      4     subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1

NameError: name 'subm' is not defined

## === cell 41
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3080640721.py in <cell line: 0>()
----> 1 subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)

NameError: name 'subm' is not defined

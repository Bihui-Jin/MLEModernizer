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

-8.1338

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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import tensorflow as tf 
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import tensorflow.io as tfio
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import glob as glob 
import seaborn as sns 
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import numpy as np
from skimage import morphology , segmentation , measure 
from sklearn.preprocessing import OneHotEncoder , LabelEncoder 
from sklearn.compose import ColumnTransformer
import os
import pydicom
!pip install dicom
import dicom 
import imageio
from IPython.display import Image
from timeit import timeit
import tensorflow.keras.backend as K
import tensorflow.keras.layers as Layers
import tensorflow.keras.models as Models
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_x = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
print('the no of rows is {} and the no of columns is {} '.format(train_x.shape[0] , train_x.shape[1]))


## === cell 3
train_x.describe()


## === cell 4
test_x = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
print('the no of rows is {} and the no of columns is {} '.format(test_x.shape[0] , test_x.shape[1]))


## === cell 5
test_x.describe()


## === cell 6
sns.countplot( x = 'Sex' , data = train_x )


## === cell 7


new_df = train_x.groupby(
    [
        train_x.Patient,
        train_x.Age,train_x.Sex, 
        train_x.SmokingStatus
    ]
)['Patient'].count()

new_df.index = new_df.index.set_names(
    [
        'id',
        'Age',
        'Sex',
        'SmokingStatus'
    ]
)

new_df = new_df.reset_index()
new_df.rename(columns = {'Patient': 'freq'},inplace = True)

fig = px.bar(new_df, x='id',y ='freq',color='freq')
fig.update_layout(
    xaxis={'categoryorder':'total ascending'},
    title='Distribution of images for each patient'
)
fig.update_xaxes(showticklabels=False)
fig.show()


## === cell 8
fig = px.histogram(
    new_df, 
    x='Age',
    nbins = 42
)

fig.update_traces(
    marker_color='rgb(158,202,225)', 
    marker_line_color='rgb(8,48,107)',
    marker_line_width=1.5, 
    opacity=0.6
)

fig.update_layout(
    title = 'Distribution of Age'
)

fig.show()


## === cell 9
sns.countplot( x = 'SmokingStatus' , data = train_x )


## === cell 10
 fig = px.histogram(
    train_x, 
    x='Age',
    color='SmokingStatus',
    color_discrete_map=
        {
            'Never smoked':'yellow',
            'Currently smokes':'cyan',
            'Ex-smoker': 'green', 
        },
    hover_data=train_x.columns
)

fig.update_layout(title='Distribution of Age w.r.t. SmokingStatus for unique patients')

fig.update_traces(
    marker_line_color='black',
    marker_line_width=1.5, 
    opacity=0.85
)

fig.show()


## === cell 11
plt.figure(figsize = (5 , 5))
sns.countplot(x = 'Sex' , hue = 'SmokingStatus' , data = train_x)


## === cell 12
fig = px.histogram(
    train_x, 
    x='Age',
    color='Sex',
    color_discrete_map=
        {
            'Male':'blue',
            'Female':'mediumturquoise'
        },
    hover_data=train_x.columns
)

fig.update_layout(title='Distribution of Age w.r.t. sex for unique patients')

fig.update_traces(
    marker_line_color='black',
    marker_line_width=1.5, 
    opacity=0.85
)

fig.show()
50
55
60
65
70
75
80
85
0
20
40
60
80
100
120
140


## === cell 13
sns.heatmap(train_x.corr() , annot = True , cmap=plt.cm.cool)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2196672447.py in <cell line: 0>()
      1 # now lets see the correlation between features using heatmap
----> 2 sns.heatmap(train_x.corr() , annot = True , cmap=plt.cm.cool)

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

## === cell 14
a= sns.distplot(train_x['FVC'] , color = 'r' , )
a.set_title('Distribution plot of SVC ' , color = 'g'  , fontsize = 18)


## === cell 15
b = sns.distplot(train_x['Percent'] , color = 'g')
b.set_title('Distribution plot of Percent' , color = 'r' , fontsize = 18)


## === cell 16
import plotly.express as px
data=px.bar(x=list(train_x['Weeks'].value_counts().keys()), y=list(train_x['Weeks'].value_counts().values) )
data


## === cell 17
fig = px.line(train_x, 'Weeks', 'FVC', line_group='Patient', color='Sex',
             title='Pulmonary Condition Progression by Sex')
fig.update_traces(mode='lines + markers')


## === cell 18
fig = px.line(train_x, 'Weeks', 'FVC', line_group='Patient', color='SmokingStatus',
             title='Pulmonary Condition Progression by Smoking Status')
fig.update_traces(mode='lines+markers')


## === cell 19
print('The Number of Unique Patients in training data are : {}'.format(len(train_x['Patient'].unique()), "\n"))


## === cell 20
data_path = '../input/osic-pulmonary-fibrosis-progression/train/'

output_path = '../input/output/'
train_image_files = sorted(glob.glob(os.path.join(data_path, '*','*.dcm')))
patients = os.listdir(data_path)
patients.sort()

print('Some sample Patient ID''s :', len(train_image_files))
print("\n".join(train_image_files[:5]))


## === cell 21
def load_scan(path):
    """
    Loads scans from a folder and into a list.
    
    Parameters: path (Folder path)
    
    Returns: slices (List of slices)
    """
    
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
def get_pixels_hu(scans):
    """
    Converts raw images to Hounsfield Units (HU).
    
    Parameters: scans (Raw images)
    
    Returns: image (NumPy array)
    """
    
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    image[image == -2000] = 0
    
    
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


## === cell 22
test_patient_scans = load_scan(data_path + patients[2])
test_patient_images = get_pixels_hu(test_patient_scans)


for imgs in range(len(test_patient_images[0:5])):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))
    ax1.imshow(test_patient_images[imgs], cmap=plt.cm.bone)
    ax1.set_title("Original Slice")
    
    ax2.imshow(test_patient_images[imgs], cmap=plt.cm.bone)
    ax2.set_title("Original Slice")
    
    ax3.imshow(test_patient_images[imgs], cmap=plt.cm.bone)
    ax3.set_title("Original Slice")
    plt.show()


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1996120606.py in <cell line: 0>()
----> 1 test_patient_scans = load_scan(data_path + patients[2])
      2 test_patient_images = get_pixels_hu(test_patient_scans)
      3 
      4 #We'll be taking a random slice to perform segmentation:
      5 

/tmp/ipykernel_11/1435689112.py in load_scan(path)
      8     """
      9 
---> 10     slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
     11     slices.sort(key = lambda x: int(x.InstanceNumber))
     12 

/tmp/ipykernel_11/1435689112.py in <listcomp>(.0)
      8     """
      9 
---> 10     slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
     11     slices.sort(key = lambda x: int(x.InstanceNumber))
     12 

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 23
def set_lungwin(img, hu=[-1200., 600.]):
    lungwin = np.array(hu)
    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


scans = load_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')
scan_array = set_lungwin(get_pixels_hu(scans))

imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.00001)
Image(filename="/tmp/gif.gif", format='png')


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/560446065.py in <cell line: 0>()
      8 
      9 
---> 10 scans = load_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')
     11 scan_array = set_lungwin(get_pixels_hu(scans))
     12 

/tmp/ipykernel_11/1435689112.py in load_scan(path)
      8     """
      9 
---> 10     slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
     11     slices.sort(key = lambda x: int(x.InstanceNumber))
     12 

/tmp/ipykernel_11/1435689112.py in <listcomp>(.0)
      8     """
      9 
---> 10     slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
     11     slices.sort(key = lambda x: int(x.InstanceNumber))
     12 

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 24
train_x.shape 


## === cell 25
test_x.shape 


## === cell 26
def eval_metric(FVC,FVC_Pred,sigma):
    n = len(sigma)
    a=np.empty(n)
    a.fill(70)
    sigma_clipped = np.maximum(sigma,a) 
    delta = np.minimum(np.abs(FVC,FVC_Pred),1000)
    eval_metric = -np.sqrt(2)*delta/sigma_clipped - np.log(np.sqrt(2)*sigma_clipped)
    return eval_metric


## === cell 27
sub_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

print(f"The sample submission contains: {sub_df.shape[0]} rows and {sub_df.shape[1]} columns.")


## === cell 28
sub_df[['Patient','Weeks']] = sub_df.Patient_Week.str.split("_",expand = True)
sub_df =  sub_df[['Patient','Weeks','Confidence', 'Patient_Week']]


## === cell 29
sub_df = sub_df.merge(test_x.drop('Weeks', axis = 1), on = "Patient")


## === cell 30
train_x['Source'] = 'train'
sub_df['Source'] = 'test'

data_df = train_x.append([sub_df])
data_df.reset_index(inplace = True)
data_df.head()


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/414196147.py in <cell line: 0>()
      3 sub_df['Source'] = 'test'
      4 
----> 5 data_df = train_x.append([sub_df])
      6 data_df.reset_index(inplace = True)
      7 data_df.head()

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 31
def get_baseline_week(df):
    _df = df.copy()
    _df['Weeks'] = _df['Weeks'].astype(int)
    _df.loc[_df.Source == 'test','min_week'] = np.nan
    _df["min_week"] = _df.groupby('Patient')['Weeks'].transform('min')
    _df['baselined_week'] = _df['Weeks'] - _df['min_week']
    
    return _df   


## === cell 32
data_df = get_baseline_week(data_df)
data_df.head()


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3396628081.py in <cell line: 0>()
----> 1 data_df = get_baseline_week(data_df)
      2 data_df.head()

NameError: name 'data_df' is not defined

## === cell 33
def get_baseline_FVC_old(df):
    _df = df.copy()
    baseline = _df.loc[_df.Weeks == _df.min_week]
    baseline = baseline[['Patient','FVC']].copy()
    baseline.columns = ['Patient','base_FVC']      
    
    for idx in _df.index:
        patient_id = _df.at[idx,'Patient']
        _df.at[idx,'base_FVC'] = baseline.loc[baseline.Patient == patient_id, 'base_FVC'].iloc[0]
    _df.drop(['min_week'], axis = 1)
    
    return _df


## === cell 34
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


## === cell 35
def old_baseline_FVC():
    return get_baseline_FVC_old(data_df)
    pass

def new_baseline_FVC():
    return get_baseline_FVC(data_df)
    

duration_old = timeit(old_baseline_FVC, number = 3)
duration_new = timeit(new_baseline_FVC, number = 3)

print(f"Taking the old, non-vectorized version took {duration_old / 3:.2f} sec, while the vectorized version only took {duration_new / 3:.3f} sec. That's {duration_old/duration_new:.0f} times faster!" )


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1260233406.py in <cell line: 0>()
      7 
      8 
----> 9 duration_old = timeit(old_baseline_FVC, number = 3)
     10 duration_new = timeit(new_baseline_FVC, number = 3)
     11 

/usr/lib/python3.11/timeit.py in timeit(stmt, setup, timer, number, globals)
    235            number=default_number, globals=None):
    236     """Convenience function to create Timer object and call timeit method."""
--> 237     return Timer(stmt, setup, timer, globals).timeit(number)
    238 
    239 

/usr/lib/python3.11/timeit.py in timeit(self, number)
    178         gc.disable()
    179         try:
--> 180             timing = self.inner(it, self.timer)
    181         finally:
    182             if gcold:

/usr/lib/python3.11/timeit.py in inner(_it, _timer, _stmt)

/tmp/ipykernel_11/1260233406.py in old_baseline_FVC()
      1 def old_baseline_FVC():
----> 2     return get_baseline_FVC_old(data_df)
      3     pass
      4 
      5 def new_baseline_FVC():

NameError: name 'data_df' is not defined

## === cell 36
data_df = get_baseline_FVC(data_df)
data_df.head()


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/117201962.py in <cell line: 0>()
----> 1 data_df = get_baseline_FVC(data_df)
      2 data_df.head()

NameError: name 'data_df' is not defined

## === cell 37
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

no_transform_attribs = ['Patient', 'Weeks', 'min_week']
num_attribs = ['FVC', 'Percent', 'Age', 'baselined_week', 'base_FVC']
cat_attribs = ['Sex', 'SmokingStatus']


## === cell 38
def own_MinMaxColumnScaler(df, columns):
    """Adds columns with scaled numeric values to range [0, 1]
    using the formula X_scld = (X - X.min) / (X.max - X.min)"""
    for col in columns:
        new_col_name = col + '_scld'
        col_min = df[col].min()
        col_max = df[col].max()        
        df[new_col_name] = (df[col] - col_min) / ( col_max - col_min )


## === cell 39
def own_OneHotColumnCreator(df, columns):
    """OneHot Encodes categorical features. Adds a column for each unique value per column"""
    for col in cat_attribs:
        for value in df[col].unique():
            df[value] = (df[col] == value).astype(int)


## === cell 40
own_MinMaxColumnScaler(data_df, num_attribs)
own_OneHotColumnCreator(data_df, cat_attribs)

data_df[data_df.Source != "train"].head()


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/554184512.py in <cell line: 0>()
      1 ## APPLY DEFINED TRANSFORMATIONS
----> 2 own_MinMaxColumnScaler(data_df, num_attribs)
      3 own_OneHotColumnCreator(data_df, cat_attribs)
      4 
      5 data_df[data_df.Source != "train"].head()

NameError: name 'data_df' is not defined

## === cell 41
train_df = data_df.loc[data_df.Source == 'train']
sub = data_df.loc[data_df.Source == 'test']


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/547538857.py in <cell line: 0>()
      1 # get back original data split
----> 2 train_df = data_df.loc[data_df.Source == 'train']
      3 sub = data_df.loc[data_df.Source == 'test']

NameError: name 'data_df' is not defined

## === cell 42

features_list = ['baselined_week_scld', 'Percent_scld', 'Age_scld', 'base_FVC_scld', 'Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes']

EPOCHS = 1000
BATCH_SIZE = 128


_lambda = 0.8 # 0.8 default


ADAM = tf.keras.optimizers.Adam(lr = 0.1,
                                beta_1 = 0.9, 
                                beta_2 = 0.999,
                                decay = 0.01)
SGD = tf.keras.optimizers.SGD()

optimizer = ADAM



lr_start   = 0.0001
lr_max     = 0.0001 * BATCH_SIZE # higher batch size --> higher lr
lr_min     = 0.00001
lr_ramp_ep = EPOCHS * 0.3
lr_sus_ep  = 0
lr_decay   = 0.992

def test_the_scheduler(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

rng = [i for i in range(EPOCHS)]
y = [test_the_scheduler(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/414312651.py in <cell line: 0>()
     14 
     15 ## Optimizers
---> 16 ADAM = tf.keras.optimizers.Adam(lr = 0.1,
     17                                 beta_1 = 0.9,
     18                                 beta_2 = 0.999,

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     60         **kwargs,
     61     ):
---> 62         super().__init__(
     63             learning_rate=learning_rate,
     64             name=name,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py in __init__(self, *args, **kwargs)
     19 class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
     20     def __init__(self, *args, **kwargs):
---> 21         super().__init__(*args, **kwargs)
     22         self._distribution_strategy = tf.distribute.get_strategy()
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in __init__(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     88             )
     89         if kwargs:
---> 90             raise ValueError(f"Argument(s) not recognized: {kwargs}")
     91 
     92         if name is None:

ValueError: Argument(s) not recognized: {'lr': 0.1}

## === cell 43
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


## === cell 44
def get_model():
    "Creates and returns a model"
    inp = Layers.Input((len(features_list),), name = "Patient")
    x = Layers.Dense(128, activation = "relu", name = "d1")(inp)
    x = Layers.Dropout(0.25)(x)
    x = Layers.Dense(128, activation = "relu", name = "d2")(x)
    x = Layers.Dropout(0.2)(x)
    p1 = Layers.Dense(3, activation = "relu", name = "p1")(x)
    p2 = Layers.Dense(3, activation = "relu", name = "p2")(x)
    preds = Layers.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis = 1), 
                     name = "preds")([p1, p2])
    
    model = Models.Model(inp, preds, name = "NeuralNet")
    model.compile(loss = mloss(_lambda), optimizer = optimizer, metrics = [score])
    
    return model


## === cell 45
neuralNet = get_model()
neuralNet.summary()


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/481959714.py in <cell line: 0>()
      1 # create neural Network
----> 2 neuralNet = get_model()
      3 neuralNet.summary()

/tmp/ipykernel_11/814802065.py in get_model()
     14 
     15     model = Models.Model(inp, preds, name = "NeuralNet")
---> 16     model.compile(loss = mloss(_lambda), optimizer = optimizer, metrics = [score])
     17 
     18     return model

NameError: name 'optimizer' is not defined

## === cell 46

y = train_df['FVC'].values.astype(float)


X_train = train_df[features_list].values
X_test = sub[features_list].values

train_preds = np.zeros((X_train.shape[0], 3))
test_preds = np.zeros((X_test.shape[0], 3))


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1185540673.py in <cell line: 0>()
      2 
      3 # get target value
----> 4 y = train_df['FVC'].values.astype(float)
      5 
      6 

NameError: name 'train_df' is not defined

## === cell 47
"""K-fold variant with non-overlapping groups.
The same group will not appear in two different folds: in this case we dont want to have overlapping patientIDs in TRAIN and VAL-Data!
The folds are approximately balanced in the sense that the number of distinct groups is approximately the same in each fold."""

NFOLDS = 10
gkf = GroupKFold(n_splits = NFOLDS)
groups = train_df['Patient'].values

count = 0
for train_idx, val_idx in gkf.split(X_train, y, groups = groups):
    count += 1
    print(f"FOLD {count}:")
    
    net = get_model()
    net.fit(X_train[train_idx], y[train_idx], batch_size = BATCH_SIZE, epochs = EPOCHS, 
            validation_data = (X_train[val_idx], y[val_idx]), verbose = 0) 
    
    print("Train:", net.evaluate(X_train[train_idx], y[train_idx], verbose = 0, batch_size = BATCH_SIZE))
    print("Val:", net.evaluate(X_train[val_idx], y[val_idx], verbose = 0, batch_size = BATCH_SIZE))
    
    train_preds[val_idx] = net.predict(X_train[val_idx], batch_size = BATCH_SIZE, verbose = 0)
    
    print("Predicting Test...")
    test_preds += net.predict(X_test, batch_size = BATCH_SIZE, verbose = 0) / NFOLDS


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3278835897.py in <cell line: 0>()
      7 gkf = GroupKFold(n_splits = NFOLDS)
      8 # extract Patient IDs for ensuring
----> 9 groups = train_df['Patient'].values
     10 
     11 count = 0

NameError: name 'train_df' is not defined

## === cell 48
sigma_opt = mean_absolute_error(y, train_preds[:,1])
sigma_uncertain = train_preds[:,2] - train_preds[:,0]
sigma_mean = np.mean(sigma_uncertain)
print(sigma_opt, sigma_mean)


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2809666322.py in <cell line: 0>()
      1 ## FIND OPTIMIZED STANDARD-DEVIATION
----> 2 sigma_opt = mean_absolute_error(y, train_preds[:,1])
      3 sigma_uncertain = train_preds[:,2] - train_preds[:,0]
      4 sigma_mean = np.mean(sigma_uncertain)
      5 print(sigma_opt, sigma_mean)

NameError: name 'y' is not defined

## === cell 49
sub.head()


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1894231914.py in <cell line: 0>()
----> 1 sub.head()

NameError: name 'sub' is not defined

## === cell 50
sub['FVC1'] = test_preds[:, 1]
sub['Confidence1'] = test_preds[:,2] - test_preds[:,0]

submission = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
submission.loc[~submission.FVC1.isnull()].head(10)


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/66350507.py in <cell line: 0>()
      1 ## PREPARE SUBMISSION FILE WITH OUR PREDICTIONS
----> 2 sub['FVC1'] = test_preds[:, 1]
      3 sub['Confidence1'] = test_preds[:,2] - test_preds[:,0]
      4 
      5 # get rid of unused data and show some non-empty data

NameError: name 'test_preds' is not defined

## === cell 51
submission.head()


## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4096176616.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined

## === cell 52
submission.describe().T


## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1258828452.py in <cell line: 0>()
----> 1 submission.describe().T

NameError: name 'submission' is not defined

## === cell 53
org_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(org_test)):
    submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'FVC'] = org_test.FVC[i]
    submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'Confidence'] = 70


## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2771494234.py in <cell line: 0>()
      2 
      3 for i in range(len(org_test)):
----> 4     submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'FVC'] = org_test.FVC[i]
      5     submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'Confidence'] = 70

NameError: name 'submission' is not defined

## === cell 54
submission[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index = False)


## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4006308278.py in <cell line: 0>()
----> 1 submission[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index = False)

NameError: name 'submission' is not defined

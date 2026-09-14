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
scikit-image==0.25.2
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

-6.866

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
import pydicom
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from skimage import morphology
from skimage import measure
from skimage.transform import resize
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.patches as patches
import tensorflow.keras.backend as k


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')


## === cell 2
train_csv


## === cell 3
base_week=train_csv.groupby('Patient')['Weeks'].min()
base_week_list=[]
for i in range(len(train_csv)):
    base_week_list.append(base_week[train_csv.iloc[i,0]])
train_csv['base_week']=base_week_list


base_week=train_csv.groupby('Patient')['Weeks'].min()
count_from_base_week=[]
for i in range(len(train_csv)):
    count_from_base_week.append(train_csv.iloc[i,1]-base_week[train_csv.iloc[i,0]])
train_csv['count_from_base_week']=count_from_base_week

confidence=np.zeros(train_csv.shape[0])
train_csv['confidence']=confidence

base_fvc_dict={}
for id in train_csv['Patient'].unique():
    base_fvc_dict[id]=np.array(train_csv[(train_csv['Patient']==id) & (train_csv['Weeks']==base_week[id])]['FVC'])[0]
base_fvc=[]
for i in range(len(train_csv)):
    base_fvc.append(base_fvc_dict[train_csv.iloc[i,0]])
train_csv['base_fvc']=base_fvc

base_fev1_dict={}
for id in train_csv['Patient'].unique():
    A=train_csv[train_csv['Patient']==id]["base_fvc"].unique()[0]
    B=train_csv[train_csv['Patient']==id]["Age"].unique()[0]
    if train_csv[train_csv['Patient']==id]["Sex"].unique()[0]=='Male':
        base_fev1_dict[id]=0.77*A+0.32+0.0069*B
    else:
        base_fev1_dict[id]=0.77*A+0.28+0.0052*B
        
base_fev1=[]
for i in range(len(train_csv)):
    base_fev1.append(base_fev1_dict[train_csv.iloc[i,0]])
train_csv['base_fev1']=base_fev1

base_week_percent_dict={}
for id in train_csv['Patient'].unique():
    base_week_percent_dict[id]=np.array(train_csv[(train_csv['Patient']==id) & (train_csv['Weeks']==base_week[id])]['Percent'])[0]
    
base_week_percent=[]
for i in range(len(train_csv)):
    base_week_percent.append(base_week_percent_dict[train_csv.iloc[i,0]])
train_csv['base_week_percent']=base_week_percent

train_csv['base fev1/base fvc']=train_csv['base_fev1']/train_csv['base_fvc']

train_csv['base_height']=(train_csv['base_fvc']+9030)/77.0


base_weight_dict={}
for id in train_csv['Patient'].unique():
    FVC=train_csv[train_csv['Patient']==id]["base_fvc"].unique()[0]
    A=train_csv[train_csv['Patient']==id]["Age"].unique()[0]
    H=train_csv[train_csv['Patient']==id]["base_height"].unique()[0]
    if train_csv[train_csv['Patient']==id]["Sex"].unique()[0]=='Male':
        base_weight_dict[id]=(FVC+5458-49*H+8*A)/12.0
    else:
        base_weight_dict[id]=(FVC+3863-37*H+6*A)/14.0
base_weight=[]
for i in range(len(train_csv)):
    base_weight.append(base_weight_dict[train_csv.iloc[i,0]])
train_csv['base_weight']=base_weight

train_csv['base_bmi']=train_csv['base_weight']/((train_csv['base_height']/100)**2)


## === cell 4
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()#sex
train_csv.iloc[:,5]=lb.fit_transform(train_csv.iloc[:,5])
lb2=LabelEncoder()#ss
train_csv.iloc[:,6]=lb2.fit_transform(train_csv.iloc[:,6])


## === cell 5
from sklearn.preprocessing import OneHotEncoder
oh1=OneHotEncoder(handle_unknown='ignore')
smoke_cat=pd.DataFrame(oh1.fit_transform(train_csv[['SmokingStatus']]).toarray(),columns=['smoking cat 0','smoking cat 1','smoking cat 2'])
train_csv=pd.concat([train_csv,smoke_cat],axis=1)


## === cell 6
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_scaled=pd.DataFrame(sc.fit_transform(train_csv[['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi']]),columns=['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi'])
train_scaled['Sex']=train_csv['Sex']
train_scaled['smoking cat 0']=train_csv['smoking cat 0']
train_scaled['smoking cat 1']=train_csv['smoking cat 1']


## === cell 7
train_scaled


## === cell 8
train_csv


## === cell 9
sub=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
test_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


## === cell 10
test_week=[]
patient_id=[]
for i in range(len(sub)):
    test_week.append(int(sub.iloc[i,0].split('_')[-1]))
    patient_id.append(sub.iloc[i,0].split('_')[0])
sub['Patient']=patient_id
sub['Weeks']=test_week
sub.drop(['FVC','Confidence'],axis=1,inplace=True)

base_fvc=test_csv.groupby('Patient')['FVC'].min()
fvc=[]
for i in range(len(sub)):
    fvc.append(base_fvc[sub.iloc[i,1]])
sub['base_fvc']=fvc

base_fev1_dict_test={}
for id in sub['Patient'].unique():
    A=sub[sub['Patient']==id]["base_fvc"].unique()[0]
    B=test_csv[test_csv['Patient']==id]["Age"].unique()[0]
    if test_csv[test_csv['Patient']==id]["Sex"].unique()[0]=='Male':
        base_fev1_dict_test[id]=0.77*A+0.32+0.0069*B
    else:
        base_fev1_dict_test[id]=0.77*A+0.28+0.0052*B
        
base_fev1_test=[]
for i in range(len(sub)):
    base_fev1_test.append(base_fev1_dict_test[sub.iloc[i,1]])
sub['base_fev1']=base_fev1_test


sub['base_height']=(sub['base_fvc']+9030)/77.0



base_weight_dict_test={}
for id in sub['Patient'].unique():
    FVC=sub[sub['Patient']==id]["base_fvc"].unique()[0]
    A=test_csv[test_csv['Patient']==id]["Age"].unique()[0]
    H=sub[sub['Patient']==id]["base_height"].unique()[0]
    if test_csv[test_csv['Patient']==id]["Sex"].unique()[0]=='Male':
        base_weight_dict_test[id]=(FVC+5458-49*H+8*A)/12.0
    else:
        base_weight_dict_test[id]=(FVC+3863-37*H+6*A)/14.0
base_weight_test=[]
for i in range(len(sub)):
    base_weight_test.append(base_weight_dict_test[sub.iloc[i,1]])
sub['base_weight']=base_weight_test


test_csv.iloc[:,5]=lb.transform(test_csv.iloc[:,5])
test_csv.iloc[:,6]=lb2.transform(test_csv.iloc[:,6])

percent_dict={}
for id in test_csv['Patient'].unique():
    percent_dict[id]=float(test_csv[test_csv['Patient']==id]['Percent'])
    
sex_dict={}
for id in test_csv['Patient'].unique():
    sex_dict[id]=int(test_csv[test_csv['Patient']==id]['Sex'])

age_dict={}
for id in test_csv['Patient'].unique():
    age_dict[id]=int(test_csv[test_csv['Patient']==id]['Age'])
    
ss_dict={}
for id in test_csv['Patient'].unique():
    ss_dict[id]=int(test_csv[test_csv['Patient']==id]['SmokingStatus'])

percent=[]
sex=[]
age=[]
ss=[]
for i in range(len(sub)):
    percent.append(percent_dict[sub.iloc[i,1]])
    sex.append(sex_dict[sub.iloc[i,1]])
    age.append(age_dict[sub.iloc[i,1]])
    ss.append(ss_dict[sub.iloc[i,1]])    
sub['base_week_percent']=percent
sub['Age']=age
sub['Sex']=sex
sub['SmokingStatus']=ss


base_week_test=test_csv.groupby('Patient')['Weeks'].min()
count_from_base_week_test=[]
base_week=[]
for i in range(len(sub)):
    count_from_base_week_test.append(sub.iloc[i,2]-base_week_test[sub.iloc[i,1]])
    base_week.append(base_week_test[sub.iloc[i,1]])
sub['count_from_base_week']=count_from_base_week_test
sub['base_week']=base_week


sub['base fev1/base fvc']=sub['base_fev1']/sub['base_fvc']

sub['base_bmi']=sub['base_weight']/((sub['base_height']/100.0)**2)


## === cell 11
smoke_cat_test=pd.DataFrame(oh1.transform(sub[['SmokingStatus']]).toarray(),columns=['smoking cat 0','smoking cat 1','smoking cat 2'])
sub=pd.concat([sub,smoke_cat_test],axis=1)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/100729771.py in <cell line: 0>()
----> 1 smoke_cat_test=pd.DataFrame(oh1.transform(sub[['SmokingStatus']]).toarray(),columns=['smoking cat 0','smoking cat 1','smoking cat 2'])
      2 sub=pd.concat([sub,smoke_cat_test],axis=1)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_encoders.py in transform(self, X)
    915             "infrequent_if_exist",
    916         }
--> 917         X_int, X_mask = self._transform(
    918             X,
    919             handle_unknown=self.handle_unknown,

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_encoders.py in _transform(self, X, handle_unknown, force_all_finite, warn_on_unknown)
    164         for i in range(n_features):
    165             Xi = X_list[i]
--> 166             diff, valid_mask = _check_unknown(Xi, self.categories_[i], return_mask=True)
    167 
    168             if not np.all(valid_mask):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _check_unknown(values, known_values, return_mask)
    301 
    302         # check for nans in the known_values
--> 303         if np.isnan(known_values).any():
    304             diff_is_nan = np.isnan(diff)
    305             if diff_is_nan.any():

TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

## === cell 12
sub_scaled=pd.DataFrame(sc.transform(sub[['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi']]),columns=['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi'])
sub_scaled['Sex']=sub['Sex']
sub_scaled['smoking cat 0']=sub['smoking cat 0']
sub_scaled['smoking cat 1']=sub['smoking cat 1']


## --- ERROR in cell 12, traceback:
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

KeyError: 'smoking cat 0'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3984344276.py in <cell line: 0>()
      1 sub_scaled=pd.DataFrame(sc.transform(sub[['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi']]),columns=['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi'])
      2 sub_scaled['Sex']=sub['Sex']
----> 3 sub_scaled['smoking cat 0']=sub['smoking cat 0']
      4 sub_scaled['smoking cat 1']=sub['smoking cat 1']

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

KeyError: 'smoking cat 0'

## === cell 13
sub_scaled


## === cell 14
x=np.array(train_csv[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_week_percent','base_height','smoking cat 0','smoking cat 1']])
y=np.array(train_csv[['FVC','confidence']])

from sklearn.model_selection import train_test_split
xtrain,xvalid,ytrain,yvalid=train_test_split(x,y,test_size=0.2)


## === cell 16
def metric(actual_fvc, predicted_fvc, confidence, return_values = False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    else:
        return np.mean(metric)
    
    


def model_loss(ytrue,ypred):   # this loss penalises both prediction and confidence value
    eps=1.0
    
    fvc_pred=ypred[:,0]
    sigmas=ypred[:,1]+eps     # so as to avoid log(0) . these are predicted connfidences
    
    ans=tf.math.log(sigmas)
    ans=ans+((ytrue[:,0]-fvc_pred)**2)/(2*sigmas**2)
    
    return tf.reduce_mean(ans)


## === cell 17
lr_scheduler=tf.keras.callbacks.ReduceLROnPlateau(factor=0.2,monitor='val_loss',mode='min',patience=40,verbose=1)
class best_weights(tf.keras.callbacks.Callback):
    def __init__(self):
        self.metric_op=-30.0
        self.weights_op=None
        self.epoch_op=-1
    def on_epoch_end(self,epoch,logs={}):
        if logs['val_metric']>=self.metric_op:
            self.metric_op=logs['val_metric']
            self.epoch_op=epoch
            self.weights_op=self.model.get_weights()
    def on_train_end(self,logs={}):
        self.model.set_weights(self.weights_op)
        print('BEST_EPOCH = {}   BEST_SCORE_ON_VALID_SET = {}'.format(self.epoch_op+1,self.metric_op))
        
        

class metrics_call(tf.keras.callbacks.Callback):
    def __init__(self,mertic,xtrain,ytrain,xvalid,yvalid):
        self.metric=metric
        self.xtrain=xtrain
        self.ytrain=ytrain
        self.xvalid=xvalid
        self.yvalid=yvalid
        
    def on_epoch_end(self,epoch,logs={}):
        train_preds=self.model.predict(self.xtrain)
        val_preds=self.model.predict(self.xvalid)
        logs['val_metric']=self.metric(self.yvalid[:,0],val_preds[:,0],val_preds[:,1])
        
        

def run_model(xtrain,ytrain,xvalid,yvalid,epoch=50):
    input=tf.keras.layers.Input(shape=xtrain.shape[1:])
    noisy=tf.keras.layers.GaussianNoise(0.3)(input)
    
    d1=tf.keras.layers.Dense(128,activation='relu')(noisy)
    d2=tf.keras.layers.Dense(128,activation='relu')(d1)
    d3=tf.keras.layers.Dense(128,activation='relu')(d2)
    mean_out1=tf.keras.layers.Dense(1)(d3)
    std_den1=tf.keras.layers.Dense(1)(d3)
    
    d4=tf.keras.layers.Dense(128,activation='relu')(noisy)
    d5=tf.keras.layers.Dense(128,activation='relu')(d4)
    d6=tf.keras.layers.Dense(128,activation='relu')(d5)
    mean_out2=tf.keras.layers.Dense(1)(d6)
    std_den2=tf.keras.layers.Dense(1)(d6)
    
    d7=tf.keras.layers.Dense(128,activation='relu')(noisy)
    d8=tf.keras.layers.Dense(128,activation='relu')(d7)
    d9=tf.keras.layers.Dense(128,activation='relu')(d8)
    mean_out3=tf.keras.layers.Dense(1)(d9)
    std_den3=tf.keras.layers.Dense(1)(d9)
    
    mean_combine=tf.keras.layers.Concatenate()([mean_out1,mean_out2,mean_out3])
    std_combine=tf.keras.layers.Concatenate()([std_den1,std_den2,std_den3])
    mean_final=tf.keras.layers.Dense(1)(mean_combine)
    std_final_den=tf.keras.layers.Dense(1)(std_combine)
    std_final=tf.keras.layers.Lambda(lambda x: tf.abs(x))(std_final_den)
    output=tf.keras.layers.Concatenate()([mean_final,std_final])
    model=tf.keras.models.Model(inputs=input,outputs=output)
    
    model.compile(loss=lambda ytrue,ypred: model_loss(ytrue,ypred),optimizer=tf.keras.optimizers.Adam(lr=0.001))
    print(model.summary())
    history=model.fit(xtrain,ytrain,epochs=epoch,batch_size=256,validation_data=(xvalid,yvalid),verbose=0,callbacks=[lr_scheduler,metrics_call(metric,xtrain,ytrain,xvalid,yvalid),best_weights()])
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.ylim(-10,10)
    plt.grid(True)

    return model


## === cell 18
model=run_model(xtrain,ytrain,xvalid,yvalid,epoch=1000)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/365991304.py in <cell line: 0>()
----> 1 model=run_model(xtrain,ytrain,xvalid,yvalid,epoch=1000)

/tmp/ipykernel_11/3183852698.py in run_model(xtrain, ytrain, xvalid, yvalid, epoch)
     63     model=tf.keras.models.Model(inputs=input,outputs=output)
     64 
---> 65     model.compile(loss=lambda ytrue,ypred: model_loss(ytrue,ypred),optimizer=tf.keras.optimizers.Adam(lr=0.001))
     66     print(model.summary())
     67     history=model.fit(xtrain,ytrain,epochs=epoch,batch_size=256,validation_data=(xvalid,yvalid),verbose=0,callbacks=[lr_scheduler,metrics_call(metric,xtrain,ytrain,xvalid,yvalid),best_weights()])

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

ValueError: Argument(s) not recognized: {'lr': 0.001}

## === cell 19
xtest=np.array(sub[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_week_percent','base_height','smoking cat 0','smoking cat 1']])


yans=model.predict(xtest)

sub['FVC']=yans[:,0]
sub['Confidence']=np.abs(yans[:,1])
sub.drop(['Patient','Weeks','Age','Sex','SmokingStatus','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi','smoking cat 0','smoking cat 1','smoking cat 2'],axis=1,inplace=True)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2783252037.py in <cell line: 0>()
      1 #xtest=np.array(sub[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi','smoking cat 0','smoking cat 1']])
----> 2 xtest=np.array(sub[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_week_percent','base_height','smoking cat 0','smoking cat 1']])
      3 
      4 #xtest=np.array(sub_scaled)
      5 

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

KeyError: "['smoking cat 0', 'smoking cat 1'] not in index"

## === cell 20
sub


## === cell 21
sub.to_csv('submission.csv',index=False)


## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame must have a 'FVC' column.

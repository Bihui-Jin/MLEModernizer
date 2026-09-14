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

No external packages required in the script and installed.

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

-7.048740974654807

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import random
import pathlib
import pickle
import time

import pydicom
import matplotlib.pyplot as plt

import tensorflow.keras.backend
from tensorflow import keras as K
from tensorflow.keras import layers as L
from skimage.transform import resize
from scipy.ndimage import zoom
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation

from math import ceil

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
raw_test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')
X_prediction = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

## === cell 2
TEST_PATH = '/kaggle/input/osic-pulmonary-fibrosis-progression/test'

DESIRED_SIZE = (30,256,256)
BATCH_SIZE = 256

clip_bounds = (-1000, 200)
pre_calculated_mean = 0.02865046213070556

## === cell 5
X_prediction['Patient'] = X_prediction['Patient_Week'].str.extract(r'(.*)_.*')
X_prediction['Weeks'] = X_prediction['Patient_Week'].str.extract(r'.*_(.*)').astype(int)
X_prediction = X_prediction[['Patient', 'Weeks', 'Patient_Week']]
rename_cols = {'Weeks_y':'Min_week', 'Weeks_x': 'Weeks', 'FVC':'Base_FVC'}
X_prediction = X_prediction.merge(raw_test, how='left', left_on='Patient', right_on='Patient').rename(columns=rename_cols)[['Patient', 'Min_week', 'Base_FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'Weeks', 'Patient_Week']].reset_index(drop=True)

## === cell 6
X_prediction['Base_week'] = X_prediction['Weeks'] - X_prediction['Min_week']

## === cell 7
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index='', name='', **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        new_columns = []
        for j in range(len(categories)):
            new_columns.append('{}_{}'.format(name, categories[j]))
        return new_columns

from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

def standardisation(x, u, s):
    return (x-u)/s

def normalization(x, ma, mi):
    return (x-mi)/(ma-mi)

class data_preparation():
    def __init__(self, bool_normalization=True,bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()
        self.standardisation = bool_standard
        self.normalization = bool_normalization
        
        
    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)
        
        try:
            data['Sex'] = self.enc_sex.transform(data['Sex'].values)
            data['SmokingStatus'] = self.enc_smok.transform(data['SmokingStatus'].values)
            data = pd.concat([data.drop(columns=['SmokingStatus']), self.onehotenc_smok.transform(data['SmokingStatus'].values.reshape(-1,1), categories=self.enc_smok.classes_, name='', index=data.index).astype(int)], axis=1)
            
            if self.standardisation:
                data['Base_week'] = standardisation(data['Base_week'],self.base_week_mean,self.base_week_std)
                data['Base_FVC'] = standardisation(data['Base_FVC'],self.base_fvc_mean,self.base_fvc_std)
                data['Base_percent'] = standardisation(data['Base_percent'],self.base_percent_mean,self.base_percent_std)
                data['Age'] = standardisation(data['Age'],self.age_mean,self.age_std)
                data['Weeks'] = standardisation(data['Weeks'],self.weeks_mean,self.weeks_std)
            
            if self.normalization:
                data['Base_week'] = normalization(data['Base_week'],self.base_week_max,self.base_week_min)
                data['Base_FVC'] = normalization(data['Base_FVC'],self.base_fvc_max,self.base_fvc_min)
                data['Percent'] = normalization(data['Percent'],self.base_percent_max,self.base_percent_min)
                data['Age'] = normalization(data['Age'],self.age_max,self.age_min)
                data['Weeks'] = normalization(data['Weeks'],self.weeks_max,self.weeks_min)
                data['Min_week'] = normalization(data['Min_week'],self.base_week_max,self.base_week_min)

        except NotFittedError:
            data['Sex'] = self.enc_sex.fit_transform(data['Sex'].values)
            data['SmokingStatus'] = self.enc_smok.fit_transform(data['SmokingStatus'].values)
            data = pd.concat([data.drop(columns=['SmokingStatus']), self.onehotenc_smok.fit_transform(data['SmokingStatus'].values.reshape(-1,1), categories=self.enc_smok.classes_, name='', index=data.index).astype(int)], axis=1)
            
            if self.standardisation:
                self.base_week_mean = data['Base_week'].mean()
                self.base_week_std = data['Base_week'].std()
                data['Base_week'] = standardisation(data['Base_week'],self.base_week_mean,self.base_week_std)

                self.base_fvc_mean = data['Base_FVC'].mean()
                self.base_fvc_std = data['Base_FVC'].std()
                data['Base_FVC'] = standardisation(data['Base_FVC'],self.base_fvc_mean,self.base_fvc_std)

                self.base_percent_mean = data['Base_percent'].mean()
                self.base_percent_std = data['Base_percent'].std()
                data['Base_percent'] = standardisation(data['Base_percent'],self.base_percent_mean,self.base_percent_std)

                self.age_mean = data['Age'].mean()
                self.age_std = data['Age'].std()
                data['Age'] = standardisation(data['Age'],self.age_mean,self.age_std)

                self.weeks_mean = data['Weeks'].mean()
                self.weeks_std = data['Weeks'].std()
                data['Weeks'] = standardisation(data['Weeks'],self.weeks_mean,self.weeks_std)

                
            if self.normalization:
                self.base_week_min = data['Base_week'].min()
                self.base_week_max = data['Base_week'].max()
                data['Base_week'] = normalization(data['Base_week'],self.base_week_max,self.base_week_min)

                self.base_fvc_min = data['Base_FVC'].min()
                self.base_fvc_max = data['Base_FVC'].max()
                data['Base_FVC'] = normalization(data['Base_FVC'],self.base_fvc_max,self.base_fvc_min)

                self.base_percent_min = data['Percent'].min()
                self.base_percent_max = data['Percent'].max()
                data['Percent'] = normalization(data['Percent'],self.base_percent_max,self.base_percent_min)

                self.age_min = data['Age'].min()
                self.age_max = data['Age'].max()
                data['Age'] = normalization(data['Age'],self.age_max,self.age_min)

                self.weeks_min = data['Weeks'].min()
                self.weeks_max = data['Weeks'].max()
                data['Weeks'] = normalization(data['Weeks'],self.weeks_max,self.weeks_min)
                
                self.base_week_min = data['Min_week'].min()
                self.base_week_max = data['Min_week'].max()
                data['Min_week'] = normalization(data['Min_week'],self.base_week_max,self.base_week_min)

            
        return data

## === cell 8
pickefile = open('/kaggle/input/data-preparation-for-osic/data_prep', 'rb')
data_prep = pickle.load(pickefile)
pickefile.close()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2383209039.py in <cell line: 0>()
----> 1 pickefile = open('/kaggle/input/data-preparation-for-osic/data_prep', 'rb')
      2 data_prep = pickle.load(pickefile)
      3 pickefile.close()

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/data-preparation-for-osic/data_prep'

## === cell 9
X_prediction = data_prep(X_prediction).sort_values('Patient')

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1036992502.py in <cell line: 0>()
----> 1 X_prediction = data_prep(X_prediction).sort_values('Patient')

NameError: name 'data_prep' is not defined

## === cell 11
class ConvertToHU:
    def __call__(self, imgs, dicom):


        intercept = dicom.RescaleIntercept
        slope = dicom.RescaleSlope
        imgs = (np.array(imgs.to_list()) * slope + intercept).astype(np.int16)
        return imgs
convertohu = ConvertToHU()

## === cell 12
class Clip:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, image):
        image[image < self.min] = self.min
        image[image > self.max] = self.max
        return image
clip = Clip(clip_bounds)

## === cell 13
class MaskWatershed:
    def __init__(self, min_hu, iterations):
        self.min_hu = min_hu
        self.iterations = iterations

    def __call__(self, image, dicom):

        stack = []
        for slice_idx in range(image.shape[0]):
            sliced = image[slice_idx]
            stack.append(self.seperate_lungs(sliced, self.min_hu,
                                             self.iterations))

        return np.stack(stack)

    @staticmethod
    def seperate_lungs(image, min_hu, iterations):
        h, w = image.shape[0], image.shape[1]

        marker_internal, marker_external, marker_watershed = MaskWatershed.generate_markers(image)

        sobel_filtered_dx = ndimage.sobel(image, 1)
        sobel_filtered_dy = ndimage.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        sobel_gradient *= 255.0 / np.max(sobel_gradient)

        watershed = morphology.watershed(sobel_gradient, marker_watershed)

        outline = ndimage.morphological_gradient(watershed, size=(3,3))
        outline = outline.astype(bool)

        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 0]]

        blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)

        outline += ndimage.black_tophat(outline, structure=blackhat_struct)

        lungfilter = np.bitwise_or(marker_internal, outline)
        lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

        segmented = np.where(lungfilter == 1, image, min_hu * np.ones((h, w)))

        return segmented  #, lungfilter, outline, watershed, sobel_gradient

    @staticmethod
    def generate_markers(image, threshold=-400):
        h, w = image.shape[0], image.shape[1]

        marker_internal = image < threshold
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)

        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()

        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0

        marker_internal = marker_internal_labels > 0

        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a

        marker_watershed = np.zeros((h, w), dtype=np.int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128

        return marker_internal, marker_external, marker_watershed
maskwatershed = MaskWatershed(min_hu=min(clip_bounds), iterations=2)

## === cell 14
class Normalize:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, image):
        image = image.astype(np.float)
        image = (image - self.min) / (self.max - self.min)
        return image
     
class ZeroCenter:
    def __init__(self, pre_calculated_mean):
        self.pre_calculated_mean = pre_calculated_mean

    def __call__(self, image):
        return image - self.pre_calculated_mean

normalize = Normalize(bounds=clip_bounds)
zerocenter = ZeroCenter(pre_calculated_mean=pre_calculated_mean)

## === cell 15
def sort_function(x): #Get the files in the right order
    return int(x.split('.')[0])

def _read(path, patients=[], desired_size=(60,512,512)):
    X = np.empty(np.concatenate(([len(patients), 1], np.array(desired_size))))
    i=0
    for patient in patients:
        dicom = pydicom.dcmread(path+'/'+patient+'/'+os.listdir(path+'/'+patient)[0])
        list_patient_files = sorted([i for i in os.listdir(path+'/'+patient)], key=sort_function)
        df = pd.DataFrame(list_patient_files).apply(lambda x:path+'/'+patient+'/'+x)
        df = convertohu(df.iloc[:,0].apply(lambda x: pydicom.dcmread(x).pixel_array), dicom)
        df = zoom(df, np.array(DESIRED_SIZE)/np.array(df.shape), mode='nearest')
        X[i,0,:,:,:] = zerocenter(normalize(maskwatershed(clip(df), dicom)))
        i+=1
            
    return X

## === cell 17
class DataGenerator(K.utils.Sequence):
    
    def on_epoch_end(self):#Indices=np.arange(size of dataset)
        self.indices = np.arange(len(self.list_IDs))
    
    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))
    
    def __init__(self, train, list_IDs, batch_size=1, desired_size=(10,512,512), img_path=TEST_PATH, *args, **kwargs):
        self.train = train
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.desired_size = desired_size
        self.img_path = img_path
        self.on_epoch_end()
    
    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        patients = self.train.loc[list_IDs_temp, 'Patient'].unique()
        imgs = _read(self.img_path, patients=patients, desired_size=self.desired_size)
        return self.train.loc[list_IDs_temp, :].reset_index(drop=True), np.transpose(imgs, (0, 2, 3, 4, 1))

## === cell 18
pred_generator = DataGenerator(X_prediction, X_prediction.index, batch_size=BATCH_SIZE, desired_size=DESIRED_SIZE, img_path=TEST_PATH)

## === cell 20
model = K.models.load_model('/kaggle/input/3d-cnn-mlp/model_9',compile=False)
CNN = K.models.load_model('/kaggle/input/3d-cnn-mlp/CNN_9',compile=False)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2538937669.py in <cell line: 0>()
      1 # model = K.models.load_model('/kaggle/input/cnn-for-latent-features/model',custom_objects={'mloss':mloss(LAMBDA_LOSS)})
----> 2 model = K.models.load_model('/kaggle/input/3d-cnn-mlp/model_9',compile=False)
      3 CNN = K.models.load_model('/kaggle/input/3d-cnn-mlp/CNN_9',compile=False)

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    204         )
    205     else:
--> 206         raise ValueError(
    207             f"File format not supported: filepath={filepath}. "
    208             "Keras 3 only supports V3 `.keras` files and "

ValueError: File format not supported: filepath=/kaggle/input/3d-cnn-mlp/model_9. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(/kaggle/input/3d-cnn-mlp/model_9, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).

## === cell 21
SELECTED_COLUMNS = ['Weeks', 'Percent', 'Age', 'Sex', 'Min_week', 'Base_FVC','Base_week', '_Currently smokes', '_Ex-smoker', '_Never smoked']
y_prediction = np.array([[0,0,0]])
for X1, X2 in pred_generator:
    out_imgs = CNN(tf.convert_to_tensor(X2))
    X_imgs = tf.concat([tf.repeat(tf.reshape(out_imgs[i], (1,-1)), X1['Patient'].value_counts()[i], axis=0) for i in range(len(X1['Patient'].value_counts()))], axis=0)
    X_tabular = model.get_layer('dense_to_freeze2')(model.get_layer('dense_to_freeze1')(tf.convert_to_tensor(np.asarray(X1[SELECTED_COLUMNS]))))
    inp_mlp = model.get_layer('concatenate')([X_imgs, X_tabular])
    inp_mlp = model.layers[-1](model.layers[-2](model.layers[-3](inp_mlp)))
    y_prediction = np.append(y_prediction, inp_mlp.numpy(), axis=0)
y_prediction = y_prediction[1:]


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2101735088.py in <cell line: 0>()
      2 SELECTED_COLUMNS = ['Weeks', 'Percent', 'Age', 'Sex', 'Min_week', 'Base_FVC','Base_week', '_Currently smokes', '_Ex-smoker', '_Never smoked']
      3 y_prediction = np.array([[0,0,0]])
----> 4 for X1, X2 in pred_generator:
      5     out_imgs = CNN(tf.convert_to_tensor(X2))
      6     X_imgs = tf.concat([tf.repeat(tf.reshape(out_imgs[i], (1,-1)), X1['Patient'].value_counts()[i], axis=0) for i in range(len(X1['Patient'].value_counts()))], axis=0)

/tmp/ipykernel_11/2563314176.py in __getitem__(self, index)
     20 
     21         patients = self.train.loc[list_IDs_temp, 'Patient'].unique()
---> 22         imgs = _read(self.img_path, patients=patients, desired_size=self.desired_size)
     23         return self.train.loc[list_IDs_temp, :].reset_index(drop=True), np.transpose(imgs, (0, 2, 3, 4, 1))

/tmp/ipykernel_11/3421092196.py in _read(path, patients, desired_size)
     14 #         df = np.array(df.iloc[:,0].apply(lambda x: resize(pydicom.dcmread(x).pixel_array, DESIRED_SIZE[1:3])).to_list())
     15 #         df = zoom(df, np.array(DESIRED_SIZE)/np.array(df.shape), mode='nearest')
---> 16         X[i,0,:,:,:] = zerocenter(normalize(maskwatershed(clip(df), dicom)))
     17         i+=1
     18 

/tmp/ipykernel_11/1681866856.py in __call__(self, image, dicom)
      9         for slice_idx in range(image.shape[0]):
     10             sliced = image[slice_idx]
---> 11             stack.append(self.seperate_lungs(sliced, self.min_hu,
     12                                              self.iterations))
     13 

/tmp/ipykernel_11/1681866856.py in seperate_lungs(image, min_hu, iterations)
     18         h, w = image.shape[0], image.shape[1]
     19 
---> 20         marker_internal, marker_external, marker_watershed = MaskWatershed.generate_markers(image)
     21 
     22         # Sobel-Gradient

/tmp/ipykernel_11/1681866856.py in generate_markers(image, threshold)
     77 
     78         # Creation of the Watershed Marker
---> 79         marker_watershed = np.zeros((h, w), dtype=np.int)
     80         marker_watershed += marker_internal * 255
     81         marker_watershed += marker_external * 128

/usr/local/lib/python3.11/dist-packages/numpy/__init__.py in __getattr__(attr)
    322 
    323         if attr in __former_attrs__:
--> 324             raise AttributeError(__former_attrs__[attr])
    325 
    326         if attr == 'testing':

AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

## === cell 22
sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':y_prediction[:,1], 'Confidence': (y_prediction[:,2]-y_prediction[:,0])})
sub.to_csv('submission.csv', index=False)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3086690166.py in <cell line: 0>()
----> 1 sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':y_prediction[:,1], 'Confidence': (y_prediction[:,2]-y_prediction[:,0])})
      2 sub.to_csv('submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    688                     f"length {len(index)}"
    689                 )
--> 690                 raise ValueError(msg)
    691         else:
    692             index = default_index(lengths[0])

ValueError: array length 1 does not match index length 1908

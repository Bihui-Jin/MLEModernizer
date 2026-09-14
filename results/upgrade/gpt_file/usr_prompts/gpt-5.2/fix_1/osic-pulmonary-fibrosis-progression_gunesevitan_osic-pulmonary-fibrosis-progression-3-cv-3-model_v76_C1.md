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

-6.859308788419815

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pickle
import random
import gc

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from scipy.stats import skew, mode
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import pydicom

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization, GaussianDropout
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import Callback

SEED = 1337

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
df_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
df_submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

print(f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}')
print(f'Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
print(f'Test Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f'Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')
print(f'Sample Submission Shape = {df_submission.shape}')
print(f'Sample Submission Memory Usage = {df_submission.memory_usage().sum() / 1024 ** 2:.2f} MB')

## === cell 2
class Preprocessor:
    
    def __init__(self, df_train, df_test, df_submission, n_folds, shuffle, resize_shape):
        
        self.df_train = df_train.copy(deep=True)
        self.df_train.sort_values(by=['Patient', 'Weeks'], inplace=True)        
        self.df_test = df_test.copy(deep=True)
        self.df_submission = df_submission.copy(deep=True)
                
        self.n_folds = n_folds
        self.shuffle = shuffle
        
        self.resize_shape = resize_shape
        
    def _drop_duplicates(self):
        
        """        
        Calculate the mean FVC and Percent of [Patient, Weeks] groups and drop duplicate rows
        This operation takes the mean of multiple measurements in a single week and uses it
        """
        
        self.df_train['FVC'] = self.df_train.groupby(['Patient', 'Weeks'])['FVC'].transform('mean')
        self.df_train['Percent'] = self.df_train.groupby(['Patient', 'Weeks'])['Percent'].transform('mean')
        self.df_train.drop_duplicates(inplace=True)
        self.df_train.reset_index(drop=True, inplace=True)
        
    def _label_encode(self):
        
        """
        Label Encode categorical features
        """
                    
        for df in [self.df_train, self.df_test]:
            df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})           
            df['SmokingStatus'] = df['SmokingStatus'].map({'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2})       
            
    def _create_folds(self):
        
        """
        Creates n number of folds for three different cross-validation schemes (n should be selected as 2 because of the low patient count)
            
        1. Double Stratified Shuffled Folds
        -----------------------------------
        Patients are stratified by Sex and SmokingStatus features and groups are shuffled
        Patients listed below are split into n folds with all of their FVC measurements
        
        Male_Ex-smoker             106 / n
        Male_Never smoked           26 / n
        Female_Never smoked         23 / n
        Female_Ex-smoker            12 / n
        Male_Currently smokes        7 / n
        Female_Currently smokes      2 / n
        
        2. Cluster Stratified Shuffled Folds
        ------------------------------------
        Patients are clustered by their last two FVC values and clusters are stratified
        Patients listed below are split into n folds with all of their FVC measurements
        
        Cluster 2    167 / n
        Cluster 3      7 / n
        Cluster 1      2 / n  
        
        3. Regular Shuffled Folds
        -------------------------
        Patient groups are shuffled into n folds        
        Patient Count 176 / n
        """
        
        self.df_train['Sex_SmokingStatus'] = self.df_train['Sex'].astype(str) + '_' + self.df_train['SmokingStatus'].astype(str)
        for group in self.df_train['Sex_SmokingStatus'].unique():
            patients = self.df_train[self.df_train['Sex_SmokingStatus'] == group]['Patient'].unique()
            
            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)
                
            for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
                self.df_train.loc[self.df_train['Patient'].isin(patient_group), 'CV1_Fold'] = fold
               
        for patient_name in self.df_train['Patient'].unique():
            z = (self.df_train[(self.df_train['Patient'] == patient_name)]['FVC'].values[-2:] - self.df_train[(self.df_train['Patient'] == patient_name)]['FVC'].values[-2:].mean()) / self.df_train[(self.df_train['Patient'] == patient_name)]['FVC'].values[-2:].std()
            reg = LinearRegression(normalize=True).fit(self.df_train[(self.df_train['Patient'] == patient_name)]['Weeks'].values[-2:].reshape(-1, 1), z)

            self.df_train.loc[self.df_train['Patient'] == patient_name, 'Intercept'] = reg.intercept_
            self.df_train.loc[self.df_train['Patient'] == patient_name, 'Coef'] = reg.coef_[0]
            
        self.df_train.loc[self.df_train['Coef'] > 0.4, 'Cluster'] = 1
        self.df_train.loc[(self.df_train['Coef'] < 0.4) & (self.df_train['Coef'] > -0.4), 'Cluster'] = 2
        self.df_train.loc[self.df_train['Coef'] < -0.4, 'Cluster'] = 3

        for group in self.df_train['Cluster'].unique():
            patients = self.df_train[self.df_train['Cluster'] == group]['Patient'].unique()

            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)

            for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
                self.df_train.loc[self.df_train['Patient'].isin(patient_group), 'CV2_Fold'] = fold
               
        patients = self.df_train['Patient'].unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)
        
        for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
            self.df_train.loc[self.df_train['Patient'].isin(patient_group), 'CV3_Fold'] = fold
            
        self.df_train.drop(columns=['Sex_SmokingStatus', 'Intercept', 'Coef', 'Cluster'], inplace=True)        
    
    def load_scan(self, dataset, patient_name):
        
        """
        Load slices of the given patient
        Sort slices by ImagePositionPatient Z if the field exists, otherwise, sort them by InstanceNumber
        Keep first slices with the same ImagePositionPatient Z or InstanceNumber values
        Stack cropped and resized slices on top of each other and create 3D volume
        Exclude all zero slices from the 3D volume and return it

        Parameters
        ----------
        dataset : str Name of the dataset (train or test)
        patient: str Name of the patient (values from Patient column)

        Returns
        -------
        scan : numpy array, shape = (n_slices, self.resize_shape[0], self.resize_shape[1])
        numpy array after the cropped and resized slices are stacked
        metadata : dict
        dictionary of processed metadata
        """
        
        patient_directory = [pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}/{s}') for s in os.listdir(f'../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}')]
        
        try:
            patient_directory.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            slice_positions = np.round([s.ImagePositionPatient[2] for s in patient_directory], 4)
            non_duplicate_idx = np.unique([np.where(slice_position == slice_positions)[0][0] for slice_position in slice_positions])
        except AttributeError:
            patient_directory.sort(key=lambda s: int(s.InstanceNumber))
            instance_numbers = np.array([int(s.InstanceNumber) for s in patient_directory])
            non_duplicate_idx = np.unique([np.where(instance_number == instance_numbers)[0][0] for instance_number in instance_numbers])

        patient_directory = list(np.array(patient_directory)[non_duplicate_idx])
                   
        metadata = {}
        pixel_spacings = np.zeros((len(patient_directory), 2))
        slice_positions = np.zeros((len(patient_directory)))
        
        for i, s in enumerate(patient_directory): 
            try:
                pixel_spacings[i, :] = np.array(s.PixelSpacing)
            except AttributeError:
                pixel_spacings[i, :] = np.nan
                
            try:
                slice_positions[i] = s.ImagePositionPatient[2]
            except AttributeError:
                pass
        
        metadata['PixelSpacing'] = list(np.round(pixel_spacings.mean(axis=0), 3))
        
        if patient_name == 'ID00128637202219474716089':
            metadata['SliceSpacing'] = 5.0
        elif patient_name == 'ID00132637202222178761324':
            metadata['SliceSpacing'] = 0.7
        else:
            metadata['SliceSpacing'] = list(mode(np.abs(np.diff(np.round(slice_positions, 3)))))[0][0]
                
        scan = np.zeros((len(patient_directory), self.resize_shape[0], self.resize_shape[1]), dtype=np.int16) 
        for i, s in enumerate(patient_directory):            
            s_cropped = self.crop_slice(s.pixel_array)
            s_resized = self.resize_slice(s_cropped)
            
            if np.all(s_resized == 0):
                continue
            else:    
                scan[i] = np.int16(s_resized)
                    
        del patient_directory
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]   
        return scan, metadata       
        
    def crop_slice(self, s):

        """
        Crop frames from slices

        Parameters
        ----------
        s : numpy array, shape = (Rows, Columns)
        numpy array of slices with frame

        Returns
        -------
        s_cropped : numpy array, shape = (Rows - All Zero Rows, Columns - All Zero Columns)
        numpy array after the all zero rows and columns are dropped
        """
        
        if np.all(s == 0):
            return s
        
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_cropped = s[~np.all(s == 0, axis=1)] # Remove all zero horizontal lines
            s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)] # Remove all zero vertical lines 
        else:
            s_cropped = s

        return s_cropped
    
    def resize_slice(self, s):
        
        """
        Resize slices to given size with nearest neighbor interpolation

        Parameters
        ----------
        s : numpy array, shape = (Rows, Columns)
        numpy array of slices

        Returns
        -------
        s_resized : numpy array, shape = (resize_shape[0], resize_shape[0])
        numpy array after resized to resize_shape
        """
        
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_resized = cv2.resize(s, self.resize_shape, interpolation=cv2.INTER_NEAREST) # Using nearest-neighbor interpolation for preserving pixel values
        else:
            s_resized = s
             
        return s_resized
            
    def _create_baseline_features(self):
            
        self.df_submission['Type'] = 'Test'
        self.df_submission['Patient'] = self.df_submission['Patient_Week'].apply(lambda x: x.split('_')[0]).astype(str)
        self.df_submission['Weeks'] = df_submission['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)
        self.df_submission.drop(columns=['Patient_Week', 'FVC', 'Confidence'], inplace=True)
        
        self.df_train['Type'] = 'Train'
        self.df_train['Weeks_Passed'] = self.df_train['Weeks'] - self.df_train.groupby('Patient')['Weeks'].transform('min')
        self.df_train['FVC_Baseline'] = self.df_train.groupby('Patient')['FVC'].transform('first')    

        for patient in self.df_test['Patient'].unique():
            self.df_submission.loc[self.df_submission['Patient'] == patient, 'FVC_Baseline'] = self.df_test[self.df_test['Patient'] == patient]['FVC'].values
            self.df_submission.loc[self.df_submission['Patient'] == patient, 'Percent'] = self.df_test[self.df_test['Patient'] == patient]['Percent'].values
            self.df_submission.loc[self.df_submission['Patient'] == patient, 'Age'] = self.df_test[self.df_test['Patient'] == patient]['Age'].values
            self.df_submission.loc[self.df_submission['Patient'] == patient, 'Sex'] = self.df_test[self.df_test['Patient'] == patient]['Sex'].values
            self.df_submission.loc[self.df_submission['Patient'] == patient, 'SmokingStatus'] = self.df_test[self.df_test['Patient'] == patient]['SmokingStatus'].values
          
        self.df_submission['Weeks_Passed'] = self.df_submission['Weeks']
        
        self.df_all = pd.concat([self.df_train, self.df_submission], ignore_index=True, axis=0)
        self.df_all['Age'] += np.int8(np.floor(self.df_all['Weeks_Passed'] / 52))
        
        self.df_all['Weeks'] = self.df_all['Weeks'].astype(np.int16)
        self.df_all['Age'] = self.df_all['Age'].astype(np.float32)
        self.df_all['FVC_Baseline'] = self.df_all['FVC_Baseline'].astype(np.float32)
        self.df_all['Percent'] = self.df_all['Percent'].astype(np.float32)
        self.df_all['Weeks_Passed'] = self.df_all['Weeks_Passed'].astype(np.float32)
        self.df_all['Sex'] = self.df_all['Sex'].astype(np.uint8)
        self.df_all['SmokingStatus'] = self.df_all['SmokingStatus'].astype(np.uint8)
        self.df_all['FVC'] = self.df_all['FVC'].astype(np.float32)
        
        self.df_train = self.df_all.loc[self.df_all['Type'] == 'Train', :].drop(columns=['Type'])
        for i in range(1, 4):
            self.df_train[f'CV{i}_Fold'] = self.df_train[f'CV{i}_Fold'].astype(np.uint8)
        self.df_test = self.df_all.loc[self.df_all['Type'] == 'Test', :].drop(columns=['Type', 'FVC', 'CV1_Fold', 'CV2_Fold', 'CV3_Fold'])    
        
    def _create_image_features(self):
        
        df_train_features = pd.read_csv('../input/osic-pulmonary-fibrosis-progression-features/df_scan_features.csv')
        scale_features = ['Fat']
        scaler = StandardScaler()
        scaler.fit(df_train_features.loc[:, scale_features])
        df_train_features.loc[:, scale_features] = scaler.transform(df_train_features.loc[:, scale_features])        
        self.df_train = self.df_train.merge(df_train_features, how='left', on='Patient')
                
        for patient_name in self.df_test['Patient'].unique():     
            
            scan, metadata = self.load_scan('test', patient_name)            
                        
            self.df_test.loc[self.df_test['Patient'] == patient_name, 'PixelSpacingX'] = metadata['PixelSpacing'][0]
            self.df_test.loc[self.df_test['Patient'] == patient_name, 'PixelSpacingY'] = metadata['PixelSpacing'][1]
            self.df_test.loc[self.df_test['Patient'] == patient_name, 'SliceSpacing'] = metadata['SliceSpacing']
            self.df_test.loc[self.df_test['Patient'] == patient_name, 'SliceCount'] = scan.shape[0]
            volume = ((metadata['SliceSpacing'] * scan.shape[0]) * (metadata['PixelSpacing'][0] * scan.shape[1]) * (metadata['PixelSpacing'][1] * scan.shape[2]))            
            self.df_test.loc[self.df_test['Patient'] == patient_name, 'VoxelVolume'] = volume / (scan.shape[0] * scan.shape[1] * scan.shape[2])           
            self.df_test.loc[self.df_test['Patient'] == patient_name, 'Fat'] = scan[(scan > -200) & (scan <= -50)].shape[0] * self.df_test.loc[self.df_test['Patient'] == patient_name, 'VoxelVolume']
            
            self.df_test.loc[self.df_test['Patient'] == patient_name, 'Scan_Skew'] = skew(scan.flatten())
            
            slice_skews = []
            
            for s in scan:
                slice_skews.append(skew(s.flatten()))
                
            self.df_test.loc[self.df_test['Patient'] == patient_name, 'Std_Slice_Skew'] = np.std(slice_skews)
            self.df_test.loc[self.df_test['Patient'] == patient_name, 'Var_Slice_Skew'] = np.var(slice_skews)
            
            del scan
            gc.collect()
            
        test_drop_cols = ['FVC', 'PixelSpacingX', 'PixelSpacingY', 'SliceSpacing', 'SliceCount', 'VoxelVolume']
        self.df_test.drop(columns=test_drop_cols, inplace=True)
        self.df_test.loc[:, scale_features] = scaler.transform(self.df_test.loc[:, scale_features])
            
    def get_data(self):
        
        self._drop_duplicates()  
        self._label_encode()
        self._create_folds()        
        self._create_baseline_features()         
        
        print(f'Preprocessed Training Set Shape = {self.df_train.shape}')
        print(f'Preprocessed Training Set Memory Usage = {self.df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
        print(f'Preprocessed Test Set Shape = {self.df_test.shape}')
        print(f'Preprocessed Test Set Memory Usage = {self.df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')
        
        return self.df_train.copy(deep=True), self.df_test.copy(deep=True)        
        

## === cell 3
preprocessor_parameters = {
    'df_train': df_train,
    'df_test': df_test,
    'df_submission': df_submission,
    'n_folds': 2,
    'shuffle': True,
    'resize_shape': (512, 512)
}

preprocessor = Preprocessor(**preprocessor_parameters)
df_train, df_test = preprocessor.get_data()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3596699974.py in <cell line: 0>()
      9 
     10 preprocessor = Preprocessor(**preprocessor_parameters)
---> 11 df_train, df_test = preprocessor.get_data()

/tmp/ipykernel_11/2935161230.py in get_data(self)
    319         self._drop_duplicates()
    320         self._label_encode()
--> 321         self._create_folds()
    322         self._create_baseline_features()
    323         #self._create_image_features()

/tmp/ipykernel_11/2935161230.py in _create_folds(self)
     82         for patient_name in self.df_train['Patient'].unique():
     83             z = (self.df_train[(self.df_train['Patient'] == patient_name)]['FVC'].values[-2:] - self.df_train[(self.df_train['Patient'] == patient_name)]['FVC'].values[-2:].mean()) / self.df_train[(self.df_train['Patient'] == patient_name)]['FVC'].values[-2:].std()
---> 84             reg = LinearRegression(normalize=True).fit(self.df_train[(self.df_train['Patient'] == patient_name)]['Weeks'].values[-2:].reshape(-1, 1), z)
     85 
     86             self.df_train.loc[self.df_train['Patient'] == patient_name, 'Intercept'] = reg.intercept_

TypeError: LinearRegression.__init__() got an unexpected keyword argument 'normalize'

## === cell 4
class QuantileRegressorMLP:
    
    def __init__(self, model, predictors, mlp_parameters, qr_parameters):
        
        self.model = model
        self.predictors = predictors

        self.mlp_parameters = mlp_parameters
        self.qr_parameters = qr_parameters
        
    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = - np.sqrt(2) * delta_clipped / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

        return np.mean(score)        
            
    def laplace_log_likelihood_loss(self, y_true, y_pred):
        
        K.cast(y_true, 'float32')
        K.cast(y_pred, 'float32')
    
        sigma_lower_bound = K.constant(70, dtype='float32')
        delta_upper_bound = K.constant(1000, dtype='float32')

        sigma = y_pred[:, 1]
        fvc_pred = y_pred[:, 0]

        sigma_clipped = K.maximum(sigma, sigma_lower_bound)
        delta = K.abs(y_true[:, 0] - fvc_pred)
        delta_clipped = K.minimum(delta, delta_upper_bound)

        score = (delta_clipped / sigma_clipped) * K.sqrt(K.cast(2, 'float32')) + K.log(sigma_clipped * K.sqrt(K.cast(2, 'float32')))
        return K.mean(score)
    
    def tilted_loss(self, y_true, y_pred):
        
        quantiles = K.constant(np.array([self.qr_parameters['quantiles']]), dtype='float32')
        error = y_true - y_pred
        return K.mean(K.maximum(quantiles * error, (quantiles - 1) * error))
    
    def hybrid_loss(self, w):
        
        def loss(y_true, y_pred):
            return w * self.tilted_loss(y_true, y_pred) + (1 - w) * self.laplace_log_likelihood_loss(y_true, y_pred)
        return loss
    
    def get_model(self, input_shape, m):
        
        model = None
    
        if m == 'MLP':            
            input_layer = Input(shape=input_shape)
            x = Dense(2 ** 7, activation='relu')(input_layer)   
            x = GaussianDropout(0.01)(x)
            x = Dense(2 ** 7, activation='relu')(x)
            x = GaussianDropout(0.01)(x)
            p1 = Dense(2, activation='linear')(x)
            p2 = Dense(2, activation='relu')(x)
            output_layer = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1))([p1, p2])

            model = Model(input_layer, output_layer)
            model.compile(loss=self.laplace_log_likelihood_loss, optimizer=tf.keras.optimizers.Adam(lr=self.mlp_parameters['lr']), metrics=[self.laplace_log_likelihood_loss])
            
        elif m == 'QR':            
            input_layer = Input(shape=input_shape)
            x = Dense(2 ** 7, activation='relu')(input_layer)   
            x = GaussianDropout(0.01)(x)
            x = Dense(2 ** 7, activation='relu')(x)
            x = GaussianDropout(0.01)(x)
            p1 = Dense(3, activation='linear')(x)
            p2 = Dense(3, activation='relu')(x)
            output_layer = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1))([p1, p2])

            model = Model(input_layer, output_layer)
            model.compile(loss=self.tilted_loss, optimizer=tf.keras.optimizers.Adam(lr=self.qr_parameters['lr']), metrics=[self.laplace_log_likelihood_loss])

        return model
        
    def train(self, X_train, y_train):
        
        self.mlp_scores = []
        self.qr_scores = []

        self.mlp_oof = pd.DataFrame(np.zeros((len(y_train), 2)))
        self.qr_oof = pd.DataFrame(np.zeros((len(y_train), len(self.qr_parameters['quantiles']))))
            
        self.mlp_models = {'CV1': [], 'CV2': [], 'CV3': []}
        self.qr_models = {'CV1': [], 'CV2': [], 'CV3': []}
        
        models = [self.model] if self.model != 'Stack' else ['MLP', 'QR']        
        for m in models:            
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')
            
            for cv in range(1, 4):               
                for fold in sorted(X_train[f'CV{cv}_Fold'].unique()):
                    trn_idx, val_idx = X_train.loc[X_train[f'CV{cv}_Fold'] != fold].index, X_train.loc[X_train[f'CV{cv}_Fold'] == fold].index

                    X_trn = X_train.loc[trn_idx, self.predictors]
                    y_trn = y_train.loc[trn_idx]
                    X_val = X_train.loc[val_idx, self.predictors]
                    y_val = y_train.loc[val_idx]

                    model = self.get_model(input_shape=X_trn.shape[1], m=m)                
                    if m ==  'MLP':                    
                        model.fit(X_trn, y_trn, epochs=self.mlp_parameters['epochs'], batch_size=self.mlp_parameters['batch_size'], verbose=0)
                        self.mlp_models[f'CV{cv}'].append(model)                    
                    elif m == 'QR':                    
                        model.fit(X_trn, y_trn, epochs=self.qr_parameters['epochs'], batch_size=self.qr_parameters['batch_size'], verbose=0)
                        self.qr_models[f'CV{cv}'].append(model)

                    predictions = model.predict(X_val)                
                    if m == 'MLP':                    
                        oof_predictions = predictions[:, 0]
                        self.mlp_oof.iloc[val_idx, 0] = oof_predictions
                        df_train.loc[val_idx, f'CV{cv}_MLP_FVC_Predictions'] = oof_predictions

                        oof_confidence = predictions[:, 1]
                        self.mlp_oof.iloc[val_idx, 1] = oof_confidence
                        df_train.loc[val_idx, f'CV{cv}_MLP_Confidence_Predictions'] = oof_confidence
                    elif m == 'QR':                    
                        oof_predictions = predictions[:, 1]
                        oof_confidence = predictions[:, 2] - predictions[:, 0]
                        for i, quantile in enumerate(self.qr_parameters['quantiles']):
                            self.qr_oof.iloc[val_idx, i] = predictions[:, i]
                            df_train.loc[val_idx, f'CV{cv}_QR_{quantile}_Predictions'] = predictions[:, i]

                    oof_score = self.laplace_log_likelihood_metric(y_val, oof_predictions, oof_confidence)
                    if m == 'MLP':
                        self.mlp_scores.append(oof_score)
                    elif m == 'QR':
                        self.qr_scores.append(oof_score)
                    print(f'CV {cv} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - Score: {oof_score:.6}')

                if m == 'MLP':
                    print(f'{"-" * 30}\nCV {cv} MLP Mean Laplace Log Likelihood {np.mean(self.mlp_scores):.6} [Std: {np.std(self.mlp_scores):.6}]')
                    print(f'CV {cv} MLP OOF Laplace Log Likelihood {self.laplace_log_likelihood_metric(y_train, self.mlp_oof.iloc[:, 0], self.mlp_oof.iloc[:, 1]):.6}\n{"-" * 30}\n')
                if m == 'QR':                    
                    print(f'{"-" * 30}\nCV {cv} QR Mean Laplace Log Likelihood {np.mean(self.qr_scores):.6} [Std: {np.std(self.qr_scores):.6}]')
                    print(f'CV {cv} QR OOF Laplace Log Likelihood {self.laplace_log_likelihood_metric(y_train, self.qr_oof.iloc[:, 1], (self.qr_oof.iloc[:, 2] - self.qr_oof.iloc[:, 0])):.6}\n{"-" * 30}\n')
                    
    def predict(self, X_test):
        
        for cv in range(1, 4):
            mlp_predictions = np.zeros((len(X_test), 2))            
            for model in self.mlp_models[f'CV{cv}']:
                mlp_predictions += model.predict(X_test[self.predictors]) / len(self.mlp_models[f'CV{cv}'])
    
            X_test[f'CV{cv}_MLP_FVC_Predictions'] = mlp_predictions[:, 0]
            X_test[f'CV{cv}_MLP_Confidence_Predictions'] = mlp_predictions[:, 1]
            
            qr_predictions = np.zeros((len(X_test), len(self.qr_parameters['quantiles'])))
            for model in self.qr_models[f'CV{cv}']:
                qr_predictions += model.predict(X_test[self.predictors]) / len(self.qr_models[f'CV{cv}'])

            for i, quantile in enumerate(self.qr_parameters['quantiles']):    
                X_test[f'CV{cv}_QR_{quantile}_Predictions'] = qr_predictions[:, i]
               
    def plot_predictions(self, df, patient): 
              
        mlp_prediction_columns = [f'CV{cv}_MLP_{target}_Predictions' for cv in range(1, 4) for target in ['FVC', 'Confidence']]
        mlp_scores = []
        for cv in range(1, 4):            
            score = self.laplace_log_likelihood_metric(df['FVC'], df[f'CV{cv}_MLP_FVC_Predictions'], df[f'CV{cv}_MLP_Confidence_Predictions'])
            mlp_scores.append(round(score, 5))
        
        qr_prediction_columns = [f'CV{cv}_QR_{quantile}_Predictions' for cv in range(1, 4) for quantile in self.qr_parameters['quantiles']]
        qr_scores = []
        for i, cv in enumerate(range(1, 4)):
            score = self.laplace_log_likelihood_metric(df['FVC'], df[qr_prediction_columns[1 + (i * 3)]], (df[qr_prediction_columns[2 + (i * 3)]] - df[qr_prediction_columns[0 + (i * 3)]]))
            qr_scores.append(round(score, 5))

        ax = df[(['Weeks', 'FVC'] + mlp_prediction_columns[::2] + qr_prediction_columns[1::3])].set_index('Weeks').plot(figsize=(30, 6), style=['-b', 'r--', 'g--', 'b--', 'r:', 'g:', 'b:'])
        ax.fill_between(df['Weeks'], df[qr_prediction_columns[2]], df[qr_prediction_columns[0]], alpha=0.1, label='CV1 QR Prediction Interval', color='red')
        ax.fill_between(df['Weeks'], df[qr_prediction_columns[5]], df[qr_prediction_columns[3]], alpha=0.1, label='CV2 QR Prediction Interval', color='green')
        ax.fill_between(df['Weeks'], df[qr_prediction_columns[8]], df[qr_prediction_columns[6]], alpha=0.1, label='CV3 QR Prediction Interval', color='blue')

        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'Patient: {patient} - MLP Scores: {mlp_scores} QR Scores: {qr_scores}', size=25, pad=25)
        ax.legend(prop={'size': 18})

        plt.show()
        
                

## === cell 5
seed_everything(SEED)

X_train = df_train.drop(columns=['FVC', 'Weeks'])
y_train = df_train['FVC'].copy(deep=True)

model_parameters = {
    'model': 'Stack',
    'predictors': ['Age', 'Sex', 'SmokingStatus', 'FVC_Baseline', 'Percent', 'Weeks_Passed'],
    'mlp_parameters': {
        'lr': 0.0005,
        'epochs': 350,
        'batch_size': 2 ** 5
    },
    'qr_parameters': {
        'quantiles': [0.25, 0.5, 0.75],
        'lr': 0.0005,
        'epochs': 150,
        'batch_size': 2 ** 5        
    }
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train)
qr_mlp.predict(df_test)

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

KeyError: 'CV1_Fold'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/164142889.py in <cell line: 0>()
     21 
     22 qr_mlp = QuantileRegressorMLP(**model_parameters)
---> 23 qr_mlp.train(X_train, y_train)
     24 qr_mlp.predict(df_test)

/tmp/ipykernel_11/730167253.py in train(self, X_train, y_train)
     95 
     96             for cv in range(1, 4):
---> 97                 for fold in sorted(X_train[f'CV{cv}_Fold'].unique()):
     98                     trn_idx, val_idx = X_train.loc[X_train[f'CV{cv}_Fold'] != fold].index, X_train.loc[X_train[f'CV{cv}_Fold'] == fold].index
     99 

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

KeyError: 'CV1_Fold'

## === cell 6
for patient, df in list(df_train.groupby('Patient'))[:10]:        
    qr_mlp.plot_predictions(df, patient)

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

KeyError: 'CV1_MLP_FVC_Predictions'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/425134564.py in <cell line: 0>()
      1 for patient, df in list(df_train.groupby('Patient'))[:10]:
----> 2     qr_mlp.plot_predictions(df, patient)

/tmp/ipykernel_11/730167253.py in plot_predictions(self, df, patient)
    163         mlp_scores = []
    164         for cv in range(1, 4):
--> 165             score = self.laplace_log_likelihood_metric(df['FVC'], df[f'CV{cv}_MLP_FVC_Predictions'], df[f'CV{cv}_MLP_Confidence_Predictions'])
    166             mlp_scores.append(round(score, 5))
    167 

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

KeyError: 'CV1_MLP_FVC_Predictions'

## === cell 7
class SubmissionPipeline:
    
    def __init__(self, df_train, df_test):
        
        self.df_train = df_train
        self.df_test = df_test
        
    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = - np.sqrt(2) * delta_clipped / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

        return np.mean(score)     
        
    def single_model(self, model):
        
        self.df_test['Patient_Week'] = self.df_test['Patient'].astype(str) + '_' + self.df_test['Weeks'].astype(str)
        
        prediction_cols = [col for col in self.df_train.columns if col.startswith(model)]
        if model.split('_')[1] == 'MLP':
            score = self.laplace_log_likelihood_metric(self.df_train['FVC'], self.df_train[prediction_cols[0]], self.df_train[prediction_cols[1]])
            print(f'Single Model {model} Score: {score:.6}')
            self.df_test['FVC'] = self.df_test[prediction_cols[0]]
            self.df_test['Confidence'] = self.df_test[prediction_cols[1]]
            
        elif model.split('_')[1] == 'QR':
            score = self.laplace_log_likelihood_metric(self.df_train['FVC'], self.df_train[prediction_cols[1]], (self.df_train[prediction_cols[2]] - self.df_train[prediction_cols[0]]))
            print(f'Single Model {model} Score: {score:.6}')
            self.df_test['FVC'] = self.df_test[prediction_cols[1]]
            self.df_test['Confidence'] = self.df_test[prediction_cols[2]] - self.df_test[prediction_cols[0]]
            
        print(f'\n{self.df_test[["Patient_Week", "FVC", "Confidence"]].describe()}')
            
        return self.df_test[['Patient_Week', 'FVC', 'Confidence']].copy(deep=True)
        
    def blend(self):
        
        qr_predictions = np.zeros((len(self.df_test), 3))
            
            
        
    

## === cell 8
sub = SubmissionPipeline(df_train, df_test)
df_submission = sub.single_model('CV3_MLP')
df_submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/339513739.py in <cell line: 0>()
      1 sub = SubmissionPipeline(df_train, df_test)
----> 2 df_submission = sub.single_model('CV3_MLP')
      3 df_submission.to_csv('submission.csv', index=False)

/tmp/ipykernel_11/1799805139.py in single_model(self, model)
     20         prediction_cols = [col for col in self.df_train.columns if col.startswith(model)]
     21         if model.split('_')[1] == 'MLP':
---> 22             score = self.laplace_log_likelihood_metric(self.df_train['FVC'], self.df_train[prediction_cols[0]], self.df_train[prediction_cols[1]])
     23             print(f'Single Model {model} Score: {score:.6}')
     24             self.df_test['FVC'] = self.df_test[prediction_cols[0]]

IndexError: list index out of range

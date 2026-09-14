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

-12.250279187279803

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
import os.path
import random
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2
import os
import math  
import pandas as pd
import sklearn.metrics as sm
import tensorflow as tf

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
curr_wkdir = os.getcwd()          # get the current working directory
print(curr_wkdir)

## === cell 3
import os.path
from tensorflow.keras.models  import  load_model

model_path = '../input/osic-model-ver3c'
model_savefile = model_path + '/osic_model_ver1.0.h5'

osic_model = load_model(model_savefile)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1011608112.py in <cell line: 0>()
      8 model_savefile = model_path + '/osic_model_ver1.0.h5'
      9 
---> 10 osic_model = load_model(model_savefile)

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/osic-model-ver3c/osic_model_ver1.0.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 4
osic_model.summary()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1172156978.py in <cell line: 0>()
----> 1 osic_model.summary()

NameError: name 'osic_model' is not defined

## === cell 5
osic_model.get_weights()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2473381295.py in <cell line: 0>()
      1 # Check model weights
----> 2 osic_model.get_weights()

NameError: name 'osic_model' is not defined

## === cell 6

test_dir ='../input/osic-pulmonary-fibrosis-progression'

maxPercent = 160
train_file = test_dir + '/train.csv'   # *** parameter : need to update with correct file name during submission
train_data = pd.read_csv(train_file)
train_data['FullFVC'] = (train_data['FVC'] * 100) / train_data['Percent'] 
train_data['PercentScaled'] = train_data['Percent'] / maxPercent
train_data['Gender'] = np.where(train_data['Sex'] == 'Male', 1, 0)


test_file = '../input/osic-pulmonary-fibrosis-progression/test.csv'   # *** parameter : need to update with correct file name during submission


test_data = pd.read_csv(test_file)

if  'ID00011637202177653955184' in test_data.values:
    indexNames = test_data[ test_data['Patient'] == "ID00011637202177653955184" ].index
    test_data.drop(indexNames , inplace=True)

test_data['FullFVC'] = (test_data['FVC'] * 100) / test_data['Percent'] 

test_data['PercentScaled'] = test_data['Percent'] / maxPercent

test_data['Gender'] = np.where(test_data['Sex'] == 'Male', 1, 0)


## === cell 7
print(test_data.columns.values)

## === cell 8
test_patient_info = test_data.drop(columns=['Weeks','FVC','Percent','Sex','PercentScaled'])
print(test_patient_info.columns.values)
print(len(test_patient_info))

## === cell 9
test_patient_info_np = np.array(test_patient_info)
patient_count = len(test_patient_info)

test_data_file = pd.DataFrame()

for i in range(patient_count):
    patient = test_patient_info_np[i,0]
    age = test_patient_info_np[i,1]
    smokingstatus = test_patient_info_np[i,2]
    fullfvc = test_patient_info_np[i,3]
    gender = test_patient_info_np[i,4]

    for wk in range(-12, 134):                     # need to generate from weeks -12 to 133
        test_data_file = (test_data_file.append({'Patient':patient, 'Weeks':wk,
                                                'Age':age, 'SmokingStatus':smokingstatus,
                                                'FullFVC':fullfvc, 'Gender':gender},
                                                ignore_index=True))
    
    
    

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3765718450.py in <cell line: 0>()
     13 
     14     for wk in range(-12, 134):                     # need to generate from weeks -12 to 133
---> 15         test_data_file = (test_data_file.append({'Patient':patient, 'Weeks':wk,
     16                                                 'Age':age, 'SmokingStatus':smokingstatus,
     17                                                 'FullFVC':fullfvc, 'Gender':gender},

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 10
print(f'test_data_file.columns.values : {test_data_file.columns.values}')
print(f'len(test_data_file): {len(test_data_file)}')


## === cell 11
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

## === cell 12
def create_montage(dir):
    inputImages = []
    outputImage = np.zeros((224,224), dtype="uint8")      
    files_list = []
    
    files_list = os.listdir(dir)

    onlyfiles = next(os.walk(dir))[2] #dir is your directory path as string
    imagecount = len(onlyfiles)

    imageinterval = math.floor(imagecount/4) 

    selectedimagesno= []           

    for i in range(4):
        imageslice = (i * imageinterval) + 1
        selectedimagesno.append(imageslice) 


    for item in selectedimagesno:
        image = io.imread(f'{dir}/{files_list[item]}')
        image = cv2.resize(image,(112,112))
        inputImages.append(image)



    outputImage[0:112, 0:112] = inputImages[0]
    outputImage[0:112, 112:224] = inputImages[1]
    outputImage[112:224, 112:224] = inputImages[2]
    outputImage[112:224, 0:112] = inputImages[3]  


    
    return outputImage

## === cell 14



path = '../input/osic-pulmonary-fibrosis-progression/test'
print(path)

images_list=[]
imagesID_list =[]
directory_list = []

directory_contents = os.listdir(path)

directory_list = directory_contents

if directory_list.count('ID00011637202177653955184') > 0:
    directory_list.remove('ID00011637202177653955184')

for item in directory_contents:
    folder = f'{path}/{item}'
    montage = create_montage(folder)
    images_list.append(montage)
    imagesID_list.append(item)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/22046009.py in <cell line: 0>()
     32 #     print(folder)
     33 #     montage = create_montage(f'{path}/{item}')
---> 34     montage = create_montage(folder)
     35     images_list.append(montage)
     36     imagesID_list.append(item)

/tmp/ipykernel_11/2362356649.py in create_montage(dir)
     39     # plotImages(inputImages)
     40 
---> 41     outputImage[0:112, 0:112] = inputImages[0]
     42     outputImage[0:112, 112:224] = inputImages[1]
     43     outputImage[112:224, 112:224] = inputImages[2]

ValueError: could not broadcast input array from shape (112,112,512) into shape (112,112)

## === cell 15
plotImages(images_list)            # for testing purpose

## === cell 17

input_images = []

for i in range(len(test_data_file)) : 
    imageloc = imagesID_list.index(test_data_file.iloc[i, 3])     #patient column in test_data_file  
    input_images.append(images_list[imageloc])

input_images = np.array(input_images)    

## === cell 18



from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

def process_input_attributes(df, inputdata):
    continuous = ['Weeks','Age','FullFVC']
    
    cs = MinMaxScaler()
    inputdataContinuous = cs.fit_transform(inputdata[continuous])
    
    zipBinarizer = LabelBinarizer().fit(df['SmokingStatus'])
    inputdataCategorical1 = zipBinarizer.transform(inputdata['SmokingStatus'])

    zipBinarizer = LabelBinarizer().fit(df['Gender'])
    inputdataCategorical2 = zipBinarizer.transform(inputdata['Gender'])
    
    inputdataX = np.hstack([inputdataCategorical1, inputdataCategorical2, inputdataContinuous])
    
    return (inputdataX)


## === cell 20

input_data = test_data_file.drop(columns=['Patient'])
input_attributes = process_input_attributes(train_data, input_data)       # chg on 4/Oct/2020

print(f'input data cols : {input_data.columns.values}')   # for checking purpose

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3301972594.py in <cell line: 0>()
      3 # ['Weeks' 'Age' 'SmokingStatus' 'FullFVC' 'Gender']
      4 
----> 5 input_data = test_data_file.drop(columns=['Patient'])
      6 input_attributes = process_input_attributes(train_data, input_data)       # chg on 4/Oct/2020
      7 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['Patient'] not found in axis"

## === cell 21
print(f'len(test_data) : {len(test_data)}')
print(f'len(test_data_file) : {len(test_data_file)}')
print(f'len(input_images) : {len(input_images)}')
print(f'len(input_attributes) : {len(input_attributes)}')

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/126665582.py in <cell line: 0>()
      3 print(f'len(test_data_file) : {len(test_data_file)}')
      4 print(f'len(input_images) : {len(input_images)}')
----> 5 print(f'len(input_attributes) : {len(input_attributes)}')

NameError: name 'input_attributes' is not defined

## === cell 22
test_predictions = osic_model.predict([input_attributes, input_images])

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3621506472.py in <cell line: 0>()
      1 # make prediction with model
----> 2 test_predictions = osic_model.predict([input_attributes, input_images])

NameError: name 'osic_model' is not defined

## === cell 25


test_data_file_np = np.array(test_data_file)
results = pd.DataFrame()
rec_count = len(test_data_file)

for i in range(rec_count):
    results = (results.append({'Age':test_data_file_np[i,0], 'FullFVC':test_data_file_np[i,1], 
                               'Gender': test_data_file_np[i,2], 'Patient': test_data_file_np[i,3], 
                               'SmokingStatus': test_data_file_np[i,4], 'Weeks': test_data_file_np[i,5], 
                               'PredictedPercentScaled': test_predictions[i]}, 
                              ignore_index=True))    
    
print('done')

## === cell 26
print(len(test_data_file))    # for testing

## === cell 28

results['PredictedFVC'] = ((results['PredictedPercentScaled'] * maxPercent) / 100) * results['FullFVC']

predictedfvc = results['PredictedFVC']

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2566337294.py in <cell line: 0>()
      4 # Changes for ver3c
      5 # results['PredictedFVC'] = results['PredictedPercentScaled'] * results['FullFVC']
----> 6 results['PredictedFVC'] = ((results['PredictedPercentScaled'] * maxPercent) / 100) * results['FullFVC']
      7 # end of changes for ver3c
      8 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'PredictedPercentScaled'

## === cell 31

results_confidence =[]
results_np = np.array(results)

for i in range(len(results_np)):
    confidence = 100
    results_confidence.append(confidence)
    

results['Confidence'] = results_confidence



## === cell 32
print(f'results columns : {results.columns.values}')

## === cell 33
results['Patient_Week'] = results['Patient'] + '_' + results['Weeks'].astype(int).astype(str)
print(f'results columns : {results.columns.values}')

## --- ERROR in cell 33, traceback:
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

KeyError: 'Patient'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/330790345.py in <cell line: 0>()
      1 # Generate the column Patient_Week
      2 # convert week to int first to remove .0 before converting to string
----> 3 results['Patient_Week'] = results['Patient'] + '_' + results['Weeks'].astype(int).astype(str)
      4 print(f'results columns : {results.columns.values}')

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

KeyError: 'Patient'

## === cell 34
results_filename = './' + 'results.csv'
results.to_csv(results_filename, index=True)

## === cell 35
print(f'results no of rows : {len(results)}')

## === cell 36
submission = pd.DataFrame()
patientwk_s = results['Patient_Week']
fvc_s       = results['PredictedFVC']
confidence_s = results['Confidence']

submission = pd.concat([patientwk_s, fvc_s, confidence_s], axis=1) 

submission_file = submission.rename(columns = {'PredictedFVC': 'FVC'}, inplace = False)

submission_file['FVC'] =  (submission_file['FVC'].str.get(0)).round(0).astype(int)     # 5/Oct/2020

print(submission_file)


output_filename = './' + 'submission.csv'

submission_file.to_csv(output_filename, index=False) 

## --- ERROR in cell 36, traceback:
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

KeyError: 'Patient_Week'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3962055309.py in <cell line: 0>()
      1 # Generate submission file
      2 submission = pd.DataFrame()
----> 3 patientwk_s = results['Patient_Week']
      4 fvc_s       = results['PredictedFVC']
      5 confidence_s = results['Confidence']

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

KeyError: 'Patient_Week'

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame must have a 'Patient_Week' column.

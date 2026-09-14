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

-7.85881723482709

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install --no-deps ../input/classification-models/classification_models-1.0.0
!pip install --no-deps ../input/keras-applications

## === cell 1
import albumentations
import cv2
import gc
import numpy as np
import os
import pandas as pd
import pydicom as dicom
import random

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import *

from classification_models.tfkeras import Classifiers

from sklearn.model_selection import KFold, train_test_split

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
MEAN_STD = {
    'Weeks': (31.861846352485475, 23.247550440817218),
    'FVC': (2690.479018721756, 832.7709592986739),
    'Percent': (77.67265350296324, 19.823261324684214),
    'Age': (67.18850871530019, 7.057394616249349),
    'typical_fvc': (3495.347708198836, 743.4071078314996),
}

MIN_MAX = {
    'Weeks': (-5., 133.),
    'FVC': (827., 6399.),
    'Percent': (28.877577, 153.145378),
    'Age': (49., 88.),
    'typical_fvc': (1598.899999999998, 4923.200000000003)
}

CATEGORICAL = {
    'Female': [1, 0],
    'Male': [0, 1],
    'Never smoked': [1, 0, 0],
    'Currently smokes': [0, 1, 0],
    'Ex-smoker': [0, 0, 1]
}

## === cell 3
IMG_SIZE = 128
BATCH_SIZE = 128
TEST_DF = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
SAMPLE_SUBMISSION = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
model_weights = [os.path.join('../input/osic-model-weights', x) for x in os.listdir('../input/osic-model-weights')]
training_features = ['Weeks', 'Age', 'Sex', 'SmokingStatus', 'typical_fvc']
num_of_features = 8

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1402817160.py in <cell line: 0>()
      4 TEST_DF = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
      5 SAMPLE_SUBMISSION = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
----> 6 model_weights = [os.path.join('../input/osic-model-weights', x) for x in os.listdir('../input/osic-model-weights')]
      7 training_features = ['Weeks', 'Age', 'Sex', 'SmokingStatus', 'typical_fvc']
      8 num_of_features = 8

FileNotFoundError: [Errno 2] No such file or directory: '../input/osic-model-weights'

## === cell 4
def create_typical_fvc(df):
    df['typical_fvc'] = df['FVC'] / df['Percent'] * 100.
    return df

def get_pixels_hu(scan):
    image = scan.pixel_array
    image = image.astype(np.int16)

    slope = scan.RescaleSlope
    intercept = scan.RescaleIntercept
    window_center = -200
    window_width = 2000
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)

    image_min = window_center - window_width//2
    image_max = window_center + window_width//2
    image[image < image_min] = image_min
    image[image > image_max] = image_max
    
    image = image.astype(np.float64)

    image = (image - image_min)/(image_max - image_min) * 255.
    
    return image.astype(np.uint8)

TEST_DF = create_typical_fvc(TEST_DF)

## === cell 5
TEST_DF.head()

## === cell 6
class Dataset(Sequence):
    def __init__(self, batch_size = BATCH_SIZE, mode=0):
        self.indices = np.arange(0, len(SAMPLE_SUBMISSION), 1)
        self.batch_size = batch_size
        self.mode = mode # 0 - Training, 1 - Test
    
    def __len__(self):
        return len(self.indices) // self.batch_size
        
    def get_tabular(self, patient, week):
        tabular = [week]
        for feature in ['Age', 'Sex', 'SmokingStatus', 'typical_fvc']:
            if feature in ['Age', 'typical_fvc']:
                tabular.append(TEST_DF[feature][TEST_DF['Patient'] == patient].values[0])
            else:
                tabular += CATEGORICAL[TEST_DF[feature][TEST_DF['Patient'] == patient].values[0]]
        return np.asarray(tabular, dtype='float32')
    
    def get_random_image(self, patient_id):
        image_folder = f'../input/osic-pulmonary-fibrosis-progression/test/{patient_id}'
        image_files = np.asarray(os.listdir(image_folder))
        if self.mode == 0:
            image_files = image_files[len(image_files) // 5 * 2:len(image_files) // 5 * 3]
            image_file = np.random.choice(image_files)
        else:
            image_file = image_files[len(image_files) // 2]
        scan = dicom.dcmread(os.path.join(image_folder, image_file))
        image = get_pixels_hu(scan)
        image = cv2.resize(image, (IMG_SIZE , IMG_SIZE))
        image = np.expand_dims(image, axis=2)
        return image
    
    def __getitem__(self, index):
        if index == self.__len__() - 1:
            indices = self.indices[index*self.batch_size:]
        else:
            indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        Patient_Week = np.asarray(SAMPLE_SUBMISSION['Patient_Week'][indices])
        patient = [x.split('_')[0] for x in Patient_Week]
        week = [x.split('_')[1] for x in Patient_Week]
        images = np.asarray([self.get_random_image(patient_id) for patient_id in patient], dtype=np.float32) / 255.
        tabulars = np.asarray([self.get_tabular(patient[i], week[i]) for i in range(len(patient))], dtype='float32')
        return [images, tabulars]

## === cell 7
def swish(x):
    return x * K.sigmoid(x)

def build_model(weights):
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    M, _ = Classifiers.get('resnet18')
    resnet18 = M(weights=None, include_top=False, input_tensor=input_img)
    x = GlobalAveragePooling2D()(resnet18.output)
    x = BatchNormalization()(x)
    input_latent = Dense(1, activation=swish)(x)
    
    input_tabular = Input(shape=(num_of_features,))
    x = BatchNormalization()(input_tabular)
    
    x = Concatenate()([input_latent, x])
    x = BatchNormalization()(x)
    x = Dense(200, activation=swish)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(180, activation=swish)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    q1 = Dense(3, activation = 'linear', name = "p1")(x)
    q_adjust = Dense(3, activation = 'linear', name = "p2")(x)
    
    preds = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis = 1), 
                     name = "preds")([q1, q_adjust])
    
    model = tf.keras.Model(inputs = [input_img, input_tabular], outputs = preds)

    model.load_weights(weights)
    return model

## === cell 8
models = [build_model(weights) for weights in model_weights]

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2358572764.py in <cell line: 0>()
----> 1 models = [build_model(weights) for weights in model_weights]

NameError: name 'model_weights' is not defined

## === cell 9
test_gen = Dataset(mode = 1)
predictions = np.zeros((len(SAMPLE_SUBMISSION), 3), dtype='float32')
for model in models:
    predictions += model.predict(test_gen, verbose = 1)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/228078769.py in <cell line: 0>()
      1 test_gen = Dataset(mode = 1)
      2 predictions = np.zeros((len(SAMPLE_SUBMISSION), 3), dtype='float32')
----> 3 for model in models:
      4     predictions += model.predict(test_gen, verbose = 1)

NameError: name 'models' is not defined

## === cell 10
predictions = predictions / len(models)
FVC = predictions[:, 1]
Confidence = predictions[:, 2] - predictions[:, 0]
SAMPLE_SUBMISSION['FVC'] = FVC.astype('int')
SAMPLE_SUBMISSION['Confidence'] = Confidence.astype('int')
SAMPLE_SUBMISSION.to_csv('submission.csv', index=False)
SAMPLE_SUBMISSION.head()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3344782689.py in <cell line: 0>()
----> 1 predictions = predictions / len(models)
      2 FVC = predictions[:, 1]
      3 Confidence = predictions[:, 2] - predictions[:, 0]
      4 SAMPLE_SUBMISSION['FVC'] = FVC.astype('int')
      5 SAMPLE_SUBMISSION['Confidence'] = Confidence.astype('int')

NameError: name 'models' is not defined

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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scipy==1.15.3
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
tqdm==4.67.1
wandb==0.21.0

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

-6.8481

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
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm
import seaborn as sns
import wandb
from wandb.keras import WandbCallback
import keras
from keras.models import Sequential

from pfutils import (get_test_data, get_train_data, get_pseudo_test_data,
                     build_model, get_cosine_annealing_lr_callback, get_fold_indices)

WANDB = False
TEST = True
DATA_GENERATOR = True
TRAIN_ON_BACKWARD_WEEKS = False

PSEUDO_TEST_PATIENTS = 0


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if TEST:
    PSEUDO_TEST_PATIENTS = 0


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2534885708.py in <cell line: 0>()
----> 1 if TEST:
      2     PSEUDO_TEST_PATIENTS = 0

NameError: name 'TEST' is not defined

## === cell 2
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("wandb_key")
assert wandb_key, "Please create a key.txt or Kaggle Secret with your W&B API key"


!pip install -q --upgrade wandb
!wandb login $wandb_key


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_11/1958285507.py in <cell line: 0>()
      2 from kaggle_secrets import UserSecretsClient
      3 user_secrets = UserSecretsClient()
----> 4 wandb_key = user_secrets.get_secret("wandb_key")
      5 assert wandb_key, "Please create a key.txt or Kaggle Secret with your W&B API key"
      6 

/usr/local/lib/python3.11/dist-packages/kaggle_secrets.py in get_secret(self, label)
     62             'Label': label,
     63         }
---> 64         response_json = self.web_client.make_post_request(request_body, self.GET_USER_SECRET_BY_LABEL_ENDPOINT)
     65         if 'secret' not in response_json:
     66             raise BackendError(

/usr/local/lib/python3.11/dist-packages/kaggle_web_client.py in make_post_request(self, data, endpoint, timeout)
     47                 response_json = json.loads(response.read())
     48                 if not response_json.get('wasSuccessful') or 'result' not in response_json:
---> 49                     raise BackendError(
     50                         f'Unexpected response from the service. Response: {response_json}.')
     51                 return response_json['result']

BackendError: Unexpected response from the service. Response: {'errors': ['Unauthenticated'], 'error': {'code': 16}, 'wasSuccessful': False}.

## === cell 4
FOLDS = 10

BATCH_SIZE = 128

NUMBER_FEATURES = 9

HIDDEN_LAYERS = [64,64]

PREDICT_SLOPE = False

USE_GAUSSIAN_ON_FVC = True 
VALUE_GAUSSIAN_NOISE_ON_FVC = 70 # Only needed when Gaussian noise = True
GAUSSIAN_NOISE_CORRELATED = False
                                     
ACTIVATION_FUNCTION = 'swish'

DROP_OUT_RATE = 0
DROP_OUT_LAYERS = [] # [0,1,2] voor dropout in de eerste 3 lagen

EPOCHS = 100
STEPS_PER_EPOCH = 100

L2_REGULARIZATION = False
REGULARIZATION_CONSTANT = 0.005

INPUT_NORMALIZATION = True
OUTPUT_NORMALIZATION = True

MAX_LEARNING_RATE = 5e-4
COSINE_CYCLES = 10

MODEL_NAME = "BaselineSubmission" 

config = dict(NUMBER_FEATURES = NUMBER_FEATURES, L2_REGULARIZATION = L2_REGULARIZATION, INPUT_NORMALIZATION =INPUT_NORMALIZATION,
              ACTIVATION_FUNCTION = ACTIVATION_FUNCTION, DROP_OUT_RATE = DROP_OUT_RATE, OUTPUT_NORMALIZATION = OUTPUT_NORMALIZATION,
              EPOCHS = EPOCHS, STEPS_PER_EPOCH = STEPS_PER_EPOCH, MAX_LEARNING_RATE = MAX_LEARNING_RATE,
              COSINE_CYCLES = COSINE_CYCLES, MODEL_NAME=MODEL_NAME, USE_GAUSSIAN_ON_FVC=USE_GAUSSIAN_ON_FVC,
              VALUE_GAUSSIAN_NOISE_ON_FVC=VALUE_GAUSSIAN_NOISE_ON_FVC, PREDICT_SLOPE = PREDICT_SLOPE,
              HIDDEN_LAYERS = HIDDEN_LAYERS, REGULARIZATION_CONSTANT = REGULARIZATION_CONSTANT,
              DROP_OUT_LAYERS = DROP_OUT_LAYERS, BATCH_SIZE = BATCH_SIZE, GAUSSIAN_NOISE_CORRELATED = GAUSSIAN_NOISE_CORRELATED )


## === cell 5
if TEST:
    test_data, submission = get_test_data("../input/osic-pulmonary-fibrosis-progression/test.csv", INPUT_NORMALIZATION)
    
train, data, labels = get_train_data('../input/osic-pulmonary-fibrosis-progression/train.csv', PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION, TRAIN_ON_BACKWARD_WEEKS)

if PSEUDO_TEST_PATIENTS > 0:
    test_data, test_check = get_pseudo_test_data('../input/osic-pulmonary-fibrosis-progression/train.csv', PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/939407962.py in <cell line: 0>()
----> 1 if TEST:
      2     test_data, submission = get_test_data("../input/osic-pulmonary-fibrosis-progression/test.csv", INPUT_NORMALIZATION)
      3 
      4 train, data, labels = get_train_data('../input/osic-pulmonary-fibrosis-progression/train.csv', PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION, TRAIN_ON_BACKWARD_WEEKS)
      5 

NameError: name 'TEST' is not defined

## === cell 6
model = build_model(config)
model.summary()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/938948086.py in <cell line: 0>()
----> 1 model = build_model(config)
      2 #tf.keras.utils.plot_model(model)
      3 model.summary()

NameError: name 'build_model' is not defined

## === cell 7
fold_pos = get_fold_indices(FOLDS, train)
print(fold_pos)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2419524396.py in <cell line: 0>()
----> 1 fold_pos = get_fold_indices(FOLDS, train)
      2 print(fold_pos)

NameError: name 'get_fold_indices' is not defined

## === cell 8
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, config, validation = False, number_of_labels = 3,
                 batch_size = 128, shuffle = True):
        self.number_features = int(config["NUMBER_FEATURES"])
        self.use_gaussian = config["USE_GAUSSIAN_ON_FVC"]
        self.gauss_std = config["VALUE_GAUSSIAN_NOISE_ON_FVC"] and not validation
        self.list_IDs = list_IDs
        self.batch_size = config["BATCH_SIZE"]
        self.labels = labels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.label_size = number_of_labels
        self.normalized = config["INPUT_NORMALIZATION"]
        self.correlated = config["GAUSSIAN_NOISE_CORRELATED"]
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, self.number_features))
        y = np.empty((self.batch_size, self.label_size), dtype=int)
        
        data = np.load("./train_data.npy", allow_pickle = True)
        lab = np.load("./train_labels.npy", allow_pickle = True)
        
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.asarray(data[ID], dtype = "float32")
            y[i,] = np.asarray(lab[ID], dtype = "float32")
        y = np.asarray(y,dtype = "float32")
            
        gauss_X = np.random.normal(0, self.gauss_std, size = self.batch_size)
        
        if self.correlated:
            gauss_y = gauss_X
        else:
            gauss_y = np.random.normal(0, self.gauss_std, size = self.batch_size)
        if self.normalized:
            gauss_X = gauss_X/5000 
            
        X[:,1] += gauss_X.astype("float32")
        y[:,2] += gauss_X.astype("float32")
        y[:,0] += gauss_y.astype("float32")
        
        return X, y


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3336470573.py in <cell line: 0>()
----> 1 class DataGenerator(keras.utils.Sequence):
      2     def __init__(self, list_IDs, config, validation = False, number_of_labels = 3,
      3                  batch_size = 128, shuffle = True):
      4         self.number_features = int(config["NUMBER_FEATURES"])
      5         self.use_gaussian = config["USE_GAUSSIAN_ON_FVC"]

NameError: name 'keras' is not defined

## === cell 9
if DATA_GENERATOR:
    train_data = train[["Weeks", "FVC", "Percent", "Age", "Sex", 
                                 "Currently smokes", "Ex-smoker", "Never smoked", "Weekdiff_target"]]
    train_labels = labels
    np.save("train_data.npy", train_data.to_numpy())
    np.save("train_labels.npy", train_labels.to_numpy())


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4076068985.py in <cell line: 0>()
----> 1 if DATA_GENERATOR:
      2     train_data = train[["Weeks", "FVC", "Percent", "Age", "Sex", 
      3                                  "Currently smokes", "Ex-smoker", "Never smoked", "Weekdiff_target"]]
      4     train_labels = labels
      5     np.save("train_data.npy", train_data.to_numpy())

NameError: name 'DATA_GENERATOR' is not defined

## === cell 10
predictions = []

for fold in range(FOLDS):
    if DATA_GENERATOR:
        train_ID = list(range(fold_pos[0],fold_pos[fold])) + list(range(fold_pos[fold+1],len(train)))
        val_ID = list(range(fold_pos[fold], fold_pos[fold+1]))
        training_generator = DataGenerator(train_ID, config)
        validation_generator = DataGenerator(val_ID, config, validation = True)
    else:
        x_train = data["input_features"][:fold_pos[fold]].append(data["input_features"][fold_pos[fold+1]:])
        y_train = labels[:fold_pos[fold]].append(labels[fold_pos[fold+1]:])
        x_val = data["input_features"][fold_pos[fold]:fold_pos[fold+1]]
        y_val = labels[fold_pos[fold]:fold_pos[fold+1]]
    
    model = build_model(config)
    
    sv = tf.keras.callbacks.ModelCheckpoint(
    'fold-%i.h5'%fold, monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='min', save_freq='epoch')
    callbacks = [sv]

    print(fold+1, "of", FOLDS)
    if WANDB:
        name = MODEL_NAME + '-F{}'.format(fold+1)
        config.update({'fold': fold+1})
        wandb.init(project="pulfib", name=name, config=config)
        wandb_cb = WandbCallback()
        callbacks.append(wandb_cb)
        
    if DATA_GENERATOR:
        history = model.fit(training_generator, validation_data = validation_generator, epochs = EPOCHS,
                            verbose = 0, callbacks = callbacks)
    else:
        history = model.fit(x_train, y_train, validation_data = (x_val,y_val), epochs = EPOCHS,
                            steps_per_epoch = STEPS_PER_EPOCH, verbose = 0, callbacks = callbacks)

    if TEST or PSEUDO_TEST_PATIENTS > 0:
        model.load_weights('fold-%i.h5'%fold)
        predictions.append(model.predict(test_data, batch_size = 256))
    
    if WANDB:
        wandb.join()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3526329267.py in <cell line: 0>()
      2 
      3 for fold in range(FOLDS):
----> 4     if DATA_GENERATOR:
      5         train_ID = list(range(fold_pos[0],fold_pos[fold])) + list(range(fold_pos[fold+1],len(train)))
      6         val_ID = list(range(fold_pos[fold], fold_pos[fold+1]))

NameError: name 'DATA_GENERATOR' is not defined

## === cell 11
if TEST:
    if PREDICT_SLOPE:
        predictions = np.mean(predictions,axis = 0)
        for i in range(1,len(test_data)+1):
            submission.loc[i,"FVC"] = test_data.loc[i-1,"FVC"] + predictions[i-1,0]*test_data.loc[i-1,"Weekdiff_target"]
            submission.loc[i, "Confidence"] = abs(predictions[i-1,1]*test_data.loc[i-1,"Weekdiff_target"])
    else:
        predictions = np.abs(predictions)
        predictions[:,:,1] = np.power(predictions[:,:,1],2)
        predictions = np.mean(predictions, axis = 0)
        predictions[:,1] = np.power(predictions[:,1],0.5)
        for i in range(1,len(test_data)+1):
            submission.loc[i,"FVC"] = predictions[i-1,0]
            submission.loc[i, "Confidence"] = predictions[i-1,1]
    submission.to_csv("submission.csv", index = False)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/223538855.py in <cell line: 0>()
----> 1 if TEST:
      2     if PREDICT_SLOPE:
      3         predictions = np.mean(predictions,axis = 0)
      4         for i in range(1,len(test_data)+1):
      5             submission.loc[i,"FVC"] = test_data.loc[i-1,"FVC"] + predictions[i-1,0]*test_data.loc[i-1,"Weekdiff_target"]

NameError: name 'TEST' is not defined

## === cell 12
backup = predictions


## === cell 13
import matplotlib.pyplot as plt
from scipy.stats import gmean
if PSEUDO_TEST_PATIENTS > 0:
    result = []
    for i in range(-20,20):
        predictions = np.abs(backup)
        if i == 0:
            predictions[:,:,1] = gmean(predictions[:,:,1], axis = 0)
            predictions = np.mean(predictions, axis = 0)
        else:
            predictions[:,:,1] = np.power(predictions[:,:,1],i)
            predictions = np.mean(predictions, axis = 0)
            predictions[:,1] = np.power(predictions[:,1],1/i)
        FVC_true = test_check["TargetFVC"].values
        FVC_pred = predictions[:,0]
        sigma = predictions[:,1]

        sigma_clip = np.maximum(np.abs(sigma), 70)
        delta = np.abs(FVC_true - FVC_pred)
        delta = np.minimum(delta, 1000)

        sq2 = np.sqrt(2)
        loss = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip * sq2)
        result.append(np.mean(loss))
    plt.plot(np.arange(-20,20),result)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1049405525.py in <cell line: 0>()
      1 import matplotlib.pyplot as plt
      2 from scipy.stats import gmean
----> 3 if PSEUDO_TEST_PATIENTS > 0:
      4     result = []
      5     for i in range(-20,20):

NameError: name 'PSEUDO_TEST_PATIENTS' is not defined

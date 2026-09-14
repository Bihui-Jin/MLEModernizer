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
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

0.1638596430578128

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from sklearn.model_selection import KFold

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.preprocessing import RobustScaler
rc = RobustScaler()

## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"

## === cell 3
def dropCols(df, cols):
    df = df.copy()
    df.drop(cols, axis=1, inplace=True)
    return df

## === cell 4
def preProcess(df):   
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1).fillna(0)
    df['diff_u_in1'] = df['u_in'] - df['u_in_lag1']    
    df['u_in_cumsum'] = df['u_in'].groupby(df['breath_id']).cumsum()
    return df

## === cell 5
train_data = pd.read_csv(train_path)
train_data = preProcess(train_data)

## === cell 6
cols_2_drop = ['id', 'breath_id', 'time_step']

## === cell 7
train_df = dropCols(train_data, cols_2_drop)

Y = train_df.pop('pressure')

## === cell 8
train_df.shape

## === cell 9
train_df.isna().sum()

## === cell 10
rc.fit(train_df)
train_df = rc.transform(train_df)

## === cell 11
train_df = train_df.reshape(-1, 80, train_df.shape[-1])
Y = Y.values.reshape(-1, 80, 1)

## === cell 12
train_df.shape, Y.shape

## === cell 13
def build_model0(input_shape, name):
    model = tf.keras.Sequential(name=name)
    
    model.add(layers.Bidirectional(layers.LSTM(440, return_sequences=True, input_shape=input_shape)))
    model.add(layers.Bidirectional(layers.LSTM(360, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(260, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(180, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True)))
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, name=name))
    return model
              
def build_model1(input_shape, name):
    model = tf.keras.Sequential(name=name)
    
    model.add(layers.Bidirectional(layers.LSTM(440, return_sequences=True, input_shape=input_shape)))
    model.add(layers.Bidirectional(layers.LSTM(360, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(260, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(180, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(100, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(80, return_sequences=True)))
    
    model.add(layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dense(1, name=name))
    return model

def build_model2(input_shape, name):
    model = tf.keras.Sequential(name=name)
    
    model.add(layers.Bidirectional(layers.LSTM(420, return_sequences=True, input_shape=input_shape)))
    model.add(layers.Bidirectional(layers.LSTM(340, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(260, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(180, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(100, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(20, return_sequences=True)))
    
    model.add(layers.TimeDistributed(layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())))
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))
    return model

def build_model3(input_shape, name):
    model = tf.keras.Sequential(name=name)
    
    model.add(layers.Bidirectional(layers.LSTM(280, return_sequences=True, input_shape=input_shape)))
    model.add(layers.Bidirectional(layers.LSTM(200, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(160, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(140, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(100, dropout=0.2, return_sequences=True, kernel_initializer='random_normal')))
    model.add(layers.Bidirectional(layers.LSTM(60, return_sequences=True)))
    
    model.add(layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dense(1, name=name))
    return model

              
def build_ensembleModel0(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(layers.Dense(8, activation='relu',  kernel_initializer=tf.keras.initializers.HeNormal(), input_shape=input_shape))
    model.add(layers.Dense(1, name=name))
    return model

def build_ensembleModel1(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(layers.Dense(8, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), input_shape=input_shape))
    model.add(layers.Dense(1, name=name))
    return model

def buildMasterEnsemble(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(layers.Dense(4, activation='relu',  kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dense(1, name=name))
    return model

## === cell 14
def build_model(input_size=[80, train_df.shape[-1]]):
    model0 = build_model0(input_size, name='model0')
    model1 = build_model1(input_size, name='model1')
    model2 = build_model2(input_size, name='model2')
    model3 = build_model2(input_size, name='model3')
    
    ensemblemodel0 = build_ensembleModel0([80,5], name='ensemble0')
    ensemblemodel1 = build_ensembleModel1([80,5], name='ensemble1')
    masterensemble = buildMasterEnsemble([80,2], name='master_ensemble')
    
    Input = tf.keras.layers.Input(input_size)
    
    model0_out = model0(Input)
    model1_out = model1(Input)
    model2_out = model2(Input)
    model3_out = model3(Input)
    
    mean_out = layers.Add()([model0_out, model1_out, model2_out, model3_out])
    mean_out = layers.Lambda(lambda x:x/4, name="mean")(mean_out)
    
    concatenateModelsout = layers.Concatenate()([model0_out, model1_out, model2_out, model3_out, mean_out])
    ensemble_out0 = ensemblemodel0(concatenateModelsout)
    ensemble_out1 = ensemblemodel1(concatenateModelsout)
    
    concatenateensemble_out = layers.Concatenate()([ensemble_out0, ensemble_out1])
    master_ensemble_out = masterensemble(concatenateensemble_out)
    
    model = tf.keras.Model(inputs=Input, outputs=[model0_out, model1_out, model2_out, model3_out, mean_out, ensemble_out0, ensemble_out1, master_ensemble_out])
    
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.MeanAbsoluteError(),
                  loss_weights={
                                'model0': 1.0,
                                'model1': 1.0,
                                'model2': 1.0,
                                'model3': 1.0,
                                'mean':0.8,
                                'ensemble0':0.6,
                                'ensemble1':0.6,
                                'master_ensemble':1
                               }
                 )
    
    return model
    

## === cell 15
build_model().summary()

## === cell 16
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_df, Y, test_size=0.2)

## === cell 17
lr_reducer_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.8,
    patience=25,
    verbose=1,
)

callback1 = tf.keras.callbacks.ModelCheckpoint(f"Bestmodel0.h5", 
                                   monitor='val_model0_loss',save_best_only=True, verbose=1)
callback2 = tf.keras.callbacks.ModelCheckpoint(f"Bestmodel1.h5", 
                                   monitor='val_model1_loss',save_best_only=True, verbose=1)
callback3 = tf.keras.callbacks.ModelCheckpoint(f"Bestmodel2.h5", 
                                   monitor='val_model2_loss',save_best_only=True, verbose=1)
callback4 = tf.keras.callbacks.ModelCheckpoint(f"Bestmodel3.h5", 
                                   monitor='val_model3_loss',save_best_only=True, verbose=1)

callback5 = tf.keras.callbacks.ModelCheckpoint(f"Bestensemble0.h5", 
                                   monitor='val_ensemble0_loss',save_best_only=True, verbose=1)
callback6 = tf.keras.callbacks.ModelCheckpoint(f"Bestensemble1.h5", 
                                   monitor='val_ensemble1_loss',save_best_only=True, verbose=1)

callback7 = tf.keras.callbacks.ModelCheckpoint(f"BestEnsembleModel.h5", 
                                   monitor='val_master_ensemble_loss',save_best_only=True, verbose=1)
callback8 = tf.keras.callbacks.ModelCheckpoint(f"BestMeanModel.h5", 
                                   monitor='val_mean_loss',save_best_only=True, verbose=1)

callbacks = [lr_reducer_callback, callback1, callback2, callback3, callback4, callback5, callback6, callback7, callback8]

## === cell 18
"""EPOCH = 500
BATCH_SIZE = 1024

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():
    model = build_model()
    his = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks)"""

## === cell 19
best_model0 = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/Bestmodel0.h5')
best_model1 = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/Bestmodel1.h5')
best_model2 = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/Bestmodel2.h5')
best_model3 = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/Bestmodel3.h5')

best_ensemble0 = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/Bestensemble0.h5')
best_ensemble1 = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/Bestensemble1.h5')

best_mean =  tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/BestMeanModel.h5')

best_master_ensemble = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/BestEnsembleModel.h5')

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2692363172.py in <cell line: 0>()
----> 1 best_model0 = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/Bestmodel0.h5')
      2 best_model1 = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/Bestmodel1.h5')
      3 best_model2 = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/Bestmodel2.h5')
      4 best_model3 = tf.keras.models.load_model('../input/ventilator-pressure-predictor-tpu/Bestmodel3.h5')
      5 

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/ventilator-pressure-predictor-tpu/Bestmodel0.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 20
test_data = pd.read_csv(test_path)
 
test_data = preProcess(test_data)

test_data = dropCols(test_data, cols_2_drop)
test_data = rc.transform(test_data)
test_data = test_data.reshape(-1, 80, test_data.shape[-1])

## === cell 21
pred_best_model0 = best_model0.predict(test_data,verbose=1)[0].reshape(-1, 1)
pred_best_model1 = best_model1.predict(test_data,verbose=1)[1].reshape(-1, 1)
pred_best_model2 = best_model2.predict(test_data,verbose=1)[2].reshape(-1, 1)
pred_best_model3 = best_model3.predict(test_data,verbose=1)[3].reshape(-1, 1)

pred_best_ensemble0 = best_ensemble0.predict(test_data,verbose=1)[5].reshape(-1, 1)
pred_best_ensemble1 = best_ensemble1.predict(test_data,verbose=1)[6].reshape(-1, 1)

pred_best_mean = best_mean.predict(test_data,verbose=1)[4].reshape(-1, 1)

pred_best_master_ensemble = best_master_ensemble.predict(test_data,verbose=1)[7].reshape(-1, 1)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3051086227.py in <cell line: 0>()
----> 1 pred_best_model0 = best_model0.predict(test_data,verbose=1)[0].reshape(-1, 1)
      2 pred_best_model1 = best_model1.predict(test_data,verbose=1)[1].reshape(-1, 1)
      3 pred_best_model2 = best_model2.predict(test_data,verbose=1)[2].reshape(-1, 1)
      4 pred_best_model3 = best_model3.predict(test_data,verbose=1)[3].reshape(-1, 1)
      5 

NameError: name 'best_model0' is not defined

## === cell 22
predictions_table = np.concatenate([pred_best_model0,pred_best_model1,pred_best_model2,pred_best_model3,pred_best_ensemble0,pred_best_ensemble1,pred_best_mean,pred_best_master_ensemble], axis=-1)
median_pre = np.median(predictions_table, axis = -1)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4023795671.py in <cell line: 0>()
----> 1 predictions_table = np.concatenate([pred_best_model0,pred_best_model1,pred_best_model2,pred_best_model3,pred_best_ensemble0,pred_best_ensemble1,pred_best_mean,pred_best_master_ensemble], axis=-1)
      2 median_pre = np.median(predictions_table, axis = -1)

NameError: name 'pred_best_model0' is not defined

## === cell 23
median_pre

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2956535175.py in <cell line: 0>()
----> 1 median_pre

NameError: name 'median_pre' is not defined

## === cell 24
median_pre.shape

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2997329941.py in <cell line: 0>()
----> 1 median_pre.shape

NameError: name 'median_pre' is not defined

## === cell 25
unique_pressures = np.unique(Y)

## === cell 26
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()

PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

## === cell 27
rounding_pre = np.round((median_pre - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1675011179.py in <cell line: 0>()
----> 1 rounding_pre = np.round((median_pre - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
      2 #Rounding With Median of Model

NameError: name 'median_pre' is not defined

## === cell 28
rounding_pre.shape

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/283169564.py in <cell line: 0>()
----> 1 rounding_pre.shape

NameError: name 'rounding_pre' is not defined

## === cell 29
submission_file = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
submission_file['pressure'] = rounding_pre
submission_file.to_csv('submission.csv', index=False)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1691462736.py in <cell line: 0>()
      1 submission_file = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
----> 2 submission_file['pressure'] = rounding_pre
      3 submission_file.to_csv('submission.csv', index=False)

NameError: name 'rounding_pre' is not defined

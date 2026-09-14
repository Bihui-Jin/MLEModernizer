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

0.9801976631902684

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

from sklearn.model_selection import KFold

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

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
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df = df.fillna(0)
    
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    

    
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
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="C_out_loss",
    factor=0.96,
    patience=4,
    verbose=1,
)

## === cell 14
def build_model():
    
    Input = layers.Input(shape=([80, train_df.shape[-1]]))
    
    m1x1 = layers.Bidirectional(layers.LSTM(units=440, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    m1x2 = layers.Bidirectional(layers.LSTM(units=360, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    m1x3 = layers.Bidirectional(layers.LSTM(units=240, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    m1x4 = layers.Bidirectional(layers.LSTM(units=180, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    m1x5 = layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    
    m2x1 = layers.Bidirectional(layers.LSTM(units=880, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    m2x3 = layers.Bidirectional(layers.LSTM(units=440, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))#
    m2x5 = layers.Bidirectional(layers.LSTM(units=360, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))#
    m2x7 = layers.Bidirectional(layers.LSTM(units=240, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))#
    m2x9 = layers.Bidirectional(layers.LSTM(units=180, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))#
    m2x11 = layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    
    m3x1 = layers.Bidirectional(layers.GRU(units=440, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    m3x2 = layers.Bidirectional(layers.GRU(units=360, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    m3x3 = layers.Bidirectional(layers.GRU(units=240, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    m3x4 = layers.Bidirectional(layers.GRU(units=180, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    m3x5 = layers.Bidirectional(layers.GRU(units=100, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    
    m3x1_out = m3x1(Input)
    m3x2_out = m3x2(m3x1_out)
    m3x3_out = m3x3(m3x2_out)
    m3x4_out = m3x4(m3x3_out)
    m3x5_out = m3x5(m3x4_out)
    
    
    m2x1_out = m2x1(Input)
    m2x3_out = m2x3(m2x1_out)
    m2x5_out = m2x5(m2x3_out)
    m2x7_out = m2x7(m2x5_out)
    m2x9_out = m2x9(m2x7_out)
    m2x11_out = m2x11(m2x9_out)
    
    m1x1_out = m1x1(Input)
    m1x1_out = layers.Multiply()([m1x1_out, m2x3_out, m3x1_out])
    m1x1_out = layers.BatchNormalization()(m1x1_out)
    
    m1x2_out = m1x2(m1x1_out)
    m1x2_out = layers.Multiply()([m1x2_out, m2x5_out, m3x2_out])
    m1x2_out = layers.BatchNormalization()(m1x2_out)
    
    m1x3_out = m1x3(m1x2_out)
    m1x3_out = layers.Multiply()([m1x3_out, m2x7_out, m3x3_out])
    m1x3_out = layers.BatchNormalization()(m1x3_out)
    
    m1x4_out = m1x4(m1x3_out)
    m1x4_out = layers.Multiply()([m1x4_out, m2x9_out, m3x4_out])
    m1x4_out = layers.BatchNormalization()(m1x4_out)
    
    m1x5_out = m1x5(m1x4_out)

    f_mul = layers.Multiply()([m1x5_out, m2x11_out, m3x5_out])
    f_mul = layers.BatchNormalization()(f_mul)
    f_mul = layers.Bidirectional(layers.LSTM(units=100, dropout=0.2, return_sequences=True))(f_mul)
    
    
    C_out = layers.Dense(units=64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(f_mul)
    C_out = layers.Dense(1, name="C_out", kernel_initializer=tf.keras.initializers.HeNormal())(C_out)
    
    m1_out = layers.Dense(units=64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(m1x5_out)
    m1_out = layers.Dense(1, name='m1_out', kernel_initializer=tf.keras.initializers.HeNormal())(m1_out)
        
    model = tf.keras.Model(inputs=Input, outputs=[C_out, m1_out], name='Base_Model')
    
    losses = {
        "C_out":tf.keras.losses.MeanAbsoluteError(),
        "m1_out":tf.keras.losses.MeanAbsoluteError()
    }
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=losses)
    
    return model

## === cell 15
test_model = build_model()
test_model.summary()

## === cell 16
tf.keras.utils.plot_model(test_model)

## === cell 17
EPOCH = 300
BATCH_SIZE = 512

tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

with tpu_strategy.scope():
    kf = KFold(n_splits=3, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_df, Y)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train_df[train_idx], train_df[test_idx]

        y_train, y_valid = Y[train_idx], Y[test_idx]

        model = build_model()

        callback0 = tf.keras.callbacks.ModelCheckpoint(f"M1_PressurePreModel{fold+1}.h5", 
                                               monitor='val_m1_out_loss',save_best_only=True, verbose=1)
        callback1 = tf.keras.callbacks.ModelCheckpoint(f"C_PressurePreModel{fold+1}.h5", 
                               monitor='val_C_out_loss',save_best_only=True, verbose=1)

        callbacks = [lr_callback, callback0, callback1]

        his = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks) 

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4252796403.py in <cell line: 0>()
      3 
      4 # detect and init the TPU
----> 5 tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
      6 
      7 # instantiate a distribution strategy

/usr/local/lib/python3.11/dist-packages/tensorflow/python/distribute/cluster_resolver/tpu/tpu_cluster_resolver.py in connect(tpu, zone, project)
    143       NotFoundError: If no TPU devices found in eager mode.
    144     """
--> 145     resolver = TPUClusterResolver(tpu, zone, project)
    146     remote.connect_to_cluster(resolver)
    147     tpu_strategy_util.initialize_tpu_system_impl(resolver, TPUClusterResolver)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/distribute/cluster_resolver/tpu/tpu_cluster_resolver.py in __init__(self, tpu, zone, project, job_name, coordinator_name, coordinator_address, credentials, service, discovery_url)
    233     if tpu != 'local':
    234       # Default Cloud environment
--> 235       self._cloud_tpu_client = client.Client(
    236           tpu=tpu,
    237           zone=zone,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/tpu/client/client.py in __init__(self, tpu, zone, project, credentials, service, discovery_url)
    157         zone = zone or tpu_node_config.get('zone')
    158       else:
--> 159         raise ValueError('Please provide a TPU Name to connect to.')
    160 
    161     self._tpu = _as_text(tpu)

ValueError: Please provide a TPU Name to connect to.

## === cell 18
models_paths = ['../input/ventilator-pressure-predictor-tpu/M1_PressurePreModel1.h5',
                '../input/ventilator-pressure-predictor-tpu/M1_PressurePreModel2.h5']
 
models = [tf.keras.models.load_model(model_path) for model_path in models_paths]

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2959633835.py in <cell line: 0>()
      2                 '../input/ventilator-pressure-predictor-tpu/M1_PressurePreModel2.h5']
      3 
----> 4 models = [tf.keras.models.load_model(model_path) for model_path in models_paths]

/tmp/ipykernel_11/2959633835.py in <listcomp>(.0)
      2                 '../input/ventilator-pressure-predictor-tpu/M1_PressurePreModel2.h5']
      3 
----> 4 models = [tf.keras.models.load_model(model_path) for model_path in models_paths]

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/ventilator-pressure-predictor-tpu/M1_PressurePreModel1.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 19
test_data = pd.read_csv(test_path)
test_data = preProcess(test_data)

test_data = dropCols(test_data, cols_2_drop)
test_data = rc.transform(test_data)
test_data = test_data.reshape(-1, 80, test_data.shape[-1])

## === cell 20
p = []

for model in models:
    p.append(model.predict(test_data, verbose=1))

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1253228363.py in <cell line: 0>()
      1 p = []
      2 
----> 3 for model in models:
      4     p.append(model.predict(test_data, verbose=1))

NameError: name 'models' is not defined

## === cell 21
predictions_table = np.concatenate(p[0], axis=-1)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3149501937.py in <cell line: 0>()
----> 1 predictions_table = np.concatenate(p[0], axis=-1)

IndexError: list index out of range

## === cell 22
predictions_table.shape

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2581216540.py in <cell line: 0>()
----> 1 predictions_table.shape

NameError: name 'predictions_table' is not defined

## === cell 23
median_pre = np.median(predictions_table, axis = -1)
mean_pre = np.mean(predictions_table, axis = -1)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3701249573.py in <cell line: 0>()
----> 1 median_pre = np.median(predictions_table, axis = -1)
      2 mean_pre = np.mean(predictions_table, axis = -1)

NameError: name 'predictions_table' is not defined

## === cell 24
unique_pressures = np.unique(Y)

## === cell 25
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()

PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

## === cell 26
PRESSURE_STEP,PRESSURE_MIN,PRESSURE_MAX

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
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX).reshape(-1,1)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/451074747.py in <cell line: 0>()
----> 1 clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX).reshape(-1,1)

NameError: name 'rounding_pre' is not defined

## === cell 30
clipped_pre.shape

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/368461661.py in <cell line: 0>()
----> 1 clipped_pre.shape

NameError: name 'clipped_pre' is not defined

## === cell 31
submission_file = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
submission_file['pressure'] = clipped_pre
submission_file.to_csv('submission_mediam_round.csv', index=False)

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3040250209.py in <cell line: 0>()
      1 submission_file = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
----> 2 submission_file['pressure'] = clipped_pre
      3 submission_file.to_csv('submission_mediam_round.csv', index=False)

NameError: name 'clipped_pre' is not defined

## === cell 32
submission_file

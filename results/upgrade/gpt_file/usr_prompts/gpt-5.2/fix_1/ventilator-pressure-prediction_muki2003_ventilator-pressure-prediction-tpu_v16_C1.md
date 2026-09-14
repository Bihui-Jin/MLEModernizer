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

0.145836685020238

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau

from sklearn.model_selection import KFold,GroupKFold
from tensorflow.keras import layers

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
rb = RobustScaler()
sc = StandardScaler()

## === cell 3
train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')

## === cell 5
train['u_in_lag1'] = train.groupby('breath_id')['u_in'].shift(1).fillna(0)
train['u_in_diff1'] = train['u_in'] - train['u_in_lag1']
train['u_in_cumsum'] = train['u_in'].groupby(train['breath_id']).cumsum()

test['u_in_lag1'] = test.groupby('breath_id')['u_in'].shift(1).fillna(0)
test['u_in_diff1'] = test['u_in'] - test['u_in_lag1']
test['u_in_cumsum'] = test['u_in'].groupby(test['breath_id']).cumsum()



## === cell 6
train

## === cell 7
targets = train['pressure'].to_numpy().reshape(-1, 80, 1)
test_id = test['id']
train.drop(columns = ['id','breath_id','pressure','time_step'], inplace = True)
test.drop(columns = ['id','breath_id','time_step'], inplace = True)

## === cell 8
rb.fit(train)

train_new = rb.transform(train)
test_new = rb.transform(test)

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])

## === cell 10
def build_model_n():
    input_layer = layers.Input(shape=([80, train_re.shape[-1]]))
    
    lstm0 = layers.Bidirectional(layers.LSTM(440, return_sequences=True, name='lstm0', kernel_initializer='random_normal'))
    lstm1 = layers.Bidirectional(layers.LSTM(360, dropout=0.2, return_sequences=True, name='lstm1', kernel_initializer='random_normal'))
    lstm2 = layers.Bidirectional(layers.LSTM(240, dropout=0.2, return_sequences=True , name='lstm2', kernel_initializer='random_normal'))
    lstm3 = layers.Bidirectional(layers.LSTM(180, dropout=0.2, return_sequences=True, name='lstm3', kernel_initializer='random_normal'))
    lstm4 = layers.Bidirectional(layers.LSTM(100, return_sequences=True, name='lstm4', kernel_initializer='random_normal'))
    
    lstm_i1 = layers.Bidirectional(layers.LSTM(360, dropout=0.2, return_sequences=True, name='lstm_i1',kernel_initializer='random_normal'))
    lstm_02 = layers.Bidirectional(layers.LSTM(240, dropout=0.2, return_sequences=True , name='lstm_02', kernel_initializer='random_normal'))
    lstm_13 = layers.Bidirectional(layers.LSTM(180, dropout=0.2, return_sequences=True, name='lstm_13', kernel_initializer='random_normal'))
    
    dense0 = layers.Dense(64, activation='relu', name='dense0', kernel_initializer='random_normal')
    dense1 = layers.Dense(1, name='dense1')
    
    
    lstm0_out = lstm0(input_layer)
    lstm_i1_out = lstm_i1(input_layer)
    
    lstm1_out = lstm1(lstm0_out)
    lstm_02_out = lstm_02(lstm0_out)
    
    lstm_13_out = lstm_13(lstm1_out)
    lstm2_out = lstm2(layers.BatchNormalization()(layers.Multiply()([lstm1_out, lstm_i1_out])))
    
    lstm3_out = lstm3(layers.BatchNormalization()(layers.Multiply()([lstm2_out, lstm_02_out])))
    
    lstm4_out = lstm4(layers.BatchNormalization()(layers.Multiply()([lstm3_out, lstm_13_out])))
    
    dense0_out = dense0(lstm4_out)
    dense1_out = dense1(dense0_out)
    
    model = tf.keras.Model(inputs=input_layer, outputs=dense1_out, name='BasicModel')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), loss=tf.keras.losses.MeanAbsoluteError())
    
    return model

## === cell 11
def build_model():
    
    x_input = layers.Input(shape=([80, train_re.shape[-1]]))
    
    x1 = layers.Bidirectional(layers.LSTM(units=440, return_sequences=True))(x_input)
    x2 = layers.Bidirectional(layers.LSTM(units=360, dropout=0.2,return_sequences=True, kernel_initializer='random_normal'))(x1)
    x3 = layers.Bidirectional(layers.LSTM(units=240, dropout=0.2, return_sequences=True, kernel_initializer='random_normal'))(x2)
    x4 = layers.Bidirectional(layers.LSTM(units=180, dropout=0.2, return_sequences=True, kernel_initializer='random_normal'))(x3)
    x5 = layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, kernel_initializer='random_normal'))(x4)
    
    z2 = layers.Bidirectional(layers.GRU(units=240, return_sequences=True))(x2)
    
    z31 = layers.Multiply()([x3, z2])
    z31 = layers.BatchNormalization()(z31)
    z3 = layers.Bidirectional(layers.GRU(units=180, return_sequences=True))(z31)
    
    z41 = layers.Multiply()([x4, z3])
    z41 = layers.BatchNormalization()(z41)
    z4 = layers.Bidirectional(layers.GRU(units=100, return_sequences=True))(z41)
    
    z51 = layers.Multiply()([x5, z4])
    z51 = layers.BatchNormalization()(z51)
    z5 = layers.Bidirectional(layers.GRU(units=64, return_sequences=True))(z51)
    
    x = layers.Concatenate(axis=2)([x5, z2, z3, z4, z5])
    
    x = layers.Dense(units=64, activation='relu')(x)
    
    x_output = layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=x_input, outputs=x_output, name='Base_Model')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), loss=tf.keras.losses.MeanAbsoluteError())
    
    return model

## === cell 12
tf.keras.utils.plot_model(build_model())

## === cell 14
EPOCH = 200
BATCH_SIZE = 512

tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/323999716.py in <cell line: 0>()
      2 BATCH_SIZE = 512
      3 
----> 4 tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
      5 
      6 tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

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

## === cell 15
import math
def exp_decay(epoch):
    initial_lrate = 0.001
    k = 0.006
    lrate = initial_lrate * math.exp(-k*epoch)
    print(f"Learning rate : {lrate}")
    return lrate

lr_schedular = tf.keras.callbacks.LearningRateScheduler(exp_decay)

## === cell 16
reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.85, patience=7)

with tpu_strategy.scope():
    kf = KFold(n_splits=8, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_re, targets)):
        print('-'*30, '>', f'Fold {fold+1}', '<', '-'*30)
        X_train, X_valid = train_re[train_idx], train_re[test_idx]

        y_train, y_valid = targets[train_idx], targets[test_idx]
        
        model = build_model()

        call_back = tf.keras.callbacks.ModelCheckpoint(f"Model{fold+1}.h5", verbose=1, monitor='val_loss',save_best_only=True)

        his = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[call_back,reduce_lr])
        stats = pd.DataFrame(his.history)
        stats.plot()
        plt.show()
        print("\n\n")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4238209375.py in <cell line: 0>()
      1 reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.85, patience=7)
      2 
----> 3 with tpu_strategy.scope():
      4     kf = KFold(n_splits=8, shuffle=True)
      5 

NameError: name 'tpu_strategy' is not defined

## === cell 18
unique_pressures = np.unique(targets)

sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()

PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

## === cell 19
models_paths = ["../input/kfolds/OldModel1.h5","../input/kfolds/OldModel2.h5","../input/kfolds/OldModel3.h5",
                "../input/kfolds/OldModel4.h5","../input/kfolds/OldModel5.h5","../input/kfolds/OldModel6.h5",
                "../input/kfolds/OldModel7.h5","../input/kfolds/OldModel8.h5","../input/kfolds/OldModel9.h5",
                "../input/kfolds/Model16614.h5","../input/kfolds/Model16517.h5","../input/kfolds/Model16468.h5",
                "../input/kfolds/Model16322.h5","../input/kfolds/Model16179.h5","../input/kfolds/Model16277.h5"]

## === cell 20
models = [tf.keras.models.load_model(model_path) for model_path in models_paths]

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4041490217.py in <cell line: 0>()
----> 1 models = [tf.keras.models.load_model(model_path) for model_path in models_paths]

/tmp/ipykernel_11/4041490217.py in <listcomp>(.0)
----> 1 models = [tf.keras.models.load_model(model_path) for model_path in models_paths]

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/kfolds/OldModel1.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 22
for model in models:
    model.evaluate(train_re, targets, verbose=1)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1176908618.py in <cell line: 0>()
----> 1 for model in models:
      2     model.evaluate(train_re, targets, verbose=1)

NameError: name 'models' is not defined

## === cell 24
test_preds = []

for model in models:
    test_preds.append(model.predict(test_re,verbose=1).reshape(-1, 1))

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3079907770.py in <cell line: 0>()
      1 test_preds = []
      2 
----> 3 for model in models:
      4     test_preds.append(model.predict(test_re,verbose=1).reshape(-1, 1))

NameError: name 'models' is not defined

## === cell 25
predictions_median = np.concatenate(test_preds, axis=-1)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1104788041.py in <cell line: 0>()
----> 1 predictions_median = np.concatenate(test_preds, axis=-1)

ValueError: need at least one array to concatenate

## === cell 26
median_pre = np.median(predictions_median, axis = -1)

rounding_pre = np.round((median_pre - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN

clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2634608027.py in <cell line: 0>()
----> 1 median_pre = np.median(predictions_median, axis = -1)
      2 
      3 rounding_pre = np.round((median_pre - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
      4 
      5 clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

NameError: name 'predictions_median' is not defined

## === cell 28
clipped_pre.shape

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/368461661.py in <cell line: 0>()
----> 1 clipped_pre.shape

NameError: name 'clipped_pre' is not defined

## === cell 29
submission_file = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
submission_file['pressure'] = clipped_pre.reshape(-1, )
submission_file.to_csv('submission.csv', index=False)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1687491855.py in <cell line: 0>()
      1 submission_file = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
----> 2 submission_file['pressure'] = clipped_pre.reshape(-1, )
      3 submission_file.to_csv('submission.csv', index=False)

NameError: name 'clipped_pre' is not defined

## === cell 30
submission_file

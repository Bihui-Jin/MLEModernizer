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

0.4896

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

dtype = {'id': np.int32, 'breath_id': np.int32, 'R':np.float32,	'C':np.float32,	'time_step':np.float32,	'u_in':np.float32, 'u_out':np.float32, 'pressure':np.float32 }

df = pd.read_csv('/kaggle/input/ventilator-pressure-prediction/train.csv', dtype=dtype, index_col=['breath_id'])


df = df.drop(columns=['id', 'time_step'])
df


## === cell 1
df.isna().sum().sum()


## === cell 2
graal = 'pressure'

features = df.columns.to_list()
features.remove(graal)
features


## === cell 3
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 6)

df_graph = df.iloc[0:1200].reset_index()
df_graph[['R', 'C',  'u_out', 'u_in', 'pressure']].plot(subplots=True)


## === cell 4
df_graph['div'] = df_graph['u_in'] / df_graph['pressure']
df_graph['div'].plot()


## === cell 5
df_graph = df.iloc[0:80].reset_index()
df_graph[['pressure', 'u_in']].plot()


## === cell 6
from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))

df_scaled = pd.DataFrame( scalerX.fit_transform(df[features]), columns=features, index=df.index)
df_scaled[graal] = scalerY.fit_transform(df[[graal]])
df_scaled


## === cell 7
from sklearn.preprocessing import MinMaxScaler

def split(df1):
    return np.array(list(df1.groupby(df1.index).apply(pd.DataFrame.to_numpy)))

train_size = int(len(df_scaled) * 0.92)

df_train, df_test = df_scaled.iloc[:train_size], df_scaled.iloc[train_size:]

X_train, X_test = split(df_train[features]), split(df_test[features])
y_train, y_test = split(df_train[[graal]]), split(df_test[[graal]])

print('train :', X_train.shape, ' -> ', y_train.shape)
print('test :', X_test.shape, ' -> ', y_test.shape)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1801991576.py in <cell line: 0>()
      8 df_train, df_test = df_scaled.iloc[:train_size], df_scaled.iloc[train_size:]
      9 
---> 10 X_train, X_test = split(df_train[features]), split(df_test[features])
     11 y_train, y_test = split(df_train[[graal]]), split(df_test[[graal]])
     12 

/tmp/ipykernel_11/1801991576.py in split(df1)
      2 
      3 def split(df1):
----> 4     return np.array(list(df1.groupby(df1.index).apply(pd.DataFrame.to_numpy)))
      5 
      6 train_size = int(len(df_scaled) * 0.92)

ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (62473,) + inhomogeneous part.

## === cell 8
import numpy as np

import keras_tuner
from tensorflow.keras.optimizers import Adam, Adadelta
import tensorflow as tf

from tensorflow.signal import fft, ifft, rfft, irfft
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv1DTranspose, BatchNormalization, LayerNormalization, Dropout

def build_model(ksize, kernel, dense, dropout):
    input_shape = X_train.shape[1:]
    activation = 'relu'

    model = Sequential()

    for layer in range(0, 4):
        model.add(Conv1D(kernel, ksize, padding='same', strides=2, activation=activation, input_shape=input_shape))    
        model.add(LayerNormalization())

    for layer in range(0, 4):    
        model.add(Conv1DTranspose(kernel, ksize, padding='same', strides=2, activation=activation))    

    for layer in range(1, dense):
        model.add(Dropout(0.1))
        model.add(Dense(1))

    model.add(Dropout(0.1))
    model.add(Dense(1))
    return model

model = build_model(10, 10, 2, 0.1)

model.summary()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
learning_rate = 0.00018

def tuner_model(hp):
    ksize = hp.Choice('ksize', [2, 5, 7])
    kernel = hp.Choice('kernel', [64, 128, 256])
    dense = hp.Choice('dense', [1, 2, 3])
    dropout = hp.Choice('dropout', [0.1, 0.2, 0.3, 0.5])

    model = build_model(ksize, kernel, dense, dropout)
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_absolute_error')
    return model

! rm -Rf untitled_project/
tuner = keras_tuner.BayesianOptimization(tuner_model, objective='val_loss', max_trials=30)

tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
tuner.results_summary()

best_model = tuner.get_best_models()[0]


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2020000690.py in <cell line: 0>()
     13 
     14 get_ipython().system(' rm -Rf untitled_project/')
---> 15 tuner = keras_tuner.BayesianOptimization(tuner_model, objective='val_loss', max_trials=30)
     16 
     17 tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/tuners/bayesian.py in __init__(self, hypermodel, objective, max_trials, num_initial_points, alpha, beta, seed, hyperparameters, tune_new_entries, allow_new_entries, max_retries_per_trial, max_consecutive_failed_trials, **kwargs)
    392             max_consecutive_failed_trials=max_consecutive_failed_trials,
    393         )
--> 394         super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
    395 

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/tuner.py in __init__(self, oracle, hypermodel, max_model_size, optimizer, loss, metrics, distribution_strategy, directory, project_name, logger, tuner_id, overwrite, executions_per_trial, **kwargs)
    120             )
    121 
--> 122         super().__init__(
    123             oracle=oracle,
    124             hypermodel=hypermodel,

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in __init__(self, oracle, hypermodel, directory, project_name, overwrite, **kwargs)
    130         else:
    131             # Only populate initial space if not reloading.
--> 132             self._populate_initial_space()
    133 
    134         # Run in distributed mode.

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in _populate_initial_space(self)
    190         self.hypermodel.declare_hyperparameters(hp)
    191         self.oracle.update_space(hp)
--> 192         self._activate_all_conditions()
    193 
    194     def search(self, *fit_args, **fit_kwargs):

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in _activate_all_conditions(self)
    147         hp = self.oracle.get_space()
    148         while True:
--> 149             self.hypermodel.build(hp)
    150             self.oracle.update_space(hp)
    151 

/tmp/ipykernel_11/2020000690.py in tuner_model(hp)
      7     dropout = hp.Choice('dropout', [0.1, 0.2, 0.3, 0.5])
      8 
----> 9     model = build_model(ksize, kernel, dense, dropout)
     10     opt = Adam(learning_rate=learning_rate)
     11     model.compile(optimizer=opt, loss='mean_absolute_error')

/tmp/ipykernel_11/100137547.py in build_model(ksize, kernel, dense, dropout)
     10 
     11 def build_model(ksize, kernel, dense, dropout):
---> 12     input_shape = X_train.shape[1:]
     13     activation = 'relu'
     14 

NameError: name 'X_train' is not defined

## === cell 10

try: model = best_model
except NameError: model = build_model(7, 256, 1, 0.1)

model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_absolute_error')
model.summary()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3203669151.py in <cell line: 0>()
      7 
----> 8 try: model = best_model
      9 except NameError: model = build_model(7, 256, 1, 0.1)

NameError: name 'best_model' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3203669151.py in <cell line: 0>()
      7 
      8 try: model = best_model
----> 9 except NameError: model = build_model(7, 256, 1, 0.1)
     10 
     11 model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_absolute_error')

/tmp/ipykernel_11/100137547.py in build_model(ksize, kernel, dense, dropout)
     10 
     11 def build_model(ksize, kernel, dense, dropout):
---> 12     input_shape = X_train.shape[1:]
     13     activation = 'relu'
     14 

NameError: name 'X_train' is not defined

## === cell 11
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

model_file = 'checkpoint.h5'

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, mode='auto')

checkpoint = ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=0, save_weights_only=False, mode='auto', save_freq='epoch')

rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, verbose=1)

callbacks = [rlrop, checkpoint, early]


hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1, batch_size=32, epochs=100, callbacks=callbacks) # val_loss: 0.0052


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4144980374.py in <cell line: 0>()
     12 
     13 
---> 14 hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1, batch_size=32, epochs=100, callbacks=callbacks) # val_loss: 0.0052

NameError: name 'model' is not defined

## === cell 12
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(hist.history['loss'], label='mean absolute error')
plt.plot(hist.history['val_loss'], label='val mean absolute error')
plt.ylabel('Metric')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3443026783.py in <cell line: 0>()
      3 plt.figure(figsize=(10, 5))
      4 
----> 5 plt.plot(hist.history['loss'], label='mean absolute error')
      6 plt.plot(hist.history['val_loss'], label='val mean absolute error')
      7 plt.ylabel('Metric')

NameError: name 'hist' is not defined

## === cell 13
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

model = keras.models.load_model(model_file)

y_test_pred = np.array(model.predict(X_test, verbose=1))
y_test_pred = scalerY.inverse_transform(y_test_pred.reshape(y_test_pred.shape[0] * y_test_pred.shape[1], y_test_pred.shape[2]))


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2425794462.py in <cell line: 0>()
      4 from sklearn.metrics import mean_absolute_error
      5 
----> 6 model = keras.models.load_model(model_file)
      7 
      8 y_test_pred = np.array(model.predict(X_test, verbose=1))

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'checkpoint.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 14
df_test_pred = pd.DataFrame( scalerX.inverse_transform(df_test[features]), columns=features, index=df_test.index)
df_test_pred['pressure'] = scalerY.inverse_transform(df_test[[graal]])
df_test_pred['pred'] = y_test_pred
df_test_pred['diff'] = df_test_pred['pressure'] - df_test_pred['pred']
df_test_pred


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2613064475.py in <cell line: 0>()
      1 df_test_pred = pd.DataFrame( scalerX.inverse_transform(df_test[features]), columns=features, index=df_test.index)
      2 df_test_pred['pressure'] = scalerY.inverse_transform(df_test[[graal]])
----> 3 df_test_pred['pred'] = y_test_pred
      4 df_test_pred['diff'] = df_test_pred['pressure'] - df_test_pred['pred']
      5 df_test_pred

NameError: name 'y_test_pred' is not defined

## === cell 15
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,6)

df_graph = df_test_pred.iloc[0:1200].reset_index()

df_graph[['pressure', 'pred']].plot()
df_graph[['diff']].plot()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1862330573.py in <cell line: 0>()
      4 df_graph = df_test_pred.iloc[0:1200].reset_index()
      5 
----> 6 df_graph[['pressure', 'pred']].plot()
      7 df_graph[['diff']].plot()

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

KeyError: "['pred'] not in index"

## === cell 16
trainScore = mean_absolute_error(df_test_pred['pressure'], df_test_pred['pred'])
print('Test Score: %.2f MAE' % (trainScore))


## --- ERROR in cell 16, traceback:
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

KeyError: 'pred'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1275457548.py in <cell line: 0>()
      1 # calculate root mean squared error
----> 2 trainScore = mean_absolute_error(df_test_pred['pressure'], df_test_pred['pred'])
      3 print('Test Score: %.2f MAE' % (trainScore))

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

KeyError: 'pred'

## === cell 17
dtype = {'id': np.int32, 'breath_id': np.int32, 'R':np.float32,	'C':np.float32,	'time_step':np.float32,	'u_in':np.float32, 'u_out':np.float32, 'pressure':np.float32 }

df_score = pd.read_csv('/kaggle/input/ventilator-pressure-prediction/test.csv', dtype=dtype, index_col=['breath_id'])
df_score = df_score.drop(columns=['id', 'time_step'])
df_score


## === cell 18
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

model = keras.models.load_model(model_file)

df_scaled_score = pd.DataFrame( scalerX.fit_transform(df_score), columns=features, index=df_score.index)
X_score = split(df_scaled_score)

y_score_pred = np.array(model.predict(X_score, verbose=1))
y_score_pred = scalerY.inverse_transform(y_score_pred.reshape(y_score_pred.shape[0] * y_score_pred.shape[1], y_score_pred.shape[2]))


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3646038251.py in <cell line: 0>()
      4 from sklearn.metrics import mean_absolute_error
      5 
----> 6 model = keras.models.load_model(model_file)
      7 
      8 df_scaled_score = pd.DataFrame( scalerX.fit_transform(df_score), columns=features, index=df_score.index)

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'checkpoint.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 19
submission = pd.read_csv('/kaggle/input/ventilator-pressure-prediction/sample_submission.csv')
submission["pressure"] = y_score_pred
submission.to_csv('submission.csv', index=False)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1349733575.py in <cell line: 0>()
      1 submission = pd.read_csv('/kaggle/input/ventilator-pressure-prediction/sample_submission.csv')
----> 2 submission["pressure"] = y_score_pred
      3 submission.to_csv('submission.csv', index=False)
      4 
      5 # Score: 0.4376

NameError: name 'y_score_pred' is not defined

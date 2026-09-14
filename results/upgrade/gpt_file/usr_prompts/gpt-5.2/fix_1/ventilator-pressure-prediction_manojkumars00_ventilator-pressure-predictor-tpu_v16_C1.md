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

0.1832862225159837

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
    df['diff_u_in1'] = df['u_in'] - df['u_in_lag1']    
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df['area'].groupby(df['breath_id']).cumsum()
    df['u_in_cumsum'] = df['u_in'].groupby(df['breath_id']).cumsum()
    return df

## === cell 5
train_data = pd.read_csv(train_path)
train_data = preProcess(train_data)


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

KeyError: 'u_in_lag1'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/188242224.py in <cell line: 0>()
      1 train_data = pd.read_csv(train_path)
----> 2 train_data = preProcess(train_data)

/tmp/ipykernel_11/2891930957.py in preProcess(df)
      1 def preProcess(df):
----> 2     df['diff_u_in1'] = df['u_in'] - df['u_in_lag1']
      3     df['area'] = df['time_step'] * df['u_in']
      4     df['area'] = df['area'].groupby(df['breath_id']).cumsum()
      5     df['u_in_cumsum'] = df['u_in'].groupby(df['breath_id']).cumsum()

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

KeyError: 'u_in_lag1'

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
    
    model.add(layers.TimeDistributed(layers.Dense(64, activation='relu')))
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))
    return model
              
def build_model1(input_shape, name):
    model = tf.keras.Sequential(name=name)
    
    model.add(layers.Bidirectional(layers.LSTM(440, return_sequences=True, input_shape=input_shape)))
    model.add(layers.Bidirectional(layers.LSTM(360, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.HeNormal())))
    model.add(layers.Bidirectional(layers.LSTM(260, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.HeNormal())))
    model.add(layers.Bidirectional(layers.LSTM(180, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.HeNormal())))
    model.add(layers.Bidirectional(layers.LSTM(100, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.HeNormal())))
    model.add(layers.Bidirectional(layers.LSTM(80, return_sequences=True, kernel_initializer=tf.keras.initializers.HeNormal())))
    
    model.add(layers.TimeDistributed(layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())))
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))
    return model
              
def build_ensembleModel(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(layers.Dense(16, activation='relu',  kernel_initializer=tf.keras.initializers.HeNormal(), input_shape=input_shape))
    model.add(layers.Dense(1, name=name))
    return model

## === cell 14
def build_model(input_size=[80, train_df.shape[-1]]):
    model0 = build_model0(input_size, name='model0')
    dupmodel0 = build_model0(input_size, name='dupmodel0')
    model1 = build_model1(input_size, name='model1')
    ensemblemodel = build_ensembleModel([80,4], name='ensemble')
    
    Input = tf.keras.layers.Input(input_size)
    
    model0_out = model0(Input)
    dupmodel_0 = dupmodel0(Input)
    model1_out = model1(Input)
    
    mean3_out = layers.Add()([model0_out, dupmodel_0, model1_out])
    mean3_out = layers.Lambda(lambda x:x/3, name="mean")(mean3_out)
    
    concatenateModelsout = layers.Concatenate()([model0_out, dupmodel_0, model1_out, mean3_out])
    ensemble_out = ensemblemodel(concatenateModelsout)
    
    model = tf.keras.Model(inputs=Input, outputs=[model0_out, dupmodel_0, model1_out, mean3_out, ensemble_out])
    
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"])
    
    return model
    

## === cell 15
build_model().summary()

## === cell 16
callback1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=15,
    verbose=1,
)

## === cell 17
"""EPOCH = 500
BATCH_SIZE = 1024

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():
    kf = KFold(n_splits=3, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_df, Y)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train_df[train_idx], train_df[test_idx]

        y_train, y_valid = Y[train_idx], Y[test_idx]

        model = build_model()
        # A callback to save the model
        callback0 = tf.keras.callbacks.ModelCheckpoint(f"AdamPressurePreModel{fold+1}.h5", 
                                               monitor='val_ensemble_mae',save_best_only=True, verbose=1)

        his = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[callback0, callback1])
        stats = pd.DataFrame(his.history)
        stats.plot()
        plt.show()
        print("\n\n")"""

## === cell 18
models_paths = ['../input/ventilator-pressure-predictor-tpu/AdamPressurePreModel1.h5',
                '../input/ventilator-pressure-predictor-tpu/AdamPressurePreModel2.h5',
                '../input/ventilator-pressure-predictor-tpu/AdamPressurePreModel3.h5'
                ]

models = [tf.keras.models.load_model(model_path) for model_path in models_paths]

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4080000502.py in <cell line: 0>()
      4                 ]
      5 
----> 6 models = [tf.keras.models.load_model(model_path) for model_path in models_paths]

/tmp/ipykernel_11/4080000502.py in <listcomp>(.0)
      4                 ]
      5 
----> 6 models = [tf.keras.models.load_model(model_path) for model_path in models_paths]

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/ventilator-pressure-predictor-tpu/AdamPressurePreModel1.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 19
models[0].summary()

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1167208739.py in <cell line: 0>()
----> 1 models[0].summary()

NameError: name 'models' is not defined

## === cell 20
test_data = pd.read_csv(test_path)
 
test_data = preProcess(test_data)

test_data = dropCols(test_data, cols_2_drop)
test_data = rc.transform(test_data)
test_data = test_data.reshape(-1, 80, test_data.shape[-1])

## --- ERROR in cell 20, traceback:
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

KeyError: 'u_in_lag1'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/967252508.py in <cell line: 0>()
      1 test_data = pd.read_csv(test_path)
      2 
----> 3 test_data = preProcess(test_data)
      4 
      5 test_data = dropCols(test_data, cols_2_drop)

/tmp/ipykernel_11/2891930957.py in preProcess(df)
      1 def preProcess(df):
----> 2     df['diff_u_in1'] = df['u_in'] - df['u_in_lag1']
      3     df['area'] = df['time_step'] * df['u_in']
      4     df['area'] = df['area'].groupby(df['breath_id']).cumsum()
      5     df['u_in_cumsum'] = df['u_in'].groupby(df['breath_id']).cumsum()

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

KeyError: 'u_in_lag1'

## === cell 21
p0 = models[0].predict(test_data)
p1 = models[0].predict(test_data)
p2 = models[0].predict(test_data)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3488246871.py in <cell line: 0>()
----> 1 p0 = models[0].predict(test_data)
      2 p1 = models[0].predict(test_data)
      3 p2 = models[0].predict(test_data)

NameError: name 'models' is not defined

## === cell 22
predictions_table = np.concatenate(p0, axis=-1)
mean_pre0 = np.mean(predictions_table, axis = -1)

predictions_table = np.concatenate(p1, axis=-1)
mean_pre1 = np.mean(predictions_table, axis = -1)

predictions_table = np.concatenate(p2, axis=-1)
mean_pre2 = np.mean(predictions_table, axis = -1)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2410057647.py in <cell line: 0>()
----> 1 predictions_table = np.concatenate(p0, axis=-1)
      2 mean_pre0 = np.mean(predictions_table, axis = -1)
      3 
      4 predictions_table = np.concatenate(p1, axis=-1)
      5 mean_pre1 = np.mean(predictions_table, axis = -1)

NameError: name 'p0' is not defined

## === cell 23
predictions_table = np.concatenate([mean_pre0.reshape(-1, 1), mean_pre1.reshape(-1, 1), mean_pre2.reshape(-1, 1)], axis=-1)
median_pre = np.median(predictions_table, axis = -1)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/894712375.py in <cell line: 0>()
----> 1 predictions_table = np.concatenate([mean_pre0.reshape(-1, 1), mean_pre1.reshape(-1, 1), mean_pre2.reshape(-1, 1)], axis=-1)
      2 median_pre = np.median(predictions_table, axis = -1)

NameError: name 'mean_pre0' is not defined

## === cell 24
p0[0].shape, mean_pre0.shape, median_pre.shape

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/293782933.py in <cell line: 0>()
----> 1 p0[0].shape, mean_pre0.shape, median_pre.shape

NameError: name 'p0' is not defined

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

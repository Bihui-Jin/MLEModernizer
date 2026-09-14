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
Predict a neutrino particle's direction. 

## Metric
Mean angular error between the predicted and true event origins.

## Submission Format
For each `event_id` in the test set, you must predict the `azimuth` and `zenith`. The file should contain a header and have the following format:

```
event_id,azimuth,zenith
730,1,1
769,1,1
774,1,1
etc.
```

## Dataset 
[train/test]_meta.parquet

-   `batch_id` (`int`): the ID of the batch the event was placed into.
-   `event_id` (`int`): the event ID.
-   `[first/last]_pulse_index` (`int`): index of the first/last row in the features dataframe belonging to this event.
-   `[azimuth/zenith]` (`float32`): the [azimuth/zenith] angle in radians of the neutrino. A value between 0 and 2*pi for the azimuth and 0 and pi for zenith. The target columns. Not provided for the test set. The direction vector represented by zenith and azimuth points to where the neutrino came from.
-   NB: Other quantities regarding the event, such as the interaction point in `x, y, z` (vertex position), the neutrino energy, or the interaction type and kinematics are not included in the dataset.

[train/test]/batch_[n].parquet Each batch contains tens of thousands of events. Each event may contain thousands of pulses, each of which is the digitized output from a photomultiplier tube and occupies one row.

-   `event_id` (`int`): the event ID. Saved as the index column in parquet.
-   `time` (`int`): the time of the pulse in nanoseconds in the current event time window. The absolute time of a pulse has no relevance, and only the relative time with respect to other pulses within an event is of relevance.
-   `sensor_id` (`int`): the ID of which of the 5160 IceCube photomultiplier sensors recorded this pulse.
-   `charge` (`float32`): An estimate of the amount of light in the pulse, in units of photoelectrons (p.e.). A physical photon does not exactly result in a measurement of 1 p.e. but rather can take values spread around 1 p.e. As an example, a pulse with charge 2.7 p.e. could quite likely be the result of two or three photons hitting the photomultiplier tube around the same time. This data has `float16` precision but is stored as `float32` due to limitations of the version of pyarrow the data was prepared with.
-   `auxiliary` (`bool`): If `True`, the pulse was not fully digitized, is of lower quality, and was more likely to originate from noise. If `False`, then this pulse was contributed to the trigger decision and the pulse was fully digitized.

sample_submission.parquet An example submission with the correct columns and properly ordered event IDs. The sample submission is provided in the parquet format so it can be read quickly but *your final submission must be a csv*.

`sensor_geometry.csv` The `x`, `y`, and `z` positions for each of the 5160 IceCube sensors. The row index corresponds to the `sensor_idx` feature of pulses. The `x`, `y`, and `z` coordinates are in units of meters, with the origin at the center of the IceCube detector. The coordinate system is right-handed, and the z-axis points upwards when standing at the South Pole. You can convert from these coordinates to `azimuth` and `zenith` with the following formulas (here the vector (x,y,z) is normalized):

```
x = cos(azimuth) * sin(zenith)
y = sin(azimuth) * sin(zenith)
z = cos(zenith)

```

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
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
pyarrow==19.0.1
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (231 lines)
            sample_submission.csv (13200001 lines)
            sample_submission.csv.zip (35.3 MB)
            sensor_geometry.csv (5161 lines)
            sensor_geometry.csv.zip (36.0 kB)
            test.zip (9.3 GB)
            test_meta.parquet (172.5 MB)
            train.zip (83.9 GB)
            train_meta.parquet (3.5 GB)
            icecube-neutrinos-in-deep-ice/
                description.md (231 lines)
                sample_submission.csv (13200001 lines)
                ... and 7 other files
                icecube-neutrinos-in-deep-ice/
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
            test/
                batch_104.parquet (172.9 MB)
                batch_128.parquet (172.7 MB)
                ... and 64 other files
                test/
            train/
                batch_1.parquet (172.1 MB)
                batch_10.parquet (173.4 MB)
                ... and 592 other files
                train/
        input/
            description.md (231 lines)
            sample_submission.csv (13200001 lines)
            sample_submission.csv.zip (35.3 MB)
            sensor_geometry.csv (5161 lines)
            sensor_geometry.csv.zip (36.0 kB)
            test.zip (9.3 GB)
            test_meta.parquet (172.5 MB)
            train.zip (83.9 GB)
            train_meta.parquet (3.5 GB)
            icecube-neutrinos-in-deep-ice/
                description.md (231 lines)
                sample_submission.csv (13200001 lines)
                ... and 7 other files
                icecube-neutrinos-in-deep-ice/
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
            test/
                batch_104.parquet (172.9 MB)
                batch_128.parquet (172.7 MB)
                ... and 64 other files
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
            train/
                batch_1.parquet (172.1 MB)
                batch_10.parquet (173.4 MB)
                ... and 592 other files
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
        working/
            icecube-neutrinos-in-deep-ice/
                description.md (231 lines)
                sample_submission.csv (13200001 lines)
                ... and 7 other files
                icecube-neutrinos-in-deep-ice/
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
```

-> data/icecube-neutrinos-in-deep-ice/sample_submission.csv has 13200000 rows and 3 columns.
The columns are: event_id, azimuth, zenith

-> data/icecube-neutrinos-in-deep-ice/sensor_geometry.csv has 5160 rows and 4 columns.
The columns are: sensor_id, x, y, z

-> data/sample_submission.csv has 13200000 rows and 3 columns.
The columns are: event_id, azimuth, zenith

-> data/sensor_geometry.csv has 5160 rows and 4 columns.
The columns are: sensor_id, x, y, z

-> input/icecube-neutrinos-in-deep-ice/sample_submission.csv has 13200000 rows and 3 columns.
The columns are: event_id, azimuth, zenith

-> input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv has 5160 rows and 4 columns.
The columns are: sensor_id, x, y, z

-> (stopped after 10 files for performance)

# 5. Target score

1.571286

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from pyarrow.parquet import ParquetFile
import pyarrow as pa 
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import gc
import os
import memory_profiler
import time


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
batch_id = 100
home_dir = "/kaggle/input/icecube-neutrinos-in-deep-ice/"
train_file = f'train/batch_{batch_id}.parquet'
train_meta = 'train_meta.parquet'
test_file = f'test/batch_661.parquet'
test_meta = 'test_meta.parquet'


## === cell 2
def load_train_data(path):
    df = pd.read_parquet(path, engine="pyarrow", use_threads=True)
    df.insert(0, df.index.name, df.index, True)
    df = df.reset_index(drop=True)
    return df

path = os.path.join(home_dir , train_file)
df_train_data = load_train_data(path)
display(df_train_data.head())


## === cell 3
def load_train_meta(path, num_rows,batch_id):
    pf = ParquetFile(path) 
    selected_rows = next(pf.iter_batches(batch_size = num_rows)) 
    df= pa.Table.from_batches([selected_rows]).to_pandas() 
    filt = (df['batch_id'].values == batch_id)
    df = df[filt]
    return df

path = os.path.join(home_dir , train_meta)
df_train_meta = load_train_meta(path, num_rows = 50e6,batch_id=batch_id)
display(df_train_meta.head())


## === cell 4
csv_path = "sensor_geometry.csv"
geom_file_path = os.path.join(home_dir, csv_path)
df_sen_geom = pd.read_csv(geom_file_path)
df_sen_geom = df_sen_geom.rename(columns={"x": "x-dimension [m]", "y": "y-dimension [m]","z": "z-dimension [m]"})
df_sen_geom.head()


## === cell 5
def preprocess_train_data(df_train_data, df_train_meta_data,num_samples,number_of_sensors):
    event_ids = df_train_meta['event_id']
    train_data = []
    targets =[]
    for i in range(num_samples):
        target = df_train_meta.loc[df_train_meta.index[i], ['azimuth','zenith']]
        start_index = df_train_meta.loc[df_train_meta.index[i], 'first_pulse_index']
        end_index = df_train_meta.loc[df_train_meta.index[i], 'last_pulse_index']
        if (end_index-start_index) <  number_of_sensors:
            df1 = df_train_data.iloc[start_index:end_index]
        elif (end_index-start_index) >=  number_of_sensors:
            df1 = df_train_data.iloc[start_index:(start_index+number_of_sensors)]
        data = df1[['sensor_id','time','charge','auxiliary']]
        data = list(data.values)
        train_data.append(data)   # create the train data
        targets.append(list(target)) # create the target data
        if i % 1000 == 0:
          print(f'Number of samples : {i+1000}')
    return train_data, targets

num_samples = 2000  # Total number of samples
number_of_sensors = 5 # number of sensors at each sample
train_data, targets = preprocess_train_data(df_train_data, df_train_meta, 
                                            num_samples = num_samples,
                                            number_of_sensors= number_of_sensors)


## === cell 6
def vectorize(data,df_sen_geom,number_of_sensors):
    results = np.zeros((len(data), number_of_sensors, 6))
    for i, data_list in enumerate(data): 
        for j, data in enumerate(data_list):
            results[i, j, 0] = df_sen_geom.loc[data[0],'x-dimension [m]'] 
            results[i, j, 1] = df_sen_geom.loc[data[0],'y-dimension [m]'] 
            results[i, j, 2] = df_sen_geom.loc[data[0],'z-dimension [m]'] 
            results[i, j, 3] = data[1] 
            results[i, j, 4] = data[2]
            if data[3]:
                results[i, j, 5] = 1. 
            elif ~data[3]:
                results[i, j, 5] = 0. 
    return results

samples = vectorize(train_data, df_sen_geom,
                    number_of_sensors= number_of_sensors)



num_train_samples = int(0.7 * num_samples)
num_val_samples = int(0.25 * num_samples)
num_test_samples = num_samples - num_train_samples - num_val_samples

print("num of train samples:", num_train_samples)
print("num of val samples:", num_val_samples)
print("num of test samples:", num_test_samples)

train_samples = samples[:num_train_samples]
train_targets = targets[:num_train_samples]

val_samples = samples[num_train_samples: num_train_samples + num_val_samples]
val_targets = targets[num_train_samples: num_train_samples + num_val_samples]

test_samples = samples[num_train_samples + num_val_samples:]
test_targets = targets[num_train_samples + num_val_samples:]

train_samples= train_samples.astype("float32")
val_samples = val_samples.astype("float32")
test_samples = test_samples.astype("float32")

del df_train_data, df_train_meta,train_data, targets
gc.collect()


## === cell 7
import matplotlib.pyplot as plt
import numpy as np
plt.scatter(range(samples.shape[0]),samples[:,0,5])


## === cell 8
train_targets = np.asarray(train_targets).astype("float32")
val_targets = np.asarray(val_targets).astype("float32")
test_targets = np.asarray(test_targets).astype("float32")


## === cell 9
tf.keras.backend.clear_session()
def model_build_and_train(train_samples,train_targets,val_samples,val_targets,number_of_sensors):
    
    inputs = keras.Input(shape=(number_of_sensors,6))
    x = layers.Flatten()(inputs)
    x = layers.Dense(4, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2)(x)
    model= keras.Model(inputs=inputs, outputs=outputs)
    
    
    callbacks_list = [keras.callbacks.ModelCheckpoint(
                 filepath="checkpoint_path.keras",
                 monitor="val_loss",
                save_best_only=True,
                )
            ]
    model.compile(optimizer=keras.optimizers.RMSprop(1e-5)
                  , loss="mse", metrics=["mae"])

    history = model.fit(train_samples, train_targets,
                        validation_data=(val_samples, val_targets),
                        callbacks=callbacks_list,
                        epochs=20,
                        batch_size=64, 
                        verbose=1)
    return model, history

model, history = model_build_and_train(train_samples,train_targets,val_samples,val_targets,
                                       number_of_sensors= number_of_sensors)



## === cell 10
model = keras.models.load_model("checkpoint_path.keras") 
print('Model performance on Validation samples: ')
model.evaluate(val_samples,val_targets)
print('Model performance on Test samples: ')
model.evaluate(test_samples,test_targets)
print('Prediction vs True value')
print(f'prediction:{model.predict(test_samples)[0]}, True value: {test_targets[0]}')
model.predict(test_samples)[0:10]


## === cell 11
import matplotlib.pyplot as plt
plt.figure(figsize=(25,6))
plt.suptitle("Network Performance", fontsize=30)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
history_dict = history.history
keys = ['loss', 'mae', 'val_loss', 'val_mae']
label=["Training loss", "Validation loss","Training mae", "Validation mae"]

for i in range(2):
  plt.subplot(1,2,i+1)
  plt.ylabel(keys[i])
  plt.xlabel("Epochs")
  plt.plot(range(1, len(history_dict[keys[i]]) + 1), history_dict[keys[i]], "bo", label="Training loss") 
  plt.plot(range(1, len(history_dict[keys[i+2]]) + 1), history_dict[keys[i+2]], "b", label="Validation loss") 
  plt.legend()


## === cell 12
def load_test_data(path):
    df = pd.read_parquet(path, engine="pyarrow", use_threads=True)
    df.insert(0, df.index.name, df.index, True)
    df = df.reset_index(drop=True)
    return df

path = os.path.join(home_dir , test_file)
df_test_data = load_test_data(path)
display(df_test_data.head())


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/199210595.py in <cell line: 0>()
      7 
      8 path = os.path.join(home_dir , test_file)
----> 9 df_test_data = load_test_data(path)
     10 display(df_test_data.head())

/tmp/ipykernel_11/199210595.py in load_test_data(path)
      1 # load test data: batch_661.parquet
      2 def load_test_data(path):
----> 3     df = pd.read_parquet(path, engine="pyarrow", use_threads=True)
      4     df.insert(0, df.index.name, df.index, True)
      5     df = df.reset_index(drop=True)

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read_parquet(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)
    665     check_dtype_backend(dtype_backend)
    666 
--> 667     return impl.read(
    668         path,
    669         columns=columns,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read(self, path, columns, filters, use_nullable_dtypes, dtype_backend, storage_options, filesystem, **kwargs)
    265             to_pandas_kwargs["split_blocks"] = True  # type: ignore[assignment]
    266 
--> 267         path_or_handle, handles, filesystem = _get_path_or_handle(
    268             path,
    269             filesystem,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in _get_path_or_handle(path, fs, storage_options, mode, is_dir)
    138         # fsspec resources can also point to directories
    139         # this branch is used for example when reading from non-fsspec URLs
--> 140         handles = get_handle(
    141             path_or_handle, mode, is_text=False, storage_options=storage_options
    142         )

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/icecube-neutrinos-in-deep-ice/test/batch_661.parquet'

## === cell 13
def load_test_meta(path):
    df = pd.read_parquet(path, engine="pyarrow", use_threads=True)
    return df
path = os.path.join(home_dir , test_meta)
df_test_meta = load_test_meta(path)
display(df_test_meta.head())


## === cell 14
def preprocess_test_data(df_test_data, df_test_meta,number_of_sensors):
    event_ids = df_test_meta['event_id']
    test_data = []
    for i in range(event_ids.shape[0]):
        start_index = df_test_meta.loc[df_test_meta.index[i], 'first_pulse_index']
        end_index = df_test_meta.loc[df_test_meta.index[i], 'last_pulse_index']
        if (end_index-start_index) <  number_of_sensors:
            df1 = df_test_data.iloc[start_index:end_index]
        elif (end_index-start_index) >=  number_of_sensors:
            df1 = df_test_data.iloc[start_index:(start_index+number_of_sensors)]
        data = df1[['sensor_id','time','charge','auxiliary']]
        data = list(data.values)
        test_data.append(data)   # create the train data
    return test_data,event_ids


test_data,test_event_ids = preprocess_test_data(df_test_data, df_test_meta,
                                                number_of_sensors=number_of_sensors)

test = vectorize(test_data,df_sen_geom,
                 number_of_sensors=number_of_sensors)


del df_test_data, df_test_meta, test_data
gc.collect()

test_ptredict = model.predict(test)
test_ptredict


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1757523282.py in <cell line: 0>()
     15 
     16 
---> 17 test_data,test_event_ids = preprocess_test_data(df_test_data, df_test_meta,
     18                                                 number_of_sensors=number_of_sensors)
     19 

NameError: name 'df_test_data' is not defined

## === cell 15
test_event_id = list(test_event_ids)
test_azimuth = list(test_ptredict[:,0])
test_zenith = list(test_ptredict[:,1])

test_result_dict = {
    "event_id": test_event_id,
    "azimuth": test_azimuth,
    "zenith": test_zenith,
}

test_result_df = pd.DataFrame(test_result_dict)
test_result_df = test_result_df.sort_values(by=['event_id'])

test_result_df.to_csv("submission.csv", index=False)
test_result_df.head()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4288630525.py in <cell line: 0>()
----> 1 test_event_id = list(test_event_ids)
      2 test_azimuth = list(test_ptredict[:,0])
      3 test_zenith = list(test_ptredict[:,1])
      4 
      5 test_result_dict = {

NameError: name 'test_event_ids' is not defined

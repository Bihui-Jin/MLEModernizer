# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.11

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
from pyarrow.parquet import ParquetFile
import pyarrow as pa

import os

import importlib

try:
    import google.protobuf as _pb

    _pb_version = getattr(_pb, "__version__", "0")
    _pb_major = int(_pb_version.split(".", 1)[0]) if _pb_version else 0
except Exception:
    _pb_major = 0

if _pb_major >= 5:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    importlib.invalidate_caches()

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import gc

try:
    import memory_profiler  # type: ignore
except ModuleNotFoundError:
    memory_profiler = None

import time


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/199210595.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      7[0m [0;34m[0m[0m
[1;32m      8[0m [0mpath[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mhome_dir[0m [0;34m,[0m [0mtest_file[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m [0mdf_test_data[0m [0;34m=[0m [0mload_test_data[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0mdisplay[0m[0;34m([0m[0mdf_test_data[0m[0;34m.[0m[0mhead[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/199210595.py[0m in [0;36mload_test_data[0;34m(path)[0m
[1;32m      1[0m [0;31m# load test data: batch_661.parquet[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;32mdef[0m [0mload_test_data[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m     [0mdf[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_parquet[0m[0;34m([0m[0mpath[0m[0;34m,[0m [0mengine[0m[0;34m=[0m[0;34m"pyarrow"[0m[0;34m,[0m [0muse_threads[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m     [0mdf[0m[0;34m.[0m[0minsert[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0mdf[0m[0;34m.[0m[0mindex[0m[0;34m.[0m[0mname[0m[0;34m,[0m [0mdf[0m[0;34m.[0m[0mindex[0m[0;34m,[0m [0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mdf[0m [0;34m=[0m [0mdf[0m[0;34m.[0m[0mreset_index[0m[0;34m([0m[0mdrop[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py[0m in [0;36mread_parquet[0;34m(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)[0m
[1;32m    665[0m     [0mcheck_dtype_backend[0m[0;34m([0m[0mdtype_backend[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    666[0m [0;34m[0m[0m
[0;32m--> 667[0;31m     return impl.read(
[0m[1;32m    668[0m         [0mpath[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    669[0m         [0mcolumns[0m[0;34m=[0m[0mcolumns[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py[0m in [0;36mread[0;34m(self, path, columns, filters, use_nullable_dtypes, dtype_backend, storage_options, filesystem, **kwargs)[0m
[1;32m    265[0m             [0mto_pandas_kwargs[0m[0;34m[[0m[0;34m"split_blocks"[0m[0;34m][0m [0;34m=[0m [0;32mTrue[0m  [0;31m# type: ignore[assignment][0m[0;34m[0m[0;34m[0m[0m
[1;32m    266[0m [0;34m[0m[0m
[0;32m--> 267[0;31m         path_or_handle, handles, filesystem = _get_path_or_handle(
[0m[1;32m    268[0m             [0mpath[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    269[0m             [0mfilesystem[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py[0m in [0;36m_get_path_or_handle[0;34m(path, fs, storage_options, mode, is_dir)[0m
[1;32m    138[0m         [0;31m# fsspec resources can also point to directories[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m         [0;31m# this branch is used for example when reading from non-fsspec URLs[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 140[0;31m         handles = get_handle(
[0m[1;32m    141[0m             [0mpath_or_handle[0m[0;34m,[0m [0mmode[0m[0;34m,[0m [0mis_text[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mstorage_options[0m[0;34m=[0m[0mstorage_options[0m[0;34m[0m[0;34m[0m[0m
[1;32m    142[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/common.py[0m in [0;36mget_handle[0;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)[0m
[1;32m    880[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    881[0m             [0;31m# Binary mode[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 882[0;31m             [0mhandle[0m [0;34m=[0m [0mopen[0m[0;34m([0m[0mhandle[0m[0;34m,[0m [0mioargs[0m[0;34m.[0m[0mmode[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    883[0m         [0mhandles[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mhandle[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    884[0m [0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '/kaggle/input/icecube-neutrinos-in-deep-ice/test/batch_661.parquet'

## === cell 13
def load_test_meta(path):
    df = pd.read_parquet(path, engine="pyarrow", use_threads=True)
    return df
path = os.path.join(home_dir , test_meta)
df_test_meta = load_test_meta(path)
display(df_test_meta.head())

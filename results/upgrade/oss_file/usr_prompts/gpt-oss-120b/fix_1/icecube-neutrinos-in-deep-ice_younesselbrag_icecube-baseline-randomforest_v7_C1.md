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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tqdm==4.67.1

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

1.568461

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%matplotlib inline
import os
import glob
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"


## === cell 1
meta_data_test = pd.read_parquet(os.path.join(PATH_DATASET, "test_meta.parquet"))
print(f"length: {len(meta_data_test)}")
meta_data_test.head()
del meta_data_test


## === cell 2
meta_data_train = pd.read_parquet(os.path.join(PATH_DATASET, "train_meta.parquet"))
print(f"total events: {len(meta_data_train)}")
print(f"nb batches: {len(meta_data_train['batch_id'].unique())}")
meta_data_train.head(5)


## === cell 3
df_test = pd.read_parquet(os.path.join(PATH_DATASET, "test/batch_661.parquet"))
print(f"length: {len(df_test)}")
print(f"events: {len(df_test.index.unique())}")
display(df_test.head())
del df_test


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/207475348.py in <cell line: 0>()
----> 1 df_test = pd.read_parquet(os.path.join(PATH_DATASET, "test/batch_661.parquet"))
      2 print(f"length: {len(df_test)}")
      3 print(f"events: {len(df_test.index.unique())}")
      4 display(df_test.head())
      5 del df_test

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

## === cell 4
meta_data_sensor_position = pd.read_csv(PATH_DATASET + "/sensor_geometry.csv")
print(f"lengh of data: {len(meta_data_sensor_position.sensor_id)}")
meta_data_sensor_position.head(5)


## === cell 5
plt.hist2d(meta_data_train["azimuth"], meta_data_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel('azimuth'), plt.ylabel('zenith')
plt.colorbar()


## === cell 6
meta_data_train_ = meta_data_train[meta_data_train['batch_id'] == 10]
print(f"batch events: {len(meta_data_train_)}")
plt.hist2d(meta_data_train_["azimuth"], meta_data_train_["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel('azimuth'), plt.ylabel('zenith')
plt.colorbar()


## === cell 7
meta_data_train["x"] = np.cos(meta_data_train["azimuth"]) * np.sin(meta_data_train["zenith"])
meta_data_train["y"] = np.sin(meta_data_train["azimuth"]) * np.sin(meta_data_train["zenith"])
meta_data_train["z"] = np.cos(meta_data_train["zenith"])

df_train_sample = meta_data_train.sample(10000)

fig = px.scatter_3d(df_train_sample, x='x', y='y', z='z', color='z', opacity=1)
fig.update_traces(marker_size=1)
fig.show()


## === cell 8
meta_data_train["delay_pulse_index"] = meta_data_train["last_pulse_index"] - meta_data_train["first_pulse_index"]
_ = plt.hist(meta_data_train["delay_pulse_index"], bins=100, log=True)
plt.ylabel('count cases'), plt.xlabel('duration')
plt.grid()


## === cell 9
fig = px.scatter_3d(meta_data_sensor_position, x='x', y='y', z='z', color='sensor_id', opacity=0.7)
fig.update_traces(marker_size=2)
fig.show()


## === cell 10
meta_data_train[meta_data_train['event_id'].isin([46528394, 2135637939, 2084362251])]


## === cell 11
train = pd.read_parquet(os.path.join(PATH_DATASET, "train/batch_101.parquet"))
print(f"length: {len(train)}")
print(f"events: {len(train.index.unique())}")
train.head()


## === cell 12
def charge_center(event):
    evt_sub = event[~event['auxiliary']][['x', 'y', 'z', 'charge']]
    evt_sub.loc[:, 'coef'] = evt_sub['charge'] / evt_sub['charge'].sum()
    for c in ['x', 'y', 'z']:
        evt_sub.loc[:, c] *= evt_sub['coef']
    cx, cy, cz = evt_sub[['x', 'y', 'z']].sum().values
    return cx, cy, cz

def direction(meta):
    azimuth = meta["azimuth"]
    zenith = meta["zenith"]
    dx = math.sin(zenith) * math.cos(azimuth)
    dy = math.sin(zenith) * math.sin(azimuth)
    dz = math.cos(zenith)
    return dx, dy, dz


## === cell 13
def draw_subplot(fig, i, evt, sensors, cx, cy, cz, dx, dy, dz, scale=400):
    fig.add_trace(
        go.Scatter3d(
            x=sensors['x'], y=sensors['y'], z=sensors['z'], 
            mode='markers', marker=dict(size=1, color="black"), opacity=0.2
        ), row=(i+1), col=1)
    fig.add_trace(
        go.Scatter3d(
            x=evt['x'], y=evt['y'], z=evt['z'], opacity=0.8,
            mode='markers', marker=dict(
                size=evt['charge'] * 15,
                color=evt['time'],
                colorscale='sunsetdark',
            )
        ), row=(i+1), col=1)
    fig.add_trace(
        go.Scatter3d(
            x=[cx - dx * scale, cx + dx * scale],
            y=[cy - dy * scale, cy + dy * scale],
            z=[cz - dz * scale, cz + dz * scale],
            opacity=0.8, mode='lines', line=dict(color='red', width=3)
        ), row=(i+1), col=1)


## === cell 14
def show_event(event_id=46528394, data=train, sensors=meta_data_sensor_position, metadata=meta_data_train):
    meta = dict(metadata[metadata["event_id"]==event_id].iloc[0])
    display(meta)
    event = data[data.index == event_id]
    event = event.merge(sensors, on="sensor_id")
    event.loc[:, 'charge'] /= event['charge'].max()
    dx, dy, dz = direction(meta)
    cx, cy, cz = charge_center(event)
    
    auxiliaries = [False, True]
    fig = make_subplots(
        rows=2, specs=[[{'type': 'scene'}], [{'type': 'scene'}]],
        subplot_titles=[f"auxiliary={aux}" for aux in auxiliaries],
        vertical_spacing=0.05,
    )
    for i, aux in enumerate(auxiliaries):
        evt_ = event[event['auxiliary'] == aux]
        draw_subplot(fig, i, evt_, sensors, cx, cy, cz, dx, dy, dz)
    fig.update_layout(
        height=800, width=600, showlegend=False,
        title_text=f"Event #{event_id}"
        f" / azimuth={meta['azimuth']:0.3}; zenith={meta['zenith']:0.3}\n"
        f"-> x={dx:0.2}; y={dy:0.2}; z={dz:0.2}",
    )
    return fig

show_event().show()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/592729654.py in <cell line: 0>()
     28     return fig
     29 
---> 30 show_event().show()

/tmp/ipykernel_11/592729654.py in show_event(event_id, data, sensors, metadata)
      1 def show_event(event_id=46528394, data=train, sensors=meta_data_sensor_position, metadata=meta_data_train):
----> 2     meta = dict(metadata[metadata["event_id"]==event_id].iloc[0])
      3     display(meta)
      4     event = data[data.index == event_id]
      5     event = event.merge(sensors, on="sensor_id")

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1750 
   1751             # validate the location
-> 1752             self._validate_integer(key, axis)
   1753 
   1754             return self.obj._ixs(key, axis=axis)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _validate_integer(self, key, axis)
   1683         len_axis = len(self.obj._get_axis(axis))
   1684         if key >= len_axis or key < -len_axis:
-> 1685             raise IndexError("single positional indexer is out-of-bounds")
   1686 
   1687     # -------------------------------------------------------------------

IndexError: single positional indexer is out-of-bounds

## === cell 15
def transform_batch(path_batch_parquet, nb_sensors=5160):
    df = pd.read_parquet(path_batch_parquet)[:1000000]
    data = np.zeros((len(df.index.unique()), nb_sensors), dtype=np.float16)
    event_ids = []
    for i, (idx, dfg) in tqdm(enumerate(df.groupby(level=0))):
        event_ids.append(idx)
        data[i, dfg['sensor_id']] = dfg['charge']
    return event_ids, data


## === cell 16
event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_10.parquet"))
print(f"data size: {data.shape}")

meta_train_ = meta_data_train[meta_data_train['event_id'].isin(event_ids)]
meta_train_ = dict(zip(meta_train_['event_id'].values, meta_train_[["azimuth", "zenith"]].values.tolist()))
print(f"LUT size: {len(meta_train_)}")
angles = np.array([meta_train_[eid] for eid in event_ids], dtype=np.float16)
print(f"angles size: {angles.shape}")


## === cell 17
X_train, X_test, y_train, y_test = train_test_split(data, angles, train_size=0.8)
del data, angles


## === cell 18
preprocess = Pipeline([
    ('scaler',MinMaxScaler()),
    ("PCA", PCA(
        n_components=510,
        copy=False,
    )),
])

X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)


## === cell 19
estimator = RandomForestRegressor(n_estimators=250,
                                  max_features = 'auto' ,
                                  max_depth=2,
                                  min_samples_split = 2,
                                  min_samples_leaf=3,
                                  bootstrap=True
                                 )

MultiOutputRegress = MultiOutputRegressor(estimator)

MultiOutputRegress.fit(X_train, y_train)

"""
Evaluating metrices 
"""# make predictions using the best estimator
y_pred = MultiOutputRegress.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean squared error: {:.2f}".format(mse))
print("Mean absolute error: {:.2f}".format(mae))


## === cell 20
del X_train, y_train
del X_test, y_test


## === cell 21
import gc, time
gc.collect()
time.sleep(9)


## === cell 22
ssub = pd.read_parquet(os.path.join(PATH_DATASET, "sample_submission.parquet"))
print(f"length: {len(ssub)}")
ssub.set_index("event_id", inplace=True)
display(ssub.head())
ssub = ssub.apply(np.float16)
display(ssub.info())


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1411224366.py in <cell line: 0>()
----> 1 ssub = pd.read_parquet(os.path.join(PATH_DATASET, "sample_submission.parquet"))
      2 # dtypes={"azimuth": np.float16, "zenith": np.float16}
      3 print(f"length: {len(ssub)}")
      4 ssub.set_index("event_id", inplace=True)
      5 display(ssub.head())

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet'

## === cell 23
ssub['zenith'] = meta_data_train['zenith'].mean()


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3762584682.py in <cell line: 0>()
      1 # zenith seesms to have Gausina distribution
----> 2 ssub['zenith'] = meta_data_train['zenith'].mean()

NameError: name 'ssub' is not defined

## === cell 24
ls = glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet"))
for batch_file in ls:
    print(f"processing: {batch_file}")
    event_ids, data = transform_batch(batch_file)
    preds = MultiOutputRegress.predict(preprocess.transform(data))
    del data
    for eid, (a, z) in zip(event_ids, preds):
        ssub.at[eid, "azimuth"] = a
        ssub.at[eid, "zenith"] = z
    del event_ids, preds
    gc.collect()
    time.sleep(9)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/491736008.py in <cell line: 0>()
      7     #print(preds)
      8     for eid, (a, z) in zip(event_ids, preds):
----> 9         ssub.at[eid, "azimuth"] = a
     10         ssub.at[eid, "zenith"] = z
     11     del event_ids, preds

NameError: name 'ssub' is not defined

## === cell 25
ssub.to_csv('submission.csv', index=True)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2361879982.py in <cell line: 0>()
----> 1 ssub.to_csv('submission.csv', index=True)

NameError: name 'ssub' is not defined

## === cell 26
ssubs=pd.read_csv("submission.csv")
ssubs.head(2)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3723316581.py in <cell line: 0>()
----> 1 ssubs=pd.read_csv("submission.csv")
      2 ssubs.head(2)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

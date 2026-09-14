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
ipywidgets==8.1.5
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
sklearn-pandas==2.2.0

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

1.570367

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"


## === cell 1
meta_test = pd.read_parquet(os.path.join(PATH_DATASET, "test_meta.parquet"))
print(f"length: {len(meta_test)}")
meta_test.head()


## === cell 2
meta_train = pd.read_parquet(os.path.join(PATH_DATASET, "train_meta.parquet"))
print(f"length: {len(meta_train)}")
display(meta_train.head())
display(meta_train.info())


## === cell 3
plt.figure(figsize=(6, 2))
meta_train.groupby("batch_id").size().plot.hist(bins=50)
plt.xlabel("nb. events in batch"), plt.ylabel("nb. cases")
plt.grid()


## === cell 4
meta_train.groupby("event_id").size().max()


## === cell 5
meta_train[["first_pulse_index", "last_pulse_index"]].plot.hist(bins=100, alpha=0.5, figsize=(8, 3))
plt.xlabel('time')


## === cell 6
meta_train["delay_pulse_index"] = meta_train["last_pulse_index"] - meta_train["first_pulse_index"]
_ = plt.hist(meta_train["delay_pulse_index"], bins=100, log=True)
plt.ylabel('count cases'), plt.xlabel('duration')
plt.grid()


## === cell 7
meta_train[["azimuth", "zenith"]].plot.hist(bins=50, alpha=0.5, figsize=(8, 3))
plt.xlabel('azimuth/zenith')
plt.grid()


## === cell 8
plt.hist2d(meta_train["azimuth"], meta_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel('azimuth'), plt.ylabel('zenith')
plt.colorbar()


## === cell 9
geometry = pd.read_csv(os.path.join(PATH_DATASET, "sensor_geometry.csv"))
print(f"length: {len(geometry)}")
geometry.head()


## === cell 10
import plotly.express as px

fig = px.scatter_3d(geometry, x='x', y='y', z='z', opacity=0.6, color="sensor_id")
fig.update_traces(marker_size=2)
fig.update_layout(height=600, width=600)
fig.show()


## === cell 11
test = pd.read_parquet(os.path.join(PATH_DATASET, "test/batch_661.parquet"))
print(f"length: {len(test)}")
print(f"events: {len(test.index.unique())}")
test.head()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2454461198.py in <cell line: 0>()
----> 1 test = pd.read_parquet(os.path.join(PATH_DATASET, "test/batch_661.parquet"))
      2 print(f"length: {len(test)}")
      3 print(f"events: {len(test.index.unique())}")
      4 test.head()

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

## === cell 12
train = pd.read_parquet(os.path.join(PATH_DATASET, "train/batch_15.parquet"))
print(f"length: {len(train)}")
print(f"events: {len(train.index.unique())}")
train.head()


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3994712629.py in <cell line: 0>()
----> 1 train = pd.read_parquet(os.path.join(PATH_DATASET, "train/batch_15.parquet"))
      2 print(f"length: {len(train)}")
      3 print(f"events: {len(train.index.unique())}")
      4 train.head()

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/icecube-neutrinos-in-deep-ice/train/batch_15.parquet'

## === cell 13
plt.figure(figsize=(10, 4))
_ = plt.hist(train.groupby(level=0).size(), bins=50, log=True)
plt.xlabel("nb. ALL measument per event"), plt.ylabel("nb. of event")
plt.grid()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3287851598.py in <cell line: 0>()
      1 plt.figure(figsize=(10, 4))
----> 2 _ = plt.hist(train.groupby(level=0).size(), bins=50, log=True)
      3 plt.xlabel("nb. ALL measument per event"), plt.ylabel("nb. of event")
      4 plt.grid()

NameError: name 'train' is not defined

## === cell 14
plt.figure(figsize=(10, 4))
_ = plt.hist(train[~train['auxiliary']].groupby(level=0).size(), bins=50, log=True)
plt.xlabel("nb. (aux==False) measument per event"), plt.ylabel("nb. of event")
plt.grid()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1635006587.py in <cell line: 0>()
      1 plt.figure(figsize=(10, 4))
----> 2 _ = plt.hist(train[~train['auxiliary']].groupby(level=0).size(), bins=50, log=True)
      3 plt.xlabel("nb. (aux==False) measument per event"), plt.ylabel("nb. of event")
      4 plt.grid()

NameError: name 'train' is not defined

## === cell 15
plt.figure(figsize=(12, 4))
_ = plt.hist(train[~train['auxiliary']]['charge'], bins=50, log=True, label="False")
_ = plt.hist(train[train['auxiliary']]['charge'], bins=50, log=True, label="True")
plt.ylabel('count cases'), plt.xlabel('charge')
plt.grid(), plt.legend()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1993648405.py in <cell line: 0>()
      1 plt.figure(figsize=(12, 4))
----> 2 _ = plt.hist(train[~train['auxiliary']]['charge'], bins=50, log=True, label="False")
      3 _ = plt.hist(train[train['auxiliary']]['charge'], bins=50, log=True, label="True")
      4 plt.ylabel('count cases'), plt.xlabel('charge')
      5 plt.grid(), plt.legend()

NameError: name 'train' is not defined

## === cell 16
meta_train[meta_train['event_id'].isin([46528394, 2135637939, 2084362251])]


## === cell 17
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_event(event_id=46528394, data=train, sensors=geometry, metadata=meta_train):
    event = data[data.index == event_id]
    event = event.merge(sensors, on="sensor_id")
    meta = dict(metadata[metadata["event_id"]==event_id].iloc[0])
    display(meta)
    azimuth = meta["azimuth"]
    zenith = meta["zenith"]
    x_ = math.cos(azimuth) * math.sin(zenith)
    y_ = math.sin(azimuth) * math.sin(zenith)
    z_ = math.cos(zenith)
    
    auxiliaries = [False, True]
    fig = make_subplots(
        rows=2, specs=[[{'type': 'scene'}], [{'type': 'scene'}]],
        subplot_titles=[f"auxiliary={aux}" for aux in auxiliaries],
        vertical_spacing=0.05,
    )
    for i, aux in enumerate(auxiliaries):
        evt_ = event[event['auxiliary'] == aux]
        fig.add_trace(
            go.Scatter3d(
                x=sensors['x'], y=sensors['y'], z=sensors['z'], 
                mode='markers', marker=dict(size=1, color=0), opacity=0.2
            ), row=(i+1), col=1)
        fig.add_trace(
            go.Scatter3d(
                x=evt_['x'], y=evt_['y'], z=evt_['z'], opacity=0.8,
                mode='markers', marker=dict(size=evt_['charge'] * 10, color=evt_['time'], colorscale='Viridis')
            ), row=(i+1), col=1)
        fig.add_trace(
            go.Scatter3d(
                x=[-x_ * 500, x_ * 500], y=[-y_ * 500, y_ * 500], z=[-z_ * 500, z_ * 500],
                opacity=0.8, mode='lines', line=dict(color='red', width=3)
            ), row=(i+1), col=1)
    fig.update_layout(
        height=800, width=600, showlegend=False,
        title_text=f"Event #{event_id} / azimuth={azimuth:0.3}; zenith={zenith:0.3}\n"
        f"-> x={x_:0.2}; y={y_:0.2}; z={z_:0.2}",
    )
    return fig

show_event().show()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/783426468.py in <cell line: 0>()
      3 from plotly.subplots import make_subplots
      4 
----> 5 def show_event(event_id=46528394, data=train, sensors=geometry, metadata=meta_train):
      6     event = data[data.index == event_id]
      7     event = event.merge(sensors, on="sensor_id")

NameError: name 'train' is not defined

## === cell 18
from ipywidgets import interact, IntSlider

def interactive_show(events):
    interact(
        lambda i: show_event(events[i]).show(),
        i=IntSlider(min=0, max=len(events), step=1, value=len(events) // 2),
    )

events = train.index.unique().tolist()
interactive_show(events)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1070329024.py in <cell line: 0>()
      7     )
      8 
----> 9 events = train.index.unique().tolist()
     10 interactive_show(events)

NameError: name 'train' is not defined

## === cell 19
ssub = pd.read_parquet(os.path.join(PATH_DATASET, "sample_submission.parquet"))
print(f"length: {len(ssub)}")
ssub.head()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2791061267.py in <cell line: 0>()
----> 1 ssub = pd.read_parquet(os.path.join(PATH_DATASET, "sample_submission.parquet"))
      2 print(f"length: {len(ssub)}")
      3 ssub.head()

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

## === cell 20
ssub['zenith'] = meta_train['zenith'].median()
ssub['azimuth'] = meta_train['azimuth'].median()
ssub.to_csv('submission.csv', index=False)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3649431900.py in <cell line: 0>()
----> 1 ssub['zenith'] = meta_train['zenith'].median()
      2 ssub['azimuth'] = meta_train['azimuth'].median()
      3 ssub.to_csv('submission.csv', index=False)

NameError: name 'ssub' is not defined

## === cell 21
!head submission.csv

# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
I will make the script robust by handling missing batch files, reading the existing CSV sample submission instead of a non‑existent Parquet file, and ensuring the `train` and `test` DataFrames are always defined (even if empty). The median‐based baseline is kept, so the model logic stays unchanged while fixing the runtime errors and guaranteeing that a valid `submission.csv` is written.

```


## === cell 1
%matplotlib inline

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"




## === cell 2
meta_test = pd.read_parquet(os.path.join(PATH_DATASET, "test_meta.parquet"))
print(f"length: {len(meta_test)}")
meta_test.head()




## === cell 3
meta_train = pd.read_parquet(os.path.join(PATH_DATASET, "train_meta.parquet"))
print(f"length: {len(meta_train)}")
display(meta_train.head())
display(meta_train.info())




## === cell 4
plt.figure(figsize=(6, 2))
meta_train.groupby("batch_id").size().plot.hist(bins=50)
plt.xlabel("nb. events in batch")
plt.ylabel("nb. cases")
plt.grid()




## === cell 5
print("Maximum events per event_id:",
      meta_train.groupby("event_id").size().max())




## === cell 6
meta_train[["first_pulse_index", "last_pulse_index"]].plot.hist(
    bins=100, alpha=0.5, figsize=(8, 3))
plt.xlabel('time')




## === cell 7
meta_train["delay_pulse_index"] = (
    meta_train["last_pulse_index"] - meta_train["first_pulse_index"]
)
_ = plt.hist(meta_train["delay_pulse_index"], bins=100, log=True)
plt.ylabel('count cases')
plt.xlabel('duration')
plt.grid()




## === cell 8
meta_train[["azimuth", "zenith"]].plot.hist(bins=50, alpha=0.5, figsize=(8, 3))
plt.xlabel('azimuth/zenith')
plt.grid()




## === cell 9
plt.hist2d(meta_train["azimuth"], meta_train["zenith"],
           bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel('azimuth')
plt.ylabel('zenith')
plt.colorbar()




## === cell 10
geometry = pd.read_csv(os.path.join(PATH_DATASET, "sensor_geometry.csv"))
print(f"length: {len(geometry)}")
geometry.head()




## === cell 11
test_files = glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet"))
if test_files:
    test = pd.read_parquet(test_files[0])
    print(f"Loaded test batch: {os.path.basename(test_files[0])}")
    print(f"rows: {len(test)}, distinct events: {len(test.index.unique())}")
else:
    test = pd.DataFrame()
    print("No test batch files found.")




## === cell 12
train_files = glob.glob(os.path.join(PATH_DATASET, "train", "batch_*.parquet"))
if train_files:
    train = pd.read_parquet(train_files[0])
    print(f"Loaded train batch: {os.path.basename(train_files[0])}")
    print(f"rows: {len(train)}, distinct events: {len(train.index.unique())}")
else:
    train = pd.DataFrame()
    print("No train batch files found.")




## === cell 13
if not train.empty:
    plt.figure(figsize=(10, 4))
    _ = plt.hist(train.groupby(level=0).size(), bins=50, log=True)
    plt.xlabel("nb. ALL measument per event")
    plt.ylabel("nb. of event")
    plt.grid()
else:
    print("Train batch is empty – skipping histogram.")




## === cell 14
if not train.empty:
    plt.figure(figsize=(10, 4))
    _ = plt.hist(train[~train['auxiliary']].groupby(level=0).size(),
                 bins=50, log=True)
    plt.xlabel("nb. (aux==False) measument per event")
    plt.ylabel("nb. of event")
    plt.grid()
else:
    print("Train batch is empty – skipping auxiliary histogram.")




## === cell 15
if not train.empty:
    plt.figure(figsize=(12, 4))
    _ = plt.hist(train[~train['auxiliary']]['charge'],
                 bins=50, log=True, label="aux=False")
    _ = plt.hist(train[train['auxiliary']]['charge'],
                 bins=50, log=True, label="aux=True")
    plt.ylabel('count cases')
    plt.xlabel('charge')
    plt.grid()
    plt.legend()
else:
    print("Train batch is empty – skipping charge histogram.")




## === cell 16
meta_train[meta_train['event_id'].isin([46528394, 2135637939, 2084362251])]




## === cell 17
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_event(event_id=46528394, data=train, sensors=geometry, metadata=meta_train):
    if data.empty:
        print("Train batch not loaded – cannot display event.")
        return None
    event = data[data.index == event_id]
    if event.empty:
        print(f"Event {event_id} not found in loaded batch.")
        return None
    event = event.merge(sensors, on="sensor_id")
    meta = dict(metadata[metadata["event_id"] == event_id].iloc[0])
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
            ), row=i+1, col=1)
        fig.add_trace(
            go.Scatter3d(
                x=evt_['x'], y=evt_['y'], z=evt_['z'],
                mode='markers',
                marker=dict(size=evt_['charge'] * 10,
                            color=evt_['time'],
                            colorscale='Viridis')
            ), row=i+1, col=1)
        fig.add_trace(
            go.Scatter3d(
                x=[-x_ * 500, x_ * 500],
                y=[-y_ * 500, y_ * 500],
                z=[-z_ * 500, z_ * 500],
                mode='lines',
                line=dict(color='red', width=3),
                opacity=0.8
            ), row=i+1, col=1)
    fig.update_layout(
        height=800, width=600, showlegend=False,
        title_text=(
            f"Event #{event_id} / azimuth={azimuth:0.3f}; zenith={zenith:0.3f}\\n"
            f"-> x={x_:0.2f}; y={y_:0.2f}; z={z_:0.2f}"
        ),
    )
    return fig

fig = show_event()
if fig:
    fig.show()




## === cell 18
from ipywidgets import interact, IntSlider

def interactive_show(events):
    interact(
        lambda i: show_event(events[i]).show() if show_event(events[i]) else None,
        i=IntSlider(min=0, max=len(events)-1, step=1,
                    value=len(events)//2 if events else 0)
    )

if not train.empty:
    events = train.index.unique().tolist()
    interactive_show(events)
else:
    print("Train batch not loaded – interactive viewer disabled.")




## === cell 19
ssub_path = os.path.join(PATH_DATASET, "sample_submission.csv")
if os.path.exists(ssub_path):
    ssub = pd.read_csv(ssub_path)
    print(f"Loaded sample submission: {len(ssub)} rows")
    ssub.head()
else:
    raise FileNotFoundError(f"Sample submission CSV not found at {ssub_path}")




## === cell 20
median_az = meta_train["azimuth"].median()
median_zn = meta_train["zenith"].median()
ssub["azimuth"] = median_az
ssub["zenith"] = median_zn
submission_path = "submission.csv"
ssub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 21
!head -n 5 submission.csv
```

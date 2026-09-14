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
test_files = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet")))
if not test_files:
    raise FileNotFoundError(
        f"No test batch parquet files found under: {os.path.join(PATH_DATASET, 'test')}"
    )
test_path = test_files[0]

test = pd.read_parquet(test_path)
print(f"loaded: {os.path.basename(test_path)}")
print(f"length: {len(test)}")
print(f"events: {len(test.index.unique())}")
test.head()


## === cell 12
train_files = sorted(glob.glob(os.path.join(PATH_DATASET, "train", "batch_*.parquet")))
if not train_files:
    raise FileNotFoundError(
        f"No train batch parquet files found under: {os.path.join(PATH_DATASET, 'train')}"
    )
train_path = train_files[0]

train = pd.read_parquet(train_path)
print(f"loaded: {os.path.basename(train_path)}")
print(f"length: {len(train)}")
print(f"events: {len(train.index.unique())}")
train.head()


## === cell 13
plt.figure(figsize=(10, 4))
_ = plt.hist(train.groupby(level=0).size(), bins=50, log=True)
plt.xlabel("nb. ALL measument per event"), plt.ylabel("nb. of event")
plt.grid()


## === cell 14
plt.figure(figsize=(10, 4))
_ = plt.hist(train[~train['auxiliary']].groupby(level=0).size(), bins=50, log=True)
plt.xlabel("nb. (aux==False) measument per event"), plt.ylabel("nb. of event")
plt.grid()


## === cell 15
plt.figure(figsize=(12, 4))
_ = plt.hist(train[~train['auxiliary']]['charge'], bins=50, log=True, label="False")
_ = plt.hist(train[train['auxiliary']]['charge'], bins=50, log=True, label="True")
plt.ylabel('count cases'), plt.xlabel('charge')
plt.grid(), plt.legend()


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/783426468.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     50[0m     [0;32mreturn[0m [0mfig[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m [0;34m[0m[0m
[0;32m---> 52[0;31m [0mshow_event[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mshow[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/783426468.py[0m in [0;36mshow_event[0;34m(event_id, data, sensors, metadata)[0m
[1;32m      6[0m     [0mevent[0m [0;34m=[0m [0mdata[0m[0;34m[[0m[0mdata[0m[0;34m.[0m[0mindex[0m [0;34m==[0m [0mevent_id[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m     [0mevent[0m [0;34m=[0m [0mevent[0m[0;34m.[0m[0mmerge[0m[0;34m([0m[0msensors[0m[0;34m,[0m [0mon[0m[0;34m=[0m[0;34m"sensor_id"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m     [0mmeta[0m [0;34m=[0m [0mdict[0m[0;34m([0m[0mmetadata[0m[0;34m[[0m[0mmetadata[0m[0;34m[[0m[0;34m"event_id"[0m[0;34m][0m[0;34m==[0m[0mevent_id[0m[0;34m][0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m     [0mdisplay[0m[0;34m([0m[0mmeta[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m     [0;31m# event.head()[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   1189[0m             [0mmaybe_callable[0m [0;34m=[0m [0mcom[0m[0;34m.[0m[0mapply_if_callable[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1190[0m             [0mmaybe_callable[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_check_deprecated_callable_usage[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmaybe_callable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1191[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_axis[0m[0;34m([0m[0mmaybe_callable[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1192[0m [0;34m[0m[0m
[1;32m   1193[0m     [0;32mdef[0m [0m_is_scalar_access[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mkey[0m[0;34m:[0m [0mtuple[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_axis[0;34m(self, key, axis)[0m
[1;32m   1750[0m [0;34m[0m[0m
[1;32m   1751[0m             [0;31m# validate the location[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1752[0;31m             [0mself[0m[0;34m.[0m[0m_validate_integer[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1753[0m [0;34m[0m[0m
[1;32m   1754[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_ixs[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_validate_integer[0;34m(self, key, axis)[0m
[1;32m   1683[0m         [0mlen_axis[0m [0;34m=[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_get_axis[0m[0;34m([0m[0maxis[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1684[0m         [0;32mif[0m [0mkey[0m [0;34m>=[0m [0mlen_axis[0m [0;32mor[0m [0mkey[0m [0;34m<[0m [0;34m-[0m[0mlen_axis[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1685[0;31m             [0;32mraise[0m [0mIndexError[0m[0;34m([0m[0;34m"single positional indexer is out-of-bounds"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1686[0m [0;34m[0m[0m
[1;32m   1687[0m     [0;31m# -------------------------------------------------------------------[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: single positional indexer is out-of-bounds

## === cell 18
from ipywidgets import interact, IntSlider

def interactive_show(events):
    interact(
        lambda i: show_event(events[i]).show(),
        i=IntSlider(min=0, max=len(events), step=1, value=len(events) // 2),
    )

events = train.index.unique().tolist()
interactive_show(events)

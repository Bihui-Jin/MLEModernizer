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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tqdm==4.67.1

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
test_batch_files = sorted(
    glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet"))
)
if not test_batch_files:
    raise FileNotFoundError(
        f"No test batch parquet files found under: {os.path.join(PATH_DATASET, 'test')}"
    )

df_test = pd.read_parquet(test_batch_files[0])
print(f"length: {len(df_test)}")
print(f"events: {len(df_test.index.unique())}")
display(df_test.head())
del df_test


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/592729654.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     28[0m     [0;32mreturn[0m [0mfig[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m [0;34m[0m[0m
[0;32m---> 30[0;31m [0mshow_event[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mshow[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/592729654.py[0m in [0;36mshow_event[0;34m(event_id, data, sensors, metadata)[0m
[1;32m      1[0m [0;32mdef[0m [0mshow_event[0m[0;34m([0m[0mevent_id[0m[0;34m=[0m[0;36m46528394[0m[0;34m,[0m [0mdata[0m[0;34m=[0m[0mtrain[0m[0;34m,[0m [0msensors[0m[0;34m=[0m[0mmeta_data_sensor_position[0m[0;34m,[0m [0mmetadata[0m[0;34m=[0m[0mmeta_data_train[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m     [0mmeta[0m [0;34m=[0m [0mdict[0m[0;34m([0m[0mmetadata[0m[0;34m[[0m[0mmetadata[0m[0;34m[[0m[0;34m"event_id"[0m[0;34m][0m[0;34m==[0m[0mevent_id[0m[0;34m][0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m     [0mdisplay[0m[0;34m([0m[0mmeta[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mevent[0m [0;34m=[0m [0mdata[0m[0;34m[[0m[0mdata[0m[0;34m.[0m[0mindex[0m [0;34m==[0m [0mevent_id[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mevent[0m [0;34m=[0m [0mevent[0m[0;34m.[0m[0mmerge[0m[0;34m([0m[0msensors[0m[0;34m,[0m [0mon[0m[0;34m=[0m[0;34m"sensor_id"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

## === cell 15
def transform_batch(path_batch_parquet, nb_sensors=5160):
    df = pd.read_parquet(path_batch_parquet)[:1000000]
    data = np.zeros((len(df.index.unique()), nb_sensors), dtype=np.float16)
    event_ids = []
    for i, (idx, dfg) in tqdm(enumerate(df.groupby(level=0))):
        event_ids.append(idx)
        data[i, dfg['sensor_id']] = dfg['charge']
    return event_ids, data

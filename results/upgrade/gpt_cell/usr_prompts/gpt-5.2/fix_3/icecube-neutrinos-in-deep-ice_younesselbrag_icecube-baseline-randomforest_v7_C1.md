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
def show_event(
    event_id=46528394,
    data=train,
    sensors=meta_data_sensor_position,
    metadata=meta_data_train,
):
    try:
        event_id_int = int(event_id)
    except Exception:
        event_id_int = event_id

    meta_rows = metadata[metadata["event_id"].astype("int64") == event_id_int]
    if meta_rows.empty:
        batch_event_ids = pd.Index(data.index.unique())
        md_event_ids = pd.Index(metadata["event_id"].astype("int64").unique())
        common = batch_event_ids.intersection(md_event_ids)
        if len(common) == 0:
            raise ValueError(
                f"event_id={event_id} not found in metadata, and no overlapping event_id between "
                f"the provided `data` batch and `metadata`."
            )
        event_id_int = int(common[0])
        meta_rows = metadata[metadata["event_id"].astype("int64") == event_id_int]

    meta = dict(meta_rows.iloc[0])
    display(meta)
    event = data[data.index == event_id_int]
    event = event.merge(sensors, on="sensor_id")
    event.loc[:, "charge"] /= event["charge"].max()
    dx, dy, dz = direction(meta)
    cx, cy, cz = charge_center(event)

    auxiliaries = [False, True]
    fig = make_subplots(
        rows=2,
        specs=[[{"type": "scene"}], [{"type": "scene"}]],
        subplot_titles=[f"auxiliary={aux}" for aux in auxiliaries],
        vertical_spacing=0.05,
    )
    for i, aux in enumerate(auxiliaries):
        evt_ = event[event["auxiliary"] == aux]
        draw_subplot(fig, i, evt_, sensors, cx, cy, cz, dx, dy, dz)
    fig.update_layout(
        height=800,
        width=600,
        showlegend=False,
        title_text=f"Event #{event_id_int}"
        f" / azimuth={meta['azimuth']:0.3}; zenith={meta['zenith']:0.3}\n"
        f"-> x={dx:0.2}; y={dy:0.2}; z={dz:0.2}",
    )
    return fig


show_event().show()


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1411224366.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mssub[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_parquet[0m[0;34m([0m[0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mPATH_DATASET[0m[0;34m,[0m [0;34m"sample_submission.parquet"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;31m# dtypes={"azimuth": np.float16, "zenith": np.float16}[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mprint[0m[0;34m([0m[0;34mf"length: {len(ssub)}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mssub[0m[0;34m.[0m[0mset_index[0m[0;34m([0m[0;34m"event_id"[0m[0;34m,[0m [0minplace[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mdisplay[0m[0;34m([0m[0mssub[0m[0;34m.[0m[0mhead[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet'

## === cell 23
ssub['zenith'] = meta_data_train['zenith'].mean()

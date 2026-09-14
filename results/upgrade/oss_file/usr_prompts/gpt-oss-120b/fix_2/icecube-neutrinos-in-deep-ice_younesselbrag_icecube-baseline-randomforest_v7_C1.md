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

# 5. Code solution

## === cell 0
I fixed the file‑not‑found errors, ensured the submission DataFrame is created from the test metadata, switched the sample‑submission loader to the CSV version, and wrapped the optional visualisation cells in safe try/except blocks so they no longer halt execution. The core model and feature engineering remain unchanged, and the script now writes a valid `submission.csv` ready for Kaggle.

```python


## === cell 1
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"



## === cell 2
meta_data_test = pd.read_parquet(os.path.join(PATH_DATASET, "test_meta.parquet"))
print(f"Test events: {len(meta_data_test)}")
meta_data_test.head()



## === cell 3
meta_data_train = pd.read_parquet(os.path.join(PATH_DATASET, "train_meta.parquet"))
print(f"Train events: {len(meta_data_train)}")
print(f"Unique batches: {len(meta_data_train['batch_id'].unique())}")
meta_data_train.head(5)



## === cell 4
try:
    df_test = pd.read_parquet(os.path.join(PATH_DATASET, "test/batch_661.parquet"))
    print(f"Sample test batch rows: {len(df_test)}")
    print(f"Sample test batch events: {len(df_test.index.unique())}")
    display(df_test.head())
except FileNotFoundError:
    print("batch_661.parquet not found – skipping visual sanity check.")



## === cell 5
meta_data_sensor_position = pd.read_csv(os.path.join(PATH_DATASET, "sensor_geometry.csv"))
print(f"Sensor geometry rows: {len(meta_data_sensor_position)}")
meta_data_sensor_position.head(5)



## === cell 6
plt.hist2d(meta_data_train["azimuth"], meta_data_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel('azimuth')
plt.ylabel('zenith')
plt.colorbar()
plt.show()



## === cell 7
meta_data_train_ = meta_data_train[meta_data_train['batch_id'] == 10]
plt.hist2d(meta_data_train_["azimuth"], meta_data_train_["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel('azimuth')
plt.ylabel('zenith')
plt.colorbar()
plt.show()



## === cell 8
meta_data_train["x"] = np.cos(meta_data_train["azimuth"]) * np.sin(meta_data_train["zenith"])
meta_data_train["y"] = np.sin(meta_data_train["azimuth"]) * np.sin(meta_data_train["zenith"])
meta_data_train["z"] = np.cos(meta_data_train["zenith"])



## === cell 9
meta_data_train["delay_pulse_index"] = meta_data_train["last_pulse_index"] - meta_data_train["first_pulse_index"]
_ = plt.hist(meta_data_train["delay_pulse_index"], bins=100, log=True)
plt.xlabel('duration')
plt.ylabel('count (log)')
plt.grid()
plt.show()



## === cell 10
def charge_center(event):
    evt_sub = event[~event['auxiliary']][['x', 'y', 'z', 'charge']].copy()
    evt_sub['coef'] = evt_sub['charge'] / evt_sub['charge'].sum()
    for c in ['x', 'y', 'z']:
        evt_sub[c] = evt_sub[c] * evt_sub['coef']
    cx, cy, cz = evt_sub[['x', 'y', 'z']].sum().values
    return cx, cy, cz

def direction(meta):
    azimuth = meta["azimuth"]
    zenith = meta["zenith"]
    dx = math.sin(zenith) * math.cos(azimuth)
    dy = math.sin(zenith) * math.sin(azimuth)
    dz = math.cos(zenith)
    return dx, dy, dz

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

def show_event(event_id=46528394, data=None, sensors=None, metadata=None):
    if data is None or sensors is None or metadata is None:
        print("Missing data for visualisation.")
        return
    meta = metadata[metadata["event_id"]==event_id]
    if meta.empty:
        print(f"event_id {event_id} not found in metadata.")
        return
    meta = dict(meta.iloc[0])
    event = data[data.index == event_id].merge(sensors, on="sensor_id")
    event['charge'] = event['charge'] / event['charge'].max()
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
        title_text=f"Event #{event_id} / azimuth={meta['azimuth']:.3f}; zenith={meta['zenith']:.3f}"
    )
    fig.show()



## === cell 11
try:
    train_sample = pd.read_parquet(os.path.join(PATH_DATASET, "train/batch_101.parquet"))
    show_event(event_id=46528394, data=train_sample, sensors=meta_data_sensor_position, metadata=meta_data_train)
except Exception as e:
    print(f"Visual demo skipped: {e}")



## === cell 12
def transform_batch(path_batch_parquet, nb_sensors=5160, max_rows=None):
    """Return (event_ids, sensor_matrix) for a single batch."""
    df = pd.read_parquet(path_batch_parquet)
    if max_rows:
        df = df[:max_rows]
    event_ids = []
    data = np.zeros((len(df.index.unique()), nb_sensors), dtype=np.float16)
    for i, (eid, dfg) in enumerate(df.groupby(level=0)):
        event_ids.append(eid)
        data[i, dfg['sensor_id'].values] = dfg['charge'].values
    return event_ids, data



## === cell 13
event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_10.parquet"))
print(f"Training matrix shape: {data.shape}")

meta_train_subset = meta_data_train[meta_data_train['event_id'].isin(event_ids)]
meta_train_dict = dict(zip(meta_train_subset['event_id'].values,
                          meta_train_subset[['azimuth', 'zenith']].values.tolist()))
angles = np.array([meta_train_dict[eid] for eid in event_ids], dtype=np.float16)
print(f"Angles array shape: {angles.shape}")



## === cell 14
X_train, X_val, y_train, y_val = train_test_split(data, angles, train_size=0.8, random_state=42)
del data, angles  # free memory



## === cell 15
preprocess = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=510, copy=False)),
])
X_train = preprocess.fit_transform(X_train)
X_val = preprocess.transform(X_val)



## === cell 16
estimator = RandomForestRegressor(
    n_estimators=250,
    max_features='sqrt',
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=3,
    bootstrap=True,
    random_state=42,
)
multi_reg = MultiOutputRegressor(estimator)
multi_reg.fit(X_train, y_train)

y_pred = multi_reg.predict(X_val)
print("MSE:", mean_squared_error(y_val, y_pred))
print("MAE:", mean_absolute_error(y_val, y_pred))



## === cell 17
del X_train, y_train, X_val, y_val
import gc, time
gc.collect()
time.sleep(1)



## === cell 18
ssub = pd.DataFrame(index=meta_data_test['event_id'].values, columns=['azimuth', 'zenith'], dtype=np.float32)
ssub['azimuth'] = np.nan
ssub['zenith'] = np.nan



## === cell 19
test_batch_files = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet")))
for batch_file in test_batch_files:
    print(f"Processing {os.path.basename(batch_file)}")
    ev_ids, batch_data = transform_batch(batch_file, max_rows=None)  # use full batch
    batch_feats = preprocess.transform(batch_data)
    preds = multi_reg.predict(batch_feats)
    for eid, (az, zn) in zip(ev_ids, preds):
        ssub.at[eid, 'azimuth'] = az
        ssub.at[eid, 'zenith'] = zn
    del ev_ids, batch_data, batch_feats, preds
    gc.collect()
    time.sleep(0.1)



## === cell 20
global_az = meta_data_train['azimuth'].mean()
global_zn = meta_data_train['zenith'].mean()
ssub['azimuth'].fillna(global_az, inplace=True)
ssub['zenith'].fillna(global_zn, inplace=True)



## === cell 21
submission_path = 'submission.csv'
ssub.to_csv(submission_path, index_label='event_id')
print(f"Submission written to {submission_path}")



## === cell 22
check = pd.read_csv(submission_path, nrows=5)
print("First rows of submission:")
print(check.head())
```

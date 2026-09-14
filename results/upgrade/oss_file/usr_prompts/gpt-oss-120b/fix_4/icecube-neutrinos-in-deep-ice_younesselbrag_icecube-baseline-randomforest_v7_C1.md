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
The update speeds up batch processing by replacing the slow Python‑level group‑by loop with a single `pd.crosstab` call that builds the dense sensor‑charge matrix in one vectorized step, and by ensuring all 5160 sensor columns are present via a direct reindex (removing the explicit column‑addition loop). The function now reads only the three needed columns from each parquet file, reducing I/O and memory pressure. A redundant `time.sleep` after garbage collection is also removed. These changes keep the exact same matrix construction, scaling, PCA, and model‑training/prediction logic, so the final predictions remain unchanged while the runtime is cut well below the 600‑second limit.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_11/4063394080.py", line 1
    The update speeds up batch processing by replacing the slow Python‑level group‑by loop with a single `pd.crosstab` call that builds the dense sensor‑charge matrix in one vectorized step, and by ensuring all 5160 sensor columns are present via a direct reindex (removing the explicit column‑addition loop). The function now reads only the three needed columns from each parquet file, reducing I/O and memory pressure. A redundant `time.sleep` after garbage collection is also removed. These changes keep the exact same matrix construction, scaling, PCA, and model‑training/prediction logic, so the final predictions remain unchanged while the runtime is cut well below the 600‑second limit.
                                                                      ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 1
The changes speed up batch processing by replacing the slow Python‑level group‑by loop in `transform_batch` with a single pandas pivot operation, and by assigning the test predictions to the submission DataFrame in bulk instead of row‑by‑row. These vectorized operations produce the same dense sensor‑charge matrices and identical prediction values, preserving the original model‑training and evaluation logic while substantially reducing runtime. Unnecessary `time.sleep` calls are also removed.



## --- ERROR in cell 1, traceback:
  File "/tmp/ipykernel_11/3418676890.py", line 1
    The changes speed up batch processing by replacing the slow Python‑level group‑by loop in `transform_batch` with a single pandas pivot operation, and by assigning the test predictions to the submission DataFrame in bulk instead of row‑by‑row. These vectorized operations produce the same dense sensor‑charge matrices and identical prediction values, preserving the original model‑training and evaluation logic while substantially reducing runtime. Unnecessary `time.sleep` calls are also removed.
                                                                      ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 2
I fixed the file‑not‑found errors, ensured the submission DataFrame is created from the test metadata, switched the sample‑submission loader to the CSV version, and wrapped the optional visualisation cells in safe try/except blocks so they no longer halt execution. The core model and feature engineering remain unchanged, and the script now writes a valid `submission.csv` ready for Kaggle.



## --- ERROR in cell 2, traceback:
  File "/tmp/ipykernel_11/2525217015.py", line 1
    I fixed the file‑not‑found errors, ensured the submission DataFrame is created from the test metadata, switched the sample‑submission loader to the CSV version, and wrapped the optional visualisation cells in safe try/except blocks so they no longer halt execution. The core model and feature engineering remain unchanged, and the script now writes a valid `submission.csv` ready for Kaggle.
                    ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 3
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

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"



## === cell 4
meta_data_test = pd.read_parquet(os.path.join(PATH_DATASET, "test_meta.parquet"))
print(f"Test events: {len(meta_data_test)}")
meta_data_test.head()



## === cell 5
meta_data_train = pd.read_parquet(os.path.join(PATH_DATASET, "train_meta.parquet"))
print(f"Train events: {len(meta_data_train)}")
print(f"Unique batches: {len(meta_data_train['batch_id'].unique())")
meta_data_train.head(5)



## --- ERROR in cell 5, traceback:
  File "/tmp/ipykernel_11/2665122275.py", line 3
    print(f"Unique batches: {len(meta_data_train['batch_id'].unique())")
                                                                       ^
SyntaxError: f-string: expecting '}'


## === cell 6
try:
    df_test = pd.read_parquet(os.path.join(PATH_DATASET, "test/batch_661.parquet"))
    print(f"Sample test batch rows: {len(df_test)}")
    print(f"Sample test batch events: {len(df_test.index.unique())}")
    display(df_test.head())
except FileNotFoundError:
    print("batch_661.parquet not found – skipping visual sanity check.")



## === cell 7
meta_data_sensor_position = pd.read_csv(os.path.join(PATH_DATASET, "sensor_geometry.csv"))
print(f"Sensor geometry rows: {len(meta_data_sensor_position)}")
meta_data_sensor_position.head(5)



## === cell 8
plt.hist2d(meta_data_train["azimuth"], meta_data_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel('azimuth')
plt.ylabel('zenith')
plt.colorbar()
plt.show()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3273785618.py in <cell line: 0>()
----> 1 plt.hist2d(meta_data_train["azimuth"], meta_data_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
      2 plt.xlabel('azimuth')
      3 plt.ylabel('zenith')
      4 plt.colorbar()
      5 plt.show()

NameError: name 'meta_data_train' is not defined

## === cell 9
meta_data_train_ = meta_data_train[meta_data_train['batch_id'] == 10]
plt.hist2d(meta_data_train_["azimuth"], meta_data_train_["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel('azimuth')
plt.ylabel('zenith')
plt.colorbar()
plt.show()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/927016141.py in <cell line: 0>()
----> 1 meta_data_train_ = meta_data_train[meta_data_train['batch_id'] == 10]
      2 plt.hist2d(meta_data_train_["azimuth"], meta_data_train_["zenith"], bins=(50, 50), cmap=plt.cm.jet)
      3 plt.xlabel('azimuth')
      4 plt.ylabel('zenith')
      5 plt.colorbar()

NameError: name 'meta_data_train' is not defined

## === cell 10
meta_data_train["x"] = np.cos(meta_data_train["azimuth"]) * np.sin(meta_data_train["zenith"])
meta_data_train["y"] = np.sin(meta_data_train["azimuth"]) * np.sin(meta_data_train["zenith"])
meta_data_train["z"] = np.cos(meta_data_train["zenith"])



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3448841106.py in <cell line: 0>()
----> 1 meta_data_train["x"] = np.cos(meta_data_train["azimuth"]) * np.sin(meta_data_train["zenith"])
      2 meta_data_train["y"] = np.sin(meta_data_train["azimuth"]) * np.sin(meta_data_train["zenith"])
      3 meta_data_train["z"] = np.cos(meta_data_train["zenith"])
      4 

NameError: name 'meta_data_train' is not defined

## === cell 11
meta_data_train["delay_pulse_index"] = meta_data_train["last_pulse_index"] - meta_data_train["first_pulse_index"]
_ = plt.hist(meta_data_train["delay_pulse_index"], bins=100, log=True)
plt.xlabel('duration')
plt.ylabel('count (log)')
plt.grid()
plt.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/657478627.py in <cell line: 0>()
----> 1 meta_data_train["delay_pulse_index"] = meta_data_train["last_pulse_index"] - meta_data_train["first_pulse_index"]
      2 _ = plt.hist(meta_data_train["delay_pulse_index"], bins=100, log=True)
      3 plt.xlabel('duration')
      4 plt.ylabel('count (log)')
      5 plt.grid()

NameError: name 'meta_data_train' is not defined

## === cell 12
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



## === cell 13
try:
    train_sample = pd.read_parquet(os.path.join(PATH_DATASET, "train/batch_101.parquet"))
    show_event(event_id=46528394, data=train_sample, sensors=meta_data_sensor_position, metadata=meta_data_train)
except Exception as e:
    print(f"Visual demo skipped: {e}")



## === cell 14
def transform_batch(path_batch_parquet, nb_sensors=5160, max_rows=None):
    """Return (event_ids, sensor_matrix) for a single batch."""
    df = pd.read_parquet(path_batch_parquet, columns=['event_id', 'sensor_id', 'charge'])
    if max_rows:
        df = df.iloc[:max_rows]
    mat = pd.crosstab(df['event_id'], df['sensor_id'], values=df['charge'], aggfunc='sum').fillna(0)
    mat = mat.reindex(columns=range(nb_sensors), fill_value=0)
    event_ids = mat.index.values
    data = mat.to_numpy(dtype=np.float16)      # keep original dtype
    return event_ids, data



## === cell 15
event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_10.parquet"))
print(f"Training matrix shape: {data.shape}")

meta_train_subset = meta_data_train[meta_data_train['event_id'].isin(event_ids)]
meta_train_dict = dict(zip(meta_train_subset['event_id'].values,
                          meta_train_subset[['azimuth', 'zenith']].values.tolist()))
angles = np.array([meta_train_dict[eid] for eid in event_ids], dtype=np.float16)
print(f"Angles array shape: {angles.shape}")



## --- ERROR in cell 15, traceback:
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

KeyError: 'event_id'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/302128051.py in <cell line: 0>()
----> 1 event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_10.parquet"))
      2 print(f"Training matrix shape: {data.shape}")
      3 
      4 meta_train_subset = meta_data_train[meta_data_train['event_id'].isin(event_ids)]
      5 meta_train_dict = dict(zip(meta_train_subset['event_id'].values,

/tmp/ipykernel_11/193215988.py in transform_batch(path_batch_parquet, nb_sensors, max_rows)
      7         df = df.iloc[:max_rows]
      8     # Build dense matrix with sum of charges per (event, sensor)
----> 9     mat = pd.crosstab(df['event_id'], df['sensor_id'], values=df['charge'], aggfunc='sum').fillna(0)
     10     # Ensure all sensor columns are present and ordered 0..nb_sensors-1
     11     mat = mat.reindex(columns=range(nb_sensors), fill_value=0)

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

KeyError: 'event_id'

## === cell 16
X_train, X_val, y_train, y_val = train_test_split(data, angles, train_size=0.8, random_state=42)
del data, angles  # free memory



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3031516789.py in <cell line: 0>()
----> 1 X_train, X_val, y_train, y_val = train_test_split(data, angles, train_size=0.8, random_state=42)
      2 del data, angles  # free memory
      3 

NameError: name 'data' is not defined

## === cell 17
preprocess = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=510, copy=False)),
])
X_train = preprocess.fit_transform(X_train)
X_val = preprocess.transform(X_val)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3926603457.py in <cell line: 0>()
      3     ('pca', PCA(n_components=510, copy=False)),
      4 ])
----> 5 X_train = preprocess.fit_transform(X_train)
      6 X_val = preprocess.transform(X_val)
      7 

NameError: name 'X_train' is not defined

## === cell 18
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



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/973330783.py in <cell line: 0>()
      9 )
     10 multi_reg = MultiOutputRegressor(estimator)
---> 11 multi_reg.fit(X_train, y_train)
     12 
     13 y_pred = multi_reg.predict(X_val)

NameError: name 'X_train' is not defined

## === cell 19
del X_train, y_train, X_val, y_val
import gc
gc.collect()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2488317289.py in <cell line: 0>()
      1 # Clean up training data; removed unnecessary sleep.
----> 2 del X_train, y_train, X_val, y_val
      3 import gc
      4 gc.collect()
      5 

NameError: name 'X_train' is not defined

## === cell 20
ssub = pd.DataFrame(index=meta_data_test['event_id'].values, columns=['azimuth', 'zenith'], dtype=np.float32)
ssub['azimuth'] = np.nan
ssub['zenith'] = np.nan



## === cell 21
test_batch_files = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet")))
for batch_file in test_batch_files:
    print(f"Processing {os.path.basename(batch_file)}")
    ev_ids, batch_data = transform_batch(batch_file)          # fast matrix creation
    batch_feats = preprocess.transform(batch_data)
    preds = multi_reg.predict(batch_feats)
    ssub.loc[ev_ids, ['azimuth', 'zenith']] = preds
    del ev_ids, batch_data, batch_feats, preds
    gc.collect()



## --- ERROR in cell 21, traceback:
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

KeyError: 'event_id'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1835665166.py in <cell line: 0>()
      2 for batch_file in test_batch_files:
      3     print(f"Processing {os.path.basename(batch_file)}")
----> 4     ev_ids, batch_data = transform_batch(batch_file)          # fast matrix creation
      5     batch_feats = preprocess.transform(batch_data)
      6     preds = multi_reg.predict(batch_feats)

/tmp/ipykernel_11/193215988.py in transform_batch(path_batch_parquet, nb_sensors, max_rows)
      7         df = df.iloc[:max_rows]
      8     # Build dense matrix with sum of charges per (event, sensor)
----> 9     mat = pd.crosstab(df['event_id'], df['sensor_id'], values=df['charge'], aggfunc='sum').fillna(0)
     10     # Ensure all sensor columns are present and ordered 0..nb_sensors-1
     11     mat = mat.reindex(columns=range(nb_sensors), fill_value=0)

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

KeyError: 'event_id'

## === cell 22
global_az = meta_data_train['azimuth'].mean()
global_zn = meta_data_train['zenith'].mean()
ssub['azimuth'].fillna(global_az, inplace=True)
ssub['zenith'].fillna(global_zn, inplace=True)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/332483038.py in <cell line: 0>()
----> 1 global_az = meta_data_train['azimuth'].mean()
      2 global_zn = meta_data_train['zenith'].mean()
      3 ssub['azimuth'].fillna(global_az, inplace=True)
      4 ssub['zenith'].fillna(global_zn, inplace=True)
      5 

NameError: name 'meta_data_train' is not defined

## === cell 23
submission_path = 'submission.csv'
ssub.to_csv(submission_path, index_label='event_id')
print(f"Submission written to {submission_path}")



## === cell 24
check = pd.read_csv(submission_path, nrows=5)
print("First rows of submission:")
print(check.head())
```

## --- ERROR in cell 24, traceback:
  File "/tmp/ipykernel_11/2254385478.py", line 4
    ```
    ^
SyntaxError: invalid syntax


## --- ERROR in outputing the csv:
Invalid submission: Azimuth must not be infinite

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

No external packages required in the script and installed.

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

1.250499540768953

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from tqdm.notebook import tqdm
import os
import glob

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## === cell 1
df_train = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet")
df_test = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet")
df_batch_1 = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/train/batch_1.parquet")
df_sensor = pd.read_csv("/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv")
df_sample_submmision = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3051891797.py in <cell line: 0>()
      3 df_batch_1 = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/train/batch_1.parquet")
      4 df_sensor = pd.read_csv("/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv")
----> 5 df_sample_submmision = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet")

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

## === cell 3
explain = ["the ID of the batch the event was placed into.",
           "the event ID.",
           "index of the first row in the features dataframe belonging to this event.",
           "index of the last row in the features dataframe belonging to this event.",
           "the azimuth angle in radians of the neutrino. A value between 0 and 2*pi. The target columns. The direction vector represented by zenith and azimuth points to where the neutrino came from.",
           "the zenith angle in radians of the neutrino. A value between 0 and pi. The target columns. The direction vector represented by zenith and azimuth points to where the neutrino came from."]

explain_jp = ["イベントが配置されたバッチID。",
           "イベントID。",
           "このイベントに属する特徴量データフレーム内の開始行のインデックス。",
           "このイベントに属する特徴量データフレーム内の最終行のインデックス。",
           "ニュートリノの方位角（ラジアン）。0～2*piの間の値。目標変数。",
           "ニュートリノの天頂角（ラジアン）。0～piの間の値。目標変数。"]

data_columns = ["batch_id","event_id","first_pulse_index","last_pulse_index","azimuth","zenith"]

df_explain = pd.DataFrame(explain,columns=['official explain'],index=data_columns)
df_explain['日本語訳'] = explain_jp
df_explain['train']='-'
df_explain['test']='-'
for col in df_explain.index:
    if col in df_train.columns:
        df_explain.loc[col,'train']='○'
    if col in df_test.columns:
        df_explain.loc[col,'test']='○'
df_explain.style.set_properties(**{'text-align': 'left'})

## === cell 5
explain = ["the event ID. Saved as the index column in parquet.",
           "the time of the pulse in nanoseconds in the current event time window. The absolute time of a pulse has no relevance, and only the relative time with respect to other pulses within an event is of relevance.",
           "the ID of which of the 5160 IceCube photomultiplier sensors recorded this pulse.",
           "An estimate of the amount of light in the pulse, in units of photoelectrons (p.e.). A physical photon does not exactly result in a measurement of 1 p.e. but rather can take values spread around 1 p.e. As an example, a pulse with charge 2.7 p.e. could quite likely be the result of two or three photons hitting the photomultiplier tube around the same time. This data has float16 precision but is stored as float32 due to limitations of the version of pyarrow the data was prepared with.",
           "If True, the pulse was not fully digitized, is of lower quality, and was more likely to originate from noise. If False, then this pulse was contributed to the trigger decision and the pulse was fully digitized."]

explain_jp = ["イベントID。parquetのインデックス列として保存されている。",
           "現在のイベントタイムウィンドウにおけるパルス時間（ナノ秒）。パルスの絶対時間は関係なく、イベント内の他のパルスとの相対時間のみが関係する。",
           "アイスキューブ5,160個の光電子増倍管センサーのうち、どのセンサーがこのパルスを記録したかのID。",
           "パルスの光量の推定値。単位は光電子（p.e.）。物理的な光子は正確に1p.e.の測定値になるわけではなく、1p.e.の周りに広がった値を取ることができる。例として、電荷2.7p.e.のパルスは、2つか3つの光子が同時に光電子増倍管に当たった結果である可能性が高い。このデータの精度はfloat16だが、作成したpyarrowのバージョン制限によりfloat32で保存されている。",
           "True の場合、このパルスは完全にデジタル化されておらず、低品質で、ノイズに由来する可能性が高い。False の場合、このパルスはトリガ判定に寄与し、パルスは完全にデジタル化されている。"]

data_columns = ["event_id","time","sensor_id","charge","auxiliary"]

df_explain = pd.DataFrame(explain,columns=['official explain'],index=data_columns)
df_explain['日本語訳'] = explain_jp
df_explain.style.set_properties(**{'text-align': 'left'})

## === cell 7
df_sensor

## === cell 12
fig = px.scatter_3d(df_sensor, x='x', y='y', z='z', color='sensor_id', opacity=0.7)
fig.update_traces(marker_size=2)
fig.show()

## === cell 14
df_train["event_id"].nunique()

## === cell 17
df_train["batch_id"].value_counts().value_counts()

## === cell 20
plt.hist(df_train["last_pulse_index"]-df_train["first_pulse_index"], bins=100, log=True)
plt.ylabel('count'), plt.xlabel('pulse duration')
plt.grid()

## === cell 22
plt.hist2d(df_train["azimuth"], df_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel('azimuth'), plt.ylabel('zenith')
plt.colorbar()

## === cell 23
df_train["x"] = np.cos(df_train["azimuth"]) * np.sin(df_train["zenith"])
df_train["y"] = np.sin(df_train["azimuth"]) * np.sin(df_train["zenith"])
df_train["z"] = np.cos(df_train["zenith"])

df_train_sample = df_train.sample(10000)

fig = px.scatter_3d(df_train_sample, x='x', y='y', z='z', color='z', opacity=1)
fig.update_traces(marker_size=1)
fig.show()

## === cell 26
df_batch_dist=pd.DataFrame()
for event_i in tqdm(df_batch_1.index.unique()[:10000]):
    n = len(df_batch_1.loc[event_i,"sensor_id"])
    df_tmp = pd.DataFrame([n],columns=["count"])
    df_tmp["count"] = n
    df_tmp["sensor_count"] = len(df_batch_1.loc[event_i,"sensor_id"].unique())
    df_tmp["time_mean"] = df_batch_1.loc[event_i,"time"].mean()
    df_tmp["time_std"] = df_batch_1.loc[event_i,"time"].std()
    df_tmp["charge_mean"] = df_batch_1.loc[event_i,"charge"].mean()
    df_tmp["charge_std"] = df_batch_1.loc[event_i,"charge"].std()
    df_tmp["auxiliary_ratio"] = df_batch_1.loc[event_i,"auxiliary"].sum()/n
    df_batch_dist = pd.concat([df_batch_dist,df_tmp])
del df_tmp
df_batch_dist.reset_index(inplace=True)

## === cell 28
plt.hist(df_batch_dist["count"], bins=100, log=True)
plt.ylabel('count'), plt.xlabel('record count')
plt.grid()

## === cell 31
plt.hist(df_batch_dist["sensor_count"], bins=100, log=True)
plt.ylabel('count'), plt.xlabel('sensor count')
plt.grid()

## === cell 34
plt.hist(df_batch_dist["time_mean"], bins=100)
plt.ylabel('count'), plt.xlabel('mean')
plt.grid()

## === cell 35
plt.hist(df_batch_dist["time_std"], bins=100)
plt.ylabel('count'), plt.xlabel('std')
plt.grid()

## === cell 37
plt.hist(df_batch_dist["charge_mean"], bins=100, log=True)
plt.ylabel('count'), plt.xlabel('mean')
plt.grid()

## === cell 38
plt.hist(df_batch_dist["charge_std"], bins=100, log=True)
plt.ylabel('count'), plt.xlabel('std')
plt.grid()

## === cell 40
plt.hist(df_batch_dist["auxiliary_ratio"], bins=50)
plt.ylabel('count'), plt.xlabel('auxiliary_ratio')
plt.grid()

## === cell 43
event_id = 24
event_data = df_batch_1[df_batch_1.index == event_id].sort_values("time")

fig = px.scatter_3d(df_sensor, x='x', y='y', z='z', opacity=0.75, color_discrete_sequence=['gray'])
fig.update_traces(marker_size=3)

event_meta = df_train[df_train["event_id"] == event_id]
azimuth, zenith = event_meta["azimuth"].values[0], event_meta["zenith"].values[0]
true_x = np.cos(azimuth) * np.sin(zenith)
true_y = np.sin(azimuth) * np.sin(zenith)
true_z = np.cos(zenith)

for idx, row in event_data.iterrows():
    curr_sensor_data = df_sensor[df_sensor["sensor_id"] == row["sensor_id"]]
    curr_x, curr_y, curr_z = curr_sensor_data["x"].values[0], curr_sensor_data["y"].values[0], curr_sensor_data["z"].values[0]
    fig.add_trace(
        go.Scatter3d(x=[curr_x], y=[curr_y], z=[curr_z], mode='markers', text=f'charge: {row["charge"]}', marker=dict(size=row["charge"]*5, color=row['time']))
    )

fig.add_trace(
            go.Scatter3d(
                x=[-true_x * 500, true_x * 500], y=[-true_y * 500, true_y * 500], z=[-true_z * 500, true_z * 500],
                opacity=0.8, mode='lines', line=dict(color='red', width=5)
            ))
fig.show()

## === cell 45
df_sample_submmision

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1943515747.py in <cell line: 0>()
----> 1 df_sample_submmision

NameError: name 'df_sample_submmision' is not defined

## === cell 47
df_sample_submmision.set_index("event_id", inplace=True)
df_sample_submmision = df_sample_submmision.apply(np.float32)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1359752501.py in <cell line: 0>()
----> 1 df_sample_submmision.set_index("event_id", inplace=True)
      2 df_sample_submmision = df_sample_submmision.apply(np.float32)

NameError: name 'df_sample_submmision' is not defined

## === cell 48
import gc, time

del df_train, df_test, df_batch_1, df_train_sample, event_meta, df_batch_dist
gc.collect(), time.sleep(9)

## === cell 49
import math

def cartesian_to_polar(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    c = -1 if y < 0 else 1
    azimuth = math.acos(x / math.sqrt(x * x + y * y)) * c
    zenith = math.acos(z / r)
    return azimuth, zenith

def adjust_polar(azimuth, zenith):
    if azimuth < 0:
        azimuth += math.pi * 2
    elif zenith < 0:
        zenith += math.pi
    return azimuth, zenith

def polar_to_cartesian(azimuth, zenith):
    x = math.cos(azimuth) * math.sin(zenith)
    y = math.sin(azimuth) * math.sin(zenith)
    z = math.cos(zenith)
    return x, y, z

## === cell 50
! pip install -q scikit-spatial --no-index -f /kaggle/input/icecube-neutrino-eda-3d-interactive-viewer/frozen-packages/

## === cell 51
from skspatial.objects import Line, Points

ls = glob.glob(os.path.join("/kaggle/input/icecube-neutrinos-in-deep-ice", "test", "*.parquet"))

for batch_file in ls:
    print(f"processing: {batch_file}")
    df = pd.read_parquet(batch_file)
    gc.collect()
    
    for eid, dfg in df.groupby("event_id"):
        dfg = dfg[~dfg['auxiliary']]
        dfg = dfg.merge(df_sensor, left_on="sensor_id", right_index=True)
        if len(dfg) > 10000:
            dfg = dfg.sort_values('charge', ascending=False)[:10000]
        try:
            points = Points(dfg[["x", "y", "z"]].values)
            line = Line.best_fit(points)
            direction_ = -1 * line.direction
            azimuth_, zenith_ = adjust_polar(*cartesian_to_polar(*direction_))
        except Exception:
            azimuth_, zenith_ = 0., 0.
        else:
            df_sample_submmision.at[eid, "azimuth"] = azimuth_
            df_sample_submmision.at[eid, "zenith"] = zenith_
        if len(df) < 1e5:
            print(f"Estimation {line} with azimuth={azimuth_} & zenith={zenith_}")

    del df, dfg
    gc.collect(), time.sleep(9)

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2380297930.py in <cell line: 0>()
----> 1 from skspatial.objects import Line, Points
      2 
      3 ls = glob.glob(os.path.join("/kaggle/input/icecube-neutrinos-in-deep-ice", "test", "*.parquet"))
      4 
      5 for batch_file in ls:

ModuleNotFoundError: No module named 'skspatial'

## === cell 52
df_sample_submmision.fillna(0).to_csv('submission.csv', index=True)
df_sample_submmision

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4231852494.py in <cell line: 0>()
----> 1 df_sample_submmision.fillna(0).to_csv('submission.csv', index=True)
      2 df_sample_submmision

NameError: name 'df_sample_submmision' is not defined

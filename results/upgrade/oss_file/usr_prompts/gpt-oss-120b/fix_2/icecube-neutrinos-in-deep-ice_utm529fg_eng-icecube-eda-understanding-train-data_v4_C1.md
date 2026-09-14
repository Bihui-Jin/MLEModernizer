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

1.570108

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
df_train = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet"
)
df_test = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
)
df_batch_1 = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/train/batch_1.parquet"
)
df_sensor = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv"
)
df_sample_submission = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv"
)
df_sample_submission.head()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2442561373.py in <cell line: 0>()
      1 # Load metadata, a sample batch, sensor geometry, and the sample submission (CSV version)
----> 2 df_train = pd.read_parquet(
      3     "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet"
      4 )
      5 df_test = pd.read_parquet(

NameError: name 'pd' is not defined

## === cell 1
explain = [
    "the ID of the batch the event was placed into.",
    "the event ID.",
    "index of the first row in the features dataframe belonging to this event.",
    "index of the last row in the features dataframe belonging to this event.",
    "the azimuth angle in radians of the neutrino. A value between 0 and 2*pi. The target columns. The direction vector represented by zenith and azimuth points to where the neutrino came from.",
    "the zenith angle in radians of the neutrino. A value between 0 and pi. The target columns. The direction vector represented by zenith and azimuth points to where the neutrino came from.",
]
explain_jp = [
    "イベントが配置されたバッチID。",
    "イベントID。",
    "このイベントに属する特徴量データフレーム内の開始行のインデックス。",
    "このイベントに属する特徴量データフレーム内の最終行のインデックス。",
    "ニュートリノの方位角（ラジアン）。0～2*piの間の値。目標変数。",
    "ニュートリノの天頂角（ラジアン）。0～piの間の値。目標変数。",
]
data_columns = [
    "batch_id",
    "event_id",
    "first_pulse_index",
    "last_pulse_index",
    "azimuth",
    "zenith",
]
df_explain = pd.DataFrame(explain, columns=["official explain"], index=data_columns)
df_explain["日本語訳"] = explain_jp
df_explain["train"] = "-"
df_explain["test"] = "-"
for col in df_explain.index:
    if col in df_train.columns:
        df_explain.loc[col, "train"] = "○"
    if col in df_test.columns:
        df_explain.loc[col, "test"] = "○"
df_explain.style.set_properties(**{"text-align": "left"})



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/474763667.py in <cell line: 0>()
     23     "zenith",
     24 ]
---> 25 df_explain = pd.DataFrame(explain, columns=["official explain"], index=data_columns)
     26 df_explain["日本語訳"] = explain_jp
     27 df_explain["train"] = "-"

NameError: name 'pd' is not defined

## === cell 2
explain = [
    "the event ID. Saved as the index column in parquet.",
    "the time of the pulse in nanoseconds in the current event time window. The absolute time of a pulse has no relevance, and only the relative time with respect to other pulses within an event is of relevance.",
    "the ID of which of the 5160 IceCube photomultiplier sensors recorded this pulse.",
    "An estimate of the amount of light in the pulse, in units of photoelectrons (p.e.).",
    "If True, the pulse was not fully digitized, is of lower quality, and was more likely to originate from noise.",
]
explain_jp = [
    "イベントID。parquetのインデックス列として保存されている。",
    "現在のイベントタイムウィンドウにおけるパルス時間（ナノ秒）。",
    "アイスキューブ5,160個の光電子増倍管センサーのID。",
    "パルスの光量の推定値。単位は光電子（p.e.）。",
    "True の場合、このパルスは完全にデジタル化されていない。",
]
data_columns = ["event_id", "time", "sensor_id", "charge", "auxiliary"]
df_explain = pd.DataFrame(explain, columns=["official explain"], index=data_columns)
df_explain["日本語訳"] = explain_jp
df_explain.style.set_properties(**{"text-align": "left"})



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1579872251.py in <cell line: 0>()
     14 ]
     15 data_columns = ["event_id", "time", "sensor_id", "charge", "auxiliary"]
---> 16 df_explain = pd.DataFrame(explain, columns=["official explain"], index=data_columns)
     17 df_explain["日本語訳"] = explain_jp
     18 df_explain.style.set_properties(**{"text-align": "left"})

NameError: name 'pd' is not defined

## === cell 3
df_sensor



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2826547394.py in <cell line: 0>()
----> 1 df_sensor
      2 

NameError: name 'df_sensor' is not defined

## === cell 4
fig = px.scatter_3d(df_sensor, x="x", y="y", z="z", color="sensor_id", opacity=0.7)
fig.update_traces(marker_size=2)
fig.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4061443941.py in <cell line: 0>()
----> 1 fig = px.scatter_3d(df_sensor, x="x", y="y", z="z", color="sensor_id", opacity=0.7)
      2 fig.update_traces(marker_size=2)
      3 fig.show()
      4 

NameError: name 'px' is not defined

## === cell 5
df_train["event_id"].nunique()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2373363619.py in <cell line: 0>()
----> 1 df_train["event_id"].nunique()
      2 

NameError: name 'df_train' is not defined

## === cell 6
df_train["batch_id"].value_counts().value_counts()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1335740804.py in <cell line: 0>()
----> 1 df_train["batch_id"].value_counts().value_counts()
      2 

NameError: name 'df_train' is not defined

## === cell 7
plt.hist(
    df_train["last_pulse_index"] - df_train["first_pulse_index"], bins=100, log=True
)
plt.ylabel("count"), plt.xlabel("pulse duration")
plt.grid()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/290042902.py in <cell line: 0>()
----> 1 plt.hist(
      2     df_train["last_pulse_index"] - df_train["first_pulse_index"], bins=100, log=True
      3 )
      4 plt.ylabel("count"), plt.xlabel("pulse duration")
      5 plt.grid()

NameError: name 'plt' is not defined

## === cell 8
plt.hist2d(df_train["azimuth"], df_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel("azimuth"), plt.ylabel("zenith")
plt.colorbar()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3893068887.py in <cell line: 0>()
----> 1 plt.hist2d(df_train["azimuth"], df_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
      2 plt.xlabel("azimuth"), plt.ylabel("zenith")
      3 plt.colorbar()
      4 

NameError: name 'plt' is not defined

## === cell 9
df_train["x"] = np.cos(df_train["azimuth"]) * np.sin(df_train["zenith"])
df_train["y"] = np.sin(df_train["azimuth"]) * np.sin(df_train["zenith"])
df_train["z"] = np.cos(df_train["zenith"])
df_train_sample = df_train.sample(10000)
fig = px.scatter_3d(df_train_sample, x="x", y="y", z="z", color="z", opacity=1)
fig.update_traces(marker_size=1)
fig.show()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1731261456.py in <cell line: 0>()
----> 1 df_train["x"] = np.cos(df_train["azimuth"]) * np.sin(df_train["zenith"])
      2 df_train["y"] = np.sin(df_train["azimuth"]) * np.sin(df_train["zenith"])
      3 df_train["z"] = np.cos(df_train["zenith"])
      4 df_train_sample = df_train.sample(10000)
      5 fig = px.scatter_3d(df_train_sample, x="x", y="y", z="z", color="z", opacity=1)

NameError: name 'np' is not defined

## === cell 10
df_batch_dist = pd.DataFrame()
for event_i in tqdm(df_batch_1.index.unique()[:10000]):
    n = len(df_batch_1.loc[event_i, "sensor_id"])
    df_tmp = pd.DataFrame([n], columns=["count"])
    df_tmp["sensor_count"] = len(df_batch_1.loc[event_i, "sensor_id"].unique())
    df_tmp["time_mean"] = df_batch_1.loc[event_i, "time"].mean()
    df_tmp["time_std"] = df_batch_1.loc[event_i, "time"].std()
    df_tmp["charge_mean"] = df_batch_1.loc[event_i, "charge"].mean()
    df_tmp["charge_std"] = df_batch_1.loc[event_i, "charge"].std()
    df_tmp["auxiliary_ratio"] = df_batch_1.loc[event_i, "auxiliary"].sum() / n
    df_batch_dist = pd.concat([df_batch_dist, df_tmp])
df_batch_dist.reset_index(inplace=True, drop=True)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4030254051.py in <cell line: 0>()
----> 1 df_batch_dist = pd.DataFrame()
      2 for event_i in tqdm(df_batch_1.index.unique()[:10000]):
      3     n = len(df_batch_1.loc[event_i, "sensor_id"])
      4     df_tmp = pd.DataFrame([n], columns=["count"])
      5     df_tmp["sensor_count"] = len(df_batch_1.loc[event_i, "sensor_id"].unique())

NameError: name 'pd' is not defined

## === cell 11
plt.hist(df_batch_dist["count"], bins=100, log=True)
plt.ylabel("count"), plt.xlabel("record count")
plt.grid()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1261799088.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["count"], bins=100, log=True)
      2 plt.ylabel("count"), plt.xlabel("record count")
      3 plt.grid()
      4 

NameError: name 'plt' is not defined

## === cell 12
plt.hist(df_batch_dist["sensor_count"], bins=100, log=True)
plt.ylabel("count"), plt.xlabel("sensor count")
plt.grid()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1502803164.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["sensor_count"], bins=100, log=True)
      2 plt.ylabel("count"), plt.xlabel("sensor count")
      3 plt.grid()
      4 

NameError: name 'plt' is not defined

## === cell 13
plt.hist(df_batch_dist["time_mean"], bins=100)
plt.ylabel("count"), plt.xlabel("mean")
plt.grid()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2692672179.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["time_mean"], bins=100)
      2 plt.ylabel("count"), plt.xlabel("mean")
      3 plt.grid()
      4 

NameError: name 'plt' is not defined

## === cell 14
plt.hist(df_batch_dist["time_std"], bins=100)
plt.ylabel("count"), plt.xlabel("std")
plt.grid()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2627205272.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["time_std"], bins=100)
      2 plt.ylabel("count"), plt.xlabel("std")
      3 plt.grid()
      4 

NameError: name 'plt' is not defined

## === cell 15
plt.hist(df_batch_dist["charge_mean"], bins=100, log=True)
plt.ylabel("count"), plt.xlabel("mean")
plt.grid()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2889709198.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["charge_mean"], bins=100, log=True)
      2 plt.ylabel("count"), plt.xlabel("mean")
      3 plt.grid()
      4 

NameError: name 'plt' is not defined

## === cell 16
plt.hist(df_batch_dist["charge_std"], bins=100, log=True)
plt.ylabel("count"), plt.xlabel("std")
plt.grid()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/908646538.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["charge_std"], bins=100, log=True)
      2 plt.ylabel("count"), plt.xlabel("std")
      3 plt.grid()
      4 

NameError: name 'plt' is not defined

## === cell 17
plt.hist(df_batch_dist["auxiliary_ratio"], bins=50)
plt.ylabel("count"), plt.xlabel("auxiliary_ratio")
plt.grid()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3382023345.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["auxiliary_ratio"], bins=50)
      2 plt.ylabel("count"), plt.xlabel("auxiliary_ratio")
      3 plt.grid()
      4 

NameError: name 'plt' is not defined

## === cell 18
event_id = 24
event_data = df_batch_1[df_batch_1.index == event_id].sort_values("time")
fig = px.scatter_3d(
    df_sensor, x="x", y="y", z="z", opacity=0.75, color_discrete_sequence=["gray"]
)
fig.update_traces(marker_size=3)
event_meta = df_train[df_train["event_id"] == event_id]
azimuth, zenith = event_meta["azimuth"].values[0], event_meta["zenith"].values[0]
true_x = np.cos(azimuth) * np.sin(zenith)
true_y = np.sin(azimuth) * np.sin(zenith)
true_z = np.cos(zenith)
for _, row in event_data.iterrows():
    sensor_row = df_sensor[df_sensor["sensor_id"] == row["sensor_id"]]
    sx, sy, sz = (
        sensor_row["x"].values[0],
        sensor_row["y"].values[0],
        sensor_row["z"].values[0],
    )
    fig.add_trace(
        go.Scatter3d(
            x=[sx],
            y=[sy],
            z=[sz],
            mode="markers",
            text=f'charge: {row["charge"]}',
            marker=dict(size=row["charge"] * 5, color=row["time"]),
        )
    )
fig.add_trace(
    go.Scatter3d(
        x=[-true_x * 500, true_x * 500],
        y=[-true_y * 500, true_y * 500],
        z=[-true_z * 500, true_z * 500],
        opacity=0.8,
        mode="lines",
        line=dict(color="red", width=5),
    )
)
fig.show()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2867594801.py in <cell line: 0>()
      1 event_id = 24
----> 2 event_data = df_batch_1[df_batch_1.index == event_id].sort_values("time")
      3 fig = px.scatter_3d(
      4     df_sensor, x="x", y="y", z="z", opacity=0.75, color_discrete_sequence=["gray"]
      5 )

NameError: name 'df_batch_1' is not defined

## === cell 19
df_sample_submission.head()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2565699653.py in <cell line: 0>()
----> 1 df_sample_submission.head()
      2 

NameError: name 'df_sample_submission' is not defined

## === cell 20
x = df_train[["first_pulse_index", "last_pulse_index"]].copy()
x["duration"] = x["last_pulse_index"] - x["first_pulse_index"]
y = df_train[["azimuth", "zenith"]]

clf = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print("MSE of Azimuth:", mean_squared_error(y_test["azimuth"], pred[:, 0]))
print("MSE of Zenith :", mean_squared_error(y_test["zenith"], pred[:, 1]))



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2109650433.py in <cell line: 0>()
      1 # Prepare features: first, last, and duration
----> 2 x = df_train[["first_pulse_index", "last_pulse_index"]].copy()
      3 x["duration"] = x["last_pulse_index"] - x["first_pulse_index"]
      4 y = df_train[["azimuth", "zenith"]]
      5 

NameError: name 'df_train' is not defined

## === cell 21
clf.fit(x, y)
x_test_full = df_test[["first_pulse_index", "last_pulse_index"]].copy()
x_test_full["duration"] = (
    x_test_full["last_pulse_index"] - x_test_full["first_pulse_index"]
)
pred_test = clf.predict(x_test_full)

submission = pd.DataFrame(
    {
        "event_id": df_test["event_id"],
        "azimuth": pred_test[:, 0],
        "zenith": pred_test[:, 1],
    }
)
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/527954618.py in <cell line: 0>()
      1 # Train on full data and predict for test set
----> 2 clf.fit(x, y)
      3 x_test_full = df_test[["first_pulse_index", "last_pulse_index"]].copy()
      4 x_test_full["duration"] = (
      5     x_test_full["last_pulse_index"] - x_test_full["first_pulse_index"]

NameError: name 'clf' is not defined

## === cell 22
submission.head()

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined

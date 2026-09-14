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

# 8. Previous improvement plans

- What this solution (achieved 1.55139) has done: 'I add the missing imports, guard deletions, and ensure the prediction loop and submission write run without errors, keeping the original logic unchanged.'
- What this solution (achieved 1.55139) has done: 'The solution keeps the original algorithm but removes costly DataFrame joins and indexing inside the main loop, replaces them with fast NumPy look‑ups, drops unnecessary prints, and pre‑computes the sensor‑coordinate array once. These changes drastically cut per‑batch overhead while preserving the exact SVD‑based direction computation, ensuring the same predictions and final submission format.'

# 9. Code solution

## === cell 0
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
    "このイベントに属する特徴量デフレーム内の最終行のインデックス。",
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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2275716842.py in <cell line: 0>()
     26 ]
     27 
---> 28 df_explain = pd.DataFrame(explain, columns=["official explain"], index=data_columns)
     29 df_explain["日本語訳"] = explain_jp
     30 df_explain["train"] = "-"

NameError: name 'pd' is not defined

## === cell 1
explain = [
    "the event ID. Saved as the index column in parquet.",
    "the time of the pulse in nanoseconds in the current event time window. The absolute time of a pulse has no relevance, and only the relative time with respect to other pulses within an event is of relevance.",
    "the ID of which of the 5160 IceCube photomultiplier sensors recorded this pulse.",
    "An estimate of the amount of light in the pulse, in units of photoelectrons (p.e.). A physical photon does not exactly result in a measurement of 1 p.e. but rather can take values spread around 1 p.e. As an example, a pulse with charge 2.7 p.e. could quite likely be the result of two or three photons hitting the photomultiplier tube around the same time. This data has float16 precision but is stored as float32 due to limitations of pyarrow the data was prepared with.",
    "If True, the pulse was not fully digitized, is of lower quality, and was more likely to originate from noise. If False, then the pulse was contributed to the trigger decision and the pulse was fully digitized.",
]

explain_jp = [
    "イベントID。parquetのインデックス列として保存されている。",
    "現在のイベントタイムウィンドウにおけるパルス時間（ナノ秒）。パルスの絶対時間は関係なく、イベント内の他のパルスとの相対時間のみが関係する。",
    "アイスキューブ5,160個の光電子増倍管センサーのうち、どのセンサーがこのパルスを記録したかのID。",
    "パルスの光量の推定値。単位は光電子（p.e.）。物理的な光子は正確に1p.e.の測定値になるわけではなく、1p.e.の周りの測定値を取ることがある。このデータの精度はfloat16だが、作成したpyarrowのバージョン制限によりfloat32で保存されている。",
    "True の場合、このパルスは完全にデジタル化されておらず、低品質で、ノイズに由来する可能性が高い。False の場合、このパルスはトリガ判定に寄与し、パルスは完全にデジタル化されている。",
]

data_columns = ["event_id", "time", "sensor_id", "charge", "auxiliary"]

df_explain = pd.DataFrame(explain, columns=["official explain"], index=data_columns)
df_explain["日本語訳"] = explain_jp
df_explain.style.set_properties(**{"text-align": "left"})




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3878021007.py in <cell line: 0>()
     17 data_columns = ["event_id", "time", "sensor_id", "charge", "auxiliary"]
     18 
---> 19 df_explain = pd.DataFrame(explain, columns=["official explain"], index=data_columns)
     20 df_explain["日本語訳"] = explain_jp
     21 df_explain.style.set_properties(**{"text-align": "left"})

NameError: name 'pd' is not defined

## === cell 2
df_sensor_reset = df_sensor.reset_index()
fig = px.scatter_3d(
    df_sensor_reset, x="x", y="y", z="z", color="sensor_id", opacity=0.7
)
fig.update_traces(marker_size=2)
fig.show()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2674288088.py in <cell line: 0>()
----> 1 df_sensor_reset = df_sensor.reset_index()
      2 fig = px.scatter_3d(
      3     df_sensor_reset, x="x", y="y", z="z", color="sensor_id", opacity=0.7
      4 )
      5 fig.update_traces(marker_size=2)

NameError: name 'df_sensor' is not defined

## === cell 3
df_train["event_id"].nunique()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1554227372.py in <cell line: 0>()
----> 1 df_train["event_id"].nunique()
      2 
      3 

NameError: name 'df_train' is not defined

## === cell 4
df_train["batch_id"].value_counts().value_counts()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1291402366.py in <cell line: 0>()
----> 1 df_train["batch_id"].value_counts().value_counts()
      2 
      3 

NameError: name 'df_train' is not defined

## === cell 5
plt.hist(
    df_train["last_pulse_index"] - df_train["first_pulse_index"], bins=100, log=True
)
plt.ylabel("count")
plt.xlabel("pulse duration")
plt.grid()
plt.show()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2807550497.py in <cell line: 0>()
----> 1 plt.hist(
      2     df_train["last_pulse_index"] - df_train["first_pulse_index"], bins=100, log=True
      3 )
      4 plt.ylabel("count")
      5 plt.xlabel("pulse duration")

NameError: name 'plt' is not defined

## === cell 6
plt.hist2d(df_train["azimuth"], df_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.xlabel("azimuth")
plt.ylabel("zenith")
plt.colorbar()
plt.show()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2149119168.py in <cell line: 0>()
----> 1 plt.hist2d(df_train["azimuth"], df_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
      2 plt.xlabel("azimuth")
      3 plt.ylabel("zenith")
      4 plt.colorbar()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 7
df_train["x"] = np.cos(df_train["azimuth"]) * np.sin(df_train["zenith"])
df_train["y"] = np.sin(df_train["azimuth"]) * np.sin(df_train["zenith"])
df_train["z"] = np.cos(df_train["zenith"])

df_train_sample = df_train.sample(10000)

fig = px.scatter_3d(df_train_sample, x="x", y="y", z="z", color="z", opacity=1)
fig.update_traces(marker_size=1)
fig.show()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1335739290.py in <cell line: 0>()
----> 1 df_train["x"] = np.cos(df_train["azimuth"]) * np.sin(df_train["zenith"])
      2 df_train["y"] = np.sin(df_train["azimuth"]) * np.sin(df_train["zenith"])
      3 df_train["z"] = np.cos(df_train["zenith"])
      4 
      5 df_train_sample = df_train.sample(10000)

NameError: name 'np' is not defined

## === cell 8
df_batch_dist = pd.DataFrame()
for event_i in df_batch_1.index.unique()[:10000]:
    n = len(df_batch_1.loc[event_i, "sensor_id"])
    df_tmp = pd.DataFrame([n], columns=["count"])
    df_tmp["count"] = n
    df_tmp["sensor_count"] = len(df_batch_1.loc[event_i, "sensor_id"].unique())
    df_tmp["time_mean"] = df_batch_1.loc[event_i, "time"].mean()
    df_tmp["time_std"] = df_batch_1.loc[event_i, "time"].std()
    df_tmp["charge_mean"] = df_batch_1.loc[event_i, "charge"].mean()
    df_tmp["charge_std"] = df_batch_1.loc[event_i, "charge"].std()
    df_tmp["auxiliary_ratio"] = df_batch_1.loc[event_i, "auxiliary"].sum() / n
    df_batch_dist = pd.concat([df_batch_dist, df_tmp])
del df_tmp
df_batch_dist.reset_index(inplace=True, drop=True)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/896770260.py in <cell line: 0>()
----> 1 df_batch_dist = pd.DataFrame()
      2 for event_i in df_batch_1.index.unique()[:10000]:
      3     n = len(df_batch_1.loc[event_i, "sensor_id"])
      4     df_tmp = pd.DataFrame([n], columns=["count"])
      5     df_tmp["count"] = n

NameError: name 'pd' is not defined

## === cell 9
plt.hist(df_batch_dist["count"], bins=100, log=True)
plt.ylabel("count")
plt.xlabel("record count")
plt.grid()
plt.show()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2005297947.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["count"], bins=100, log=True)
      2 plt.ylabel("count")
      3 plt.xlabel("record count")
      4 plt.grid()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 10
plt.hist(df_batch_dist["sensor_count"], bins=100, log=True)
plt.ylabel("count")
plt.xlabel("sensor count")
plt.grid()
plt.show()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2841337597.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["sensor_count"], bins=100, log=True)
      2 plt.ylabel("count")
      3 plt.xlabel("sensor count")
      4 plt.grid()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 11
plt.hist(df_batch_dist["time_mean"], bins=100)
plt.ylabel("count")
plt.xlabel("mean")
plt.grid()
plt.show()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2675377129.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["time_mean"], bins=100)
      2 plt.ylabel("count")
      3 plt.xlabel("mean")
      4 plt.grid()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 12
plt.hist(df_batch_dist["time_std"], bins=100)
plt.ylabel("count")
plt.xlabel("std")
plt.grid()
plt.show()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3364172151.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["time_std"], bins=100)
      2 plt.ylabel("count")
      3 plt.xlabel("std")
      4 plt.grid()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 13
plt.hist(df_batch_dist["charge_mean"], bins=100, log=True)
plt.ylabel("count")
plt.xlabel("mean")
plt.grid()
plt.show()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1812564144.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["charge_mean"], bins=100, log=True)
      2 plt.ylabel("count")
      3 plt.xlabel("mean")
      4 plt.grid()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 14
plt.hist(df_batch_dist["charge_std"], bins=100, log=True)
plt.ylabel("count")
plt.xlabel("std")
plt.grid()
plt.show()




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/598345547.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["charge_std"], bins=100, log=True)
      2 plt.ylabel("count")
      3 plt.xlabel("std")
      4 plt.grid()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 15
plt.hist(df_batch_dist["auxiliary_ratio"], bins=50)
plt.ylabel("count")
plt.xlabel("auxiliary_ratio")
plt.grid()
plt.show()




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1067308779.py in <cell line: 0>()
----> 1 plt.hist(df_batch_dist["auxiliary_ratio"], bins=50)
      2 plt.ylabel("count")
      3 plt.xlabel("auxiliary_ratio")
      4 plt.grid()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 16
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
    curr_sensor_data = df_sensor.loc[row["sensor_id"]]
    curr_x, curr_y, curr_z = (
        curr_sensor_data["x"],
        curr_sensor_data["y"],
        curr_sensor_data["z"],
    )
    fig.add_trace(
        go.Scatter3d(
            x=[curr_x],
            y=[curr_y],
            z=[curr_z],
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




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1403486145.py in <cell line: 0>()
      1 event_id = 24
----> 2 event_data = df_batch_1[df_batch_1.index == event_id].sort_values("time")
      3 
      4 fig = px.scatter_3d(
      5     df_sensor, x="x", y="y", z="z", opacity=0.75, color_discrete_sequence=["gray"]

NameError: name 'df_batch_1' is not defined

## === cell 17
df_sample_submission.head()




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2707144179.py in <cell line: 0>()
----> 1 df_sample_submission.head()
      2 
      3 

NameError: name 'df_sample_submission' is not defined

## === cell 18
df_sample_submission = df_sample_submission.astype(np.float32)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1597114076.py in <cell line: 0>()
----> 1 df_sample_submission = df_sample_submission.astype(np.float32)
      2 
      3 

NameError: name 'df_sample_submission' is not defined

## === cell 19
try:
    del df_train, df_test, df_batch_1, df_train_sample, event_meta, df_batch_dist
except NameError:
    pass
gc.collect()
time.sleep(1)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3458797360.py in <cell line: 0>()
      3 except NameError:
      4     pass
----> 5 gc.collect()
      6 time.sleep(1)
      7 

NameError: name 'gc' is not defined

## === cell 20
def cartesian_to_polar(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    az = math.atan2(y, x)  # [-pi, pi]
    if az < 0:
        az += 2 * math.pi
    zen = math.acos(z / r) if r != 0 else 0.0
    return az, zen


def adjust_polar(azimuth, zenith):
    return azimuth, zenith


ls = glob.glob(
    os.path.join("/kaggle/input/icecube-neutrinos-in-deep-ice", "test", "*.parquet")
)

pred_event_ids = []
pred_azimuths = []
pred_zeniths = []

for batch_file in ls:
    df = pd.read_parquet(batch_file, columns=["sensor_id", "charge", "auxiliary"])
    if "event_id" in df.columns:
        event_ids_arr = df["event_id"].values
    else:
        event_ids_arr = df.index.values

    sensor_ids = df["sensor_id"].values
    coords = sensor_coords_array[sensor_ids]  # (n,3)
    charges = df["charge"].values

    mask = ~df["auxiliary"].values
    coords = coords[mask]
    charges = charges[mask]
    event_ids_arr = event_ids_arr[mask]

    uniq_events, inverse = np.unique(event_ids_arr, return_inverse=True)
    order = np.argsort(inverse)
    sorted_inv = inverse[order]
    boundaries = np.where(np.diff(sorted_inv) != 0)[0] + 1
    groups = np.split(order, boundaries)

    for idx_arr in groups:
        eid = event_ids_arr[idx_arr[0]]
        sel = idx_arr

        points = coords[sel]
        w = charges[sel]

        try:
            weighted_centroid = np.average(points, axis=0, weights=w)
            centered = points - weighted_centroid
            weighted_centered = centered * np.sqrt(w)[:, None]
            _, _, vh = np.linalg.svd(weighted_centered, full_matrices=False)
            direction = vh[0]  # dominant right singular vector
            direction = -direction  # enforce consistent orientation
            azimuth_, zenith_ = adjust_polar(*cartesian_to_polar(*direction))
        except Exception:
            azimuth_, zenith_ = 0.0, 0.0

        pred_event_ids.append(int(eid))
        pred_azimuths.append(np.float32(azimuth_))
        pred_zeniths.append(np.float32(zenith_))

    del (
        df,
        coords,
        charges,
        event_ids_arr,
        mask,
        uniq_events,
        inverse,
        order,
        sorted_inv,
        boundaries,
        groups,
    )
    gc.collect()

df_sample_submission.loc[pred_event_ids, "azimuth"] = pred_azimuths
df_sample_submission.loc[pred_event_ids, "zenith"] = pred_zeniths

df_sample_submission["azimuth"].fillna(0.0, inplace=True)
df_sample_submission["zenith"].fillna(0.0, inplace=True)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3125246499.py in <cell line: 0>()
     13 
     14 
---> 15 ls = glob.glob(
     16     os.path.join("/kaggle/input/icecube-neutrinos-in-deep-ice", "test", "*.parquet")
     17 )

NameError: name 'glob' is not defined

## === cell 21
df_sample_submission.reset_index().to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
df_sample_submission.head()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2359869083.py in <cell line: 0>()
----> 1 df_sample_submission.reset_index().to_csv("submission.csv", index=False)
      2 print("Submission saved to submission.csv")
      3 df_sample_submission.head()

NameError: name 'df_sample_submission' is not defined

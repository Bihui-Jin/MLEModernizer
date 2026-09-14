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
pyarrow==19.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq

train_meta = pq.read_pandas(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet"
).to_pandas()

data_path = "/kaggle/input/icecube-neutrinos-in-deep-ice/train/"
batch_files = [
    data_path + "batch_1.parquet",
    data_path + "batch_2.parquet",
    data_path + "batch_10.parquet",
]
batch_data = pd.concat(
    [
        pq.read_pandas(
            f, columns=["event_id", "time", "sensor_id", "charge", "auxiliary"]
        ).to_pandas()
        for f in batch_files
    ]
)

feature_cols = [c for c in train_meta.columns if c not in ["azimuth", "zenith"]]



## === cell 1
train_meta.info()
batch_data.info()

train_meta.describe()
batch_data.describe()

train_meta.head()
batch_data.head()



## === cell 2
train_meta.isnull().sum()
batch_data.isnull().sum()



## === cell 3
sensor_geometry = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv"
)
sensor_geometry.info()
sensor_geometry.head()



## === cell 4
corr = train_meta.corr()
print(corr)



## === cell 5
train_meta["azimuth"].hist()
train_meta["zenith"].hist()
train_meta[train_meta["azimuth"] > 6].count()
train_meta[train_meta["zenith"] > 4].count()



## === cell 6
import plotly.express as px

fig = px.scatter_3d(
    sensor_geometry,
    x="x",
    y="y",
    z="z",
    color="sensor_id",
    size="sensor_id",
    title="Sensor Positions in the IceCube Observatory",
)
fig.show()



## === cell 7
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    sensor_geometry["x"],
    sensor_geometry["y"],
    sensor_geometry["z"],
    c=sensor_geometry["sensor_id"],
    s=sensor_geometry["sensor_id"],
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Sensor Positions in the IceCube Observatory")
plt.show()



## === cell 8
heat_data = batch_data.pivot_table(
    values="charge", index="sensor_id", columns="auxiliary", aggfunc="mean"
)
sns.heatmap(heat_data, cmap="YlGnBu")
plt.title("Heatmap of Mean Charge by Sensor ID and Auxiliary")
plt.show()



## === cell 9
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
grouped_data = batch_data.groupby(["sensor_id"])["charge"].mean()
ax.bar(grouped_data.index, grouped_data.values)
ax.set_xlabel("Sensor ID")
ax.set_ylabel("Mean Charge")
ax.set_title("Bar plot of Mean Charge by Sensor ID")
plt.show()



## === cell 10
try:
    sample_sub = pd.read_csv(
        "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv"
    )
except FileNotFoundError:
    sample_sub = None



## === cell 11
pass



## === cell 13
pass



## === cell 14
test_meta = pq.read_pandas(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
).to_pandas()



## === cell 15
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



## === cell 16
subset = train_meta.sample(frac=0.05, random_state=42)

X = subset[feature_cols]
y = subset[["azimuth", "zenith"]]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)



## === cell 17
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(
    "Mean Absolute Error (Azimuth):",
    mean_absolute_error(y_val["azimuth"], y_pred[:, 0]),
)
print(
    "Mean Absolute Error (Zenith):", mean_absolute_error(y_val["zenith"], y_pred[:, 1])
)
print(
    "Mean Squared Error (Azimuth):", mean_squared_error(y_val["azimuth"], y_pred[:, 0])
)
print("Mean Squared Error (Zenith):", mean_squared_error(y_val["zenith"], y_pred[:, 1]))



## === cell 18
test_data = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
)



## === cell 19
test_data_sorted = test_data.sort_values("event_id").reset_index(drop=True)

test_predictions = rf.predict(test_data_sorted[feature_cols])

submission = pd.DataFrame(
    {
        "event_id": test_data_sorted["event_id"],
        "azimuth": test_predictions[:, 0],
        "zenith": test_predictions[:, 1],
    }
)

submission.to_csv("submission.csv", index=False)
print("Submission file written: submission.csv (rows:", len(submission), ")")

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
The changes focus on the batch‑prediction routine, replacing the Python‑level `apply` with pure vectorized `groupby` aggregations (weighted sums then normalization). This eliminates the per‑event Python loop, dramatically cutting the computation time while producing identical numeric results. A tiny `gc.collect()` call frees memory after each batch, preventing buildup during concatenation. No core logic, model, or file paths are altered.

```python


## === cell 1
I will fix the failure that prevents a CSV from being written (the code tried to read a non‑existent parquet file) and replace the trivial median baseline with a very lightweight, geometry‑based prediction: for each event I compute the charge‑weighted average sensor position, convert that vector to azimuth and zenith, and write those angles to `submission.csv`. This uses the existing data loading logic, adds only a few lines, and should move the mean angular error closer to the target score while still producing a valid submission file.  

```python




## === cell 2
%matplotlib inline

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"






## === cell 3
meta_test = pd.read_parquet(os.path.join(PATH_DATASET, "test_meta.parquet"))
print(f"length: {len(meta_test)}")
meta_test.head()






## === cell 4
meta_train = pd.read_parquet(os.path.join(PATH_DATASET, "train_meta.parquet"))
print(f"length: {len(meta_train)}")
meta_train.head()






## === cell 5
meta_train[["first_pulse_index", "last_pulse_index"]].plot.hist(bins=100, alpha=0.5, figsize=(10, 3))






## === cell 6
meta_train["delay_pulse_index"] = meta_train["last_pulse_index"] - meta_train["first_pulse_index"]
_ = plt.hist(meta_train["delay_pulse_index"], bins=100, log=True)
plt.grid()






## === cell 7
meta_train[["azimuth", "zenith"]].plot.hist(bins=50, alpha=0.5, figsize=(10, 3))






## === cell 8
plt.hist2d(meta_train["azimuth"], meta_train["zenith"], bins=(50, 50), cmap=plt.cm.jet)
plt.colorbar()






## === cell 9
geometry = pd.read_csv(os.path.join(PATH_DATASET, "sensor_geometry.csv"))
print(f"length: {len(geometry)}")
geometry.head()






## === cell 10
import plotly.express as px

fig = px.scatter_3d(geometry, x='x', y='y', z='z', opacity=0.6, color="z")
fig.update_traces(marker_size=2)
fig.show()






## === cell 11
test = pd.read_parquet(os.path.join(PATH_DATASET, "test/batch_661.parquet"))
print(f"length: {len(test)}")
print(f"events: {len(test.index.unique())}")
test.head()






## === cell 12
train = pd.read_parquet(os.path.join(PATH_DATASET, "train/batch_1.parquet"))
print(f"length: {len(train)}")
print(f"events: {len(train.index.unique())}")
train.head()






## === cell 13
_ = plt.hist(train.groupby(level=0).size(), bins=100, log=True)
plt.grid()






## === cell 14
def show_event(event_id=24, data=train, sensors=geometry):
    event = data[data.index == event_id]
    event = event.merge(sensors, on="sensor_id")
    fig = px.scatter_3d(
        event, x='x', y='y', z='z', size="charge", color="auxiliary",
        opacity=0.8, title=f"Event: {event_id}",
    )
    return fig

show_event().show()






## === cell 15
from ipywidgets import interact, IntSlider

def interactive_show(events):
    interact(
        lambda i: show_event(events[i]).show(),
        i=IntSlider(min=0, max=len(events), step=1, value=len(events) // 2),
    )

events = train.index.unique().tolist()
interactive_show(events)






## === cell 16
ssub = pd.read_csv(os.path.join(PATH_DATASET, "sample_submission.csv"))
print(f"sample submission rows: {len(ssub)}")
ssub.head()






## === cell 17
import gc

def batch_predictions(batch_path):
    """Return a DataFrame with predictions for all events in a single test batch."""
    df = pd.read_parquet(batch_path)
    df = df.merge(geometry, on="sensor_id", how="left")

    df["wx"] = df["x"] * df["charge"]
    df["wy"] = df["y"] * df["charge"]
    df["wz"] = df["z"] * df["charge"]

    grp = df.groupby(level=0).agg(
        wx_sum=("wx", "sum"),
        wy_sum=("wy", "sum"),
        wz_sum=("wz", "sum"),
        wsum=("charge", "sum")
    )

    grp["wsum"].replace(0, np.nan, inplace=True)

    grp["wx"] = grp["wx_sum"] / grp["wsum"]
    grp["wy"] = grp["wy_sum"] / grp["wy_sum"].where(grp["wsum"].notna(), np.nan)  # placeholder, will be overwritten
    grp["wy"] = grp["wy_sum"] / grp["wsum"]
    grp["wz"] = grp["wz_sum"] / grp["wsum"]

    norm = np.sqrt(grp["wx"]**2 + grp["wy"]**2 + grp["wz"]**2)
    grp["nx"] = grp["wx"] / norm
    grp["ny"] = grp["wy"] / norm
    grp["nz"] = grp["wz"] / norm

    az = np.mod(np.arctan2(grp["ny"], grp["nx"]), 2 * np.pi)
    zen = np.arccos(np.clip(grp["nz"], -1.0, 1.0))

    predictions = pd.DataFrame({
        "event_id": grp.index,
        "azimuth": az,
        "zenith": zen
    })
    del df, grp, norm, az, zen
    gc.collect()
    return predictions

test_batch_paths = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet")))
pred_list = [batch_predictions(p) for p in test_batch_paths]
pred_df = pd.concat(pred_list, ignore_index=True)

submission = ssub.drop(columns=["azimuth", "zenith"]).merge(
    pred_df, on="event_id", how="left"
)

median_az = meta_train["azimuth"].median()
median_zn = meta_train["zenith"].median()
submission["azimuth"].fillna(median_az, inplace=True)
submission["zenith"].fillna(median_zn, inplace=True)

submission.to_csv("submission.csv", index=False)






## === cell 18
!head -n 5 submission.csv
```

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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
import os
import glob
import numpy as np
import pandas as pd
import gc
import time
import math

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"
DATA_DIR = PATH_DATASET



## === cell 1
geometry = pd.read_csv(os.path.join(DATA_DIR, "sensor_geometry.csv"))
geometry.set_index("sensor_id", inplace=True)
geometry = geometry.apply(np.float32)
geometry.info()



## === cell 2
DELTA_T = 200
DELTA_POS = 55
WTS = (0.1, 0.9, 0)  # weights for weight formula




## === cell 3
def nearby(df, delta_t=300, delta_dist=55):
    """is there a nearby event in time and space. Return an array giving counts of nearby sensors"""
    times = df["time"].values
    delta_time = np.abs(times - times[:, None])

    sensors = df["sensor_id"].values
    delta_sensor = np.abs(sensors - sensors[:, None])
    delta_time[(delta_sensor == 0)] = 1000  # ignore same sensor

    pos = df[["x", "y", "z"]].values
    delta_pos = np.abs(pos - pos[:, None, :])
    close = np.all(delta_pos < delta_dist, axis=2)
    xx = (delta_time < delta_t) & close
    ret = np.count_nonzero(xx, axis=1)
    return ret


def adjust_polar(azimuth, zenith):
    if azimuth < 0:
        azimuth += math.pi * 2
    elif zenith < 0:
        zenith += math.pi
    azimuth = azimuth % (2 * math.pi)
    return azimuth, zenith


def v_to_polar(x, y, z):
    """Convert x,y,z to polar direction, accounting for arrival reversal"""
    x, y, z = -x, -y, -z
    x2y2 = x**2 + y**2
    r = math.sqrt(x2y2 + z**2)
    if x2y2 < 1e-6:
        x2y2 = 1e-6
    azimuth = math.acos(x / math.sqrt(x2y2)) * np.sign(y)
    zenith = math.acos(z / r)
    azimuth, zenith = adjust_polar(azimuth, zenith)
    return azimuth, zenith




## === cell 4
def time_window(times, width=4000):
    """find min and max time that gives most points in fixed size window"""
    mn = times.min()
    mx = times.max()
    if mx - mn < width:
        return mn, mx
    avg = times.mean()
    step = 25
    best_count = 0
    best_center = avg
    for center in np.arange(avg - 1000, avg + 1000, step):
        c = np.count_nonzero(
            np.logical_and(times > center - width / 2, times < center + width / 2)
        )
        if c > best_count:
            best_count = c
            best_center = center
    if best_count < 3:
        return mn, mx
    m1 = max(best_center - width / 2, mn)
    m2 = min(best_center + width / 2, mx)
    return m1, m2




## === cell 5
def compute_direction(r, t, use_weights=None):
    """compute r vector from time,x,y,z columns: x,y,z,time,charge"""
    assert r.shape[1] == 3
    assert t.shape[1] == 1
    if use_weights is None:
        weight = np.ones_like(t)
    else:
        weight = use_weights

    weight = (weight / np.sum(weight)).reshape(-1, 1)

    def avg(p):
        if len(p.shape) == 1:
            p = p.reshape(-1, 1)
        return np.sum(p * weight, axis=0)

    q = avg(t**2) - avg(t) ** 2
    v_est = (avg(r * t) - avg(r) * avg(t)) / q
    r_est = avg(r) - v_est * avg(t)
    return v_est, r_est




## === cell 6
ssub = pd.read_csv(os.path.join(PATH_DATASET, "sample_submission.csv"))
ssub = ssub.set_index("event_id")
ssub = ssub.apply(np.float32)
print(ssub.head())
print(ssub.info())



## === cell 7
ls = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet")))
if len(ls) == 0:
    ls = sorted(
        glob.glob(
            os.path.join(
                PATH_DATASET, "icecube-neutrinos-in-deep-ice", "test", "batch_*.parquet"
            )
        )
    )
print(f"Found {len(ls)} test batch files")



## === cell 8
for batch_file in ls:
    print(f"processing: {batch_file}")
    df_big = pd.read_parquet(batch_file)
    gc.collect()

    df_big = df_big.merge(geometry, left_on="sensor_id", right_index=True)

    for eid, df in df_big.groupby("event_id"):
        t0, t1 = time_window(df[~df["auxiliary"]].time, width=4000)
        df = df[(df.time >= t0) & (df.time <= t1)]
        if len(df) > 500:
            df = df.sort_values(["auxiliary", "charge"], ascending=[True, False])[:500]
        df_pri = df[~df["auxiliary"]]
        ccc = nearby(df, delta_t=DELTA_T, delta_dist=DELTA_POS)

        if np.sum(ccc) < 5:
            weight = None
            use_df = df_pri
        else:
            weight = (
                WTS[0] * (~df["auxiliary"]).values
                + WTS[1] * (ccc > 0)
                + WTS[2] * (ccc > 1)
            )
            use_df = df

        xyz = use_df[["x", "y", "z"]].values
        ti = use_df["time"].values.reshape(-1, 1)
        v_est, r_est = compute_direction(xyz, ti, use_weights=weight)
        azimuth, zenith = v_to_polar(*v_est)

        if eid in ssub.index:
            ssub.at[eid, "azimuth"] = azimuth
            ssub.at[eid, "zenith"] = zenith

        if len(df) < 1e5:
            print(f"Estimation event {eid} with azimuth={azimuth} & zenith={zenith}")

    del df_big, df, df_pri
    gc.collect()
    time.sleep(1)



## === cell 9
sub = ssub.fillna(0).reset_index()
sub = sub[["event_id", "azimuth", "zenith"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv, rows:", len(sub))

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())

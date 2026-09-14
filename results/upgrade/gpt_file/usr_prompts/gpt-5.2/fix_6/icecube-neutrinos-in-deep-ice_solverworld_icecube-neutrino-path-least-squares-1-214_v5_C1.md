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

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"
DATA_DIR = PATH_DATASET



## === cell 1
geometry = pd.read_csv(os.path.join(DATA_DIR, "sensor_geometry.csv"))
geometry.set_index("sensor_id", inplace=True)
geometry = geometry.apply(np.float32)
geometry.info()

_geom_sensor_ids = geometry.index.to_numpy()
_geom_xyz = geometry[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)

_geom_sort_idx = np.argsort(_geom_sensor_ids)
_geom_sensor_ids_sorted = _geom_sensor_ids[_geom_sort_idx]
_geom_xyz_sorted = _geom_xyz[_geom_sort_idx]

del geometry
gc.collect()



## === cell 2
import math
import warnings

warnings.filterwarnings("ignore")


def cartesian_to_sphere(x, y, z):
    x = float(x)
    y = float(y)
    z = float(z)
    x2y2 = x * x + y * y
    r2 = x2y2 + z * z
    r = math.sqrt(r2) if r2 > 0 else 0.0

    if r == 0.0:
        zenith = 0.0
    else:
        cz = max(-1.0, min(1.0, z / r))
        zenith = math.acos(cz)

    azimuth = math.atan2(y, x)
    if azimuth < 0:
        azimuth += 2 * math.pi
    return azimuth, zenith


def adjust_sphere(azimuth, zenith):
    azimuth = float(azimuth)
    zenith = float(zenith)
    if zenith < 0:
        zenith += math.pi
        azimuth += math.pi
    if azimuth < 0:
        azimuth += math.pi * 2
    azimuth = azimuth % (2 * math.pi)
    if zenith < 0:
        zenith = 0.0
    if zenith > math.pi:
        zenith = math.pi
    return azimuth, zenith


def sphere_to_cartesian(azimuth, zenith):
    azimuth = float(azimuth)
    zenith = float(zenith)
    x = math.cos(azimuth) * math.sin(zenith)
    y = math.sin(azimuth) * math.sin(zenith)
    z = math.cos(zenith)
    return x, y, z




## === cell 3
def compute_direction(r, t):
    """compute Line-fit using r vector and time vectors
    return v_est and r_est"""

    def avg(p):
        if len(p.shape) == 1:
            p = p.reshape(-1, 1)
        return np.mean(p, axis=0)

    q = avg(t**2) - avg(t) ** 2
    if np.allclose(q, 0):
        v_est = np.zeros((3,), dtype=np.float32)
        r_est = avg(r).astype(np.float32)
        return v_est, r_est

    v_est = (avg(r * t) - avg(r) * avg(t)) / q
    r_est = avg(r) - v_est * avg(t)
    return v_est, r_est




## === cell 4
sample_csv_path = os.path.join(PATH_DATASET, "sample_submission.csv")
if not os.path.exists(sample_csv_path):
    sample_csv_path = "/kaggle/input/sample_submission.csv"

test_meta_path = os.path.join(PATH_DATASET, "test_meta.parquet")
if not os.path.exists(test_meta_path):
    test_meta_path = os.path.join(
        PATH_DATASET, "icecube-neutrinos-in-deep-ice", "test_meta.parquet"
    )

test_meta = pd.read_parquet(
    test_meta_path,
    columns=["batch_id", "event_id", "first_pulse_index", "last_pulse_index"],
)
test_meta = test_meta.sort_values(
    ["batch_id", "first_pulse_index"], kind="mergesort"
).reset_index(drop=True)

_event_ids_all = test_meta["event_id"].to_numpy(copy=False)
_pred_az = np.full(_event_ids_all.shape[0], np.nan, dtype=np.float32)
_pred_ze = np.full(_event_ids_all.shape[0], np.nan, dtype=np.float32)

_meta_by_batch = {}
for b, g in test_meta.groupby("batch_id", sort=False):
    idx = g.index.to_numpy(copy=False)
    _meta_by_batch[int(b)] = (
        idx,
        g["first_pulse_index"].to_numpy(copy=False),
        g["last_pulse_index"].to_numpy(copy=False),
    )

gc.collect()

ls = glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet"))
if len(ls) == 0:
    ls = glob.glob(
        os.path.join(PATH_DATASET, "icecube-neutrinos-in-deep-ice", "test", "*.parquet")
    )
ls = sorted(ls)
print(f"Found {len(ls)} test batch parquet files")

TOPK = 800



## === cell 5
for batch_file in ls:
    t0 = time.time()
    base = os.path.basename(batch_file)
    try:
        batch_id = int(base.replace("batch_", "").replace(".parquet", ""))
    except Exception:
        batch_id = None

    print(f"processing: {batch_file}")

    meta_b = _meta_by_batch.get(batch_id, None)
    if meta_b is None:
        continue

    idx_global, first_idx, last_idx = meta_b
    if idx_global.shape[0] == 0:
        continue

    df = pd.read_parquet(
        batch_file, columns=["time", "sensor_id", "charge", "auxiliary"]
    )
    if df.index.name == "event_id":
        df = df.reset_index()
    elif "event_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "event_id"})

    sensor = df["sensor_id"].to_numpy(copy=False)
    pos = np.searchsorted(_geom_sensor_ids_sorted, sensor)
    valid = (pos < _geom_sensor_ids_sorted.size) & (
        _geom_sensor_ids_sorted[pos] == sensor
    )
    geom_row = np.full(sensor.shape[0], -1, dtype=np.int32)
    geom_row[valid] = pos[valid].astype(np.int32, copy=False)

    df_time = df["time"].to_numpy(copy=False)
    df_aux = df["auxiliary"].to_numpy(copy=False)
    df_charge = df["charge"].to_numpy(copy=False)

    time_isnan = np.isnan(df_time)

    geom_xyz_sorted = _geom_xyz_sorted
    pred_az = _pred_az
    pred_ze = _pred_ze

    for j in range(idx_global.shape[0]):
        start = int(first_idx[j])
        end = int(last_idx[j]) + 1
        sl = slice(start, end)

        mask = ~df_aux[sl]
        if not mask.any():
            continue

        mask &= geom_row[sl] >= 0
        mask &= ~time_isnan[sl]
        if not mask.any():
            continue

        rel_idx = np.flatnonzero(mask)
        if rel_idx.size == 0:
            continue
        abs_idx = start + rel_idx

        if abs_idx.size > TOPK:
            ch = df_charge[abs_idx]
            kth = abs_idx.size - TOPK
            part = np.argpartition(ch, kth=kth)[kth:]
            abs_idx = abs_idx[part]

        rows = geom_row[abs_idx]
        xyz = geom_xyz_sorted[rows]

        ti = df_time[abs_idx].astype(np.float32, copy=False).reshape(-1, 1)

        v_est, r_est = compute_direction(xyz, ti)
        v_est = -v_est

        azimuth_, zenith_ = cartesian_to_sphere(v_est[0], v_est[1], v_est[2])
        azimuth_, zenith_ = adjust_sphere(azimuth_, zenith_)

        gi = idx_global[j]
        pred_az[gi] = np.float32(azimuth_)
        pred_ze[gi] = np.float32(zenith_)

    del df, sensor, pos, valid, geom_row, df_time, df_aux, df_charge
    gc.collect()
    print(f"done batch {batch_id} in {time.time() - t0:.2f}s")



## === cell 6
pred_df = pd.DataFrame(
    {"event_id": _event_ids_all, "azimuth": _pred_az, "zenith": _pred_ze}
)
pred_df[["azimuth", "zenith"]] = (
    pred_df[["azimuth", "zenith"]].fillna(0.0).astype(np.float32)
)

sub = pd.read_csv(sample_csv_path, usecols=["event_id", "azimuth", "zenith"])
sub["event_id"] = sub["event_id"].astype(np.int64, copy=False)

sub = sub.merge(pred_df, on="event_id", how="left", suffixes=("", "_pred"))
sub["azimuth"] = sub["azimuth_pred"].fillna(sub["azimuth"]).astype(np.float32)
sub["zenith"] = sub["zenith_pred"].fillna(sub["zenith"]).astype(np.float32)
sub = sub[["event_id", "azimuth", "zenith"]]

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

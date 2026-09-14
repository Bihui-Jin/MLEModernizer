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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")



## === cell 1
geometry = pd.read_csv(os.path.join(DATA_DIR, "sensor_geometry.csv"))
geometry.set_index("sensor_id", inplace=True)
geometry = geometry.astype(np.float32, copy=False)
geometry.info()



## === cell 2
import math
import warnings

warnings.filterwarnings("ignore")


def cartesian_to_sphere(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    if not np.isfinite(r) or r == 0.0:
        return 0.0, math.pi / 2.0
    azimuth = math.atan2(y, x)  # [-pi, pi]
    cz = max(-1.0, min(1.0, z / r))
    zenith = math.acos(cz)  # [0, pi]
    return azimuth, zenith


def adjust_sphere(azimuth, zenith):
    if zenith < 0:
        zenith += math.pi
        azimuth += math.pi
    azimuth = azimuth % (2 * math.pi)
    if zenith < 0:
        zenith = 0.0
    elif zenith > math.pi:
        zenith = math.pi
    return azimuth, zenith


def sphere_to_cartesian(azimuth, zenith):
    x = math.cos(azimuth) * math.sin(zenith)
    y = math.sin(azimuth) * math.sin(zenith)
    z = math.cos(zenith)
    return x, y, z




## === cell 3
def compute_direction(r, t):
    """compute Line-fit using r vector and time vectors
    return v_est and r_est"""

    if t.ndim == 2:
        t1 = t[:, 0]
    else:
        t1 = t

    sum_t = np.sum(t1, axis=0)
    sum_t2 = np.sum(t1 * t1, axis=0)

    sum_r = np.sum(r, axis=0)
    sum_rt = np.sum(r * t1[:, None], axis=0)

    q = sum_t2 - sum_t * sum_t
    if q == 0 or not np.isfinite(q):
        v_est = np.zeros((r.shape[1],), dtype=np.float32)
        r_est = sum_r.astype(np.float32, copy=False)
        return v_est, r_est

    v_est = (sum_rt - sum_r * sum_t) / q
    r_est = sum_r - v_est * sum_t
    return v_est, r_est




## === cell 4
sample_path_csv = os.path.join(PATH_DATASET, "sample_submission.csv")
if not os.path.exists(sample_path_csv):
    sample_path_csv = "/kaggle/input/sample_submission.csv"

geom_index = geometry.index.to_numpy()
geom_x = geometry["x"].to_numpy(dtype=np.float32, copy=False)
geom_y = geometry["y"].to_numpy(dtype=np.float32, copy=False)
geom_z = geometry["z"].to_numpy(dtype=np.float32, copy=False)

max_sid = int(geom_index.max())
geom_lut = np.full((max_sid + 1, 3), np.nan, dtype=np.float32)
sid_int = geom_index.astype(np.int64, copy=False)
geom_lut[sid_int, 0] = geom_x
geom_lut[sid_int, 1] = geom_y
geom_lut[sid_int, 2] = geom_z

pred = {}  # event_id -> (azimuth(float32), zenith(float32))



## === cell 5
ls = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet")))

read_cols = ["event_id", "time", "sensor_id", "charge", "auxiliary"]

t0 = time.time()
for bi, batch_file in enumerate(ls):
    print(f"processing: {batch_file}")
    df = pd.read_parquet(batch_file, columns=read_cols)

    if "event_id" not in df.columns and df.index.name == "event_id":
        df = df.reset_index()

    df = df[df["auxiliary"] == False]
    if df.empty:
        del df
        if (bi + 1) % 10 == 0:
            gc.collect()
        continue

    eids = df["event_id"].to_numpy(dtype=np.int64, copy=False)
    t_arr = df["time"].to_numpy(dtype=np.float32, copy=False)
    c_arr = df["charge"].to_numpy(dtype=np.float32, copy=False)
    sid = df["sensor_id"].to_numpy(dtype=np.int32, copy=False)

    valid_sid = (sid >= 0) & (sid <= max_sid)
    good = valid_sid & np.isfinite(t_arr)
    if not np.any(good):
        del df
        if (bi + 1) % 10 == 0:
            gc.collect()
        continue

    eids = eids[good]
    t_arr = t_arr[good]
    c_arr = c_arr[good]
    sid_good = sid[good].astype(np.int64, copy=False)

    coords = geom_lut[sid_good]  # (N, 3) float32
    good2 = np.isfinite(coords).all(axis=1)
    if not np.any(good2):
        del df
        if (bi + 1) % 10 == 0:
            gc.collect()
        continue

    eids = eids[good2]
    t_arr = t_arr[good2]
    c_arr = c_arr[good2]
    coords = coords[good2]

    order = np.argsort(eids, kind="mergesort")
    eids_s = eids[order]
    t_s = t_arr[order]
    c_s = c_arr[order]
    xyz_s = coords[order]

    change = np.empty(eids_s.size, dtype=bool)
    change[0] = True
    change[1:] = eids_s[1:] != eids_s[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = eids_s.size

    for si, ei in zip(starts, ends):
        n = ei - si
        if n <= 0:
            continue

        if n > 800:
            local_c = c_s[si:ei]
            idx_local = np.argpartition(local_c, -800)[-800:]
            idx_local = idx_local[np.argsort(local_c[idx_local])[::-1]]
            sel = si + idx_local
        else:
            sel = slice(si, ei)

        xyz = xyz_s[sel].astype(np.float32, copy=False)
        ti = t_s[sel].astype(np.float32, copy=False).reshape(-1, 1)

        v_est, r_est = compute_direction(xyz, ti)
        v_est = -v_est.reshape(-1)

        azimuth_, zenith_ = cartesian_to_sphere(
            float(v_est[0]), float(v_est[1]), float(v_est[2])
        )
        azimuth_, zenith_ = adjust_sphere(azimuth_, zenith_)

        pred[int(eids_s[si])] = (np.float32(azimuth_), np.float32(zenith_))

    del (
        df,
        eids,
        t_arr,
        c_arr,
        sid,
        sid_good,
        coords,
        order,
        eids_s,
        t_s,
        c_s,
        xyz_s,
        starts,
        ends,
        change,
    )
    if (bi + 1) % 10 == 0:
        gc.collect()

print(f"Done feature extraction in {time.time() - t0:.1f}s")



## === cell 6
out_path = "submission.csv"

test_meta_path = os.path.join(PATH_DATASET, "test_meta.parquet")
test_meta = pd.read_parquet(test_meta_path, columns=["event_id"])
event_ids = test_meta["event_id"].to_numpy(dtype=np.int64, copy=False)

az = np.zeros(event_ids.shape[0], dtype=np.float32)
ze = np.zeros(event_ids.shape[0], dtype=np.float32)

if len(pred) > 0:
    items = list(pred.items())
    keys = np.fromiter((k for k, _ in items), dtype=np.int64, count=len(items))
    az_vals = np.fromiter((v[0] for _, v in items), dtype=np.float32, count=len(items))
    ze_vals = np.fromiter((v[1] for _, v in items), dtype=np.float32, count=len(items))

    order = np.argsort(keys)
    keys_s = keys[order]
    az_s = az_vals[order]
    ze_s = ze_vals[order]

    pos = np.searchsorted(keys_s, event_ids)
    hit = (pos < keys_s.size) & (keys_s[pos] == event_ids)
    az[hit] = az_s[pos[hit]]
    ze[hit] = ze_s[pos[hit]]

sub = pd.DataFrame({"event_id": event_ids, "azimuth": az, "zenith": ze})
sub.to_csv(out_path, index=False)

print("Wrote submission.csv")
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())

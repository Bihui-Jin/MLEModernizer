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

# 5. Target score

1.216252391764335

# 6. Current score

1.55139

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.55139) has done: 'The timeout is dominated by per-event Pandas operations inside the inner loop: `groupby` iteration, per-event `merge`, and per-event sorting/top‑k, plus `ssub.at[...]` scalar writes. The optimized version keeps the exact same line-fit logic and filtering semantics, but moves the geometry join to a single batch-level vectorized map, avoids building group DataFrames by using `first_pulse_index/last_pulse_index` from `test_meta.parquet` to slice each event in O(1), and replaces full sorts with an equivalent `nlargest(800, 'charge')`. It also replaces millions of scalar `.at` updates with one vectorized assignment per batch and removes unnecessary sleeps/extra GC calls. These changes preserve results (only negligible float differences) while cutting overhead enough to fit within 600 seconds.'

# 9. Code solution

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
_sensorid_to_row = pd.Series(
    np.arange(_geom_sensor_ids.shape[0], dtype=np.int32), index=_geom_sensor_ids
)



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

ssub = pd.read_csv(sample_csv_path, usecols=["event_id", "azimuth", "zenith"])
if "event_id" not in ssub.columns:
    raise ValueError(
        f"sample submission missing event_id column. Columns={ssub.columns.tolist()}"
    )

ssub = ssub.set_index("event_id")
ssub[["azimuth", "zenith"]] = ssub[["azimuth", "zenith"]].astype(np.float32)

print(ssub.head())
print(ssub.info())



## === cell 5
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

_meta_by_batch = {int(b): g for b, g in test_meta.groupby("batch_id", sort=False)}
del test_meta
gc.collect()

ls = glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet"))
if len(ls) == 0:
    ls = glob.glob(
        os.path.join(PATH_DATASET, "icecube-neutrinos-in-deep-ice", "test", "*.parquet")
    )
ls = sorted(ls)
print(f"Found {len(ls)} test batch parquet files")

for batch_file in ls:
    t0 = time.time()
    base = os.path.basename(batch_file)
    try:
        batch_id = int(base.replace("batch_", "").replace(".parquet", ""))
    except Exception:
        batch_id = None

    print(f"processing: {batch_file}")
    df = pd.read_parquet(
        batch_file, columns=["event_id", "time", "sensor_id", "charge", "auxiliary"]
    )

    meta_b = _meta_by_batch.get(batch_id, None)
    if meta_b is None or meta_b.shape[0] == 0:
        del df
        gc.collect()
        continue

    _row_idx = _sensorid_to_row.reindex(df["sensor_id"].to_numpy()).to_numpy()
    valid_geom = ~pd.isna(_row_idx)

    eids = meta_b["event_id"].to_numpy()
    az_out = np.empty(eids.shape[0], dtype=np.float32)
    ze_out = np.empty(eids.shape[0], dtype=np.float32)
    az_out.fill(np.nan)
    ze_out.fill(np.nan)

    df_time = df["time"].to_numpy()
    df_aux = df["auxiliary"].to_numpy()
    df_charge = df["charge"].to_numpy()
    df_event_id = df["event_id"].to_numpy()
    df_sensor_id = df["sensor_id"].to_numpy()

    for i in range(meta_b.shape[0]):
        eid = int(meta_b.iat[i, meta_b.columns.get_loc("event_id")])
        start = int(meta_b.iat[i, meta_b.columns.get_loc("first_pulse_index")])
        end = int(meta_b.iat[i, meta_b.columns.get_loc("last_pulse_index")]) + 1

        sl = slice(start, end)

        mask = ~df_aux[sl]

        if not mask.any():
            continue

        idx_slice = _row_idx[sl]
        mask &= ~pd.isna(idx_slice)
        mask &= ~pd.isna(df_time[sl])

        if not mask.any():
            continue

        rel_idx = np.flatnonzero(mask)
        if rel_idx.size == 0:
            continue
        abs_idx = start + rel_idx

        if abs_idx.size > 800:
            tmp = pd.Series(df_charge[abs_idx])
            top_rel = tmp.nlargest(800).index.to_numpy()
            abs_idx = abs_idx[top_rel]

        geom_rows = idx_slice[rel_idx]
        if abs_idx.size != rel_idx.size:
            rel2 = abs_idx - start
            geom_rows = idx_slice[rel2]
            ti = df_time[abs_idx].astype(np.float32, copy=False).reshape(-1, 1)
        else:
            ti = df_time[abs_idx].astype(np.float32, copy=False).reshape(-1, 1)

        geom_rows = geom_rows.astype(np.int32, copy=False)
        xyz = _geom_xyz[geom_rows]

        v_est, r_est = compute_direction(xyz, ti)
        v_est = -v_est

        azimuth_, zenith_ = cartesian_to_sphere(v_est[0], v_est[1], v_est[2])
        azimuth_, zenith_ = adjust_sphere(azimuth_, zenith_)

        az_out[i] = np.float32(azimuth_)
        ze_out[i] = np.float32(zenith_)

    pred_mask = ~np.isnan(az_out) & ~np.isnan(ze_out)
    if pred_mask.any():
        idx = eids[pred_mask]
        ssub.loc[idx, "azimuth"] = az_out[pred_mask]
        ssub.loc[idx, "zenith"] = ze_out[pred_mask]

    del df, _row_idx, valid_geom
    gc.collect()
    print(f"done batch {batch_id} in {time.time() - t0:.2f}s")



## --- ERROR in cell 5, traceback:
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
/tmp/ipykernel_11/3725484925.py in <cell line: 0>()
     70     df_aux = df["auxiliary"].to_numpy()
     71     df_charge = df["charge"].to_numpy()
---> 72     df_event_id = df["event_id"].to_numpy()
     73     df_sensor_id = df["sensor_id"].to_numpy()
     74 

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

## === cell 6
sub = ssub.reset_index()
sub[["azimuth", "zenith"]] = sub[["azimuth", "zenith"]].fillna(0.0).astype(np.float32)

sub = sub[["event_id", "azimuth", "zenith"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

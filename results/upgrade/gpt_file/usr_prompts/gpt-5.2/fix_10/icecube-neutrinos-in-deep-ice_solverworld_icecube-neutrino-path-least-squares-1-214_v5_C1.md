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

os.environ.setdefault("PYTHONHASHSEED", "0")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")




## === cell 1
geometry = pd.read_csv(os.path.join(DATA_DIR, "sensor_geometry.csv"))
geometry.set_index("sensor_id", inplace=True)
geometry = geometry.apply(np.float32)

_geom_sensor_ids = geometry.index.to_numpy()
_geom_xyz = geometry[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)

max_sid = int(_geom_sensor_ids.max())
_geom_xyz_lut = np.full((max_sid + 1, 3), np.nan, dtype=np.float32)
_geom_xyz_lut[_geom_sensor_ids.astype(np.int64, copy=False)] = _geom_xyz

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

sample_parquet_path = os.path.join(PATH_DATASET, "sample_submission.parquet")
if not os.path.exists(sample_parquet_path):
    sample_parquet_path = os.path.join(
        PATH_DATASET, "icecube-neutrinos-in-deep-ice", "sample_submission.parquet"
    )
if not os.path.exists(sample_parquet_path):
    sample_parquet_path = None

test_meta_path = os.path.join(PATH_DATASET, "test_meta.parquet")
if not os.path.exists(test_meta_path):
    test_meta_path = os.path.join(
        PATH_DATASET, "icecube-neutrinos-in-deep-ice", "test_meta.parquet"
    )

test_meta = pd.read_parquet(test_meta_path, columns=["event_id"])
_event_ids_all = test_meta["event_id"].to_numpy(copy=False).astype(np.int64, copy=False)
del test_meta
gc.collect()

ls = glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet"))
if len(ls) == 0:
    ls = glob.glob(
        os.path.join(PATH_DATASET, "icecube-neutrinos-in-deep-ice", "test", "*.parquet")
    )
ls = sorted(ls)
print(f"Found {len(ls)} test batch parquet files")

TOPK = 800
_pred_by_event = {}

_TWO_PI = np.float32(2.0 * np.pi)
_PI = np.float32(np.pi)


def _cart_to_sphere_adjust_np(vx, vy, vz):
    vx = vx.astype(np.float32, copy=False)
    vy = vy.astype(np.float32, copy=False)
    vz = vz.astype(np.float32, copy=False)

    r = np.sqrt(vx * vx + vy * vy + vz * vz, dtype=np.float32)
    zenith = np.zeros_like(r, dtype=np.float32)
    nz = r > 0
    cz = np.empty_like(r, dtype=np.float32)
    cz[nz] = vz[nz] / r[nz]
    np.clip(cz, -1.0, 1.0, out=cz)
    zenith[nz] = np.arccos(cz[nz]).astype(np.float32, copy=False)

    azimuth = np.arctan2(vy, vx).astype(np.float32, copy=False)
    azimuth = np.where(azimuth < 0, azimuth + _TWO_PI, azimuth).astype(
        np.float32, copy=False
    )

    negz = zenith < 0
    if np.any(negz):
        zenith = np.where(negz, zenith + _PI, zenith).astype(np.float32, copy=False)
        azimuth = np.where(negz, azimuth + _PI, azimuth).astype(np.float32, copy=False)

    negaz = azimuth < 0
    if np.any(negaz):
        azimuth = np.where(negaz, azimuth + _TWO_PI, azimuth).astype(
            np.float32, copy=False
        )

    azimuth = np.mod(azimuth, _TWO_PI).astype(np.float32, copy=False)

    zenith = np.clip(zenith, 0.0, _PI).astype(np.float32, copy=False)
    return azimuth, zenith




## === cell 5
for batch_file in ls:
    t0b = time.time()
    base = os.path.basename(batch_file)
    try:
        batch_id = int(base.replace("batch_", "").replace(".parquet", ""))
    except Exception:
        batch_id = None

    print(f"processing: {batch_file}")

    df = pd.read_parquet(
        batch_file,
        columns=["time", "sensor_id", "charge", "auxiliary"],
        engine="pyarrow",
    )

    if df.index.name == "event_id":
        event = df.index.to_numpy(copy=False)
    elif "event_id" in df.columns:
        event = df["event_id"].to_numpy(copy=False)
    else:
        df = df.reset_index().rename(columns={"index": "event_id"})
        event = df["event_id"].to_numpy(copy=False)

    sensor = df["sensor_id"].to_numpy(copy=False)
    df_time = df["time"].to_numpy(copy=False)
    df_aux = df["auxiliary"].to_numpy(copy=False)
    df_charge = df["charge"].to_numpy(copy=False)

    xyz_all = _geom_xyz_lut[sensor.astype(np.int64, copy=False)]
    geom_valid = ~np.isnan(xyz_all[:, 0])
    time_isnan = np.isnan(df_time)
    base_mask = (~df_aux) & geom_valid & (~time_isnan)

    if not np.any(base_mask):
        del (
            df,
            event,
            sensor,
            df_time,
            df_aux,
            df_charge,
            xyz_all,
            geom_valid,
            time_isnan,
            base_mask,
        )
        gc.collect()
        print(f"done batch {batch_id} in {time.time() - t0b:.2f}s (no valid pulses)")
        continue

    idx0 = np.flatnonzero(base_mask)
    ev0 = event[idx0].astype(np.int64, copy=False)
    inv0, uniq_ev = pd.factorize(ev0, sort=False)
    inv0 = inv0.astype(np.int64, copy=False)
    n_events = int(uniq_ev.size)

    ch0 = df_charge[idx0]
    if idx0.size > 0:
        order = np.lexsort(((-ch0).astype(np.float32, copy=False), inv0))
        inv_s = inv0[order]
        idx_s = idx0[order]

        starts = np.empty(inv_s.size, dtype=bool)
        starts[0] = True
        starts[1:] = inv_s[1:] != inv_s[:-1]
        grp_start_pos = np.flatnonzero(starts)
        grp_start_for_pos = np.empty(inv_s.size, dtype=np.int64)
        grp_start_for_pos[grp_start_pos] = grp_start_pos
        np.maximum.accumulate(grp_start_for_pos, out=grp_start_for_pos)
        rank_in_group = np.arange(inv_s.size, dtype=np.int64) - grp_start_for_pos

        keep = rank_in_group < TOPK
        idx_sel = idx_s[keep]
        inv_sel = inv_s[keep]

        t_sel = df_time[idx_sel].astype(np.float32, copy=False)
        xyz_sel = xyz_all[idx_sel].astype(np.float32, copy=False)
    else:
        idx_sel = idx0
        inv_sel = inv0
        t_sel = df_time[idx0].astype(np.float32, copy=False)
        xyz_sel = xyz_all[idx0].astype(np.float32, copy=False)

    cnt = np.bincount(inv_sel, minlength=n_events).astype(np.float32, copy=False)
    valid_ev_mask = cnt > 0

    if not np.any(valid_ev_mask):
        del (
            df,
            event,
            sensor,
            df_time,
            df_aux,
            df_charge,
            xyz_all,
            geom_valid,
            time_isnan,
            base_mask,
            idx0,
            ev0,
            inv0,
            uniq_ev,
            ch0,
            order,
            inv_s,
            idx_s,
            starts,
            grp_start_pos,
            grp_start_for_pos,
            rank_in_group,
            keep,
            idx_sel,
            inv_sel,
            t_sel,
            xyz_sel,
            cnt,
            valid_ev_mask,
        )
        gc.collect()
        print(
            f"done batch {batch_id} in {time.time() - t0b:.2f}s (no events after TOPK)"
        )
        continue

    sum_t = np.bincount(inv_sel, weights=t_sel, minlength=n_events).astype(
        np.float32, copy=False
    )
    sum_t2 = np.bincount(inv_sel, weights=t_sel * t_sel, minlength=n_events).astype(
        np.float32, copy=False
    )

    x = xyz_sel[:, 0]
    y = xyz_sel[:, 1]
    z = xyz_sel[:, 2]

    sum_x = np.bincount(inv_sel, weights=x, minlength=n_events).astype(
        np.float32, copy=False
    )
    sum_y = np.bincount(inv_sel, weights=y, minlength=n_events).astype(
        np.float32, copy=False
    )
    sum_z = np.bincount(inv_sel, weights=z, minlength=n_events).astype(
        np.float32, copy=False
    )

    sum_xt = np.bincount(inv_sel, weights=x * t_sel, minlength=n_events).astype(
        np.float32, copy=False
    )
    sum_yt = np.bincount(inv_sel, weights=y * t_sel, minlength=n_events).astype(
        np.float32, copy=False
    )
    sum_zt = np.bincount(inv_sel, weights=z * t_sel, minlength=n_events).astype(
        np.float32, copy=False
    )

    inv_cnt = np.zeros_like(cnt, dtype=np.float32)
    inv_cnt[valid_ev_mask] = 1.0 / cnt[valid_ev_mask]

    mean_t = sum_t * inv_cnt
    mean_t2 = sum_t2 * inv_cnt

    mean_x = sum_x * inv_cnt
    mean_y = sum_y * inv_cnt
    mean_z = sum_z * inv_cnt

    mean_xt = sum_xt * inv_cnt
    mean_yt = sum_yt * inv_cnt
    mean_zt = sum_zt * inv_cnt

    q = mean_t2 - mean_t * mean_t
    q_zero = np.abs(q) <= 1e-8
    denom = np.where(q_zero, 1.0, q).astype(np.float32, copy=False)

    vx = (mean_xt - mean_x * mean_t) / denom
    vy = (mean_yt - mean_y * mean_t) / denom
    vz = (mean_zt - mean_z * mean_t) / denom

    vx[q_zero] = 0.0
    vy[q_zero] = 0.0
    vz[q_zero] = 0.0

    vx = -vx
    vy = -vy
    vz = -vz

    valid_idx = np.flatnonzero(valid_ev_mask)
    az_v, ze_v = _cart_to_sphere_adjust_np(vx[valid_idx], vy[valid_idx], vz[valid_idx])
    ev_out = uniq_ev[valid_idx].astype(np.int64, copy=False)

    for e, a, z_ in zip(ev_out, az_v, ze_v):
        _pred_by_event[int(e)] = (np.float32(a), np.float32(z_))

    del (
        df,
        event,
        sensor,
        df_time,
        df_aux,
        df_charge,
        xyz_all,
        geom_valid,
        time_isnan,
        base_mask,
        idx0,
        ev0,
        inv0,
        uniq_ev,
        ch0,
        order,
        inv_s,
        idx_s,
        starts,
        grp_start_pos,
        grp_start_for_pos,
        rank_in_group,
        keep,
        idx_sel,
        inv_sel,
        t_sel,
        xyz_sel,
        cnt,
        valid_ev_mask,
        sum_t,
        sum_t2,
        sum_x,
        sum_y,
        sum_z,
        sum_xt,
        sum_yt,
        sum_zt,
        inv_cnt,
        mean_t,
        mean_t2,
        mean_x,
        mean_y,
        mean_z,
        mean_xt,
        mean_yt,
        mean_zt,
        q,
        q_zero,
        denom,
        vx,
        vy,
        vz,
        x,
        y,
        z,
        valid_idx,
        az_v,
        ze_v,
        ev_out,
    )
    gc.collect()
    print(f"done batch {batch_id} in {time.time() - t0b:.2f}s")




## === cell 6
event_ids = _event_ids_all  # already int64
n = event_ids.shape[0]

az = np.zeros(n, dtype=np.float32)
ze = np.zeros(n, dtype=np.float32)

pred_keys = np.fromiter(
    _pred_by_event.keys(), dtype=np.int64, count=len(_pred_by_event)
)
pred_az = np.fromiter(
    (v[0] for v in _pred_by_event.values()), dtype=np.float32, count=len(_pred_by_event)
)
pred_ze = np.fromiter(
    (v[1] for v in _pred_by_event.values()), dtype=np.float32, count=len(_pred_by_event)
)

if pred_keys.size:
    sort_idx = np.argsort(pred_keys)
    k_sorted = pred_keys[sort_idx]
    az_sorted = pred_az[sort_idx]
    ze_sorted = pred_ze[sort_idx]

    pos = np.searchsorted(k_sorted, event_ids)
    in_range = pos < k_sorted.size
    match = np.zeros_like(in_range, dtype=bool)
    match[in_range] = k_sorted[pos[in_range]] == event_ids[in_range]

    az[match] = az_sorted[pos[match]]
    ze[match] = ze_sorted[pos[match]]

out_path = "submission.csv"

with open(out_path, "w") as f:
    f.write("event_id,azimuth,zenith\n")
    chunk = 1_000_000
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        block = np.column_stack((event_ids[i:j], az[i:j], ze[i:j]))
        np.savetxt(
            f,
            block,
            fmt=["%d", "%.9g", "%.9g"],
            delimiter=",",
        )

print("Wrote submission.csv with rows:", n)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

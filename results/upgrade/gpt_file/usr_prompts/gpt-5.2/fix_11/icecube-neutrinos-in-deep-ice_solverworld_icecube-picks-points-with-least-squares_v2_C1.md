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

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)

pd.options.mode.chained_assignment = None



## === cell 1
geometry = pd.read_csv(
    os.path.join(DATA_DIR, "sensor_geometry.csv"), usecols=["sensor_id", "x", "y", "z"]
).astype({"sensor_id": np.int32, "x": np.float32, "y": np.float32, "z": np.float32})
geometry.set_index("sensor_id", inplace=True)

max_sid = int(geometry.index.max())
geom_x = np.empty(max_sid + 1, dtype=np.float32)
geom_y = np.empty(max_sid + 1, dtype=np.float32)
geom_z = np.empty(max_sid + 1, dtype=np.float32)
geom_x[:] = np.nan
geom_y[:] = np.nan
geom_z[:] = np.nan
idxs = geometry.index.to_numpy()
geom_x[idxs] = geometry["x"].to_numpy()
geom_y[idxs] = geometry["y"].to_numpy()
geom_z[idxs] = geometry["z"].to_numpy()

geometry.info()



## === cell 2
DELTA_T = 200
DELTA_POS = 55
WTS = (0.1, 0.9, 0)  # weights for weight formula




## === cell 3
def nearby(times, sensors, x, y, z, delta_t=300, delta_dist=55):
    n = times.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int32)

    order = np.argsort(times, kind="mergesort")
    t = times[order]
    s = sensors[order]
    xx = x[order]
    yy = y[order]
    zz = z[order]

    counts_sorted = np.zeros(n, dtype=np.int32)

    dd = float(delta_dist)
    dt = float(delta_t)

    dx = np.empty(n, dtype=np.float32)
    dy = np.empty(n, dtype=np.float32)
    dz = np.empty(n, dtype=np.float32)
    ds = np.empty(n, dtype=bool)

    left = 0
    right = 0

    for i in range(n):
        ti = t[i]

        while left < n and t[left] <= ti - dt:
            left += 1
        if right < i + 1:
            right = i + 1
        while right < n and t[right] < ti + dt:
            right += 1

        wL = left
        wR = right
        wlen = wR - wL
        if wlen <= 1:
            continue

        wi = s[i]

        np.not_equal(s[wL:wR], wi, out=ds[:wlen])
        if not ds[:wlen].any():
            continue

        np.subtract(xx[wL:wR], xx[i], out=dx[:wlen])
        np.subtract(yy[wL:wR], yy[i], out=dy[:wlen])
        np.subtract(zz[wL:wR], zz[i], out=dz[:wlen])

        close = (
            (np.abs(dx[:wlen]) < dd)
            & (np.abs(dy[:wlen]) < dd)
            & (np.abs(dz[:wlen]) < dd)
            & ds[:wlen]
        )
        counts_sorted[i] = int(np.count_nonzero(close))

    counts = np.empty(n, dtype=np.int32)
    counts[order] = counts_sorted
    return counts


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
    x2y2 = x * x + y * y
    r2 = x2y2 + z * z

    if (not math.isfinite(r2)) or r2 <= 1e-24:
        return 0.0, 0.0

    r = math.sqrt(r2)

    if x2y2 < 1e-12:
        azimuth = 0.0
    else:
        denom_xy = math.sqrt(x2y2)
        c = x / denom_xy
        if c > 1.0:
            c = 1.0
        elif c < -1.0:
            c = -1.0
        azimuth = math.acos(c) * float(np.sign(y))

    cz = z / r
    if cz > 1.0:
        cz = 1.0
    elif cz < -1.0:
        cz = -1.0
    zenith = math.acos(cz)

    azimuth, zenith = adjust_polar(azimuth, zenith)
    return azimuth, zenith




## === cell 4
def time_window_sorted(times_sorted, width=4000):
    """find min and max time that gives most points in fixed size window
    Assumes times_sorted is sorted ascending (as in batch parquet within event).
    """
    n = times_sorted.shape[0]
    if n == 0:
        return 0.0, 0.0

    mn = float(times_sorted[0])
    mx = float(times_sorted[-1])
    if mx - mn < width:
        return mn, mx

    w = float(width)
    best_count = 0
    best_l = 0

    r = 0
    for l in range(n):
        if r < l:
            r = l
        limit = times_sorted[l] + w
        while r < n and times_sorted[r] < limit:
            r += 1
        c = r - l
        if c > best_count:
            best_count = c
            best_l = l

    if best_count < 3:
        return mn, mx

    half = w / 2.0
    best_center = float(times_sorted[best_l]) + half
    m1 = max(best_center - half, mn)
    m2 = min(best_center + half, mx)
    return m1, m2




## === cell 5
def compute_direction(r, t, use_weights=None):
    """compute r vector from time,x,y,z columns: x,y,z,time,charge

    Bugfix: ensure weights broadcast correctly with r (n,3) and t (n,1) by enforcing (n,1).
    Robustness: if time variance is degenerate, fall back to a safe non-NaN direction.
    """
    assert r.shape[1] == 3
    assert t.shape[1] == 1

    if use_weights is None:
        weight = np.ones_like(t, dtype=np.float32)
    else:
        weight = use_weights
        if weight.ndim == 1:
            weight = weight.reshape(-1, 1)
        weight = weight.astype(np.float32, copy=False)

    sw = float(np.sum(weight))
    if not np.isfinite(sw) or sw <= 0:
        weight = np.ones_like(t, dtype=np.float32)
        sw = float(weight.shape[0])

    w = weight / sw  # shape (n,1)

    avg_t = float(np.sum(t * w))
    avg_t2 = float(np.sum((t * t) * w))
    avg_r = np.sum(r * w, axis=0)
    avg_rt = np.sum((r * t) * w, axis=0)

    q = avg_t2 - avg_t * avg_t
    if (not np.isfinite(q)) or abs(q) <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32), avg_r.astype(
            np.float32, copy=False
        )

    v_est = (avg_rt - avg_r * avg_t) / q
    r_est = avg_r - v_est * avg_t
    return v_est.astype(np.float32, copy=False), r_est.astype(np.float32, copy=False)




## === cell 6
ssub = pd.read_csv(
    os.path.join(PATH_DATASET, "sample_submission.csv"),
    usecols=["event_id"],
    dtype={"event_id": np.int64},
)
event_ids = ssub["event_id"].to_numpy(np.int64, copy=False)

out_az = np.zeros(event_ids.shape[0], dtype=np.float32)
out_ze = np.zeros(event_ids.shape[0], dtype=np.float32)

min_eid = int(event_ids.min())
max_eid = int(event_ids.max())
eid_to_row_arr = np.full(max_eid - min_eid + 1, -1, dtype=np.int32)
eid_to_row_arr[event_ids - min_eid] = np.arange(event_ids.shape[0], dtype=np.int32)

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
needed_cols = ["time", "sensor_id", "auxiliary", "charge"]

t_start = time.time()
events_done = 0
batches_done = 0

xyz_buf = np.empty((500, 3), dtype=np.float32)

_NEARBY_SUM_THRESHOLD = 5

for batch_file in ls:
    batches_done += 1
    print(f"processing: {batch_file}")
    df_big = pd.read_parquet(batch_file, columns=needed_cols)

    if "event_id" not in df_big.columns:
        if df_big.index.name == "event_id":
            ev = df_big.index.to_numpy(copy=False)
        else:
            df_big = df_big.reset_index().rename(columns={"index": "event_id"})
            ev = df_big["event_id"].to_numpy(copy=False)
    else:
        ev = df_big["event_id"].to_numpy(copy=False)

    t_all = df_big["time"].to_numpy(copy=False)
    sid_all = df_big["sensor_id"].to_numpy(copy=False).astype(np.int32, copy=False)
    aux_all = df_big["auxiliary"].to_numpy(copy=False)
    chg_all = df_big["charge"].to_numpy(copy=False)

    x_all = geom_x[sid_all]
    y_all = geom_y[sid_all]
    z_all = geom_z[sid_all]

    boundaries = np.flatnonzero(ev[1:] != ev[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, ev.shape[0]]

    for s, e in zip(starts, ends):
        eid = int(ev[s])

        tt = t_all[s:e]
        ss = sid_all[s:e]
        aa = aux_all[s:e]
        cc = chg_all[s:e]
        xx = x_all[s:e]
        yy = y_all[s:e]
        zz = z_all[s:e]

        pri_mask = ~aa

        if np.any(pri_mask):
            tw_source = tt[pri_mask]
        else:
            tw_source = tt
        t0, t1 = time_window_sorted(tw_source, width=4000)

        in_tw = (tt >= t0) & (tt <= t1)
        if not np.all(in_tw):
            tt = tt[in_tw]
            ss = ss[in_tw]
            aa = aa[in_tw]
            cc = cc[in_tw]
            xx = xx[in_tw]
            yy = yy[in_tw]
            zz = zz[in_tw]
            pri_mask = ~aa  # update after filtering

        if tt.shape[0] > 500:
            ord2 = np.lexsort((-cc, aa))
            ord2 = ord2[:500]
            tt = tt[ord2]
            ss = ss[ord2]
            aa = aa[ord2]
            cc = cc[ord2]
            xx = xx[ord2]
            yy = yy[ord2]
            zz = zz[ord2]
            pri_mask = ~aa

        n_evt = tt.shape[0]
        if n_evt * (n_evt - 1) < _NEARBY_SUM_THRESHOLD:
            ccc_sum = 0
            ccc = None
        else:
            ccc = nearby(tt, ss, xx, yy, zz, delta_t=DELTA_T, delta_dist=DELTA_POS)
            ccc_sum = int(np.sum(ccc))

        if ccc_sum < 5:
            weight = None
            use_mask = pri_mask
        else:
            weight = (
                WTS[0] * pri_mask.astype(np.float32, copy=False)
                + WTS[1] * (ccc > 0).astype(np.float32, copy=False)
                + WTS[2] * (ccc > 1).astype(np.float32, copy=False)
            )
            use_mask = slice(None)

        if isinstance(use_mask, slice):
            ux, uy, uz = xx, yy, zz
            ut = tt
            uw = weight
        else:
            ux, uy, uz = xx[use_mask], yy[use_mask], zz[use_mask]
            ut = tt[use_mask]
            uw = None if weight is None else weight[use_mask]

        m = ut.shape[0]
        if m <= 500:
            xyz = xyz_buf[:m]
        else:
            xyz = np.empty((m, 3), dtype=np.float32)
        xyz[:, 0] = ux
        xyz[:, 1] = uy
        xyz[:, 2] = uz

        ti = ut.reshape(-1, 1)

        v_est, r_est = compute_direction(xyz, ti, use_weights=uw)
        azimuth, zenith = v_to_polar(float(v_est[0]), float(v_est[1]), float(v_est[2]))

        idx = eid - min_eid
        if 0 <= idx < eid_to_row_arr.shape[0]:
            row = int(eid_to_row_arr[idx])
            if row != -1:
                out_az[row] = np.float32(azimuth)
                out_ze[row] = np.float32(zenith)

        events_done += 1
        if (events_done % 200000) == 0:
            dt = time.time() - t_start
            print(f"  processed events: {events_done:,}  elapsed: {dt:.1f}s")

    del (
        df_big,
        ev,
        t_all,
        sid_all,
        aux_all,
        chg_all,
        x_all,
        y_all,
        z_all,
        boundaries,
        starts,
        ends,
    )
    if (batches_done % 8) == 0:
        gc.collect()

print(f"Done. Total events processed: {events_done:,} in {time.time() - t_start:.1f}s")



## === cell 9
sub = pd.DataFrame({"event_id": event_ids, "azimuth": out_az, "zenith": out_ze})
sub = sub[["event_id", "azimuth", "zenith"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv, rows:", len(sub))

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())

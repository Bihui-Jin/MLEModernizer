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

# 5. Code solution

## === cell 0
import os, glob, gc, time, math
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/icecube-neutrinos-in-deep-ice"

df_test = pd.read_parquet(
    os.path.join(INPUT_DIR, "test_meta.parquet"),
    columns=["event_id"],
    engine="pyarrow",
)
sub_event_ids = df_test["event_id"].to_numpy(np.int64, copy=True)
n_sub = sub_event_ids.shape[0]
sub_az = np.zeros(n_sub, dtype=np.float32)
sub_ze = np.zeros(n_sub, dtype=np.float32)

sub_event_ids_sorted = np.sort(sub_event_ids)
sub_sorted_to_pos = np.empty_like(sub_event_ids_sorted, dtype=np.int64)
orig_order = np.argsort(sub_event_ids, kind="mergesort")
sub_sorted_to_pos[:] = orig_order.astype(np.int64, copy=False)
del orig_order

df_sensor = pd.read_csv(os.path.join(INPUT_DIR, "sensor_geometry.csv"))
if "sensor_id" not in df_sensor.columns:
    raise ValueError("sensor_geometry.csv must contain 'sensor_id' column.")
df_sensor = df_sensor.set_index("sensor_id")[["x", "y", "z"]].astype(np.float32)

del df_test
gc.collect()




## === cell 1
def cartesian_to_polar(x: float, y: float, z: float):
    r = math.sqrt(x * x + y * y + z * z) + 1e-12
    azimuth = math.atan2(y, x)  # [-pi, pi]
    zenith = math.acos(max(-1.0, min(1.0, z / r)))  # [0, pi]
    return azimuth, zenith


def adjust_polar(azimuth: float, zenith: float):
    azimuth = azimuth % (2.0 * math.pi)
    if zenith < 0:
        zenith = 0.0
    if zenith > math.pi:
        zenith = math.pi
    return azimuth, zenith


def polar_to_cartesian(azimuth: float, zenith: float):
    x = math.cos(azimuth) * math.sin(zenith)
    y = math.sin(azimuth) * math.sin(zenith)
    z = math.cos(zenith)
    return x, y, z




## === cell 2
def _principal_eigenvector_sym_3x3(xx, yy, zz, xy, xz, yz):
    m00 = float(xx)
    m11 = float(yy)
    m22 = float(zz)
    m01 = float(xy)
    m02 = float(xz)
    m12 = float(yz)

    trace = m00 + m11 + m22
    q = trace / 3.0

    a00 = m00 - q
    a11 = m11 - q
    a22 = m22 - q
    p2 = a00 * a00 + a11 * a11 + a22 * a22 + 2.0 * (m01 * m01 + m02 * m02 + m12 * m12)
    p = math.sqrt(max(p2 / 6.0, 0.0))

    if p < 1e-18:
        return 1.0, 0.0, 0.0

    invp = 1.0 / p
    b00 = a00 * invp
    b11 = a11 * invp
    b22 = a22 * invp
    b01 = m01 * invp
    b02 = m02 * invp
    b12 = m12 * invp

    detB = (
        b00 * (b11 * b22 - b12 * b12)
        - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02)
    )
    r = detB / 2.0
    if r <= -1.0:
        phi = math.pi / 3.0
    elif r >= 1.0:
        phi = 0.0
    else:
        phi = math.acos(r) / 3.0

    eig1 = q + 2.0 * p * math.cos(phi)  # largest

    r0x, r0y, r0z = (m00 - eig1), m01, m02
    r1x, r1y, r1z = m01, (m11 - eig1), m12
    r2x, r2y, r2z = m02, m12, (m22 - eig1)

    c01x = r0y * r1z - r0z * r1y
    c01y = r0z * r1x - r0x * r1z
    c01z = r0x * r1y - r0y * r1x
    n01 = c01x * c01x + c01y * c01y + c01z * c01z

    c02x = r0y * r2z - r0z * r2y
    c02y = r0z * r2x - r0x * r2z
    c02z = r0x * r2y - r0y * r2x
    n02 = c02x * c02x + c02y * c02y + c02z * c02z

    c12x = r1y * r2z - r1z * r2y
    c12y = r1z * r2x - r1x * r2z
    c12z = r1x * r2y - r1y * r2x
    n12 = c12x * c12x + c12y * c12y + c12z * c12z

    if n01 >= n02 and n01 >= n12 and n01 > 1e-30:
        vx, vy, vz = c01x, c01y, c01z
    elif n02 >= n12 and n02 > 1e-30:
        vx, vy, vz = c02x, c02y, c02z
    elif n12 > 1e-30:
        vx, vy, vz = c12x, c12y, c12z
    else:
        if (r0x * r0x + r0y * r0y + r0z * r0z) >= (r1x * r1x + r1y * r1y + r1z * r1z):
            ax, ay, az = r0x, r0y, r0z
        else:
            ax, ay, az = r1x, r1y, r1z
        if abs(ax) < abs(ay):
            bx, by, bz = 1.0, 0.0, 0.0
        else:
            bx, by, bz = 0.0, 1.0, 0.0
        vx = ay * bz - az * by
        vy = az * bx - ax * bz
        vz = ax * by - ay * bx

    norm = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-12
    return vx / norm, vy / norm, vz / norm


def best_fit_line_direction_from_sums(
    n: int,
    sx: float,
    sy: float,
    sz: float,
    sxx: float,
    syy: float,
    szz: float,
    sxy: float,
    sxz: float,
    syz: float,
) -> np.ndarray:
    if n < 2:
        raise ValueError("Need at least 2 points for line fit.")
    invn = 1.0 / float(n)
    mx = sx * invn
    my = sy * invn
    mz = sz * invn

    xx = sxx - float(n) * mx * mx
    yy = syy - float(n) * my * my
    zz = szz - float(n) * mz * mz
    xy = sxy - float(n) * mx * my
    xz = sxz - float(n) * mx * mz
    yz = syz - float(n) * my * mz

    vx, vy, vz = _principal_eigenvector_sym_3x3(xx, yy, zz, xy, xz, yz)
    return np.array([vx, vy, vz], dtype=np.float64)




## === cell 3
gc.collect()



## === cell 4
test_dir = os.path.join(INPUT_DIR, "test")
batch_files = sorted(glob.glob(os.path.join(test_dir, "batch_*.parquet")))
if len(batch_files) == 0:
    raise FileNotFoundError(f"No test batch parquet files found in: {test_dir}")

sensor_idx = df_sensor.index.to_numpy(np.int64, copy=False)
max_sid = int(sensor_idx.max())
x_lut = np.full(max_sid + 1, np.nan, dtype=np.float32)
y_lut = np.full(max_sid + 1, np.nan, dtype=np.float32)
z_lut = np.full(max_sid + 1, np.nan, dtype=np.float32)
x_lut[sensor_idx] = df_sensor["x"].to_numpy(np.float32, copy=False)
y_lut[sensor_idx] = df_sensor["y"].to_numpy(np.float32, copy=False)
z_lut[sensor_idx] = df_sensor["z"].to_numpy(np.float32, copy=False)

_two_pi = 2.0 * math.pi
_pi = math.pi

import pyarrow.parquet as pq


def _event_pos(eid: int) -> int:
    j = np.searchsorted(sub_event_ids_sorted, eid)
    if j < n_sub and int(sub_event_ids_sorted[j]) == eid:
        return int(sub_sorted_to_pos[j])
    return -1


def _sums_xyz(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    sx = float(xs.sum())
    sy = float(ys.sum())
    sz = float(zs.sum())
    sxx = float(np.einsum("i,i->", xs, xs))
    syy = float(np.einsum("i,i->", ys, ys))
    szz = float(np.einsum("i,i->", zs, zs))
    sxy = float(np.einsum("i,i->", xs, ys))
    sxz = float(np.einsum("i,i->", xs, zs))
    syz = float(np.einsum("i,i->", ys, zs))
    return sx, sy, sz, sxx, syy, szz, sxy, sxz, syz


t0 = time.time()
for batch_file in batch_files:
    table = pq.read_table(
        batch_file,
        columns=["event_id", "sensor_id", "auxiliary", "charge", "time"],
        memory_map=True,
        use_threads=True,
    )
    cols = table.column_names

    if "event_id" not in cols:
        df = table.to_pandas(types_mapper=None)
        if "event_id" not in df.columns:
            df = df.reset_index()
        eids = df["event_id"].to_numpy(np.int64, copy=False)
        sid = df["sensor_id"].to_numpy(np.int64, copy=False)
        charges = (
            df["charge"].to_numpy(np.float32, copy=False)
            if "charge" in df.columns
            else None
        )
        aux = (
            df["auxiliary"].to_numpy(bool, copy=False)
            if "auxiliary" in df.columns
            else None
        )
        times = (
            df["time"].to_numpy(np.int64, copy=False) if "time" in df.columns else None
        )
        del df, table
    else:
        eids = table["event_id"].to_numpy(zero_copy_only=False)
        if eids.dtype != np.int64:
            eids = eids.astype(np.int64, copy=False)
        sid = table["sensor_id"].to_numpy(zero_copy_only=False)
        if sid.dtype != np.int64:
            sid = sid.astype(np.int64, copy=False)

        charges = None
        if "charge" in cols:
            charges = table["charge"].to_numpy(zero_copy_only=False)
            if charges.dtype != np.float32:
                charges = charges.astype(np.float32, copy=False)

        aux = None
        if "auxiliary" in cols:
            aux = table["auxiliary"].to_numpy(zero_copy_only=False)
            if aux.dtype != np.bool_:
                aux = aux.astype(bool, copy=False)

        times = None
        if "time" in cols:
            times = table["time"].to_numpy(zero_copy_only=False)
            if times.dtype != np.int64:
                times = times.astype(np.int64, copy=False)

        del table

    if eids.shape[0] < 2:
        continue

    if aux is not None:
        m = ~aux
        if not m.all():
            eids = eids[m]
            sid = sid[m]
            if charges is not None:
                charges = charges[m]
            if times is not None:
                times = times[m]
        del aux

    if eids.shape[0] < 2:
        continue

    valid_sid = (sid >= 0) & (sid <= max_sid)
    if not valid_sid.all():
        eids = eids[valid_sid]
        sid = sid[valid_sid]
        if charges is not None:
            charges = charges[valid_sid]
        if times is not None:
            times = times[valid_sid]
        if eids.shape[0] < 2:
            continue

    x = x_lut[sid]
    y = y_lut[sid]
    z = z_lut[sid]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not ok.all():
        eids = eids[ok]
        x = x[ok]
        y = y[ok]
        z = z[ok]
        if charges is not None:
            charges = charges[ok]
        if times is not None:
            times = times[ok]
        if eids.shape[0] < 2:
            continue

    if charges is not None:
        order = np.lexsort((-charges, eids))
        c_sorted = charges[order]
    else:
        order = np.argsort(eids, kind="mergesort")
        c_sorted = None

    e_sorted = eids[order]
    x_sorted = x[order].astype(np.float64, copy=False)
    y_sorted = y[order].astype(np.float64, copy=False)
    z_sorted = z[order].astype(np.float64, copy=False)
    t_sorted = times[order] if times is not None else None

    n_rows = e_sorted.shape[0]
    if n_rows < 2:
        continue

    boundaries = np.flatnonzero(e_sorted[1:] != e_sorted[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n_rows]))
    counts = ends - starts
    unique_ids = e_sorted[starts]
    n_events = unique_ids.shape[0]

    for i in range(n_events):
        cnt = int(counts[i])
        if cnt < 2:
            continue

        eid = int(unique_ids[i])
        pos = _event_pos(eid)
        if pos < 0:
            continue

        st = int(starts[i])
        en = int(ends[i])

        if cnt > 10000 and c_sorted is not None:
            en_eff = st + 10000
            n = 10000
        else:
            en_eff = en
            n = cnt

        xs = x_sorted[st:en_eff]
        ys = y_sorted[st:en_eff]
        zs = z_sorted[st:en_eff]

        sx, sy, sz, sxx, syy, szz, sxy, sxz, syz = _sums_xyz(xs, ys, zs)

        direction = best_fit_line_direction_from_sums(
            n, sx, sy, sz, sxx, syy, szz, sxy, sxz, syz
        )

        vx, vy, vz = float(direction[0]), float(direction[1]), float(direction[2])
        if t_sorted is not None:
            ts = t_sorted[st:en_eff].astype(np.float64, copy=False)
            t_mean = float(ts.mean())
            mx = sx / float(n)
            my = sy / float(n)
            mz = sz / float(n)

            dt = ts - t_mean
            proj = (xs - mx) * vx + (ys - my) * vy + (zs - mz) * vz

            if c_sorted is not None:
                ws = c_sorted[st:en_eff].astype(np.float64, copy=False)
                wsum = float(ws.sum()) + 1e-12
                cov = float(np.einsum("i,i->", ws, proj * dt) / wsum)
            else:
                cov = float(np.einsum("i,i->", proj, dt) / float(n))

            if cov < 0.0:
                vx, vy, vz = -vx, -vy, -vz

        r = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-12
        az = math.atan2(vy, vx) % _two_pi
        ze = math.acos(max(-1.0, min(1.0, vz / r)))
        if ze < 0.0:
            ze = 0.0
        elif ze > _pi:
            ze = _pi

        sub_az[pos] = np.float32(az)
        sub_ze[pos] = np.float32(ze)

    del (
        eids,
        sid,
        charges,
        times,
        x,
        y,
        z,
        valid_sid,
        ok,
        order,
        e_sorted,
        x_sorted,
        y_sorted,
        z_sorted,
        t_sorted,
        c_sorted,
        unique_ids,
        starts,
        ends,
        counts,
        boundaries,
    )

gc.collect()

print(f"Done inference in {time.time() - t0:.1f}s across {len(batch_files)} batches.")



## === cell 5
az = (sub_az.astype(np.float64) % (2.0 * np.pi)).astype(np.float32)
ze = np.clip(sub_ze.astype(np.float64), 0.0, np.pi).astype(np.float32)

az = np.nan_to_num(az, nan=0.0).astype(np.float32)
ze = np.nan_to_num(ze, nan=0.0).astype(np.float32)

sub_df = pd.DataFrame({"event_id": sub_event_ids, "azimuth": az, "zenith": ze})

submission_path = "submission.csv"
sub_df.to_csv(submission_path, index=False)

print(
    f"Wrote {submission_path} with shape={sub_df.shape} and columns={['event_id','azimuth','zenith']}"
)
sub_df.head()

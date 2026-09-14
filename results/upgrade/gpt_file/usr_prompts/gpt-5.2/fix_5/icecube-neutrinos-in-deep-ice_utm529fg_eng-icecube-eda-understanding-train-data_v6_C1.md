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
import numpy as np
import pandas as pd
import os, glob, gc, time, math

INPUT_DIR = "/kaggle/input/icecube-neutrinos-in-deep-ice"

df_test = pd.read_parquet(os.path.join(INPUT_DIR, "test_meta.parquet"))

df_sensor = pd.read_csv(os.path.join(INPUT_DIR, "sensor_geometry.csv"))
if "sensor_id" not in df_sensor.columns:
    raise ValueError("sensor_geometry.csv must contain 'sensor_id' column.")
df_sensor = df_sensor.set_index("sensor_id")[["x", "y", "z"]].astype(np.float32)

sample_csv_path = os.path.join(INPUT_DIR, "sample_submission.csv")
df_sample_submission = pd.read_csv(sample_csv_path, usecols=["event_id"])
if "event_id" not in df_sample_submission.columns:
    raise ValueError("sample_submission.csv must contain 'event_id' column.")

sub_event_ids = df_sample_submission["event_id"].to_numpy(np.int64, copy=True)
n_sub = sub_event_ids.shape[0]
sub_az = np.zeros(n_sub, dtype=np.float32)
sub_ze = np.zeros(n_sub, dtype=np.float32)

sub_index = pd.Index(sub_event_ids)

del df_sample_submission
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

    xtx = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float64)

    w, v = np.linalg.eigh(xtx)  # ascending eigenvalues
    direction = v[:, -1]
    norm = np.linalg.norm(direction) + 1e-12
    return (direction / norm).astype(np.float64)




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

t0 = time.time()
for batch_file in batch_files:
    df = pd.read_parquet(
        batch_file,
        columns=["event_id", "sensor_id", "auxiliary", "charge", "time"],
    )

    if "event_id" not in df.columns:
        df = df.reset_index()

    eids = df["event_id"].to_numpy(np.int64, copy=False)
    sid = df["sensor_id"].to_numpy(np.int64, copy=False)

    if "auxiliary" in df.columns:
        aux = df["auxiliary"].to_numpy(bool, copy=False)
        m = ~aux
        if not m.all():
            eids = eids[m]
            sid = sid[m]
            if "charge" in df.columns:
                charges = df["charge"].to_numpy(np.float32, copy=False)[m]
            else:
                charges = None
    else:
        charges = (
            df["charge"].to_numpy(np.float32, copy=False)
            if "charge" in df.columns
            else None
        )

    del df  # free early
    if eids.shape[0] < 2:
        gc.collect()
        continue

    valid_sid = (sid >= 0) & (sid <= max_sid)
    if not valid_sid.all():
        eids = eids[valid_sid]
        sid = sid[valid_sid]
        if charges is not None:
            charges = charges[valid_sid]
        if eids.shape[0] < 2:
            del eids, sid, charges, valid_sid
            gc.collect()
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
        if eids.shape[0] < 2:
            del eids, sid, charges, x, y, z, ok, valid_sid
            gc.collect()
            continue

    order = np.argsort(eids, kind="mergesort")
    e_sorted = eids[order]
    x_sorted = x[order]
    y_sorted = y[order]
    z_sorted = z[order]
    c_sorted = charges[order] if charges is not None else None

    unique_ids, start_idx, counts = np.unique(
        e_sorted, return_index=True, return_counts=True
    )
    pos_arr = sub_index.get_indexer(unique_ids)  # -1 if missing

    n_events = unique_ids.shape[0]
    dir_xyz = np.full((n_events, 3), np.nan, dtype=np.float64)

    for i in range(n_events):
        cnt = int(counts[i])
        if cnt < 2:
            continue
        pos = int(pos_arr[i])
        if pos < 0:
            continue

        st = int(start_idx[i])
        en = st + cnt

        xs = x_sorted[st:en]
        ys = y_sorted[st:en]
        zs = z_sorted[st:en]

        if cnt > 10000 and c_sorted is not None:
            cs = c_sorted[st:en]
            k = 10000
            idx = np.argpartition(cs, cnt - k)[cnt - k :]
            xs = xs[idx]
            ys = ys[idx]
            zs = zs[idx]
            n = int(xs.shape[0])
        else:
            n = cnt

        try:
            xs64 = xs.astype(np.float64, copy=False)
            ys64 = ys.astype(np.float64, copy=False)
            zs64 = zs.astype(np.float64, copy=False)

            sx = float(xs64.sum())
            sy = float(ys64.sum())
            sz = float(zs64.sum())
            sxx = float((xs64 * xs64).sum())
            syy = float((ys64 * ys64).sum())
            szz = float((zs64 * zs64).sum())
            sxy = float((xs64 * ys64).sum())
            sxz = float((xs64 * zs64).sum())
            syz = float((ys64 * zs64).sum())

            direction = best_fit_line_direction_from_sums(
                n, sx, sy, sz, sxx, syy, szz, sxy, sxz, syz
            )
            dir_xyz[i, :] = -direction
        except Exception:
            continue

    vx = dir_xyz[:, 0]
    vy = dir_xyz[:, 1]
    vz = dir_xyz[:, 2]
    good = (
        np.isfinite(vx)
        & np.isfinite(vy)
        & np.isfinite(vz)
        & (pos_arr >= 0)
        & (counts >= 2)
    )
    if np.any(good):
        vxg = vx[good]
        vyg = vy[good]
        vzg = vz[good]
        r = np.sqrt(vxg * vxg + vyg * vyg + vzg * vzg) + 1e-12
        az = np.arctan2(vyg, vxg)
        cz = np.clip(vzg / r, -1.0, 1.0)
        ze = np.arccos(cz)

        az = np.mod(az, _two_pi)
        ze = np.clip(ze, 0.0, _pi)

        out_pos = pos_arr[good].astype(np.int64, copy=False)
        sub_az[out_pos] = az.astype(np.float32, copy=False)
        sub_ze[out_pos] = ze.astype(np.float32, copy=False)

    del (
        eids,
        sid,
        charges,
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
        c_sorted,
        unique_ids,
        start_idx,
        counts,
        pos_arr,
        dir_xyz,
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

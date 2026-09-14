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

1.1764789678613623

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
geometry = pd.read_csv(os.path.join(DATA_DIR, "sensor_geometry.csv"))
geometry.set_index("sensor_id", inplace=True)
geometry = geometry.apply(np.float32)

_SENSOR_POS = geometry[["x", "y", "z"]].values  # (N_sensors, 3)
_DELTA_POS = 55.0
_sq_dist = np.sum((_SENSOR_POS[:, None, :] - _SENSOR_POS[None, :, :]) ** 2, axis=2)
_SENSOR_CLOSE = _sq_dist < (_DELTA_POS**2)  # bool matrix, shape (N, N)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3552333582.py in <cell line: 0>()
----> 1 geometry = pd.read_csv(os.path.join(DATA_DIR, "sensor_geometry.csv"))
      2 geometry.set_index("sensor_id", inplace=True)
      3 geometry = geometry.apply(np.float32)
      4 
      5 _SENSOR_POS = geometry[["x", "y", "z"]].values  # (N_sensors, 3)

NameError: name 'pd' is not defined

## === cell 1
def nearby(df, delta_t=300, delta_dist=55):
    """Count nearby sensors in time and space for each pulse.

    Optimized: sorts by time and uses a sliding window so that only pulses
    within `delta_t` are compared, keeping the original counting semantics.
    """
    times = df["time"].values
    sids = df["sensor_id"].values
    idx = geometry.index.get_indexer(sids)

    order = np.argsort(times)
    times_sorted = times[order]
    idx_sorted = idx[order]

    n = len(times_sorted)
    counts = np.zeros(n, dtype=np.int32)

    start = 0
    for i in range(n):
        while times_sorted[i] - times_sorted[start] > delta_t:
            start += 1
        if start < i:
            close_mask = _SENSOR_CLOSE[idx_sorted[start:i], idx_sorted[i]]
            cnt = close_mask.sum()
            counts[order[i]] += cnt
            counts[order[start:i]] += close_mask.astype(np.int32)
    return counts


def adjust_polar(azimuth, zenith):
    """Wrap angles into the proper range."""
    if azimuth < 0:
        azimuth += 2 * math.pi
    if zenith < 0:
        zenith += math.pi
    azimuth = azimuth % (2 * math.pi)
    return azimuth, zenith


def v_to_polar(x, y, z):
    """Convert a direction vector to azimuth and zenith."""
    x, y, z = -x, -y, -z  # arrival reversal
    x2y2 = x**2 + y**2
    r = math.sqrt(x2y2 + z**2)
    if x2y2 < 1e-6:
        x2y2 = 1e-6
    azimuth = math.acos(x / math.sqrt(x2y2)) * np.sign(y)
    zenith = math.acos(z / r)
    return adjust_polar(azimuth, zenith)


def time_window(times, width=4000):
    """Find the time window containing the most points."""
    mn, mx = times.min(), times.max()
    if mx - mn < width:
        return mn, mx
    avg = times.mean()
    best_count, best_center = 0, avg
    for center in np.arange(avg - 1000, avg + 1000, 25):
        c = np.count_nonzero(
            (times > center - width / 2) & (times < center + width / 2)
        )
        if c > best_count:
            best_count, best_center = c, center
    if best_count < 3:
        return mn, mx
    return max(best_center - width / 2, mn), min(best_center + width / 2, mx)


def compute_direction(r, t, use_weights=None):
    """Estimate direction (v) and origin (r0) from positions and times."""
    assert r.shape[1] == 3
    assert t.shape[1] == 1
    weight = np.ones_like(t) if use_weights is None else use_weights
    weight = (weight / np.sum(weight)).reshape(-1, 1)

    def avg(p):
        p = p.reshape(-1, 1) if p.ndim == 1 else p
        return np.sum(p * weight, axis=0)

    q = avg(t**2) - avg(t) ** 2
    v_est = (avg(r * t) - avg(r) * avg(t)) / q
    r_est = avg(r) - v_est * avg(t)
    return v_est, r_est




## === cell 2
PARQUET_COLS = ["event_id", "time", "sensor_id", "charge", "auxiliary"]

test_files = glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet"))

for batch_file in test_files:
    df_big = pd.read_parquet(batch_file, columns=PARQUET_COLS)

    coords = geometry.loc[df_big["sensor_id"]].values
    df_big[["x", "y", "z"]] = coords

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

        v_est, _ = compute_direction(xyz, ti, use_weights=weight)
        azimuth, zenith = v_to_polar(*v_est)

        ssub.at[eid, "azimuth"] = azimuth
        ssub.at[eid, "zenith"] = zenith

    del df_big

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3588245182.py in <cell line: 0>()
      1 PARQUET_COLS = ["event_id", "time", "sensor_id", "charge", "auxiliary"]
      2 
----> 3 test_files = glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet"))
      4 
      5 for batch_file in test_files:

NameError: name 'glob' is not defined

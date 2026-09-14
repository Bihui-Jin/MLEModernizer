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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





## === cell 1
def compute_angle(x, y, z):
    zc = z - z.mean()
    xc = x - x.mean()
    yc = y - y.mean()

    vz = np.dot(zc, zc)
    mx = np.dot(zc, xc) / vz
    my = np.dot(zc, yc) / vz

    zr = 1.0
    xr = mx * zr
    yr = my * zr
    r = np.sqrt(zr * zr + xr * xr + yr * yr)

    azimuth = np.arctan2(yr, xr)
    if azimuth < 0:
        azimuth = 2 * np.pi + azimuth
    zenith = np.arccos(zr / r)
    return azimuth, zenith




## === cell 2
sensor_geometry = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv",
    index_col="sensor_id",
)
geom_xyz = sensor_geometry[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)



## === cell 3
meta_df = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet"
)
batch_id = meta_df.batch_id.unique()[0]

batch_df = meta_df[meta_df.batch_id == batch_id].set_index("event_id", drop=True)
batch_features = pd.read_parquet(
    f"/kaggle/input/icecube-neutrinos-in-deep-ice/train/batch_{batch_id}.parquet"
)

event_id = batch_df.index[6]

event_features = batch_features.iloc[
    batch_df.loc[event_id, "first_pulse_index"] : batch_df.loc[
        event_id, "last_pulse_index"
    ]
    + 1
]
azimuth = batch_df.loc[event_id, "azimuth"]
zenith = batch_df.loc[event_id, "zenith"]

position = geom_xyz[event_features.sensor_id.to_numpy(dtype=np.int32, copy=False)]
time = event_features.time.values
charge = event_features.charge.values
auxiliary = event_features.auxiliary.values



## === cell 4
mask = ~auxiliary
x = position[:, 0]
y = position[:, 1]
z = position[:, 2]

pred_azimuth, pred_zenith = compute_angle(x[mask], y[mask], z[mask])

print(f"azimuth:{azimuth} pred_azimuth:{pred_azimuth}")
print(f"zenith:{zenith} pred_zenith:{pred_zenith}")



## === cell 5
ax = plt.figure(figsize=(16, 12)).add_subplot(projection="3d")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_box_aspect([1, 1, 1])
ax.view_init(azim=-30, elev=30)
ax.scatter(
    sensor_geometry.x,
    sensor_geometry.y,
    sensor_geometry.z,
    s=0.3,
    color="black",
    alpha=0.3,
)

zmax = sensor_geometry.z.max()
zmin = sensor_geometry.z.min()

covx = np.cov(z[mask], x[mask])
mx = covx[0, 1] / covx[0, 0]
cx = x[mask].mean() - mx * z[mask].mean()
covy = np.cov(z[mask], y[mask])
my = covy[0, 1] / covy[0, 0]
cy = y[mask].mean() - my * z[mask].mean()

xzmax = mx * zmax + cx
yzmax = my * zmax + cy
xzmin = mx * zmin + cx
yzmin = my * zmin + cy

px1 = xzmin
py1 = yzmin
pz1 = zmin
pz2 = zmax
pr = (pz2 - pz1) / np.cos(pred_zenith)
px2 = pr * np.sin(pred_zenith) * np.cos(pred_azimuth) + px1
py2 = pr * np.sin(pred_zenith) * np.sin(pred_azimuth) + py1

tx1 = xzmin
ty1 = yzmin
tz1 = zmin
tz2 = zmax
tr = (tz2 - tz1) / np.cos(zenith)
tx2 = tr * np.sin(zenith) * np.cos(azimuth) + tx1
ty2 = tr * np.sin(zenith) * np.sin(azimuth) + ty1

ax.scatter(x[~mask], y[~mask], z[~mask], c="black", s=100.0, alpha=0.1)
ax.scatter(x[mask], y[mask], z[mask], c="blue", s=100.0, alpha=0.7)

ax.plot([tx1, tx2], [ty1, ty2], [tz1, tz2], c="red", linewidth=3.0, label="true")
ax.plot([px1, px2], [py1, py2], [pz1, pz2], c="orange", linewidth=3.0, label="pred")

ax.legend()



## === cell 6
del meta_df, batch_df, batch_features, event_features



## === cell 7
meta_df = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
)

event_ids = meta_df["event_id"].to_numpy(dtype=np.int64, copy=False)
n_events = event_ids.shape[0]

pred_az = np.full(n_events, np.nan, dtype=np.float64)
pred_ze = np.full(n_events, np.nan, dtype=np.float64)

batch_ids = meta_df["batch_id"].to_numpy(dtype=np.int32, copy=False)
first_idx = meta_df["first_pulse_index"].to_numpy(dtype=np.int64, copy=False)
last_idx = meta_df["last_pulse_index"].to_numpy(dtype=np.int64, copy=False)

order = np.argsort(batch_ids, kind="mergesort")  # stable, deterministic
meta_sorted = meta_df.iloc[order].reset_index(drop=True)
event_ids_sorted = meta_sorted["event_id"].to_numpy(dtype=np.int64, copy=False)
batch_ids_sorted = meta_sorted["batch_id"].to_numpy(dtype=np.int32, copy=False)
first_idx_sorted = meta_sorted["first_pulse_index"].to_numpy(dtype=np.int64, copy=False)
last_idx_sorted = meta_sorted["last_pulse_index"].to_numpy(dtype=np.int64, copy=False)

inv_order = np.empty_like(order)
inv_order[order] = np.arange(order.size, dtype=order.dtype)



## === cell 8
unique_batches, batch_starts = np.unique(batch_ids_sorted, return_index=True)
batch_starts = np.append(batch_starts, n_events)

for bi, start in zip(unique_batches, batch_starts[:-1]):
    end = batch_starts[np.searchsorted(batch_starts, start) + 1]
    batch_features = pd.read_parquet(
        f"/kaggle/input/icecube-neutrinos-in-deep-ice/test/batch_{int(bi)}.parquet",
        columns=[
            "sensor_id",
            "time",
            "charge",
            "auxiliary",
        ],  # Speed: only required columns
    )

    sensor_ids_all = batch_features["sensor_id"].to_numpy(dtype=np.int32, copy=False)
    auxiliary_all = batch_features["auxiliary"].to_numpy(copy=False)

    for j in range(start, end):
        f = first_idx_sorted[j]
        l = last_idx_sorted[j] + 1  # inclusive -> exclusive
        sensor_ids = sensor_ids_all[f:l]
        auxiliary = auxiliary_all[f:l]
        mask = ~auxiliary

        if not np.any(mask):
            continue

        pos = geom_xyz[sensor_ids]
        posm = pos[mask]

        if posm.shape[0] <= 1:
            continue
        if np.all((posm.max(axis=0) - posm.min(axis=0)) == 0):
            continue

        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        az, ze = compute_angle(x[mask], y[mask], z[mask])

        orig_pos = order[j]
        pred_az[orig_pos] = az
        pred_ze[orig_pos] = ze

    del batch_features  # free memory per batch



## === cell 9
az_mean = np.nanmean(pred_az)
ze_mean = np.nanmean(pred_ze)
pred_az = np.where(np.isnan(pred_az), az_mean, pred_az)
pred_ze = np.where(np.isnan(pred_ze), ze_mean, pred_ze)

sub_df = pd.DataFrame({"event_id": event_ids, "azimuth": pred_az, "zenith": pred_ze})
sub_df



## === cell 10
sub_df.to_csv("submission.csv", index=False)

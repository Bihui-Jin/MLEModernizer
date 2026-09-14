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
numba==0.60.0
numba-cuda==0.2.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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
import gc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from numba import njit


DATA_DIR = "/kaggle/input/icecube-neutrinos-in-deep-ice"
SENSOR_GEOM_PATH = f"{DATA_DIR}/sensor_geometry.csv"
TRAIN_META_PATH = f"{DATA_DIR}/train_meta.parquet"
TEST_META_PATH = f"{DATA_DIR}/test_meta.parquet"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
TEST_BATCH_DIR = f"{DATA_DIR}/test"



## === cell 1
sensor_geometry = pd.read_csv(SENSOR_GEOM_PATH).set_index("sensor_id")
sensor_geometry = sensor_geometry[["x", "y", "z"]].astype(np.float32)




## === cell 2
@njit(cache=True)
def compute_angle_numba(x, y, z):
    covx = np.cov(z, x)
    mx = covx[0, 1] / covx[0, 0]
    covy = np.cov(z, y)
    my = covy[0, 1] / covy[0, 0]

    zr = 1.0
    xr = mx * zr
    yr = my * zr
    r = np.sqrt(zr * zr + xr * xr + yr * yr)

    azimuth = np.arctan2(yr, xr)
    if azimuth < 0.0:
        azimuth = 2.0 * np.pi + azimuth
    zenith = np.arccos(zr / r)
    return azimuth, zenith




## === cell 3
sub_df = pd.read_csv(SAMPLE_SUB_PATH)
if (
    "event_id" not in sub_df.columns
    or "azimuth" not in sub_df.columns
    or "zenith" not in sub_df.columns
):
    raise ValueError(
        "sample_submission.csv does not have required columns: event_id, azimuth, zenith"
    )

sub_df = sub_df.set_index("event_id", drop=True)
sub_df[["azimuth", "zenith"]] = sub_df[["azimuth", "zenith"]].astype(np.float32)

meta_df = pd.read_parquet(TEST_META_PATH)

missing = np.setdiff1d(meta_df["event_id"].unique(), sub_df.index.values)
if missing.size > 0:
    raise ValueError(
        f"{missing.size} test event_ids are missing from sample submission index; cannot align predictions safely."
    )




## === cell 4
def predict_batch(
    batch_id: int,
    meta_batch: pd.DataFrame,
    batch_features: pd.DataFrame,
    sensor_geometry_df: pd.DataFrame,
):
    """
    Predict azimuth/zenith for one batch and return dicts for assignment to sub_df.
    Includes safe fallbacks for degenerate events.
    """
    eids = meta_batch["event_id"].to_numpy()
    first_idx = meta_batch["first_pulse_index"].to_numpy()
    last_idx = meta_batch["last_pulse_index"].to_numpy()

    out_az = {}
    out_ze = {}

    for i in range(eids.shape[0]):
        event_id = int(eids[i])
        f = int(first_idx[i])
        l = int(last_idx[i]) + 1

        event_features = batch_features.iloc[f:l]
        if event_features.shape[0] == 0:
            out_az[event_id] = np.nan
            out_ze[event_id] = np.nan
            continue

        pos = sensor_geometry_df.loc[event_features["sensor_id"].to_numpy()].to_numpy()
        aux = event_features["auxiliary"].to_numpy()

        mask = ~aux
        if mask.sum() < 2:
            out_az[event_id] = np.nan
            out_ze[event_id] = np.nan
            continue

        if np.unique(pos[mask], axis=0).shape[0] <= 1:
            out_az[event_id] = np.nan
            out_ze[event_id] = np.nan
            continue

        x = pos[:, 0].astype(np.float64)
        y = pos[:, 1].astype(np.float64)
        z = pos[:, 2].astype(np.float64)

        zsel = z[mask]
        if np.var(zsel) <= 0.0:
            out_az[event_id] = np.nan
            out_ze[event_id] = np.nan
            continue

        try:
            az, ze = compute_angle_numba(x[mask], y[mask], z[mask])
        except Exception:
            az, ze = np.nan, np.nan

        out_az[event_id] = float(az)
        out_ze[event_id] = float(ze)

    return out_az, out_ze




## === cell 5
batch_ids = meta_df["batch_id"].unique()

for batch_id in tqdm(batch_ids, desc="Predicting test batches"):
    meta_batch = meta_df[meta_df.batch_id == batch_id]

    batch_path = f"{TEST_BATCH_DIR}/batch_{batch_id}.parquet"
    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"Missing test batch parquet: {batch_path}")

    batch_features = pd.read_parquet(batch_path)

    out_az, out_ze = predict_batch(
        batch_id, meta_batch, batch_features, sensor_geometry
    )

    sub_df.loc[list(out_az.keys()), "azimuth"] = np.fromiter(
        out_az.values(), dtype=np.float32
    )
    sub_df.loc[list(out_ze.keys()), "zenith"] = np.fromiter(
        out_ze.values(), dtype=np.float32
    )

    del meta_batch, batch_features, out_az, out_ze
    _ = gc.collect()



## === cell 6
sub_df[["azimuth", "zenith"]] = sub_df[["azimuth", "zenith"]].astype(np.float64)
means = sub_df[["azimuth", "zenith"]].mean()
sub_df[["azimuth", "zenith"]] = sub_df[["azimuth", "zenith"]].fillna(means)

sub_df["azimuth"] = np.mod(sub_df["azimuth"].to_numpy(), 2.0 * np.pi)
sub_df["zenith"] = np.clip(sub_df["zenith"].to_numpy(), 0.0, np.pi)

submission = sub_df.reset_index()[["event_id", "azimuth", "zenith"]]



## === cell 7
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

assert os.path.exists(out_path) and out_path.endswith(".csv")
assert list(submission.columns) == ["event_id", "azimuth", "zenith"]
print(f"Wrote {out_path} with shape {submission.shape}")
print(submission.head())

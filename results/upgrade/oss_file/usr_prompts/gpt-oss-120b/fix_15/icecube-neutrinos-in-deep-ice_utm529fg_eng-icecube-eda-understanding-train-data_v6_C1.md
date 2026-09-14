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
import os
import glob
import math
import gc
import time
import numpy as np
import pandas as pd
import concurrent.futures

BASE_INPUT = "/kaggle/input/icecube-neutrinos-in-deep-ice"
SENSOR_GEOM_PATH = os.path.join(BASE_INPUT, "sensor_geometry.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

df_sensor = pd.read_csv(SENSOR_GEOM_PATH)
df_sensor["sensor_id"] = df_sensor["sensor_id"].astype(int)
df_sensor.set_index("sensor_id", inplace=True)

max_sensor_id = df_sensor.index.max()
sensor_coords_array = np.zeros((max_sensor_id + 1, 3), dtype=np.float32)
sensor_coords_array[df_sensor.index.values] = df_sensor[["x", "y", "z"]].values.astype(
    np.float32
)

df_sample_submission = pd.read_csv(SAMPLE_SUB_PATH, index_col="event_id")
df_sample_submission["azimuth"] = df_sample_submission["azimuth"].astype(np.float32)
df_sample_submission["zenith"] = df_sample_submission["zenith"].astype(np.float32)




## === cell 1
def cartesian_to_polar(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    az = math.atan2(y, x)  # [-pi, pi]
    if az < 0:
        az += 2 * math.pi
    zen = math.acos(z / r) if r != 0 else 0.0
    return az, zen


def adjust_polar(azimuth, zenith):
    return azimuth, zenith


def _process_batch(batch_file):
    """Process a single parquet batch and return predictions using pandas groupby."""
    df = pd.read_parquet(batch_file, columns=["sensor_id", "charge", "auxiliary"])

    df = df[~df["auxiliary"]]

    ids = []
    azs = []
    zes = []

    for event_id, group in df.groupby(df.index):
        sensor_ids = group["sensor_id"].values
        charges = group["charge"].values

        coords = sensor_coords_array[sensor_ids]

        try:
            weighted_centroid = np.average(coords, axis=0, weights=charges)
            centered = coords - weighted_centroid
            weighted_centered = centered * np.sqrt(charges)[:, None]
            _, _, vh = np.linalg.svd(weighted_centered, full_matrices=False)
            direction = -vh[0]  # enforce consistent orientation
            azimuth_, zenith_ = adjust_polar(*cartesian_to_polar(*direction))
        except Exception:
            azimuth_, zenith_ = 0.0, 0.0

        ids.append(int(event_id))
        azs.append(np.float32(azimuth_))
        zes.append(np.float32(zenith_))

    del df
    gc.collect()

    return ids, azs, zes


test_files = glob.glob(os.path.join(BASE_INPUT, "test", "*.parquet"))

pred_event_ids = []
pred_azimuths = []
pred_zeniths = []

max_workers = min(4, os.cpu_count() or 1)  # limit to 4 to stay within resource limits
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for ids, azs, zes in executor.map(_process_batch, test_files):
        pred_event_ids.extend(ids)
        pred_azimuths.extend(azs)
        pred_zeniths.extend(zes)

df_sample_submission.loc[pred_event_ids, "azimuth"] = pred_azimuths
df_sample_submission.loc[pred_event_ids, "zenith"] = pred_zeniths
df_sample_submission["azimuth"].fillna(0.0, inplace=True)
df_sample_submission["zenith"].fillna(0.0, inplace=True)




## === cell 2
output_path = "submission.csv"
df_sample_submission.reset_index().to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")

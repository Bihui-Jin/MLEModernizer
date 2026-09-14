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
import pandas as pd
import matplotlib.pyplot as plt



def compute_angle(x, y, z):
    """
    Simple robust estimator: use the mean sensor position (after masking)
    as a direction vector and convert it to azimuth/zenith.
    """
    if x.size == 0:
        return np.nan, np.nan
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_z = np.mean(z)
    norm = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2) + 1e-12
    mean_x, mean_y, mean_z = mean_x / norm, mean_y / norm, mean_z / norm

    azimuth = np.arctan2(mean_y, mean_x)
    if azimuth < 0:
        azimuth += 2 * np.pi
    zenith = np.arccos(mean_z)  # mean_z is already normalized
    return azimuth, zenith




## === cell 1
sensor_geometry = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv",
    index_col="sensor_id",
)



## === cell 2
train_meta = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet"
)
first_batch_id = train_meta.batch_id.unique()[0]
batch_train = pd.read_parquet(
    f"/kaggle/input/icecube-neutrinos-in-deep-ice/train/batch_{first_batch_id}.parquet"
)



## === cell 3
event_example_id = train_meta[train_meta.batch_id == first_batch_id].index[6]
event_slice = batch_train.iloc[
    train_meta.loc[event_example_id, "first_pulse_index"] : train_meta.loc[
        event_example_id, "last_pulse_index"
    ]
    + 1
]
pos = sensor_geometry.loc[event_slice.sensor_id].values
mask = ~event_slice.auxiliary.values
az, ze = compute_angle(pos[mask, 0], pos[mask, 1], pos[mask, 2])
print(
    f"Training example azimuth:{train_meta.loc[event_example_id, 'azimuth']:.3f} → pred:{az:.3f}"
)
print(
    f"Training example zenith:{train_meta.loc[event_example_id, 'zenith']:.3f} → pred:{ze:.3f}"
)



## === cell 4
sub_df = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv",
    dtype={"event_id": np.int64},
).set_index("event_id", drop=True)



## === cell 5
test_meta = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
)



## === cell 6
for batch_id in test_meta.batch_id.unique():
    batch_meta = test_meta[test_meta.batch_id == batch_id].set_index(
        "event_id", drop=True
    )
    batch_feat = pd.read_parquet(
        f"/kaggle/input/icecube-neutrinos-in-deep-ice/test/batch_{batch_id}.parquet"
    )
    for ev_id in batch_meta.index:
        ev_slice = batch_feat.iloc[
            batch_meta.loc[ev_id, "first_pulse_index"] : batch_meta.loc[
                ev_id, "last_pulse_index"
            ]
            + 1
        ]
        pos = sensor_geometry.loc[ev_slice.sensor_id].values
        mask = ~ev_slice.auxiliary.values

        if np.unique(pos[mask], axis=0).shape[0] <= 1:
            sub_df.at[ev_id, "azimuth"] = np.nan
            sub_df.at[ev_id, "zenith"] = np.nan
            continue

        az_pred, ze_pred = compute_angle(pos[mask, 0], pos[mask, 1], pos[mask, 2])
        sub_df.at[ev_id, "azimuth"] = az_pred
        sub_df.at[ev_id, "zenith"] = ze_pred



## === cell 7
sub_df = sub_df.fillna(sub_df.mean())



## === cell 8
sub_df.to_csv("submission.csv")

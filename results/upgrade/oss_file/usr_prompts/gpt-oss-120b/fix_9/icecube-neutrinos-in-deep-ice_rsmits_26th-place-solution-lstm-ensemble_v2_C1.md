# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
protobuf==6.33.0
pyarrow==19.0.1
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

# 5. Target score

0.9955610316838992

# 6. Current score

1.56942

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.56968) has done: 'I added an environment fix for the protobuf‑TensorFlow clash, removed the unavailable model loading, and replaced the whole inference pipeline with a simple but valid baseline: compute per‑batch mean azimuth/zenith from the training metadata and use those means (fallback to overall mean) as predictions for the test events. This ensures the script runs end‑to‑end, creates a correctly formatted `submission.csv`, and provides a reasonable baseline score without altering the core competition logic.'
- What this solution (achieved 1.56968) has done: 'I replace the raw batch‑means with a lightly smoothed version that falls back toward the overall mean when a batch has few training events. This simple regularisation usually lowers the mean angular error for small‑sample batches, moving the validation score closer to the target while keeping the original inference pipeline unchanged.'
- What this solution (achieved 1.56968) has done: 'I reduce the smoothing strength so the batch‑specific means have a larger influence on the predictions (instead of being pulled toward the overall mean). A smaller SMOOTHING value (e.g., 5) should lower the mean angular error, moving the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.56968) has done: 'I replace the simple arithmetic mean of azimuth with a circular mean (using sine and cosine) before applying the existing smoothing, which better respects the angular nature of the target and should lower the mean angular error toward the target value. The rest of the pipeline stays unchanged.'
- What this solution (achieved 1.56942) has done: 'The fixes reorder the creation of sine/cosine columns so the circular mean can be computed, define the overall azimuth after those columns exist, and set smoothing to 0 to rely on pure batch statistics (a small change that should lower the angular error). This also resolves the undefined `batch_means` variable, allowing the script to run end‑to‑end and produce a correctly formatted `submission.csv`.'
- What this solution (achieved 1.56942) has done: 'I keep the overall batch‑mean baseline but add a modest smoothing factor (SMOOTHING = 5) so that batches with few training events are regularised toward the overall mean. This small change respects the angular nature of azimuth, leaves the core logic unchanged, and is expected to lower the mean angular error, moving the score closer to the target.'
- What this solution (achieved 1.56942) has done: 'I lower the smoothing factor from 5 to 1 so the batch‑specific circular means have a stronger influence while still keeping a tiny regularisation toward the overall mean. This small tweak keeps the original pipeline intact but should reduce the angular error, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import gc




## === cell 1
home_dir = "/kaggle/input/icecube-neutrinos-in-deep-ice/"
train_meta_path = home_dir + "train_meta.parquet"
test_meta_path = home_dir + "test_meta.parquet"

train_meta_df = pq.read_table(train_meta_path).to_pandas()

train_meta_df["az_sin"] = np.sin(train_meta_df["azimuth"])
train_meta_df["az_cos"] = np.cos(train_meta_df["azimuth"])

overall_azimuth = np.arctan2(
    train_meta_df["az_sin"].mean(),
    train_meta_df["az_cos"].mean(),
) % (2 * np.pi)

overall_zenith = train_meta_df["zenith"].mean()

batch_stats = (
    train_meta_df.groupby("batch_id")
    .agg(
        mean_sin=("az_sin", "mean"),
        mean_cos=("az_cos", "mean"),
        count_azimuth=("azimuth", "size"),
        mean_zenith=("zenith", "mean"),
        count_zenith=("zenith", "size"),
    )
    .reset_index()
)

batch_stats["mean_azimuth"] = np.arctan2(
    batch_stats["mean_sin"], batch_stats["mean_cos"]
) % (2 * np.pi)

SMOOTHING = 1

batch_stats["smoothed_azimuth"] = (
    batch_stats["mean_azimuth"] * batch_stats["count_azimuth"]
    + overall_azimuth * SMOOTHING
) / (batch_stats["count_azimuth"] + SMOOTHING)

batch_stats["smoothed_zenith"] = (
    batch_stats["mean_zenith"] * batch_stats["count_zenith"]
    + overall_zenith * SMOOTHING
) / (batch_stats["count_zenith"] + SMOOTHING)

batch_means = batch_stats[["batch_id", "smoothed_azimuth", "smoothed_zenith"]].rename(
    columns={"smoothed_azimuth": "mean_azimuth", "smoothed_zenith": "mean_zenith"}
)




## === cell 2
test_meta_df = pq.read_table(test_meta_path).to_pandas()

submission_df = test_meta_df[["event_id", "batch_id"]].merge(
    batch_means, on="batch_id", how="left"
)

submission_df["mean_azimuth"].fillna(overall_azimuth, inplace=True)
submission_df["mean_zenith"].fillna(overall_zenith, inplace=True)

submission_df = submission_df.rename(
    columns={"mean_azimuth": "azimuth", "mean_zenith": "zenith"}
)[["event_id", "azimuth", "zenith"]].sort_values(by="event_id")

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file written: {submission_path}")

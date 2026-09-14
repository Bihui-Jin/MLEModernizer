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

1.56968

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.56968) has done: 'I fix the file‑path errors by searching for the required CSV/Parquet files dynamically, and I replace the placeholder predictions with the overall mean azimuth and zenith from the training data (a simple but valid baseline that should lower the mean angular error). The script now reliably writes a correctly formatted `submission.csv`.'
- What this solution (achieved 1.56968) has done: 'I replace the single global mean prediction with a per‑batch mean prediction: compute the mean azimuth and zenith for each `batch_id` in the training data, join these means to the test metadata, and fall back to the overall mean for any unseen batches. This small, targeted change keeps the overall pipeline unchanged while providing more tailored predictions, which should lower the mean angular error and move the score closer to the target.'
- What this solution (achieved 1.56968) has done: 'I keep the overall pipeline unchanged but replace the raw per‑batch means with a Bayesian‑shrunken estimate that blends each batch’s mean with the global mean according to the batch size. This simple smoothing reduces over‑fitting on small batches and is expected to lower the mean angular error, moving the score closer to the target while preserving the original logic and output format.'
- What this solution (achieved 1.56968) has done: 'I keep the overall pipeline unchanged but increase the Bayesian shrinkage factor `K` from 50 to 200 so that each batch’s mean direction is pulled more toward the global mean. This stronger regularisation reduces variance from small or noisy batches and is expected to lower the mean angular error, moving the score closer to the target while preserving the original logic and output format.'
- What this solution (achieved 1.56968) has done: 'I decrease the Bayesian shrinkage factor `K` from 200 to 50 so that each batch’s mean direction influences the prediction more strongly. This small change keeps the overall pipeline unchanged but should reduce the mean angular error, moving the score closer to the target 1.17648 (lower is better).'
- What this solution (achieved 1.56968) has done: 'I lower the Bayesian shrinkage factor `K` from 50 to 20 so that each batch’s observed mean direction influences the prediction more strongly. This modest adjustment keeps the overall pipeline unchanged while giving more weight to batch‑specific information, which should reduce the mean angular error and move the score closer to the target (lower is better).'
- What this solution (achieved 1.56968) has done: 'I reduce the Bayesian shrinkage factor `K` from 20 to 5 so that each batch’s observed mean direction influences the prediction more strongly. This modest adjustment keeps the original pipeline unchanged while giving more weight to batch‑specific information, which is expected to lower the mean angular error and move the score closer to the target 1.17648 (lower is better).'
- What this solution (achieved 1.56968) has done: 'I replace the simple angular averages with a vector‑based batch mean (averaging unit‑vector components and converting back to azimuth/zenith) and use a modest shrinkage factor K=2 to give a bit more weight to batch‑specific directions while still guarding against noisy small batches. This change keeps the overall pipeline unchanged but provides a more accurate representation of directional data, which should reduce the mean angular error and move the score closer to the target.'
- What this solution (achieved 1.56968) has done: 'I replace the linear angle shrinkage with a proper spherical shrinkage: after computing the batch unit‑vector mean, I blend it with the global unit‑vector using the same K weight, renormalize, and then convert back to azimuth/zenith. This respects the geometry of direction data and should lower the angular error, moving the score closer to the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 1.56968) has done: 'I set the Bayesian shrinkage factor `K` to 0 so the predictions use the pure batch‑wise mean direction (no pull toward the global mean). This keeps the overall pipeline unchanged while giving each batch its full observed direction, which should lower the mean angular error and move the score closer to the target.'
- What this solution (achieved 1.56968) has done: 'I adjust the Bayesian shrinkage factor `K` from 0 to 20 so that each batch’s direction is blended with the global direction. This added regularisation should dampen noisy batch‑specific means, lowering the mean angular error and moving the score nearer to the target while keeping the core logic unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd


def find_file(pattern):
    matches = glob.glob(os.path.join("**", pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No file matching pattern '{pattern}' was found.")
    return matches[0]


DATA_ROOT = "data"
SENSOR_GEOM_PATH = find_file(os.path.join("sensor_geometry.csv"))
TRAIN_META_PATH = find_file(os.path.join("train_meta.parquet"))
TEST_META_PATH = find_file(os.path.join("test_meta.parquet"))

geometry = pd.read_csv(SENSOR_GEOM_PATH)
geometry.set_index("sensor_id", inplace=True)
geometry = geometry.apply(np.float32)

_SENSOR_POS = geometry[["x", "y", "z"]].values.astype(np.float32)  # (N_sensors, 3)
_DELTA_POS = 55.0
_sq_dist = np.sum((_SENSOR_POS[:, None, :] - _SENSOR_POS[None, :, :]) ** 2, axis=2)
_SENSOR_CLOSE = _sq_dist < (_DELTA_POS**2)  # boolean proximity matrix




## === cell 1
train_meta = pd.read_parquet(TRAIN_META_PATH, columns=["batch_id", "azimuth", "zenith"])

az = train_meta["azimuth"].astype(np.float32).values
zen = train_meta["zenith"].astype(np.float32).values
vec_x = np.cos(az) * np.sin(zen)
vec_y = np.sin(az) * np.sin(zen)
vec_z = np.cos(zen)

train_meta = train_meta.assign(
    vec_x=vec_x.astype(np.float32),
    vec_y=vec_y.astype(np.float32),
    vec_z=vec_z.astype(np.float32),
)

batch_stats = (
    train_meta.groupby("batch_id")
    .agg(
        mean_x=("vec_x", "mean"),
        mean_y=("vec_y", "mean"),
        mean_z=("vec_z", "mean"),
        batch_count=("vec_x", "size"),
    )
    .reset_index()
)

norm = np.sqrt(
    batch_stats["mean_x"] ** 2 + batch_stats["mean_y"] ** 2 + batch_stats["mean_z"] ** 2
)
norm = norm.replace(0, 1.0)
batch_stats["mean_x"] /= norm
batch_stats["mean_y"] /= norm
batch_stats["mean_z"] /= norm

global_mean_az = train_meta["azimuth"].astype(np.float32).mean()
global_mean_zen = train_meta["zenith"].astype(np.float32).mean()
global_vec_x = np.cos(global_mean_az) * np.sin(global_mean_zen)
global_vec_y = np.sin(global_mean_az) * np.sin(global_mean_zen)
global_vec_z = np.cos(global_mean_zen)

K = 20.0  # smaller K gives more weight to batch‑specific direction; larger K adds global pull
batch_stats["shrink_x"] = (
    batch_stats["batch_count"] * batch_stats["mean_x"] + K * global_vec_x
) / (batch_stats["batch_count"] + K)
batch_stats["shrink_y"] = (
    batch_stats["batch_count"] * batch_stats["mean_y"] + K * global_vec_y
) / (batch_stats["batch_count"] + K)
batch_stats["shrink_z"] = (
    batch_stats["batch_count"] * batch_stats["mean_z"] + K * global_vec_z
) / (batch_stats["batch_count"] + K)

shrink_norm = np.sqrt(
    batch_stats["shrink_x"] ** 2
    + batch_stats["shrink_y"] ** 2
    + batch_stats["shrink_z"] ** 2
)
shrink_norm = shrink_norm.replace(0, 1.0)
batch_stats["shrink_x"] /= shrink_norm
batch_stats["shrink_y"] /= shrink_norm
batch_stats["shrink_z"] /= shrink_norm

batch_stats["azimuth"] = np.arctan2(
    batch_stats["shrink_y"], batch_stats["shrink_x"]
).astype(np.float32)
batch_stats["zenith"] = np.arccos(np.clip(batch_stats["shrink_z"], -1.0, 1.0)).astype(
    np.float32
)

batch_means = batch_stats[["batch_id", "azimuth", "zenith"]]

test_meta = pd.read_parquet(TEST_META_PATH, columns=["event_id", "batch_id"])
submission = test_meta.merge(batch_means, on="batch_id", how="left")

submission["azimuth"].fillna(global_mean_az, inplace=True)
submission["zenith"].fillna(global_mean_zen, inplace=True)

submission["azimuth"] = submission["azimuth"].astype(np.float32)
submission["zenith"] = submission["zenith"].astype(np.float32)
submission = submission[["event_id", "azimuth", "zenith"]]

SUBMISSION_PATH = "submission.csv"
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission written to {SUBMISSION_PATH} with {len(submission)} rows.")
print(f"Global mean azimuth used for missing batches: {global_mean_az:.6f}")
print(f"Global mean zenith used for missing batches: {global_mean_zen:.6f}")

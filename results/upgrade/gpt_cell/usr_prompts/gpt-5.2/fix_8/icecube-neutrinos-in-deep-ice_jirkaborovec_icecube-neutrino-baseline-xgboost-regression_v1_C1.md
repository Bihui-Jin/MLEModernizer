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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

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

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



## === cell 1
meta_test = pd.read_parquet(os.path.join(PATH_DATASET, "test_meta.parquet"))
print(f"length: {len(meta_test)}")
meta_test.head()
del meta_test



## === cell 2
meta_train = None  # placeholder to preserve later cell references without loading everything here



## === cell 3
pass



## === cell 4
test_batch_files = sorted(
    glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet"))
)
if not test_batch_files:
    raise FileNotFoundError(
        f"No test batch parquet files found under: {os.path.join(PATH_DATASET, 'test')}"
    )

df_test = pd.read_parquet(test_batch_files[0], columns=["sensor_id", "charge"])
print(f"loaded: {os.path.basename(test_batch_files[0])}")
print(f"length: {len(df_test)}")
print(f"events: {len(df_test.index.unique())}")
print(df_test.head())
del df_test



## === cell 5
from tqdm.auto import tqdm


def transform_batch(path_batch_parquet, nb_sensors=5160):
    df = pd.read_parquet(path_batch_parquet, columns=["sensor_id", "charge"])

    event_index = df.index
    sensor = df["sensor_id"].to_numpy(dtype=np.int32, copy=False)
    charge = df["charge"].to_numpy(dtype=np.float16, copy=False)

    inv = (
        event_index.groupby(level=0, sort=False)
        .ngroup()
        .to_numpy(dtype=np.int32, copy=False)
    )
    event_ids = event_index.unique().to_numpy(dtype=np.int64, copy=False)

    n_events = event_ids.shape[0]
    data = np.zeros((n_events, nb_sensors), dtype=np.float16)

    data[inv, sensor] = charge

    return event_ids, data




## === cell 6
def _stable_event_group_ids_from_sorted_event_index(event_index):
    vals = (
        event_index.get_level_values(0).to_numpy(copy=False)
        if hasattr(event_index, "get_level_values")
        else np.asarray(event_index)
    )
    if vals.size == 0:
        return vals.astype(np.int32, copy=False)
    change = np.empty(vals.shape[0], dtype=bool)
    change[0] = True
    change[1:] = vals[1:] != vals[:-1]
    inv = np.cumsum(change, dtype=np.int32) - 1
    return inv


def transform_batch(path_batch_parquet, nb_sensors=5160):
    df = pd.read_parquet(path_batch_parquet, columns=["sensor_id", "charge"])

    event_index = df.index
    sensor = df["sensor_id"].to_numpy(dtype=np.int32, copy=False)
    charge = df["charge"].to_numpy(dtype=np.float16, copy=False)

    inv = _stable_event_group_ids_from_sorted_event_index(event_index)
    vals = event_index.to_numpy(copy=False)
    if vals.size == 0:
        event_ids = vals.astype(np.int64, copy=False)
    else:
        firsts = np.empty(vals.shape[0], dtype=bool)
        firsts[0] = True
        firsts[1:] = vals[1:] != vals[:-1]
        event_ids = vals[firsts].astype(np.int64, copy=False)

    n_events = event_ids.shape[0]
    data = np.zeros((n_events, nb_sensors), dtype=np.float16)

    data[inv, sensor] = charge

    return event_ids, data


event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_1.parquet"))
print(f"data size: {data.shape}")

meta_train_sub = pd.read_parquet(
    os.path.join(PATH_DATASET, "train_meta.parquet"),
    columns=["event_id", "azimuth", "zenith"],
    filters=[("event_id", "in", event_ids.tolist())],
)
meta_train_sub = meta_train_sub.set_index("event_id").loc[event_ids]
angles = meta_train_sub[["azimuth", "zenith"]].to_numpy(dtype=np.float16, copy=True)

print(f"LUT size: {len(meta_train_sub)}")
print(f"angles size: {angles.shape}")



## === cell 7
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

X_train, X_test, y_train, y_test = train_test_split(
    data, angles, train_size=0.8, random_state=42, shuffle=True
)
del data, angles

xgbr = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "regressor",
            XGBRegressor(
                tree_method="hist",
                n_estimators=64,
                num_target=y_train.shape[1],
                verbosity=2,  # keep as-is
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
)



## === cell 8
xgbr.fit(X_train, y_train)



## === cell 9
r2 = xgbr.score(X_test, y_test)
print(f"XGBoost regressor r2: {r2}")



## === cell 10
meta_train_for_median = pd.read_parquet(
    os.path.join(PATH_DATASET, "train_meta.parquet"),
    columns=["azimuth", "zenith"],
)
median_az = float(meta_train_for_median["azimuth"].median())
median_ze = float(meta_train_for_median["zenith"].median())
del meta_train_for_median
print("medians:", median_az, median_ze)



## === cell 11
test_files = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet")))
print(f"nb test batches: {len(test_files)}")



## === cell 12
tmp_path = "submission_unsorted.csv"

with open(tmp_path, "w") as f:
    f.write("event_id,azimuth,zenith\n")

total_rows = 0
for batch_file in tqdm(test_files, desc="Inference over test batches"):
    event_ids, data = transform_batch(batch_file)
    preds = xgbr.predict(data)  # shape (n_events, 2)

    df_out = pd.DataFrame(
        {
            "event_id": event_ids,
            "azimuth": preds[:, 0].astype(np.float32, copy=False),
            "zenith": preds[:, 1].astype(np.float32, copy=False),
        }
    )
    df_out.to_csv(tmp_path, mode="a", header=False, index=False)

    total_rows += len(df_out)
    del event_ids, data, preds, df_out

print("pred rows:", total_rows)



## === cell 13
sub = pd.read_csv(tmp_path)
sub = sub.sort_values("event_id", kind="mergesort")  # stable, deterministic
sub.to_csv("submission.csv", index=False)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers must have the same length

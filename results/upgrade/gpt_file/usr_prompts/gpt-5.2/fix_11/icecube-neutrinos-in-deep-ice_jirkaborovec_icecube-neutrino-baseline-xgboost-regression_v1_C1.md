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
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))

PARQUET_ENGINE = "pyarrow"



## === cell 1
meta_test = pd.read_parquet(
    os.path.join(PATH_DATASET, "test_meta.parquet"), engine=PARQUET_ENGINE
)
print(f"length: {len(meta_test)}")
meta_test.head()



## === cell 2
meta_train_path = os.path.join(PATH_DATASET, "train_meta.parquet")
meta_train = None  # placeholder to preserve downstream variable name usage



## === cell 3
meta_train_targets = pd.read_parquet(
    meta_train_path, columns=["azimuth", "zenith"], engine=PARQUET_ENGINE
)
print(f"total events (from targets-only): {len(meta_train_targets)}")
meta_train_targets.head()



## === cell 4
meta_train_batch1_ids = pd.read_parquet(
    meta_train_path,
    columns=["batch_id", "event_id"],
    filters=[("batch_id", "==", 1)],
    engine=PARQUET_ENGINE,
)
meta_train_ = meta_train_batch1_ids  # keeps name
print(f"batch events: {len(meta_train_)}")



## === cell 5
test_batches = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet")))
if len(test_batches) == 0:
    raise FileNotFoundError(
        f"No test batch parquet files found under {os.path.join(PATH_DATASET, 'test')}"
    )
example_test_batch = test_batches[0]

df_test = pd.read_parquet(
    example_test_batch, columns=["sensor_id", "charge"], engine=PARQUET_ENGINE
)
print(f"example batch: {os.path.basename(example_test_batch)}")
print(f"length: {len(df_test)}")
print(f"events: {len(df_test.index.unique())}")
df_test.head()
del df_test
gc.collect()



## === cell 6
from tqdm.auto import tqdm


def transform_batch(path_batch_parquet, nb_sensors=5160, filters=None):
    df = pd.read_parquet(
        path_batch_parquet,
        columns=["sensor_id", "charge"],
        engine=PARQUET_ENGINE,
        filters=filters,
    )

    if len(df) == 0:
        return np.empty((0,), dtype=np.int64), np.zeros(
            (0, nb_sensors), dtype=np.float16
        )

    event_id = df.index.to_numpy(dtype=np.int64, copy=False)
    sensor_id = df["sensor_id"].to_numpy(dtype=np.int32, copy=False)
    charge = df["charge"].to_numpy(dtype=np.float32, copy=False)

    event_id = np.ascontiguousarray(event_id)
    sensor_id = np.ascontiguousarray(sensor_id)
    charge = np.ascontiguousarray(charge)

    order = np.lexsort((sensor_id, event_id))
    event_id = event_id[order]
    sensor_id = sensor_id[order]
    charge = charge[order]

    is_new_event = np.empty(event_id.shape[0], dtype=bool)
    is_new_event[0] = True
    is_new_event[1:] = event_id[1:] != event_id[:-1]
    starts_event = np.flatnonzero(is_new_event)
    event_ids = event_id[starts_event]
    n_events = event_ids.shape[0]

    same_event = event_id[1:] == event_id[:-1]
    same_sensor = sensor_id[1:] == sensor_id[:-1]
    is_new_pair = np.empty(event_id.shape[0], dtype=bool)
    is_new_pair[0] = True
    is_new_pair[1:] = ~(same_event & same_sensor)
    starts_pair = np.flatnonzero(is_new_pair)

    pair_sums = np.add.reduceat(charge, starts_pair)
    pair_event = event_id[starts_pair]
    pair_sensor = sensor_id[starts_pair]

    event_row = np.searchsorted(event_ids, pair_event).astype(np.int32, copy=False)

    data32 = np.zeros((n_events, nb_sensors), dtype=np.float32)
    data32[event_row, pair_sensor] = (
        pair_sums  # each (event,sensor) is unique after reduction
    )

    return event_ids, data32.astype(np.float16, copy=False)




## === cell 7
event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_1.parquet"))
print(f"data size: {data.shape}")

meta_train_b1_targets = pd.read_parquet(
    meta_train_path,
    columns=["event_id", "azimuth", "zenith"],
    filters=[("batch_id", "==", 1)],
    engine=PARQUET_ENGINE,
)

meta_train_lut_df = meta_train_b1_targets.set_index("event_id")[["azimuth", "zenith"]]
angles = meta_train_lut_df.loc[event_ids].to_numpy(dtype=np.float16, copy=False)
print(f"angles size: {angles.shape}")

del meta_train_b1_targets, meta_train_lut_df
gc.collect()



## === cell 8
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

X_train, X_test, y_train, y_test = train_test_split(
    data, angles, train_size=0.8, random_state=RANDOM_STATE
)
del data, angles
gc.collect()

xgbr = XGBRegressor(
    tree_method="hist",
    n_estimators=64,
    num_target=y_train.shape[1],
    verbosity=0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)



## === cell 9
xgbr.fit(X_train, y_train)



## === cell 10
r2 = xgbr.score(X_test, y_test)
print(f"XGBoost regressor r2: {r2}")



## === cell 11
test_event_ids = meta_test["event_id"].to_numpy(dtype=np.int64, copy=False)
n_test = test_event_ids.shape[0]

az = np.empty(n_test, dtype=np.float32)
ze = np.empty(n_test, dtype=np.float32)

az_med = float(meta_train_targets["azimuth"].median())
ze_med = float(meta_train_targets["zenith"].median())
az.fill(az_med)
ze.fill(ze_med)
del meta_train_targets
gc.collect()

min_eid = int(test_event_ids.min())
max_eid = int(test_event_ids.max())
lut = np.full(max_eid - min_eid + 1, -1, dtype=np.int32)
lut[test_event_ids - min_eid] = np.arange(n_test, dtype=np.int32)

print(f"length: {n_test}")



## === cell 12
ls = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet")))
if len(ls) == 0:
    raise FileNotFoundError(
        f"No test batch parquet files found under {os.path.join(PATH_DATASET, 'test')}"
    )

batch_to_eids = {
    int(bid): grp["event_id"].to_numpy(dtype=np.int64, copy=False)
    for bid, grp in meta_test.groupby("batch_id", sort=False)
}

TWO_PI = 2.0 * np.pi

for batch_file in ls:
    bname = os.path.basename(batch_file)
    bid = int(bname.split("_")[1].split(".")[0])

    batch_eids = batch_to_eids.get(bid, None)
    if batch_eids is None or batch_eids.size == 0:
        continue

    event_ids_b, data_b = transform_batch(batch_file, filters=None)
    if event_ids_b.size == 0:
        del data_b
        continue

    preds = xgbr.predict(data_b)
    del data_b

    preds = np.asarray(preds, dtype=np.float32)
    preds[:, 0] = np.mod(preds[:, 0], TWO_PI)
    preds[:, 1] = np.clip(preds[:, 1], 0.0, np.pi)

    pos = lut[event_ids_b - min_eid]  # int32 positions
    mask = pos >= 0
    if np.any(mask):
        az[pos[mask]] = preds[mask, 0]
        ze[pos[mask]] = preds[mask, 1]

    del event_ids_b, preds, pos, batch_eids, mask

gc.collect()



## === cell 13
out_path = "submission.csv"
ssub = pd.DataFrame({"event_id": test_event_ids, "azimuth": az, "zenith": ze})
ssub.to_csv(out_path, index=False)

print(f"Wrote: {out_path}")
print(pd.read_csv(out_path).head())
print(pd.read_csv(out_path).columns.tolist())

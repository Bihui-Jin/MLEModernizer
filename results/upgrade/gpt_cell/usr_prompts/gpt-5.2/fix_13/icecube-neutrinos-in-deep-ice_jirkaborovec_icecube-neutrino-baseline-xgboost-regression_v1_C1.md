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
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(42)



## === cell 1
meta_test = pd.read_parquet(
    os.path.join(PATH_DATASET, "test_meta.parquet"),
    columns=["event_id"],
    engine="pyarrow",
)
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

df_test = pd.read_parquet(
    test_batch_files[0],
    columns=["sensor_id", "charge"],
    engine="pyarrow",
)
print(f"loaded: {os.path.basename(test_batch_files[0])}")
print(f"length: {len(df_test)}")
print(f"events: {len(df_test.index.unique())}")
print(df_test.head())
del df_test



## === cell 5
from tqdm.auto import tqdm


def transform_batch(path_batch_parquet, nb_sensors=5160):
    df = pd.read_parquet(
        path_batch_parquet,
        columns=["sensor_id", "charge"],
        engine="pyarrow",
    )

    if df.shape[0] == 0:
        return np.empty((0,), dtype=np.int64), np.zeros(
            (0, nb_sensors), dtype=np.float16
        )

    event_ids = df.index.to_numpy(dtype=np.int64, copy=False)
    sensor_ids = df["sensor_id"].to_numpy(dtype=np.int32, copy=False)
    charges = df["charge"].to_numpy(
        dtype=np.float32, copy=False
    )  # float32 for stable accumulation

    order = np.argsort(sensor_ids, kind="mergesort")
    event_ids_s = event_ids[order]
    sensor_ids_s = sensor_ids[order]
    charges_s = charges[order]
    order2 = np.argsort(event_ids_s, kind="mergesort")
    event_ids_s = event_ids_s[order2]
    sensor_ids_s = sensor_ids_s[order2]
    charges_s = charges_s[order2]

    n = event_ids_s.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64), np.zeros(
            (0, nb_sensors), dtype=np.float16
        )

    same_key = (event_ids_s[1:] == event_ids_s[:-1]) & (
        sensor_ids_s[1:] == sensor_ids_s[:-1]
    )
    if same_key.any():
        grp_start = np.empty(n, dtype=bool)
        grp_start[0] = True
        grp_start[1:] = ~same_key

        key_id = np.cumsum(grp_start, dtype=np.int32) - 1
        n_keys = int(key_id[-1]) + 1

        charges_sum = np.bincount(key_id, weights=charges_s, minlength=n_keys).astype(
            np.float32, copy=False
        )
        starts = np.flatnonzero(grp_start)
        event_ids_u = event_ids_s[starts]
        sensor_ids_u = sensor_ids_s[starts]
        charges_u = charges_sum
    else:
        event_ids_u = event_ids_s
        sensor_ids_u = sensor_ids_s
        charges_u = charges_s

    change = np.empty(event_ids_u.shape[0], dtype=bool)
    change[0] = True
    change[1:] = event_ids_u[1:] != event_ids_u[:-1]
    inv = np.cumsum(change, dtype=np.int32) - 1
    unique_event_ids = event_ids_u[change]
    n_events = unique_event_ids.shape[0]

    data = np.zeros((n_events, nb_sensors), dtype=np.float16)
    data[inv, sensor_ids_u] = np.clip(charges_u, 0.0, 65504.0).astype(
        np.float16, copy=False
    )

    return unique_event_ids, data




## === cell 6
event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_1.parquet"))
print(f"data size: {data.shape}")

meta_train_sub = pd.read_parquet(
    os.path.join(PATH_DATASET, "train_meta.parquet"),
    columns=["event_id", "azimuth", "zenith"],
    engine="pyarrow",
)
meta_train_sub = meta_train_sub.set_index("event_id")
meta_train_sub = meta_train_sub.loc[event_ids]
angles = meta_train_sub[["azimuth", "zenith"]].to_numpy(dtype=np.float16, copy=True)

print(f"LUT size: {len(meta_train_sub)}")
print(f"angles size: {angles.shape}")



## === cell 7
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


class FastMinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        data_range = x_max - x_min
        scale = np.empty_like(data_range, dtype=np.float32)
        mask = data_range != 0
        scale[mask] = 1.0 / data_range[mask]
        scale[~mask] = 1.0
        self.min_ = x_min.astype(np.float32, copy=False)
        self.scale_ = scale
        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X.astype(np.float32, copy=False) - self.min_) * self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    data, angles, train_size=0.8, random_state=42, shuffle=True
)
del data, angles, meta_train_sub

xgbr = Pipeline(
    [
        ("scaler", FastMinMaxScaler()),
        (
            "regressor",
            XGBRegressor(
                tree_method="hist",
                n_estimators=64,
                num_target=y_train.shape[1],
                verbosity=2,  # keep as-is
                n_jobs=-1,
                random_state=42,
                predictor="cpu_predictor",
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
print("Skipping unused median computation to save time.")



## === cell 11
test_files = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet")))
print(f"nb test batches: {len(test_files)}")



## === cell 12
TWO_PI = np.float32(2.0 * np.pi)
PI = np.float32(np.pi)

meta_test = pd.read_parquet(
    os.path.join(PATH_DATASET, "test_meta.parquet"),
    columns=["event_id"],
    engine="pyarrow",
)
event_order = meta_test["event_id"].to_numpy(dtype=np.int64, copy=False)
del meta_test
n_total = event_order.shape[0]

az_out = np.empty(n_total, dtype=np.float32)
ze_out = np.empty(n_total, dtype=np.float32)

pos = 0
for batch_file in tqdm(test_files, desc="Inference over test batches"):
    event_ids_b, data_b = transform_batch(batch_file)
    preds_b = xgbr.predict(data_b)  # shape (n_events, 2)

    idx = np.searchsorted(event_order, event_ids_b)
    if not np.all(event_order[idx] == event_ids_b):
        raise ValueError(
            "Event ID alignment failed: some event_ids_b not found in event_order."
        )

    az = preds_b[:, 0].astype(np.float32, copy=False)
    ze = preds_b[:, 1].astype(np.float32, copy=False)

    az = np.mod(az, TWO_PI)  # [0, 2*pi)
    ze = np.clip(ze, 0.0, PI)  # [0, pi]

    az_out[idx] = az
    ze_out[idx] = ze

    pos += event_ids_b.shape[0]
    del event_ids_b, data_b, preds_b, idx, az, ze

print("filled rows:", pos, "of", n_total)

sub = pd.DataFrame({"event_id": event_order, "azimuth": az_out, "zenith": ze_out})
sub.to_csv("submission.csv", index=False)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())

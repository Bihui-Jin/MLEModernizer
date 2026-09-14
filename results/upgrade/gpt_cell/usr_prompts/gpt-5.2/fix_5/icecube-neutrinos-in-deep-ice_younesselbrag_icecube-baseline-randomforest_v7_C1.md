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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Target score

1.568461

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob, math, gc, time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
meta_data_train = pd.read_parquet(os.path.join(PATH_DATASET, "train_meta.parquet"))
print(f"total events: {len(meta_data_train)}")
print(f"nb batches: {meta_data_train['batch_id'].nunique()}")



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass




## === cell 11
def charge_center(event):
    evt_sub = event[~event["auxiliary"]][["x", "y", "z", "charge"]]
    evt_sub.loc[:, "coef"] = evt_sub["charge"] / evt_sub["charge"].sum()
    for c in ["x", "y", "z"]:
        evt_sub.loc[:, c] *= evt_sub["coef"]
    cx, cy, cz = evt_sub[["x", "y", "z"]].sum().values
    return cx, cy, cz


def direction(meta):
    azimuth = meta["azimuth"]
    zenith = meta["zenith"]
    dx = math.sin(zenith) * math.cos(azimuth)
    dy = math.sin(zenith) * math.sin(azimuth)
    dz = math.cos(zenith)
    return dx, dy, dz




## === cell 12
pass



## === cell 13
pass




## === cell 14
def transform_batch(path_batch_parquet, nb_sensors=5160, limit_rows=1000000):
    df = pd.read_parquet(path_batch_parquet)
    if limit_rows is not None:
        df = df.iloc[:limit_rows]

    sensor = df["sensor_id"].to_numpy(dtype=np.int32, copy=False)
    charge = df["charge"].to_numpy(dtype=np.float16, copy=False)

    event = df.index.to_numpy(copy=False)

    unique_event_ids, inv = np.unique(event, return_inverse=True)
    n_events = unique_event_ids.shape[0]

    data = np.zeros((n_events, nb_sensors), dtype=np.float16)

    order = np.arange(inv.size, dtype=np.int64)
    data[inv[order], sensor[order]] = charge[order]

    return unique_event_ids.tolist(), data




## === cell 15
event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_10.parquet"))

meta_train_ = meta_data_train.loc[
    meta_data_train["event_id"].isin(event_ids), ["event_id", "azimuth", "zenith"]
]
meta_train_ = meta_train_.set_index("event_id").loc[
    event_ids
]  # preserve event_ids order
angles = meta_train_[["azimuth", "zenith"]].to_numpy(dtype=np.float16, copy=False)

print(f"data size: {data.shape}")
print(f"angles size: {angles.shape}")



## === cell 16
X_train, X_test, y_train, y_test = train_test_split(
    data, angles, train_size=0.8, random_state=RANDOM_STATE, shuffle=True
)
del data, angles
gc.collect()



## === cell 17
preprocess = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("PCA", PCA(n_components=510, copy=False, random_state=RANDOM_STATE)),
    ]
)

X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)



## === cell 18
estimator = RandomForestRegressor(
    n_estimators=250,
    max_features="auto",
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=3,
    bootstrap=True,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

MultiOutputRegress = MultiOutputRegressor(estimator, n_jobs=-1)

MultiOutputRegress.fit(X_train, y_train)

y_pred = MultiOutputRegress.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean squared error: {:.6f}".format(mse))
print("Mean absolute error: {:.6f}".format(mae))



## === cell 19
del X_train, y_train, X_test, y_test, y_pred
gc.collect()



## === cell 20
gc.collect()



## === cell 21
test_files = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet")))
if not test_files:
    raise FileNotFoundError(
        f"No test batch parquet files found under: {os.path.join(PATH_DATASET, 'test')}"
    )

out_path = "submission.csv"
if os.path.exists(out_path):
    os.remove(out_path)



## === cell 22
default_zenith = float(meta_data_train["zenith"].mean())
default_azimuth = 0.0  # original initialized from sample_submission (likely 0); will be overwritten for predicted ids.



## === cell 23
header_written = False
all_parts = (
    []
)  # keep small list of per-batch outputs for a final sort without holding full 13.2M sample submission

for batch_file in test_files:
    print(f"processing: {os.path.basename(batch_file)}")
    event_ids, data = transform_batch(batch_file)
    Xb = preprocess.transform(data)
    preds = MultiOutputRegress.predict(Xb)

    part = pd.DataFrame(
        {
            "event_id": np.asarray(event_ids, dtype=np.int64),
            "azimuth": preds[:, 0].astype(np.float32, copy=False),
            "zenith": preds[:, 1].astype(np.float32, copy=False),
        }
    )
    all_parts.append(part)

    del event_ids, data, Xb, preds, part
    gc.collect()

pred_df = pd.concat(all_parts, ignore_index=True)
del all_parts
pred_df.sort_values("event_id", inplace=True, kind="mergesort")
pred_df.to_csv(out_path, index=False)



## === cell 24
print(
    f"Wrote {out_path} with {sum(1 for _ in open(out_path)) - 1} rows (excluding header)."
)



## === cell 25
ssubs = pd.read_csv("submission.csv", nrows=5)
print(ssubs.head())

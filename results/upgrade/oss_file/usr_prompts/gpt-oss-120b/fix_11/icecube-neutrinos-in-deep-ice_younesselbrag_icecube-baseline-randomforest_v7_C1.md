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

# 5. Code solution

## === cell 0
import os
import glob
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc
from joblib import Parallel, delayed

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"



## === cell 1
meta_data_train = pd.read_parquet(os.path.join(PATH_DATASET, "train_meta.parquet"))
meta_data_test = pd.read_parquet(os.path.join(PATH_DATASET, "test_meta.parquet"))
print(f"Train events: {len(meta_data_train)}, Test events: {len(meta_data_test)}")



## === cell 2
sensor_geom = pd.read_csv(os.path.join(PATH_DATASET, "sensor_geometry.csv"))
print(f"Sensor geometry rows: {len(sensor_geom)}")




## === cell 3
def transform_batch(path_batch_parquet, nb_sensors=5160, max_rows=None):
    """
    Fast construction of the (events × sensors) charge matrix.
    Uses NumPy's in‑place addition and float32 for speed.
    Returns (event_ids, matrix) where event_ids are sorted unique IDs.
    """
    df = pd.read_parquet(path_batch_parquet, columns=["sensor_id", "charge"])
    if max_rows:
        df = df.iloc[:max_rows]

    row_idx, uniq_event_ids = pd.factorize(df.index, sort=True)
    n_events = len(uniq_event_ids)

    mat = np.zeros((n_events, nb_sensors), dtype=np.float32)

    cols = df["sensor_id"].values.astype(np.int64)
    vals = df["charge"].values.astype(np.float32)

    np.add.at(mat, (row_idx, cols), vals)

    return uniq_event_ids.astype(np.int64).values, mat




## === cell 4
train_batch_path = os.path.join(PATH_DATASET, "train", "batch_10.parquet")
event_ids_train, X_batch = transform_batch(train_batch_path)

train_subset = meta_data_train[meta_data_train["event_id"].isin(event_ids_train)]
train_dict = dict(
    zip(
        train_subset["event_id"].values,
        train_subset[["azimuth", "zenith"]].values.tolist(),
    )
)
y_batch = np.array([train_dict[eid] for eid in event_ids_train], dtype=np.float32)

print(f"Training matrix shape: {X_batch.shape}, Targets shape: {y_batch.shape}")

preprocess = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=510, copy=False, random_state=42)),
    ]
)

X_processed = preprocess.fit_transform(X_batch)

_scaler = preprocess.named_steps["scaler"]
_pca = preprocess.named_steps["pca"]


def numpy_preprocess_transform(X):
    """
    Apply the fitted MinMaxScaler and PCA to raw feature matrix X.
    Handles constant‑feature columns (scale == 0) to avoid NaNs.
    """
    scale_safe = np.where(_scaler.scale_ == 0, 1, _scaler.scale_)
    X_scaled = (X - _scaler.data_min_) / scale_safe
    X_centered = X_scaled - _pca.mean_
    return np.dot(X_centered, _pca.components_.T)


estimator = RandomForestRegressor(
    n_estimators=250,
    max_features="sqrt",
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=3,
    bootstrap=True,
    random_state=42,
    n_jobs=5,
)
multi_reg = MultiOutputRegressor(estimator)
multi_reg.fit(X_processed, y_batch)

del X_batch, y_batch, train_subset, train_dict, X_processed, preprocess
gc.collect()



## === cell 5
ssub = None

test_batch_files = sorted(glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet")))


def process_one_batch(batch_file):
    ev_ids, batch_data = transform_batch(batch_file)
    batch_feats = numpy_preprocess_transform(batch_data)
    preds = multi_reg.predict(batch_feats)
    preds = np.nan_to_num(preds, nan=0.0)
    batch_df = pd.DataFrame(
        preds,
        index=ev_ids,
        columns=["azimuth", "zenith"],
        dtype=np.float32,
    )
    return batch_df


batch_pred_dfs = Parallel(n_jobs=5, backend="loky")(
    delayed(process_one_batch)(bf) for bf in tqdm(test_batch_files, desc="Test batches")
)

ssub = pd.concat(batch_pred_dfs).reindex(meta_data_test["event_id"].values)



## === cell 6
global_az = meta_data_train["azimuth"].mean()
global_zn = meta_data_train["zenith"].mean()
ssub["azimuth"].fillna(global_az, inplace=True)
ssub["zenith"].fillna(global_zn, inplace=True)



## === cell 7
submission_path = "submission.csv"
ssub.reset_index().rename(columns={"index": "event_id"}).to_csv(
    submission_path, index=False
)
print(f"Submission written to {submission_path}")



## === cell 8
check = pd.read_csv(submission_path, nrows=5)
print(check.head())

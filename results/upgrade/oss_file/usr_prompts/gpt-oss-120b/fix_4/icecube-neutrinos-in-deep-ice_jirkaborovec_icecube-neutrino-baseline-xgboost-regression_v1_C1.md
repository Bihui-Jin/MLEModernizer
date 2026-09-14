# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

1.568381

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from xgboost import XGBRegressor

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"

META_COLS = ["event_id", "batch_id", "azimuth", "zenith"]
meta_train = pd.read_parquet(
    os.path.join(PATH_DATASET, "train_meta.parquet"), columns=META_COLS
)
print(f"total events: {len(meta_train)}")
print(f"nb batches: {len(meta_train['batch_id'].unique())}")




## === cell 1
def transform_batch(path_batch_parquet, nb_sensors=5160):
    """
    Reads a batch parquet and returns:
    - list of event_ids (preserving order of first appearance)
    - 2‑D numpy array [n_events, nb_sensors] with summed charge per sensor
    This implementation uses NumPy indexed‑add instead of a Python loop,
    yielding a large speedup while producing exactly the same result.
    """
    df = pd.read_parquet(
        path_batch_parquet, columns=["event_id", "sensor_id", "charge"]
    )
    event_ids_unique = pd.unique(df["event_id"])
    event_idx = pd.Categorical(
        df["event_id"], categories=event_ids_unique, ordered=False
    ).codes
    data = np.zeros((len(event_ids_unique), nb_sensors), dtype=np.float32)
    np.add.at(
        data,
        (event_idx, df["sensor_id"].values),
        df["charge"].values.astype(np.float32),
    )
    return list(event_ids_unique), data




## === cell 2
train_batch_files = sorted(
    glob.glob(os.path.join(PATH_DATASET, "train", "batch_*.parquet"))
)
N_TRAIN_BATCHES = 20  # adjust if more time/memory is available
X_parts = []
y_parts = []

print(f"Building training data from first {N_TRAIN_BATCHES} batches...")
for batch_file in tqdm(train_batch_files[:N_TRAIN_BATCHES], desc="Train batches"):
    event_ids_batch, data_batch = transform_batch(
        batch_file
    )  # data_batch shape (n_events, 5160)
    batch_targets = (
        meta_train.set_index("event_id")
        .loc[event_ids_batch, ["azimuth", "zenith"]]
        .values
    )
    X_parts.append(data_batch)
    y_parts.append(batch_targets.astype(np.float32))
    del event_ids_batch, data_batch, batch_targets

X_train = np.vstack(X_parts)
y_train = np.vstack(y_parts)
del X_parts, y_parts

print(f"Training matrix shape: {X_train.shape}, target shape: {y_train.shape}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'event_id'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/567211233.py in <cell line: 0>()
     10 print(f"Building training data from first {N_TRAIN_BATCHES} batches...")
     11 for batch_file in tqdm(train_batch_files[:N_TRAIN_BATCHES], desc="Train batches"):
---> 12     event_ids_batch, data_batch = transform_batch(
     13         batch_file
     14     )  # data_batch shape (n_events, 5160)

/tmp/ipykernel_11/1615681138.py in transform_batch(path_batch_parquet, nb_sensors)
     10         path_batch_parquet, columns=["event_id", "sensor_id", "charge"]
     11     )
---> 12     event_ids_unique = pd.unique(df["event_id"])
     13     # map each row to the index of its event in the unique list
     14     event_idx = pd.Categorical(

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'event_id'

## === cell 3
xgbr = XGBRegressor(
    n_estimators=120,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=5,
    random_state=42,
    reg_lambda=1.0,
    verbosity=0,
)

print("Fitting XGBoost model...")
xgbr.fit(X_train, y_train)
print("Model training completed.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4218112067.py in <cell line: 0>()
     14 
     15 print("Fitting XGBoost model...")
---> 16 xgbr.fit(X_train, y_train)
     17 print("Model training completed.")
     18 

NameError: name 'X_train' is not defined

## === cell 4
sample_sub_path = os.path.join(PATH_DATASET, "sample_submission.csv")
submission_df = pd.read_csv(sample_sub_path)
submission_df.set_index("event_id", inplace=True)

test_batch_files = sorted(
    glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet"))
)
print(f"Found {len(test_batch_files)} test batch files. Generating predictions...")

for batch_file in tqdm(test_batch_files, desc="Test batches"):
    event_ids_batch, data_batch = transform_batch(batch_file)
    preds = xgbr.predict(data_batch)  # shape (n_events, 2)
    submission_df.loc[event_ids_batch, ["azimuth", "zenith"]] = preds
    del event_ids_batch, data_batch, preds  # free memory each iteration

submission_df.fillna(0, inplace=True)

output_path = "submission.csv"
submission_df.reset_index().to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'event_id'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1052009536.py in <cell line: 0>()
     11 
     12 for batch_file in tqdm(test_batch_files, desc="Test batches"):
---> 13     event_ids_batch, data_batch = transform_batch(batch_file)
     14     preds = xgbr.predict(data_batch)  # shape (n_events, 2)
     15     # assign predictions directly into the submission dataframe

/tmp/ipykernel_11/1615681138.py in transform_batch(path_batch_parquet, nb_sensors)
     10         path_batch_parquet, columns=["event_id", "sensor_id", "charge"]
     11     )
---> 12     event_ids_unique = pd.unique(df["event_id"])
     13     # map each row to the index of its event in the unique list
     14     event_idx = pd.Categorical(

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'event_id'

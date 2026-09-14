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

1.571286

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error




## === cell 1
batch_id = 100
home_dir = "/kaggle/input/icecube-neutrinos-in-deep-ice/"
train_file = f"train/batch_{batch_id}.parquet"
train_meta = "train_meta.parquet"
test_file = f"test/batch_104.parquet"
test_meta = "test_meta.parquet"




## === cell 2
def load_train_data(path):
    df = pd.read_parquet(path, engine="pyarrow", use_threads=True)
    df.insert(0, df.index.name, df.index, True)
    df = df.reset_index(drop=True)
    return df


path = os.path.join(home_dir, train_file)
df_train_data = load_train_data(path)
display(df_train_data.head())




## === cell 3
def load_train_meta(path, batch_id):
    needed_cols = [
        "batch_id",
        "event_id",
        "first_pulse_index",
        "last_pulse_index",
        "azimuth",
        "zenith",
    ]
    df = pd.read_parquet(
        path,
        engine="pyarrow",
        columns=needed_cols,
        filters=[("batch_id", "=", batch_id)],
        use_threads=True,
    )
    return df


path = os.path.join(home_dir, train_meta)
df_train_meta = load_train_meta(path, batch_id=batch_id)
display(df_train_meta.head())




## === cell 4
csv_path = "sensor_geometry.csv"
geom_file_path = os.path.join(home_dir, csv_path)
df_sen_geom = pd.read_csv(geom_file_path)
df_sen_geom.set_index("sensor_id", inplace=True)

max_sensor_id = df_sen_geom.index.max()
geom_array = np.zeros((max_sensor_id + 1, 3), dtype=np.float32)
geom_array[df_sen_geom.index.values] = df_sen_geom[["x", "y", "z"]].values.astype(
    np.float32
)

df_sen_geom.head()




## === cell 5
def preprocess_train_data(
    df_train_data, df_train_meta_data, num_samples, number_of_sensors
):
    train_vals = df_train_data[["sensor_id", "time", "charge", "auxiliary"]].to_numpy()
    first_idx = df_train_meta_data["first_pulse_index"].values
    last_idx = df_train_meta_data["last_pulse_index"].values
    azimuth = df_train_meta_data["azimuth"].values
    zenith = df_train_meta_data["zenith"].values

    train_data = []
    targets = np.empty((num_samples, 2), dtype=np.float32)

    for i in range(num_samples):
        targets[i] = np.array([azimuth[i], zenith[i]], dtype=np.float32)
        start = first_idx[i]
        end = last_idx[i]
        slice_end = (
            start + number_of_sensors if (end - start) >= number_of_sensors else end
        )
        train_data.append(train_vals[start:slice_end])
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} samples")
    return train_data, targets


num_samples = 2000  # keep small for quick run
number_of_sensors = 5
train_data, targets = preprocess_train_data(
    df_train_data,
    df_train_meta,
    num_samples=num_samples,
    number_of_sensors=number_of_sensors,
)




## === cell 6
def vectorize(data, geom_array, number_of_sensors):
    results = np.empty((len(data), number_of_sensors, 6), dtype=np.float32)
    for i, pulse_arr in enumerate(data):
        sensor_ids = pulse_arr[:, 0].astype(int)
        results[i, :, 0:3] = geom_array[sensor_ids]
        results[i, :, 3] = pulse_arr[:, 1].astype(np.float32)  # time
        results[i, :, 4] = pulse_arr[:, 2].astype(np.float32)  # charge
        results[i, :, 5] = pulse_arr[:, 3].astype(np.float32)  # auxiliary
    return results


samples = vectorize(train_data, geom_array, number_of_sensors=number_of_sensors)




## === cell 7
num_train = int(0.7 * num_samples)
num_val = int(0.25 * num_samples)
num_test = num_samples - num_train - num_val

train_samples = samples[:num_train].astype("float32")
val_samples = samples[num_train : num_train + num_val].astype("float32")
test_samples = samples[num_train + num_val :].astype("float32")

train_targets = targets[:num_train]
val_targets = targets[num_train : num_train + num_val]
test_targets = targets[num_train + num_val :]

del df_train_data, df_train_meta, train_data, targets
gc.collect()




## === cell 8
train_X = train_samples.reshape(train_samples.shape[0], -1)
val_X = val_samples.reshape(val_samples.shape[0], -1)
test_X = test_samples.reshape(test_samples.shape[0], -1)

mlp = MLPRegressor(
    hidden_layer_sizes=(4,),
    activation="relu",
    solver="adam",
    learning_rate_init=1e-5,
    max_iter=200,
    random_state=42,
    verbose=False,
)

mlp.fit(train_X, train_targets)

val_pred = mlp.predict(val_X)
test_pred = mlp.predict(test_X)

print("Validation MAE:", mean_absolute_error(val_targets, val_pred))
print("Test MAE:", mean_absolute_error(test_targets, test_pred))
print("Sample prediction vs true:", val_pred[0], val_targets[0])




## === cell 9
def load_test_data(path):
    df = pd.read_parquet(path, engine="pyarrow", use_threads=True)
    df.insert(0, df.index.name, df.index, True)
    df = df.reset_index(drop=True)
    return df


path = os.path.join(home_dir, test_file)
df_test_data = load_test_data(path)
display(df_test_data.head())




## === cell 10
def load_test_meta(path):
    return pd.read_parquet(path, engine="pyarrow", use_threads=True)


df_test_meta = load_test_meta(os.path.join(home_dir, test_meta))
display(df_test_meta.head())




## === cell 11
def preprocess_test_data(df_test_data, df_test_meta, number_of_sensors):
    test_vals = df_test_data[["sensor_id", "time", "charge", "auxiliary"]].to_numpy()
    event_ids = df_test_meta["event_id"].values
    first_idx = df_test_meta["first_pulse_index"].values
    last_idx = df_test_meta["last_pulse_index"].values

    test_data = []
    for i in range(event_ids.shape[0]):
        start = first_idx[i]
        end = last_idx[i]
        slice_end = (
            start + number_of_sensors if (end - start) >= number_of_sensors else end
        )
        test_data.append(test_vals[start:slice_end])
    return test_data, event_ids


test_data, test_event_ids = preprocess_test_data(
    df_test_data, df_test_meta, number_of_sensors=number_of_sensors
)

test_vectors = vectorize(test_data, geom_array, number_of_sensors=number_of_sensors)
test_X_full = test_vectors.reshape(test_vectors.shape[0], -1)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_57/1048244669.py in <cell line: 0>()
     22 )
     23 
---> 24 test_vectors = vectorize(test_data, geom_array, number_of_sensors=number_of_sensors)
     25 test_X_full = test_vectors.reshape(test_vectors.shape[0], -1)
     26 

/tmp/ipykernel_57/3571249212.py in vectorize(data, geom_array, number_of_sensors)
      5         # pulse_arr is a NumPy array with shape (<=number_of_sensors, 4)
      6         sensor_ids = pulse_arr[:, 0].astype(int)
----> 7         results[i, :, 0:3] = geom_array[sensor_ids]
      8         results[i, :, 3] = pulse_arr[:, 1].astype(np.float32)  # time
      9         results[i, :, 4] = pulse_arr[:, 2].astype(np.float32)  # charge

ValueError: could not broadcast input array from shape (0,3) into shape (5,3)

## === cell 12
test_predictions = mlp.predict(test_X_full)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/2099868961.py in <cell line: 0>()
----> 1 test_predictions = mlp.predict(test_X_full)
      2 
      3 

NameError: name 'test_X_full' is not defined

## === cell 13
submission_df = pd.DataFrame(
    {
        "event_id": test_event_ids,
        "azimuth": test_predictions[:, 0],
        "zenith": test_predictions[:, 1],
    }
)
submission_df = submission_df.sort_values("event_id")
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
submission_df.head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/1631732318.py in <cell line: 0>()
      2     {
      3         "event_id": test_event_ids,
----> 4         "azimuth": test_predictions[:, 0],
      5         "zenith": test_predictions[:, 1],
      6     }

NameError: name 'test_predictions' is not defined

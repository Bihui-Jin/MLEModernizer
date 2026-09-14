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

# 5. Code solution

## === cell 0
import os
import gc
import time

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)
print("Pandas:", pd.__version__)



## === cell 1
batch_id = 100
home_dir = "/kaggle/input/icecube-neutrinos-in-deep-ice/"
train_file = f"train/batch_{batch_id}.parquet"
train_meta = "train_meta.parquet"
test_file = None
test_meta = "test_meta.parquet"




## === cell 2
def load_train_data(path):
    df = pd.read_parquet(path, engine="pyarrow")
    df.insert(0, df.index.name, df.index, True)
    df = df.reset_index(drop=True)
    return df


path = os.path.join(home_dir, train_file)
df_train_data = load_train_data(path)
print(df_train_data.head())
print(df_train_data.shape)




## === cell 3
def load_train_meta(path, batch_id):
    df = pd.read_parquet(path, engine="pyarrow")
    df = df[df["batch_id"] == batch_id].reset_index(drop=True)
    return df


path = os.path.join(home_dir, train_meta)
df_train_meta = load_train_meta(path, batch_id=batch_id)
print(df_train_meta.head())
print(df_train_meta.shape)



## === cell 4
csv_path = "sensor_geometry.csv"
geom_file_path = os.path.join(home_dir, csv_path)
df_sen_geom = pd.read_csv(geom_file_path)

if "sensor_id" in df_sen_geom.columns:
    df_sen_geom = df_sen_geom.set_index("sensor_id", drop=False)

df_sen_geom = df_sen_geom.rename(
    columns={"x": "x-dimension [m]", "y": "y-dimension [m]", "z": "z-dimension [m]"}
)
print(df_sen_geom.head())
print(df_sen_geom.shape)




## === cell 5
def preprocess_train_data(
    df_train_data, df_train_meta_data, num_samples, number_of_sensors
):
    train_data = []
    targets = []
    n = min(num_samples, len(df_train_meta_data))
    for i in range(n):
        target = df_train_meta_data.loc[
            df_train_meta_data.index[i], ["azimuth", "zenith"]
        ]
        start_index = df_train_meta_data.loc[
            df_train_meta_data.index[i], "first_pulse_index"
        ]
        end_index = df_train_meta_data.loc[
            df_train_meta_data.index[i], "last_pulse_index"
        ]
        if (end_index - start_index) < number_of_sensors:
            df1 = df_train_data.iloc[start_index:end_index]
        else:
            df1 = df_train_data.iloc[start_index : (start_index + number_of_sensors)]
        data = df1[["sensor_id", "time", "charge", "auxiliary"]]
        data = list(data.values)
        train_data.append(data)
        targets.append(list(target))
        if i % 1000 == 0:
            print(f"Number of samples : {i+1000}")
    return train_data, targets


num_samples = 2000  # Total number of samples
number_of_sensors = 5  # number of sensors at each sample
train_data, targets = preprocess_train_data(
    df_train_data,
    df_train_meta,
    num_samples=num_samples,
    number_of_sensors=number_of_sensors,
)




## === cell 6
def vectorize(data, df_sen_geom, number_of_sensors):
    results = np.zeros((len(data), number_of_sensors, 6), dtype=np.float32)
    for i, data_list in enumerate(data):
        if len(data_list) < number_of_sensors:
            pad = [[0, 0, 0.0, False]] * (number_of_sensors - len(data_list))
            data_list = list(data_list) + pad
        elif len(data_list) > number_of_sensors:
            data_list = list(data_list)[:number_of_sensors]

        for j, row in enumerate(data_list):
            sensor_id = int(row[0])
            if sensor_id in df_sen_geom.index:
                results[i, j, 0] = df_sen_geom.loc[sensor_id, "x-dimension [m]"]
                results[i, j, 1] = df_sen_geom.loc[sensor_id, "y-dimension [m]"]
                results[i, j, 2] = df_sen_geom.loc[sensor_id, "z-dimension [m]"]
            else:
                results[i, j, 0] = 0.0
                results[i, j, 1] = 0.0
                results[i, j, 2] = 0.0

            results[i, j, 3] = float(row[1])
            results[i, j, 4] = float(row[2])
            aux = bool(row[3])
            results[i, j, 5] = 1.0 if aux else 0.0
    return results


samples = vectorize(train_data, df_sen_geom, number_of_sensors=number_of_sensors)

num_train_samples = int(0.7 * num_samples)
num_val_samples = int(0.25 * num_samples)
num_test_samples = num_samples - num_train_samples - num_val_samples

print("num of train samples:", num_train_samples)
print("num of val samples:", num_val_samples)
print("num of test samples:", num_test_samples)

train_samples = samples[:num_train_samples].astype("float32")
train_targets = targets[:num_train_samples]

val_samples = samples[num_train_samples : num_train_samples + num_val_samples].astype(
    "float32"
)
val_targets = targets[num_train_samples : num_train_samples + num_val_samples]

test_samples = samples[num_train_samples + num_val_samples :].astype("float32")
test_targets = targets[num_train_samples + num_val_samples :]

del df_train_data, train_data, targets
gc.collect()



## === cell 7
import matplotlib.pyplot as plt

plt.scatter(range(samples.shape[0]), samples[:, 0, 5])



## === cell 8
train_targets = np.asarray(train_targets).astype("float32")
val_targets = np.asarray(val_targets).astype("float32")
test_targets = np.asarray(test_targets).astype("float32")



## === cell 9
tf.keras.backend.clear_session()


def model_build_and_train(
    train_samples, train_targets, val_samples, val_targets, number_of_sensors
):
    inputs = keras.Input(shape=(number_of_sensors, 6))
    x = layers.Flatten()(inputs)
    x = layers.Dense(4, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath="checkpoint_path.keras",
            monitor="val_loss",
            save_best_only=True,
        )
    ]
    model.compile(optimizer=keras.optimizers.RMSprop(1e-5), loss="mse", metrics=["mae"])

    history = model.fit(
        train_samples,
        train_targets,
        validation_data=(val_samples, val_targets),
        callbacks=callbacks_list,
        epochs=20,
        batch_size=64,
        verbose=1,
    )
    return model, history


model, history = model_build_and_train(
    train_samples,
    train_targets,
    val_samples,
    val_targets,
    number_of_sensors=number_of_sensors,
)



## === cell 10
model = keras.models.load_model("checkpoint_path.keras")
print("Model performance on Validation samples: ")
model.evaluate(val_samples, val_targets, verbose=1)
print("Model performance on Test samples: ")
model.evaluate(test_samples, test_targets, verbose=1)
print("Prediction vs True value")
print(
    f"prediction:{model.predict(test_samples, verbose=0)[0]}, True value: {test_targets[0]}"
)
print(model.predict(test_samples, verbose=0)[0:10])



## === cell 11
import matplotlib.pyplot as plt

plt.figure(figsize=(25, 6))
plt.suptitle("Network Performance", fontsize=30)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
history_dict = history.history
keys = ["loss", "mae", "val_loss", "val_mae"]

for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.ylabel(keys[i])
    plt.xlabel("Epochs")
    plt.plot(
        range(1, len(history_dict[keys[i]]) + 1),
        history_dict[keys[i]],
        "bo",
        label="Training",
    )
    plt.plot(
        range(1, len(history_dict[keys[i + 2]]) + 1),
        history_dict[keys[i + 2]],
        "b",
        label="Validation",
    )
    plt.legend()




## === cell 12
def load_test_meta(path):
    df = pd.read_parquet(path, engine="pyarrow")
    return df


path = os.path.join(home_dir, test_meta)
df_test_meta_all = load_test_meta(path)
print(df_test_meta_all.head())
print("Unique test batches:", df_test_meta_all["batch_id"].nunique())




## === cell 13
def load_test_data(path):
    df = pd.read_parquet(path, engine="pyarrow")
    df.insert(0, df.index.name, df.index, True)
    df = df.reset_index(drop=True)
    return df


def preprocess_test_data(df_test_data, df_test_meta, number_of_sensors):
    event_ids = df_test_meta["event_id"].values
    test_data = []
    for i in range(event_ids.shape[0]):
        start_index = df_test_meta.loc[df_test_meta.index[i], "first_pulse_index"]
        end_index = df_test_meta.loc[df_test_meta.index[i], "last_pulse_index"]
        if (end_index - start_index) < number_of_sensors:
            df1 = df_test_data.iloc[start_index:end_index]
        else:
            df1 = df_test_data.iloc[start_index : (start_index + number_of_sensors)]
        data = df1[["sensor_id", "time", "charge", "auxiliary"]]
        data = list(data.values)
        test_data.append(data)
    return test_data, event_ids


all_results = []
batch_ids = df_test_meta_all["batch_id"].unique()
batch_ids.sort()

t0 = time.time()
for k, test_batch_id in enumerate(batch_ids, start=1):
    test_file = f"test/batch_{int(test_batch_id)}.parquet"
    df_test_meta = df_test_meta_all[
        df_test_meta_all["batch_id"] == test_batch_id
    ].reset_index(drop=True)

    path = os.path.join(home_dir, test_file)
    df_test_data = load_test_data(path)

    test_data, test_event_ids = preprocess_test_data(
        df_test_data, df_test_meta, number_of_sensors=number_of_sensors
    )
    test_vec = vectorize(test_data, df_sen_geom, number_of_sensors=number_of_sensors)

    test_predict = model.predict(test_vec, verbose=0)

    batch_df = pd.DataFrame(
        {
            "event_id": test_event_ids.astype(np.int64),
            "azimuth": test_predict[:, 0].astype(np.float32),
            "zenith": test_predict[:, 1].astype(np.float32),
        }
    )
    all_results.append(batch_df)

    del df_test_data, df_test_meta, test_data, test_vec, test_predict, batch_df
    gc.collect()

    if k % 5 == 0 or k == len(batch_ids):
        print(f"Processed {k}/{len(batch_ids)} test batches in {time.time() - t0:.1f}s")

del df_test_meta_all
gc.collect()

test_result_df = pd.concat(all_results, axis=0, ignore_index=True)
print("Total predicted rows:", len(test_result_df))
print(test_result_df.head())



## === cell 14
sample_path = os.path.join(home_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path, usecols=["event_id"])
sample_sub["event_id"] = sample_sub["event_id"].astype(np.int64)

test_result_df["event_id"] = test_result_df["event_id"].astype(np.int64)

submission = sample_sub.merge(test_result_df, on="event_id", how="left")

submission["azimuth"] = submission["azimuth"].astype(np.float32).fillna(0.0)
submission["zenith"] = submission["zenith"].astype(np.float32).fillna(0.0)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print("Missing preds:", submission[["azimuth", "zenith"]].isna().sum().to_dict())

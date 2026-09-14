# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.11

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
from pyarrow.parquet import ParquetFile
import pyarrow as pa

import os
import importlib

try:
    import google.protobuf as _pb

    _pb_version = getattr(_pb, "__version__", "0")
    _pb_major = int(_pb_version.split(".", 1)[0]) if _pb_version else 0
except Exception:
    _pb_major = 0

if _pb_major >= 5:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    importlib.invalidate_caches()

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import gc
import time

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

np.random.seed(42)
tf.random.set_seed(42)

try:
    import memory_profiler  # type: ignore
except ModuleNotFoundError:
    memory_profiler = None




## === cell 1
batch_id = 100
home_dir = "/kaggle/input/icecube-neutrinos-in-deep-ice/"
train_file = f"train/batch_{batch_id}.parquet"
train_meta = "train_meta.parquet"
test_file = "test/batch_661.parquet"
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
def load_train_meta_filtered(path, batch_id):
    pf = ParquetFile(path)
    cols = [
        "batch_id",
        "event_id",
        "first_pulse_index",
        "last_pulse_index",
        "azimuth",
        "zenith",
    ]
    out_batches = []
    for rb in pf.iter_batches(columns=cols, batch_size=1_000_000):
        t = pa.Table.from_batches([rb])
        bcol = t.column("batch_id").to_numpy(zero_copy_only=False)
        mask = bcol == batch_id
        if mask.any():
            out_batches.append(t.filter(pa.array(mask)))
    if not out_batches:
        return pd.DataFrame(columns=cols)
    return pa.concat_tables(out_batches, promote=True).to_pandas()


path = os.path.join(home_dir, train_meta)
df_train_meta = load_train_meta_filtered(path, batch_id=batch_id)
display(df_train_meta.head())




## === cell 4
csv_path = "sensor_geometry.csv"
geom_file_path = os.path.join(home_dir, csv_path)
df_sen_geom = pd.read_csv(geom_file_path)
df_sen_geom = df_sen_geom.rename(
    columns={"x": "x-dimension [m]", "y": "y-dimension [m]", "z": "z-dimension [m]"}
)
df_sen_geom.head()




## === cell 5
def build_samples_and_targets_fast(
    df_pulses, df_meta, num_samples, number_of_sensors, geom_df
):
    meta = df_meta.iloc[:num_samples].reset_index(drop=True)

    targets = meta[["azimuth", "zenith"]].to_numpy(dtype=np.float32)

    first = meta["first_pulse_index"].to_numpy(dtype=np.int64)
    last = meta["last_pulse_index"].to_numpy(dtype=np.int64)
    lengths = (last - first).astype(np.int64)
    take = np.minimum(lengths, number_of_sensors)

    geom_sorted = geom_df.sort_values("sensor_id", kind="mergesort")
    max_sid = int(geom_sorted["sensor_id"].max())
    xg = np.empty(max_sid + 1, dtype=np.float32)
    yg = np.empty(max_sid + 1, dtype=np.float32)
    zg = np.empty(max_sid + 1, dtype=np.float32)
    sid_arr = geom_sorted["sensor_id"].to_numpy(dtype=np.int64)
    xg[sid_arr] = geom_sorted["x-dimension [m]"].to_numpy(dtype=np.float32)
    yg[sid_arr] = geom_sorted["y-dimension [m]"].to_numpy(dtype=np.float32)
    zg[sid_arr] = geom_sorted["z-dimension [m]"].to_numpy(dtype=np.float32)

    sensor_id = df_pulses["sensor_id"].to_numpy(dtype=np.int64, copy=False)
    timev = df_pulses["time"].to_numpy(dtype=np.float32, copy=False)
    charge = df_pulses["charge"].to_numpy(dtype=np.float32, copy=False)
    aux = df_pulses["auxiliary"].to_numpy(copy=False)
    auxf = aux.astype(
        np.float32, copy=False
    )  # bool->0/1 float, same meaning as original

    offsets = np.arange(number_of_sensors, dtype=np.int64)[None, :]
    idx = first[:, None] + offsets  # (num_samples, number_of_sensors)
    valid = offsets < take[:, None]

    samples = np.zeros((meta.shape[0], number_of_sensors, 6), dtype=np.float32)

    r, c = np.nonzero(valid)
    flat_idx = idx[r, c]

    sid = sensor_id[flat_idx]
    samples[r, c, 0] = xg[sid]
    samples[r, c, 1] = yg[sid]
    samples[r, c, 2] = zg[sid]
    samples[r, c, 3] = timev[flat_idx]
    samples[r, c, 4] = charge[flat_idx]
    samples[r, c, 5] = auxf[flat_idx]

    return samples, targets


num_samples = 2000  # Total number of samples
number_of_sensors = 5  # number of sensors at each sample

samples, targets = build_samples_and_targets_fast(
    df_train_data,
    df_train_meta,
    num_samples=num_samples,
    number_of_sensors=number_of_sensors,
    geom_df=df_sen_geom,
)

num_train_samples = int(0.7 * num_samples)
num_val_samples = int(0.25 * num_samples)
num_test_samples = num_samples - num_train_samples - num_val_samples

print("num of train samples:", num_train_samples)
print("num of val samples:", num_val_samples)
print("num of test samples:", num_test_samples)

train_samples = samples[:num_train_samples].astype("float32", copy=False)
train_targets = targets[:num_train_samples].astype("float32", copy=False)

val_samples = samples[num_train_samples : num_train_samples + num_val_samples].astype(
    "float32", copy=False
)
val_targets = targets[num_train_samples : num_train_samples + num_val_samples].astype(
    "float32", copy=False
)

test_samples = samples[num_train_samples + num_val_samples :].astype(
    "float32", copy=False
)
test_targets = targets[num_train_samples + num_val_samples :].astype(
    "float32", copy=False
)

del df_train_data, df_train_meta
gc.collect()




## === cell 6
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(range(samples.shape[0]), samples[:, 0, 5])




## === cell 7
train_targets = np.asarray(train_targets).astype("float32", copy=False)
val_targets = np.asarray(val_targets).astype("float32", copy=False)
test_targets = np.asarray(test_targets).astype("float32", copy=False)




## === cell 8
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




## === cell 9
model = keras.models.load_model("checkpoint_path.keras")
print("Model performance on Validation samples: ")
model.evaluate(val_samples, val_targets)
print("Model performance on Test samples: ")
model.evaluate(test_samples, test_targets)
print("Prediction vs True value")
print(f"prediction:{model.predict(test_samples)[0]}, True value: {test_targets[0]}")
model.predict(test_samples)[0:10]




## === cell 10
import matplotlib.pyplot as plt

plt.figure(figsize=(25, 6))
plt.suptitle("Network Performance", fontsize=30)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
history_dict = history.history
keys = ["loss", "mae", "val_loss", "val_mae"]
label = ["Training loss", "Validation loss", "Training mae", "Validation mae"]

for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.ylabel(keys[i])
    plt.xlabel("Epochs")
    plt.plot(
        range(1, len(history_dict[keys[i]]) + 1),
        history_dict[keys[i]],
        "bo",
        label="Training loss",
    )
    plt.plot(
        range(1, len(history_dict[keys[i + 2]]) + 1),
        history_dict[keys[i + 2]],
        "b",
        label="Validation loss",
    )
    plt.legend()




## === cell 11
def load_test_data(path):
    df = pd.read_parquet(path, engine="pyarrow", use_threads=True)
    df.insert(0, df.index.name, df.index, True)
    df = df.reset_index(drop=True)
    return df


path = os.path.join(home_dir, test_file)
if not os.path.exists(path):
    test_dir = os.path.join(home_dir, "test")
    candidates = []
    if os.path.isdir(test_dir):
        candidates = sorted(
            f
            for f in os.listdir(test_dir)
            if f.startswith("batch_") and f.endswith(".parquet")
        )
    if not candidates:
        raise FileNotFoundError(
            f"Requested test file not found: {path}. Also no batch_*.parquet files found in {test_dir}."
        )
    path = os.path.join(test_dir, candidates[0])

df_test_data = load_test_data(path)
display(df_test_data.head())




## === cell 12
def load_test_meta(path):
    df = pd.read_parquet(path, engine="pyarrow", use_threads=True)
    return df


path = os.path.join(home_dir, test_meta)
df_test_meta = load_test_meta(path)
display(df_test_meta.head())




## === cell 13
def build_test_samples_fast(df_pulses, df_meta, number_of_sensors, geom_df):
    meta = df_meta.reset_index(drop=True)
    event_ids = meta["event_id"].to_numpy(copy=False)

    first = meta["first_pulse_index"].to_numpy(dtype=np.int64)
    last = meta["last_pulse_index"].to_numpy(dtype=np.int64)
    lengths = (last - first).astype(np.int64)
    take = np.minimum(lengths, number_of_sensors)

    geom_sorted = geom_df.sort_values("sensor_id", kind="mergesort")
    max_sid = int(geom_sorted["sensor_id"].max())
    xg = np.empty(max_sid + 1, dtype=np.float32)
    yg = np.empty(max_sid + 1, dtype=np.float32)
    zg = np.empty(max_sid + 1, dtype=np.float32)
    sid_arr = geom_sorted["sensor_id"].to_numpy(dtype=np.int64)
    xg[sid_arr] = geom_sorted["x-dimension [m]"].to_numpy(dtype=np.float32)
    yg[sid_arr] = geom_sorted["y-dimension [m]"].to_numpy(dtype=np.float32)
    zg[sid_arr] = geom_sorted["z-dimension [m]"].to_numpy(dtype=np.float32)

    sensor_id = df_pulses["sensor_id"].to_numpy(dtype=np.int64, copy=False)
    timev = df_pulses["time"].to_numpy(dtype=np.float32, copy=False)
    charge = df_pulses["charge"].to_numpy(dtype=np.float32, copy=False)
    auxf = df_pulses["auxiliary"].to_numpy(copy=False).astype(np.float32, copy=False)

    offsets = np.arange(number_of_sensors, dtype=np.int64)[None, :]
    idx = first[:, None] + offsets
    valid = offsets < take[:, None]

    samples = np.zeros((meta.shape[0], number_of_sensors, 6), dtype=np.float32)
    r, c = np.nonzero(valid)
    flat_idx = idx[r, c]

    sid = sensor_id[flat_idx]
    samples[r, c, 0] = xg[sid]
    samples[r, c, 1] = yg[sid]
    samples[r, c, 2] = zg[sid]
    samples[r, c, 3] = timev[flat_idx]
    samples[r, c, 4] = charge[flat_idx]
    samples[r, c, 5] = auxf[flat_idx]

    return samples, event_ids


test, test_event_ids = build_test_samples_fast(
    df_test_data, df_test_meta, number_of_sensors=number_of_sensors, geom_df=df_sen_geom
)

del df_test_data, df_test_meta
gc.collect()

test_ptredict = model.predict(test, batch_size=4096, verbose=0)
test_ptredict




## --- ERROR in cell 13, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1423049284.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     43[0m [0;34m[0m[0m
[1;32m     44[0m [0;34m[0m[0m
[0;32m---> 45[0;31m test, test_event_ids = build_test_samples_fast(
[0m[1;32m     46[0m     [0mdf_test_data[0m[0;34m,[0m [0mdf_test_meta[0m[0;34m,[0m [0mnumber_of_sensors[0m[0;34m=[0m[0mnumber_of_sensors[0m[0;34m,[0m [0mgeom_df[0m[0;34m=[0m[0mdf_sen_geom[0m[0;34m[0m[0;34m[0m[0m
[1;32m     47[0m )

[0;32m/tmp/ipykernel_11/1423049284.py[0m in [0;36mbuild_test_samples_fast[0;34m(df_pulses, df_meta, number_of_sensors, geom_df)[0m
[1;32m     32[0m     [0mflat_idx[0m [0;34m=[0m [0midx[0m[0;34m[[0m[0mr[0m[0;34m,[0m [0mc[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     33[0m [0;34m[0m[0m
[0;32m---> 34[0;31m     [0msid[0m [0;34m=[0m [0msensor_id[0m[0;34m[[0m[0mflat_idx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     35[0m     [0msamples[0m[0;34m[[0m[0mr[0m[0;34m,[0m [0mc[0m[0;34m,[0m [0;36m0[0m[0;34m][0m [0;34m=[0m [0mxg[0m[0;34m[[0m[0msid[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     36[0m     [0msamples[0m[0;34m[[0m[0mr[0m[0;34m,[0m [0mc[0m[0;34m,[0m [0;36m1[0m[0;34m][0m [0;34m=[0m [0myg[0m[0;34m[[0m[0msid[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: index 33168996 is out of bounds for axis 0 with size 33168971

## === cell 14
test_event_id = list(test_event_ids)
test_azimuth = list(test_ptredict[:, 0])
test_zenith = list(test_ptredict[:, 1])

test_result_dict = {
    "event_id": test_event_id,
    "azimuth": test_azimuth,
    "zenith": test_zenith,
}

test_result_df = pd.DataFrame(test_result_dict)
test_result_df = test_result_df.sort_values(by=["event_id"])

test_result_df.to_csv("submission.csv", index=False)
test_result_df.head()

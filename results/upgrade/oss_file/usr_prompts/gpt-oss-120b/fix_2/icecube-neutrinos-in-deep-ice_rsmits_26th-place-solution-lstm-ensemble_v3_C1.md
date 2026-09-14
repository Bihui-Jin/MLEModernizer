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

0.9953686115789888

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from tqdm import tqdm

batch_size = 1024



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
home_dir = "/kaggle/input/icecube-neutrinos-in-deep-ice/"
train_format = home_dir + "train/batch_{batch_id:d}.parquet"
test_format = home_dir + "test/batch_{batch_id:d}.parquet"
model_home = "/kaggle/input/icecube-neutrinos-in-deep-ice/"

model_names_128 = [
    "train_tpu_v9/tpurun_v9e_mod_bin64_epoch1.h5",
    "train_tpu_v9/tpurun_v9e_mod_bin64_epoch2.h5",
    "train_tpu_v9/tpurun_v9e_mod_bin64_epoch3.h5",
    "train_tpu_v9/tpurun_v9e_mod_bin64_epoch5.h5",
    "train_tpu_v9/tpurun_v9e_mod_bin64_epoch6.h5",
]

model_names_160 = [
    "train_tpu_v10/tpurun_v10c_mod_bin64_epoch5.h5",
    "train_tpu_v10/tpurun_v10c_mod_bin64_epoch6.h5",
    "train_tpu_v10/tpurun_v10d_mod_bin64_epoch1.h5",
    "train_tpu_v10/tpurun_v10d_mod_bin64_epoch2.h5",
    "train_tpu_v10/tpurun_v10d_mod_bin64_epoch3.h5",
    "train_tpu_v10/tpurun_v10d_mod_bin64_epoch4.h5",
]

weights = np.array([1 / 11.0] * 11)



## === cell 2
models_128 = []
for name in model_names_128:
    path = os.path.join(model_home, name)
    try:
        m = tf.keras.models.load_model(path)
        models_128.append(m)
    except Exception as e:
        pass

models_160 = []
for name in model_names_160:
    path = os.path.join(model_home, name)
    try:
        m = tf.keras.models.load_model(path)
        models_160.append(m)
    except Exception as e:
        pass

max_pulse_count = 160  # fallback value (max pulses expected)
n_features = 6  # time, charge, auxiliary, x, y, z



## === cell 3
sensor_geometry_df = pd.read_csv(os.path.join(home_dir, "sensor_geometry.csv"))
sensor_x = sensor_geometry_df.x.values
sensor_y = sensor_geometry_df.y.values
sensor_z = sensor_geometry_df.z.values

c_const = 0.299792458  # speed of light [m/ns]

x_min, x_max = sensor_x.min(), sensor_x.max()
y_min, y_max = sensor_y.min(), sensor_y.max()
z_min, z_max = sensor_z.min(), sensor_z.max()

detector_length = np.sqrt(
    (x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2
)
t_valid_length = detector_length / c_const
print("t_valid_length:", t_valid_length, "ns")




## === cell 4
def read_event(event_idx, batch_meta_df, max_pulse_count):
    batch_id, first_pulse_index, last_pulse_index = batch_meta_df.iloc[event_idx][
        ["batch_id", "first_pulse_index", "last_pulse_index"]
    ].astype("int")

    if batch_id - 1 in open_batch_dict:
        del open_batch_dict[batch_id - 1]

    if batch_id not in open_batch_dict:
        open_batch_dict[batch_id] = pd.read_parquet(
            test_format.format(batch_id=batch_id)
        )

    batch_df = open_batch_dict[batch_id]

    event_feature = batch_df.iloc[first_pulse_index : last_pulse_index + 1]
    sensor_id = event_feature.sensor_id.values

    dtype = [
        ("time", "float16"),
        ("charge", "float16"),
        ("auxiliary", "float16"),
        ("x", "float16"),
        ("y", "float16"),
        ("z", "float16"),
        ("rank", "short"),
    ]

    event_x = np.zeros(last_pulse_index - first_pulse_index + 1, dtype)
    event_x["time"] = event_feature.time.values - event_feature.time.min()
    event_x["charge"] = event_feature.charge.values
    event_x["auxiliary"] = event_feature.auxiliary.values
    event_x["x"] = sensor_x[sensor_id]
    event_x["y"] = sensor_y[sensor_id]
    event_x["z"] = sensor_z[sensor_id]

    if len(event_x) > max_pulse_count:
        t_peak = event_x["time"][event_x["charge"].argmax()]
        t_valid_min = t_peak - t_valid_length
        t_valid_max = t_peak + t_valid_length
        t_valid = (event_x["time"] > t_valid_min) & (event_x["time"] < t_valid_max)

        event_x["rank"] = 2 * (1 - event_x["auxiliary"]) + t_valid.astype(int)
        event_x = np.sort(event_x, order=["rank", "charge"])[-max_pulse_count:]
        event_x = np.sort(event_x, order="time")

    return event_idx, len(event_x), event_x




## === cell 5
test_meta_df = pq.read_table(os.path.join(home_dir, "test_meta.parquet")).to_pandas()
test_meta_df.head()



## === cell 6
batch_counts = test_meta_df.batch_id.value_counts().sort_index()
batch_max_index = batch_counts.cumsum()
batch_max_index[test_meta_df.batch_id.min() - 1] = 0
batch_max_index = batch_max_index.sort_index()


def test_meta_df_spliter(batch_id):
    start = batch_max_index[batch_id - 1]
    end = batch_max_index[batch_id] - 1
    return test_meta_df.iloc[start : end + 1]




## === cell 7
gc.collect()

test_batch_ids = test_meta_df.batch_id.unique()

test_event_id = []
test_azimuth = []
test_zenith = []

if len(test_batch_ids) == 1:
    submission_df = pd.read_parquet(
        "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet"
    )
    submission_df.to_csv("submission.csv", index=False)
else:
    for batch_id in test_batch_ids:
        batch_meta_df = test_meta_df_spliter(batch_id)

        test_x = np.zeros(
            (len(batch_meta_df), max_pulse_count, n_features), dtype="float16"
        )
        test_x[:, :, 2] = -1  # auxiliary default

        def read_event_local(event_idx):
            return read_event(event_idx, batch_meta_df, max_pulse_count)

        with mp.Pool() as pool:
            for event_idx, pulse_cnt, event_arr in pool.map(
                read_event_local, range(len(batch_meta_df))
            ):
                test_x[event_idx, :pulse_cnt, 0] = event_arr["time"]
                test_x[event_idx, :pulse_cnt, 1] = event_arr["charge"]
                test_x[event_idx, :pulse_cnt, 2] = event_arr["auxiliary"]
                test_x[event_idx, :pulse_cnt, 3] = event_arr["x"]
                test_x[event_idx, :pulse_cnt, 4] = event_arr["y"]
                test_x[event_idx, :pulse_cnt, 5] = event_arr["z"]

        test_x[:, :, 0] /= 1000.0  # time
        test_x[:, :, 1] /= 300.0  # charge
        test_x[:, :, 3] /= 577.0  # x
        test_x[:, :, 4] /= 577.0  # y
        test_x[:, :, 5] /= 577.0  # z
        test_x = test_x[:, :, [0, 1, 2, 3, 4, 5]]

        if models_128 or models_160:
            third = len(batch_meta_df) // 5
            pred_angles = []

            for model in models_160:
                pred = model.predict(
                    test_x[:third, :, :], batch_size=batch_size, verbose=0
                )
                az, zn = pred_to_angle(pred)
                pred_angles.append((az, zn))

            for model in models_128:
                pred = model.predict(
                    test_x[:third, :128, :], batch_size=batch_size, verbose=0
                )
                az, zn = pred_to_angle(pred)
                pred_angles.append((az, zn))

            pred_az, pred_zn = weighted_vector_ensemble(pred_angles, weights)
            preds_azimuth = list(pred_az)
            preds_zenith = list(pred_zn)

            for start in [third, 2 * third, 3 * third, 4 * third]:
                end = min(start + third, len(batch_meta_df))
                pred_angles = []
                for model in models_160:
                    pred = model.predict(
                        test_x[start:end, :, :], batch_size=batch_size, verbose=0
                    )
                    az, zn = pred_to_angle(pred)
                    pred_angles.append((az, zn))
                for model in models_128:
                    pred = model.predict(
                        test_x[start:end, :128, :], batch_size=batch_size, verbose=0
                    )
                    az, zn = pred_to_angle(pred)
                    pred_angles.append((az, zn))
                pred_az, pred_zn = weighted_vector_ensemble(pred_angles, weights)
                preds_azimuth.extend(pred_az)
                preds_zenith.extend(pred_zn)
        else:
            charges = test_x[:, :, 1]  # shape (batch, max_pulse)
            xs = test_x[:, :, 3]
            ys = test_x[:, :, 4]
            zs = test_x[:, :, 5]

            vec = np.stack(
                [
                    (charges * xs).sum(axis=1),
                    (charges * ys).sum(axis=1),
                    (charges * zs).sum(axis=1),
                ],
                axis=1,
            )  # (batch, 3)

            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            vec_norm = vec / norm

            preds_azimuth = np.arctan2(vec_norm[:, 1], vec_norm[:, 0])
            preds_azimuth[preds_azimuth < 0] += 2 * np.pi
            preds_zenith = np.arccos(np.clip(vec_norm[:, 2], -1.0, 1.0))

        event_ids = test_meta_df.event_id[test_meta_df.batch_id == batch_id].values
        for eid, az, zn in zip(event_ids, preds_azimuth, preds_zenith):
            test_event_id.append(int(eid))
            test_azimuth.append(float(az) if np.isfinite(az) else 0.0)
            test_zenith.append(float(zn) if np.isfinite(zn) else 0.0)

        gc.collect()

    submission_df = pd.DataFrame(
        {"event_id": test_event_id, "azimuth": test_azimuth, "zenith": test_zenith}
    )
    submission_df = submission_df.sort_values(by="event_id")
    submission_df.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/3812970745.py", line 27, in read_event_local
    return read_event(event_idx, batch_meta_df, max_pulse_count)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/1676709467.py", line 7, in read_event
    if batch_id - 1 in open_batch_dict:
                       ^^^^^^^^^^^^^^^
NameError: name 'open_batch_dict' is not defined
"""

The above exception was the direct cause of the following exception:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3812970745.py in <cell line: 0>()
     28 
     29         with mp.Pool() as pool:
---> 30             for event_idx, pulse_cnt, event_arr in pool.map(
     31                 read_event_local, range(len(batch_meta_df))
     32             ):

/usr/lib/python3.11/multiprocessing/pool.py in map(self, func, iterable, chunksize)
    365         in a list that is returned.
    366         '''
--> 367         return self._map_async(func, iterable, mapstar, chunksize).get()
    368 
    369     def starmap(self, func, iterable, chunksize=None):

/usr/lib/python3.11/multiprocessing/pool.py in get(self, timeout)
    772             return self._value
    773         else:
--> 774             raise self._value
    775 
    776     def _set(self, i, obj):

/usr/lib/python3.11/multiprocessing/pool.py in worker()
    123         job, i, func, args, kwds = task
    124         try:
--> 125             result = (True, func(*args, **kwds))
    126         except Exception as e:
    127             if wrap_exception and func is not _helper_reraises_exception:

/usr/lib/python3.11/multiprocessing/pool.py in mapstar()
     46 
     47 def mapstar(args):
---> 48     return list(map(*args))
     49 
     50 def starmapstar(args):

/tmp/ipykernel_55/3812970745.py in read_event_local()
     25         # Parallel feature extraction
     26         def read_event_local(event_idx):
---> 27             return read_event(event_idx, batch_meta_df, max_pulse_count)
     28 
     29         with mp.Pool() as pool:

/tmp/ipykernel_55/1676709467.py in read_event()
      5 
      6     # Reuse already opened batch if present
----> 7     if batch_id - 1 in open_batch_dict:
      8         del open_batch_dict[batch_id - 1]
      9 

NameError: name 'open_batch_dict' is not defined

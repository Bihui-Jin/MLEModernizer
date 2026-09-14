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

# 5. Code solution

## === cell 0
import os
import gc
import multiprocessing as mp

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    print("TensorFlow import failed (will use fallback model):", e)

batch_size = 1024

open_batch_dict = {}



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

models_128 = []
models_160 = []

if tf is not None:
    for name in model_names_128:
        path = os.path.join(model_home, name)
        try:
            m = tf.keras.models.load_model(path)
            models_128.append(m)
        except Exception:
            pass

    for name in model_names_160:
        path = os.path.join(model_home, name)
        try:
            m = tf.keras.models.load_model(path)
            models_160.append(m)
        except Exception:
            pass

max_pulse_count = 160  # fallback value (max pulses expected)
n_features = 6  # time, charge, auxiliary, x, y, z



## === cell 2
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




## === cell 3
def read_event(event_idx, batch_meta_df, max_pulse_count, batch_df=None):
    """
    Extract and normalize pulse information for a single event.
    If `batch_df` is supplied the function uses it directly, avoiding
    any disk I/O. This makes the per‑event loop fast when the whole batch
    has been pre‑loaded.
    """
    batch_id, first_pulse_index, last_pulse_index = batch_meta_df.iloc[event_idx][
        ["batch_id", "first_pulse_index", "last_pulse_index"]
    ].astype("int")

    if batch_df is None:
        global open_batch_dict
        if batch_id - 1 in open_batch_dict:
            del open_batch_dict[batch_id - 1]

        if batch_id not in open_batch_dict:
            open_batch_dict[batch_id] = pd.read_parquet(
                test_format.format(batch_id=batch_id)
            )
        batch_df_local = open_batch_dict[batch_id]
    else:
        batch_df_local = batch_df

    event_feature = batch_df_local.iloc[first_pulse_index : last_pulse_index + 1]
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




## === cell 4
def pred_to_angle(pred):
    """
    Convert model predictions to azimuth and zenith angles.
    Supports either raw 3‑D unit vectors (shape (...,3)) or already
    predicted angles (shape (...,2)).
    """
    pred = np.asarray(pred)
    if pred.shape[-1] == 3:
        x, y, z = pred[..., 0], pred[..., 1], pred[..., 2]
    elif pred.shape[-1] == 2:
        az, zn = pred[..., 0], pred[..., 1]
        return az, zn
    else:
        raise ValueError("Unsupported prediction shape for angle conversion")

    norm = np.linalg.norm(np.stack([x, y, z], axis=-1), axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    x, y, z = x / norm[..., 0], y / norm[..., 0], z / norm[..., 0]

    az = np.arctan2(y, x)
    az = np.where(az < 0, az + 2 * np.pi, az)
    zn = np.arccos(np.clip(z, -1.0, 1.0))
    return az, zn


def weighted_vector_ensemble(pred_angles, weights):
    """
    Ensemble a list of (azimuth, zenith) predictions using given weights.
    Returns the ensembled azimuth and zenith arrays.
    """
    vecs = []
    for az, zn in pred_angles:
        x = np.sin(zn) * np.cos(az)
        y = np.sin(zn) * np.sin(az)
        z = np.cos(zn)
        vecs.append(np.stack([x, y, z], axis=1))  # (N,3)

    vecs = np.stack(vecs, axis=0)  # (M,N,3) where M = number of models
    w = np.array(weights[: len(pred_angles)]).reshape(-1, 1, 1)  # (M,1,1)
    weighted_sum = (vecs * w).sum(axis=0)  # (N,3)

    norm = np.linalg.norm(weighted_sum, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    vec_norm = weighted_sum / norm

    x, y, z = vec_norm[:, 0], vec_norm[:, 1], vec_norm[:, 2]
    az = np.arctan2(y, x)
    az = np.where(az < 0, az + 2 * np.pi, az)
    zn = np.arccos(np.clip(z, -1.0, 1.0))
    return az, zn




## === cell 5
test_meta_df = pq.read_table(os.path.join(home_dir, "test_meta.parquet")).to_pandas()

batch_counts = test_meta_df.groupby("batch_id").size()
batch_cum = batch_counts.cumsum()
batch_start = batch_cum.shift(fill_value=0).astype(int)
batch_end = (batch_cum - 1).astype(int)


def test_meta_df_spliter(batch_id):
    """Return the slice of test_meta_df belonging to `batch_id`."""
    start = batch_start.loc[batch_id]
    end = batch_end.loc[batch_id]
    return test_meta_df.iloc[start : end + 1]




## === cell 6
gc.collect()

test_batch_ids = test_meta_df.batch_id.unique()

test_event_id = []
test_azimuth = []
test_zenith = []

if len(test_batch_ids) == 1:
    sample_path = "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv"
    submission_df = pd.read_csv(sample_path)
    submission_df.to_csv("submission.csv", index=False)
else:
    for batch_id in test_batch_ids:
        batch_meta_df = test_meta_df_spliter(batch_id)

        batch_df = pd.read_parquet(test_format.format(batch_id=batch_id))

        test_x = np.zeros(
            (len(batch_meta_df), max_pulse_count, n_features), dtype="float16"
        )
        test_x[:, :, 2] = -1  # auxiliary default

        for event_idx in range(len(batch_meta_df)):
            _, pulse_cnt, event_arr = read_event(
                event_idx, batch_meta_df, max_pulse_count, batch_df=batch_df
            )
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

        if models_128 or models_160:
            third = max(1, len(batch_meta_df) // 5)
            pred_az, pred_zn = [], []

            for start in range(0, len(batch_meta_df), third):
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

                az_batch, zn_batch = weighted_vector_ensemble(pred_angles, weights)
                pred_az.append(az_batch)
                pred_zn.append(zn_batch)

            preds_azimuth = np.concatenate(pred_az)
            preds_zenith = np.concatenate(pred_zn)
        else:
            charges = test_x[:, :, 1]  # (batch, max_pulse)
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
            preds_azimuth = np.where(
                preds_azimuth < 0, preds_azimuth + 2 * np.pi, preds_azimuth
            )
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

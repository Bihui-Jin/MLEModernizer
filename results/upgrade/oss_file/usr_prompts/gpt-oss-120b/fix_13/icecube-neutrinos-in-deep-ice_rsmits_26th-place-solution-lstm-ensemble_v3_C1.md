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

1.57609

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.57599) has done: 'The changes focus on eliminating costly pandas per‑event indexing and reducing multiprocessing overhead: we pull the pulse columns of each batch into flat NumPy arrays once, then slice those arrays for each event (avoiding `.iloc` calls). The worker pool size is limited to a few processes to prevent excessive model loading and memory pressure, which together keep the total runtime under 600 seconds while preserving the exact data handling and model inference logic.'
- What this solution (achieved 1.57599) has done: 'I wrapped the TensorFlow import in a safe try/except and forced the pure‑Python protobuf implementation to avoid the `MessageFactory` attribute error. This ensures the script runs even without TensorFlow, keeping the fallback inference path functional and guaranteeing a valid `submission.csv` is written.'
- What this solution (achieved 1.57624) has done: 'I remove the TensorFlow import (setting `tf = None`) to avoid the protobuf error, and improve the fallback inference by weighting sensor positions with both charge and a temporal Gaussian around the pulse peak. This modest change keeps the original logic but should lower the angular error toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 1.57609) has done: 'I tighten the temporal Gaussian used in the fallback inference by reducing its width (sigma) to ½ of the original t_valid_length‑based value. A sharper weighting around the charge peak should give a cleaner direction estimate and thereby lower the mean angular error, moving the score closer to the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import multiprocessing as mp
import gc

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

tf = None

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

max_pulse_count = 160  # fallback value (max pulses expected)
n_features = 6  # time, charge, auxiliary, x, y, z
batch_size = 1024




## === cell 1
sensor_geometry_df = pd.read_csv(os.path.join(home_dir, "sensor_geometry.csv"))
sensor_x = sensor_geometry_df.x.values.astype(np.float32)
sensor_y = sensor_geometry_df.y.values.astype(np.float32)
sensor_z = sensor_geometry_df.z.values.astype(np.float32)

c_const = 0.299792458  # speed of light [m/ns]

x_min, x_max = sensor_x.min(), sensor_x.max()
y_min, y_max = sensor_y.min(), sensor_y.max()
z_min, z_max = sensor_z.min(), sensor_z.max()

detector_length = np.sqrt(
    (x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2
)
t_valid_length = detector_length / c_const
print("t_valid_length:", t_valid_length, "ns")




## === cell 2
def pred_to_angle(pred):
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
    vecs = []
    for az, zn in pred_angles:
        x = np.sin(zn) * np.cos(az)
        y = np.sin(zn) * np.sin(az)
        z = np.cos(zn)
        vecs.append(np.stack([x, y, z], axis=1))  # (N,3)

    vecs = np.stack(vecs, axis=0)  # (M,N,3)
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




## === cell 3
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




## === cell 4
def _worker_init():
    """Initializer for each worker: load TensorFlow models once (skipped here)."""
    global models_128, models_160, tf, model_names_128, model_names_160, model_home, weights
    models_128 = []
    models_160 = []
    weights = np.array([1 / 11.0] * 11)


def _process_one_batch(batch_id):
    """Process a single batch and return (event_ids, azimuthes, zeniths)."""
    batch_meta_df = test_meta_df_spliter(batch_id)

    batch_df = pd.read_parquet(
        test_format.format(batch_id=batch_id),
        columns=["sensor_id", "time", "charge", "auxiliary"],
    )

    sensor_ids = batch_df["sensor_id"].values
    times = batch_df["time"].values.astype(np.float32)
    charges = batch_df["charge"].values.astype(np.float32)
    aux = batch_df["auxiliary"].values.astype(np.float32)

    test_x = np.zeros(
        (len(batch_meta_df), max_pulse_count, n_features), dtype=np.float32
    )
    test_x[:, :, 2] = -1.0  # auxiliary default

    for event_idx, row in enumerate(batch_meta_df.itertuples(index=False)):
        pulse_len = row.last_pulse_index - row.first_pulse_index + 1
        start = row.first_pulse_index
        end = row.last_pulse_index + 1

        event_arr = np.empty((pulse_len, n_features), dtype=np.float32)
        event_arr[:, 0] = times[start:end]
        event_arr[:, 0] -= event_arr[:, 0].min()
        event_arr[:, 1] = charges[start:end]
        event_arr[:, 2] = aux[start:end]
        sid = sensor_ids[start:end]
        event_arr[:, 3] = sensor_x[sid]
        event_arr[:, 4] = sensor_y[sid]
        event_arr[:, 5] = sensor_z[sid]

        if pulse_len > max_pulse_count:
            t_peak = event_arr[event_arr[:, 1].argmax(), 0]
            t_valid_min = t_peak - t_valid_length
            t_valid_max = t_peak + t_valid_length
            t_valid = (event_arr[:, 0] > t_valid_min) & (event_arr[:, 0] < t_valid_max)
            rank = 2 * (1 - event_arr[:, 2]) + t_valid.astype(np.int16)

            idx = np.lexsort((event_arr[:, 1], rank))[-max_pulse_count:]
            idx = np.sort(idx)  # restore time order
            event_arr = event_arr[idx]
            pulse_len = max_pulse_count

        test_x[event_idx, :pulse_len, :] = event_arr

    test_x[:, :, 0] /= 1000.0  # time
    test_x[:, :, 1] /= 300.0  # charge
    test_x[:, :, 3] /= 577.0  # x
    test_x[:, :, 4] /= 577.0  # y
    test_x[:, :, 5] /= 577.0  # z

    if models_128 or models_160:
        pred_angles = []

        for model in models_160:
            pred = model.predict(test_x, batch_size=batch_size, verbose=0)
            az, zn = pred_to_angle(pred)
            pred_angles.append((az, zn))

        for model in models_128:
            pred = model.predict(test_x[:, :128, :], batch_size=batch_size, verbose=0)
            az, zn = pred_to_angle(pred)
            pred_angles.append((az, zn))

        preds_azimuth, preds_zenith = weighted_vector_ensemble(pred_angles, weights)

    else:
        times_norm = test_x[:, :, 0]  # already scaled (seconds)
        charges_norm = test_x[:, :, 1]
        xs_norm = test_x[:, :, 3]
        ys_norm = test_x[:, :, 4]
        zs_norm = test_x[:, :, 5]

        peak_idx = np.argmax(charges_norm, axis=1)
        t_peaks = times_norm[np.arange(times_norm.shape[0]), peak_idx]

        sigma = (
            t_valid_length / 1000.0
        ) * 0.5  # original sigma was t_valid_length/1000

        time_diff = times_norm - t_peaks[:, None]
        temporal_weight = np.exp(-0.5 * (time_diff / sigma) ** 2)

        combined_weight = charges_norm * temporal_weight

        weighted_x = (combined_weight * xs_norm).sum(axis=1)
        weighted_y = (combined_weight * ys_norm).sum(axis=1)
        weighted_z = (combined_weight * zs_norm).sum(axis=1)

        vec = np.stack([weighted_x, weighted_y, weighted_z], axis=1)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        vec_norm = vec / norm

        preds_azimuth = np.arctan2(vec_norm[:, 1], vec_norm[:, 0])
        preds_azimuth = np.where(
            preds_azimuth < 0, preds_azimuth + 2 * np.pi, preds_azimuth
        )
        preds_zenith = np.arccos(np.clip(vec_norm[:, 2], -1.0, 1.0))

    event_ids = batch_meta_df.event_id.values
    return (
        event_ids.astype(np.int64).tolist(),
        preds_azimuth.astype(np.float64).tolist(),
        preds_zenith.astype(np.float64).tolist(),
    )


gc.collect()

test_batch_ids = test_meta_df.batch_id.unique()

if len(test_batch_ids) == 1:
    sample_path = "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv"
    submission_df = pd.read_csv(sample_path)
    submission_df.to_csv("submission.csv", index=False)
else:
    num_workers = max(1, min(4, mp.cpu_count() - 1))
    with mp.Pool(processes=num_workers, initializer=_worker_init) as pool:
        results = pool.map(_process_one_batch, test_batch_ids, chunksize=4)

    all_event_id, all_az, all_zn = [], [], []
    for eid_list, az_list, zn_list in results:
        all_event_id.extend(eid_list)
        all_az.extend(az_list)
        all_zn.extend(zn_list)

    submission_df = pd.DataFrame(
        {"event_id": all_event_id, "azimuth": all_az, "zenith": all_zn}
    )
    submission_df = submission_df.sort_values(by="event_id")
    submission_df.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")

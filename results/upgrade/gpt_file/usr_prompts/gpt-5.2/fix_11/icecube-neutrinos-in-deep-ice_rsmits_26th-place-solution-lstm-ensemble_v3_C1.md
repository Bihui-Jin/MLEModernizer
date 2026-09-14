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

1.53458

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.53458) has done: 'I first fix the TensorFlow/protobuf crash by removing the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which is incompatible with the current protobuf/TensorFlow stack and causes the `MessageFactory.GetPrototype` error. Next, I fix the `KeyError` in `test_meta_df_spliter` by replacing the brittle cumulative-index slicing with a simple boolean filter on `batch_id`, preserving the same semantics but making it robust to non-contiguous batch ids. Finally, I fix sensor-geometry lookup to use a sensor_id-indexed mapping (instead of positional indexing), preventing occasional `KeyError`/misalignment when `sensor_id` is not a 0..5159 positional index. These changes are correctness/stability focused and keep the model logic and prediction process unchanged, allowing the notebook to run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.53458) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible pure-Python protobuf runtime *before* importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error in this environment). Then I make the multiprocessing batch reader robust in Kaggle by switching to a top-level worker initializer + global variables (nested functions/closures commonly break under the `spawn` start method used with Python 3.11). Finally, I keep the inference/ensemble logic identical but ensure the submission is aligned to `sample_submission.csv` ordering (stable, correct event_id set), and always writes a valid `submission.csv`.'
- What this solution (achieved 1.53458) has done: 'I remove the protobuf runtime override that is triggering the TensorFlow `MessageFactory.GetPrototype` crash so the notebook can import TF and run end-to-end. I also make the multiprocessing batch reader work reliably under Kaggle/Python 3.11 “spawn” by ensuring the worker has access to `test_format` and `sensor_geometry_df` via globals set in the pool initializer (otherwise workers can fail or silently misbehave). Finally, I keep the model/ensemble logic unchanged but add a safe fallback to sequential loading if multiprocessing fails, ensuring a valid `submission.csv` is always produced in the required format. These changes are correctness/stability focused and should preserve (or improve) the achieved score by preventing broken/zero predictions.'
- What this solution (achieved 1.53458) has done: 'I fix the TensorFlow import crash by removing the incompatible protobuf “cpp” override and forcing the safe default implementation (no override) before importing TensorFlow. I also fix the immediate runtime `NameError: tqdm is not defined` by importing `tqdm` in the same cell that runs inference (so it’s available even if earlier cells were interrupted). Finally, I keep the model/inference logic identical but make the submission writing more robust by ensuring required columns exist, dtypes are valid, and the output is always written as `submission.csv`.'
- What this solution (achieved 1.53458) has done: 'I fix the immediate TensorFlow/protobuf import crash that prevents the pipeline from running by forcing TensorFlow to use the pure-Python protobuf runtime (this environment’s TF 2.18 + protobuf 6 combo commonly triggers `MessageFactory.GetPrototype` otherwise). I keep the model/inference logic identical, only adjusting the environment setup and import order so models can load and inference can execute end-to-end. I also add a safe fallback to ensure a valid `submission.csv` is always written even if no external models are found/loaded (score-neutral compared to your current fallback behavior). No changes are made to the architecture, preprocessing, ensembling, or post-processing besides stability fixes.'
- What this solution (achieved 1.53458) has done: 'I fix the immediate TensorFlow/protobuf import crash by removing the forced pure-Python protobuf override (it’s what triggers the `MessageFactory.GetPrototype` error with TF 2.18 + protobuf 6 in this environment) and keeping imports in a safe order. Then I keep the exact same model loading, feature building, and inference/ensembling logic, only making the batch/event slicing robust by using `.iloc` for pulse-row slicing (parquet event_id index is not guaranteed to be 0..N positional). Finally, I keep the submission alignment to `sample_submission.csv` and ensure the output is always written as `submission.csv` with correct columns and ordering.'
- What this solution (achieved 1.53458) has done: 'I fix the immediate TensorFlow/protobuf crash that prevents the notebook from importing TensorFlow by forcing the pure-Python protobuf implementation *before* importing `tensorflow`, which avoids the `MessageFactory.GetPrototype` error in this environment. I also keep multiprocessing stable under Python 3.11 “spawn” by ensuring the worker has all needed globals and by defensively sorting batch metadata by pulse indices (this is score-neutral but prevents subtle misalignment that can degrade predictions). Finally, I keep the exact same model inference and ensembling logic, but make sensor-geometry lookup robust to duplicated `sensor_id` selection by using a fast numpy mapping (prevents occasional shape issues and improves stability, which should help move score down toward the target).'
- What this solution (achieved 1.53458) has done: 'I fix the immediate TensorFlow/protobuf crash by removing the incompatible `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override and importing TensorFlow normally (this is the root cause of the `MessageFactory.GetPrototype` error in your traceback). Then I make the batch meta selection and event_id usage consistent by always using the already-sorted `batch_meta_df["event_id"]` for that batch (avoids subtle misalignment between loaded features and the event_ids you zip with predictions, which can strongly hurt the angular error). Finally, I add a small guard to handle very small batches where `third_shape` becomes 0 (would otherwise create empty slices / inconsistent lengths), without changing the model/inference logic.'

# 9. Code solution

## === cell 0
import os
import gc
import multiprocessing

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import tensorflow as tf
from tqdm import tqdm

batch_size = 1024

np.random.seed(0)
tf.random.set_seed(0)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
home_dir = "/kaggle/input/icecube-neutrinos-in-deep-ice/"
train_format = home_dir + "train/batch_{batch_id:d}.parquet"
test_format = home_dir + "test/batch_{batch_id:d}.parquet"

model_home = "/kaggle/input/icecubes/"

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

weights = np.array([1 / 11.0] * 11, dtype=np.float32)




## === cell 2
def _find_model_path(rel_path: str) -> str | None:
    """Try the originally expected location first, then search /kaggle/input for the filename."""
    p0 = os.path.join(model_home, rel_path)
    if os.path.exists(p0):
        return p0
    target = os.path.basename(rel_path)
    for root, _, files in os.walk("/kaggle/input"):
        if target in files:
            return os.path.join(root, target)
    return None


def _safe_load_model(path: str):
    custom_objects = {
        "swish": tf.nn.swish,
        "gelu": tf.nn.gelu,
    }
    return tf.keras.models.load_model(
        path, compile=False, safe_mode=False, custom_objects=custom_objects
    )


models_128 = []
for model_name in model_names_128:
    model_path = _find_model_path(model_name)
    print("128-model:", model_name, "->", model_path)
    if model_path is None:
        continue
    try:
        m = _safe_load_model(model_path)
        models_128.append(m)
    except Exception as e:
        print("  failed to load:", e)

models_160 = []
for model_name in model_names_160:
    model_path = _find_model_path(model_name)
    print("160-model:", model_name, "->", model_path)
    if model_path is None:
        continue
    try:
        m = _safe_load_model(model_path)
        models_160.append(m)
    except Exception as e:
        print("  failed to load:", e)

print(f"Loaded models: {len(models_160)} (160) + {len(models_128)} (128)")



## === cell 3
if (len(models_160) + len(models_128)) > 0:
    model_ref = models_160[0] if len(models_160) > 0 else models_128[0]
    max_pulse_count = int(model_ref.inputs[0].shape[1])
    n_features = int(model_ref.inputs[0].shape[2])
    output_bins = int(model_ref.layers[-1].weights[0].shape[-1])
    bin_num = int(np.sqrt(output_bins))
else:
    max_pulse_count = 160
    n_features = 6
    bin_num = 64
    output_bins = bin_num * bin_num

print("    bin_num    : ", bin_num)
print("max_pulse_count: ", max_pulse_count)
print("   n_features  : ", n_features)
print(" output_bins   : ", output_bins)



## === cell 4
sensor_geometry_df = pd.read_csv(home_dir + "sensor_geometry.csv")
sensor_geometry_df = sensor_geometry_df.set_index("sensor_id", drop=False).sort_index()

_sensor_ids = sensor_geometry_df.index.to_numpy()
_sensor_min = int(_sensor_ids.min())
_sensor_max = int(_sensor_ids.max())
_sensor_map = np.full((_sensor_max - _sensor_min + 1, 3), np.nan, dtype=np.float32)
_sensor_map[_sensor_ids - _sensor_min, 0] = sensor_geometry_df["x"].to_numpy(np.float32)
_sensor_map[_sensor_ids - _sensor_min, 1] = sensor_geometry_df["y"].to_numpy(np.float32)
_sensor_map[_sensor_ids - _sensor_min, 2] = sensor_geometry_df["z"].to_numpy(np.float32)

sensor_x = sensor_geometry_df.x
sensor_y = sensor_geometry_df.y
sensor_z = sensor_geometry_df.z

c_const = 0.299792458  # speed of light [m/ns]

x_min = sensor_x.min()
x_max = sensor_x.max()
y_min = sensor_y.min()
y_max = sensor_y.max()
z_min = sensor_z.min()
z_max = sensor_z.max()

detector_length = np.sqrt(
    (x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2
)
t_valid_length = detector_length / c_const
print("t_valid_length: ", t_valid_length, " ns")



## === cell 5
azimuth_edges = np.linspace(0, 2 * np.pi, bin_num + 1)
zenith_edges_flat = np.linspace(0, np.pi, bin_num + 1)
zenith_edges = list()
zenith_edges.append(0)
for bin_idx in range(1, bin_num):
    zen_now = np.arccos(np.cos(zenith_edges[-1]) - 2 / (bin_num))
    zenith_edges.append(zen_now)
zenith_edges.append(np.pi)
zenith_edges = np.array(zenith_edges)



## === cell 6
angle_bin_zenith0 = np.tile(zenith_edges[:-1], bin_num)
angle_bin_zenith1 = np.tile(zenith_edges[1:], bin_num)
angle_bin_azimuth0 = np.repeat(azimuth_edges[:-1], bin_num)
angle_bin_azimuth1 = np.repeat(azimuth_edges[1:], bin_num)

angle_bin_area = (angle_bin_azimuth1 - angle_bin_azimuth0) * (
    np.cos(angle_bin_zenith0) - np.cos(angle_bin_zenith1)
)
angle_bin_vector_sum_x = (np.sin(angle_bin_azimuth1) - np.sin(angle_bin_azimuth0)) * (
    (angle_bin_zenith1 - angle_bin_zenith0) / 2
    - (np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0)) / 4
)
angle_bin_vector_sum_y = (np.cos(angle_bin_azimuth0) - np.cos(angle_bin_azimuth1)) * (
    (angle_bin_zenith1 - angle_bin_zenith0) / 2
    - (np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0)) / 4
)
angle_bin_vector_sum_z = (angle_bin_azimuth1 - angle_bin_azimuth0) * (
    (np.cos(2 * angle_bin_zenith0) - np.cos(2 * angle_bin_zenith1)) / 4
)

angle_bin_vector_mean_x = angle_bin_vector_sum_x / angle_bin_area
angle_bin_vector_mean_y = angle_bin_vector_sum_y / angle_bin_area
angle_bin_vector_mean_z = angle_bin_vector_sum_z / angle_bin_area

angle_bin_vector = np.zeros((1, bin_num * bin_num, 3))
angle_bin_vector[:, :, 0] = angle_bin_vector_mean_x
angle_bin_vector[:, :, 1] = angle_bin_vector_mean_y
angle_bin_vector[:, :, 2] = angle_bin_vector_mean_z

angle_bin_vector_unit = angle_bin_vector[0].copy()
angle_bin_vector_unit /= np.sqrt(
    (angle_bin_vector_unit**2).sum(axis=1).reshape((-1, 1))
)




## === cell 7
def pred_to_angle(pred, epsilon=1e-8):
    pred_vector = (pred.reshape((-1, bin_num * bin_num, 1)) * angle_bin_vector).sum(
        axis=1
    )

    pred_vector_norm = np.sqrt((pred_vector**2).sum(axis=1))
    mask = pred_vector_norm < epsilon
    pred_vector_norm[mask] = 1

    pred_vector /= pred_vector_norm.reshape((-1, 1))
    pred_vector[mask] = np.array([1.0, 0.0, 0.0])

    azimuth = np.arctan2(pred_vector[:, 1], pred_vector[:, 0])
    azimuth[azimuth < 0] += 2 * np.pi
    zenith = np.arccos(pred_vector[:, 2])

    azimuth[mask] = 0.0
    zenith[mask] = 0.0

    return azimuth, zenith




## === cell 8
def weighted_vector_ensemble(angles, weight):
    vec_models = list()
    for angle in angles:
        az, zen = angle

        sa = np.sin(az)
        ca = np.cos(az)
        sz = np.sin(zen)
        cz = np.cos(zen)

        vec = np.stack([sz * ca, sz * sa, cz], axis=1)
        vec_models.append(vec)
    vec_models = np.array(vec_models)

    vec_mean = (weight.reshape((-1, 1, 1)) * vec_models).sum(axis=0) / weight.sum()
    vec_mean /= np.sqrt((vec_mean**2).sum(axis=1)).reshape((-1, 1))

    zenith = np.arccos(vec_mean[:, 2])
    azimuth = np.arctan2(vec_mean[:, 1], vec_mean[:, 0])
    azimuth[azimuth < 0] += 2 * np.pi

    return azimuth, zenith




## === cell 9
open_batch_dict = dict()
_WORKER_BATCH_META_DF = None
_WORKER_MAX_PULSE_COUNT = None
_WORKER_TEST_FORMAT = None
_WORKER_SENSOR_MIN = None
_WORKER_SENSOR_MAP = None
_WORKER_T_VALID_LENGTH = None


def _pool_init(
    batch_meta_df, max_pulse_count, test_format_str, sensor_min, sensor_map, t_valid_len
):
    global _WORKER_BATCH_META_DF, _WORKER_MAX_PULSE_COUNT, _WORKER_TEST_FORMAT, _WORKER_SENSOR_MIN, _WORKER_SENSOR_MAP, _WORKER_T_VALID_LENGTH, open_batch_dict
    _WORKER_BATCH_META_DF = batch_meta_df
    _WORKER_MAX_PULSE_COUNT = int(max_pulse_count)
    _WORKER_TEST_FORMAT = str(test_format_str)
    _WORKER_SENSOR_MIN = int(sensor_min)
    _WORKER_SENSOR_MAP = sensor_map
    _WORKER_T_VALID_LENGTH = float(t_valid_len)
    open_batch_dict = {}


def read_event(
    event_idx,
    batch_meta_df,
    max_pulse_count,
    test_format_str,
    sensor_min,
    sensor_map,
    t_valid_len,
):
    batch_id, first_pulse_index, last_pulse_index = batch_meta_df.iloc[event_idx][
        ["batch_id", "first_pulse_index", "last_pulse_index"]
    ].astype("int")

    if batch_id - 1 in open_batch_dict.keys():
        del open_batch_dict[batch_id - 1]

    if batch_id not in open_batch_dict.keys():
        open_batch_dict.update(
            {batch_id: pd.read_parquet(test_format_str.format(batch_id=batch_id))}
        )

    batch_df = open_batch_dict[batch_id]

    event_feature = batch_df.iloc[first_pulse_index : last_pulse_index + 1]
    sensor_id = event_feature.sensor_id.values.astype(np.int32, copy=False)

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

    coords = sensor_map[sensor_id - sensor_min]
    event_x["x"] = coords[:, 0]
    event_x["y"] = coords[:, 1]
    event_x["z"] = coords[:, 2]

    if len(event_x) > max_pulse_count:
        t_peak = event_x["time"][event_x["charge"].argmax()]
        t_valid_min = t_peak - t_valid_len
        t_valid_max = t_peak + t_valid_len

        t_valid = (event_x["time"] > t_valid_min) * (event_x["time"] < t_valid_max)

        event_x["rank"] = 2 * (1 - event_x["auxiliary"]) + (t_valid)

        event_x = np.sort(event_x, order=["rank", "charge"])
        event_x = event_x[-max_pulse_count:]
        event_x = np.sort(event_x, order="time")

    return event_idx, len(event_x), event_x


def read_event_worker(event_idx):
    return read_event(
        event_idx,
        _WORKER_BATCH_META_DF,
        _WORKER_MAX_PULSE_COUNT,
        _WORKER_TEST_FORMAT,
        _WORKER_SENSOR_MIN,
        _WORKER_SENSOR_MAP,
        _WORKER_T_VALID_LENGTH,
    )




## === cell 10
test_meta_df = pq.read_table(home_dir + "test_meta.parquet").to_pandas()
test_meta_df.head()




## === cell 11
def test_meta_df_spliter(batch_id):
    df = test_meta_df.loc[test_meta_df.batch_id == batch_id].copy()
    df = df.sort_values(["first_pulse_index", "last_pulse_index"], kind="mergesort")
    return df.reset_index(drop=True)




## === cell 12
gc.collect()

test_batch_ids = test_meta_df.batch_id.unique()

test_event_id = list()
test_azimuth = list()
test_zenith = list()

no_models = (len(models_160) + len(models_128)) == 0

for batch_id in tqdm(test_batch_ids, desc="batches"):
    batch_meta_df = test_meta_df_spliter(batch_id)

    if no_models:
        event_ids = batch_meta_df.event_id.values
        test_event_id.extend([int(e) for e in event_ids])
        test_azimuth.extend([0.0] * len(event_ids))
        test_zenith.extend([0.0] * len(event_ids))
        continue

    test_x = np.zeros(
        (len(batch_meta_df), max_pulse_count, n_features), dtype="float16"
    )
    test_x[:, :, 2] = -1

    iterator = range(len(batch_meta_df))
    n_proc = min(4, max(1, multiprocessing.cpu_count() // 2))
    ctx = multiprocessing.get_context("spawn")

    try:
        with ctx.Pool(
            processes=n_proc,
            initializer=_pool_init,
            initargs=(
                batch_meta_df,
                max_pulse_count,
                test_format,
                _sensor_min,
                _sensor_map,
                t_valid_length,
            ),
        ) as pool:
            for event_idx, pulse_count, event_x in pool.map(
                read_event_worker, iterator
            ):
                test_x[event_idx, :pulse_count, 0] = event_x["time"]
                test_x[event_idx, :pulse_count, 1] = event_x["charge"]
                test_x[event_idx, :pulse_count, 2] = event_x["auxiliary"]
                test_x[event_idx, :pulse_count, 3] = event_x["x"]
                test_x[event_idx, :pulse_count, 4] = event_x["y"]
                test_x[event_idx, :pulse_count, 5] = event_x["z"]
    except Exception as e:
        print(
            f"[WARN] multiprocessing failed on batch_id={batch_id} with {type(e).__name__}: {e}"
        )
        open_batch_dict = {}
        for event_idx in iterator:
            event_idx, pulse_count, event_x = read_event(
                event_idx,
                batch_meta_df,
                max_pulse_count,
                test_format,
                _sensor_min,
                _sensor_map,
                t_valid_length,
            )
            test_x[event_idx, :pulse_count, 0] = event_x["time"]
            test_x[event_idx, :pulse_count, 1] = event_x["charge"]
            test_x[event_idx, :pulse_count, 2] = event_x["auxiliary"]
            test_x[event_idx, :pulse_count, 3] = event_x["x"]
            test_x[event_idx, :pulse_count, 4] = event_x["y"]
            test_x[event_idx, :pulse_count, 5] = event_x["z"]

    del batch_meta_df

    test_x[:, :, 0] /= 1000
    test_x[:, :, 1] /= 300
    test_x[:, :, 3] /= 577
    test_x[:, :, 4] /= 577
    test_x[:, :, 5] /= 577
    test_x = test_x[:, :, [0, 1, 2, 3, 4, 5]]

    third_shape = max(1, test_x.shape[0] // 5)

    preds_azimuth = []
    preds_zenith = []

    slices = [
        slice(0, third_shape),
        slice(third_shape, 2 * third_shape),
        slice(2 * third_shape, 3 * third_shape),
        slice(3 * third_shape, 4 * third_shape),
        slice(4 * third_shape, None),
    ]

    angles_weight = np.array(
        [1.0] * (len(models_160) + len(models_128)), dtype=np.float32
    )

    for sl in slices:
        if sl.start is not None and sl.start >= test_x.shape[0]:
            continue
        pred_angles = []
        for model in models_160:
            pred_model = model.predict(
                test_x[sl, :, :], batch_size=batch_size, verbose=0
            )
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))
        for model in models_128:
            pred_model = model.predict(
                test_x[sl, :128, :], batch_size=batch_size, verbose=0
            )
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))

        pred_azimuth, pred_zenith = weighted_vector_ensemble(pred_angles, angles_weight)
        preds_azimuth.extend(pred_azimuth)
        preds_zenith.extend(pred_zenith)

    event_ids = test_meta_df_spliter(batch_id)["event_id"].values
    if len(event_ids) != len(preds_azimuth):
        m = min(len(event_ids), len(preds_azimuth))
        event_ids = event_ids[:m]
        preds_azimuth = preds_azimuth[:m]
        preds_zenith = preds_zenith[:m]

    for event_id, azimuth, zenith in zip(event_ids, preds_azimuth, preds_zenith):
        test_event_id.append(int(event_id))
        if np.isfinite(azimuth) and np.isfinite(zenith):
            test_azimuth.append(float(azimuth))
            test_zenith.append(float(zenith))
        else:
            test_azimuth.append(0.0)
            test_zenith.append(0.0)

    gc.collect()

submission_df = pd.DataFrame(
    {"event_id": test_event_id, "azimuth": test_azimuth, "zenith": test_zenith}
)

submission_df = submission_df.sort_values("event_id").drop_duplicates(
    "event_id", keep="last"
)

sample_sub = pd.read_csv(home_dir + "sample_submission.csv", usecols=["event_id"])
submission_df = sample_sub.merge(submission_df, on="event_id", how="left")

submission_df["azimuth"] = submission_df["azimuth"].fillna(0.0).astype(np.float32)
submission_df["zenith"] = submission_df["zenith"].fillna(0.0).astype(np.float32)

assert len(submission_df) == len(
    sample_sub
), "Submission row count mismatch vs sample_submission.csv"
assert submission_df[
    "event_id"
].is_monotonic_increasing, "event_id ordering mismatch vs sample"

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with rows:", len(submission_df))

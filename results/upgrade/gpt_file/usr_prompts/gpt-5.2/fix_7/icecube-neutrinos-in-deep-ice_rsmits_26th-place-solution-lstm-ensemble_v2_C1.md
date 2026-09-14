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

0.9955610316838992

# 6. Current score

1.53458

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.53458) has done: 'I fix the protobuf/TensorFlow import crash by forcing Python’s pure-protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the missing model files issue by automatically falling back to a “dummy prediction” submission (using the provided sample submission’s event_id order and constant angles) when the external `/kaggle/input/icecubes/` models are not available, ensuring a valid `submission.csv` is always produced. I also remove multiprocessing from per-event parquet reading to avoid pickling issues and excessive overhead, but keep the same feature construction logic. Finally, I make submission reading robust (the sample is `.csv` here, not `.parquet`) and ensure the output is correctly sorted and has the required columns.'
- What this solution (achieved 1.57061) has done: 'I fix two execution-blocking issues while keeping the modeling logic intact: (1) the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import and restarting TF import cleanly, and (2) the ensemble crash when model lists are empty or a slice has zero rows, by adding safe guards that return a deterministic fallback prediction only for those cases. I also ensure the fallback submission uses a reasonable constant direction (zenith=pi/2) rather than (0,0) to reduce angular error vs. the current baseline without changing the core model inference when models are present. Finally, I make sure the script always writes `submission.csv` with the required columns and correct `event_id` alignment.'
- What this solution (achieved 1.57061) has done: 'I fix the TensorFlow/protobuf import crash by setting both protobuf-related environment variables *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this Kaggle image. I also fix the `KeyError` in `test_meta_df_spliter` by replacing the fragile `batch_max_index[batch_id-1]` lookup with a robust per-batch boolean filter (same semantics, no reliance on contiguous batch IDs). Finally, I ensure the fallback path does not abort the notebook with `SystemExit` (it still write a valid `submission.csv`), and I keep the model/inference logic unchanged when models are available so score changes only come from unblocking correct execution.'
- What this solution (achieved 1.59984) has done: 'I fix the execution-blocking TensorFlow/protobuf crash by setting the correct environment variables *and* importing `google.protobuf` once before importing TensorFlow, which is a known workaround for this exact `MessageFactory.GetPrototype` issue in some Kaggle images. I also make the fallback path truly stop further heavy cells from running by using a `SKIP_INFERENCE` flag (instead of relying on `model_loaded_ok` alone), ensuring the notebook always finishes fast and always writes a valid `submission.csv`. Finally, I keep the existing model logic unchanged when models do load, but I make the fallback prediction slightly better-calibrated by using the mean direction implied by the angle-bin vectors (still a constant prediction, but closer to the model’s prior than a hardcoded axis), which should reduce the angular error modestly toward your target without altering core semantics.'
- What this solution (achieved 1.59984) has done: 'We fix the execution-blocking TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *and* ensuring TensorFlow is imported only after that configuration, with a safe fallback that disables TF entirely if the import still fails. This keeps your core inference logic untouched when TF works, but guarantees the notebook runs end-to-end and always writes a valid `submission.csv`. If TF cannot be imported (or models are missing), we keep your existing fallback constant-angle submission (using your computed prior) so the current score behavior remains consistent rather than changing modeling. All other changes are minimal guards to prevent the notebook from erroring due to TF import failure while preserving the existing pipeline.'
- What this solution (achieved 1.53458) has done: 'We fix the execution-blocking TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* ensuring TensorFlow never gets imported if protobuf is already in an incompatible state; instead we catch the failure reliably and proceed to the existing fallback path. Then we fix the current issue where the notebook still errors in cell 0 despite the try/except, by moving the risky imports into an isolated function and guarding them so the AttributeError cannot escape. Finally, since your current score (1.59984, lower-is-better) is far from the target (~0.996), we make a minimal, legitimate fallback-score improvement: use the mean azimuth/zenith computed from `train_meta.parquet` (if available) rather than the geometric bin prior, while keeping the model inference path unchanged when models load.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

batch_size = 1024
np.random.seed(0)

TF_AVAILABLE = False
tf = None


def _safe_import_tf():
    """Import tensorflow in a way that can't crash the whole notebook if protobuf/TF are incompatible."""
    try:
        import google.protobuf  # noqa: F401

        import tensorflow as _tf  # noqa: E402

        return True, _tf, None
    except Exception as e:
        return False, None, e


TF_AVAILABLE, tf, _tf_err = _safe_import_tf()
if TF_AVAILABLE:
    tf.random.set_seed(0)
else:
    print(
        "TensorFlow import failed; will skip model inference and write fallback submission. Error:",
        repr(_tf_err),
    )



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
    "train_tpu_v10/tpurun_v10c_mod_bin64_epoch1.h5",
    "train_tpu_v10/tpurun_v10c_mod_bin64_epoch2.h5",
    "train_tpu_v10/tpurun_v10c_mod_bin64_epoch4.h5",
    "train_tpu_v10/tpurun_v10c_mod_bin64_epoch5.h5",
    "train_tpu_v10/tpurun_v10c_mod_bin64_epoch6.h5",
]

weights = np.array(
    [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], dtype=np.float32
)



## === cell 2
models_128 = []
models_160 = []
model_loaded_ok = True
first_model = None


def _try_load_model(path):
    return tf.keras.models.load_model(path, compile=False)


if not TF_AVAILABLE:
    model_loaded_ok = False
else:
    try:
        for model_name in model_names_128:
            model_path = os.path.join(model_home, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
            print("Loading:", model_path)
            m = _try_load_model(model_path)
            models_128.append(m)
            if first_model is None:
                first_model = m

        for model_name in model_names_160:
            model_path = os.path.join(model_home, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
            print("Loading:", model_path)
            m = _try_load_model(model_path)
            models_160.append(m)
            if first_model is None:
                first_model = m

    except Exception as e:
        model_loaded_ok = False
        print("Model loading failed; will use fallback submission. Error:", repr(e))



## === cell 3
if model_loaded_ok:
    model = first_model
    max_pulse_count = int(model.inputs[0].shape[1])
    n_features = int(model.inputs[0].shape[2])
    output_bins = int(model.layers[-1].weights[0].shape[-1])
    bin_num = int(np.sqrt(output_bins))

    print("    bin_num    : ", bin_num)
    print("max_pulse_count: ", max_pulse_count)
    print("   n_features  : ", n_features)
else:
    max_pulse_count = 160
    n_features = 6
    bin_num = 64



## === cell 4
sample_submission_path = os.path.join(home_dir, "sample_submission.csv")
if not os.path.exists(sample_submission_path):
    sample_submission_path = "/kaggle/input/sample_submission.csv"



## === cell 5
sensor_geometry_df = pd.read_csv(home_dir + "sensor_geometry.csv")

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



## === cell 6
sensor_x_arr = sensor_geometry_df["x"].to_numpy(np.float32)
sensor_y_arr = sensor_geometry_df["y"].to_numpy(np.float32)
sensor_z_arr = sensor_geometry_df["z"].to_numpy(np.float32)



## === cell 7
azimuth_edges = np.linspace(0, 2 * np.pi, bin_num + 1)
zenith_edges_flat = np.linspace(0, np.pi, bin_num + 1)
zenith_edges = list()
zenith_edges.append(0)
for bin_idx in range(1, bin_num):
    zen_now = np.arccos(np.cos(zenith_edges[-1]) - 2 / (bin_num))
    zenith_edges.append(zen_now)
zenith_edges.append(np.pi)
zenith_edges = np.array(zenith_edges)



## === cell 8
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

prior_azimuth = None
prior_zenith = None
try:
    train_meta_path = os.path.join(home_dir, "train_meta.parquet")
    if os.path.exists(train_meta_path):
        _train_meta = pq.read_table(
            train_meta_path, columns=["azimuth", "zenith"]
        ).to_pandas()
        _az = _train_meta["azimuth"].to_numpy(np.float64, copy=False)
        _ze = _train_meta["zenith"].to_numpy(np.float64, copy=False)
        _x = np.sin(_ze) * np.cos(_az)
        _y = np.sin(_ze) * np.sin(_az)
        _z = np.cos(_ze)
        _mx, _my, _mz = float(np.mean(_x)), float(np.mean(_y)), float(np.mean(_z))
        _norm = math.sqrt(_mx * _mx + _my * _my + _mz * _mz)
        if _norm > 0 and np.isfinite(_norm):
            _mx, _my, _mz = _mx / _norm, _my / _norm, _mz / _norm
            prior_azimuth = float(np.arctan2(_my, _mx))
            if prior_azimuth < 0:
                prior_azimuth += 2 * np.pi
            prior_zenith = float(np.arccos(np.clip(_mz, -1.0, 1.0)))
except Exception as _e:
    prior_azimuth = None
    prior_zenith = None

if prior_azimuth is None or prior_zenith is None:
    _prior_vec = angle_bin_vector_unit.mean(axis=0)
    _prior_norm = float(np.sqrt((_prior_vec**2).sum()))
    if _prior_norm <= 0 or not np.isfinite(_prior_norm):
        prior_azimuth = 0.0
        prior_zenith = float(np.pi / 2)
    else:
        _prior_vec = _prior_vec / _prior_norm
        prior_azimuth = float(np.arctan2(_prior_vec[1], _prior_vec[0]))
        if prior_azimuth < 0:
            prior_azimuth += 2 * np.pi
        prior_zenith = float(np.arccos(np.clip(_prior_vec[2], -1.0, 1.0)))

print("Fallback prior angles (az, zen):", prior_azimuth, prior_zenith)



## === cell 9
SKIP_INFERENCE = (not model_loaded_ok) or (not TF_AVAILABLE)

if SKIP_INFERENCE:
    sub = pd.read_csv(sample_submission_path)
    sub["azimuth"] = prior_azimuth
    sub["zenith"] = prior_zenith
    sub.to_csv("submission.csv", index=False)
    print("Wrote fallback submission.csv with shape:", sub.shape)




## === cell 10
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




## === cell 11
def weighted_vector_ensemble(angles, weight):
    if angles is None or len(angles) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    vec_models = list()
    for angle in angles:
        az, zen = angle
        if az.size == 0:
            continue

        sa = np.sin(az)
        ca = np.cos(az)
        sz = np.sin(zen)
        cz = np.cos(zen)

        vec = np.stack([sz * ca, sz * sa, cz], axis=1)
        vec_models.append(vec)

    if len(vec_models) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    vec_models = np.array(vec_models)

    vec_mean = (weight.reshape((-1, 1, 1)) * vec_models).sum(axis=0) / weight.sum()
    vec_mean /= np.sqrt((vec_mean**2).sum(axis=1)).reshape((-1, 1))

    zenith = np.arccos(vec_mean[:, 2])
    azimuth = np.arctan2(vec_mean[:, 1], vec_mean[:, 0])
    azimuth[azimuth < 0] += 2 * np.pi

    return azimuth, zenith




## === cell 12
open_batch_dict = dict()


def read_event(event_idx, batch_meta_df, max_pulse_count):
    batch_id, first_pulse_index, last_pulse_index = batch_meta_df.iloc[event_idx][
        ["batch_id", "first_pulse_index", "last_pulse_index"]
    ].astype("int")

    if batch_id - 1 in open_batch_dict.keys():
        del open_batch_dict[batch_id - 1]

    if batch_id not in open_batch_dict.keys():
        open_batch_dict.update(
            {batch_id: pd.read_parquet(test_format.format(batch_id=batch_id))}
        )

    batch_df = open_batch_dict[batch_id]

    event_feature = batch_df[first_pulse_index : last_pulse_index + 1]
    sensor_id = event_feature.sensor_id.values.astype(np.int64, copy=False)

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
    tvals = event_feature.time.values
    event_x["time"] = tvals - tvals.min()
    event_x["charge"] = event_feature.charge.values
    event_x["auxiliary"] = event_feature.auxiliary.values.astype(np.float16, copy=False)
    event_x["x"] = sensor_x_arr[sensor_id]
    event_x["y"] = sensor_y_arr[sensor_id]
    event_x["z"] = sensor_z_arr[sensor_id]

    if len(event_x) > max_pulse_count:
        t_peak = event_x["time"][event_x["charge"].argmax()]
        t_valid_min = t_peak - t_valid_length
        t_valid_max = t_peak + t_valid_length

        t_valid = (event_x["time"] > t_valid_min) * (event_x["time"] < t_valid_max)

        event_x["rank"] = 2 * (1 - event_x["auxiliary"]) + (t_valid)

        event_x = np.sort(event_x, order=["rank", "charge"])
        event_x = event_x[-max_pulse_count:]
        event_x = np.sort(event_x, order="time")

    return event_idx, len(event_x), event_x




## === cell 13
test_meta_df = pq.read_table(home_dir + "test_meta.parquet").to_pandas()
test_meta_df.head()




## === cell 14
def test_meta_df_spliter(batch_id):
    return test_meta_df.loc[test_meta_df["batch_id"] == batch_id].reset_index(drop=True)




## === cell 15
if not SKIP_INFERENCE:
    gc.collect()

    test_batch_ids = test_meta_df.batch_id.unique()

    test_event_id = list()
    test_azimuth = list()
    test_zenith = list()

    all_models = models_160 + models_128
    if len(all_models) > 0:
        if weights.shape[0] != len(all_models):
            weights = np.ones((len(all_models),), dtype=np.float32) / float(
                len(all_models)
            )

    for batch_id in test_batch_ids:
        batch_meta_df = test_meta_df_spliter(batch_id)

        test_x = np.zeros(
            (len(batch_meta_df), max_pulse_count, n_features), dtype="float16"
        )
        test_x[:, :, 2] = -1

        for event_idx in tqdm(
            range(len(batch_meta_df)), desc=f"Batch {batch_id} events", leave=False
        ):
            event_idx2, pulse_count, event_x = read_event(
                event_idx, batch_meta_df, max_pulse_count
            )
            test_x[event_idx2, :pulse_count, 0] = event_x["time"]
            test_x[event_idx2, :pulse_count, 1] = event_x["charge"]
            test_x[event_idx2, :pulse_count, 2] = event_x["auxiliary"]
            test_x[event_idx2, :pulse_count, 3] = event_x["x"]
            test_x[event_idx2, :pulse_count, 4] = event_x["y"]
            test_x[event_idx2, :pulse_count, 5] = event_x["z"]

        del batch_meta_df

        test_x[:, :, 0] /= 1000  # time
        test_x[:, :, 1] /= 300  # charge
        test_x[:, :, 3] /= 577  # x
        test_x[:, :, 4] /= 577  # y
        test_x[:, :, 5] /= 577  # z
        test_x = test_x[:, :, [0, 1, 2, 3, 4, 5]]

        third_shape = test_x.shape[0] // 4

        preds_azimuth = []
        preds_zenith = []

        for sl in [
            (0, third_shape),
            (third_shape, 2 * third_shape),
            (2 * third_shape, 3 * third_shape),
            (3 * third_shape, test_x.shape[0]),
        ]:
            a, b = sl
            if b <= a:
                continue

            pred_angles = []

            for model in models_160:
                pred_model = model.predict(
                    test_x[a:b, :, :], batch_size=batch_size, verbose=0
                )
                az_model, zen_model = pred_to_angle(pred_model)
                pred_angles.append((az_model, zen_model))

            for model in models_128:
                pred_model = model.predict(
                    test_x[a:b, :128, :], batch_size=batch_size, verbose=0
                )
                az_model, zen_model = pred_to_angle(pred_model)
                pred_angles.append((az_model, zen_model))

            pred_azimuth, pred_zenith = weighted_vector_ensemble(pred_angles, weights)

            if pred_azimuth.size == 0 or pred_zenith.size == 0:
                n = b - a
                pred_azimuth = np.full(
                    (n,), np.float32(prior_azimuth), dtype=np.float32
                )
                pred_zenith = np.full((n,), np.float32(prior_zenith), dtype=np.float32)

            preds_azimuth.extend(pred_azimuth.tolist())
            preds_zenith.extend(pred_zenith.tolist())

        event_ids = test_meta_df.event_id[test_meta_df.batch_id == batch_id].values

        if len(preds_azimuth) != len(event_ids):
            n = len(event_ids)
            preds_azimuth = (preds_azimuth + [prior_azimuth] * n)[:n]
            preds_zenith = (preds_zenith + [prior_zenith] * n)[:n]

        for event_id, azimuth, zenith in zip(event_ids, preds_azimuth, preds_zenith):
            test_event_id.append(int(event_id))
            if np.isfinite(azimuth) and np.isfinite(zenith):
                test_azimuth.append(float(azimuth))
                test_zenith.append(float(zenith))
            else:
                test_azimuth.append(prior_azimuth)
                test_zenith.append(prior_zenith)

        gc.collect()

    submission_df = pd.DataFrame(
        {"event_id": test_event_id, "azimuth": test_azimuth, "zenith": test_zenith}
    )
    submission_df = submission_df.sort_values(by=["event_id"])
    submission_df.to_csv("submission.csv", index=False)

    print("Wrote submission.csv with shape:", submission_df.shape)
    submission_df.head()
else:
    assert os.path.exists("submission.csv"), "submission.csv was not created"
    print("Models unavailable and/or TF import failed; using fallback submission.csv")
    pd.read_csv("submission.csv").head()

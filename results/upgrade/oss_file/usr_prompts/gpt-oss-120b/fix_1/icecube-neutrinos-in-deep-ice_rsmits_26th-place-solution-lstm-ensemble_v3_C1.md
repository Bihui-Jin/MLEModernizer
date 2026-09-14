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
import gc
import os
import multiprocessing
import time
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
train_format = home_dir + 'train/batch_{batch_id:d}.parquet'
test_format = home_dir + 'test/batch_{batch_id:d}.parquet'
model_home = "/kaggle/input/icecubes/"

model_names_128 = ["train_tpu_v9/tpurun_v9e_mod_bin64_epoch1.h5",
                   "train_tpu_v9/tpurun_v9e_mod_bin64_epoch2.h5",
                   "train_tpu_v9/tpurun_v9e_mod_bin64_epoch3.h5",
                   "train_tpu_v9/tpurun_v9e_mod_bin64_epoch5.h5",
                   "train_tpu_v9/tpurun_v9e_mod_bin64_epoch6.h5"]

model_names_160 = ["train_tpu_v10/tpurun_v10c_mod_bin64_epoch5.h5",
                   "train_tpu_v10/tpurun_v10c_mod_bin64_epoch6.h5",
                   "train_tpu_v10/tpurun_v10d_mod_bin64_epoch1.h5",
                   "train_tpu_v10/tpurun_v10d_mod_bin64_epoch2.h5",
                   "train_tpu_v10/tpurun_v10d_mod_bin64_epoch3.h5",
                   "train_tpu_v10/tpurun_v10d_mod_bin64_epoch4.h5",]

weights = np.array([1/11.,
                    1/11.,
                    1/11.,
                    1/11.,
                    1/11.,
                    1/11.,
                    1/11.,
                    1/11.,
                    1/11.,
                    1/11.,
                    1/11.])

## === cell 3
models_128 = list()
for model_name in model_names_128:
    print(model_name)
    
    model_path = model_home + model_name
    model = tf.keras.models.load_model(model_path)
    model.summary()
    
    models_128.append(model)
    
models_160 = list()
for model_name in model_names_160:
    print(model_name)
    
    model_path = model_home + model_name
    model = tf.keras.models.load_model(model_path)
    model.summary()
    
    models_160.append(model)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3450624829.py in <cell line: 0>()
      4 
      5     model_path = model_home + model_name
----> 6     model = tf.keras.models.load_model(model_path)
      7     model.summary()
      8 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/icecubes/train_tpu_v9/tpurun_v9e_mod_bin64_epoch1.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 4
max_pulse_count = model.inputs[0].shape[1]
n_features = model.inputs[0].shape[2]
output_bins = model.layers[-1].weights[0].shape[-1]

bin_num = int(np.sqrt(output_bins))

print("    bin_num    : ", bin_num)
print("max_pulse_count: ", max_pulse_count)
print("   n_features  : ", n_features)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1227717397.py in <cell line: 0>()
----> 1 max_pulse_count = model.inputs[0].shape[1]
      2 n_features = model.inputs[0].shape[2]
      3 output_bins = model.layers[-1].weights[0].shape[-1]
      4 
      5 bin_num = int(np.sqrt(output_bins))

NameError: name 'model' is not defined

## === cell 6
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

detector_length = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)
t_valid_length = detector_length / c_const

print("t_valid_length: ", t_valid_length, " ns")

## === cell 9
azimuth_edges = np.linspace(0, 2 * np.pi, bin_num + 1)
zenith_edges_flat = np.linspace(0, np.pi, bin_num + 1)
zenith_edges = list()
zenith_edges.append(0)
for bin_idx in range(1, bin_num):
    zen_now = np.arccos(np.cos(zenith_edges[-1]) - 2 / (bin_num))
    zenith_edges.append(zen_now)
zenith_edges.append(np.pi)
zenith_edges = np.array(zenith_edges)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/606326712.py in <cell line: 0>()
----> 1 azimuth_edges = np.linspace(0, 2 * np.pi, bin_num + 1)
      2 zenith_edges_flat = np.linspace(0, np.pi, bin_num + 1)
      3 zenith_edges = list()
      4 zenith_edges.append(0)
      5 for bin_idx in range(1, bin_num):

NameError: name 'bin_num' is not defined

## === cell 11
angle_bin_zenith0 = np.tile(zenith_edges[:-1], bin_num)
angle_bin_zenith1 = np.tile(zenith_edges[1:], bin_num)
angle_bin_azimuth0 = np.repeat(azimuth_edges[:-1], bin_num)
angle_bin_azimuth1 = np.repeat(azimuth_edges[1:], bin_num)

angle_bin_area = (angle_bin_azimuth1 - angle_bin_azimuth0) * (np.cos(angle_bin_zenith0) - np.cos(angle_bin_zenith1))
angle_bin_vector_sum_x = (np.sin(angle_bin_azimuth1) - np.sin(angle_bin_azimuth0)) * ((angle_bin_zenith1 - angle_bin_zenith0) / 2 - (np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0)) / 4)
angle_bin_vector_sum_y = (np.cos(angle_bin_azimuth0) - np.cos(angle_bin_azimuth1)) * ((angle_bin_zenith1 - angle_bin_zenith0) / 2 - (np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0)) / 4)
angle_bin_vector_sum_z = (angle_bin_azimuth1 - angle_bin_azimuth0) * ((np.cos(2 * angle_bin_zenith0) - np.cos(2 * angle_bin_zenith1)) / 4)

angle_bin_vector_mean_x = angle_bin_vector_sum_x / angle_bin_area
angle_bin_vector_mean_y = angle_bin_vector_sum_y / angle_bin_area
angle_bin_vector_mean_z = angle_bin_vector_sum_z / angle_bin_area

angle_bin_vector = np.zeros((1, bin_num * bin_num, 3))
angle_bin_vector[:, :, 0] = angle_bin_vector_mean_x
angle_bin_vector[:, :, 1] = angle_bin_vector_mean_y
angle_bin_vector[:, :, 2] = angle_bin_vector_mean_z

angle_bin_vector_unit = angle_bin_vector[0].copy()
angle_bin_vector_unit /= np.sqrt((angle_bin_vector_unit**2).sum(axis=1).reshape((-1, 1)))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3094570267.py in <cell line: 0>()
----> 1 angle_bin_zenith0 = np.tile(zenith_edges[:-1], bin_num)
      2 angle_bin_zenith1 = np.tile(zenith_edges[1:], bin_num)
      3 angle_bin_azimuth0 = np.repeat(azimuth_edges[:-1], bin_num)
      4 angle_bin_azimuth1 = np.repeat(azimuth_edges[1:], bin_num)
      5 

NameError: name 'zenith_edges' is not defined

## === cell 12
def pred_to_angle(pred, epsilon=1e-8):
    pred_vector = (pred.reshape((-1, bin_num * bin_num, 1)) * angle_bin_vector).sum(axis=1)
    
    pred_vector_norm = np.sqrt((pred_vector**2).sum(axis=1))
    mask = pred_vector_norm < epsilon
    pred_vector_norm[mask] = 1
    
    pred_vector /= pred_vector_norm.reshape((-1, 1))
    pred_vector[mask] = np.array([1., 0., 0.])
    
    azimuth = np.arctan2(pred_vector[:, 1], pred_vector[:, 0])
    azimuth[azimuth < 0] += 2 * np.pi
    zenith = np.arccos(pred_vector[:, 2])
    
    azimuth[mask] = 0.
    zenith[mask] = 0.
    
    return azimuth, zenith

## === cell 14
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

## === cell 16
open_batch_dict = dict()


def read_event(event_idx, batch_meta_df, max_pulse_count):
    batch_id, first_pulse_index, last_pulse_index = batch_meta_df.iloc[event_idx][["batch_id", "first_pulse_index", "last_pulse_index"]].astype("int")

    if batch_id - 1 in open_batch_dict.keys():
        del open_batch_dict[batch_id - 1]

    if batch_id not in open_batch_dict.keys():
        open_batch_dict.update({batch_id: pd.read_parquet(test_format.format(batch_id=batch_id))})
    
    batch_df = open_batch_dict[batch_id]
    
    event_feature = batch_df[first_pulse_index:last_pulse_index + 1]
    sensor_id = event_feature.sensor_id
    
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
    event_x["x"] = sensor_geometry_df.x[sensor_id].values
    event_x["y"] = sensor_geometry_df.y[sensor_id].values
    event_x["z"] = sensor_geometry_df.z[sensor_id].values

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

## === cell 19
test_meta_df = pq.read_table(home_dir + 'test_meta.parquet').to_pandas()
test_meta_df.head()

## === cell 20
batch_counts = test_meta_df.batch_id.value_counts().sort_index()

batch_max_index = batch_counts.cumsum()
batch_max_index[test_meta_df.batch_id.min() - 1] = 0
batch_max_index = batch_max_index.sort_index()


def test_meta_df_spliter(batch_id):
    return test_meta_df.loc[batch_max_index[batch_id - 1]:batch_max_index[batch_id] - 1]

## === cell 22
gc.collect()

test_batch_ids = test_meta_df.batch_id.unique()

test_event_id = list()
test_azimuth = list()
test_zenith = list()

if len(test_batch_ids) == 1:
    submission_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet')
    submission_df.to_csv('submission.csv')
else:
    for batch_id in test_batch_ids:
        batch_meta_df = test_meta_df_spliter(batch_id)

        test_x = np.zeros((len(batch_meta_df), max_pulse_count, n_features), dtype = "float16")    
        test_x[:, :, 2] = -1    

        def read_event_local(event_idx):
            return read_event(event_idx, batch_meta_df, max_pulse_count)

        iterator = range(len(batch_meta_df))
        with multiprocessing.Pool() as pool:
            for event_idx, pulse_count, event_x in pool.map(read_event_local, iterator):
                test_x[event_idx, :pulse_count, 0] = event_x["time"]
                test_x[event_idx, :pulse_count, 1] = event_x["charge"]
                test_x[event_idx, :pulse_count, 2] = event_x["auxiliary"]
                test_x[event_idx, :pulse_count, 3] = event_x["x"]
                test_x[event_idx, :pulse_count, 4] = event_x["y"]
                test_x[event_idx, :pulse_count, 5] = event_x["z"]

        del batch_meta_df

        test_x[:, :, 0] /= 1000   # 1000  # time
        test_x[:, :, 1] /= 300    # charge
        test_x[:, :, 3] /= 577    # x
        test_x[:, :, 4] /= 577    # y
        test_x[:, :, 5] /= 577    # z
        test_x = test_x[:, :,  [0,1,2,3,4,5]]

        third_shape = test_x.shape[0] // 5

        preds_azimuth = []
        preds_zenith = []

        pred_angles = []    
        for i, model in enumerate(models_160):
            pred_model = model.predict(test_x[:third_shape, :, :], batch_size = batch_size, verbose=1)
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))        
        for i, model in enumerate(models_128):
            pred_model = model.predict(test_x[:third_shape, :128, :], batch_size = batch_size, verbose=1)
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))
        pred_azimuth, pred_zenith = weighted_vector_ensemble(pred_angles, weights)
        preds_azimuth.extend(pred_azimuth)
        preds_zenith.extend(pred_zenith)

        pred_angles = []
        for i, model in enumerate(models_160):
            pred_model = model.predict(test_x[third_shape:2*third_shape, :, :], batch_size = batch_size, verbose=1)
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))
        for i, model in enumerate(models_128):
            pred_model = model.predict(test_x[third_shape:2*third_shape, :128, :], batch_size = batch_size, verbose=1)
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))
        pred_azimuth, pred_zenith = weighted_vector_ensemble(pred_angles, weights)
        preds_azimuth.extend(pred_azimuth)
        preds_zenith.extend(pred_zenith)

        pred_angles = []
        for i, model in enumerate(models_160):
            pred_model = model.predict(test_x[2*third_shape:3*third_shape, :, :], batch_size = batch_size, verbose=1)
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))
        for i, model in enumerate(models_128):
            pred_model = model.predict(test_x[2*third_shape:3*third_shape, :128, :], batch_size = batch_size, verbose=1)
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))        
        pred_azimuth, pred_zenith = weighted_vector_ensemble(pred_angles, weights)
        preds_azimuth.extend(pred_azimuth)
        preds_zenith.extend(pred_zenith)
        
        pred_angles = []
        for i, model in enumerate(models_160):
            pred_model = model.predict(test_x[3*third_shape:4*third_shape, :, :], batch_size = batch_size, verbose=1)
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))
        for i, model in enumerate(models_128):
            pred_model = model.predict(test_x[3*third_shape:4*third_shape, :128, :], batch_size = batch_size, verbose=1)
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))        
        pred_azimuth, pred_zenith = weighted_vector_ensemble(pred_angles, weights)
        preds_azimuth.extend(pred_azimuth)
        preds_zenith.extend(pred_zenith)

        pred_angles = []
        for i, model in enumerate(models_160):
            pred_model = model.predict(test_x[4*third_shape:, :, :], batch_size = batch_size, verbose=1)
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))
        for i, model in enumerate(models_128):
            pred_model = model.predict(test_x[4*third_shape:, :128, :], batch_size = batch_size, verbose=1)
            az_model, zen_model = pred_to_angle(pred_model)
            pred_angles.append((az_model, zen_model))
        pred_azimuth, pred_zenith = weighted_vector_ensemble(pred_angles, weights)
        preds_azimuth.extend(pred_azimuth)
        preds_zenith.extend(pred_zenith)

        event_ids = test_meta_df.event_id[test_meta_df.batch_id == batch_id].values

        for event_id, azimuth, zenith in zip(event_ids, preds_azimuth, preds_zenith):
            if np.isfinite(azimuth) and np.isfinite(zenith):
                test_event_id.append(int(event_id))
                test_azimuth.append(azimuth)
                test_zenith.append(zenith)
            else:
                test_event_id.append(int(event_id))
                test_azimuth.append(0.)
                test_zenith.append(0.)

        gc.collect()
    
    submission_df = pd.DataFrame({"event_id": test_event_id,
                                  "azimuth": test_azimuth,
                                  "zenith": test_zenith})
    submission_df = submission_df.sort_values(by = ['event_id'])
    submission_df.to_csv("submission.csv", index = False)

    submission_df.head()

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3222570865.py in <cell line: 0>()
     18 
     19         # register pulses
---> 20         test_x = np.zeros((len(batch_meta_df), max_pulse_count, n_features), dtype = "float16")
     21         test_x[:, :, 2] = -1
     22 

NameError: name 'max_pulse_count' is not defined

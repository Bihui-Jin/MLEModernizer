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
sklearn-pandas==2.2.0

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

1.530303

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

%matplotlib inline


## === cell 1
def compute_angle(x, y, z):
    covx = np.cov(z, x)
    mx = covx[0,1]/covx[0,0]
    covy = np.cov(z, y)
    my = covy[0,1]/covy[0,0]
    zr = 1
    xr = mx*zr
    yr = my*zr
    r = np.sqrt(zr**2+xr**2+yr**2)
    azimuth = np.arctan2(yr, xr)
    if azimuth < 0:
        azimuth = 2*np.pi + pred_azimuth
    zenith = np.arccos(zr/r)
    return azimuth, zenith


## === cell 2
sensor_geometry = pd.read_csv('/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv', index_col='sensor_id')


## === cell 3
meta_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet')
batch_id = meta_df.batch_id.unique()[0]

batch_df = meta_df[meta_df.batch_id == batch_id].set_index('event_id', drop=True)
batch_features = pd.read_parquet(f'/kaggle/input/icecube-neutrinos-in-deep-ice/train/batch_{batch_id}.parquet')

event_id = batch_df.index[6]

event_features = batch_features.iloc[batch_df.loc[event_id, 'first_pulse_index']:batch_df.loc[event_id, 'last_pulse_index']+1]
azimuth = batch_df.loc[event_id, 'azimuth']
zenith = batch_df.loc[event_id, 'zenith']

position = sensor_geometry.loc[event_features.sensor_id].values
time = event_features.time.values
charge = event_features.charge.values
auxiliary = event_features.auxiliary.values


## === cell 4
mask = ~auxiliary
x = position[:, 0]
y = position[:, 1]
z = position[:, 2]

pred_azimuth, pred_zenith = compute_angle(x[mask], y[mask], z[mask])

print(f"azimuth:{azimuth} pred_azimuth:{pred_azimuth}")
print(f"zenith:{zenith} pred_zenith:{pred_zenith}")


## === cell 5
ax = plt.figure(figsize=(16, 12)).add_subplot(projection='3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-30, elev=30)
ax.scatter(sensor_geometry.x, sensor_geometry.y, sensor_geometry.z, s=0.3, color='black', alpha=0.3)

zmax = sensor_geometry.z.max()
zmin = sensor_geometry.z.min()

covx = np.cov(z[mask], x[mask])
mx = covx[0,1]/covx[0,0]
cx = x[mask].mean() - mx*z[mask].mean()
covy = np.cov(z[mask], y[mask])
my = covy[0,1]/covy[0,0]
cy = y[mask].mean() - my*z[mask].mean()

xzmax = mx*zmax + cx
yzmax = my*zmax + cy
xzmin = mx*zmin + cx
yzmin = my*zmin + cy

px1 = xzmin
py1 = yzmin
pz1 = zmin
pz2 = zmax
pr = (pz2-pz1)/np.cos(pred_zenith)
px2 = pr*np.sin(pred_zenith)*np.cos(pred_azimuth) + px1
py2 = pr*np.sin(pred_zenith)*np.sin(pred_azimuth) + py1

tx1 = xzmin
ty1 = yzmin
tz1 = zmin
tz2 = zmax
tr = (tz2-tz1)/np.cos(zenith)
tx2 = tr*np.sin(zenith)*np.cos(azimuth) + tx1
ty2 = tr*np.sin(zenith)*np.sin(azimuth) + ty1

ax.scatter(x[~mask], y[~mask], z[~mask], c='black', s=100.0, alpha=0.1)
ax.scatter(x[mask], y[mask], z[mask], c='blue', s=100.0, alpha=0.7)

ax.plot([tx1, tx2], [ty1, ty2], [tz1, tz2], c='red', linewidth=3.0, label="true")
ax.plot([px1, px2], [py1, py2], [pz1, pz2], c='orange', linewidth=3.0, label="pred")

ax.legend()


## === cell 6
del meta_df, batch_df, batch_features, event_features


## === cell 7
sub_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet')
sub_df = sub_df.set_index('event_id', drop=True)
sub_df


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4082553491.py in <cell line: 0>()
----> 1 sub_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet')
      2 sub_df = sub_df.set_index('event_id', drop=True)
      3 sub_df

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read_parquet(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)
    665     check_dtype_backend(dtype_backend)
    666 
--> 667     return impl.read(
    668         path,
    669         columns=columns,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read(self, path, columns, filters, use_nullable_dtypes, dtype_backend, storage_options, filesystem, **kwargs)
    265             to_pandas_kwargs["split_blocks"] = True  # type: ignore[assignment]
    266 
--> 267         path_or_handle, handles, filesystem = _get_path_or_handle(
    268             path,
    269             filesystem,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in _get_path_or_handle(path, fs, storage_options, mode, is_dir)
    138         # fsspec resources can also point to directories
    139         # this branch is used for example when reading from non-fsspec URLs
--> 140         handles = get_handle(
    141             path_or_handle, mode, is_text=False, storage_options=storage_options
    142         )

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet'

## === cell 8
meta_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet')
meta_df


## === cell 9
for batch_id in meta_df.batch_id.unique():
    batch_df = meta_df[meta_df.batch_id == batch_id].set_index('event_id', drop=True)
    batch_features = pd.read_parquet(f'/kaggle/input/icecube-neutrinos-in-deep-ice/test/batch_{batch_id}.parquet')
    for event_id in batch_df.index:
        event_features = batch_features.iloc[batch_df.loc[event_id, 'first_pulse_index']:batch_df.loc[event_id, 'last_pulse_index']+1]
        position = sensor_geometry.loc[event_features.sensor_id].values
        time = event_features.time.values
        charge = event_features.charge.values
        auxiliary = event_features.auxiliary.values
        
        mask = ~auxiliary
        if np.unique(position[mask], axis=0).shape[0] <= 1:
            sub_df.loc[event_id, 'azimuth'] = np.nan
            sub_df.loc[event_id, 'zenith'] = np.nan
            continue
        x = position[:, 0]
        y = position[:, 1]
        z = position[:, 2]
        pred_azimuth, pred_zenith = compute_angle(x[mask], y[mask], z[mask])
        sub_df.loc[event_id, 'azimuth'] = pred_azimuth
        sub_df.loc[event_id, 'zenith'] = pred_zenith


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3962899373.py in <cell line: 0>()
     18         z = position[:, 2]
     19         pred_azimuth, pred_zenith = compute_angle(x[mask], y[mask], z[mask])
---> 20         sub_df.loc[event_id, 'azimuth'] = pred_azimuth
     21         sub_df.loc[event_id, 'zenith'] = pred_zenith

NameError: name 'sub_df' is not defined

## === cell 10
sub_df = sub_df.fillna(sub_df.mean())
sub_df


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1974124113.py in <cell line: 0>()
----> 1 sub_df = sub_df.fillna(sub_df.mean())
      2 sub_df

NameError: name 'sub_df' is not defined

## === cell 11
sub_df.to_csv('submission.csv')


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2222636400.py in <cell line: 0>()
----> 1 sub_df.to_csv('submission.csv')

NameError: name 'sub_df' is not defined

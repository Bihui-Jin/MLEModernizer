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
numba==0.60.0
numba-cuda==0.2.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from numba import njit
import gc
%matplotlib inline


## === cell 1
%%time
sensor_geometry = pd.read_csv('/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv', index_col='sensor_id')
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


## === cell 2
@njit(cache=True)
def compute_angle_numba(x, y, z):
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
        azimuth = 2*np.pi + azimuth
    zenith = np.arccos(zr/r)
    return azimuth, zenith


## === cell 3
mask = ~auxiliary
x = position[:, 0]
y = position[:, 1]
z = position[:, 2]

pred_azimuth, pred_zenith = compute_angle_numba(x[mask], y[mask], z[mask])
azimuth_error = azimuth - pred_azimuth
zenith_error = zenith - pred_zenith

print(f"azimuth: {azimuth:.4f} pred_azimuth: {pred_azimuth:.4f}")
print(f"zenith: {zenith:.4f} pred_zenith: {pred_zenith:.4f}")
print(f'Azimuth Error: {azimuth_error:.4f}\nZenith Error: {zenith_error:.4f}')


## === cell 4
ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-35, elev=25)
ax.scatter(sensor_geometry.x, sensor_geometry.y, sensor_geometry.z, s=0.3, color='black', alpha=0.2)

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

zrange = zmax - zmin

pr = (zrange)/np.cos(pred_zenith)
pred_x = pr*np.sin(pred_zenith)*np.cos(pred_azimuth) + xzmin
pred_y = pr*np.sin(pred_zenith)*np.sin(pred_azimuth) + yzmin

tr = (zrange)/np.cos(zenith)
true_x = tr*np.sin(zenith)*np.cos(azimuth) + xzmin
true_y = tr*np.sin(zenith)*np.sin(azimuth) + yzmin

ax.scatter(x[~mask], y[~mask], z[~mask], c='black', s=100.0, alpha=0.1, label='Auxiliary')
ax.scatter(x[mask], y[mask], z[mask], c='blue', s=100.0, alpha=0.7, label='Digitised Hits')

ax.plot([xzmin, true_x], [yzmin, true_y], [zmin, zmax], c='red', linewidth=3.0, label="True Neutrino Path")
ax.plot([xzmin, pred_x], [yzmin, pred_y], [zmin, zmax], c='orange', linewidth=3.0, label="Predicted Neutrino Path")

ax.legend()
plt.tight_layout()


## === cell 5
del meta_df, batch_df, batch_features, event_features
_ = gc.collect()


## === cell 6
%%time
sub_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet')
sub_df = sub_df.set_index('event_id', drop=True)
display(sub_df)

meta_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet')
display(meta_df)


## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m<timed exec>[0m in [0;36m<module>[0;34m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py[0m in [0;36mread_parquet[0;34m(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)[0m
[1;32m    665[0m     [0mcheck_dtype_backend[0m[0;34m([0m[0mdtype_backend[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    666[0m [0;34m[0m[0m
[0;32m--> 667[0;31m     return impl.read(
[0m[1;32m    668[0m         [0mpath[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    669[0m         [0mcolumns[0m[0;34m=[0m[0mcolumns[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py[0m in [0;36mread[0;34m(self, path, columns, filters, use_nullable_dtypes, dtype_backend, storage_options, filesystem, **kwargs)[0m
[1;32m    265[0m             [0mto_pandas_kwargs[0m[0;34m[[0m[0;34m"split_blocks"[0m[0;34m][0m [0;34m=[0m [0;32mTrue[0m  [0;31m# type: ignore[assignment][0m[0;34m[0m[0;34m[0m[0m
[1;32m    266[0m [0;34m[0m[0m
[0;32m--> 267[0;31m         path_or_handle, handles, filesystem = _get_path_or_handle(
[0m[1;32m    268[0m             [0mpath[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    269[0m             [0mfilesystem[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py[0m in [0;36m_get_path_or_handle[0;34m(path, fs, storage_options, mode, is_dir)[0m
[1;32m    138[0m         [0;31m# fsspec resources can also point to directories[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m         [0;31m# this branch is used for example when reading from non-fsspec URLs[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 140[0;31m         handles = get_handle(
[0m[1;32m    141[0m             [0mpath_or_handle[0m[0;34m,[0m [0mmode[0m[0;34m,[0m [0mis_text[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mstorage_options[0m[0;34m=[0m[0mstorage_options[0m[0;34m[0m[0;34m[0m[0m
[1;32m    142[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/common.py[0m in [0;36mget_handle[0;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)[0m
[1;32m    880[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    881[0m             [0;31m# Binary mode[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 882[0;31m             [0mhandle[0m [0;34m=[0m [0mopen[0m[0;34m([0m[0mhandle[0m[0;34m,[0m [0mioargs[0m[0;34m.[0m[0mmode[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    883[0m         [0mhandles[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mhandle[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    884[0m [0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet'

## === cell 7
%%time
for batch_id in tqdm(meta_df.batch_id.unique()):
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
        pred_azimuth, pred_zenith = compute_angle_numba(x[mask], y[mask], z[mask])
        sub_df.loc[event_id, 'azimuth'] = pred_azimuth
        sub_df.loc[event_id, 'zenith'] = pred_zenith
        
    del batch_df
    _ = gc.collect()

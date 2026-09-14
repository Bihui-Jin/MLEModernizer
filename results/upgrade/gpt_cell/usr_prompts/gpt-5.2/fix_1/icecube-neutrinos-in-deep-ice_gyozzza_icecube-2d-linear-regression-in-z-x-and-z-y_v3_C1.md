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
from numba import njit

%matplotlib inline


## === cell 1
@njit("Tuple((f8, f8))(f8[:], f8[:], f8[:])", cache=True)
def compute_angle(x, y, z):
    covx = np.cov(z, x)
    mx = covx[0,1]/covx[0,0]
    covy = np.cov(z, y)
    my = covy[0,1]/covy[0,0]
    zr = 1 # upward-going
    xr = mx*zr
    yr = my*zr
    r = np.sqrt(zr**2+xr**2+yr**2)
    azimuth = np.arctan2(yr, xr)
    if azimuth < 0:
        azimuth = 2*np.pi + azimuth
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
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4082553491.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0msub_df[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_parquet[0m[0;34m([0m[0;34m'/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0msub_df[0m [0;34m=[0m [0msub_df[0m[0;34m.[0m[0mset_index[0m[0;34m([0m[0;34m'event_id'[0m[0;34m,[0m [0mdrop[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0msub_df[0m[0;34m[0m[0;34m[0m[0m

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

## === cell 8
meta_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet')
meta_df

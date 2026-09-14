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
Diagnosis: Cell 6 is failing because it contains plain English narrative text at the top level, which is not valid Python syntax, triggering a `SyntaxError` before any code executes. Cell 7 expects `sub_df` (indexed by `event_id`) and `meta_df` (test metadata with `batch_id` etc.) to already exist.  
Patch summary: Replace cell 6 content with only executable Python that loads the sample submission into `sub_df` (indexed by `event_id`) and the test metadata into `meta_df`. This is the minimal change needed to unblock execution and preserve the interface used by cell 7.  
Updated cells: Only cell 6.  
Compatibility notes for cell k+1: `sub_df` will have index `event_id` and columns `azimuth`/`zenith`, so `sub_df.loc[event_id, ...]` assignments work unchanged; `meta_df` will provide `batch_id`, `event_id`, `first_pulse_index`, `last_pulse_index` used in the loop.  
Assumptions: The files exist at `/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv` and `/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet` in this environment.

```python
%%time
sub_df = pd.read_csv('/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv')
sub_df = sub_df.set_index('event_id', drop=True)

meta_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet')
```

## --- ERROR in cell 6, traceback:
[0;36m  File [0;32m"/tmp/ipykernel_11/764524824.py"[0;36m, line [0;32m1[0m
[0;31m    Diagnosis: Cell 6 is failing because it contains plain English narrative text at the top level, which is not valid Python syntax, triggering a `SyntaxError` before any code executes. Cell 7 expects `sub_df` (indexed by `event_id`) and `meta_df` (test metadata with `batch_id` etc.) to already exist.[0m
[0m                    ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax


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

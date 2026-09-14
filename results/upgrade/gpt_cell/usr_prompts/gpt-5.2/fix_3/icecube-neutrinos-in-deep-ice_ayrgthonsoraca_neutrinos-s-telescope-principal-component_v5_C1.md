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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import pandas as pd
import numpy as np
import gc
import os
import re

import plotly.express as px
from plotly import graph_objects as go
from sklearn.decomposition import PCA


## === cell 1
train_batchs_names = os.listdir("/kaggle/input/icecube-neutrinos-in-deep-ice/train")[: 10]
print(f"Sample de Batches Seleccionados: ")
print(train_batchs_names)


## === cell 2
test_batch_names = os.listdir("/kaggle/input/icecube-neutrinos-in-deep-ice/test")
print(test_batch_names)


## === cell 3
def GetBatchId(nombre_archivo):
    numero = re.findall(r'\d+', nombre_archivo)
    return int(numero[0]) if numero else None

def Batch2Path(batch_id_name, is_train = True):
    path = "/kaggle/input/icecube-neutrinos-in-deep-ice/"
    
    if is_train:
        
        path += "train/" + batch_id_name
    else:
        
        path += "test/" + batch_id_name
        
    return path


## === cell 4
meta_train = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet")
meta_test = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet")

print(f"Cantidad de eventos de entrenamiento: {len(meta_train)}")
print(f"Cantidad de eventos a Predecir: {len(meta_test)}")


## === cell 5
sensor_geometry = pd.read_csv("/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv")

x = sensor_geometry.x
y = sensor_geometry.y
z = sensor_geometry.z

d = np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2 + (z.max() - z.min())**2)

c = (299792.458 * 1000)/10**9

time_valid = d/c

print(f"time valid: {time_valid}m/ns")


## === cell 6
def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    '''
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse 
    cosine (arccos) thereof is then the angle between the two input vectors
    
    Parameters:
    -----------
    
    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian
    
    Returns:
    --------
    
    dist : float
        mean over the angular distance(s) in radian
    '''
    
    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")
    
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)
    
    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)
    
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    scalar_prod =  np.clip(scalar_prod, -1, 1)
    
    return np.average(np.abs(np.arccos(scalar_prod)))


## === cell 7
def Load_event(idx, batch_names, meta, is_train = True):
    batch_df = pd.read_parquet(Batch2Path(batch_names, is_train)).reset_index()
    batch_id = GetBatchId(batch_names)
    
    if is_train:

        relevant_meta = meta[meta["batch_id"] == batch_id].reset_index(drop=True)
        event_count = len(relevant_meta)
    else:
        relevant_meta = meta
        

    first_pulse_index, last_pulse_index, event_id = relevant_meta.iloc[idx][["first_pulse_index", "last_pulse_index", "event_id"]].astype(int)
    event_feature = batch_df[first_pulse_index: last_pulse_index + 1]
    event_feature = event_feature.merge(sensor_geometry, on="sensor_id", how="left")[["time", "charge", "auxiliary", "x", "y", "z"]]
    event_feature["time"] -= event_feature["time"].min()
    
    event_feature.x = event_feature.x - event_feature.x.mean()
    event_feature.y = event_feature.y - event_feature.y.mean()
    event_feature.z = event_feature.z - event_feature.z.mean()
    
    if is_train:
        
        azimuth, zenith = relevant_meta.iloc[idx][["azimuth", "zenith"]].values.T.astype('float16')
        true_angles = np.array([zenith, azimuth])
        vector = np.array([
            np.sin(zenith) * np.cos(azimuth),
            np.sin(zenith) * np.sin(azimuth),
            np.cos(azimuth)
        ])
    
        vector_escalar = np.array([-500, 500])
        x = vector_escalar * vector[0]
        y = vector_escalar * vector[1]
        z = vector_escalar * vector[2]
        true_direction = pd.DataFrame({'x': x, 'y': y, 'z': z})
    
    vector_escalar = np.array([-500, 500])
    pca = PCA(n_components = 1).fit(event_feature.loc[~event_feature.auxiliary][["x", "y", "z"]])
    vector_pca = pca.components_[0]
    x_pred = vector_escalar *vector_pca[0]
    y_pred = vector_escalar *vector_pca[1]
    z_pred = vector_escalar *vector_pca[2]
    pca_direction = pd.DataFrame({'x_pred': x_pred, 'y_pred': y_pred, 'z_pred': z_pred})
    
    zenith_pca = np.arccos(vector_pca[2])
    azimuth_pca = np.arctan2(vector_pca[1], vector_pca[0])
    if azimuth_pca < 0:
        azimuth_pca = 2*np.pi + azimuth_pca
        
    pca_angles = np.array([zenith_pca, azimuth_pca])
    
    if is_train:
        true_angles_direction = [true_angles, true_direction]
    pca_angles_direction = [pca_angles, pca_direction]
    
    
    
    print("")
    print("==" * 20)
    if is_train:
        print(f"Event info.\n1.Batch Number: {batch_id}\n2.Event Index of batch_{batch_id}: {idx}\n3. Event_id: {event_id}\n4.Zenith_true: {zenith} - Zenith_pca: {zenith_pca}\n5.Azimuth: {azimuth} - Azimuth_pca: {azimuth_pca}")
    else:
        print(f"Event info.\n1.Batch Number: {batch_id}\n2.Event Index of batch_{batch_id}: {idx}\n3. Event_id: {event_id}\n4.Zenith_pca: {zenith_pca}\n5.Azimuth_pca: {azimuth_pca}")
        
    print("")
    print("==" * 20)  
    del batch_df
    gc.collect()
    
    pulses = px.scatter_3d(event_feature, x="x", y="y", z="z", opacity = .5 ,color="auxiliary",
                           color_discrete_map={True: "steelblue", False: "aquamarine"})
    
    if is_train:
        pulses.update_traces(marker_size = event_feature.charge * 10)
        true_direction = px.line_3d(true_direction, x = "x", y = "y", z = "z", color_discrete_sequence=['aquamarine'])
    
        pc1_direction = px.line_3d(pca_direction, x = "x_pred", y = "y_pred", z = "z_pred", color_discrete_sequence=['white'])

        fig = go.Figure(data = pulses.data + true_direction.data + pc1_direction.data)
        fig.update_layout(template='plotly_dark')
        fig.show()
    
    else:
        pulses.update_traces(marker_size = event_feature.charge * 10)
    
        pc1_direction = px.line_3d(pca_direction, x = "x_pred", y = "y_pred", z = "z_pred", color_discrete_sequence=['white'])

        fig = go.Figure(data = pulses.data  + pc1_direction.data)
        fig.update_layout(template='plotly_dark')
        fig.show()
        
    
    
    print("")
    print("==" * 20)
    print(f"the Varianza explained by PC1 is: {pca.explained_variance_ratio_[0]:.2%}")
    if is_train:
        print(f'The value of error: {angular_dist_score(azimuth, zenith,azimuth_pca, zenith_pca ):.2f}')
        return event_feature, true_angles_direction, pca_angles_direction
    else:
        return azimuth_pca, zenith_pca


## === cell 8
for i in range(2):
    event, y_true, y_pred = Load_event(i, train_batchs_names[0], meta_train, is_train = True)
    


## === cell 9
meta_test = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
)

test_dir = "/kaggle/input/icecube-neutrinos-in-deep-ice/test"
available_test_batches = sorted(
    [
        f
        for f in os.listdir(test_dir)
        if f.startswith("batch_") and f.endswith(".parquet")
    ]
)
if not available_test_batches:
    raise FileNotFoundError(f"No batch_*.parquet files found in: {test_dir}")

test_batch_name = available_test_batches[0]
test = pd.read_parquet(os.path.join(test_dir, test_batch_name))

globals()["batch_661.parquet"] = test_batch_name


## === cell 10
batch_name = globals().get("batch_661.parquet", "batch_661.parquet")
azimuth, zenith = Load_event(2, batch_name, meta_test, is_train=False)


## === cell 11
submission = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet")
for i in range(3):
    azimuth, zenith = Load_event(i, "batch_661.parquet", meta_test, is_train = False)
    submission.iloc[i, 1] = azimuth
    submission.iloc[i, 2] = zenith


## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/880626775.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0msubmission[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_parquet[0m[0;34m([0m[0;34m"/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m3[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mazimuth[0m[0;34m,[0m [0mzenith[0m [0;34m=[0m [0mLoad_event[0m[0;34m([0m[0mi[0m[0;34m,[0m [0;34m"batch_661.parquet"[0m[0;34m,[0m [0mmeta_test[0m[0;34m,[0m [0mis_train[0m [0;34m=[0m [0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0msubmission[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mi[0m[0;34m,[0m [0;36m1[0m[0;34m][0m [0;34m=[0m [0mazimuth[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0msubmission[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mi[0m[0;34m,[0m [0;36m2[0m[0;34m][0m [0;34m=[0m [0mzenith[0m[0;34m[0m[0;34m[0m[0m

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

## === cell 12
submission.to_csv("submission.csv", index = False)

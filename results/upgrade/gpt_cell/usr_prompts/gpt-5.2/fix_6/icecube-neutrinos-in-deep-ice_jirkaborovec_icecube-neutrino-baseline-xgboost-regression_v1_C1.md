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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

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
import os
import glob
import numpy as np
import pandas as pd

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



## === cell 1
meta_test = pd.read_parquet(os.path.join(PATH_DATASET, "test_meta.parquet"))
print(f"length: {len(meta_test)}")
meta_test.head()
del meta_test



## === cell 2
meta_train = pd.read_parquet(os.path.join(PATH_DATASET, "train_meta.parquet"))
print(f"total events: {len(meta_train)}")
print(f"nb batches: {len(meta_train['batch_id'].unique())}")
meta_train.head()



## === cell 3
pass



## === cell 4
pass



## === cell 5
test_batch_files = sorted(
    glob.glob(os.path.join(PATH_DATASET, "test", "batch_*.parquet"))
)
if not test_batch_files:
    raise FileNotFoundError(
        f"No test batch parquet files found under: {os.path.join(PATH_DATASET, 'test')}"
    )

df_test = pd.read_parquet(test_batch_files[0], columns=["sensor_id", "charge"])
print(f"loaded: {os.path.basename(test_batch_files[0])}")
print(f"length: {len(df_test)}")
print(f"events: {len(df_test.index.unique())}")
print(df_test.head())
del df_test



## === cell 6
from tqdm.auto import tqdm


def transform_batch(path_batch_parquet, nb_sensors=5160):
    df = pd.read_parquet(path_batch_parquet, columns=["sensor_id", "charge"])

    event_index = df.index
    sensor = df["sensor_id"].to_numpy(dtype=np.int32, copy=False)
    charge = df["charge"].to_numpy(dtype=np.float16, copy=False)

    inv = (
        event_index.groupby(level=0, sort=False)
        .ngroup()
        .to_numpy(dtype=np.int32, copy=False)
    )
    event_ids = event_index.unique().to_numpy(dtype=np.int64, copy=False)

    n_events = event_ids.shape[0]
    data = np.zeros((n_events, nb_sensors), dtype=np.float16)

    data[inv, sensor] = charge

    return event_ids, data




## === cell 7
event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_1.parquet"))
print(f"data size: {data.shape}")

meta_train_sub = meta_train.loc[
    meta_train["event_id"].isin(event_ids), ["event_id", "azimuth", "zenith"]
]
meta_train_sub = meta_train_sub.set_index("event_id").loc[event_ids]
angles = meta_train_sub[["azimuth", "zenith"]].to_numpy(dtype=np.float16, copy=True)

print(f"LUT size: {len(meta_train_sub)}")
print(f"angles size: {angles.shape}")



## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3747596181.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mevent_ids[0m[0;34m,[0m [0mdata[0m [0;34m=[0m [0mtransform_batch[0m[0;34m([0m[0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mPATH_DATASET[0m[0;34m,[0m [0;34m"train/batch_1.parquet"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mprint[0m[0;34m([0m[0;34mf"data size: {data.shape}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m [0;31m# SPEED: avoid building a dict and a Python loop; use a merge/join to align targets to event_ids.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;31m# This is equivalent (same mapping event_id -> (azimuth, zenith)) but vectorized.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3505122160.py[0m in [0;36mtransform_batch[0;34m(path_batch_parquet, nb_sensors)[0m
[1;32m     17[0m     [0;31m# Fast event grouping without sorting[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m     inv = (
[0;32m---> 19[0;31m         [0mevent_index[0m[0;34m.[0m[0mgroupby[0m[0;34m([0m[0mlevel[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m [0msort[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m         [0;34m.[0m[0mngroup[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m         [0;34m.[0m[0mto_numpy[0m[0;34m([0m[0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mint32[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Index.groupby() got an unexpected keyword argument 'level'

## === cell 8
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

X_train, X_test, y_train, y_test = train_test_split(
    data, angles, train_size=0.8, random_state=42, shuffle=True
)
del data, angles

xgbr = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "regressor",
            XGBRegressor(
                tree_method="hist",
                n_estimators=64,
                num_target=y_train.shape[1],
                verbosity=2,  # keep as-is
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
)

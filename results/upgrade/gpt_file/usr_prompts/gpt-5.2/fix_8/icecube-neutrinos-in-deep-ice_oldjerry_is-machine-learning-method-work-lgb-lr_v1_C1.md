# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
lightgbm==4.6.0
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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

PATH_DATASET = "/kaggle/input/icecube-neutrinos-in-deep-ice"

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pass



## === cell 2
meta_train = pd.read_parquet(
    os.path.join(PATH_DATASET, "train_meta.parquet"),
    columns=["event_id", "azimuth", "zenith"],
)
meta_train.head()



## === cell 3
from tqdm.auto import tqdm


def transform_batch(path_batch_parquet, nb_sensors=5160):
    df = pd.read_parquet(path_batch_parquet, columns=["sensor_id", "charge"])
    df = df.reset_index(names="event_id")

    g = df.groupby(["event_id", "sensor_id"], sort=False, observed=True)[
        "charge"
    ].last()

    ev = g.index.get_level_values(0).to_numpy(copy=False)
    sensor = g.index.get_level_values(1).to_numpy(np.int64, copy=False)
    charge = g.to_numpy(np.float16, copy=False)

    ev_codes, event_ids_np = pd.factorize(ev, sort=False)
    ev_codes = ev_codes.astype(np.int64, copy=False)

    data = np.zeros((len(event_ids_np), nb_sensors), dtype=np.float16)
    data[ev_codes, sensor] = charge
    return event_ids_np, data




## === cell 4
event_ids, data = transform_batch(os.path.join(PATH_DATASET, "train/batch_10.parquet"))
print(f"data size: {data.shape}")

mt = meta_train.set_index("event_id").loc[event_ids, ["azimuth", "zenith"]]
angles = mt.to_numpy(np.float16, copy=True)
print(f"angles size: {angles.shape}")



## === cell 5
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, angles, train_size=0.8, random_state=42
)
del data, angles



## === cell 6
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

preprocess = Pipeline(
    [
        ("scaler", StandardScaler(copy=True)),
        (
            "PCA",
            PCA(n_components=510, copy=False, svd_solver="randomized", random_state=42),
        ),
    ]
)

X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)



## === cell 7
from lightgbm import LGBMRegressor
import lightgbm as lgbm



## === cell 8
params_lgb = {
    "colsample_bytree": 0.8825309754631472,
    "learning_rate": 0.13432792418150802,
    "max_depth": 4,
    "n_estimators": 165,
    "num_leaves": 15,
    "reg_alpha": 0.676985054081822,
    "reg_lambda": 580.8583113057583,
    "subsample": 0.9447267339812311,
    "n_jobs": -1,
    "random_state": 42,
}
lgb = LGBMRegressor(**params_lgb)



## === cell 9
callbacks = [
    lgbm.early_stopping(stopping_rounds=5, verbose=True),
    lgbm.log_evaluation(period=50),
]

lgb.fit(
    X_train,
    y_train[:, 0],
    eval_set=[(X_test, y_test[:, 0])],
    eval_metric="rmse",
    callbacks=callbacks,
)



## === cell 10
lgb1 = LGBMRegressor(**params_lgb)
callbacks = [
    lgbm.early_stopping(stopping_rounds=5, verbose=True),
    lgbm.log_evaluation(period=50),
]
lgb1.fit(
    X_train,
    y_train[:, 1],
    eval_set=[(X_test, y_test[:, 1])],
    eval_metric="rmse",
    callbacks=callbacks,
)



## === cell 11
import gc, time

gc.collect()
time.sleep(1)

sample_path = os.path.join(PATH_DATASET, "sample_submission.csv")
meta_test = pd.read_csv(sample_path, usecols=["event_id"])
test_event_ids = meta_test["event_id"].to_numpy(np.int64, copy=False)
n_test = test_event_ids.shape[0]

max_eid = int(test_event_ids.max())
eid_to_pos = np.full(max_eid + 1, -1, dtype=np.int32)
eid_to_pos[test_event_ids] = np.arange(n_test, dtype=np.int32)

az_arr = np.empty(n_test, dtype=np.float32)
ze_arr = np.empty(n_test, dtype=np.float32)
del meta_test



## === cell 12
az_mean = float(meta_train["azimuth"].mean())
ze_mean = float(meta_train["zenith"].mean())
az_arr.fill(az_mean)
ze_arr.fill(ze_mean)



## === cell 13
from scipy import sparse

scaler = preprocess.named_steps["scaler"]
pca = preprocess.named_steps["PCA"]

mean_ = scaler.mean_.astype(np.float32, copy=False)  # (5160,)
scale_ = scaler.scale_.astype(np.float32, copy=False)  # (5160,)
components_ = pca.components_.astype(np.float32, copy=False)  # (510, 5160)

const_term = (-mean_ / scale_) @ components_.T  # (510,) float32
pca_w = (components_ / scale_[None, :]).T  # (5160, 510) float32


def transform_batch_to_pca(path_batch_parquet, nb_sensors=5160):
    df = pd.read_parquet(path_batch_parquet, columns=["sensor_id", "charge"])
    df = df.reset_index(names="event_id")

    g = df.groupby(["event_id", "sensor_id"], sort=False, observed=True)[
        "charge"
    ].last()

    ev = g.index.get_level_values(0).to_numpy(copy=False)
    sensor = g.index.get_level_values(1).to_numpy(np.int32, copy=False)
    charge = g.to_numpy(np.float32, copy=False)

    ev_codes, event_ids_np = pd.factorize(ev, sort=False)
    ev_codes = ev_codes.astype(np.int32, copy=False)
    n_events = len(event_ids_np)

    Xs = sparse.csr_matrix(
        (charge, (ev_codes, sensor)),
        shape=(n_events, nb_sensors),
        dtype=np.float32,
    )

    Xt = Xs @ pca_w  # (n_events, 510) float32 dense
    Xt += const_term  # broadcast add, float32

    return event_ids_np, Xt




## === cell 14
ls = glob.glob(os.path.join(PATH_DATASET, "test", "*.parquet"))
ls.sort()
print("num test batches:", len(ls))

for i, batch_file in enumerate(ls):
    print(f"processing: {batch_file}")

    event_ids, Xt = transform_batch_to_pca(batch_file)

    az_pred = lgb.predict(Xt)
    ze_pred = lgb1.predict(Xt)

    az_pred = np.mod(az_pred, 2.0 * np.pi)
    ze_pred = np.clip(ze_pred, 0.0, np.pi)

    pos = eid_to_pos[event_ids.astype(np.int64, copy=False)]
    valid = pos >= 0
    if not np.all(valid):
        pos = pos[valid]
        az_pred = az_pred[valid]
        ze_pred = ze_pred[valid]

    az_arr[pos] = az_pred.astype(np.float32, copy=False)
    ze_arr[pos] = ze_pred.astype(np.float32, copy=False)

    del event_ids, Xt, az_pred, ze_pred, pos, valid
    if (i % 8) == 7:
        gc.collect()



## === cell 15
sub = pd.DataFrame({"event_id": test_event_ids, "azimuth": az_arr, "zenith": ze_arr})
sub = sub[["event_id", "azimuth", "zenith"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

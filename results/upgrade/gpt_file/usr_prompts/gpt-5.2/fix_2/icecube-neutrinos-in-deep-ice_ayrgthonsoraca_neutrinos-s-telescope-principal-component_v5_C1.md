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
import pandas as pd
import numpy as np
import gc
import os
import re

import plotly.express as px
from plotly import graph_objects as go
from sklearn.decomposition import PCA



## === cell 1
train_batchs_names = os.listdir("/kaggle/input/icecube-neutrinos-in-deep-ice/train")[
    :10
]
print(f"Sample de Batches Seleccionados: ")
print(train_batchs_names)



## === cell 2
test_batch_names = os.listdir("/kaggle/input/icecube-neutrinos-in-deep-ice/test")
print(test_batch_names[:10], "...", f"(total={len(test_batch_names)})")




## === cell 3
def GetBatchId(nombre_archivo):
    numero = re.findall(r"\d+", nombre_archivo)
    return int(numero[0]) if numero else None


def Batch2Path(batch_id_name, is_train=True):
    path = "/kaggle/input/icecube-neutrinos-in-deep-ice/"
    if is_train:
        path += "train/" + batch_id_name
    else:
        path += "test/" + batch_id_name
    return path




## === cell 4
meta_train = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet"
)
meta_test = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
)

print(f"Cantidad de eventos de entrenamiento: {len(meta_train)}")
print(f"Cantidad de eventos a Predecir: {len(meta_test)}")



## === cell 5
sensor_geometry = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv"
)

x = sensor_geometry.x
y = sensor_geometry.y
z = sensor_geometry.z

d = np.sqrt(
    (x.max() - x.min()) ** 2 + (y.max() - y.min()) ** 2 + (z.max() - z.min()) ** 2
)
c = (299792.458 * 1000) / 10**9
time_valid = d / c
print(f"time valid: {time_valid}m/ns")




## === cell 6
def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    if not (
        np.all(np.isfinite(az_true))
        and np.all(np.isfinite(zen_true))
        and np.all(np.isfinite(az_pred))
        and np.all(np.isfinite(zen_pred))
    ):
        raise ValueError("All arguments must be finite")

    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)

    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)
    scalar_prod = np.clip(scalar_prod, -1, 1)
    return np.average(np.abs(np.arccos(scalar_prod)))




## === cell 7
def Load_event(idx, batch_names, meta, is_train=True, do_plot=True):
    batch_path = Batch2Path(batch_names, is_train)
    batch_df = pd.read_parquet(
        batch_path
    ).reset_index()  # event_id comes as index in parquet
    batch_id = GetBatchId(batch_names)

    if is_train:
        relevant_meta = meta[meta["batch_id"] == batch_id].reset_index(drop=True)
    else:
        relevant_meta = meta[meta["batch_id"] == batch_id].reset_index(drop=True)

    first_pulse_index, last_pulse_index, event_id = relevant_meta.iloc[idx][
        ["first_pulse_index", "last_pulse_index", "event_id"]
    ].astype(int)

    event_feature = batch_df.iloc[first_pulse_index : last_pulse_index + 1].copy()
    event_feature = event_feature.merge(sensor_geometry, on="sensor_id", how="left")[
        ["time", "charge", "auxiliary", "x", "y", "z"]
    ]

    event_feature["time"] = event_feature["time"] - event_feature["time"].min()

    event_feature["x"] = event_feature["x"] - event_feature["x"].mean()
    event_feature["y"] = event_feature["y"] - event_feature["y"].mean()
    event_feature["z"] = event_feature["z"] - event_feature["z"].mean()

    if is_train:
        azimuth, zenith = relevant_meta.iloc[idx][
            ["azimuth", "zenith"]
        ].values.T.astype("float32")
        true_angles = np.array([zenith, azimuth])

        vector = np.array(
            [
                np.sin(zenith) * np.cos(azimuth),
                np.sin(zenith) * np.sin(azimuth),
                np.cos(azimuth),
            ]
        )

        vector_escalar = np.array([-500, 500])
        x_t = vector_escalar * vector[0]
        y_t = vector_escalar * vector[1]
        z_t = vector_escalar * vector[2]
        true_direction = pd.DataFrame({"x": x_t, "y": y_t, "z": z_t})

    non_aux = event_feature.loc[~event_feature["auxiliary"], ["x", "y", "z"]]
    if len(non_aux) < 2:
        non_aux = event_feature[["x", "y", "z"]]

    pca = PCA(n_components=1).fit(non_aux)
    vector_pca = pca.components_[0]

    zenith_pca = np.arccos(np.clip(vector_pca[2], -1.0, 1.0))
    azimuth_pca = np.arctan2(vector_pca[1], vector_pca[0])
    if azimuth_pca < 0:
        azimuth_pca = 2 * np.pi + azimuth_pca

    if do_plot:
        print("")
        print("==" * 20)
        if is_train:
            print(
                f"Event info.\n1.Batch Number: {batch_id}\n2.Event Index of batch_{batch_id}: {idx}\n3. Event_id: {event_id}"
                f"\n4.Zenith_true: {zenith} - Zenith_pca: {zenith_pca}\n5.Azimuth: {azimuth} - Azimuth_pca: {azimuth_pca}"
            )
        else:
            print(
                f"Event info.\n1.Batch Number: {batch_id}\n2.Event Index of batch_{batch_id}: {idx}\n3. Event_id: {event_id}"
                f"\n4.Zenith_pca: {zenith_pca}\n5.Azimuth_pca: {azimuth_pca}"
            )

        pulses = px.scatter_3d(
            event_feature,
            x="x",
            y="y",
            z="z",
            opacity=0.5,
            color="auxiliary",
            color_discrete_map={True: "steelblue", False: "aquamarine"},
        )
        pulses.update_traces(
            marker_size=np.clip(event_feature["charge"].to_numpy(), 0, None) * 10
        )

        vector_escalar = np.array([-500, 500])
        pca_direction = pd.DataFrame(
            {
                "x_pred": vector_escalar * vector_pca[0],
                "y_pred": vector_escalar * vector_pca[1],
                "z_pred": vector_escalar * vector_pca[2],
            }
        )
        pc1_direction = px.line_3d(
            pca_direction,
            x="x_pred",
            y="y_pred",
            z="z_pred",
            color_discrete_sequence=["white"],
        )

        if is_train:
            true_direction_fig = px.line_3d(
                true_direction,
                x="x",
                y="y",
                z="z",
                color_discrete_sequence=["aquamarine"],
            )
            fig = go.Figure(
                data=pulses.data + true_direction_fig.data + pc1_direction.data
            )
        else:
            fig = go.Figure(data=pulses.data + pc1_direction.data)

        fig.update_layout(template="plotly_dark")
        fig.show()

        print("")
        print("==" * 20)
        print(
            f"the Varianza explained by PC1 is: {pca.explained_variance_ratio_[0]:.2%}"
        )
        if is_train:
            print(
                f"The value of error: {angular_dist_score(azimuth, zenith, azimuth_pca, zenith_pca):.2f}"
            )

    del batch_df
    gc.collect()

    if is_train:
        return (
            event_feature,
            [true_angles, true_direction],
            [np.array([zenith_pca, azimuth_pca]), None],
        )
    else:
        return azimuth_pca, zenith_pca, int(event_id)




## === cell 8
for i in range(2):
    event, y_true, y_pred = Load_event(
        i, train_batchs_names[0], meta_train, is_train=True, do_plot=False
    )
    del event
    gc.collect()



## === cell 9
test_batch_file = sorted(
    os.listdir("/kaggle/input/icecube-neutrinos-in-deep-ice/test")
)[0]
print("Using test batch:", test_batch_file)

_az, _ze, _eid = Load_event(
    0, test_batch_file, meta_test, is_train=False, do_plot=False
)
print("Example prediction:", _eid, _az, _ze)



## === cell 10
submission = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv"
)
print(submission.head())



## === cell 11
test_dir = "/kaggle/input/icecube-neutrinos-in-deep-ice/test"
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".parquet")])

pred_az = np.empty(len(submission), dtype=np.float32)
pred_ze = np.empty(len(submission), dtype=np.float32)

event_to_row = pd.Series(
    np.arange(len(submission), dtype=np.int64), index=submission["event_id"].values
)

filled = 0
for bf_i, batch_file in enumerate(test_files):
    batch_id = GetBatchId(batch_file)
    relevant_meta = meta_test[meta_test["batch_id"] == batch_id].reset_index(drop=True)
    if len(relevant_meta) == 0:
        continue

    batch_path = os.path.join(test_dir, batch_file)
    batch_df = pd.read_parquet(batch_path).reset_index()
    batch_df = batch_df.merge(sensor_geometry, on="sensor_id", how="left")

    for i in range(len(relevant_meta)):
        first_idx, last_idx, event_id = relevant_meta.iloc[i][
            ["first_pulse_index", "last_pulse_index", "event_id"]
        ].astype(int)

        ef = batch_df.iloc[first_idx : last_idx + 1][
            ["time", "charge", "auxiliary", "x", "y", "z"]
        ].copy()
        ef["time"] = ef["time"] - ef["time"].min()
        ef["x"] = ef["x"] - ef["x"].mean()
        ef["y"] = ef["y"] - ef["y"].mean()
        ef["z"] = ef["z"] - ef["z"].mean()

        non_aux = ef.loc[~ef["auxiliary"], ["x", "y", "z"]]
        if len(non_aux) < 2:
            non_aux = ef[["x", "y", "z"]]

        pca = PCA(n_components=1).fit(non_aux)
        v = pca.components_[0]

        zenith_pca = np.arccos(np.clip(v[2], -1.0, 1.0))
        azimuth_pca = np.arctan2(v[1], v[0])
        if azimuth_pca < 0:
            azimuth_pca = 2 * np.pi + azimuth_pca

        row = event_to_row.get(event_id)
        if row is not None:
            pred_az[row] = azimuth_pca
            pred_ze[row] = zenith_pca
            filled += 1

    del batch_df
    gc.collect()

    if (bf_i + 1) % 5 == 0:
        print(
            f"Processed {bf_i+1}/{len(test_files)} test batches; filled {filled}/{len(submission)} events"
        )

print("Filled:", filled, "of", len(submission))



## === cell 12
submission["azimuth"] = pred_az
submission["zenith"] = pred_ze

submission["azimuth"] = (
    submission["azimuth"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)
submission["zenith"] = (
    submission["zenith"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

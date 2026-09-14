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

1.470661

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from pathlib import Path
import os, sys

META_ROOT = Path("/kaggle/working/meta_partitions")

print(f"pid: {os.getpid()}")
print("META_ROOT:", META_ROOT)


def ensure_partitioned(split: str):
    print(
        f"Skipping meta partitioning for {split}; will read {split}_meta.parquet directly."
    )


ensure_partitioned("train")
ensure_partitioned("test")



## === cell 1
from pathlib import Path
import gc
import os
from functools import lru_cache

import polars as pl
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
import pytorch_lightning as ptl

event_accumulator = None

np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
torch.set_num_interop_threads(1)



## === cell 2
bp = Path("/kaggle/input/icecube-neutrinos-in-deep-ice")

sensors_df = pl.read_csv(bp / "sensor_geometry.csv").with_columns(
    pl.col("sensor_id").cast(pl.Int16)
)
print("sensors shape", sensors_df.shape)



## === cell 3
_META_CACHE = {}  # folder -> (meta_df, batch_id -> row indices)


def _load_full_meta(folder: str) -> pl.DataFrame:
    src_file = bp / f"{folder}_meta.parquet"
    cols = ["batch_id", "event_id", "first_pulse_index", "last_pulse_index"]
    if folder == "train":
        cols += ["azimuth", "zenith"]
    return pl.read_parquet(src_file, columns=cols)


def _get_meta_index(folder: str):
    if folder in _META_CACHE:
        return _META_CACHE[folder]
    meta = _load_full_meta(folder)
    gb = (
        meta.select(["batch_id"])
        .with_row_index("row_idx")
        .group_by("batch_id")
        .agg(pl.col("row_idx"))
    )
    batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
    _META_CACHE[folder] = (meta, batch_to_rows)
    return _META_CACHE[folder]


@lru_cache(maxsize=512)
def load_meta_for_batch(folder: str, b_id: int) -> pl.DataFrame:
    meta, batch_to_rows = _get_meta_index(folder)
    rows = batch_to_rows.get(int(b_id), [])
    if not rows:
        return meta.head(0)
    return meta.take(rows)


def load_batch(folder, b_id):
    """
    Load a single batch of pulses and its corresponding meta.
    """
    pulses = pl.read_parquet(bp / folder / f"batch_{b_id}.parquet")
    pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")

    meta = load_meta_for_batch(folder, int(b_id))
    return pulses, meta


load_batch_c = lru_cache(maxsize=1)(load_batch)




## === cell 4
def get_batch_ids_from_folder(folder):
    fnames = [x.stem for x in (bp / folder).glob("batch_*.parquet")]
    assert all(x.startswith("batch_") for x in fnames)
    return np.sort(np.array([int(x[6:]) for x in fnames]))


TRAIN_BIDS = get_batch_ids_from_folder("train")
TEST_BIDS = get_batch_ids_from_folder("test")
print("num train batches:", len(TRAIN_BIDS), "num test batches:", len(TEST_BIDS))




## === cell 5
def preview_data():
    b_pulses_train, b_meta_train = load_batch_c("train", TRAIN_BIDS[0])
    b_pulses_test, b_meta_test = load_batch("test", TEST_BIDS[0])

    print("Train meta head:\n", b_meta_train.head().to_pandas())
    print("Test meta head:\n", b_meta_test.head().to_pandas())
    print("Sensors head:\n", sensors_df.head().to_pandas())
    print("Pulses train head:\n", b_pulses_train.head().to_pandas())
    print("Pulses test head:\n", b_pulses_test.head().to_pandas())
    print(
        "Sample submission head:\n", pd.read_csv(bp / "sample_submission.csv", nrows=5)
    )


preview_data()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2682047245.py in <cell line: 0>()
     13 
     14 
---> 15 preview_data()
     16 
     17 

/tmp/ipykernel_11/2682047245.py in preview_data()
      1 def preview_data():
----> 2     b_pulses_train, b_meta_train = load_batch_c("train", TRAIN_BIDS[0])
      3     b_pulses_test, b_meta_test = load_batch("test", TEST_BIDS[0])
      4 
      5     print("Train meta head:\n", b_meta_train.head().to_pandas())

/tmp/ipykernel_11/1383333814.py in load_batch(folder, b_id)
     47     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     48 
---> 49     meta = load_meta_for_batch(folder, int(b_id))
     50     return pulses, meta
     51 

/tmp/ipykernel_11/1383333814.py in load_meta_for_batch(folder, b_id)
     33 @lru_cache(maxsize=512)
     34 def load_meta_for_batch(folder: str, b_id: int) -> pl.DataFrame:
---> 35     meta, batch_to_rows = _get_meta_index(folder)
     36     rows = batch_to_rows.get(int(b_id), [])
     37     if not rows:

/tmp/ipykernel_11/1383333814.py in _get_meta_index(folder)
     26         .agg(pl.col("row_idx"))
     27     )
---> 28     batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
     29     _META_CACHE[folder] = (meta, batch_to_rows)
     30     return _META_CACHE[folder]

/tmp/ipykernel_11/1383333814.py in <dictcomp>(.0)
     26         .agg(pl.col("row_idx"))
     27     )
---> 28     batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
     29     _META_CACHE[folder] = (meta, batch_to_rows)
     30     return _META_CACHE[folder]

AttributeError: 'list' object has no attribute 'to_list'

## === cell 6
def load_single_event(folder, i_batch, i_evt, ignore_aux=True):
    bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
    pulses, meta = load_batch_c(folder, bids[i_batch])

    meta_event = meta[i_evt]
    (event_id,) = meta_event["event_id"]

    event_pulses = pulses.filter(pl.col("event_id") == event_id)
    if ignore_aux:
        event_pulses = event_pulses.filter(~pl.col("auxiliary")).drop("auxiliary")

    target = None
    if "azimuth" in meta.columns:
        (azimuth,) = meta_event["azimuth"]
        (zenith,) = meta_event["zenith"]
        target = (azimuth, zenith)

    return event_id, event_pulses, target




## === cell 7
import plotly.graph_objects as go
import plotly.express as px


def plot_event(event_id, event_pulses, y=None):
    fig = px.scatter_3d(
        x=event_pulses["x"],
        y=event_pulses["y"],
        z=event_pulses["z"],
        color=event_pulses["time"],
        size=event_pulses["charge"],
    )

    if y is not None:
        (azimuth, zenith) = y
        xyz = event_pulses[["x", "y", "z"]]
        xyz_mean = xyz.mean(axis=0)
        r = 1200

        (x0,) = xyz_mean["x"]
        (y0,) = xyz_mean["y"]
        (z0,) = xyz_mean["z"]
        (x1,) = xyz_mean["x"] + r * np.cos(azimuth) * np.sin(zenith)
        (y1,) = xyz_mean["y"] + r * np.sin(azimuth) * np.sin(zenith)
        (z1,) = xyz_mean["z"] + r * np.cos(zenith)

        fig.add_trace(
            go.Scatter3d(
                x=np.linspace(x0, x1, 100),
                y=np.linspace(y0, y1, 100),
                z=np.linspace(z0, z1, 100),
                marker=go.scatter3d.Marker(size=0.001),
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-600, 600]),
            yaxis=dict(range=[-600, 600]),
            zaxis=dict(range=[-600, 600]),
        ),
        scene_aspectmode="cube",
        title=dict(text=f"evt #{event_id}"),
    )
    return fig




## === cell 8
print("Single-event loader smoke test:", load_single_event("train", 0, 0)[0])




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1742663794.py in <cell line: 0>()
----> 1 print("Single-event loader smoke test:", load_single_event("train", 0, 0)[0])
      2 
      3 

/tmp/ipykernel_11/1735953756.py in load_single_event(folder, i_batch, i_evt, ignore_aux)
      1 def load_single_event(folder, i_batch, i_evt, ignore_aux=True):
      2     bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
----> 3     pulses, meta = load_batch_c(folder, bids[i_batch])
      4 
      5     meta_event = meta[i_evt]

/tmp/ipykernel_11/1383333814.py in load_batch(folder, b_id)
     47     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     48 
---> 49     meta = load_meta_for_batch(folder, int(b_id))
     50     return pulses, meta
     51 

/tmp/ipykernel_11/1383333814.py in load_meta_for_batch(folder, b_id)
     33 @lru_cache(maxsize=512)
     34 def load_meta_for_batch(folder: str, b_id: int) -> pl.DataFrame:
---> 35     meta, batch_to_rows = _get_meta_index(folder)
     36     rows = batch_to_rows.get(int(b_id), [])
     37     if not rows:

/tmp/ipykernel_11/1383333814.py in _get_meta_index(folder)
     26         .agg(pl.col("row_idx"))
     27     )
---> 28     batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
     29     _META_CACHE[folder] = (meta, batch_to_rows)
     30     return _META_CACHE[folder]

/tmp/ipykernel_11/1383333814.py in <dictcomp>(.0)
     26         .agg(pl.col("row_idx"))
     27     )
---> 28     batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
     29     _META_CACHE[folder] = (meta, batch_to_rows)
     30     return _META_CACHE[folder]

AttributeError: 'list' object has no attribute 'to_list'

## === cell 9
def normalize_time(df: pl.DataFrame) -> pl.DataFrame:
    g = df.select(["event_id", "time"]).group_by("event_id")
    times = (
        g.quantile(0.5)
        .rename({"time": "t_mid"})
        .with_columns(pl.col("t_mid").cast(pl.Int64))
    )
    return df.join(times, on="event_id").with_columns(
        (pl.col("time") - pl.col("t_mid")).alias("time_norm")
    )


NBINS_T = 10
NBINS_S = 10


def preprocess_batch(batch_df: pl.DataFrame) -> pl.DataFrame:
    tbins = np.linspace(-5000, 10000, NBINS_T + 1)
    tbin_width = tbins[1] - tbins[0]
    tbin_centers = (tbins[:-1] + tbins[1:]) / 2

    sbins = np.linspace(-600, 600, NBINS_S + 1)
    sbin_width = sbins[1] - sbins[0]
    sbin_centers = (sbins[:-1] + sbins[1:]) / 2

    return (
        normalize_time(batch_df)
        .group_by("event_id")
        .agg(
            [
                (
                    (
                        np.exp(
                            -(
                                ((pl.col("time_norm") - tmid) / tbin_width) ** 2
                                + ((pl.col(ax) - smid) / sbin_width) ** 2
                            )
                        )
                    )
                    * pl.col("charge")
                )
                .sum()
                .alias(f"t_{int(tmid)}_{ax}_{int(smid)}".replace("-", "m"))
                for ax in ["x", "y", "z"]
                for tmid in tbin_centers
                for smid in sbin_centers
            ]
        )
    )




## === cell 10
def plot_example(i_batch=0, i_event=4):
    b_pulses, _ = load_batch_c("train", TRAIN_BIDS[i_batch])
    e_id = b_pulses["event_id"].unique().sort()[i_event]
    prep_batch = preprocess_batch(b_pulses.filter(pl.col("event_id") == e_id))
    print("Preprocessed batch shape:", prep_batch.shape)
    img = prep_batch.drop("event_id").to_numpy().reshape(3, NBINS_T, NBINS_S)

    plt.figure(figsize=(14, 4))
    plt.imshow(
        np.concatenate(
            [np.pad(np.log1p(x), 1, constant_values=np.nan) for x in img], axis=-1
        )
    )
    plt.colorbar()


print("Preprocess smoke test OK")



## === cell 11
from pathlib import Path

preproc_cache_script = r'''
import os
import numpy as np
import polars as pl
from joblib import Memory

os.makedirs("cache", exist_ok=True)
memory = Memory("cache/")

def _assert_event_id_alignment(batch_pulses: pl.DataFrame, batch_meta: pl.DataFrame) -> None:
    a = batch_pulses["event_id"].to_numpy()
    b = batch_meta["event_id"].to_numpy()
    assert a.shape == b.shape and np.array_equal(a, b), "event_id mismatch between pulses and meta after sorting"

@memory.cache(ignore=["load_batch", "preprocess_batch"])
def load_and_preprocess_batch(
    folder, batch_id, load_batch, preprocess_batch, num_subsample=None
):
    """
    Load and preprocess a batch, possibly taking only a subsample of a batch.
    """
    batch_pulses, batch_meta = load_batch(folder, batch_id)
    if num_subsample is not None:
        ids = batch_pulses["event_id"].unique()
        # Note: upstream sets global seed; choice is deterministic given seed.
        ids = np.random.choice(ids, num_subsample, replace=False)
        batch_pulses = batch_pulses.filter(pl.col("event_id").is_in(ids.tolist()))
        batch_meta = batch_meta.filter(pl.col("event_id").is_in(ids.tolist()))
    batch_pulses = preprocess_batch(batch_pulses).sort(by="event_id")
    batch_meta = batch_meta.sort(by="event_id")

    _assert_event_id_alignment(batch_pulses, batch_meta)
    return batch_pulses, batch_meta
'''
Path("preprocessing_cache.py").write_text(preproc_cache_script)
print("Wrote preprocessing_cache.py")



## === cell 12
import preprocessing_cache


def load_and_preprocess_batch(folder, i_batch, num_subsample=None):
    bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
    return preprocessing_cache.load_and_preprocess_batch(
        folder, bids[i_batch], load_batch, preprocess_batch, num_subsample
    )




## === cell 13
del load_batch_c
gc.collect()




## === cell 14
class Dataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        folder,
        file_ids,
        batch_size,
        drop_each_last=True,
        num_subsample=None,
        with_evt_id=False,
    ):
        super().__init__()
        self.folder = folder
        self.file_ids = file_ids
        self.batch_size = batch_size
        self.drop_each_last = drop_each_last
        self.num_subsample = num_subsample
        self.with_evt_id = with_evt_id

    def __iter__(self):
        for i in np.random.choice(
            len(self.file_ids), len(self.file_ids), replace=False
        ):
            batch_x, batch_y = load_and_preprocess_batch(
                self.folder, self.file_ids[i], num_subsample=self.num_subsample
            )
            if "azimuth" in batch_y.columns:
                batch_y = batch_y[["azimuth", "zenith"]]
            else:
                batch_y = None

            evt_ids = batch_x["event_id"]
            batch_x = batch_x.drop("event_id")

            if self.num_subsample is not None:
                assert len(batch_x) == self.num_subsample

            x_np = (
                batch_x.to_numpy().astype(np.float32).reshape(-1, 3, NBINS_T, NBINS_S)
            )
            y_np = None if batch_y is None else batch_y.to_numpy().astype(np.float32)

            for i_evt in range(0, len(x_np), self.batch_size):
                minibatch_x = x_np[i_evt : i_evt + self.batch_size]
                if self.drop_each_last and len(minibatch_x) < self.batch_size:
                    continue

                minibatch_x = torch.from_numpy(minibatch_x)
                minibatch_y = (
                    None
                    if y_np is None
                    else torch.from_numpy(y_np[i_evt : i_evt + self.batch_size])
                )

                if self.with_evt_id:
                    yield minibatch_x, minibatch_y, evt_ids[
                        i_evt : i_evt + self.batch_size
                    ]
                else:
                    yield minibatch_x, minibatch_y




## === cell 15
assert NBINS_T == 10 and NBINS_S == 10, "Our model expects 10x10 representation"


class ConvPredictor(torch.nn.Module):
    def __init__(self, activation=torch.nn.ELU()):
        super().__init__()
        (self.model_tx, self.model_ty, self.model_tz) = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3),
                activation,
                torch.nn.Conv2d(32, 64, 3),
                activation,
                torch.nn.Conv2d(64, 128, 3),
                activation,
                torch.nn.Conv2d(128, 256, 3),
                activation,
            )
            for _ in range(3)
        ]
        self.head = torch.nn.Sequential(
            torch.nn.Linear(3 * 256 * 2 * 2, 32), activation, torch.nn.Linear(32, 3)
        )

    def forward(self, x):
        x = torch.log(1.0 + x)
        tx, ty, tz = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        tx = self.model_tx(tx).view(x.shape[0], 256 * 4)
        ty = self.model_ty(ty).view(x.shape[0], 256 * 4)
        tz = self.model_tz(tz).view(x.shape[0], 256 * 4)
        pred = self.head(torch.cat([tx, ty, tz], axis=1))
        return pred


class LitModel(ptl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate_loss(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X, Y = batch[0], batch[1]
        else:
            raise ValueError(f"Unexpected batch structure: {type(batch)}")

        (azimuth, zenith) = Y.T
        vx = torch.cos(azimuth) * torch.sin(zenith)
        vy = torch.sin(azimuth) * torch.sin(zenith)
        vz = torch.cos(zenith)
        v = torch.stack([vx, vy, vz], axis=1)

        pred_v = self.model(X)
        return torch.nn.functional.mse_loss(pred_v, v)

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return opt

    def on_validation_end(self):
        return


model = ConvPredictor()
lit_model = LitModel(model)



## === cell 16
gc.collect()

np.random.seed(42)
use_n_files = 3
chosen_file_ids = np.random.choice(len(TRAIN_BIDS), use_n_files + 1, replace=False)
train_file_ids = chosen_file_ids[:-1]
val_file_ids = chosen_file_ids[-1:]
print("Train file ids:", train_file_ids)
print("Validation file ids:", val_file_ids)

train_dataset = Dataset(
    folder="train",
    file_ids=train_file_ids,
    batch_size=32,
    drop_each_last=True,
    num_subsample=5000,
)
val_dataset = Dataset(
    folder="train",
    file_ids=val_file_ids,
    batch_size=500,
    num_subsample=500,
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=0)
val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=0)

trainer = ptl.Trainer(
    max_epochs=30,
    callbacks=ptl.callbacks.ModelCheckpoint(monitor="val_loss"),
    accelerator="cpu",
    devices=1,
    logger=ptl.loggers.CSVLogger(save_dir="lightning_logs", name="icecube"),
    enable_checkpointing=True,
    enable_model_summary=False,
)
trainer.fit(
    model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2118303610.py in <cell line: 0>()
     36     enable_model_summary=False,
     37 )
---> 38 trainer.fit(
     39     model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
     40 )

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in fit(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
    558         self.training = True
    559         self.should_stop = False
--> 560         call._call_and_handle_interrupt(
    561             self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
    562         )

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
     47         if trainer.strategy.launcher is not None:
     48             return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
---> 49         return trainer_fn(*args, **kwargs)
     50 
     51     except _TunerExitException:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _fit_impl(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
    596             model_connected=self.lightning_module is not None,
    597         )
--> 598         self._run(model, ckpt_path=ckpt_path)
    599 
    600         assert self.state.stopped

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run(self, model, ckpt_path)
   1009         # RUN THE TRAINER
   1010         # ----------------------------
-> 1011         results = self._run_stage()
   1012 
   1013         # ----------------------------

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run_stage(self)
   1051         if self.training:
   1052             with isolate_rng():
-> 1053                 self._run_sanity_check()
   1054             with torch.autograd.set_detect_anomaly(self._detect_anomaly):
   1055                 self.fit_loop.run()

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run_sanity_check(self)
   1080 
   1081             # run eval step
-> 1082             val_loop.run()
   1083 
   1084             call._call_callback_hooks(self, "on_sanity_check_end")

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/utilities.py in _decorator(self, *args, **kwargs)
    177             context_manager = torch.no_grad
    178         with context_manager():
--> 179             return loop_run(self, *args, **kwargs)
    180 
    181     return _decorator

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/evaluation_loop.py in run(self)
    136                 else:
    137                     dataloader_iter = None
--> 138                     batch, batch_idx, dataloader_idx = next(data_fetcher)
    139                 if previous_dataloader_idx != dataloader_idx:
    140                     # the dataloader has changed, notify the logger connector

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py in __next__(self)
    132         elif not self.done:
    133             # this will run only when no pre-fetching was done.
--> 134             batch = super().__next__()
    135         else:
    136             # the iterator is empty

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py in __next__(self)
     59         self._start_profiler()
     60         try:
---> 61             batch = next(self.iterator)
     62         except StopIteration:
     63             self.done = True

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py in __next__(self)
    339     def __next__(self) -> _ITERATOR_RETURN:
    340         assert self._iterator is not None
--> 341         out = next(self._iterator)
    342         if isinstance(self._iterator, _Sequential):
    343             return out

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py in __next__(self)
    140 
    141         try:
--> 142             out = next(self.iterators[0])
    143         except StopIteration:
    144             # try the next iterator

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     31             for _ in possibly_batched_index:
     32                 try:
---> 33                     data.append(next(self.dataset_iter))
     34                 except StopIteration:
     35                     self.ended = True

/tmp/ipykernel_11/1234483560.py in __iter__(self)
     23             len(self.file_ids), len(self.file_ids), replace=False
     24         ):
---> 25             batch_x, batch_y = load_and_preprocess_batch(
     26                 self.folder, self.file_ids[i], num_subsample=self.num_subsample
     27             )

/tmp/ipykernel_11/2469455004.py in load_and_preprocess_batch(folder, i_batch, num_subsample)
      4 def load_and_preprocess_batch(folder, i_batch, num_subsample=None):
      5     bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
----> 6     return preprocessing_cache.load_and_preprocess_batch(
      7         folder, bids[i_batch], load_batch, preprocess_batch, num_subsample
      8     )

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    605     def __call__(self, *args, **kwargs):
    606         # Return the output, without the metadata
--> 607         return self._cached_call(args, kwargs, shelving=False)[0]
    608 
    609     def __getstate__(self):

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in _cached_call(self, args, kwargs, shelving)
    560 
    561         # Returns the output but not the metadata
--> 562         return self._call(call_id, args, kwargs, shelving)
    563 
    564     @property

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in _call(self, call_id, args, kwargs, shelving)
    830         self._before_call(args, kwargs)
    831         start_time = time.time()
--> 832         output = self.func(*args, **kwargs)
    833         return self._after_call(call_id, args, kwargs, shelving, output, start_time)
    834 

/kaggle/working/preprocessing_cache.py in load_and_preprocess_batch(folder, batch_id, load_batch, preprocess_batch, num_subsample)
     20     Load and preprocess a batch, possibly taking only a subsample of a batch.
     21     """
---> 22     batch_pulses, batch_meta = load_batch(folder, batch_id)
     23     if num_subsample is not None:
     24         ids = batch_pulses["event_id"].unique()

/tmp/ipykernel_11/1383333814.py in load_batch(folder, b_id)
     47     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     48 
---> 49     meta = load_meta_for_batch(folder, int(b_id))
     50     return pulses, meta
     51 

/tmp/ipykernel_11/1383333814.py in load_meta_for_batch(folder, b_id)
     33 @lru_cache(maxsize=512)
     34 def load_meta_for_batch(folder: str, b_id: int) -> pl.DataFrame:
---> 35     meta, batch_to_rows = _get_meta_index(folder)
     36     rows = batch_to_rows.get(int(b_id), [])
     37     if not rows:

/tmp/ipykernel_11/1383333814.py in _get_meta_index(folder)
     26         .agg(pl.col("row_idx"))
     27     )
---> 28     batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
     29     _META_CACHE[folder] = (meta, batch_to_rows)
     30     return _META_CACHE[folder]

/tmp/ipykernel_11/1383333814.py in <dictcomp>(.0)
     26         .agg(pl.col("row_idx"))
     27     )
---> 28     batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
     29     _META_CACHE[folder] = (meta, batch_to_rows)
     30     return _META_CACHE[folder]

AttributeError: 'list' object has no attribute 'to_list'

## === cell 17
print("best_model_score:", trainer.checkpoint_callback.best_model_score)
print("best_model_path:", trainer.checkpoint_callback.best_model_path)



## === cell 18
best_path = trainer.checkpoint_callback.best_model_path
if best_path and Path(best_path).exists():
    checkpoint = torch.load(best_path, map_location="cpu")
    lit_model.load_state_dict(checkpoint["state_dict"])
else:
    print("No checkpoint found; using current in-memory model weights.")

lit_model.eval()
model.eval()



## === cell 19
trainer.validate(lit_model, val_dataloader)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/993188157.py in <cell line: 0>()
----> 1 trainer.validate(lit_model, val_dataloader)
      2 
      3 

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in validate(self, model, dataloaders, ckpt_path, verbose, datamodule)
    661         self.state.status = TrainerStatus.RUNNING
    662         self.validating = True
--> 663         return call._call_and_handle_interrupt(
    664             self, self._validate_impl, model, dataloaders, ckpt_path, verbose, datamodule
    665         )

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
     47         if trainer.strategy.launcher is not None:
     48             return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
---> 49         return trainer_fn(*args, **kwargs)
     50 
     51     except _TunerExitException:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _validate_impl(self, model, dataloaders, ckpt_path, verbose, datamodule)
    703             self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
    704         )
--> 705         results = self._run(model, ckpt_path=ckpt_path)
    706         # remove the tensors from the validation results
    707         results = convert_tensors_to_scalars(results)

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run(self, model, ckpt_path)
   1009         # RUN THE TRAINER
   1010         # ----------------------------
-> 1011         results = self._run_stage()
   1012 
   1013         # ----------------------------

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run_stage(self)
   1046 
   1047         if self.evaluating:
-> 1048             return self._evaluation_loop.run()
   1049         if self.predicting:
   1050             return self.predict_loop.run()

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/utilities.py in _decorator(self, *args, **kwargs)
    177             context_manager = torch.no_grad
    178         with context_manager():
--> 179             return loop_run(self, *args, **kwargs)
    180 
    181     return _decorator

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/evaluation_loop.py in run(self)
    121         if self.skip:
    122             return []
--> 123         self.reset()
    124         self.on_run_start()
    125         data_fetcher = self._data_fetcher

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/evaluation_loop.py in reset(self)
    257         combined_loader.limits = self.max_batches
    258         data_fetcher.setup(combined_loader)
--> 259         iter(data_fetcher)  # creates the iterator inside the fetcher
    260 
    261         # add the previous `fetched` value to properly track `is_last_batch` with no prefetching

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py in __iter__(self)
    110         for _ in range(self.prefetch_batches):
    111             try:
--> 112                 batch = super().__next__()
    113                 self.batches.append(batch)
    114             except StopIteration:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py in __next__(self)
     59         self._start_profiler()
     60         try:
---> 61             batch = next(self.iterator)
     62         except StopIteration:
     63             self.done = True

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py in __next__(self)
    339     def __next__(self) -> _ITERATOR_RETURN:
    340         assert self._iterator is not None
--> 341         out = next(self._iterator)
    342         if isinstance(self._iterator, _Sequential):
    343             return out

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py in __next__(self)
    140 
    141         try:
--> 142             out = next(self.iterators[0])
    143         except StopIteration:
    144             # try the next iterator

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     31             for _ in possibly_batched_index:
     32                 try:
---> 33                     data.append(next(self.dataset_iter))
     34                 except StopIteration:
     35                     self.ended = True

/tmp/ipykernel_11/1234483560.py in __iter__(self)
     23             len(self.file_ids), len(self.file_ids), replace=False
     24         ):
---> 25             batch_x, batch_y = load_and_preprocess_batch(
     26                 self.folder, self.file_ids[i], num_subsample=self.num_subsample
     27             )

/tmp/ipykernel_11/2469455004.py in load_and_preprocess_batch(folder, i_batch, num_subsample)
      4 def load_and_preprocess_batch(folder, i_batch, num_subsample=None):
      5     bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
----> 6     return preprocessing_cache.load_and_preprocess_batch(
      7         folder, bids[i_batch], load_batch, preprocess_batch, num_subsample
      8     )

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    605     def __call__(self, *args, **kwargs):
    606         # Return the output, without the metadata
--> 607         return self._cached_call(args, kwargs, shelving=False)[0]
    608 
    609     def __getstate__(self):

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in _cached_call(self, args, kwargs, shelving)
    560 
    561         # Returns the output but not the metadata
--> 562         return self._call(call_id, args, kwargs, shelving)
    563 
    564     @property

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in _call(self, call_id, args, kwargs, shelving)
    830         self._before_call(args, kwargs)
    831         start_time = time.time()
--> 832         output = self.func(*args, **kwargs)
    833         return self._after_call(call_id, args, kwargs, shelving, output, start_time)
    834 

/kaggle/working/preprocessing_cache.py in load_and_preprocess_batch(folder, batch_id, load_batch, preprocess_batch, num_subsample)
     20     Load and preprocess a batch, possibly taking only a subsample of a batch.
     21     """
---> 22     batch_pulses, batch_meta = load_batch(folder, batch_id)
     23     if num_subsample is not None:
     24         ids = batch_pulses["event_id"].unique()

/tmp/ipykernel_11/1383333814.py in load_batch(folder, b_id)
     47     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     48 
---> 49     meta = load_meta_for_batch(folder, int(b_id))
     50     return pulses, meta
     51 

/tmp/ipykernel_11/1383333814.py in load_meta_for_batch(folder, b_id)
     33 @lru_cache(maxsize=512)
     34 def load_meta_for_batch(folder: str, b_id: int) -> pl.DataFrame:
---> 35     meta, batch_to_rows = _get_meta_index(folder)
     36     rows = batch_to_rows.get(int(b_id), [])
     37     if not rows:

/tmp/ipykernel_11/1383333814.py in _get_meta_index(folder)
     26         .agg(pl.col("row_idx"))
     27     )
---> 28     batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
     29     _META_CACHE[folder] = (meta, batch_to_rows)
     30     return _META_CACHE[folder]

/tmp/ipykernel_11/1383333814.py in <dictcomp>(.0)
     26         .agg(pl.col("row_idx"))
     27     )
---> 28     batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
     29     _META_CACHE[folder] = (meta, batch_to_rows)
     30     return _META_CACHE[folder]

AttributeError: 'list' object has no attribute 'to_list'

## === cell 20
def angles2vec(azimuth, zenith):
    x = torch.cos(azimuth) * torch.sin(zenith)
    y = torch.sin(azimuth) * torch.sin(zenith)
    z = torch.cos(zenith)
    return torch.stack([x, y, z], axis=1)


def vec2angles(vec):
    norm = ((vec**2).sum(axis=1) ** 0.5)[:, None]
    vec = vec / norm
    zenith = torch.acos(vec[:, 2])
    sin_zenith = torch.sin(zenith)
    sin_zenith = torch.where(
        sin_zenith == 0, torch.full_like(sin_zenith, 1e-8), sin_zenith
    )
    cos_azimuth = vec[:, 0] / sin_zenith
    sin_azimuth = vec[:, 1] / sin_zenith
    azimuth = torch.atan2(sin_azimuth, cos_azimuth)
    return azimuth, zenith




## === cell 21
test_ds = Dataset(
    "test",
    list(range(len(TEST_BIDS))),
    batch_size=1000,
    drop_each_last=False,
    with_evt_id=True,
)


def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


out_path = Path("submission.csv")
with out_path.open("w") as f:
    f.write("event_id,azimuth,zenith\n")
    with torch.no_grad():
        for i_batch in range(len(TEST_BIDS)):
            batch_x, batch_meta = load_and_preprocess_batch(
                "test", i_batch, num_subsample=None
            )
            evt_ids = batch_x["event_id"].to_numpy()
            x_np = (
                batch_x.drop("event_id")
                .to_numpy()
                .astype(np.float32)
                .reshape(-1, 3, NBINS_T, NBINS_S)
            )
            X = torch.from_numpy(x_np)
            pred_vec = model(X)
            az, ze = vec2angles(pred_vec)

            az = torch.remainder(az, 2 * torch.pi)
            ze = torch.clamp(ze, 0.0, torch.pi)

            az_np = torch2numpy(az)
            ze_np = torch2numpy(ze)

            for eid, a, z in zip(evt_ids, az_np, ze_np):
                f.write(f"{int(eid)},{float(a)},{float(z)}\n")

print(
    "Wrote submission.csv:", out_path.exists(), "size(bytes):", out_path.stat().st_size
)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/744093448.py in <cell line: 0>()
     24         for i_batch in range(len(TEST_BIDS)):
     25             # Use cached preprocessing; deterministic and identical computations.
---> 26             batch_x, batch_meta = load_and_preprocess_batch(
     27                 "test", i_batch, num_subsample=None
     28             )

/tmp/ipykernel_11/2469455004.py in load_and_preprocess_batch(folder, i_batch, num_subsample)
      4 def load_and_preprocess_batch(folder, i_batch, num_subsample=None):
      5     bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
----> 6     return preprocessing_cache.load_and_preprocess_batch(
      7         folder, bids[i_batch], load_batch, preprocess_batch, num_subsample
      8     )

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    605     def __call__(self, *args, **kwargs):
    606         # Return the output, without the metadata
--> 607         return self._cached_call(args, kwargs, shelving=False)[0]
    608 
    609     def __getstate__(self):

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in _cached_call(self, args, kwargs, shelving)
    560 
    561         # Returns the output but not the metadata
--> 562         return self._call(call_id, args, kwargs, shelving)
    563 
    564     @property

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in _call(self, call_id, args, kwargs, shelving)
    830         self._before_call(args, kwargs)
    831         start_time = time.time()
--> 832         output = self.func(*args, **kwargs)
    833         return self._after_call(call_id, args, kwargs, shelving, output, start_time)
    834 

/kaggle/working/preprocessing_cache.py in load_and_preprocess_batch(folder, batch_id, load_batch, preprocess_batch, num_subsample)
     20     Load and preprocess a batch, possibly taking only a subsample of a batch.
     21     """
---> 22     batch_pulses, batch_meta = load_batch(folder, batch_id)
     23     if num_subsample is not None:
     24         ids = batch_pulses["event_id"].unique()

/tmp/ipykernel_11/1383333814.py in load_batch(folder, b_id)
     47     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     48 
---> 49     meta = load_meta_for_batch(folder, int(b_id))
     50     return pulses, meta
     51 

/tmp/ipykernel_11/1383333814.py in load_meta_for_batch(folder, b_id)
     33 @lru_cache(maxsize=512)
     34 def load_meta_for_batch(folder: str, b_id: int) -> pl.DataFrame:
---> 35     meta, batch_to_rows = _get_meta_index(folder)
     36     rows = batch_to_rows.get(int(b_id), [])
     37     if not rows:

/tmp/ipykernel_11/1383333814.py in _get_meta_index(folder)
     26         .agg(pl.col("row_idx"))
     27     )
---> 28     batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
     29     _META_CACHE[folder] = (meta, batch_to_rows)
     30     return _META_CACHE[folder]

/tmp/ipykernel_11/1383333814.py in <dictcomp>(.0)
     26         .agg(pl.col("row_idx"))
     27     )
---> 28     batch_to_rows = {int(b): rows.to_list() for b, rows in gb.iter_rows()}
     29     _META_CACHE[folder] = (meta, batch_to_rows)
     30     return _META_CACHE[folder]

AttributeError: 'list' object has no attribute 'to_list'

## === cell 22
import pandas as pd
from pathlib import Path

sub_head = pd.read_csv("submission.csv", nrows=5)
print(sub_head)
print("Columns:", list(sub_head.columns))
print(
    "File exists:",
    Path("submission.csv").exists(),
    "size(bytes):",
    Path("submission.csv").stat().st_size,
)

## --- ERROR in outputing the csv:
Invalid submission: Azimuth must be a number

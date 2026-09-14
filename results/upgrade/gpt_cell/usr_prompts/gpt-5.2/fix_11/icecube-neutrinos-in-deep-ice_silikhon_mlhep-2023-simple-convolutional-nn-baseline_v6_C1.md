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

No external packages required in the script and installed.

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
%%file partition_meta.py
"""
Script that partitions large metadata file into smaller files
- separate file per each batch.

Usage:
    python partition_meta.py SPLIT
where SPLIT is either `train` or `test`
"""

from pathlib import Path
import sys, gc

import polars as pl
import pyarrow.parquet as pq
from tqdm import trange

bp = Path("/kaggle/input/icecube-neutrinos-in-deep-ice")

def iter_through_meta(folder, chunk_size=100):
    """
    Read the large metadata file in chunks of batches.
    
    Parameters
    ----------
    folder : str
        Which part of the data to process. Should be either "train" or "test".

    Returns
    -------
    generator
        A generator of chunks of batches in pyarrow.Table format.
    """
    assert folder in ["train", "test"], "Argument `folder` should either be 'train' or 'test'"
    src_file = bp / f"{folder}_meta.parquet"
    all_batch_ids = pl.read_parquet(src_file, columns=["batch_id"]).unique().sort("batch_id")
    for chunk_first in trange(0, len(all_batch_ids), chunk_size):
        selection = all_batch_ids[chunk_first: chunk_first + chunk_size]["batch_id"]
        first, last = selection.min(), selection.max()
        yield pq.read_table(src_file, filters=[
            ("batch_id", ">=", first),
            ("batch_id", "<=", last),
        ])
        gc.collect()

def write_meta_batches(meta, folder):
    """
    Take a chunk of metadata info and write it to disk with separate file per batch.

    Parameters
    ----------
    meta : pyarrow.Table
        A table with a chunk of batches with metadata.
    folder : str
        Which to write into. Should be either "train" or "test".
    """
    folder.mkdir(exist_ok=True)
    pq.write_to_dataset(meta, root_path=folder, partition_cols=['batch_id'], flavor='spark')

def main(folder):
    """
    Read metatdata in chunks and save separate files per batch.
    
    Parameters
    ----------
    folder : str
        Which part of the data to process. Should be either "train" or "test".
    """
    print(f"Working on {folder}")
    for table in iter_through_meta(folder):
        write_meta_batches(table, Path(folder))
    print(f"Done ({folder})")
    
if __name__ == "__main__":
    try:
        _, folder = sys.argv
    except ValueError:
        print('Usage: "python partition_meta.py SPLIT", where SPLIT = "train" or "test"')
        exit(0)
    main(folder)


## === cell 1
from pathlib import Path

if not Path("train").exists():
    !python partition_meta.py train
if not Path("test").exists():
    !python partition_meta.py test


## === cell 2
!top -bn1 -o '%MEM'

import os
print(f"(our pid: {os.getpid()})")


## === cell 3
bp = Path("/kaggle/input/icecube-neutrinos-in-deep-ice")

def check_num_batches(split):
    num_actual = len(list(Path(split).glob("batch_id=*")))
    num_expected = len(list((bp / split).glob("batch_*.parquet")))

    if num_actual != num_expected:
        print(
            f"WARNING!!! Found {num_actual} batch files when expected {num_expected} for "
            f'split "{split}". Check that partitioning code ran ok.'
        )
    else:
        print(f"Check ok ({split})")

for split in ["train", "test"]:
    check_num_batches(split)


## === cell 4
from pathlib import Path
import gc
import os
from functools import lru_cache

import polars as pl
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True) # https://stackoverflow.com/questions/67419817/uncaught-error-script-error-for-plotly-http-requirejs-org-docs-errors-html
import plotly.graph_objects as go
import plotly.express as px
import torch
import pytorch_lightning as ptl
from tensorboard.backend.event_processing import event_accumulator


## === cell 5
sensors_df = pl.read_csv(bp / "sensor_geometry.csv").with_columns(pl.col("sensor_id").cast(pl.Int16))

print("sesnsors shape", sensors_df.shape)


## === cell 6
def load_batch(folder, b_id):
    """
    Load a single batch of pulses.
    
    Parameters
    ----------
    folder : str
        Which part of the data to process. Should be either "train" or "test".
    b_id : int
        Index of the batch (as in the `batch_id` column).

    Returns
    -------
    polars.DataFrame
        Data frame with the loaded pulses.
    polars.DataFrame
        Data frame with corresponding metadata.
    """
    pulses = pl.read_parquet(bp / folder / f"batch_{b_id}.parquet")
    pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")

    meta = pl.read_parquet(f"{folder}/batch_id={b_id}/*.parquet")

    return pulses, meta

load_batch_c = lru_cache(maxsize=1)(load_batch)


## === cell 7
def get_batch_ids_from_folder(folder):
    fnames = [x.stem for x in (bp / folder).glob("*.parquet")]
    assert all(x.startswith("batch_") for x in fnames)
    return np.sort(np.array([int(x[6:]) for x in fnames]))

TRAIN_BIDS = get_batch_ids_from_folder("train")
TEST_BIDS = get_batch_ids_from_folder("test")


## === cell 8
def load_batch(folder, b_id):
    """
    Load a single batch of pulses.

    Parameters
    ----------
    folder : str
        Which part of the data to process. Should be either "train" or "test".
    b_id : int
        Index of the batch (as in the `batch_id` column).

    Returns
    -------
    polars.DataFrame
        Data frame with the loaded pulses.
    polars.DataFrame
        Data frame with corresponding metadata.
    """
    pulses = pl.read_parquet(bp / folder / f"batch_{b_id}.parquet")
    pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")

    meta = pl.read_parquet(f"{folder}/batch_id={b_id}/*.parquet", glob=True)

    return pulses, meta


load_batch_c = lru_cache(maxsize=1)(load_batch)


## === cell 9
def load_single_event(folder, i_batch, i_evt, ignore_aux=True):
    """
    Load a single event (using positional indexing).

    Parameters
    ----------
    folder : str
        Which part of the data to process. Should be either "train" or "test".
    i_batch : int
        Positional index of the batch. Should be in a range from 0 (inclusive)
        to the number of batches in `folder` (exclusive).
    i_evt : int
        Positional index of the event within the batch.
    ignore_aux : bool
        Whether to exclude auxiliary pulses.

    Returns
    -------
    int
        The id of the read event (as in `event_id` column).
    polars.DataFrame
        Table with pulses.
    Tuple[float, float] | None
        Azimuth and zenith (if `folder` is 'train') or `None` (if `folder` is 'test').
    """
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


## === cell 10
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
        xyz = event_pulses[['x', 'y', 'z']]
        xyz_mean = xyz.mean(axis=0)
        r = 1200

        (x0,) = xyz_mean['x']
        (y0,) = xyz_mean['y']
        (z0,) = xyz_mean['z']
        (x1,) = xyz_mean['x'] + r * np.cos(azimuth) * np.sin(zenith)
        (y1,) = xyz_mean['y'] + r * np.sin(azimuth) * np.sin(zenith)
        (z1,) = xyz_mean['z'] + r * np.cos(zenith)

        fig.add_trace(
            go.Scatter3d(
                x=np.linspace(x0, x1, 100),
                y=np.linspace(y0, y1, 100),
                z=np.linspace(z0, z1, 100),
                marker=go.scatter3d.Marker(size=0.001)
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-600,600],),
            yaxis=dict(range=[-600,600],),
            zaxis=dict(range=[-600,600],),
        ),
        scene_aspectmode='cube',
        title=dict(text=f"evt #{event_id}")
    )

    return fig


## === cell 11
def load_batch(folder, b_id):
    """
    Load a single batch of pulses.

    Parameters
    ----------
    folder : str
        Which part of the data to process. Should be either "train" or "test".
    b_id : int
        Index of the batch (as in the `batch_id` column).

    Returns
    -------
    polars.DataFrame
        Data frame with the loaded pulses.
    polars.DataFrame
        Data frame with corresponding metadata.
    """
    pulses = pl.read_parquet(bp / folder / f"batch_{b_id}.parquet")
    pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")

    meta = pl.read_parquet(f"{folder}/batch_id={b_id}/**/*.parquet", glob=True)

    return pulses, meta


load_batch_c = lru_cache(maxsize=1)(load_batch)


## === cell 12
def load_batch(folder, b_id):
    """
    Load a single batch of pulses.

    Parameters
    ----------
    folder : str
        Which part of the data to process. Should be either "train" or "test".
    b_id : int
        Index of the batch (as in the `batch_id` column).

    Returns
    -------
    polars.DataFrame
        Data frame with the loaded pulses.
    polars.DataFrame
        Data frame with corresponding metadata.
    """
    pulses = pl.read_parquet(bp / folder / f"batch_{b_id}.parquet")
    pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")

    base_dir = Path("/kaggle/working")
    if not (base_dir / folder).exists():
        base_dir = Path(".")

    meta_dir = base_dir / folder / f"batch_id={b_id}"
    pattern_recursive = str(meta_dir / "**" / "*.parquet")
    pattern_flat = str(meta_dir / "*.parquet")

    meta_files = list(meta_dir.glob("**/*.parquet"))
    if meta_files:
        meta = pl.read_parquet(pattern_recursive, glob=True)
    else:
        meta_files = list(meta_dir.glob("*.parquet"))
        if meta_files:
            meta = pl.read_parquet(pattern_flat, glob=True)
        else:
            src_file = bp / f"{folder}_meta.parquet"
            meta = (
                pl.scan_parquet(src_file).filter(pl.col("batch_id") == b_id).collect()
            )

    return pulses, meta


load_batch_c = lru_cache(maxsize=1)(load_batch)


def _plot_event_patched(event_id, event_pulses, y=None):
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
        xyz_mean = xyz.select(pl.all().mean())
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
            xaxis=dict(
                range=[-600, 600],
            ),
            yaxis=dict(
                range=[-600, 600],
            ),
            zaxis=dict(
                range=[-600, 600],
            ),
        ),
        scene_aspectmode="cube",
        title=dict(text=f"evt #{event_id}"),
    )

    return fig


plot_event = _plot_event_patched

iplot(plot_event(*load_single_event("train", 0, 21)))


## === cell 13
iplot(plot_event(*load_single_event("train", 0, 21, ignore_aux=False)))


## === cell 14
def normalize_time(df):
    g = df[["event_id", "time"]].group_by("event_id")
    times = (
        g.quantile(0.5)
        .rename(dict(time="t_mid"))
        .with_columns(pl.col("t_mid").cast(pl.Int64))
    )
    return df.join(times, on="event_id").with_columns(
        (pl.col("time") - pl.col("t_mid")).alias("time_norm")
    )


NBINS_T = 10
NBINS_S = 10


def preprocess_batch(batch_df):
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
                            -(  #
                                ((pl.col("time_norm") - tmid) / tbin_width)
                                ** 2  #  <== Calculate the gaussian-smoothed
                                + ((pl.col(ax) - smid) / sbin_width)
                                ** 2  #  <== contribution of each pulse to each bin
                            )
                        )
                    )
                    * pl.col("charge")
                )
                .sum()
                .alias(f"t_{int(tmid)}_{ax}_{int(smid)}".replace("-", "m"))
                for ax in ["x", "y", "z"]  #  <== Iterate over axes
                for tmid in tbin_centers
                for smid in sbin_centers  #  <== Iterate over bins
            ]
        )
    )


def plot_example(i_batch=0, i_event=4):
    b_pulses, _ = load_batch_c("train", TRAIN_BIDS[i_batch])
    e_id = b_pulses["event_id"].unique().sort()[i_event]
    prep_batch = preprocess_batch(b_pulses.filter(pl.col("event_id") == e_id))
    print("Preprocessed batch shape:", prep_batch.shape)
    img = (
        prep_batch.drop("event_id").to_numpy().reshape(3, NBINS_T, NBINS_S)
    )  # 3 images (TX, TY, TZ) of NBINS_T by NBINS_S

    plt.figure(figsize=(14, 4))
    plt.imshow(
        np.concatenate(
            [np.pad(np.log1p(x), 1, constant_values=np.nan) for x in img], axis=-1
        )
    )
    plt.colorbar()


plot_example()


## === cell 15
%%file preprocessing_cache.py

import numpy as np
import polars as pl
from joblib import Memory
memory = Memory("cache/")

@memory.cache(ignore=["load_batch", "preprocess_batch"])
def load_and_preprocess_batch(
    folder, batch_id, load_batch, preprocess_batch, num_subsample=None
):
    """
    Load and preprocess a batch, possibly taking only a subsample of a batch.
    
    Parameters
    ----------
    folder : str
        Which part of the data to process. Should be either "train" or "test".
    batch_id : int
        Index of the batch (as in the `batch_id` column).
    load_batch : Callable
        Function loading a batch (to pass the interactively defined `load_batch` function).
    preprocess_batch : Callable
        Function preprocessing a batch (to pass the interactively defined `preprocess_batch`
        function).
    num_subsample : int | None
        If provided, only preprocess a random subsample of this size from the batch.

    Returns
    -------
    polars.DataFrame
        Preprocessed pulses.
    polars.DataFrame
        Corresponding meta info.
    """
    batch_pulses, batch_meta = load_batch(folder, batch_id)
    if num_subsample is not None:
        ids = batch_pulses["event_id"].unique()
        ids = np.random.choice(ids, num_subsample, replace=False)
        batch_pulses = batch_pulses.filter(pl.col("event_id").is_in(ids.tolist()))
        batch_meta = batch_meta.filter(pl.col("event_id").is_in(ids.tolist()))
    batch_pulses = preprocess_batch(batch_pulses).sort(by="event_id")
    batch_meta = batch_meta.sort(by="event_id")

    assert batch_pulses["event_id"].series_equal(batch_meta["event_id"])
    return batch_pulses, batch_meta


## === cell 16
import preprocessing_cache

def load_and_preprocess_batch(folder, i_batch, num_subsample=None):
    bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
    return preprocessing_cache.load_and_preprocess_batch(
        folder, bids[i_batch], load_batch, preprocess_batch, num_subsample
    )


## === cell 17

del load_batch_c
gc.collect()


## === cell 18
class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, folder, file_ids, batch_size, drop_each_last=True, num_subsample=None, with_evt_id=False):
        """
        Parameters
        ----------
        folder : str
            Which part of the data to process. Should be either "train" or "test".
        file_ids : Sequence[int]
            A collection of batch-file indices (positional) to use.
        batch_size : int
            Size of the output batches (not to be confused with the batches in which the pulses
            are provided by the competition).
        drop_each_last : bool
            Whether to avoid batches of size smaller than `batch_size`
        num_subsample : int | None
            Take this many elements from each file (`None` = take all).
        with_evt_id : bool
            Output event ids (useful for making and submitting the prediction).
        """
        super().__init__()
        self.folder = folder
        self.file_ids = file_ids
        self.batch_size = batch_size
        self.drop_each_last = drop_each_last
        self.num_subsample = num_subsample
        self.with_evt_id = with_evt_id

    def __iter__(self):
        for i in np.random.choice(len(self.file_ids), len(self.file_ids), replace=False):
            gc.collect()
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

            for i_evt in range(0, len(batch_x), self.batch_size):
                minibatch_x = batch_x[i_evt: i_evt + self.batch_size]
                if self.drop_each_last and len(minibatch_x) < self.batch_size:
                    continue

                minibatch_x = torch.from_numpy(
                    minibatch_x.to_numpy().astype(np.float32).reshape(-1, 3, NBINS_T, NBINS_S)
                )
                minibatch_y = None if batch_y is None else (
                    torch.from_numpy(
                        batch_y[i_evt: i_evt + self.batch_size].to_numpy().astype(np.float32)
                    )
                )
                if self.with_evt_id:
                    yield minibatch_x, minibatch_y, evt_ids[i_evt: i_evt + self.batch_size]
                else:
                    yield minibatch_x, minibatch_y


## === cell 19
assert NBINS_T == 10 and NBINS_S == 10, "Our model expects 10x10 representation"

class ConvPredictor(torch.nn.Module):
    def __init__(self, activation=torch.nn.ELU()):
        super().__init__()
        (self.model_tx, self.model_ty, self.model_tz) = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3), activation, # 1x10 -> 32x8
                torch.nn.Conv2d(32, 64, 3), activation, # -> 64x6
                torch.nn.Conv2d(64, 128, 3), activation, # -> 128x4
                torch.nn.Conv2d(128, 256, 3), activation, # -> 256x2
            ) for _ in range(3)
        ]
        self.head = torch.nn.Sequential(
            torch.nn.Linear(3 * 256 * 2 * 2, 32), activation,
            torch.nn.Linear(32, 3)
        )

    def forward(self, x):
        x = torch.log(1.0 + x)
        tx, ty, tz = x[:, 0: 1], x[:, 1: 2], x[:, 2: 3]
        tx = self.model_tx(tx).view(x.shape[0], 256 * 4)
        ty = self.model_tx(ty).view(x.shape[0], 256 * 4)
        tz = self.model_tx(tz).view(x.shape[0], 256 * 4)
        pred = self.head(torch.cat([tx, ty, tz], axis=1))
        return pred

class LitModel(ptl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate_loss(self, batch):
        (X,), (Y,) = batch
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
        super().on_validation_end()

        self.logger.experiment.flush()
        if not hasattr(self, "tb_reader"):
            self.tb_reader = event_accumulator.EventAccumulator(self.logger.log_dir)
            self.hdisplay = display("", display_id=True)
        ea = self.tb_reader
        ea.Reload()

        try:
            train_values = [(x.step, x.value) for x in ea.Scalars("train_loss")]
            val_values = [(x.step, x.value) for x in ea.Scalars("val_loss")]

            if hasattr(self, "last_fig"):
                plt.close(self.last_fig)

            self.last_fig = plt.figure()
            plt.plot(*zip(*train_values), label="Train")
            plt.plot(*zip(*val_values), label="Validation");
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.legend()
            self.hdisplay.update(self.last_fig)
        except KeyError:
            pass


model = ConvPredictor()
lit_model = LitModel(model)


## === cell 20
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
train_dataloader = torch.utils.data.DataLoader(train_dataset)
val_dataloader = torch.utils.data.DataLoader(val_dataset)

_accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = ptl.Trainer(
    max_epochs=30,
    callbacks=ptl.callbacks.ModelCheckpoint(monitor="val_loss"),
    accelerator=_accelerator,
    devices=1,
    logger=False,
)
trainer.fit(
    model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)


## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/331735951.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     36[0m     [0mlogger[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m )
[0;32m---> 38[0;31m trainer.fit(
[0m[1;32m     39[0m     [0mmodel[0m[0;34m=[0m[0mlit_model[0m[0;34m,[0m [0mtrain_dataloaders[0m[0;34m=[0m[0mtrain_dataloader[0m[0;34m,[0m [0mval_dataloaders[0m[0;34m=[0m[0mval_dataloader[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m )

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36mfit[0;34m(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)[0m
[1;32m    558[0m         [0mself[0m[0;34m.[0m[0mtraining[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m    559[0m         [0mself[0m[0;34m.[0m[0mshould_stop[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 560[0;31m         call._call_and_handle_interrupt(
[0m[1;32m    561[0m             [0mself[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_fit_impl[0m[0;34m,[0m [0mmodel[0m[0;34m,[0m [0mtrain_dataloaders[0m[0;34m,[0m [0mval_dataloaders[0m[0;34m,[0m [0mdatamodule[0m[0;34m,[0m [0mckpt_path[0m[0;34m[0m[0;34m[0m[0m
[1;32m    562[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py[0m in [0;36m_call_and_handle_interrupt[0;34m(trainer, trainer_fn, *args, **kwargs)[0m
[1;32m     47[0m         [0;32mif[0m [0mtrainer[0m[0;34m.[0m[0mstrategy[0m[0;34m.[0m[0mlauncher[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m             [0;32mreturn[0m [0mtrainer[0m[0;34m.[0m[0mstrategy[0m[0;34m.[0m[0mlauncher[0m[0;34m.[0m[0mlaunch[0m[0;34m([0m[0mtrainer_fn[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0mtrainer[0m[0;34m=[0m[0mtrainer[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 49[0;31m         [0;32mreturn[0m [0mtrainer_fn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;34m[0m[0m
[1;32m     51[0m     [0;32mexcept[0m [0m_TunerExitException[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36m_fit_impl[0;34m(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)[0m
[1;32m    596[0m             [0mmodel_connected[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mlightning_module[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    597[0m         )
[0;32m--> 598[0;31m         [0mself[0m[0;34m.[0m[0m_run[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mckpt_path[0m[0;34m=[0m[0mckpt_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    599[0m [0;34m[0m[0m
[1;32m    600[0m         [0;32massert[0m [0mself[0m[0;34m.[0m[0mstate[0m[0;34m.[0m[0mstopped[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36m_run[0;34m(self, model, ckpt_path)[0m
[1;32m   1009[0m         [0;31m# RUN THE TRAINER[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1010[0m         [0;31m# ----------------------------[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1011[0;31m         [0mresults[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_run_stage[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1012[0m [0;34m[0m[0m
[1;32m   1013[0m         [0;31m# ----------------------------[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36m_run_stage[0;34m(self)[0m
[1;32m   1051[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mtraining[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1052[0m             [0;32mwith[0m [0misolate_rng[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1053[0;31m                 [0mself[0m[0;34m.[0m[0m_run_sanity_check[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1054[0m             [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0mautograd[0m[0;34m.[0m[0mset_detect_anomaly[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_detect_anomaly[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1055[0m                 [0mself[0m[0;34m.[0m[0mfit_loop[0m[0;34m.[0m[0mrun[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36m_run_sanity_check[0;34m(self)[0m
[1;32m   1080[0m [0;34m[0m[0m
[1;32m   1081[0m             [0;31m# run eval step[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1082[0;31m             [0mval_loop[0m[0;34m.[0m[0mrun[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1083[0m [0;34m[0m[0m
[1;32m   1084[0m             [0mcall[0m[0;34m.[0m[0m_call_callback_hooks[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m"on_sanity_check_end"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/utilities.py[0m in [0;36m_decorator[0;34m(self, *args, **kwargs)[0m
[1;32m    177[0m             [0mcontext_manager[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mno_grad[0m[0;34m[0m[0;34m[0m[0m
[1;32m    178[0m         [0;32mwith[0m [0mcontext_manager[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 179[0;31m             [0;32mreturn[0m [0mloop_run[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    180[0m [0;34m[0m[0m
[1;32m    181[0m     [0;32mreturn[0m [0m_decorator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/evaluation_loop.py[0m in [0;36mrun[0;34m(self)[0m
[1;32m    136[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    137[0m                     [0mdataloader_iter[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 138[0;31m                     [0mbatch[0m[0;34m,[0m [0mbatch_idx[0m[0;34m,[0m [0mdataloader_idx[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mdata_fetcher[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    139[0m                 [0;32mif[0m [0mprevious_dataloader_idx[0m [0;34m!=[0m [0mdataloader_idx[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    140[0m                     [0;31m# the dataloader has changed, notify the logger connector[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    132[0m         [0;32melif[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0mdone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    133[0m             [0;31m# this will run only when no pre-fetching was done.[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 134[0;31m             [0mbatch[0m [0;34m=[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__next__[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    135[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    136[0m             [0;31m# the iterator is empty[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m     59[0m         [0mself[0m[0;34m.[0m[0m_start_profiler[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     60[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 61[0;31m             [0mbatch[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mself[0m[0;34m.[0m[0miterator[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     62[0m         [0;32mexcept[0m [0mStopIteration[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     63[0m             [0mself[0m[0;34m.[0m[0mdone[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    339[0m     [0;32mdef[0m [0m__next__[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0m_ITERATOR_RETURN[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    340[0m         [0;32massert[0m [0mself[0m[0;34m.[0m[0m_iterator[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 341[0;31m         [0mout[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_iterator[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    342[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_iterator[0m[0;34m,[0m [0m_Sequential[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    343[0m             [0;32mreturn[0m [0mout[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    140[0m [0;34m[0m[0m
[1;32m    141[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 142[0;31m             [0mout[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mself[0m[0;34m.[0m[0miterators[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    143[0m         [0;32mexcept[0m [0mStopIteration[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m             [0;31m# try the next iterator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    706[0m                 [0;31m# TODO(https://github.com/pytorch/pytorch/issues/76750)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    707[0m                 [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 708[0;31m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    709[0m             [0mself[0m[0;34m.[0m[0m_num_yielded[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    710[0m             if (

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_next_data[0;34m(self)[0m
[1;32m    762[0m     [0;32mdef[0m [0m_next_data[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    763[0m         [0mindex[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_index[0m[0;34m([0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 764[0;31m         [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_dataset_fetcher[0m[0;34m.[0m[0mfetch[0m[0;34m([0m[0mindex[0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    765[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_pin_memory[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    766[0m             [0mdata[0m [0;34m=[0m [0m_utils[0m[0;34m.[0m[0mpin_memory[0m[0;34m.[0m[0mpin_memory[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_pin_memory_device[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py[0m in [0;36mfetch[0;34m(self, possibly_batched_index)[0m
[1;32m     31[0m             [0;32mfor[0m [0m_[0m [0;32min[0m [0mpossibly_batched_index[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     32[0m                 [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 33[0;31m                     [0mdata[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mnext[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdataset_iter[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m                 [0;32mexcept[0m [0mStopIteration[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m                     [0mself[0m[0;34m.[0m[0mended[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3638244729.py[0m in [0;36m__iter__[0;34m(self)[0m
[1;32m     29[0m         [0;32mfor[0m [0mi[0m [0;32min[0m [0mnp[0m[0;34m.[0m[0mrandom[0m[0;34m.[0m[0mchoice[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mfile_ids[0m[0;34m)[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mfile_ids[0m[0;34m)[0m[0;34m,[0m [0mreplace[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     30[0m             [0mgc[0m[0;34m.[0m[0mcollect[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 31[0;31m             batch_x, batch_y = load_and_preprocess_batch(
[0m[1;32m     32[0m                 [0mself[0m[0;34m.[0m[0mfolder[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mfile_ids[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m,[0m [0mnum_subsample[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mnum_subsample[0m[0;34m[0m[0;34m[0m[0m
[1;32m     33[0m             )

[0;32m/tmp/ipykernel_11/113142534.py[0m in [0;36mload_and_preprocess_batch[0;34m(folder, i_batch, num_subsample)[0m
[1;32m      3[0m [0;32mdef[0m [0mload_and_preprocess_batch[0m[0;34m([0m[0mfolder[0m[0;34m,[0m [0mi_batch[0m[0;34m,[0m [0mnum_subsample[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mbids[0m [0;34m=[0m [0mdict[0m[0;34m([0m[0mtrain[0m[0;34m=[0m[0mTRAIN_BIDS[0m[0;34m,[0m [0mtest[0m[0;34m=[0m[0mTEST_BIDS[0m[0;34m)[0m[0;34m[[0m[0mfolder[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m     return preprocessing_cache.load_and_preprocess_batch(
[0m[1;32m      6[0m         [0mfolder[0m[0;34m,[0m [0mbids[0m[0;34m[[0m[0mi_batch[0m[0;34m][0m[0;34m,[0m [0mload_batch[0m[0;34m,[0m [0mpreprocess_batch[0m[0;34m,[0m [0mnum_subsample[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m     )

[0;32m/usr/local/lib/python3.11/dist-packages/joblib/memory.py[0m in [0;36m__call__[0;34m(self, *args, **kwargs)[0m
[1;32m    605[0m     [0;32mdef[0m [0m__call__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    606[0m         [0;31m# Return the output, without the metadata[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 607[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_cached_call[0m[0;34m([0m[0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m,[0m [0mshelving[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    608[0m [0;34m[0m[0m
[1;32m    609[0m     [0;32mdef[0m [0m__getstate__[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/joblib/memory.py[0m in [0;36m_cached_call[0;34m(self, args, kwargs, shelving)[0m
[1;32m    560[0m [0;34m[0m[0m
[1;32m    561[0m         [0;31m# Returns the output but not the metadata[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 562[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call[0m[0;34m([0m[0mcall_id[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m,[0m [0mshelving[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    563[0m [0;34m[0m[0m
[1;32m    564[0m     [0;34m@[0m[0mproperty[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/joblib/memory.py[0m in [0;36m_call[0;34m(self, call_id, args, kwargs, shelving)[0m
[1;32m    830[0m         [0mself[0m[0;34m.[0m[0m_before_call[0m[0;34m([0m[0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    831[0m         [0mstart_time[0m [0;34m=[0m [0mtime[0m[0;34m.[0m[0mtime[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 832[0;31m         [0moutput[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    833[0m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_after_call[0m[0;34m([0m[0mcall_id[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m,[0m [0mshelving[0m[0;34m,[0m [0moutput[0m[0;34m,[0m [0mstart_time[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    834[0m [0;34m[0m[0m

[0;32m/kaggle/working/preprocessing_cache.py[0m in [0;36mload_and_preprocess_batch[0;34m(folder, batch_id, load_batch, preprocess_batch, num_subsample)[0m
[1;32m     42[0m     [0mbatch_meta[0m [0;34m=[0m [0mbatch_meta[0m[0;34m.[0m[0msort[0m[0;34m([0m[0mby[0m[0;34m=[0m[0;34m"event_id"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     43[0m [0;34m[0m[0m
[0;32m---> 44[0;31m     [0;32massert[0m [0mbatch_pulses[0m[0;34m[[0m[0;34m"event_id"[0m[0;34m][0m[0;34m.[0m[0mseries_equal[0m[0;34m([0m[0mbatch_meta[0m[0;34m[[0m[0;34m"event_id"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     45[0m     [0;32mreturn[0m [0mbatch_pulses[0m[0;34m,[0m [0mbatch_meta[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'Series' object has no attribute 'series_equal'

## === cell 21
print(trainer.checkpoint_callback.best_model_score.detach().cpu().numpy())
print(trainer.checkpoint_callback.best_model_path)

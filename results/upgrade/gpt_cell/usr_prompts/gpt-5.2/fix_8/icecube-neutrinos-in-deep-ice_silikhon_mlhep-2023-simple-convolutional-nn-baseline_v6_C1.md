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
    g = df[["event_id", "time"]].groupby("event_id")
    times = g.quantile(0.5).rename(dict(time="t_mid")).with_columns(pl.col("t_mid").cast(pl.Int64))
    return df.join(times, on="event_id").with_columns((pl.col("time") - pl.col("t_mid")).alias("time_norm"))

NBINS_T = 10
NBINS_S = 10

def preprocess_batch(batch_df):
    tbins = np.linspace(-5000, 10000, NBINS_T + 1)
    tbin_width = tbins[1] - tbins[0]
    tbin_centers = (tbins[:-1] + tbins[1:]) / 2

    sbins = np.linspace(-600, 600, NBINS_S + 1)
    sbin_width = sbins[1] - sbins[0]
    sbin_centers = (sbins[:-1] + sbins[1:]) / 2

    return normalize_time(batch_df).groupby("event_id").agg([
        ((
            np.exp(-(                                           #
                ((pl.col("time_norm") - tmid) / tbin_width)**2  #  <== Calculate the gaussian-smoothed
                + ((pl.col(ax) - smid) / sbin_width)**2         #  <== contribution of each pulse to each bin
            ))
        ) * pl.col("charge")).sum().alias(f"t_{int(tmid)}_{ax}_{int(smid)}".replace('-', "m"))
        for ax in ["x", "y", "z"]                          #  <== Iterate over axes
        for tmid in tbin_centers for smid in sbin_centers  #  <== Iterate over bins
    ])

def plot_example(i_batch=0, i_event=4):
    b_pulses, _ = load_batch_c("train", TRAIN_BIDS[i_batch])
    e_id = b_pulses["event_id"].unique().sort()[i_event]
    prep_batch = preprocess_batch(b_pulses.filter(pl.col("event_id") == e_id))
    print("Preprocessed batch shape:", prep_batch.shape)
    img = prep_batch.drop("event_id").to_numpy().reshape(3, NBINS_T, NBINS_S)  # 3 images (TX, TY, TZ) of NBINS_T by NBINS_S

    plt.figure(figsize=(14, 4))
    plt.imshow(np.concatenate([np.pad(np.log1p(x), 1, constant_values=np.nan) for x in img], axis=-1))
    plt.colorbar();

plot_example()


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/830251705.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     45[0m     [0mplt[0m[0;34m.[0m[0mcolorbar[0m[0;34m([0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[1;32m     46[0m [0;34m[0m[0m
[0;32m---> 47[0;31m [0mplot_example[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/830251705.py[0m in [0;36mplot_example[0;34m(i_batch, i_event)[0m
[1;32m     37[0m     [0mb_pulses[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0mload_batch_c[0m[0;34m([0m[0;34m"train"[0m[0;34m,[0m [0mTRAIN_BIDS[0m[0;34m[[0m[0mi_batch[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     38[0m     [0me_id[0m [0;34m=[0m [0mb_pulses[0m[0;34m[[0m[0;34m"event_id"[0m[0;34m][0m[0;34m.[0m[0munique[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0msort[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mi_event[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 39[0;31m     [0mprep_batch[0m [0;34m=[0m [0mpreprocess_batch[0m[0;34m([0m[0mb_pulses[0m[0;34m.[0m[0mfilter[0m[0;34m([0m[0mpl[0m[0;34m.[0m[0mcol[0m[0;34m([0m[0;34m"event_id"[0m[0;34m)[0m [0;34m==[0m [0me_id[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     40[0m     [0mprint[0m[0;34m([0m[0;34m"Preprocessed batch shape:"[0m[0;34m,[0m [0mprep_batch[0m[0;34m.[0m[0mshape[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     41[0m     [0mimg[0m [0;34m=[0m [0mprep_batch[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0;34m"event_id"[0m[0;34m)[0m[0;34m.[0m[0mto_numpy[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mreshape[0m[0;34m([0m[0;36m3[0m[0;34m,[0m [0mNBINS_T[0m[0;34m,[0m [0mNBINS_S[0m[0;34m)[0m  [0;31m# 3 images (TX, TY, TZ) of NBINS_T by NBINS_S[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/830251705.py[0m in [0;36mpreprocess_batch[0;34m(batch_df)[0m
[1;32m     20[0m     [0;31m# Normalize time and calculate a gaussian-like contribution of each pulse to each bin.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m     [0;31m# We group all the data by `event_id` and then aggregate each group with the code below.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 22[0;31m     return normalize_time(batch_df).groupby("event_id").agg([
[0m[1;32m     23[0m         ((
[1;32m     24[0m             np.exp(-(                                           #

[0;32m/tmp/ipykernel_11/830251705.py[0m in [0;36mnormalize_time[0;34m(df)[0m
[1;32m      1[0m [0;32mdef[0m [0mnormalize_time[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m     [0mg[0m [0;34m=[0m [0mdf[0m[0;34m[[0m[0;34m[[0m[0;34m"event_id"[0m[0;34m,[0m [0;34m"time"[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mgroupby[0m[0;34m([0m[0;34m"event_id"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m     [0mtimes[0m [0;34m=[0m [0mg[0m[0;34m.[0m[0mquantile[0m[0;34m([0m[0;36m0.5[0m[0;34m)[0m[0;34m.[0m[0mrename[0m[0;34m([0m[0mdict[0m[0;34m([0m[0mtime[0m[0;34m=[0m[0;34m"t_mid"[0m[0;34m)[0m[0;34m)[0m[0;34m.[0m[0mwith_columns[0m[0;34m([0m[0mpl[0m[0;34m.[0m[0mcol[0m[0;34m([0m[0;34m"t_mid"[0m[0;34m)[0m[0;34m.[0m[0mcast[0m[0;34m([0m[0mpl[0m[0;34m.[0m[0mInt64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0;32mreturn[0m [0mdf[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mtimes[0m[0;34m,[0m [0mon[0m[0;34m=[0m[0;34m"event_id"[0m[0;34m)[0m[0;34m.[0m[0mwith_columns[0m[0;34m([0m[0;34m([0m[0mpl[0m[0;34m.[0m[0mcol[0m[0;34m([0m[0;34m"time"[0m[0;34m)[0m [0;34m-[0m [0mpl[0m[0;34m.[0m[0mcol[0m[0;34m([0m[0;34m"t_mid"[0m[0;34m)[0m[0;34m)[0m[0;34m.[0m[0malias[0m[0;34m([0m[0;34m"time_norm"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;31mAttributeError[0m: 'DataFrame' object has no attribute 'groupby'

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

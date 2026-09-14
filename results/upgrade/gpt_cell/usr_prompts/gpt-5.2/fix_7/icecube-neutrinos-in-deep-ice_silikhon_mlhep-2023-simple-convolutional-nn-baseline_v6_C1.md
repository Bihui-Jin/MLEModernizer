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


iplot(plot_event(*load_single_event("train", 0, 21)))


## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3121139717.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     49[0m [0;34m[0m[0m
[1;32m     50[0m [0;34m[0m[0m
[0;32m---> 51[0;31m [0miplot[0m[0;34m([0m[0mplot_event[0m[0;34m([0m[0;34m*[0m[0mload_single_event[0m[0;34m([0m[0;34m"train"[0m[0;34m,[0m [0;36m0[0m[0;34m,[0m [0;36m21[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/4227782466.py[0m in [0;36mplot_event[0;34m(event_id, event_pulses, y)[0m
[1;32m     11[0m         [0;34m([0m[0mazimuth[0m[0;34m,[0m [0mzenith[0m[0;34m)[0m [0;34m=[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m         [0mxyz[0m [0;34m=[0m [0mevent_pulses[0m[0;34m[[0m[0;34m[[0m[0;34m'x'[0m[0;34m,[0m [0;34m'y'[0m[0;34m,[0m [0;34m'z'[0m[0;34m][0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 13[0;31m         [0mxyz_mean[0m [0;34m=[0m [0mxyz[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m         [0mr[0m [0;34m=[0m [0;36m1200[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m [0;34m[0m[0m

[0;31mTypeError[0m: DataFrame.mean() got an unexpected keyword argument 'axis'

## === cell 13
iplot(plot_event(*load_single_event("train", 0, 21, ignore_aux=False)))

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

No external packages required in the script and installed.

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
def preview_data():
    b_pulses_train, b_meta_train = load_batch_c("train", TRAIN_BIDS[0])
    b_pulses_test, b_meta_test = load_batch("test", TEST_BIDS[0])

    display(HTML("Train meta:"))
    display(b_meta_train.head())

    display(HTML("Test meta:"))
    display(b_meta_test.head())

    display(HTML("Sensors:"))
    display(sensors_df.head())

    display(HTML("Pulses batch train:"))
    display(b_pulses_train.head())

    display(HTML("Pulses batch test:"))
    display(b_pulses_test.head())

    display(HTML("Sample submission:"))
    display(pl.read_parquet(bp / "sample_submission.parquet").head())

preview_data()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_11/90547586.py in <cell line: 0>()
     21     display(pl.read_parquet(bp / "sample_submission.parquet").head())
     22 
---> 23 preview_data()

/tmp/ipykernel_11/90547586.py in preview_data()
      1 def preview_data():
----> 2     b_pulses_train, b_meta_train = load_batch_c("train", TRAIN_BIDS[0])
      3     b_pulses_test, b_meta_test = load_batch("test", TEST_BIDS[0])
      4 
      5     display(HTML("Train meta:"))

/tmp/ipykernel_11/3314951761.py in load_batch(folder, b_id)
     20     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     21 
---> 22     meta = pl.read_parquet(f"{folder}/batch_id={b_id}/*.parquet")
     23 
     24     return pulses, meta

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/io/parquet/functions.py in read_parquet(source, columns, n_rows, row_index_name, row_index_offset, parallel, use_statistics, hive_partitioning, glob, schema, hive_schema, try_parse_hive_dates, rechunk, low_memory, storage_options, credential_provider, retries, use_pyarrow, pyarrow_options, memory_map, include_file_paths, allow_missing_columns)
    250             lf = lf.select(columns)
    251 
--> 252     return lf.collect()
    253 
    254 

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ComputeError: expected at least 1 source

This error occurred with the following context stack:
	[1] 'parquet scan'
	[2] 'sink'


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
iplot(plot_event(*load_single_event("train", 0, 18)))


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_11/264326845.py in <cell line: 0>()
----> 1 iplot(plot_event(*load_single_event("train", 0, 18)))

/tmp/ipykernel_11/1430077469.py in load_single_event(folder, i_batch, i_evt, ignore_aux)
     25     """
     26     bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
---> 27     pulses, meta = load_batch_c(folder, bids[i_batch])
     28 
     29     meta_event = meta[i_evt]

/tmp/ipykernel_11/3314951761.py in load_batch(folder, b_id)
     20     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     21 
---> 22     meta = pl.read_parquet(f"{folder}/batch_id={b_id}/*.parquet")
     23 
     24     return pulses, meta

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/io/parquet/functions.py in read_parquet(source, columns, n_rows, row_index_name, row_index_offset, parallel, use_statistics, hive_partitioning, glob, schema, hive_schema, try_parse_hive_dates, rechunk, low_memory, storage_options, credential_provider, retries, use_pyarrow, pyarrow_options, memory_map, include_file_paths, allow_missing_columns)
    250             lf = lf.select(columns)
    251 
--> 252     return lf.collect()
    253 
    254 

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ComputeError: expected at least 1 source

This error occurred with the following context stack:
	[1] 'parquet scan'
	[2] 'sink'


## === cell 12
iplot(plot_event(*load_single_event("train", 0, 21)))


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_11/637419598.py in <cell line: 0>()
----> 1 iplot(plot_event(*load_single_event("train", 0, 21)))

/tmp/ipykernel_11/1430077469.py in load_single_event(folder, i_batch, i_evt, ignore_aux)
     25     """
     26     bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
---> 27     pulses, meta = load_batch_c(folder, bids[i_batch])
     28 
     29     meta_event = meta[i_evt]

/tmp/ipykernel_11/3314951761.py in load_batch(folder, b_id)
     20     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     21 
---> 22     meta = pl.read_parquet(f"{folder}/batch_id={b_id}/*.parquet")
     23 
     24     return pulses, meta

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/io/parquet/functions.py in read_parquet(source, columns, n_rows, row_index_name, row_index_offset, parallel, use_statistics, hive_partitioning, glob, schema, hive_schema, try_parse_hive_dates, rechunk, low_memory, storage_options, credential_provider, retries, use_pyarrow, pyarrow_options, memory_map, include_file_paths, allow_missing_columns)
    250             lf = lf.select(columns)
    251 
--> 252     return lf.collect()
    253 
    254 

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ComputeError: expected at least 1 source

This error occurred with the following context stack:
	[1] 'parquet scan'
	[2] 'sink'


## === cell 13
iplot(plot_event(*load_single_event("train", 0, 21, ignore_aux=False)))


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2155523880.py in <cell line: 0>()
----> 1 iplot(plot_event(*load_single_event("train", 0, 21, ignore_aux=False)))

/tmp/ipykernel_11/1430077469.py in load_single_event(folder, i_batch, i_evt, ignore_aux)
     25     """
     26     bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
---> 27     pulses, meta = load_batch_c(folder, bids[i_batch])
     28 
     29     meta_event = meta[i_evt]

/tmp/ipykernel_11/3314951761.py in load_batch(folder, b_id)
     20     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     21 
---> 22     meta = pl.read_parquet(f"{folder}/batch_id={b_id}/*.parquet")
     23 
     24     return pulses, meta

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/io/parquet/functions.py in read_parquet(source, columns, n_rows, row_index_name, row_index_offset, parallel, use_statistics, hive_partitioning, glob, schema, hive_schema, try_parse_hive_dates, rechunk, low_memory, storage_options, credential_provider, retries, use_pyarrow, pyarrow_options, memory_map, include_file_paths, allow_missing_columns)
    250             lf = lf.select(columns)
    251 
--> 252     return lf.collect()
    253 
    254 

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ComputeError: expected at least 1 source

This error occurred with the following context stack:
	[1] 'parquet scan'
	[2] 'sink'


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
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_11/830251705.py in <cell line: 0>()
     45     plt.colorbar();
     46 
---> 47 plot_example()

/tmp/ipykernel_11/830251705.py in plot_example(i_batch, i_event)
     35 # Note that we are plotting log(1 + amplitude).
     36 def plot_example(i_batch=0, i_event=4):
---> 37     b_pulses, _ = load_batch_c("train", TRAIN_BIDS[i_batch])
     38     e_id = b_pulses["event_id"].unique().sort()[i_event]
     39     prep_batch = preprocess_batch(b_pulses.filter(pl.col("event_id") == e_id))

/tmp/ipykernel_11/3314951761.py in load_batch(folder, b_id)
     20     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     21 
---> 22     meta = pl.read_parquet(f"{folder}/batch_id={b_id}/*.parquet")
     23 
     24     return pulses, meta

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/io/parquet/functions.py in read_parquet(source, columns, n_rows, row_index_name, row_index_offset, parallel, use_statistics, hive_partitioning, glob, schema, hive_schema, try_parse_hive_dates, rechunk, low_memory, storage_options, credential_provider, retries, use_pyarrow, pyarrow_options, memory_map, include_file_paths, allow_missing_columns)
    250             lf = lf.select(columns)
    251 
--> 252     return lf.collect()
    253 
    254 

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ComputeError: expected at least 1 source

This error occurred with the following context stack:
	[1] 'parquet scan'
	[2] 'sink'


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

trainer = ptl.Trainer(
    max_epochs=30,
    callbacks=ptl.callbacks.ModelCheckpoint(monitor="val_loss"),
    accelerator='gpu', devices=1,
)
trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
MisconfigurationException                 Traceback (most recent call last)
/tmp/ipykernel_11/3493076192.py in <cell line: 0>()
     25 val_dataloader = torch.utils.data.DataLoader(val_dataset)
     26 
---> 27 trainer = ptl.Trainer(
     28     max_epochs=30,
     29     callbacks=ptl.callbacks.ModelCheckpoint(monitor="val_loss"),

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/argparse.py in insert_env_defaults(self, *args, **kwargs)
     68 
     69         # all args were already moved to kwargs
---> 70         return fn(self, **kwargs)
     71 
     72     return cast(_T, insert_env_defaults)

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in __init__(self, accelerator, strategy, devices, num_nodes, precision, logger, callbacks, fast_dev_run, max_epochs, min_epochs, max_steps, min_steps, max_time, limit_train_batches, limit_val_batches, limit_test_batches, limit_predict_batches, overfit_batches, val_check_interval, check_val_every_n_epoch, num_sanity_val_steps, log_every_n_steps, enable_checkpointing, enable_progress_bar, enable_model_summary, accumulate_grad_batches, gradient_clip_val, gradient_clip_algorithm, deterministic, benchmark, inference_mode, use_distributed_sampler, profiler, detect_anomaly, barebones, plugins, sync_batchnorm, reload_dataloaders_every_n_epochs, default_root_dir, model_registry)
    402         self._data_connector = _DataConnector(self)
    403 
--> 404         self._accelerator_connector = _AcceleratorConnector(
    405             devices=devices,
    406             accelerator=accelerator,

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py in __init__(self, devices, num_nodes, accelerator, strategy, plugins, precision, sync_batchnorm, benchmark, use_distributed_sampler, deterministic)
    142             self._accelerator_flag = self._choose_auto_accelerator()
    143         elif self._accelerator_flag == "gpu":
--> 144             self._accelerator_flag = self._choose_gpu_accelerator_backend()
    145 
    146         self._check_device_config_and_set_final_flags(devices=devices, num_nodes=num_nodes)

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py in _choose_gpu_accelerator_backend()
    346         if CUDAAccelerator.is_available():
    347             return "cuda"
--> 348         raise MisconfigurationException("No supported gpu backend found!")
    349 
    350     def _set_parallel_devices_and_init_accelerator(self) -> None:

MisconfigurationException: No supported gpu backend found!

## === cell 21
print(trainer.checkpoint_callback.best_model_score.detach().cpu().numpy())
print(trainer.checkpoint_callback.best_model_path)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2410093261.py in <cell line: 0>()
----> 1 print(trainer.checkpoint_callback.best_model_score.detach().cpu().numpy())
      2 print(trainer.checkpoint_callback.best_model_path)

NameError: name 'trainer' is not defined

## === cell 22
checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
lit_model.load_state_dict(checkpoint["state_dict"])


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4272834056.py in <cell line: 0>()
----> 1 checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
      2 lit_model.load_state_dict(checkpoint["state_dict"])

NameError: name 'trainer' is not defined

## === cell 23
trainer.validate(lit_model, val_dataloader)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2074042176.py in <cell line: 0>()
----> 1 trainer.validate(lit_model, val_dataloader)

NameError: name 'trainer' is not defined

## === cell 24
def angles2vec(azimuth, zenith):
    x = torch.cos(azimuth) * torch.sin(zenith)
    y = torch.sin(azimuth) * torch.sin(zenith)
    z = torch.cos(zenith)
    return torch.stack([x, y, z], axis=1)

def vec2angles(vec):
    norm = ((vec**2).sum(axis=1)**0.5)[:, None]
    vec = vec / norm
    zenith = torch.acos(vec[:, 2])
    sin_zenith = torch.sin(zenith)
    cos_azimuth = vec[:, 0] / sin_zenith
    sin_azimuth = vec[:, 1] / sin_zenith
    azimuth = torch.atan2(sin_azimuth, cos_azimuth)
    return azimuth, zenith


## === cell 25
test_ds = Dataset("test", list(range(len(TEST_BIDS))), batch_size=1000, drop_each_last=False, with_evt_id=True)

def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return x
with torch.no_grad():
    predictions_df = pd.concat([
        pd.DataFrame({
            name: torch2numpy(value) for name, value in zip(
                ["event_id", "azimuth", "zenith"],
                (eids.to_numpy(),) + vec2angles(model(X))
            )
        }) for X, _, eids in test_ds
    ])

predictions_df.to_csv("submission.csv", index=False)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2484084218.py in <cell line: 0>()
      6     return x
      7 with torch.no_grad():
----> 8     predictions_df = pd.concat([
      9         pd.DataFrame({
     10             name: torch2numpy(value) for name, value in zip(

/tmp/ipykernel_11/2484084218.py in <listcomp>(.0)
      6     return x
      7 with torch.no_grad():
----> 8     predictions_df = pd.concat([
      9         pd.DataFrame({
     10             name: torch2numpy(value) for name, value in zip(

/tmp/ipykernel_11/3638244729.py in __iter__(self)
     29         for i in np.random.choice(len(self.file_ids), len(self.file_ids), replace=False):
     30             gc.collect()
---> 31             batch_x, batch_y = load_and_preprocess_batch(
     32                 self.folder, self.file_ids[i], num_subsample=self.num_subsample
     33             )

/tmp/ipykernel_11/113142534.py in load_and_preprocess_batch(folder, i_batch, num_subsample)
      3 def load_and_preprocess_batch(folder, i_batch, num_subsample=None):
      4     bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
----> 5     return preprocessing_cache.load_and_preprocess_batch(
      6         folder, bids[i_batch], load_batch, preprocess_batch, num_subsample
      7     )

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
     33         Corresponding meta info.
     34     """
---> 35     batch_pulses, batch_meta = load_batch(folder, batch_id)
     36     if num_subsample is not None:
     37         ids = batch_pulses["event_id"].unique()

/tmp/ipykernel_11/3314951761.py in load_batch(folder, b_id)
     20     pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
     21 
---> 22     meta = pl.read_parquet(f"{folder}/batch_id={b_id}/*.parquet")
     23 
     24     return pulses, meta

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/io/parquet/functions.py in read_parquet(source, columns, n_rows, row_index_name, row_index_offset, parallel, use_statistics, hive_partitioning, glob, schema, hive_schema, try_parse_hive_dates, rechunk, low_memory, storage_options, credential_provider, retries, use_pyarrow, pyarrow_options, memory_map, include_file_paths, allow_missing_columns)
    250             lf = lf.select(columns)
    251 
--> 252     return lf.collect()
    253 
    254 

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ComputeError: expected at least 1 source

This error occurred with the following context stack:
	[1] 'parquet scan'
	[2] 'sink'


## === cell 26
!head submission.csv

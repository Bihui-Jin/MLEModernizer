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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass



## === cell 1
batch_size = 32
block_size = 256  # maximum context length
max_iters = 5000  # number of iterations
change_batch_every = 256  # change batch of data every change_batch_every steps
device = "cuda" if torch.cuda.is_available() else "cpu"
evaluate_every_step = 500
learning_rate = 1e-4
eval_iters = 32  # number of batches to process for evaluation

embed_dim = 64 - 5
n_layers = 6
num_heads = 6
dropout = 0.2
max_time = 77785
dt = 10  # resolution in nanoseconds. By lowering this number you increase your resolution.
t_max = 4e6  # maximum cumulative time

num_time = int(max_time / dt + 1)
num_sensors = 5160



## === cell 2
df_sensor_geometry = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv"
)
sensor_ids = sorted(list(set(df_sensor_geometry["sensor_id"].to_list())))
sensor_ids[:10], len(sensor_ids)



## === cell 3
_geo = df_sensor_geometry[["sensor_id", "x", "y", "z"]].copy()
_geo["sensor_id"] = _geo["sensor_id"].astype(np.int32)
_max_sensor_id = int(_geo["sensor_id"].max())
SENSOR_XYZ_LUT = np.zeros((_max_sensor_id + 1, 3), dtype=np.float32)
SENSOR_XYZ_LUT[_geo["sensor_id"].to_numpy()] = _geo[["x", "y", "z"]].to_numpy(
    np.float32
)

_INV_500 = np.float32(1.0 / 500.0)
_TIME_SCALE = np.float32(1.0 / (dt * t_max))



## === cell 4
try:
    from numba import njit
except Exception:
    njit = None

if njit is not None:

    @njit(cache=True, fastmath=False)
    def _build_features_batch_numba(
        first_idx_batch,
        last_idx_batch,
        sensor_all,
        time_all,
        charge_all,
        sensor_xyz_lut,
        inv_500,
        time_scale,
        out_xyz,
        out_time,
        out_charge,
    ):
        bs = first_idx_batch.shape[0]
        n_rows = sensor_all.shape[0]
        T = out_time.shape[1]  # block_size
        for b in range(bs):
            for t in range(T):
                out_time[b, t] = 0.0
                out_charge[b, t] = 0.0
                out_xyz[b, t, 0] = 0.0
                out_xyz[b, t, 1] = 0.0
                out_xyz[b, t, 2] = 0.0

            start = int(first_idx_batch[b])
            end_excl = int(last_idx_batch[b]) + 1
            if start < 0 or end_excl <= start or start >= n_rows:
                continue
            if end_excl > n_rows:
                end_excl = n_rows

            event_size = end_excl - start
            n = T if event_size >= T else event_size
            if n <= 0:
                continue

            mx = 0.0
            my = 0.0
            mz = 0.0
            for i in range(n):
                sid = int(sensor_all[start + i])
                mx += sensor_xyz_lut[sid, 0]
                my += sensor_xyz_lut[sid, 1]
                mz += sensor_xyz_lut[sid, 2]
            invn = 1.0 / n
            mx *= invn
            my *= invn
            mz *= invn

            csum = 0.0
            for i in range(n):
                sid = int(sensor_all[start + i])
                x = (sensor_xyz_lut[sid, 0] - mx) * inv_500
                y = (sensor_xyz_lut[sid, 1] - my) * inv_500
                z = (sensor_xyz_lut[sid, 2] - mz) * inv_500
                out_xyz[b, i, 0] = x
                out_xyz[b, i, 1] = y
                out_xyz[b, i, 2] = z

                csum += float(time_all[start + i]) * time_scale
                out_time[b, i] = csum

                c = float(charge_all[start + i])
                if c < 0.0:
                    c = 0.0
                elif c > 5.0:
                    c = 5.0
                out_charge[b, i] = c




## === cell 5
try:
    import fastparquet  # noqa: F401

    _PARQUET_ENGINE = "fastparquet"
except Exception:
    _PARQUET_ENGINE = "auto"


class EventDataLoader:
    """
    Speed-critical refactor (correctness-preserving):
    - Keep batch parquet loaded and cache the needed columns as contiguous NumPy arrays.
    - Provide a Numba-accelerated batch feature builder (same math as before).
    """

    def __init__(self, batch_cache_size=2):
        np.random.seed(42)
        print("Loading large input test meta parquet file ...")
        df_meta_eval = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet",
            engine=_PARQUET_ENGINE if _PARQUET_ENGINE != "auto" else None,
        )

        self.change_batch_every = change_batch_every
        self.meta_data = {"eval": df_meta_eval.reset_index(drop=True)}

        _batch_ids = self.meta_data["eval"]["batch_id"].to_numpy(np.int32, copy=False)
        self._unique_batch_ids = np.unique(_batch_ids).astype(np.int32, copy=False)

        self.batch_id_current = {"eval": int(self._unique_batch_ids[0])}
        self.meta_data_current = {"eval": self._select_meta_batch("eval")}

        self._events_np = {"eval": None}

        self._batch_cache_size = int(batch_cache_size)
        self._batch_cache = {}  # batch_id -> dict(sensor_id,time,charge)
        self._batch_cache_order = []  # insertion order

        self.load_batch_events("eval")
        self.counter = 0  # API compatibility

    def _select_meta_batch(self, split):
        assert split == "eval"
        batch_id = self.batch_id_current[split]
        dfm = self.meta_data[split]
        mask = dfm["batch_id"].to_numpy(np.int32, copy=False) == batch_id
        return dfm.loc[mask].reset_index(drop=True)

    def shuffle_metadata_batch(self, split):
        assert split == "eval"
        return self._select_meta_batch(split)

    def _cache_put(self, batch_id, arrays):
        if batch_id in self._batch_cache:
            return
        self._batch_cache[batch_id] = arrays
        self._batch_cache_order.append(batch_id)
        if len(self._batch_cache_order) > self._batch_cache_size:
            old = self._batch_cache_order.pop(0)
            self._batch_cache.pop(old, None)

    def load_batch_events(self, split):
        """Load a batch parquet file and cache relevant columns as NumPy arrays."""
        assert split == "eval"
        batch_id = int(self.batch_id_current[split])

        if batch_id in self._batch_cache:
            self._events_np[split] = self._batch_cache[batch_id]
            return None

        special_dir = "test"
        df = pd.read_parquet(
            f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet",
            columns=["time", "sensor_id", "charge"],
            engine=_PARQUET_ENGINE if _PARQUET_ENGINE != "auto" else None,
        )

        arrays = {
            "sensor_id": np.ascontiguousarray(
                df["sensor_id"].to_numpy(np.int32, copy=False)
            ),
            "time": np.ascontiguousarray(df["time"].to_numpy(np.float32, copy=False)),
            "charge": np.ascontiguousarray(
                df["charge"].to_numpy(np.float32, copy=False)
            ),
        }
        self._events_np[split] = arrays
        self._cache_put(batch_id, arrays)
        return df

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        assert split in ["train", "test", "eval"]

        start = int(first_pulse_index)
        end_excl = int(last_pulse_index) + 1

        sensor_all = self._events_np[split]["sensor_id"]
        time_all = self._events_np[split]["time"]
        charge_all = self._events_np[split]["charge"]

        event_size = max(0, end_excl - start)
        n = block_size if event_size >= block_size else event_size

        xyz = np.zeros((block_size, 3), dtype=np.float32)
        tt = np.zeros((block_size,), dtype=np.float32)
        ch = np.zeros((block_size,), dtype=np.float32)

        if n > 0:
            sensor = sensor_all[start : start + n]
            traw = time_all[start : start + n]
            craw = charge_all[start : start + n]

            _xyz = SENSOR_XYZ_LUT[sensor]
            m = _xyz.mean(axis=0, dtype=np.float32)
            xyz[:n] = (_xyz - m) * _INV_500

            tt[:n] = np.cumsum(traw * _TIME_SCALE, dtype=np.float32)
            np.clip(craw, 0.0, 5.0, out=ch[:n])

        return (torch.from_numpy(xyz), torch.from_numpy(tt), torch.from_numpy(ch))

    def fill_batch_tensors(
        self,
        first_idx_batch,
        last_idx_batch,
        out_xyz,
        out_time,
        out_charge,
        split="eval",
    ):
        """
        Speed-critical (correctness-preserving):
        - Use a compiled Numba kernel when available to build the whole mini-batch.
        - Avoid per-event torch ops/allocations; fill the provided buffers directly.
        """
        assert split == "eval"
        bs = int(first_idx_batch.shape[0])
        if bs == 0:
            return

        sensor_all = self._events_np[split]["sensor_id"]
        time_all = self._events_np[split]["time"]
        charge_all = self._events_np[split]["charge"]

        if njit is not None:
            _build_features_batch_numba(
                first_idx_batch.astype(np.int64, copy=False),
                last_idx_batch.astype(np.int64, copy=False),
                sensor_all,
                time_all,
                charge_all,
                SENSOR_XYZ_LUT,
                float(_INV_500),
                float(_TIME_SCALE),
                out_xyz,  # numpy arrays here
                out_time,
                out_charge,
            )
            return

        n_rows = int(sensor_all.shape[0])
        out_xyz.fill(0.0)
        out_time.fill(0.0)
        out_charge.fill(0.0)
        for b in range(bs):
            start = int(first_idx_batch[b])
            end_excl = int(last_idx_batch[b]) + 1
            if start < 0 or end_excl <= start or start >= n_rows:
                continue
            if end_excl > n_rows:
                end_excl = n_rows

            event_size = end_excl - start
            n = block_size if event_size >= block_size else event_size
            if n <= 0:
                continue

            sensor = sensor_all[start : start + n]
            traw = time_all[start : start + n]
            craw = charge_all[start : start + n]

            xyz = SENSOR_XYZ_LUT[sensor]
            m = xyz.mean(axis=0, dtype=np.float32)
            out_xyz[b, :n, :] = (xyz - m) * _INV_500
            out_time[b, :n] = np.cumsum(traw * _TIME_SCALE, dtype=np.float32)
            np.clip(craw, 0.0, 5.0, out=out_charge[b, :n])

    def change_event_batch(self):
        cur = self.batch_id_current["eval"]
        idx = int(np.searchsorted(self._unique_batch_ids, cur))
        idx = min(idx + 1, len(self._unique_batch_ids) - 1)
        self.batch_id_current["eval"] = int(self._unique_batch_ids[idx])

        self.meta_data_current = {"eval": self.shuffle_metadata_batch("eval")}
        self.load_batch_events("eval")
        self.counter = 0

    def get_xy(self, split):
        raise RuntimeError(
            "Training API disabled in this inference-only optimized script."
        )




## === cell 6
try:
    import fastparquet  # noqa: F401

    _PARQUET_ENGINE = "fastparquet"
except Exception:
    try:
        import pyarrow  # noqa: F401

        _PARQUET_ENGINE = "pyarrow"
    except Exception as e:
        raise ImportError(
            "No supported parquet engine found. Please install 'pyarrow' or 'fastparquet'."
        ) from e


class EventDataLoader:
    """
    Speed-critical refactor (correctness-preserving):
    - Keep batch parquet loaded and cache the needed columns as contiguous NumPy arrays.
    - Provide a Numba-accelerated batch feature builder (same math as before).
    """

    def __init__(self, batch_cache_size=2):
        np.random.seed(42)
        print("Loading large input test meta parquet file ...")
        df_meta_eval = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet",
            engine=_PARQUET_ENGINE,
        )

        self.change_batch_every = change_batch_every
        self.meta_data = {"eval": df_meta_eval.reset_index(drop=True)}

        _batch_ids = self.meta_data["eval"]["batch_id"].to_numpy(np.int32, copy=False)
        self._unique_batch_ids = np.unique(_batch_ids).astype(np.int32, copy=False)

        self.batch_id_current = {"eval": int(self._unique_batch_ids[0])}
        self.meta_data_current = {"eval": self._select_meta_batch("eval")}

        self._events_np = {"eval": None}

        self._batch_cache_size = int(batch_cache_size)
        self._batch_cache = {}  # batch_id -> dict(sensor_id,time,charge)
        self._batch_cache_order = []  # insertion order

        self.load_batch_events("eval")
        self.counter = 0  # API compatibility

    def _select_meta_batch(self, split):
        assert split == "eval"
        batch_id = self.batch_id_current[split]
        dfm = self.meta_data[split]
        mask = dfm["batch_id"].to_numpy(np.int32, copy=False) == batch_id
        return dfm.loc[mask].reset_index(drop=True)

    def shuffle_metadata_batch(self, split):
        assert split == "eval"
        return self._select_meta_batch(split)

    def _cache_put(self, batch_id, arrays):
        if batch_id in self._batch_cache:
            return
        self._batch_cache[batch_id] = arrays
        self._batch_cache_order.append(batch_id)
        if len(self._batch_cache_order) > self._batch_cache_size:
            old = self._batch_cache_order.pop(0)
            self._batch_cache.pop(old, None)

    def load_batch_events(self, split):
        """Load a batch parquet file and cache relevant columns as NumPy arrays."""
        assert split == "eval"
        batch_id = int(self.batch_id_current[split])

        if batch_id in self._batch_cache:
            self._events_np[split] = self._batch_cache[batch_id]
            return None

        special_dir = "test"
        df = pd.read_parquet(
            f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet",
            columns=["time", "sensor_id", "charge"],
            engine=_PARQUET_ENGINE,
        )

        arrays = {
            "sensor_id": np.ascontiguousarray(
                df["sensor_id"].to_numpy(np.int32, copy=False)
            ),
            "time": np.ascontiguousarray(df["time"].to_numpy(np.float32, copy=False)),
            "charge": np.ascontiguousarray(
                df["charge"].to_numpy(np.float32, copy=False)
            ),
        }
        self._events_np[split] = arrays
        self._cache_put(batch_id, arrays)
        return df

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        assert split in ["train", "test", "eval"]

        start = int(first_pulse_index)
        end_excl = int(last_pulse_index) + 1

        sensor_all = self._events_np[split]["sensor_id"]
        time_all = self._events_np[split]["time"]
        charge_all = self._events_np[split]["charge"]

        event_size = max(0, end_excl - start)
        n = block_size if event_size >= block_size else event_size

        xyz = np.zeros((block_size, 3), dtype=np.float32)
        tt = np.zeros((block_size,), dtype=np.float32)
        ch = np.zeros((block_size,), dtype=np.float32)

        if n > 0:
            sensor = sensor_all[start : start + n]
            traw = time_all[start : start + n]
            craw = charge_all[start : start + n]

            _xyz = SENSOR_XYZ_LUT[sensor]
            m = _xyz.mean(axis=0, dtype=np.float32)
            xyz[:n] = (_xyz - m) * _INV_500

            tt[:n] = np.cumsum(traw * _TIME_SCALE, dtype=np.float32)
            np.clip(craw, 0.0, 5.0, out=ch[:n])

        return (torch.from_numpy(xyz), torch.from_numpy(tt), torch.from_numpy(ch))

    def fill_batch_tensors(
        self,
        first_idx_batch,
        last_idx_batch,
        out_xyz,
        out_time,
        out_charge,
        split="eval",
    ):
        """
        Speed-critical (correctness-preserving):
        - Use a compiled Numba kernel when available to build the whole mini-batch.
        - Avoid per-event torch ops/allocations; fill the provided buffers directly.
        """
        assert split == "eval"
        bs = int(first_idx_batch.shape[0])
        if bs == 0:
            return

        sensor_all = self._events_np[split]["sensor_id"]
        time_all = self._events_np[split]["time"]
        charge_all = self._events_np[split]["charge"]

        if njit is not None:
            _build_features_batch_numba(
                first_idx_batch.astype(np.int64, copy=False),
                last_idx_batch.astype(np.int64, copy=False),
                sensor_all,
                time_all,
                charge_all,
                SENSOR_XYZ_LUT,
                float(_INV_500),
                float(_TIME_SCALE),
                out_xyz,  # numpy arrays here
                out_time,
                out_charge,
            )
            return

        n_rows = int(sensor_all.shape[0])
        out_xyz.fill(0.0)
        out_time.fill(0.0)
        out_charge.fill(0.0)
        for b in range(bs):
            start = int(first_idx_batch[b])
            end_excl = int(last_idx_batch[b]) + 1
            if start < 0 or end_excl <= start or start >= n_rows:
                continue
            if end_excl > n_rows:
                end_excl = n_rows

            event_size = end_excl - start
            n = block_size if event_size >= block_size else event_size
            if n <= 0:
                continue

            sensor = sensor_all[start : start + n]
            traw = time_all[start : start + n]
            craw = charge_all[start : start + n]

            xyz = SENSOR_XYZ_LUT[sensor]
            m = xyz.mean(axis=0, dtype=np.float32)
            out_xyz[b, :n, :] = (xyz - m) * _INV_500
            out_time[b, :n] = np.cumsum(traw * _TIME_SCALE, dtype=np.float32)
            np.clip(craw, 0.0, 5.0, out=out_charge[b, :n])

    def change_event_batch(self):
        cur = self.batch_id_current["eval"]
        idx = int(np.searchsorted(self._unique_batch_ids, cur))
        idx = min(idx + 1, len(self._unique_batch_ids) - 1)
        self.batch_id_current["eval"] = int(self._unique_batch_ids[idx])

        self.meta_data_current = {"eval": self.shuffle_metadata_batch("eval")}
        self.load_batch_events("eval")
        self.counter = 0

    def get_xy(self, split):
        raise RuntimeError(
            "Training API disabled in this inference-only optimized script."
        )


## === cell 7
_df0 = data_loader.meta_data["eval"].iloc[0]
xyz0, time0, charge0 = data_loader.get_single_event(
    int(_df0["first_pulse_index"]), int(_df0["last_pulse_index"]), split="eval"
)
xyz0[:2], time0[:5], charge0[:5]




## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3179556211.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0m_df0[0m [0;34m=[0m [0mdata_loader[0m[0;34m.[0m[0mmeta_data[0m[0;34m[[0m[0;34m"eval"[0m[0;34m][0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m xyz0, time0, charge0 = data_loader.get_single_event(
[1;32m      3[0m     [0mint[0m[0;34m([0m[0m_df0[0m[0;34m[[0m[0;34m"first_pulse_index"[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0mint[0m[0;34m([0m[0m_df0[0m[0;34m[[0m[0;34m"last_pulse_index"[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0msplit[0m[0;34m=[0m[0;34m"eval"[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m )
[1;32m      5[0m [0mxyz0[0m[0;34m[[0m[0;34m:[0m[0;36m2[0m[0;34m][0m[0;34m,[0m [0mtime0[0m[0;34m[[0m[0;34m:[0m[0;36m5[0m[0;34m][0m[0;34m,[0m [0mcharge0[0m[0;34m[[0m[0;34m:[0m[0;36m5[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'data_loader' is not defined

## === cell 8
def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    if not (
        torch.all(torch.isfinite(az_true))
        and torch.all(torch.isfinite(zen_true))
        and torch.all(torch.isfinite(az_pred))
        and torch.all(torch.isfinite(zen_pred))
    ):
        raise ValueError("All arguments must be finite")

    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)

    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)

    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)
    scalar_prod = torch.clamp(scalar_prod, -1, 1)
    return torch.mean(torch.abs(torch.acos(scalar_prod)))

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
import os
import glob
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict



## === cell 1
batch_size = 32
block_size = 256  # maximum context length
max_iters = 5000  # number of iterations (not used in this notebook as provided)
change_batch_every = 256  # change batch of data every change_batch_every steps
device = "cuda" if torch.cuda.is_available() else "cpu"
evaluate_every_step = 500
learning_rate = 1e-4
eval_iters = 32  # number of batches to process for evaluation

embed_dim = 55

n_layers = 6
num_heads = 6
dropout = 0.2
max_time = 77785
dt = 10  # resolution in nanoseconds. By lowering this number you increase your resolution.
t_max = 4e6  # maximum cumulative time

num_time = int(max_time / dt + 1)
num_sensors = 5160

torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass



## === cell 2
DATA_ROOT = "/kaggle/input/icecube-neutrinos-in-deep-ice"


def _resolve_batch_dir(split: str) -> str:
    candidates = [
        os.path.join(DATA_ROOT, split),
        os.path.join(DATA_ROOT, "icecube-neutrinos-in-deep-ice", split),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


TRAIN_DIR = _resolve_batch_dir("train")
TEST_DIR = _resolve_batch_dir("test")


def _existing_batch_ids(batch_dir: str) -> list[int]:
    files = glob.glob(os.path.join(batch_dir, "batch_*.parquet"))
    ids = []
    for f in files:
        base = os.path.basename(f)
        try:
            ids.append(int(base.replace("batch_", "").replace(".parquet", "")))
        except Exception:
            pass
    ids.sort()
    return ids


TRAIN_BATCH_IDS = _existing_batch_ids(TRAIN_DIR)
TEST_BATCH_IDS = _existing_batch_ids(TEST_DIR)

if len(TRAIN_BATCH_IDS) == 0 or len(TEST_BATCH_IDS) == 0:
    raise RuntimeError(
        f"Could not find batch parquet files. TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}"
    )

print(
    "Resolved TRAIN_DIR:",
    TRAIN_DIR,
    "num_batches:",
    len(TRAIN_BATCH_IDS),
    "min/max:",
    (TRAIN_BATCH_IDS[0], TRAIN_BATCH_IDS[-1]),
)
print(
    "Resolved TEST_DIR :",
    TEST_DIR,
    "num_batches:",
    len(TEST_BATCH_IDS),
    "min/max:",
    (TEST_BATCH_IDS[0], TEST_BATCH_IDS[-1]),
)



## === cell 3
df_sensor_geometry = pd.read_csv(os.path.join(DATA_ROOT, "sensor_geometry.csv"))
sensor_ids = sorted(list(set(df_sensor_geometry["sensor_id"].to_list())))
sensor_ids[:10], len(sensor_ids)




## === cell 4
class EventDataLoader:
    def __init__(self, n_test_data=500_000):
        np.random.seed(42)

        meta_cols_train = [
            "batch_id",
            "event_id",
            "first_pulse_index",
            "last_pulse_index",
            "azimuth",
            "zenith",
        ]
        meta_cols_test = [
            "batch_id",
            "event_id",
            "first_pulse_index",
            "last_pulse_index",
        ]

        print("Loading large input train meta parquet file (columns pruned) ...")
        df_meta = pd.read_parquet(
            os.path.join(DATA_ROOT, "train_meta.parquet"), columns=meta_cols_train
        )

        print("Loading large input test meta parquet file (columns pruned) ...")
        df_meta_eval = pd.read_parquet(
            os.path.join(DATA_ROOT, "test_meta.parquet"), columns=meta_cols_test
        )

        n_train_data = len(df_meta) - n_test_data
        self.change_batch_every = change_batch_every
        self.meta_data = {
            "train": df_meta[:n_train_data].reset_index(drop=True),
            "test": df_meta[n_train_data:].reset_index(drop=True),
            "eval": df_meta_eval.reset_index(drop=True),
        }
        self.batch_id_current = {"train": 0, "test": 0, "eval": 0}

        geom = df_sensor_geometry[["sensor_id", "x", "y", "z"]].copy()
        self._geom_min_id = int(geom["sensor_id"].min())
        self._geom_max_id = int(geom["sensor_id"].max())
        lut_len = self._geom_max_id - self._geom_min_id + 1
        self._xyz_lut = np.full((lut_len, 3), np.nan, dtype=np.float32)
        idx = geom["sensor_id"].values.astype(np.int64) - self._geom_min_id
        self._xyz_lut[idx] = geom[["x", "y", "z"]].values.astype(np.float32, copy=False)

        self._events_np = {}
        self._events_cache = {
            "train": OrderedDict(),
            "test": OrderedDict(),
            "eval": OrderedDict(),
        }
        self._events_cache_max = {"train": 2, "test": 2, "eval": 8}

        self._buf_xyz = np.zeros((block_size, 3), dtype=np.float32)
        self._buf_time = np.zeros((block_size,), dtype=np.float32)
        self._buf_charge = np.zeros((block_size,), dtype=np.float32)

        self._pin = device.startswith("cuda")
        self._t_xyz = torch.empty(
            (block_size, 3), dtype=torch.float32, pin_memory=self._pin
        )
        self._t_time = torch.empty(
            (block_size,), dtype=torch.float32, pin_memory=self._pin
        )
        self._t_charge = torch.empty(
            (block_size,), dtype=torch.float32, pin_memory=self._pin
        )
        self._t_xyz_np = self._t_xyz.numpy()
        self._t_time_np = self._t_time.numpy()
        self._t_charge_np = self._t_charge.numpy()

        self._time_scale = np.float32(1.0 / (float(dt) * float(t_max)))

        self.meta_data_current = {
            "train": self.shuffle_metadata_batch("train"),
            "test": self.shuffle_metadata_batch("test"),
            "eval": self.shuffle_metadata_batch("eval"),
        }
        self.events = {
            "train": self.load_batch_events("train"),
            "test": self.load_batch_events("test"),
            "eval": self.load_batch_events("eval"),
        }
        self.counter = 0

    def shuffle_metadata_batch(self, split):
        if split in ["train", "test"]:
            available = sorted(set(self.meta_data[split]["batch_id"].unique().tolist()))
            if len(available) == 0:
                raise RuntimeError(f"No batch_id found in meta for split={split}")
            exists = set(TRAIN_BATCH_IDS)
            available = [b for b in available if b in exists]
            if len(available) == 0:
                available = TRAIN_BATCH_IDS
            random_batch_id = int(np.random.choice(available))
            self.batch_id_current[split] = random_batch_id
        else:
            available = sorted(set(self.meta_data[split]["batch_id"].unique().tolist()))
            exists = set(TEST_BATCH_IDS)
            available = [b for b in available if b in exists]
            if len(available) == 0:
                random_batch_id = int(TEST_BATCH_IDS[0])
            else:
                random_batch_id = int(available[0])
            self.batch_id_current[split] = random_batch_id

        return self.meta_data[split][
            self.meta_data[split]["batch_id"] == self.batch_id_current[split]
        ].reset_index(drop=True)

    def _get_batch_path(self, split: str, batch_id: int) -> str:
        special_dir = TRAIN_DIR if split in ["train", "test"] else TEST_DIR
        return os.path.join(special_dir, f"batch_{int(batch_id)}.parquet")

    def _cache_put(self, split: str, batch_id: int, ev: dict):
        cache = self._events_cache[split]
        bid = int(batch_id)
        if bid in cache:
            cache.move_to_end(bid)
            cache[bid] = ev
        else:
            cache[bid] = ev
            cache.move_to_end(bid)
            max_keep = int(self._events_cache_max[split])
            while len(cache) > max_keep:
                cache.popitem(last=False)

    def load_batch_events(self, split):
        batch_id = self.batch_id_current[split]
        path = self._get_batch_path(split, batch_id)
        if not os.path.exists(path):
            valid_ids = sorted(
                set(self.meta_data_current[split]["batch_id"].unique().tolist())
            )
            valid_ids = [
                b for b in valid_ids if os.path.exists(self._get_batch_path(split, b))
            ]
            if len(valid_ids) == 0:
                fallback = (
                    TRAIN_BATCH_IDS if split in ["train", "test"] else TEST_BATCH_IDS
                )
                batch_id = int(fallback[0])
            else:
                batch_id = int(valid_ids[0])
            self.batch_id_current[split] = batch_id
            path = self._get_batch_path(split, batch_id)

        df = pd.read_parquet(path, columns=["time", "sensor_id", "charge"])
        ev = {
            "sensor_id": df["sensor_id"].to_numpy(dtype=np.int64, copy=False),
            "time": df["time"].to_numpy(dtype=np.float32, copy=False),
            "charge": df["charge"].to_numpy(dtype=np.float32, copy=False),
        }
        self._events_np[split] = ev
        self._cache_put(split, int(batch_id), ev)
        return None  # not used downstream

    def load_batch_events_by_id(self, split: str, batch_id: int):
        bid = int(batch_id)
        cache = self._events_cache[split]
        if bid in cache:
            cache.move_to_end(bid)
            self._events_np[split] = cache[bid]
            self.batch_id_current[split] = bid
            return

        path = self._get_batch_path(split, bid)
        if not os.path.exists(path):
            fallback = TRAIN_BATCH_IDS if split in ["train", "test"] else TEST_BATCH_IDS
            bid = int(fallback[0])
            path = self._get_batch_path(split, bid)

        df = pd.read_parquet(path, columns=["time", "sensor_id", "charge"])
        ev = {
            "sensor_id": df["sensor_id"].to_numpy(dtype=np.int64, copy=False),
            "time": df["time"].to_numpy(dtype=np.float32, copy=False),
            "charge": df["charge"].to_numpy(dtype=np.float32, copy=False),
        }
        self._cache_put(split, int(bid), ev)
        self._events_np[split] = ev
        self.batch_id_current[split] = int(bid)

    def _sensor_to_xyz(self, sensor_id_arr: np.ndarray) -> np.ndarray:
        sid = sensor_id_arr.astype(np.int64, copy=False)
        idx = sid - self._geom_min_id
        idx = np.clip(idx, 0, self._xyz_lut.shape[0] - 1)
        return self._xyz_lut[idx]

    def fill_single_event_np(
        self,
        first_pulse_index: int,
        last_pulse_index: int,
        split: str,
        xyz_out: np.ndarray,
        time_out: np.ndarray,
        charge_out: np.ndarray,
    ):
        assert split in ["train", "test", "eval"]
        start = int(first_pulse_index)
        end = int(last_pulse_index) + 1

        event_size = end - start
        T = block_size
        take = event_size if event_size < T else T

        xyz_out.fill(0.0)
        time_out.fill(0.0)
        charge_out.fill(0.0)

        if take > 0:
            ev = self._events_np[split]
            n_rows = int(ev["sensor_id"].shape[0])

            if start < 0 or start >= n_rows:
                take = 0
            else:
                take = min(take, n_rows - start)

            if take > 0:
                sensor_ids = ev["sensor_id"][start : start + take]
                xyz = self._sensor_to_xyz(sensor_ids)

                if not np.isfinite(xyz).all():
                    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0).astype(
                        np.float32, copy=False
                    )

                col_avg = xyz.mean(axis=0, dtype=np.float32)
                xyz_out[:take] = (xyz.astype(np.float32, copy=False) - col_avg) / 500.0

                time_scaled = ev["time"][start : start + take] * self._time_scale
                time_out[:take] = np.cumsum(time_scaled, dtype=np.float32)

                ch = ev["charge"][start : start + take].astype(np.float32, copy=False)
                np.clip(ch, 0.0, 5.0, out=charge_out[:take])

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        self.fill_single_event_np(
            first_pulse_index,
            last_pulse_index,
            split,
            self._t_xyz_np,
            self._t_time_np,
            self._t_charge_np,
        )
        return self._t_xyz.clone(), self._t_time.clone(), self._t_charge.clone()

    def change_event_batch(self):
        self.meta_data_current = {
            "train": self.shuffle_metadata_batch("train"),
            "test": self.shuffle_metadata_batch("test"),
            "eval": self.shuffle_metadata_batch("eval"),
        }
        self.events["train"] = self.load_batch_events("train")
        self.events["test"] = self.load_batch_events("test")
        self.events["eval"] = self.load_batch_events("eval")
        self.counter = 0

    def get_xy(self, split):
        assert split in ["train", "test"], "split must be either 'train' or 'test'"

        mcur = self.meta_data_current[split]
        idxs = np.random.randint(0, len(mcur), size=batch_size)
        df_meta_sample = mcur.iloc[idxs].reset_index(drop=True)

        first_pulse_indices = df_meta_sample["first_pulse_index"].to_numpy(
            dtype=np.int64, copy=False
        )
        last_pulse_indices = df_meta_sample["last_pulse_index"].to_numpy(
            dtype=np.int64, copy=False
        )

        if not hasattr(self, "_batch_buf"):
            self._batch_buf = {
                "xyz": torch.empty(
                    (batch_size, block_size, 3),
                    dtype=torch.float32,
                    pin_memory=self._pin,
                ),
                "time": torch.empty(
                    (batch_size, block_size), dtype=torch.float32, pin_memory=self._pin
                ),
                "charge": torch.empty(
                    (batch_size, block_size), dtype=torch.float32, pin_memory=self._pin
                ),
            }
            self._batch_buf_np = {
                "xyz": self._batch_buf["xyz"].numpy(),
                "time": self._batch_buf["time"].numpy(),
                "charge": self._batch_buf["charge"].numpy(),
            }

        xyz_cpu = self._batch_buf["xyz"]
        time_cpu = self._batch_buf["time"]
        charge_cpu = self._batch_buf["charge"]

        xyz_np = self._batch_buf_np["xyz"]
        time_np = self._batch_buf_np["time"]
        charge_np = self._batch_buf_np["charge"]

        for bi, (first_pulse_index, last_pulse_index) in enumerate(
            zip(first_pulse_indices, last_pulse_indices)
        ):
            self.fill_single_event_np(
                int(first_pulse_index),
                int(last_pulse_index),
                split,
                xyz_np[bi],
                time_np[bi],
                charge_np[bi],
            )

        y = torch.tensor(
            df_meta_sample[["azimuth", "zenith"]].values, dtype=torch.float32
        )

        if device.startswith("cuda"):
            xyz = xyz_cpu.to(device, non_blocking=True)
            time = time_cpu.to(device, non_blocking=True)
            charge = charge_cpu.to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            xyz = xyz_cpu.to(device)
            time = time_cpu.to(device)
            charge = charge_cpu.to(device)
            y = y.to(device)

        self.counter += 1
        if self.counter == self.change_batch_every:
            self.change_event_batch()
            self.counter = 0

        return (xyz, time, charge), y

    def fill_many_events_from_current_batch(
        self,
        fp: np.ndarray,
        lp: np.ndarray,
        split: str,
        xyz_out: np.ndarray,
        time_out: np.ndarray,
        charge_out: np.ndarray,
    ):
        assert split in ["train", "test", "eval"]
        ev = self._events_np[split]
        n_rows = int(ev["sensor_id"].shape[0])

        fp = fp.astype(np.int64, copy=False)
        lp = lp.astype(np.int64, copy=False)
        starts = fp
        ends = lp + 1

        valid_start = (starts >= 0) & (starts < n_rows)
        starts_clamped = starts.copy()
        starts_clamped[~valid_start] = 0
        ends_clamped = ends.copy()
        ends_clamped[~valid_start] = 0

        take = (ends_clamped - starts_clamped).clip(min=0)
        take = np.minimum(take, block_size)
        take = np.minimum(take, n_rows - starts_clamped)

        B = int(fp.shape[0])
        xyz_out.fill(0.0)
        time_out.fill(0.0)
        charge_out.fill(0.0)

        sensor_id_all = ev["sensor_id"]
        time_all = ev["time"]
        charge_all = ev["charge"]
        time_scale = self._time_scale

        for i in range(B):
            ti = int(take[i])
            if ti <= 0:
                continue
            s = int(starts_clamped[i])

            sensor_ids = sensor_id_all[s : s + ti]
            xyz = self._sensor_to_xyz(sensor_ids)
            if not np.isfinite(xyz).all():
                xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0).astype(
                    np.float32, copy=False
                )
            col_avg = xyz.mean(axis=0, dtype=np.float32)
            xyz_out[i, :ti, :] = (xyz.astype(np.float32, copy=False) - col_avg) / 500.0

            time_scaled = time_all[s : s + ti] * time_scale
            time_out[i, :ti] = np.cumsum(time_scaled, dtype=np.float32)

            ch = charge_all[s : s + ti].astype(np.float32, copy=False)
            np.clip(ch, 0.0, 5.0, out=charge_out[i, :ti])




## === cell 5
data_loader = EventDataLoader()



## === cell 6
(xyz, time, charge), y = data_loader.get_xy(split="train")
xyz[0].shape, time[0].shape, charge[0].shape, y[0]




## === cell 7
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




## === cell 8
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters, device="cpu")
        for k in range(eval_iters):
            (xyz, time, charge), y = data_loader.get_xy(split)
            logits, loss = model(xyz, time, charge, y)
            losses[k] = loss.detach().float().cpu()
        out[split] = losses.mean()
    model.train()
    return out




## === cell 9
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, n_embd, head_size, dropout):
        super().__init__()
        assert n_embd % num_heads == 0
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd
        self.dropout_p = float(dropout)

        self.qkv = nn.Linear(n_embd, 3 * num_heads * head_size, bias=False)
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B,T,3*H*D)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B,H,T,D)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0.0
        )  # (B,H,T,D)
        out = (
            attn.transpose(1, 2)
            .contiguous()
            .view(B, T, self.num_heads * self.head_size)
        )
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, num_heads, n_embd, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, n_embd, head_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, num_sensor, n_embd, num_time, num_heads, n_layer, dropout=0.2):
        super().__init__()
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(num_heads, n_embd + 5, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd + 5)
        self.classifier = nn.Linear(n_embd + 5, 2)

        self.register_buffer(
            "_pos_idx", torch.arange(block_size, dtype=torch.long), persistent=False
        )

        self.apply(self._init_weights)

    def forward(self, xyz, time, charge, targets=None):
        x = torch.cat([xyz, time.unsqueeze(-1), charge.unsqueeze(-1)], dim=-1)

        pos_emb = self.position_embedding_table(self._pos_idx)  # (T, n_embd)
        x = torch.cat([x, pos_emb.unsqueeze(0).expand(x.shape[0], -1, -1)], dim=-1)

        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)
        pred = self.classifier(x)

        if targets is None:
            loss = None
        else:
            loss = torch.mean(
                (targets[:, 0] - pred[:, 0]) ** 2 + (targets[:, 1] - pred[:, 1]) ** 2
            )
        return pred, loss




## === cell 10
model = TransformerModel(
    num_sensors, embed_dim, num_time, num_heads, n_layers, dropout=0.2
).to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")



## === cell 11
(xyz, time, charge), y = data_loader.get_xy("train")
print(xyz.shape, time.shape, charge.shape)
pred, loss = model(xyz, time, charge, y)
pred.shape, float(loss.item())



## === cell 12
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_steps = 2000  # keeps within typical Kaggle notebook time; adjust only if needed
model.train()
for step in tqdm(range(train_steps), desc="Training"):
    (xyz, time, charge), y = data_loader.get_xy("train")
    _, loss = model(xyz, time, charge, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if (step + 1) % evaluate_every_step == 0:
        losses = estimate_loss()
        print(
            f"step {step+1}: train_loss={losses['train']:.6f}, test_loss={losses['test']:.6f}"
        )



## === cell 13
torch.save(model.state_dict(), "model.pth")



## === cell 14
df_eval = data_loader.meta_data["eval"].copy()
df_eval.head(), len(df_eval)



## === cell 15
model.eval()

infer_bs = 2048  # semantics identical

N = len(df_eval)
predictions = np.zeros((N, 2), dtype=np.float32)

batch_all = df_eval["batch_id"].to_numpy(dtype=np.int64, copy=False)
fp_all = df_eval["first_pulse_index"].to_numpy(dtype=np.int64, copy=False)
lp_all = df_eval["last_pulse_index"].to_numpy(dtype=np.int64, copy=False)

order = np.argsort(batch_all, kind="mergesort")
batch_sorted = batch_all[order]

pin = device.startswith("cuda")

xyz_cpu_buf = torch.empty(
    (infer_bs, block_size, 3), dtype=torch.float32, pin_memory=pin
)
time_cpu_buf = torch.empty((infer_bs, block_size), dtype=torch.float32, pin_memory=pin)
charge_cpu_buf = torch.empty(
    (infer_bs, block_size), dtype=torch.float32, pin_memory=pin
)

xyz_np_view = xyz_cpu_buf.numpy()
time_np_view = time_cpu_buf.numpy()
charge_np_view = charge_cpu_buf.numpy()

if device.startswith("cuda"):
    xyz_dev_buf = torch.empty(
        (infer_bs, block_size, 3), dtype=torch.float32, device=device
    )
    time_dev_buf = torch.empty(
        (infer_bs, block_size), dtype=torch.float32, device=device
    )
    charge_dev_buf = torch.empty(
        (infer_bs, block_size), dtype=torch.float32, device=device
    )

with torch.no_grad():
    pbar = tqdm(total=N, desc="Inference (grouped by batch_id)")
    start = 0
    while start < N:
        bid = int(batch_sorted[start])
        end = start + 1
        while end < N and int(batch_sorted[end]) == bid:
            end += 1

        idxs_global = order[start:end]  # indices into original df_eval
        fp_seg = fp_all[idxs_global]
        lp_seg = lp_all[idxs_global]

        data_loader.load_batch_events_by_id("eval", bid)

        seg_len = idxs_global.shape[0]
        seg_off = 0
        while seg_off < seg_len:
            m = min(infer_bs, seg_len - seg_off)
            idxs_mb = idxs_global[seg_off : seg_off + m]

            data_loader.fill_many_events_from_current_batch(
                fp_seg[seg_off : seg_off + m],
                lp_seg[seg_off : seg_off + m],
                "eval",
                xyz_np_view[:m],
                time_np_view[:m],
                charge_np_view[:m],
            )

            if device.startswith("cuda"):
                xyz_t = xyz_dev_buf[:m]
                time_t = time_dev_buf[:m]
                charge_t = charge_dev_buf[:m]
                xyz_t.copy_(xyz_cpu_buf[:m], non_blocking=True)
                time_t.copy_(time_cpu_buf[:m], non_blocking=True)
                charge_t.copy_(charge_cpu_buf[:m], non_blocking=True)
            else:
                xyz_t, time_t, charge_t = (
                    xyz_cpu_buf[:m],
                    time_cpu_buf[:m],
                    charge_cpu_buf[:m],
                )

            pred_t, _ = model(xyz_t, time_t, charge_t, targets=None)
            predictions[idxs_mb] = pred_t.detach().float().cpu().numpy()

            seg_off += m
            pbar.update(m)

        start = end
    pbar.close()

predictions[:, 0] = np.mod(
    predictions[:, 0], np.float32(2.0 * np.pi)
)  # azimuth [0,2pi)
predictions[:, 1] = np.clip(
    predictions[:, 1], np.float32(0.0), np.float32(np.pi)
)  # zenith [0,pi]

df_pred = pd.DataFrame(
    {
        "event_id": df_eval["event_id"].values,
        "azimuth": predictions[:, 0],
        "zenith": predictions[:, 1],
    }
)
df_pred.head()



## === cell 16
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")

event_id_pred = df_pred["event_id"].to_numpy(dtype=np.int64, copy=False)
az_pred = df_pred["azimuth"].to_numpy(dtype=np.float32, copy=False)
zen_pred = df_pred["zenith"].to_numpy(dtype=np.float32, copy=False)

pred_order = np.argsort(event_id_pred, kind="mergesort")
eid_sorted = event_id_pred[pred_order]
az_sorted = az_pred[pred_order]
zen_sorted = zen_pred[pred_order]

out_path = "submission.csv"
chunksize = 1_000_000

reader = pd.read_csv(
    sample_path,
    usecols=["event_id"],
    chunksize=chunksize,
)

with open(out_path, "w", newline="") as f:
    f.write("event_id,azimuth,zenith\n")
    j = 0
    M = eid_sorted.shape[0]

    for chunk in tqdm(reader, desc="Writing submission.csv (stream merge)"):
        eids = chunk["event_id"].to_numpy(dtype=np.int64, copy=False)
        az_out = np.zeros(eids.shape[0], dtype=np.float32)
        zen_out = np.zeros(eids.shape[0], dtype=np.float32)

        for i, eid in enumerate(eids):
            while j < M and eid_sorted[j] < eid:
                j += 1
            if j < M and eid_sorted[j] == eid:
                a = az_sorted[j]
                z = zen_sorted[j]
                if not np.isfinite(a):
                    a = 0.0
                if not np.isfinite(z):
                    z = 0.0
                az_out[i] = np.float32(a)
                zen_out[i] = np.float32(z)

        for eid, a, z in zip(eids, az_out, zen_out):
            f.write(f"{int(eid)},{float(a)},{float(z)}\n")

df_out_head = pd.read_csv(out_path, nrows=5)
(df_out_head, (sum(1 for _ in open(out_path)) - 1))

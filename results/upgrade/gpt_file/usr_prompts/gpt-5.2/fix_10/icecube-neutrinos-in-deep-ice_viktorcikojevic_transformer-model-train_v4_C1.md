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
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time as _time



## === cell 1
batch_size = 32
block_size = 256  # maximum context length

max_iters = 200
change_batch_every = 256  # change batch of data every change_batch_every steps
device = "cuda" if torch.cuda.is_available() else "cpu"
evaluate_every_step = 100
learning_rate = 1e-4
eval_iters = 16  # number of batches to process for evaluation

embed_dim = 64 - 5
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

torch.backends.cudnn.benchmark = True

PIN_MEMORY = device == "cuda"



## === cell 2
DATA_ROOT = "/kaggle/input/icecube-neutrinos-in-deep-ice"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = (
        "/kaggle/input/icecube-neutrinos-in-deep-ice/icecube-neutrinos-in-deep-ice"
    )

TRAIN_META_PATH = os.path.join(DATA_ROOT, "train_meta.parquet")
TEST_META_PATH = os.path.join(DATA_ROOT, "test_meta.parquet")
SENSOR_GEOM_PATH = os.path.join(DATA_ROOT, "sensor_geometry.csv")

if not os.path.exists(TRAIN_META_PATH):
    TRAIN_META_PATH = "/kaggle/input/train_meta.parquet"
if not os.path.exists(TEST_META_PATH):
    TEST_META_PATH = "/kaggle/input/test_meta.parquet"
if not os.path.exists(SENSOR_GEOM_PATH):
    SENSOR_GEOM_PATH = "/kaggle/input/sensor_geometry.csv"

print("Resolved paths:")
print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_META_PATH exists:", os.path.exists(TRAIN_META_PATH), TRAIN_META_PATH)
print("TEST_META_PATH exists:", os.path.exists(TEST_META_PATH), TEST_META_PATH)
print("SENSOR_GEOM_PATH exists:", os.path.exists(SENSOR_GEOM_PATH), SENSOR_GEOM_PATH)

df_sensor_geometry = pd.read_csv(SENSOR_GEOM_PATH)
sensor_ids = sorted(list(set(df_sensor_geometry["sensor_id"].to_list())))
print(sensor_ids[:10], len(sensor_ids))

max_sid = int(df_sensor_geometry["sensor_id"].max())
_geom_arr = np.zeros((max_sid + 1, 3), dtype=np.float32)
_sid = df_sensor_geometry["sensor_id"].to_numpy(np.int32, copy=False)
_geom_arr[_sid, 0] = df_sensor_geometry["x"].to_numpy(np.float32, copy=False)
_geom_arr[_sid, 1] = df_sensor_geometry["y"].to_numpy(np.float32, copy=False)
_geom_arr[_sid, 2] = df_sensor_geometry["z"].to_numpy(np.float32, copy=False)




## === cell 3
class EventDataLoader:
    """
    Speed fixes (core semantics preserved):
    - Precompute per-loaded-parquet pulse arrays (xyz/time/charge) once, instead of re-deriving xyz via geometry
      lookup on every get_xy call.
    - Vectorize per-batch normalization and cumulative time computation with NumPy using scatter-add style ops,
      reducing Python loop overhead while keeping identical math per event.
    - Reuse pinned CPU tensors to avoid repeated pin_memory() allocations in the hot path.
    """

    def __init__(self, n_test_data=500_000):
        np.random.seed(42)

        meta_cols_train = [
            "batch_id",
            "first_pulse_index",
            "last_pulse_index",
            "azimuth",
            "zenith",
        ]
        meta_cols_eval = [
            "batch_id",
            "first_pulse_index",
            "last_pulse_index",
        ]

        print("Loading train meta parquet (columns only; keep index as event_id) ...")
        df_meta_full = pd.read_parquet(TRAIN_META_PATH, columns=meta_cols_train)
        df_meta_full = df_meta_full.reset_index().rename(columns={"index": "event_id"})

        print("Loading test meta parquet (minimal columns; keep/derive event_id) ...")
        df_meta_eval = pd.read_parquet(TEST_META_PATH, columns=meta_cols_eval)
        df_meta_eval = df_meta_eval.reset_index().rename(columns={"index": "event_id"})

        n_test_data = int(min(n_test_data, max(0, len(df_meta_full) - 1)))
        n_train_data = len(df_meta_full) - n_test_data
        self.change_batch_every = change_batch_every

        self.meta_data = {
            "train": df_meta_full.iloc[:n_train_data].reset_index(drop=True),
            "test": df_meta_full.iloc[n_train_data:].reset_index(drop=True),
            "eval": df_meta_eval.reset_index(drop=True),
        }
        self.batch_id_current = {"train": 0, "test": 0, "eval": 0}

        train_batch_files = glob.glob(
            os.path.join(DATA_ROOT, "train", "batch_*.parquet")
        )
        test_batch_files = glob.glob(os.path.join(DATA_ROOT, "test", "batch_*.parquet"))
        self._existing_batches_disk = {
            "train": sorted(
                [
                    int(os.path.basename(p).split("_")[1].split(".")[0])
                    for p in train_batch_files
                ]
            ),
            "test": sorted(
                [
                    int(os.path.basename(p).split("_")[1].split(".")[0])
                    for p in test_batch_files
                ]
            ),
        }
        self._existing_batches_disk["eval"] = self._existing_batches_disk["test"]

        self._existing_batches = {}
        for split in ["train", "test", "eval"]:
            special_dir = "train" if split in ["train", "test"] else "test"
            on_disk = set(
                self._existing_batches_disk[special_dir]
                if split in ["train", "test"]
                else self._existing_batches_disk["test"]
            )
            in_meta = set(self.meta_data[split]["batch_id"].unique().tolist())
            valid = sorted(list(on_disk.intersection(in_meta)))
            if len(valid) == 0:
                valid = sorted(list(on_disk))
            self._existing_batches[split] = valid

        self._meta_by_batch = {}
        for split in ["train", "test", "eval"]:
            gb = self.meta_data[split].groupby("batch_id", sort=False)
            self._meta_by_batch[split] = {
                int(k): v.reset_index(drop=True) for k, v in gb
            }

        self._meta_cache = {"train": None, "test": None, "eval": None}

        self._pulse_cache = {"train": None, "test": None, "eval": None}

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

        self._buf = {
            "xyz": np.empty((batch_size, block_size, 3), dtype=np.float32),
            "time": np.empty((batch_size, block_size), dtype=np.float32),
            "charge": np.empty((batch_size, block_size), dtype=np.float32),
        }

        if PIN_MEMORY:
            self._cpu_tensors = {
                "xyz": torch.empty(
                    (batch_size, block_size, 3), dtype=torch.float32, pin_memory=True
                ),
                "time": torch.empty(
                    (batch_size, block_size), dtype=torch.float32, pin_memory=True
                ),
                "charge": torch.empty(
                    (batch_size, block_size), dtype=torch.float32, pin_memory=True
                ),
                "y": torch.empty((batch_size, 2), dtype=torch.float32, pin_memory=True),
            }
        else:
            self._cpu_tensors = None

        self.counter = 0

        self._inv_dt_tmax = np.float32(1.0 / (float(dt) * float(t_max)))
        self._inv_500 = np.float32(1.0 / 500.0)

    def shuffle_metadata_batch(self, split):
        valid_batches = self._existing_batches[split]
        if len(valid_batches) == 0:
            raise RuntimeError(
                f"No valid batch parquet files found for split={split} under {DATA_ROOT}."
            )

        random_batch_id = int(np.random.choice(valid_batches))
        self.batch_id_current[split] = random_batch_id

        df = self._meta_by_batch[split].get(random_batch_id)
        if df is None:
            df = self.meta_data[split][
                self.meta_data[split]["batch_id"] == random_batch_id
            ].reset_index(drop=True)

        self._meta_cache[split] = {
            "first": df["first_pulse_index"].to_numpy(np.int64, copy=False),
            "last": df["last_pulse_index"].to_numpy(np.int64, copy=False),
            "event_id": df["event_id"].to_numpy(np.int64, copy=False),
        }
        if split in ["train", "test"]:
            self._meta_cache[split]["y"] = df[["azimuth", "zenith"]].to_numpy(
                np.float32, copy=False
            )

        return df

    def _build_pulse_cache(self, split):
        ev = self.events[split]
        sensor = ev["sensor_id"].to_numpy(np.int32, copy=False)
        sidx = np.clip(sensor, 0, _geom_arr.shape[0] - 1)
        xyz = _geom_arr[sidx].astype(np.float32, copy=False)

        self._pulse_cache[split] = {
            "xyz": xyz,  # (N_pulses,3)
            "time": ev["time"].to_numpy(np.float32, copy=False),
            "charge": ev["charge"].to_numpy(np.float32, copy=False),
        }

    def load_batch_events(self, split):
        special_dir = "train" if split in ["train", "test"] else "test"
        batch_id = int(self.batch_id_current[split])
        path = os.path.join(DATA_ROOT, special_dir, f"batch_{batch_id}.parquet")

        if not os.path.exists(path):
            valid_batches = self._existing_batches[split]
            found = False
            for _ in range(200):
                batch_id = int(np.random.choice(valid_batches))
                path = os.path.join(DATA_ROOT, special_dir, f"batch_{batch_id}.parquet")
                if os.path.exists(path):
                    self.batch_id_current[split] = batch_id
                    df = self._meta_by_batch[split].get(batch_id)
                    if df is None:
                        df = self.meta_data[split][
                            self.meta_data[split]["batch_id"] == batch_id
                        ].reset_index(drop=True)
                    self.meta_data_current[split] = df
                    found = True
                    break
            if not found:
                raise FileNotFoundError(
                    f"Could not find any readable parquet for split={split} under {DATA_ROOT}/{special_dir}."
                )

        df = pd.read_parquet(path, columns=["event_id", "time", "sensor_id", "charge"])
        self._pulse_cache[split] = None
        return df

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

    def _fill_batch_arrays(self, split, first, last):
        if self._pulse_cache[split] is None:
            self._build_pulse_cache(split)
        pc = self._pulse_cache[split]

        xyz_np = self._buf["xyz"]
        time_np = self._buf["time"]
        charge_np = self._buf["charge"]
        xyz_np.fill(0.0)
        time_np.fill(0.0)
        charge_np.fill(0.0)

        xyz_all = pc["xyz"]
        t_all = pc["time"]
        c_all = pc["charge"]

        inv_dt_tmax = self._inv_dt_tmax
        inv_500 = self._inv_500

        lengths = np.zeros((batch_size,), dtype=np.int32)

        for b in range(batch_size):
            fp = int(first[b])
            lp = int(last[b])
            if lp < fp:
                continue
            event_size = lp - fp + 1
            if event_size <= 0:
                continue
            if event_size > block_size:
                event_size = block_size

            sl = slice(fp, fp + event_size)
            xyz_np[b, :event_size, :] = xyz_all[sl]
            time_np[b, :event_size] = t_all[sl] * inv_dt_tmax
            charge_np[b, :event_size] = c_all[sl]
            lengths[b] = event_size

        if np.any(lengths > 0):
            mask = (
                np.arange(block_size, dtype=np.int32)[None, :] < lengths[:, None]
            ).astype(np.float32, copy=False)
            denom = lengths.astype(np.float32)
            denom[denom == 0] = 1.0

            m = mask[:, :, None]
            sums = (xyz_np * m).sum(axis=1, dtype=np.float32)  # (B,3)
            mu = sums / denom[:, None]  # (B,3)
            xyz_np[:] = (
                (xyz_np - mu[:, None, :]) * inv_500 * m
            )  # keep zeros outside prefix

            time_np[:] = np.cumsum(time_np, axis=1, dtype=np.float32) * mask

            np.clip(charge_np, 0.0, 5.0, out=charge_np)
            charge_np[:] = charge_np * mask

        return xyz_np, time_np, charge_np

    def get_xy(self, split):
        assert split in ["train", "test"]

        meta_cache = self._meta_cache[split]
        n = meta_cache["first"].shape[0]
        idx = np.random.randint(0, n, size=batch_size)

        first = meta_cache["first"][idx]
        last = meta_cache["last"][idx]
        y_np = meta_cache["y"][idx]

        xyz_np, time_np, charge_np = self._fill_batch_arrays(split, first, last)

        if PIN_MEMORY:
            self._cpu_tensors["xyz"].copy_(torch.from_numpy(xyz_np), non_blocking=False)
            self._cpu_tensors["time"].copy_(
                torch.from_numpy(time_np), non_blocking=False
            )
            self._cpu_tensors["charge"].copy_(
                torch.from_numpy(charge_np), non_blocking=False
            )
            self._cpu_tensors["y"].copy_(torch.from_numpy(y_np), non_blocking=False)

            xyz = self._cpu_tensors["xyz"].to(device, non_blocking=True)
            time = self._cpu_tensors["time"].to(device, non_blocking=True)
            charge = self._cpu_tensors["charge"].to(device, non_blocking=True)
            y = self._cpu_tensors["y"].to(device, non_blocking=True)
        else:
            xyz = torch.from_numpy(xyz_np).to(device)
            time = torch.from_numpy(time_np).to(device)
            charge = torch.from_numpy(charge_np).to(device)
            y = torch.from_numpy(y_np).to(device)

        self.counter += 1
        if self.counter == self.change_batch_every:
            self.change_event_batch()
            self.counter = 0

        return (xyz, time, charge), y




## === cell 4
data_loader = EventDataLoader()




## === cell 5
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




## === cell 6
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ["train", "test"]:
            losses = torch.zeros(eval_iters, device="cpu")
            for k in range(eval_iters):
                (xyz, time, charge), y = data_loader.get_xy(split)
                _, loss = model(xyz, time, charge, y)
                losses[k] = float(loss.detach().cpu().item())
            out[split] = float(losses.mean().item())
    model.train()
    return out




## === cell 7
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        v = self.value(x)  # (B,T,hs)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, n_embd, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
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
        pos_idx = self._pos_idx
        if pos_idx.device != x.device:
            pos_idx = pos_idx.to(x.device)
        pos_emb = self.position_embedding_table(pos_idx)
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




## === cell 8
model = TransformerModel(
    num_sensors, embed_dim, num_time, num_heads, n_layers, dropout=0.2
).to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")



## === cell 9
(xyz, time, charge), y = data_loader.get_xy("train")
print(xyz.shape, time.shape, charge.shape, y.shape)
pred, loss = model(xyz, time, charge, y)
print(pred.shape, loss.item())



## === cell 10
if device == "cuda":
    torch.cuda.synchronize()
t0 = _time.time()
_ = model(xyz, time, charge)
if device == "cuda":
    torch.cuda.synchronize()
print("Forward pass seconds:", _time.time() - t0)



## === cell 11
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in tqdm(range(max_iters), desc="training"):
    (xyz, time, charge), y = data_loader.get_xy("train")
    pred, loss = model(xyz, time, charge, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % evaluate_every_step == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train_loss {losses['train']:.4f}, test_loss {losses['test']:.4f}"
        )



## === cell 12
torch.save(model.state_dict(), "model.pth")



## === cell 13
df = data_loader.meta_data["eval"].copy()
print(df.head())




## === cell 14
def postprocess_angles_np(pred_np: np.ndarray) -> np.ndarray:
    pred_np = pred_np.astype(np.float64, copy=False)
    two_pi = 2.0 * math.pi
    pred_np[:, 0] = np.mod(pred_np[:, 0], two_pi)
    pred_np[:, 1] = np.clip(pred_np[:, 1], 0.0, math.pi)
    return pred_np.astype(np.float32)




## === cell 15
model.eval()

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"

sample_sub = pd.read_csv(sample_path, usecols=["event_id"])
sample_event_ids = sample_sub["event_id"].to_numpy(np.int64, copy=False)
sample_n = len(sample_sub)
print("sample_submission rows:", sample_n)

order = np.argsort(sample_event_ids, kind="mergesort")
sorted_eids = sample_event_ids[order]


def _lookup_rows(eids: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_eids, eids)
    ok = (pos < sorted_eids.size) & (sorted_eids[pos] == eids)
    rows = np.full(eids.shape[0], -1, dtype=np.int64)
    rows[ok] = order[pos[ok]]
    return rows


az_out = np.zeros(sample_n, dtype=np.float32)
zen_out = np.zeros(sample_n, dtype=np.float32)

test_meta = data_loader.meta_data["eval"]
test_meta = test_meta.sort_values(
    ["batch_id", "event_id"], kind="mergesort"
).reset_index(drop=True)
batch_ids = test_meta["batch_id"].to_numpy(np.int32, copy=False)

batch_change = np.flatnonzero(np.r_[True, batch_ids[1:] != batch_ids[:-1], True])
seg_starts = batch_change[:-1]
seg_ends = batch_change[1:]

inv_dt_tmax = np.float32(1.0 / (float(dt) * float(t_max)))
inv_500 = np.float32(1.0 / 500.0)

xyz_np = np.zeros((batch_size, block_size, 3), dtype=np.float32)
time_np = np.zeros((batch_size, block_size), dtype=np.float32)
charge_np = np.zeros((batch_size, block_size), dtype=np.float32)

if PIN_MEMORY:
    pin_xyz = torch.empty(
        (batch_size, block_size, 3), dtype=torch.float32, pin_memory=True
    )
    pin_time = torch.empty(
        (batch_size, block_size), dtype=torch.float32, pin_memory=True
    )
    pin_charge = torch.empty(
        (batch_size, block_size), dtype=torch.float32, pin_memory=True
    )
else:
    pin_xyz = pin_time = pin_charge = None

for s0, s1 in tqdm(list(zip(seg_starts, seg_ends)), desc="predicting test batches"):
    batch_id = int(batch_ids[s0])
    path = os.path.join(DATA_ROOT, "test", f"batch_{batch_id}.parquet")
    if not os.path.exists(path):
        continue

    meta_b = test_meta.iloc[s0:s1]
    first_idx = meta_b["first_pulse_index"].to_numpy(np.int64, copy=False)
    last_idx = meta_b["last_pulse_index"].to_numpy(np.int64, copy=False)
    event_ids = meta_b["event_id"].to_numpy(np.int64, copy=False)

    rows = _lookup_rows(event_ids)

    events_df = pd.read_parquet(
        path, columns=["event_id", "time", "sensor_id", "charge"]
    )

    sensor = events_df["sensor_id"].to_numpy(np.int32, copy=False)
    sidx = np.clip(sensor, 0, _geom_arr.shape[0] - 1)
    xyz_all = _geom_arr[sidx].astype(np.float32, copy=False)

    t_all = events_df["time"].to_numpy(np.float32, copy=False)
    c_all = events_df["charge"].to_numpy(np.float32, copy=False)

    start = 0
    while start < len(meta_b):
        end = min(start + batch_size, len(meta_b))
        bs = end - start

        xyz_np[:bs].fill(0.0)
        time_np[:bs].fill(0.0)
        charge_np[:bs].fill(0.0)

        lengths = np.zeros((bs,), dtype=np.int32)

        for b, (fp, lp) in enumerate(zip(first_idx[start:end], last_idx[start:end])):
            fp = int(fp)
            lp = int(lp)
            if lp < fp:
                continue
            event_size = lp - fp + 1
            if event_size <= 0:
                continue
            if event_size > block_size:
                event_size = block_size

            sl = slice(fp, fp + event_size)
            xyz_np[b, :event_size, :] = xyz_all[sl]
            time_np[b, :event_size] = t_all[sl] * inv_dt_tmax
            charge_np[b, :event_size] = c_all[sl]
            lengths[b] = event_size

        if np.any(lengths > 0):
            mask = (
                np.arange(block_size, dtype=np.int32)[None, :] < lengths[:, None]
            ).astype(np.float32, copy=False)
            denom = lengths.astype(np.float32)
            denom[denom == 0] = 1.0

            m = mask[:, :, None]
            sums = (xyz_np[:bs] * m).sum(axis=1, dtype=np.float32)
            mu = sums / denom[:, None]
            xyz_np[:bs] = (xyz_np[:bs] - mu[:, None, :]) * inv_500 * m

            time_np[:bs] = np.cumsum(time_np[:bs], axis=1, dtype=np.float32) * mask

            np.clip(charge_np[:bs], 0.0, 5.0, out=charge_np[:bs])
            charge_np[:bs] *= mask

        if PIN_MEMORY:
            pin_xyz[:bs].copy_(torch.from_numpy(xyz_np[:bs]), non_blocking=False)
            pin_time[:bs].copy_(torch.from_numpy(time_np[:bs]), non_blocking=False)
            pin_charge[:bs].copy_(torch.from_numpy(charge_np[:bs]), non_blocking=False)

            xyz_t = pin_xyz[:bs].to(device, non_blocking=True)
            time_t = pin_time[:bs].to(device, non_blocking=True)
            charge_t = pin_charge[:bs].to(device, non_blocking=True)
        else:
            xyz_t = torch.from_numpy(xyz_np[:bs]).to(device)
            time_t = torch.from_numpy(time_np[:bs]).to(device)
            charge_t = torch.from_numpy(charge_np[:bs]).to(device)

        with torch.inference_mode():
            pred_t, _ = model(xyz_t, time_t, charge_t, targets=None)

        pred_np = pred_t.detach().cpu().numpy()
        pred_np = postprocess_angles_np(pred_np)

        r = rows[start:end]
        valid = r >= 0
        if np.any(valid):
            az_out[r[valid]] = pred_np[valid, 0]
            zen_out[r[valid]] = pred_np[valid, 1]

        start = end

missing = int(np.count_nonzero((az_out == 0.0) & (zen_out == 0.0)))
print("Missing event predictions filled with defaults:", missing)

sub_df = pd.DataFrame(
    {"event_id": sample_event_ids, "azimuth": az_out, "zenith": zen_out}
)
assert len(sub_df) == sample_n, "Submission row count mismatch vs sample_submission."

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("submission.csv path:", os.path.abspath(sub_path))

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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "1337")

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import random



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
torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print("Warning: could not enable torch deterministic algorithms:", repr(e))
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

try:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
except Exception:
    pass



## === cell 2
df_sensor_geometry = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv"
)

_sensor_geom = df_sensor_geometry.sort_values("sensor_id").reset_index(drop=True)
_sensor_id_min = int(_sensor_geom["sensor_id"].min())
_sensor_id_max = int(_sensor_geom["sensor_id"].max())
_sensor_id_span = _sensor_id_max - _sensor_id_min + 1

_geom_x = np.full(_sensor_id_span, np.nan, dtype=np.float32)
_geom_y = np.full(_sensor_id_span, np.nan, dtype=np.float32)
_geom_z = np.full(_sensor_id_span, np.nan, dtype=np.float32)

_idx = _sensor_geom["sensor_id"].values.astype(np.int64) - _sensor_id_min
_geom_x[_idx] = _sensor_geom["x"].values.astype(np.float32, copy=False)
_geom_y[_idx] = _sensor_geom["y"].values.astype(np.float32, copy=False)
_geom_z[_idx] = _sensor_geom["z"].values.astype(np.float32, copy=False)

sensor_ids = sorted(list(set(df_sensor_geometry["sensor_id"].to_list())))
sensor_ids[:10], len(sensor_ids)




## === cell 3
class EventDataLoader:
    def __init__(self, n_test_data=500_000):
        np.random.seed(42)

        meta_cols = [
            "batch_id",
            "event_id",
            "first_pulse_index",
            "last_pulse_index",
            "azimuth",
            "zenith",
        ]
        print("Loading input train meta parquet file (columns only) ...")
        df_meta_all = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet",
            columns=meta_cols,
        )
        print("Loading input test meta parquet file (columns only) ...")
        df_meta_eval = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet",
            columns=["batch_id", "event_id", "first_pulse_index", "last_pulse_index"],
        )

        self.test_batch_ids = sorted(df_meta_eval["batch_id"].unique().tolist())
        if len(self.test_batch_ids) == 0:
            raise RuntimeError("No batch_id found in test_meta.parquet")

        self.change_batch_every = change_batch_every

        n_total = len(df_meta_all)
        n_test_data = int(min(max(n_test_data, 0), n_total))
        n_train_data = n_total - n_test_data

        self.meta_data = {
            "train_all": df_meta_all.iloc[:n_train_data].reset_index(drop=True),
            "test": df_meta_all.iloc[n_train_data:].reset_index(drop=True),
            "eval": df_meta_eval.reset_index(drop=True),
        }

        self._batch_rows = {"train": {}, "test": {}, "eval": {}}
        for split, key in [("train", "train_all"), ("test", "test"), ("eval", "eval")]:
            b = self.meta_data[key]["batch_id"].to_numpy(dtype=np.int64, copy=False)
            order = np.argsort(b, kind="mergesort")
            b_sorted = b[order]
            uniq, start = np.unique(b_sorted, return_index=True)
            end = np.r_[start[1:], b_sorted.size]
            for ub, s, e in zip(uniq.tolist(), start.tolist(), end.tolist()):
                self._batch_rows[split][int(ub)] = order[s:e]

        self._batch_minmax = {
            "train": (
                int(self.meta_data["train_all"]["batch_id"].min()),
                int(self.meta_data["train_all"]["batch_id"].max()),
            ),
            "test": (
                int(self.meta_data["test"]["batch_id"].min()),
                int(self.meta_data["test"]["batch_id"].max()),
            ),
        }

        self.batch_id_current = {"train": 0, "test": 0, "eval": 0}

        self.meta_data_current = {"train": None, "test": None, "eval": None}
        self._meta_np = {"train": None, "test": None, "eval": None}
        self._row_pool = {"train": None, "test": None, "eval": None}

        self._events_np = {"train": None, "test": None, "eval": None}
        self._events_cache = {"train": {}, "test": {}, "eval": {}}
        self._events_cache_order = {"train": [], "test": [], "eval": []}
        self._events_cache_max = 2  # small bounded cache to control RAM

        self.meta_data_current["train"] = self.shuffle_metadata_batch("train")
        self.meta_data_current["test"] = self.shuffle_metadata_batch("test")
        self.meta_data_current["eval"] = self.shuffle_metadata_batch("eval")

        self._load_batch_events_np("train")
        self._load_batch_events_np("test")
        self._load_batch_events_np("eval")

        self._refresh_meta_np("train")
        self._refresh_meta_np("test")
        self._refresh_meta_np("eval")

        self.counter = 0

        self._cpu_xyz = torch.empty(
            (batch_size, block_size, 3), dtype=torch.float32
        ).pin_memory()
        self._cpu_time = torch.empty(
            (batch_size, block_size), dtype=torch.float32
        ).pin_memory()
        self._cpu_charge = torch.empty(
            (batch_size, block_size), dtype=torch.float32
        ).pin_memory()

        self._np_xyz = np.zeros((batch_size, block_size, 3), dtype=np.float32)
        self._np_time = np.zeros((batch_size, block_size), dtype=np.float32)
        self._np_charge = np.zeros((batch_size, block_size), dtype=np.float32)

        self._pos = np.arange(block_size, dtype=np.int64)[None, :]

    def shuffle_metadata_batch(self, split):
        if split in ["train", "test"]:
            bmin, bmax = self._batch_minmax[split]
            src_key = "train_all" if split == "train" else "test"

            for _ in range(32):
                bid = int(np.random.randint(bmin, bmax + 1))
                rows = self._batch_rows[split].get(bid, None)
                if rows is not None and rows.size:
                    self.batch_id_current[split] = bid
                    return self.meta_data[src_key].iloc[rows].reset_index(drop=True)

            raise RuntimeError(f"Could not sample a non-empty batch for split={split}")
        else:
            bid = int(np.random.choice(self.test_batch_ids))
            self.batch_id_current[split] = bid
            rows = self._batch_rows["eval"].get(bid, None)
            if rows is None or rows.size == 0:
                raise RuntimeError(f"Empty eval batch_id={bid}")
            return self.meta_data["eval"].iloc[rows].reset_index(drop=True)

    def _load_batch_events_np(self, split):
        special_dir = "train" if split in ["train", "test"] else "test"
        batch_id = int(self.batch_id_current[split])

        cache = self._events_cache[split]
        if batch_id in cache:
            self._events_np[split] = cache[batch_id]
            return

        path = f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet"
        ev = pd.read_parquet(
            path, columns=["time", "sensor_id", "charge"]
        ).reset_index()

        np_pack = {
            "time": ev["time"].to_numpy(dtype=np.float32, copy=False),
            "sensor_id": ev["sensor_id"].to_numpy(dtype=np.int64, copy=False),
            "charge": ev["charge"].to_numpy(dtype=np.float32, copy=False),
        }

        cache[batch_id] = np_pack
        self._events_cache_order[split].append(batch_id)
        if len(self._events_cache_order[split]) > self._events_cache_max:
            old = self._events_cache_order[split].pop(0)
            if old in cache and old != batch_id:
                del cache[old]

        self._events_np[split] = np_pack

    def _refresh_meta_np(self, split):
        m = self.meta_data_current[split]
        if split in ["train", "test"]:
            self._meta_np[split] = {
                "first_pulse_index": m["first_pulse_index"].to_numpy(
                    dtype=np.int64, copy=False
                ),
                "last_pulse_index": m["last_pulse_index"].to_numpy(
                    dtype=np.int64, copy=False
                ),
                "azimuth": m["azimuth"].to_numpy(dtype=np.float32, copy=False),
                "zenith": m["zenith"].to_numpy(dtype=np.float32, copy=False),
            }
        else:
            self._meta_np[split] = {
                "first_pulse_index": m["first_pulse_index"].to_numpy(
                    dtype=np.int64, copy=False
                ),
                "last_pulse_index": m["last_pulse_index"].to_numpy(
                    dtype=np.int64, copy=False
                ),
            }
        n = len(m)
        if n <= 0:
            raise RuntimeError(f"Empty meta_data_current for split={split}")
        self._row_pool[split] = np.arange(n, dtype=np.int64)

    def _build_batch_features(self, split, first_pulse_indices, last_pulse_indices):
        np_ev = self._events_np[split]
        time_arr = np_ev["time"]
        sid_arr = np_ev["sensor_id"]
        charge_arr = np_ev["charge"]

        fp = first_pulse_indices.astype(np.int64, copy=False)
        lp = last_pulse_indices.astype(np.int64, copy=False)

        lens = (lp - fp + 1).astype(np.int64, copy=False)
        n_take = np.minimum(lens, block_size).astype(np.int64, copy=False)

        idx = fp[:, None] + self._pos  # (B,T)
        idx = np.minimum(idx, lp[:, None])  # clamp within event
        valid = self._pos < n_take[:, None]  # (B,T)

        t_raw = time_arr[idx]  # (B,T)
        sid = sid_arr[idx]  # (B,T)
        c_raw = charge_arr[idx]  # (B,T)

        sid0 = (sid - _sensor_id_min).astype(np.int64, copy=False)
        x = _geom_x[sid0]
        y = _geom_y[sid0]
        z = _geom_z[sid0]

        x = np.where(valid, x, 0.0).astype(np.float32, copy=False)
        y = np.where(valid, y, 0.0).astype(np.float32, copy=False)
        z = np.where(valid, z, 0.0).astype(np.float32, copy=False)

        xyz = self._np_xyz
        xyz.fill(0.0)
        xyz[..., 0] = x
        xyz[..., 1] = y
        xyz[..., 2] = z

        mean = xyz.sum(axis=1, dtype=np.float32) / float(block_size)
        xyz -= mean[:, None, :]
        xyz *= 1.0 / 500.0
        xyz *= valid[..., None].astype(np.float32, copy=False)

        t = (t_raw / float(dt) / float(t_max)).astype(np.float32, copy=False)
        t = np.where(valid, t, 0.0).astype(np.float32, copy=False)
        time_feat = self._np_time
        time_feat[:] = np.cumsum(t, axis=1, dtype=np.float32)
        time_feat *= valid.astype(np.float32, copy=False)

        charge_feat = self._np_charge
        charge_feat[:] = np.clip(c_raw, 0.0, 5.0).astype(np.float32, copy=False)
        charge_feat *= valid.astype(np.float32, copy=False)

        return xyz, time_feat, charge_feat

    def change_event_batch(self):
        self.meta_data_current = {
            "train": self.shuffle_metadata_batch("train"),
            "test": self.shuffle_metadata_batch("test"),
            "eval": self.shuffle_metadata_batch("eval"),
        }

        self._load_batch_events_np("train")
        self._load_batch_events_np("test")
        self._load_batch_events_np("eval")

        self._refresh_meta_np("train")
        self._refresh_meta_np("test")
        self._refresh_meta_np("eval")

        self.counter = 0

    def get_xy(self, split, advance=True):
        assert split in ["train", "test"]

        pool = self._row_pool[split]
        if pool.shape[0] < batch_size:
            idx = np.random.randint(0, pool.shape[0], size=batch_size, dtype=np.int64)
        else:
            idx = np.random.choice(pool, size=batch_size, replace=False)

        mnp = self._meta_np[split]
        first_pulse_indices = mnp["first_pulse_index"][idx]
        last_pulse_indices = mnp["last_pulse_index"][idx]

        xyz_np, time_np, charge_np = self._build_batch_features(
            split, first_pulse_indices, last_pulse_indices
        )

        self._cpu_xyz.copy_(torch.from_numpy(xyz_np), non_blocking=True)
        self._cpu_time.copy_(torch.from_numpy(time_np), non_blocking=True)
        self._cpu_charge.copy_(torch.from_numpy(charge_np), non_blocking=True)

        xyz_batch = self._cpu_xyz.to(device, non_blocking=True)
        time_batch = self._cpu_time.to(device, non_blocking=True)
        charge_batch = self._cpu_charge.to(device, non_blocking=True)

        y_cpu = np.empty((batch_size, 2), dtype=np.float32)
        y_cpu[:, 0] = mnp["azimuth"][idx]
        y_cpu[:, 1] = mnp["zenith"][idx]
        y = torch.from_numpy(y_cpu).to(device, non_blocking=True)

        if advance:
            self.counter += 1
            if self.counter == self.change_batch_every:
                self.change_event_batch()
                self.counter = 0

        return (xyz_batch, time_batch, charge_batch), y




## === cell 4
data_loader = EventDataLoader()



## === cell 5
(xyz, time, charge), y = data_loader.get_xy(split="train")
xyz[0], time[0], charge[0], y[0]




## === cell 6
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




## === cell 7
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters, device="cpu")
        for k in range(eval_iters):
            (xyz, time, charge), y = data_loader.get_xy(split, advance=False)
            logits, loss = model(xyz, time, charge, y)
            losses[k] = float(loss.detach().cpu().item())
        out[split] = losses.mean().item()
    model.train()
    return out




## === cell 8
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
        self.num_heads = num_heads
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size * num_heads, bias=False)
        self.query = nn.Linear(n_embd, head_size * num_heads, bias=False)
        self.value = nn.Linear(n_embd, head_size * num_heads, bias=False)
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = (
            out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size)
        )
        out = self.proj(out)
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
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, n_embd))
        self.blocks = nn.Sequential(
            *[Block(num_heads, n_embd + 5, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd + 5)
        self.classifier = nn.Linear(n_embd + 5, 2)
        self.apply(self._init_weights)

    def forward(self, xyz, time, charge, targets=None):
        x = torch.cat([xyz, time.unsqueeze(-1), charge.unsqueeze(-1)], dim=-1)
        pos_embedding_expanded = self.pos_embedding.expand(x.size(0), x.size(1), -1)
        x = torch.cat([x, pos_embedding_expanded], dim=-1)
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




## === cell 9
model = TransformerModel(
    num_sensors, embed_dim, num_time, num_heads, n_layers, dropout=0.2
).to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")



## === cell 10
(xyz, time, charge), y = data_loader.get_xy("train")
print(xyz.shape, time.shape, charge.shape)
pred, _ = model(xyz, time, charge, y)
pred.shape



## === cell 11
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

eager_model_for_infer = model

try:
    compiled = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        (xyz_t, time_t, charge_t), y_t = data_loader.get_xy("train")
        _ = compiled(xyz_t, time_t, charge_t, y_t)
    model = compiled
    print("torch.compile enabled")
except Exception as e:
    print("torch.compile disabled (fallback to eager):", repr(e))

model.train()
for iter in tqdm(range(max_iters), desc="training"):
    (xyz, time, charge), y = data_loader.get_xy("train")
    logits, loss = model(xyz, time, charge, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % evaluate_every_step == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}"
        )



## === cell 12
try:
    torch.save(model.state_dict(), "model.pth")
except Exception:
    pass



## === cell 13
test_meta = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet",
    columns=["batch_id", "event_id", "first_pulse_index", "last_pulse_index"],
)

sample_sub_path = "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv"

test_batch_ids = sorted(test_meta["batch_id"].unique().tolist())
len(test_batch_ids), test_batch_ids[:5], test_batch_ids[-5:]



## === cell 14
model.eval()


def _normalize_angles(pred_az: np.ndarray, pred_zen: np.ndarray):
    az = np.mod(pred_az, 2.0 * np.pi).astype(np.float32, copy=False)
    zen = np.clip(pred_zen, 0.0, np.pi).astype(np.float32, copy=False)
    return az, zen


_infer_pos = np.arange(block_size, dtype=np.int64)[None, :]

_infer_xyz_buf = np.empty((batch_size, block_size, 3), dtype=np.float32)
_infer_time_buf = np.empty((batch_size, block_size), dtype=np.float32)
_infer_charge_buf = np.empty((batch_size, block_size), dtype=np.float32)


def _build_features_vectorized_into(
    time_arr, sid_arr, charge_arr, fp, lp, xyz_out, time_out, charge_out
):
    T = block_size
    lens = (lp - fp + 1).astype(np.int64, copy=False)
    n_take = np.minimum(lens, T).astype(np.int64, copy=False)

    pos = _infer_pos
    idx = fp[:, None] + pos
    idx = np.minimum(idx, lp[:, None])

    valid = pos < n_take[:, None]

    t_raw = time_arr[idx]
    sid = sid_arr[idx]
    c_raw = charge_arr[idx]

    sid0 = (sid - _sensor_id_min).astype(np.int64, copy=False)
    x = _geom_x[sid0]
    y = _geom_y[sid0]
    z = _geom_z[sid0]

    xyz_out[..., 0] = np.where(valid, x, 0.0).astype(np.float32, copy=False)
    xyz_out[..., 1] = np.where(valid, y, 0.0).astype(np.float32, copy=False)
    xyz_out[..., 2] = np.where(valid, z, 0.0).astype(np.float32, copy=False)

    mean = xyz_out.sum(axis=1, dtype=np.float32) / float(T)
    xyz_out -= mean[:, None, :]
    xyz_out *= 1.0 / 500.0
    xyz_out *= valid[..., None].astype(np.float32, copy=False)

    t = (t_raw / float(dt) / float(t_max)).astype(np.float32, copy=False)
    t *= valid.astype(np.float32, copy=False)
    np.cumsum(t, axis=1, dtype=np.float32, out=time_out)
    time_out *= valid.astype(np.float32, copy=False)

    np.clip(c_raw, 0.0, 5.0, out=charge_out)
    charge_out *= valid.astype(np.float32, copy=False)


_test_meta_by_batch = {
    int(b): df.reset_index(drop=True)
    for b, df in test_meta.groupby("batch_id", sort=False)[
        ["event_id", "first_pulse_index", "last_pulse_index"]
    ]
}

_infer_cpu_xyz = torch.empty(
    (batch_size, block_size, 3), dtype=torch.float32
).pin_memory()
_infer_cpu_time = torch.empty(
    (batch_size, block_size), dtype=torch.float32
).pin_memory()
_infer_cpu_charge = torch.empty(
    (batch_size, block_size), dtype=torch.float32
).pin_memory()

_test_pulse_cache = {}
_test_pulse_cache_order = []
_test_pulse_cache_max = 2


def _load_test_pulses_np(batch_id: int):
    bid = int(batch_id)
    if bid in _test_pulse_cache:
        return _test_pulse_cache[bid]

    pulses = pd.read_parquet(
        f"/kaggle/input/icecube-neutrinos-in-deep-ice/test/batch_{bid}.parquet",
        columns=["time", "sensor_id", "charge"],
    ).reset_index()

    pack = (
        pulses["time"].to_numpy(dtype=np.float32, copy=False),
        pulses["sensor_id"].to_numpy(dtype=np.int64, copy=False),
        pulses["charge"].to_numpy(dtype=np.float32, copy=False),
    )
    _test_pulse_cache[bid] = pack
    _test_pulse_cache_order.append(bid)
    if len(_test_pulse_cache_order) > _test_pulse_cache_max:
        old = _test_pulse_cache_order.pop(0)
        if old in _test_pulse_cache and old != bid:
            del _test_pulse_cache[old]
    return pack


def predict_batch(batch_id: int, infer_model: nn.Module):
    time_arr, sid_arr, charge_arr = _load_test_pulses_np(int(batch_id))

    meta_b = _test_meta_by_batch[int(batch_id)]
    n_events = len(meta_b)
    preds = np.zeros((n_events, 2), dtype=np.float32)

    fp_all = meta_b["first_pulse_index"].to_numpy(dtype=np.int64, copy=False)
    lp_all = meta_b["last_pulse_index"].to_numpy(dtype=np.int64, copy=False)

    with torch.no_grad():
        for start in range(0, n_events, batch_size):
            end = min(start + batch_size, n_events)
            cur_bs = end - start

            fp = fp_all[start:end]
            lp = lp_all[start:end]

            _build_features_vectorized_into(
                time_arr,
                sid_arr,
                charge_arr,
                fp,
                lp,
                _infer_xyz_buf[:cur_bs],
                _infer_time_buf[:cur_bs],
                _infer_charge_buf[:cur_bs],
            )

            _infer_cpu_xyz[:cur_bs].copy_(
                torch.from_numpy(_infer_xyz_buf[:cur_bs]), non_blocking=True
            )
            _infer_cpu_time[:cur_bs].copy_(
                torch.from_numpy(_infer_time_buf[:cur_bs]), non_blocking=True
            )
            _infer_cpu_charge[:cur_bs].copy_(
                torch.from_numpy(_infer_charge_buf[:cur_bs]), non_blocking=True
            )

            xyz_t = _infer_cpu_xyz[:cur_bs].to(device, non_blocking=True)
            time_t = _infer_cpu_time[:cur_bs].to(device, non_blocking=True)
            charge_t = _infer_cpu_charge[:cur_bs].to(device, non_blocking=True)

            out, _ = infer_model(xyz_t, time_t, charge_t, targets=None)
            preds[start:end] = out.detach().cpu().numpy().astype(np.float32, copy=False)

    az, zen = _normalize_angles(preds[:, 0], preds[:, 1])
    return meta_b["event_id"].to_numpy(dtype=np.int64, copy=False), az, zen


infer_model = model
try:
    _ = next(infer_model.parameters())
except Exception:
    infer_model = eager_model_for_infer
infer_model.eval()

all_event_ids = []
all_az = []
all_zen = []

for b in tqdm(test_batch_ids, desc="predicting test batches"):
    eids_b, az_b, zen_b = predict_batch(int(b), infer_model=infer_model)
    all_event_ids.append(eids_b)
    all_az.append(az_b)
    all_zen.append(zen_b)

pred_event_id = np.concatenate(all_event_ids)
pred_az = np.concatenate(all_az).astype(np.float32, copy=False)
pred_zen = np.concatenate(all_zen).astype(np.float32, copy=False)



## === cell 15
order = np.argsort(pred_event_id, kind="mergesort")
pred_event_id_sorted = pred_event_id[order]
pred_az_sorted = pred_az[order]
pred_zen_sorted = pred_zen[order]

final_path = "submission.csv"
with open(final_path, "w", newline="") as f_out:
    f_out.write("event_id,azimuth,zenith\n")
    for eid, az, zen in zip(pred_event_id_sorted, pred_az_sorted, pred_zen_sorted):
        az_n, zen_n = _normalize_angles(
            np.array([az], dtype=np.float32), np.array([zen], dtype=np.float32)
        )
        f_out.write(f"{int(eid)},{float(az_n[0])},{float(zen_n[0])}\n")

final_path

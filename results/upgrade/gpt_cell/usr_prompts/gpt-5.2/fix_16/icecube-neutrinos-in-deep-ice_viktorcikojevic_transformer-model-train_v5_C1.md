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
                out_xyz,
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
data_loader = EventDataLoader()

_df0 = data_loader.meta_data["eval"].iloc[0]
xyz0, time0, charge0 = data_loader.get_single_event(
    int(_df0["first_pulse_index"]), int(_df0["last_pulse_index"]), split="eval"
)
xyz0[:2], time0[:5], charge0[:5]




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
    raise RuntimeError(
        "Training/eval loss estimation disabled in this inference-only optimized script."
    )




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
        self.register_buffer("_pos_idx", torch.arange(block_size), persistent=False)
        self.apply(self._init_weights)

    def forward(self, xyz, time, charge, targets=None):
        x = torch.cat([xyz, time.unsqueeze(-1), charge.unsqueeze(-1)], dim=-1)

        pos = self._pos_idx.to(x.device)
        pos_emb = self.position_embedding_table(pos)  # (T,C)
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
)
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")



## === cell 11
model.eval()
xyz = xyz0.unsqueeze(0).to(device)
time = time0.unsqueeze(0).to(device)
charge = charge0.unsqueeze(0).to(device)
with torch.inference_mode():
    pred0, _ = model(xyz, time, charge)
pred0.shape



## === cell 12
_ = model(xyz, time, charge)



## === cell 13
torch.save(model.state_dict(), "model.pth")



## === cell 14
df = data_loader.meta_data["eval"][
    ["event_id", "first_pulse_index", "last_pulse_index", "batch_id"]
]
df



## === cell 15
if njit is not None:
    _dummy_bs = 1
    _dummy_first = np.zeros((_dummy_bs,), dtype=np.int64)
    _dummy_last = np.zeros((_dummy_bs,), dtype=np.int64)
    _dummy_sensor = np.zeros((1,), dtype=np.int32)
    _dummy_time = np.zeros((1,), dtype=np.float32)
    _dummy_charge = np.zeros((1,), dtype=np.float32)
    _dummy_out_xyz = np.zeros((_dummy_bs, block_size, 3), dtype=np.float32)
    _dummy_out_time = np.zeros((_dummy_bs, block_size), dtype=np.float32)
    _dummy_out_charge = np.zeros((_dummy_bs, block_size), dtype=np.float32)
    _build_features_batch_numba(
        _dummy_first,
        _dummy_last,
        _dummy_sensor,
        _dummy_time,
        _dummy_charge,
        SENSOR_XYZ_LUT,
        float(_INV_500),
        float(_TIME_SCALE),
        _dummy_out_xyz,
        _dummy_out_time,
        _dummy_out_charge,
    )



## === cell 16
model.eval()

infer_batch_size = 256  # keep same core inference batch size

event_id_all = df["event_id"].to_numpy(np.int64, copy=False)
first_all = df["first_pulse_index"].to_numpy(np.int64, copy=False)
last_all = df["last_pulse_index"].to_numpy(np.int64, copy=False)
batch_all = df["batch_id"].to_numpy(np.int32, copy=False)

order = np.argsort(batch_all, kind="mergesort")  # stable, deterministic
event_id_s = event_id_all[order]
first_s = first_all[order]
last_s = last_all[order]
batch_s = batch_all[order]

n_total = len(df)
pred_az = np.empty((n_total,), dtype=np.float32)
pred_ze = np.empty((n_total,), dtype=np.float32)

xyz_np = np.empty((infer_batch_size, block_size, 3), dtype=np.float32)
time_np = np.empty((infer_batch_size, block_size), dtype=np.float32)
charge_np = np.empty((infer_batch_size, block_size), dtype=np.float32)
xyz_np_t = torch.from_numpy(xyz_np)
time_np_t = torch.from_numpy(time_np)
charge_np_t = torch.from_numpy(charge_np)

xyz_pin = torch.empty((infer_batch_size, block_size, 3), dtype=torch.float32)
time_pin = torch.empty((infer_batch_size, block_size), dtype=torch.float32)
charge_pin = torch.empty((infer_batch_size, block_size), dtype=torch.float32)
if device == "cuda":
    xyz_pin = xyz_pin.pin_memory()
    time_pin = time_pin.pin_memory()
    charge_pin = charge_pin.pin_memory()
    xyz_gpu = torch.empty(
        (infer_batch_size, block_size, 3), device="cuda", dtype=torch.float32
    )
    time_gpu = torch.empty(
        (infer_batch_size, block_size), device="cuda", dtype=torch.float32
    )
    charge_gpu = torch.empty(
        (infer_batch_size, block_size), device="cuda", dtype=torch.float32
    )

pred_az_s = np.empty((n_total,), dtype=np.float32)
pred_ze_s = np.empty((n_total,), dtype=np.float32)

unique_batches = np.unique(batch_s)

with torch.inference_mode():
    pos0 = 0
    for batch_id in tqdm(unique_batches, desc="Batches"):
        batch_id = int(batch_id)
        if data_loader.batch_id_current["eval"] != batch_id:
            data_loader.batch_id_current["eval"] = batch_id
            data_loader.meta_data_current = {
                "eval": data_loader._select_meta_batch("eval")
            }
            data_loader.load_batch_events("eval")

        pos1 = pos0
        while pos1 < n_total and int(batch_s[pos1]) == batch_id:
            pos1 += 1
        if pos1 == pos0:
            continue

        for i in range(pos0, pos1, infer_batch_size):
            j = min(i + infer_batch_size, pos1)
            bs = j - i

            data_loader.fill_batch_tensors(
                first_s[i:j],
                last_s[i:j],
                xyz_np[:bs],
                time_np[:bs],
                charge_np[:bs],
                split="eval",
            )

            if device == "cuda":
                xyz_pin[:bs].copy_(xyz_np_t[:bs], non_blocking=True)
                time_pin[:bs].copy_(time_np_t[:bs], non_blocking=True)
                charge_pin[:bs].copy_(charge_np_t[:bs], non_blocking=True)

                xyz_gpu[:bs].copy_(xyz_pin[:bs], non_blocking=True)
                time_gpu[:bs].copy_(time_pin[:bs], non_blocking=True)
                charge_gpu[:bs].copy_(charge_pin[:bs], non_blocking=True)

                pred, _ = model(xyz_gpu[:bs], time_gpu[:bs], charge_gpu[:bs])
            else:
                pred, _ = model(xyz_np_t[:bs], time_np_t[:bs], charge_np_t[:bs])

            pred_np = pred.detach().cpu().numpy().astype(np.float32, copy=False)
            pred_az_s[i:j] = pred_np[:, 0]
            pred_ze_s[i:j] = pred_np[:, 1]

        pos0 = pos1

inv_order = np.empty_like(order)
inv_order[order] = np.arange(n_total, dtype=order.dtype)
pred_az[:] = pred_az_s[inv_order]
pred_ze[:] = pred_ze_s[inv_order]

sub = pd.DataFrame({"event_id": event_id_all, "azimuth": pred_az, "zenith": pred_ze})
sub = sub.sort_values(["event_id"])
sub.to_csv("submission.csv", index=False)
sub

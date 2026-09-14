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
class EventDataLoader:
    """
    Speed-critical refactor (correctness-preserving):
    - Avoid per-event DataFrame slicing (.iloc) and per-event torch.from_numpy allocations.
    - Load the current parquet batch once, keep pulse columns as contiguous NumPy arrays.
    - Build events directly into preallocated output tensors/arrays in-place.
    Core feature logic is identical: same truncation to block_size, same XYZ centering+scale,
    same cumulative-time transform, same charge clipping.
    """

    def __init__(self, n_test_data=500_000):
        np.random.seed(42)
        print("Loading large input test meta parquet file ...")
        df_meta_eval = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
        )

        self.change_batch_every = change_batch_every
        self.meta_data = {
            "eval": df_meta_eval.reset_index(drop=True),
        }

        _batch_ids = self.meta_data["eval"]["batch_id"].to_numpy(np.int32, copy=False)
        self._unique_batch_ids, self._batch_start = np.unique(
            _batch_ids, return_index=True
        )
        self._unique_batch_ids = self._unique_batch_ids.astype(np.int32, copy=False)

        self.batch_id_current = {"eval": int(self._unique_batch_ids[0])}
        self.meta_data_current = {"eval": self._select_meta_batch("eval")}
        self.events = {"eval": None}
        self._events_np = {"eval": None}  # dict of numpy arrays for columns
        self._events_event_ids = {"eval": None}  # numpy array of event_id per pulse row
        self._events_group = {
            "eval": None
        }  # mapping event_id -> (start,end) in this batch file
        self.load_batch_events("eval")
        self.counter = 0  # kept for API compatibility

    def _select_meta_batch(self, split):
        assert split == "eval"
        batch_id = self.batch_id_current[split]
        dfm = self.meta_data[split]
        mask = dfm["batch_id"].to_numpy(np.int32, copy=False) == batch_id
        return dfm.loc[mask].reset_index(drop=True)

    def shuffle_metadata_batch(self, split):
        assert split == "eval"
        return self._select_meta_batch(split)

    def load_batch_events(self, split):
        """Load a batch parquet file and cache relevant columns as NumPy arrays."""
        assert split == "eval"
        special_dir = "test"
        batch_id = self.batch_id_current[split]

        df = pd.read_parquet(
            f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet",
            columns=["time", "sensor_id", "charge"],
        )
        event_id = df.index.to_numpy(np.int64, copy=False)

        self.events[split] = df  # keep for any compatibility/debug usage

        self._events_event_ids[split] = event_id
        self._events_np[split] = {
            "sensor_id": np.ascontiguousarray(
                df["sensor_id"].to_numpy(np.int32, copy=False)
            ),
            "time": np.ascontiguousarray(df["time"].to_numpy(np.float32, copy=False)),
            "charge": np.ascontiguousarray(
                df["charge"].to_numpy(np.float32, copy=False)
            ),
        }

        eid = self._events_event_ids[split]
        if eid.size == 0:
            self._events_group[split] = (
                np.empty((0,), np.int64),
                np.empty((0,), np.int64),
                np.empty((0,), np.int64),
            )
        else:
            change = np.nonzero(eid[1:] != eid[:-1])[0] + 1
            starts = np.concatenate(([0], change)).astype(np.int64, copy=False)
            ends = np.concatenate((change - 1, [eid.size - 1])).astype(
                np.int64, copy=False
            )
            uniq = eid[starts]
            self._events_group[split] = (uniq, starts, ends)

        return df

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        assert split in [
            "train",
            "test",
            "eval",
        ], "split must be either 'train', 'test' or 'eval'"

        start = int(first_pulse_index)
        end = int(last_pulse_index) + 1

        sensor = self._events_np[split]["sensor_id"][start:end]
        time_raw = self._events_np[split]["time"][start:end]
        charge_raw = self._events_np[split]["charge"][start:end]

        event_size = int(end - start)
        n = block_size if event_size >= block_size else event_size

        xyz = np.zeros((block_size, 3), dtype=np.float32)
        time = np.zeros((block_size,), dtype=np.float32)
        charge = np.zeros((block_size,), dtype=np.float32)

        if n > 0:
            xyz[:n] = SENSOR_XYZ_LUT[sensor[:n]]
            m = xyz[:n].mean(axis=0, dtype=np.float32)
            xyz[:n] = (xyz[:n] - m) * _INV_500

            time[:n] = np.cumsum(time_raw[:n] * _TIME_SCALE, dtype=np.float32)
            np.clip(charge_raw[:n], 0.0, 5.0, out=charge[:n])

        return (
            torch.from_numpy(xyz),
            torch.from_numpy(time),
            torch.from_numpy(charge),
        )

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
        Speed-critical helper (correctness-preserving):
        Fill preallocated CPU tensors in-place for a batch of events.
        This avoids allocating 3 new torch tensors per event and reduces Python overhead.
        """
        assert split == "eval"
        bs = int(first_idx_batch.shape[0])
        sensor_all = self._events_np[split]["sensor_id"]
        time_all = self._events_np[split]["time"]
        charge_all = self._events_np[split]["charge"]

        for b in range(bs):
            start = int(first_idx_batch[b])
            end = int(last_idx_batch[b]) + 1
            event_size = end - start
            n = block_size if event_size >= block_size else event_size

            out_xyz[b].zero_()
            out_time[b].zero_()
            out_charge[b].zero_()

            if n <= 0:
                continue

            sensor = sensor_all[start : start + n]
            traw = time_all[start : start + n]
            craw = charge_all[start : start + n]

            xyz_np = SENSOR_XYZ_LUT[sensor]  # (n,3) float32 view/copy
            m = xyz_np.mean(axis=0, dtype=np.float32)
            xyz_np = (xyz_np - m) * _INV_500

            t_np = np.cumsum(traw * _TIME_SCALE, dtype=np.float32)

            c_np = np.clip(craw, 0.0, 5.0).astype(np.float32, copy=False)

            out_xyz[b, :n].copy_(torch.from_numpy(xyz_np), non_blocking=False)
            out_time[b, :n].copy_(torch.from_numpy(t_np), non_blocking=False)
            out_charge[b, :n].copy_(torch.from_numpy(c_np), non_blocking=False)

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




## === cell 5
data_loader = EventDataLoader()



## === cell 6
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
df = data_loader.meta_data["eval"].copy()
df



## === cell 15
model.eval()

infer_batch_size = 256
n = len(df)
pred_arr = np.empty((n, 2), dtype=np.float32)

first_idx_all = df["first_pulse_index"].to_numpy(np.int64, copy=False)
last_idx_all = df["last_pulse_index"].to_numpy(np.int64, copy=False)

xyz_cpu = torch.empty((infer_batch_size, block_size, 3), dtype=torch.float32)
time_cpu = torch.empty((infer_batch_size, block_size), dtype=torch.float32)
charge_cpu = torch.empty((infer_batch_size, block_size), dtype=torch.float32)

if device == "cuda":
    xyz_cpu = xyz_cpu.pin_memory()
    time_cpu = time_cpu.pin_memory()
    charge_cpu = charge_cpu.pin_memory()

with torch.inference_mode():
    for i in range(0, n, infer_batch_size):
        j = min(i + infer_batch_size, n)
        bs = j - i

        data_loader.fill_batch_tensors(
            first_idx_all[i:j],
            last_idx_all[i:j],
            xyz_cpu,
            time_cpu,
            charge_cpu,
            split="eval",
        )

        xyz_t = xyz_cpu[:bs]
        time_t = time_cpu[:bs]
        charge_t = charge_cpu[:bs]

        if device == "cuda":
            xyz_t = xyz_t.to(device, non_blocking=True)
            time_t = time_t.to(device, non_blocking=True)
            charge_t = charge_t.to(device, non_blocking=True)
        else:
            xyz_t = xyz_t.to(device)
            time_t = time_t.to(device)
            charge_t = charge_t.to(device)

        pred, _ = model(xyz_t, time_t, charge_t)
        pred_arr[i:j] = pred.detach().cpu().numpy().astype(np.float32, copy=False)

df["azimuth"] = pred_arr[:, 0]
df["zenith"] = pred_arr[:, 1]
df = df[["event_id", "azimuth", "zenith"]]

df = df.sort_values(["event_id"])
df.to_csv("submission.csv", index=False)
df


## --- ERROR in cell 15, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/810661639.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     25[0m         [0mbs[0m [0;34m=[0m [0mj[0m [0;34m-[0m [0mi[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m [0;34m[0m[0m
[0;32m---> 27[0;31m         data_loader.fill_batch_tensors(
[0m[1;32m     28[0m             [0mfirst_idx_all[0m[0;34m[[0m[0mi[0m[0;34m:[0m[0mj[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m             [0mlast_idx_all[0m[0;34m[[0m[0mi[0m[0;34m:[0m[0mj[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2231096976.py[0m in [0;36mfill_batch_tensors[0;34m(self, first_idx_batch, last_idx_batch, out_xyz, out_time, out_charge, split)[0m
[1;32m    195[0m [0;34m[0m[0m
[1;32m    196[0m             [0;31m# write into tensors[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 197[0;31m             [0mout_xyz[0m[0;34m[[0m[0mb[0m[0;34m,[0m [0;34m:[0m[0mn[0m[0;34m][0m[0;34m.[0m[0mcopy_[0m[0;34m([0m[0mtorch[0m[0;34m.[0m[0mfrom_numpy[0m[0;34m([0m[0mxyz_np[0m[0;34m)[0m[0;34m,[0m [0mnon_blocking[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    198[0m             [0mout_time[0m[0;34m[[0m[0mb[0m[0;34m,[0m [0;34m:[0m[0mn[0m[0;34m][0m[0;34m.[0m[0mcopy_[0m[0;34m([0m[0mtorch[0m[0;34m.[0m[0mfrom_numpy[0m[0;34m([0m[0mt_np[0m[0;34m)[0m[0;34m,[0m [0mnon_blocking[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    199[0m             [0mout_charge[0m[0;34m[[0m[0mb[0m[0;34m,[0m [0;34m:[0m[0mn[0m[0;34m][0m[0;34m.[0m[0mcopy_[0m[0;34m([0m[0mtorch[0m[0;34m.[0m[0mfrom_numpy[0m[0;34m([0m[0mc_np[0m[0;34m)[0m[0;34m,[0m [0mnon_blocking[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: The size of tensor a (256) must match the size of tensor b (0) at non-singleton dimension 0

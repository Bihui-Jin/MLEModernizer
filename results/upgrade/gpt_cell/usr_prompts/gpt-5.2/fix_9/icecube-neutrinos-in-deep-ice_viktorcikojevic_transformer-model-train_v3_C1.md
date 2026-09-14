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
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 1
batch_size = 32
block_size = 256  # maximum context length
max_iters = 5000  # number of iterations (not used in this inference-only script)
change_batch_every = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
evaluate_every_step = 500
learning_rate = 1e-4
eval_iters = 32

embed_dim = 64 - 5
n_layers = 6
num_heads = 6
dropout = 0.2
max_time = 77785
dt = 10
t_max = 4e6

num_time = int(max_time / dt + 1)
num_sensors = 5160




## === cell 2
df_sensor_geometry = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv"
)
sensor_ids = sorted(list(set(df_sensor_geometry["sensor_id"].to_list())))
sensor_ids[:10], len(sensor_ids)




## === cell 3
_sensor_xyz = np.empty((df_sensor_geometry["sensor_id"].max() + 1, 3), dtype=np.float32)
_sensor_xyz[:] = 0.0
_sensor_xyz[df_sensor_geometry["sensor_id"].to_numpy()] = df_sensor_geometry[
    ["x", "y", "z"]
].to_numpy(np.float32)




## === cell 4
class EventDataLoader:
    """
    Inference-only loader.
    Core feature logic preserved:
      - xyz lookup by sensor_id, then per-event mean-centering and /500 scaling
      - time: (time/dt)/t_max then cumulative sum
      - charge: clipped to [0,5]
      - truncation/padding to block_size with zeros
    Speed changes:
      - Use per-batch NumPy arrays instead of pandas slicing for each event.
      - Cache current batch arrays and reuse.
    """

    def __init__(self):
        np.random.seed(42)
        print("Loading large input test meta parquet file ...")
        df_meta_eval = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
        )
        self.meta_data = {"eval": df_meta_eval.reset_index(drop=True)}
        if (
            "batch_id" in self.meta_data["eval"].columns
            and len(self.meta_data["eval"]) > 0
        ):
            self.batch_id_current = {
                "eval": int(self.meta_data["eval"]["batch_id"].min())
            }
        else:
            self.batch_id_current = {"eval": 0}

        self.events = {"eval": None}
        self._cache = {"eval": {}}
        self._set_batch_arrays("eval", self.batch_id_current["eval"])

    def _set_batch_arrays(self, split, batch_id: int):
        special_dir = "test"
        df_events = pd.read_parquet(
            f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet",
            columns=["event_id", "time", "sensor_id", "charge"],
        ).reset_index(drop=True)

        self._cache[split]["sensor_id"] = df_events["sensor_id"].to_numpy(
            np.int64, copy=False
        )
        self._cache[split]["time"] = df_events["time"].to_numpy(np.float32, copy=False)
        self._cache[split]["charge"] = df_events["charge"].to_numpy(
            np.float32, copy=False
        )

    def load_batch_events(self, split):
        batch_id = self.batch_id_current[split]
        self._set_batch_arrays(split, batch_id)
        return None  # kept for compatibility; not used downstream

    def fill_single_event_np(
        self,
        first_pulse_index: int,
        last_pulse_index: int,
        xyz_out: np.ndarray,  # (block_size,3) float32
        time_out: np.ndarray,  # (block_size,) float32
        charge_out: np.ndarray,  # (block_size,) float32
        split: str = "eval",
    ):
        sensor_id_arr = self._cache[split]["sensor_id"]
        time_arr = self._cache[split]["time"]
        charge_arr = self._cache[split]["charge"]

        s0 = int(first_pulse_index)
        s1 = int(last_pulse_index) + 1

        xyz_out.fill(0.0)
        time_out.fill(0.0)
        charge_out.fill(0.0)

        sensor_id = sensor_id_arr[s0:s1]
        n_total = sensor_id.shape[0]
        n = n_total if n_total < block_size else block_size
        if n <= 0:
            return

        xyz_raw = _sensor_xyz[sensor_id[:n]]  # (n,3) float32
        xyz_out[:n, :] = xyz_raw

        mu = xyz_out[:n, :].mean(axis=0, dtype=np.float32)
        xyz_out[:n, :] = (xyz_out[:n, :] - mu) / 500.0

        t = (time_arr[s0 : s0 + n] / dt) / t_max
        np.cumsum(t, dtype=np.float32, out=time_out[:n])

        np.clip(charge_arr[s0 : s0 + n], 0.0, 5.0, out=charge_out[:n])

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        xyz = np.zeros((block_size, 3), dtype=np.float32)
        time = np.zeros((block_size,), dtype=np.float32)
        charge = np.zeros((block_size,), dtype=np.float32)
        self.fill_single_event_np(
            first_pulse_index, last_pulse_index, xyz, time, charge, split=split
        )
        return torch.from_numpy(xyz), torch.from_numpy(time), torch.from_numpy(charge)




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
    raise RuntimeError(
        "estimate_loss() not supported in this inference-only optimized script."
    )




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




## === cell 8
model = TransformerModel(
    num_sensors, embed_dim, num_time, num_heads, n_layers, dropout=0.2
)
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")




## === cell 9
if "data_loader" not in globals():
    data_loader = EventDataLoader()

df_eval = data_loader.meta_data["eval"]
row0 = df_eval.iloc[0]
xyz, time, charge = data_loader.get_single_event(
    int(row0["first_pulse_index"]), int(row0["last_pulse_index"]), split="eval"
)
xyz.shape




## === cell 10
model.eval()
with torch.inference_mode():
    xyz_b = xyz.to(device).unsqueeze(0)
    time_b = time.to(device).unsqueeze(0)
    charge_b = charge.to(device).unsqueeze(0)
    pred, _ = model(xyz_b, time_b, charge_b, targets=None)
pred.shape




## === cell 11
pass




## === cell 12
torch.save(model.state_dict(), "model.pth")




## === cell 13
df = data_loader.meta_data["eval"]
df.head()




## === cell 14
model.eval()


@torch.no_grad()
def predict_dataframe_batched(df_meta, batch_events=1024):
    n = len(df_meta)
    az = np.empty(n, dtype=np.float32)
    ze = np.empty(n, dtype=np.float32)

    if "batch_id" in df_meta.columns:
        groups = df_meta.groupby("batch_id", sort=False)
    else:
        groups = [(data_loader.batch_id_current["eval"], df_meta)]

    pin = torch.cuda.is_available()
    xyz_batch_cpu = torch.empty(
        (batch_events, block_size, 3), dtype=torch.float32, pin_memory=pin
    )
    time_batch_cpu = torch.empty(
        (batch_events, block_size), dtype=torch.float32, pin_memory=pin
    )
    charge_batch_cpu = torch.empty(
        (batch_events, block_size), dtype=torch.float32, pin_memory=pin
    )

    xyz_np = xyz_batch_cpu.numpy()
    time_np = time_batch_cpu.numpy()
    charge_np = charge_batch_cpu.numpy()

    for batch_id, g in groups:
        batch_id = int(batch_id)
        if data_loader.batch_id_current["eval"] != batch_id:
            data_loader.batch_id_current["eval"] = batch_id
            data_loader.load_batch_events("eval")

        idxs = g.index.to_numpy()
        fp = g["first_pulse_index"].to_numpy(np.int64, copy=False)
        lp = g["last_pulse_index"].to_numpy(np.int64, copy=False)

        for s in tqdm(
            range(0, len(g), batch_events),
            desc=f"Predict batch_id={batch_id}",
            leave=False,
        ):
            e = min(s + batch_events, len(g))
            cur_idx = idxs[s:e]
            b = e - s

            for i in range(b):
                data_loader.fill_single_event_np(
                    int(fp[s + i]),
                    int(lp[s + i]),
                    xyz_np[i],
                    time_np[i],
                    charge_np[i],
                    split="eval",
                )

            with torch.inference_mode():
                pred, _ = model(
                    xyz_batch_cpu[:b].to(device, non_blocking=True),
                    time_batch_cpu[:b].to(device, non_blocking=True),
                    charge_batch_cpu[:b].to(device, non_blocking=True),
                    targets=None,
                )
            pred = pred.detach().cpu().numpy()
            az[cur_idx] = pred[:, 0]
            ze[cur_idx] = pred[:, 1]

    return az, ze


az, ze = predict_dataframe_batched(df, batch_events=1024)
df = df.copy()
df["azimuth"] = az
df["zenith"] = ze
df = df[["event_id", "azimuth", "zenith"]]
df.head()




## === cell 15
df = df.sort_values(["event_id"])
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(df), "rows")

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

# 5. Target score

1.533632

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
        print("Loading large input train meta parquet file ...")
        df_meta = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet"
        )
        print("Loading large input test meta parquet file ...")
        df_meta_eval = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
        )

        self.test_batch_ids = sorted(df_meta_eval["batch_id"].unique().tolist())
        if len(self.test_batch_ids) == 0:
            raise RuntimeError("No batch_id found in test_meta.parquet")

        n_train_data = len(df_meta) - n_test_data
        self.change_batch_every = change_batch_every
        self.meta_data = {
            "train": df_meta[:n_train_data].reset_index(drop=True),
            "test": df_meta[n_train_data:].reset_index(drop=True),
            "eval": df_meta_eval.reset_index(drop=True),
        }
        self.batch_id_current = {"train": 0, "test": 0, "eval": 0}

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
        self.counter = 0  # used to keep track of the current batch

        self._events_np = {"train": None, "test": None, "eval": None}
        self._refresh_events_np("train")
        self._refresh_events_np("test")
        self._refresh_events_np("eval")

    def shuffle_metadata_batch(self, split):
        if split in ["train", "test"]:
            random_batch_id = np.random.randint(
                self.meta_data[split]["batch_id"].min(),
                self.meta_data[split]["batch_id"].max() + 1,
            )
            self.batch_id_current[split] = int(random_batch_id)
        else:
            self.batch_id_current[split] = int(np.random.choice(self.test_batch_ids))
        return self.meta_data[split][
            self.meta_data[split]["batch_id"] == self.batch_id_current[split]
        ].reset_index(drop=True)

    def load_batch_events(self, split):
        """load a batch of events from the parquet file"""
        special_dir = "train" if split in ["train", "test"] else "test"
        batch_id = self.batch_id_current[split]
        path = f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet"
        return pd.read_parquet(path).reset_index()

    def _refresh_events_np(self, split):
        ev = self.events[split]
        self._events_np[split] = {
            "time": ev["time"].to_numpy(dtype=np.float32, copy=False),
            "sensor_id": ev["sensor_id"].to_numpy(dtype=np.int64, copy=False),
            "charge": ev["charge"].to_numpy(dtype=np.float32, copy=False),
        }

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        assert split in [
            "train",
            "test",
            "eval",
        ], "split must be either 'train', 'test' or 'eval'"

        np_ev = self._events_np[split]
        fp = int(first_pulse_index)
        lp = int(last_pulse_index)
        sl = slice(fp, lp + 1)

        time_raw = np_ev["time"][sl]  # float32
        sid = np_ev["sensor_id"][sl]  # int64
        charge_raw = np_ev["charge"][sl]  # float32

        event_size = time_raw.shape[0]
        T = block_size
        n_take = event_size if event_size < T else T

        xyz = np.zeros((T, 3), dtype=np.float32)
        time = np.zeros((T,), dtype=np.float32)
        charge = np.zeros((T,), dtype=np.float32)

        if n_take > 0:
            sid0 = sid[:n_take] - _sensor_id_min
            x = _geom_x[sid0]
            y = _geom_y[sid0]
            z = _geom_z[sid0]
            xyz[:n_take, 0] = x
            xyz[:n_take, 1] = y
            xyz[:n_take, 2] = z

            mean = xyz[:n_take].sum(axis=0) / float(T)
            xyz[:n_take] = (xyz[:n_take] - mean) / 500.0

            t = (time_raw[:n_take] / float(dt) / float(t_max)).astype(
                np.float32, copy=False
            )
            time[:n_take] = np.cumsum(t, dtype=np.float32)

            c = np.clip(charge_raw[:n_take], 0.0, 5.0).astype(np.float32, copy=False)
            charge[:n_take] = c

        xyz_t = torch.from_numpy(xyz)
        time_t = torch.from_numpy(time)
        charge_t = torch.from_numpy(charge)
        return xyz_t, time_t, charge_t

    def change_event_batch(self):
        """change both the train and test event batches"""
        self.meta_data_current = {
            "train": self.shuffle_metadata_batch("train"),
            "test": self.shuffle_metadata_batch("test"),
            "eval": self.shuffle_metadata_batch("eval"),
        }
        self.events["train"] = self.load_batch_events("train")
        self.events["test"] = self.load_batch_events("test")
        self.events["eval"] = self.load_batch_events("eval")
        self._refresh_events_np("train")
        self._refresh_events_np("test")
        self._refresh_events_np("eval")
        self.counter = 0

    def get_xy(self, split):
        assert split in ["train", "test"], "split must be either 'train' or 'test'"

        df_meta_sample = (
            self.meta_data_current[split].sample(n=batch_size).reset_index(drop=True)
        )
        first_pulse_indices = df_meta_sample["first_pulse_index"].to_list()
        last_pulse_indices = df_meta_sample["last_pulse_index"].to_list()

        xyz_batch = torch.empty((batch_size, block_size, 3), dtype=torch.float32)
        time_batch = torch.empty((batch_size, block_size), dtype=torch.float32)
        charge_batch = torch.empty((batch_size, block_size), dtype=torch.float32)

        for j, (first_pulse_index, last_pulse_index) in enumerate(
            zip(first_pulse_indices, last_pulse_indices)
        ):
            xyz, time, charge = self.get_single_event(
                first_pulse_index, last_pulse_index, split
            )
            xyz_batch[j].copy_(xyz)
            time_batch[j].copy_(time)
            charge_batch[j].copy_(charge)

        xyz_batch = xyz_batch.to(device, non_blocking=True)
        time_batch = time_batch.to(device, non_blocking=True)
        charge_batch = charge_batch.to(device, non_blocking=True)

        y = torch.tensor(
            df_meta_sample[["azimuth", "zenith"]].values,
            dtype=torch.float32,
            device=device,
        )

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
            (xyz, time, charge), y = data_loader.get_xy(split)
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



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3333897894.py in <cell line: 0>()
      1 (xyz, time, charge), y = data_loader.get_xy("train")
      2 print(xyz.shape, time.shape, charge.shape)
----> 3 pred, _ = model(xyz, time, charge, y)
      4 pred.shape
      5 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_11/436576058.py in forward(self, xyz, time, charge, targets)
     92         pos_embedding_expanded = self.pos_embedding.expand(x.size(0), x.size(1), -1)
     93         x = torch.cat([x, pos_embedding_expanded], dim=-1)
---> 94         x = self.blocks(x)
     95         x = self.ln_f(x)
     96         x = x.mean(dim=1)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_11/436576058.py in forward(self, x)
     66 
     67     def forward(self, x):
---> 68         x = x + self.sa(self.ln1(x))
     69         x = x + self.ffwd(self.ln2(x))
     70         return x

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_11/436576058.py in forward(self, x)
     33 
     34     def forward(self, x):
---> 35         out = torch.cat([h(x) for h in self.heads], dim=-1)
     36         out = self.dropout(self.proj(out))
     37         return out

/tmp/ipykernel_11/436576058.py in <listcomp>(.0)
     33 
     34     def forward(self, x):
---> 35         out = torch.cat([h(x) for h in self.heads], dim=-1)
     36         out = self.dropout(self.proj(out))
     37         return out

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_11/436576058.py in forward(self, x)
     11     def forward(self, x):
     12         B, T, C = x.shape
---> 13         k = self.key(x)
     14         q = self.query(x)
     15         wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

## === cell 11
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

try:
    model = torch.compile(model, mode="reduce-overhead")
except Exception:
    pass

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



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_11/2981693681.py in <cell line: 0>()
     11 for iter in tqdm(range(max_iters), desc="training"):
     12     (xyz, time, charge), y = data_loader.get_xy("train")
---> 13     logits, loss = model(xyz, time, charge, y)
     14 
     15     optimizer.zero_grad(set_to_none=True)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in RETURN_VALUE(self, inst)
   3046 
   3047     def RETURN_VALUE(self, inst):
-> 3048         self._return(inst)
   3049 
   3050     def RETURN_CONST(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _return(self, inst)
   3031         )
   3032         log.debug("%s triggered compile", inst.opname)
-> 3033         self.output.compile_subgraph(
   3034             self,
   3035             reason=GraphCompileReason(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_subgraph(self, tx, partial_convert, reason)
   1134             if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
   1135                 output.extend(
-> 1136                     self.compile_and_call_fx_graph(
   1137                         tx, pass2.graph_output_vars(), root, output_replacements
   1138                     )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_and_call_fx_graph(self, tx, rv, root, replaced_outputs)
   1380 
   1381             with self.restore_global_state():
-> 1382                 compiled_fn = self.call_user_compiler(gm)
   1383 
   1384             from torch.fx._lazy_graph_module import _LazyGraphModule

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in call_user_compiler(self, gm)
   1430             dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
   1431         ):
-> 1432             return self._call_user_compiler(gm)
   1433 
   1434     def _call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1481             raise e
   1482         except Exception as e:
-> 1483             raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
   1484                 e.__traceback__
   1485             ) from None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1460             if config.verify_correctness:
   1461                 compiler_fn = WrapperBackend(compiler_fn)
-> 1462             compiled_fn = compiler_fn(gm, self.example_inputs())
   1463             _step_logger()(logging.INFO, f"done compiler function {name}")
   1464             assert callable(compiled_fn), "compiler_fn did not return callable"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_dynamo.py in __call__(self, gm, example_inputs, **kwargs)
    128                     raise
    129         else:
--> 130             compiled_gm = compiler_fn(gm, example_inputs)
    131 
    132         return compiled_gm

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in __call__(self, model_, inputs_)
   2338         from torch._inductor.compile_fx import compile_fx
   2339 
-> 2340         return compile_fx(model_, inputs_, config_patches=self.config)
   2341 
   2342     def get_compiler_config(self):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1550     if config_patches:
   1551         with config.patch(config_patches):
-> 1552             return compile_fx(
   1553                 model_,
   1554                 example_inputs_,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1861             unlift_effect_tokens=True
   1862         ):
-> 1863             return aot_autograd(
   1864                 fw_compiler=fw_compiler,
   1865                 bw_compiler=bw_compiler,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/backends/common.py in __call__(self, gm, example_inputs, **kwargs)
     81             # NB: NOT cloned!
     82             with enable_aot_logging(), patch_config:
---> 83                 cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
     84                 counters["aot_autograd"]["ok"] += 1
     85                 return disable(cg)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations, inference_compiler, cudagraphs)
   1153         )
   1154     else:
-> 1155         compiled_fn = dispatch_and_compile()
   1156 
   1157     if isinstance(mod, torch._dynamo.utils.GmWrapper):

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in dispatch_and_compile()
   1129         functional_call = create_functional_call(mod, params_spec, params_len)
   1130         with compiled_autograd._disable():
-> 1131             compiled_fn, _ = create_aot_dispatcher_function(
   1132                 functional_call,
   1133                 fake_flat_args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    578 ) -> Tuple[Callable, ViewAndMutationMeta]:
    579     with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
--> 580         return _create_aot_dispatcher_function(
    581             flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
    582         )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in _create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    828         compiler_fn = choose_dispatcher(needs_autograd, aot_config)
    829 
--> 830         compiled_fn, fw_metadata = compiler_fn(
    831             flat_fn,
    832             _dup_fake_script_obj(fake_flat_args),

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py in aot_dispatch_autograd(flat_fn, flat_args, aot_config, fw_metadata)
    447             if fake_mode is not None and fake_mode.shape_env is not None:
    448                 tensorify_python_scalars(fx_g, fake_mode.shape_env, fake_mode)
--> 449             fw_module, bw_module = aot_config.partition_fn(
    450                 fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs
    451             )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in partition_fn(gm, joint_inputs, **kwargs)
   1777             cuda_context = get_cuda_device_context(gm)
   1778             with cuda_context:
-> 1779                 _recursive_joint_graph_passes(gm)
   1780             return min_cut_rematerialization_partition(
   1781                 gm, joint_inputs, **kwargs, compiler="inductor"

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in _recursive_joint_graph_passes(gm)
    320             subgraph = getattr(gm, subgraph_name)
    321             _recursive_joint_graph_passes(subgraph)
--> 322         joint_graph_passes(gm)
    323 
    324 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/joint_graph.py in joint_graph_passes(graph)
    466             maybe_count = GraphTransformObserver(
    467                 graph, f"pass_pattern_{i}"
--> 468             ).apply_graph_pass(patterns.apply)
    469             count += maybe_count if maybe_count is not None else 0
    470 

/usr/local/lib/python3.11/dist-packages/torch/fx/passes/graph_transform_observer.py in apply_graph_pass(self, pass_fn)
     68         with self:
     69             if not self._check_disable_pass():
---> 70                 return pass_fn(self.gm.graph)
     71 
     72         return None

/usr/local/lib/python3.11/dist-packages/torch/_inductor/pattern_matcher.py in apply(self, gm)
   1771                     if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
   1772                         log.warning("%s%s %s %s", node, node.args, m, entry.pattern)
-> 1773                     if is_match(m) and entry.extra_check(m):
   1774                         count += 1
   1775                         entry.apply(m, graph, node)  # type: ignore[arg-type]

/usr/local/lib/python3.11/dist-packages/torch/_inductor/pattern_matcher.py in check_fn(match)
   1350             specific_pattern_match = specific_pattern.match(node)
   1351 
-> 1352             if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
   1353                 # trace the pattern using the shapes from the user program
   1354                 match.replacement_graph = trace_fn(replace_fn, args)

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in should_pad_mm(match)
    706 def should_pad_mm(match: Match) -> bool:
    707     mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
--> 708     return should_pad_common(mat1, mat2) and should_pad_bench(
    709         match, mat1, mat2, torch.ops.aten.mm
    710     )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in should_pad_bench(*args, **kwargs)
    384 def should_pad_bench(*args, **kwargs):
    385     with dynamo_timed("pad_mm_benchmark"):
--> 386         return _should_pad_bench(*args, **kwargs)
    387 
    388 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in _should_pad_bench(match, mat1, mat2, op, input)
    592 
    593         if ori_time is None:
--> 594             ori_time = do_bench(orig_bench_fn)
    595             set_cached_base_mm_benchmark_time(ori_time_key, ori_time)
    596 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/benchmarking.py in wrapper(self, *args, **kwargs)
     64             "benchmarking." + self.__class__.__name__ + "." + fn.__name__
     65         ] += 1
---> 66         return fn(self, *args, **kwargs)
     67 
     68     return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/benchmarking.py in benchmark_gpu(self, _callable, **kwargs)
    200         elif "return_mode" in kwargs:
    201             return self.triton_do_bench(_callable, **kwargs)
--> 202         return self.triton_do_bench(_callable, **kwargs, return_mode="median")
    203 
    204 

/usr/local/lib/python3.11/dist-packages/triton/testing.py in do_bench(fn, warmup, rep, grad_to_none, quantiles, return_mode)
    115     di = runtime.driver.active.get_device_interface()
    116 
--> 117     fn()
    118     di.synchronize()
    119 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in orig_bench_fn()
    561         def orig_bench_fn():
    562             if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
--> 563                 op(mat1, mat2)
    564             else:
    565                 op(input, mat1, mat2)

/usr/local/lib/python3.11/dist-packages/torch/_ops.py in __call__(self, *args, **kwargs)
   1121         if self._has_torchbind_op_overload and _must_dispatch_in_python(args, kwargs):
   1122             return _call_overload_packet_from_python(self, args, kwargs)
-> 1123         return self._op(*args, **(kwargs or {}))
   1124 
   1125     # TODO: use this to make a __dir__

BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


## === cell 12
torch.save(model.state_dict(), "model.pth")



## === cell 13
test_meta = pd.read_parquet(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
)
sample_sub_path = "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv"
test_batch_ids = sorted(test_meta["batch_id"].unique().tolist())
len(test_batch_ids), test_batch_ids[:5], test_batch_ids[-5:]



## === cell 14
model.eval()


def predict_batch(batch_id: int) -> pd.DataFrame:
    pulses = pd.read_parquet(
        f"/kaggle/input/icecube-neutrinos-in-deep-ice/test/batch_{batch_id}.parquet"
    ).reset_index()

    time_arr = pulses["time"].to_numpy(dtype=np.float32, copy=False)
    sid_arr = pulses["sensor_id"].to_numpy(dtype=np.int64, copy=False)
    charge_arr = pulses["charge"].to_numpy(dtype=np.float32, copy=False)

    meta_b = test_meta[test_meta["batch_id"] == batch_id][
        ["event_id", "first_pulse_index", "last_pulse_index"]
    ].reset_index(drop=True)

    n_events = len(meta_b)
    preds = np.zeros((n_events, 2), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_events, batch_size):
            end = min(start + batch_size, n_events)
            chunk = meta_b.iloc[start:end]

            bsz = end - start
            xyz_np = np.zeros((bsz, block_size, 3), dtype=np.float32)
            time_np = np.zeros((bsz, block_size), dtype=np.float32)
            charge_np = np.zeros((bsz, block_size), dtype=np.float32)

            for j, (fp, lp) in enumerate(
                zip(
                    chunk["first_pulse_index"].to_list(),
                    chunk["last_pulse_index"].to_list(),
                )
            ):
                fp = int(fp)
                lp = int(lp)
                sl = slice(fp, lp + 1)
                t_raw = time_arr[sl]
                sid = sid_arr[sl]
                c_raw = charge_arr[sl]

                event_size = t_raw.shape[0]
                T = block_size
                n_take = event_size if event_size < T else T

                if n_take > 0:
                    sid0 = sid[:n_take] - _sensor_id_min
                    x = _geom_x[sid0]
                    y = _geom_y[sid0]
                    z = _geom_z[sid0]
                    xyz_np[j, :n_take, 0] = x
                    xyz_np[j, :n_take, 1] = y
                    xyz_np[j, :n_take, 2] = z

                    mean = xyz_np[j, :n_take].sum(axis=0) / float(T)
                    xyz_np[j, :n_take] = (xyz_np[j, :n_take] - mean) / 500.0

                    t = (t_raw[:n_take] / float(dt) / float(t_max)).astype(
                        np.float32, copy=False
                    )
                    time_np[j, :n_take] = np.cumsum(t, dtype=np.float32)

                    charge_np[j, :n_take] = np.clip(c_raw[:n_take], 0.0, 5.0).astype(
                        np.float32, copy=False
                    )

            xyz_t = torch.from_numpy(xyz_np).to(device, non_blocking=True)
            time_t = torch.from_numpy(time_np).to(device, non_blocking=True)
            charge_t = torch.from_numpy(charge_np).to(device, non_blocking=True)

            out, _ = model(xyz_t, time_t, charge_t, targets=None)
            preds[start:end] = out.detach().cpu().numpy().astype(np.float32, copy=False)

    return pd.DataFrame(
        {
            "event_id": meta_b["event_id"].values,
            "azimuth": preds[:, 0],
            "zenith": preds[:, 1],
        }
    )


out_path = "submission.csv"
if os.path.exists(out_path):
    os.remove(out_path)

first = True
for b in tqdm(test_batch_ids, desc="predicting test batches"):
    df_b = predict_batch(int(b))
    df_b.to_csv(out_path, index=False, mode="w" if first else "a", header=first)
    first = False

out_path



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_11/3813749224.py in <cell line: 0>()
     90 first = True
     91 for b in tqdm(test_batch_ids, desc="predicting test batches"):
---> 92     df_b = predict_batch(int(b))
     93     df_b.to_csv(out_path, index=False, mode="w" if first else "a", header=first)
     94     first = False

/tmp/ipykernel_11/3813749224.py in predict_batch(batch_id)
     72             charge_t = torch.from_numpy(charge_np).to(device, non_blocking=True)
     73 
---> 74             out, _ = model(xyz_t, time_t, charge_t, targets=None)
     75             preds[start:end] = out.detach().cpu().numpy().astype(np.float32, copy=False)
     76 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in RETURN_VALUE(self, inst)
   3046 
   3047     def RETURN_VALUE(self, inst):
-> 3048         self._return(inst)
   3049 
   3050     def RETURN_CONST(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _return(self, inst)
   3031         )
   3032         log.debug("%s triggered compile", inst.opname)
-> 3033         self.output.compile_subgraph(
   3034             self,
   3035             reason=GraphCompileReason(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_subgraph(self, tx, partial_convert, reason)
   1134             if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
   1135                 output.extend(
-> 1136                     self.compile_and_call_fx_graph(
   1137                         tx, pass2.graph_output_vars(), root, output_replacements
   1138                     )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_and_call_fx_graph(self, tx, rv, root, replaced_outputs)
   1380 
   1381             with self.restore_global_state():
-> 1382                 compiled_fn = self.call_user_compiler(gm)
   1383 
   1384             from torch.fx._lazy_graph_module import _LazyGraphModule

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in call_user_compiler(self, gm)
   1430             dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
   1431         ):
-> 1432             return self._call_user_compiler(gm)
   1433 
   1434     def _call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1481             raise e
   1482         except Exception as e:
-> 1483             raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
   1484                 e.__traceback__
   1485             ) from None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1460             if config.verify_correctness:
   1461                 compiler_fn = WrapperBackend(compiler_fn)
-> 1462             compiled_fn = compiler_fn(gm, self.example_inputs())
   1463             _step_logger()(logging.INFO, f"done compiler function {name}")
   1464             assert callable(compiled_fn), "compiler_fn did not return callable"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_dynamo.py in __call__(self, gm, example_inputs, **kwargs)
    128                     raise
    129         else:
--> 130             compiled_gm = compiler_fn(gm, example_inputs)
    131 
    132         return compiled_gm

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in __call__(self, model_, inputs_)
   2338         from torch._inductor.compile_fx import compile_fx
   2339 
-> 2340         return compile_fx(model_, inputs_, config_patches=self.config)
   2341 
   2342     def get_compiler_config(self):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1550     if config_patches:
   1551         with config.patch(config_patches):
-> 1552             return compile_fx(
   1553                 model_,
   1554                 example_inputs_,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1861             unlift_effect_tokens=True
   1862         ):
-> 1863             return aot_autograd(
   1864                 fw_compiler=fw_compiler,
   1865                 bw_compiler=bw_compiler,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/backends/common.py in __call__(self, gm, example_inputs, **kwargs)
     81             # NB: NOT cloned!
     82             with enable_aot_logging(), patch_config:
---> 83                 cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
     84                 counters["aot_autograd"]["ok"] += 1
     85                 return disable(cg)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations, inference_compiler, cudagraphs)
   1153         )
   1154     else:
-> 1155         compiled_fn = dispatch_and_compile()
   1156 
   1157     if isinstance(mod, torch._dynamo.utils.GmWrapper):

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in dispatch_and_compile()
   1129         functional_call = create_functional_call(mod, params_spec, params_len)
   1130         with compiled_autograd._disable():
-> 1131             compiled_fn, _ = create_aot_dispatcher_function(
   1132                 functional_call,
   1133                 fake_flat_args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    578 ) -> Tuple[Callable, ViewAndMutationMeta]:
    579     with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
--> 580         return _create_aot_dispatcher_function(
    581             flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
    582         )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in _create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    828         compiler_fn = choose_dispatcher(needs_autograd, aot_config)
    829 
--> 830         compiled_fn, fw_metadata = compiler_fn(
    831             flat_fn,
    832             _dup_fake_script_obj(fake_flat_args),

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py in aot_dispatch_base(flat_fn, flat_args, aot_config, fw_metadata)
    201                 assert isinstance(fw_module, GraphModule)
    202                 tensorify_python_scalars(fw_module, fake_mode.shape_env, fake_mode)
--> 203             compiled_fw = compiler(fw_module, updated_flat_args)
    204 
    205         if fakified_out_wrapper.needs_post_compile:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in __call__(self, gm, example_inputs)
    487         example_inputs: Sequence[InputType],
    488     ) -> OutputCode:
--> 489         return self.compiler_fn(gm, example_inputs)
    490 
    491 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fw_compiler_base(gm, example_inputs, is_inference)
   1677                 if is_inference:
   1678                     # partition_fn won't be called
-> 1679                     _recursive_joint_graph_passes(gm)
   1680 
   1681                 fixed = torch._inductor.utils.num_fw_fixed_arguments(

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in _recursive_joint_graph_passes(gm)
    320             subgraph = getattr(gm, subgraph_name)
    321             _recursive_joint_graph_passes(subgraph)
--> 322         joint_graph_passes(gm)
    323 
    324 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/joint_graph.py in joint_graph_passes(graph)
    466             maybe_count = GraphTransformObserver(
    467                 graph, f"pass_pattern_{i}"
--> 468             ).apply_graph_pass(patterns.apply)
    469             count += maybe_count if maybe_count is not None else 0
    470 

/usr/local/lib/python3.11/dist-packages/torch/fx/passes/graph_transform_observer.py in apply_graph_pass(self, pass_fn)
     68         with self:
     69             if not self._check_disable_pass():
---> 70                 return pass_fn(self.gm.graph)
     71 
     72         return None

/usr/local/lib/python3.11/dist-packages/torch/_inductor/pattern_matcher.py in apply(self, gm)
   1771                     if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
   1772                         log.warning("%s%s %s %s", node, node.args, m, entry.pattern)
-> 1773                     if is_match(m) and entry.extra_check(m):
   1774                         count += 1
   1775                         entry.apply(m, graph, node)  # type: ignore[arg-type]

/usr/local/lib/python3.11/dist-packages/torch/_inductor/pattern_matcher.py in check_fn(match)
   1350             specific_pattern_match = specific_pattern.match(node)
   1351 
-> 1352             if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
   1353                 # trace the pattern using the shapes from the user program
   1354                 match.replacement_graph = trace_fn(replace_fn, args)

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in should_pad_bmm(match)
    777 def should_pad_bmm(match: Match) -> bool:
    778     mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
--> 779     return should_pad_common(mat1, mat2) and should_pad_bench(
    780         match, mat1, mat2, torch.ops.aten.bmm
    781     )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in should_pad_bench(*args, **kwargs)
    384 def should_pad_bench(*args, **kwargs):
    385     with dynamo_timed("pad_mm_benchmark"):
--> 386         return _should_pad_bench(*args, **kwargs)
    387 
    388 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in _should_pad_bench(match, mat1, mat2, op, input)
    592 
    593         if ori_time is None:
--> 594             ori_time = do_bench(orig_bench_fn)
    595             set_cached_base_mm_benchmark_time(ori_time_key, ori_time)
    596 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/benchmarking.py in wrapper(self, *args, **kwargs)
     64             "benchmarking." + self.__class__.__name__ + "." + fn.__name__
     65         ] += 1
---> 66         return fn(self, *args, **kwargs)
     67 
     68     return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/benchmarking.py in benchmark_gpu(self, _callable, **kwargs)
    200         elif "return_mode" in kwargs:
    201             return self.triton_do_bench(_callable, **kwargs)
--> 202         return self.triton_do_bench(_callable, **kwargs, return_mode="median")
    203 
    204 

/usr/local/lib/python3.11/dist-packages/triton/testing.py in do_bench(fn, warmup, rep, grad_to_none, quantiles, return_mode)
    115     di = runtime.driver.active.get_device_interface()
    116 
--> 117     fn()
    118     di.synchronize()
    119 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in orig_bench_fn()
    561         def orig_bench_fn():
    562             if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
--> 563                 op(mat1, mat2)
    564             else:
    565                 op(input, mat1, mat2)

/usr/local/lib/python3.11/dist-packages/torch/_ops.py in __call__(self, *args, **kwargs)
   1121         if self._has_torchbind_op_overload and _must_dispatch_in_python(args, kwargs):
   1122             return _call_overload_packet_from_python(self, args, kwargs)
-> 1123         return self._op(*args, **(kwargs or {}))
   1124 
   1125     # TODO: use this to make a __dir__

BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


## === cell 15
pred = pd.read_csv("submission.csv", usecols=["event_id", "azimuth", "zenith"])
pred = pred.drop_duplicates("event_id", keep="last")
pred = pred.set_index("event_id")

sample_sub_path = "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv"
final_path = "submission.csv"
tmp_path = "submission_tmp.csv"
if os.path.exists(tmp_path):
    os.remove(tmp_path)

chunksize = 1_000_000
first = True
for chunk in tqdm(
    pd.read_csv(sample_sub_path, usecols=["event_id"], chunksize=chunksize),
    desc="writing final submission",
):
    joined = chunk.join(pred, on="event_id", how="left")
    joined["azimuth"] = joined["azimuth"].fillna(0.0).astype(np.float32)
    joined["zenith"] = joined["zenith"].fillna(0.0).astype(np.float32)
    joined.to_csv(tmp_path, index=False, mode="w" if first else "a", header=first)
    first = False

os.replace(tmp_path, final_path)

sub_head = pd.read_csv(final_path, nrows=5)
sub_head, sum(1 for _ in open(final_path, "rb")) - 1

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/245648493.py in <cell line: 0>()
      1 # --- Correctness + speed: fix length/order mismatch without loading the full 13.2M-row sample_submission into memory.
      2 # We stream the sample_submission.csv and map predictions by event_id using a pandas join on a smaller index.
----> 3 pred = pd.read_csv("submission.csv", usecols=["event_id", "azimuth", "zenith"])
      4 pred = pred.drop_duplicates("event_id", keep="last")
      5 pred = pred.set_index("event_id")

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

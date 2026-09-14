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



## === cell 2
DATA_ROOT = "/kaggle/input/icecube-neutrinos-in-deep-ice"

df_sensor_geometry = pd.read_csv(f"{DATA_ROOT}/sensor_geometry.csv")
sensor_ids = sorted(list(set(df_sensor_geometry["sensor_id"].to_list())))
print(sensor_ids[:10], len(sensor_ids))




## === cell 3
class EventDataLoader:
    def __init__(self, n_test_data=500_000):
        np.random.seed(42)
        print("Loading large input train meta parquet file ...")
        df_meta = pd.read_parquet(f"{DATA_ROOT}/train_meta.parquet")
        print("Loading large input test meta parquet file ...")
        df_meta_eval = pd.read_parquet(f"{DATA_ROOT}/test_meta.parquet")

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

    def shuffle_metadata_batch(self, split):
        if split in ["train", "test"]:
            random_batch_id = np.random.randint(
                self.meta_data[split]["batch_id"].min(),
                self.meta_data[split]["batch_id"].max() + 1,
            )
            self.batch_id_current[split] = int(random_batch_id)
            return self.meta_data[split][
                self.meta_data[split]["batch_id"] == random_batch_id
            ].reset_index(drop=True)

        valid_batches = self.meta_data["eval"]["batch_id"].unique()
        random_batch_id = int(np.random.choice(valid_batches))
        self.batch_id_current[split] = random_batch_id
        return self.meta_data[split][
            self.meta_data[split]["batch_id"] == random_batch_id
        ].reset_index(drop=True)

    def load_batch_events(self, split):
        """load a batch of events from the parquet file"""
        special_dir = "train" if split in ["train", "test"] else "test"
        batch_id = self.batch_id_current[split]
        path = f"{DATA_ROOT}/{special_dir}/batch_{batch_id}.parquet"
        if not os.path.exists(path):
            if split == "eval":
                valid_batches = self.meta_data["eval"]["batch_id"].unique()
                for _ in range(50):
                    batch_id = int(np.random.choice(valid_batches))
                    path = f"{DATA_ROOT}/{special_dir}/batch_{batch_id}.parquet"
                    if os.path.exists(path):
                        self.batch_id_current[split] = batch_id
                        self.meta_data_current[split] = self.meta_data[split][
                            self.meta_data[split]["batch_id"] == batch_id
                        ].reset_index(drop=True)
                        break
        return pd.read_parquet(path).reset_index()

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        """get a single event from the dataframe (train or test)"""
        assert split in [
            "train",
            "test",
            "eval",
        ], "split must be either 'train', 'test' or 'eval'"
        event = self.events[split].iloc[first_pulse_index : last_pulse_index + 1]
        event = pd.merge(event, df_sensor_geometry, on="sensor_id", how="left")

        event_size = len(event)
        if event_size < block_size:
            pad = pd.DataFrame(
                np.zeros((block_size - event_size, len(event.columns))),
                columns=event.columns,
            )
            event = pd.concat([event, pad], ignore_index=True)
        event = event.iloc[:block_size]

        xyz = event[["x", "y", "z"]].values.astype(np.float32, copy=False)
        for i in range(3):
            xyz[:, i] = (xyz[:, i] - np.average(xyz[:, i])) / 500.0
            xyz[event_size:, i] = 0.0

        time = (
            event["time"].values.astype(np.float32, copy=False)
            / float(dt)
            / float(t_max)
        )
        time = np.cumsum(time)
        time[event_size:] = 0.0

        charge = event["charge"].values.astype(np.float32, copy=False)
        charge = np.clip(charge, 0, 5)

        xyz = torch.tensor(xyz, dtype=torch.float32)
        time = torch.tensor(time, dtype=torch.float32)
        charge = torch.tensor(charge, dtype=torch.float32)

        return xyz, time, charge

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
        self.counter = 0

    def get_xy(self, split):
        assert split in ["train", "test"], "split must be either 'train' or 'test'"
        df_meta_sample = (
            self.meta_data_current[split].sample(n=batch_size).reset_index(drop=True)
        )

        first_pulse_indices = df_meta_sample["first_pulse_index"].to_list()
        last_pulse_indices = df_meta_sample["last_pulse_index"].to_list()

        xyz_batch, time_batch, charge_batch = [], [], []
        for first_pulse_index, last_pulse_index in zip(
            first_pulse_indices, last_pulse_indices
        ):
            xyz, time, charge = self.get_single_event(
                first_pulse_index, last_pulse_index, split
            )
            xyz_batch.append(xyz)
            time_batch.append(time)
            charge_batch.append(charge)

        xyz = torch.stack(xyz_batch)
        time = torch.stack(time_batch)
        charge = torch.stack(charge_batch)
        y = torch.tensor(
            df_meta_sample[["azimuth", "zenith"]].values, dtype=torch.float32
        )

        xyz = xyz.to(device)
        time = time.to(device)
        charge = charge.to(device)
        y = y.to(device)

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
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters, device="cpu")
        for k in range(eval_iters):
            (xyz, time, charge), y = data_loader.get_xy(split)
            logits, loss = model(xyz, time, charge, y)
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
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
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
        self.apply(self._init_weights)

    def forward(self, xyz, time, charge, targets=None):
        x = torch.cat([xyz, time.unsqueeze(-1), charge.unsqueeze(-1)], dim=-1)
        pos_emb = self.position_embedding_table(torch.arange(block_size, device=device))
        x = torch.cat([x, pos_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)], dim=-1)
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
import time as _time

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
df.head()



## === cell 14

model.eval()

sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"
sample = pd.read_csv(sample_sub_path, usecols=["event_id"])
event_id_to_row = pd.Series(
    np.arange(len(sample), dtype=np.int64), index=sample["event_id"].values
)

pred_az = np.zeros(len(sample), dtype=np.float32)
pred_zen = np.zeros(len(sample), dtype=np.float32)

test_meta = data_loader.meta_data["eval"]
batch_ids = sorted(test_meta["batch_id"].unique().tolist())

for batch_id in tqdm(batch_ids, desc="predicting test batches"):
    meta_b = test_meta[test_meta["batch_id"] == batch_id].reset_index(drop=True)
    path = f"{DATA_ROOT}/test/batch_{batch_id}.parquet"
    if not os.path.exists(path):
        continue

    events_df = pd.read_parquet(path).reset_index()
    events_df = pd.merge(events_df, df_sensor_geometry, on="sensor_id", how="left")

    first_idx = meta_b["first_pulse_index"].to_numpy()
    last_idx = meta_b["last_pulse_index"].to_numpy()
    event_ids = meta_b["event_id"].to_numpy()

    start = 0
    while start < len(meta_b):
        end = min(start + batch_size, len(meta_b))
        xyz_batch, time_batch, charge_batch = [], [], []
        for fp, lp in zip(first_idx[start:end], last_idx[start:end]):
            event = events_df.iloc[fp : lp + 1]

            event_size = len(event)
            if event_size < block_size:
                pad = np.zeros((block_size - event_size, 3), dtype=np.float32)
                xyz = event[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
                xyz = np.vstack([xyz, pad])
            else:
                xyz = event[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)[
                    :block_size
                ]
            for i in range(3):
                xyz[:, i] = (
                    xyz[:, i] - float(np.average(xyz[:event_size, i]))
                    if event_size > 0
                    else xyz[:, i]
                ) / 500.0
                if event_size < block_size:
                    xyz[event_size:, i] = 0.0

            t = (
                event["time"].to_numpy(dtype=np.float32, copy=False)
                / float(dt)
                / float(t_max)
            )
            t = np.cumsum(t)
            if event_size < block_size:
                t = np.concatenate(
                    [t, np.zeros(block_size - event_size, dtype=np.float32)], axis=0
                )
            else:
                t = t[:block_size]
            if event_size < block_size:
                t[event_size:] = 0.0

            c = event["charge"].to_numpy(dtype=np.float32, copy=False)
            c = np.clip(c, 0, 5)
            if event_size < block_size:
                c = np.concatenate(
                    [c, np.zeros(block_size - event_size, dtype=np.float32)], axis=0
                )
            else:
                c = c[:block_size]

            xyz_batch.append(torch.tensor(xyz, dtype=torch.float32))
            time_batch.append(torch.tensor(t, dtype=torch.float32))
            charge_batch.append(torch.tensor(c, dtype=torch.float32))

        xyz_t = torch.stack(xyz_batch).to(device)
        time_t = torch.stack(time_batch).to(device)
        charge_t = torch.stack(charge_batch).to(device)

        with torch.no_grad():
            pred_t, _ = model(xyz_t, time_t, charge_t, targets=None)
        pred_np = pred_t.detach().cpu().numpy()

        rows = event_id_to_row.loc[event_ids[start:end]].to_numpy()
        pred_az[rows] = pred_np[:, 0].astype(np.float32)
        pred_zen[rows] = pred_np[:, 1].astype(np.float32)

        start = end



## === cell 15
submission = pd.DataFrame(
    {
        "event_id": sample["event_id"].values,
        "azimuth": pred_az,
        "zenith": pred_zen,
    }
)

submission = submission.sort_values("event_id").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

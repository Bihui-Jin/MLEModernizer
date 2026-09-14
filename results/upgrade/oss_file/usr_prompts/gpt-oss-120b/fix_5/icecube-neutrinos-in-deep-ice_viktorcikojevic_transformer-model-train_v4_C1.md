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
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

batch_size = 32
block_size = 256  # maximum context length
max_iters = 200  # training iterations
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
dt = 10  # time resolution (ns)
t_max = 4e6
num_time = int(max_time / dt + 1)
num_sensors = 5160
torch.manual_seed(1337)
np.random.seed(42)
random.seed(42)



## === cell 1
df_sensor_geometry = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv"
)
sensor_ids = sorted(df_sensor_geometry["sensor_id"].unique())




## === cell 2
class EventDataLoader:
    def __init__(self, n_test_data=500_000):
        print("Loading train and test meta parquet files...")
        self.meta_train = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet"
        )
        self.meta_test = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
        )
        n_train_data = len(self.meta_train) - n_test_data
        self.meta_data = {
            "train": self.meta_train[:n_train_data].reset_index(drop=True),
            "test": self.meta_train[n_train_data:].reset_index(drop=True),
            "eval": self.meta_test.reset_index(drop=True),
        }
        self.batch_id_current = {"train": 0, "test": 0, "eval": 0}
        self.change_batch_every = change_batch_every
        self.counter = 0

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

    def shuffle_metadata_batch(self, split):
        batch_ids = self.meta_data[split]["batch_id"].unique()
        random_batch_id = np.random.choice(batch_ids)
        self.batch_id_current[split] = int(random_batch_id)
        return self.meta_data[split][
            self.meta_data[split]["batch_id"] == random_batch_id
        ].reset_index(drop=True)

    def load_batch_events(self, split):
        folder = "train" if split in ["train", "test"] else "test"
        batch_id = self.batch_id_current[split]
        path = f"/kaggle/input/icecube-neutrinos-in-deep-ice/{folder}/batch_{batch_id}.parquet"
        df = pd.read_parquet(path).reset_index(drop=True)
        df = pd.merge(df, df_sensor_geometry, on="sensor_id", how="left")
        return df

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        assert split in ["train", "test", "eval"]
        sub = self.events[split].iloc[first_pulse_index : last_pulse_index + 1]
        raw_event_size = len(sub)

        event_size = min(raw_event_size, block_size)

        xyz = np.zeros((block_size, 3), dtype=np.float32)
        time = np.zeros((block_size,), dtype=np.float32)
        charge = np.zeros((block_size,), dtype=np.float32)

        if event_size > 0:
            sub_trunc = sub.iloc[:event_size]
            xyz[:event_size] = sub_trunc[["x", "y", "z"]].values.astype(np.float32)
            mean_xyz = xyz[:event_size].mean(axis=0, keepdims=True)
            xyz[:event_size] = (xyz[:event_size] - mean_xyz) / 500.0

            time[:event_size] = sub_trunc["time"].values.astype(np.float32) / dt / t_max
            time[:event_size] = np.cumsum(time[:event_size])

            charge[:event_size] = np.clip(
                sub_trunc["charge"].values.astype(np.float32), 0, 5
            )

        xyz_t = torch.tensor(xyz, dtype=torch.float32, device=device)
        time_t = torch.tensor(time, dtype=torch.float32, device=device)
        charge_t = torch.tensor(charge, dtype=torch.float32, device=device)

        return xyz_t, time_t, charge_t

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
        assert split in ["train", "test"]
        df_sample = (
            self.meta_data_current[split].sample(n=batch_size).reset_index(drop=True)
        )
        xyz_batch, time_batch, charge_batch = [], [], []
        for fp, lp in zip(
            df_sample["first_pulse_index"], df_sample["last_pulse_index"]
        ):
            xyz, time, charge = self.get_single_event(fp, lp, split)
            xyz_batch.append(xyz)
            time_batch.append(time)
            charge_batch.append(charge)

        xyz = torch.stack(xyz_batch)
        time = torch.stack(time_batch)
        charge = torch.stack(charge_batch)
        y = torch.tensor(
            df_sample[["azimuth", "zenith"]].values, dtype=torch.float32, device=device
        )

        self.counter += 1
        if self.counter == self.change_batch_every:
            self.change_event_batch()
        return (xyz, time, charge), y

    def batch_tensors(self, df_batch, split):
        xyz_list, time_list, charge_list = [], [], []
        for fp, lp in zip(df_batch["first_pulse_index"], df_batch["last_pulse_index"]):
            xyz, time, charge = self.get_single_event(fp, lp, split)
            xyz_list.append(xyz)
            time_list.append(time)
            charge_list.append(charge)
        return torch.stack(xyz_list), torch.stack(time_list), torch.stack(charge_list)




## === cell 3
data_loader = EventDataLoader()




## === cell 4
class Head(nn.Module):
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
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
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


class FeedForward(nn.Module):
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
    def __init__(self, num_heads, n_embd, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, n_embd, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, n_embd, num_heads, n_layer, dropout):
        super().__init__()
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.register_buffer("position_ids", torch.arange(block_size, device=device))
        self.blocks = nn.Sequential(
            *[Block(num_heads, n_embd + 5, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd + 5)
        self.classifier = nn.Linear(n_embd + 5, 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, xyz, time, charge, targets=None):
        x = torch.cat(
            [xyz, time.unsqueeze(-1), charge.unsqueeze(-1)], dim=-1
        )  # (B,T,5)
        pos = self.position_embedding(self.position_ids)
        x = torch.cat(
            [x, pos.unsqueeze(0).repeat(x.shape[0], 1, 1)], dim=-1
        )  # (B,T,5+n_embd)

        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)  # (B, n_embd+5)
        pred = self.classifier(x)  # (B,2)

        loss = None
        if targets is not None:
            loss = torch.mean(
                (targets[:, 0] - pred[:, 0]) ** 2 + (targets[:, 1] - pred[:, 1]) ** 2
            )
        return pred, loss




## === cell 5
model = TransformerModel(embed_dim, num_heads, n_layers, dropout).to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")



## === cell 6
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in tqdm(range(max_iters), desc="training"):
    model.train()
    (xyz, time, charge), y = data_loader.get_xy("train")
    pred, loss = model(xyz, time, charge, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % evaluate_every_step == 0 or step == max_iters - 1:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(eval_iters):
                (v_xyz, v_time, v_charge), v_y = data_loader.get_xy("test")
                _, v_loss = model(v_xyz, v_time, v_charge, v_y)
                val_losses.append(v_loss.item())
        print(
            f"\nStep {step+1}: train loss {loss.item():.4f}, val loss {np.mean(val_losses):.4f}"
        )



## === cell 7
model.eval()
df_eval = data_loader.meta_data["eval"].copy()


def batch_predict(df_batch):
    xyz_tensor, time_tensor, charge_tensor = data_loader.batch_tensors(df_batch, "eval")
    with torch.no_grad():
        preds, _ = model(xyz_tensor, time_tensor, charge_tensor)
    return preds.cpu().numpy()


chunk_size = 8192
preds_all = np.empty((len(df_eval), 2), dtype=np.float32)

for start in range(0, len(df_eval), chunk_size):
    end = min(start + chunk_size, len(df_eval))
    chunk_df = df_eval.iloc[start:end]
    preds_all[start:end] = batch_predict(chunk_df)

df_eval["azimuth"] = preds_all[:, 0]
df_eval["zenith"] = preds_all[:, 1]

submission = df_eval[["event_id", "azimuth", "zenith"]].sort_values("event_id")
submission.to_csv("submission.csv", index=False)
print("submission.csv written, shape:", submission.shape)

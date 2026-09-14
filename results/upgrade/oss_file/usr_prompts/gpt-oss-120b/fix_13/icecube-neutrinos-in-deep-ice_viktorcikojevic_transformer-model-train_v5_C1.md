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

1.533571

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
dropout = 0.1
max_time = 77785
dt = 10  # resolution in nanoseconds. By lowering this number you increase your resolution.
t_max = 4e6  # maximum cumulative time

num_time = int(max_time / dt + 1)
num_sensors = 5160
torch.manual_seed(1337)




## === cell 1
df_sensor_geometry = pd.read_csv(
    "/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv",
    usecols=["sensor_id", "x", "y", "z"],  # only needed columns
)
sensor_ids = sorted(list(set(df_sensor_geometry["sensor_id"].to_list())))
print(sensor_ids[:10], len(sensor_ids))




## === cell 2
class EventDataLoader:
    """
    Loads metadata and pulse data for train/test/eval splits.
    For inference we can skip loading the massive train metadata (and its batches) to save time.
    """

    def __init__(self, n_test_data=500_000, load_train=True, load_test=True):
        np.random.seed(42)
        if load_train:
            print("Loading large input train meta parquet file ...")
            df_meta = pd.read_parquet(
                "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet",
                columns=[
                    "batch_id",
                    "first_pulse_index",
                    "last_pulse_index",
                    "azimuth",
                    "zenith",
                ],
            )
        else:
            df_meta = pd.DataFrame()

        if load_test:
            print("Loading large input test meta parquet file ...")
            df_meta_eval = pd.read_parquet(
                "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet",
                columns=["batch_id", "first_pulse_index", "last_pulse_index"],
            )
        else:
            df_meta_eval = pd.DataFrame()

        n_train_data = len(df_meta) - n_test_data if load_train else 0
        self.change_batch_every = change_batch_every
        self.meta_data = {
            "train": (
                df_meta[:n_train_data].reset_index(drop=True)
                if load_train
                else pd.DataFrame()
            ),
            "test": (
                df_meta[n_train_data:].reset_index(drop=True)
                if load_train
                else pd.DataFrame()
            ),
            "eval": df_meta_eval.reset_index(drop=True),
        }

        self.batch_id_current = {"train": 0, "test": 0, "eval": 0}

        self.events = {
            "train": self.load_batch_events("train") if load_train else pd.DataFrame(),
            "test": self.load_batch_events("test") if load_test else pd.DataFrame(),
            "eval": self.load_batch_events("eval"),
        }

        self.merged_events = {}
        for split in ["train", "test", "eval"]:
            self._merge_events(split)

        self.meta_data_current = {
            "train": (
                self.shuffle_metadata_batch("train")
                if not self.meta_data["train"].empty
                else self.meta_data["train"]
            ),
            "test": (
                self.shuffle_metadata_batch("test")
                if not self.meta_data["test"].empty
                else self.meta_data["test"]
            ),
            "eval": self.shuffle_metadata_batch("eval"),
        }
        self.counter = 0  # used to keep track of the current batch

    def shuffle_metadata_batch(self, split):
        """Return a random batch of metadata for the given split."""
        if split in ["train", "test"]:
            if "batch_id" not in self.meta_data[split].columns:
                return self.meta_data[split].reset_index(drop=True)
            min_id = int(self.meta_data[split]["batch_id"].min())
            max_id = int(self.meta_data[split]["batch_id"].max())
            random_batch_id = (
                min_id if min_id == max_id else np.random.randint(min_id, max_id + 1)
            )
            self.batch_id_current[split] = random_batch_id
        else:
            self.batch_id_current[split] = 661

        meta = self.meta_data[split]
        if "batch_id" not in meta.columns:
            return meta.reset_index(drop=True)
        return meta[meta["batch_id"] == self.batch_id_current[split]].reset_index(
            drop=True
        )

    def load_batch_events(self, split):
        """Load a batch of events from the parquet file, with fallback to any existing file."""
        special_dir = "train" if split in ["train", "test"] else "test"
        batch_id = self.batch_id_current[split]
        path = f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet"
        if not os.path.exists(path):
            dir_path = f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}"
            available = [
                f
                for f in os.listdir(dir_path)
                if f.startswith("batch_") and f.endswith(".parquet")
            ]
            if not available:
                raise FileNotFoundError(f"No batch parquet files found in {dir_path}")
            path = os.path.join(dir_path, sorted(available)[0])
            print(f"[Info] Fallback batch file used for split '{split}': {path}")
        return pd.read_parquet(
            path, columns=["sensor_id", "time", "charge"]
        ).reset_index()

    def _merge_events(self, split):
        """Merge sensor geometry into the current batch once."""
        if self.events[split].empty:
            self.merged_events[split] = pd.DataFrame()
        else:
            self.merged_events[split] = pd.merge(
                self.events[split], df_sensor_geometry, on="sensor_id", how="left"
            )

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        """Extract a single event as tensors, truncating to block_size if necessary."""
        assert split in ["train", "test", "eval"]
        event = (
            self.merged_events[split]
            .iloc[first_pulse_index : last_pulse_index + 1]
            .copy()
        )
        event = event.iloc[:block_size]
        event_size = len(event)

        raw_xyz = event[["x", "y", "z"]].values.astype(np.float32)  # (event_size,3)
        xyz = np.zeros((block_size, 3), dtype=np.float32)
        xyz[:event_size] = raw_xyz
        if event_size > 0:
            avg = np.mean(xyz[:event_size], axis=0, keepdims=True)
            xyz[:event_size] = (xyz[:event_size] - avg) / 500.0

        raw_time = event["time"].values.astype(np.float32) / dt / t_max
        cum_time = np.cumsum(raw_time)
        time = np.zeros(block_size, dtype=np.float32)
        time[:event_size] = cum_time

        raw_charge = np.clip(event["charge"].values.astype(np.float32), 0.0, 5.0)
        charge = np.zeros(block_size, dtype=np.float32)
        charge[:event_size] = raw_charge

        xyz = torch.from_numpy(xyz)
        time = torch.from_numpy(time)
        charge = torch.from_numpy(charge)

        return xyz, time, charge

    def change_event_batch(self):
        """Refresh both metadata and the underlying batch (including merged geometry)."""
        self.meta_data_current = {
            "train": (
                self.shuffle_metadata_batch("train")
                if not self.meta_data["train"].empty
                else self.meta_data["train"]
            ),
            "test": (
                self.shuffle_metadata_batch("test")
                if not self.meta_data["test"].empty
                else self.meta_data["test"]
            ),
            "eval": self.shuffle_metadata_batch("eval"),
        }
        if not self.meta_data["train"].empty:
            self.events["train"] = self.load_batch_events("train")
        if not self.meta_data["test"].empty:
            self.events["test"] = self.load_batch_events("test")
        self.events["eval"] = self.load_batch_events("eval")
        for split in ["train", "test", "eval"]:
            self._merge_events(split)
        self.counter = 0

    def get_xy(self, split):
        """Return a batch of (xyz, time, charge) tensors and target angles."""
        assert split in ["train", "test", "eval"]
        df_meta_source = (
            self.meta_data[split]
            if self.meta_data_current[split].empty
            else self.meta_data_current[split]
        )

        if df_meta_source.empty:
            raise ValueError(f"No metadata available for split '{split}'")

        sample_size = min(batch_size, len(df_meta_source))
        df_meta_sample = df_meta_source.sample(
            n=sample_size, random_state=42
        ).reset_index(drop=True)

        first_pulse_indices = df_meta_sample["first_pulse_index"].tolist()
        last_pulse_indices = df_meta_sample["last_pulse_index"].tolist()

        xyz_batch, time_batch, charge_batch = [], [], []
        for first, last in zip(first_pulse_indices, last_pulse_indices):
            xyz, time, charge = self.get_single_event(first, last, split)
            xyz_batch.append(xyz)
            time_batch.append(time)
            charge_batch.append(charge)

        xyz = torch.stack(xyz_batch).to(device)
        time = torch.stack(time_batch).to(device)
        charge = torch.stack(charge_batch).to(device)

        if {"azimuth", "zenith"}.issubset(df_meta_sample.columns):
            y = torch.tensor(
                df_meta_sample[["azimuth", "zenith"]].values, dtype=torch.float32
            ).to(device)
        else:
            y = torch.zeros((sample_size, 2), device=device)

        self.counter += 1
        if self.counter == self.change_batch_every:
            self.change_event_batch()
            self.counter = 0

        return (xyz, time, charge), y




## === cell 3
data_loader = EventDataLoader(load_train=True, load_test=True)




## === cell 4
def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    """
    Calculate the mean angular distance (MAE) between true and predicted directions.
    """
    if not (
        torch.all(torch.isfinite(az_true))
        and torch.all(torch.isfinite(zen_true))
        and torch.all(torch.isfinite(az_pred))
        and torch.all(torch.isfinite(zen_pred))
    ):
        raise ValueError("All arguments must be finite")

    sa1, ca1 = torch.sin(az_true), torch.cos(az_true)
    sz1, cz1 = torch.sin(zen_true), torch.cos(zen_true)
    sa2, ca2 = torch.sin(az_pred), torch.cos(az_pred)
    sz2, cz2 = torch.sin(zen_pred), torch.cos(zen_pred)

    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + cz1 * cz2
    scalar_prod = torch.clamp(scalar_prod, -1, 1)
    return torch.mean(torch.abs(torch.acos(scalar_prod)))




## === cell 5
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train"]:
        if data_loader.meta_data[split].empty:
            out[split] = float("nan")
            continue
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            (xyz, time, charge), y = data_loader.get_xy(split)
            _, loss = model(xyz, time, charge, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out




## === cell 6
class Head(nn.Module):
    """One head of self‑attention."""

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
    """Multiple heads of self‑attention in parallel."""

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
    """A simple linear layer followed by a non‑linearity."""

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
    """Transformer block: communication followed by computation."""

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
        self.register_buffer(
            "pos_emb",
            self.position_embedding_table(
                torch.arange(block_size, device=device)
            ).unsqueeze(0),
        )

        self.blocks = nn.Sequential(
            *[Block(num_heads, n_embd + 5, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd + 5)
        self.classifier = nn.Linear(n_embd + 5, 2)

        self.apply(self._init_weights)

    def forward(self, xyz, time, charge, targets=None):
        x = torch.cat(
            [xyz, time.unsqueeze(-1), charge.unsqueeze(-1)], dim=-1
        )  # (B,T,5)

        pos_emb = self.pos_emb  # (1, block_size, n_embd)
        x = torch.cat([x, pos_emb.expand(x.shape[0], -1, -1)], dim=-1)  # (B,T,5+n_embd)

        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)  # (B, n_embd+5)
        pred = self.classifier(x)  # (B,2)

        loss = None
        if targets is not None:
            loss = angular_dist_score(
                targets[:, 0], targets[:, 1], pred[:, 0], pred[:, 1]
            )
        return pred, loss




## === cell 7
model = TransformerModel(
    num_sensors, embed_dim, num_time, num_heads, n_layers, dropout=dropout
).to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2288138182.py in <cell line: 0>()
----> 1 model = TransformerModel(
      2     num_sensors, embed_dim, num_time, num_heads, n_layers, dropout=dropout
      3 ).to(device)
      4 print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
      5 

/tmp/ipykernel_55/4120265368.py in __init__(self, num_sensor, n_embd, num_time, num_heads, n_layer, dropout)
     84         self.register_buffer(
     85             "pos_emb",
---> 86             self.position_embedding_table(
     87                 torch.arange(block_size, device=device)
     88             ).unsqueeze(0),

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/sparse.py in forward(self, input)
    188 
    189     def forward(self, input: Tensor) -> Tensor:
--> 190         return F.embedding(
    191             input,
    192             self.weight,

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
   2549         # remove once script supports set_grad_enabled
   2550         _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
-> 2551     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
   2552 
   2553 

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper_CUDA__index_select)

## === cell 8
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
model.train()
train_steps = 1500
for step in range(train_steps):
    (xyz, time, charge), y = data_loader.get_xy("train")
    optimizer.zero_grad()
    _, loss = model(xyz, time, charge, y)
    loss.backward()
    optimizer.step()
    if (step + 1) % 100 == 0:
        print(f"Step {step+1}/{train_steps}, loss: {loss.item():.4f}")




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3329548123.py in <cell line: 0>()
----> 1 optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
      2 model.train()
      3 train_steps = 1500
      4 for step in range(train_steps):
      5     (xyz, time, charge), y = data_loader.get_xy("train")

NameError: name 'model' is not defined

## === cell 9
df = data_loader.meta_data["eval"].copy()
predictions = []
batch_size = 256
for i in range(0, len(df), batch_size):
    first_idxs = df["first_pulse_index"].iloc[i : i + batch_size].tolist()
    last_idxs = df["last_pulse_index"].iloc[i : i + batch_size].tolist()
    xyz_batch, time_batch, charge_batch = [], [], []
    for f, l in zip(first_idxs, last_idxs):
        xyz, time, charge = data_loader.get_single_event(f, l, split="eval")
        xyz_batch.append(xyz)
        time_batch.append(time)
        charge_batch.append(charge)

    xyz_tensor = torch.stack(xyz_batch).to(device)
    time_tensor = torch.stack(time_batch).to(device)
    charge_tensor = torch.stack(charge_batch).to(device)

    with torch.no_grad():
        pred_batch, _ = model(xyz_tensor, time_tensor, charge_tensor)

    pred_batch[:, 0] = torch.remainder(pred_batch[:, 0], 2 * np.pi)  # azimuth ∈ [0, 2π)
    pred_batch[:, 1] = torch.clamp(pred_batch[:, 1], 0.0, np.pi)  # zenith ∈ [0, π]

    predictions.extend(pred_batch.cpu().numpy())

predictions = np.nan_to_num(np.array(predictions), nan=0.0, posinf=0.0, neginf=0.0)

if "event_id" not in df.columns:
    df = df.reset_index().rename(columns={"index": "event_id"})

df["azimuth"] = predictions[:, 0]
df["zenith"] = predictions[:, 1]
df = df[["event_id", "azimuth", "zenith"]]




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2141623213.py in <cell line: 0>()
     17 
     18     with torch.no_grad():
---> 19         pred_batch, _ = model(xyz_tensor, time_tensor, charge_tensor)
     20 
     21     pred_batch[:, 0] = torch.remainder(pred_batch[:, 0], 2 * np.pi)  # azimuth ∈ [0, 2π)

NameError: name 'model' is not defined

## === cell 10
df = df.sort_values("event_id")
submission_path = "submission.csv"
df.to_csv(submission_path, index=False, columns=["event_id", "azimuth", "zenith"])
print(f"Submission written to {submission_path}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1875488766.py in <cell line: 0>()
----> 1 df = df.sort_values("event_id")
      2 submission_path = "submission.csv"
      3 df.to_csv(submission_path, index=False, columns=["event_id", "azimuth", "zenith"])
      4 print(f"Submission written to {submission_path}")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in sort_values(self, by, axis, ascending, inplace, kind, na_position, ignore_index, key)
   7187             # len(by) == 1
   7188 
-> 7189             k = self._get_label_or_level_values(by[0], axis=axis)
   7190 
   7191             # need to rewrap column in Series to apply key function

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1909             values = self.axes[axis].get_level_values(key)._values
   1910         else:
-> 1911             raise KeyError(key)
   1912 
   1913         # Check for duplicates

KeyError: 'event_id'

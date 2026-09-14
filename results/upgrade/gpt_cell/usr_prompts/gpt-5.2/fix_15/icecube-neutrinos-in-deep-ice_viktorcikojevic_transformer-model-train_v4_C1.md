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
import math
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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



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
_sensor_xyz = df_sensor_geometry.set_index("sensor_id")[["x", "y", "z"]].astype(
    np.float32
)
_sensor_id_min = int(_sensor_xyz.index.min())
_sensor_id_max = int(_sensor_xyz.index.max())
_sensor_xyz_lut = np.zeros((_sensor_id_max - _sensor_id_min + 1, 3), dtype=np.float32)
_sensor_xyz_lut[_sensor_xyz.index.values - _sensor_id_min] = _sensor_xyz.values


def _lookup_xyz(sensor_id_arr: np.ndarray) -> np.ndarray:
    return _sensor_xyz_lut[sensor_id_arr.astype(np.int64) - _sensor_id_min]




## === cell 4
class EventDataLoader:
    def __init__(self, n_test_data=500_000):
        np.random.seed(42)
        print("Loading large input train meta parquet file (columns-pruned) ...")
        df_meta = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet",
            columns=[
                "batch_id",
                "event_id",
                "first_pulse_index",
                "last_pulse_index",
                "azimuth",
                "zenith",
            ],
        )
        print("Loading large input test meta parquet file (columns-pruned) ...")
        df_meta_eval = pd.read_parquet(
            "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet",
            columns=[
                "batch_id",
                "event_id",
                "first_pulse_index",
                "last_pulse_index",
            ],
        )

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
        self.counter = 0

    def shuffle_metadata_batch(self, split):
        if split in ["train", "test"]:
            random_batch_id = np.random.randint(
                self.meta_data[split]["batch_id"].min(),
                self.meta_data[split]["batch_id"].max() + 1,
            )
            self.batch_id_current[split] = int(random_batch_id)
        else:
            random_batch_id = int(self.meta_data[split]["batch_id"].min())
            self.batch_id_current[split] = int(random_batch_id)

        return self.meta_data[split][
            self.meta_data[split]["batch_id"] == self.batch_id_current[split]
        ].reset_index(drop=True)

    def load_batch_events(self, split):
        """load a batch of events from the parquet file"""

        special_dir = "train" if split in ["train", "test"] else "test"
        batch_id = self.batch_id_current[split]

        df = pd.read_parquet(
            f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet",
            columns=["event_id", "time", "sensor_id", "charge"],
        )

        self._events_np = getattr(self, "_events_np", {})
        self._events_np[split] = {
            "sensor_id": np.ascontiguousarray(df["sensor_id"].to_numpy(np.int32)),
            "time": np.ascontiguousarray(df["time"].to_numpy(np.float32)),
            "charge": np.ascontiguousarray(df["charge"].to_numpy(np.float32)),
        }
        return df

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        """get a single event from the dataframe (train or test)"""

        assert split in [
            "train",
            "test",
            "eval",
        ], "split must be either 'train', 'test' or 'eval'"

        ev = self._events_np[split]
        fi = int(first_pulse_index)
        li = int(last_pulse_index) + 1  # slice end-exclusive

        sensor_id = ev["sensor_id"][fi:li]
        xyz_src = _lookup_xyz(sensor_id)  # (n,3) float32

        event_size = int(xyz_src.shape[0])
        use_n = event_size if event_size < block_size else block_size

        xyz = np.zeros((block_size, 3), dtype=np.float32)
        if use_n > 0:
            xyz[:use_n] = xyz_src[:use_n]

            m0 = float(np.average(xyz[:use_n, 0]))
            m1 = float(np.average(xyz[:use_n, 1]))
            m2 = float(np.average(xyz[:use_n, 2]))
            xyz[:use_n, 0] = (xyz[:use_n, 0] - m0) / 500.0
            xyz[:use_n, 1] = (xyz[:use_n, 1] - m1) / 500.0
            xyz[:use_n, 2] = (xyz[:use_n, 2] - m2) / 500.0

        t_src = ev["time"][fi:li]
        if use_n > 0:
            t = (t_src[:use_n] / dt / t_max).astype(np.float32, copy=False)
            t = np.cumsum(t, dtype=np.float32)
        else:
            t = np.empty((0,), dtype=np.float32)

        time = np.zeros((block_size,), dtype=np.float32)
        if use_n > 0:
            time[:use_n] = t

        ch_src = ev["charge"][fi:li]
        charge = np.zeros((block_size,), dtype=np.float32)
        if use_n > 0:
            ch = np.clip(ch_src[:use_n], 0.0, 5.0).astype(np.float32, copy=False)
            charge[:use_n] = ch

        return (
            torch.from_numpy(xyz),
            torch.from_numpy(time),
            torch.from_numpy(charge),
        )

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

        dfm = self.meta_data_current[split]
        n_rows = len(dfm)
        idx = np.random.choice(n_rows, size=batch_size, replace=False)
        df_meta_sample = dfm.iloc[idx]

        first_pulse_indices = df_meta_sample["first_pulse_index"].to_numpy(
            np.int64, copy=False
        )
        last_pulse_indices = df_meta_sample["last_pulse_index"].to_numpy(
            np.int64, copy=False
        )

        xyz_batch = []
        time_batch = []
        charge_batch = []
        for fi, li in zip(first_pulse_indices, last_pulse_indices):
            xyz, time, charge = self.get_single_event(fi, li, split)
            xyz_batch.append(xyz)
            time_batch.append(time)
            charge_batch.append(charge)

        xyz = torch.stack(xyz_batch).to(device, non_blocking=True)
        time = torch.stack(time_batch).to(device, non_blocking=True)
        charge = torch.stack(charge_batch).to(device, non_blocking=True)
        y = torch.tensor(
            df_meta_sample[["azimuth", "zenith"]].to_numpy(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
        )

        self.counter += 1
        if self.counter == self.change_batch_every:
            self.change_event_batch()
            self.counter = 0

        return (xyz, time, charge), y




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
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            (xyz, time, charge), y = data_loader.get_xy(split)
            logits, loss = model(xyz, time, charge, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
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

        self.register_buffer(
            "_pos_idx", torch.arange(block_size, dtype=torch.long), persistent=False
        )

        self.apply(self._init_weights)

    def forward(self, xyz, time, charge, targets=None):
        x = torch.cat([xyz, time.unsqueeze(-1), charge.unsqueeze(-1)], dim=-1)

        pos_emb = self.position_embedding_table(self._pos_idx.to(x.device))
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
data_loader = EventDataLoader()

(xyz, time, charge), y = data_loader.get_xy("train")
print(xyz.shape, time.shape, charge.shape)
x = model(xyz, time, charge)
x[0].shape



## === cell 10
torch.save(model.state_dict(), "model.pth")



## === cell 11
model.eval()


@torch.inference_mode()
def make_predictions_streaming_full_test(
    out_csv_path: str,
    infer_batch_size: int = 512,
):
    test_meta_path = "/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet"
    test_batches_dir = "/kaggle/input/icecube-neutrinos-in-deep-ice/test"
    sample_sub_path = (
        "/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.csv"
    )

    df_meta = pd.read_parquet(
        test_meta_path,
        columns=["event_id", "batch_id", "first_pulse_index", "last_pulse_index"],
    )

    meta_event = df_meta["event_id"].to_numpy(np.int64, copy=False)
    meta_batch = df_meta["batch_id"].to_numpy(np.int32, copy=False)
    meta_fi = df_meta["first_pulse_index"].to_numpy(np.int64, copy=False)
    meta_li = df_meta["last_pulse_index"].to_numpy(np.int64, copy=False)

    n_meta = meta_event.shape[0]

    pred_path = "test_preds_memmap.float32"
    if os.path.exists(pred_path):
        os.remove(pred_path)
    preds_mm = np.memmap(pred_path, mode="w+", dtype=np.float32, shape=(n_meta, 2))

    use_pinned = torch.cuda.is_available()
    xyz_cpu = torch.empty(
        (infer_batch_size, block_size, 3), dtype=torch.float32, pin_memory=use_pinned
    )
    time_cpu = torch.empty(
        (infer_batch_size, block_size), dtype=torch.float32, pin_memory=use_pinned
    )
    charge_cpu = torch.empty(
        (infer_batch_size, block_size), dtype=torch.float32, pin_memory=use_pinned
    )

    xyz_np_buf = xyz_cpu.numpy()
    time_np_buf = time_cpu.numpy()
    charge_np_buf = charge_cpu.numpy()

    lookup_xyz = _lookup_xyz
    _dt_tmax = float(dt * t_max)

    unique_batches = np.unique(meta_batch)
    for b in tqdm(
        unique_batches, desc="Predicting test (one parquet load per batch_id)"
    ):
        mask = meta_batch == b
        idxs_in_meta = np.nonzero(mask)[0]
        if idxs_in_meta.size == 0:
            continue

        df = pd.read_parquet(
            f"{test_batches_dir}/batch_{int(b)}.parquet",
            columns=["event_id", "time", "sensor_id", "charge"],
        )
        ev_sensor = np.ascontiguousarray(df["sensor_id"].to_numpy(np.int32, copy=False))
        ev_time = np.ascontiguousarray(df["time"].to_numpy(np.float32, copy=False))
        ev_charge = np.ascontiguousarray(df["charge"].to_numpy(np.float32, copy=False))

        for start in range(0, idxs_in_meta.shape[0], infer_batch_size):
            idxs = idxs_in_meta[start : start + infer_batch_size]
            bs = idxs.shape[0]

            xyz_np_buf[:bs].fill(0.0)
            time_np_buf[:bs].fill(0.0)
            charge_np_buf[:bs].fill(0.0)

            for j in range(bs):
                pos = int(idxs[j])
                fi = int(meta_fi[pos])
                li = int(meta_li[pos]) + 1

                sensor_id = ev_sensor[fi:li]
                xyz_src = lookup_xyz(sensor_id)  # (n,3) float32

                use_n = xyz_src.shape[0]
                if use_n > block_size:
                    use_n = block_size
                if use_n <= 0:
                    continue

                xyz_tmp = xyz_src[:use_n].astype(np.float32, copy=True)
                m = xyz_tmp.mean(axis=0, dtype=np.float32)
                xyz_tmp = (xyz_tmp - m) / 500.0
                xyz_np_buf[j, :use_n, :] = xyz_tmp

                t = (ev_time[fi:li][:use_n] / _dt_tmax).astype(np.float32, copy=False)
                np.cumsum(t, dtype=np.float32, out=t)
                time_np_buf[j, :use_n] = t

                ch = np.clip(ev_charge[fi:li][:use_n], 0.0, 5.0).astype(
                    np.float32, copy=False
                )
                charge_np_buf[j, :use_n] = ch

            xyz_dev = xyz_cpu[:bs].to(device, non_blocking=True)
            time_dev = time_cpu[:bs].to(device, non_blocking=True)
            charge_dev = charge_cpu[:bs].to(device, non_blocking=True)

            pred, _ = model(xyz_dev, time_dev, charge_dev, targets=None)
            preds_mm[idxs, :] = (
                pred.detach().cpu().numpy().astype(np.float32, copy=False)
            )

    preds_mm.flush()

    with open(out_csv_path, "w") as f:
        f.write("event_id,azimuth,zenith\n")

    meta_pos = pd.Series(
        np.arange(n_meta, dtype=np.int64), index=pd.Index(meta_event, name="event_id")
    )

    chunk_iter = pd.read_csv(sample_sub_path, usecols=["event_id"], chunksize=1_000_000)
    n_written = 0
    for df_chunk in tqdm(
        chunk_iter, desc="Writing submission in sample_submission order"
    ):
        ev_ids = df_chunk["event_id"].to_numpy(np.int64, copy=False)

        pos = meta_pos.reindex(ev_ids).to_numpy(
            dtype=np.float64, copy=False
        )  # float with NaN for missing
        out_az = np.zeros(ev_ids.shape[0], dtype=np.float32)
        out_ze = np.zeros(ev_ids.shape[0], dtype=np.float32)

        valid = ~np.isnan(pos)
        if valid.any():
            pos_i = pos[valid].astype(np.int64, copy=False)
            out_az[valid] = preds_mm[pos_i, 0]
            out_ze[valid] = preds_mm[pos_i, 1]

        out_df = pd.DataFrame({"event_id": ev_ids, "azimuth": out_az, "zenith": out_ze})
        out_df.to_csv(out_csv_path, mode="a", header=False, index=False)
        n_written += len(out_df)

    return n_written


n_written = make_predictions_streaming_full_test("submission.csv", infer_batch_size=512)
print("Wrote predictions for", n_written, "rows (matches sample_submission length)")

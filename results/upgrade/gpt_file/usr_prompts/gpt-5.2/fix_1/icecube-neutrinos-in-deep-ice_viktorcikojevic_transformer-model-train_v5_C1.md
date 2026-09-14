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
block_size = 256 # maximum context length
max_iters = 5000 # number of iterations
change_batch_every = 256 # change batch of data every change_batch_every steps
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluate_every_step = 500 
learning_rate = 1e-4
eval_iters = 32 # number of batches to process for evaluation

embed_dim= 64-5
n_layers = 6
num_heads = 6
dropout = 0.2
max_time = 77785
dt = 10 # resolution in nanoseconds. By lowering this number you increase your resolution.
t_max = 4e6 # maximum cumulative time

num_time = int(max_time / dt + 1)
num_sensors = 5160
torch.manual_seed(1337);


## === cell 2
df_sensor_geometry = pd.read_csv("/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv")
sensor_ids = sorted(list(set(df_sensor_geometry["sensor_id"].to_list())))
sensor_ids[:10], len(sensor_ids)


## === cell 3
class EventDataLoader:
    def __init__(self, n_test_data=500_000):
        np.random.seed(42)
        print("Loading large input train meta parquet file ...")

        
        df_meta = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet")
        print("Loading large input test meta parquet file ...")
        df_meta_eval = pd.read_parquet("/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet")
        n_train_data = len(df_meta) - n_test_data
        self.change_batch_every = change_batch_every
        self.meta_data = {
            "train": df_meta[:n_train_data].reset_index(drop=True),
            "test": df_meta[n_train_data:].reset_index(drop=True),
            "eval": df_meta_eval.reset_index(drop=True)
        }
        self.batch_id_current = {
            "train": 0,
            "test": 0,
            "eval": 0
        }
        self.meta_data_current = {
            "train":self.shuffle_metadata_batch("train"),
            "test":self.shuffle_metadata_batch("test"),
            "eval":self.shuffle_metadata_batch("eval")
        }
        self.events = {
            "train": self.load_batch_events("train"),
            "test": self.load_batch_events("test"),
            "eval": self.load_batch_events("eval")
        }
        self.counter=0 # used to keep track of the current batch
    
    def shuffle_metadata_batch(self, split):
        if split in ["train", "test"]:
            random_batch_id = np.random.randint(self.meta_data[split]["batch_id"].min(),
                                                self.meta_data[split]["batch_id"].max()+1)
            self.batch_id_current[split] = random_batch_id
        else:
            random_batch_id = 661
            self.batch_id_current[split] = random_batch_id
        return self.meta_data[split][self.meta_data[split]["batch_id"]==random_batch_id].reset_index(drop=True)
    
    
    def load_batch_events(self,split):
        """ load a batch of events from the parquet file """
        special_dir = "train" if split in ["train", "test"] else "test"
        batch_id = self.batch_id_current[split]
        return pd.read_parquet(f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet").reset_index()
        
    
    

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        """ get a single event from the dataframe (train or test) """
        
        assert split in ["train", "test", "eval"], "split must be either 'train', 'test' or 'eval'"
        
        event = self.events[split].iloc[first_pulse_index:last_pulse_index+1]
        
        event = pd.merge(event, df_sensor_geometry, on="sensor_id")
        
        
        event_size = len(event)
        if event_size < block_size:
            event = event.append(pd.DataFrame(np.zeros((block_size - event_size, len(event.columns))), columns=event.columns))
        event = event[:block_size]
        
        xyz = event[["x", "y", "z"]].values
        for i in range(3):
            xyz[:,i] = (xyz[:,i] - np.average(xyz[:,i])) / 500
            xyz[:,i][event_size:] = 0
        
        time = event["time"].values / dt / t_max
        time = np.cumsum(time)
        time[event_size:] = 0
        
        charge = event["charge"].values
        charge = np.clip(charge, 0, 5)
        
        
        xyz = torch.tensor(xyz, dtype=torch.float32)
        time = torch.tensor(time, dtype=torch.float32)
        charge = torch.tensor(charge, dtype=torch.float32)

        return xyz, time, charge

    def change_event_batch(self):
        """ change both the train and test event batches """
        
        self.meta_data_current = {
            "train":self.shuffle_metadata_batch("train"),
            "test":self.shuffle_metadata_batch("test"),
            "eval":self.shuffle_metadata_batch("eval")
        }
        self.events["train"] = self.load_batch_events("train")
        self.events["test"] = self.load_batch_events("test")
        self.events["eval"] = self.load_batch_events("eval")
        self.counter = 0

    def get_xy(self, split):
        
        
        assert split in ["train", "test"], "split must be either 'train' or 'test'"


        df_meta_sample = self.meta_data_current[split].sample(n=batch_size).reset_index(drop=True)

        first_pulse_indices = df_meta_sample["first_pulse_index"].to_list()
        last_pulse_indices = df_meta_sample["last_pulse_index"].to_list()

        xyz_batch = []
        time_batch = []
        charge_batch = []
        for first_pulse_index, last_pulse_index in zip(first_pulse_indices, last_pulse_indices):
                xyz, time, charge = self.get_single_event(first_pulse_index, last_pulse_index, split)
                xyz_batch.append(xyz)
                time_batch.append(time)
                charge_batch.append(charge)

        xyz = torch.stack(xyz_batch)
        time = torch.stack(time_batch)
        charge = torch.stack(charge_batch)
        y = torch.tensor(df_meta_sample[["azimuth", "zenith"]].values)
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


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1132833988.py in <cell line: 0>()
----> 1 data_loader = EventDataLoader()

/tmp/ipykernel_11/991368643.py in __init__(self, n_test_data)
     37             "train": self.load_batch_events("train"),
     38             "test": self.load_batch_events("test"),
---> 39             "eval": self.load_batch_events("eval")
     40         }
     41         self.counter=0 # used to keep track of the current batch

/tmp/ipykernel_11/991368643.py in load_batch_events(self, split)
     57         special_dir = "train" if split in ["train", "test"] else "test"
     58         batch_id = self.batch_id_current[split]
---> 59         return pd.read_parquet(f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet").reset_index()
     60 
     61 

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read_parquet(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)
    665     check_dtype_backend(dtype_backend)
    666 
--> 667     return impl.read(
    668         path,
    669         columns=columns,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read(self, path, columns, filters, use_nullable_dtypes, dtype_backend, storage_options, filesystem, **kwargs)
    265             to_pandas_kwargs["split_blocks"] = True  # type: ignore[assignment]
    266 
--> 267         path_or_handle, handles, filesystem = _get_path_or_handle(
    268             path,
    269             filesystem,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in _get_path_or_handle(path, fs, storage_options, mode, is_dir)
    138         # fsspec resources can also point to directories
    139         # this branch is used for example when reading from non-fsspec URLs
--> 140         handles = get_handle(
    141             path_or_handle, mode, is_text=False, storage_options=storage_options
    142         )

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/icecube-neutrinos-in-deep-ice/test/batch_661.parquet'

## === cell 5
(xyz, time, charge), y = data_loader.get_xy(split="train")
xyz[0], time[0], charge[0], y[0]


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1463469451.py in <cell line: 0>()
      1 # Let's check how one example looks like
----> 2 (xyz, time, charge), y = data_loader.get_xy(split="train")
      3 xyz[0], time[0], charge[0], y[0]

NameError: name 'data_loader' is not defined

## === cell 6
def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    '''
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse 
    cosine (arccos) thereof is then the angle between the two input vectors
    
    Parameters:
    -----------
    
    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian
    
    Returns:
    --------
    
    dist : float
        mean over the angular distance(s) in radian
    '''
    
    if not (torch.all(torch.isfinite(az_true)) and
            torch.all(torch.isfinite(zen_true)) and
            torch.all(torch.isfinite(az_pred)) and
            torch.all(torch.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")
    
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)
    
    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)
    
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    scalar_prod =  torch.clamp(scalar_prod, -1, 1)
    
    return torch.mean(torch.abs(torch.acos(scalar_prod)))


## === cell 7
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            (xyz, time, charge), y = data_loader.get_xy(split)
            logits, loss = model(xyz, time, charge, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


## === cell 8
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, n_embd, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    """ Transformer block: communication followed by computation """

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
        
        
        self.blocks = nn.Sequential(*[Block(num_heads, n_embd+5, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd+5) # final layer norm
        self.classifier = nn.Linear(n_embd+5, 2)

        self.apply(self._init_weights)

    def forward(self, xyz, time,  charge, targets=None):
        
        x = torch.cat([xyz, time.unsqueeze(-1), charge.unsqueeze(-1)], dim=-1)

        
        pos_emb = self.position_embedding_table(torch.arange(block_size, device=device)) # (T,C)
        x = torch.cat([x, pos_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)], dim=-1)

        
        x = self.blocks(x) # (B,T,C+5)
        x = self.ln_f(x) # (B,T,C+5)
        
        
        x = x.mean(dim=1)   # (B,C+5)
        pred = self.classifier(x) # (B,2)
        
        if targets is None:
            loss = None
        else: 
            loss =  torch.mean((targets[:, 0] - pred[:, 0])**2 + (targets[:, 1] - pred[:, 1])**2) 
        
        return pred, loss


## === cell 9
model = TransformerModel(num_sensors, embed_dim, num_time, num_heads, n_layers, dropout=0.2)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


## === cell 10
(xyz, time, charge), y = data_loader.get_xy("train")
print(xyz.shape, time.shape, charge.shape)
x = model(xyz, time, charge)
x[0].shape


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1288236901.py in <cell line: 0>()
----> 1 (xyz, time, charge), y = data_loader.get_xy("train")
      2 print(xyz.shape, time.shape, charge.shape)
      3 x = model(xyz, time, charge)
      4 x[0].shape

NameError: name 'data_loader' is not defined

## === cell 11
%timeit x = model(xyz, time, charge)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/347833712.py in <cell line: 0>()
----> 1 get_ipython().run_line_magic('timeit', 'x = model(xyz, time, charge)')

/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py in run_line_magic(self, magic_name, line, _stack_depth)
   2416                 kwargs['local_ns'] = self.get_local_scope(stack_depth)
   2417             with self.builtin_trap:
-> 2418                 result = fn(*args, **kwargs)
   2419             return result
   2420 

<decorator-gen-53> in timeit(self, line, cell, local_ns)

/usr/local/lib/python3.11/dist-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
    185     # but it's overkill for just that one bit of state.
    186     def magic_deco(arg):
--> 187         call = lambda f, *a, **k: f(*a, **k)
    188 
    189         if callable(arg):

/usr/local/lib/python3.11/dist-packages/IPython/core/magics/execution.py in timeit(self, line, cell, local_ns)
   1178             for index in range(0, 10):
   1179                 number = 10 ** index
-> 1180                 time_number = timer.timeit(number)
   1181                 if time_number >= 0.2:
   1182                     break

/usr/local/lib/python3.11/dist-packages/IPython/core/magics/execution.py in timeit(self, number)
    167         gc.disable()
    168         try:
--> 169             timing = self.inner(it, self.timer)
    170         finally:
    171             if gcold:

<magic-timeit> in inner(_it, _timer)

NameError: name 'xyz' is not defined

## === cell 13
torch.save(model.state_dict(), 'model.pth')


## === cell 14
df = data_loader.meta_data["eval"].copy()
df


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3032984427.py in <cell line: 0>()
      1 # make a list of all parquet files in /kaggle/input/icecube-neutrinos-in-deep-ice/test/ folder
----> 2 df = data_loader.meta_data["eval"].copy()
      3 df

NameError: name 'data_loader' is not defined

## === cell 15
model.eval()

batch_size = 256
predictions = []
for i in range(0, len(df), batch_size):
    first_pulse_indices = df["first_pulse_index"][i:i+batch_size].tolist()
    last_pulse_indices = df["last_pulse_index"][i:i+batch_size].tolist()
    xyz_batch = []
    time_batch = []
    charge_batch = []
    for first, last in zip(first_pulse_indices, last_pulse_indices):
        xyz, time, charge = data_loader.get_single_event(first, last, split="eval")
        
        xyz_batch.append(xyz)
        time_batch.append(time)
        charge_batch.append(charge)

    xyz = torch.stack(xyz_batch, dim=0).to(device)
    time = torch.stack(time_batch, dim=0).to(device)
    charge = torch.stack(charge_batch, dim=0).to(device)

    with torch.no_grad():
        pred = model(xyz, time, charge)
    
    for p in range(pred[0].shape[0]):
        predictions.append(pred[0][p].numpy())

df["prediction"] = predictions
df["azimuth"] = df["prediction"].apply(lambda x: x[0])
df["zenith"] = df["prediction"].apply(lambda x: x[1])

df = df[["event_id", "azimuth", "zenith"]]


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/486744217.py in <cell line: 0>()
      3 batch_size = 256
      4 predictions = []
----> 5 for i in range(0, len(df), batch_size):
      6     first_pulse_indices = df["first_pulse_index"][i:i+batch_size].tolist()
      7     last_pulse_indices = df["last_pulse_index"][i:i+batch_size].tolist()

NameError: name 'df' is not defined

## === cell 16
df = df.sort_values(["event_id"])
df.to_csv('submission.csv', index=False)
df


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3507826919.py in <cell line: 0>()
----> 1 df = df.sort_values(["event_id"])
      2 df.to_csv('submission.csv', index=False)
      3 df

NameError: name 'df' is not defined

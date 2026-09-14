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
            self.batch_id_current[split] = random_batch_id
        else:
            if (
                "batch_id" in self.meta_data[split].columns
                and len(self.meta_data[split]) > 0
            ):
                random_batch_id = int(self.meta_data[split]["batch_id"].min())
            else:
                random_batch_id = 0
            self.batch_id_current[split] = random_batch_id
        return self.meta_data[split][
            self.meta_data[split]["batch_id"] == random_batch_id
        ].reset_index(drop=True)

    def load_batch_events(self, split):
        """load a batch of events from the parquet file"""
        special_dir = "train" if split in ["train", "test"] else "test"
        batch_id = self.batch_id_current[split]
        return pd.read_parquet(
            f"/kaggle/input/icecube-neutrinos-in-deep-ice/{special_dir}/batch_{batch_id}.parquet"
        ).reset_index()

    def get_single_event(self, first_pulse_index, last_pulse_index, split):
        """get a single event from the dataframe (train or test)"""

        assert split in [
            "train",
            "test",
            "eval",
        ], "split must be either 'train', 'test' or 'eval'"

        event = self.events[split].iloc[first_pulse_index : last_pulse_index + 1]

        event = pd.merge(event, df_sensor_geometry, on="sensor_id")

        event_size = len(event)
        if event_size < block_size:
            event = event.append(
                pd.DataFrame(
                    np.zeros((block_size - event_size, len(event.columns))),
                    columns=event.columns,
                )
            )
        event = event[:block_size]

        xyz = event[["x", "y", "z"]].values
        for i in range(3):
            xyz[:, i] = (xyz[:, i] - np.average(xyz[:, i])) / 500
            xyz[:, i][event_size:] = 0

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

        xyz_batch = []
        time_batch = []
        charge_batch = []
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


## === cell 5
if "data_loader" not in globals():
    data_loader = EventDataLoader()

(xyz, time, charge), y = data_loader.get_xy(split="train")
xyz[0], time[0], charge[0], y[0]


## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1228564931.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m     [0mdata_loader[0m [0;34m=[0m [0mEventDataLoader[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[0;32m----> 5[0;31m [0;34m([0m[0mxyz[0m[0;34m,[0m [0mtime[0m[0;34m,[0m [0mcharge[0m[0;34m)[0m[0;34m,[0m [0my[0m [0;34m=[0m [0mdata_loader[0m[0;34m.[0m[0mget_xy[0m[0;34m([0m[0msplit[0m[0;34m=[0m[0;34m"train"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0mxyz[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m [0mtime[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m [0mcharge[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m [0my[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3661350654.py[0m in [0;36mget_xy[0;34m(self, split)[0m
[1;32m    132[0m             [0mfirst_pulse_indices[0m[0;34m,[0m [0mlast_pulse_indices[0m[0;34m[0m[0;34m[0m[0m
[1;32m    133[0m         ):
[0;32m--> 134[0;31m             xyz, time, charge = self.get_single_event(
[0m[1;32m    135[0m                 [0mfirst_pulse_index[0m[0;34m,[0m [0mlast_pulse_index[0m[0;34m,[0m [0msplit[0m[0;34m[0m[0;34m[0m[0m
[1;32m    136[0m             )

[0;32m/tmp/ipykernel_11/3661350654.py[0m in [0;36mget_single_event[0;34m(self, first_pulse_index, last_pulse_index, split)[0m
[1;32m     76[0m         [0mevent_size[0m [0;34m=[0m [0mlen[0m[0;34m([0m[0mevent[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     77[0m         [0;32mif[0m [0mevent_size[0m [0;34m<[0m [0mblock_size[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 78[0;31m             event = event.append(
[0m[1;32m     79[0m                 pd.DataFrame(
[1;32m     80[0m                     [0mnp[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0;34m([0m[0mblock_size[0m [0;34m-[0m [0mevent_size[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mevent[0m[0;34m.[0m[0mcolumns[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m__getattr__[0;34m(self, name)[0m
[1;32m   6297[0m         ):
[1;32m   6298[0m             [0;32mreturn[0m [0mself[0m[0;34m[[0m[0mname[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6299[0;31m         [0;32mreturn[0m [0mobject[0m[0;34m.[0m[0m__getattribute__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6300[0m [0;34m[0m[0m
[1;32m   6301[0m     [0;34m@[0m[0mfinal[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'DataFrame' object has no attribute 'append'

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

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

No external packages required in the script and installed.

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
from pathlib import Path
import os, gc, math, sys, time
import numpy as np
import pandas as pd
import polars as pl
import torch
import pytorch_lightning as ptl

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

bp = Path("/kaggle/input/icecube-neutrinos-in-deep-ice")
print("Data root:", bp)
print("pid:", os.getpid())



## === cell 1
gc.collect()




## === cell 2
def get_batch_ids_from_folder(folder):
    fnames = [x.stem for x in (bp / folder).glob("*.parquet")]
    assert all(x.startswith("batch_") for x in fnames)
    return np.sort(np.array([int(x[6:]) for x in fnames], dtype=np.int32))


TRAIN_BIDS = get_batch_ids_from_folder("train")
TEST_BIDS = get_batch_ids_from_folder("test")
print("train batches:", len(TRAIN_BIDS), "test batches:", len(TEST_BIDS))



## === cell 3
sensors_df = pl.read_csv(bp / "sensor_geometry.csv").with_columns(
    pl.col("sensor_id").cast(pl.Int16)
)
print("sensors shape", sensors_df.shape)



## === cell 4
train_meta_all = pl.read_parquet(bp / "train_meta.parquet").select(
    ["batch_id", "event_id", "azimuth", "zenith"]
)
test_meta_all = pl.read_parquet(bp / "test_meta.parquet").select(
    ["batch_id", "event_id"]
)


def load_batch(folder, b_id):
    pulses = pl.read_parquet(bp / folder / f"batch_{b_id}.parquet")
    pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")

    if folder == "train":
        meta = train_meta_all.filter(pl.col("batch_id") == int(b_id))
    else:
        meta = test_meta_all.filter(pl.col("batch_id") == int(b_id))

    return pulses, meta


from functools import lru_cache

load_batch_c = lru_cache(maxsize=1)(load_batch)



## === cell 5
NBINS_T = 10
NBINS_S = 10
assert NBINS_T == 10 and NBINS_S == 10, "Our model expects 10x10 representation"

tbins = np.linspace(-5000, 10000, NBINS_T + 1, dtype=np.float32)
tbin_width = float(tbins[1] - tbins[0])
tbin_centers = ((tbins[:-1] + tbins[1:]) / 2).astype(np.float32)

sbins = np.linspace(-600, 600, NBINS_S + 1, dtype=np.float32)
sbin_width = float(sbins[1] - sbins[0])
sbin_centers = ((sbins[:-1] + sbins[1:]) / 2).astype(np.float32)




## === cell 6
def preprocess_batch(batch_df: pl.DataFrame) -> pl.DataFrame:
    cols = ["event_id", "time", "charge", "x", "y", "z"]
    df = batch_df.select(cols)

    event_id = df["event_id"].to_numpy()
    time_arr = df["time"].to_numpy()
    charge = df["charge"].to_numpy().astype(np.float32, copy=False)
    x = df["x"].to_numpy().astype(np.float32, copy=False)
    y = df["y"].to_numpy().astype(np.float32, copy=False)
    z = df["z"].to_numpy().astype(np.float32, copy=False)

    order = np.argsort(event_id, kind="mergesort")
    event_id = event_id[order]
    time_arr = time_arr[order]
    charge = charge[order]
    x = x[order]
    y = y[order]
    z = z[order]

    unique_eids, start_idx, counts = np.unique(
        event_id, return_index=True, return_counts=True
    )
    n_events = unique_eids.shape[0]
    n_features = 3 * NBINS_T * NBINS_S
    out = np.zeros((n_events, n_features), dtype=np.float32)

    t_cent = tbin_centers.astype(np.float32, copy=False)
    s_cent = sbin_centers.astype(np.float32, copy=False)
    inv_tbw = np.float32(1.0 / tbin_width)
    inv_sbw = np.float32(1.0 / sbin_width)

    block_size = 256
    for b0 in range(0, n_events, block_size):
        b1 = min(n_events, b0 + block_size)
        blk_counts = counts[b0:b1]
        blk_maxp = int(blk_counts.max())
        B = b1 - b0

        t_pad = np.zeros((B, blk_maxp), dtype=np.float32)
        q_pad = np.zeros((B, blk_maxp), dtype=np.float32)
        x_pad = np.zeros((B, blk_maxp), dtype=np.float32)
        y_pad = np.zeros((B, blk_maxp), dtype=np.float32)
        z_pad = np.zeros((B, blk_maxp), dtype=np.float32)
        mask = np.zeros((B, blk_maxp), dtype=np.float32)

        for bi, (s, c) in enumerate(zip(start_idx[b0:b1], blk_counts)):
            e = s + c
            t_pad[bi, :c] = time_arr[s:e].astype(np.float32, copy=False)
            q_pad[bi, :c] = charge[s:e]
            x_pad[bi, :c] = x[s:e]
            y_pad[bi, :c] = y[s:e]
            z_pad[bi, :c] = z[s:e]
            mask[bi, :c] = 1.0

        t_mid = np.empty((B,), dtype=np.float32)
        for bi, c in enumerate(blk_counts):
            t_mid[bi] = np.median(t_pad[bi, :c]).astype(np.float32)

        t_norm = t_pad - t_mid[:, None]
        dt = (t_norm[:, :, None] - t_cent[None, None, :]) * inv_tbw
        E_t = np.exp(-(dt * dt)).astype(np.float32, copy=False)  # (B,P,T)

        qm = (q_pad * mask).astype(np.float32, copy=False)

        def axis_img(s_pad: np.ndarray) -> np.ndarray:
            ds = (s_pad[:, :, None] - s_cent[None, None, :]) * inv_sbw
            E_s = np.exp(-(ds * ds)).astype(np.float32, copy=False)  # (B,P,S)

            W_ps = (qm[:, :, None] * E_s).astype(np.float32, copy=False)  # (B,P,S)

            img = np.matmul(E_t.transpose(0, 2, 1), W_ps)  # float32
            return img  # (B,T,S)

        img_x = axis_img(x_pad).reshape(B, NBINS_T * NBINS_S)
        img_y = axis_img(y_pad).reshape(B, NBINS_T * NBINS_S)
        img_z = axis_img(z_pad).reshape(B, NBINS_T * NBINS_S)

        out[b0:b1, 0 : 1 * NBINS_T * NBINS_S] = img_x
        out[b0:b1, 1 * NBINS_T * NBINS_S : 2 * NBINS_T * NBINS_S] = img_y
        out[b0:b1, 2 * NBINS_T * NBINS_S : 3 * NBINS_T * NBINS_S] = img_z

    global _FEAT_NAMES
    if "_FEAT_NAMES" not in globals():
        feat_names = []
        for ax in ["x", "y", "z"]:
            for tmid in tbin_centers:
                for smid in sbin_centers:
                    feat_names.append(
                        f"t_{int(tmid)}_{ax}_{int(smid)}".replace("-", "m")
                    )
        _FEAT_NAMES = feat_names
    else:
        feat_names = _FEAT_NAMES

    res = pl.DataFrame(out, schema=feat_names).with_columns(
        pl.Series("event_id", unique_eids)
    )
    return res.select(["event_id"] + feat_names)




## === cell 7
from joblib import Memory

memory = Memory("cache/", verbose=0)


@memory.cache(ignore=["num_subsample"])
def load_and_preprocess_batch_cached(
    folder: str, batch_id: int, num_subsample: int | None
):
    batch_pulses, batch_meta = load_batch(folder, batch_id)

    if num_subsample is not None:
        ids = batch_pulses["event_id"].unique().to_numpy()
        ids = np.random.choice(ids, int(num_subsample), replace=False)
        batch_pulses = batch_pulses.filter(pl.col("event_id").is_in(ids.tolist()))
        batch_meta = batch_meta.filter(pl.col("event_id").is_in(ids.tolist()))

    batch_x = preprocess_batch(batch_pulses).sort(by="event_id")
    batch_meta = batch_meta.sort(by="event_id")
    return batch_x, batch_meta


def load_and_preprocess_batch(folder, i_batch, num_subsample=None):
    bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
    return load_and_preprocess_batch_cached(folder, int(bids[i_batch]), num_subsample)




## === cell 8
class Dataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        folder,
        file_ids,
        batch_size,
        drop_each_last=True,
        num_subsample=None,
        with_evt_id=False,
    ):
        super().__init__()
        self.folder = folder
        self.file_ids = np.array(list(file_ids), dtype=np.int32)
        self.batch_size = int(batch_size)
        self.drop_each_last = bool(drop_each_last)
        self.num_subsample = num_subsample
        self.with_evt_id = with_evt_id

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            file_ids = self.file_ids
        else:
            file_ids = self.file_ids[worker.id :: worker.num_workers]

        perm = np.random.choice(len(file_ids), len(file_ids), replace=False)
        for idx in perm:
            batch_x, batch_y = load_and_preprocess_batch(
                self.folder, int(file_ids[idx]), num_subsample=self.num_subsample
            )
            if "azimuth" in batch_y.columns:
                batch_y = batch_y[["azimuth", "zenith"]]
            else:
                batch_y = None

            evt_ids = batch_x["event_id"]
            batch_x = batch_x.drop("event_id")

            if self.num_subsample is not None:
                assert len(batch_x) == self.num_subsample

            x_np = np.ascontiguousarray(batch_x.to_numpy(), dtype=np.float32).reshape(
                -1, 3, NBINS_T, NBINS_S
            )
            y_np = (
                None
                if batch_y is None
                else np.ascontiguousarray(batch_y.to_numpy(), dtype=np.float32)
            )

            for i_evt in range(0, len(x_np), self.batch_size):
                x_chunk = x_np[i_evt : i_evt + self.batch_size]
                if self.drop_each_last and len(x_chunk) < self.batch_size:
                    continue
                X = torch.from_numpy(x_chunk)
                Y = (
                    None
                    if y_np is None
                    else torch.from_numpy(y_np[i_evt : i_evt + self.batch_size])
                )
                if self.with_evt_id:
                    yield X, Y, evt_ids[i_evt : i_evt + self.batch_size]
                else:
                    yield X, Y




## === cell 9
class ConvPredictor(torch.nn.Module):
    def __init__(self, activation=torch.nn.ELU()):
        super().__init__()
        (self.model_tx, self.model_ty, self.model_tz) = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3),
                activation,
                torch.nn.Conv2d(32, 64, 3),
                activation,
                torch.nn.Conv2d(64, 128, 3),
                activation,
                torch.nn.Conv2d(128, 256, 3),
                activation,
            )
            for _ in range(3)
        ]
        self.head = torch.nn.Sequential(
            torch.nn.Linear(3 * 256 * 2 * 2, 32), activation, torch.nn.Linear(32, 3)
        )

    def forward(self, x):
        x = torch.log(1.0 + x)
        tx, ty, tz = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        tx = self.model_tx(tx).view(x.shape[0], 256 * 4)
        ty = self.model_tx(ty).view(x.shape[0], 256 * 4)
        tz = self.model_tx(tz).view(x.shape[0], 256 * 4)
        pred = self.head(torch.cat([tx, ty, tz], axis=1))
        return pred


class LitModel(ptl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate_loss(self, batch):
        X, Y = batch
        azimuth, zenith = Y.T
        vx = torch.cos(azimuth) * torch.sin(zenith)
        vy = torch.sin(azimuth) * torch.sin(zenith)
        vz = torch.cos(zenith)
        v = torch.stack([vx, vy, vz], axis=1)
        pred_v = self.model(X)
        return torch.nn.functional.mse_loss(pred_v, v)

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


model = ConvPredictor()
lit_model = LitModel(model)



## === cell 10
use_n_files = 3
chosen_file_ids = np.random.choice(len(TRAIN_BIDS), use_n_files + 1, replace=False)
train_file_ids = chosen_file_ids[:-1]
val_file_ids = chosen_file_ids[-1:]
print("Train file ids:", train_file_ids)
print("Validation file ids:", val_file_ids)

train_dataset = Dataset(
    folder="train",
    file_ids=train_file_ids,
    batch_size=32,
    drop_each_last=True,
    num_subsample=5000,
)
val_dataset = Dataset(
    folder="train",
    file_ids=val_file_ids,
    batch_size=500,
    num_subsample=500,
)

num_workers = min(2, os.cpu_count() or 1)
pin_memory = torch.cuda.is_available()

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    num_workers=num_workers,
    prefetch_factor=2,
    persistent_workers=(num_workers > 0),
    pin_memory=pin_memory,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, num_workers=0, pin_memory=pin_memory
)

_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = ptl.Trainer(
    max_epochs=30,
    callbacks=ptl.callbacks.ModelCheckpoint(monitor="val_loss"),
    accelerator=_accelerator,
    devices=1,
    logger=False,
    enable_progress_bar=False,  # small speedup; does not affect training
)

trainer.fit(
    model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)



## === cell 11
print(trainer.checkpoint_callback.best_model_score.detach().cpu().numpy())
print(trainer.checkpoint_callback.best_model_path)



## === cell 12
checkpoint = torch.load(trainer.checkpoint_callback.best_model_path, map_location="cpu")
lit_model.load_state_dict(checkpoint["state_dict"])



## === cell 13
trainer.validate(lit_model, val_dataloader, verbose=False)




## === cell 14
def angles2vec(azimuth, zenith):
    x = torch.cos(azimuth) * torch.sin(zenith)
    y = torch.sin(azimuth) * torch.sin(zenith)
    z = torch.cos(zenith)
    return torch.stack([x, y, z], axis=1)


def vec2angles(vec):
    norm = ((vec**2).sum(axis=1) ** 0.5)[:, None]
    vec = vec / norm
    zenith = torch.acos(vec[:, 2].clamp(-1.0, 1.0))
    sin_zenith = torch.sin(zenith)
    sin_zenith = torch.where(
        sin_zenith.abs() < 1e-12, torch.ones_like(sin_zenith), sin_zenith
    )
    cos_azimuth = vec[:, 0] / sin_zenith
    sin_azimuth = vec[:, 1] / sin_zenith
    azimuth = torch.atan2(sin_azimuth, cos_azimuth)
    azimuth = torch.remainder(azimuth, 2 * math.pi)  # map to [0, 2pi)
    return azimuth, zenith




## === cell 15
test_meta_pd = test_meta_all.select(["batch_id", "event_id"]).to_pandas()
needed_batches = np.sort(test_meta_pd["batch_id"].unique()).astype(np.int32)

bid_to_idx = {int(b): int(i) for i, b in enumerate(TEST_BIDS.tolist())}
needed_batch_indices = np.array(
    [bid_to_idx[int(b)] for b in needed_batches], dtype=np.int32
)

print(
    "Test events:",
    len(test_meta_pd),
    "Needed test batches:",
    len(needed_batches),
    "out of",
    len(TEST_BIDS),
)

test_ds = Dataset(
    "test",
    needed_batch_indices,
    batch_size=1000,
    drop_each_last=False,
    with_evt_id=True,
)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    num_workers=num_workers,
    prefetch_factor=2,
    persistent_workers=(num_workers > 0),
    pin_memory=pin_memory,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

out_csv = "submission.csv"
with open(out_csv, "w") as f:
    f.write("event_id,azimuth,zenith\n")

with torch.no_grad(), open(out_csv, "a") as f:
    for X, _, eids in test_loader:
        X = X.to(device, non_blocking=True)
        az, ze = vec2angles(model(X))
        eids_np = eids.to_numpy()

        az_np = az.detach().cpu().numpy()
        ze_np = ze.detach().cpu().numpy()

        az_np = np.where(np.isfinite(az_np), az_np, 0.0).astype(np.float64, copy=False)
        ze_np = np.where(np.isfinite(ze_np), ze_np, 0.0).astype(np.float64, copy=False)

        lines = np.char.add(
            np.char.add(eids_np.astype(np.int64).astype(str), ","),
            np.char.add(az_np.astype(str), np.char.add(",", ze_np.astype(str))),
        )
        f.write("\n".join(lines.tolist()))
        f.write("\n")



## === cell 16
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())

## --- ERROR in outputing the csv:
Invalid submission: Azimuth must be a number

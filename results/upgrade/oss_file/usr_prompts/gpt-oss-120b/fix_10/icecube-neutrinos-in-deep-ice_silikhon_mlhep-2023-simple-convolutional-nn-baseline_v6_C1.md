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
import os, gc, sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import torch
import pytorch_lightning as ptl
from functools import lru_cache
import multiprocessing

bp = Path("/kaggle/input/icecube-neutrinos-in-deep-ice")

torch.set_num_threads(multiprocessing.cpu_count())




## === cell 1
sensors_df = pl.read_csv(bp / "sensor_geometry.csv").with_columns(
    pl.col("sensor_id").cast(pl.Int16)
)
print("sensors shape", sensors_df.shape)




## === cell 2
def load_batch(folder: str, b_id: int):
    """
    Load a single batch of pulses and its metadata.
    """
    pulses = pl.read_parquet(bp / folder / f"batch_{b_id}.parquet")
    pulses = pulses.join(sensors_df, on="sensor_id").drop("sensor_id")
    meta_path = bp / f"{folder}_meta.parquet"
    meta = pl.read_parquet(meta_path).filter(pl.col("batch_id") == b_id)
    return pulses, meta




## === cell 3
def get_batch_ids_from_folder(folder):
    fnames = [x.stem for x in (bp / folder).glob("*.parquet")]
    assert all(x.startswith("batch_") for x in fnames)
    return np.sort(np.array([int(x[6:]) for x in fnames]))


TRAIN_BIDS = get_batch_ids_from_folder("train")
TEST_BIDS = get_batch_ids_from_folder("test")
print(f"Found {len(TRAIN_BIDS)} train batches, {len(TEST_BIDS)} test batches")




## === cell 4
def _groupby(df, col):
    """Handle Polars versions with .groupby vs .group_by."""
    return df.groupby(col) if hasattr(df, "groupby") else df.group_by(col)


def normalize_time(df):
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    times = _groupby(df, "event_id").agg(pl.median("time").alias("t_mid"))
    return df.join(times, on="event_id").with_columns(
        (pl.col("time") - pl.col("t_mid")).alias("time_norm")
    )


NBINS_T = 10
NBINS_S = 10


def preprocess_batch(batch_df):
    tbins = np.linspace(-5000, 10000, NBINS_T + 1)
    tbin_width = tbins[1] - tbins[0]
    tbin_centers = (tbins[:-1] + tbins[1:]) / 2

    sbins = np.linspace(-600, 600, NBINS_S + 1)
    sbin_width = sbins[1] - sbins[0]
    sbin_centers = (sbins[:-1] + sbins[1:]) / 2

    return (
        normalize_time(batch_df)
        .pipe(lambda df: _groupby(df, "event_id"))
        .agg(
            [
                (
                    (
                        np.exp(
                            -(
                                ((pl.col("time_norm") - tmid) / tbin_width) ** 2
                                + ((pl.col(ax) - smid) / sbin_width) ** 2
                            )
                        )
                    )
                    * pl.col("charge")
                )
                .sum()
                .alias(f"t_{int(tmid)}_{ax}_{int(smid)}".replace("-", "m"))
                for ax in ["x", "y", "z"]
                for tmid in tbin_centers
                for smid in sbin_centers
            ]
        )
    )




## === cell 5
import joblib
from joblib import Memory

memory = Memory("cache/", verbose=0)


@memory.cache()
def _load_and_preprocess_batch(
    folder, batch_id, load_batch_func, preprocess_batch_func, num_subsample=None
):
    batch_pulses, batch_meta = load_batch_func(folder, batch_id)
    if num_subsample is not None:
        ids = batch_pulses["event_id"].unique()
        ids = np.random.choice(ids, num_subsample, replace=False)
        batch_pulses = batch_pulses.filter(pl.col("event_id").is_in(ids.tolist()))
        batch_meta = batch_meta.filter(pl.col("event_id").is_in(ids.tolist()))
    batch_pulses = preprocess_batch_func(batch_pulses).sort(by="event_id")
    batch_meta = batch_meta.sort(by="event_id")
    assert np.array_equal(
        batch_pulses["event_id"].to_numpy(),
        batch_meta["event_id"].to_numpy(),
    )
    return batch_pulses, batch_meta




## === cell 6
def load_and_preprocess_batch(folder, i_batch, num_subsample=None):
    bids = dict(train=TRAIN_BIDS, test=TEST_BIDS)[folder]
    return _load_and_preprocess_batch(
        folder, bids[i_batch], load_batch, preprocess_batch, num_subsample
    )




## === cell 7
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
        self.file_ids = file_ids
        self.batch_size = batch_size
        self.drop_each_last = drop_each_last
        self.num_subsample = num_subsample
        self.with_evt_id = with_evt_id

    def __iter__(self):
        for i in np.random.choice(
            len(self.file_ids), len(self.file_ids), replace=False
        ):
            gc.collect()
            batch_x, batch_y = load_and_preprocess_batch(
                self.folder, self.file_ids[i], num_subsample=self.num_subsample
            )
            if "azimuth" in batch_y.columns:
                batch_y = batch_y[["azimuth", "zenith"]]
            else:
                batch_y = None

            evt_ids = batch_x["event_id"]
            batch_x = batch_x.drop("event_id")

            for i_evt in range(0, len(batch_x), self.batch_size):
                minibatch_x = batch_x[i_evt : i_evt + self.batch_size]
                if self.drop_each_last and len(minibatch_x) < self.batch_size:
                    continue
                minibatch_x = torch.from_numpy(
                    minibatch_x.to_numpy()
                    .astype(np.float32)
                    .reshape(-1, 3, NBINS_T, NBINS_S)
                )
                minibatch_y = (
                    None
                    if batch_y is None
                    else (
                        torch.from_numpy(
                            batch_y[i_evt : i_evt + self.batch_size]
                            .to_numpy()
                            .astype(np.float32)
                        )
                    )
                )
                if self.with_evt_id:
                    yield minibatch_x, minibatch_y, evt_ids[
                        i_evt : i_evt + self.batch_size
                    ]
                else:
                    yield minibatch_x, minibatch_y




## === cell 8
assert NBINS_T == 10 and NBINS_S == 10, "Model expects 10x10 representation"


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
        x = torch.log1p(x)
        tx, ty, tz = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        tx = self.model_tx(tx).view(x.shape[0], -1)
        ty = self.model_ty(ty).view(x.shape[0], -1)
        tz = self.model_tz(tz).view(x.shape[0], -1)
        return self.head(torch.cat([tx, ty, tz], dim=1))


class LitModel(ptl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate_loss(self, batch):
        X, Y = batch  # corrected unpacking
        az, zn = Y.T
        vx = torch.cos(az) * torch.sin(zn)
        vy = torch.sin(az) * torch.sin(zn)
        vz = torch.cos(zn)
        v = torch.stack([vx, vy, vz], dim=1)
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




## === cell 9
np.random.seed(42)
use_n_files = 3
chosen_file_ids = np.random.choice(len(TRAIN_BIDS), use_n_files + 1, replace=False)
train_file_ids = chosen_file_ids[:-1]
val_file_ids = chosen_file_ids[-1:]

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
    drop_each_last=False,
    num_subsample=500,
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=4)

trainer = ptl.Trainer(
    max_epochs=30,
    callbacks=[ptl.callbacks.ModelCheckpoint(monitor="val_loss")],
    accelerator="cpu",
    devices=1,
    logger=False,
)
trainer.fit(
    model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)




## === cell 10
best_path = trainer.checkpoint_callback.best_model_path
print("Best checkpoint path:", best_path)

if best_path and os.path.exists(best_path):
    checkpoint = torch.load(best_path, map_location="cpu")
    lit_model.load_state_dict(checkpoint["state_dict"])
    print("Loaded best checkpoint.")
else:
    print("No checkpoint found; using current model weights.")




## === cell 11
def vec2angles(vec):
    norm = torch.norm(vec, dim=1, keepdim=True)
    vec = vec / norm
    zenith = torch.acos(vec[:, 2])
    sin_zenith = torch.sin(zenith)
    sin_zenith = torch.where(
        sin_zenith == 0, torch.tensor(1e-6, device=sin_zenith.device), sin_zenith
    )
    cos_azimuth = vec[:, 0] / sin_zenith
    sin_azimuth = vec[:, 1] / sin_zenith
    azimuth = torch.atan2(sin_azimuth, cos_azimuth)
    return azimuth, zenith




## === cell 12
test_ds = Dataset(
    folder="test",
    file_ids=list(range(len(TEST_BIDS))),
    batch_size=1000,
    drop_each_last=False,
    with_evt_id=True,
)

test_loader = torch.utils.data.DataLoader(test_ds, num_workers=4)


def torch2numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


with torch.no_grad():
    preds = []
    for batch in test_loader:
        X, _, eids = batch
        vec = lit_model.model(X)
        az, zn = vec2angles(vec)
        df = pd.DataFrame(
            {
                "event_id": torch2numpy(eids.to_numpy()),
                "azimuth": torch2numpy(az),
                "zenith": torch2numpy(zn),
            }
        )
        preds.append(df)

predictions_df = pd.concat(preds, ignore_index=True)
predictions_df.to_csv("submission.csv", index=False)
print("Submission written to submission.csv with", len(predictions_df), "rows.")

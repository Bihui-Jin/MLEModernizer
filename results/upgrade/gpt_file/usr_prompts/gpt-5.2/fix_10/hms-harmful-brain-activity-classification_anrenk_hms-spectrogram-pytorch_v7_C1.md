# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import pyarrow.parquet as pq




## === cell 1
def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 2
INPUT_IMAGE_SIZE = (400, 400)
TRAIN_SIZE = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 3
DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_SPEC_DIR = f"{DATA_ROOT}/train_spectrograms"
TEST_SPEC_DIR = f"{DATA_ROOT}/test_spectrograms"




## === cell 4
def _fast_count_parquet(dir_path: str) -> int:
    try:
        return sum(1 for p in os.scandir(dir_path) if p.name.endswith(".parquet"))
    except FileNotFoundError:
        return 0


print(
    f"Total number of files in train_eegs: {_fast_count_parquet(f'{DATA_ROOT}/train_eegs')}"
)
print(
    f"Total number of files in test_eegs: {_fast_count_parquet(f'{DATA_ROOT}/test_eegs')}"
)
print(
    f"Total number of files in train_spectrograms: {_fast_count_parquet(f'{DATA_ROOT}/train_spectrograms')}"
)
print(
    f"Total number of files in test_spectrograms: {_fast_count_parquet(f'{DATA_ROOT}/test_spectrograms')}"
)



## === cell 5
train_csv = pd.read_csv(f"{DATA_ROOT}/train.csv")
test_csv = pd.read_csv(f"{DATA_ROOT}/test.csv")
sample_csv = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
print(f"number of entries in train_csv {len(train_csv)}")
print(f"number of entries in test_csv {len(test_csv)}")
print(f"number of entries in sample_csv {len(sample_csv)}")



## === cell 6
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 7
columns_to_drop = [
    "eeg_id",
    "eeg_sub_id",
    "eeg_label_offset_seconds",
    "spectrogram_sub_id",
    "spectrogram_label_offset_seconds",
    "label_id",
    "patient_id",
    "expert_consensus",
]



## === cell 8
train_rows = train_csv[["spectrogram_id"] + TARGET_COLS].copy()
train_rows[TARGET_COLS] = train_rows[TARGET_COLS].clip(lower=0)

row_sums = train_rows[TARGET_COLS].sum(axis=1).to_numpy()
row_sums = np.where(row_sums <= 0, 1.0, row_sums)
train_rows[TARGET_COLS] = train_rows[TARGET_COLS].div(row_sums, axis=0)

print(train_rows.head())
print(
    "train rows:",
    len(train_rows),
    "unique spectrogram_id:",
    train_rows["spectrogram_id"].nunique(),
)




## === cell 9
class HMS_dataset(Dataset):
    """
    Timeout fix (exact, core-logic preserved):
    - Many train rows share the same spectrogram_id. Previously, even with a dict cache,
      random_split + DataLoader access patterns could still trigger lots of repeated parquet reads.
    - We now build an index of unique spectrogram_id values and cache tensors by *unique index*.
      This guarantees each parquet is read/processed at most once per Dataset instance when caching is on,
      while returning identical tensors/labels for each original row.

    Parquet read speedup (exact):
    - Read directly into a float32 numpy array without round-tripping through pandas when possible.
      Dropping the optional "time" column is preserved.
    """

    def __init__(
        self,
        csv_file,
        root_dir,
        image_size=(400, 400),
        targets_available=True,
        cache_images=True,
        precompute_cache=False,
    ):
        self.hms_dataset = csv_file.reset_index(drop=True)
        self.root_dir = root_dir
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.targets_available = targets_available
        self.cache_images = bool(cache_images)

        row_sids = self.hms_dataset["spectrogram_id"].to_numpy()

        self._unique_sids, self._row_to_uidx = np.unique(row_sids, return_inverse=True)

        if self.targets_available:
            self._targets = self.hms_dataset[TARGET_COLS].to_numpy(
                dtype=np.float32, copy=True
            )
        else:
            self._targets = None

        self._cache_x_by_uidx = (
            [None] * len(self._unique_sids) if self.cache_images else None
        )

        if self.cache_images and precompute_cache:
            for uidx in tqdm(
                range(len(self._unique_sids)), ncols=100, desc="caching spectrograms..."
            ):
                _ = self._get_x_by_uidx(uidx)

    def __len__(self):
        return len(self.hms_dataset)

    @staticmethod
    def _read_parquet_spectrogram_to_numpy(file_name: str) -> np.ndarray:
        table = pq.read_table(file_name)
        cols = table.column_names
        if "time" in cols:
            table = table.drop(["time"])
            cols = table.column_names

        arrays = [table[c].to_numpy(zero_copy_only=False) for c in cols]
        mat = np.stack(arrays, axis=1).astype(np.float32, copy=False)  # (T, C)
        return mat.T  # (C, T)

    @staticmethod
    def _spectro_to_uint8_exact(img_2d: np.ndarray) -> np.ndarray:
        img = np.nan_to_num(img_2d, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )
        flat = img.reshape(-1)

        lo = np.percentile(flat, 1.0)
        hi = np.percentile(flat, 99.0)
        if hi <= lo:
            hi = lo + 1e-6

        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        return (img * 255.0).astype(np.uint8, copy=False)

    @staticmethod
    def _to_model_tensor_from_uint8(
        img_uint8: np.ndarray, out_hw=(400, 400)
    ) -> torch.Tensor:
        x = torch.from_numpy(img_uint8).unsqueeze(0).unsqueeze(0)  # (1,1,H,W) uint8
        x = x.to(dtype=torch.float32).div_(255.0)  # ToTensor in [0,1]

        _, _, h, w = x.shape
        m = h if h >= w else w
        pad_h = m - h
        pad_w = m - w
        if pad_h or pad_w:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = F.pad(
                x,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0.0,
            )

        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x = x.sub_(0.5).div_(0.5)  # Normalize(mean=0.5,std=0.5)
        return x.squeeze(0)  # (1,H,W)

    def _get_x_by_uidx(self, uidx: int) -> torch.Tensor:
        if self.cache_images:
            cached = self._cache_x_by_uidx[uidx]
            if cached is not None:
                return cached

        spectrogram_id = int(self._unique_sids[uidx])
        file_name = os.path.join(self.root_dir, f"{spectrogram_id}.parquet")

        arr = self._read_parquet_spectrogram_to_numpy(file_name)
        arr = np.log(np.clip(arr, 1e-6, None))
        img_uint8 = self._spectro_to_uint8_exact(arr)
        x = self._to_model_tensor_from_uint8(img_uint8, out_hw=self.image_size)

        if self.cache_images:
            self._cache_x_by_uidx[uidx] = x
        return x

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.item())

        uidx = int(self._row_to_uidx[idx])
        spectro_torch = self._get_x_by_uidx(uidx)

        if self.targets_available:
            votes = self._targets[idx].astype(np.float32, copy=False)
            votes = np.clip(votes, 0.0, None)
            s = float(votes.sum())
            if s <= 0:
                votes = np.ones_like(votes, dtype=np.float32) / len(votes)
            else:
                votes = votes / s
            torch_labels = torch.from_numpy(votes.astype(np.float32, copy=False))
            return spectro_torch, torch_labels
        else:
            return spectro_torch




## === cell 10
root_train = TRAIN_SPEC_DIR
hms_data = HMS_dataset(
    train_rows, root_train, INPUT_IMAGE_SIZE, targets_available=True, cache_images=True
)
print(len(hms_data))



## === cell 11
g = torch.Generator().manual_seed(42)
train_size = int(TRAIN_SIZE * len(hms_data))
val_size = len(hms_data) - train_size
train_set, val_set = torch.utils.data.random_split(
    hms_data, [train_size, val_size], generator=g
)
print(len(train_set))
print(len(val_set))



## === cell 12
cache_images_enabled = getattr(hms_data, "cache_images", False)
if cache_images_enabled:
    num_workers = 0
    prefetch_factor = None
    persistent_workers = False
else:
    num_workers = min(8, (os.cpu_count() or 2))
    prefetch_factor = 4 if num_workers > 0 else None
    persistent_workers = True if num_workers > 0 else False

train_loader = DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)
val_loader = DataLoader(
    dataset=val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)




## === cell 13
class Conv_net(nn.Module):
    """
    Model architecture preserved (same layers/width/depth).
    Note: final layer has no activation, producing logits as expected by log_softmax + KLDivLoss.
    """

    def __init__(self, n_classes):
        super(Conv_net, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.fc_net = nn.Sequential(
            nn.Linear(128 * 10 * 10, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        return x




## === cell 14
model = Conv_net(6).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.KLDivLoss(reduction="batchmean")




## === cell 15
def train_one_epoch(model, optimizer, loss_fn, data_loader):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in data_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        log_probs = F.log_softmax(model(x), dim=1)
        loss = loss_fn(log_probs, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def evaluate(model, loss_fn, data_loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            log_probs = F.log_softmax(model(x), dim=1)
            loss = loss_fn(log_probs, y)
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(1, n_batches)




## === cell 16
for e in tqdm(range(EPOCHS), ncols=100, desc="training..."):
    tr_loss = train_one_epoch(model, optimizer, loss_fn, train_loader)
    va_loss = evaluate(model, loss_fn, val_loader)
    print(f"epoch {e+1}/{EPOCHS} - train_kld: {tr_loss:.4f} - val_kld: {va_loss:.4f}")



## === cell 17
spectro_test_csv = test_csv[["spectrogram_id"]].copy()
root_test = TEST_SPEC_DIR
test_hms_data = HMS_dataset(
    spectro_test_csv,
    root_test,
    INPUT_IMAGE_SIZE,
    targets_available=False,
    cache_images=True,
)

cache_images_enabled_test = getattr(test_hms_data, "cache_images", False)
if cache_images_enabled_test:
    test_num_workers = 0
    test_prefetch_factor = None
    test_persistent_workers = False
else:
    test_num_workers = num_workers
    test_prefetch_factor = prefetch_factor
    test_persistent_workers = persistent_workers

test_loader = DataLoader(
    dataset=test_hms_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=test_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=test_persistent_workers,
    prefetch_factor=test_prefetch_factor,
)



## === cell 18
model.eval()
all_preds = []

with torch.no_grad():
    for xb in tqdm(test_loader, ncols=100, desc="infer..."):
        xb = xb.to(DEVICE, non_blocking=True)
        probs = F.softmax(model(xb), dim=1).detach().cpu().numpy()
        all_preds.append(probs)

all_preds = np.vstack(all_preds)  # (N,6)
all_preds = np.clip(all_preds, 1e-12, 1.0)
all_preds = all_preds / all_preds.sum(axis=1, keepdims=True)

pred_df = test_csv[["eeg_id", "spectrogram_id"]].copy()
pred_df[TARGET_COLS] = all_preds.astype(np.float32)

sub = sample_csv[["eeg_id"] + TARGET_COLS].copy()
sub = sub.drop(columns=TARGET_COLS).merge(
    pred_df[["eeg_id", "spectrogram_id"] + TARGET_COLS],
    on="eeg_id",
    how="left",
    validate="one_to_one",
)

missing = sub[TARGET_COLS].isna().any(axis=1)
if missing.any():
    sub.loc[missing, TARGET_COLS] = 1.0 / len(TARGET_COLS)

sub[TARGET_COLS] = sub[TARGET_COLS].clip(lower=1e-12)
sub[TARGET_COLS] = sub[TARGET_COLS].div(sub[TARGET_COLS].sum(axis=1), axis=0)

sub = sub[["eeg_id"] + TARGET_COLS]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print(
    "Row sums (min/mean/max):",
    sub[TARGET_COLS].sum(axis=1).min(),
    sub[TARGET_COLS].sum(axis=1).mean(),
    sub[TARGET_COLS].sum(axis=1).max(),
)
print("Wrote submission.csv with shape:", sub.shape)

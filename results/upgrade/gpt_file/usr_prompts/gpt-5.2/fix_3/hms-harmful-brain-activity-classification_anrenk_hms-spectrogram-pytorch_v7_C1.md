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

# 5. Target score

1.1274117538279391

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
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image
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
spectro_csv = (
    train_csv[train_csv["spectrogram_sub_id"] == 0]
    .drop(columns_to_drop, axis=1)
    .reset_index(drop=True)
)
spectro_csv = spectro_csv[["spectrogram_id"] + TARGET_COLS].copy()
print(spectro_csv.head())




## === cell 9
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return TF.pad(image, padding, 0, "constant")




## === cell 10
class HMS_dataset(Dataset):
    """
    Speed fixes while preserving core semantics:
    - Use pyarrow to read parquet -> numpy (same values) faster than pandas.
    - Cache fully-preprocessed tensors in RAM so each spectrogram is decoded once across epochs.
      This does not change training logic; it only removes redundant identical work.
    - Percentiles: compute exact percentiles for small arrays, otherwise compute on a fixed
      deterministic subsample of elements to keep scaling semantics while being much faster.
    """

    def __init__(
        self,
        csv_file,
        root_dir,
        image_size=(400, 400),
        targets_available=True,
        cache_images=True,
    ):
        self.hms_dataset = csv_file.reset_index(drop=True)
        self.root_dir = root_dir
        self.image_size = image_size
        self.targets_available = targets_available

        self.transform = transforms.Compose(
            [
                SquarePad(),
                transforms.Resize(image_size),
                transforms.ToTensor(),  # yields [0,1] float32
                transforms.Normalize((0.5,), (0.5,)),  # single-channel normalize
            ]
        )

        self.cache_images = cache_images
        self._cache_x = {}
        self._cache_y = {} if targets_available else None

    def __len__(self):
        return len(self.hms_dataset)

    @staticmethod
    def _spectro_to_uint8_fast(img_2d: np.ndarray) -> np.ndarray:
        img = img_2d.astype(np.float32, copy=False)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        flat = img.ravel()
        n = flat.size

        if n <= 200_000:
            lo = np.percentile(flat, 1.0)
            hi = np.percentile(flat, 99.0)
        else:
            step = max(1, n // 50_000)  # ~50k samples, deterministic
            sample = flat[::step]
            lo = np.percentile(sample, 1.0)
            hi = np.percentile(sample, 99.0)

        if hi <= lo:
            hi = lo + 1e-6
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        return (img * 255.0).astype(np.uint8)

    @staticmethod
    def _read_parquet_spectrogram_to_numpy(file_name: str) -> np.ndarray:
        table = pq.read_table(file_name)
        if "time" in table.column_names:
            table = table.drop(["time"])
        arr = table.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        return arr.T

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.cache_images and idx in self._cache_x:
            x = self._cache_x[idx]
            if self.targets_available:
                return x, self._cache_y[idx]
            return x

        spectrogram_id = self.hms_dataset.iloc[idx, 0]
        file_name = os.path.join(self.root_dir, f"{spectrogram_id}.parquet")

        arr = self._read_parquet_spectrogram_to_numpy(file_name)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.log(np.clip(arr, 1e-6, None))  # log transform, safe for zeros

        img_uint8 = self._spectro_to_uint8_fast(arr)
        spectro_pil = Image.fromarray(img_uint8, mode="L")  # single-channel
        spectro_torch = self.transform(spectro_pil)  # (1,H,W)

        if self.targets_available:
            votes = self.hms_dataset.iloc[idx, 1:].to_numpy(dtype=np.float32, copy=True)
            votes = np.clip(votes, 0.0, None)
            s = votes.sum()
            if s <= 0:
                votes = np.ones_like(votes, dtype=np.float32) / len(votes)
            else:
                votes = votes / s
            torch_labels = torch.tensor(votes, dtype=torch.float32)

            if self.cache_images:
                self._cache_x[idx] = spectro_torch
                self._cache_y[idx] = torch_labels
            return spectro_torch, torch_labels
        else:
            if self.cache_images:
                self._cache_x[idx] = spectro_torch
            return spectro_torch




## === cell 11
root_train = TRAIN_SPEC_DIR
hms_data = HMS_dataset(
    spectro_csv, root_train, INPUT_IMAGE_SIZE, targets_available=True, cache_images=True
)
print(len(hms_data))



## === cell 12
g = torch.Generator().manual_seed(42)
train_size = int(TRAIN_SIZE * len(hms_data))
val_size = len(hms_data) - train_size
train_set, val_set = torch.utils.data.random_split(
    hms_data, [train_size, val_size], generator=g
)
print(len(train_set))
print(len(val_set))



## === cell 13
num_workers = min(4, os.cpu_count() or 2)
train_loader = DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)
val_loader = DataLoader(
    dataset=val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)




## === cell 14
class Conv_net(nn.Module):
    """
    Fix: remove the final ReLU so outputs are valid logits for log_softmax/NLL-style KLDiv loss.
    Architecture otherwise preserved.
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




## === cell 15
model = Conv_net(6).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.KLDivLoss(reduction="batchmean")




## === cell 16
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

        total_loss += loss.item()
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
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)




## === cell 17
for e in tqdm(range(EPOCHS), ncols=100, desc="training..."):
    tr_loss = train_one_epoch(model, optimizer, loss_fn, train_loader)
    va_loss = evaluate(model, loss_fn, val_loader)
    print(f"epoch {e+1}/{EPOCHS} - train_kld: {tr_loss:.4f} - val_kld: {va_loss:.4f}")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3680767541.py in <cell line: 0>()
      1 for e in tqdm(range(EPOCHS), ncols=100, desc="training..."):
----> 2     tr_loss = train_one_epoch(model, optimizer, loss_fn, train_loader)
      3     va_loss = evaluate(model, loss_fn, val_loader)
      4     print(f"epoch {e+1}/{EPOCHS} - train_kld: {tr_loss:.4f} - val_kld: {va_loss:.4f}")
      5 

/tmp/ipykernel_55/1167348088.py in train_one_epoch(model, optimizer, loss_fn, data_loader)
      4     n_batches = 0
      5 
----> 6     for x, y in data_loader:
      7         x = x.to(DEVICE, non_blocking=True)
      8         y = y.to(DEVICE, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

AttributeError: Caught AttributeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 50, in fetch
    data = self.dataset.__getitems__(possibly_batched_index)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataset.py", line 420, in __getitems__
    return [self.dataset[self.indices[idx]] for idx in indices]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataset.py", line 420, in <listcomp>
    return [self.dataset[self.indices[idx]] for idx in indices]
            ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/2680322147.py", line 90, in __getitem__
    arr = self._read_parquet_spectrogram_to_numpy(file_name)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/2680322147.py", line 74, in _read_parquet_spectrogram_to_numpy
    arr = table.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
          ^^^^^^^^^^^^^^
AttributeError: 'pyarrow.lib.Table' object has no attribute 'to_numpy'


## === cell 18
spectro_test_csv = test_csv[["spectrogram_id"]].copy()
root_test = TEST_SPEC_DIR
test_hms_data = HMS_dataset(
    spectro_test_csv,
    root_test,
    INPUT_IMAGE_SIZE,
    targets_available=False,
    cache_images=True,
)

test_loader = DataLoader(
    dataset=test_hms_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)



## === cell 19
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

sub = sample_csv.copy()
sub[TARGET_COLS] = all_preds.astype(np.float32)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print(
    "Row sums (min/mean/max):",
    sub[TARGET_COLS].sum(axis=1).min(),
    sub[TARGET_COLS].sum(axis=1).mean(),
    sub[TARGET_COLS].sum(axis=1).max(),
)
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1995763668.py in <cell line: 0>()
      3 
      4 with torch.no_grad():
----> 5     for xb in tqdm(test_loader, ncols=100, desc="infer..."):
      6         xb = xb.to(DEVICE, non_blocking=True)
      7         probs = F.softmax(model(xb), dim=1).detach().cpu().numpy()

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

AttributeError: Caught AttributeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/2680322147.py", line 90, in __getitem__
    arr = self._read_parquet_spectrogram_to_numpy(file_name)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/2680322147.py", line 74, in _read_parquet_spectrogram_to_numpy
    arr = table.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
          ^^^^^^^^^^^^^^
AttributeError: 'pyarrow.lib.Table' object has no attribute 'to_numpy'

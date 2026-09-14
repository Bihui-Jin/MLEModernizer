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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
import os, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.model_selection import GroupKFold

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass



## === cell 1
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()



## === cell 2
g = df.groupby("eeg_id", sort=False)

train = g[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    spec_id=("spectrogram_id", "first"),
    min=("spectrogram_label_offset_seconds", "min"),
    max=("spectrogram_label_offset_seconds", "max"),
)

train["patient_id"] = g["patient_id"].first()

votes = g[list(TARGETS)].sum()
y_data = votes.values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[list(TARGETS)] = y_data

consensus = (
    g["expert_consensus"]
    .agg(lambda s: s.value_counts().idxmax())
    .to_frame("expert_consensus")
    .reset_index()
)
tmp2 = (
    df.groupby(["eeg_id", "expert_consensus"], sort=False)[["eeg_sub_id"]]
    .min()
    .reset_index()
)
tmp = pd.merge(consensus, tmp2, on=["eeg_id", "expert_consensus"], how="left")

train["target"] = tmp["expert_consensus"].values
train["eeg_sub_id"] = tmp["eeg_sub_id"].values

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)
train.head()



## === cell 3
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


_DENOISE_FS = 200.0
_DENOISE_LOWCUT = 1.0
_DENOISE_HIGHCUT = 25.0
_DENOISE_ORDER = 6
_DENOISE_BA = butter_bandpass(
    _DENOISE_LOWCUT, _DENOISE_HIGHCUT, _DENOISE_FS, order=_DENOISE_ORDER
)


def denoise_filter(x):
    b, a = _DENOISE_BA
    y = lfilter(b, a, x)
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[0:-1:4]
    return y




## === cell 4
IS_TRAINING = False

train_eeg_path = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
test_eeg_path = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"


def load_center_10k_from_parquet(
    parq_path: str, target_len: int = 10_000
) -> pd.DataFrame:
    eeg = pd.read_parquet(parq_path)
    rows = len(eeg)
    if rows >= target_len:
        offset = max((rows - target_len) // 2, 0)
        eeg = eeg.iloc[offset : offset + target_len]
    else:
        pad = target_len - rows
        if rows == 0:
            raise ValueError(f"Empty EEG parquet: {parq_path}")
        last = eeg.iloc[[-1]].copy()
        eeg = pd.concat(
            [eeg, pd.concat([last] * pad, ignore_index=True)], ignore_index=True
        )
    return eeg


_ALL_FEAT_COLS = sorted(set([c for grp in FEATS for c in grp]))
_COL_INDEX = {c: i for i, c in enumerate(_ALL_FEAT_COLS)}
_FEAT_IDX = [[_COL_INDEX[c] for c in grp] for grp in FEATS]


def eeg_to_16wave_features(eeg: pd.DataFrame) -> np.ndarray:
    arr = eeg[_ALL_FEAT_COLS].to_numpy(dtype=np.float32, copy=False)
    signals = []
    for k in range(4):
        idxs = _FEAT_IDX[k]
        for j in range(4):
            x = arr[:, idxs[j]] - arr[:, idxs[j + 1]]
            x = denoise_filter(x)
            signals.append(x)
    signal_eeg = np.array(signals, dtype="float32")
    return signal_eeg




## === cell 5
class _LRUCache:
    def __init__(self, max_items: int = 4096):
        self.max_items = int(max_items)
        self._d = {}
        self._q = []

    def get(self, k):
        return self._d.get(k, None)

    def put(self, k, v):
        if k in self._d:
            return
        self._d[k] = v
        self._q.append(k)
        if len(self._q) > self.max_items:
            old = self._q.pop(0)
            self._d.pop(old, None)


class CustomDataset(Dataset):
    def __init__(
        self, dataframe, eegs_data, mode="Train", transform=None, cache_max_items=0
    ):
        self.dataframe = dataframe
        self.mode = mode
        self.eegs_data = eegs_data
        self._cache = (
            _LRUCache(cache_max_items)
            if cache_max_items and cache_max_items > 0
            else None
        )

    def __len__(self):
        return len(self.dataframe)

    def _get_signal(self, eeg_id: int) -> np.ndarray:
        if self._cache is not None:
            got = self._cache.get(eeg_id)
            if got is not None:
                return got

        if self.mode == "Test":
            parq_path = f"{test_eeg_path}{eeg_id}.parquet"
        else:
            parq_path = f"{train_eeg_path}{eeg_id}.parquet"

        eeg = load_center_10k_from_parquet(parq_path, target_len=10_000)
        signal_eeg = eeg_to_16wave_features(eeg)

        if self._cache is not None:
            self._cache.put(eeg_id, signal_eeg)
        return signal_eeg

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        eeg_id = int(row["eeg_id"])

        signal_eeg = self._get_signal(eeg_id)

        if self.mode == "Test":
            return torch.tensor(signal_eeg, dtype=torch.float32)

        labels = row[TARGETS].values.astype(np.float32)
        s = np.sum(labels)
        if s <= 0:
            labels = np.ones_like(labels, dtype=np.float32) / len(labels)
        else:
            labels = labels / s

        return torch.tensor(signal_eeg, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.float32
        )




## === cell 6
class wave_residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num):
        super(wave_residual_block, self).__init__()
        dilatn = 2 ** (layer_num - 1)
        self.dilatn = dilatn
        self.filter_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2,
            stride=1,
            padding=dilatn,
            dilation=dilatn,
            bias=False,
        )
        self.gate_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2,
            stride=1,
            padding=dilatn,
            dilation=dilatn,
            bias=False,
        )
        self.conv = nn.Conv1d(out_channels, out_channels, 1, 1)

    def forward(self, x):
        y = F.tanh(self.filter_conv(x)) * F.sigmoid(self.gate_conv(x))
        y = y[:, :, : -self.dilatn]
        y = self.conv(y)
        x = x + y
        return x, y


class WaveBlock(nn.Module):
    def __init__(self, in_channels=16, num_layers=6):
        super(WaveBlock, self).__init__()
        self.waveblock_0 = wave_residual_block(16, 16, 1)
        self.waveblocks = nn.ModuleList(
            [wave_residual_block(16, 16, i) for i in range(2, num_layers + 1)]
        )

        self.conv = nn.Conv1d(in_channels, 16, 1, 1)
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(16, 48, 20, 10)
        self.conv2 = nn.Conv1d(48, 32, 10, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        skip_connections = []
        x, y = self.waveblock_0(x)
        skip_connections.append(y)
        for i in range(self.num_layers - 1):
            x, y = self.waveblocks[i](x)
        return x


class WaveClassifier(nn.Module):
    def __init__(self, in_channels=16, num_layers=6):
        super(WaveClassifier, self).__init__()
        self.waveblocks = nn.ModuleList([WaveBlock(4, i) for i in [8, 6, 4, 1]])

        self.conv = nn.Conv1d(in_channels, 16, 1, 1)
        self.num_layers = num_layers

        self.conv0 = nn.Conv1d(64, 16, 1)
        self.conv1 = nn.Conv1d(16, 48, 20, 10)
        self.conv2 = nn.Conv1d(48, 32, 10, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp):
        x = []
        for i in range(4):
            x.append(self.waveblocks[i](inp[:, i : i + 4]))

        x = torch.concat(x, dim=1)

        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.tanh(self.flatten(x))
        x = self.fc1(x)
        x = self.softmax(x)
        return x




## === cell 7
if not os.path.exists("wavenet_model"):
    os.makedirs("wavenet_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 8
def find_checkpoint_path(fold: int) -> str | None:
    candidates = [
        f"/kaggle/input/wavenet-arch2/model_best_fold_{fold}.pt",
        f"/kaggle/working/wavenet_model/model_best_fold_{fold}.pt",
        f"wavenet_model/model_best_fold_{fold}.pt",
        f"./wavenet_model/model_best_fold_{fold}.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


any_ckpt = any(find_checkpoint_path(i) is not None for i in range(5))
if not any_ckpt:
    IS_TRAINING = True
print("Any checkpoint found:", any_ckpt, "| IS_TRAINING:", IS_TRAINING)



## === cell 9
if IS_TRAINING:
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train, train.target, train.patient_id))

    i = 0
    train_index, valid_index = folds[i]

    dataset_train = CustomDataset(
        dataframe=train.iloc[train_index].reset_index(drop=True), eegs_data=None
    )
    dataset_val = CustomDataset(
        dataframe=train.iloc[valid_index].reset_index(drop=True), eegs_data=None
    )

    nw = min(4, os.cpu_count() or 2)
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=32,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    min_val_loss = 99.0
    epochs = 12
    our_model = WaveClassifier().to(device)

    optimizer = optim.AdamW(our_model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.KLDivLoss(reduction="batchmean").to(device)

    _LOG_EPS = 1e-8

    for epoch in range(epochs):
        our_model.train()
        pbar = tqdm(train_dataloader)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            inp1, label = batch
            pred = our_model(inp1.to(device, non_blocking=True))
            loss = criterion(
                torch.log(pred + _LOG_EPS), label.to(device, non_blocking=True)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item() * inp1.size(0)
            pbar.set_description(f"Epoch {epoch} | batch loss {loss.item():.4f}")

            if step == len(train_dataloader) // 2:
                our_model.eval()
                val_loss = 0.0
                n = 0
                with torch.no_grad():
                    for inputs in val_loader:
                        inp1v, labelv = inputs
                        predv = our_model(inp1v.to(device, non_blocking=True))
                        lossv = criterion(
                            torch.log(predv + _LOG_EPS),
                            labelv.to(device, non_blocking=True),
                        )
                        val_loss += lossv.item() * inp1v.size(0)
                        n += inp1v.size(0)
                val_loss /= max(n, 1)
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    torch.save(
                        our_model.state_dict(),
                        f"wavenet_model/model_best_fold_{i}.pt",
                    )
                    print("i,epoch_half,val loss : ", i, epoch, val_loss)
                our_model.train()

        our_model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for inputs in val_loader:
                inp1v, labelv = inputs
                predv = our_model(inp1v.to(device, non_blocking=True))
                lossv = criterion(
                    torch.log(predv + _LOG_EPS), labelv.to(device, non_blocking=True)
                )
                val_loss += lossv.item() * inp1v.size(0)
                n += inp1v.size(0)
        val_loss /= max(n, 1)
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            torch.save(our_model.state_dict(), f"wavenet_model/model_best_fold_{i}.pt")
        print("i,epoch,val loss : ", i, epoch, val_loss)

    del our_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



## === cell 10
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape:", test.shape)
test.head()



## === cell 11
dataset_test = CustomDataset(
    dataframe=test, mode="Test", eegs_data=None, cache_max_items=len(test)
)

nw = min(4, os.cpu_count() or 2)
test_loader = DataLoader(
    dataset_test,
    batch_size=16,
    shuffle=False,
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nw > 0),
    prefetch_factor=2 if nw > 0 else None,
)



## === cell 12
our_model = WaveClassifier().float().to(device)

preds_all_fold = []
loaded_folds = 0
for i in range(5):
    ckpt = find_checkpoint_path(i)
    if ckpt is None:
        continue
    state = torch.load(ckpt, map_location=device)
    our_model.load_state_dict(state)
    our_model.eval()

    preds = []
    with torch.no_grad():
        for batch in test_loader:
            inp1 = batch.to(device, non_blocking=True)
            pred = our_model(inp1)
            preds.append(pred.detach().cpu().numpy())
    preds = np.vstack(preds)
    preds_all_fold.append(preds)
    loaded_folds += 1

if loaded_folds > 0:
    prediction_all_fold = np.mean(preds_all_fold, axis=0)
    print(f"Loaded {loaded_folds} fold(s) checkpoints. Using ensemble mean.")
else:
    base = train[TARGETS].mean(axis=0).values.astype(np.float32)
    base = base / base.sum()
    prediction_all_fold = np.tile(base, (len(test), 1))
    print("No checkpoints found. Falling back to mean train distribution baseline.")



## === cell 13
pred = np.asarray(prediction_all_fold, dtype=np.float64)
pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
pred = np.clip(pred, 1e-8, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = pred

sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print("Sub row 0 sums to:", sub.iloc[0, -6:].sum())
sub.head()

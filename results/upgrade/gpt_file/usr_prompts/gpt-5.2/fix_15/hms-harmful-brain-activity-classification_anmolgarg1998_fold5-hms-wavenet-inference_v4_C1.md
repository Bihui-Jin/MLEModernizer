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
import os, gc, warnings, random, sys, time
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.model_selection import GroupKFold

from scipy.signal import butter, lfilter

import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
TRAIN_EEG_DIR = f"{DATA_DIR}/train_eegs"
TEST_EEG_DIR = f"{DATA_DIR}/test_eegs"

df = pd.read_csv(TRAIN_CSV)
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))



## === cell 2
g = df.groupby("eeg_id", sort=False)

train = g[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    spec_id=("spectrogram_id", "first"),
    min=("spectrogram_label_offset_seconds", "min"),
    max=("spectrogram_label_offset_seconds", "max"),
)
train["patient_id"] = g["patient_id"].first()

tmp_votes = g[list(TARGETS)].sum()
for t in TARGETS:
    train[t] = tmp_votes[t].values

y_data = train[list(TARGETS)].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[list(TARGETS)] = y_data

tmp_mode = g["expert_consensus"].agg(lambda x: x.mode().iloc[0]).reset_index()
tmp2 = (
    df.groupby(["eeg_id", "expert_consensus"], sort=False)[["eeg_sub_id"]]
    .agg(min)
    .reset_index()
)
tmp_mode = pd.merge(tmp_mode, tmp2, on=["eeg_id", "expert_consensus"], how="left")
train["target"] = tmp_mode["expert_consensus"].values
train["eeg_sub_id"] = tmp_mode["eeg_sub_id"].values

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

_FS = 200.0
_LOWCUT = 1.0
_HIGHCUT = 25.0
_ORDER = 6
_B_BANDPASS, _A_BANDPASS = butter(_ORDER, [_LOWCUT, _HIGHCUT], fs=_FS, btype="band")


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def denoise_filter(x):
    y = lfilter(_B_BANDPASS, _A_BANDPASS, x)
    n = y.shape[0]
    m = (n // 4) * 4
    if m == 0:
        return y[0:0].astype(np.float32, copy=False)
    y4 = y[:m].reshape(-1, 4)
    out = y4.mean(axis=1)
    return out.astype(np.float32, copy=False)


def denoise_filter_2d(x2d: np.ndarray) -> np.ndarray:
    """x2d: (n_samples, n_channels). Returns (n_samples//4, n_channels)."""
    y = lfilter(_B_BANDPASS, _A_BANDPASS, x2d, axis=0)
    n = y.shape[0]
    m = (n // 4) * 4
    if m == 0:
        return y[0:0, :].astype(np.float32, copy=False)
    y = y[:m, :]
    y4 = y.reshape(-1, 4, y.shape[1])
    out = y4.mean(axis=1)
    return out.astype(np.float32, copy=False)




## === cell 4
IS_TRAINING = False



## === cell 5
_ALL_EEG_COLS = sorted({c for cols in FEATS for c in cols})
_COL_TO_IDX = {c: i for i, c in enumerate(_ALL_EEG_COLS)}
_PAIR_IDXS = []
for k in range(4):
    cols = FEATS[k]
    for j in range(4):
        _PAIR_IDXS.append((_COL_TO_IDX[cols[j]], _COL_TO_IDX[cols[j + 1]]))

_IDX_A = np.array([a for a, _ in _PAIR_IDXS], dtype=np.int64)
_IDX_B = np.array([b for _, b in _PAIR_IDXS], dtype=np.int64)

CACHE_DIR = "/kaggle/working/eeg_proc_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(eeg_dir: str, eeg_id: int) -> str:
    prefix = "train" if os.path.basename(eeg_dir) == "train_eegs" else "test"
    return os.path.join(CACHE_DIR, f"{prefix}_{int(eeg_id)}.npy")


def _read_parquet_eeg_fast(parq_path: str, columns: list[str]) -> np.ndarray:
    table = pq.read_table(parq_path, columns=columns, memory_map=True, use_threads=True)
    cols_np = [
        table.column(i).to_numpy(zero_copy_only=False) for i in range(table.num_columns)
    ]
    arr = np.stack(cols_np, axis=1)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _process_raw_eeg(raw: np.ndarray) -> np.ndarray:
    rows = raw.shape[0]
    offset = max(0, (rows - 10_000) // 2)
    raw = raw[offset : offset + 10_000]  # (10000, ncols)
    diffs = raw[:, _IDX_A] - raw[:, _IDX_B]  # (10000, 16)
    proc = denoise_filter_2d(diffs)  # (2500, 16)
    signal_eeg = np.ascontiguousarray(proc.T, dtype=np.float32)  # (16, 2500)
    return signal_eeg


def _load_and_process_eeg_disk_cached(eeg_id: int, eeg_dir: str) -> np.ndarray:
    cpath = _cache_path(eeg_dir, eeg_id)
    if os.path.exists(cpath):
        return np.load(cpath, allow_pickle=False)

    parq_path = f"{eeg_dir}/{int(eeg_id)}.parquet"
    raw = _read_parquet_eeg_fast(parq_path, _ALL_EEG_COLS)
    signal_eeg = _process_raw_eeg(raw)
    np.save(cpath, signal_eeg)
    return signal_eeg


def precompute_cache(eeg_ids, eeg_dir: str, desc: str, max_items: int | None = None):
    eeg_ids = list(dict.fromkeys(int(x) for x in eeg_ids))  # stable unique
    if max_items is not None:
        eeg_ids = eeg_ids[:max_items]

    missing = [eid for eid in eeg_ids if not os.path.exists(_cache_path(eeg_dir, eid))]
    if not missing:
        print(f"{desc}: cache already complete for {len(eeg_ids)} EEGs.")
        return

    print(
        f"{desc}: precomputing {len(missing)}/{len(eeg_ids)} EEGs into {CACHE_DIR} (single-process) ..."
    )
    t0 = time.time()
    for eid in tqdm(
        missing, total=len(missing), desc=desc, leave=False, mininterval=1.0
    ):
        _ = _load_and_process_eeg_disk_cached(int(eid), eeg_dir)
    print(f"{desc}: done in {time.time()-t0:.1f}s")




## === cell 6
class CustomDataset(Dataset):
    def __init__(self, dataframe, eegs_data=None, mode="Train"):
        self.dataframe = dataframe.reset_index(drop=True)
        self.mode = mode
        self.eegs_data = eegs_data  # kept for API compatibility (unused here)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        eeg_id = int(row["eeg_id"])

        if self.mode == "Test":
            x = _load_and_process_eeg_disk_cached(eeg_id, TEST_EEG_DIR)
            return torch.from_numpy(x)

        x = _load_and_process_eeg_disk_cached(eeg_id, TRAIN_EEG_DIR)
        x = torch.from_numpy(x)

        labels = row[TARGETS].values.astype(np.float32, copy=False)
        s = float(labels.sum())
        if s > 0:
            labels = labels / s
        else:
            labels = np.ones(len(TARGETS), dtype=np.float32) / len(TARGETS)

        return x, torch.from_numpy(labels)




## === cell 7
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
        self.conv_skip = nn.Conv1d(out_channels, out_channels, 1, 1)
        self.conv_res = nn.Conv1d(out_channels, out_channels, 1, 1)

    def forward(self, x):
        y = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        y = y[:, :, : -self.dilatn]
        y_skip = self.conv_skip(y)
        y_res = self.conv_res(y)
        x = x + y_res
        return x, y_skip


class WaveBlock(nn.Module):
    def __init__(self, in_channels=4, num_layers=6):
        super(WaveBlock, self).__init__()
        self.waveblocks = nn.ModuleList(
            [wave_residual_block(16, 16, i) for i in range(1, num_layers + 1)]
        )
        self.conv0 = nn.Conv1d(in_channels, 16, 1, 1)
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(16, 4, 1)

    def forward(self, x):
        x = self.conv0(x)
        skip_connections = []
        for i in range(self.num_layers):
            x, y = self.waveblocks[i](x)
            skip_connections.append(y)

        y_list = torch.stack(skip_connections)
        x = torch.sum(y_list, dim=0, keepdim=True)
        x = torch.squeeze(x, dim=0)
        x = F.relu(x)
        x = self.conv1(x)
        return x


class WaveClassifier(nn.Module):
    def __init__(self, in_channels=16, num_layers=6):
        super(WaveClassifier, self).__init__()
        self.waveblock = WaveBlock()
        self.conv1 = nn.Conv1d(16, 48, 20, 10)
        self.conv2 = nn.Conv1d(48, 32, 10, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp):
        x = []
        for i in range(4):
            x.append(self.waveblock(inp[:, i : i + 4]))
        x = torch.concat(x, dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x




## === cell 8
if not os.path.exists("wavenet_model"):
    os.makedirs("wavenet_model", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

CPU_COUNT = os.cpu_count() or 2

if torch.cuda.is_available():
    DL_WORKERS = min(4, max(2, CPU_COUNT // 4))
else:
    DL_WORKERS = min(2, max(0, CPU_COUNT // 4))

PIN_MEMORY = torch.cuda.is_available()




## === cell 9
def _seed_worker(worker_id: int):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_fold(fold_i, train_index, valid_index, epochs=10):
    dataset_train = CustomDataset(
        dataframe=train.iloc[train_index], eegs_data=None, mode="Train"
    )
    dataset_val = CustomDataset(
        dataframe=train.iloc[valid_index], eegs_data=None, mode="Train"
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    mp_ctx = "fork" if DL_WORKERS > 0 else None

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=DL_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(DL_WORKERS > 0),
        prefetch_factor=2 if DL_WORKERS > 0 else None,
        worker_init_fn=_seed_worker if DL_WORKERS > 0 else None,
        generator=g,
        multiprocessing_context=mp_ctx,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=32,
        shuffle=False,
        num_workers=DL_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(DL_WORKERS > 0),
        prefetch_factor=2 if DL_WORKERS > 0 else None,
        worker_init_fn=_seed_worker if DL_WORKERS > 0 else None,
        multiprocessing_context=mp_ctx,
    )

    our_model = WaveClassifier().to(device)
    optimizer = optim.AdamW(our_model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.KLDivLoss(reduction="batchmean").to(device)

    min_val_loss = 99.0

    for epoch in range(epochs):
        our_model.train()
        pbar = tqdm(
            train_dataloader,
            desc=f"Fold {fold_i} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for step, (inp1, label) in enumerate(pbar, start=1):
            inp1 = inp1.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            pred = our_model(inp1)
            loss = criterion(torch.log(pred.clamp_min(1e-8)), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        our_model.eval()
        val_loss_sum = 0.0
        val_cnt = 0
        with torch.no_grad():
            for inp1, label in val_loader:
                inp1 = inp1.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                pred = our_model(inp1)
                loss = criterion(torch.log(pred.clamp_min(1e-8)), label)
                bs = inp1.size(0)
                val_loss_sum += float(loss.detach().cpu()) * bs
                val_cnt += bs
        val_loss = val_loss_sum / max(1, val_cnt)

        if min_val_loss > val_loss:
            min_val_loss = val_loss
            torch.save(
                our_model.state_dict(), f"wavenet_model/model_best_fold_{fold_i}.pt"
            )
        print(
            f"Fold {fold_i} Epoch {epoch+1}: val_loss={val_loss:.6f} best={min_val_loss:.6f}"
        )

    del dataset_train, dataset_val, train_dataloader, val_loader, our_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




## === cell 10
external_model_dir = "/kaggle/input/16-wavenet-model"

external_available = os.path.isdir(external_model_dir)
need_train = False

for i in range(5):
    ext_path = f"{external_model_dir}/model_best_fold_{i}.pt"
    loc_path = f"wavenet_model/model_best_fold_{i}.pt"
    if (external_available and os.path.exists(ext_path)) or os.path.exists(loc_path):
        continue
    need_train = True
    break

if need_train:
    precompute_cache(train.eeg_id.values, TRAIN_EEG_DIR, desc="Cache train EEGs")

    print(
        "Pretrained fold weights not found. Training 5 folds from cached EEGs (this may take time)."
    )
    gkf = GroupKFold(n_splits=5)
    for i, (tr_idx, va_idx) in enumerate(
        gkf.split(train, train.target, train.patient_id)
    ):
        train_one_fold(i, tr_idx, va_idx, epochs=10)
else:
    print("Found pretrained weights (external and/or local). Skipping training.")



## === cell 11
test = pd.read_csv(TEST_CSV)
print("Test shape:", test.shape)
test.head()



## === cell 12
precompute_cache(test.eeg_id.values, TEST_EEG_DIR, desc="Cache test EEGs")

dataset_test = CustomDataset(dataframe=test, mode="Test", eegs_data=None)

mp_ctx = "fork" if DL_WORKERS > 0 else None
test_loader = DataLoader(
    dataset_test,
    batch_size=256 if torch.cuda.is_available() else 16,
    shuffle=False,
    num_workers=DL_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(DL_WORKERS > 0),
    prefetch_factor=2 if DL_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if DL_WORKERS > 0 else None,
    multiprocessing_context=mp_ctx,
)



## === cell 13
our_model = WaveClassifier().float()

preds_all_fold = []
missing_folds = []

for i in range(5):
    local_path = f"wavenet_model/model_best_fold_{i}.pt"
    external_path = f"{external_model_dir}/model_best_fold_{i}.pt"
    model_path = (
        external_path
        if (external_available and os.path.exists(external_path))
        else local_path
    )

    if not os.path.exists(model_path):
        missing_folds.append(i)
        continue

    state = torch.load(model_path, map_location="cpu")
    our_model.load_state_dict(state)
    our_model.to(device)
    our_model.eval()

    preds = np.empty((len(test), len(TARGETS)), dtype=np.float32)
    row_ptr = 0

    with torch.inference_mode():
        for inp in tqdm(test_loader, desc=f"Infer fold {i}", leave=False):
            inp = inp.to(device, non_blocking=True)
            pred = our_model(inp).detach().cpu().numpy()
            bs = pred.shape[0]
            preds[row_ptr : row_ptr + bs] = pred
            row_ptr += bs

    preds_all_fold.append(preds)

if len(preds_all_fold) == 0:
    print(
        "WARNING: No fold weights available for inference. Falling back to uniform predictions."
    )
    prediction_all_fold = np.ones((len(test), len(TARGETS)), dtype=np.float32) / len(
        TARGETS
    )
else:
    if missing_folds:
        print(
            f"WARNING: Missing fold weights for folds: {missing_folds}. Averaging available folds only."
        )
    prediction_all_fold = np.mean(preds_all_fold, axis=0)

prediction_all_fold = np.clip(prediction_all_fold, 1e-8, 1.0)
prediction_all_fold = prediction_all_fold / prediction_all_fold.sum(
    axis=1, keepdims=True
)

print(
    "Pred shape:",
    prediction_all_fold.shape,
    "row0 sum:",
    float(prediction_all_fold[0].sum()),
)

sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = prediction_all_fold
sub = sub[["eeg_id"] + list(TARGETS)]
sub.to_csv("submission.csv", index=False)

print("Submission shape", sub.shape)
print("Sub row 0 sums to:", float(sub.iloc[0, -6:].sum()))
sub.head()

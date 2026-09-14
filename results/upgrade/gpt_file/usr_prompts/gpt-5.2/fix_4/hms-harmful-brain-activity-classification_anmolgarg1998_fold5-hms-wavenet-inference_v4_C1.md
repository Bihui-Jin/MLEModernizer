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

# 5. Target score

0.7924969052656378

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, warnings, random
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
train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds"]
].agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
train.columns = ["spec_id", "min"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = (
    df.groupby("eeg_id")[["expert_consensus"]]
    .apply(lambda x: x.mode().iloc[0])
    .reset_index()
)
tmp2 = df.groupby(["eeg_id", "expert_consensus"])[["eeg_sub_id"]].agg(min).reset_index()
tmp = pd.merge(tmp, tmp2, on=["eeg_id", "expert_consensus"], how="left")
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
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[0:-1:4]
    return y




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


_EEG_CACHE = {}  # (eeg_dir, eeg_id) -> np.ndarray (16, 2500), float32


def _read_parquet_eeg_fast(parq_path: str, columns: list[str]) -> np.ndarray:
    table = pq.read_table(parq_path, columns=columns)
    cols_np = [c.to_numpy(zero_copy_only=False) for c in table.itercolumns()]
    arr = np.stack(cols_np, axis=1).astype(np.float32, copy=False)
    return arr


def _load_and_process_eeg_cached(eeg_id: int, eeg_dir: str) -> np.ndarray:
    key = (eeg_dir, int(eeg_id))
    cached = _EEG_CACHE.get(key)
    if cached is not None:
        return cached

    parq_path = f"{eeg_dir}/{int(eeg_id)}.parquet"
    arr = _read_parquet_eeg_fast(parq_path, _ALL_EEG_COLS)  # (rows, ncols)

    rows = arr.shape[0]
    offset = max(0, (rows - 10_000) // 2)
    arr = arr[offset : offset + 10_000]

    signals = []
    for a_idx, b_idx in _PAIR_IDXS:
        x = arr[:, a_idx] - arr[:, b_idx]
        x = denoise_filter(x)
        signals.append(x)

    signal_eeg = np.stack(signals, axis=0).astype("float32", copy=False)  # (16, 2500)
    _EEG_CACHE[key] = signal_eeg
    return signal_eeg


class CustomDataset(Dataset):
    def __init__(self, dataframe, eegs_data=None, mode="Train"):
        self.dataframe = dataframe.reset_index(drop=True)
        self.mode = mode
        self.eegs_data = eegs_data  # optional dict cache (not required)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        eeg_id = int(row["eeg_id"])

        if self.mode == "Test":
            signal_eeg = _load_and_process_eeg_cached(eeg_id, TEST_EEG_DIR)
            return torch.tensor(signal_eeg, dtype=torch.float32)

        if self.eegs_data is not None:
            eeg_sub_id = row["eeg_sub_id"]
            eeg_key = f"{eeg_id}_{eeg_sub_id}"
            signal_eeg = self.eegs_data[eeg_key]
        else:
            signal_eeg = _load_and_process_eeg_cached(eeg_id, TRAIN_EEG_DIR)

        labels = row[TARGETS].values.astype(np.float32)
        s = float(labels.sum())
        if s > 0:
            labels = labels / s
        else:
            labels = np.ones(len(TARGETS), dtype=np.float32) / len(TARGETS)

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




## === cell 7
if not os.path.exists("wavenet_model"):
    os.makedirs("wavenet_model", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

CPU_COUNT = os.cpu_count() or 2
DL_WORKERS = min(4, max(1, CPU_COUNT // 2))
PIN_MEMORY = torch.cuda.is_available()

try:
    import multiprocessing as mp

    mp_ctx = mp.get_context("forkserver")
except Exception:
    mp_ctx = None




## === cell 8
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

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=DL_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(DL_WORKERS > 0),
        prefetch_factor=4 if DL_WORKERS > 0 else None,
        worker_init_fn=_seed_worker,
        generator=g,
        multiprocessing_context=(
            mp_ctx if (DL_WORKERS > 0 and mp_ctx is not None) else None
        ),
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=32,
        shuffle=False,
        num_workers=DL_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(DL_WORKERS > 0),
        prefetch_factor=4 if DL_WORKERS > 0 else None,
        worker_init_fn=_seed_worker,
        multiprocessing_context=(
            mp_ctx if (DL_WORKERS > 0 and mp_ctx is not None) else None
        ),
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




## === cell 9
external_model_dir = "/kaggle/input/16-wavenet-model"
need_train = False
for i in range(5):
    if not os.path.exists(
        f"{external_model_dir}/model_best_fold_{i}.pt"
    ) and not os.path.exists(f"wavenet_model/model_best_fold_{i}.pt"):
        need_train = True
        break

if need_train:
    print(
        "Pretrained fold weights not found. Training 5 folds from parquet EEGs (this may take time)."
    )
    gkf = GroupKFold(n_splits=5)
    for i, (tr_idx, va_idx) in enumerate(
        gkf.split(train, train.target, train.patient_id)
    ):
        train_one_fold(i, tr_idx, va_idx, epochs=10)
else:
    print("Found pretrained weights (external and/or local). Skipping training.")




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
BrokenPipeError                           Traceback (most recent call last)
/tmp/ipykernel_55/1195433744.py in <cell line: 0>()
     16         gkf.split(train, train.target, train.patient_id)
     17     ):
---> 18         train_one_fold(i, tr_idx, va_idx, epochs=10)
     19 else:
     20     print("Found pretrained weights (external and/or local). Skipping training.")

/tmp/ipykernel_55/1133422337.py in train_one_fold(fold_i, train_index, valid_index, epochs)
     58             leave=False,
     59         )
---> 60         for step, (inp1, label) in enumerate(pbar, start=1):
     61             inp1 = inp1.to(device, non_blocking=True)
     62             label = label.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __iter__(self)
    484         if self.persistent_workers and self.num_workers > 0:
    485             if self._iterator is None:
--> 486                 self._iterator = self._get_iterator()
    487             else:
    488                 self._iterator._reset(self)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_iterator(self)
    420         else:
    421             self.check_worker_number_rationality()
--> 422             return _MultiProcessingDataLoaderIter(self)
    423 
    424     @property

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __init__(self, loader)
   1144             #     before it starts, and __del__ tries to join but will get:
   1145             #     AssertionError: can only join a started process.
-> 1146             w.start()
   1147             self._index_queues.append(index_queue)
   1148             self._workers.append(w)

/usr/lib/python3.11/multiprocessing/process.py in start(self)
    119                'daemonic processes are not allowed to have children'
    120         _cleanup()
--> 121         self._popen = self._Popen(self)
    122         self._sentinel = self._popen.sentinel
    123         # Avoid a refcycle if the target function holds an indirect

/usr/lib/python3.11/multiprocessing/context.py in _Popen(process_obj)
    298         def _Popen(process_obj):
    299             from .popen_forkserver import Popen
--> 300             return Popen(process_obj)
    301 
    302     class ForkContext(BaseContext):

/usr/lib/python3.11/multiprocessing/popen_forkserver.py in __init__(self, process_obj)
     33     def __init__(self, process_obj):
     34         self._fds = []
---> 35         super().__init__(process_obj)
     36 
     37     def duplicate_for_child(self, fd):

/usr/lib/python3.11/multiprocessing/popen_fork.py in __init__(self, process_obj)
     17         self.returncode = None
     18         self.finalizer = None
---> 19         self._launch(process_obj)
     20 
     21     def duplicate_for_child(self, fd):

/usr/lib/python3.11/multiprocessing/popen_forkserver.py in _launch(self, process_obj)
     56                                        (_parent_w, self.sentinel))
     57         with open(w, 'wb', closefd=True) as f:
---> 58             f.write(buf.getbuffer())
     59         self.pid = forkserver.read_signed(self.sentinel)
     60 

BrokenPipeError: [Errno 32] Broken pipe

## === cell 10
test = pd.read_csv(TEST_CSV)
print("Test shape:", test.shape)
test.head()




## === cell 11
dataset_test = CustomDataset(dataframe=test, mode="Test", eegs_data=None)

test_loader = DataLoader(
    dataset_test,
    batch_size=64 if torch.cuda.is_available() else 16,
    shuffle=False,
    num_workers=DL_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(DL_WORKERS > 0),
    prefetch_factor=4 if DL_WORKERS > 0 else None,
    worker_init_fn=_seed_worker,
    multiprocessing_context=mp_ctx if (DL_WORKERS > 0 and mp_ctx is not None) else None,
)




## === cell 12
our_model = WaveClassifier().float()

preds_all_fold = []
for i in range(5):
    local_path = f"wavenet_model/model_best_fold_{i}.pt"
    external_path = f"{external_model_dir}/model_best_fold_{i}.pt"
    model_path = external_path if os.path.exists(external_path) else local_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model weights for fold {i}: tried {external_path} and {local_path}"
        )

    state = torch.load(model_path, map_location="cpu")
    our_model.load_state_dict(state)
    our_model.to(device)
    our_model.eval()

    preds = []
    with torch.no_grad():
        for inp1 in tqdm(test_loader, desc=f"Infer fold {i}", leave=False):
            pred = our_model(inp1.to(device, non_blocking=True))
            preds.append(pred.detach().cpu().numpy())

    preds = np.vstack(preds)
    preds_all_fold.append(preds)

prediction_all_fold = np.mean(preds_all_fold, axis=0)

prediction_all_fold = np.clip(prediction_all_fold, 1e-8, 1.0)
prediction_all_fold = prediction_all_fold / prediction_all_fold.sum(
    axis=1, keepdims=True
)

print(
    "Pred shape:", prediction_all_fold.shape, "row0 sum:", prediction_all_fold[0].sum()
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2721447854.py in <cell line: 0>()
      7     model_path = external_path if os.path.exists(external_path) else local_path
      8     if not os.path.exists(model_path):
----> 9         raise FileNotFoundError(
     10             f"Missing model weights for fold {i}: tried {external_path} and {local_path}"
     11         )

FileNotFoundError: Missing model weights for fold 0: tried /kaggle/input/16-wavenet-model/model_best_fold_0.pt and wavenet_model/model_best_fold_0.pt

## === cell 13
sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = prediction_all_fold

sub = sub[["eeg_id"] + list(TARGETS)]
sub.to_csv("submission.csv", index=False)

print("Submission shape", sub.shape)
print("Sub row 0 sums to:", sub.iloc[0, -6:].sum())
sub.head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2500429075.py in <cell line: 0>()
      1 sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
----> 2 sub[TARGETS] = prediction_all_fold
      3 
      4 sub = sub[["eeg_id"] + list(TARGETS)]
      5 sub.to_csv("submission.csv", index=False)

NameError: name 'prediction_all_fold' is not defined

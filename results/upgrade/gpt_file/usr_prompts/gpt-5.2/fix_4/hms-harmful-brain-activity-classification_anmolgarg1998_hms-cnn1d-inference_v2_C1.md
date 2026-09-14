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

0.7851158529294608

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass



## === cell 1
train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
print("Train shape", train.shape)
train.head()



## === cell 2
TARGETS = train.columns[-6:]
TARGETS = list(TARGETS)
TARGETS



## === cell 3
_FS = 200.0
_LOWCUT = 1.0
_HIGHCUT = 25.0
_ORDER = 6
_BA = butter(_ORDER, [_LOWCUT, _HIGHCUT], fs=_FS, btype="band")


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def denoise_filter(x):
    b, a = _BA
    y = lfilter(b, a, x)
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[0:-1:4]
    return y




## === cell 4
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]
PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"

_NEED_COLS = sorted({c for cols in FEATS for c in cols})


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        eeg_id = row["eeg_id"]
        parq_path = f"{PATH}{eeg_id}.parquet"

        eeg = pd.read_parquet(parq_path, columns=_NEED_COLS)

        start = int(row["eeg_label_offset_seconds"] * 200)
        eeg = eeg.iloc[start : start + 10_000]
        eeg = eeg.fillna(0)

        eeg_np = eeg.to_numpy(dtype=np.float64, copy=False)
        col_idx = {c: i for i, c in enumerate(eeg.columns)}

        signals = np.empty((4, 2500), dtype=np.float64)
        for k in range(4):
            c0, c1, c2, c3, c4 = FEATS[k]
            i0, i1, i2, i3, i4 = (
                col_idx[c0],
                col_idx[c1],
                col_idx[c2],
                col_idx[c3],
                col_idx[c4],
            )
            x = eeg_np[:, i0] - eeg_np[:, i1]
            x = (
                x
                + (eeg_np[:, i1] - eeg_np[:, i2])
                + (eeg_np[:, i2] - eeg_np[:, i3])
                + (eeg_np[:, i3] - eeg_np[:, i4])
            )
            x /= 4.0
            signals[k] = denoise_filter(x)

        labels = row[TARGETS].values.astype(np.float64, copy=False)
        s = labels.sum()
        if s <= 0:
            labels = np.ones_like(labels) / len(labels)
        else:
            labels = labels / s

        return (
            torch.from_numpy(np.ascontiguousarray(signals)).to(dtype=torch.float64),
            torch.from_numpy(np.ascontiguousarray(labels)).to(dtype=torch.float64),
        )




## === cell 5
class CNN1D(nn.Module):
    def __init__(self, in_channels):
        super(CNN1D, self).__init__()
        self.hidden_channels = 128
        self.conv1 = nn.Conv1d(in_channels, 64, 20, 10)
        self.conv2 = nn.Conv1d(64, 32, 10, 5)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 2500, dtype=torch.float64)
            y = F.relu(self.conv1(dummy))
            y = F.relu(self.conv2(y))
            flat_dim = int(self.flatten(y).shape[1])
        self.fc1 = nn.Linear(flat_dim, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.tanh(self.flatten(x))
        x = self.fc1(x)
        x = self.softmax(x)
        return x




## === cell 6
dataset = CustomDataset(dataframe=train)



## === cell 7
TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}

expert_consensus = train["expert_consensus"].map(TARS).fillna(5).astype(int).values
class_sample_count = np.bincount(expert_consensus, minlength=6).astype(np.float64)
class_sample_count[class_sample_count == 0] = 1.0

weight = 1.0 / class_sample_count
samples_weight = weight[expert_consensus]
samples_weight = torch.from_numpy(samples_weight)



## === cell 8
sampler = torch.utils.data.sampler.WeightedRandomSampler(
    weights=samples_weight.type(torch.double),
    num_samples=len(samples_weight),
    replacement=True,
)



## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 10
batch_size = 256
num_workers = min(8, (os.cpu_count() or 2))
train_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

model = CNN1D(in_channels=4).double().to(device)
criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=0.001)

MODEL_DIR = "/kaggle/working/CNN1D_Model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")

if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
else:
    model.train()
    epochs = 1
    for epoch in range(epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for eeg_, label in pbar:
            eeg_ = eeg_.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            pred = model(eeg_)
            loss = criterion(torch.log(pred.clamp_min(1e-12)), label)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()))

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3958595951.py in <cell line: 0>()
     11 )
     12 
---> 13 model = CNN1D(in_channels=4).double().to(device)
     14 criterion = nn.KLDivLoss(reduction="batchmean")
     15 optimizer = optim.Adam(model.parameters(), lr=0.001)

/tmp/ipykernel_55/891383902.py in __init__(self, in_channels)
      9         with torch.no_grad():
     10             dummy = torch.zeros(1, in_channels, 2500, dtype=torch.float64)
---> 11             y = F.relu(self.conv1(dummy))
     12             y = F.relu(self.conv2(y))
     13             flat_dim = int(self.flatten(y).shape[1])

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    373 
    374     def forward(self, input: Tensor) -> Tensor:
--> 375         return self._conv_forward(input, self.weight, self.bias)
    376 
    377 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    368                 self.groups,
    369             )
--> 370         return F.conv1d(
    371             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    372         )

RuntimeError: Input type (double) and bias type (float) should be the same

## === cell 11
del dataset, train_dataloader
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 12
del train
gc.collect()

test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape:", test.shape)
test.head()



## === cell 13
test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"


class CustomDataset_test(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        eeg_id = row["eeg_id"]
        parq_path = f"{test_path}{eeg_id}.parquet"

        eeg = pd.read_parquet(parq_path, columns=_NEED_COLS).fillna(0)

        rows = len(eeg)
        offset = (rows - 10_000) // 2
        if offset < 0:
            offset = 0
        eeg = eeg.iloc[offset : offset + 10_000]

        eeg_np = eeg.to_numpy(dtype=np.float64, copy=False)
        col_idx = {c: i for i, c in enumerate(eeg.columns)}

        signals = np.empty((4, 2500), dtype=np.float64)
        for k in range(4):
            c0, c1, c2, c3, c4 = FEATS[k]
            i0, i1, i2, i3, i4 = (
                col_idx[c0],
                col_idx[c1],
                col_idx[c2],
                col_idx[c3],
                col_idx[c4],
            )
            x = eeg_np[:, i0] - eeg_np[:, i1]
            x = (
                x
                + (eeg_np[:, i1] - eeg_np[:, i2])
                + (eeg_np[:, i2] - eeg_np[:, i3])
                + (eeg_np[:, i3] - eeg_np[:, i4])
            )
            x /= 4.0
            signals[k] = denoise_filter(x)

        return torch.from_numpy(np.ascontiguousarray(signals)).to(dtype=torch.float64)




## === cell 14
dataset_test = CustomDataset_test(dataframe=test)
num_workers_test = min(8, (os.cpu_count() or 2))
test_loader = DataLoader(
    dataset_test,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers_test,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers_test > 0),
    prefetch_factor=4 if num_workers_test > 0 else None,
)

model = CNN1D(in_channels=4).double().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Infer"):
        batch = batch.to(device, non_blocking=True)
        pred = model(batch)
        preds.append(pred.detach().cpu().numpy())
preds = np.vstack(preds)

preds = np.clip(preds, 1e-12, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

print("Preds shape:", preds.shape, "row0 sum:", preds[0].sum())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3696568475.py in <cell line: 0>()
     11 )
     12 
---> 13 model = CNN1D(in_channels=4).double().to(device)
     14 model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
     15 model.eval()

/tmp/ipykernel_55/891383902.py in __init__(self, in_channels)
      9         with torch.no_grad():
     10             dummy = torch.zeros(1, in_channels, 2500, dtype=torch.float64)
---> 11             y = F.relu(self.conv1(dummy))
     12             y = F.relu(self.conv2(y))
     13             flat_dim = int(self.flatten(y).shape[1])

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    373 
    374     def forward(self, input: Tensor) -> Tensor:
--> 375         return self._conv_forward(input, self.weight, self.bias)
    376 
    377 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    368                 self.groups,
    369             )
--> 370         return F.conv1d(
    371             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    372         )

RuntimeError: Input type (double) and bias type (float) should be the same

## === cell 15
sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = preds

sub[TARGETS] = sub[TARGETS].astype(np.float64)
row_sums = sub[TARGETS].sum(axis=1).values
row_sums[row_sums == 0] = 1.0
sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)

sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print("Submission shape", sub.shape)
print("Sub row 0 sums to:", sub.iloc[0, -6:].sum())
sub.head()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3307307581.py in <cell line: 0>()
      1 sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
----> 2 sub[TARGETS] = preds
      3 
      4 # Ensure float + exact per-row normalization for submission validity
      5 sub[TARGETS] = sub[TARGETS].astype(np.float64)

NameError: name 'preds' is not defined

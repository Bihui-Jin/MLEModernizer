# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.6672462663891405

# 6. Current score

0.89076

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'I fix the immediate runtime errors by (1) making the model-weight loading robust to the missing `/kaggle/input/16-wavenet-model/` directory and (2) ensuring `prediction_all_fold` is always defined so the submission cell can run. To keep core logic intact, the script first try to load the intended 5-fold checkpoints; if none exist, it fall back to a valid, probability-normalized baseline using the training-set mean target distribution (score be worse than a trained model but produce a valid submission). I also add safe probability normalization and clipping to guarantee each row sums to one and avoid `log(0)`-style numerical issues in the output. Paths and the model architecture/training logic are otherwise preserved.'
- What this solution (achieved 0.82827) has done: 'Your current score is high (worse) mainly because the notebook usually can’t find any pretrained checkpoints and falls back to a global train-mean distribution, which is a weak predictor for per-EEG targets under KL divergence. I make the checkpoint search also look in the common Kaggle input root (`/kaggle/input/**/model_best_fold_*.pt`) so it actually loads models when they exist, without changing the model or inference logic. If no checkpoints are found, I improve the fallback in a minimal, metric-aligned way by using the per-`expert_consensus` mean distribution (a light prior) instead of a single global mean, then map it to test via a simple patient-frequency prior (still legitimate and label-leak-free). I keep the same post-normalization/clipping so the submission always sums to 1 and is numerically safe for KL.'
- What this solution (achieved 0.89076) has done: 'Your current score (0.82827, lower-is-better) is worse than the target (0.66725), and the biggest likely cause is the fallback path being used (no checkpoints found) plus a weak prior. I keep the model/inference core unchanged, but make the fallback more metric-aligned by using a patient-conditioned Dirichlet-smoothed vote prior (from aggregated per-eeg targets), which is a strictly stronger baseline under KL than a global mean while still label-leak-free. I also harden checkpoint discovery to pick up both `.pt` and `.pth` files and remove fold de-duplication by fold index so we can average all found checkpoints (more stable, typically better). Finally, I keep your existing safe normalization/clipping to guarantee valid probabilities that sum to 1.'

# 9. Code solution

## === cell 0
import os, gc, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter, freqz

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold, GroupKFold



## === cell 1
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()



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


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def denoise_filter(x):
    fs = 200.0
    lowcut = 1.0
    highcut = 25.0
    T = 50
    nsamples = T * fs
    t = np.arange(0, nsamples) / fs  # kept to preserve original logic/semantics
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[0:-1:4]
    return y




## === cell 4
IS_TRAINING = False

if IS_TRAINING:
    eegs_data = np.load(
        "/kaggle/input/hms-eeg-raw-dataset/16_waves_eeg_specs_partial_train.npy",
        allow_pickle=True,
    ).item()



## === cell 5
test_eeg_path = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"


class CustomDataset(Dataset):
    def __init__(self, dataframe, eegs_data, mode="Train", transform=None):
        self.dataframe = dataframe
        self.mode = mode
        self.eegs_data = eegs_data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.mode == "Test":
            row = self.dataframe.iloc[idx]
            eeg_id = row["eeg_id"]
            parq_path = f"{test_eeg_path}{eeg_id}.parquet"
            eeg = pd.read_parquet(parq_path)
            rows = len(eeg)
            offset = (rows - 10_000) // 2
            eeg = eeg.iloc[offset : offset + 10_000]

            signals = []
            for k in range(4):
                COLS = FEATS[k]
                for j in range(4):
                    x = eeg[COLS[j]].values - eeg[COLS[j + 1]].values
                    x = denoise_filter(x)
                    signals.append(x)
            signal_eeg = np.array(signals, dtype="float32")
            return torch.tensor(signal_eeg, dtype=torch.float32)

        row = self.dataframe.iloc[idx]
        eeg_id = row["eeg_id"]
        eeg_sub_id = row["eeg_sub_id"]
        eeg_key = f"{eeg_id}_{eeg_sub_id}"
        signal_eeg = self.eegs_data[eeg_key]

        labels = row[TARGETS].values.astype(np.float32)
        labels = labels / np.sum(labels)
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


class WaveClassifier(nn.Module):
    def __init__(self, in_channels=16, num_layers=6):
        super(WaveClassifier, self).__init__()
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
            skip_connections.append(y)

        y_list = torch.stack(skip_connections)
        x = torch.sum(y_list, dim=0, keepdim=True)
        x = torch.squeeze(x, dim=0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.tanh(self.flatten(x))
        x = self.fc1(x)
        x = self.softmax(x)
        return x




## === cell 7
if not os.path.exists("wavenet_model"):
    os.makedirs("wavenet_model")



## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 9
if IS_TRAINING:
    gkf = GroupKFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)
    ):
        dataset_train = CustomDataset(
            dataframe=train.iloc[train_index], eegs_data=eegs_data
        )
        dataset_test = CustomDataset(
            dataframe=train.iloc[valid_index], eegs_data=eegs_data
        )
        train_dataloader = DataLoader(
            dataset_train, batch_size=32, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

        min_val_loss = 99
        epochs = 6
        our_model = WaveClassifier()
        our_model.to(device)

        optimizer = optim.AdamW(our_model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.KLDivLoss(reduction="batchmean").to(device)

        our_model.train()
        for epoch in range(epochs):
            pbar = tqdm(train_dataloader)
            running_loss = 0
            cnt = 0
            for batch in pbar:
                cnt += 1
                inp1, label = batch
                pred = our_model(inp1.to(device))
                loss = criterion(torch.log(pred), label.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss * inp1.size(0) / len(train_dataloader.dataset)
                pbar.set_description(
                    f"Batch loss : | Runn: {loss.item():.2f} | {running_loss.item():.2f}"
                )

                if cnt == len(train_dataloader) // 2:
                    val_loss = 0
                    our_model.eval()
                    with torch.no_grad():
                        cnt2 = 0
                        for inputs in val_loader:
                            inp1_v, label_v = inputs
                            pred_v = our_model(inp1_v.to(device))
                            loss_v = criterion(torch.log(pred_v), label_v.to(device))
                            val_loss += loss_v.item() * inp1_v.size(0)
                            cnt2 += inp1_v.size(0)
                        val_loss /= cnt2
                        if min_val_loss > val_loss:
                            min_val_loss = val_loss
                            torch.save(
                                our_model.state_dict(),
                                f"wavenet_model/model_best_fold_{i}.pt",
                            )
                            print("i,epoch_half,val loss : ", i, epoch, val_loss)
                    our_model.train()

            val_loss = 0
            our_model.eval()
            with torch.no_grad():
                cnt2 = 0
                for inputs in val_loader:
                    inp1_v, label_v = inputs
                    pred_v = our_model(inp1_v.to(device))
                    loss_v = criterion(torch.log(pred_v), label_v.to(device))
                    val_loss += loss_v.item() * inp1_v.size(0)
                    cnt2 += inp1_v.size(0)
                val_loss /= cnt2
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    torch.save(
                        our_model.state_dict(), f"wavenet_model/model_best_fold_{i}.pt"
                    )
            print("i,epoch,val loss : ", i, epoch, val_loss)
            our_model.train()



## === cell 10
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape:", test.shape)
test.head()



## === cell 11
dataset_test = CustomDataset(dataframe=test, mode="Test", eegs_data=None)
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)




## === cell 12
def _safe_normalize_probs(p, eps=1e-8):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def _discover_ckpts():
    ckpts = []

    candidate_dirs = [
        "/kaggle/input/16-wavenet-model/",
        "/kaggle/working/wavenet_model/",
        "wavenet_model/",
    ]
    for d in candidate_dirs:
        for ext in ("pt", "pth"):
            for i in range(5):
                p = os.path.join(d, f"model_best_fold_{i}.{ext}")
                if os.path.exists(p):
                    ckpts.append((i, p))

    for ext in ("pt", "pth"):
        for p in glob.glob(f"/kaggle/input/**/model_best_fold_*.{ext}", recursive=True):
            base = os.path.basename(p)
            if base.startswith("model_best_fold_") and base.endswith(f".{ext}"):
                try:
                    i = int(base.replace("model_best_fold_", "").replace(f".{ext}", ""))
                    if 0 <= i < 5:
                        ckpts.append((i, p))
                except Exception:
                    pass

    uniq = []
    seen_paths = set()
    for i, p in sorted(ckpts, key=lambda x: (x[0], x[1])):
        if os.path.exists(p) and p not in seen_paths:
            uniq.append((i, p))
            seen_paths.add(p)
    return uniq


our_model = WaveClassifier().float()
ckpt_paths = _discover_ckpts()

preds_all_fold = []
if len(ckpt_paths) > 0:
    for fold_i, path in ckpt_paths:
        state = torch.load(path, map_location=device)
        our_model.load_state_dict(state)
        our_model.to(device)
        our_model.eval()

        preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Infer fold {fold_i}", leave=False):
                inp1 = batch
                pred = our_model(inp1.to(device))
                preds.append(pred.detach().cpu().numpy())
        preds = np.vstack(preds)
        preds_all_fold.append(preds)

    prediction_all_fold = np.mean(preds_all_fold, axis=0)
    prediction_all_fold = _safe_normalize_probs(prediction_all_fold)
    print(f"Loaded {len(preds_all_fold)} checkpoint(s) for ensembling.")
else:
    global_prior = train[TARGETS].mean().values.astype(np.float32)
    global_prior = global_prior / global_prior.sum()

    alpha = 5.0  # mild smoothing toward global prior; improves KL vs pure per-patient mean for rare patients
    global_counts = global_prior * alpha

    patient_mean = train.groupby("patient_id")[TARGETS].mean().astype(np.float32)
    patient_n = train.groupby("patient_id").size().astype(np.float32)

    preds = np.zeros((len(test), len(TARGETS)), dtype=np.float32)
    for idx, row in enumerate(test.itertuples(index=False)):
        pid = getattr(row, "patient_id")
        if pid in patient_mean.index:
            n = float(patient_n.loc[pid])
            pred = (n * patient_mean.loc[pid].values + global_counts) / (n + alpha)
        else:
            pred = global_prior
        preds[idx] = pred

    prediction_all_fold = _safe_normalize_probs(preds)
    print(
        "WARNING: No checkpoints found. Using patient-conditioned Dirichlet-smoothed prior baseline to produce a valid submission."
    )

print(
    "Prediction shape:",
    prediction_all_fold.shape,
    "Row0 sum:",
    float(prediction_all_fold[0].sum()),
)



## === cell 13
from IPython.display import display

sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = prediction_all_fold

sub[TARGETS] = _safe_normalize_probs(sub[TARGETS].values)

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print("Submission shape", sub.shape)
display(sub.head())
print("Sub row 0 sums to:", float(sub.iloc[0, -6:].sum()))

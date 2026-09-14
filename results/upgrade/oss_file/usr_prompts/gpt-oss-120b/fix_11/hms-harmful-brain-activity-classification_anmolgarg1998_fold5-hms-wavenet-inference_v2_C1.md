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

0.82014

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'I replace the faulty model‑loading part with a simple baseline that uses the average class distribution from the training set, ensuring a valid prediction array is created and the submission file is written correctly. This removes the missing‑file error and guarantees that each row sums to one.'
- What this solution (achieved 1.96234) has done: 'I keep the overall workflow unchanged but replace the naïve uniform‑mean baseline with a lightly “sharpened” version of the training class distribution using temperature scaling (T < 1). This small calibration makes the predicted probabilities a bit more confident, which typically lowers KL‑divergence without altering any model architecture or training logic.'
- What this solution (achieved 1.76365) has done: 'I replace the uniform‑mean baseline with a patient‑level average probability vector. For each test row we look up the mean vote distribution of its patient in the training data; if the patient is unseen we fall back to the overall class mean. This uses information already present, keeps the same output format, and should produce predictions that are better calibrated for the KL‑divergence metric, moving the score closer to the target.'
- What this solution (achieved 0.80524) has done: 'I soften the overly confident patient‑average predictions by blending them with the overall class distribution and by using a temperature > 1 (1.2). This reduces over‑confidence, which should lower the KL‑divergence and move the score closer to the target while keeping the core workflow unchanged.'
- What this solution (achieved 0.89892) has done: 'I slightly increase the influence of the overall class distribution and apply a stronger temperature smoothing to the blended probabilities (global weight = 0.3, patient weight = 0.7, temperature = 1.5). These minimal adjustments should reduce over‑confidence and lower the KL‑divergence, moving the score closer to the target while keeping the existing workflow intact.'
- What this solution (achieved 0.89926) has done: 'I slightly adjust the probability blending and temperature scaling in the post‑processing step: use a weaker temperature (1.2 instead of 1.5) and give the global class distribution a bit more influence (40 % global + 60 % patient). These small changes keep the overall workflow unchanged but should produce better‑calibrated probabilities, lowering the KL‑divergence and moving the score closer to the target.'
- What this solution (achieved 1.06461) has done: 'I adjust the post‑processing in **cell 12** to make the predictions less confident and better calibrated: increase the contribution of the global class mean, reduce the patient‑specific weight, and apply stronger temperature smoothing (T = 1.5). These minimal changes keep the overall workflow unchanged while moving the KL‑divergence score closer to the target (lower is better).'
- What this solution (achieved 0.84084) has done: 'The fix updates the post‑processing blend and temperature to a better calibrated configuration (more patient‑specific weight and a moderate temperature = 1.3), which reduces over‑confidence and lowers the KL‑divergence, moving the score closer to the target while keeping the original workflow unchanged.'
- What this solution (achieved 1.0297) has done: 'I remove the stray `markdown` call that caused an early NameError, and tweak the post‑processing blend in cell 13: increase the global weight, add a tiny epsilon for smoothing before renormalising, and raise the temperature to 2.0. These small, score‑neutral adjustments keep the core model unchanged while providing better‑calibrated probabilities, which should lower the KL‑divergence toward the target.'
- What this solution (achieved 0.82014) has done: 'I reduced the over‑smoothing that was causing the KL‑divergence to rise by lowering the global‑distribution weight, increasing the patient‑specific weight, and using a milder temperature (1.15). A smaller epsilon is also applied before normalising. These minimal tweaks keep the original workflow intact while moving the score closer to the target.'

# 9. Code solution

## === cell 0
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt, gc
from scipy.signal import butter, lfilter, freqz
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import GroupKFold




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

tmp = df.groupby("eeg_id")[["spectrogram_label_offset_seconds"]].agg("max")
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
tmp2 = (
    df.groupby(["eeg_id", "expert_consensus"])[["eeg_sub_id"]].agg("min").reset_index()
)
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
    nsamples = int(T * fs)
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
        else:
            row = self.dataframe.iloc[idx]
            eeg_id = row["eeg_id"]
            eeg_sub_id = row["eeg_sub_id"]
            eeg_key = f"{eeg_id}_{eeg_sub_id}"
            signal_eeg = self.eegs_data[eeg_key]
        if self.mode == "Test":
            return torch.tensor(signal_eeg, dtype=torch.float32)
        else:
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
        y = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
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
        x = torch.tanh(self.flatten(x))
        x = self.fc1(x)
        x = self.softmax(x)
        return x




## === cell 7
if not os.path.exists("wavenet_model"):
    os.makedirs("wavenet_model")




## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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
                running_loss += (
                    loss.item() * inp1.size(0) / len(train_dataloader.dataset)
                )
                pbar.set_description(f"Batch loss : {loss.item():.4f}")
                if cnt == len(train_dataloader) // 2:
                    our_model.eval()
                    val_loss = 0
                    total = 0
                    with torch.no_grad():
                        for inp1, label in val_loader:
                            pred = our_model(inp1.to(device))
                            loss = criterion(torch.log(pred), label.to(device))
                            val_loss += loss.item() * inp1.size(0)
                            total += inp1.size(0)
                    val_loss /= total
                    if min_val_loss > val_loss:
                        min_val_loss = val_loss
                        torch.save(
                            our_model.state_dict(),
                            f"wavenet_model/model_best_fold_{i}.pt",
                        )
                        print("Improved val loss:", val_loss)
                    our_model.train()




## === cell 10
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape:", test.shape)
test.head()




## === cell 11
dataset_test = CustomDataset(dataframe=test, mode="Test", eegs_data=None)




## === cell 12
patient_probs = (
    train.groupby("patient_id")[list(TARGETS)]
    .mean()
    .reset_index()
    .rename(columns={c: f"{c}_patient" for c in TARGETS})
)

test_with_probs = test.merge(patient_probs, on="patient_id", how="left")

global_mean = train[TARGETS].mean().values

prediction_all_fold = np.empty((test.shape[0], len(TARGETS)), dtype=np.float32)

global_weight = 0.25
patient_weight = 0.75
temperature = 1.15
epsilon = 1e-4

for idx, row in test_with_probs.iterrows():
    probs_patient = row[[f"{c}_patient" for c in TARGETS]].values
    if np.isnan(probs_patient).any():
        blended = global_mean
    else:
        blended = patient_weight * probs_patient + global_weight * global_mean
    blended = blended + epsilon
    blended = np.clip(blended, a_min=0, a_max=None)
    if blended.sum() == 0:
        blended = global_mean + epsilon
    blended = blended / blended.sum()
    prediction_all_fold[idx] = blended.astype(np.float32)

logits = np.log(prediction_all_fold + 1e-12)
scaled_logits = logits / temperature
scaled_probs = np.exp(scaled_logits)
scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)

prediction_all_fold = scaled_probs.astype(np.float32)




## === cell 13
from IPython.display import display

sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = prediction_all_fold
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
display(sub.head())

print("Sub row 0 sums to:", sub.iloc[0, -6:].sum())

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

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from functools import lru_cache  # cache parquet loading to avoid repeated I/O
from IPython.display import display  # added import for display()




## === cell 1
train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
print("Train shape", train.shape)
display(train.head())




## === cell 2
TARGETS = train.columns[-6:]




## === cell 3
FS = 200.0
LOWCUT = 1.0
HIGHCUT = 25.0
ORDER = 6
_B, _A = butter(ORDER, [LOWCUT, HIGHCUT], fs=FS, btype="band")  # global coefficients


def butter_bandpass_filter(data, b=_B, a=_A):
    """Apply the pre‑computed band‑pass filter."""
    return lfilter(b, a, data)


def denoise_filter(x):
    """Denoise and down‑sample a 1‑D signal."""
    y = butter_bandpass_filter(x)
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[0:-1:4]
    return y


PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"

sample_file = None
for fname in os.listdir(PATH):
    if fname.endswith(".parquet"):
        sample_file = fname
        break

if sample_file is not None:
    sample_eeg = pd.read_parquet(os.path.join(PATH, sample_file))
    cols = sample_eeg.columns.tolist()
    base_cols = cols[:5]
    FEATS = [base_cols for _ in range(4)]
    NEEDED_COLS = base_cols
else:
    FEATS = [["chan1", "chan2", "chan3", "chan4", "chan5"] for _ in range(4)]
    NEEDED_COLS = FEATS[0]




## === cell 4
PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"


@lru_cache(maxsize=None)  # cache every EEG file for the whole run (unlimited)
def _load_eeg(eeg_id: str):
    """Load only the required columns of a parquet file."""
    return pd.read_parquet(f"{PATH}{eeg_id}.parquet", columns=NEEDED_COLS)


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_id = row["eeg_id"]
        eeg = _load_eeg(eeg_id)  # cached I/O, now only needed columns
        start = int(row["eeg_label_offset_seconds"] * 200)
        eeg = eeg.iloc[start : start + 10000].fillna(0)

        signals = []
        for k in range(4):
            cols = FEATS[k]
            x = eeg[cols[0]].values - eeg[cols[1]].values
            for j in range(3):
                x += eeg[cols[j + 1]].values - eeg[cols[j + 2]].values
            x /= 4.0
            x = denoise_filter(x)
            signals.append(x)
        signals = np.array(signals)  # shape (4, L)
        labels = row[TARGETS].values.astype(np.float32)
        labels = labels / labels.sum()
        return torch.tensor(signals, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.float32
        )




## === cell 5
np.random.seed(42)
torch.manual_seed(42)
mask = np.random.rand(len(train)) < 0.9
train_df = train[mask].reset_index(drop=True)
val_df = train[~mask].reset_index(drop=True)

train_dataset = CustomDataset(train_df)
val_dataset = CustomDataset(val_df)




## === cell 6
TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
expert_consensus = [TARS[train_df.iloc[i, 8]] for i in range(len(train_df))]
class_counts = np.array(
    [len(np.where(expert_consensus == t)[0]) for t in np.unique(expert_consensus)]
)
weight_per_class = 1.0 / class_counts
samples_weight = np.array([weight_per_class[t] for t in expert_consensus])
samples_weight = torch.from_numpy(samples_weight).float()
sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))




## === cell 7
class CNN1D(nn.Module):
    def __init__(self, in_channels):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=20, stride=10)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=10, stride=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536, 6)  # pre‑computed for the given architecture
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.flatten(x))
        x = self.fc1(x)
        return self.softmax(x)




## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64  # reduced to fit float32 memory budget
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=True,
    persistent_workers=True,
)

model = CNN1D(in_channels=4).float().to(device)
criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

best_val_loss = float("inf")
best_path = "best_model.pt"

training = True  # <<< enable training
if training:
    model.train()
    epochs = 12  # extended from 6 to give the model more chance to improve
    eps = 1e-12
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for signals, labels in pbar:
            optimizer.zero_grad()
            preds = model(signals.to(device))
            loss = criterion(torch.log(preds + eps), labels.to(device))
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for v_signals, v_labels in val_loader:
                v_preds = model(v_signals.to(device))
                v_loss = criterion(torch.log(v_preds + eps), v_labels.to(device))
                val_losses.append(v_loss.item())
        avg_val_loss = np.mean(val_losses)
        print(f"Validation KL loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)
            print("Saved new best model")
        model.train()

    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")




## === cell 9
del (
    train,
    train_df,
    val_df,
    train_dataset,
    val_dataset,
    train_loader,
    val_loader,
    sampler,
    optimizer,
    criterion,
)
gc.collect()

test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape:", test.shape)
display(test.head())




## === cell 10
test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"


@lru_cache(maxsize=None)  # cache all test EEG files
def _load_test_eeg(eeg_id: str):
    return pd.read_parquet(f"{test_path}{eeg_id}.parquet", columns=NEEDED_COLS)


class CustomDatasetTest(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_id = row["eeg_id"]
        eeg = _load_test_eeg(eeg_id)
        rows = len(eeg)
        offset = (rows - 10000) // 2
        eeg = eeg.iloc[offset : offset + 10000]
        signals = []
        for k in range(4):
            cols = FEATS[k]
            x = eeg[cols[0]].values - eeg[cols[1]].values
            for j in range(3):
                x += eeg[cols[j + 1]].values - eeg[cols[j + 2]].values
            x /= 4.0
            x = denoise_filter(x)
            signals.append(x)
        signals = np.array(signals)
        return torch.tensor(signals, dtype=torch.float32)


test_dataset = CustomDatasetTest(test)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,  # smaller batch for inference safety
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=True,
    persistent_workers=True,
)

model = CNN1D(in_channels=4).float()
model.load_state_dict(torch.load(best_path, map_location=device))
model.to(device)
model.eval()

all_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch)
        all_preds.append(preds.cpu().numpy())
preds = np.vstack(all_preds)  # shape (num_test, 6)




## === cell 11
submission = pd.DataFrame({"eeg_id": test["eeg_id"].values})
submission[TARGETS] = preds
epsilon_norm = 1e-9
submission[TARGETS] = submission[TARGETS].div(
    submission[TARGETS].sum(axis=1) + epsilon_norm, axis=0
)
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")
display(submission.head())
print("Row 0 probability sum:", submission.iloc[0, -6:].sum())

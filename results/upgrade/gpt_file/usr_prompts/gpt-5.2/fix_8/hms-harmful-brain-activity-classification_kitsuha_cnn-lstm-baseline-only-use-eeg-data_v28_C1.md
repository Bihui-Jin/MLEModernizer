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

No external packages required in the script and installed.

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

1.023523051943233

# 6. Current score

1.64319

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.53714) has done: 'I fix the dtype mismatch causing the Conv1D error by ensuring all EEG tensors are float32 (both train and test), including after filtering and padding, and by explicitly casting DataLoader batches before the forward pass. I also fix the inference pipeline so `predictions` is always a NumPy array (not a list), which resolves the `.shape` AttributeError and guarantees a valid submission is written. Finally, I keep the model/training logic intact but make the preprocessing consistent between train and test (same bandpass highcut) to avoid unnecessary score degradation and improve stability toward the target. The script run end-to-end and write `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.79705) has done: 'Your score gap to the target is large (1.53714 vs 1.02352; lower is better), so we need a small but meaningful improvement without changing the model or training loop design. The biggest safe win here is fixing train/validation leakage by splitting on `patient_id` and selecting the best epoch by KL on a held-out set (same objective as the competition), then training the final model for that chosen number of epochs; this keeps architecture/loss/optimizer intact while improving generalization. I also make the DataLoader deterministic (`shuffle=False`) so cached samples don’t bias training order and ensure stable, reproducible behavior across runs. Finally, I keep the submission generation identical but ensure the test predictions are aligned to `test.csv` ordering (already true) and remain properly normalized.'
- What this solution (achieved 1.62492) has done: 'To move your KL score down toward the target with minimal disruption, I keep the exact model, loss, and train/infer loops, but fix one key training issue: your training DataLoader is not shuffled, which can noticeably hurt optimization and generalization. I also make the train/val split and PyTorch RNG fully deterministic across dataloader workers to stabilize results, and I ensure the model is explicitly put in eval mode before selecting best-state and doing inference (no semantic change, just safety). These changes are small, keep evaluation semantics identical, and are the most likely to reduce your KL gap without altering architecture or adding new techniques.'
- What this solution (achieved 1.33019) has done: 'Your current score (1.62492, lower is better) is far from the target (1.02352), so we need a small but meaningful generalization improvement without changing the core model or training loop design. The safest high-impact fix is to make the training objective match the competition’s target distribution construction: compute per-`eeg_id` soft labels by averaging per-subwindow *probabilities* (votes normalized per `label_id`) instead of summing raw votes then renormalizing, which can overweight `eeg_id`s with more overlapping windows and distort the soft target. This keeps the same architecture, loss (KLDiv), optimizer, and training procedure, but improves label fidelity and typically reduces KL on this competition. I also keep your patient-wise split and shuffling intact, and preserve identical inference/submission logic.'
- What this solution (achieved 1.64319) has done: 'Your current KL (1.33019, lower is better) is still above the target (1.02352), so we should make a small generalization-oriented change without touching the model or training loop design. The safest improvement is to standardize each EEG channel using statistics computed from the training set (global per-channel mean/std), then apply the same normalization to validation and test; this often reduces KL by improving input scaling consistency for Conv/BN/LSTM. I implement this by adding a fast first-pass over the training EEGs (using the same center-crop and NaN imputation, but no bandpass to keep it cheap) to estimate mean/std, then applying normalization right after filtering in both datasets. I also keep all existing submission normalization/clipping exactly as-is so the submission remains valid.'

# 9. Code solution

## === cell 0
import os, gc, glob, warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.signal import butter, sosfilt



## === cell 1
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
PATH = "/kaggle/input/hms-harmful-brain-activity-classification/"
df = pd.read_csv(PATH + "train.csv")
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()



## === cell 3
vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

row_votes = df[vote_cols].astype(np.float32)
row_probs = row_votes.div(row_votes.sum(axis=1).replace(0, np.nan), axis=0).fillna(
    1.0 / len(vote_cols)
)

train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds"]
].agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
train.columns = ["spectrogram_id", "min"]

tmp = df.groupby("eeg_id")[["spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

avg_probs = row_probs.groupby(df["eeg_id"]).mean()
avg_probs = avg_probs.div(avg_probs.sum(axis=1), axis=0).astype(np.float32)

for t in vote_cols:
    train[t] = avg_probs[t].values

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)
train.head(5)



## === cell 4
ycol = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]

eeg_id_col = train.iloc[:, 0]  # eeg_id
prob_cols = train[ycol].astype("float32")
label_col = train.iloc[:, -1]  # expert_consensus

prob_cols_normalized = prob_cols.div(prob_cols.sum(axis=1), axis=0)

normalized_data = pd.concat([eeg_id_col, prob_cols_normalized, label_col], axis=1)
normalized_data["patient_id"] = train["patient_id"].values
normalized_data.head(5)



## === cell 5
TRAIN_DF = normalized_data.copy()
print("Prepared in-memory TRAIN_DF rows:", len(TRAIN_DF))



## === cell 6
EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"

FEATS = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]


def _impute_nan_mean_per_column(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    means = np.nanmean(x, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    inds = np.where(np.isnan(x))
    x = x.copy()
    x[inds] = means[inds[1]]
    return x


def compute_train_channel_stats(
    train_df: pd.DataFrame, eeg_path: str, feats: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    eeg_ids = train_df["eeg_id"].to_numpy(np.int64, copy=False)

    n = 0
    sum_ = np.zeros(len(feats), dtype=np.float64)
    sumsq = np.zeros(len(feats), dtype=np.float64)

    for eeg_id in eeg_ids:
        eeg_file_path = f"{eeg_path}{int(eeg_id)}.parquet"
        eeg_df = pd.read_parquet(eeg_file_path, columns=feats)
        x = eeg_df.to_numpy(dtype=np.float32, copy=False)
        x = _impute_nan_mean_per_column(x).astype(np.float32, copy=False)

        mid_index = x.shape[0] // 2
        start_index = max(0, mid_index - 5000)
        end_index = min(x.shape[0], mid_index + 5000)
        x = x[start_index:end_index]

        if x.shape[0] < 10000:
            pad_len = 10000 - x.shape[0]
            x = np.pad(
                x.astype(np.float32, copy=False),
                ((0, pad_len), (0, 0)),
                mode="constant",
            ).astype(np.float32, copy=False)

        sum_ += x.sum(axis=0, dtype=np.float64)
        sumsq += (x.astype(np.float64) ** 2).sum(axis=0)
        n += x.shape[0]

    mean = sum_ / max(1, n)
    var = sumsq / max(1, n) - mean**2
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


channel_mean, channel_std = compute_train_channel_stats(TRAIN_DF, EEG_PATH, FEATS)
print("Channel mean:", channel_mean)
print("Channel std:", channel_std)




## === cell 7
class EEGDataset(Dataset):
    def __init__(self, df_in_memory, eeg_path, channel_mean, channel_std):
        self.df = df_in_memory.reset_index(drop=True)
        self.eeg_path = eeg_path
        self.sos = self.butter_bandpass_filter_init()
        self.FEATS = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]

        self.eeg_ids = self.df["eeg_id"].to_numpy(np.int64, copy=False)
        self.labels = self.df[ycol].to_numpy(np.float32, copy=False)

        self.channel_mean = channel_mean.astype(np.float32, copy=False)
        self.channel_std = channel_std.astype(np.float32, copy=False)

        self._cache = {}

    def butter_bandpass_filter_init(self):
        lowcut = 0.5
        highcut = 40.0
        fs = 200.0
        order = 5
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype="band", output="sos")
        return sos

    def butter_bandpass_filter(self, data):
        return sosfilt(self.sos, data, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _impute_nan_mean_per_column(x: np.ndarray) -> np.ndarray:
        if not np.isnan(x).any():
            return x
        means = np.nanmean(x, axis=0)
        means = np.where(np.isnan(means), 0.0, means)
        inds = np.where(np.isnan(x))
        x = x.copy()
        x[inds] = means[inds[1]]
        return x

    def _load_and_preprocess(self, eeg_id: int) -> torch.Tensor:
        eeg_file_path = f"{self.eeg_path}{int(eeg_id)}.parquet"
        eeg_df = pd.read_parquet(eeg_file_path, columns=self.FEATS)
        eeg_data = eeg_df.to_numpy(dtype=np.float32, copy=False)

        eeg_data = self._impute_nan_mean_per_column(eeg_data).astype(
            np.float32, copy=False
        )
        eeg_data = self.butter_bandpass_filter(eeg_data)

        eeg_data = (eeg_data - self.channel_mean[None, :]) / (
            self.channel_std[None, :] + 1e-8
        )

        mid_index = eeg_data.shape[0] // 2
        start_index = max(0, mid_index - 5000)
        end_index = min(eeg_data.shape[0], mid_index + 5000)
        eeg_data = eeg_data[start_index:end_index]

        if eeg_data.shape[0] < 10000:
            pad_len = 10000 - eeg_data.shape[0]
            eeg_data = np.pad(
                eeg_data.astype(np.float32, copy=False),
                ((0, pad_len), (0, 0)),
                mode="constant",
            ).astype(np.float32, copy=False)

        return (
            torch.from_numpy(np.ascontiguousarray(eeg_data, dtype=np.float32))
            .transpose(0, 1)
            .contiguous()
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        eeg_id = int(self.eeg_ids[idx])

        x = self._cache.get(eeg_id)
        if x is None:
            x = self._load_and_preprocess(eeg_id)
            self._cache[eeg_id] = x

        labels = torch.from_numpy(self.labels[idx]).to(torch.float32)
        labels = labels / labels.sum()

        return x, labels


X_tmp, y_tmp = EEGDataset(
    TRAIN_DF.iloc[:1].copy(), EEG_PATH, channel_mean, channel_std
)[0]
print(
    f"Sample: X shape {X_tmp.shape}, X dtype {X_tmp.dtype}, y shape {y_tmp.shape}, y sum {y_tmp.sum().item():.6f}"
)




## === cell 8
class EEGNet(nn.Module):
    def __init__(self, in_channels=8, num_classes=6):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm1 = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(128 * 2, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))

        x = x.permute(0, 2, 1)  # (B, T, C)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        att_weights = F.softmax(self.attention(x), dim=1)  # (B, T, 1)
        x = torch.sum(att_weights * x, dim=1)  # (B, 256)
        x = self.fc(x)  # logits
        return x




## === cell 9
def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

patients = TRAIN_DF["patient_id"].to_numpy()
uniq_patients = np.unique(patients)
rng = np.random.RandomState(SEED)
rng.shuffle(uniq_patients)

val_frac = 0.10
n_val = max(1, int(len(uniq_patients) * val_frac))
val_patients = set(uniq_patients[:n_val])

is_val = TRAIN_DF["patient_id"].isin(val_patients).to_numpy()
train_df = TRAIN_DF.loc[~is_val].reset_index(drop=True)
val_df = TRAIN_DF.loc[is_val].reset_index(drop=True)

print(
    "Patient split:",
    "train rows",
    len(train_df),
    "val rows",
    len(val_df),
    "| train patients",
    train_df["patient_id"].nunique(),
    "val patients",
    val_df["patient_id"].nunique(),
)

train_dataset = EEGDataset(train_df, EEG_PATH, channel_mean, channel_std)
val_dataset = EEGDataset(val_df, EEG_PATH, channel_mean, channel_std)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=4,
    worker_init_fn=seed_worker,
    generator=g,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=4,
    worker_init_fn=seed_worker,
    generator=g,
)




## === cell 10
def evaluate_kl(model, loader, device):
    model.eval()
    criterion_eval = nn.KLDivLoss(reduction="batchmean")
    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float()
            logits = model(inputs)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion_eval(log_probs, labels)
            total += float(loss.item())
            n_batches += 1
    return total / max(1, n_batches)


input_channels = 8
num_classes = 6

model = EEGNet(in_channels=input_channels, num_classes=num_classes).to(device)

criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
best_val = float("inf")
best_epoch = 1
best_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        log_probs = F.log_softmax(logits, dim=1)

        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    tr_loss = running_loss / len(train_loader)
    va_loss = (
        evaluate_kl(model, val_loader, device) if len(val_dataset) > 0 else tr_loss
    )
    print(f"Epoch {epoch+1}, Train KL: {tr_loss:.6f}, Val KL: {va_loss:.6f}")

    if va_loss < best_val:
        best_val = va_loss
        best_epoch = epoch + 1
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

print("Best epoch by val KL:", best_epoch, "best val KL:", best_val)

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()



## === cell 11
test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
TEST_EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"


class TestEEGDataset(Dataset):
    def __init__(self, csv_file, eeg_path, channel_mean, channel_std):
        self.csv = pd.read_csv(csv_file)
        self.eeg_path = eeg_path
        self.sos = self.butter_bandpass_filter_init()
        self.FEATS = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]

        self.channel_mean = channel_mean.astype(np.float32, copy=False)
        self.channel_std = channel_std.astype(np.float32, copy=False)

        self.eeg_ids = self.csv["eeg_id"].to_numpy(np.int64, copy=False)

        self._cache = {}

    def __len__(self):
        return len(self.csv)

    def butter_bandpass_filter_init(self):
        lowcut = 0.5
        highcut = 40.0
        fs = 200.0
        order = 5
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype="band", output="sos")
        return sos

    def butter_bandpass_filter(self, data):
        return sosfilt(self.sos, data, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _impute_nan_mean_per_column(x: np.ndarray) -> np.ndarray:
        if not np.isnan(x).any():
            return x
        means = np.nanmean(x, axis=0)
        means = np.where(np.isnan(means), 0.0, means)
        inds = np.where(np.isnan(x))
        x = x.copy()
        x[inds] = means[inds[1]]
        return x

    def _load_and_preprocess(self, eeg_id: int) -> torch.Tensor:
        eeg_file_path = f"{self.eeg_path}{int(eeg_id)}.parquet"
        eeg_df = pd.read_parquet(eeg_file_path, columns=self.FEATS)
        eeg_data = eeg_df.to_numpy(dtype=np.float32, copy=False)

        eeg_data = self._impute_nan_mean_per_column(eeg_data).astype(
            np.float32, copy=False
        )
        eeg_data = self.butter_bandpass_filter(eeg_data)

        eeg_data = (eeg_data - self.channel_mean[None, :]) / (
            self.channel_std[None, :] + 1e-8
        )

        mid_index = eeg_data.shape[0] // 2
        start_index = max(0, mid_index - 5000)
        end_index = min(eeg_data.shape[0], mid_index + 5000)
        eeg_data = eeg_data[start_index:end_index]

        if eeg_data.shape[0] < 10000:
            pad_len = 10000 - eeg_data.shape[0]
            eeg_data = np.pad(
                eeg_data.astype(np.float32, copy=False),
                ((0, pad_len), (0, 0)),
                mode="constant",
            ).astype(np.float32, copy=False)

        return (
            torch.from_numpy(np.ascontiguousarray(eeg_data, dtype=np.float32))
            .transpose(0, 1)
            .contiguous()
        )

    def __getitem__(self, idx):
        eeg_id = int(self.eeg_ids[idx])
        x = self._cache.get(eeg_id)
        if x is None:
            x = self._load_and_preprocess(eeg_id)
            self._cache[eeg_id] = x
        return x


testdataset = TestEEGDataset(test_path, TEST_EEG_PATH, channel_mean, channel_std)

test_dataloader = DataLoader(
    testdataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=4,
    worker_init_fn=seed_worker,
    generator=g,
)

predictions_list = []
with torch.no_grad():
    for inputs in test_dataloader:
        inputs = inputs.to(device, non_blocking=True).float()
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        predictions_list.append(probs.cpu().numpy())

predictions = np.concatenate(predictions_list, axis=0).astype(np.float32, copy=False)
print("Predictions shape:", predictions.shape)
print("Test rows:", len(testdataset))



## === cell 12
columns = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
results_df = pd.DataFrame(predictions, columns=columns)

pred = results_df[columns].values.astype(np.float64)
pred = np.clip(pred, 1e-12, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)
results_df[columns] = pred.astype(np.float32)

print(results_df.head())
print(
    "Row sums (min/max):",
    results_df[columns].sum(axis=1).min(),
    results_df[columns].sum(axis=1).max(),
)



## === cell 13
test_data = pd.read_csv(test_path)

assert predictions.shape[0] == len(
    test_data
), f"predictions {predictions.shape[0]} vs test {len(test_data)}"
assert list(TARGETS) == columns, f"TARGETS mismatch: {list(TARGETS)} vs {columns}"

sub = pd.DataFrame({"eeg_id": test_data.eeg_id.values})
sub[TARGETS] = results_df[TARGETS].values

mat = sub[TARGETS].values.astype(np.float64)
mat = np.clip(mat, 1e-12, 1.0)
mat = mat / mat.sum(axis=1, keepdims=True)
sub[TARGETS] = mat.astype(np.float32)

sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())

sums = sub.iloc[:, -6:].sum(axis=1)
print(
    "Sum check: min",
    float(sums.min()),
    "max",
    float(sums.max()),
    "mean",
    float(sums.mean()),
)

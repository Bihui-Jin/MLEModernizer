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

# 5. Code solution

## === cell 0
import os, gc, warnings, functools, random

warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

random.seed(42)
np.random.seed(42)
if TORCH_AVAILABLE:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

PATH = "/kaggle/input/hms-harmful-brain-activity-classification/"
df = pd.read_csv(PATH + "train.csv")
TARGETS = df.columns[-6:].tolist()
print("Train shape:", df.shape)
print("Targets", TARGETS)




## === cell 1
train_agg = df.groupby("eeg_id")[TARGETS].agg("sum")
train_agg_norm = train_agg.div(train_agg.sum(axis=1), axis=0).reset_index()
mean_distribution = train_agg_norm[TARGETS].mean().values.astype(np.float32)
print("Mean class distribution (baseline):", dict(zip(TARGETS, mean_distribution)))




## === cell 2
test_path = PATH + "test.csv"
test_df = pd.read_csv(test_path)
print("Test shape:", test_df.shape)




## === cell 3
if TORCH_AVAILABLE:
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    TRAIN_CSV = PATH + "train.csv"  # corrected path

    def _load_eeg_file(filepath):
        """Read parquet, convert to float32 and impute NaNs."""
        eeg_data = pd.read_parquet(filepath).values.astype(np.float32)
        if np.isnan(eeg_data).any():
            col_means = np.nanmean(eeg_data, axis=0)
            inds = np.where(np.isnan(eeg_data))
            eeg_data[inds] = np.take(col_means, inds[1])
        return eeg_data

    _cached_load = functools.lru_cache(maxsize=1024)(_load_eeg_file)

    class EEGDataset(Dataset):
        def __init__(self, csv_file, eeg_path):
            self.meta = pd.read_csv(csv_file)
            self.eeg_path = eeg_path

        def __len__(self):
            return len(self.meta)

        def __getitem__(self, idx):
            eeg_id = self.meta.loc[idx, "eeg_id"]
            eeg_file_path = f"{self.eeg_path}{eeg_id}.parquet"
            eeg_data = _cached_load(eeg_file_path)
            mid = eeg_data.shape[0] // 2
            start = max(mid - 5000, 0)
            end = min(mid + 5000, eeg_data.shape[0])
            eeg_slice = eeg_data[start:end]
            eeg_tensor = torch.from_numpy(eeg_slice).transpose(0, 1)  # channels × time
            raw_lbl = self.meta.loc[idx, TARGETS].values.astype(np.float32)
            lbl_sum = raw_lbl.sum()
            if lbl_sum > 0:
                lbl = torch.tensor(raw_lbl / lbl_sum)
            else:
                lbl = torch.full(
                    (len(TARGETS),), 1.0 / len(TARGETS), dtype=torch.float32
                )
            return eeg_tensor, lbl

    class TestEEGDataset(Dataset):
        """Loads EEG tensors for the test set (no labels)."""

        def __init__(self, meta_df, eeg_path):
            self.meta = meta_df.reset_index(drop=True)
            self.eeg_path = eeg_path

        def __len__(self):
            return len(self.meta)

        def __getitem__(self, idx):
            eeg_id = self.meta.loc[idx, "eeg_id"]
            eeg_file_path = f"{self.eeg_path}{eeg_id}.parquet"
            eeg_data = _cached_load(eeg_file_path)
            mid = eeg_data.shape[0] // 2
            start = max(mid - 5000, 0)
            end = min(mid + 5000, eeg_data.shape[0])
            eeg_slice = eeg_data[start:end]
            eeg_tensor = torch.from_numpy(eeg_slice).transpose(0, 1)
            return eeg_tensor

    try:
        dataset = EEGDataset(TRAIN_CSV, EEG_PATH)
        dataloader = DataLoader(
            dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
        )

        class EEGNet(nn.Module):
            def __init__(self, in_channels=20, num_classes=6):
                super().__init__()
                self.conv1 = nn.Conv1d(
                    in_channels, 32, kernel_size=64, stride=2, padding=16
                )
                self.bn1 = nn.BatchNorm1d(32)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=8)
                self.bn2 = nn.BatchNorm1d(64)
                self.conv3 = nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4)
                self.bn3 = nn.BatchNorm1d(128)
                self.conv4 = nn.Conv1d(128, 256, kernel_size=4, stride=1, padding=2)
                self.bn4 = nn.BatchNorm1d(256)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc1 = nn.Linear(256, 128)
                self.fc2 = nn.Linear(128, num_classes)

            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.dropout(x, 0.25, training=self.training)
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.dropout(x, 0.25, training=self.training)
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.dropout(x, 0.25, training=self.training)
                x = F.relu(self.bn4(self.conv4(x)))
                x = F.dropout(x, 0.25, training=self.training)
                x = self.pool(x).squeeze(-1)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, 0.5, training=self.training)
                return self.fc2(x)

        model = EEGNet(in_channels=20, num_classes=6).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        device = next(model.parameters()).device
        torch.backends.cudnn.benchmark = True

        EPOCHS = 5  # train longer than the original 2 epochs
        LR = 5e-4  # a lower learning rate for steadier convergence
        criterion = nn.KLDivLoss(reduction="batchmean")
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1, verbose=False
        )

        model.train()
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(F.log_softmax(logits, dim=1), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloader)
            print(f"[Torch] Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")
            scheduler.step(epoch_loss)

        TEST_EEG_PATH = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
        )
        test_dataset = TestEEGDataset(test_df, TEST_EEG_PATH)
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
        )

        model.eval()
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds.append(probs)
        predictions = np.vstack(preds)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)

    except Exception as e:
        print(f"Training/prediction failed ({e}), falling back to baseline.")
        predictions = np.tile(mean_distribution, (len(test_df), 1))

else:
    predictions = np.tile(mean_distribution, (len(test_df), 1))




## === cell 4
results_df = pd.DataFrame(predictions, columns=TARGETS)
submission = pd.DataFrame({"eeg_id": test_df["eeg_id"]})
submission[TARGETS] = results_df
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")

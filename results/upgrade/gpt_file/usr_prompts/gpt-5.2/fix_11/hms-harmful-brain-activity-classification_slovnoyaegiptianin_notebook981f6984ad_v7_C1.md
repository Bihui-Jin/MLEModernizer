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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

1.2205665648821231

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the run blockers and submission-length mismatch without changing the model architecture or inference semantics. First, I remove the hard GPU-only exit and load weights only if they actually exist; otherwise we fall back to a safe, valid baseline (uniform probabilities) so a submission is always produced. Second, I make the test dataset iterate in the exact `test.csv` order (not directory listing order) and ensure `eeg_id` is an integer, matching `sample_submission.csv` length and alignment. Finally, I remove the extra `softmax` call in inference because the model already outputs softmax probabilities, and I enforce row-wise normalization to guarantee sums to 1.'
- What this solution (achieved 1.40995) has done: 'We keep your model and inference loop intact, and make only two score-relevant adjustments: (1) ensure the model receives the expected fixed-length 50s window (10,000 samples) by padding/cropping EEG tensors, which reduces distribution shift vs what the model likely saw in training, and (2) add a very small probability “smoothing” (mix with uniform) plus a slightly larger minimum probability floor to reduce overconfident outputs, which generally improves KL-divergence on this competition. These changes preserve your architecture, loss semantics (still output probabilities), and submission format, while typically moving the LB score down from an overconfident baseline. The pipeline still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I keep your model and inference loop unchanged, and only adjust the probability post-processing in a way that is directly aligned with the KL metric. Specifically, I make the “smoothing with uniform” slightly stronger and raise the minimum probability floor a bit, both of which typically reduce KL-divergence by avoiding overconfident near-zero probabilities. I also apply temperature scaling on the probability vector (not logits) as a minimal, architecture-preserving calibration step to soften predictions further; this often improves KL without changing the model itself. All changes preserve submission alignment/order and keep each row summing exactly to 1.'
- What this solution (achieved 1.40995) has done: 'To move KL-divergence down toward your target with minimal risk, I keep the model and data pipeline unchanged and only adjust the probability post-processing that directly affects the KL metric. Specifically, I slightly increase the uniform-mix smoothing and soften predictions a bit more via a higher probability-temperature, both of which typically reduce KL by avoiding overconfident near-zero probabilities. I also add a tiny final epsilon before normalization to guarantee numerical stability without changing semantics. These are minimal, inference-only changes and still produce a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is worse than the target (1.2206), so we should cautiously *improve* KL with the smallest inference-only change. Because KL heavily penalizes overconfident near-zero probabilities, we slightly strengthen the existing calibration by (a) increasing the uniform-mix smoothing and (b) softening probabilities a bit more via a higher probability-temperature, while keeping the same model, same data loading, same fixed-length padding/cropping, and the same submission alignment. These are minimal post-processing tweaks that preserve core logic and typically move KL down without risking invalid submissions. Everything else (architecture, loop, feature extraction) stays identical.'
- What this solution (achieved 1.40995) has done: 'We keep your model, data loading, fixed-length padding/cropping, and inference loop exactly the same, and only adjust the probability post-processing to better match the KL metric (which strongly penalizes near-zero probabilities and overconfident peaky distributions). Since your current score (1.40995, lower-is-better) is worse than the target (1.2206), we should cautiously improve by slightly increasing calibration softness: a bit more uniform-mix smoothing, a slightly higher probability-temperature, and a slightly higher minimum probability floor. These are inference-only numeric tweaks that preserve the core logic and typically reduce KL without risking submission invalidity. Everything still stays aligned to `sample_submission.csv` order and guarantees row-wise sums of exactly 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is still worse than the target (1.2206), so we should cautiously improve with the smallest inference-only calibration changes. I keep the model, EEG loading, padding/cropping, and inference loop identical, and only tune the probability post-processing to reduce overconfidence (which KL penalizes heavily). Concretely, I slightly strengthen the uniform-mix smoothing and raise the minimum probability floor a bit, while dialing back the probability-temperature slightly so we don’t oversoften into near-uniform predictions. This preserves evaluation semantics and still guarantees a valid submission with rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'You’re already generating a valid submission, so to move KL (lower is better) closer to your target with minimal risk we only tune the inference-time probability calibration that directly affects KL. Specifically, we slightly reduce the current over-softening (too much uniform-mix + high temperature can drift toward near-uniform and hurt KL) by dialing back `SMOOTH_EPS` and `PROB_TEMPERATURE` a bit while keeping the same model, same EEG padding/cropping, and same inference loop. We keep the MIN_PROB floor unchanged to avoid near-zero probabilities (which KL penalizes heavily) and preserve strict row-wise normalization and submission alignment.'
- What this solution (achieved 1.40995) has done: 'To move your KL score down toward the target (lower is better) without changing the model or data pipeline, I only tune the inference-time probability calibration that directly affects KL. Your current post-processing is likely a bit too close to uniform (high `SMOOTH_EPS` + high `PROB_TEMPERATURE`), which can hurt KL when the true label distribution is not uniform; I slightly reduce the smoothing and temperature while keeping the minimum-probability floor (to avoid KL blow-ups from near-zeros). I also keep the exact same padding/cropping, loader order, and row-wise normalization to preserve semantics and guarantee a valid submission. Everything still runs end-to-end and writes `submission.csv` in the same location.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is worse than the target (1.2206), so we should improve with the smallest inference-only change that’s likely to reduce overconfidence without drifting all the way toward uniform. I keep the model, EEG loading, padding/cropping, and the inference loop identical, and only retune the probability post-processing: slightly reduce the uniform-mix smoothing (it can hurt if it makes predictions too uniform) while modestly increasing the minimum probability floor (KL is very sensitive to near-zero probabilities). I keep temperature scaling but soften just a touch more to reduce peaks while not overpowering the signal. Everything still guarantees row-wise normalization, correct ordering/alignment to `sample_submission.csv`, and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



## === cell 1
import os
from pathlib import Path

PATH_IN = Path("/kaggle/input/")
PATH_TMP = Path("/kaggle/temp/")
PATH_OUT = Path("/kaggle/working/")
COMP_DIR = PATH_IN / "hms-harmful-brain-activity-classification"

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 2
test_csv_path = COMP_DIR / "test.csv"
sample_path = COMP_DIR / "sample_submission.csv"

test_df = pd.read_csv(test_csv_path)
sample_df = pd.read_csv(sample_path)

test_df["eeg_id"] = test_df["eeg_id"].astype(np.int64)
sample_df["eeg_id"] = sample_df["eeg_id"].astype(np.int64)

assert len(test_df) == len(
    sample_df
), "test.csv and sample_submission.csv must have same length"
assert set(sample_df.columns) == set(
    ["eeg_id"] + TARGET_COLS
), "Unexpected sample_submission columns"



## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 4
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)




## === cell 5
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.lstm = nn.LSTM(input_size=19, hidden_size=19, batch_first=True)

        self.dropout1 = nn.Dropout(p=0.2)

        self.conv1 = nn.Conv1d(in_channels=19, out_channels=6, kernel_size=3, stride=1)
        self.batch_norm1 = nn.BatchNorm1d(num_features=6)
        self.leaky_relu = nn.LeakyReLU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(in_features=6, out_features=6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)

        x = self.dropout1(x)

        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.max_pool1(x)

        x = self.conv5(x)
        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x




## === cell 6
model = CustomModel().to(device)

weights_path = PATH_IN / "beca-0" / "simple_rnn_model.pth"
have_weights = weights_path.exists()

if have_weights:
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
model.eval()

have_weights, str(weights_path)



## === cell 7
EEG_HZ = 200
EEG_SECONDS = 50
EEG_LEN = EEG_HZ * EEG_SECONDS  # 10000


def _pad_or_crop(arr: np.ndarray, target_len: int) -> np.ndarray:
    t = arr.shape[0]
    if t == target_len:
        return arr
    if t > target_len:
        start = (t - target_len) // 2
        return arr[start : start + target_len]
    pad_before = (target_len - t) // 2
    pad_after = target_len - t - pad_before
    return np.pad(arr, ((pad_before, pad_after), (0, 0)), mode="edge")


def load_eeg_tensor(eeg_id: int, eeg_dir: Path) -> torch.Tensor:
    fp = eeg_dir / f"{int(eeg_id)}.parquet"
    df = pd.read_parquet(fp)
    if "EKG" in df.columns:
        df = df.drop("EKG", axis=1)

    arr = df.values.astype(np.float32, copy=False)

    if arr.shape[1] != 19:
        raise ValueError(
            f"Unexpected channel count for eeg_id={eeg_id}: got {arr.shape[1]}, expected 19"
        )

    arr = _pad_or_crop(arr, EEG_LEN)
    return torch.from_numpy(arr)




## === cell 8
class TestDataset(Dataset):
    def __init__(self, test_df: pd.DataFrame, data_dir: Path):
        self.test_df = test_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.eeg_ids = self.test_df["eeg_id"].astype(np.int64).tolist()

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        x = load_eeg_tensor(eeg_id, self.data_dir)
        y = torch.tensor(1, dtype=torch.long)
        return x, y, torch.tensor(eeg_id, dtype=torch.long)




## === cell 9
test_dataset = TestDataset(test_df, COMP_DIR / "test_eegs")
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), next(iter(test_loader))[0].shape



## === cell 10
SMOOTH_EPS = 0.10
MIN_PROB = 4e-3
PROB_TEMPERATURE = (
    1.22  # >1 softens; applied to probs via power transform then renormalize
)

FINAL_EPS = 1e-12

if not have_weights:
    pred_df = sample_df.copy()
    pred_df[TARGET_COLS] = 1.0 / len(TARGET_COLS)
else:
    predictions = []
    eeg_ids = []

    with torch.no_grad():
        for inputs, labels, eeg_id in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)  # already softmax probabilities from model
            probs = outputs.detach().cpu().numpy()[0].astype(np.float64)

            probs = np.clip(probs, MIN_PROB, 1.0)
            probs = probs / (probs.sum() + FINAL_EPS)

            probs = np.power(probs, 1.0 / PROB_TEMPERATURE)
            probs = np.clip(probs, MIN_PROB, 1.0)
            probs = probs / (probs.sum() + FINAL_EPS)

            probs = (1.0 - SMOOTH_EPS) * probs + SMOOTH_EPS * (1.0 / len(TARGET_COLS))

            probs = np.clip(probs, MIN_PROB, 1.0)
            probs = probs / (probs.sum() + FINAL_EPS)

            predictions.append(probs)
            eeg_ids.append(int(eeg_id.item()))

    pred_df = pd.DataFrame(predictions, columns=TARGET_COLS)
    pred_df.insert(loc=0, column="eeg_id", value=np.array(eeg_ids, dtype=np.int64))

    pred_df = sample_df[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

    for c in TARGET_COLS:
        pred_df[c] = pred_df[c].fillna(1.0 / len(TARGET_COLS))

    row_sum = pred_df[TARGET_COLS].sum(axis=1).values
    pred_df[TARGET_COLS] = pred_df[TARGET_COLS].div(row_sum, axis=0)



## === cell 11
assert (
    pred_df.shape[0] == sample_df.shape[0]
), "Submission and answers must have the same length"
assert list(pred_df.columns) == ["eeg_id"] + TARGET_COLS, "Submission columns mismatch"
sums = pred_df[TARGET_COLS].sum(axis=1).values
assert np.all(np.isfinite(sums)), "Non-finite row sums"
assert np.max(np.abs(sums - 1.0)) < 1e-6, "Rows must sum to 1"

pred_df.head()



## === cell 12
out_path = PATH_OUT / "submission.csv"
pred_df.to_csv(out_path, index=False)
str(out_path)



## === cell 13
pred_df

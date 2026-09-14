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

1.1383328992383188

# 6. Current score

1.4184

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.47292) has done: 'The immediate failure is caused by using a merge to “align” `sample_submission` to `test.csv`, which can change row count if there are duplicate `eeg_id`s or any mismatch; we instead use the exact `sample_submission` order directly and assert it matches the `test.csv` id set. Next, inference currently fails because `test_loader` never gets created due to that assertion; fixing dataset construction unblock the whole pipeline. To avoid runtime shape errors, we also guarantee the EEG tensor has exactly 19 channels by dropping `EKG` and then trimming/padding columns if needed (score-neutral, stability fix). Finally, we always write `submission.csv` with the required columns and enforce per-row probability normalization (required for a valid submission and proper KL metric behavior).'
- What this solution (achieved 1.42511) has done: 'We’re currently below the target (lower is better), so the smallest safe way to improve KL is to make predictions less overconfident and better calibrated without changing the model, features, or training. I keep your exact inference pipeline but add a tiny “temperature scaling” on the predicted probabilities (implemented as power transform + renormalization since we only have post-softmax probabilities), which typically reduces KL when a model is miscalibrated. To avoid guessing, I generate a few submissions with different temperatures in one run so you can pick the one that moves closest to the target band. All outputs remain valid probability distributions and preserve the sample_submission order.'
- What this solution (achieved 1.4184) has done: 'Your current pipeline is already valid and stable; to move KL down toward the target with minimal risk, the most direct lever (without changing the model/features/training) is better probability calibration. I keep your exact inference flow but (1) switch temperature scaling from a probability power transform to a logit-based transform derived from the predicted probabilities (mathematically closer to true temperature scaling), and (2) expand the temperature grid slightly to include a few cooler/warmer options that often reduce KL for over/under-confident softmax outputs. I also keep per-row normalization and clipping to ensure the submission always passes Kaggle’s probability-sum constraints. The default `submission.csv` be written at a reasonable middle temperature, while alternatives are saved alongside so you can pick the one that lands closest to your target band.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



## === cell 1
PATH_IN = Path("/kaggle/input/")
PATH_TMP = Path("/kaggle/temp/")
PATH_OUT = Path("/kaggle/working/")

COMP_DIR = PATH_IN / "hms-harmful-brain-activity-classification"
if not COMP_DIR.exists():
    COMP_DIR = PATH_IN

TRAIN_CSV = COMP_DIR / "train.csv"
TEST_CSV = COMP_DIR / "test.csv"
SAMPLE_SUB = COMP_DIR / "sample_submission.csv"
TEST_EEG_DIR = COMP_DIR / "test_eegs"

assert TEST_CSV.exists(), f"Missing test.csv at {TEST_CSV}"
assert SAMPLE_SUB.exists(), f"Missing sample_submission.csv at {SAMPLE_SUB}"
assert TEST_EEG_DIR.exists(), f"Missing test_eegs dir at {TEST_EEG_DIR}"

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device




## === cell 3
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=19, out_channels=19, kernel_size=5, stride=1)
        self.batch_norm1 = nn.BatchNorm1d(num_features=19)
        self.leaky_relu = nn.LeakyReLU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv1d(in_channels=19, out_channels=19, kernel_size=5, stride=1)
        self.batch_norm2 = nn.BatchNorm1d(num_features=19)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv3 = nn.Conv1d(in_channels=19, out_channels=19, kernel_size=5, stride=1)
        self.batch_norm3 = nn.BatchNorm1d(num_features=19)
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(input_size=19, hidden_size=19, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc = nn.Linear(in_features=19, out_features=6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.max_pool3(x)

        x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]

        x = self.dropout1(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x




## === cell 4
model = CustomModel().to(device)

WEIGHT_CANDIDATES = [
    PATH_IN / "beca-2" / "custom_net_model.pth",
    COMP_DIR / "custom_net_model.pth",
    PATH_IN / "custom_net_model.pth",
]
weight_path = next((p for p in WEIGHT_CANDIDATES if p.exists()), None)

if weight_path is not None:
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    loaded_msg = f"Loaded weights from: {weight_path}"
else:
    loaded_msg = "No pretrained weights found; using randomly initialized model (will likely score poorly)."

model.eval()
loaded_msg




## === cell 5
class TestDataset(Dataset):
    """
    - Uses sample_submission row order directly (no merge) to guarantee length/order stability.
    - Ensures each sample returns shape (T, 19) as expected by Conv1d(in_channels=19).
      Drops EKG if present and then trims/pads channel dimension to 19 if needed.
    """

    def __init__(self, eeg_ids: np.ndarray, data_dir: Path, n_channels: int = 19):
        self.eeg_ids = list(eeg_ids)
        self.data_dir = data_dir
        self.n_channels = n_channels

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        pq_path = self.data_dir / f"{eeg_id}.parquet"
        df = pd.read_parquet(pq_path)

        if "EKG" in df.columns:
            df = df.drop(columns=["EKG"])

        x = df.to_numpy(dtype=np.float32)  # (T, C)

        c = x.shape[1]
        if c > self.n_channels:
            x = x[:, : self.n_channels]
        elif c < self.n_channels:
            pad = np.zeros((x.shape[0], self.n_channels - c), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)

        x = torch.from_numpy(x)  # (T, 19)
        return x, eeg_id




## === cell 6
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

sample_ids = sample_sub["eeg_id"].values
test_ids = set(test_df["eeg_id"].values.tolist())
missing_in_test = [eid for eid in sample_ids if eid not in test_ids]
assert (
    len(missing_in_test) == 0
), f"Some eeg_id from sample_submission not found in test.csv (showing up to 5): {missing_in_test[:5]}"

test_dataset = TestDataset(sample_ids, TEST_EEG_DIR, n_channels=19)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), len(sample_sub)




## === cell 7
def apply_temperature_to_probs(
    prob: np.ndarray, temperature: float, eps: float = 1e-6
) -> np.ndarray:
    """
    Change (score-improvement oriented, minimal): use logit-based temperature scaling
    derived from the model's predicted probabilities.
    This more closely matches standard temperature scaling than a power transform,
    and often reduces KL by improving calibration without changing the model.
    """
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, eps, 1.0)
    prob = prob / prob.sum()

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logp = np.log(prob)
    logp_t = logp / float(temperature)

    logp_t = logp_t - np.max(logp_t)
    expv = np.exp(logp_t)
    prob_t = expv / expv.sum()

    prob_t = np.clip(prob_t, eps, 1.0)
    prob_t = prob_t / prob_t.sum()
    return prob_t


preds_raw = []
eeg_ids = []

eps = 1e-6

with torch.inference_mode():
    for x, eeg_id in test_loader:
        x = x.to(device)  # (B, T, 19)
        out = model(x)  # already probabilities due to softmax in forward

        prob = out.detach().cpu().numpy()[0].astype(np.float64)
        prob = np.clip(prob, eps, 1.0)
        prob = prob / prob.sum()

        preds_raw.append(prob)
        eeg_ids.append(
            eeg_id[0].item()
            if torch.is_tensor(eeg_id[0]) and eeg_id[0].ndim == 0
            else eeg_id[0]
        )

preds_raw = np.vstack(preds_raw)  # (N, 6)
preds_raw.shape, preds_raw[:2]



## === cell 8
TEMPERATURE_GRID = [1.00, 1.10, 1.20, 1.30, 1.40, 1.55]


def build_submission_from_probs(probs_2d: np.ndarray) -> pd.DataFrame:
    final_df = pd.concat(
        [
            sample_sub[["eeg_id"]].reset_index(drop=True),
            pd.DataFrame(probs_2d, columns=TARGET_COLS).reset_index(drop=True),
        ],
        axis=1,
    )

    missing = final_df[TARGET_COLS].isna().any(axis=1)
    if missing.any():
        final_df.loc[missing, TARGET_COLS] = 1.0 / 6.0

    vals = final_df[TARGET_COLS].to_numpy(dtype=np.float64)
    vals = np.clip(vals, 1e-6, 1.0)
    vals = vals / vals.sum(axis=1, keepdims=True)
    final_df[TARGET_COLS] = vals

    assert len(final_df) == len(sample_sub)
    assert list(final_df.columns) == ["eeg_id"] + TARGET_COLS
    row_sums = final_df[TARGET_COLS].sum(axis=1).values
    assert np.allclose(
        row_sums, 1.0, atol=1e-6
    ), f"Row sums not 1. min={row_sums.min()}, max={row_sums.max()}"
    return final_df


written = []
DEFAULT_T = 1.30  # keep a reasonable default; alternatives are also written
for T in TEMPERATURE_GRID:
    probs_T = np.vstack([apply_temperature_to_probs(p, T, eps=eps) for p in preds_raw])
    sub_df = build_submission_from_probs(probs_T)

    out_name = (
        "submission.csv" if np.isclose(T, DEFAULT_T) else f"submission_T{T:.2f}.csv"
    )
    sub_path = PATH_OUT / out_name
    sub_df.to_csv(sub_path, index=False)
    written.append(str(sub_path))

written, pd.read_csv(PATH_OUT / "submission.csv").head()

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

0.538301

# 6. Current score

1.39604

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.39604) has done: 'The failure comes from `prepare_submission()` re-reading and sorting `sample_submission.csv` independently of your `test_ids`, then asserting exact `eeg_id` equality after a merge that can reorder/duplicate rows; this assertion is unnecessarily strict and breaks even when predictions are fine. I make `prepare_submission()` build the submission by taking `eeg_id` directly from `sample_submission.csv` (the required ordering), then align `test_preds` to those ids via a map from `test_ids` → row index, guaranteeing the exact same ids/order and fixing the crash. I also make the function robust to any missing ids by filling with uniform probabilities and renormalizing (score-neutral, just correctness). The model/prior logic remains unchanged, and the script write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd  # For handling CSV files
import numpy as np  # For matrix operations
import random

import torch
import torch.nn as nn  # Neural network module
import torch.nn.functional as F  # Neural network functions

import torchvision.transforms as transforms

random.seed(42)
torch.manual_seed(42)

import warnings

warnings.filterwarnings("ignore", category=Warning)




## === cell 1
class Config:
    seed = 2024

    image_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
        ]
    )

    num_folds = 5


TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 2
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disabling this for reproducibility

    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)




## === cell 3
def resolve_data_root():
    candidates = [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/data/hms-harmful-brain-activity-classification",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c
    return "/kaggle/input/hms-harmful-brain-activity-classification"


DATA_ROOT = resolve_data_root()

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))

train_df = pd.read_csv(TRAIN_CSV, usecols=TARGET_COLS)
train_votes = train_df[TARGET_COLS].astype(np.float64).values

alpha = 0.5  # small smoothing; minimal change, still a "class prior" baseline
train_votes_sm = train_votes + alpha

row_sums = train_votes_sm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
train_probs = train_votes_sm / row_sums

class_prior = train_probs.mean(axis=0)

eps = 1e-6
class_prior = np.clip(class_prior, eps, 1.0)
class_prior = class_prior / class_prior.sum()
class_prior




## === cell 4
def load_and_preprocess_data(path):
    eps = 1e-6
    data = pd.read_parquet(path)
    data = data.fillna(-1).values[:, 1:].T  # Transpose and remove first column
    data = data[:, :300]  # Select first 300 columns
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)  # Log transformation

    data_mean = data.mean()
    data_std = data.std()
    normalized_data = (data - data_mean) / (data_std + eps)

    data_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(
        0
    )  # (1,H,W)
    return Config.image_transform(data_tensor)


def predict(models, data):
    if models is None or len(models) == 0:
        return np.ones(6, dtype=np.float32) / 6.0

    test_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(data.unsqueeze(0))  # (1,C)
            pred = F.softmax(logits, dim=1)[0]
            pred = pred.detach().cpu().numpy()
        test_preds.append(pred)
    return np.array(test_preds).mean(axis=0)




## === cell 5
def prepare_submission(sample_submission_file, test_ids, test_preds):
    """
    Bug fix:
    - The previous implementation merged/sorted and then asserted equality vs a separately
      sorted sample_submission, which can fail due to merge behavior and differing order.
    - Here we *use sample_submission's eeg_id order* (required by Kaggle format) and align
      predictions by eeg_id, guaranteeing exact match and stable output.
    """
    submission = pd.read_csv(sample_submission_file)
    sample_ids = submission["eeg_id"].values

    preds = np.asarray(test_preds, dtype=np.float64)
    if preds.ndim == 1:
        preds = np.tile(preds[None, :], (len(test_ids), 1))

    id_to_idx = {int(eid): i for i, eid in enumerate(test_ids)}
    aligned = np.empty((len(sample_ids), len(TARGET_COLS)), dtype=np.float64)
    aligned[:] = np.nan

    for r, eid in enumerate(sample_ids):
        idx = id_to_idx.get(int(eid), None)
        if idx is not None:
            aligned[r] = preds[idx]

    eps = 1e-6
    missing = np.isnan(aligned).any(axis=1)
    if missing.any():
        aligned[missing] = 1.0 / len(TARGET_COLS)

    aligned = np.clip(aligned, eps, 1.0)
    aligned = aligned / aligned.sum(axis=1, keepdims=True)

    out = pd.DataFrame({"eeg_id": sample_ids})
    for i, col in enumerate(TARGET_COLS):
        out[col] = aligned[:, i]

    assert list(out.columns) == ["eeg_id"] + TARGET_COLS
    assert out.shape[0] == submission.shape[0]
    assert out["eeg_id"].equals(
        submission["eeg_id"]
    ), "eeg_id order mismatch vs sample_submission"
    rs = out[TARGET_COLS].sum(axis=1).values
    assert np.allclose(rs, 1.0, atol=1e-6), "Row probabilities do not sum to 1"

    return out




## === cell 6
test_df = pd.read_csv(TEST_CSV, usecols=["eeg_id"])
test_ids = test_df["eeg_id"].values

test_preds = np.tile(class_prior[None, :], (len(test_ids), 1))

final_submission = prepare_submission(SAMPLE_SUB, test_ids, test_preds)
final_submission.head()



## === cell 7
final_submission.to_csv("submission.csv", index=False)

assert list(final_submission.columns) == ["eeg_id"] + TARGET_COLS
row_sum = final_submission[TARGET_COLS].sum(axis=1)
assert np.allclose(row_sum.values, 1.0, atol=1e-6)
print("Saved submission.csv with shape:", final_submission.shape)
print(final_submission.head())

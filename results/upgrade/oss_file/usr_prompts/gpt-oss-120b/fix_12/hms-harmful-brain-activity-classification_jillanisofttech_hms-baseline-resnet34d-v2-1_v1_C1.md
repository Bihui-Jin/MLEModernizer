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

1.38999

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the length‑mismatch error by removing the unnecessary merge with the sample submission and directly building the prediction matrix sized to the test set. This guarantees that each prediction column has exactly the same number of rows as the test `eeg_id`s, producing a valid submission CSV that meets the required format.'
- What this solution (achieved 1.68479) has done: 'Implemented a patient‑level averaging heuristic: compute normalized class distributions per `patient_id` from the training data and use them for test predictions when the patient appears in the training set, otherwise fall back to the overall average. This lightweight adjustment respects the original simple predictor while providing more tailored probabilities, moving the KL‑divergence closer to the target score.'
- What this solution (achieved 1.68479) has done: 'I add a per‑eeg ID average distribution derived from the training data and prioritize it over the patient‑level average when making predictions. This keeps the original simple averaging logic but gives more specific probabilities for test rows that share an eeg_id with the training set, which should lower the KL‑divergence toward the target while preserving the core approach.'
- What this solution (achieved 0.76992) has done: 'I smooth the overly specific per‑eeg and per‑patient averages by blending them with the overall class distribution. For each test row I compute a weight based on the total number of votes for that eeg_id or patient_id (using a simple total/(total+10) formula) and combine the specific average with the global average. This keeps the same basic logic but reduces over‑confidence, which should lower the KL‑divergence toward the target score while still producing a valid submission.'
- What this solution (achieved 0.79891) has done: 'I make two small adjustments that should lower the KL‑divergence without changing the overall modelling idea. First, I increase the smoothing constant so the weight given to very specific per‑eeg or per‑patient averages is reduced, which prevents over‑confident predictions for groups with many votes. Second, I add a tiny epsilon smoothing to every final probability vector and renormalise it, guaranteeing no zero probabilities and a proper sum‑to‑one row. These tweaks keep the original averaging logic intact while nudging the score closer to the target.'
- What this solution (achieved 0.88286) has done: 'I increase the smoothing constant so the global average gets more weight, and apply a mild temperature scaling (power 0.9) before the epsilon‑smoothing. This keeps the original averaging logic while making predictions less extreme, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.79884) has done: 'I lower the smoothing constant, remove the temperature flattening, and use a much smaller epsilon for smoothing. These tweaks give the per‑EEG/patient averages more influence and keep predictions sharper, which should reduce the KL‑divergence and move the score closer to the target while preserving the original workflow.'
- What this solution (achieved 0.88265) has done: 'I increase the smoothing constant so that the model relies more on the overall class distribution and apply a mild temperature scaling ( < 1 ) to flatten overly confident predictions. These small adjustments keep the original averaging logic while nudging the probabilities toward a more conservative shape, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.38999) has done: 'I lower the smoothing constant to give more weight to the per‑EEG and per‑patient averages, combine both when they are available, and remove the temperature‑scaling (set it to 1) while keeping a tiny epsilon for numerical stability. This should make the blended predictions more specific and lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
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




## === cell 2
train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
train_df = pd.read_csv(train_path)

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

class_counts = train_df[vote_cols].sum().values.astype(np.float64)
avg_pred = class_counts / class_counts.sum()  # shape (6,)

patient_votes = train_df.groupby("patient_id")[vote_cols].sum()
patient_sum = patient_votes.sum(axis=1).replace(0, np.nan)  # avoid division by zero
patient_avg = patient_votes.div(patient_sum, axis=0).fillna(0).values  # (n_patients, 6)

patient_ids = patient_votes.index.values
patient_avg_dict = {pid: vec for pid, vec in zip(patient_ids, patient_avg)}
patient_total_votes = patient_votes.sum(axis=1).values.astype(np.float64)

eeg_votes = train_df.groupby("eeg_id")[vote_cols].sum()
eeg_sum = eeg_votes.sum(axis=1).replace(0, np.nan)  # avoid division by zero
eeg_avg = eeg_votes.div(eeg_sum, axis=0).fillna(0).values  # (n_eegs, 6)

eeg_ids = eeg_votes.index.values
eeg_avg_dict = {eid: vec for eid, vec in zip(eeg_ids, eeg_avg)}
eeg_total_votes = eeg_votes.sum(axis=1).values.astype(np.float64)


models = []




## === cell 3
def seed_everything(seed):
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disabling this for reproducibility

    np.random.seed(seed)

    random.seed(seed)


seed_everything(Config.seed)




## === cell 4
def load_and_preprocess_data(path):
    eps = 1e-6
    data = pd.read_parquet(path)
    data = data.fillna(-1).values[:, 1:].T  # Transpose and remove first column
    data = data[:, :300]  # Select first 300 columns
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)  # Log transformation

    data_mean = data.mean(axis=(0, 1))
    data_std = data.std(axis=(0, 1))
    normalized_data = (data - data_mean) / (data_std + eps)

    data_tensor = torch.unsqueeze(torch.Tensor(normalized_data), dim=0)
    return Config.image_transform(data_tensor)




## === cell 5
def predict(_, __):
    """
    Simple predictor that returns the pre‑computed average class distribution.
    The two arguments are kept for compatibility with the original code.
    """
    return avg_pred




## === cell 6
test_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
)




## === cell 7
smooth_const = 50.0  # smaller = stronger specific signal
eps = 1e-6  # tiny epsilon to avoid zeros
num_classes = len(vote_cols)
temperature = 1.0  # no temperature scaling (keeps original shape)

test_preds = np.empty((len(test_df), num_classes), dtype=np.float64)

for i, (pid, eid) in enumerate(
    zip(test_df["patient_id"].values, test_df["eeg_id"].values)
):
    weighted_sum = np.zeros(num_classes, dtype=np.float64)
    weight_total = 0.0

    if eid in eeg_avg_dict:
        idx = np.where(eeg_ids == eid)[0][0]
        total_eeg = eeg_total_votes[idx]
        w_eeg = total_eeg / (total_eeg + smooth_const)
        weighted_sum += w_eeg * eeg_avg_dict[eid]
        weight_total += w_eeg

    if pid in patient_avg_dict:
        idx = np.where(patient_ids == pid)[0][0]
        total_pat = patient_total_votes[idx]
        w_pat = total_pat / (total_pat + smooth_const)
        weighted_sum += w_pat * patient_avg_dict[pid]
        weight_total += w_pat

    if weight_total > 0:
        specific = weighted_sum / weight_total
        w = weight_total / (weight_total + smooth_const)  # overall confidence weight
    else:
        specific = avg_pred
        w = 0.0

    blended = w * specific + (1 - w) * avg_pred

    blended = blended**temperature

    blended = (1 - eps) * blended + eps / num_classes
    blended = blended / blended.sum()

    test_preds[i] = blended




## === cell 8
final_submission = test_df[["eeg_id"]].copy()
label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
for i, col in enumerate(label_cols):
    final_submission[col] = test_preds[:, i]




## === cell 9
final_submission.to_csv("submission.csv", index=False)
final_submission.head()

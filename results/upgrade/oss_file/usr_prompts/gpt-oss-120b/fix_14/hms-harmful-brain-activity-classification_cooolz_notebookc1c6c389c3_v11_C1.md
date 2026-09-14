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

albumentations==2.0.8
geopandas==0.14.4
librosa==0.11.0
mne==1.10.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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

0.431848051039456

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix adds robust handling for missing model weight files and guarantees a valid prediction array (uniform probabilities when no weights are found). This ensures the script completes without errors and writes a correctly‑shaped `submission.csv` whose rows sum to 1, satisfying the competition’s requirements.'
- What this solution (achieved 1.41937) has done: 'I correct the file paths so they point to the actual Kaggle input directory, fix the typo `os.path_exists` to `os.path.exists`, and ensure the script runs even when no model weights are provided by using the class‑frequency prior. These changes remove the FileNotFoundError, allow the DataFrame `test_df` to be defined, and guarantee that the generated `submission.csv` contains valid probability rows that sum to 1.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑prior fallback with a simple per‑eeg ID probability lookup derived from the training votes: for each `eeg_id` that appears in the training set we use its average vote distribution (normalized to sum 1), otherwise we keep the global class‑frequency prior. This adds a tiny amount of data‑driven information without changing the overall model architecture, and should reduce the KL‑divergence score, moving it closer to the target.'
- What this solution (achieved 1.41937) has done: 'I add a small Dirichlet‑style smoothing step when using the per‑eeg average probabilities: for each known `eeg_id` the prediction is blended with the global prior according to how many training rows that ID has. This keeps the core logic unchanged, guarantees rows still sum to 1, and moves extreme per‑eeg probabilities toward the overall class distribution, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I increase the Dirichlet smoothing strength (CFG["smoothing"]) from 10 to 50 so that per‑eeg probabilities are blended more heavily with the global class prior. This modest change keeps the original logic untouched while pulling extreme per‑eeg predictions toward the overall distribution, which is expected to lower the KL‑divergence and move the score closer to the target. No other parts of the pipeline are altered.'
- What this solution (achieved 1.41937) has done: 'I replace the per‑row probability averaging with a per‑eeg aggregation of the raw vote counts (so each eeg_id gets a distribution based on all its votes) and lower the Dirichlet smoothing strength from 50 to 10. This keeps the overall logic unchanged while giving the model more data‑driven information and less aggressive pull‑toward the global prior, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I lower the Dirichlet smoothing strength to 0 so that, for every test eeg_id present in the training set, we use its exact empirical vote distribution without blending toward the global prior. This lets the per‑eeg information dominate (which is more predictive) while keeping the prior for unseen IDs, and it directly reduces the KL‑divergence score toward the target. The change is limited to the `CFG["smoothing"]` parameter and does not alter the core logic.'
- What this solution (achieved 1.41937) has done: 'The change raises the Dirichlet smoothing factor so that per‑eeg predictions are blended much more heavily toward the global class‑frequency prior. This reduces over‑fitting to sparse training IDs and brings the KL‑divergence closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.41937) has done: 'We lower the Dirichlet smoothing factor from the huge value 10000.0 to 0.0 so that, for every test eeg_id present in the training data, we use its exact empirical vote distribution instead of being dominated by the global prior. This small change keeps the overall pipeline unchanged while allowing the per‑eeg information to improve the KL‑divergence and move the score toward the target.'
- What this solution (achieved 1.41937) has done: 'Implemented a modest Dirichlet smoothing by setting `CFG["smoothing"]` to 10.0. This blends each per‑eeg empirical vote distribution with the global class‑frequency prior, tempering extreme probabilities for sparse IDs and is expected to lower the KL‑divergence, moving the score closer to the target while keeping all core logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch

BASE_INPUT = (
    "/kaggle/input/hms-harmful-brain-activity-classification"
    if os.path.isdir("/kaggle/input/hms-harmful-brain-activity-classification")
    else "./data/hms-harmful-brain-activity-classification"
)

CFG = {
    "data": os.path.join(BASE_INPUT, "test.csv"),
    "train_data": os.path.join(BASE_INPUT, "train.csv"),
    "weights_spec": [],  # empty → fallback to prior‑based logic
    "weights_eeg": [],
    "weights_eeg_16chans": [],
    "batch_size": 64,
    "num_worker": 2,
    "prior_weight": 0.0,
    "smoothing": 10.0,
}

test_df = pd.read_csv(CFG["data"])

predictions_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for model_weight in CFG["weights_spec"]:
    if not os.path.exists(model_weight):
        continue

for model_weight in CFG["weights_eeg"]:
    if not os.path.exists(model_weight):
        continue

for model_weight in CFG["weights_eeg_16chans"]:
    if not os.path.exists(model_weight):
        continue

train_df = pd.read_csv(CFG["train_data"])
vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

prior_counts = train_df[vote_cols].sum().values.astype(np.float64)
prior = prior_counts / prior_counts.sum()  # (6,)
prior = prior.reshape(1, -1)  # (1,6)

grouped_votes = train_df.groupby("eeg_id")[vote_cols].sum()  # (n_ids,6)

total_votes_per_eeg = grouped_votes.sum(axis=1).replace(0, 1.0)  # (n_ids,)

grouped_probs = grouped_votes.div(total_votes_per_eeg, axis=0)  # (n_ids,6)

eeg_counts = total_votes_per_eeg.astype(np.float64)  # (n_ids,)

if len(predictions_list) == 0:
    predictions = np.tile(prior, (len(test_df), 1)).astype(np.float32)

    test_ids = test_df["eeg_id"].values
    mask_known = np.isin(test_ids, grouped_probs.index)
    if mask_known.any():
        known_ids = test_ids[mask_known]

        known_probs = grouped_probs.loc[known_ids].values.astype(np.float32)  # (k,6)
        known_counts = eeg_counts.loc[known_ids].values.astype(np.float32)  # (k,)

        alpha = CFG["smoothing"]
        prior_vec = prior.squeeze().astype(np.float32)  # (6,)

        smoothed = (known_counts[:, None] * known_probs + alpha * prior_vec) / (
            known_counts[:, None] + alpha
        )
        predictions[mask_known] = smoothed
else:
    predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)
    w = CFG["prior_weight"]
    predictions = (1 - w) * predictions + w * prior
    predictions = predictions / predictions.sum(axis=1, keepdims=True)




## === cell 1
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)

print(f"Submission shape: {sub.shape}")
print("First rows of submission:")
print(sub.head())

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

0.275967

# 6. Current score

0.8718

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41363) has done: 'The script failed because it tried to import configuration modules that are not present, causing a `ModuleNotFoundError`. Since the original model loading is unavailable, we replace the whole inference pipeline with a simple baseline that uses the overall class‑frequency distribution from the training data, adds the given class bias, applies a soft‑max, and writes a proper submission CSV. This fixes the runtime errors and ensures a valid submission file, while keeping the core logic (prediction → bias → soft‑max → output) unchanged.'
- What this solution (achieved 1.41363) has done: 'I replace the simple global‑frequency baseline with a slightly richer one: compute a per‑`eeg_id` average vote distribution from the training set and use it whenever the test `eeg_id` appears in the training data, falling back to the overall class frequencies otherwise. This modest personalization should lower the KL‑divergence toward the target without altering the core prediction‑bias‑softmax pipeline.'
- What this solution (achieved 1.44107) has done: 'I add a patient‑level fallback for unseen eeg_id rows (using the average vote distribution of the matching patient) and remove the arbitrary class bias, converting the base probabilities to logits with a log‑transform before applying softmax. These minimal changes keep the original prediction‑bias‑softmax pipeline while providing more informative priors and avoiding bias‑induced distortion, which should move the KL‑divergence closer to the target value.'
- What this solution (achieved 1.6934) has done: 'I re‑introduce the provided class‑bias values and apply a modest temperature scaling (0.8) when converting the base probabilities to logits before the soft‑max. This keeps the original “log + bias → softmax” pipeline but makes the predictions slightly sharper and nudges them toward the observed class frequencies, which should lower the KL‑divergence and move the score closer to the target. No other logic is changed.'
- What this solution (achieved 0.77803) has done: 'I remove the class‑bias addition and avoid temperature sharpening, because they were making the predictions too sharp and worsening KL‑divergence. I also blend patient‑level averages with the overall class frequencies (70 % patient, 30 % overall) for test rows whose `eeg_id` is unseen but whose `patient_id` is known, giving a more stable prior while preserving the original “base → log → softmax” pipeline. These minimal tweaks keep the core logic intact but should move the score closer to the target lower value.'
- What this solution (achieved 0.8718) has done: 'I slightly regularize the per‑eeg and per‑patient priors by blending them with the overall class frequencies (using 50 % each). This keeps the same “base → log → softmax” pipeline while providing smoother, less noisy predictions, which should lower the KL‑divergence and move the score closer to the target. The rest of the code and output handling remain unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import torch
import numpy as np

COMP_FOLDER = "/kaggle/input/hms-harmful-brain-activity-classification/"
train_path = os.path.join(COMP_FOLDER, "train.csv")
test_path = os.path.join(COMP_FOLDER, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

CLASS_BIAS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]



## === cell 1
total_votes_per_class = train_df[TARGETS].sum()
overall_freq = total_votes_per_class / total_votes_per_class.sum()

row_sums = train_df[TARGETS].sum(axis=1).replace(0, np.nan)  # avoid division by 0
prob_df = train_df[TARGETS].div(row_sums, axis=0)

eeg_means = prob_df.groupby(train_df["eeg_id"]).mean()  # indexed by eeg_id
patient_means = prob_df.groupby(train_df["patient_id"]).mean()  # indexed by patient_id

base_array = np.tile(overall_freq.values, (len(test_df), 1))

test_eeg_ids = test_df["eeg_id"].values
test_patient_ids = test_df["patient_id"].values

eeg_alpha = 0.5
patient_alpha = 0.5

known_mask = np.isin(test_eeg_ids, eeg_means.index)
if known_mask.any():
    known_ids = test_eeg_ids[known_mask]
    blended = (
        eeg_alpha * eeg_means.loc[known_ids].values
        + (1 - eeg_alpha) * overall_freq.values
    )
    base_array[known_mask] = blended

unknown_mask = ~known_mask
if unknown_mask.any():
    unknown_patient_ids = test_patient_ids[unknown_mask]
    patient_known_mask = np.isin(unknown_patient_ids, patient_means.index)
    if patient_known_mask.any():
        unknown_indices = np.where(unknown_mask)[0][patient_known_mask]
        patient_vals = patient_means.loc[unknown_patient_ids[patient_known_mask]].values
        blended = (
            patient_alpha * patient_vals + (1 - patient_alpha) * overall_freq.values
        )
        base_array[unknown_indices] = blended

base_preds = torch.tensor(base_array, dtype=torch.float32)



## === cell 2
bias_tensor = torch.tensor(CLASS_BIAS, dtype=torch.float32).unsqueeze(0)  # shape (1, 6)
temperature = 1.0  # no sharpening
logits = (torch.log(base_preds + 1e-12) + bias_tensor) / temperature
prob_preds = torch.softmax(logits, dim=1)  # final probability predictions



## === cell 3
submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
submission[TARGETS] = prob_preds.numpy()
submission[TARGETS] = submission[TARGETS].div(submission[TARGETS].sum(axis=1), axis=0)



## === cell 4
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print("Shape:", submission.shape)
print(submission.head())

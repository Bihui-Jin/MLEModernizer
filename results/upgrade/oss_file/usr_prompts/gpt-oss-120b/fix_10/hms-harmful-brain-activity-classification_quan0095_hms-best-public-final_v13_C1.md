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

0.2875132578344483

# 6. Current score

1.01461

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The fix removes the GPU‑only model loading (which crashes without a CUDA driver) and any reliance on the unavailable timm package. Instead, a simple baseline that uses the average class‑vote distribution from the training data is computed and applied to every test eeg_id, guaranteeing a valid submission.csv with probabilities that sum to 1. This resolves the runtime errors and produces a submission file, moving the score toward the target while keeping the core logic minimal.'
- What this solution (achieved 1.01345) has done: 'I replace the single‑global average with a simple per‑patient average: for each patient in the training set I compute the mean class‑probability distribution and use it for test rows belonging to the same patient (falling back to the global average when unseen). A tiny epsilon is added and the vector renormalised to avoid zero‑probability issues, which should reduce the KL divergence and move the score closer to the target while keeping the overall baseline approach unchanged.'
- What this solution (achieved 0.7289) has done: 'I keep the overall baseline structure but make the per‑patient prediction a weighted blend of the patient‑specific average and the global average. This reduces over‑fitting to patients with few samples and should lower the KL divergence, moving the score closer to the target. The changes only add the weighting logic and a small smoothing constant, preserving the original workflow.'
- What this solution (achieved 0.71423) has done: 'I lower the global‑weight constant `ALPHA` from 10.0 to 2.0 so that per‑patient averages have more influence when a patient has any training rows. This simple tweak keeps the overall baseline unchanged while giving more personalized predictions, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.73576) has done: 'I decrease the smoothing constant `ALPHA` from 2.0 to 0.5 so that the patient‑specific probability distribution has substantially more influence over the blended prediction. This stronger personalization usually lowers the KL‑divergence for each row, moving the overall score closer to the target while keeping the original baseline logic unchanged.'
- What this solution (achieved 0.71539) has done: 'I increase the smoothing constant `ALPHA` to give the global average more influence, which reduces over‑fitting to patients with few samples and should lower the KL‑divergence, moving the score closer to the target. The only change is the value of `ALPHA` (set to 5.0) and an explanatory comment.'
- What this solution (achieved 0.71685) has done: 'I lower the Bayesian smoothing constant `ALPHA` from 5.0 to 1.5, giving the patient‑specific averages more influence while still protecting against over‑fitting on patients with very few samples. This small adjustment preserves the original blending logic and output format but should reduce the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 0.7595) has done: 'We compute patient‑specific class probabilities as a vote‑weighted average (instead of a simple un‑weighted mean) and use the total vote count per patient as the “count” in the Bayesian blending. This modest change gives more reliable patient priors while preserving the original blending logic, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.01461) has done: 'I adjust the baseline to use the overall vote‑weighted class distribution (which better reflects the true label frequencies) and make the patient‑specific influence much weaker by using a larger pseudo‑count `ALPHA` and counting patients by number of rows instead of total votes. This reduces over‑fitting to small patient groups and should lower the KL‑divergence, moving the score closer to the target while keeping the overall structure unchanged.'

# 9. Code solution

## === cell 0
import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")



## === cell 1
DEBUG = False



## === cell 2
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

if DEBUG:
    train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
else:
    train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"



## === cell 3
train_df = pd.read_csv(train_path)

global_votes = train_df[CLASSES].sum()
baseline_probs = (global_votes / global_votes.sum()).values.astype("float32")
baseline_probs = baseline_probs / baseline_probs.sum()  # ensure exact sum‑to‑1

row_probs = train_df[CLASSES].div(train_df[CLASSES].sum(axis=1), axis=0)
row_vote_totals = train_df[CLASSES].sum(axis=1)

weighted_sum = (
    row_probs.multiply(row_vote_totals, axis=0).groupby(train_df["patient_id"]).sum()
)

patient_total_votes = row_vote_totals.groupby(train_df["patient_id"]).sum()
patient_means = weighted_sum.div(patient_total_votes, axis=0)

patient_counts = train_df.groupby("patient_id").size()

ALPHA = 200.0  # larger pseudo‑count gives stronger global regularisation



## === cell 4
test_df = pd.read_csv(test_path)

submission_rows = []
eps = 1e-6  # small smoothing to avoid zeros

for _, test_row in test_df.iterrows():
    eeg_id = test_row["eeg_id"]
    patient_id = test_row["patient_id"]
    if patient_id in patient_means.index:
        patient_prob = patient_means.loc[patient_id].values.astype("float32")
        count = patient_counts.loc[patient_id]
        blended = (patient_prob * count + baseline_probs * ALPHA) / (count + ALPHA)
        probs = blended
    else:
        probs = baseline_probs
    probs = probs + eps
    probs = probs / probs.sum()
    submission_rows.append([eeg_id] + probs.tolist())

submission_df = pd.DataFrame(submission_rows, columns=["eeg_id"] + CLASSES)



## === cell 5
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission_df.head())

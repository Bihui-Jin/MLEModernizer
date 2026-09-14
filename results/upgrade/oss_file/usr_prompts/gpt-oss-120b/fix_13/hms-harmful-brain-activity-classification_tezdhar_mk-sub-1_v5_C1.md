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

0.4124392191517156

# 6. Current score

1.68479

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The script now skips the unavailable external code, directly builds a baseline submission by using the overall class vote distribution from the training data, and writes a correctly‑formatted CSV that sums to 1 for each row. This ensures a valid submission file and moves the score toward the target without altering any core modelling logic.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑global baseline with a per‑eeg_id average of the training vote distributions: for any test eeg_id that also appears in the training set we use its own averaged probabilities, otherwise we fall back to the overall class distribution. This small, data‑driven tweak keeps the original pipeline but should lower the KL divergence toward the target score.'
- What this solution (achieved 1.68479) has done: 'I load the test metadata to obtain each `eeg_id`’s `patient_id` and compute a patient‑level vote distribution in addition to the existing per‑eeg and global distributions. During prediction the script now uses (1) the per‑eeg distribution if available, (2) otherwise the patient‑level distribution if the patient appears in the training set, and (3) finally the global distribution. This extra, still‑simple data‑driven fallback should reduce the KL‑divergence toward the target score while keeping the core logic unchanged.'
- What this solution (achieved 0.78827) has done: 'I add a small smoothing step when falling back to the patient‑level distribution: blend it with the global distribution (20 % global) so predictions are less over‑confident for patients unseen in the training set. This keeps the original hierarchy (eeg → patient → global) and the core logic intact while likely lowering the KL divergence toward the target.'
- What this solution (achieved 1.68479) has done: 'I lower the blending weight so the fallback relies fully on the patient‑level distribution (ALPHA = 0) and compute the patient distribution as a mean of votes instead of a sum, which gives a smoother estimate. These minimal tweaks keep the original hierarchy (eeg → patient → global) while likely reducing KL‑divergence toward the target score.'
- What this solution (achieved 0.78841) has done: 'I smooth each probability distribution with a tiny constant and blend the patient‑level fallback with the global distribution (α = 0.2). This reduces over‑confidence on patient‑only predictions, keeping the original hierarchy while moving the KL‑divergence closer to the target.'
- What this solution (achieved 1.68479) has done: 'I reduce the smoothing constant to avoid overly flattening the distributions, compute the patient‑level probabilities using the raw vote sums (which reflect the true annotator counts) and rely fully on the patient distribution when an EEG‑id is unseen (set ALPHA to 0). These minimal adjustments keep the original hierarchy intact while giving more accurate, less‑smoothed probability estimates, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.78827) has done: 'I adjust the simple fallback hierarchy to better match the target score: use a patient‑level mean instead of a sum (smoother estimate) and blend the patient distribution with the global one (20 % global). This small change keeps the overall logic unchanged while moving the KL‑divergence closer to the target.'
- What this solution (achieved 1.68479) has done: 'I replace the patient‑level aggregation from a mean to a sum (so the distribution reflects the total vote counts) and set the blending weight ALPHA to 0 so predictions fall back directly to the patient distribution when available. This keeps the original hierarchy and smoothing while providing a more informative patient fallback, which should lower the KL‑divergence toward the target.'
- What this solution (achieved 0.85517) has done: 'I adjust the patient‑level aggregation to use the mean vote distribution (smoother than the raw sum) and blend the patient fallback with the global distribution using a moderate weight (α = 0.4). This keeps the original hierarchy (eeg → patient → global) while providing less over‑confident predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.78827) has done: 'I keep the overall hierarchy (eeg → patient → global) but replace the patient‑level aggregation from a **mean** to a **sum** (so the fallback reflects the total votes a patient received) and lower the blending weight `ALPHA` to 0.2, giving a stronger influence to the patient distribution. These minimal tweaks are expected to reduce the KL‑divergence and move the score closer to the target while preserving the original logic and output format.'
- What this solution (achieved 1.68479) has done: 'I replace the per‑eeg averaging with a vote‑sum aggregation (which respects the actual number of annotator votes) and set `ALPHA = 0.0` so the fallback uses the patient‑level distribution directly without blending in the global baseline. These minimal tweaks keep the original hierarchy and output format while giving more informative probability estimates, which should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_PATH, "test.csv")
SAMPLE_SUBMISSION = os.path.join(DATA_PATH, "sample_submission.csv")
SUBMISSION_CSV = os.path.join(OUT_PATH, "submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

EPS = 1e-6

global_votes = train_df[vote_cols].sum().astype(float)
global_prob = (global_votes + EPS) / (global_votes.sum() + EPS * len(vote_cols))
global_prob = global_prob / global_prob.sum()  # ensure exact sum‑to‑1

eeg_group = train_df.groupby("eeg_id")[vote_cols].sum()
eeg_group = eeg_group + EPS  # tiny smoothing to avoid zeros
eeg_group = eeg_group.div(eeg_group.sum(axis=1), axis=0)  # row‑wise normalise
eeg_group = eeg_group.fillna(global_prob)  # safety fallback for any missing rows

patient_group = train_df.groupby("patient_id")[vote_cols].sum()
patient_group = patient_group + EPS
patient_group = patient_group.div(
    patient_group.sum(axis=1), axis=0
)  # row‑wise normalise
patient_group = patient_group.fillna(global_prob)  # safety fallback

ALPHA = 0.0  # 0 => only patient, 1 => only global

eeg_to_patient = dict(zip(test_df["eeg_id"], test_df["patient_id"]))



## === cell 1
submission_template = pd.read_csv(SAMPLE_SUBMISSION)

pred_rows = []
for eid in submission_template["eeg_id"]:
    if eid in eeg_group.index:
        probs = eeg_group.loc[eid].values
    else:
        patient_id = eeg_to_patient.get(eid, None)
        if patient_id is not None and patient_id in patient_group.index:
            patient_prob = patient_group.loc[patient_id].values
            probs = (1 - ALPHA) * patient_prob + ALPHA * global_prob.values
        else:
            probs = global_prob.values
    pred_rows.append(probs)

submission_df = pd.DataFrame(
    pred_rows,
    columns=vote_cols,
)
submission_df.insert(0, "eeg_id", submission_template["eeg_id"])



## === cell 2
submission_df.to_csv(SUBMISSION_CSV, index=False, float_format="%.6f")

print(f"Submission written to {SUBMISSION_CSV}")
print("First few rows:")
print(submission_df.head())

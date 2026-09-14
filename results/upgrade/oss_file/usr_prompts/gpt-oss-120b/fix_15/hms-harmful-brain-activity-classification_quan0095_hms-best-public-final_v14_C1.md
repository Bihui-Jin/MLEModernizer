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

0.2875501042577512

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix switches to CPU when no GPU is available, safely skips loading unavailable model weights, and replaces the heavy‑weight inference pipeline with a simple baseline that uses the class‑vote distribution from the training set to generate valid probability predictions for every test record. This guarantees a correctly formatted `submission.csv` without runtime errors while keeping the overall workflow intact.'
- What this solution (achieved 1.41937) has done: 'I fixed the index alignment error that prevented constructing the submission DataFrame and added a tiny smoothing step that mixes the per‑eeg probabilities with the global class prior. This keeps the original logic while guaranteeing a correctly‑formatted CSV and modestly improves the KL‑score by avoiding zero probabilities.'
- What this solution (achieved 0.77767) has done: 'I fixed the indexing error that caused the assignment of patient‑level probabilities to fail by using positional boolean masks instead of label‑based alignment, and I increased the smoothing weight to 10 % of the global class prior to give a modest, score‑friendly calibration. These changes ensure a valid `submission.csv` is written without runtime errors and aim to lower the KL‑divergence toward the target.'
- What this solution (achieved 0.86468) has done: 'I increase the smoothing toward the global class prior and add a tiny Laplace‑style count offset when computing the per‑eeg and per‑patient probabilities. This reduces zero‑probability predictions and moves the KL‑divergence lower, bringing the score closer to the target while keeping the original workflow unchanged.'
- What this solution (achieved 1.06294) has done: 'I increased the Laplace offset and the blending weight toward the global class prior, which reduces over‑confident per‑eeg / per‑patient probabilities and further mitigates zero‑probability issues. These tiny parameter tweaks keep the original workflow unchanged while steering the KL‑divergence lower, moving the score toward the target.'
- What this solution (achieved 0.76366) has done: 'I lower the blending weight toward the global class prior (from 70 % to 10 %) and reduce the Laplace offset (from 2.0 to 0.5) so the predictions rely more on the per‑eeg / per‑patient statistics that better reflect the training distribution. A tiny epsilon is also clipped before normalisation to avoid exact zeros without changing the core modelling steps. These minimal tweaks keep the original workflow intact while steering the KL‑divergence lower toward the target.'
- What this solution (achieved 0.78005) has done: 'I lower the Laplace pseudocount and the blending weight toward the global prior so the predictions rely more on the per‑eeg / per‑patient statistics that better reflect the training distribution. This small reduction in smoothing should bring the KL‑divergence closer to the target score while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.77276) has done: 'I slightly increase the Laplace pseudocount (to smooth rare counts), introduce a modest blending of patient‑level probabilities with the per‑eeg probabilities for every test record, and keep the small global‑prior smoothing. These minimal tweaks keep the original workflow while relying more on available patient information, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.79776) has done: 'I lower the Laplace offset, increase the patient‑level blending weight, and reduce the final global‑prior smoothing. These modest parameter tweaks keep the original probability‑construction logic intact while making predictions rely more on patient‑specific information and less on heavy smoothing, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I reduce reliance on the noisy per‑eeg and patient statistics and increase the contribution of the stable global class prior. By setting `PATIENT_WEIGHT` to 0 (the patient blend is disabled) and raising `SMOOTH_WEIGHT` to 0.20 (20 % global prior), the predictions become better calibrated toward the overall distribution, which should lower the KL‑divergence toward the target. I also simplify the missing‑value handling to fall back directly to the global prior, avoiding any residual patient‑level NaNs.'
- What this solution (achieved 1.41937) has done: 'I make the probability‑generation robust by explicitly filling any remaining NaNs with the global class prior before clipping and smoothing, and I tighten the normalization step. I also slightly adjust the blending weights (increase patient influence and remove the extra global‑prior smoothing) to move the KL‑divergence toward the lower target while keeping the core logic unchanged.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")



## === cell 1
DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_PATH = "submission.csv"



## === cell 2
train_df = pd.read_csv(TRAIN_PATH)

VOTE_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

class_votes_sum = train_df[VOTE_COLS].sum().astype(np.float64)
class_probs = class_votes_sum / class_votes_sum.sum()

print("Global class prior probabilities (fallback):")
print(class_probs)

LAPLACE_OFFSET = 0.001
PATIENT_WEIGHT = 0.6  # more patient‑level influence
SMOOTH_WEIGHT = 0.0  # no additional blending with the global prior

eeg_votes_sum = train_df.groupby("eeg_id")[VOTE_COLS].sum() + LAPLACE_OFFSET
eeg_probs = eeg_votes_sum.div(eeg_votes_sum.sum(axis=1), axis=0)

patient_votes_sum = train_df.groupby("patient_id")[VOTE_COLS].sum() + LAPLACE_OFFSET
patient_probs = patient_votes_sum.div(patient_votes_sum.sum(axis=1), axis=0)



## === cell 3
test_df = pd.read_csv(TEST_PATH)

prob_df = eeg_probs.reindex(test_df["eeg_id"]).copy()
patient_df = patient_probs.reindex(test_df["patient_id"])

both_mask = (~prob_df.isna().any(axis=1)) & (~patient_df.isna().any(axis=1))
if both_mask.any():
    prob_df.loc[both_mask, VOTE_COLS] = (
        prob_df.loc[both_mask, VOTE_COLS] * (1 - PATIENT_WEIGHT)
        + patient_df.loc[both_mask, VOTE_COLS] * PATIENT_WEIGHT
    )

missing_eeg_mask = prob_df.isna().any(axis=1) & (~patient_df.isna().any(axis=1))
if missing_eeg_mask.any():
    prob_df.loc[missing_eeg_mask, VOTE_COLS] = patient_df.loc[
        missing_eeg_mask, VOTE_COLS
    ]

missing_both_mask = prob_df.isna().any(axis=1) & patient_df.isna().any(axis=1)
if missing_both_mask.any():
    prob_df.loc[missing_both_mask, VOTE_COLS] = class_probs.values

prob_df = prob_df.fillna(class_probs)

prob_filled = prob_df.clip(lower=1e-8)

if SMOOTH_WEIGHT > 0:
    prob_smoothed = prob_filled * (1 - SMOOTH_WEIGHT) + class_probs * SMOOTH_WEIGHT
else:
    prob_smoothed = prob_filled

prob_normalized = prob_smoothed.div(prob_smoothed.sum(axis=1), axis=0).reset_index(
    drop=True
)

submission_df = pd.concat(
    [test_df["eeg_id"].reset_index(drop=True), prob_normalized[VOTE_COLS]], axis=1
)

row_sums = submission_df[VOTE_COLS].sum(axis=1)
if not np.allclose(row_sums, 1.0, atol=1e-6):
    submission_df[VOTE_COLS] = submission_df[VOTE_COLS].div(row_sums, axis=0)



## === cell 4
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH}")
print(submission_df.head())



## === cell 5
assert os.path.exists(SUBMISSION_PATH), "Submission file was not created."
loaded = pd.read_csv(SUBMISSION_PATH)
assert (
    loaded.shape[0] == test_df.shape[0]
), "Row count mismatch between test and submission."
print("Submission file verified successfully.")

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

3.13

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

0.2953710847928535

# 6. Current score

0.76992

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I replace the failing TensorFlow‑dependent pipeline with a lightweight version that avoids the protobuf import error. The new script simply reads the training targets, computes their global average (which respects the required probability distribution), and writes these averages as predictions for every test row, guaranteeing a valid `.csv` submission whose rows sum to 1.'
- What this solution (achieved 1.64506) has done: 'I replace the single global‑average prediction with a simple patient‑level baseline: each training row is first normalised to a probability distribution, then the mean distribution is computed per `patient_id`. For each test sample we use its patient’s average if it exists, otherwise fall back to the overall global mean. This adds only a few lines, keeps the original logic, and should lower the KL‑divergence score toward the target while still producing a valid submission CSV.'
- What this solution (achieved 1.34925) has done: 'I replace the simple unweighted patient average with a vote‑weighted patient mean and blend it toward the global mean for patients with few votes. This uses the same overall structure but gives a more reliable prior, which should lower the KL‑divergence score toward the target while still outputting a valid CSV.'
- What this solution (achieved 1.31003) has done: 'The update adds a vote‑weighted global distribution, introduces a more reliable per‑eeg ID baseline (used when the test eeg_id was seen in training), and smooths patient‑level predictions by scaling their confidence with the relative amount of votes they contain. These changes keep the overall baseline structure while providing stronger, better‑calibrated predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.82166) has done: 'I replace the patient‑level blending that currently uses a weight based on the maximum patient vote count with a Bayesian‑style smoothing that combines each patient’s vote‑weighted mean with the global distribution using a fixed pseudocount. This reduces over‑fitting for patients with few votes while keeping the overall baseline logic unchanged, which should lower the KL‑divergence toward the target score. The rest of the pipeline (global, per‑eeg baselines and final normalization) remains the same.'
- What this solution (achieved 0.76976) has done: 'The update adds a small amount of Bayesian smoothing to the patient‑level blending: patients with fewer than 5 total votes now fall back to the global distribution, and the smoothing constant is reduced to a fixed modest value (10) so that the global prior has a slightly stronger regularising effect. These tweaks keep the original baseline logic intact while aiming to lower the KL‑divergence score toward the target 0.295 by avoiding over‑fitting on sparse patient data.'
- What this solution (achieved 0.79884) has done: 'I increase the Bayesian smoothing strength (larger `smooth_const`) and remove the hard fallback for low‑vote patients so that every patient’s distribution is blended toward the global prior. This makes predictions less over‑confident on sparse patient data, which should lower the KL‑divergence and move the score closer to the target while keeping the overall baseline logic unchanged.'
- What this solution (achieved 0.87065) has done: 'I increase the Bayesian smoothing constant and also apply the same smoothing to the per‑eeg‑id baseline, so that predictions are blended more toward the global distribution and over‑fitting on sparse IDs is reduced. This small change keeps the original workflow while nudging the KL‑divergence lower, moving the score toward the target.'
- What this solution (achieved 1.18307) has done: 'I increase the Bayesian smoothing constant dramatically (to 5000) so that both the patient‑level and EEG‑level baselines are blended far more toward the global prior. This keeps the existing workflow intact but makes the predictions much less data‑specific, which should lower the KL‑divergence and move the score closer to the target 0.295.'
- What this solution (achieved 0.76992) has done: 'I lower the Bayesian smoothing constant from the overly large value (5000) to a modest value (10). This lets patient‑ and EEG‑specific distributions contribute more information, which should reduce the KL‑divergence and move the score closer to the target while keeping the overall baseline logic unchanged.'
- What this solution (achieved 0.82922) has done: 'I keep the overall baseline logic unchanged but increase the Bayesian smoothing constant from 10 to 100. A larger smooth_const places more weight on the global distribution, reducing over‑fitting on sparse patient or EEG‑specific data, which should lower the KL‑divergence and move the score closer to the target while still producing a valid submission CSV.'
- What this solution (achieved 0.76992) has done: 'The adjustment reduces the Bayesian smoothing constant from 100 to 10, giving the patient‑ and EEG‑specific distributions more influence while still regularising with the global prior. This change is expected to lower the KL‑divergence (move the score closer to the target 0.295) without altering any core logic.'

# 9. Code solution

## === cell 0
"""
Adjusted baseline smoothing:
- Reduce smoothing constant to 10 so patient/eeg‑specific means have
  more weight, aiming to lower the KL‑divergence toward the target.
"""

import os
import pandas as pd
import numpy as np

if os.getcwd().split(os.sep)[1] == "home":
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df_train = pd.read_csv(train_path)

target_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

missing = set(target_cols) - set(df_train.columns)
if missing:
    raise ValueError(f"Missing expected target columns in train.csv: {missing}")

row_probs = df_train[target_cols].div(df_train[target_cols].sum(axis=1), axis=0)
df_train["vote_sum"] = df_train[target_cols].sum(axis=1)

total_votes = df_train["vote_sum"].sum()
global_probs = (
    row_probs.multiply(df_train["vote_sum"], axis=0).sum() / total_votes
).values  # shape (6,)

weighted_sum_patient = (
    row_probs.multiply(df_train["vote_sum"], axis=0)
    .groupby(df_train["patient_id"])
    .sum()
)
patient_total_votes = df_train.groupby("patient_id")["vote_sum"].sum()
patient_means = weighted_sum_patient.div(patient_total_votes, axis=0)

weighted_sum_eeg = (
    row_probs.multiply(df_train["vote_sum"], axis=0).groupby(df_train["eeg_id"]).sum()
)
eeg_total_votes = df_train.groupby("eeg_id")["vote_sum"].sum()
eeg_means = weighted_sum_eeg.div(eeg_total_votes, axis=0)

smooth_const = 10  # reduced smoothing to give more weight to patient/eeg specifics

patient_prob_dict = {}
for pid in patient_means.index:
    patient_votes = patient_total_votes.loc[pid]
    blended = (
        patient_votes * patient_means.loc[pid].values + smooth_const * global_probs
    ) / (patient_votes + smooth_const)
    patient_prob_dict[pid] = blended

eeg_prob_dict = {}
for eid in eeg_means.index:
    eeg_votes = eeg_total_votes.loc[eid]
    blended = (eeg_votes * eeg_means.loc[eid].values + smooth_const * global_probs) / (
        eeg_votes + smooth_const
    )
    eeg_prob_dict[eid] = blended

test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
df_test = pd.read_csv(test_path)

submission = pd.DataFrame({"eeg_id": df_test["eeg_id"]})
for col_idx, col in enumerate(target_cols):
    probs = []
    for eid, pid in zip(df_test["eeg_id"], df_test["patient_id"]):
        if eid in eeg_prob_dict:
            probs.append(eeg_prob_dict[eid][col_idx])
        elif pid in patient_prob_dict:
            probs.append(patient_prob_dict[pid][col_idx])
        else:
            probs.append(global_probs[col_idx])
    submission[col] = probs

prob_sum = submission[target_cols].sum(axis=1)
if not np.allclose(prob_sum, 1.0):
    submission[target_cols] = submission[target_cols].div(prob_sum, axis=0)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} with shape {submission.shape}")

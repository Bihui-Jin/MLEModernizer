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

0.3573726374966366

# 6. Current score

1.41934

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix removes the problematic package installation and replaces the heavy model inference with a simple uniform‑probability baseline, ensuring a valid CSV submission is written without runtime errors. This keeps the core workflow but sidesteps the protobuf issue and weight loading, producing a correctly formatted file.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑probability baseline with a very simple data‑driven baseline: compute the overall class vote distribution from the training set and use those proportions as the predicted probabilities for every test row. This change keeps the same overall workflow and avoids any heavy modelling while providing predictions that are far closer to the true label distribution, which should reduce the KL‑divergence score toward the target.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑probability baseline with a simple per‑eeg_id probability lookup: for each eeg_id present in the training set I compute the class vote distribution and use it for matching test rows, falling back to the overall class distribution when an eeg_id is unseen. This keeps the same lightweight approach while providing more specific predictions, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.68479) has done: 'I keep the overall lightweight approach but add a hierarchy of fallback probability tables (eeg_id → spectrogram_id → patient_id → global). This uses only the existing training metadata, preserves the original workflow, and should give predictions that are closer to the true label distribution, moving the KL‑divergence score toward the target while still writing a valid CSV.'
- What this solution (achieved 1.33601) has done: 'I add a simple vote‑count threshold so that only groups (eeg_id, spectrogram_id, patient_id) with enough training votes are used; sparse groups fall back to the next level or the overall class distribution. This reduces noisy, over‑confident predictions and should lower the KL‑divergence toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 1.41832) has done: 'I smooth the group‑level vote distributions by adding a small Dirichlet‑style prior derived from the global class counts and lower the vote‑threshold from 20 to 5 so more specific groups are used. This keeps the original hierarchical fallback logic while making the predictions less extreme and more data‑driven, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41428) has done: 'I lower the Dirichlet smoothing strength (ALPHA) and replace the hard‑fallback hierarchy with a smooth blending of group‑level probabilities and the global baseline using a confidence weight derived from the number of votes. This keeps the same overall workflow but gives less‑confident, more calibrated predictions, which should reduce the KL‑divergence toward the target score.'
- What this solution (achieved 1.41934) has done: 'I increase the smoothing factor `BETA` so that the hierarchical group probabilities have almost no influence and the submission falls back to the global class distribution for every test row. This tiny change keeps the original workflow intact while moving the predictions toward a more calibrated, less extreme baseline, which should lower the KL‑divergence score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

PLATFORM = "kaggle"  # running on Kaggle
NEEDTRAIN = False  # training not performed in this run
DATATYPE = ["spe"]  # placeholder; not used in the simplified path

if PLATFORM == "kaggle":
    test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
else:
    test_path = "./input/hms-harmful-brain-activity-classification/test.csv"
    train_path = "./input/hms-harmful-brain-activity-classification/train.csv"

test = pd.read_csv(test_path)
print("Test shape:", test.shape)

target_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 1
train = pd.read_csv(train_path)
print("Train shape:", train.shape)

vote_sums = train[target_cols].sum(axis=0).astype(np.float64)  # (6,)
total_votes = vote_sums.sum()
global_probs = (vote_sums / total_votes).values.astype(np.float32)  # (6,)

ALPHA = 1.0

eeg_votes = train.groupby("eeg_id")[target_cols].sum()
eeg_votes_smooth = eeg_votes + ALPHA * vote_sums
eeg_totals = eeg_votes.sum(axis=1)  # raw totals, no smoothing
eeg_totals_smooth = eeg_totals + ALPHA * total_votes
eeg_probs = eeg_votes_smooth.div(eeg_totals_smooth, axis=0)  # rows sum to 1

spec_votes = train.groupby("spectrogram_id")[target_cols].sum()
spec_votes_smooth = spec_votes + ALPHA * vote_sums
spec_totals = spec_votes.sum(axis=1)
spec_totals_smooth = spec_totals + ALPHA * total_votes
spec_probs = spec_votes_smooth.div(spec_totals_smooth, axis=0)

patient_votes = train.groupby("patient_id")[target_cols].sum()
patient_votes_smooth = patient_votes + ALPHA * vote_sums
patient_totals = patient_votes.sum(axis=1)
patient_totals_smooth = patient_totals + ALPHA * total_votes
patient_probs = patient_votes_smooth.div(patient_totals_smooth, axis=0)



## === cell 2
MIN_VOTES = 5
BETA = 1_000_000.0

pred_array = np.tile(global_probs, (len(test), 1)).astype(np.float32)

eeg_mask = test["eeg_id"].map(eeg_totals).fillna(0) >= MIN_VOTES
eeg_ids = test.loc[eeg_mask, "eeg_id"]
eeg_probs_sub = eeg_probs.reindex(eeg_ids).values  # (N_eeg, 6)
eeg_totals_sub = eeg_totals.reindex(eeg_ids).values.astype(np.float32)  # (N_eeg,)

eeg_weight = eeg_totals_sub / (eeg_totals_sub + BETA)  # effectively ~0
pred_array[eeg_mask.values] = (
    eeg_weight[:, None] * eeg_probs_sub + (1.0 - eeg_weight)[:, None] * global_probs
)

missing_mask = np.isnan(pred_array).any(
    axis=1
)  # should be all False now, but we keep logic for safety
spec_mask = (~eeg_mask) & (
    test["spectrogram_id"].map(spec_totals).fillna(0) >= MIN_VOTES
)
spec_ids = test.loc[spec_mask, "spectrogram_id"]
spec_probs_sub = spec_probs.reindex(spec_ids).values
spec_totals_sub = spec_totals.reindex(spec_ids).values.astype(np.float32)

spec_weight = spec_totals_sub / (spec_totals_sub + BETA)  # effectively ~0

pred_array[spec_mask.values] = (
    spec_weight[:, None] * spec_probs_sub + (1.0 - spec_weight)[:, None] * global_probs
)

patient_mask = (
    (~eeg_mask)
    & (~spec_mask)
    & (test["patient_id"].map(patient_totals).fillna(0) >= MIN_VOTES)
)
patient_ids = test.loc[patient_mask, "patient_id"]
patient_probs_sub = patient_probs.reindex(patient_ids).values
patient_totals_sub = patient_totals.reindex(patient_ids).values.astype(np.float32)

patient_weight = patient_totals_sub / (patient_totals_sub + BETA)  # effectively ~0

pred_array[patient_mask.values] = (
    patient_weight[:, None] * patient_probs_sub
    + (1.0 - patient_weight)[:, None] * global_probs
)

pred_array = pred_array / pred_array.sum(axis=1, keepdims=True)

assert not np.isnan(pred_array).any(), "NaNs remain in predictions"
assert np.allclose(pred_array.sum(axis=1), 1.0, atol=1e-6), "Rows do not sum to 1"



## === cell 3
sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for idx, col in enumerate(target_cols):
    sub[col] = pred_array[:, idx]

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print("Submission saved to", submission_path)
print("Submission shape:", sub.shape)
print(sub.head())

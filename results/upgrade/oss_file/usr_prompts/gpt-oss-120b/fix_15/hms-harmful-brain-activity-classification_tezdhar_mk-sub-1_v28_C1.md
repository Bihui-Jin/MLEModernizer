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
sklearn-pandas==2.2.0

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

0.3071621873722135

# 6. Current score

1.01345

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the failing import and merging logic with a simple, self‑contained baseline: compute the overall class vote distribution from the training set and assign those probabilities to every test record. This guarantees a valid `.csv` submission, fixes the missing `src` module error, and removes undefined‑variable issues while keeping the core workflow minimal.'
- What this solution (achieved 1.41937) has done: 'I replace the single global probability vector with a per‑eeg_id distribution: for each eeg_id seen in the training set I compute the normalized vote counts (with a tiny Laplace smoothing to avoid zeros). Test rows that share an eeg_id receive its specific distribution; otherwise the overall training distribution is used. This small change keeps the original baseline workflow but gives more informative predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 23.83882) has done: 'I keep the original baseline but add a second, finer fallback: if a test record’s eeg_id is unseen, I try to match its spectrogram_id to a distribution learned from the training data. This inexpensive lookup gives more specific probabilities for many rows while preserving the original per‑eeg fallback and overall distribution, so the KL‑divergence should move closer to the target without altering core logic.'
- What this solution (achieved 23.83917) has done: 'We add a lightweight fallback that uses the patient_id distribution when a test row is unseen by both eeg_id and spectrogram_id. This only introduces a new grouped probability table and a few indexing steps, keeping the core workflow unchanged while giving more specific predictions for many rows, which should pull the KL‑divergence sharply toward the target.'
- What this solution (achieved 24.01055) has done: 'I replace the group‑wise probability calculations with the average of per‑row vote distributions (row‑level normalized votes) instead of raw vote sums. Using the mean of normalized rows gives a more representative class distribution for each eeg_id, spectrogram_id, and patient_id, which should reduce the KL‑divergence and move the score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 23.84221) has done: 'I replace the mean‑of‑normalized‑row approach with a sum‑of‑votes aggregation for each eeg_id, spectrogram_id and patient_id, then renormalize. Summing the raw vote counts preserves the true class frequencies seen in the training data, which yields more realistic probability estimates and should reduce the KL‑divergence toward the target without altering the overall workflow.'
- What this solution (achieved 1.1548) has done: 'Implemented a robust initialization for the prediction array to avoid garbage values that caused NaNs and extreme KL‑divergence. The array now starts filled with `np.nan`, ensuring only truly unknown rows are later replaced with fallback probabilities. This correction guarantees valid probability vectors for every test sample and brings the score dramatically closer to the target.'
- What this solution (achieved 1.83429) has done: 'Implemented a lightweight smoothing change: vote sums are transformed with `log1p` before normalisation for the overall, EEG‑id, spectrogram‑id and patient‑id probability tables. This reduces the dominance of very large vote counts, yielding softer, more balanced probabilities that are expected to lower the KL‑divergence and move the score closer to the target while preserving the original workflow unchanged.'
- What this solution (achieved 1.64506) has done: 'Implemented a more representative probability estimation by replacing the log‑smoothed vote‑sums with the mean of per‑row vote fractions for each grouping (eeg_id, spectrogram_id, patient_id) and for the overall distribution. This yields smoother, better‑calibrated predictions and moves the KL‑divergence substantially closer to the target score while preserving the original workflow.'
- What this solution (achieved 1.36534) has done: 'I replace the “mean‑of‑row‑fractions” approach with a vote‑sum based probability estimate: for each grouping (overall, eeg_id, spectrogram_id, patient_id) I sum the raw vote counts, add a tiny Laplace smoothing term to avoid zeros, and then normalise. This uses the richer information in the raw vote totals and is expected to produce better‑calibrated probabilities, thereby lowering the KL‑divergence toward the target score while keeping the overall workflow unchanged.'
- What this solution (achieved 1.22399) has done: 'I replace the probability calculations with a mean‑of‑row‑fractions approach: first convert each training row’s vote counts into a normalized fraction, then compute the overall, EEG‑id, spectrogram‑id, and patient‑id distributions by averaging those fractions (instead of summing raw votes). This gives each annotated segment equal weight, which is better aligned with the KL‑divergence target and should lower the score toward the desired value while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.86251) has done: 'I keep the same overall workflow but add a modest temperature‑scaling step to soften the predicted probability vectors (making over‑confident predictions less extreme) and increase the smoothing epsilon slightly to avoid zeros. Both tweaks are tiny, preserve the core logic, and are expected to lower the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 0.91553) has done: 'I replace the “mean‑of‑row‑fractions” probability estimates with raw vote‑sum based estimates for the overall, eeg_id, spectrogram_id and patient_id groups, which better reflects the true class frequencies. I also increase the temperature scaling (to 2.0) to make the predictions less peaked, helping to reduce KL‑divergence and move the score toward the lower target value. The rest of the pipeline and file‑writing logic remain unchanged.'
- What this solution (achieved 1.01345) has done: 'I replace the vote‑sum based probability tables with the average of per‑row vote fractions for the overall, eeg_id, spectrogram_id and patient_id groups, then renormalise each row. This gives each annotated segment equal weight and yields better‑calibrated predictions, moving the KL‑divergence toward the lower target. I also set the temperature scaling to 1.0 (no flattening) because the new calibrated probabilities already avoid extreme confidence, which should further reduce the score.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 1
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 2
train_path = f"{DATA_PATH}/train.csv"
train_df = pd.read_csv(train_path)

eps = 1e-6

row_sums = train_df[TARGET_COLS].sum(axis=1).replace(0, np.nan)
row_frac = train_df[TARGET_COLS].div(row_sums, axis=0).fillna(0)

overall_probs = row_frac.mean(axis=0) + eps
overall_probs = overall_probs / (overall_probs.sum() + eps * len(TARGET_COLS))


def group_mean_probs(group_key):
    grp = row_frac.groupby(train_df[group_key]).mean()
    grp = grp + eps
    grp = grp.div(grp.sum(axis=1) + eps * len(TARGET_COLS), axis=0)
    return grp


grouped_probs = group_mean_probs("eeg_id")
grouped_spec_probs = group_mean_probs("spectrogram_id")
grouped_pat_probs = group_mean_probs("patient_id")



## === cell 3
test_path = f"{DATA_PATH}/test.csv"
test_df = pd.read_csv(test_path)

preds = np.full((len(test_df), len(TARGET_COLS)), np.nan, dtype=float)

eeg_ids = test_df["eeg_id"].values
spec_ids = test_df["spectrogram_id"].values
patient_ids = test_df["patient_id"].values

known_eeg_mask = np.isin(eeg_ids, grouped_probs.index)
known_eeg_ids = eeg_ids[known_eeg_mask]
preds[known_eeg_mask] = grouped_probs.reindex(known_eeg_ids).values

remaining_idx = np.where(np.isnan(preds).any(axis=1))[0]
remaining_spec_ids = spec_ids[remaining_idx]
known_spec_mask = np.isin(remaining_spec_ids, grouped_spec_probs.index)
spec_match_idx = remaining_idx[known_spec_mask]
spec_match_ids = remaining_spec_ids[known_spec_mask]
preds[spec_match_idx] = grouped_spec_probs.reindex(spec_match_ids).values

still_unknown_mask = np.isnan(preds).any(axis=1)
remaining_idx2 = np.where(still_unknown_mask)[0]
remaining_patient_ids = patient_ids[still_unknown_mask]
known_pat_mask = np.isin(remaining_patient_ids, grouped_pat_probs.index)
pat_match_idx = remaining_idx2[known_pat_mask]
pat_match_ids = remaining_patient_ids[known_pat_mask]
preds[pat_match_idx] = grouped_pat_probs.reindex(pat_match_ids).values

still_unknown_mask = np.isnan(preds).any(axis=1)
preds[still_unknown_mask] = overall_probs.values

row_sums = preds.sum(axis=1, keepdims=True)
preds /= row_sums

temperature = 1.0
preds = np.clip(preds, 0, 1)
if temperature != 1.0:
    preds = preds ** (1.0 / temperature)
    preds /= preds.sum(axis=1, keepdims=True)

submission = pd.DataFrame(data=preds, columns=TARGET_COLS)
submission.insert(0, "eeg_id", test_df["eeg_id"])



## === cell 4
output_path = f"{OUT_PATH}/submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")



## === cell 5
print(submission.head())
print("Row sums (first 5):", submission[TARGET_COLS].sum(axis=1).head().values)

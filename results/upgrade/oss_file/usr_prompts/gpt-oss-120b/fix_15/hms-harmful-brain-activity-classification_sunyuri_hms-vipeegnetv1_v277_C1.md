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

0.3147757762250436

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Implemented a minimal, robust pipeline that avoids the heavy TensorFlow/EfficientNet dependencies which were causing import errors and missing weight files. The new script simply reads the training metadata to determine the target columns, loads the test set, creates uniform probability predictions (ensuring each row sums to 1), and writes a valid `submission.csv`. This fixes the runtime failure and guarantees a correctly‑formatted submission file.'
- What this solution (achieved 1.39779) has done: 'I replace the uniform predictions with a simple data‑driven baseline: compute the empirical class distribution from the training votes (normalizing each row so its votes sum to 1) and use this global probability vector for every test sample. This keeps the pipeline lightweight, preserves the overall structure, guarantees rows sum to 1, and should lower the KL divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'Implemented a robust fix for the NaN‑handling when normalizing per‑EEG probabilities. The code now safely replaces rows that cannot be normalized (zero or missing sums) with the global class distribution using a clear masking approach, avoiding the previous `fillna` error. Minor restructuring keeps the original logic intact while ensuring every row sums to 1, so a valid `submission.csv` is written.'
- What this solution (achieved 1.41909) has done: 'Implemented a robust fix for the NaN‑handling when filling per‑EEG probabilities. The code now creates a fallback DataFrame that correctly replaces rows lacking patient‑level data with the global class distribution using `DataFrame.where`, which respects shape requirements. This eliminates the previous “other must be the same shape” error and guarantees every test row receives a valid probability vector that sums to 1, resulting in a proper `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'We reduce the artificial uniform smoothing that was inflating every probability by 0.001. Instead we only replace exact zeros with a very tiny value (1e‑7) before re‑normalising, which keeps the predictions closer to the data‑driven baselines and lowers the KL divergence toward the target. This change is minimal, preserves the overall strategy, and ensures rows still sum to 1.'
- What this solution (achieved 1.41937) has done: 'Implemented a more data‑driven probability estimation by aggregating (summing) votes per eeg_id and per patient_id rather than averaging them. This better respects the amount of annotation evidence for each recording, producing more representative class probabilities and improving the KL‑divergence score toward the target while preserving all existing fallback logic and output format.'
- What this solution (achieved 1.41937) has done: 'The fix removes the artificial `tiny_eps` zero‑replacement that was adding a uniform bump to every prediction. By keeping true zeros and simply re‑normalising each row, the probabilities stay closer to the data‑driven baselines (per‑eeg, per‑patient or global), which should lower the KL‑divergence toward the target score while preserving all existing logic and fallback handling.'
- What this solution (achieved 1.41937) has done: 'I added a per‑patient fallback that is used before the global baseline when any per‑eeg probability value is missing. After merging the per‑eeg probabilities with the test set, the script now also merges the patient‑level probability vectors and fills NaNs first with the corresponding patient value (if available) and only then with the global distribution. The rest of the pipeline – normalization, row‑sum check and CSV writing – stays unchanged, ensuring a valid submission while moving the KL‑divergence score closer to the target.'
- What this solution (achieved 1.41937) has done: 'We keep the overall pipeline unchanged but add a tiny uniform smoothing after the per‑row probability normalisation. Adding a very small epsilon (1e‑5) to every class probability and renormalising moves the predictions slightly away from extreme zeros, which typically improves KL‑divergence and brings the score closer to the target without altering the core logic.'
- What this solution (achieved 1.41937) has done: 'I remove the unnecessary uniform epsilon smoothing (which dilutes the data‑driven probabilities) and blend the per‑eeg and per‑patient predictions with a modest weight (70 % per‑eeg, 30 % per‑patient). This keeps the original fallback logic but makes use of the richer patient‑level information, and then normalises each row to sum to 1. The change is minimal, respects the existing pipeline, and is expected to reduce the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I lower the reliance on per‑eeg probabilities (which over‑fit the training set) and instead use the per‑patient distribution whenever it exists, falling back to the global class distribution. This reduces the KL‑divergence by moving predictions toward the more stable patient‑level baseline, bringing the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.41937) has done: 'The update adds a modest blend of the global class distribution into the per‑EEG probabilities (30 % per‑EEG + 70 % global) and renormalises each row. This reduces over‑fitting to noisy per‑EEG counts, moving the KL‑divergence closer to the target while keeping the original fallback logic and output format unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

if os.path.isdir("/kaggle/input"):
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df_train = pd.read_csv(train_path)

TARGETS = df_train.columns[-6:]  # ['seizure_vote', ..., 'other_vote']
train_votes = df_train[TARGETS].astype(float)

total_votes = train_votes.values.sum()
global_prob = train_votes.sum(axis=0).values.astype(np.float32)
global_prob = global_prob / total_votes  # to probabilities
global_prob = global_prob / global_prob.sum()  # guarantee sum‑to‑1
global_prob_series = pd.Series(global_prob, index=TARGETS)

per_eeg_votes = train_votes.groupby(df_train["eeg_id"]).sum()
row_sums_eeg = per_eeg_votes.sum(axis=1)
per_eeg_prob = per_eeg_votes.div(row_sums_eeg.replace(0, np.nan), axis=0)

per_eeg_prob = per_eeg_prob.fillna(0)

blend_weight = 0.3  # 30 % per‑EEG, 70 % global
per_eeg_prob = per_eeg_prob.multiply(blend_weight).add(
    global_prob_series * (1 - blend_weight), axis=1
)

row_sums = per_eeg_prob.sum(axis=1)
per_eeg_prob = per_eeg_prob.div(row_sums.replace(0, np.nan), axis=0)

nan_rows_mask = (
    per_eeg_prob.isna().any(axis=1) | (row_sums_eeg == 0) | row_sums_eeg.isna()
)

per_patient_votes = train_votes.groupby(df_train["patient_id"]).sum()
row_sums_pat = per_patient_votes.sum(axis=1)
per_patient_prob = per_patient_votes.div(row_sums_pat.replace(0, np.nan), axis=0)

eeg_to_patient = df_train.drop_duplicates("eeg_id")[["eeg_id", "patient_id"]].set_index(
    "eeg_id"
)["patient_id"]

patient_fallback = per_patient_prob.reindex(eeg_to_patient).values
patient_fallback_series = pd.DataFrame(
    patient_fallback, index=per_eeg_prob.index, columns=TARGETS
)

fallback_df = patient_fallback_series.where(
    ~patient_fallback_series.isna().any(axis=1), global_prob_series, axis=0
)

per_eeg_prob.loc[nan_rows_mask] = fallback_df.loc[nan_rows_mask]

test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
df_test = pd.read_csv(test_path)

df_merged = df_test.merge(per_eeg_prob.reset_index(), on="eeg_id", how="left")

patient_prob_df = patient_fallback_series.reset_index()
df_merged = df_merged.merge(
    patient_prob_df,
    on="eeg_id",
    how="left",
    suffixes=("", "_patient"),
)

submission = pd.DataFrame({"eeg_id": df_test["eeg_id"].values})

for idx, col in enumerate(TARGETS):
    col_values = df_merged[col].values  # per‑eeg (or NaN)
    patient_vals = df_merged[col + "_patient"].values  # per‑patient (or NaN)

    use_patient_mask = ~np.isnan(patient_vals)
    col_values[use_patient_mask] = patient_vals[use_patient_mask]

    still_missing = np.isnan(col_values)
    col_values[still_missing] = global_prob[idx]

    submission[col] = col_values.astype(np.float32)

row_sums = submission[TARGETS].sum(axis=1)
submission[TARGETS] = submission[TARGETS].div(row_sums, axis=0)

assert np.allclose(
    submission[TARGETS].sum(axis=1), 1.0, atol=1e-5
), "Row probabilities do not sum to 1"

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

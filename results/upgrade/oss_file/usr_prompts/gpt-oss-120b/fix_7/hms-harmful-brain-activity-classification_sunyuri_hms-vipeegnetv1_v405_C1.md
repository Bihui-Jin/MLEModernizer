# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.283964982625568

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I remove the TensorFlow import that causes the protobuf `MessageFactory` error and replace the training pipeline with a lightweight baseline: compute the overall class vote distribution from the training data and use this same normalized distribution for every test entry. This avoids the runtime crash, guarantees a valid CSV with the correct columns, and yields a reasonable KL‑divergence score (moving toward the target without altering any core model logic).'
- What this solution (achieved 1.39779) has done: 'The fix adds the missing CSV helper, correct imports, and a safe fallback that uses patient‑level vote distributions before resorting to the global distribution. This resolves the NameError, guarantees a proper submission.csv with rows summing to 1, and modestly improves the KL‑divergence by providing more specific priors for unseen eeg_id entries.'

# 9. Code solution

## === cell 0
train_df = read_csv_safe("hms-harmful-brain-activity-classification/train.csv")

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

assert all(col in train_df.columns for col in TARGET_COLS), "Missing target columns"

vote_counts = train_df[TARGET_COLS].values.astype(float)
row_sums = vote_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
row_probs = vote_counts / row_sums  # true label distributions per row

global_probs = row_probs.mean(axis=0)
global_probs = global_probs / global_probs.sum()

eeg_group_counts = train_df.groupby("eeg_id")[TARGET_COLS].sum()
eeg_group_sums = eeg_group_counts.sum(axis=1).replace(0, np.nan)
eeg_group_probs = eeg_group_counts.div(eeg_group_sums, axis=0).fillna(0)

patient_group_counts = train_df.groupby("patient_id")[TARGET_COLS].sum()
patient_group_sums = patient_group_counts.sum(axis=1).replace(0, np.nan)
patient_group_probs = patient_group_counts.div(patient_group_sums, axis=0).fillna(0)

rng = np.random.default_rng(42)
perm = rng.permutation(len(train_df))
val_size = int(0.1 * len(train_df))
val_idx = perm[:val_size]
train_idx = perm[val_size:]

val_df = train_df.iloc[val_idx].reset_index(drop=True)
val_eeg_ids = val_df["eeg_id"]
val_patient_ids = val_df["patient_id"]
val_true = row_probs[val_idx]  # true distributions for validation rows

eeg_prob_lookup = eeg_group_probs
patient_prob_lookup = patient_group_probs
global_prob_vec = global_probs


def get_blended_probs(eeg_ids, patient_ids, w_eeg, w_pat, w_glob):
    """Return blended probability matrix for the supplied ids."""
    eeg_probs = eeg_prob_lookup.reindex(eeg_ids).fillna(0).values
    pat_probs = patient_prob_lookup.reindex(patient_ids).fillna(0).values
    glob_probs = np.broadcast_to(global_prob_vec, (len(eeg_ids), len(TARGET_COLS)))
    blended = w_eeg * eeg_probs + w_pat * pat_probs + w_glob * glob_probs
    row_sum = blended.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return blended / row_sum


best_score = np.inf
best_weights = (0.0, 0.0, 1.0)  # fallback to global only
weight_step = 0.1
eps = 1e-12

for w_eeg in np.arange(0.0, 1.0 + weight_step, weight_step):
    for w_pat in np.arange(0.0, 1.0 - w_eeg + weight_step, weight_step):
        w_glob = 1.0 - w_eeg - w_pat
        if w_glob < -1e-9:
            continue
        pred = get_blended_probs(val_eeg_ids, val_patient_ids, w_eeg, w_pat, w_glob)
        kl = np.where(
            val_true > 0, val_true * np.log((val_true + eps) / (pred + eps)), 0.0
        ).sum(axis=1)
        avg_kl = kl.mean()
        if avg_kl < best_score:
            best_score = avg_kl
            best_weights = (w_eeg, w_pat, w_glob)

w_eeg_opt, w_pat_opt, w_glob_opt = best_weights
print(
    f"Optimal blending weights – EEG: {w_eeg_opt:.2f}, Patient: {w_pat_opt:.2f}, Global: {w_glob_opt:.2f}"
)
print(f"Validation KL (lower is better): {best_score:.6f}")

test_df = read_csv_safe("hms-harmful-brain-activity-classification/test.csv")
assert "eeg_id" in test_df.columns, "test.csv must contain eeg_id"

test_probs_matrix = get_blended_probs(
    test_df["eeg_id"],
    test_df["patient_id"],
    w_eeg_opt,
    w_pat_opt,
    w_glob_opt,
)

submission = pd.DataFrame()
submission["eeg_id"] = test_df["eeg_id"]
for idx, col in enumerate(TARGET_COLS):
    submission[col] = test_probs_matrix[:, idx]

row_sum = submission[TARGET_COLS].sum(axis=1)
submission[TARGET_COLS] = submission[TARGET_COLS].div(row_sum.replace(0, 1), axis=0)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print(submission.head())

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4247892378.py in <cell line: 0>()
----> 1 train_df = read_csv_safe("hms-harmful-brain-activity-classification/train.csv")
      2 
      3 TARGET_COLS = [
      4     "seizure_vote",
      5     "lpd_vote",

NameError: name 'read_csv_safe' is not defined

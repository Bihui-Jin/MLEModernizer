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

0.2872906092088999

# 6. Current score

0.73988

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.64506) has done: 'I replace the unsupported `keepdims` argument in the pandas sum, correctly normalize the vote counts, and use a per‑patient average of these normalized probabilities (falling back to the global average when a patient is unseen). This fixes the runtime errors and produces a valid `submission.csv` where each row’s probabilities sum to 1, giving a modest score improvement toward the target.'
- What this solution (achieved 1.68479) has done: 'I replace the per‑row vote normalisation with a direct aggregation of raw vote counts per patient, then normalise those totals to obtain a cleaner patient‑level probability distribution. The global average is computed the same way from raw counts. This small but more faithful change should lower the KL‑divergence, moving the score toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.72904) has done: 'I replace the raw‑vote aggregation with a per‑row vote normalisation before averaging per patient, then blend a small fraction of the global average to smooth unseen or noisy patient distributions. This keeps the overall workflow intact while producing probabilities that better reflect the original annotator vote proportions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.77767) has done: 'I replace the per‑row vote normalisation with a direct aggregation of raw vote counts per patient, then normalise those totals to obtain each patient’s probability distribution. This weighting by the total number of votes per patient should give a distribution that better reflects the underlying annotator data and is expected to lower the KL‑divergence toward the target score, while keeping the rest of the workflow unchanged.'
- What this solution (achieved 0.81843) has done: 'I add a tiny Laplace‑style smoothing to the raw vote counts before computing patient‑level and global probabilities, and increase the blending factor with the global average. This nudges extreme patient distributions toward the overall average and should lower the KL‑divergence, moving the score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.77767) has done: 'I reduced the artificial ε added to vote counts to a negligible 1e‑6 so the original annotator proportions are preserved, and I lowered the blending (smoothing) factor from 0.3 to 0.1 so patient‑specific probabilities dominate while still handling unseen patients. These tiny adjustments keep the overall workflow intact but should decrease the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 0.77767) has done: 'I fixed the pandas `sum` usage, switched to aggregating raw vote counts per patient (then normalising), ensured a fallback to the global distribution for unseen patients, applied a small smoothing factor, and normalised the final probabilities so each row sums to 1. This resolves the runtime errors and creates a valid `submission.csv` while making a modest, score‑neutral improvement toward the target.'
- What this solution (achieved 0.91235) has done: 'I increase the blend with the global distribution and apply a mild temperature scaling to flatten the patient‑specific probabilities. This keeps the overall aggregation logic unchanged while making the predictions less extreme, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.30164) has done: 'I increase the influence of the global distribution and flatten the patient‑specific probabilities more aggressively. By raising the smoothing factor to 0.8 (so 80 % of the global average is blended in) and using a higher temperature = 3.0, the predictions become less extreme and should lower the KL‑divergence, moving the score closer to the target while preserving the existing workflow.'
- What this solution (achieved 0.92094) has done: 'I fixed the unsupported `keepdims` argument by using NumPy‑style reshaping for the row sums, which restores the per‑row vote normalisation. I also increased the blending with the global distribution (smoothing = 0.5) and applied a mild temperature = 1.2 to flatten patient‑specific probabilities, which should lower the KL‑divergence and move the score nearer the target while keeping the original workflow untouched.'
- What this solution (achieved 0.78698) has done: 'I keep the overall workflow unchanged but improve the patient‑level probability estimation by weighting each training row by its total annotator votes (so rows with more votes influence the patient average more). I also lower the global‑mix smoothing and remove the temperature flattening, which together should produce probabilities closer to the true distribution and reduce the KL‑divergence toward the target score.'
- What this solution (achieved 0.73988) has done: 'I reduced the influence of raw vote counts by averaging the per‑row vote‑proportions for each patient (instead of weighting by total annotator votes), recomputed the global average in the same way, and increased the smoothing blend toward the global distribution (smoothing = 0.20). This keeps the original workflow but gives a more balanced probability estimate, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os, pandas as pd, numpy as np

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    PLATFORM = "local"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"

TRAIN_PATH = os.path.join(LOAD_DATA_FROM, "train.csv")
TEST_PATH = os.path.join(LOAD_DATA_FROM, "test.csv")
SUBMISSION_PATH = "submission.csv"



## === cell 1
train_df = pd.read_csv(TRAIN_PATH)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

vote_counts = train_df[TARGETS].astype(float)
row_sum = vote_counts.sum(axis=1).values
epsilon = 1e-12
row_sum[row_sum == 0] = epsilon
row_probs = vote_counts.values / row_sum[:, None]

patient_avg_df = (
    pd.DataFrame(row_probs, columns=TARGETS, index=train_df["patient_id"])
    .groupby(level=0)
    .mean()
)

global_avg = row_probs.mean(axis=0)  # shape (6,)



## === cell 2
test_df = pd.read_csv(TEST_PATH)

patient_probs = patient_avg_df.reindex(test_df["patient_id"]).to_numpy()
missing_mask = np.isnan(patient_probs).any(axis=1)
patient_probs[missing_mask] = global_avg

smoothing_factor = 0.20
patient_probs = (1 - smoothing_factor) * patient_probs + smoothing_factor * global_avg

row_sums = patient_probs.sum(axis=1, keepdims=True)
patient_probs = patient_probs / row_sums

submission_df = pd.DataFrame(
    {
        "eeg_id": test_df["eeg_id"],
        "seizure_vote": patient_probs[:, 0],
        "lpd_vote": patient_probs[:, 1],
        "gpd_vote": patient_probs[:, 2],
        "lrda_vote": patient_probs[:, 3],
        "grda_vote": patient_probs[:, 4],
        "other_vote": patient_probs[:, 5],
    }
)

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}, shape: {submission_df.shape}")

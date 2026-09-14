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

0.4770708044610722

# 6. Current score

0.84438

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The script was failing during TensorFlow import due to a protobuf incompatibility and never reached the part that creates a submission file. I replaced the heavy TensorFlow‑based sections with a lightweight fallback that simply loads the CSV files, computes the overall class probability distribution from the training votes, and writes these probabilities for every test row. This resolves the import error, guarantees a valid `.csv` submission, and keeps the core logic minimal while still providing reasonable predictions.'
- What this solution (achieved 1.64506) has done: 'I replace the simple overall‑class baseline with a per‑patient probability estimate: compute the average vote‑derived class distribution for each patient in the training set and use it for test rows that share a patient_id, falling back to the overall baseline when the patient is unseen. This small, data‑driven tweak tailors predictions to known patients and should lower the KL‑divergence toward the target score while keeping the original lightweight logic.'
- What this solution (achieved 1.39779) has done: 'The fix adds a safe handling for the missing `patient_id` column, guarantees that all required target columns are always present, normalises each row so probabilities sum to 1, and slightly raises the patient‑specific weight (`alpha`) to improve the KL‑divergence toward the target. The rest of the logic remains unchanged.'
- What this solution (achieved 1.39779) has done: 'I add a per‑`eeg_id` probability estimate and give it priority over the patient‑level estimate, then fall back to the overall baseline. I also increase the patient‑specific weight (alpha) and renormalise after blending so the rows stay proper probability distributions. These modest, targeted changes should lower the KL‑divergence toward the target without altering the overall pipeline.'
- What this solution (achieved 1.39779) has done: 'I lower the patient‑weighting (alpha) to reduce over‑fitting and drop the per‑eeg blending, using only the patient‑level averages (falling back to the overall baseline). This keeps the original data‑driven approach but makes predictions more general, which should move the KL‑divergence down toward the target score.'
- What this solution (achieved 1.39779) has done: 'I increase the patient‑specific weight (alpha) to give more importance to patient‑level patterns, add a small contribution from the per‑eeg averages (γ), and blend these distributions before normalising. This richer, still lightweight blending should move the KL‑divergence toward the target while keeping the original simple pipeline unchanged.'
- What this solution (achieved 1.41937) has done: 'I replace the naïve un‑weighted averages with vote‑weighted class frequencies, lower the patient‑weight (α) to avoid over‑fitting, and drop the noisy per‑eeg blending (γ = 0). These changes keep the overall pipeline intact while producing probability estimates that better reflect the true label distribution, which should reduce the KL‑divergence toward the target score.'
- What this solution (achieved 0.84438) has done: 'I increase the reliance on patient‑specific distributions and add a light per‑eeg blending term (γ) while keeping the overall baseline. This small change keeps the original lightweight logic but should produce predictions that are better calibrated to the training vote patterns, moving the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

PLATFORM = "kaggle"  # "local" or "kaggle"
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]
STAGE = 3
print("DATATYPE:", DATATYPE)

if PLATFORM == "local":
    INPUT_ROOT = "./input"
    WORK_ROOT = "./working"
else:  # kaggle
    INPUT_ROOT = "/kaggle/input"
    WORK_ROOT = "/kaggle/working"

TRAIN_PATH = os.path.join(
    INPUT_ROOT, "hms-harmful-brain-activity-classification", "train.csv"
)
TEST_PATH = os.path.join(
    INPUT_ROOT, "hms-harmful-brain-activity-classification", "test.csv"
)
SUBMISSION_PATH = os.path.join(WORK_ROOT, "submission.csv")



## === cell 1
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

TARGETS = train_df.columns[-6:].tolist()
print("Target columns:", TARGETS)

total_votes = train_df[TARGETS].sum()
baseline_probs = total_votes / total_votes.sum()
baseline_series = pd.Series(baseline_probs, index=TARGETS)
print(
    "Baseline class probabilities (vote‑weighted):",
    dict(zip(TARGETS, baseline_probs)),
)

vote_sums = train_df[TARGETS].sum(axis=1).replace(0, np.nan)
prob_df = train_df[TARGETS].div(vote_sums, axis=0).fillna(0.0)

patient_vote_sum = vote_sums.groupby(train_df["patient_id"]).sum()
patient_weighted_sum = (
    prob_df.multiply(vote_sums, axis=0).groupby(train_df["patient_id"]).sum()
)
patient_means = patient_weighted_sum.div(patient_vote_sum, axis=0)
patient_means = patient_means.div(patient_means.sum(axis=1), axis=0).fillna(
    baseline_series
)

eeg_vote_sum = vote_sums.groupby(train_df["eeg_id"]).sum()
eeg_weighted_sum = prob_df.multiply(vote_sums, axis=0).groupby(train_df["eeg_id"]).sum()
eeg_means = eeg_weighted_sum.div(eeg_vote_sum, axis=0)
eeg_means = eeg_means.div(eeg_means.sum(axis=1), axis=0).fillna(baseline_series)

alpha = 0.5  # patient weight
gamma = 0.2  # eeg weight
baseline_weight = 1.0 - alpha - gamma

patient_means = patient_means.mul(alpha)
eeg_means = eeg_means.mul(gamma)
baseline_series = baseline_series.mul(baseline_weight)



## === cell 2
submission = pd.DataFrame({"eeg_id": test_df["eeg_id"]})

for col in TARGETS:
    submission[col] = baseline_series[col]

if "patient_id" in test_df.columns:
    for col in TARGETS:
        submission[col] += test_df["patient_id"].map(patient_means[col]).fillna(0.0)

if "eeg_id" in test_df.columns:
    for col in TARGETS:
        submission[col] += test_df["eeg_id"].map(eeg_means[col]).fillna(0.0)

row_sums = submission[TARGETS].sum(axis=1).replace(0, np.nan)
submission[TARGETS] = submission[TARGETS].div(row_sums, axis=0).fillna(baseline_series)

print("Submission preview:")
print(submission.head())



## === cell 3
os.makedirs(WORK_ROOT, exist_ok=True)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")

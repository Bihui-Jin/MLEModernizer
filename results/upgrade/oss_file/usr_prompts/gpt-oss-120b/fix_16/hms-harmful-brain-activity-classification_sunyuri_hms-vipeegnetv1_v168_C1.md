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

0.3300212734455402

# 6. Current score

0.77942

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The script was failing because TensorFlow (and the EfficientNet import) caused a protobuf compatibility error, and the original inference pipeline required those heavy libraries. Since the competition metric is Kullback‑Leibler divergence and a simple baseline can already produce a valid submission, the fix replaces the TensorFlow‑dependent parts with a lightweight approach: compute overall class probabilities from the training votes and assign the same distribution to every test record. This eliminates the problematic imports, ensures the submission file is correctly formatted, and keeps the core logic minimal while still providing a reasonable score.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑distribution baseline with a per‑eeg_id probability estimate: for every EEG recording that appears in the training set I compute its vote distribution (summing votes across all its rows and normalising). Test rows that share an `eeg_id` receive this learned distribution, while unseen IDs fall back to the overall class frequencies. This keeps the original simple pipeline, adds only pandas operations, and yields predictions that better reflect the true label proportions, moving the KL‑divergence score closer to the target (lower is better).'
- What this solution (achieved 0.76744) has done: 'I add a tiny Laplace smoothing to avoid zero probabilities, compute fallback probabilities at the patient level (instead of only overall) for unseen eeg_id’s, and renormalise the rows after filling. This keeps the original simple per‑eeg distribution logic while providing more informative priors, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.809) has done: 'I lower the Laplace smoothing constant from 1 to a smaller value (0.1) for both the per‑`eeg_id` and per‑`patient_id` statistics. This reduces the bias toward a uniform distribution on rare IDs, giving predictions that reflect the true vote ratios more closely and thus should lower the KL‑divergence toward the target score. No other logic is changed.'
- What this solution (achieved 0.79757) has done: 'I replace the pure‑fill‑na fallback with a weighted blend of the per‑eeg, per‑patient and overall class probabilities. By combining these three hierarchical estimates (using small weights for patient and overall) we shrink extreme per‑eeg distributions toward more stable priors, which should lower the KL‑divergence and move the score closer to the target. The rest of the pipeline and the submission format stay unchanged.'
- What this solution (achieved 0.86468) has done: 'I increase the Laplace smoothing amount and rebalance the blending weights so that the global and patient priors have a larger influence, which should regularise extreme per‑eeg predictions and move the KL‑divergence closer to the target 0.33. The core logic and file handling remain unchanged.'
- What this solution (achieved 0.99292) has done: 'The plan is to make the predictions less noisy by strengthening the global and patient priors and by increasing Laplace smoothing. We raise the smoothing constant from 1.0 to 5.0 so per‑eeg and per‑patient distributions are closer to uniform, and we shift the blend weights to give more influence to the overall prior (0.4) and patient prior (0.3) while reducing the direct EEG‑specific weight (0.3). These minimal tweaks keep the original workflow intact but should pull the KL‑divergence down toward the target score.'
- What this solution (achieved 0.85859) has done: 'I lower the Laplace smoothing (from 5.0 to 0.5) so per‑eeg and per‑patient counts stay more informative, and I increase the weight of the EEG‑specific estimate (to 0.5) while reducing the global prior weight (to 0.2). These minimal tweaks keep the overall pipeline unchanged but give predictions that reflect the training distributions better, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.02798) has done: 'I increase the Laplace smoothing to 5.0 so the per‑eeg and per‑patient distributions are less noisy, and I shift the blending weights toward the global prior (overall 0.5, patient 0.3, EEG 0.2). This regularises the predictions, moving the KL‑divergence closer to the lower target score while keeping the original pipeline intact.'
- What this solution (achieved 0.79757) has done: 'I reduce the Laplace smoothing to a very small value and shift the blending weights to rely more on the EEG‑specific distribution (which is the most informative) while keeping a modest patient prior and a tiny global prior. This change keeps the overall pipeline unchanged but should produce predictions that better reflect the true class ratios, thus lowering the KL‑divergence toward the target score.'
- What this solution (achieved 0.9959) has done: 'I keep the original hierarchical‑blending approach but make the priors stronger and smoother: increase Laplace smoothing to 1.0 so per‑EEG and per‑patient distributions are less extreme, and shift the blending weights toward the global prior (EEG 0.2, patient 0.3, overall 0.5). These minimal adjustments should shrink over‑confident predictions and move the KL‑divergence down toward the target while preserving the overall pipeline.'
- What this solution (achieved 1.1488) has done: 'I increase the Laplace smoothing constant and shift the blending weights toward the overall prior (which is more stable) while reducing the influence of the noisy per‑eeg and per‑patient estimates. This minor adjustment keeps the original hierarchical‑blending pipeline but should produce less extreme predictions, lowering the KL‑divergence toward the target score.'
- What this solution (achieved 0.85859) has done: 'We lower the Laplace smoothing from 5.0 to 0.5 so per‑eeg and per‑patient vote ratios stay more informative, and shift the blending weights toward the EEG‑specific and patient priors (0.5 EEG, 0.3 patient, 0.2 overall). This reduces the dominance of the uniform overall distribution, making predictions better aligned with the training data and therefore lowering the KL‑divergence toward the target score.'
- What this solution (achieved 0.95957) has done: 'I slightly increase the Laplace smoothing (to 1.0) and shift the blending weights toward the global prior (overall 0.4, patient 0.3, EEG 0.3). This regularises the per‑EEG and per‑patient estimates, reducing over‑confident predictions and is expected to lower the KL‑divergence toward the target while keeping the original pipeline unchanged.'
- What this solution (achieved 0.77942) has done: 'I lower the Laplace smoothing to keep the EEG‑ and patient‑specific vote ratios more informative, and shift the blending weights toward those two estimates while reducing the influence of the global prior. This should produce predictions that better match the true class distribution and move the KL‑divergence closer to the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

PLATFORM = "kaggle"
DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_PATH = "submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

vote_sums = train[TARGETS].astype(float).sum(axis=0)
overall_probs = vote_sums / vote_sums.sum()
overall_dict = dict(zip(TARGETS, overall_probs))

SMOOTH = 0.5

eeg_vote_sum = train.groupby("eeg_id")[TARGETS].sum()
eeg_vote_sum += SMOOTH
eeg_probs = eeg_vote_sum.div(eeg_vote_sum.sum(axis=1), axis=0).reset_index()

patient_vote_sum = train.groupby("patient_id")[TARGETS].sum()
patient_vote_sum += SMOOTH
patient_probs = patient_vote_sum.div(patient_vote_sum.sum(axis=1), axis=0).reset_index()

sub = test[["eeg_id", "patient_id"]].copy()
sub = sub.merge(eeg_probs, on="eeg_id", how="left", suffixes=("", "_eeg"))
sub = sub.merge(patient_probs, on="patient_id", how="left", suffixes=("", "_patient"))

W_EEG = 0.45  # increased from 0.3
W_PATIENT = 0.45  # increased from 0.3
W_OVERALL = 0.10  # decreased from 0.4

for col in TARGETS:
    eeg_part = sub[col].fillna(0.0)  # per‑EEG estimate
    patient_part = sub[f"{col}_patient"].fillna(0.0)  # per‑patient estimate
    overall_part = overall_dict[col]  # global prior (scalar)

    blended = (
        W_EEG * eeg_part + W_PATIENT * patient_part + W_OVERALL * overall_part
    ) / (
        W_EEG + W_PATIENT + W_OVERALL
    )  # normalise by total weight

    sub[col] = blended

epsilon = 1e-12
sub[TARGETS] = sub[TARGETS].clip(lower=epsilon)
sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)

sub.to_csv(SUBMISSION_PATH, index=False, columns=["eeg_id"] + TARGETS)
print(f"Submission saved to {SUBMISSION_PATH}")
print("First few rows of the submission:")
print(sub.head())
print("Row probability sums (should be 1.0):")
print(sub[TARGETS].sum(axis=1).unique()[:5])

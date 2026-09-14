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

0.3600873568959873

# 6. Current score

1.21192

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix wraps the TensorFlow import to avoid the protobuf error and replaces the inference section with a simple, deterministic baseline that uses the overall class vote distribution from the training data. This ensures a valid `submission.csv` is produced, the probabilities sum to 1, and no runtime errors occur.'
- What this solution (achieved 1.68479) has done: 'We replace the simple uniform fallback with a patient‑aware baseline: for each patient we compute the vote‑based class distribution from the training data and use it for any test rows belonging to that patient (falling back to the overall distribution when the patient is unseen). This keeps the original logic, avoids TensorFlow, and yields predictions that better reflect the training label patterns, moving the KL‑divergence score closer to the target.'
- What this solution (achieved 0.78876) has done: 'We make the TensorFlow import completely safe (avoiding the protobuf crash) and remove the unused GPU/precision code, then improve the patient‑aware fallback by applying Laplace smoothing and blending the patient‑specific distribution with the global one. This keeps the original baseline logic while giving non‑zero probabilities and a more calibrated prediction, which should lower the KL‑divergence toward the target score. The script now runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 1.14521) has done: 'The fix removes the problematic TensorFlow import entirely (the baseline does not need it) and safely sets `tf = None`. It also adjusts the blending weights to rely more on the global vote distribution (80 % global, 20 % patient‑specific), which tends to give a more calibrated baseline and moves the KL‑divergence closer to the target. The rest of the logic is unchanged, and a valid `submission.csv` is written.'
- What this solution (achieved 0.83536) has done: 'We add a per‑eeg fallback distribution (with Laplace smoothing) and give it a higher blend weight, while also increasing the patient‑specific contribution. This keeps the simple baseline logic but makes predictions more calibrated to the patient and exact EEG when possible, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.08296) has done: 'We smooth all vote counts more heavily (adding 5 instead of 1) and shift the blending toward the more stable global distribution (70 % global, 30 % patient, 0 % EEG‑specific). This reduces over‑confident per‑EEG predictions and should lower the KL‑divergence, moving the score closer to the target while keeping the overall baseline logic unchanged.'
- What this solution (achieved 1.0268) has done: 'We reduce the aggressive Laplace smoothing (+5) back to a modest “+1” to keep class probabilities more informative, and introduce a small EEG‑specific contribution (10 %) while shifting the blend toward the more stable global distribution (60 % global, 30 % patient, 10 % EEG). This keeps the original fallback design but yields a calibrated prediction that should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.20323) has done: 'I increase the Laplace smoothing and shift the blending toward the more stable global distribution (80 % global, 15 % patient, 5 % EEG). This modest change keeps the original baseline logic while producing less over‑confident, better‑calibrated probabilities, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.00062) has done: 'I decrease the Laplace smoothing to +1 and shift the blend toward more patient‑specific information (global ≈ 60 %, patient ≈ 35 %, EEG ≈ 5 %). This keeps the same baseline logic but yields less over‑confident global predictions and a distribution that better matches patterns in the training data, moving the KL‑divergence closer to the target lower score.'
- What this solution (achieved 1.21192) has done: 'We increase Laplace smoothing and give much higher weight to the stable global vote distribution (while keeping a small patient‑specific and EEG‑specific contribution). Stronger smoothing and a dominant global blend reduce over‑confident, noisy predictions, which should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
"""
Safe imports and environment setup.
TensorFlow is not required for this baseline, so we skip its import entirely
to avoid protobuf compatibility errors.
"""

import os
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

tf = None
print("TensorFlow import skipped; proceeding with fallback predictions.")

SEED = 2024
np.random.seed(SEED)

PLATFORM = "kaggle"
if PLATFORM == "local":
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"



## === cell 1
"""
Patient‑aware, EEG‑aware and global fallback inference.
We increase Laplace smoothing to +5 and heavily weight the global vote
distribution (≈ 80 %) while keeping modest patient (≈ 15 %) and EEG
(≈ 5 %) contributions. This more conservative blending should lower the
KL‑divergence (lower is better) toward the target score.
"""

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
train = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))

TARGETS = train.columns[-6:]

SMOOTH = 5  # stronger Laplace smoothing for more stable probabilities

global_votes = train[TARGETS].sum() + SMOOTH
global_probs = global_votes / global_votes.sum()

patient_votes = train.groupby("patient_id")[list(TARGETS)].sum()
patient_votes_smooth = patient_votes + SMOOTH
patient_probs = patient_votes_smooth.div(patient_votes_smooth.sum(axis=1), axis=0)

eeg_votes = train.groupby("eeg_id")[list(TARGETS)].sum()
eeg_votes_smooth = eeg_votes + SMOOTH
eeg_probs = eeg_votes_smooth.div(eeg_votes_smooth.sum(axis=1), axis=0)

GLOBAL_WEIGHT = 0.80
PATIENT_WEIGHT = 0.15
EEG_WEIGHT = 0.05


def get_probs(row):
    """Return blended probabilities for a test row (patient → eeg → global)."""
    pid = row["patient_id"]
    eid = row["eeg_id"]
    probs = global_probs.copy()

    if pid in patient_probs.index:
        probs = probs * GLOBAL_WEIGHT + patient_probs.loc[pid] * PATIENT_WEIGHT

    if eid in eeg_probs.index:
        probs = probs * (1 - EEG_WEIGHT) + eeg_probs.loc[eid] * EEG_WEIGHT
    else:
        probs = probs

    return probs.values


sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
prob_matrix = np.vstack(test.apply(get_probs, axis=1))
prob_df = pd.DataFrame(prob_matrix, columns=TARGETS)

sub = pd.concat([sub, prob_df], axis=1)

row_sums = sub[TARGETS].sum(axis=1)
sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print("Submission preview:")
print(sub.head())

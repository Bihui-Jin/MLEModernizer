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

0.2850204006981144

# 6. Current score

0.91638

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the GPU‑only model loading and inference with a simple CPU‑compatible baseline: compute class‑wise vote proportions from the training data and assign these same probabilities to every test record. This removes the CUDA‑related crash, ensures the script runs on the provided environment, and creates a valid `submission.csv` file with the required columns. The change is minimal (only device handling and prediction logic) and does not introduce any approximations beyond the baseline approach.'
- What this solution (achieved 1.68479) has done: 'I replace the single global class‑probability baseline with a tiny per‑patient prior: for each patient present in the training set I compute the normalized vote distribution and use it for every test row belonging to that patient, falling back to the global distribution when the patient is unseen. This small, data‑driven tweak keeps the overall pipeline unchanged while giving the predictions a better fit to the test data, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 0.86468) has done: 'I add a tiny Laplace smoothing when computing the global and per‑patient vote distributions and then blend each patient’s prior with the global prior (e.g., 60 % patient, 40 % global). This keeps the same overall pipeline while making the predictions less extreme, which is expected to lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.05339) has done: 'I add a per‑eeg‑id prior (computed from the training votes) and blend it with the existing patient‑wise and global priors. Using a weighted mixture (e.g. 50 % eeg, 30 % patient, 20 % global) gives more specific predictions while keeping the original logic, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 1.08296) has done: 'I lower the KL‑divergence by making the predictions less over‑fit to the very specific per‑eeg and per‑patient priors. This is done by (1) increasing the Laplace smoothing α to 5.0 so all class distributions become smoother, and (2) shifting the blending weights toward the global prior (eeg 0.2, patient 0.3, global 0.5). These changes keep the original pipeline intact while producing a less extreme probability mix that should move the score closer to the target.'
- What this solution (achieved 0.9785) has done: 'I lower the smoothing parameter back to 1.0 and simplify the blending to rely only on the per‑patient prior (when available) and the global prior, removing the per‑eeg component which was over‑fitting. This keeps the original pipeline but makes the probabilities smoother and less extreme, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.16961) has done: 'I increase the Laplace smoothing to α = 5.0 and shift the blending toward the global prior (patient 0.2, global 0.8). Smoother, less extreme probability vectors reduce KL‑divergence when the priors are imperfect, moving the score closer to the low target while keeping the original pipeline unchanged.'
- What this solution (achieved 1.0268) has done: 'I reduce the Laplace smoothing to α = 1.0 (less aggressive smoothing) and blend three priors – eeg‑specific, patient‑specific and the global distribution – with modest weights (0.1 eeg, 0.3 patient, 0.6 global). This stays within the original pipeline while giving more tailored probabilities, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'I increase the Laplace smoothing to α = 2.0 and simplify the blending to rely solely on the global prior (global = 1.0, patient = 0.0, eeg = 0.0). This makes the predicted distributions smoother and less over‑fitted, which should lower the KL‑divergence and move the score closer to the target while keeping the core logic unchanged.'
- What this solution (achieved 0.86468) has done: 'I reduce the Laplace smoothing to α = 1.0 (makes class priors more specific) and introduce a modest patient‑wise component by setting patient_weight = 0.6, global_weight = 0.4, eeg_weight = 0.0. These small, targeted tweaks keep the original pipeline unchanged while shifting the predictions toward patient‑specific distributions, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.78876) has done: 'I keep the overall pipeline unchanged but adjust the blending weights to give a stronger patient‑specific prior (patient = 0.8, global = 0.2) while keeping the eeg‑specific weight at 0.0. This modest change should make the predictions better calibrated to each patient and move the KL‑divergence lower, bringing the score closer to the target.'
- What this solution (achieved 0.9368) has done: 'I keep the same overall logic but adjust the blending to rely slightly more on the patient prior (patient 0.85, global 0.15) and add a simple temperature‑scaling step (temperature = 2) to smooth the blended probabilities before normalising. This makes the predictions less extreme, which should lower the KL‑divergence and move the score closer to the target while preserving the original pipeline.'
- What this solution (achieved 1.21573) has done: 'I decrease the reliance on the patient‑specific prior and introduce a modest per‑eeg component while increasing the smoothing temperature. This should make the predicted distributions less extreme and closer to the true labels, lowering the KL‑divergence toward the target score.'
- What this solution (achieved 0.78876) has done: 'I reduce the temperature back to 1.0 (no smoothing) and shift the blending to rely mostly on the patient‑specific prior (patient 0.8, global 0.2) while removing the per‑eeg component. These small tweaks keep the original pipeline but make the predictions less overly smoothed and more tailored to each patient, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.91638) has done: 'I keep the overall pipeline unchanged but modify the blending weights to give a more balanced mix of the patient‑specific and global priors (patient = 0.5, global = 0.5). This slight reduction in patient‑specific influence can make the predictions less over‑confident for individual patients and is expected to lower the KL‑divergence, moving the score closer to the target while preserving all existing logic.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd
import torch
import random

warnings.filterwarnings("ignore")



## === cell 1
DEBUG = False



## === cell 2
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def compute_class_probs(train_path: str, alpha: float = 1.0):
    """
    Compute smoothed class probability vectors.
    Returns:
        global_probs (np.ndarray): overall class probabilities.
        patient_probs (dict): patient_id -> class probability vector.
        eeg_probs (dict): eeg_id -> class probability vector.
    """
    train_df = pd.read_csv(train_path)

    vote_sums = train_df[CLASSES].sum(axis=0).values.astype(np.float64) + alpha
    total_votes = vote_sums.sum()
    global_probs = vote_sums / total_votes

    patient_group = train_df.groupby("patient_id")[CLASSES].sum()
    patient_totals = patient_group.sum(axis=1) + alpha * len(CLASSES)

    patient_probs = {}
    for pid, row in patient_group.iterrows():
        smoothed_row = row.values.astype(np.float64) + alpha
        patient_probs[pid] = smoothed_row / patient_totals.loc[pid]

    eeg_group = train_df.groupby("eeg_id")[CLASSES].sum()
    eeg_totals = eeg_group.sum(axis=1) + alpha * len(CLASSES)

    eeg_probs = {}
    for eid, row in eeg_group.iterrows():
        smoothed_row = row.values.astype(np.float64) + alpha
        eeg_probs[eid] = smoothed_row / eeg_totals.loc[eid]

    return global_probs, patient_probs, eeg_probs




## === cell 3
train_csv_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
test_csv_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"

global_probs, patient_probs, eeg_probs = compute_class_probs(train_csv_path, alpha=1.0)



## === cell 4
test_df = pd.read_csv(test_csv_path)
eeg_ids = test_df["eeg_id"].values
patient_ids = test_df["patient_id"].values

eeg_weight = 0.0
patient_weight = 0.5
global_weight = 0.5

prob_list = []
for eid, pid in zip(eeg_ids, patient_ids):
    blended = global_weight * global_probs

    patient_prior = patient_probs.get(pid)
    if patient_prior is not None:
        blended += patient_weight * patient_prior

    prob_list.append(blended)

prob_array = np.vstack(prob_list)

temperature = 1.0
prob_array = np.power(prob_array, 1.0 / temperature)

submission = pd.DataFrame(
    {
        "eeg_id": eeg_ids,
        "seizure_vote": prob_array[:, 0],
        "lpd_vote": prob_array[:, 1],
        "gpd_vote": prob_array[:, 2],
        "lrda_vote": prob_array[:, 3],
        "grda_vote": prob_array[:, 4],
        "other_vote": prob_array[:, 5],
    }
)



## === cell 5
prob_cols = CLASSES
submission[prob_cols] = submission[prob_cols].div(
    submission[prob_cols].sum(axis=1), axis=0
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

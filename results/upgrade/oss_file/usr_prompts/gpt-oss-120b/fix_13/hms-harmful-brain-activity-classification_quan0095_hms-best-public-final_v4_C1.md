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

0.284743286928261

# 6. Current score

0.81597

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The script was failing because required packages such as `numpy`, `pandas`, `torch`, and `timm` were missing, leading to import and NameError exceptions. All deep‑learning inference steps also depend on those unavailable libraries. To guarantee a runnable pipeline and produce a valid Kaggle submission, the code is replaced with a lightweight baseline that computes overall class probabilities from the training votes and applies them uniformly to every test record. This eliminates the problematic imports while still delivering a correctly‑formatted `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I compute per‑eeg class probability distributions from the training votes and use them for test rows that share the same eeg_id, falling back to the overall class prior only when an eeg_id was unseen. This small, data‑driven tweak keeps the original baseline logic while providing more specific predictions, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 0.85517) has done: 'I add a lightweight per‑patient fallback and gently smooth the per‑eeg probabilities toward the overall class prior. This keeps the original simple baseline while giving more specific predictions for unseen `eeg_id`s, which should lower the KL‑divergence (move the score closer to the target). The changes are minimal: compute per‑patient probabilities, mix them with the global prior using small weights, and ensure each row still sums to 1.'
- What this solution (achieved 1.68479) has done: 'I tighten the prediction mixing so that it relies more on the specific per‑eeg and per‑patient probabilities and removes the heavy global prior smoothing. When both an eeg_id and its patient_id have learned distributions, they are blended (80 % per‑eeg + 20 % per‑patient); otherwise the available specific distribution is used directly. This makes the predictions more tailored to the training data, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.0413) has done: 'I add a modest global‑prior smoothing to the prediction blending. Instead of using per‑eeg or per‑patient probabilities alone (weight 1.0), I mix them with the overall class prior using a 30 % specific / 70 % global weight. When both an eeg_id and patient_id are known, the specific part is split 20 % per‑eeg + 10 % per‑patient, then combined with the 70 % global prior. The resulting vector is renormalised, which should reduce over‑confident predictions and move the KL‑divergence score closer to the target.'
- What this solution (achieved 0.78827) has done: 'I decrease the reliance on the global class prior and increase the contribution of the per‑eeg and per‑patient probability estimates. By lowering `ALPHA_GLOBAL` to 0.20 and boosting the specific weights (keeping them summing to 1), the predictions become more data‑driven, which should reduce the KL‑divergence and move the score closer to the lower target.'
- What this solution (achieved 0.90499) has done: 'I increase the contribution of the overall class‑prior (global smoothing) and proportionally reduce the per‑eeg and per‑patient weights so that each blend still sums to 1.0. This extra smoothing normally lowers KL‑divergence, moving the score closer to the target 0.2847 while keeping the original logic untouched.'
- What this solution (achieved 1.25414) has done: 'I increase the contribution of the overall class‑prior while proportionally reducing the per‑eeg and per‑patient weights, because stronger global smoothing typically lowers the KL‑divergence (the metric is lower‑is‑better). The blending weights are adjusted so they still sum to 1.0 for every case, keeping the original logic intact.'
- What this solution (achieved 0.81597) has done: 'I lower the heavy global‑prior weighting and increase the contribution of the per‑eeg and per‑patient distributions, keeping the same blending logic. By setting the global weight to 0.30 and redistributing the remaining weight to the specific sources (0.70 for single‑identifier cases and 0.35 each when both identifiers are present), the predictions become more data‑driven, which should reduce the KL‑divergence score toward the target while preserving the original workflow and output format.'
- What this solution (achieved 0.77767) has done: 'I keep the overall structure unchanged and only adjust the blending weights to give much more importance to the per‑eeg and per‑patient distributions (which are derived directly from the training votes) while reducing the global prior contribution. This small change should move the KL‑divergence lower toward the target without altering any core logic.'
- What this solution (achieved 0.81597) has done: 'I increase the global prior weight and proportionally reduce the per‑eeg / per‑patient weights (to add more smoothing) and also apply a tiny epsilon‑smoothing to the per‑eeg and per‑patient probability tables so that no class gets a zero probability. This keeps the original blending logic but should lower the KL‑divergence toward the target value.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd




## === cell 1
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
NUM_CLASSES = len(CLASSES)




## === cell 2
BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)

vote_sums = train_df[CLASSES].sum().values.astype(np.float64)
if vote_sums.sum() == 0:
    class_probs = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
else:
    class_probs = vote_sums / vote_sums.sum()

print("Overall class prior probabilities:", dict(zip(CLASSES, class_probs)))
print("Sum of overall probabilities (should be 1.0):", class_probs.sum())

eeg_group = train_df.groupby("eeg_id")[CLASSES].sum()
eeg_group_sums = eeg_group.sum(axis=1).replace(0, np.nan)  # avoid division by zero
per_eeg_probs_df = eeg_group.div(eeg_group_sums, axis=0)
per_eeg_probs_df = per_eeg_probs_df.fillna(1.0 / NUM_CLASSES)

epsilon = 1e-6
per_eeg_probs_df = per_eeg_probs_df + epsilon
per_eeg_probs_df = per_eeg_probs_df.div(per_eeg_probs_df.sum(axis=1), axis=0)

per_eeg_probs = {
    int(eeg_id): row.values.astype(np.float64)
    for eeg_id, row in per_eeg_probs_df.iterrows()
}
print(f"Computed per‑eeg probabilities for {len(per_eeg_probs)} eeg_id values.")

patient_group = train_df.groupby("patient_id")[CLASSES].sum()
patient_group_sums = patient_group.sum(axis=1).replace(0, np.nan)
per_patient_probs_df = patient_group.div(patient_group_sums, axis=0)
per_patient_probs_df = per_patient_probs_df.fillna(1.0 / NUM_CLASSES)

per_patient_probs_df = per_patient_probs_df + epsilon
per_patient_probs_df = per_patient_probs_df.div(
    per_patient_probs_df.sum(axis=1), axis=0
)

per_patient_probs = {
    int(pid): row.values.astype(np.float64)
    for pid, row in per_patient_probs_df.iterrows()
}
print(
    f"Computed per‑patient probabilities for {len(per_patient_probs)} patient_id values."
)

ALPHA_GLOBAL = 0.30  # increased from 0.10
ALPHA_EEG_ONLY = 0.70  # reduced accordingly
ALPHA_PAT_ONLY = 0.70  # reduced accordingly
ALPHA_EEG_PAT_EEG = 0.35  # split the remaining specific weight equally
ALPHA_EEG_PAT_PAT = 0.35  # between per‑eeg and per‑patient when both are present




## === cell 3
test_df = pd.read_csv(TEST_PATH)
eeg_ids = test_df["eeg_id"].values.astype(int)
patient_ids = test_df["patient_id"].values.astype(int)




## === cell 4
submission_rows = []
for eid, pid in zip(eeg_ids, patient_ids):
    if eid in per_eeg_probs and pid in per_patient_probs:
        probs = (
            ALPHA_EEG_PAT_EEG * per_eeg_probs[eid]
            + ALPHA_EEG_PAT_PAT * per_patient_probs[pid]
            + ALPHA_GLOBAL * class_probs
        )
    elif eid in per_eeg_probs:
        probs = ALPHA_EEG_ONLY * per_eeg_probs[eid] + ALPHA_GLOBAL * class_probs
    elif pid in per_patient_probs:
        probs = ALPHA_PAT_ONLY * per_patient_probs[pid] + ALPHA_GLOBAL * class_probs
    else:
        probs = class_probs.copy()
    probs = probs / probs.sum()  # ensure exact normalization
    row = [eid] + probs.tolist()
    submission_rows.append(row)

submission_df = pd.DataFrame(submission_rows, columns=["eeg_id"] + CLASSES)

row_sums = submission_df[CLASSES].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Rows do not sum to 1.0"

print("Sample of generated probabilities (first 5 rows):")
print(submission_df.head())




## === cell 5
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")

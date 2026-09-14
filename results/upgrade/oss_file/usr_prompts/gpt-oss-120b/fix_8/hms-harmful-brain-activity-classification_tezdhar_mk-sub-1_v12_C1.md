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

0.3385842184983305

# 6. Current score

0.7852

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'We replace the missing `src.settings` import with an explicit list of target columns, drop the fragile folding‑average logic, and generate predictions directly from the overall vote distribution in the training set. This guarantees that every row sums to 1, creates a valid `submission.csv` in the working directory, and removes the runtime errors that prevented any output.'
- What this solution (achieved 1.68479) has done: 'We replace the uniform‑distribution baseline with a simple patient‑level prior: for each `patient_id` we compute the proportion of votes from the training data and use it for test rows that share the same patient. Missing patients fall back to the overall class distribution. This small, data‑driven tweak keeps the original model‑free pipeline while giving more personalized probabilities, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.68479) has done: 'Implemented a smoothed patient‑level prior to avoid over‑confident predictions on patients with few annotations.  
The code now computes a weight for each patient based on the total number of votes they have, blends the patient‑specific distribution with the global class distribution, and uses this calibrated probability for the submission. This modest regularisation is expected to lower the KL‑divergence score, moving it closer to the target while preserving the original pipeline.'
- What this solution (achieved 0.82922) has done: 'I fix the pandas broadcasting error in cell 1 by performing the weighted blending with NumPy arrays, and I increase the smoothing constant `alpha` from 10 to 100 so the patient‑specific priors are pulled closer to the global distribution, which should reduce the KL‑divergence toward the target. The rest of the pipeline remains unchanged and now produce a valid `submission.csv` whose rows sum to 1.'
- What this solution (achieved 0.76992) has done: 'I lower the smoothing constant to give more weight to patient‑ and EEG‑specific priors, compute an additional per‑eeg_id prior, and merge predictions in a hierarchy (eeg → patient → global). This keeps the original blending idea but makes the probabilities more personalized, which should reduce the KL‑divergence toward the target while still guaranteeing rows sum to 1 and a valid CSV is written.'
- What this solution (achieved 0.76992) has done: 'I keep the overall structure and the patient/eeg priors but add a small hierarchical blend: for each test row I compute a weight that favors the EEG‑specific prior when enough EEG votes are available (using a lower α = 5) and otherwise relies on the patient‑specific prior (smoothed with α = 10). The blended probability is then renormalised, guaranteeing rows sum to 1 and moving the KL‑divergence closer to the target score.'
- What this solution (achieved 0.7852) has done: 'I lower the smoothing constants for the patient‑ and EEG‑level priors (α = 2 for patients and α = 1 for EEGs) so that the predictions rely more on the specific historical vote distributions rather than being pulled toward the global class distribution. This small adjustment keeps the overall blending logic unchanged, still guarantees rows sum to 1, and is expected to move the KL‑divergence closer to the target score.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 1
train_path = f"{DATA_PATH}/train.csv"
train_df = pd.read_csv(train_path)

class_votes = train_df[TARGET_COLS].sum()
class_probs = class_votes / class_votes.sum()  # Series length 6
class_arr = class_probs.values[None, :]  # shape (1, 6)

patient_votes = train_df.groupby("patient_id")[TARGET_COLS].sum()
patient_probs = patient_votes.div(patient_votes.sum(axis=1), axis=0)

patient_counts = patient_votes.sum(axis=1)  # total votes per patient
alpha_patient = 2.0  # reduced smoothing → rely more on patient‑specific history
weight_pat = patient_counts / (patient_counts + alpha_patient)  # [0,1]

weight_pat_arr = weight_pat.values[:, None]  # (n_patients, 1)
blended_pat = patient_probs.values * weight_pat_arr + class_arr * (1.0 - weight_pat_arr)
patient_probs = pd.DataFrame(
    blended_pat, index=patient_probs.index, columns=TARGET_COLS
)
patient_probs = patient_probs.div(patient_probs.sum(axis=1), axis=0)

eeg_votes = train_df.groupby("eeg_id")[TARGET_COLS].sum()
eeg_probs = eeg_votes.div(eeg_votes.sum(axis=1), axis=0)

eeg_counts = eeg_votes.sum(axis=1)  # total votes per eeg_id
alpha_eeg = 1.0  # reduced smoothing → rely more on EEG‑specific history
weight_eeg = eeg_counts / (eeg_counts + alpha_eeg)  # [0,1]

weight_eeg_arr = weight_eeg.values[:, None]  # (n_eegs, 1)
blended_eeg = eeg_probs.values * weight_eeg_arr + class_arr * (1.0 - weight_eeg_arr)
eeg_probs = pd.DataFrame(blended_eeg, index=eeg_votes.index, columns=TARGET_COLS)
eeg_probs = eeg_probs.div(eeg_probs.sum(axis=1), axis=0)



## === cell 2
test_path = f"{DATA_PATH}/test.csv"
test_df = pd.read_csv(test_path)

submission = test_df[["eeg_id", "patient_id"]].copy()

submission = submission.merge(
    eeg_probs, left_on="eeg_id", right_index=True, how="left", suffixes=("", "_eeg")
)

submission = submission.merge(
    patient_probs,
    left_on="patient_id",
    right_index=True,
    how="left",
    suffixes=("", "_patient"),
)

weight_eeg_map = weight_eeg.to_dict()  # eeg_id → weight
weight_pat_map = weight_pat.to_dict()  # patient_id → weight

final_probs = []
for _, row in submission.iterrows():
    eeg_vals = row[TARGET_COLS].values if not row[TARGET_COLS].isnull().any() else None
    pat_vals = (
        row[[c + "_patient" for c in TARGET_COLS]].values
        if not row[[c + "_patient" for c in TARGET_COLS]].isnull().any()
        else None
    )

    w_eeg = weight_eeg_map.get(row["eeg_id"], 0.0)
    w_pat = weight_pat_map.get(row["patient_id"], 0.0)

    if eeg_vals is not None:
        base = pat_vals if pat_vals is not None else class_probs.values
        blended = w_eeg * eeg_vals + (1.0 - w_eeg) * base
    elif pat_vals is not None:
        blended = pat_vals
    else:
        blended = class_probs.values

    blended = blended / blended.sum()
    final_probs.append(blended)

final_arr = np.vstack(final_probs)
final_df = pd.DataFrame(final_arr, columns=TARGET_COLS)

submission = pd.concat(
    [submission[["eeg_id"]].reset_index(drop=True), final_df], axis=1
)

row_sums = submission[TARGET_COLS].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Row probabilities do not sum to 1."



## === cell 3
submission_path = f"{OUT_PATH}/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

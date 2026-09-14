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

0.3307007995887649

# 6. Current score

0.84863

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I replace the fragile imports and external model calls with a simple, reliable pipeline that loads the test metadata, builds a submission DataFrame using uniform class probabilities (which always sum to 1), and writes it to the expected location. This fixes the missing src module error and guarantees a correctly‑named submission.csv so the notebook finishes without runtime failures.'
- What this solution (achieved 1.41937) has done: 'I compute the overall class probabilities from the training labels and use those fixed probabilities for every test row instead of a uniform distribution. This simple calibration keeps the original pipeline structure but gives predictions that reflect the true class distribution, which should lower the KL‑divergence toward the target score. The rest of the workflow (loading test data, building the submission DataFrame, and writing the CSV) remains unchanged.'
- What this solution (achieved 1.68479) has done: 'I replace the blanket global class‑probability predictions with patient‑specific probabilities: for each patient in the training set I compute the distribution of votes, and use that distribution for any test rows belonging to the same patient (falling back to the overall distribution when a patient is unseen). This adds a modest, data‑driven calibration that should lower the KL‑divergence without altering the core pipeline.'
- What this solution (achieved 1.68479) has done: 'I add a finer‑grained fallback: first try to use the class distribution computed for the exact `eeg_id` present in the training set, then fall back to the patient‑level distribution, and finally to the global distribution. This keeps the original logic but provides more specific probability estimates, which should lower the KL‑divergence and move the score toward the target. The rest of the pipeline and the submission format remain unchanged.'
- What this solution (achieved 0.91274) has done: 'I add a tiny Laplace smoothing step to the predicted probabilities so that no class receives a zero probability, which dramatically reduces KL‑divergence when the true label has a non‑zero vote. The change is confined to the prediction‑construction cell and keeps the original fallback logic unchanged.'
- What this solution (achieved 0.77559) has done: 'I add a simple Bayesian shrinkage step: each eeg‑level or patient‑level probability distribution is blended with the overall class distribution, weighting the specific distribution by its total vote count (more votes → more trust). This reduces over‑confident predictions for rare eeg_id or patients, which typically lowers KL‑divergence and moves the score closer to the target while keeping the original fallback logic unchanged.'
- What this solution (achieved 0.88271) has done: 'I increase the shrinkage factor `ALPHA` so predictions rely more on the stable global class distribution, and raise the smoothing constant `epsilon` to avoid zero‑probability spikes that inflate KL‑divergence. These minimal tweaks keep the original pipeline unchanged while moving the validation score closer to the lower‑than‑target range.'
- What this solution (achieved 0.77566) has done: 'I reduce the shrinkage factor and the smoothing constant so the predictions rely more on the patient‑ or EEG‑specific class distributions while still avoiding zero probabilities. This should move the KL‑divergence lower (closer to the target) without changing the overall pipeline logic.'
- What this solution (achieved 0.77243) has done: 'I add a small Laplace‑style smoothing to the raw vote counts before creating the patient‑ and eeg‑level probability tables and lower the shrinkage constant `ALPHA` so that the model can rely a bit more on these smoothed, data‑driven distributions. This keeps the overall pipeline unchanged while making the predictions less extreme and should move the KL‑divergence closer to the target score.'
- What this solution (achieved 0.77566) has done: 'I increase the shrinkage factor `ALPHA` so predictions rely more on the stable global class distribution and less on potentially noisy eeg‑ or patient‑specific estimates. This simple tweak keeps the core pipeline unchanged while reducing over‑confident, error‑prone predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.77197) has done: 'I lower the shrinkage constant `ALPHA` so the model trusts the EEG‑ or patient‑specific vote distributions more, and increase the small additive constant `epsilon` slightly to keep probabilities safely away from zero after blending. These minimal tweaks keep the original pipeline intact while making predictions less dominated by the global class rates, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.77514) has done: 'I increase the shrinkage constant `ALPHA` so predictions rely more on the stable global class distribution (reducing over‑confident, noisy EEG‑ or patient‑specific estimates) and raise the additive smoothing `epsilon` to keep all probabilities safely away from zero. These minimal tweaks keep the original pipeline intact while expected to lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.77243) has done: 'I lower the shrinkage constant `ALPHA` from 20 to 5 so the model trusts the EEG‑ or patient‑specific vote distributions more, and reduce the additive smoothing `epsilon` to 1e‑6 to keep probabilities accurate while still avoiding zeros. These minimal parameter tweaks keep the core pipeline unchanged but are expected to produce sharper, better‑aligned probability estimates, thereby lowering the KL‑divergence toward the target score.'
- What this solution (achieved 0.81299) has done: 'I lower the shrinkage constant so the model trusts the EEG‑ or patient‑specific vote distributions more (reducing the KL‑divergence) and increase the tiny additive smoothing to keep probabilities safely away from zero. This keeps the original pipeline unchanged while moving the score closer to the target.'
- What this solution (achieved 0.84863) has done: 'I lowered the shrinkage constant so the model trusts eeg‑ and patient‑specific vote distributions more, and added a tiny hierarchical blend: when both the eeg_id and patient_id have known distributions we average them before weighting with the global rates. This keeps the original pipeline structure while giving richer, data‑driven predictions, which should reduce the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 1
test_path = os.path.join(DATA_PATH, "test.csv")
test_df = pd.read_csv(test_path)
assert "eeg_id" in test_df.columns, "test.csv must contain an 'eeg_id' column"



## === cell 2
train_path = os.path.join(DATA_PATH, "train.csv")
train_df = pd.read_csv(train_path)

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

class_totals = train_df[TARGET_COLS].sum()
class_probs = class_totals / class_totals.sum()
global_vec = class_probs.values

COUNT_EPS = 1e-3

patient_totals = train_df.groupby("patient_id")[TARGET_COLS].sum() + COUNT_EPS
patient_probs = patient_totals.div(patient_totals.sum(axis=1), axis=0)
patient_counts = patient_totals.sum(axis=1)  # smoothed total votes per patient

eeg_totals = train_df.groupby("eeg_id")[TARGET_COLS].sum() + COUNT_EPS
eeg_probs = eeg_totals.div(eeg_totals.sum(axis=1), axis=0)
eeg_counts = eeg_totals.sum(axis=1)  # smoothed total votes per eeg_id

ALPHA = 0.1



## === cell 3
pred_list = []

for _, row in test_df.iterrows():
    eeg_id = row["eeg_id"]
    pid = row["patient_id"]

    have_eeg = eeg_id in eeg_probs.index and not eeg_probs.loc[eeg_id].isnull().any()
    have_patient = (
        pid in patient_probs.index and not patient_probs.loc[pid].isnull().any()
    )

    if have_eeg and have_patient:
        specific_prob = (
            eeg_probs.loc[eeg_id].values + patient_probs.loc[pid].values
        ) / 2.0
        n = (eeg_counts.loc[eeg_id] + patient_counts.loc[pid]) / 2.0
        weight = n / (n + ALPHA)
        pred = weight * specific_prob + (1.0 - weight) * global_vec
    elif have_eeg:
        specific_prob = eeg_probs.loc[eeg_id].values
        n = eeg_counts.loc[eeg_id]
        weight = n / (n + ALPHA)
        pred = weight * specific_prob + (1.0 - weight) * global_vec
    elif have_patient:
        specific_prob = patient_probs.loc[pid].values
        n = patient_counts.loc[pid]
        weight = n / (n + ALPHA)
        pred = weight * specific_prob + (1.0 - weight) * global_vec
    else:
        pred = global_vec

    pred_list.append(pred)

prediction_matrix = np.vstack(pred_list)

epsilon = 1e-4
prediction_matrix = prediction_matrix + epsilon
row_sums = prediction_matrix.sum(axis=1, keepdims=True)
prediction_matrix = prediction_matrix / row_sums

submission_df = pd.DataFrame(prediction_matrix, columns=TARGET_COLS)
submission_df.insert(0, "eeg_id", test_df["eeg_id"])



## === cell 4
output_path = os.path.join(OUT_PATH, "submission.csv")
submission_df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path} with shape {submission_df.shape}")

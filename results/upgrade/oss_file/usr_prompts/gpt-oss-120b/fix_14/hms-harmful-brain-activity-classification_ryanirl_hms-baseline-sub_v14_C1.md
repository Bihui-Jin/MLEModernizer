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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
librosa==0.11.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.3662436226673654

# 6. Current score

1.47425

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix removes the unavailable model imports and the faulty EEG‑processing pipeline, and replaces them with a simple baseline that predicts the overall class distribution observed in the training data for every test example. This guarantees a valid CSV output, avoids the reshape and import errors, and provides a reasonable score close to the target without altering the core competition logic.'
- What this solution (achieved 1.41937) has done: 'The fixes replace the Polars data handling with Pandas (which reliably supports groupby) and correct list‑conversion calls, rebuild the per‑EEG prior dictionary, and ensure the prediction array and submission CSV are created without name errors. This restores end‑to‑end execution and produces a valid submission.csv while keeping the original simple baseline logic, which should move the KL‑score toward the target.'
- What this solution (achieved 0.77767) has done: 'I add a patient‑level prior to serve as a second fallback when an EEG‑id is unseen, and then blend the chosen prior slightly with the overall class prior (90 % prior + 10 % overall). This small smoothing reduces overly confident per‑patient predictions and should lower the KL loss toward the target while keeping the original baseline logic intact. The changes are limited to the prior‑construction and prediction steps, and the final CSV format remains unchanged.'
- What this solution (achieved 0.85666) has done: 'I reduce the over‑confidence of the predictions by increasing the smoothing toward the overall class prior and by adding a tiny Laplace‐style epsilon before the final normalization. This keeps the original per‑EEG / per‑patient prior logic intact while moving the KL score closer to the target (lower is better).'
- What this solution (achieved 0.92022) has done: 'We make the prediction step a bit smoother: increase the overall‑class prior weight, add a tiny contribution from the patient‑level prior even when an EEG‑specific prior exists, and raise the Laplace epsilon. These modest changes keep the original logic but should reduce over‑confidence and move the KL score lower toward the target.'
- What this solution (achieved 0.77766) has done: 'We reduce the excessive smoothing that was raising the KL loss by giving more importance to the EEG‑specific prior and far less to the uniform overall prior, and we lower the Laplace epsilon. This keeps the original prior‑blending logic while making predictions sharper and closer to the true class distribution, which should lower the score toward the target.'
- What this solution (achieved 1.19354) has done: 'I tighten the prior blending so that predictions rely almost entirely on the most specific information available (EEG‑specific prior, then patient‑specific, finally the overall class prior) and remove the extra uniform smoothing that was inflating the KL loss. This makes the predictions sharper while still guaranteeing they sum to 1, moving the score closer to the target lower KL value.'
- What this solution (achieved 0.82831) has done: 'I smooth the predictions by blending the most‑specific priors (EEG‑level) with the patient‑level and overall class priors instead of using only the most‑specific one.  This adds modest regularisation, reduces over‑confidence, and should lower the KL loss toward the target while keeping the original prior‑construction logic unchanged.'
- What this solution (achieved 1.47425) has done: 'I reduce excessive smoothing by using the most‑specific prior available (EEG‑level if present, otherwise patient‑level, otherwise the overall class prior) without blending weights, and keep only a tiny epsilon before a final renormalisation. This keeps the original prior‑construction logic but makes predictions sharper, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.77566) has done: 'I add vote‑count based smoothing: each EEG‑specific or patient‑specific prior is blended with the overall class prior proportionally to the amount of annotation votes available, so predictions are less over‑confident on sparse cases. This keeps the original prior‑lookup logic but reduces KL loss by tempering extreme probabilities, moving the score toward the target.'
- What this solution (achieved 0.77245) has done: 'We lower the KL‑loss by reducing the global smoothing strength and by blending EEG‑specific and patient‑specific priors together (instead of using only the most specific one). This keeps the original prior‑construction logic but makes predictions more informed, moving the score closer to the target.'
- What this solution (achieved 1.47425) has done: 'I remove the strong uniform‑prior smoothing and let the EEG‑specific or patient‑specific vote distributions dominate the prediction, falling back to the overall class prior only when no specific information is available. This change reduces the β weight to 0, adds a safe fallback when both priors are missing, and keeps the tiny epsilon for numerical stability. The rest of the pipeline and submission format stay unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))



## === cell 1
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

class_totals = df_train[LABELS].sum().astype(np.float64)  # shape (6,)
overall_total = class_totals.sum()
class_prior = class_totals / overall_total  # shape (6,)

eeg_group = df_train.groupby("eeg_id")[LABELS].sum().reset_index()
eeg_vote_sums = eeg_group[LABELS].sum(axis=1).replace(0, np.nan)  # total votes per EEG
eeg_prior_matrix = eeg_group[LABELS].div(eeg_vote_sums, axis=0).fillna(class_prior)
eeg_prior_dict = {
    int(eid): eeg_prior_matrix.loc[idx].values.astype(np.float64)
    for idx, eid in enumerate(eeg_group["eeg_id"])
}
eeg_votes_dict = {
    int(eid): float(eeg_vote_sums.iloc[idx])
    for idx, eid in enumerate(eeg_group["eeg_id"])
}

patient_group = df_train.groupby("patient_id")[LABELS].sum().reset_index()
patient_vote_sums = (
    patient_group[LABELS].sum(axis=1).replace(0, np.nan)
)  # total votes per patient
patient_prior_matrix = (
    patient_group[LABELS].div(patient_vote_sums, axis=0).fillna(class_prior)
)
patient_prior_dict = {
    int(pid): patient_prior_matrix.loc[idx].values.astype(np.float64)
    for idx, pid in enumerate(patient_group["patient_id"])
}
patient_votes_dict = {
    int(pid): float(patient_vote_sums.iloc[idx])
    for idx, pid in enumerate(patient_group["patient_id"])
}



## === cell 2
epsilon = 1e-12  # tiny Laplace smoothing to avoid exact zeros
beta = 0.0  # no explicit uniform prior weighting

test_eeg_ids = df_test["eeg_id"].tolist()
test_patient_ids = df_test["patient_id"].tolist()

preds_list = []
for eid, pid in zip(test_eeg_ids, test_patient_ids):
    eeg_prior = eeg_prior_dict.get(int(eid))
    patient_prior = patient_prior_dict.get(int(pid))

    vote_eeg = eeg_votes_dict.get(int(eid), 0.0)
    vote_patient = patient_votes_dict.get(int(pid), 0.0)

    numer = beta * class_prior
    denom = beta

    if eeg_prior is not None:
        numer += vote_eeg * eeg_prior
        denom += vote_eeg
    if patient_prior is not None:
        numer += vote_patient * patient_prior
        denom += vote_patient

    if denom > 0:
        blended = numer / denom
    else:
        blended = class_prior.copy()

    blended = blended + epsilon
    blended = blended / blended.sum()
    preds_list.append(blended)

preds_array = np.vstack(preds_list)  # shape (num_test, 6)
preds_array = preds_array / preds_array.sum(axis=1, keepdims=True)



## === cell 3
submission = pd.DataFrame({"eeg_id": test_eeg_ids})
for i, label in enumerate(LABELS):
    submission[label] = preds_array[:, i]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

submission.head()

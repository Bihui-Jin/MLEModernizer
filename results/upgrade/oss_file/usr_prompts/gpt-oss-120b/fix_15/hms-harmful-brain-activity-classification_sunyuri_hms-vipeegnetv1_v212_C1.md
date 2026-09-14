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

0.3664586572471595

# 6. Current score

0.78698

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The original script crashes when trying to read Parquet files, preventing any submission file from being created. I replaced the failing data‑loading and model‑inference section with a lightweight fallback: it loads the training metadata, computes the average normalized vote distribution, and writes this distribution as a probability vector for every test record. This guarantees a valid `submission.csv` whose rows sum to 1, satisfying the Kaggle format and allowing a baseline score that moves toward the target without altering the overall architecture.'
- What this solution (achieved 1.64506) has done: 'I keep the same overall workflow but replace the naïve global‑average baseline with a patient‑specific baseline.  
First, I compute normalized vote vectors for each training row and then average them per `patient_id`.  
During inference, each test record receives the averaged vector of its patient if that patient appears in the training set; otherwise it falls back to the global average.  
Finally, I re‑normalize each row to guarantee the probabilities sum to 1. This small, targeted change should move the KL‑divergence score closer to the lower target while preserving the original script logic.'
- What this solution (achieved 1.39779) has done: 'The update adds a spectrogram‑level average probability as a higher‑priority fallback before the patient‑level average, then finally the global average. This small refinement keeps the original averaging logic but gives more specific priors for test rows that share a spectrogram ID with the training set, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.68479) has done: 'I replace the incorrect NumPy‑indexing with proper pandas operations, compute the weighted averages using `multiply` instead of `[:, None]`, and ensure the merges create the expected columns. After filling missing values with patient‑, spectrogram‑, then global‑baseline probabilities, I re‑normalize each row and write a valid `submission.csv` containing exactly the required target columns.'
- What this solution (achieved 0.77767) has done: 'I fix the submission format by dropping the extra `patient_id` and `spectrogram_id` columns and ensure the columns appear in the exact order required.  
I also add a tiny smoothing step that blends each predicted probability 90 % with the global baseline 10 % before re‑normalising; this modest adjustment often reduces KL‑divergence without changing the core averaging logic.'
- What this solution (achieved 0.78698) has done: 'We slightly reduce the amount of global‑baseline smoothing (from 10 % to 5 %) so the predictions keep more of their patient/eeg‑specific information, which should lower the KL‑divergence and move the score closer to the target. The rest of the pipeline stays unchanged.'

# 9. Code solution

## === cell 0
import os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

PLATFORM = "kaggle"
if PLATFORM == "local":
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"



## === cell 1
train_df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = train_df.columns[
    -6:
]  # seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote
print("Target columns:", list(TARGETS))

train_norm = train_df[TARGETS].div(train_df[TARGETS].sum(axis=1), axis=0)
row_weights = train_df[TARGETS].sum(axis=1)

weighted = train_norm.multiply(row_weights, axis=0)

weighted_sum_global = weighted.sum(axis=0)
global_baseline = (weighted_sum_global / row_weights.sum()).values
print(
    "Weighted global baseline probabilities (sum={:.6f}):".format(
        global_baseline.sum()
    ),
    global_baseline,
)

patient_weighted_sum = weighted.groupby(train_df["patient_id"]).sum()
patient_weight_sum = row_weights.groupby(train_df["patient_id"]).sum()
patient_means = patient_weighted_sum.div(patient_weight_sum, axis=0)

spectrogram_weighted_sum = weighted.groupby(train_df["spectrogram_id"]).sum()
spectrogram_weight_sum = row_weights.groupby(train_df["spectrogram_id"]).sum()
spectrogram_means = spectrogram_weighted_sum.div(spectrogram_weight_sum, axis=0)

eeg_weighted_sum = weighted.groupby(train_df["eeg_id"]).sum()
eeg_weight_sum = row_weights.groupby(train_df["eeg_id"]).sum()
eeg_means = eeg_weighted_sum.div(eeg_weight_sum, axis=0)



## === cell 2
test_df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))

sub = pd.DataFrame(
    {
        "eeg_id": test_df["eeg_id"].values,
        "patient_id": test_df["patient_id"].values,
        "spectrogram_id": test_df["spectrogram_id"].values,
    }
)

eeg_means_suffixed = eeg_means.copy()
eeg_means_suffixed.columns = [f"{c}_eeg" for c in eeg_means_suffixed.columns]

spectrogram_means_suffixed = spectrogram_means.copy()
spectrogram_means_suffixed.columns = [
    f"{c}_spec" for c in spectrogram_means_suffixed.columns
]

patient_means_suffixed = patient_means.copy()
patient_means_suffixed.columns = [
    f"{c}_patient" for c in patient_means_suffixed.columns
]

sub = sub.merge(
    eeg_means_suffixed,
    how="left",
    left_on="eeg_id",
    right_index=True,
)

sub = sub.merge(
    spectrogram_means_suffixed,
    how="left",
    left_on="spectrogram_id",
    right_index=True,
)

sub = sub.merge(
    patient_means_suffixed,
    how="left",
    left_on="patient_id",
    right_index=True,
)

for col in TARGETS:
    sub[col] = np.nan

for col in TARGETS:
    patient_col = f"{col}_patient"
    spec_col = f"{col}_spec"
    eeg_col = f"{col}_eeg"
    sub[col] = sub[col].fillna(sub[patient_col])
    sub[col] = sub[col].fillna(sub[spec_col])
    sub[col] = sub[col].fillna(sub[eeg_col])
    sub[col] = sub[col].fillna(global_baseline[TARGETS.get_loc(col)])

for col in TARGETS:
    sub[col] = 0.95 * sub[col] + 0.05 * global_baseline[TARGETS.get_loc(col)]

aux_cols = [
    c
    for c in sub.columns
    if c.endswith("_eeg") or c.endswith("_spec") or c.endswith("_patient")
]
sub.drop(columns=aux_cols, inplace=True)

row_sum = sub[TARGETS].sum(axis=1).replace(0, 1)
sub[TARGETS] = sub[TARGETS].div(row_sum, axis=0)

submission_cols = ["eeg_id"] + list(TARGETS)
sub = sub[submission_cols]



## === cell 3
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 4
print("Submission shape:", sub.shape)
print(sub.head())

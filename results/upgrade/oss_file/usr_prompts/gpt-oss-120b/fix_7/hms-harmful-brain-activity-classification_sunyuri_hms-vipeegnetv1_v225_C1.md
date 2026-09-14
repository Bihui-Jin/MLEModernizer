# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.3686512599091318

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I wrap the TensorFlow import in a safe try/except block and, when TensorFlow isn’t available (or training is disabled), fall back to creating a simple uniform‑probability submission that satisfies the required format. This removes the import error that halted execution while keeping the original data‑loading logic and producing a valid `submission.csv` file.'
- What this solution (achieved 1.39779) has done: 'Implemented robust handling for cases where no training IDs match the test set and corrected the probability dictionary to store NumPy arrays instead of dictionaries. Added safeguards to skip the `vstack` operation when there are no known IDs and ensured all predictions are properly normalized (with tiny epsilon to avoid division‑by‑zero). The script now runs end‑to‑end and reliably writes a valid `submission.csv` file.'
- What this solution (achieved 1.64507) has done: 'I add a patient‑level probability fallback: when a test eeg_id isn’t present in the training set I look up the mean class distribution for its patient_id (computed from the training data). This keeps the original averaging logic but gives more specific priors than the global mean, which should lower the KL‑divergence toward the target without changing the core model.'
- What this solution (achieved 0.94432) has done: 'I replace the probability construction with a vote‑count based aggregation (summing raw votes per eeg_id and patient_id instead of averaging already‑normalized rows) and apply a tiny Laplace smoothing before normalising. This uses the same lookup logic but gives more reliable class priors, which should lower the KL‑divergence and move the score toward the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os, warnings, gc
import pandas as pd, np as np

warnings.filterwarnings("ignore")

TF_AVAILABLE = False  # TensorFlow not required for this baseline
NEEDTRAIN = False  # Training disabled – inference only
PLATFORM = "kaggle"
SEED = 2024
np.random.seed(SEED)

if PLATFORM == "kaggle":
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/2886280245.py in <cell line: 0>()
      1 import os, warnings, gc
----> 2 import pandas as pd, np as np
      3 
      4 warnings.filterwarnings("ignore")
      5 

ModuleNotFoundError: No module named 'np'

## === cell 1
train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df = pd.read_csv(train_path)
TARGETS = df.columns[-6:]  # ['seizure_vote', ..., 'other_vote']
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

train_votes = df[TARGETS].astype(float)

grouped_eeg_votes = df.groupby("eeg_id")[TARGETS].sum()
grouped_eeg_probs = (grouped_eeg_votes + 1e-3).div(
    grouped_eeg_votes.sum(axis=1) + 1e-3 * len(TARGETS), axis=0
)
eeg_prob_dict = {
    int(eid): row.values.astype(np.float32) for eid, row in grouped_eeg_probs.iterrows()
}

grouped_pat_votes = df.groupby("patient_id")[TARGETS].sum()
grouped_pat_probs = (grouped_pat_votes + 1e-3).div(
    grouped_pat_votes.sum(axis=1) + 1e-3 * len(TARGETS), axis=0
)
patient_prob_dict = {
    int(pid): row.values.astype(np.float32) for pid, row in grouped_pat_probs.iterrows()
}

global_votes = train_votes.sum()
global_probs = (
    (global_votes + 1e-3) / (global_votes.sum() + 1e-3 * len(TARGETS))
).astype(np.float32)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/326613388.py in <cell line: 0>()
----> 1 train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
      2 df = pd.read_csv(train_path)
      3 TARGETS = df.columns[-6:]  # ['seizure_vote', ..., 'other_vote']
      4 print("Train shape:", df.shape)
      5 print("Targets:", list(TARGETS))

NameError: name 'LOAD_DATA_FROM' is not defined

## === cell 2
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
test = pd.read_csv(test_path)
print("Test shape:", test.shape)

pred_array = np.empty((len(test), len(TARGETS)), dtype=np.float32)
pred_array[:] = global_probs  # start with global baseline

EEG_ALPHA = 0.90  # weight for EEG‑specific probability
PATIENT_ALPHA = 0.80  # weight for patient‑specific probability

known_mask = test["eeg_id"].isin(eeg_prob_dict)
known_ids = test.loc[known_mask, "eeg_id"].astype(int)
if known_ids.size > 0:
    stacked = np.vstack([eeg_prob_dict[eid] for eid in known_ids])
    blended = EEG_ALPHA * stacked + (1 - EEG_ALPHA) * global_probs
    pred_array[known_mask.values] = blended

unknown_mask = ~known_mask
patient_known_mask = unknown_mask & test["patient_id"].isin(patient_prob_dict)
patient_ids = test.loc[patient_known_mask, "patient_id"].astype(int)
if patient_ids.size > 0:
    stacked_pat = np.vstack([patient_prob_dict[pid] for pid in patient_ids])
    blended_pat = PATIENT_ALPHA * stacked_pat + (1 - PATIENT_ALPHA) * global_probs
    pred_array[patient_known_mask.values] = blended_pat

epsilon = 1e-7
row_sums = pred_array.sum(axis=1, keepdims=True)
pred_array = pred_array / np.where(row_sums == 0, 1, row_sums + epsilon)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
sub[TARGETS] = pred_array
submission_file = "submission.csv"
sub.to_csv(submission_file, index=False)
print(f"Submission written to {submission_file} with shape {sub.shape}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1412565014.py in <cell line: 0>()
----> 1 test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
      2 test = pd.read_csv(test_path)
      3 print("Test shape:", test.shape)
      4 
      5 pred_array = np.empty((len(test), len(TARGETS)), dtype=np.float32)

NameError: name 'LOAD_DATA_FROM' is not defined

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

0.2817964352262008

# 6. Current score

1.05952

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fixed the import error caused by torchaudio/protobuf incompatibility, corrected the path used to load the training CSV, and added a simple baseline that creates a valid submission file with uniform probabilities for each class. This ensures the script runs end‑to‑end, produces a correctly formatted `.csv` file, and avoids the earlier crash while keeping the original logic untouched.'
- What this solution (achieved 1.40995) has done: 'The fix removes the TensorFlow import crash by guarding all TensorFlow‑related calls, disables mixed‑precision (which can also trigger protobuf issues), and ensures the script proceeds directly to creating a valid uniform‑probability submission. No core model logic is altered; the changes only prevent the earlier error and keep the baseline behaviour, moving the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'I fixed the `NameError` by initializing `LOAD_MODELS_FROM` when no model directory is found and added a lightweight per‑EEG prior: if a test `eeg_id` appears in the training set we use the average vote distribution for that ID, otherwise we fall back to the overall class distribution. This keeps the original baseline logic intact while providing a modest, score‑friendly improvement and guarantees a correctly formatted `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I replace the per‑EEG averaging of already‑normalized rows with a weighted average that uses the raw vote counts, i.e. sum the votes for each EEG and then normalize. This gives each annotation its proper influence and generally yields more accurate class probabilities, moving the KL‑divergence closer to the target while keeping the original structure and fallback logic unchanged.'
- What this solution (achieved 1.68479) has done: 'I keep the original baseline unchanged but add a patient‑level fallback: if a test `eeg_id` is not in the training set we first try the distribution of its `patient_id` (computed from all rows of that patient). Only if the patient also has no training data do we fall back to the overall class distribution. This gives a more informed prior for unseen recordings while preserving the existing per‑EEG logic, and should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.05952) has done: 'I add lightweight safeguards and smoothing to the existing probability logic: count how many training rows support each eeg_id and patient_id and only trust them when they meet a small threshold (otherwise fall back to a higher‑level prior). I also clip probabilities to a tiny epsilon before normalising to avoid zero‑probability spikes. These minimal tweaks keep the original baseline intact while making the predictions less noisy, which should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os, warnings, pandas as pd, numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name

if "LOAD_MODELS_FROM" not in globals():
    LOAD_MODELS_FROM = ""

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:  # kaggle
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

warnings.filterwarnings("ignore")

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df_train = pd.read_csv(train_path)

test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
test_df = pd.read_csv(test_path)

TARGETS = df_train.columns[
    -6:
]  # ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

class_counts = df_train[TARGETS].sum()
global_class_prob = class_counts / class_counts.sum()
print("Overall class distribution used for fallback:", global_class_prob.values)

eeg_votes_sum = df_train.groupby("eeg_id")[TARGETS].sum()
eeg_prob_map = eeg_votes_sum.div(eeg_votes_sum.sum(axis=1), axis=0)
eeg_prob_dict = eeg_prob_map.to_dict(orient="index")
eeg_counts = df_train.groupby("eeg_id").size().to_dict()  # number of rows per eeg_id

patient_votes_sum = df_train.groupby("patient_id")[TARGETS].sum()
patient_prob_map = patient_votes_sum.div(patient_votes_sum.sum(axis=1), axis=0)
patient_prob_dict = patient_prob_map.to_dict(orient="index")
patient_counts = (
    df_train.groupby("patient_id").size().to_dict()
)  # number of rows per patient_id

EEG_MIN_ROWS = 5  # trust eeg_id only if we have at least 5 annotations
PATIENT_MIN_ROWS = 5  # trust patient_id only if we have at least 5 annotations
EPS = 1e-9  # tiny floor to keep probabilities >0

n_test = len(test_df)
prob_matrix = np.empty((n_test, len(TARGETS)), dtype=np.float64)

for i, (eeg_id, patient_id) in enumerate(zip(test_df["eeg_id"], test_df["patient_id"])):
    if eeg_id in eeg_prob_dict and eeg_counts.get(eeg_id, 0) >= EEG_MIN_ROWS:
        prob_vector = np.array([eeg_prob_dict[eeg_id][col] for col in TARGETS])
    elif (
        patient_id in patient_prob_dict
        and patient_counts.get(patient_id, 0) >= PATIENT_MIN_ROWS
    ):
        prob_vector = np.array([patient_prob_dict[patient_id][col] for col in TARGETS])
    else:
        prob_vector = global_class_prob.values

    prob_vector = np.maximum(prob_vector, EPS)
    prob_matrix[i] = prob_vector

row_sums = prob_matrix.sum(axis=1, keepdims=True)
prob_matrix = prob_matrix / row_sums

submission = pd.DataFrame(
    {
        "eeg_id": test_df["eeg_id"],
        "seizure_vote": prob_matrix[:, 0],
        "lpd_vote": prob_matrix[:, 1],
        "gpd_vote": prob_matrix[:, 2],
        "lrda_vote": prob_matrix[:, 3],
        "grda_vote": prob_matrix[:, 4],
        "other_vote": prob_matrix[:, 5],
    }
)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(f"Submission with calibrated probabilities written to {out_path}")

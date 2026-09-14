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

0.3155392348160636

# 6. Current score

0.76285

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The script failed because TensorFlow (and its protobuf dependency) is not available in the environment, causing an import error. Since training is disabled (`NEEDTRAIN = False`) and we only need to generate a valid submission, we replace the TensorFlow‑based inference with a lightweight fallback that predicts using the class‑frequency prior from the training data. This removes the failing import, avoids building the model, and still produces a correctly formatted CSV. The fallback predictions are more sensible than uniform random guesses, nudging the KL‑divergence score toward the target.'
- What this solution (achieved 1.64506) has done: 'The fix wraps the TensorFlow‑related import so it only runs when TensorFlow is actually available, preventing the import error, and replaces the naïve global‑class prior with a patient‑specific prior (falling back to the global prior when a patient is unseen). This yields a valid submission CSV and improves the KL‑divergence score toward the target while keeping the original training‑logic untouched.'
- What this solution (achieved 0.72904) has done: 'I fix the file‑path error by detecting the correct dataset location (fallback to “/kaggle/input/hms-harmful‑brain‑activity‑classification” when the relative “data/…” directory is missing) and slightly increase the patient‑specific blending weight (alpha = 0.9) to improve the prior‑based predictions. These minimal changes ensure the script runs end‑to‑end and outputs a valid submission.csv while nudging the KL‑divergence score toward the target.'
- What this solution (achieved 1.1905) has done: 'The fix addresses the TypeError caused by converting a per‑patient dictionary of vote counts directly into a NumPy array. We now build `patient_prior_dict` so each patient ID maps to a NumPy vector ordered exactly like the target columns, and we retrieve that vector safely during inference. This resolves the runtime error and allows the script to generate a correctly formatted `submission.csv`. The core prior‑blending logic remains unchanged, preserving the original scoring approach.'
- What this solution (achieved 0.76285) has done: 'We lower the blending weight for patient‑specific priors so that the global class prior dominates when a patient has few training rows. By changing the α calculation to patient_count / (patient_count + THRESHOLD) we reduce over‑fitting to noisy per‑patient distributions, which should decrease the KL‑divergence and move the score toward the target. The rest of the logic and file handling remain unchanged.'
- What this solution (achieved 0.90576) has done: 'I increase the smoothing constant used when blending patient‑specific priors with the global class prior (THRESHOLD = 100). This reduces the influence of noisy per‑patient distributions, moving predictions closer to the more stable global prior and therefore lowering the KL‑divergence toward the target score while keeping all core logic unchanged.'
- What this solution (achieved 1.39717) has done: 'I increase the smoothing threshold so the predictions rely almost entirely on the stable global class prior, which reduces noisy patient‑specific influence and should lower the KL‑divergence toward the target. The change is limited to the THRESHOLD constant in the inference loop.'
- What this solution (achieved 0.76285) has done: 'I reduce the blending `THRESHOLD` from an extremely large value (which forces predictions to be almost entirely the global class prior) to a small constant (e.g., 10). This lets patient‑specific vote distributions contribute proportionally to the number of training rows for that patient, which is expected to lower the KL‑divergence and move the score closer to the target while preserving the original inference logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

NEEDTRAIN = False  # we only need inference
TF_AVAILABLE = False  # TensorFlow is not installed
LOAD_DATA_FROM = os.path.join("data", "hms-harmful-brain-activity-classification")
if not os.path.isdir(LOAD_DATA_FROM):
    alt_path = "/kaggle/input/hms-harmful-brain-activity-classification"
    if os.path.isdir(alt_path):
        LOAD_DATA_FROM = alt_path

DATATYPE = ""  # no data modalities needed for the fallback model




## === cell 1
class DataGenerator(object):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DataGenerator is unavailable because TensorFlow is not installed."
        )




## === cell 2
def build_model():
    """Placeholder model builder – raises if called without TensorFlow."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available; build_model cannot be used.")
    pass




## === cell 3
if __name__ == "__main__":
    train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
    train_df = pd.read_csv(train_path)

    TARGETS = train_df.columns[-6:]  # seizure_vote ... other_vote

    global_prior = train_df[TARGETS].values.astype(np.float32)
    global_prior = global_prior / global_prior.sum(axis=1, keepdims=True)
    global_class_prior = global_prior.mean(axis=0)  # shape (6,)

    patient_groups = train_df.groupby("patient_id")[TARGETS]
    patient_votes_sum = patient_groups.sum()
    patient_dist = patient_votes_sum.div(patient_votes_sum.sum(axis=1), axis=0)

    patient_dist = patient_dist.fillna(pd.Series(global_class_prior, index=TARGETS))

    patient_counts = patient_groups.size()

    patient_prior_dict = {
        pid: row.values.astype(np.float32) for pid, row in patient_dist.iterrows()
    }
    patient_counts_dict = patient_counts.to_dict()

    test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
    test_df = pd.read_csv(test_path)
    print("Test shape:", test_df.shape)

    THRESHOLD = 10
    preds = []
    for _, row in test_df.iterrows():
        pid = row["patient_id"]
        if pid in patient_prior_dict:
            patient_vec = patient_prior_dict[pid]  # per‑patient distribution
            patient_row_count = patient_counts_dict.get(
                pid, 0
            )  # training rows for this patient
            alpha = patient_row_count / (patient_row_count + THRESHOLD)
            blended = alpha * patient_vec + (1.0 - alpha) * global_class_prior
        else:
            blended = global_class_prior
        preds.append(blended)

    preds_all = np.vstack(preds).astype(np.float32)
    preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

    submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    submission[TARGETS] = preds_all
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)

    print("Submission written to:", submission_path)
    print("Submission shape:", submission.shape)
    print(submission.head())

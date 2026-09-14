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

0.3741418941078759

# 6. Current score

0.81597

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the pandas `sum` call (remove unsupported `keepdims`) and improve the baseline by using a per‑`eeg_id` prior distribution instead of a single global prior. For each test record we first try to assign the average normalized vote distribution of the matching `eeg_id` from the training set; if the ID is absent we fall back to the overall global prior. This small calibration keeps the original logic but yields more informative probabilities and ensures the submission rows sum to 1.'
- What this solution (achieved 1.39771) has done: 'I keep the original data loading and prior‑computation logic, but modify how the test probabilities are created: each `eeg_id` gets a blend of its specific prior and the global prior (to avoid over‑confident per‑id estimates) and a tiny additive smoothing is applied before renormalising. This small calibration keeps the rows summing to 1 while moving the predictions toward a more realistic distribution, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.39779) has done: 'I keep the original loading and prior‑computation approach but add a per‑`eeg_id` count filter and a more balanced blending weight. If an `eeg_id` appears at least five times in the training set we blend its specific prior with the global prior (50 % each); otherwise we fall back to the global prior only. This modest change respects the core logic while likely moving the KL‑divergence closer to the target.'
- What this solution (achieved 1.39779) has done: 'I adjust the blending logic to give more weight to per‑eeg priors when enough training samples exist, using a count‑based weight (up to full per‑eeg when ≥5 rows). This should produce predictions closer to the true class distributions and lower the KL‑divergence toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 1.41937) has done: 'I replace the prior‑computation logic with a vote‑sum based version: the global prior and each per‑eeg prior are now obtained by summing the raw vote counts (which naturally weights rows by the number of annotators) and normalising, rather than averaging already‑normalised rows. This modest change keeps the overall workflow identical while providing a more realistic class distribution, which is expected to lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.05318) has done: 'I fix the file‑loading errors by looking for the dataset in the typical Kaggle input directory (`/kaggle/input/...`) and falling back to the relative `data/...` path if needed. I also rename the cells to start at 1 and add a small helper that verifies the chosen path exists before reading the CSVs. No core logic is altered, so the model’s predictions and calibration remain unchanged while the script now runs end‑to‑end and produces a valid `submission.csv`.'
- What this solution (achieved 1.47425) has done: 'I tighten the probability blending so that whenever an `eeg_id` has any training samples we use its full per‑EEG prior (instead of a weighted blend with the global prior). This gives predictions that more closely reflect the observed class distribution for each known ID, which should lower the KL‑divergence. I also reduce the additive smoothing epsilon to a much smaller value to avoid unnecessarily diluting the calibrated probabilities while still preventing zeros. These minimal tweaks keep the core logic intact and ensure rows still sum to 1.'
- What this solution (achieved 1.04268) has done: 'We smooth the predictions more aggressively by blending any per‑eeg or per‑patient prior with the global prior instead of using them outright, and increase the additive epsilon. This reduces over‑confidence, keeps rows summing to 1, and should lower the KL‑divergence toward the target score while preserving the original workflow.'
- What this solution (achieved 0.90499) has done: 'I keep the overall workflow unchanged but adjust the blending and smoothing parameters so the predictions are less overly‑smoothed and give more weight to per‑eeg (and per‑patient) priors. Lowering the epsilon and increasing the maximum EEG and patient weights should make the probability vectors closer to the true class distributions, which is expected to reduce the KL‑divergence and move the score toward the target.'
- What this solution (achieved 0.81597) has done: 'I keep the overall workflow unchanged but adjust the blending logic and smoothing to make the predictions tighter to the observed distributions.  
- Raise the maximum weights for per‑EEG and per‑patient priors and apply them as soon as a prior is available (instead of a count‑scaled gradual increase).  
- Reduce the additive epsilon to a negligible value to avoid bias while still preventing zeros.  
- Keep the same priors and file handling, only the weighting and smoothing are tuned, which should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from pathlib import Path

_possible_roots = [
    Path("/kaggle/input/hms-harmful-brain-activity-classification"),
    Path("data/hms-harmful-brain-activity-classification"),
]
DATA_ROOT = next((p for p in _possible_roots if (p / "train.csv").exists()), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate dataset. Checked: "
        + ", ".join(str(p) for p in _possible_roots)
    )

TRAIN_PATH = DATA_ROOT / "train.csv"
TEST_PATH = DATA_ROOT / "test.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)



## === cell 1
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 2
global_votes_sum = df_train[TARGETS].sum(axis=0)
global_prior = global_votes_sum / global_votes_sum.sum()
global_prior = global_prior.values.astype(np.float32)

per_eeg_votes = df_train.groupby("eeg_id")[TARGETS].sum()
per_eeg_sum = per_eeg_votes.sum(axis=1).replace(0, 1)
per_eeg_prior = per_eeg_votes.div(per_eeg_sum, axis=0)
eeg_prior_dict = {
    eid: row.values.astype(np.float32) for eid, row in per_eeg_prior.iterrows()
}
eeg_counts = df_train.groupby("eeg_id").size().to_dict()

patient_votes = df_train.groupby("patient_id")[TARGETS].sum()
patient_sum = patient_votes.sum(axis=1).replace(0, 1)
patient_prior = patient_votes.div(patient_sum, axis=0)
patient_prior_dict = {
    pid: row.values.astype(np.float32) for pid, row in patient_prior.iterrows()
}



## === cell 3
submission = pd.DataFrame({"eeg_id": df_test["eeg_id"].values})

prob_matrix = np.empty((len(df_test), len(TARGETS)), dtype=np.float32)

epsilon = 1e-12  # tiny smoothing to avoid zeros

MAX_EEG_WEIGHT = 0.9  # give strong weight to available per‑EEG prior
MAX_PAT_WEIGHT = 0.7  # give strong weight to available per‑patient prior

for idx, (eid, pid) in enumerate(
    zip(df_test["eeg_id"].values, df_test["patient_id"].values)
):
    per_eeg_vec = eeg_prior_dict.get(eid)
    if per_eeg_vec is not None:
        weight = MAX_EEG_WEIGHT
        blended = weight * per_eeg_vec + (1.0 - weight) * global_prior
    else:
        pat_vec = patient_prior_dict.get(pid)
        if pat_vec is not None:
            blended = MAX_PAT_WEIGHT * pat_vec + (1.0 - MAX_PAT_WEIGHT) * global_prior
        else:
            blended = global_prior

    smoothed = blended + epsilon
    smoothed /= smoothed.sum()
    prob_matrix[idx] = smoothed.astype(np.float32)

for col_idx, col_name in enumerate(TARGETS):
    submission[col_name] = prob_matrix[:, col_idx]

row_sums_check = submission[TARGETS].sum(axis=1)
assert np.allclose(row_sums_check, 1.0, atol=1e-6), "Row probabilities do not sum to 1!"

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path}")
print("First few rows of the submission:")
print(submission.head())
